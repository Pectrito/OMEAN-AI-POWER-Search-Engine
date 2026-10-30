from __future__ import annotations

import re
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


VOCAB_SIZE = 4096
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
SEP_ID = 3
UNK_ID = 4
FIRST_WORD_ID = 5
MAX_SOURCE_LENGTH = 128
MAX_SUMMARY_LENGTH = 40
MODEL_VERSION = 1
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def token_id(token: str) -> int:
    return FIRST_WORD_ID + zlib.crc32(token.encode("utf-8")) % (VOCAB_SIZE - FIRST_WORD_ID)


def encode_text(
    text: str,
    max_length: int = MAX_SOURCE_LENGTH,
    vocabulary: Mapping[str, int] | None = None,
) -> list[int]:
    if max_length < 2:
        raise ValueError("max_length must leave room for beginning and end tokens.")
    tokens = TOKEN_PATTERN.findall(text.casefold())
    if vocabulary is None:
        word_ids = [token_id(token) for token in tokens]
    else:
        word_ids = [vocabulary.get(token, UNK_ID) for token in tokens]
    return [BOS_ID, *word_ids[: max_length - 2], EOS_ID]


def encode_query_document(
    query: str,
    document: str,
    vocabulary: Mapping[str, int] | None = None,
) -> list[int]:
    query_ids = encode_text(query, max_length=MAX_SOURCE_LENGTH, vocabulary=vocabulary)
    document_ids = encode_text(document, max_length=MAX_SOURCE_LENGTH, vocabulary=vocabulary)
    combined = [BOS_ID, *query_ids[1:-1], SEP_ID, *document_ids[1:-1], EOS_ID]
    return combined[:MAX_SOURCE_LENGTH - 1] + [EOS_ID]


def pad_sequences(sequences: Sequence[Sequence[int]], device: torch.device) -> Tensor:
    if not sequences:
        raise ValueError("At least one sequence is required.")
    width = max(len(sequence) for sequence in sequences)
    result = torch.full((len(sequences), width), PAD_ID, dtype=torch.long, device=device)
    for row, sequence in enumerate(sequences):
        result[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
    return result


class MultiTaskSearchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        dimension = 128
        self.token_embedding = nn.Embedding(VOCAB_SIZE, dimension, padding_idx=PAD_ID)
        self.position_embedding = nn.Embedding(MAX_SOURCE_LENGTH, dimension)
        self.input_norm = nn.LayerNorm(dimension)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=4,
            dim_feedforward=384,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=1, enable_nested_tensor=False
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dimension,
            nhead=4,
            dim_feedforward=384,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.rank_head = nn.Sequential(
            nn.Linear(dimension, 446),
            nn.GELU(),
            nn.Linear(446, 1),
        )
        self.output_bias = nn.Parameter(torch.zeros(VOCAB_SIZE))
        self.vocabulary: dict[int, str] = {UNK_ID: "<unk>"}
        self.token_to_id: dict[str, int] = {}

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _embed(self, token_ids: Tensor) -> Tensor:
        if token_ids.shape[1] > MAX_SOURCE_LENGTH:
            raise ValueError(f"Input exceeds {MAX_SOURCE_LENGTH} tokens.")
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        return self.input_norm(
            self.token_embedding(token_ids) + self.position_embedding(positions).unsqueeze(0)
        )

    def _encode(self, source_ids: Tensor) -> tuple[Tensor, Tensor]:
        padding_mask = source_ids.eq(PAD_ID)
        memory = self.encoder(self._embed(source_ids), src_key_padding_mask=padding_mask)
        return memory, padding_mask

    def rank_logits(self, pair_ids: Tensor) -> Tensor:
        memory, _ = self._encode(pair_ids)
        return self.rank_head(memory[:, 0]).squeeze(-1)

    def summary_logits(self, source_ids: Tensor, decoder_ids: Tensor) -> Tensor:
        memory, source_padding_mask = self._encode(source_ids)
        target_padding_mask = decoder_ids.eq(PAD_ID)
        target = self._embed(decoder_ids)
        causal_mask = torch.triu(
            torch.ones(
                decoder_ids.shape[1],
                decoder_ids.shape[1],
                dtype=torch.bool,
                device=decoder_ids.device,
            ),
            diagonal=1,
        )
        decoded = self.decoder(
            target,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=source_padding_mask,
        )
        return F.linear(decoded, self.token_embedding.weight, self.output_bias)

    @torch.inference_mode()
    def rank_score(self, query: str, body: str) -> float:
        self.eval()
        pair = encode_query_document(query, body[:3000], self.token_to_id)
        ids = torch.tensor(pair, dtype=torch.long, device=next(self.parameters()).device).unsqueeze(0)
        return float(torch.sigmoid(self.rank_logits(ids))[0].item())

    @torch.inference_mode()
    def summarize(
        self, body: str, max_tokens: int = 32, vocabulary: Mapping[int, str] | None = None
    ) -> str:
        self.eval()
        device = next(self.parameters()).device
        source = torch.tensor(
            encode_text(body[:6000], vocabulary=self.token_to_id),
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        memory, source_padding_mask = self._encode(source)
        generated = [BOS_ID]
        for _ in range(min(max_tokens, MAX_SUMMARY_LENGTH - 1)):
            decoder_ids = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)
            target = self._embed(decoder_ids)
            causal_mask = torch.triu(
                torch.ones(
                    len(generated),
                    len(generated),
                    dtype=torch.bool,
                    device=device,
                ),
                diagonal=1,
            )
            decoded = self.decoder(
                target,
                memory,
                tgt_mask=causal_mask,
                memory_key_padding_mask=source_padding_mask,
            )
            logits = F.linear(decoded[:, -1], self.token_embedding.weight, self.output_bias)
            token_id = int(logits.argmax(dim=-1).item())
            if token_id in {EOS_ID, PAD_ID, BOS_ID, SEP_ID}:
                break
            generated.append(token_id)
        token_names = vocabulary if vocabulary is not None else self.vocabulary
        decoded = [
            "" if token_id == UNK_ID else token_names.get(token_id, "")
            for token_id in generated[1:]
        ]
        output = ""
        for token in decoded:
            if not token:
                continue
            if token in {".", ",", "!", "?", ":", ";", ")", "]"}:
                output = output.rstrip() + token
            elif token in {"(", "["}:
                output += (" " if output else "") + token
            else:
                output += (" " if output else "") + token
        return output.strip()


def load_model(path: Path) -> tuple[MultiTaskSearchModel, dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("version") != MODEL_VERSION:
        raise ValueError(f"Unsupported search model version in {path}.")
    model = MultiTaskSearchModel()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if model.parameter_count() != checkpoint.get("parameter_count"):
        raise ValueError(f"Model parameter count does not match checkpoint {path}.")
    model.eval()
    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid training metadata in {path}.")
    saved_vocabulary = checkpoint.get("vocabulary", {})
    if not isinstance(saved_vocabulary, dict):
        raise ValueError(f"Invalid model vocabulary in {path}.")
    model.vocabulary = {int(key): str(value) for key, value in saved_vocabulary.items()}
    model.token_to_id = {
        token: token_id for token_id, token in model.vocabulary.items() if token_id >= FIRST_WORD_ID
    }
    return model, metadata
