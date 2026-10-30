from __future__ import annotations

import hashlib
import html
import importlib.util
import ipaddress
import json
import logging
import math
import re
import socket
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.robotparser
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable


SUPPORTED_EXTENSIONS = {".txt", ".md", ".rst", ".csv", ".json", ".html", ".htm"}
USER_AGENT = "OmeanSearchBot/0.1 (+local search prototype)"
MAX_DOCUMENT_BYTES = 2 * 1024 * 1024
MAX_PAGE_BYTES = 1024 * 1024
MAX_CRAWL_PAGES = 50
CRAWL_DELAY_SECONDS = 1.0
TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
SUMMARY_STOP_WORDS = {
    "also",
    "are",
    "been",
    "being",
    "but",
    "for",
    "from",
    "had",
    "has",
    "have",
    "her",
    "his",
    "into",
    "its",
    "not",
    "one",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "they",
    "this",
    "those",
    "was",
    "were",
    "which",
    "who",
    "with",
    "about",
    "an",
    "and",
    "as",
    "at",
    "be",
    "can",
    "do",
    "does",
    "how",
    "in",
    "of",
    "on",
    "or",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "you",
}

QUESTION_STOP_WORDS = SUMMARY_STOP_WORDS | {
    "all",
    "any",
    "could",
    "did",
    "i",
    "is",
    "it",
    "me",
    "my",
    "please",
    "should",
    "tell",
    "the",
    "to",
    "would",
}


class DocumentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self._in_title = False
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in {"p", "div", "li", "br", "h1", "h2", "h3", "article"}:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in {"p", "div", "li", "br", "h1", "h2", "h3", "article"}:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        cleaned = " ".join(data.split())
        if not cleaned:
            return
        if self._in_title:
            self.title_parts.append(cleaned)
        self.text_parts.append(cleaned)


class HTTPStatusError(ValueError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class PublicRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, allowed_origin: tuple[str, str]) -> None:
        super().__init__()
        self.allowed_origin = allowed_origin

    def redirect_request(self, request, response, code, message, headers, new_url):
        parsed = _validate_public_url(new_url)
        origin = (parsed.scheme, parsed.netloc.casefold())
        if origin != self.allowed_origin:
            raise ValueError("Redirect left the seed origin.")
        return super().redirect_request(request, response, code, message, headers, new_url)


def tokenize(text: str) -> list[str]:
    return [match.group(0).casefold() for match in TOKEN_PATTERN.finditer(text)]


def search_terms(query: str) -> list[str]:
    terms = [term for term in tokenize(query) if term not in QUESTION_STOP_WORDS]
    if "ai" in terms:
        terms.extend(("artificial", "intelligence"))
    return list(dict.fromkeys(terms))[:20]


def is_question(query: str) -> bool:
    return bool(re.match(r"^\s*(what|who|when|where|why|how|which|define)\b", query, re.IGNORECASE))


def extract_text(path: Path, raw: str) -> tuple[str, str]:
    suffix = path.suffix.casefold()
    if suffix in {".html", ".htm"}:
        parser = DocumentParser()
        parser.feed(raw)
        parser.close()
        title = " ".join(parser.title_parts).strip() or path.stem
        body = " ".join(" ".join(parser.text_parts).split())
        return title, html.unescape(body)

    if suffix == ".md":
        title = path.stem
        for line in raw.splitlines():
            heading = re.match(r"^\s*#\s+(.+?)\s*#*\s*$", line)
            if heading:
                title = heading.group(1)
                break
        body = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", raw)
        body = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", body)
        body = re.sub(r"^\s{0,3}#{1,6}\s*", "", body, flags=re.MULTILINE)
        body = re.sub(r"[*_`~>|]", " ", body)
        return title, " ".join(body.split())

    return path.stem, " ".join(raw.split())


@dataclass(frozen=True)
class SearchResult:
    title: str
    path: str
    summary: str
    score: float
    source: str
    source_id: str


class LocalSearchEngine:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir.resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = self.data_dir / "search-index.sqlite3"
        self._connection = sqlite3.connect(self.database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                source_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                source TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS app_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._connection.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                title,
                body,
                content='documents',
                content_rowid='rowid',
                tokenize='unicode61 remove_diacritics 2'
            )
            """
        )
        self._connection.executescript(
            """
            CREATE TRIGGER IF NOT EXISTS documents_fts_insert AFTER INSERT ON documents BEGIN
                INSERT INTO documents_fts(rowid, title, body)
                VALUES (new.rowid, new.title, new.body);
            END;
            CREATE TRIGGER IF NOT EXISTS documents_fts_delete AFTER DELETE ON documents BEGIN
                INSERT INTO documents_fts(documents_fts, rowid, title, body)
                VALUES ('delete', old.rowid, old.title, old.body);
            END;
            CREATE TRIGGER IF NOT EXISTS documents_fts_update AFTER UPDATE OF title, body ON documents BEGIN
                INSERT INTO documents_fts(documents_fts, rowid, title, body)
                VALUES ('delete', old.rowid, old.title, old.body);
                INSERT INTO documents_fts(rowid, title, body)
                VALUES (new.rowid, new.title, new.body);
            END;
            """
        )
        initialized = self._connection.execute(
            "SELECT value FROM app_metadata WHERE key = 'documents_fts_initialized'"
        ).fetchone()
        if initialized is None:
            self._connection.execute("INSERT INTO documents_fts(documents_fts) VALUES ('rebuild')")
            self._connection.execute(
                "INSERT INTO app_metadata(key, value) VALUES ('documents_fts_initialized', '1')"
            )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS relevance_feedback (
                query TEXT NOT NULL,
                source_id TEXT NOT NULL,
                relevant INTEGER NOT NULL CHECK (relevant IN (0, 1)),
                PRIMARY KEY (query, source_id)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS search_model (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        self._connection.commit()
        self._ranker_weights: list[float] | None = None
        self._train_ranker()
        self._neural_model = None
        self._neural_model_error: str | None = None
        self.model_metadata: dict[str, object] = {}
        training_model_path = self.data_dir / "training-output" / "multitask-search-model.pt"
        published_model_path = self.data_dir.parent / "omean" / "multitask-search-model.pt"
        standalone_model_path = self.data_dir.parent / "multitask-search-model.pt"
        legacy_bundled_model_path = self.data_dir / "omean" / "multitask-search-model.pt"
        legacy_model_path = self.data_dir / "multitask-search-model.pt"
        self.model_path = next(
            (
                path
                for path in (
                    training_model_path,
                    published_model_path,
                    standalone_model_path,
                    legacy_bundled_model_path,
                    legacy_model_path,
                )
                if path.is_file()
            ),
            published_model_path,
        )
        if self.model_path.is_file():
            try:
                if importlib.util.find_spec("torch") is None:
                    raise ModuleNotFoundError("No module named 'torch'", name="torch")
                try:
                    from model_runtime import load_model
                except ModuleNotFoundError as error:
                    if error.name != "model_runtime":
                        raise
                    from neural_model import load_model

                self._neural_model, self.model_metadata = load_model(self.model_path)
            except ModuleNotFoundError as error:
                if error.name != "torch":
                    raise
                self._neural_model_error = (
                    "The trained neural model needs PyTorch, which is not installed in this "
                    "Python environment. Using BM25 and extractive summaries instead."
                )
                logging.getLogger(__name__).warning(
                    "%s Start the app with the Python environment used for training, "
                    "or install a PyTorch build compatible with this Python version.",
                    self._neural_model_error,
                )

    def close(self) -> None:
        self._connection.close()

    def index_directory(self) -> dict[str, int | list[str]]:
        indexed = 0
        errors: list[str] = []
        observed: set[str] = set()

        for path in sorted(self.data_dir.rglob("*")):
            if not path.is_file() or path.is_symlink() or path.suffix.casefold() not in SUPPORTED_EXTENSIONS:
                continue
            relative_path = path.relative_to(self.data_dir).as_posix()
            if relative_path.startswith("web/"):
                continue
            source_id = f"file:{relative_path}"
            observed.add(source_id)
            try:
                size = path.stat().st_size
                if size > MAX_DOCUMENT_BYTES:
                    errors.append(f"{relative_path}: larger than {MAX_DOCUMENT_BYTES} bytes")
                    continue
                raw = path.read_text(encoding="utf-8", errors="replace")
                title, body = extract_text(path, raw)
                if not tokenize(body):
                    continue
                self._connection.execute(
                    """
                    INSERT INTO documents(source_id, title, body, source)
                    VALUES (?, ?, ?, 'local')
                    ON CONFLICT(source_id) DO UPDATE SET
                        title = excluded.title, body = excluded.body, source = excluded.source
                    """,
                    (source_id, title, body),
                )
                indexed += 1
            except OSError as error:
                errors.append(f"{relative_path}: {error}")

        existing = self._connection.execute(
            "SELECT source_id FROM documents WHERE source_id LIKE 'file:%'"
        ).fetchall()
        stale = [row["source_id"] for row in existing if row["source_id"] not in observed]
        self._connection.executemany("DELETE FROM documents WHERE source_id = ?", ((item,) for item in stale))
        self._connection.commit()
        return {"indexed": indexed, "removed": len(stale), "errors": errors}

    def add_web_page(self, source_id: str, title: str, body: str) -> None:
        self._connection.execute(
            """
            INSERT INTO documents(source_id, title, body, source)
            VALUES (?, ?, ?, 'web')
            ON CONFLICT(source_id) DO UPDATE SET
                title = excluded.title, body = excluded.body, source = excluded.source
            """,
            (source_id, title, body),
        )
        self._connection.commit()

    def import_wikipedia_articles(
        self,
        articles: Iterable[tuple[str, str, str]],
        batch_size: int = 500,
    ) -> int:
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        imported = 0
        batch: list[tuple[str, str, str, str]] = []
        for article_id, title, body in articles:
            if not title.strip() or not body.strip():
                continue
            batch.append(
                (f"wikipedia:20231101-en:{article_id}", title.strip(), body.strip(), "wikipedia")
            )
            if len(batch) >= batch_size:
                self._connection.executemany(
                    """
                    INSERT INTO documents(source_id, title, body, source)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(source_id) DO UPDATE SET
                        title = excluded.title, body = excluded.body, source = excluded.source
                    """,
                    batch,
                )
                self._connection.commit()
                imported += len(batch)
                batch.clear()
        if batch:
            self._connection.executemany(
                """
                INSERT INTO documents(source_id, title, body, source)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    title = excluded.title, body = excluded.body, source = excluded.source
                """,
                batch,
            )
            self._connection.commit()
            imported += len(batch)
        return imported

    def model_status(self) -> dict[str, int | bool]:
        row = self._connection.execute("SELECT COUNT(*) AS count FROM relevance_feedback").fetchone()
        count = int(row["count"])
        return {
            "trained": self._ranker_weights is not None,
            "feedback_examples": count,
            "parameters": len(self._ranker_weights) if self._ranker_weights is not None else 0,
        }

    def document_count(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) AS count FROM documents").fetchone()
        return int(row["count"])

    def rate_result(self, query: str, source_id: str, relevant: bool) -> dict[str, int | bool]:
        if not isinstance(query, str) or not isinstance(source_id, str):
            raise ValueError("query and source_id must be strings.")
        clean_query = " ".join(query.split())
        if not clean_query or len(clean_query) > 200:
            raise ValueError("A query between 1 and 200 characters is required.")
        if not source_id:
            raise ValueError("A document source_id is required.")
        if not isinstance(relevant, bool):
            raise ValueError("relevant must be true or false.")
        exists = self._connection.execute(
            "SELECT 1 FROM documents WHERE source_id = ?", (source_id,)
        ).fetchone()
        if exists is None:
            raise ValueError("The rated document is no longer in the index.")
        self._connection.execute(
            """
            INSERT INTO relevance_feedback(query, source_id, relevant) VALUES (?, ?, ?)
            ON CONFLICT(query, source_id) DO UPDATE SET relevant = excluded.relevant
            """,
            (clean_query, source_id, int(relevant)),
        )
        self._connection.commit()
        self._train_ranker()
        return self.model_status()

    def _feature_vector(self, query: str, title: str, body: str) -> list[float]:
        terms = set(search_terms(query))
        if not terms:
            return [0.0, 0.0, 0.0, 0.0, 1.0]
        title_terms = set(tokenize(title))
        body_terms = set(tokenize(body))
        folded_query = " ".join(query.casefold().split())
        folded_body = body.casefold()
        body_length = len(tokenize(body))
        return [
            len(terms & title_terms) / len(terms),
            len(terms & body_terms) / len(terms),
            float(len(folded_query) > 2 and folded_query in folded_body),
            min(math.log1p(body_length) / 10, 1.0),
            1.0,
        ]

    def _train_ranker(self) -> None:
        rows = self._connection.execute(
            """
            SELECT f.query, d.title, d.body, f.relevant
            FROM relevance_feedback AS f
            JOIN documents AS d ON d.source_id = f.source_id
            """
        ).fetchall()
        if len(rows) < 8 or {int(row["relevant"]) for row in rows} != {0, 1}:
            self._ranker_weights = None
            self._connection.execute("DELETE FROM search_model WHERE key = 'ranker_weights'")
            self._connection.commit()
            return

        examples = [
            (self._feature_vector(row["query"], row["title"], row["body"]), int(row["relevant"]))
            for row in rows
        ]
        weights = [0.0] * len(examples[0][0])
        learning_rate = 0.2
        for _ in range(400):
            gradients = [0.0] * len(weights)
            for features, label in examples:
                logit = max(-20.0, min(20.0, sum(w * x for w, x in zip(weights, features))))
                probability = 1.0 / (1.0 + math.exp(-logit))
                error = probability - label
                for index, value in enumerate(features):
                    gradients[index] += error * value
            for index in range(len(weights) - 1):
                gradients[index] = gradients[index] / len(examples) + 0.01 * weights[index]
            gradients[-1] /= len(examples)
            weights = [weight - learning_rate * gradient for weight, gradient in zip(weights, gradients)]

        self._ranker_weights = weights
        self._connection.execute(
            """
            INSERT INTO search_model(key, value) VALUES ('ranker_weights', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (json.dumps(weights),),
        )
        self._connection.commit()

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        base_terms = [term for term in tokenize(query) if term not in QUESTION_STOP_WORDS]
        terms = search_terms(query)
        if not terms:
            return []
        match_query = " OR ".join(f'"{term}"' for term in dict.fromkeys(terms))
        candidate_limit = max(50, min(limit * 10, 200))
        rows = self._connection.execute(
            """
            SELECT d.source_id, d.title, d.body, d.source,
                   bm25(documents_fts, 5.0, 1.0) AS fts_score
            FROM documents_fts
            JOIN documents AS d ON d.rowid = documents_fts.rowid
            WHERE documents_fts MATCH ?
            ORDER BY fts_score
            LIMIT ?
            """,
            (match_query, candidate_limit),
        ).fetchall()
        if not rows:
            return []

        scored: list[tuple[float, sqlite3.Row]] = []
        for candidate_index, row in enumerate(rows):
            score = -float(row["fts_score"]) * 1_000_000
            title_terms = set(tokenize(row["title"]))
            score += 1.5 * sum(1 for term in terms if term in title_terms)
            title_phrase = " ".join(tokenize(row["title"]))
            preferred_phrase = " ".join(base_terms)
            if base_terms == ["ai"]:
                preferred_phrase = "artificial intelligence"
            if title_phrase == preferred_phrase:
                score += 10_000_000 if base_terms == ["ai"] else 5_000_000
            elif base_terms == ["ai"] and title_phrase == "ai":
                score += 1_000_000
            folded_body = row["body"].casefold()
            folded_query = " ".join(query.casefold().split())
            if len(folded_query) > 2 and folded_query in folded_body:
                score += 1.0
            if self._ranker_weights is not None:
                features = self._feature_vector(query, row["title"], row["body"])
                logit = max(-20.0, min(20.0, sum(w * x for w, x in zip(self._ranker_weights, features))))
                score += 2.0 / (1.0 + math.exp(-logit))
            if self._neural_model is not None and candidate_index < 20:
                neural_score = self._neural_model.rank_score(query, row["body"])
                score *= 0.8 + 0.4 * neural_score
            if score:
                scored.append((score, row))

        scored.sort(key=lambda result: (-result[0], result[1]["title"].casefold()))
        results: list[SearchResult] = []
        for score, row in scored[: max(1, min(limit, 50))]:
            summary = summarize(row["body"], terms, max_sentences=1 if is_question(query) else 2)
            if self._neural_model is not None:
                generated = self._neural_model.summarize(row["body"])
                if generated and is_grounded_summary(generated, row["body"]):
                    summary = generated
            results.append(
                SearchResult(
                    title=row["title"],
                    path=row["title"] if row["source"] == "wikipedia" else row["source_id"].removeprefix("file:"),
                    summary=summary,
                    score=round(score, 4),
                    source=row["source"],
                    source_id=row["source_id"],
                )
            )
        return results


def summarize(
    text: str,
    query_terms: Iterable[str],
    max_chars: int = 360,
    max_sentences: int = 2,
) -> str:
    sentences = [sentence.strip() for sentence in SENTENCE_PATTERN.split(text) if sentence.strip()]
    if not sentences:
        return ""
    terms = set(query_terms)
    ranked: list[tuple[float, int, str]] = []
    for index, sentence in enumerate(sentences):
        words = tokenize(sentence)
        overlap = sum(1 for word in set(words) if word in terms)
        lead_bonus = 0.15 if index == 0 else 0
        length_bonus = min(len(words), 30) / 300
        ranked.append((overlap + lead_bonus + length_bonus, index, sentence))
    selected = sorted(
        sorted(ranked, key=lambda item: (-item[0], item[1]))[:max(1, max_sentences)],
        key=lambda item: item[1],
    )
    summary = " ".join(sentence for _, _, sentence in selected)
    if len(summary) > max_chars:
        summary = summary[: max_chars - 1].rsplit(" ", 1)[0] + "…"
    return summary


def is_grounded_summary(candidate: str, source: str) -> bool:
    content_terms = {
        term
        for term in tokenize(candidate)
        if len(term) > 2 and term not in SUMMARY_STOP_WORDS
    }
    if len(content_terms) < 3:
        return False
    source_terms = set(tokenize(source[:6000]))
    return len(content_terms & source_terms) / len(content_terms) >= 0.9


def _validate_public_url(url: str) -> urllib.parse.ParseResult:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only absolute http:// and https:// URLs are allowed.")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not allowed.")
    hostname = parsed.hostname.rstrip(".").casefold()
    if hostname == "localhost" or hostname.endswith(".localhost") or hostname.endswith(".local"):
        raise ValueError("Local and intranet hosts are not allowed.")
    try:
        addresses = {ipaddress.ip_address(item[4][0]) for item in socket.getaddrinfo(hostname, None)}
    except socket.gaierror as error:
        raise ValueError(f"Could not resolve host {hostname}: {error}") from error
    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError("Only publicly routable hosts are allowed.")
    return parsed


def _fetch(url: str, allowed_origin: tuple[str, str]) -> tuple[bytes, str, str]:
    parsed_url = _validate_public_url(url)
    if (parsed_url.scheme, parsed_url.netloc.casefold()) != allowed_origin:
        raise ValueError("URL is outside the seed origin.")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        opener = urllib.request.build_opener(PublicRedirectHandler(allowed_origin))
        with opener.open(request, timeout=15) as response:
            final_url = response.geturl()
            final = _validate_public_url(final_url)
            if (final.scheme, final.netloc.casefold()) != allowed_origin:
                raise ValueError("Redirect left the seed origin.")
            content_type = response.headers.get_content_type()
            data = response.read(MAX_PAGE_BYTES + 1)
            if len(data) > MAX_PAGE_BYTES:
                raise ValueError(f"Response from {url} exceeded {MAX_PAGE_BYTES} bytes.")
            return data, content_type, final_url
    except urllib.error.HTTPError as error:
        raise HTTPStatusError(error.code, f"HTTP {error.code} while fetching {url}") from error
    except urllib.error.URLError as error:
        raise ValueError(f"Could not fetch {url}: {error}") from error


def crawl_web(seed_url: str, data_dir: Path, max_pages: int = 10) -> dict[str, int | list[str]]:
    if not 1 <= max_pages <= MAX_CRAWL_PAGES:
        raise ValueError(f"max_pages must be between 1 and {MAX_CRAWL_PAGES}.")
    seed = _validate_public_url(seed_url)
    origin = (seed.scheme, seed.netloc.casefold())
    base = f"{seed.scheme}://{seed.netloc}"
    robots_url = urllib.parse.urljoin(base, "/robots.txt")
    robots = urllib.robotparser.RobotFileParser()
    robots.set_url(robots_url)
    try:
        robots_data, _, _ = _fetch(robots_url, origin)
        robots.parse(robots_data.decode("utf-8", errors="replace").splitlines())
    except HTTPStatusError as error:
        if error.status != 404:
            raise
        robots.parse([])

    queue = [urllib.parse.urlunparse(seed._replace(fragment=""))]
    visited: set[str] = set()
    stored = 0
    errors: list[str] = []
    engine = LocalSearchEngine(data_dir)
    try:
        while queue and stored < max_pages:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)
            try:
                parsed = _validate_public_url(url)
                if (parsed.scheme, parsed.netloc.casefold()) != origin:
                    continue
                if not robots.can_fetch(USER_AGENT, url):
                    errors.append(f"{url}: disallowed by robots.txt")
                    continue
                data, content_type, final_url = _fetch(url, origin)
                final = urllib.parse.urlparse(final_url)
                if (final.scheme, final.netloc.casefold()) != origin:
                    errors.append(f"{url}: redirect left the seed origin")
                    continue
                if content_type not in {"text/html", "application/xhtml+xml"}:
                    continue
                raw = data.decode("utf-8", errors="replace")
                page_path = Path("web") / (hashlib.sha256(final_url.encode("utf-8")).hexdigest() + ".html")
                destination = data_dir / page_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                title, body = extract_text(destination, raw)
                if tokenize(body):
                    engine.add_web_page(f"web:{page_path.as_posix()}", title, body)
                    stored += 1

                parser = LinkParser(final_url)
                parser.feed(raw)
                for link in parser.links:
                    absolute = urllib.parse.urlparse(link)
                    if (absolute.scheme, absolute.netloc.casefold()) == origin:
                        normalized = urllib.parse.urlunparse(absolute._replace(fragment=""))
                        if normalized not in visited and normalized not in queue:
                            queue.append(normalized)
            except (OSError, ValueError, UnicodeError) as error:
                errors.append(f"{url}: {error}")
            if queue and stored < max_pages:
                time.sleep(CRAWL_DELAY_SECONDS)
    finally:
        engine.close()
    return {"downloaded": stored, "visited": len(visited), "errors": errors}


class LinkParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(urllib.parse.urljoin(self.base_url, href))
