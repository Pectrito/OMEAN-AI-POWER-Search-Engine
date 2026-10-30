# Omean Search — Omean-v1

Omean is a local-first search prototype. It indexes documents on your computer,
searches them with BM25 and SQLite full-text search, and presents result
summaries. Search data stays local; the app does not send your documents to a
remote search service.

## Omean-v1 model

The published checkpoint is `data/multitask-search-model.pt`.

- **Trainable parameters:** 1,000,061
- **Model:** compact Transformer encoder/decoder with a relevance-ranking head
  and a summary-generation head
- **Training data:** 20,000 sampled articles from one English Wikipedia shard
- **Training steps:** 5,000
- **Validation ranking accuracy:** 53.91% on the project’s automatically
  generated validation examples

This is an experimental, weakly supervised model—not a Google-scale search
engine or a general-purpose chatbot. Ranking and generated summaries can be
wrong; check important claims against the source documents. Search remains
available with BM25 and extractive summaries when neural inference is
unavailable.

The app itself uses Python's standard library. Neural inference is optional;
install a PyTorch build compatible with your Python version to load the
checkpoint:

```powershell
python -m pip install torch
```

## Run the app

Requirements: Python 3.10 or newer. Clone with Git LFS enabled to retrieve the
published search index and model files:

```powershell
git lfs install
git clone https://github.com/gammabyblackskyofficial1/OMEAN-AI-POWER-Search-Engine.git
cd OMEAN-AI-POWER-Search-Engine
python app.py
```

If you use GitHub's **Download ZIP** instead, Git LFS files may be placeholder
pointers rather than the full SQLite database. The app detects the pointer,
creates a fresh local index, and continues. To get the prebuilt index, use the
Git LFS clone commands above.

Open <http://127.0.0.1:8000>. By default, the app indexes documents under
`data/`. Supported file types include `.txt`, `.md`, `.rst`, `.csv`, `.json`,
`.html`, and `.htm`. Keep the server on localhost unless you add suitable
authentication and network protections.

The starter `data/` folder contains four searchable Markdown documents. Startup
reports the number of local documents separately from the total index (which
may include the prebuilt Wikipedia corpus). The web page also shows the local
document count, including after a search. A query can still return zero
matches; that is the number of matching results, not the number of indexed
documents. Add or edit supported files in `data/` to grow your collection.

## Download and import the Wikipedia data

The original dataset shard is not included in the repository. Download this
file (about 401 MiB):

[Download the English Wikipedia shard](https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.en/train-00000-of-00041.parquet)

Save it as
`data/training/wikipedia-20231101-en-00000.parquet`, then install the importer
dependency and build or update the local search index:

```powershell
New-Item -ItemType Directory -Force .\data\training
python -m pip install -r .\requirements-data.txt
python .\import_wikipedia.py
```

The importer writes the SQLite index to `data/search-index.sqlite3`. The
repository also stores a prebuilt index with Git LFS; importing the shard is
optional if that index was downloaded successfully. The shard is one part of
the Wikipedia dataset, not the complete encyclopedia.

The source dataset and its attribution and reuse terms are documented in
[`training/NOTICE.txt`](training/NOTICE.txt). Keep that notice with
redistributed dataset-derived materials.

## maker and story 
It's make by om tayade. it's just 14 years child.on it development phase we have many bug and stop developing this but some things happened and we strat to make it.open-source to this model motive is that I want it free to people and i think i get best support from people we definitely make india own llm model in large-scale and the omean-v2 is coming-soon.
## support as
download this GitHub repos and use it 
# thank you #
