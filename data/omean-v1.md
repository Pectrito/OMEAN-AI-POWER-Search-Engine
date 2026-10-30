# Omean-v1

Omean-v1 is a small local search and summarization prototype. It combines
SQLite full-text search and BM25 ranking with an optional trained neural
checkpoint.

The checkpoint contains 1,000,061 trainable parameters. It was trained with
weakly supervised examples derived from article titles and introductory text.
It is an educational experiment, not a general-purpose chatbot or a
Google-scale search engine. Search can fall back to BM25 and extractive
summaries when neural inference is unavailable.
