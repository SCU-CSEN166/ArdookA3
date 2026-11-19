"""
Shared pieces for the first pass (finding eligibility sentences).

Both notebooks import this file, so training and inference split and score
sentences in exactly the same way. Keep it in the same folder as the
notebooks.
"""

import re
from pathlib import Path

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# One column per eligibility label in the labeling CSV.
LABEL_COLUMNS = ["location", "business_type", "audience"]

# Zero-shot queries. Used to pre-fill the labeling sheet, as the fallback
# for any label without a trained model, and as the baseline the trained
# model has to beat.
QUERIES = {
    "location": [
        "Which countries or regions must applicants be located in?",
        "Applicants must be based in or registered in this country.",
    ],
    "business_type": [
        "What type of organization can apply, nonprofit or for-profit?",
        "How many years must the business have been operating?",
    ],
    "audience": [
        "Who is this grant intended for?",
        "Priority is given to women-led or underrepresented founders "
        "in a specific sector.",
    ],
}


def read_grant(path):
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def split_sentences(text, min_words=4):
    """Split grant text into sentences. Handles bullet lists by splitting on
    newlines first, then on sentence-ending punctuation."""
    sentences = []
    for line in text.splitlines():
        line = line.strip(" \t•-*")
        if not line:
            continue
        for s in re.split(r"(?<=[.!?])\s+", line):
            s = s.strip()
            if len(s.split()) >= min_words:
                sentences.append(s)
    return sentences


def embed(embedder, texts):
    """Embeddings as a numpy array (one row per text). They are normalized,
    so a dot product between two rows is their cosine similarity."""
    return embedder.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def zero_shot_scores(embedder, sentence_vecs, queries):
    """For each sentence, its best cosine similarity to any of the queries."""
    query_vecs = embed(embedder, queries)
    return (sentence_vecs @ query_vecs.T).max(axis=1)
