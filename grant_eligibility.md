# Eligibility check: Ardook's profile vs. one grant text file

Pass 1 (FIND):  for each label, find the grant sentences that state that
                kind of requirement. Uses the trained classifier from
                models/first_pass.joblib if it has one for the label,
                otherwise zero-shot similarity to the queries.
Pass 2 (JUDGE): cross-encoder/nli-MiniLM2-L6-H768 reads each found
                sentence next to a statement about Ardook and scores
                entailment / contradiction / neutral.

Needs grant_utils.py in the same folder.


```python
# One-time setup: run this, then Kernel -> Restart. Skip it after that.
%pip install --user sentence-transformers scikit-learn
```


```python
# ---------------------------------------------------------------------------
# CELL 1: setup, profile, labels, functions.
# Re-run this cell after editing ARDOOK, LABELS or the thresholds.
# ---------------------------------------------------------------------------

# If the model download fails because the compute node has no internet,
# download the models once from a login node, then uncomment these two lines
# and point HF_HOME at that cache folder. Must come before the imports below.
# import os
# os.environ["HF_HOME"] = "/path/to/your/scratch/hf_cache"

from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from grant_utils import (
    EMBED_MODEL, QUERIES, embed, read_grant, split_sentences, zero_shot_scores,
)

# Ardook reference profile.
# PLACEHOLDER VALUES - replace these with the real fact sheet.
ARDOOK = {
    "country": "Armenia",
    "legal_type": "for-profit company",
    "stage": "early-stage startup",
    "founded": 2023,
    "women_led": True,
    "focus": "data and AI tools for small businesses",
}

# The three labels. "column" links each one to the trained classifier and
# the zero-shot queries in grant_utils.QUERIES.
LABELS = {
    "1) Location": {
        "column": "location",
        "claim": (
            f"A {ARDOOK['legal_type']} based in {ARDOOK['country']} "
            f"is eligible to apply."
        ),
    },
    "2) Business type / length of operation": {
        "column": "business_type",
        "claim": (
            f"A {ARDOOK['stage']} that is a {ARDOOK['legal_type']}, "
            f"founded in {ARDOOK['founded']}, is eligible to apply."
        ),
    },
    "3) Preferred audience": {
        "column": "audience",
        "claim": (
            f"A {'women-led ' if ARDOOK['women_led'] else ''}company working on "
            f"{ARDOOK['focus']} is the kind of applicant this grant is meant for."
        ),
    },
}

NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
FIRST_PASS_PATH = Path("models/first_pass.joblib")

TOP_K = 5        # most evidence sentences kept per label
MIN_SIM = 0.30   # zero-shot only: similarity below this = not about the label


def find_evidence(column, sentence_vecs):
    """Pass 1. Returns ([(sentence_index, score)], source), best first.
    Trained: score is the classifier's probability, kept if >= its threshold.
    Zero-shot: score is cosine similarity, kept if >= MIN_SIM."""
    if column in first_pass:
        entry = first_pass[column]
        scores = entry["clf"].predict_proba(sentence_vecs)[:, 1]
        cutoff, source = entry["threshold"], "trained"
    else:
        scores = zero_shot_scores(embedder, sentence_vecs, QUERIES[column])
        cutoff, source = MIN_SIM, "zero-shot"

    kept = [i for i in np.argsort(-scores) if scores[i] >= cutoff][:TOP_K]
    return [(int(i), float(scores[i])) for i in kept], source


def label_order(nli):
    """Read which output column is entailment / contradiction / neutral.
    Falls back to the order documented on the model card."""
    try:
        id2label = {
            int(i): name.lower()
            for i, name in nli.model.config.id2label.items()
        }
        if set(id2label.values()) == {"contradiction", "entailment", "neutral"}:
            return id2label
    except AttributeError:
        pass
    return {0: "contradiction", 1: "entailment", 2: "neutral"}


def judge(nli, id2label, evidence_sentences, claim):
    """Pass 2. Score each evidence sentence (premise) against the Ardook
    claim (hypothesis). Returns one dict of probabilities per sentence."""
    pairs = [(sentence, claim) for sentence in evidence_sentences]
    probs = nli.predict(pairs, apply_softmax=True)
    return [
        {id2label[j]: float(row[j]) for j in range(len(row))}
        for row in probs
    ]


def verdict(scores):
    """Combine sentence-level scores into one verdict for the label.
    The 0-1 'likelihood' is a rough signal, NOT a calibrated probability."""
    top_entail = max(s["entailment"] for s in scores)
    top_contra = max(s["contradiction"] for s in scores)
    likelihood = (top_entail - top_contra + 1) / 2

    if top_contra > 0.5 and top_contra > top_entail:
        return "likely NOT met", likelihood
    if top_entail > 0.5:
        return "likely met", likelihood
    return "unclear - needs a human check", likelihood
```


```python
# ---------------------------------------------------------------------------
# CELL 2: load the models and the trained first pass.
# Run once; re-run only after retraining the first pass.
# ---------------------------------------------------------------------------
embedder = SentenceTransformer(EMBED_MODEL)
nli = CrossEncoder(NLI_MODEL)
id2label = label_order(nli)

first_pass = {}
if FIRST_PASS_PATH.exists():
    saved = joblib.load(FIRST_PASS_PATH)
    if saved["embed_model"] != EMBED_MODEL:
        print(f"Warning: {FIRST_PASS_PATH} was trained on {saved['embed_model']}, "
              f"not {EMBED_MODEL}. Ignoring it.")
    else:
        first_pass = saved["labels"]

for spec in LABELS.values():
    col = spec["column"]
    print(f"{col:<14} first pass: {'trained' if col in first_pass else 'zero-shot'}")
```


    Loading weights:   0%|          | 0/103 [00:00<?, ?it/s]


    Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.



    config.json:   0%|          | 0.00/875 [00:00<?, ?B/s]



    model.safetensors: reconstructing file:   0%|          |  0.00B /  328MB            



    model.safetensors: downloading bytes:           |  0.00B            



    Loading weights:   0%|          | 0/105 [00:00<?, ?it/s]



    tokenizer_config.json:   0%|          | 0.00/330 [00:00<?, ?B/s]



    vocab.json:   0%|          | 0.00/798k [00:00<?, ?B/s]



    merges.txt:   0%|          | 0.00/456k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/1.36M [00:00<?, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/239 [00:00<?, ?B/s]


    location       first pass: trained
    business_type  first pass: trained
    audience       first pass: trained



```python
# ---------------------------------------------------------------------------
# CELL 3: load one grant. Change GRANT_PATH and re-run for each grant.
# ---------------------------------------------------------------------------
GRANT_PATH = "grants/grantTest.txt"

sentences = split_sentences(read_grant(GRANT_PATH))
sentence_vecs = embed(embedder, sentences)
print(f"{len(sentences)} sentences found in {GRANT_PATH}")
```

    88 sentences found in grants/grantTest.txt



```python
# ---------------------------------------------------------------------------
# CELL 4: check each eligibility label. Re-run after changing Cell 1 or 3.
# ---------------------------------------------------------------------------
for label, spec in LABELS.items():
    print("\n" + "=" * 70)
    print(label)
    print("Ardook claim:", spec["claim"])

    evidence, source = find_evidence(spec["column"], sentence_vecs)
    if not evidence:
        print(f"Verdict: not mentioned ({source} first pass found nothing; "
              f"may mean no restriction - verify)")
        continue

    evidence_text = [sentences[i] for i, _ in evidence]
    scores = judge(nli, id2label, evidence_text, spec["claim"])
    result, likelihood = verdict(scores)

    print(f"Verdict: {result}   (rough likelihood {likelihood:.2f})")
    print(f"Evidence found by: {source} first pass")
    for (i, score), s in zip(evidence, scores):
        print(
            f'  - "{sentences[i]}"\n'
            f"    pass-1 score {score:.2f} | entail {s['entailment']:.2f} "
            f"| contradict {s['contradiction']:.2f} "
            f"| neutral {s['neutral']:.2f}"
        )
```

    
    ======================================================================
    1) Location
    Ardook claim: A for-profit company based in Armenia is eligible to apply.
    Verdict: unclear - needs a human check   (rough likelihood 0.41)
    Evidence found by: trained first pass
      - "Eligible associated countries: Albania, Armenia, Bosnia and Herzegovina, Egypt, Faroe"
        pass-1 score 0.68 | entail 0.00 | contradict 0.01 | neutral 0.99
      - "3.Established country: the applicant must be established in an EU Member State"
        pass-1 score 0.67 | entail 0.00 | contradict 0.06 | neutral 0.94
      - "Eligible EU countries: Austria, Belgium, Bulgaria, Croatia, Republic of Cyprus, Czech"
        pass-1 score 0.61 | entail 0.00 | contradict 0.17 | neutral 0.83
      - "Latvia, Lithuania, Luxembourg, Malta, Netherlands, Poland, Portugal, Romania, Slovakia,"
        pass-1 score 0.61 | entail 0.00 | contradict 0.18 | neutral 0.82
      - "Islands, Georgia, Iceland, Israel, Kosovo, Moldova, Montenegro, North Macedonia,"
        pass-1 score 0.56 | entail 0.00 | contradict 0.05 | neutral 0.95
    
    ======================================================================
    2) Business type / length of operation
    Ardook claim: A early-stage startup that is a for-profit company, founded in 2023, is eligible to apply.
    Verdict: likely NOT met   (rough likelihood 0.22)
    Evidence found by: trained first pass
      - "The legal entity must be registered for a minimum of six months and a"
        pass-1 score 0.67 | entail 0.00 | contradict 0.05 | neutral 0.95
      - "The founder or co-founder of the company must"
        pass-1 score 0.63 | entail 0.00 | contradict 0.01 | neutral 0.99
      - "completing the CAP table if the founder or investor is a holding company owned"
        pass-1 score 0.59 | entail 0.00 | contradict 0.56 | neutral 0.43
      - "Under Horizon Europe, a 'startup' should be understood as an SME in the"
        pass-1 score 0.55 | entail 0.00 | contradict 0.10 | neutral 0.89
      - "We consider a woman co-founder if she acquired shares"
        pass-1 score 0.54 | entail 0.00 | contradict 0.03 | neutral 0.96
    
    ======================================================================
    3) Preferred audience
    Ardook claim: A women-led company working on data and AI tools for small businesses is the kind of applicant this grant is meant for.
    Verdict: unclear - needs a human check   (rough likelihood 0.49)
    Evidence found by: trained first pass
      - "Women TechEU targets early-stage women-led deep tech startups."
        pass-1 score 0.75 | entail 0.01 | contradict 0.01 | neutral 0.99
      - "startup which is adopting industry standard technologies or modifying content,"
        pass-1 score 0.74 | entail 0.00 | contradict 0.01 | neutral 0.99
      - "Using technology and being a deep tech company are very different concepts."
        pass-1 score 0.71 | entail 0.00 | contradict 0.02 | neutral 0.98
      - "Startups are before reaching a high fidelity Minimum Viable Product,"
        pass-1 score 0.68 | entail 0.00 | contradict 0.01 | neutral 0.99
      - "Deep tech: Deep tech innovation aims to provide concrete solutions to our"
        pass-1 score 0.65 | entail 0.00 | contradict 0.01 | neutral 0.99



```python

```
