# BGE hidden-test binary classification report

## Status

**COMPLETED with real BGE inference on all 10 hidden grants.**

The professor's expected hidden labels were not included, so hidden-test accuracy, precision, recall, and F1 are not yet available. The table below contains predictions, not verified outcomes.

## Method

- Model: `BAAI/bge-small-en-v1.5` only; no MiniLM inference was used.
- Model input fields: `Program`, `Type`, and `Core Requirements` only.
- Deadlines and current status were ignored exactly as instructed.
- BGE's retrieval instruction was applied to the mission query only.
- Embeddings were normalized and scored with normalized dot product (cosine similarity).
- Binary target used for calibration: synthetic Train row is Relevant when `Eligibility label == Eligible` and `Thematic fit` is Excellent or Strong. Availability was ignored.
- Calibration set: 160 synthetic Train rows, including 72 positive rows.
- Frozen decision threshold: **0.518063154**. Score at or above the threshold maps to Relevant/Good; lower maps to Not relevant/Bad.
- Threshold selection maximized positive-class F1 on the synthetic Train split, with accuracy, precision, and then the higher threshold as deterministic tie-breakers.

## Synthetic Train calibration metrics

These are development/calibration metrics, not hidden-test metrics.

| Accuracy | Precision | Recall | F1 | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.468750 | 0.458599 | 1.000000 | 0.628821 | 72 | 85 | 3 | 0 |

## Hidden predictions

BGE classified 6 of 10 grants as Relevant/Good and 4 as Not relevant/Bad.

| ID | Program | BGE score | Binary class | Submission label |
|---:|---|---:|---|---|
| 1 | Cisco Global Problem Solver Challenge 2027 | 0.637993 | Good | Relevant |
| 2 | Women's Fund Armenia - Small Grants for Women-Led Initiatives in Aragatsotn | 0.515913 | Bad | Not relevant |
| 3 | Tory Burch Foundation Fellows Program | 0.517910 | Bad | Not relevant |
| 4 | AWS Activate (Founders tier) | 0.529717 | Good | Relevant |
| 5 | Women Founders Network Fast Pitch 2027 | 0.539750 | Good | Relevant |
| 6 | IOM ELEVATE Business Grants | 0.568105 | Good | Relevant |
| 7 | TiE Women Global Pitch Competition 2026 | 0.481710 | Bad | Not relevant |
| 8 | D-Prize Global Competition | 0.618019 | Good | Relevant |
| 9 | People in Need - Grant Call for Armenian Civil Society Organisations | 0.517357 | Bad | Not relevant |
| 10 | Women Who Tech - Tech Startup Grant | 0.567728 | Good | Relevant |

## Interpretation and limitations

The binary output is produced solely by thresholding BGE semantic similarity. BGE is not a verified rules engine and may miss hard eligibility restrictions that appear in the requirements. Therefore, these predictions satisfy the requested BGE-only experiment but should not be treated as verified grant eligibility decisions.

The synthetic calibration was weak: accuracy was 0.468750, precision was 0.458599, and there were 85 false positives. This is evidence that a single BGE similarity threshold does not model the combined eligibility-and-fit decision reliably.

The threshold was calibrated on synthetic labels tied to a hypothetical profile. That calibration may not transfer to the professor's real hidden examples. Once expected labels are released, compute accuracy, precision, recall, and F1 with Relevant as the positive class and analyze every false positive and false negative.

## Reproducibility

- Python: `3.10.19`
- sentence-transformers: `5.7.0`
- transformers: `4.57.6`
- torch: `2.2.2`
- CPU encode/classify time, excluding model download/loading: `4.372658` seconds

```bash
python -m src.classify_hidden_bge \
  --hidden-xlsx Ardook_hidden_test_grants.xlsx \
  --synthetic data/raw/Synthetic_Ardook_ML_Ready_200.csv \
  --labels data/raw/Synthetic_Ardook_Ground_Truth_200.csv \
  --mission config/hypothetical_mission.txt \
  --output-dir results/ \
  --low-disk-mode
```
