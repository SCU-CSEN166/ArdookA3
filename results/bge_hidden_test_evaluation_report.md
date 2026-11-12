# BGE hidden-test evaluation against released labels

## Verified metrics

Good is the positive class.

| Accuracy | Precision | Recall | F1 | Specificity | NPV |
|---:|---:|---:|---:|---:|---:|
| 0.700000 | 0.666667 | 0.800000 | 0.727273 | 0.600000 | 0.750000 |

## Confusion matrix

| Actual / Predicted | Good | Bad |
|---|---:|---:|
| Good | 4 | 1 |
| Bad | 2 | 3 |

The model correctly classified 7 of 10 grants. It produced 2 false positives and 1 false negative.

## Errors

- Grant 6 — IOM ELEVATE Business Grants: False positive; predicted Good, actual Bad. Released reason: Only for SMEs in specific Armenian border regions
- Grant 8 — D-Prize Global Competition: False positive; predicted Good, actual Bad. Released reason: For new ventures distributing proven poverty interventions, not AI/data products
- Grant 7 — TiE Women Global Pitch Competition 2026: False negative; predicted Bad, actual Good. Released reason: Woman-founded, early-stage startup with a validated offering; equity-free prize

## Interpretation

BGE recovered 4 of the 5 truly Good grants, but two grants with hard eligibility or program-purpose conflicts were incorrectly labeled Good. This confirms the earlier concern that semantic similarity can detect topical alignment while missing categorical eligibility restrictions.

The false negative was the TiE Women competition, which the released labels mark Good because it fits a woman-founded early-stage startup with a validated offering. Its BGE score fell below the synthetic calibration threshold, illustrating imperfect transfer from synthetic training examples to real hidden grants.
