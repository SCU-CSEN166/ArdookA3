# MiniLM versus BGE grant-relevance experiment

## Status and conclusion

**COMPLETED with real local inference.** MiniLM ranked better on the primary synthetic metric by 0.241376 NDCG@5.
This synthetic-only comparison does not establish real-world grant recommendation performance.

## Data and assumptions

- Synthetic rows: 200 (160 predefined Train, 40 predefined Test).
- Eligibility counts: 120 Eligible, 60 Ineligible, and 20 Needs review; validation found no duplicate IDs or missing required values.
- Primary candidate pool: 18 held-out grants after an **oracle eligibility/availability filter** requiring ground-truth `Eligible` and `Open`.
- Original demonstration rows: 37; their `Fit` and status values are unverified, mixed/conditional, and not accuracy ground truth.
- Hypothetical evaluation query: “An early-stage AI and data analytics venture developing responsible technology for social impact, economic opportunity, and underserved communities.” This is a fictional assessment profile, not a verified statement about Ardook.
- Model input is restricted to `Program`, `Type`, and `Core Requirements`, joined with field labels. No outcome, rationale, status, split, source, or human-conclusion field is embedded.
- The allowlist prevents label leakage but also omits potentially useful neutral purpose detail; rankings may therefore reflect sparse or eligibility-oriented descriptions.
- BGE query-only retrieval instruction used: **True**. It was not added to documents or MiniLM.
- Ground truth was joined by `Grant ID`, with unique-ID, schema, missing-field, expected-count, and shared-column agreement checks.

## Primary held-out metrics

Excellent=3, Strong=2, Moderate=1, Weak=0. Precision@5 treats Excellent/Strong as relevant. Raw cosine values are not compared across model families.

| Model | Parameters | Test rows | Eligible/open | NDCG@5 | Precision@5 | Spearman | Inference seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sentence-transformers/all-MiniLM-L6-v2` | 22713216 | 40 | 18 | 0.828410 | 1.000000 | 0.581156 | 0.883772 |
| `BAAI/bge-small-en-v1.5` | 33360000 | 40 | 18 | 0.587034 | 0.600000 | 0.077488 | 1.373303 |

### Primary top five

**minilm:**

- SYN-0054 — Starling Field Lab — Discovery AI For Economic Mobility Innovation Sprint 2027 — Excellent (score 0.544839)
- SYN-0142 — Skyline Meadow Collaborative — Trustworthy AI Product Development Award — Strong (score 0.503714)
- SYN-0191 — Amber Current Collaborative — Early-Stage Digital Innovation Opportunity Fund — Strong (score 0.501857)
- SYN-0199 — Ridgewater Innovation Fund — Elevate Small-Business Opportunity Analytics Seed Fund — Excellent (score 0.448756)
- SYN-0111 — ClearTide Trust — NextWave Data-Driven Social Enterprises Pitch Competition 2027 — Strong (score 0.446979)

**bge:**

- SYN-0191 — Amber Current Collaborative — Early-Stage Digital Innovation Opportunity Fund — Strong (score 0.695831)
- SYN-0142 — Skyline Meadow Collaborative — Trustworthy AI Product Development Award — Strong (score 0.664294)
- SYN-0054 — Starling Field Lab — Discovery AI For Economic Mobility Innovation Sprint 2027 — Excellent (score 0.643885)
- SYN-0059 — Tandem Bay Collaborative — Elevate Ventures Pitch Competition — Moderate (score 0.598652)
- SYN-0103 — Cirrus Point Lab — Sustainable Fashion Commerce Award — Weak (score 0.598632)

## Unfiltered diagnostic (not eligibility classification)

All 40 test rows were ranked without the oracle filter. Counts below show constraint violations in each top five; they are not classifier accuracy.

| Model | Top-5 Ineligible | Top-5 non-open |
|---|---:|---:|
| `sentence-transformers/all-MiniLM-L6-v2` | 2 | 1 |
| `BAAI/bge-small-en-v1.5` | 2 | 1 |

**minilm unfiltered top five:**

- SYN-0054 — Starling Field Lab — Discovery AI For Economic Mobility Innovation Sprint 2027 — eligibility=Eligible, availability=Open, fit=Excellent
- SYN-0146 — Aurora Path Initiative — Catalyst Startup AI Infrastructure Pilot Program — eligibility=Ineligible, availability=Closed, fit=Strong
- SYN-0142 — Skyline Meadow Collaborative — Trustworthy AI Product Development Award — eligibility=Eligible, availability=Open, fit=Strong
- SYN-0191 — Amber Current Collaborative — Early-Stage Digital Innovation Opportunity Fund — eligibility=Eligible, availability=Open, fit=Strong
- SYN-0116 — Newhaven Partnership — Accessible Data Literacy Award — eligibility=Ineligible, availability=Open, fit=Excellent

**bge unfiltered top five:**

- SYN-0191 — Amber Current Collaborative — Early-Stage Digital Innovation Opportunity Fund — eligibility=Eligible, availability=Open, fit=Strong
- SYN-0116 — Newhaven Partnership — Accessible Data Literacy Award — eligibility=Ineligible, availability=Open, fit=Excellent
- SYN-0142 — Skyline Meadow Collaborative — Trustworthy AI Product Development Award — eligibility=Eligible, availability=Open, fit=Strong
- SYN-0146 — Aurora Path Initiative — Catalyst Startup AI Infrastructure Pilot Program — eligibility=Ineligible, availability=Closed, fit=Strong
- SYN-0054 — Starling Field Lab — Discovery AI For Economic Mobility Innovation Sprint 2027 — eligibility=Eligible, availability=Open, fit=Excellent

## Real-data demonstration (no accuracy claim)

**minilm top five:**

- ORIG-0021 — Google.org GenAI / AI Social Impact Accelerator — unverified tracker fit: Excellent if matching call opens
- ORIG-0023 — Google.org Impact Challenge – AI for Science — unverified tracker fit: Moderate
- ORIG-0022 — Google.org Impact Challenge – AI for Government Innovation — unverified tracker fit: Moderate–Strong thematically
- ORIG-0029 — Rural AI Catalyst Fund — unverified tracker fit: Weak–Moderate
- ORIG-0034 — NetHope AI Accelerator — unverified tracker fit: Strong

**bge top five:**

- ORIG-0033 — Humanity AI funding/program — unverified tracker fit: Moderate
- ORIG-0022 — Google.org Impact Challenge – AI for Government Innovation — unverified tracker fit: Moderate–Strong thematically
- ORIG-0023 — Google.org Impact Challenge – AI for Science — unverified tracker fit: Moderate
- ORIG-0021 — Google.org GenAI / AI Social Impact Accelerator — unverified tracker fit: Excellent if matching call opens
- ORIG-0029 — Rural AI Catalyst Fund — unverified tracker fit: Weak–Moderate

## Error and disagreement analysis

Largest test-set rank disagreements (unfiltered pool):

- SYN-0103 — Cirrus Point Lab — Sustainable Fashion Commerce Award: MiniLM #39, BGE #13 (|Δ|=26); fit=Weak, eligibility=Eligible.
- SYN-0081 — Verdant North Trust — Language-Access AI Pilot Program 2027: MiniLM #10, BGE #33 (|Δ|=23); fit=Excellent, eligibility=Eligible.
- SYN-0131 — Oakstrand Collaborative — Open Digital Public Goods Prototype Award: MiniLM #29, BGE #6 (|Δ|=23); fit=Strong, eligibility=Eligible.
- SYN-0113 — Riverstone Foundation — Pathways Ventures Opportunity Fund: MiniLM #12, BGE #34 (|Δ|=22); fit=Strong, eligibility=Eligible.
- SYN-0133 — BlueJuniper Innovation Fund — Pathways Ventures Pilot Program: MiniLM #30, BGE #11 (|Δ|=19); fit=Strong, eligibility=Ineligible.

High-relevance but ineligible examples:

- SYN-0008 — Orchid Vale Trust — Fair Access Analytics Venture Fellowship: Excellent; MiniLM #18, BGE #19. This illustrates why semantic relevance cannot enforce eligibility.
- SYN-0075 — Wavelength Lab — Regional AI Commercialization Founder Prize 2027: Strong; MiniLM #19, BGE #8. This illustrates why semantic relevance cannot enforce eligibility.
- SYN-0087 — Cobalt Pine Initiative — Community Data Infrastructure Opportunity Fund 2027: Excellent; MiniLM #6, BGE #10. This illustrates why semantic relevance cannot enforce eligibility.
- SYN-0116 — Newhaven Partnership — Accessible Data Literacy Award: Excellent; MiniLM #5, BGE #2. This illustrates why semantic relevance cannot enforce eligibility.
- SYN-0120 — Juniper Trail Partnership — AI For Inclusive Employment Pilot Program 2027: Excellent; MiniLM #27, BGE #32. This illustrates why semantic relevance cannot enforce eligibility.

Because only three neutral source fields are permitted, some disagreements may arise from short or requirement-heavy descriptions rather than genuine mission mismatch. This is a qualitative hypothesis, not a measured causal explanation. Human review and richer, non-leaking purpose text are needed.

## Reproducibility

- Python: `3.10.19`
- Platform: `macOS-15.4.1-x86_64-i386-64bit`
- Device: `cpu`
- Ephemeral low-disk model cache: `True`
- sentence-transformers: `5.7.0`
- transformers: `4.57.6`
- torch: `2.2.2`
- numpy: `1.26.4`
- synthetic input: `data/raw/Synthetic_Ardook_ML_Ready_200.csv`
- ground truth: `data/raw/Synthetic_Ardook_Ground_Truth_200.csv`
- original input: `data/raw/Original_Ardook_Grant_Tracker_37.csv`

```bash
python -m src.compare_grant_embeddings \
  --synthetic data/raw/Synthetic_Ardook_ML_Ready_200.csv \
  --labels data/raw/Synthetic_Ardook_Ground_Truth_200.csv \
  --original data/raw/Original_Ardook_Grant_Tracker_37.csv \
  --mission config/hypothetical_mission.txt \
  --output-dir results/ \
  --low-disk-mode
```

Model download/loading time is excluded from reported inference seconds. The timed section includes normalized query/document encoding and deterministic ranking for the 40 synthetic test and 37 original records.

## Limitations

Synthetic labels encode a fictional profile and scenario design; they may reward dataset artifacts and do not prove benefit, fairness, current availability, or legal eligibility in practice. The oracle filter uses truth labels only to isolate thematic ranking and is not deployable as-is. The original tracker is small, possibly stale, and lacks independently verified labels. A human must verify official sources, deadlines, restrictions, and organizational facts before acting.
