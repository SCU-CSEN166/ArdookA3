# CSEN 166 — Term Team Project for Social Impact (TTP4SI)

**Team:** [A3]  
**Challenge:** [Ardook]  
**Team members:** [Naomi Wiesen, Olivia Desire, David Lopez-Ryan]  
**Repository:** [https://github.com/SCU-CSEN166/ArdookA3]  
**Last updated:** [10/28/26]

> **How to use this README:** Maintain this single file throughout the project. Complete the Week 5 project plan first, then update the appropriate weekly section before each milestone. Be specific and honest: describe what you attempted, what worked, what did not work, and what evidence supports your conclusions. For every week, document each member's actual contribution. If a task was not completed, explain the blocker and next step. Do not include private or sensitive data in this public README.

## Remember - Best Practices

### Data access and privacy
[Explain where authorized data can be accessed. Do not commit personal information, private WhatsApp exports, API keys, credentials, or restricted grant/application records. Use synthetic or anonymized examples in the README.]

### Known limitations
[Keep a current list of what the system cannot reliably do.]

## Week 5 — Scope: Project Plan

**Deliverable during Week 6 Lab:** A project plan (write in this README file) describing your problem formulation, proposed approach, each team member's responsibilities, proposed baseline models with Hugging Face links, proposed evaluation metrics, proposed data with links, and goals for each week.

### 1. Problem formulation and social impact
- **Problem:** Creating an evaluation system to judge the recommendation value of a given grant opportunity. 
- **Intended users:** Ardook will hopefully be able to utilize this process to find useful grants more easily. 
- **Inputs and outputs:** What will the system receive and produce? Give one concrete example. The system will take in grant information files and output a value corresponding to how recommended the grant is.  
- **Success criteria:** What would make the system useful and reliable? Multiple faucets of ranking factors (e.g. splitting up how a grant is judged into multiple qualifiers) and being able to then output a ranked list of recommended grants to apply for. 
- **Constraints and risks:** [e.g., privacy, missing data, hallucinations, language coverage, human verification] : Given data is very small and limited, so its more than likely that synthetic data would need to be created (oversampling) to better train the model in some way. Also, there are certain features that can completely disqualify a grant in terms of eligibility, so there's a risk of the model being able to find a list of great grant opportunities that aren't actually real, or aren't eligible at all. 

### 2. Proposed technical approach (Info Retrieval style :D)
- **Pipeline:**
     - Create a similarly structured user profile for Ardook for comparison/evaluation stuff
     - One model that would filter through the data (spreadsheet) and gather relevant information
         - Train + test validation --> 80% train, 20% (10% validation and 10% test)
     - E.g. data extraction-y models to be used to format additional (and/or current) grant opportunities into a spreadsheet that can be treated as a corpus, possibly using SQL(?) or some other basic dictionary format for organizing the grant information
     - Second model takes in the data spreadsheet to develop a score + compare to the Ardook mission statement to see if the grants align with their values
     - Rank the score based on evaluation metrics:
            1) Eligibility (yes/no) , can be used to filter out invalid grants
                 -> Eligibility will be first checked based on the following:
                      a) Valid Location (e.g. Europe, US, country) 
                      b) Business type (start up vs established group)
                      c) Primary Audience (preferred recipient, e.g. women leaders, or topic-specific preferences) 
            2) Relevance, which for the baseline will likely be regarding the grant's purpose compared with Ardook's profile(?)
                      or:
                           - Based on thematic relevance, e.g. some form of relevance score based on the primary audience qualifier (separately calculated)
       
- **Baseline(s):** Manually extract key information from a small set of grants into a CSV or JSON file, including funding priorities, supported activities, eligible locations, applicant types, and deadlines. Use simple rules to exclude grants that Ardook is not eligible for and flag missing eligibility information for review. Use the pretrained all-MiniLM-L6-v2 model to compare each eligible grant's priorities and activities with Ardook's verified mission and funding needs, then rank grants by cosine similarity. No model training or synthetic data is needed for this baseline. Team members will review the grants and measure how many of the top five recommendations are relevant. This baseline will provide a comparison for the later extraction and fit-assessment models.
- **Proposed models/tools:** [Name, purpose, and link to each Hugging Face model or other resource.]
  - [Model name](https://huggingface.co/models) — [What it will do; why selected]
  - [All-mini LM] https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 - Sentence Similarity/transformer, 22.7M size
  - [Qwen3 VL-Embedding-8B] https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B - Sentence similarity
  - [BAAI] https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B
  - [Sim LM] https://huggingface.co/intfloat/simlm-msmarco-reranker - Text Ranking 
- **Alternative approach:** [What will you try if the baseline underperforms?] If the baseline underperforms, we would change the evaluation criteria to see if that helps with performance. In addition to that, we can also look at more synthetic data to see if that is effective too.

### 3. Proposed data
| Dataset/source | Link or access instructions | Purpose | Size / languages | Privacy or usage considerations |
|---|---|---|---|---|
| Ardook_Data_All_Grant_Recommendations_UPDATED_v_3.xlsx | [URL](https://drive.google.com/file/d/10EcaXrXUItPKAJwdqx6BTepwhGWMQIUn/view?usp=sharing) | Developing the baseline and evaluate recommendations using existing fit ranges | 37 opportunities, 13 columns, English | Provided from instructor.  |

- **Data preparation:** Convert to CSV, standardize fit labels, check missing values and duplicates, seperate development and evaluation examples.
- **Data limitations:** Small dataset, conditional fit labels, and potentially outdated deadlines/status. Need human verification. 

### 4. Proposed evaluation
| Metric / test | What it measures | How it will be calculated | Target / success criterion |
|---|---|---|---|
| [Metric - e.g Accuracy, F1, AoC] | [Quality dimension] | [Method/ Formula] | [Target] |
| Accuracy | This metric will measure how close the data input is to the expected score | (TP + TN)/(TP + TN + FP + FN)| Above 60% accuracy to start, and will increase with future implementations|
| Precision | This metric will be the ability to filter to find the good grants |TP/(TP + FP)| Above 60% correct filter identification of grants for baseline success |
| Recall | This metric will look at how many of these good grants are actually viable | TP/(TP + FN)| Above 60% success when identifying viable grants for the baseline |


- **Evaluation examples:** [How will you build or select a development set?]
    - Compare the data to the expected score to see if it was accurate. For example, when it comes to determining the validity of grants. We can use binary classification to determine if they are valid or not. If they are valid, the grant will be added to the filtered list to check the relevance to Ardook
- **Human verification:** [If relevant, who will check outputs and how?]
    - One or two teammates can ckeck if the outputs are relevant based on the keywords written in the Ardook mission statement. Based on the similarity between the grant and the mission statement, we can have a treshold to determine how well they match to each other.
- **Failure cases to watch:** [At least two anticipated risks]
    - An anticipated risk to look out for is having an uneven balance in terms of which grants are valid and which ones are not. This could lead to overfitting if the amount of valid grants are small.
    - Another risk to watch out for is with the relevance check in terms of matching word by word to the Ardook profile instead of the grant's overall relevance.

### 5. Team responsibilities
| Member | Primary responsibility | Specific Weekly contribution | Planned next task |
|---|---|---|---|
| [Name] | [Data / models / evaluation / integration / etc.] | [Actual work completed] | [Next action] |
| Naomi |Explore models that relate to sentence similarity|Developed the problem formulation in the README|Transform the data into a value to help with ranking|
| Olivia |Analyzing ways to extract data so that it's more readable + searching for text classification models|Worked on the pipeline and alternative approach in the README |Looking for existing models that transform data in a dictionary-like structure|
| David |Developing Synthetic Data to add to the spreadsheet |Contributed to the Baseline and Proposed Data in the README|Normalizing the data|
---

## Week 6 — Build: Working Baseline

**Deliverable during Week 7 Lab:** A runnable baseline, setup/run instructions, and 1–2 example inputs and outputs. Update this section with what worked, what failed, and individual contributions.

### Implementation status
- **What runs end-to-end now?** [Describe exactly.]
       - We currently have the inputs from the Ardook spreadsheet and were able to use fit and the text prompt to decipher the relevance score values.
- **How to run it:** [Commands, dependencies, configuration; refer to Setup below.]

- **Model(s) and data actually used:** [Names, versions, links, and any changes from the plan.]
    - Used SimLM model to help with generating the relevance score: https://huggingface.co/intfloat/simlm-msmarco-reranker
        - Initially, we didn't have a clear criteria for the relevance scoring. However, now we are evaluating the relevance score based on the Ardook profile we created and some eligibility criteria including location, business type, and the intended audience.
    - 
- **Example 1 (SimLM):** Input: bat_di = encode(tokenizer, 'List the scores of the grant available to women entrepreneurs based on the fit', str(row[TEXT_COL]), str(row['Fit'])) → Output: see screenshot below (look at TTP4SI_SimLM_model.ipynb for the full outputs)
<img width="137" height="235" alt="Screenshot 2026-11-10 at 7 25 37 PM" src="https://github.com/user-attachments/assets/5ccd526b-7b00-4b22-9f50-57e7baeee44a" />

  
- **Example 2:** Input: [non-sensitive example] → Output: [result]

### Attempts and outcomes
| Attempt / change | Worked? (Yes / Partly / No) | Evidence (output, metric, link) | What we learned / next step |
|---|---|---|---|
| [Attempt] | [Status] | [Evidence] | [Learning] |
|Attempted to connect a score to the Ardook data |Partly, presents an output but all the scores are negative though| https://colab.research.google.com/drive/10tfed4YgQFChgk21SibdX6NMZlkokp6t?usp=sharing |I learned that since the data is coming from the same source there could be underfitting due to the familarity in score values.|

### Individual contributions during that week
| Member | What I personally implemented/tested | Result (worked / partly / failed) | Evidence (commit, PR, experiment) | Blocker / next step |
|---|---|---|---|---|
| [Name] | [Specific contribution] | [Status] | [Link] | [Next step] |
| Olivia | I tried to implement relevance scoring based on the data in the ardook spreadsheet | partly |https://colab.research.google.com/drive/10tfed4YgQFChgk21SibdX6NMZlkokp6t?usp=sharing|Continue to work on implentation specifically with output scores. |
| Naomi | Attempted to check against core requirements in the spreadsheet against Ardook's profile via sentence similarity to try and get a relevant score | dropped/switching to different strategy | See Testing Models.ipynb | Changing strategy- use model to identify some set of eligibility-related statements |

## Week 7 — Evaluate: Results Checkpoint

**Deliverable during Week 8 Lab:** Preliminary evaluation results, metrics, development-set details, and 2–3 representative successes or failures. Include what you will improve next.

### Evaluation setup
- **Development data:** [Source, size, selection, annotations, link or access instructions]
- **Metrics and rationale:** [Why these metrics reflect the intended use]
- **Comparison:** [Baseline versus current approach, if different]
  - Compared to Week 6, the Week 7 code includes...
    - SimLM model: has normalized outputs for the scores so the values are between 0 and 1 rather than between negative infinity and positive infinity, model predictions in the form of percentage values, and evaluation metrics such as accuracy, precision, and recall too.

| System / version | Metric 1 (Accuracy) | Metric 2 (Precision)| Metric 3 (Recall) | Notes |
|---|---:|---:|---:|---|
| SimLM Baseline | 62.16% | 30%| 30% | The accuracy was above the 60% threshold so it was successful for the baseline. However, the precision and recall values were on the lower side. This implies that model's strength is identifying similar keywords, but when faced with comparing it to the Ardook profile that is where the model struggles to find matching grants.|
| Baseline | [Value] | [Value] | [Value] | [Notes] |
| Current | [Value] | [Value] | [Value] | [Notes] |

### Successes, failures, and interpretation
| Example / category | Expected behavior | Actual behavior | Success or failure? | Likely reason |
|---|---|---|---|---|
| [Case] | [Expected] | [Observed] | [Status] | [Hypothesis] |
| [Case] | [Expected] | [Observed] | [Status] | [Hypothesis] |
| [Case] | [Expected] | [Observed] | [Status] | [Hypothesis] |

- **What improved?** [Evidence]
- **What did not improve?** [Evidence]
- **Next experiment:** [Specific change and how you will measure it]

### Individual contributions during that week
| Member | Work completed | Result (worked / partly / failed) | Evidence | Next step |
|---|---|---|---|---|
| Olivia | I worked on normalizing the score values from last week, created prediction values for the model, and developed the evaluation metrics as well. | partially, the accuracy was above the goal threshold but precision and recall were not| https://colab.research.google.com/drive/17F2tcFgMOSILFkfg9LDv9CIuKJQ7tjf9?usp=sharing|Combine the results from the summarization model into the input of the SimLM model to create a single baseline. As well as, look into ways to improve the precision and recall outputs too.|
| [Name] | [Specific work] | [Status] | [Link] | [Next step] |
| [Name] | [Specific work] | [Status] | [Link] | [Next step] |

---

## Week 8 — Stress-Test: Reliability on Hidden Data

**Deliverable during Week 9 Lab:** Hidden-test results (as provided by the TA), analysis of at least three important failure cases, and a prioritized improvement plan.

- **Submitted system version / commit:** `experiment/bge-vs` / `454472f`
- **Hidden-test metrics and leaderboard result:** No TA-provided hidden dataset or leaderboard result was available for this run. The results below are from the predefined 40-row synthetic `Test` split and must not be described as hidden-test performance.
- **Difference from development results:** The 160 `Train` rows were not used for supervised training or query tuning. Both pretrained models were evaluated once on the same fixed test records, mission query, candidate pools, and ranking code.

### Held-out synthetic stress test

The experiment compared `sentence-transformers/all-MiniLM-L6-v2` with
`BAAI/bge-small-en-v1.5`. Only `Program`, `Type`, and `Core Requirements` were
embedded, preventing outcome labels and human-written fit conclusions from
leaking into model input. The primary pool used an oracle filter requiring the
ground-truth labels `Eligible` and `Open`; this left 18 candidates from the 40
test rows and isolated thematic ranking from eligibility checking.

| Model | NDCG@5 | Precision@5 | Spearman | CPU encode/rank time |
|---|---:|---:|---:|---:|
| all-MiniLM-L6-v2 | 0.828410 | 1.000000 | 0.581156 | 0.883772 s |
| bge-small-en-v1.5 | 0.587034 | 0.600000 | 0.077488 | 1.373303 s |

MiniLM won this synthetic held-out comparison by **0.241376 NDCG@5** and
**0.400000 Precision@5**. MiniLM placed five Excellent/Strong grants in its
primary top five, while BGE placed three. Raw cosine scores were not compared
across model families because their score scales are not directly meaningful.

| Failure case / category | What went wrong | Possible cause | Severity / user impact | Proposed fix |
|---|---|---|---|---|
| Eligibility and availability | Without the oracle filter, each model's top five contained two Ineligible grants and one non-open grant. | Semantic similarity measures thematic language, not hard constraints such as geography, applicant type, or deadline status. | High: a user could spend time on an unavailable or disallowed grant. | Keep eligibility/availability as a separate verified rules or extraction stage before relevance ranking. |
| BGE thematic ordering | BGE placed a Moderate and a Weak grant in the primary top five, lowering Precision@5 to 0.60. | The short, requirement-heavy permitted text may not contain enough neutral purpose detail; the fixed hypothetical query may also favor MiniLM's representation. | Medium: relevant grants can be displaced near the top of the review queue. | Test richer non-leaking purpose descriptions and have humans review top-ranked results. Do not tune on the held-out labels. |
| Large model disagreement | For example, an Excellent language-access AI grant ranked #10 with MiniLM but #33 with BGE; other records moved more than 20 positions. | Embedding models emphasize different words and semantic relationships, especially when descriptions are sparse. | Medium: recommendations are unstable across reasonable model choices. | Add human relevance judgments, report rank uncertainty, and evaluate on an independently labeled real-world set. |

### Changes attempted after stress-testing

| Change | Worked? (Yes / Partly / No) | Evidence | Keep, revise, or revert? |
|---|---|---|---|
| Strict three-field input allowlist | Yes | Twelve unit tests passed, including checks that forbidden label/status fields never enter model text. | Keep; expand only with neutral fields reviewed for leakage. |
| Same fixed query, test split, candidate pools, and metric code | Yes | Both models ranked the identical 18-record primary pool and identical 40-record diagnostic pool. | Keep for fair comparisons. |
| BGE retrieval instruction on the query only | Partly | It followed the BGE model guidance, but BGE still underperformed MiniLM on NDCG@5 and Precision@5. | Keep as the documented BGE configuration; do not select BGE based on this run. |
| Unfiltered diagnostic in addition to oracle-filtered ranking | Yes | It exposed two ineligible and one non-open top-five result for each model. | Keep; use it to demonstrate why a separate eligibility stage is required. |

### Individual contributions during that week

| Member | Work completed | Result (worked / partly / failed) | Evidence | Next step |
|---|---|---|---|---|
| David | Implemented and ran the reproducible MiniLM-versus-BGE comparison; validated and joined the 200-row synthetic data by Grant ID; prevented label leakage; calculated NDCG@5, Precision@5, and Spearman; exported rankings; analyzed failures; added 12 tests and documentation. | Worked for the synthetic held-out evaluation; real-world accuracy remains unverified. | [`results/model_comparison_report.md`](results/model_comparison_report.md), [`results/model_comparison_metrics.csv`](results/model_comparison_metrics.csv), and [`tests/test_grant_comparison.py`](tests/test_grant_comparison.py) | Build a separately verified eligibility stage and obtain independent human labels for real-grant evaluation. |

---

## Week 9 — Responsible AI, Peer Testing, and Final Delivery

**Deliverable during Week 10 Lab:** Document peer-testing feedback, Responsible AI concerns, changes made in response, and the final runnable system. Submit the final project report as instructed.

### Peer testing
- **Team evaluated:** [Team ID]
- **One thing that worked well:** [Evidence]
- **One meaningful failure:** [Reproduction steps or example]
- **One actionable recommendation:** [Specific fix]
- **Feedback received from another team:** [Summary]

### Responsible AI assessment
- **Potential harms or affected users:** [Who could be affected and how?]
- **Privacy and data handling:** [How sensitive information is protected]
- **Reliability and unsupported claims:** [How outputs are verified or uncertainty is communicated]
- **Fairness, accessibility, and usability:** [Relevant concerns and tests]
- **Human oversight:** [Where a human should review or correct outputs]
- **Remaining limitations:** [What should not be trusted or deployed yet]

### Final improvements and outcomes
| Feedback / issue | Change made | Worked? (Yes / Partly / No) | Evidence | Remaining limitation |
|---|---|---|---|---|
| [Issue] | [Change] | [Status] | [Link or metric] | [Limitation] |

### Individual contributions during that week
| Member | Final implementation / evaluation / documentation contribution | Result (worked / partly / failed) | Evidence | Reflection / remaining work |
|---|---|---|---|---|
| [Name] | [Specific work] | [Status] | [Link] | [Reflection] |



### Final submission checklist
- [ ] Code and final system/prototype are in the repository
- [ ] Setup and run instructions are complete and tested
- [ ] Data sources, model versions, and dependencies are documented
- [ ] Evaluation results are reported and compared with the baseline
- [ ] Important failures and limitations are explained
- [ ] Hidden-test and peer-feedback findings are addressed
- [ ] Responsible AI considerations and human oversight are discussed
- [ ] Every team member's contributions are documented for every week

**Final commit / release:** [Commit hash or tag]

---

## Setup and Reproducibility (keep updated)

### Requirements

Python 3.10–3.11, CPU by default, and the pinned packages in
`requirements.txt`. First-time execution requires network access to download the
two Hugging Face models.

### Installation

```bash
/usr/local/bin/python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Run the system

```bash
python -m src.compare_grant_embeddings \
  --synthetic data/raw/Synthetic_Ardook_ML_Ready_200.csv \
  --labels data/raw/Synthetic_Ardook_Ground_Truth_200.csv \
  --original data/raw/Original_Ardook_Grant_Tracker_37.csv \
  --mission config/hypothetical_mission.txt \
  --output-dir results/
```

### Run evaluation

```bash
python -m pytest -q
```

### Repository structure
```text
project/
├── README.md
├── config/
├── data/raw/          # local-only inputs; ignored by Git
├── requirements.txt
├── results/
├── src/
├── tests/
└── docs/
```

## MiniLM versus BGE grant-relevance experiment

This repository now includes a reproducible, CPU-first comparison of
`sentence-transformers/all-MiniLM-L6-v2` and `BAAI/bge-small-en-v1.5`. It ranks
grants for thematic relevance to a configurable **hypothetical** Ardook-like
profile. It does not predict eligibility. The primary held-out comparison first
uses an oracle filter (`Eligible` and `Open`) and then evaluates ranking quality
with NDCG@5 and Precision@5. A separate unfiltered ranking demonstrates why
semantic similarity cannot enforce hard grant constraints.

To prevent label leakage, model text is built only from `Program`, `Type`, and
`Core Requirements`. Labels, status, fit explanations, recommended actions,
scenario metadata, and official-source fields are never embedded. This strict
allowlist also limits how much neutral purpose information the models can use.

The instructor-provided original tracker and local experiment CSVs are ignored
by Git. The derived `results/original_grant_rankings.csv` is also local-only by
default because it reproduces source descriptions. Copy authorized files into
`data/raw/` without committing them:

```bash
cp /path/to/Synthetic_Ardook_ML_Ready_200.csv data/raw/
cp /path/to/Synthetic_Ardook_Ground_Truth_200.csv data/raw/
cp /path/to/Original_Ardook_Grant_Tracker_37.csv data/raw/
```

Use Python 3.10–3.11 (3.10 is tested and shown here), create an isolated environment, and
run the tests before downloading either model:

```bash
/usr/local/bin/python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

Run the fixed primary experiment (BGE's retrieval instruction is applied to its
query only and is enabled by default):

```bash
python -m src.compare_grant_embeddings \
  --synthetic data/raw/Synthetic_Ardook_ML_Ready_200.csv \
  --labels data/raw/Synthetic_Ardook_Ground_Truth_200.csv \
  --original data/raw/Original_Ardook_Grant_Tracker_37.csv \
  --mission config/hypothetical_mission.txt \
  --output-dir results/
```

On a machine with very little free disk space, append `--low-disk-mode`. This
uses a task-created temporary cache for one model at a time and removes that
cache after its scores have been retained; later runs must download the models
again.

The command validates IDs, schemas, expected label counts, missing values, and
the predefined 160/40 split before inference. It writes metrics, filtered and
unfiltered synthetic rankings, original-data demo rankings, and a detailed
report under `results/`. Model loading/download time is excluded from the
reported encode-and-rank runtime. Results are meaningful only for this synthetic
profile and label design; real deadlines, eligibility, and organizational facts
require official-source and human verification.

### Measured result

| Model | NDCG@5 | Precision@5 | Spearman |
|---|---:|---:|---:|
| all-MiniLM-L6-v2 | 0.828410 | 1.000000 | 0.581156 |
| bge-small-en-v1.5 | 0.587034 | 0.600000 | 0.077488 |

On the 18-candidate oracle-filtered test pool, MiniLM performed better by
0.241376 NDCG@5 and 0.400000 Precision@5. This is a synthetic-only result, not
proof of real-world performance. See the
[full report](results/model_comparison_report.md) and
[Week 8 presentation notes](docs/week8_presentation_notes.md).


