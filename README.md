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
      -  Download the files from github
     - When it comes to TTP4SI_SimLM_model, use Google Colab to run with CPU computing resources. Also make sure to change the file path, before running the code
     - <img width="845" height="83" alt="Screenshot 2026-11-12 at 1 57 00 PM" src="https://github.com/user-attachments/assets/938e43bc-7281-4337-95d3-73f553d71db1" />

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
- **Development data:** A 200-row synthetic grant dataset based on a hypothetical Ardook-like profile, with a predefined split of 160 `Train` and 40 `Test` rows. No supervised training or fine-tuning was performed for the MiniLM/BGE comparison.
- **Input controls:** The embedding experiment uses only `Program`, `Type`, and `Core Requirements`. Eligibility labels, thematic-fit labels, recommendations, status fields, split metadata, and human-written fit explanations are excluded to prevent label leakage.
- **Metrics and rationale:** NDCG@5 measures whether the highest-grade grants appear near the top in the correct order. Precision@5 measures how many of the top five are labeled Excellent or Strong. Spearman correlation measures overall agreement between similarity scores and thematic-fit grades.
- **Comparison:** The earlier SimLM work treated the problem more like similarity classification, using accuracy, precision, and recall. The new experiment treats thematic relevance as a ranking problem and compares `all-MiniLM-L6-v2` with `bge-small-en-v1.5` under identical conditions.

The existing SimLM checkpoint produced:

| System / version | Metric 1 (Accuracy) | Metric 2 (Precision)| Metric 3 (Recall) | Notes |
|---|---:|---:|---:|---|
| SimLM | 62.16% | 30%| 30% | The accuracy was above the 60% threshold so it was successful for the baseline. However, the precision and recall values were on the lower side. This implies that model's strength is identifying similar keywords, but when faced with comparing it to the Ardook profile that is where the model struggles to find matching grants.|

The embedding-ranking follow-up produced:

| Model | NDCG@5 | Precision@5 | Spearman | CPU encode/rank time |
|---|---:|---:|---:|---:|
| all-MiniLM-L6-v2 | 0.828410 | 1.000000 | 0.581156 | 0.883772 s |
| bge-small-en-v1.5 | 0.587034 | 0.600000 | 0.077488 | 1.373303 s |

The SimLM and embedding metrics measure different tasks and should not be
treated as a direct numerical model-to-model comparison. Within the controlled
embedding experiment, MiniLM performed better than BGE by 0.241376 NDCG@5 and
0.400000 Precision@5.

Mini LM Eligibility Training- Current evaluation
<img width="618" height="301" alt="image" src="https://github.com/user-attachments/assets/5a9a8ae2-f8a5-4af8-aece-deaaafe73fad" />


### Successes, failures, and interpretation
| Example / category | Expected behavior | Actual behavior | Success or failure? | Likely reason |
|---|---|---|---|---|
| SimLM predictions| Calculates predictions based on the relevance scores from the SimLM model | Produces predictions/percentage values for the top 10 eligible grants| Success | The reason being is that the prediction values are mainly based on the prompt of the model which influences these outputs. |
| MiniLM primary top five | Excellent and Strong eligible/open grants should appear first. | All five top-ranked grants were Excellent or Strong, producing Precision@5 of 1.0. | Success | MiniLM represented the hypothetical mission and permitted grant descriptions effectively on this synthetic test pool. |
| BGE primary top five | BGE should place the same highly relevant grants near the top. | BGE placed one Moderate and one Weak grant in its top five, producing Precision@5 of 0.6. | Partial failure | The short, requirement-heavy permitted text may not provide enough neutral purpose detail for BGE's representation. |
| Unfiltered ranking | Relevant grants should rank highly, but hard constraints still need separate checking. | Both models placed two Ineligible grants and one non-open grant in their unfiltered top five. | Expected limitation | Embedding similarity measures themes, not geography, applicant restrictions, deadlines, or availability. |
| Model stability | Reasonable embedding models should produce broadly similar rankings. | Several grants moved more than 20 positions between MiniLM and BGE. | Failure / uncertainty | The models emphasize different terms and semantic relationships, especially when descriptions are sparse. |

- **What improved?** The new evaluation uses a fixed held-out split, identical candidate pools, ranking-specific metrics, deterministic ordering, explicit schema validation, and protections against label leakage. Twelve unit tests passed.
- **What did not improve?** Neither embedding model can reliably determine eligibility or availability. The synthetic labels and hypothetical mission also do not establish performance on verified real grants.
- **Next experiment:** Build a separate eligibility/availability stage, obtain independent human relevance judgments for real grants, and evaluate MiniLM on that set without changing the query or feature selection based on test results.

### Individual contributions during that week
| Member | Work completed | Result (worked / partly / failed) | Evidence | Next step |
|---|---|---|---|---|
| Olivia | I worked on normalizing the score values from last week, created prediction values for the model, and developed the evaluation metrics as well. | partially, the accuracy was above the goal threshold but precision and recall were not| https://colab.research.google.com/drive/17F2tcFgMOSILFkfg9LDv9CIuKJQ7tjf9?usp=sharing|Combine the results from the summarization model into the input of the SimLM model to create a single baseline. As well as, look into ways to improve the precision and recall outputs too.|
| David | Prepared and validated the 200-row synthetic dataset and predefined 160/40 split; implemented and ran the leakage-safe MiniLM-versus-BGE ranking evaluation; calculated NDCG@5, Precision@5, and Spearman; exported rankings; analyzed failures; and added 12 tests and reproducibility documentation. | Worked for the synthetic held-out evaluation; MiniLM outperformed BGE, but real-world accuracy remains unverified. | [`results/model_comparison_report.md`](results/model_comparison_report.md), [`results/model_comparison_metrics.csv`](results/model_comparison_metrics.csv), and [`tests/test_grant_comparison.py`](tests/test_grant_comparison.py) | Add a separately verified eligibility stage and evaluate on independently labeled real grants. |

---

## Week 8 — Stress-Test: Reliability on Hidden Data

**Deliverable during Week 9 Lab:** Hidden-test results (as provided by the TA), analysis of at least three important failure cases, and a prioritized improvement plan.

- **Submitted system version / commit:** [Commit hash]
- **Hidden-test metrics and leaderboard result:** [Results, when available]
- **Difference from development results:** [Compare; do not assume the hidden set is identical]
- David will add failure cases from evaluation report for hidden tests

| Failure case / category | What went wrong | Possible cause | Severity / user impact | Proposed fix |
|---|---|---|---|---|
| [Case 1] | [Observation] | [Hypothesis] | [Impact] | [Fix] |
| [Case 2] | [Observation] | [Hypothesis] | [Impact] | [Fix] |
| [Case 3] | [Observation] | [Hypothesis] | [Impact] | [Fix] |

### Changes attempted after stress-testing
| Change | Worked? (Yes / Partly / No) | Evidence | Keep, revise, or revert? |
|---|---|---|---|
| [Change] | [Status] | [Result] | [Decision] |

### Individual contributions during that week
| Member | Work completed | Result (worked / partly / failed) | Evidence | Next step |
|---|---|---|---|---|
| [Name] | [Specific work] | [Status] | [Link] | [Next step] |

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
[Week 7 presentation notes](docs/week7_presentation_notes.md).


