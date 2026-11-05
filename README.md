# CSEN 166 — Term Team Project for Social Impact (TTP4SI)

**Team:** [A3]  
**Challenge:** [Ardook]  
**Team members:** [Naomi Wiesen, Olivia Desire, David Lopez-Ryan]  
**Repository:** [Link]  
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
         - Train + test validation --> 80% train, 20% test
     - E.g. data extraction-y models to be used to format additional (and/or current) grant opportunities into a spreadsheet that can be treated as a corpus, possibly using SQL(?) or some other basic dictionary format for organizing the grant information
     - Second model takes in the data spreadsheet to develop a score + compare to the Ardook mission statement to see if the grants align with their values
     - Rank the score based on evaluation metrics:
            1) Eligibility (yes/no) , can be used to filter out invalid grants
            2) Relevance, which for the baseline will likely be regarding the grant's purpose compared with Ardook's profile
       
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
- **How to run it:** [Commands, dependencies, configuration; refer to Setup below.]
- **Model(s) and data actually used:** [Names, versions, links, and any changes from the plan.]
- **Example 1:** Input: [non-sensitive example] → Output: [result]
- **Example 2:** Input: [non-sensitive example] → Output: [result]

### Attempts and outcomes
| Attempt / change | Worked? (Yes / Partly / No) | Evidence (output, metric, link) | What we learned / next step |
|---|---|---|---|
| [Attempt] | [Status] | [Evidence] | [Learning] |

### Individual contributions during that week
| Member | What I personally implemented/tested | Result (worked / partly / failed) | Evidence (commit, PR, experiment) | Blocker / next step |
|---|---|---|---|---|
| [Name] | [Specific contribution] | [Status] | [Link] | [Next step] |

---

## Week 7 — Evaluate: Results Checkpoint

**Deliverable during Week 8 Lab:** Preliminary evaluation results, metrics, development-set details, and 2–3 representative successes or failures. Include what you will improve next.

### Evaluation setup
- **Development data:** [Source, size, selection, annotations, link or access instructions]
- **Metrics and rationale:** [Why these metrics reflect the intended use]
- **Comparison:** [Baseline versus current approach, if different]

| System / version | Metric 1 | Metric 2 | Notes |
|---|---:|---:|---|
| Baseline | [Value] | [Value] | [Notes] |
| Current | [Value] | [Value] | [Notes] |

### Successes, failures, and interpretation
| Example / category | Expected behavior | Actual behavior | Success or failure? | Likely reason |
|---|---|---|---|---|
| [Case] | [Expected] | [Observed] | [Status] | [Hypothesis] |

- **What improved?** [Evidence]
- **What did not improve?** [Evidence]
- **Next experiment:** [Specific change and how you will measure it]

### Individual contributions during that week
| Member | Work completed | Result (worked / partly / failed) | Evidence | Next step |
|---|---|---|---|---|
| [Name] | [Specific work] | [Status] | [Link] | [Next step] |

---

## Week 8 — Stress-Test: Reliability on Hidden Data

**Deliverable during Week 9 Lab:** Hidden-test results (as provided by the TA), analysis of at least three important failure cases, and a prioritized improvement plan.

- **Submitted system version / commit:** [Commit hash]
- **Hidden-test metrics and leaderboard result:** [Results, when available]
- **Difference from development results:** [Compare; do not assume the hidden set is identical]

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
[Python version, libraries, compute requirements, model access]

### Installation
```bash
# Add installation commands here
```

### Run the system
```bash
# Add an example command here
```

### Run evaluation
```bash
# Add an example evaluation command here
```

### Repository structure
```text
project/
├── README.md
├── requirements.txt
├── src/
├── tests/
├── evaluation/
└── examples/
```


