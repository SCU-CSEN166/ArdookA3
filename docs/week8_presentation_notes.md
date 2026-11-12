# Week 8 presentation notes — MiniLM versus BGE

## Two-to-three-minute explanation

This week, I built and ran a reproducible experiment to compare two lightweight
embedding models for grant relevance: all-MiniLM-L6-v2, which is our baseline,
and BGE-small-en-v1.5, which is the candidate model.

The goal was specifically to rank grants by thematic relevance. I kept
eligibility separate because similarity models cannot reliably decide hard
requirements like geography, applicant type, or whether a grant is still open.

I validated 200 synthetic grant records and confirmed the predefined split of
160 train records and 40 test records. I did not train either model and I did not
use the test labels to adjust the query. Both pretrained models received the
same hypothetical mission, the same test records, the same candidate pools, and
the same metric code.

To prevent label leakage, I allowed only three fields into the models: Program,
Type, and Core Requirements. I excluded fields such as Thematic fit, Why It Fits
Ardook, Recommended Action, eligibility labels, and train/test metadata. Those
excluded fields would have revealed the expected answer or introduced human
conclusions into the input.

For the primary comparison, I used an oracle filter to select grants whose
ground-truth labels said Eligible and Open. This left 18 candidates in the test
pool. The filter was used only to isolate thematic ranking; it is not something
the embedding models predicted.

MiniLM scored 0.828410 on NDCG at 5 and 1.0 on Precision at 5. BGE scored
0.587034 on NDCG at 5 and 0.6 on Precision at 5. In plain language, all five of
MiniLM's top five results were labeled Excellent or Strong, while only three of
BGE's top five were. MiniLM therefore won this synthetic comparison by about
0.241 NDCG points and 0.40 Precision points.

The most important failure appeared when I removed the oracle filter. Both
models placed two ineligible grants and one non-open grant in their top five.
That shows that thematic similarity cannot replace an eligibility checker. I
also found large disagreements between the models; some grants moved more than
20 ranking positions. A likely reason is that our safe input fields contain
short, requirement-heavy descriptions and not always enough neutral purpose
detail.

My conclusion is that MiniLM is the better model for the next version of this
specific synthetic ranking experiment, but the result does not prove that it is
better for real Ardook decisions. The labels assume a hypothetical profile, and
the original 37-grant tracker does not have independently verified ground
truth. My next step would be to build a separate rules-based or verified
eligibility stage, collect human relevance judgments on real grants, and then
evaluate the ranking on that independent set.

## One-minute version

This week, I compared MiniLM and BGE for thematic grant ranking. I validated 200
synthetic grants, used the fixed 40-row test split, and prevented label leakage
by embedding only Program, Type, and Core Requirements. After using an oracle
Eligible-and-Open filter, 18 test candidates remained. MiniLM achieved 0.828
NDCG at 5 and 1.0 Precision at 5, compared with BGE's 0.587 NDCG and 0.6
Precision. That means MiniLM put five relevant grants in its top five, while BGE
put three. However, without the oracle filter, both models recommended two
ineligible and one non-open grant in their top five. My main finding is that
MiniLM is the stronger thematic ranker on this synthetic test, but embeddings
cannot replace a separate eligibility checker, and synthetic performance is not
verified real-world performance.

## Likely questions and answers

### Why did you use NDCG@5?

NDCG rewards putting the highest-grade grants near the top, so it evaluates the
ordering of Excellent, Strong, Moderate, and Weak items rather than treating all
relevant items as identical.

### What does Precision@5 mean here?

Excellent and Strong count as relevant. MiniLM's Precision@5 of 1.0 means all
five top results were relevant. BGE's 0.6 means three of five were relevant.

### Did you train the models?

No. This experiment compared two pretrained embedding models. The 160 train
rows were available as a development partition but were not needed because I
did not fine-tune or tune the fixed query.

### Why is the filter called an oracle filter?

It uses the known test labels Eligible and Open to create the evaluation pool.
That isolates relevance ranking, but it is not deployable because a real system
would need to determine those constraints independently.

### Why might MiniLM have beaten BGE?

The safe source text is short and requirement-heavy, and each model represents
that text differently. The result establishes what happened on this dataset; it
does not prove a general causal explanation or that MiniLM always wins.

### What was the biggest limitation?

The labels and mission are synthetic and hypothetical. The real tracker is
small and has conditional or potentially stale judgments, so we cannot claim
verified real-world accuracy yet.

### What did you personally complete?

I implemented the comparison pipeline, data validation, leakage protections,
ranking metrics, filtered and unfiltered diagnostics, exports, error analysis,
12 unit tests, and the reproducibility documentation. I also ran both models on
CPU and recorded the actual results rather than estimating them.
