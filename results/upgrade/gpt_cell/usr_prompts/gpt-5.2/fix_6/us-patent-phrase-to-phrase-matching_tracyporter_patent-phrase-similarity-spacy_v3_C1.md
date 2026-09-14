# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Given pairs of phrases (an `anchor` and a `target` phrase), build a model to rate how similar they are.  

## Metric
Pearson correlation coefficient.

## Submission Format
For each `id` (representing a pair of phrases) in the test set, you must predict the similarity `score`. The file should contain a header and have the following format:

```
id,score
4112d61851461f60,0
09e418c93a776564,0.25
36baf228038e314b,1
etc.

```

## Dataset
The scores are in the 0-1 range with increments of 0.25 with the following meanings:

- **1.0** - Very close match. This is typically an exact match except possibly for differences in conjugation, quantity (e.g. singular vs. plural), and addition or removal of stopwords (e.g. "the", "and", "or").
- **0.75** - Close synonym, e.g. "mobile phone" vs. "cellphone". This also includes abbreviations, e.g. "TCP" -> "transmission control protocol".
- **0.5** - Synonyms which don't have the same meaning (same function, same properties). This includes broad-narrow (hyponym) and narrow-broad (hypernym) matches.
- **0.25** - Somewhat related, e.g. the two phrases are in the same high level domain but are not synonyms. This also includes antonyms.
- **0.0** - Unrelated.

Files
-----

- **train.csv** - the training set, containing phrases, contexts, and their similarity scores
- **test.csv** - the test set set, identical in structure to the training set but without the score
- **sample_submission.csv** - a sample submission file in the correct format

Columns
-------

- `id` - a unique identifier for a pair of phrases
- `anchor` - the first phrase
- `target` - the second phrase
- `context` - the [CPC classification (version 2021.05)](https://en.wikipedia.org/wiki/Cooperative_Patent_Classification), which indicates the subject within which the similarity is to be scored
- `score` - the similarity. This is sourced from a combination of one or more manual expert ratings.

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
seaborn==0.12.2
sklearn-pandas==2.2.0
spacy==3.8.7
spacy-legacy==3.0.12
spacy-loggers==1.0.5

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        input/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        working/
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
```

-> data/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> data/us-patent-phrase-to-phrase-matching/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/us-patent-phrase-to-phrase-matching/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/us-patent-phrase-to-phrase-matching/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> input/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> (stopped after 10 files for performance)

# 5. Target score

0.444

# 6. Current score

0.12909

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.28726) has done: 'Diagnosis: Cell 13 crashes because `X_test["anchor"][i]` uses label-based indexing on a Series whose index is inherited from `combi` (it continues from `len(train)`), so label `0` does not exist and raises `KeyError: 0`. This is a pandas indexing bug (mixing positional loop with label indexing), not a spaCy issue.

Patch summary: In cell 13, switch to positional indexing via `.iloc[i]` for both `anchor` and `target` when iterating over `range(len(X_test))`. This preserves identical logic while making indexing deterministic and compatible with any inherited index.

Updated cells: Only cell 13 is modified.

Compatibility notes for cell k+1: `simularity` remains a Python list of length `len(X_test)`, so cell 14 (`submission['score'] = simularity`) continues to work unchanged.

Assumptions: `X_test` is a DataFrame with columns `anchor` and `target`, and its length matches the submission template length.'
- What this solution (achieved 0.354) has done: 'To move your Pearson score up toward the 0.444 target without changing the overall approach (spaCy semantic similarity), I make two minimal, metric-relevant fixes: (1) ensure we always use a real vectors model (`en_core_web_md` or `en_core_web_lg`) instead of silently falling back to a blank model (which hurts similarity quality), and (2) include the `context` string in both texts when computing similarity, which is a small, legitimate feature addition that often improves correlation for this dataset. I also keep the same train/test reading and submission writing, and speed up inference (without changing semantics) by using `nlp.pipe` in batches. These changes are expected to increase the score from ~0.287 toward your 0.444 target while staying within the same core logic.'
- What this solution (achieved 0.12909) has done: 'We keep your spaCy similarity approach intact and make two small, metric-relevant tweaks that typically raise Pearson correlation on this competition: (1) rescale cosine similarities from the vectors model into the target’s 0–1 range (clipping to valid bounds), and (2) lightly calibrate predictions by snapping to the nearest allowed label level {0, 0.25, 0.5, 0.75, 1.0}, which matches how the ground-truth scores are distributed. These changes don’t alter the model/loop/feature extraction; they only adjust post-processing to better align with the evaluation target distribution. We also avoid accidentally using `en_core_web_sm` (no vectors) to prevent low-quality similarity scores. The script still run end-to-end and write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



## === cell 1
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 2
train = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv")
test = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv")
submission = pd.read_csv(
    "/kaggle/input/us-patent-phrase-to-phrase-matching/sample_submission.csv"
)



## === cell 3
train



## === cell 4
test



## === cell 5
submission



## === cell 6
sns.distplot(train.score)



## === cell 7
plt.boxplot(train.score)



## === cell 8
target = train.score



## === cell 9
combi = pd.concat([train.drop(["score"], axis=1), test], axis=0, ignore_index=True)
combi



## === cell 10
y = target
X = combi[: len(train)]
X_test = combi[len(train) :]



## === cell 11
import sys, subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "spacy"])
subprocess.check_call(
    [sys.executable, "-m", "spacy", "download", "en_core_web_md", "-q"]
)



## === cell 12
import spacy

nlp = None
for model_name in ("en_core_web_lg", "en_core_web_md"):
    try:
        nlp = spacy.load(model_name)
        break
    except OSError:
        continue

if nlp is None:
    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

test_texts_anchor = (
    X_test["anchor"].astype(str) + " [CTX] " + X_test["context"].astype(str)
).tolist()
test_texts_target = (
    X_test["target"].astype(str) + " [CTX] " + X_test["context"].astype(str)
).tolist()

disable = []
for comp in ("parser", "tagger", "ner", "lemmatizer", "attribute_ruler"):
    if comp in nlp.pipe_names:
        disable.append(comp)

anchors = list(nlp.pipe(test_texts_anchor, batch_size=128, disable=disable))
targets = list(nlp.pipe(test_texts_target, batch_size=128, disable=disable))

simularity = [a.similarity(t) for a, t in zip(anchors, targets)]

print(len(simularity))
print(simularity[:10])

pred = (np.array(simularity, dtype=np.float64) + 1.0) / 2.0
pred = np.clip(pred, 0.0, 1.0)

levels = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
pred = levels[np.argmin(np.abs(pred[:, None] - levels[None, :]), axis=1)]

simularity = pred.tolist()



## === cell 13
submission["score"] = simularity
submission.to_csv("submission.csv", index=False)
submission
