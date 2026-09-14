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
Given a dataset of comments from Wikipedia's talk page edits, predict the probability of each comment being toxic.

## Metric
Mean column-wise ROC AUC; the average of the individual AUCs of each predicted column.

## Submission Format
For each `id` in the test set, you must predict a probability for each of the six possible types of comment toxicity (toxic, severe_toxic, obscene, threat, insult, identity_hate). The columns must be in the same order as shown below. The file should contain a header and have the following format:

```
id,toxic,severe_toxic,obscene,threat,insult,identity_hate
00001cee341fdb12,0.5,0.5,0.5,0.5,0.5,0.5
0000247867823ef7,0.5,0.5,0.5,0.5,0.5,0.5
etc.
```

## Dataset 
- **train.csv** - the training set, contains comments with their binary labels
- **test.csv** - the test set, you must predict the toxicity probabilities for these comments.
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        input/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        working/
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
```

-> data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/jigsaw-toxic-comment-classification-challenge/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-toxic-comment-classification-challenge/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> (stopped after 10 files for performance)

# 5. Target score

0.9865778215228144

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'The error is because the script tries to read out-of-notebook “../input/module-*” blend files that do not exist in your provided environment, so nothing downstream is defined and submission writing fails. To keep the core intent (produce a valid submission) with minimal change, I switch the pipeline to read the competition’s provided `sample_submission.csv` and generate a stable baseline prediction for all six labels. This run end-to-end, produce a correctly formatted `submission.csv`, and avoid any dependency on missing external model outputs. I also make the input path robust by auto-detecting whether the data is under `/kaggle/input/...` or `/kaggle/data/...` in your environment.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path

CANDIDATES = [
    Path("/kaggle/input/jigsaw-toxic-comment-classification-challenge"),
    Path("/kaggle/data/jigsaw-toxic-comment-classification-challenge"),
    Path("/kaggle/input"),
    Path("/kaggle/data"),
]

base_dir = None
for p in CANDIDATES:
    if p.exists():
        base_dir = p
        break

print("Using base_dir:", base_dir)
if base_dir is not None and base_dir.is_dir():
    print("Top-level entries:", sorted([x.name for x in base_dir.iterdir()])[:30])



## === cell 1
import numpy as np
import pandas as pd
from pathlib import Path


label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

if base_dir is None:
    raise FileNotFoundError(
        "Could not locate /kaggle/input or /kaggle/data in this environment."
    )

comp_dir = base_dir
if (base_dir / "sample_submission.csv").exists() and (base_dir / "test.csv").exists():
    comp_dir = base_dir
elif (
    base_dir / "jigsaw-toxic-comment-classification-challenge" / "sample_submission.csv"
).exists():
    comp_dir = base_dir / "jigsaw-toxic-comment-classification-challenge"

sample_path = comp_dir / "sample_submission.csv"
test_path = comp_dir / "test.csv"

sample_sub = pd.read_csv(sample_path)
test_df = pd.read_csv(test_path, usecols=["id"])

sub = test_df.merge(sample_sub[["id"]], on="id", how="left")
if sub.shape[0] != test_df.shape[0]:
    sub = test_df.copy()

train_path = comp_dir / "train.csv"
if train_path.exists():
    train_df = pd.read_csv(train_path, usecols=label_cols)
    priors = train_df[label_cols].mean().clip(1e-6, 1 - 1e-6)
else:
    priors = pd.Series([0.5] * len(label_cols), index=label_cols)

for c in label_cols:
    sub[c] = float(priors[c])

sub = sub[["id"] + label_cols]

print(sub.head())
print("Submission shape:", sub.shape)



## === cell 2
out_path = Path("submission.csv")
sub.to_csv(out_path, index=False)
print("Wrote:", out_path.resolve())
print("Columns:", list(sub.columns))
print("Dtypes:", sub.dtypes.to_dict())
