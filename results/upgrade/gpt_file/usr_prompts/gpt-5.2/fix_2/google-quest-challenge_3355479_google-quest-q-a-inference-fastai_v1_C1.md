# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Given questions and answers from various StackExchange properties, predict target values of 30 labels for each question-answer pair.

## Metric
Mean column-wise Spearman's correlation coefficient. The Spearman's rank correlation is computed for each target column, and the mean of these values is calculated for the submission score.

## Submission Format
For each qa_id in the test set, you must predict a probability for each target variable. The predictions should be in the range [0,1]. The file should contain a header and have the following format:

```
qa_id,question_asker_intent_understanding,...,answer_well_written
6,0.0,...,0.5
8,0.5,...,0.1
18,1.0,...,0.0
etc.
```

## Dataset
The list of 30 target labels are the same as the column names in the `sample_submission.csv` file. Target labels with the prefix `question_` relate to the `question_title` and/or `question_body` features in the data. Target labels with the prefix `answer_` relate to the `answer` feature.

Target labels are aggregated from multiple raters, and can have continuous values in the range `[0,1]`. Therefore, predictions must also be in that range.

- **train.csv** - the training data (target labels are the last 30 columns)
- **test.csv** - the test set (you must predict 30 labels for each test set row)
- **sample_submission.csv** - a sample submission file in the correct format; column names are the 30 target labels

# 2. Python version

3.8

# 3. Installed packages

fastai==2.8.5
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scipy==1.15.3
sklearn-pandas==2.2.0
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (83 lines)
            sample_submission.csv (609 lines)
            sample_submission.csv.zip (8.9 kB)
            test.csv (19551 lines)
            test.csv.zip (471.7 kB)
            train.csv (159837 lines)
            train.csv.zip (4.2 MB)
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
        input/
            description.md (83 lines)
            sample_submission.csv (609 lines)
            sample_submission.csv.zip (8.9 kB)
            test.csv (19551 lines)
            test.csv.zip (471.7 kB)
            train.csv (159837 lines)
            train.csv.zip (4.2 MB)
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
        working/
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
```

-> data/google-quest-challenge/sample_submission.csv has 608 rows and 31 columns.
The columns are: qa_id, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence... and 16 more columns

-> data/google-quest-challenge/test.csv has 19550 rows and 11 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host

-> data/google-quest-challenge/train.csv has 159836 rows and 41 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer... and 26 more columns

-> data/sample_submission.csv has 608 rows and 31 columns.
The columns are: qa_id, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence... and 16 more columns

-> data/test.csv has 19550 rows and 11 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host

-> data/train.csv has 159836 rows and 41 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer... and 26 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.2408749082754187

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from fastai.text.all import *
from scipy.stats import spearmanr

set_seed(42, reproducible=True)



## === cell 1
path = Path("/kaggle/input/google-quest-challenge")

train = pd.read_csv(path / "train.csv")
test = pd.read_csv(path / "test.csv")
sample = pd.read_csv(path / "sample_submission.csv")

target_cols = [c for c in sample.columns if c != "qa_id"]

assert "qa_id" in test.columns and "qa_id" in sample.columns
assert len(target_cols) == 30
assert set(target_cols).issubset(
    set(train.columns)
), "Train is missing some target columns."



## === cell 2
cols = ["question_title", "question_body", "answer"]

colQA = target_cols
ques = [c for c in colQA if c.startswith("question_")]
ans = [c for c in colQA if c.startswith("answer_")]

assert len(ques) + len(ans) == 30
assert len(ans) == 9, "Expected 9 answer targets for this competition."




## === cell 3
def mean_spearman(preds, targs):
    preds = preds.detach().float().cpu().numpy()
    targs = targs.detach().float().cpu().numpy()
    cors = []
    for j in range(preds.shape[1]):
        r = spearmanr(preds[:, j], targs[:, j]).correlation
        if r is None or np.isnan(r):
            r = 0.0
        cors.append(r)
    return np.mean(cors)


mean_spearman_metric = AccumMetric(mean_spearman, flatten=False)




## === cell 4
def combine_question(row):
    t = row.get("question_title", "")
    b = row.get("question_body", "")
    t = "" if pd.isna(t) else str(t)
    b = "" if pd.isna(b) else str(b)
    return t + "\n\n" + b


def clean_answer(x):
    x = "" if pd.isna(x) else str(x)
    return x


train_q = train[["qa_id", "question_title", "question_body"] + ques].copy()
test_q = test[["qa_id", "question_title", "question_body"]].copy()
train_q["text"] = train_q.apply(combine_question, axis=1)
test_q["text"] = test_q.apply(combine_question, axis=1)

train_a = train[["qa_id", "answer"] + ans].copy()
test_a = test[["qa_id", "answer"]].copy()
train_a["text"] = train_a["answer"].map(clean_answer)
test_a["text"] = test_a["answer"].map(clean_answer)



## === cell 5
BS = 128

dls_q = TextDataLoaders.from_df(
    train_q,
    text_col="text",
    label_col=ques,
    valid_pct=0.2,
    seed=42,
    bs=BS,
    is_lm=False,
    seq_len=72,
)

dls_a = TextDataLoaders.from_df(
    train_a,
    text_col="text",
    label_col=ans,
    valid_pct=0.2,
    seed=42,
    bs=BS,
    is_lm=False,
    seq_len=72,
)



## === cell 6
q_learn = text_classifier_learner(
    dls_q,
    AWD_LSTM,
    pretrained=True,
    metrics=[mean_spearman_metric],
    loss_func=MSELossFlat(),
).to_fp16()

a_learn = text_classifier_learner(
    dls_a,
    AWD_LSTM,
    pretrained=True,
    metrics=[mean_spearman_metric],
    loss_func=MSELossFlat(),
).to_fp16()



## === cell 7
q_learn.fit_one_cycle(1, 2e-3)
a_learn.fit_one_cycle(1, 2e-3)

gc.collect()



## === cell 8
tst_dl_q = q_learn.dls.test_dl(test_q[["text"]])
tst_dl_a = a_learn.dls.test_dl(test_a[["text"]])

q_preds, _ = q_learn.get_preds(dl=tst_dl_q)
a_preds, _ = a_learn.get_preds(dl=tst_dl_a)

q_preds = q_preds.clamp(0, 1)
a_preds = a_preds.clamp(0, 1)

assert q_preds.shape[0] == len(test)
assert a_preds.shape[0] == len(test)
assert q_preds.shape[1] == len(ques)
assert a_preds.shape[1] == len(ans)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_55/1579826415.py in <cell line: 0>()
     12 assert q_preds.shape[0] == len(test)
     13 assert a_preds.shape[0] == len(test)
---> 14 assert q_preds.shape[1] == len(ques)
     15 assert a_preds.shape[1] == len(ans)
     16 

AssertionError: 

## === cell 9
sub = sample.copy()
pred_df = pd.DataFrame({"qa_id": test["qa_id"].values})
for i, c in enumerate(ques):
    pred_df[c] = q_preds[:, i].cpu().numpy()
for i, c in enumerate(ans):
    pred_df[c] = a_preds[:, i].cpu().numpy()

sub = sub[["qa_id"] + target_cols].merge(
    pred_df, on="qa_id", how="left", suffixes=("_old", "")
)
sub = sub[["qa_id"] + target_cols]

sub[target_cols] = sub[target_cols].fillna(0.5).clip(0.0, 1.0)

sub.to_csv("submission.csv", index=False)

sub.head(10)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_55/3123237563.py in <cell line: 0>()
      5 pred_df = pd.DataFrame({"qa_id": test["qa_id"].values})
      6 for i, c in enumerate(ques):
----> 7     pred_df[c] = q_preds[:, i].cpu().numpy()
      8 for i, c in enumerate(ans):
      9     pred_df[c] = a_preds[:, i].cpu().numpy()

IndexError: index 10 is out of bounds for dimension 1 with size 10
