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

0.3163193312381239

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved nan) has done: 'The crash comes from a shape mismatch: `q_preds`/`a_preds` are not coming out with exactly `len(ques)` / `len(ans)` columns, so assigning them into `pred_df[ques]` / `pred_df[ans]` raises “Columns must be same length as key”. I keep the same two-model setup and training loop, but make the head explicitly multi-target regression by setting `n_out` to the number of labels for each learner. I also make the submission assembly robust by aligning prediction columns to the expected target columns (and safely falling back to 0.5 if something is still off), ensuring a valid `submission.csv` is always written.'

# 9. Code solution

## === cell 0
import os, random, gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fastai.text.all import *
from scipy.stats import spearmanr


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)



## === cell 1
base = Path("/kaggle/data/google-quest-challenge")
if not base.exists():
    base = Path("/kaggle/input/google-quest-challenge")
if not base.exists():
    base = Path("../input/google-quest-challenge")
if not base.exists():
    base = Path("../data/google-quest-challenge")

train_path = base / "train.csv"
test_path = base / "test.csv"
sample_path = base / "sample_submission.csv"

assert train_path.exists(), f"train.csv not found at {train_path}"
assert test_path.exists(), f"test.csv not found at {test_path}"
assert sample_path.exists(), f"sample_submission.csv not found at {sample_path}"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample = pd.read_csv(sample_path)

train.shape, test.shape, sample.shape



## === cell 2
cols = ["question_title", "question_body", "answer"]

target_cols = [c for c in sample.columns if c != "qa_id"]

colQA = target_cols
ques = [c for c in colQA if c.startswith("question_")]
ans = [c for c in colQA if c.startswith("answer_")]

len(ques), len(ans), len(colQA)




## === cell 3
class AvgSpearman(Metric):
    def __init__(self):
        self._name = "avg_spearman"

    def reset(self):
        self.preds = []
        self.targs = []

    def accumulate(self, learn):
        self.preds.append(learn.pred.detach().float().cpu())
        self.targs.append(learn.y.detach().float().cpu())

    @property
    def value(self):
        if len(self.preds) == 0:
            return None
        p = torch.cat(self.preds, dim=0).numpy()
        t = torch.cat(self.targs, dim=0).numpy()
        s = 0.0
        ncols = p.shape[1]
        for j in range(ncols):
            corr = spearmanr(p[:, j], t[:, j]).correlation
            if np.isnan(corr):
                corr = 0.0
            s += corr
        return s / ncols




## === cell 4
for c in cols:
    train[c] = train[c].fillna("")
    test[c] = test[c].fillna("")

train_q = train.copy()
test_q = test.copy()
train_q["text"] = (
    train_q["question_title"].astype(str) + " " + train_q["question_body"].astype(str)
).str.strip()
test_q["text"] = (
    test_q["question_title"].astype(str) + " " + test_q["question_body"].astype(str)
).str.strip()

train_a = train.copy()
test_a = test.copy()
train_a["text"] = train_a["answer"].astype(str)
test_a["text"] = test_a["answer"].astype(str)

train_q[["text"] + ques].head(2)



## === cell 5
BS = 128

for c in ques:
    train_q[c] = train_q[c].astype("float32")
for c in ans:
    train_a[c] = train_a[c].astype("float32")


def make_reg_dls(df, text_col, y_cols, bs=128, seq_len=72, valid_pct=0.2, seed=42):
    def _get_y(r):
        return np.asarray([r[c] for c in y_cols], dtype=np.float32)

    splitter = RandomSplitter(valid_pct=valid_pct, seed=seed)

    dblock = DataBlock(
        blocks=(
            TextBlock.from_df(text_col=text_col, seq_len=seq_len),
            RegressionBlock(n_out=len(y_cols)),
        ),
        get_x=ColReader(text_col),
        get_y=_get_y,
        splitter=splitter,
    )
    return dblock.dataloaders(df, bs=bs)


dls_q = make_reg_dls(
    train_q, text_col="text", y_cols=ques, bs=BS, seq_len=72, valid_pct=0.2, seed=42
)
dls_a = make_reg_dls(
    train_a, text_col="text", y_cols=ans, bs=BS, seq_len=72, valid_pct=0.2, seed=42
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1484720292.py in <cell line: 0>()
     29 
     30 
---> 31 dls_q = make_reg_dls(
     32     train_q, text_col="text", y_cols=ques, bs=BS, seq_len=72, valid_pct=0.2, seed=42
     33 )

/tmp/ipykernel_55/1484720292.py in make_reg_dls(df, text_col, y_cols, bs, seq_len, valid_pct, seed)
     19     dblock = DataBlock(
     20         blocks=(
---> 21             TextBlock.from_df(text_col=text_col, seq_len=seq_len),
     22             RegressionBlock(n_out=len(y_cols)),
     23         ),

TypeError: TextBlock.from_df() missing 1 required positional argument: 'text_cols'

## === cell 6
learn_q = text_classifier_learner(
    dls_q, AWD_LSTM, drop_mult=0.5, metrics=[AvgSpearman()], n_out=len(ques)
)
learn_q.loss_func = MSELossFlat()

learn_a = text_classifier_learner(
    dls_a, AWD_LSTM, drop_mult=0.5, metrics=[AvgSpearman()], n_out=len(ans)
)
learn_a.loss_func = MSELossFlat()

try:
    learn_q = learn_q.to_fp16()
    learn_a = learn_a.to_fp16()
except Exception:
    pass



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/191732996.py in <cell line: 0>()
      1 learn_q = text_classifier_learner(
----> 2     dls_q, AWD_LSTM, drop_mult=0.5, metrics=[AvgSpearman()], n_out=len(ques)
      3 )
      4 learn_q.loss_func = MSELossFlat()
      5 

NameError: name 'dls_q' is not defined

## === cell 7
learn_q.fit_one_cycle(1, 2e-3)
learn_a.fit_one_cycle(1, 2e-3)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2580677861.py in <cell line: 0>()
----> 1 learn_q.fit_one_cycle(1, 2e-3)
      2 learn_a.fit_one_cycle(1, 2e-3)
      3 

NameError: name 'learn_q' is not defined

## === cell 8
dl_q_test = learn_q.dls.test_dl(test_q)
dl_a_test = learn_a.dls.test_dl(test_a)

q_preds, _ = learn_q.get_preds(dl=dl_q_test)
a_preds, _ = learn_a.get_preds(dl=dl_a_test)

q_preds.shape, a_preds.shape



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/395052110.py in <cell line: 0>()
----> 1 dl_q_test = learn_q.dls.test_dl(test_q)
      2 dl_a_test = learn_a.dls.test_dl(test_a)
      3 
      4 q_preds, _ = learn_q.get_preds(dl=dl_q_test)
      5 a_preds, _ = learn_a.get_preds(dl=dl_a_test)

NameError: name 'learn_q' is not defined

## === cell 9
sub = sample.copy()

pred_df = pd.DataFrame({"qa_id": test["qa_id"].values})

q_np = q_preds.float().cpu().numpy()
a_np = a_preds.float().cpu().numpy()


def fix_pred_shape(arr, n_rows, n_cols, fill=0.5):
    arr = np.asarray(arr)
    if arr.ndim != 2:
        return np.full((n_rows, n_cols), fill, dtype=np.float32)
    if arr.shape[0] != n_rows:
        if arr.shape[0] > n_rows:
            arr = arr[:n_rows, :]
        else:
            pad = np.full((n_rows - arr.shape[0], arr.shape[1]), fill, dtype=np.float32)
            arr = np.vstack([arr, pad])
    if arr.shape[1] != n_cols:
        if arr.shape[1] > n_cols:
            arr = arr[:, :n_cols]
        else:
            pad = np.full((arr.shape[0], n_cols - arr.shape[1]), fill, dtype=np.float32)
            arr = np.hstack([arr, pad])
    return arr.astype(np.float32, copy=False)


q_np = fix_pred_shape(q_np, len(test), len(ques), fill=0.5)
a_np = fix_pred_shape(a_np, len(test), len(ans), fill=0.5)

pred_df[ques] = q_np
pred_df[ans] = a_np

for c in target_cols:
    pred_df[c] = pred_df[c].clip(0.0, 1.0)

sub = sub.drop(columns=target_cols).merge(pred_df, on="qa_id", how="left")

for c in target_cols:
    if c not in sub.columns:
        sub[c] = 0.5
sub[target_cols] = sub[target_cols].fillna(0.5)

sub = sub[["qa_id"] + target_cols]

out_path = Path("submission.csv")
sub.to_csv(out_path, index=False)
out_path, sub.shape



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2161532459.py in <cell line: 0>()
      3 pred_df = pd.DataFrame({"qa_id": test["qa_id"].values})
      4 
----> 5 q_np = q_preds.float().cpu().numpy()
      6 a_np = a_preds.float().cpu().numpy()
      7 

NameError: name 'q_preds' is not defined

## === cell 10
sub.head(10)
