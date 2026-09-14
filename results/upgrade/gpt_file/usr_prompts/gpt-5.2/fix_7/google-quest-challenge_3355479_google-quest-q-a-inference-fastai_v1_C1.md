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

0.18645

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.12778) has done: 'I fix the fastai API error by switching `TextBlock.from_df` to use the correct `text_cols` argument, which unblocks dataloader creation and therefore the entire training/inference pipeline. I also make the DataBlock use the existing `text` and `y` columns directly (keeping the same modeling approach and targets) and add a tiny safety cast so regression targets become a float tensor with the right shape. Finally, I keep your submission-building logic but ensure predictions always exist and align to `qa_id`, writing a valid `submission.csv` with the required 31 columns.'
- What this solution (achieved 0.13754) has done: 'Your current score (0.12778) is well below the target (0.24087), so we should make a small, low-risk improvement that better matches the competition’s Spearman (rank) metric without changing your model architecture or training loop. The biggest issue is that `RegressionBlock` without an explicit `n_out` can silently behave like single-target regression; we make the output dimensionality explicit for both question and answer models so they truly learn all targets. We also remove `to_fp16()` (mixed precision can hurt ranking stability for tiny 1-epoch runs) and add a very light post-processing step: per-column rank normalization to `[0,1]`, which is monotonic and therefore aligned with Spearman while keeping outputs in-range. These are minimal changes that preserve the approach (AWD_LSTM + MSE, same split, same epochs) but should move the score upward toward your target.'
- What this solution (achieved 0.16018) has done: 'Your current score (0.13754) is far below the target (0.24087), so we should nudge performance upward with very small, low-risk changes that keep your two-model AWD_LSTM + MSE setup intact. The biggest win without changing the approach is to train a bit longer (more than 1 epoch) so the model actually learns useful ranking signal; this preserves the same training loop and architecture. Because Spearman is rank-based, we keep your per-column rank normalization, but we also apply it to validation metric computation by keeping the metric as-is (it’s fine) and only post-process test predictions (as you already do). Finally, we reduce random split variance by using a stratification-free but more stable validation setup (same RandomSplitter seed, but we also set `num_workers=0` to improve determinism on Kaggle), which typically improves leaderboard reliability without changing semantics.'
- What this solution (achieved 0.18645) has done: 'Your gap to the target is large (0.16018 → 0.24087), so we should make a small, low-risk improvement that preserves your exact two-model AWD_LSTM + MSE setup. The most impactful minimal change is to use the full `question_title + question_body + answer` context for both models (while still predicting question targets vs answer targets separately), which often improves rank ordering without changing architecture or training semantics. Additionally, we slightly increase `seq_len` so the model can “see” more text (still the same tokenizer/model), and we train a bit longer with the same `fit_one_cycle` call to reduce underfitting. Everything else (splitting, metric, rank-normalization postprocess, submission writing) remains the same.'

# 9. Code solution

## === cell 0
import os
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from fastai.text.all import *
from scipy.stats import spearmanr, rankdata

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
def _safe_str(x):
    return "" if pd.isna(x) else str(x)


def combine_all(row):
    t = _safe_str(row.get("question_title", ""))
    b = _safe_str(row.get("question_body", ""))
    a = _safe_str(row.get("answer", ""))
    return t + "\n\n" + b + "\n\n" + a


train_q = train[["qa_id", "question_title", "question_body", "answer"] + ques].copy()
test_q = test[["qa_id", "question_title", "question_body", "answer"]].copy()
train_q["text"] = train_q.apply(combine_all, axis=1)
test_q["text"] = test_q.apply(combine_all, axis=1)

train_a = train[["qa_id", "question_title", "question_body", "answer"] + ans].copy()
test_a = test[["qa_id", "question_title", "question_body", "answer"]].copy()
train_a["text"] = train_a.apply(combine_all, axis=1)
test_a["text"] = test_a.apply(combine_all, axis=1)

train_q["y"] = train_q[ques].astype("float32").values.tolist()
train_a["y"] = train_a[ans].astype("float32").values.tolist()



## === cell 5
BS = 128


def make_dls(df, text_col, y_col, bs, seq_len=96, seed=42, valid_pct=0.2):
    n_out = len(df.iloc[0][y_col])
    dblock = DataBlock(
        blocks=(
            TextBlock.from_df(text_cols=text_col, is_lm=False, seq_len=seq_len),
            RegressionBlock(n_out=n_out),
        ),
        get_x=ColReader(text_col),
        get_y=ColReader(y_col),
        splitter=RandomSplitter(valid_pct=valid_pct, seed=seed),
    )
    return dblock.dataloaders(df, bs=bs, num_workers=0)


dls_q = make_dls(
    train_q[["text", "y"]].copy(), text_col="text", y_col="y", bs=BS, seq_len=96
)
dls_a = make_dls(
    train_a[["text", "y"]].copy(), text_col="text", y_col="y", bs=BS, seq_len=96
)



## === cell 6
q_learn = text_classifier_learner(
    dls_q,
    AWD_LSTM,
    pretrained=True,
    metrics=[mean_spearman_metric],
    loss_func=MSELossFlat(),
)

a_learn = text_classifier_learner(
    dls_a,
    AWD_LSTM,
    pretrained=True,
    metrics=[mean_spearman_metric],
    loss_func=MSELossFlat(),
)



## === cell 7
q_learn.fit_one_cycle(5, 2e-3)
a_learn.fit_one_cycle(5, 2e-3)

gc.collect()



## === cell 8
tst_dl_q = q_learn.dls.test_dl(test_q[["text"]], num_workers=0)
tst_dl_a = a_learn.dls.test_dl(test_a[["text"]], num_workers=0)

q_preds, _ = q_learn.get_preds(dl=tst_dl_q)
a_preds, _ = a_learn.get_preds(dl=tst_dl_a)


def rank_norm_cols(t: torch.Tensor) -> torch.Tensor:
    arr = t.detach().float().cpu().numpy()
    out = np.zeros_like(arr, dtype=np.float32)
    n = arr.shape[0]
    denom = max(n - 1, 1)
    for j in range(arr.shape[1]):
        r = rankdata(arr[:, j], method="average")  # 1..n
        out[:, j] = (r - 1.0) / denom
    return torch.from_numpy(out)


q_preds = rank_norm_cols(q_preds)
a_preds = rank_norm_cols(a_preds)

q_preds = q_preds.clamp(0, 1)
a_preds = a_preds.clamp(0, 1)

assert q_preds.shape[0] == len(test)
assert a_preds.shape[0] == len(test)
assert q_preds.shape[1] == len(ques), (q_preds.shape, len(ques))
assert a_preds.shape[1] == len(ans), (a_preds.shape, len(ans))



## === cell 9
sub = sample.copy()
pred_df = pd.DataFrame({"qa_id": test["qa_id"].values})

for i, c in enumerate(ques):
    pred_df[c] = q_preds[:, i].numpy()
for i, c in enumerate(ans):
    pred_df[c] = a_preds[:, i].numpy()

sub = sub[["qa_id"] + target_cols].merge(
    pred_df, on="qa_id", how="left", suffixes=("_old", "")
)
sub = sub[["qa_id"] + target_cols]
sub[target_cols] = sub[target_cols].fillna(0.5).clip(0.0, 1.0)

sub.to_csv("submission.csv", index=False)

sub.head(10)
