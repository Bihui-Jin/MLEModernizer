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

No external packages required in the script and installed.

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

0.3655933516945904

# 6. Current score

0.25853

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.03533) has done: 'I fix the import errors, make tokenisation fallback to the standard HuggingFace model, guard directory creation, skip saving/loading non‑existent pre‑processed files, and build a simple dataset that tokenises the test rows on the fly. This lets the script run end‑to‑end, load the available pretrained model weights, generate predictions for the test set, and write a correctly formatted `submission.csv` file.'
- What this solution (achieved 0.16444) has done: 'I replace the failing Transformer imports with safe fallbacks, add a simple deterministic tokeniser for cases without a tokenizer, and compute predictions using a lightweight linear‑regression model built from question/answer text lengths. This removes the crash‑prone BERT model while still providing varied predictions that correlate better with the targets, moving the score toward the target range. The script now writes a proper `submission.csv` with the required columns.'
- What this solution (achieved 0.23157) has done: 'I added the missing `os` import to fix the directory‑creation errors, and expanded the feature engineering in `build_features` to include word counts and simple punctuation flags for title, body and answer texts. These extra numeric signals give the linear regression a richer representation, which should raise the Spearman correlation toward the target while keeping the core model unchanged. The rest of the pipeline remains the same, now successfully creating the required `submission.csv`.'
- What this solution (achieved 0.25194) has done: 'I add a few informative numeric features (log‑scaled lengths and average word lengths) to the existing feature set used by the linear‑regression model. These extra monotonic signals often improve the rank‑based Spearman metric while keeping the original linear‑regression approach unchanged. The change is limited to the feature‑building function, so the overall pipeline, model architecture, and output format remain the same.'
- What this solution (achieved 0.28013) has done: 'The fix adds the missing imports, loads the data, builds numeric features, fits a simple linear‑regression model (using NumPy least‑squares) on the training set, generates predictions for the test set, clips them to the required [0, 1] range and writes a correctly‑formatted `submission.csv` with the exact column order from the sample submission. No core modelling logic is changed, only the surrounding pipeline is completed so the script runs end‑to‑end and can achieve a score closer to the target.'
- What this solution (achieved 0.28529) has done: 'The update adds a few extra numeric signals that are monotonic (digit ratios, punctuation counts, and log‑scaled word counts) which tend to improve rank‑based metrics like Spearman while keeping the linear‑regression pipeline unchanged. The regularisation strength is also increased slightly (ridge_alpha = 1e‑2) to reduce over‑fitting, which should raise the validation correlation toward the target score.'
- What this solution (achieved 0.28532) has done: 'I add two simple monotonic ratio features (title‑to‑body length and answer‑to‑body length) to the feature matrix, which often improve rank‑based metrics, and I slightly reduce the ridge regularisation strength (from 1e‑2 to 1e‑3) to let the model fit the training data a bit better. These minimal changes keep the overall linear‑regression pipeline unchanged while moving the Spearman score closer to the target.'
- What this solution (achieved 0.28803) has done: 'I make the script robust to the actual data location by trying several common Kaggle input directories and picking the first one that exists. The loading block is moved to the first cell, then the categorical frequency dictionaries are built, followed by the feature‑engineering function and the training/prediction pipeline. This fixes the FileNotFoundError, ensures all variables are defined before use, and guarantees a correctly‑named `submission.csv` with the required columns. No core modeling logic is changed.'
- What this solution (achieved 0.25954) has done: 'I add feature scaling (zero‑mean, unit‑variance) using the training split statistics and include a finer regularisation option (5e‑5) so the ridge model can fit the data slightly better. These small, monotonic‑preserving changes are expected to raise the Spearman correlation toward the target without altering the overall linear‑regression pipeline.'
- What this solution (achieved 0.25853) has done: 'I add a few more monotonic numeric signals (total lengths, total word counts, their logs, overall average word length and a title‑to‑answer length ratio) to the feature matrix, because such extra rank‑preserving features often lift Spearman correlation. I also broaden the ridge‑regularisation search to include slightly smaller and larger alphas, giving the model a better chance to find an optimal regularisation strength. These changes keep the linear‑ridge pipeline unchanged while aiming to raise the validation score toward the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

possible_roots = [
    "data/google-quest-challenge",
    "input/google-quest-challenge",
    "working/google-quest-challenge",
    "/kaggle/input/google-quest-challenge",
]
DATA_ROOT = None
for root in possible_roots:
    if os.path.isdir(root):
        DATA_ROOT = root
        break
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate the 'google-quest-challenge' data directory."
    )

TRAIN_PATH = os.path.join(DATA_ROOT, "train.csv")
TEST_PATH = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
SUBMISSION_PATH = "submission.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

target_cols = train_df.columns[-30:].tolist()
y_all = train_df[target_cols].values.astype(np.float32)

_category_counts = train_df["category"].value_counts().to_dict()
_host_counts = train_df["host"].value_counts().to_dict()




## === cell 1
def build_features(df):
    title_len = df.question_title.astype(str).apply(len).values
    body_len = df.question_body.astype(str).apply(len).values
    answer_len = df.answer.astype(str).apply(len).values

    title_wc = df.question_title.astype(str).apply(lambda x: len(x.split())).values
    body_wc = df.question_body.astype(str).apply(lambda x: len(x.split())).values
    answer_wc = df.answer.astype(str).apply(lambda x: len(x.split())).values

    title_q = df.question_title.astype(str).apply(lambda x: int("?" in x)).values
    body_q = df.question_body.astype(str).apply(lambda x: int("?" in x)).values
    answer_q = df.answer.astype(str).apply(lambda x: int("?" in x)).values

    title_exc = df.question_title.astype(str).apply(lambda x: int("!" in x)).values
    body_exc = df.question_body.astype(str).apply(lambda x: int("!" in x)).values
    answer_exc = df.answer.astype(str).apply(lambda x: int("!" in x)).values

    title_up = (
        df.question_title.astype(str)
        .apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
        .values
    )
    body_up = (
        df.question_body.astype(str)
        .apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
        .values
    )
    answer_up = (
        df.answer.astype(str)
        .apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
        .values
    )

    title_len_log = np.log1p(title_len)
    body_len_log = np.log1p(body_len)
    answer_len_log = np.log1p(answer_len)

    title_avg_word_len = title_len / np.maximum(title_wc, 1)
    body_avg_word_len = body_len / np.maximum(body_wc, 1)
    answer_avg_word_len = answer_len / np.maximum(answer_wc, 1)

    title_digit_ratio = (
        df.question_title.astype(str)
        .apply(lambda x: sum(c.isdigit() for c in x) / max(len(x), 1))
        .values
    )
    body_digit_ratio = (
        df.question_body.astype(str)
        .apply(lambda x: sum(c.isdigit() for c in x) / max(len(x), 1))
        .values
    )
    answer_digit_ratio = (
        df.answer.astype(str)
        .apply(lambda x: sum(c.isdigit() for c in x) / max(len(x), 1))
        .values
    )

    title_punct = (
        df.question_title.astype(str)
        .apply(lambda x: x.count("?") + x.count("!"))
        .values
    )
    body_punct = (
        df.question_body.astype(str).apply(lambda x: x.count("?") + x.count("!")).values
    )
    answer_punct = (
        df.answer.astype(str).apply(lambda x: x.count("?") + x.count("!")).values
    )

    title_wc_log = np.log1p(title_wc)
    body_wc_log = np.log1p(body_wc)
    answer_wc_log = np.log1p(answer_wc)

    cat_freq = df["category"].map(_category_counts).fillna(0).values
    host_freq = df["host"].map(_host_counts).fillna(0).values

    title_body_ratio = title_len / (body_len + 1)
    answer_body_ratio = answer_len / (body_len + 1)
    title_body_ratio_log = np.log1p(title_body_ratio)
    answer_body_ratio_log = np.log1p(answer_body_ratio)

    url_len = df["url"].astype(str).apply(len).values
    host_len = df["host"].astype(str).apply(len).values

    total_len = title_len + body_len + answer_len
    total_len_log = np.log1p(total_len)

    total_wc = title_wc + body_wc + answer_wc
    total_wc_log = np.log1p(total_wc)

    avg_word_len_total = total_len / np.maximum(total_wc, 1)

    title_answer_ratio = title_len / (answer_len + 1)


    bias = np.ones_like(title_len)

    feats = np.vstack(
        [
            title_len,
            body_len,
            answer_len,
            title_wc,
            body_wc,
            answer_wc,
            title_q,
            body_q,
            answer_q,
            title_exc,
            body_exc,
            answer_exc,
            title_up,
            body_up,
            answer_up,
            title_len_log,
            body_len_log,
            answer_len_log,
            title_avg_word_len,
            body_avg_word_len,
            answer_avg_word_len,
            title_digit_ratio,
            body_digit_ratio,
            answer_digit_ratio,
            title_punct,
            body_punct,
            answer_punct,
            title_wc_log,
            body_wc_log,
            answer_wc_log,
            cat_freq,
            host_freq,
            title_body_ratio,
            answer_body_ratio,
            title_body_ratio_log,
            answer_body_ratio_log,
            url_len,
            host_len,
            total_len,
            total_len_log,
            total_wc,
            total_wc_log,
            avg_word_len_total,
            title_answer_ratio,
            bias,
        ]
    ).T
    return feats




## === cell 2
X_all = build_features(train_df)
X_test = build_features(test_df)

rng = np.random.default_rng(42)
perm = rng.permutation(X_all.shape[0])
split_idx = int(0.8 * len(perm))
train_idx, val_idx = perm[:split_idx], perm[split_idx:]

X_train, X_val = X_all[train_idx], X_all[val_idx]
y_train, y_val = y_all[train_idx], y_all[val_idx]

mean_feat = X_train.mean(axis=0, keepdims=True)
std_feat = X_train.std(axis=0, keepdims=True) + 1e-6  # avoid division by zero

X_train = (X_train - mean_feat) / std_feat
X_val = (X_val - mean_feat) / std_feat
X_all = (X_all - mean_feat) / std_feat
X_test = (X_test - mean_feat) / std_feat

candidate_alphas = [1e-6, 5e-5, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
best_alpha = candidate_alphas[0]
best_score = -np.inf


def mean_spearman(y_true, y_pred):
    corrs = []
    for i in range(y_true.shape[1]):
        corr, _ = spearmanr(y_true[:, i], y_pred[:, i])
        if np.isnan(corr):
            corr = 0.0
        corrs.append(corr)
    return np.mean(corrs)


for alpha in candidate_alphas:
    A = X_train.T @ X_train + alpha * np.eye(X_train.shape[1])
    B = X_train.T @ y_train
    w = np.linalg.solve(A, B)
    val_pred = X_val @ w
    score = mean_spearman(y_val, val_pred)
    if score > best_score:
        best_score = score
        best_alpha = alpha

ridge_alpha = best_alpha

A_full = X_all.T @ X_all + ridge_alpha * np.eye(X_all.shape[1])
B_full = X_all.T @ y_all
weights = np.linalg.solve(A_full, B_full)  # shape (features, targets)

test_pred = X_test @ weights
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = pd.DataFrame(test_pred, columns=target_cols)
submission.insert(0, "qa_id", test_df["qa_id"].values)
submission.to_csv(SUBMISSION_PATH, index=False)

print(f"Submission written to {SUBMISSION_PATH} using ridge_alpha={ridge_alpha:.5f}")
