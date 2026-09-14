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

0.3697054710738545

# 6. Current score

0.25134

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved nan) has done: 'The script was failing early because the tokenizer and model paths were invalid, required packages were missing, and later steps depended on objects that were never created. To get a complete, runnable pipeline we replace the broken BERT preprocessing and model inference with a simple baseline: predict the average value of each target column from the training data. This guarantees predictions are in the required [0, 1] range, creates a correctly‑formatted `submission.csv`, and eliminates all import and runtime errors while keeping the overall workflow (data loading → prediction → submission) intact.'
- What this solution (achieved 0.18408) has done: 'I make the file loading robust by checking both the original relative path and the typical Kaggle “/kaggle/input” location, and replace the simple global‑mean baseline with a lightweight per‑category mean model: for each target column we use the mean value computed within the same `category` as the test row (falling back to the overall mean when a category is unseen). This adds modest, data‑driven variation while keeping the core logic unchanged, and should move the Spearman score closer to the target.'
- What this solution (achieved 0.23861) has done: 'The changes add simple numeric text‑length features (question title, body and answer lengths) and fit a cheap linear regression for each target column using these features.  
Predictions from the linear model are blended 50 % with the existing per‑category mean baseline, then clipped to the required [0, 1] range. This adds useful signal while preserving the original workflow and keeps the implementation lightweight.'
- What this solution (achieved 0.23352) has done: 'The changes add categorical one‑hot features to the linear regression (so the model can learn category‑specific effects) and increase the weight of the linear model in the final blend, which should raise the Spearman correlation toward the target while keeping the original simple pipeline intact.'
- What this solution (achieved 0.25087) has done: 'I keep the overall pipeline but improve the numeric text‑length features by applying a log‑1p transform and standardising them (zero‑mean, unit‑variance). This gives the linear regression a more stable basis and usually raises correlation. I also shift the blend toward the linear model (80 % linear + 20 % category mean) to let the better‑fitted predictor dominate while still retaining the safe category fallback.'
- What this solution (achieved 0.2518) has done: 'I add a combined text‑length feature (`total_len`) to give the linear model a bit more signal and shift the blend toward the linear predictions (90 % linear + 10 % category mean). These small, targeted changes should raise the Spearman correlation toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.25024) has done: 'I keep the overall pipeline unchanged but replace the ordinary least‑squares fit with a tiny ridge regularisation (λ = 0.01) to make the linear model less noisy, and I shift the blending towards the more stable category‑mean predictions (70 % linear + 30 % category). These small, targeted tweaks are expected to raise the Spearman score toward the target without altering the core logic.'
- What this solution (achieved 0.2494) has done: 'I add simple interaction features between the standardized text‑length columns, slightly reduce the ridge regularisation, and shift the blend more toward the linear model. These minimal tweaks keep the overall pipeline unchanged while giving the regression extra signal, which should raise the Spearman correlation toward the target score.'
- What this solution (achieved 0.24938) has done: 'The update slightly lowers the ridge regularisation (λ = 0.0005) to let the linear model fit the data a bit more closely and increases the blend weight toward the linear predictions (92 % linear + 8 % category mean). These minimal tweaks are expected to raise the Spearman correlation toward the target while preserving the original pipeline.'
- What this solution (achieved 0.25057) has done: 'I slightly reduce regularisation (λ = 0) and use a more robust pseudo‑inverse, then shift the blending strongly toward the linear model (98 % linear + 2 % category mean). I also add the raw (non‑log) length features, standardised, to give the linear model a bit more signal while keeping the overall pipeline unchanged. These minimal tweaks should raise the Spearman correlation toward the target score.'
- What this solution (achieved 0.25071) has done: 'I keep the overall pipeline unchanged but add a small ridge regularisation (λ = 0.01) to make the linear regression less noisy and shift the blend slightly toward the safer category‑mean predictions (85 % linear + 15 % category). These minimal hyper‑parameter tweaks are expected to raise the Spearman correlation toward the target without altering the core logic.'
- What this solution (achieved 0.25058) has done: 'I slightly reduce the ridge regularisation (set λ to 0.0) and give the linear model a larger share in the final blend (95 % linear + 5 % category‑mean). These minimal tweaks keep the overall pipeline unchanged while allowing the model to fit the data a bit more closely, which should raise the Spearman‑correlation toward the target score.'
- What this solution (achieved 0.24891) has done: 'I keep the overall pipeline intact but add a modest ridge regularisation (λ = 0.1) to make the linear regression less noisy and adjust the blending to rely more on the safe category‑mean baseline (60 % linear + 40 % category). These small, targeted tweaks should raise the Spearman correlation toward the target without altering the core logic.'
- What this solution (achieved 0.25058) has done: 'Implemented a small but effective tweak: removed ridge regularisation (`lambda_reg = 0.0`) to let the linear model fit the data more closely and shifted the blending to rely heavily on the linear predictions (`weight_lin = 0.95`, `weight_cat = 0.05`). These adjustments are expected to raise the Spearman correlation toward the target while preserving the original pipeline logic.'
- What this solution (achieved 0.25134) has done: 'I add a few lightweight ratio features (answer‑to‑total length and title‑to‑body length), standardise them like the existing log‑features, and use a modest ridge regularisation (λ = 0.1). I also shift the blending to give the safe category‑mean baseline a slightly larger influence (80 % linear + 20 % category). These changes keep the overall linear‑regression pipeline intact while providing extra signal that should raise the Spearman score toward the target.'

# 9. Code solution

## === cell 0
import os, pandas as pd, numpy as np

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
import os, pandas as pd


def resolve_path(rel_path):
    """
    Return an existing path for the given relative path.
    Tries the original relative location and the Kaggle input mount.
    """
    candidates = [
        rel_path,
        os.path.join(
            "/kaggle/input/google-quest-challenge", os.path.basename(rel_path)
        ),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find file for {rel_path}")


train_path = resolve_path("../input/google-quest-challenge/train.csv")
test_path = resolve_path("../input/google-quest-challenge/test.csv")
sample_sub_path = resolve_path("../input/google-quest-challenge/sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_sub_path)




## === cell 2
def text_features(df):
    title_len = df["question_title"].fillna("").astype(str).apply(len)
    body_len = df["question_body"].fillna("").astype(str).apply(len)
    answer_len = df["answer"].fillna("").astype(str).apply(len)
    total_len = title_len + body_len + answer_len
    feats = pd.DataFrame(
        {
            "title_len": np.log1p(title_len),
            "body_len": np.log1p(body_len),
            "answer_len": np.log1p(answer_len),
            "total_len": np.log1p(total_len),
            "title_len_raw": title_len,
            "body_len_raw": body_len,
            "answer_len_raw": answer_len,
            "total_len_raw": total_len,
        }
    )
    return feats


target_cols = train.columns[11:]

overall_means = train[target_cols].mean()
category_means = train.groupby("category")[target_cols].mean().reset_index()

category_dummies_train = pd.get_dummies(train["category"], prefix="cat")
category_dummies_test = pd.get_dummies(test["category"], prefix="cat")
category_dummies_test = category_dummies_test.reindex(
    columns=category_dummies_train.columns, fill_value=0
)

num_feat_train_raw = text_features(train)
num_feat_test_raw = text_features(test)

log_feat_cols = ["title_len", "body_len", "answer_len", "total_len"]
raw_feat_cols = ["title_len_raw", "body_len_raw", "answer_len_raw", "total_len_raw"]

log_means = num_feat_train_raw[log_feat_cols].mean()
log_stds = num_feat_train_raw[log_feat_cols].std(ddof=0).replace(0, 1)

log_feat_train = (num_feat_train_raw[log_feat_cols] - log_means) / log_stds
log_feat_test = (num_feat_test_raw[log_feat_cols] - log_means) / log_stds

raw_means = num_feat_train_raw[raw_feat_cols].mean()
raw_stds = num_feat_train_raw[raw_feat_cols].std(ddof=0).replace(0, 1)

raw_feat_train = (num_feat_train_raw[raw_feat_cols] - raw_means) / raw_stds
raw_feat_test = (num_feat_test_raw[raw_feat_cols] - raw_means) / raw_stds

extra_raw_train = pd.DataFrame(
    {
        "ans_to_total": num_feat_train_raw["answer_len_raw"]
        / (num_feat_train_raw["total_len_raw"] + 1),
        "title_to_body": num_feat_train_raw["title_len_raw"]
        / (num_feat_train_raw["body_len_raw"] + 1),
    }
)
extra_raw_test = pd.DataFrame(
    {
        "ans_to_total": num_feat_test_raw["answer_len_raw"]
        / (num_feat_test_raw["total_len_raw"] + 1),
        "title_to_body": num_feat_test_raw["title_len_raw"]
        / (num_feat_test_raw["body_len_raw"] + 1),
    }
)

extra_log_train = np.log1p(extra_raw_train)
extra_log_test = np.log1p(extra_raw_test)

extra_log_means = extra_log_train.mean()
extra_log_stds = extra_log_train.std(ddof=0).replace(0, 1)

extra_log_train_std = (extra_log_train - extra_log_means) / extra_log_stds
extra_log_test_std = (extra_log_test - extra_log_means) / extra_log_stds

train_inter = np.column_stack(
    [
        log_feat_train["title_len"].values * log_feat_train["body_len"].values,
        log_feat_train["title_len"].values * log_feat_train["answer_len"].values,
        log_feat_train["body_len"].values * log_feat_train["answer_len"].values,
    ]
)

test_inter = np.column_stack(
    [
        log_feat_test["title_len"].values * log_feat_test["body_len"].values,
        log_feat_test["title_len"].values * log_feat_test["answer_len"].values,
        log_feat_test["body_len"].values * log_feat_test["answer_len"].values,
    ]
)

X_train = np.hstack(
    [
        np.ones((log_feat_train.shape[0], 1)),  # intercept
        log_feat_train.values,  # log‑standardised lengths
        raw_feat_train.values,  # raw‑standardised lengths
        extra_log_train_std.values,  # new ratio features
        train_inter,  # interaction terms
        category_dummies_train.values,  # category one‑hot
    ]
)
X_test = np.hstack(
    [
        np.ones((log_feat_test.shape[0], 1)),
        log_feat_test.values,
        raw_feat_test.values,
        extra_log_test_std.values,
        test_inter,
        category_dummies_test.values,
    ]
)

lambda_reg = 0.1
coeffs = {}
I = np.eye(X_train.shape[1])
XtX = X_train.T @ X_train + lambda_reg * I
XtX_inv = np.linalg.pinv(XtX)

for col in target_cols:
    y = train[col].values
    Xty = X_train.T @ y
    coeff = XtX_inv @ Xty
    coeffs[col] = coeff

lin_pred = pd.DataFrame({"qa_id": test["qa_id"]})
for col in target_cols:
    pred = X_test @ coeffs[col]
    pred = np.clip(pred, 0.0, 1.0)
    lin_pred[col] = pred

submission = test[["qa_id", "category"]].merge(
    category_means, on="category", how="left"
)
for col in target_cols:
    submission[col].fillna(overall_means[col], inplace=True)

weight_lin = 0.80
weight_cat = 0.20
for col in target_cols:
    submission[col] = weight_lin * lin_pred[col] + weight_cat * submission[col]
    submission[col] = np.clip(submission[col], 0.0, 1.0)

submission = submission.drop(columns=["category"])




## === cell 3
output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission file written to {output_path}")
