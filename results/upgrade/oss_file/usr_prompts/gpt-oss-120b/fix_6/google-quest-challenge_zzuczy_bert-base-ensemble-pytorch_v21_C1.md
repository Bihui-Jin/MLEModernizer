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

0.25087

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved nan) has done: 'The script was failing early because the tokenizer and model paths were invalid, required packages were missing, and later steps depended on objects that were never created. To get a complete, runnable pipeline we replace the broken BERT preprocessing and model inference with a simple baseline: predict the average value of each target column from the training data. This guarantees predictions are in the required [0, 1] range, creates a correctly‑formatted `submission.csv`, and eliminates all import and runtime errors while keeping the overall workflow (data loading → prediction → submission) intact.'
- What this solution (achieved 0.18408) has done: 'I make the file loading robust by checking both the original relative path and the typical Kaggle “/kaggle/input” location, and replace the simple global‑mean baseline with a lightweight per‑category mean model: for each target column we use the mean value computed within the same `category` as the test row (falling back to the overall mean when a category is unseen). This adds modest, data‑driven variation while keeping the core logic unchanged, and should move the Spearman score closer to the target.'
- What this solution (achieved 0.23861) has done: 'The changes add simple numeric text‑length features (question title, body and answer lengths) and fit a cheap linear regression for each target column using these features.  
Predictions from the linear model are blended 50 % with the existing per‑category mean baseline, then clipped to the required [0, 1] range. This adds useful signal while preserving the original workflow and keeps the implementation lightweight.'
- What this solution (achieved 0.23352) has done: 'The changes add categorical one‑hot features to the linear regression (so the model can learn category‑specific effects) and increase the weight of the linear model in the final blend, which should raise the Spearman correlation toward the target while keeping the original simple pipeline intact.'
- What this solution (achieved 0.25087) has done: 'I keep the overall pipeline but improve the numeric text‑length features by applying a log‑1p transform and standardising them (zero‑mean, unit‑variance). This gives the linear regression a more stable basis and usually raises correlation. I also shift the blend toward the linear model (80 % linear + 20 % category mean) to let the better‑fitted predictor dominate while still retaining the safe category fallback.'

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
    feats = pd.DataFrame(
        {
            "title_len": np.log1p(title_len),
            "body_len": np.log1p(body_len),
            "answer_len": np.log1p(answer_len),
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

feat_means = num_feat_train_raw.mean()
feat_stds = num_feat_train_raw.std(ddof=0).replace(0, 1)  # avoid division by 0

num_feat_train = (num_feat_train_raw - feat_means) / feat_stds
num_feat_test = (num_feat_test_raw - feat_means) / feat_stds

X_train = np.hstack(
    [
        np.ones((num_feat_train.shape[0], 1)),  # intercept
        num_feat_train.values,  # standardized lengths
        category_dummies_train.values,  # category one‑hot
    ]
)
X_test = np.hstack(
    [
        np.ones((num_feat_test.shape[0], 1)),
        num_feat_test.values,
        category_dummies_test.values,
    ]
)

coeffs = {}
for col in target_cols:
    y = train[col].values
    coef, _, _, _ = np.linalg.lstsq(X_train, y, rcond=None)
    coeffs[col] = coef

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

weight_lin = 0.8
weight_cat = 0.2
for col in target_cols:
    submission[col] = weight_lin * lin_pred[col] + weight_cat * submission[col]
    submission[col] = np.clip(submission[col], 0.0, 1.0)

submission = submission.drop(columns=["category"])



## === cell 3
output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission file written to {output_path}")
