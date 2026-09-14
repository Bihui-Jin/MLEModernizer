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

3.14

# 3. Installed packages

geopandas==0.14.4
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

0.9856857135374236

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I adjust the script to (1) use the correct absolute path for the sample submission file, (2) blend whatever model prediction files are actually present instead of requiring all five, and (3) renormalize the blending weights so the final probabilities stay in a valid range and a CSV is always written. This fixes the “no CSV generated” issue and lets the ensemble work with the available predictions, moving the score toward the target.'
- What this solution (achieved 0.5) has done: 'I align predictions by the `id` column instead of relying on row order, ensuring each model’s probabilities are added to the correct test rows. This prevents the previous mis‑alignment that yielded a flat 0.5 score. I also keep the weight normalization and clipping unchanged, so the ensemble still respects the original weighting scheme while producing a valid, better‑aligned submission.'

# 9. Code solution

## === cell 0
import pandas as pd
import os

model_paths = {
    "BART": "/kaggle/input/toxic-torii-v5-bart-ver1-result/submission.csv",
    "TF-IDF": "/kaggle/input/toxic-torii-v3-tfidf-ver1-result/submission.csv",
    "GRU": "/kaggle/input/toxic-torii-v8-gru-ver1-result/submission_gru.csv",
    "LSTM": "/kaggle/input/toxic-torii-v9-lstm-ver1-result/submission_lstm.csv",
    "FeatureEngineering": "/kaggle/input/toxic-torii-v7-featureengineering2-ver1-result/submission_andre_features_enhanced.csv",
}

print("ファイルを読み込んでいます...")


def load_submission(path, name):
    if os.path.exists(path):
        print(f" - {name}: 読み込み成功 ({path})")
        return pd.read_csv(path)
    else:
        print(f" - {name}: ファイルが見つかりません。パスを確認してください: {path}")
        return None


model_dfs = {}
for name, path in model_paths.items():
    df = load_submission(path, name)
    if df is not None:
        model_dfs[name] = df




## === cell 1
base_weights = {
    "BART": 0.70,
    "TF-IDF": 0.15,
    "GRU": 0.05,
    "LSTM": 0.05,
    "FeatureEngineering": 0.05,
}

available_weights = {name: w for name, w in base_weights.items() if name in model_dfs}
total_weight = sum(available_weights.values())
normalized_weights = {name: w / total_weight for name, w in available_weights.items()}

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]




## === cell 2
sample_submission_path = (
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv"
)
if not os.path.exists(sample_submission_path):
    sample_submission_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip"

submission_df = pd.read_csv(sample_submission_path).set_index("id")




## === cell 3
print("\n★利用可能なモデルをブレンドします")
blend_preds = pd.DataFrame(
    0.0, index=submission_df.index, columns=label_cols, dtype=float
)

for name, df in model_dfs.items():
    w = normalized_weights[name]
    model_df = df.set_index("id")
    aligned = model_df[label_cols].reindex(submission_df.index).fillna(0.0)
    blend_preds += aligned * w
    print(f" - {name}: 重み {w:.4f}")

blend_preds = blend_preds.clip(0.0, 1.0)

submission_df[label_cols] = blend_preds

output_path = "submission_final_ensemble.csv"
submission_df.reset_index().to_csv(output_path, index=False)
print(f"\n完了！ '{output_path}' を作成しました。")




## === cell 4
print(submission_df.head())
