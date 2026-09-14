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

0.9854984171236082

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import pandas as pd
from pathlib import Path

base_paths = {
    "BART": "/kaggle/input/toxic-torii-v5-bart-ver1-result/submission.csv",
    "TF-IDF": "/kaggle/input/toxic-torii-v3-tfidf-ver1-result/submission.csv",
    "GRU": "/kaggle/input/toxic-torii-v8-gru-ver1-result/submission_gru.csv",
    "LSTM": "/kaggle/input/toxic-torii-v9-lstm-ver1-result/submission_lstm.csv",
    "FeatureEngineering": "/kaggle/input/toxic-torii-v7-featureengineering2-ver1-result/submission_andre_features_enhanced.csv",
}


def load_submission(path: str, name: str) -> pd.DataFrame | None:
    """Load a CSV (plain or zipped). Returns None if the file cannot be read."""
    p = Path(path)
    if not p.exists():
        print(f" - {name}: ファイルが見つかりません ({path})")
        return None
    try:
        if p.suffix == ".zip":
            return pd.read_csv(p, compression="zip")
        else:
            return pd.read_csv(p)
    except Exception as e:
        print(f" - {name}: 読み込み失敗 ({e})")
        return None


print("ファイルを読み込んでいます...")
df_bart = load_submission(base_paths["BART"], "BART")
df_tfidf = load_submission(base_paths["TF-IDF"], "TF-IDF")
df_gru = load_submission(base_paths["GRU"], "GRU")
df_lstm = load_submission(base_paths["LSTM"], "LSTM")
df_feature = load_submission(base_paths["FeatureEngineering"], "FeatureEngineering")

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

sample_path_candidates = [
    "../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip",
    "../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv",
    "sample_submission.csv.zip",
    "sample_submission.csv",
]
submission_df = None
for sp in sample_path_candidates:
    if os.path.exists(sp):
        try:
            submission_df = pd.read_csv(
                sp, compression="zip" if sp.endswith(".zip") else None
            )
            print(f"サンプル提出ファイルをロードしました: {sp}")
            break
        except Exception:
            continue
if submission_df is None:
    raise FileNotFoundError(
        "サンプル提出ファイルが見つかりませんでした。パスを確認してください。"
    )



## === cell 1
available_dfs = {
    "BART": df_bart,
    "TF-IDF": df_tfidf,
    "GRU": df_gru,
    "LSTM": df_lstm,
    "FeatureEngineering": df_feature,
}
orig_weights = {
    "BART": 0.50,
    "TF-IDF": 0.15,
    "GRU": 0.15,
    "LSTM": 0.15,
    "FeatureEngineering": 0.05,
}

valid_weights = {k: w for k, w in orig_weights.items() if available_dfs[k] is not None}
total_weight = sum(valid_weights.values())
if total_weight == 0:
    raise RuntimeError(
        "全ての予測ファイルの読み込みに失敗しました。少なくとも1つは必要です。"
    )

normalized_weights = {k: w / total_weight for k, w in valid_weights.items()}

print("\n★利用可能なモデルでブレンドします")
print("使用するモデルと重み:", normalized_weights)

blend_preds = pd.DataFrame(
    0, index=submission_df.index, columns=label_cols, dtype=float
)

for model_name, df in available_dfs.items():
    if df is None:
        continue
    weight = normalized_weights[model_name]
    blend_preds += df[label_cols] * weight

submission_df[label_cols] = blend_preds

output_path = Path("submission_final_ensemble.csv")
submission_df.to_csv(output_path, index=False)
print(f"\n完了！ '{output_path}' を作成しました。")

print(submission_df.head())

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/638754577.py in <cell line: 0>()
     20 total_weight = sum(valid_weights.values())
     21 if total_weight == 0:
---> 22     raise RuntimeError(
     23         "全ての予測ファイルの読み込みに失敗しました。少なくとも1つは必要です。"
     24     )

RuntimeError: 全ての予測ファイルの読み込みに失敗しました。少なくとも1つは必要です。
