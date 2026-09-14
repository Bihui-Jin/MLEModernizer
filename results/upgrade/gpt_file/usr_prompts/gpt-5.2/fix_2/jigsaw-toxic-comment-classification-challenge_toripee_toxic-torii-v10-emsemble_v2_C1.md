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

0.9854984171236082

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'You’re currently not getting a Kaggle score because the notebook depends on five external `/kaggle/input/...` submissions that are not present in your provided environment, so it never produces a usable blended file. I keep your “blend multiple models’ submission.csv files” core logic, but make it robust by (1) auto-discovering those submissions if they exist, (2) aligning/merging by `id` to avoid row-order mismatches, and (3) falling back to a valid baseline submission (all 0.5s from `sample_submission.csv`) when some files are missing so you always get a valid `.csv` for scoring. This should move you from “Not yielded” to a real score (likely >0), which is necessarily closer to your target than having no score. The changes are minimal and only address execution + valid submission creation.'

# 9. Code solution

## === cell 0
import os
import glob
import pandas as pd


LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

SAMPLE_SUB_PATH = (
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv"
)
if not os.path.exists(SAMPLE_SUB_PATH):
    SAMPLE_SUB_PATH = "/kaggle/data/sample_submission.csv"

path_bart = "/kaggle/input/toxic-torii-v5-bart-ver1-result/submission.csv"
path_tfidf = "/kaggle/input/toxic-torii-v3-tfidf-ver1-result/submission.csv"
path_gru = "/kaggle/input/toxic-torii-v8-gru-ver1-result/submission_gru.csv"
path_lstm = "/kaggle/input/toxic-torii-v9-lstm-ver1-result/submission_lstm.csv"
path_feature = "/kaggle/input/toxic-torii-v7-featureengineering2-ver1-result/submission_andre_features_enhanced.csv"

print("ファイルを読み込んでいます...")


def _find_submission_like(path_hint: str):
    """
    Minimal robustness: if the hinted file doesn't exist, try to find a submission*.csv
    inside the hinted dataset folder (same parent), without changing the blending logic.
    """
    if os.path.exists(path_hint):
        return path_hint

    parent = os.path.dirname(path_hint)
    if os.path.isdir(parent):
        candidates = []
        exact = os.path.join(parent, "submission.csv")
        if os.path.exists(exact):
            candidates.append(exact)
        candidates.extend(sorted(glob.glob(os.path.join(parent, "submission*.csv"))))
        if candidates:
            return candidates[0]
    return None


def load_submission(path_hint, name):
    path = _find_submission_like(path_hint)
    if path and os.path.exists(path):
        try:
            df = pd.read_csv(path)
            missing = [c for c in (["id"] + LABEL_COLS) if c not in df.columns]
            if missing:
                print(
                    f" - {name}: 読み込み失敗（必要列が不足） missing={missing} ({path})"
                )
                return None
            print(f" - {name}: 読み込み成功 ({path}) shape={df.shape}")
            return df[["id"] + LABEL_COLS].copy()
        except Exception as e:
            print(f" - {name}: 読み込み失敗（例外） ({path}) error={e}")
            return None
    else:
        print(
            f" - {name}: ファイルが見つかりません。パスを確認してください: {path_hint}"
        )
        return None


df_bart = load_submission(path_bart, "BART")
df_tfidf = load_submission(path_tfidf, "TF-IDF")
df_gru = load_submission(path_gru, "GRU")
df_lstm = load_submission(path_lstm, "LSTM")
df_feature = load_submission(path_feature, "FeatureEngineering")

if not os.path.exists(SAMPLE_SUB_PATH):
    raise FileNotFoundError(f"sample_submission.csv not found at {SAMPLE_SUB_PATH}")

submission_df = pd.read_csv(SAMPLE_SUB_PATH)
assert (
    submission_df.columns.tolist() == ["id"] + LABEL_COLS
), "Unexpected sample_submission format"



## === cell 1


def _merge_on_id(base_ids: pd.DataFrame, pred_df: pd.DataFrame, name: str):
    """
    Left-merge predictions onto base ids, keeping order of sample_submission.
    Any missing ids get NaNs; we'll handle that downstream.
    """
    merged = base_ids[["id"]].merge(pred_df, on="id", how="left", validate="one_to_one")
    n_missing = merged[LABEL_COLS].isna().any(axis=1).sum()
    if n_missing > 0:
        print(
            f" ! {name}: {n_missing} rows have missing predictions after id-merge (will fallback to 0.5 for those rows)."
        )
    return merged


available = []
if df_bart is not None:
    available.append(("BART", df_bart))
if df_tfidf is not None:
    available.append(("TF-IDF", df_tfidf))
if df_gru is not None:
    available.append(("GRU", df_gru))
if df_lstm is not None:
    available.append(("LSTM", df_lstm))
if df_feature is not None:
    available.append(("FeatureEngineering", df_feature))

weights = {
    "BART": 0.50,
    "TF-IDF": 0.15,
    "GRU": 0.15,
    "LSTM": 0.15,
    "FeatureEngineering": 0.05,
}

if len(available) == 0:
    print("\nエラー: ブレンド用のsubmissionファイルが1つも読み込めませんでした。")
    print(
        "→ 有効な提出ファイルを作るため、sample_submission（全て0.5）のまま出力します。"
    )
    out_path = "submission_final_ensemble.csv"
    submission_df.to_csv(out_path, index=False)
    print(f"完了！ '{out_path}' を作成しました。")
else:
    print(f"\n★{len(available)}個のモデルをブレンドします（読み込めたもののみ使用）")

    w_sum = sum(weights[name] for name, _ in available)
    if w_sum <= 0:
        raise ValueError("Sum of available weights is non-positive, cannot blend.")

    blended = pd.DataFrame({"id": submission_df["id"]})
    for col in LABEL_COLS:
        blended[col] = 0.0

    for name, df in available:
        merged = _merge_on_id(submission_df, df, name)
        preds = merged[LABEL_COLS].fillna(0.5)
        w = weights[name] / w_sum  # renormalize only when some models missing
        for col in LABEL_COLS:
            blended[col] += preds[col].astype("float64") * w

    blended[LABEL_COLS] = blended[LABEL_COLS].clip(0.0, 1.0)

    submission_df[LABEL_COLS] = blended[LABEL_COLS].values
    out_path = "submission_final_ensemble.csv"
    submission_df.to_csv(out_path, index=False)
    print(f"完了！ '{out_path}' を作成しました。")

print(submission_df.head())
print("Saved submission:", os.path.abspath("submission_final_ensemble.csv"))
print("Rows:", len(submission_df), "Cols:", submission_df.shape[1])
