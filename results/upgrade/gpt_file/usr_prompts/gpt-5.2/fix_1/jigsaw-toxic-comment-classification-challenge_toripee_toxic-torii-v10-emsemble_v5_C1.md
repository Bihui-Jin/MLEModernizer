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

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import os


path_bart = "/kaggle/input/toxic-torii-v5-bart-ver1-result/submission.csv"

path_tfidf = "/kaggle/input/toxic-torii-v3-tfidf-ver1-result/submission.csv"

path_gru = "/kaggle/input/toxic-torii-v8-gru-ver1-result/submission_gru.csv"

path_lstm = "/kaggle/input/toxic-torii-v9-lstm-ver1-result/submission_lstm.csv"

path_feature = "/kaggle/input/toxic-torii-v7-featureengineering2-ver1-result/submission_andre_features_enhanced.csv"

print("ファイルを読み込んでいます...")

def load_submission(path, name):
    if os.path.exists(path):
        print(f" - {name}: 読み込み成功 ({path})")
        return pd.read_csv(path)
    else:
        print(f" - {name}: ファイルが見つかりません。パスを確認してください: {path}")
        return None

df_bart = load_submission(path_bart, "BART")
df_tfidf = load_submission(path_tfidf, "TF-IDF")
df_gru = load_submission(path_gru, "GRU")
df_lstm = load_submission(path_lstm, "LSTM")
df_feature = load_submission(path_feature, "FeatureEngineering")

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
submission_df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip")

if df_bart is not None and df_tfidf is not None and df_gru is not None and df_lstm is not None and df_feature is not None:
    print("\n★5つのモデルをブレンドします")
    
    blend_preds = (df_bart[label_cols] * 0.70) + \
                  (df_tfidf[label_cols] * 0.15) + \
                  (df_gru[label_cols] * 0.05) + \
                  (df_lstm[label_cols] * 0.05) + \
                  (df_feature[label_cols] * 0.05)
    
    submission_df[label_cols] = blend_preds
    submission_df.to_csv('submission_final_ensemble.csv', index=False)
    print("完了！ 'submission_final_ensemble.csv' を作成しました。")
    
else:
    print("\nエラー: 全てのファイルが正しく読み込めませんでした。パスを修正してください。")

print(submission_df.head())
