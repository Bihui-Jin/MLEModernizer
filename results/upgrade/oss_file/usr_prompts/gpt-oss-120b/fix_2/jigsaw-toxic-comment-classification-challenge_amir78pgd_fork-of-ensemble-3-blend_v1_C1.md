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

3.8

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
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

0.986512541976308

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'I make the script robust by loading each external prediction file inside a try‑except block; any missing files are simply skipped instead of raising an error. If no external predictions are available, the code falls back to a baseline that uses the overall mean label frequencies from the training data. The ensemble (or baseline) is then merged with the test IDs and saved as `submission.csv`, guaranteeing a valid CSV output without changing the original modelling logic.'

# 9. Code solution

## === cell 0
import os, numpy as np, pandas as pd

print("input dirs:", os.listdir("../input"))



## === cell 1
pred_files = [
    "../input/module-12-attn-bi/10fold_attn_post_am (1).csv",
    "../input/module-10-dpcnn/10fold_dpcnn_test (1).csv",
    "../input/module-1-nbsvm/Module_1_submission.csv",
    "../input/module-2-bilstm-f-pl/Module_2_submission.csv",
    "../input/module-7-capsule-gru/Module_7_submission.csv",
    "../input/module-6-mish/Mish_submission.csv",
    "../input/module-3-bilstm-g-tta/Module_3_submission.csv",
    "../input/module-11-capsule/10fold_capsule_am.csv",
    "../input/module-13-bilstm-f/Module_13_submission.csv",
    "../input/module-9-dmcnn/10fold_dmcnn_am.csv",
    "../input/module-4-dehyph/Module_4_submission.csv",
    "../input/module-5-bi-lstm-duembd-pl/Module_5_submission.csv",
    "../input/module-8-bi-lstm-pp/10fold_lstmpp_am (1).csv",
]

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

pred_dfs = []
for fp in pred_files:
    try:
        df = pd.read_csv(fp)
        if set(label_cols).issubset(df.columns):
            pred_dfs.append(df[[*label_cols]])
        else:
            print(
                f"Warning: file {fp} does not contain required label columns, skipped."
            )
    except FileNotFoundError:
        print(f"File not found (skipped): {fp}")
    except Exception as e:
        print(f"Error loading {fp}: {e}")



## === cell 2
train_path = "../input/jigsaw-toxic-comment-classification-challenge/train.csv"
test_path = "../input/jigsaw-toxic-comment-classification-challenge/test.csv"

test_df = pd.read_csv(test_path, usecols=["id"])

if pred_dfs:  # at least one external prediction is available
    avg_pred = pred_dfs[0].astype(np.float64).copy()
    for df in pred_dfs[1:]:
        avg_pred += df.astype(np.float64)
    avg_pred /= len(pred_dfs)
    submission = pd.concat(
        [test_df.reset_index(drop=True), avg_pred.reset_index(drop=True)], axis=1
    )
else:
    train_df = pd.read_csv(train_path, usecols=label_cols)
    mean_probs = train_df.mean()
    avg_pred = pd.DataFrame([mean_probs] * len(test_df), columns=label_cols)
    submission = pd.concat(
        [test_df.reset_index(drop=True), avg_pred.reset_index(drop=True)], axis=1
    )



## === cell 3
submission = submission[["id"] + label_cols]

output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}, shape: {submission.shape}")
