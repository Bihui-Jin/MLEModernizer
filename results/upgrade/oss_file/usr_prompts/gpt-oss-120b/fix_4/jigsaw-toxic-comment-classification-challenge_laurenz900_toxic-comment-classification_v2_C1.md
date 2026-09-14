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

3.6

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
nltk==3.9.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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

0.6648

# 6. Current score

0.93043

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.93043) has done: 'The fix replaces the fragile regex‐extract approach with a safe count‑based method for detecting smileys, ensuring the new feature columns are created as integers. This resolves the ValueError during type conversion and the subsequent missing‑column errors, allowing the pipeline to run end‑to‑end and generate a complete submission CSV.'

# 9. Code solution

## === cell 0
import pathlib, pandas as pd, numpy as np
from scipy.special import expit
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score


def resolve_path(relative_path: str) -> pathlib.Path:
    """Return an existing Path for the given relative location, trying common Kaggle directories."""
    candidates = [
        pathlib.Path(relative_path),
        pathlib.Path("/kaggle/input") / relative_path,
        pathlib.Path("kaggle/input") / relative_path,
        pathlib.Path("data") / relative_path,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Unable to locate {relative_path}")


base_rel = "jigsaw-toxic-comment-classification-challenge"
train_path = resolve_path(f"{base_rel}/train.csv")
test_path = resolve_path(f"{base_rel}/test.csv")

df_train = pd.read_csv(train_path)
df_predict = pd.read_csv(test_path)

all_text = pd.concat([df_train["comment_text"], df_predict["comment_text"]])




## === cell 1
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["ex_mark"] = df["comment_text"].str.count("!").clip(0, 1)
    df["qu_mark"] = df["comment_text"].str.count("\\?").clip(0, 1)
    smileys_good = r"((:|;)-?(\)|P|D))"
    smileys_bad = r"((:|;)-?\'?(\())"
    df["smileys_good"] = (
        df["comment_text"].str.count(smileys_good).clip(0, 1).astype(int)
    )
    df["smileys_bad"] = df["comment_text"].str.count(smileys_bad).clip(0, 1).astype(int)
    return df


df_train = add_features(df_train)
df_predict = add_features(df_predict)



## === cell 2
vect = CountVectorizer(min_df=4, ngram_range=(1, 3), stop_words="english")
vect.fit(all_text)

X_train_text = vect.transform(df_train["comment_text"])
X_test_text = vect.transform(df_predict["comment_text"])

extra_train = df_train[["smileys_good", "smileys_bad", "ex_mark", "qu_mark"]].astype(
    "int64"
)
extra_test = df_predict[["smileys_good", "smileys_bad", "ex_mark", "qu_mark"]].astype(
    "int64"
)

train_features = hstack([X_train_text, extra_train])
predict_features = hstack([X_test_text, extra_test])

Y = df_train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]



## === cell 3
model = LinearSVC()
params = {"C": [1], "random_state": [0]}

Y_predicted = pd.DataFrame({"id": df_predict["id"]})
scores = []

for col in Y.columns:
    gs = GridSearchCV(model, params, scoring="roc_auc", cv=3, n_jobs=-1)
    gs.fit(train_features, Y[col])
    prob = expit(gs.decision_function(predict_features))
    Y_predicted[col] = prob
    scores.append(gs.best_score_)
    print(f"{col}: {gs.best_score_:.4f}")

print("mean score:", np.mean(scores))



## === cell 4
submission_path = "submission.csv"
Y_predicted.to_csv(submission_path, index=False)
