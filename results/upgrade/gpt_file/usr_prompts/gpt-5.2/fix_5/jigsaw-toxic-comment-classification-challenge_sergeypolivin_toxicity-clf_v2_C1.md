# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.12

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
seaborn==0.12.2
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

# 5. Code solution

## === cell 0
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import nltk
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

DATA_DIR = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/"
OUTPUT_DIR = "/kaggle/working/"
RANDOM_STATE = 42

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

try:
    nltk.download("stopwords", quiet=True)
    stopwords = list(nltk_stopwords.words("english"))
except Exception:
    stopwords = None  # let TfidfVectorizer handle default if needed




## === cell 1
os.listdir(DATA_DIR)




## === cell 2
os.listdir(OUTPUT_DIR)




## === cell 3
def unpack_zipfile(filename):
    """Unpacks zip-file by name from DATA_DIR to OUTPUT_DIR."""
    try:
        shutil.unpack_archive(
            filename=DATA_DIR + filename,
            extract_dir=OUTPUT_DIR,
            format="zip",
        )
    except Exception as e:
        print(e)
    else:
        print(f"Archive file '{filename}' has been unpacked successfully.")




## === cell 4
unpack_zipfile(filename="train.csv.zip")
unpack_zipfile(filename="test.csv.zip")
unpack_zipfile(filename="sample_submission.csv.zip")




## === cell 5
os.listdir(OUTPUT_DIR)




## === cell 6
def _find_file_recursive(root_dir, filename):
    for r, _, files in os.walk(root_dir):
        if filename in files:
            return os.path.join(r, filename)
    return None


def read_csv_fallback(name):
    out_path = os.path.join(OUTPUT_DIR, name)
    in_path = os.path.join(DATA_DIR, name)
    if os.path.exists(out_path):
        return pd.read_csv(out_path)
    if os.path.exists(in_path):
        return pd.read_csv(in_path)

    found = _find_file_recursive(OUTPUT_DIR, name)
    if found is not None:
        return pd.read_csv(found)

    found = _find_file_recursive(DATA_DIR, name)
    if found is not None:
        return pd.read_csv(found)

    raise FileNotFoundError(f"Could not locate {name} under {OUTPUT_DIR} or {DATA_DIR}")


train_df = read_csv_fallback("train.csv")
test_df = read_csv_fallback("test.csv")
sample_sub = read_csv_fallback("sample_submission.csv")




## === cell 7
test_labels = None
test_labels_info = None




## === cell 8
train_df.sample(n=5, random_state=RANDOM_STATE)




## === cell 9
train_df.info()




## === cell 10
cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

df = pd.Series(dtype="int64")
for col in cols:
    temp = int(train_df[col].value_counts().get(1, 0))
    df.loc[col] = temp




## === cell 11
df.sort_values(ascending=False).plot(kind="bar")
plt.title("Labels occurrences (train set)", fontsize=15)
plt.xlabel("Classes (comment toxicity degree)")
plt.ylabel("Number of examples")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




## === cell 12
test_df.sample(n=5, random_state=RANDOM_STATE)




## === cell 13
test_df.info()




## === cell 14
corpus_train = train_df["comment_text"].values.astype("U")
corpus_train[:5]




## === cell 15
target_train = train_df[cols].values
target_train[:5]




## === cell 16
corpus_test = test_df["comment_text"].values.astype("U")
corpus_test[:5]




## === cell 17
vectorizer = TfidfVectorizer(
    stop_words=stopwords,
    strip_accents="unicode",
    lowercase=True,
    ngram_range=(1, 2),
    analyzer="word",
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
)

vectorizer_char = TfidfVectorizer(
    strip_accents="unicode",
    lowercase=True,
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
)




## === cell 18
from scipy.sparse import hstack

features_train_word = vectorizer.fit_transform(corpus_train)
features_test_word = vectorizer.transform(corpus_test)

features_train_char = vectorizer_char.fit_transform(corpus_train)
features_test_char = vectorizer_char.transform(corpus_test)

features_train = hstack([features_train_word, features_train_char]).tocsr()
features_test = hstack([features_test_word, features_test_char]).tocsr()

features_train.shape, features_test.shape




## === cell 19
base_estimator = LogisticRegression(
    class_weight="balanced",
    max_iter=2000,
    solver="saga",
    random_state=RANDOM_STATE,
    n_jobs=1,  # keep deterministic per-estimator; OVR parallelism handled by OneVsRestClassifier
)

classifier = OneVsRestClassifier(
    estimator=base_estimator,
    n_jobs=-1,
)




## === cell 20
classifier.fit(features_train, target_train)




## === cell 21
test_ids = test_df["id"].values.astype("U")
test_ids[:5]




## === cell 22
try:
    idx_tr, idx_va = train_test_split(
        np.arange(features_train.shape[0]),
        test_size=0.1,
        random_state=RANDOM_STATE,
    )
    X_tr_vec = features_train[idx_tr]
    X_va_vec = features_train[idx_va]
    y_tr = target_train[idx_tr]
    y_va = target_train[idx_va]

    clf_tmp = OneVsRestClassifier(
        estimator=LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="saga",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        n_jobs=-1,
    )
    clf_tmp.fit(X_tr_vec, y_tr)
    va_proba = clf_tmp.predict_proba(X_va_vec)

    aucs = []
    for i, c in enumerate(cols):
        if len(np.unique(y_va[:, i])) < 2:
            continue
        aucs.append(roc_auc_score(y_va[:, i], va_proba[:, i]))
    if len(aucs) > 0:
        print(f"Validation mean ROC AUC (sanity-check, split): {np.mean(aucs):.5f}")
except Exception as e:
    print("Sanity-check skipped due to:", repr(e))




## === cell 23
predictions_train = classifier.predict(features_train)
f1_micro = f1_score(predictions_train, target_train, average="micro")
f1_macro = f1_score(predictions_train, target_train, average="macro")
f1_weighted = f1_score(predictions_train, target_train, average="weighted")
print(f"F1-score (micro): {f1_micro:.4f}")
print(f"F1-score (macro): {f1_macro:.4f}")
print(f"F1-score (weighted): {f1_weighted:.4f}")




## === cell 24
proba_predictions_test = classifier.predict_proba(features_test)




## === cell 25
sample_sub.head()




## === cell 26
submission = pd.DataFrame(
    proba_predictions_test,
    columns=cols,
)
submission.insert(0, "id", test_df["id"].values)

submission = submission[sample_sub.columns]

submission.head()




## === cell 27
sample_sub.info()




## === cell 28
submission.info()




## === cell 29
out_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(out_path, index=False)
print(f"The submission has been successfully saved to: {out_path}")




## === cell 30
with open(out_path, "r", encoding="utf-8") as f:
    for _ in range(5):
        print(f.readline().rstrip("\n"))

## --- ERROR in outputing the csv:
Invalid submission: Submission is missing the following columns: {'threat', 'toxic', 'identity_hate', 'severe_toxic', 'obscene', 'insult'}
