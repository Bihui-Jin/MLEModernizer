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

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.93441

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import string
import re
from statistics import mean
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

pd.options.display.float_format = "{:,.3f}".format

STOP_WORDS = ENGLISH_STOP_WORDS

_CLEAN_RE = re.compile(rf"[{re.escape(string.punctuation)}0-9]")


def clean_series(series: pd.Series) -> pd.Series:
    """
    Vectorized text cleaning:
    - strip punctuation & digits,
    - lower‑case,
    - collapse whitespace.
    Stop‑words are removed later by CountVectorizer.
    """
    return (
        series.str.replace(_CLEAN_RE, " ", regex=True)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def locate_file(*candidates):
    """Return the first existing path among candidates."""
    for cand in candidates:
        p = Path(cand)
        if p.is_file():
            return str(p)
    raise FileNotFoundError(f"None of the candidate files exist: {candidates}")




## === cell 1
train_path = locate_file(
    "kaggle/data/jigsaw-toxic-comment-classification-challenge/train.csv",
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv",
    "data/jigsaw-toxic-comment-classification-challenge/train.csv",
)
test_path = locate_file(
    "kaggle/data/jigsaw-toxic-comment-classification-challenge/test.csv",
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv",
    "data/jigsaw-toxic-comment-classification-challenge/test.csv",
)
sample_sub_path = locate_file(
    "kaggle/data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv",
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv",
    "data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv",
)

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_sub_path)

print(train_df.shape, test_df.shape, sample_submission.shape)



## === cell 2
X = train_df["comment_text"]
y = train_df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=123, shuffle=True
)

X_train_clean = clean_series(X_train)
X_val_clean = clean_series(X_val)
test_clean = clean_series(test_df["comment_text"])



## === cell 3
vectorizer = CountVectorizer(max_features=5000, stop_words=STOP_WORDS)
X_train_dtm = vectorizer.fit_transform(X_train_clean)
X_val_dtm = vectorizer.transform(X_val_clean)

print("DTM shapes:", X_train_dtm.shape, X_val_dtm.shape)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
InvalidParameterError                     Traceback (most recent call last)
/tmp/ipykernel_11/1631099708.py in <cell line: 0>()
      1 # Let CountVectorizer drop stop‑words; this matches the prior manual removal.
      2 vectorizer = CountVectorizer(max_features=5000, stop_words=STOP_WORDS)
----> 3 X_train_dtm = vectorizer.fit_transform(X_train_clean)
      4 X_val_dtm = vectorizer.transform(X_val_clean)
      5 

/usr/local/lib/python3.11/dist-packages/sklearn/feature_extraction/text.py in fit_transform(self, raw_documents, y)
   1367             )
   1368 
-> 1369         self._validate_params()
   1370         self._validate_ngram_range()
   1371         self._warn_for_unused_params()

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_params(self)
    598         accepted constraints.
    599         """
--> 600         validate_parameter_constraints(
    601             self._parameter_constraints,
    602             self.get_params(deep=False),

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_param_validation.py in validate_parameter_constraints(parameter_constraints, params, caller_name)
     95                 )
     96 
---> 97             raise InvalidParameterError(
     98                 f"The {param_name!r} parameter of {caller_name} must be"
     99                 f" {constraints_str}. Got {param_val!r} instead."

InvalidParameterError: The 'stop_words' parameter of CountVectorizer must be a str among {'english'}, an instance of 'list' or None. Got frozenset({'could', 'someone', 'is', 'were', 'whereupon', 'toward', 'yourselves', 'among', 'are', 'due', 'beforehand', 'much', 'somewhere', 'which', 'further', 'not', 'whence', 'empty', 'forty', 'therein', 'nothing', 'me', 'off', 'we', 'formerly', 'he', 'ltd', 'his', 'over', 'so', 'across', 'the', 'indeed', 'nobody', 'de', 'take', 'out', 'whatever', 'during', 'yet', 'any', 'between', 'mostly', 'mine', 'whole', 'fifteen', 'your', 'latter', 'besides', 'at', 'such', 'sixty', 'was', 'although', 'myself', 'though', 'hundred', 'made', 'either', 'none', 'side', 'hereby', 'moreover', 'there', 'also', 'hereafter', 'latterly', 'ourselves', 'an', 'in', 'nine', 'within', 'thence', 'system', 'yours', 'four', 'because', 'before', 'behind', 'find', 'third', 'most', 'may', 'too', 'once', 'rather', 'very', 'except', 'has', 'neither', 'well', 'a', 'whither', 'twenty', 'ie', 'move', 'together', 'detail', 'her', 'through', 'twelve', 'co', 'into', 'they', 'elsewhere', 'former', 'herein', 'hasnt', 'often', 'she', 'their', 'sometimes', 'eleven', 'cant', 'thereby', 'will', 'them', 'itself', 'whose', 'became', 'who', 'must', 'already', 'whenever', 'you', 'next', 'where', 'namely', 'everyone', 'or', 'than', 'whom', 'somehow', 'that', 'noone', 'seem', 'beyond', 'give', 'perhaps', 'therefore', 'never', 'those', 'our', 'wherever', 'but', 'about', 'down', 'it', 'see', 'seems', 'describe', 'anyhow', 'else', 'no', 'can', 'herself', 'top', 'all', 'put', 'becomes', 'please', 'first', 'from', 'mill', 'am', 'by', 'done', 'get', 'us', 'interest', 'same', 'two', 'many', 'as', 'yourself', 'call', 'keep', 'had', 'fire', 'go', 'on', 'should', 'how', 'another', 'and', 'then', 'throughout', 'wherein', 'fifty', 'when', 'everything', 'front', 'some', 'full', 'nevertheless', 'couldnt', 'however', 'since', 'with', 'themselves', 'hers', 'last', 'every', 'become', 'eg', 'cry', 'more', 'thereafter', 'inc', 'show', 'nor', 'be', 'bill', 'seeming', 'afterwards', 'this', 'whereby', 'something', 'for', 'below', 'seemed', 'himself', 'via', 'fill', 'hence', 'around', 'nowhere', 'upon', 'only', 'thereupon', 'towards', 'thin', 'i', 'of', 'serious', 'each', 'found', 'ten', 'bottom', 'been', 'hereupon', 're', 'one', 'beside', 'amongst', 'whether', 'always', 'until', 'up', 'while', 'least', 'what', 'whereas', 'whoever', 'why', 'cannot', 'part', 'con', 'do', 'my', 'whereafter', 'otherwise', 'less', 'sincere', 'amount', 'etc', 'enough', 'its', 'onto', 'several', 'under', 'still', 'five', 'anyone', 'anything', 'others', 'against', 'back', 'have', 'after', 'without', 'these', 'being', 'thru', 'name', 'other', 'ever', 'ours', 'if', 'meanwhile', 'above', 'eight', 'own', 'almost', 'thick', 'anywhere', 'becoming', 'sometime', 'now', 'alone', 'amoungst', 'would', 'along', 'both', 'everywhere', 'un', 'anyway', 'again', 'even', 'might', 'to', 'thus', 'few', 'per', 'six', 'him', 'here', 'three'}) instead.

## === cell 4
nb_model = MultiOutputClassifier(MultinomialNB(), n_jobs=-1).fit(X_train_dtm, y_train)
lr_model = MultiOutputClassifier(
    LogisticRegression(class_weight="balanced", max_iter=3000, solver="saga", n_jobs=-1)
).fit(X_train_dtm, y_train)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2887635024.py in <cell line: 0>()
      1 # Parallelize over the six output classifiers.
----> 2 nb_model = MultiOutputClassifier(MultinomialNB(), n_jobs=-1).fit(X_train_dtm, y_train)
      3 lr_model = MultiOutputClassifier(
      4     LogisticRegression(class_weight="balanced", max_iter=3000, solver="saga", n_jobs=-1)
      5 ).fit(X_train_dtm, y_train)

NameError: name 'X_train_dtm' is not defined

## === cell 5
def calculate_roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> list:
    """Return list of ROC‑AUC for each column."""
    return [roc_auc_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]


y_val_np = y_val.to_numpy()

for model, name in [(nb_model, "NaiveBayes"), (lr_model, "LogisticRegression")]:
    probs = np.transpose(np.array(model.predict_proba(X_val_dtm))[:, :, 1])
    mean_auc = mean(calculate_roc_auc(y_val_np, probs))
    print(f"{name} Mean AUC: {mean_auc:.4f}")



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2964746807.py in <cell line: 0>()
      6 y_val_np = y_val.to_numpy()
      7 
----> 8 for model, name in [(nb_model, "NaiveBayes"), (lr_model, "LogisticRegression")]:
      9     probs = np.transpose(np.array(model.predict_proba(X_val_dtm))[:, :, 1])
     10     mean_auc = mean(calculate_roc_auc(y_val_np, probs))

NameError: name 'nb_model' is not defined

## === cell 6
X_test_dtm = vectorizer.transform(test_clean)
test_probs = np.transpose(np.array(lr_model.predict_proba(X_test_dtm))[:, :, 1])

submission = pd.DataFrame(
    data=test_probs,
    columns=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
)
submission.insert(0, "id", test_df["id"])

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, shape: {submission.shape}")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/127043553.py in <cell line: 0>()
----> 1 X_test_dtm = vectorizer.transform(test_clean)
      2 test_probs = np.transpose(np.array(lr_model.predict_proba(X_test_dtm))[:, :, 1])
      3 
      4 submission = pd.DataFrame(
      5     data=test_probs,

/usr/local/lib/python3.11/dist-packages/sklearn/feature_extraction/text.py in transform(self, raw_documents)
   1428                 "Iterable over raw text documents expected, string object received."
   1429             )
-> 1430         self._check_vocabulary()
   1431 
   1432         # use the same matrix-building strategy as fit_transform

/usr/local/lib/python3.11/dist-packages/sklearn/feature_extraction/text.py in _check_vocabulary(self)
    508             self._validate_vocabulary()
    509             if not self.fixed_vocabulary_:
--> 510                 raise NotFittedError("Vocabulary not fitted or provided")
    511 
    512         if len(self.vocabulary_) == 0:

NotFittedError: Vocabulary not fitted or provided
