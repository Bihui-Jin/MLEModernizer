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

3.12

# 3. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
sklearn-pandas==2.2.0
tqdm==4.67.1

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

0.94991

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import re
import shutil
from collections import Counter

import numpy as np
import pandas as pd
from IPython.display import display
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
from tqdm import tqdm

DATA_DIR = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/"
OUTPUT_DIR = "/kaggle/working/"




## === cell 1
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




## === cell 2
unpack_zipfile(filename="train.csv.zip")
unpack_zipfile(filename="test.csv.zip")
unpack_zipfile(filename="test_labels.csv.zip")



## === cell 3
train_df = pd.read_csv(OUTPUT_DIR + "train.csv")
test_df = pd.read_csv(OUTPUT_DIR + "test.csv")



## === cell 4
train_df.head()



## === cell 5
test_df.head()



## === cell 6
for df in (train_df, test_df):
    df["comment_text_preprocessed"] = df["comment_text"].str.lower()



## === cell 7
cols = ["comment_text", "comment_text_preprocessed"]
display(train_df[cols].sample(5))
display(test_df[cols].sample(5))



## === cell 8
eng_stopwords = set(nltk_stopwords.words("english"))
eng_stopwords.update(["i'm", "that's", "can't"])
eng_stopwords




## === cell 9
def clear_stopwords(comment_text, stopwords=eng_stopwords):
    """Removes stopwords from the commentary text."""
    comment_text_cleared = [
        word for word in str(comment_text).split() if word not in stopwords
    ]

    return " ".join(comment_text_cleared)




## === cell 10
train_text_2 = train_df["comment_text_preprocessed"].iloc[2]

print("Inp:\n\n{}\n".format(train_text_2))
print("Out:\n\n{}".format(clear_stopwords(train_text_2)))



## === cell 11
for df in (train_df, test_df):
    df["comment_text_preprocessed"] = df["comment_text_preprocessed"].apply(
        lambda comment_text: clear_stopwords(comment_text)
    )



## === cell 12
display(train_df[cols].sample(5))
display(test_df[cols].sample(5))



## === cell 13
word_counter = Counter()
for comment_text in train_df["comment_text_preprocessed"].values:
    for word in comment_text.split():
        word_counter[word] += 1

word_counter.most_common(10)



## === cell 14
freq_words = set([word for (word, word_count) in word_counter.most_common(10)])
freq_words




## === cell 15
def clear_freqwords(comment_text, freqwords=freq_words):
    """Removes top-10 frequent words."""

    comment_text_cleared = [
        word for word in str(comment_text).split() if word not in freq_words
    ]

    return " ".join(comment_text_cleared)




## === cell 16
for df in (train_df, test_df):
    df["comment_text_preprocessed"] = df["comment_text_preprocessed"].apply(
        lambda comment_text: clear_freqwords(comment_text)
    )



## === cell 17
display(train_df[cols].sample(5))
display(test_df[cols].sample(5))



## === cell 18
rare_words_num = 10
rare_words = set(
    [
        word
        for (word, word_count) in word_counter.most_common()[: -rare_words_num - 1 : -1]
    ]
)
rare_words




## === cell 19
def clear_rarewords(comment_text, rarewords=rare_words):
    """Removes top-10 rarest words."""

    comment_text_cleared = [
        word for word in str(comment_text).split() if word not in rare_words
    ]

    return " ".join(comment_text_cleared)




## === cell 20
for df in (train_df, test_df):
    df["comment_text_preprocessed"] = df["comment_text_preprocessed"].apply(
        lambda comment_text: clear_rarewords(comment_text)
    )



## === cell 21
display(train_df[cols].sample(5))
display(test_df[cols].sample(5))




## === cell 22
def clear_urls(comment_text):
    """Clears the comment text from URLs."""

    url_regex_pattern = re.compile(r"https?://\S+|www\.\S+")

    return url_regex_pattern.sub(r"", comment_text)




## === cell 23
train_text_900 = train_df["comment_text_preprocessed"].iloc[-900]

print("Inp:\n\n{}\n".format(train_text_900))
print("Out:\n\n{}".format(clear_urls(train_text_900)))



## === cell 24
for df in (train_df, test_df):
    df["comment_text_preprocessed"] = df["comment_text_preprocessed"].apply(
        lambda comment_text: clear_urls(comment_text)
    )



## === cell 25
display(train_df[cols].sample(5))
display(test_df[cols].sample(5))



## === cell 26
regex = re.compile(r"[a-zA-Z]+")


def leave_words_only(comment_text, regex=regex):
    """Removes non-word inclusions."""

    return " ".join(regex.findall(comment_text))




## === cell 27
train_text = train_df["comment_text_preprocessed"].iloc[-1]

print("Inp:\n\n{}\n".format(train_text))
print("Out:\n\n{}".format(leave_words_only(train_text)))



## === cell 28
for df in (train_df, test_df):
    df["comment_text_preprocessed"] = df["comment_text_preprocessed"].apply(
        lambda comment_text: leave_words_only(comment_text)
    )



## === cell 29
display(train_df[cols].sample(5))
display(test_df[cols].sample(5))



## === cell 30
other_eng_stopwords = [word for word in eng_stopwords if "'" in word]
other_eng_stopwords



## === cell 31
other_eng_stopwords = [word.replace("'", "") for word in other_eng_stopwords]
other_eng_stopwords



## === cell 32
word_counter = Counter()
for comment_text in train_df["comment_text_preprocessed"].values:
    for word in comment_text.split():
        word_counter[word] += 1

word_counter.most_common(100)



## === cell 33
eng_stopwords.update(
    [
        "utc",
        "eg",
        "jpg",
        "didnt",
        "th",
        "oh",
        "im",
        "cant",
        "wp",
        "hi",
    ]
)
eng_stopwords.update(other_eng_stopwords)
eng_stopwords



## === cell 34
for df in (train_df, test_df):
    df["comment_text_preprocessed"] = df["comment_text_preprocessed"].apply(
        lambda comment_text: clear_stopwords(
            comment_text,
            stopwords=eng_stopwords,
        )
    )



## === cell 35
for df in (train_df, test_df):
    df["comment_text_preprocessed"] = df["comment_text_preprocessed"].apply(
        lambda comment_text: clear_freqwords(comment_text)
    )



## === cell 36
display(train_df[cols].sample(5))
display(test_df[cols].sample(5))



## === cell 37
target_cols = train_df.columns[2:]  # ['toxic', 'severe_toxic', ..., 'identity_hate']
target_train = train_df[target_cols].values
target_train[:5]



## === cell 38
corpus_train = train_df["comment_text_preprocessed"].values.astype("U")
corpus_train[:5]



## === cell 39
corpus_test = test_df["comment_text_preprocessed"].values.astype("U")
corpus_test[:5]



## === cell 40
vectorizer = TfidfVectorizer(
    max_features=1700,
    min_df=0.0011,
    max_df=0.35,
    norm="l2",
)



## === cell 41
features_train = vectorizer.fit_transform(corpus_train)
features_train.shape



## === cell 42
features_test = vectorizer.transform(corpus_test)
features_test.shape



## === cell 43
base_estimator = LogisticRegression(
    class_weight="balanced",
    max_iter=10000,
    multi_class="multinomial",
    C=0.009,
    penalty="l2",
    n_jobs=-1,
)



## === cell 44
chains = [
    ClassifierChain(
        base_estimator=base_estimator,
        order="random",
        random_state=i,
    )
    for i in range(10)
]

for i in tqdm(range(len(chains))):
    chains[i].fit(features_train, target_train)
print()



## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2827043506.py in <cell line: 0>()
      9 
     10 for i in tqdm(range(len(chains))):
---> 11     chains[i].fit(features_train, target_train)
     12 print()
     13 

/usr/local/lib/python3.11/dist-packages/sklearn/multioutput.py in fit(self, X, Y)
    811         self._validate_params()
    812 
--> 813         super().fit(X, Y)
    814         self.classes_ = [
    815             estimator.classes_ for chain_idx, estimator in enumerate(self.estimators_)

/usr/local/lib/python3.11/dist-packages/sklearn/multioutput.py in fit(self, X, Y, **fit_params)
    607             Y_pred_chain = Y[:, self.order_]
    608             if sp.issparse(X):
--> 609                 X_aug = sp.hstack((X, Y_pred_chain), format="lil")
    610                 X_aug = X_aug.tocsr()
    611             else:

/usr/local/lib/python3.11/dist-packages/scipy/sparse/_construct.py in hstack(blocks, format, dtype)
    754         return _block([blocks], format, dtype)
    755     else:
--> 756         return _block([blocks], format, dtype, return_spmatrix=True)
    757 
    758 

/usr/local/lib/python3.11/dist-packages/scipy/sparse/_construct.py in _block(blocks, format, dtype, return_spmatrix)
    959         for j in range(N):
    960             if blocks[i,j] is not None:
--> 961                 A = coo_array(blocks[i,j])
    962                 blocks[i,j] = A
    963                 block_mask[i,j] = True

/usr/local/lib/python3.11/dist-packages/scipy/sparse/_coo.py in __init__(self, arg1, shape, dtype, copy, maxprint)
     93                 self.coords = tuple(idx.astype(index_dtype, copy=False)
     94                                      for idx in coords)
---> 95                 self.data = getdata(M[coords], copy=copy, dtype=dtype)
     96                 self.has_canonical_format = True
     97 

/usr/local/lib/python3.11/dist-packages/scipy/sparse/_sputils.py in getdata(obj, dtype, copy)
    148     # Defer to getdtype for checking that the dtype is OK.
    149     # This is called for the validation only; we don't need the return value.
--> 150     getdtype(data.dtype)
    151     return data
    152 

/usr/local/lib/python3.11/dist-packages/scipy/sparse/_sputils.py in getdtype(dtype, a, default)
    135     if newdtype not in supported_dtypes:
    136         supported_dtypes_fmt = ", ".join(t.__name__ for t in supported_dtypes)
--> 137         raise ValueError(f"scipy.sparse does not support dtype {newdtype.name}. "
    138                          f"The only supported types are: {supported_dtypes_fmt}.")
    139     return newdtype

ValueError: scipy.sparse does not support dtype object. The only supported types are: bool_, int8, uint8, int16, uint16, int32, uint32, int64, uint64, longlong, ulonglong, float32, float64, longdouble, complex64, complex128, clongdouble.

## === cell 45
predictions = np.array([chain.predict_proba(features_test) for chain in chains])
proba_predictions_test = predictions.mean(axis=0)
proba_predictions_test



## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/2642976640.py in <cell line: 0>()
----> 1 predictions = np.array([chain.predict_proba(features_test) for chain in chains])
      2 proba_predictions_test = predictions.mean(axis=0)
      3 proba_predictions_test
      4 

/tmp/ipykernel_11/2642976640.py in <listcomp>(.0)
----> 1 predictions = np.array([chain.predict_proba(features_test) for chain in chains])
      2 proba_predictions_test = predictions.mean(axis=0)
      3 proba_predictions_test
      4 

/usr/local/lib/python3.11/dist-packages/sklearn/multioutput.py in predict_proba(self, X)
    840             else:
    841                 X_aug = np.hstack((X, previous_predictions))
--> 842             Y_prob_chain[:, chain_idx] = estimator.predict_proba(X_aug)[:, 1]
    843             Y_pred_chain[:, chain_idx] = estimator.predict(X_aug)
    844         inv_order = np.empty_like(self.order_)

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_logistic.py in predict_proba(self, X)
   1360             where classes are ordered as they are in ``self.classes_``.
   1361         """
-> 1362         check_is_fitted(self)
   1363 
   1364         ovr = self.multi_class in ["ovr", "warn"] or (

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1388 
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 
   1392 

NotFittedError: This LogisticRegression instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.

## === cell 46
submission = pd.DataFrame(proba_predictions_test, columns=target_cols)
submission.insert(0, "id", test_df["id"].values)
submission.head()



## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/998420011.py in <cell line: 0>()
      1 # Build a submission that matches the required format:
----> 2 submission = pd.DataFrame(proba_predictions_test, columns=target_cols)
      3 # Insert the correct id column as the first column
      4 submission.insert(0, "id", test_df["id"].values)
      5 submission.head()

NameError: name 'proba_predictions_test' is not defined

## === cell 47
submission.info()



## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2614940646.py in <cell line: 0>()
----> 1 submission.info()
      2 

NameError: name 'submission' is not defined

## === cell 48
submission.to_csv("submission.csv", index=False)
print("The submission has been successfully saved.")

## --- ERROR in cell 48, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2490249171.py in <cell line: 0>()
----> 1 submission.to_csv("submission.csv", index=False)
      2 print("The submission has been successfully saved.")

NameError: name 'submission' is not defined

## --- ERROR in outputing the csv:
Invalid submission: Submission is missing the following columns: {'insult', 'obscene', 'identity_hate', 'threat', 'severe_toxic', 'toxic'}
