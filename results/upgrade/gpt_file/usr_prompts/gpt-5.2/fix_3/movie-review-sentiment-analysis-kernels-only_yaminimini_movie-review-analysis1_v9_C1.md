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
Predict the sentiment of phrases.

## Metric
Classification accuracy.

## Submission Format
For each phrase in the test set, predict a label for the sentiment. Your submission should have a header and look like the following:

```
PhraseId,Sentiment
156061,2
156062,2
156063,2
...
```

## Dataset
The dataset is comprised of tab-separated files with phrases. Each phrase has a PhraseId. Each sentence has a SentenceId.

The sentiment labels are:

0 - negative

1 - somewhat negative

2 - neutral

3 - somewhat positive

4 - positive

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sampleSubmission.csv (46819 lines)
            sampleSubmission.csv.zip (146.0 kB)
            test.tsv (46819 lines)
            test.tsv.zip (1.1 MB)
            train.tsv (109243 lines)
            train.tsv.zip (2.7 MB)
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
        input/
            description.md (72 lines)
            sampleSubmission.csv (46819 lines)
            sampleSubmission.csv.zip (146.0 kB)
            test.tsv (46819 lines)
            test.tsv.zip (1.1 MB)
            train.tsv (109243 lines)
            train.tsv.zip (2.7 MB)
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
        working/
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
```

-> data/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv has 46818 rows and 2 columns.
Here is some information about the columns:
PhraseId (int64) has range: 29.00 - 156030.00, 0 nan values
Sentiment (int64) has 1 unique values: [2]

-> data/sampleSubmission.csv has 46818 rows and 2 columns.
Here is some information about the columns:
PhraseId (int64) has range: 29.00 - 156030.00, 0 nan values
Sentiment (int64) has 1 unique values: [2]

-> (stopped after 10 files for performance)

# 5. Target score

0.16143

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.51711) has done: 'I fix the runtime blockers that prevent training/prediction and ensure a valid `submission.csv` is always written. The main issues are: converting sparse matrices to `np.matrix` via `.todense()` (scikit-learn rejects this), using the deprecated `keras.preprocessing` imports in Keras 3, and outdated APIs like `Adam(lr=...)` and `predict_classes`. To keep changes minimal and stable, I keep the TF-IDF + MultinomialNB core path (and avoid the broken Keras LSTM branch), generate predictions on the test set, and write the submission with the exact required columns and a `.csv` suffix.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing
import os

print(os.listdir("../input"))



## === cell 1
import pandas as pd



## === cell 2
train = pd.read_csv("../input/train.tsv", sep="\t")



## === cell 3
train.head()



## === cell 4
test = pd.read_csv("../input/test.tsv", sep="\t")



## === cell 5
test.head()



## === cell 6
train["Sentiment"].unique()



## === cell 7
train.shape



## === cell 8
test.shape



## === cell 9
train.isnull().sum(axis=0)



## === cell 10
test.isnull().sum(axis=0)



## === cell 11
train["SentenceId"].value_counts()[0:5]



## === cell 12
test["SentenceId"].value_counts()[0:5]



## === cell 13
len(train["SentenceId"].unique()) + len(test["SentenceId"].unique())



## === cell 14
len(train["PhraseId"].unique()) + len(test["PhraseId"].unique())



## === cell 15
from sklearn.feature_extraction.text import TfidfVectorizer



## === cell 16
tfidf = TfidfVectorizer(
    analyzer="word",
    stop_words="english",
    min_df=0.20,
    max_df=0.50,
    ngram_range=(1, 1),
)



## === cell 17
tfidf.fit(train["Phrase"])



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/320214275.py in <cell line: 0>()
----> 1 tfidf.fit(train["Phrase"])
      2 

/usr/local/lib/python3.11/dist-packages/sklearn/feature_extraction/text.py in fit(self, raw_documents, y)
   2101             sublinear_tf=self.sublinear_tf,
   2102         )
-> 2103         X = super().fit_transform(raw_documents)
   2104         self._tfidf.fit(X)
   2105         return self

/usr/local/lib/python3.11/dist-packages/sklearn/feature_extraction/text.py in fit_transform(self, raw_documents, y)
   1399             if max_features is not None:
   1400                 X = self._sort_features(X, vocabulary)
-> 1401             X, self.stop_words_ = self._limit_features(
   1402                 X, vocabulary, max_doc_count, min_doc_count, max_features
   1403             )

/usr/local/lib/python3.11/dist-packages/sklearn/feature_extraction/text.py in _limit_features(self, X, vocabulary, high, low, limit)
   1251         kept_indices = np.where(mask)[0]
   1252         if len(kept_indices) == 0:
-> 1253             raise ValueError(
   1254                 "After pruning, no terms remain. Try a lower min_df or a higher max_df."
   1255             )

ValueError: After pruning, no terms remain. Try a lower min_df or a higher max_df.

## === cell 18
train_tfidf = tfidf.transform(train["Phrase"])



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/2791519848.py in <cell line: 0>()
----> 1 train_tfidf = tfidf.transform(train["Phrase"])
      2 

/usr/local/lib/python3.11/dist-packages/sklearn/feature_extraction/text.py in transform(self, raw_documents)
   2155         check_is_fitted(self, msg="The TF-IDF vectorizer is not fitted")
   2156 
-> 2157         X = super().transform(raw_documents)
   2158         return self._tfidf.transform(X, copy=False)
   2159 

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

## === cell 19
X_train = train_tfidf  # sparse csr_matrix



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1925786741.py in <cell line: 0>()
----> 1 X_train = train_tfidf  # sparse csr_matrix
      2 

NameError: name 'train_tfidf' is not defined

## === cell 20
Y_train = train["Sentiment"].astype(int)



## === cell 21
from sklearn.naive_bayes import MultinomialNB



## === cell 22
NB = MultinomialNB()



## === cell 23
NB.fit(X_train, Y_train)



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3442454686.py in <cell line: 0>()
----> 1 NB.fit(X_train, Y_train)
      2 

NameError: name 'X_train' is not defined

## === cell 24
test_tfidf = tfidf.transform(test["Phrase"])



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/1794134294.py in <cell line: 0>()
----> 1 test_tfidf = tfidf.transform(test["Phrase"])
      2 

/usr/local/lib/python3.11/dist-packages/sklearn/feature_extraction/text.py in transform(self, raw_documents)
   2155         check_is_fitted(self, msg="The TF-IDF vectorizer is not fitted")
   2156 
-> 2157         X = super().transform(raw_documents)
   2158         return self._tfidf.transform(X, copy=False)
   2159 

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

## === cell 25
x_test = test_tfidf  # sparse csr_matrix



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2098840106.py in <cell line: 0>()
----> 1 x_test = test_tfidf  # sparse csr_matrix
      2 

NameError: name 'test_tfidf' is not defined

## === cell 26
x_test.shape



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/422321314.py in <cell line: 0>()
----> 1 x_test.shape
      2 

NameError: name 'x_test' is not defined

## === cell 27
y_pred = NB.predict(x_test)



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2887879950.py in <cell line: 0>()
----> 1 y_pred = NB.predict(x_test)
      2 

NameError: name 'x_test' is not defined

## === cell 28
type(y_pred)



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2391300876.py in <cell line: 0>()
----> 1 type(y_pred)
      2 

NameError: name 'y_pred' is not defined

## === cell 29
y_pred_df = pd.DataFrame(y_pred.astype(int), columns=["Sentiment"])



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4071958366.py in <cell line: 0>()
----> 1 y_pred_df = pd.DataFrame(y_pred.astype(int), columns=["Sentiment"])
      2 

NameError: name 'y_pred' is not defined

## === cell 30
sub = pd.concat([test["PhraseId"].astype(int), y_pred_df], axis=1)



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2718338746.py in <cell line: 0>()
----> 1 sub = pd.concat([test["PhraseId"].astype(int), y_pred_df], axis=1)
      2 

NameError: name 'y_pred_df' is not defined

## === cell 31
sub.head()



## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3518946013.py in <cell line: 0>()
----> 1 sub.head()
      2 

NameError: name 'sub' is not defined

## === cell 32
sub.columns = ["PhraseId", "Sentiment"]
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)



## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2165715783.py in <cell line: 0>()
----> 1 sub.columns = ["PhraseId", "Sentiment"]
      2 sub.to_csv("submission.csv", index=False)
      3 print("Wrote submission.csv with shape:", sub.shape)
      4 

NameError: name 'sub' is not defined

## === cell 33
pass
