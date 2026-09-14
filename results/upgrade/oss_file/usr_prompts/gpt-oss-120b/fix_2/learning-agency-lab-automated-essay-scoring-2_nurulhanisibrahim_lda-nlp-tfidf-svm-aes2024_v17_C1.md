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
Predict the score of student essays.

## Metric
Quadratic weighted kappa.

## Submission Format
For each `essay_id` in the test set, you must predict the corresponding `score` (between 1-6, see [rubric](https://storage.googleapis.com/kaggle-forum-message-attachments/2733927/20538/Rubric_%20Holistic%20Essay%20Scoring.pdf) for more details). The file should contain a header and have the following format:

```
essay_id,score
000d118,3
000fe60,3
001ab80,4
...
```

## Dataset
- **train.csv** - Essays and scores to be used as training data.
    - `essay_id` - The unique ID of the essay
    - `full_text` - The full essay response
    - `score` - Holistic score of the essay on a 1-6 scale
- **test.csv** - The essays to be used as test data. Contains the same fields as `train.csv`, aside from exclusion of `score`.
- **sample_submission.csv** - A submission file in the correct format.
    - `essay_id` - The unique ID of the essay
    - `score` - The predicted holistic score of the essay on a 1-6 scale

# 2. Python version

3.12

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
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        input/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        working/
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
```

-> data/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/learning-agency-lab-automated-essay-scoring-2/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/learning-agency-lab-automated-essay-scoring-2/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> data/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> (stopped after 10 files for performance)

# 5. Target score

0.72791

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from sklearn.svm import LinearSVC



## === cell 1
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 2
train_df1 = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
)



## === cell 3
print(train_df1["score"].dtypes)



## === cell 4
print(train_df1.head(10))



## === cell 5
print(train_df1["score"].value_counts())



## === cell 6
submission_template = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
)
print(submission_template.head())



## === cell 7
cList = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there had",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'alls": "you alls",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
}



## === cell 8
c_re = re.compile("(%s)" % "|".join(cList.keys()))




## === cell 9
def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]

    return c_re.sub(replace, text)




## === cell 10
def removeHTML(x):
    html = re.compile(r"<.*?>")
    return html.sub(r"", x)




## === cell 11
def dataPreprocessing(series):
    series = series.apply(lambda s: s.lower())
    series = series.apply(removeHTML)
    series = series.apply(lambda s: re.sub("@\\w+", "", s))
    series = series.apply(lambda s: re.sub("'\\d+", "", s))
    series = series.apply(lambda s: re.sub("\\d+", "", s))
    series = series.apply(lambda s: re.sub("http\\w+", "", s))
    series = series.apply(lambda s: re.sub(r"\\s+", " ", s))
    series = series.apply(expandContractions)
    series = series.apply(lambda s: re.sub(r"\\.+", ".", s))
    series = series.apply(lambda s: re.sub(r"\\,+", ",", s))
    series = series.apply(lambda s: re.sub("\\n", "", s))
    series = series.apply(lambda s: re.sub("[^\\w\\s]", "", s))
    series = series.apply(lambda s: s.strip())
    return series




## === cell 12
x = dataPreprocessing(train_df1["full_text"])
print("Processed training texts:", x.shape)



## === cell 13
test_df1 = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)
x0 = dataPreprocessing(test_df1["full_text"])
print("Processed test texts:", x0.shape)



## === cell 14
y = train_df1["score"]
print("Target vector shape:", y.shape)



## === cell 15
X_train, X_val, y_train, y_val = train_test_split(
    x, y, test_size=0.1, random_state=123, stratify=y
)



## === cell 16
print(
    "Train/validation split sizes:",
    X_train.shape,
    X_val.shape,
    y_train.shape,
    y_val.shape,
)



## === cell 17
text_vectorizer = TfidfVectorizer(
    stop_words="english",
    sublinear_tf=False,
    strip_accents="unicode",
    binary=True,
    analyzer="word",
    token_pattern=r"\\w{2,}",
    ngram_range=(1, 1),
    norm="l1",
    use_idf=False,
    smooth_idf=False,
    max_features=50000,  # reduced to keep memory reasonable
    min_df=20,
)



## === cell 18
X_train_features = text_vectorizer.fit_transform(X_train)
X_val_features = text_vectorizer.transform(X_val)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3850395222.py in <cell line: 0>()
----> 1 X_train_features = text_vectorizer.fit_transform(X_train)
      2 X_val_features = text_vectorizer.transform(X_val)
      3 

/usr/local/lib/python3.11/dist-packages/sklearn/feature_extraction/text.py in fit_transform(self, raw_documents, y)
   2131             sublinear_tf=self.sublinear_tf,
   2132         )
-> 2133         X = super().fit_transform(raw_documents)
   2134         self._tfidf.fit(X)
   2135         # X is already a transformed view of raw_documents so

/usr/local/lib/python3.11/dist-packages/sklearn/feature_extraction/text.py in fit_transform(self, raw_documents, y)
   1386                     break
   1387 
-> 1388         vocabulary, X = self._count_vocab(raw_documents, self.fixed_vocabulary_)
   1389 
   1390         if self.binary:

/usr/local/lib/python3.11/dist-packages/sklearn/feature_extraction/text.py in _count_vocab(self, raw_documents, fixed_vocab)
   1292             vocabulary = dict(vocabulary)
   1293             if not vocabulary:
-> 1294                 raise ValueError(
   1295                     "empty vocabulary; perhaps the documents only contain stop words"
   1296                 )

ValueError: empty vocabulary; perhaps the documents only contain stop words

## === cell 19
clf = LinearSVC(C=1.0, max_iter=2000, dual=False)
clf.fit(X_train_features, y_train)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/629318857.py in <cell line: 0>()
      1 # Train a linear SVM that works with sparse input
      2 clf = LinearSVC(C=1.0, max_iter=2000, dual=False)
----> 3 clf.fit(X_train_features, y_train)
      4 

NameError: name 'X_train_features' is not defined

## === cell 20
y_pred_val = clf.predict(X_val_features)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/212337313.py in <cell line: 0>()
----> 1 y_pred_val = clf.predict(X_val_features)
      2 

NameError: name 'X_val_features' is not defined

## === cell 21
print("Confusion Matrix (validation):")
print(confusion_matrix(y_val, y_pred_val))



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2000218341.py in <cell line: 0>()
      1 print("Confusion Matrix (validation):")
----> 2 print(confusion_matrix(y_val, y_pred_val))
      3 

NameError: name 'y_pred_val' is not defined

## === cell 22
print("\nClassification Report (validation):")
print(classification_report(y_val, y_pred_val, digits=4))



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4057989078.py in <cell line: 0>()
      1 print("\nClassification Report (validation):")
----> 2 print(classification_report(y_val, y_pred_val, digits=4))
      3 

NameError: name 'y_pred_val' is not defined

## === cell 23
kappa = cohen_kappa_score(y_val, y_pred_val, weights="quadratic")
print("\nCohen's quadratic kappa (validation):", kappa)



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3421997966.py in <cell line: 0>()
----> 1 kappa = cohen_kappa_score(y_val, y_pred_val, weights="quadratic")
      2 print("\nCohen's quadratic kappa (validation):", kappa)
      3 

NameError: name 'y_pred_val' is not defined

## === cell 24
test_features = text_vectorizer.transform(x0)
test_predictions = clf.predict(test_features)



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/1941591115.py in <cell line: 0>()
      1 # Predict on the official test set
----> 2 test_features = text_vectorizer.transform(x0)
      3 test_predictions = clf.predict(test_features)
      4 

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
submission = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
)
submission["score"] = test_predictions.astype(int)
submission.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")
display(submission.head())

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/244867695.py in <cell line: 0>()
      3     "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
      4 )
----> 5 submission["score"] = test_predictions.astype(int)
      6 submission.to_csv("submission.csv", index=False)
      7 print("Submission file written to submission.csv")

NameError: name 'test_predictions' is not defined
