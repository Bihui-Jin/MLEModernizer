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
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC



## === cell 1
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 2
train_df = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
)



## === cell 3
train_df["score"].dtypes



## === cell 4
train_df.head(10)



## === cell 5
train_df["score"].value_counts()



## === cell 6
submission_template = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
)
submission_template.head()



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
def dataPreprocessing(x):
    x = x.apply(lambda s: s.lower())
    x = x.apply(removeHTML)
    x = x.apply(lambda s: re.sub("@\\w+", "", s))
    x = x.apply(lambda s: re.sub("'\\d+", "", s))
    x = x.apply(lambda s: re.sub("\\d+", "", s))
    x = x.apply(lambda s: re.sub("http\\w+", "", s))
    x = x.apply(lambda s: re.sub(r"\\s+", " ", s))
    x = x.apply(expandContractions)
    x = x.apply(lambda s: re.sub(r"\\.+", ".", s))
    x = x.apply(lambda s: re.sub(r"\\,+", ",", s))
    x = x.apply(lambda s: re.sub("\\n", "", s))
    x = x.apply(lambda s: re.sub("[^\\w\\s]", "", s))
    x = x.apply(lambda s: s.strip())
    return x




## === cell 12
X_text = dataPreprocessing(train_df["full_text"])



## === cell 13
test_df = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)
X_test_text = dataPreprocessing(test_df["full_text"])



## === cell 14
y = train_df["score"]



## === cell 15
X_train_text, X_valid_text, y_train, y_valid = train_test_split(
    X_text, y, test_size=0.1, random_state=123, stratify=y
)



## === cell 16
print("Train size:", X_train_text.shape[0])
print("Valid size:", X_valid_text.shape[0])



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
    max_features=90000,
    min_df=20,
)



## === cell 18
X_train_features = text_vectorizer.fit_transform(X_train_text)
X_valid_features = text_vectorizer.transform(X_valid_text)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/842454531.py in <cell line: 0>()
----> 1 X_train_features = text_vectorizer.fit_transform(X_train_text)
      2 X_valid_features = text_vectorizer.transform(X_valid_text)
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
clf = make_pipeline(LinearSVC(C=1.75, dual=False, max_iter=2000))



## === cell 20
clf.fit(X_train_features, y_train)
y_pred_valid = clf.predict(X_valid_features)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2638354326.py in <cell line: 0>()
----> 1 clf.fit(X_train_features, y_train)
      2 y_pred_valid = clf.predict(X_valid_features)
      3 

NameError: name 'X_train_features' is not defined

## === cell 21
print(confusion_matrix(y_valid, y_pred_valid))
print(classification_report(y_valid, y_pred_valid))



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1778032660.py in <cell line: 0>()
----> 1 print(confusion_matrix(y_valid, y_pred_valid))
      2 print(classification_report(y_valid, y_pred_valid))
      3 

NameError: name 'y_pred_valid' is not defined

## === cell 22
kappa = cohen_kappa_score(y_valid, y_pred_valid, weights="quadratic")
print("Cohen's kappa score:", kappa)



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1311100305.py in <cell line: 0>()
----> 1 kappa = cohen_kappa_score(y_valid, y_pred_valid, weights="quadratic")
      2 print("Cohen's kappa score:", kappa)
      3 

NameError: name 'y_pred_valid' is not defined

## === cell 23
X_test_features = text_vectorizer.transform(X_test_text)
test_predictions = clf.predict(X_test_features)



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/127745953.py in <cell line: 0>()
      1 # Prepare test predictions
----> 2 X_test_features = text_vectorizer.transform(X_test_text)
      3 test_predictions = clf.predict(X_test_features)
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

## === cell 24
submission = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
)
submission["score"] = test_predictions
submission.to_csv("submission.csv", index=False)
display(submission.head())

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1577368732.py in <cell line: 0>()
      2     "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
      3 )
----> 4 submission["score"] = test_predictions
      5 submission.to_csv("submission.csv", index=False)
      6 display(submission.head())

NameError: name 'test_predictions' is not defined
