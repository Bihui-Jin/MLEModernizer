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



## === cell 1
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_df1 = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
)



## === cell 2
train_df1["score"].dtypes



## === cell 3
train_df1.head(10)



## === cell 4
train_df1["score"].value_counts()



## === cell 5
submission = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
)
submission



## === cell 6
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
    "you'll": "you you will",
    "you'll've": "you you will have",
    "you're": "you are",
    "you've": "you have",
}



## === cell 7
c_re = re.compile("(%s)" % "|".join(cList.keys()))




## === cell 8
def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]

    return c_re.sub(replace, text)




## === cell 9
def removeHTML(x):
    html = re.compile(r"<.*?>")
    return html.sub(r"", x)




## === cell 10
def dataPreprocessing(x: pd.Series) -> pd.Series:
    x = x.fillna("").astype(str)
    x = x.apply(lambda s: s.lower())
    x = x.apply(removeHTML)
    x = x.apply(lambda s: re.sub("@\w+", "", s))
    x = x.apply(lambda s: re.sub("'\d+", "", s))
    x = x.apply(lambda s: re.sub("\d+", "", s))
    x = x.apply(lambda s: re.sub("http\w+", "", s))
    x = x.apply(lambda s: re.sub(r"\s+", " ", s))
    x = x.apply(expandContractions)
    x = x.apply(lambda s: re.sub(r"\.+", ".", s))
    x = x.apply(lambda s: re.sub(r"\,+", ",", s))
    x = x.apply(lambda s: re.sub("\n", "", s))
    x = x.apply(lambda s: re.sub(r"[^\w\s]", "", s))
    x = x.apply(lambda s: s.strip())
    return x




## === cell 11
x = dataPreprocessing(train_df1["full_text"])



## === cell 12
x



## === cell 13
test_df1 = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)



## === cell 14
x0 = dataPreprocessing(test_df1["full_text"])



## === cell 15
y = train_df1["score"].astype(int)



## === cell 16
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=123, stratify=y
)



## === cell 17
y_train



## === cell 18
X_train



## === cell 19
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



## === cell 20
text_vectorizer = TfidfVectorizer(
    stop_words="english",
    sublinear_tf=False,
    strip_accents="unicode",
    binary=True,
    analyzer="word",
    token_pattern=r"\w{2,}",
    ngram_range=(1, 1),
    norm="l1",
    use_idf=False,
    smooth_idf=False,
    max_features=90000,
    min_df=20,
    sparse_output=True,
)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1921733605.py in <cell line: 0>()
      1 # BUGFIX: ensure TF-IDF output is CSR (stable for downstream scalers) via sparse_output=True.
      2 # Keep core TF-IDF settings the same.
----> 3 text_vectorizer = TfidfVectorizer(
      4     stop_words="english",
      5     sublinear_tf=False,

TypeError: TfidfVectorizer.__init__() got an unexpected keyword argument 'sparse_output'

## === cell 21
X_train_features = text_vectorizer.fit_transform(X_train).tocsr()
X_train_features



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/592393520.py in <cell line: 0>()
      1 # Keep this for inspection (not required for the pipeline fit), and force CSR to avoid coo_matrix issues.
----> 2 X_train_features = text_vectorizer.fit_transform(X_train).tocsr()
      3 X_train_features
      4 

NameError: name 'text_vectorizer' is not defined

## === cell 22
X_train_features



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2388575743.py in <cell line: 0>()
----> 1 X_train_features
      2 

NameError: name 'X_train_features' is not defined

## === cell 23
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import SVC

clf = make_pipeline(
    text_vectorizer,
    MaxAbsScaler(),
    SVC(
        C=1.75,
        kernel="rbf",
        gamma="scale",
        decision_function_shape="ovr",
        random_state=123,
        tol=1e-5,
        shrinking=True,
        verbose=True,
        break_ties=True,
    ),
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1057722222.py in <cell line: 0>()
      7 # so each step sees the expected input and is properly fitted.
      8 clf = make_pipeline(
----> 9     text_vectorizer,
     10     MaxAbsScaler(),
     11     SVC(

NameError: name 'text_vectorizer' is not defined

## === cell 24
print(confusion_matrix(y_test.values.ravel(), y_pred.ravel()))



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4127076281.py in <cell line: 0>()
----> 1 print(confusion_matrix(y_test.values.ravel(), y_pred.ravel()))
      2 

NameError: name 'y_pred' is not defined

## === cell 25
print(classification_report(y_test.values.ravel(), y_pred.ravel()))



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2810632315.py in <cell line: 0>()
----> 1 print(classification_report(y_test.values.ravel(), y_pred.ravel()))
      2 

NameError: name 'y_pred' is not defined

## === cell 26
kappa = cohen_kappa_score(y_test.values.ravel(), y_pred.ravel(), weights="quadratic")
print("Cohen's kappa score: ", kappa)



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1659918473.py in <cell line: 0>()
----> 1 kappa = cohen_kappa_score(y_test.values.ravel(), y_pred.ravel(), weights="quadratic")
      2 print("Cohen's kappa score: ", kappa)
      3 

NameError: name 'y_pred' is not defined

## === cell 27
test_predictions = clf.predict(x0)



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1012539354.py in <cell line: 0>()
      1 # Predict on the competition test set
----> 2 test_predictions = clf.predict(x0)
      3 

NameError: name 'clf' is not defined

## === cell 28
submission = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
)
submission = submission.merge(test_df1[["essay_id"]], on="essay_id", how="right")

pred = np.asarray(test_predictions, dtype=int)
pred = np.clip(pred, 1, 6)

submission["score"] = pred
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote:", os.path.abspath("submission.csv"), "rows:", len(submission))

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3424646002.py in <cell line: 0>()
      5 submission = submission.merge(test_df1[["essay_id"]], on="essay_id", how="right")
      6 
----> 7 pred = np.asarray(test_predictions, dtype=int)
      8 pred = np.clip(pred, 1, 6)
      9 

NameError: name 'test_predictions' is not defined
