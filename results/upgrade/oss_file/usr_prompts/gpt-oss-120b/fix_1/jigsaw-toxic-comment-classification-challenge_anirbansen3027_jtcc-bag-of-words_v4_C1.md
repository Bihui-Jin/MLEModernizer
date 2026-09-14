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
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from statistics import mean 

import string 

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

pd.options.display.float_format = "{:,.3f}".format


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/2756011002.py in <cell line: 0>()
      8 #Sklearn Libraries
      9 from sklearn.model_selection import train_test_split
---> 10 from sklearn.feature_extraction import stop_words
     11 from sklearn.feature_extraction.text import CountVectorizer
     12 from sklearn.multioutput import MultiOutputClassifier

ImportError: cannot import name 'stop_words' from 'sklearn.feature_extraction' (/usr/local/lib/python3.11/dist-packages/sklearn/feature_extraction/__init__.py)

## === cell 1
!unzip -o '/kaggle/input/jigsaw-toxic-comment-classification-challenge/*.zip' -d /kaggle/working > /dev/null


## === cell 2
train_text = pd.read_csv("train.csv")
test_text = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

print(train_text.shape, test_text.shape, sample_submission.shape)
train_text.head()


## === cell 3
test_text.head()


## === cell 4
sample_submission.head()


## === cell 5
train_text[["toxic","severe_toxic","obscene","threat","insult","identity_hate"]].apply(pd.Series.value_counts, args = (True, True, False, None, False))


## === cell 6
X = train_text.comment_text
y = train_text[["toxic","severe_toxic","obscene","threat","insult","identity_hate"]]
X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle = True, random_state = 123)


## === cell 7
X_train.shape, X_val.shape


## === cell 8
stop_words = stop_words.ENGLISH_STOP_WORDS
def clean(doc):
    doc = "".join([char for char in doc if char not in string.punctuation and not char.isdigit()])
    doc = " ".join([token for token in doc.split() if token not in stop_words])
    return doc.lower()


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/802116497.py in <cell line: 0>()
      1 #importing stop words like in, the, of so that these can be removed from texts
      2 #as these words dont help in determining the classes(Whether a sentence is toxic or not)
----> 3 stop_words = stop_words.ENGLISH_STOP_WORDS
      4 #Function for basic cleaning/preprocessing texts
      5 def clean(doc):

NameError: name 'stop_words' is not defined

## === cell 9
vect = CountVectorizer(max_features= 5000, preprocessor=clean)
X_train_dtm = vect.fit_transform(X_train)
X_val_dtm = vect.transform(X_val)

print(X_train_dtm.shape, X_val_dtm.shape)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3337243498.py in <cell line: 0>()
----> 1 vect = CountVectorizer(max_features= 5000, preprocessor=clean)
      2 X_train_dtm = vect.fit_transform(X_train)
      3 X_val_dtm = vect.transform(X_val)
      4 
      5 print(X_train_dtm.shape, X_val_dtm.shape)

NameError: name 'CountVectorizer' is not defined

## === cell 10
pd.DataFrame(X_train_dtm.A[:5], columns = vect.get_feature_names())


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2044087620.py in <cell line: 0>()
----> 1 pd.DataFrame(X_train_dtm.A[:5], columns = vect.get_feature_names())

NameError: name 'X_train_dtm' is not defined

## === cell 11
nb = MultiOutputClassifier(MultinomialNB()).fit(X_train_dtm, y_train)
lr = MultiOutputClassifier(LogisticRegression(class_weight='balanced', max_iter=3000)) \
                    .fit(X_train_dtm, y_train)


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1646984612.py in <cell line: 0>()
      1 #Initializing and fitting models on Training and Test Data
      2 #Naive Bayes Model
----> 3 nb = MultiOutputClassifier(MultinomialNB()).fit(X_train_dtm, y_train)
      4 #Logistic Regression Model (As we have unbalanced dataset, we use class_weight which will use inverse
      5 #of counts of that class. It penalizes mistakes in samples of class[i] with class_weight[i] instead of 1)

NameError: name 'MultiOutputClassifier' is not defined

## === cell 12
def calculate_roc_auc(y_test, y_pred):
    aucs = []
    for col in range(y_test.shape[1]):
        aucs.append(roc_auc_score(y_test[:,col],y_pred[:,col]))
    return aucs


## === cell 13
results = []
for model in [nb,lr]:
    est = type(model.estimator).__name__
    y_vals = y_val.to_numpy()
    y_preds = np.transpose(np.array(model.predict_proba(X_val_dtm))[:,:,1])
    mean_auc = mean(calculate_roc_auc(y_vals,y_preds))
    results.append([est, mean_auc])
    
pd.DataFrame(results, columns = ["Model","Mean AUC"])


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4127907918.py in <cell line: 0>()
      2 results = []
      3 #Making predictions from all the trained models and measure performance for each
----> 4 for model in [nb,lr]:
      5     #Extracting name of the model
      6     est = type(model.estimator).__name__

NameError: name 'nb' is not defined

## === cell 14
df_test = pd.merge(test_text, sample_submission, on = "id")
X_test_dtm = vect.transform(df_test["comment_text"])
y_preds = np.transpose(np.array(lr.predict_proba(X_test_dtm))[:,:,1])
df_test[["toxic","severe_toxic","obscene","threat","insult","identity_hate"]] = y_preds
df_test.drop(["comment_text"], axis = 1, inplace = True)
df_test.to_csv("sample_submission.csv", index = False)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/748425637.py in <cell line: 0>()
      4 #Transform the test dataset as well based on Bag of Words/ Count Vectorizer as the Logistic model would
      5 #expect the same
----> 6 X_test_dtm = vect.transform(df_test["comment_text"])
      7 #Use the Logistic Regression model to output probabilities and take the probability for class 1
      8 y_preds = np.transpose(np.array(lr.predict_proba(X_test_dtm))[:,:,1])

NameError: name 'vect' is not defined

## --- ERROR in outputing the csv:
Invalid submission: Submission is missing the following columns: {'identity_hate', 'toxic', 'obscene', 'severe_toxic', 'threat', 'insult'}
