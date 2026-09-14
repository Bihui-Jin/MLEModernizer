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
Given some text, predict the author.

## Metric
Multi-class logarithmic loss. 

The submitted probabilities for a given sentences are not required to sum to one because they are rescaled prior to being scored (each row is divided by the row sum).

In order to avoid the extremes of the log function, predicted probabilities are replaced with \\(max(min(p,1-10^{-15}),10^{-15})\\).

## Submission Format
You must submit a csv file with the id, and a probability for each of the three classes. The order of the rows does not matter. The file must have a header and should look like the following:

```
id,EAP,HPL,MWS
id07943,0.33,0.33,0.33
...
```

## Dataset 
### File descriptions
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Data fields
- **id** - a unique identifier for each sentence
- **text** - some text written by one of the authors
- **author** - the author of the sentence (EAP: Edgar Allan Poe, HPL: HP Lovecraft; MWS: Mary Wollstonecraft Shelley)

# 2. Python version

3.6

# 3. Installed packages

gensim==4.4.0
geopandas==0.14.4
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
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (98 lines)
            sample_submission.csv (1959 lines)
            sample_submission.csv.zip (7.4 kB)
            test.csv (1959 lines)
            test.csv.zip (133.3 kB)
            train.csv (17622 lines)
            train.csv.zip (1.2 MB)
            train.zip (1.2 MB)
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
        input/
            description.md (98 lines)
            sample_submission.csv (1959 lines)
            sample_submission.csv.zip (7.4 kB)
            test.csv (1959 lines)
            test.csv.zip (133.3 kB)
            train.csv (17622 lines)
            train.csv.zip (1.2 MB)
            train.zip (1.2 MB)
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
        working/
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
```

-> data/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> data/spooky-author-identification/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> data/spooky-author-identification/test.csv has 1958 rows and 2 columns.
The columns are: id, text

-> data/spooky-author-identification/train.csv has 17621 rows and 3 columns.
The columns are: id, text, author

-> data/test.csv has 1958 rows and 2 columns.
The columns are: id, text

-> data/train.csv has 17621 rows and 3 columns.
The columns are: id, text, author

-> input/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> (stopped after 10 files for performance)

# 5. Target score

1.05046

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from subprocess import check_output
from gensim.models import Word2Vec

from nltk.tokenize import RegexpTokenizer
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline

import xgboost as xgb

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer,
    HashingVectorizer,
)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

alpha_tokenizer = RegexpTokenizer("[A-Za-z]\\w+")
lemmatizer = WordNetLemmatizer()
stop = stopwords.words("english")




## === cell 1
import os

train_path = os.path.join("..", "input", "train.csv")
test_path = os.path.join("..", "input", "test.csv")
if not os.path.exists(train_path):
    train_path = os.path.join("input", "train.csv")
if not os.path.exists(test_path):
    test_path = os.path.join("input", "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

author_mapping = {"EAP": 0, "HPL": 1, "MWS": 2}
y_train = train["author"].map(author_mapping).values




## === cell 2
data = [
    [
        lemmatizer.lemmatize(word.lower())
        for word in alpha_tokenizer.tokenize(sent)
        if word.lower() not in stop
    ]
    for sent in train.text.values
]




## === cell 3
vectorizers = [
    (
        "3-gram TF-IDF Vectorizer on words",
        TfidfVectorizer(ngram_range=(1, 3), analyzer="word", binary=False),
    ),
    (
        "3-gram Count Vectorizer on words",
        CountVectorizer(ngram_range=(1, 3), analyzer="word", binary=False),
    ),
    (
        "3-gram Hashing Vectorizer on words",
        HashingVectorizer(ngram_range=(1, 5), analyzer="word", binary=False),
    ),
    (
        "TF-IDF + SVD",
        Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(ngram_range=(1, 3), analyzer="word", binary=False),
                ),
                ("svd", TruncatedSVD(n_components=150)),
            ]
        ),
    ),
    (
        "TF-IDF + SVD + Normalizer",
        Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(ngram_range=(1, 3), analyzer="word", binary=False),
                ),
                ("svd", TruncatedSVD(n_components=150)),
                ("norm", Normalizer()),
            ]
        ),
    ),
]




## === cell 4
estimators = [
    (KNeighborsClassifier(n_neighbors=3), "K-Nearest Neighbors", "yellow"),
    (
        SVC(
            C=1,
            cache_size=200,
            class_weight=None,
            coef0=0.0,
            decision_function_shape="ovr",
            degree=3,
            gamma="auto",
            kernel="linear",
            max_iter=-1,
            probability=False,
            random_state=None,
            shrinking=True,
            tol=0.001,
            verbose=False,
        ),
        "Support Vector Machine",
        "red",
    ),
    (LogisticRegression(tol=1e-8, penalty="l2", C=0.1), "Logistic Regression", "green"),
    (MultinomialNB(), "Naive Bayes", "magenta"),
    (
        RandomForestClassifier(n_estimators=10, criterion="gini"),
        "Random Forest",
        "gray",
    ),
    (None, "XGBoost", "pink"),
]




## === cell 5
params = {
    "objective": "multi:softprob",
    "eta": 0.1,
    "max_depth": 3,
    "silent": 1,
    "num_class": 3,
    "eval_metric": "mlogloss",
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.3,
    "seed": 0,
}




## === cell 6
NUM_FEATURES = 150
model = Word2Vec(
    sentences=data,
    vector_size=NUM_FEATURES,
    window=5,
    min_count=3,
    sg=1,
    alpha=1e-4,
    workers=4,
)




## === cell 9
def get_feature_vec(tokens, num_features, wv_model):
    """
    Compute the average Word2Vec vector for a list of tokens.
    """
    feature_vec = np.zeros(num_features, dtype="float32")
    missed = 0
    for word in tokens:
        try:
            feature_vec += wv_model[word]
        except KeyError:
            missed += 1
    valid_cnt = len(tokens) - missed
    if valid_cnt == 0:
        return np.zeros(num_features, dtype="float32")
    return feature_vec / valid_cnt




## === cell 10
train_vectors = []
for txt in train.text.values:
    tokens = [
        lemmatizer.lemmatize(word.lower())
        for word in alpha_tokenizer.tokenize(txt)
        if word.lower() not in stop
    ]
    train_vectors.append(get_feature_vec(tokens, NUM_FEATURES, model.wv))
train_vectors = np.vstack(train_vectors)  # shape (n_samples, NUM_FEATURES)




## === cell 11
test_vectors = []
for txt in test.text.values:
    tokens = [
        lemmatizer.lemmatize(word.lower())
        for word in alpha_tokenizer.tokenize(txt)
        if word.lower() not in stop
    ]
    test_vectors.append(get_feature_vec(tokens, NUM_FEATURES, model.wv))
test_vectors = np.vstack(test_vectors)  # shape (n_test, NUM_FEATURES)




## === cell 12
dtrain = xgb.DMatrix(train_vectors, label=y_train)
dtest = xgb.DMatrix(test_vectors)
xgb_model = xgb.train(params=params, dtrain=dtrain, num_boost_round=40)
probs = xgb_model.predict(dtest, ntree_limit=xgb_model.best_ntree_limit)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/663011473.py in <cell line: 0>()
      2 dtest = xgb.DMatrix(test_vectors)
      3 xgb_model = xgb.train(params=params, dtrain=dtrain, num_boost_round=40)
----> 4 probs = xgb_model.predict(dtest, ntree_limit=xgb_model.best_ntree_limit)
      5 
      6 

AttributeError: 'Booster' object has no attribute 'best_ntree_limit'

## === cell 13
author = pd.DataFrame(probs, columns=["EAP", "HPL", "MWS"])
submission = pd.DataFrame()
submission["id"] = test["id"]
submission["EAP"] = author["EAP"]
submission["HPL"] = author["HPL"]
submission["MWS"] = author["MWS"]
submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3036826166.py in <cell line: 0>()
----> 1 author = pd.DataFrame(probs, columns=["EAP", "HPL", "MWS"])
      2 submission = pd.DataFrame()
      3 submission["id"] = test["id"]
      4 submission["EAP"] = author["EAP"]
      5 submission["HPL"] = author["HPL"]

NameError: name 'probs' is not defined
