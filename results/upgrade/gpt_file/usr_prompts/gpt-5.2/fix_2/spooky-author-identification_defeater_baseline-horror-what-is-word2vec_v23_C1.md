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

1.09129

# 6. Current score

0.92147

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 0.92147) has done: 'I fix the pipeline so it runs end-to-end and actually writes a valid `submission.csv`. The main runtime error comes from splitting `X` independently from `y`, so the XGBoost `DMatrix` gets labels with a mismatched length; I switch to a single consistent train/validation split on indices and then fit on full training data before predicting on the real test set. I also update the XGBoost parameters for xgboost==2.0.3 compatibility (remove deprecated `silent`, avoid `best_ntree_limit` without eval/early stopping) while keeping the same model/training core. Finally, I make NLTK stopwords/wordnet downloading robust in Kaggle so preprocessing won’t crash.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import xgboost as xgb

from gensim.models import Word2Vec

from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

import nltk

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer,
    HashingVectorizer,
)
from sklearn.model_selection import train_test_split

for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords

alpha_tokenizer = RegexpTokenizer("[A-Za-z]\\w+")
lemmatizer = WordNetLemmatizer()
stop = set(stopwords.words("english"))




## === cell 1
def _read_csv_fallback(rel_paths):
    last_err = None
    for p in rel_paths:
        try:
            return pd.read_csv(p)
        except Exception as e:
            last_err = e
    raise FileNotFoundError(
        f"Could not read any of: {rel_paths}. Last error: {last_err}"
    )


train = _read_csv_fallback(
    [
        "/kaggle/input/train.csv",
        "/kaggle/data/train.csv",
        "../input/train.csv",
    ]
)
test = _read_csv_fallback(
    [
        "/kaggle/input/test.csv",
        "/kaggle/data/test.csv",
        "../input/test.csv",
    ]
)

author_mapping = {"EAP": 0, "HPL": 1, "MWS": 2}
inv_author_mapping = {v: k for k, v in author_mapping.items()}

y_all = train["author"].map(author_mapping).values.astype(np.int32)



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
    (
        LogisticRegression(
            tol=1e-8, penalty="l2", C=0.1, max_iter=1000, multi_class="auto"
        ),
        "Logistic Regression",
        "green",
    ),
    (MultinomialNB(), "Naive Bayes", "magenta"),
    (
        RandomForestClassifier(n_estimators=10, criterion="gini", random_state=0),
        "Random Forest",
        "gray",
    ),
    (None, "XGBoost", "pink"),
]



## === cell 5
params = {}
params["objective"] = "multi:softprob"
params["eta"] = 0.1
params["max_depth"] = 3
params["num_class"] = 3
params["eval_metric"] = "mlogloss"
params["min_child_weight"] = 1
params["subsample"] = 0.8
params["colsample_bytree"] = 0.3
params["seed"] = 0
params["verbosity"] = 0



## === cell 6
vec = vectorizers[4][1]

X_all = vec.fit_transform(np.hstack((train.text.values, test.text.values)))

n_train = train.shape[0]
X_train_full = X_all[:n_train]
X_test_comp = X_all[n_train:]

idx_tr, idx_va = train_test_split(
    np.arange(n_train), test_size=0.3, random_state=0, stratify=y_all
)
X_tr = X_train_full[idx_tr]
y_tr = y_all[idx_tr]
X_va = X_train_full[idx_va]
y_va = y_all[idx_va]

dtrain = xgb.DMatrix(X_tr, label=y_tr)
dvalid = xgb.DMatrix(X_va, label=y_va)

model = xgb.train(
    params=list(params.items()),
    dtrain=dtrain,
    num_boost_round=40,
    evals=[(dtrain, "train"), (dvalid, "valid")],
    verbose_eval=False,
)

dtrain_full = xgb.DMatrix(X_train_full, label=y_all)
model_full = xgb.train(
    params=list(params.items()),
    dtrain=dtrain_full,
    num_boost_round=40,
    verbose_eval=False,
)

dtest = xgb.DMatrix(X_test_comp)
probs = model_full.predict(dtest)

final = pd.DataFrame(
    {
        "id": test["id"].values,
        "EAP": probs[:, 0],
        "HPL": probs[:, 1],
        "MWS": probs[:, 2],
    }
)
final.to_csv("submission.csv", index=False)

print(final.head())
print("Wrote submission.csv with shape:", final.shape)



## === cell 7
NUM_FEATURES = 150
w2v_model = Word2Vec(
    sentences=data,
    min_count=3,
    vector_size=NUM_FEATURES,
    window=5,
    sg=1,
    alpha=1e-4,
    workers=4,
)



## === cell 8
len(w2v_model.wv.key_to_index)



## === cell 9
word = "raven"
if word in w2v_model.wv:
    print(w2v_model.wv.most_similar(word))
else:
    print(f"'{word}' not in vocabulary")




## === cell 10
def get_feature_vec(tokens, num_features, model):
    featureVec = np.zeros(shape=(1, num_features), dtype="float32")
    missed = 0
    for w in tokens:
        try:
            featureVec = np.add(featureVec, model.wv[w])
        except KeyError:
            missed += 1
    if len(tokens) - missed == 0:
        return np.zeros(shape=(num_features,), dtype="float32")
    return np.divide(featureVec, len(tokens) - missed).squeeze()




## === cell 11
vectors = []
for sent in train.text.values:
    toks = [
        lemmatizer.lemmatize(word.lower())
        for word in alpha_tokenizer.tokenize(sent)
        if word.lower() not in stop
    ]
    vectors.append(get_feature_vec(toks, NUM_FEATURES, w2v_model))
vectors = np.vstack(vectors)



## === cell 12
estimator = LogisticRegression(C=1, max_iter=1000, multi_class="auto")
estimator.fit(vectors, y_all)



## === cell 13
test_vectors = []
for sent in test.text.values:
    toks = [
        lemmatizer.lemmatize(word.lower())
        for word in alpha_tokenizer.tokenize(sent)
        if word.lower() not in stop
    ]
    test_vectors.append(get_feature_vec(toks, NUM_FEATURES, w2v_model))
test_vectors = np.vstack(test_vectors)



## === cell 14
w2v_probs = estimator.predict_proba(test_vectors)



## === cell 15
w2v_final = pd.DataFrame(
    {
        "id": test["id"].values,
        "EAP": w2v_probs[:, 0],
        "HPL": w2v_probs[:, 1],
        "MWS": w2v_probs[:, 2],
    }
)
print(w2v_final.head())
