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

0.36146

# 6. Current score

0.39847

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 0.39847) has done: 'I fix the runtime errors caused by API changes in gensim (Word2Vec `size`/vocab access and word lookup) and xgboost (missing `best_ntree_limit` when no early stopping is used), while keeping the feature engineering and training approach intact. I also correct two logic bugs that currently break execution: the mean word length feature mistakenly uses the full row instead of the text, and the train/test split in `vectorize()` was misaligned (splitting `X` and `y` separately). Finally, I ensure the code writes a valid `submission.csv` with the exact required columns (`id,EAP,HPL,MWS`) by making `pred_full_test` a proper 2D numpy array and aligning prediction rows with `test_id`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, train_test_split

from sklearn.metrics import accuracy_score

import xgboost as xgb

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

for pkg, path in [
    ("stopwords", "corpora/stopwords"),
    ("wordnet", "corpora/wordnet"),
    ("omw-1.4", "corpora/omw-1.4"),
]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg, quiet=True)

alpha_tokenizer = RegexpTokenizer(r"[A-Za-z]\w+")
lemmatizer = WordNetLemmatizer()
stop = stopwords.words("english")



## === cell 1
TRAIN_PATH = "/kaggle/input/train.csv"
TEST_PATH = "/kaggle/input/test.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
test_id = test["id"].values

author_mapping = {"EAP": 0, "HPL": 1, "MWS": 2}
y_train = train["author"].map(author_mapping).values



## === cell 2
vectorizers = [
    (
        "TF-IDF + SVD",
        Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(ngram_range=(1, 3), analyzer="word", binary=False),
                ),
                ("svd", TruncatedSVD(n_components=150, random_state=42)),
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
                ("svd", TruncatedSVD(n_components=150, random_state=42)),
                ("norm", Normalizer()),
            ]
        ),
    ),
]



## === cell 3
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
        LogisticRegression(tol=1e-8, penalty="l2", C=0.1, max_iter=200),
        "Logistic Regression",
        "green",
    ),
    (MultinomialNB(), "Naive Bayes", "magenta"),
    (
        RandomForestClassifier(n_estimators=10, criterion="gini", random_state=42),
        "Random Forest",
        "gray",
    ),
    (None, "XGBoost", "pink"),
]



## === cell 4
params = {}
params["objective"] = "multi:softprob"
params["eta"] = 0.1
params["max_depth"] = 3
params["silent"] = 1
params["num_class"] = 3
params["eval_metric"] = "mlogloss"
params["min_child_weight"] = 1
params["subsample"] = 0.8
params["colsample_bytree"] = 0.3
params["seed"] = 0




## === cell 5
def vectorize():
    test_size = 0.3
    tr_idx, va_idx = train_test_split(
        np.arange(train.shape[0]), test_size=test_size, random_state=42, shuffle=True
    )

    y_tr = y_train[tr_idx]
    y_va = y_train[va_idx]

    for vec_name, vec in vectorizers:
        print(vec_name + "\n")
        X = vec.fit_transform(train.text.values)
        X_tr, X_va = X[tr_idx], X[va_idx]

        for est, est_name, _ in estimators:
            if est_name == "XGBoost":
                xgtrain = xgb.DMatrix(X_tr, label=y_tr)
                xgval = xgb.DMatrix(X_va)
                booster = xgb.train(
                    params=list(params.items()), dtrain=xgtrain, num_boost_round=40
                )
                preds = booster.predict(xgval).argmax(axis=1)
            else:
                est.fit(X_tr, y_tr)
                preds = est.predict(X_va)
            print(accuracy_score(y_va, preds), est_name)





## === cell 6
train["num_words"] = train.text.apply(lambda x: len(str(x).split()))
test["num_words"] = test.text.apply(lambda x: len(str(x).split()))

train["num_unique_words"] = train.text.apply(lambda x: len(set(str(x).split())))
test["num_unique_words"] = test.text.apply(lambda x: len(set(str(x).split())))

train["num_chars"] = train.text.apply(lambda x: len(str(x)))
test["num_chars"] = test.text.apply(lambda x: len(str(x)))

train["num_stopwords"] = train.text.apply(
    lambda x: len([w for w in str(x).lower().split() if w in stop])
)
test["num_stopwords"] = test.text.apply(
    lambda x: len([w for w in str(x).lower().split() if w in stop])
)

train["mean_word_len"] = train.text.apply(
    lambda s: (
        float(np.mean([len(w) for w in str(s).split()])) if len(str(s).split()) else 0.0
    )
)
test["mean_word_len"] = test.text.apply(
    lambda s: (
        float(np.mean([len(w) for w in str(s).split()])) if len(str(s).split()) else 0.0
    )
)



## === cell 7
train_text = [
    " ".join(
        [
            lemmatizer.lemmatize(word.lower())
            for word in alpha_tokenizer.tokenize(sent)
            if word.lower() not in stop
        ]
    )
    for sent in train.text.values
]
test_text = [
    " ".join(
        [
            lemmatizer.lemmatize(word.lower())
            for word in alpha_tokenizer.tokenize(sent)
            if word.lower() not in stop
        ]
    )
    for sent in test.text.values
]



## === cell 8
vectorizer = CountVectorizer(ngram_range=(1, 7), analyzer="char")

full = vectorizer.fit_transform(train_text + test_text)
X_train_cv = vectorizer.transform(train_text)
X_test_cv = vectorizer.transform(test_text)

pred_full_test = np.zeros((test.shape[0], 3), dtype=np.float64)
pred_train = np.zeros((train.shape[0], 3), dtype=np.float64)

for dev_index, val_index in KFold(n_splits=5, shuffle=True, random_state=42).split(
    np.arange(train.shape[0])
):
    dev_X, val_X = X_train_cv[dev_index], X_train_cv[val_index]
    dev_y, val_y = y_train[dev_index], y_train[val_index]
    model_nb = MultinomialNB()
    model_nb.fit(dev_X, dev_y)
    pred_full_test += model_nb.predict_proba(X_test_cv)
    pred_train[val_index, :] = model_nb.predict_proba(val_X)

pred_full_test /= 5.0

train["CH_EAP"] = pred_train[:, 0]
train["CH_HPL"] = pred_train[:, 1]
train["CH_MWS"] = pred_train[:, 2]
test["CH_EAP"] = pred_full_test[:, 0]
test["CH_HPL"] = pred_full_test[:, 1]
test["CH_MWS"] = pred_full_test[:, 2]



## === cell 9
vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 3))
full = vectorizer.fit_transform(train_text + test_text)
X_train_cv = vectorizer.transform(train_text)
X_test_cv = vectorizer.transform(test_text)

pred_full_test = np.zeros((test.shape[0], 3), dtype=np.float64)
pred_train = np.zeros((train.shape[0], 3), dtype=np.float64)

for dev_index, val_index in KFold(n_splits=5, shuffle=True, random_state=42).split(
    np.arange(train.shape[0])
):
    dev_X, val_X = X_train_cv[dev_index], X_train_cv[val_index]
    dev_y, val_y = y_train[dev_index], y_train[val_index]
    model_nb = MultinomialNB()
    model_nb.fit(dev_X, dev_y)
    pred_full_test += model_nb.predict_proba(X_test_cv)
    pred_train[val_index, :] = model_nb.predict_proba(val_X)

pred_full_test /= 5.0

train["C_EAP"] = pred_train[:, 0]
train["C_HPL"] = pred_train[:, 1]
train["C_MWS"] = pred_train[:, 2]
test["C_EAP"] = pred_full_test[:, 0]
test["C_HPL"] = pred_full_test[:, 1]
test["C_MWS"] = pred_full_test[:, 2]



## === cell 10
vectorizer = TfidfVectorizer(ngram_range=(1, 5), analyzer="char")
full = vectorizer.fit_transform(train_text + test_text)
X_train_tv = vectorizer.transform(train_text)
X_test_tv = vectorizer.transform(test_text)

pred_full_test = np.zeros((test.shape[0], 3), dtype=np.float64)
pred_train = np.zeros((train.shape[0], 3), dtype=np.float64)

for dev_index, val_index in KFold(n_splits=5, shuffle=True, random_state=42).split(
    np.arange(train.shape[0])
):
    dev_X, val_X = X_train_tv[dev_index], X_train_tv[val_index]
    dev_y, val_y = y_train[dev_index], y_train[val_index]
    model_nb = MultinomialNB()
    model_nb.fit(dev_X, dev_y)
    pred_full_test += model_nb.predict_proba(X_test_tv)
    pred_train[val_index, :] = model_nb.predict_proba(val_X)

pred_full_test /= 5.0

train["T_EAP"] = pred_train[:, 0]
train["T_HPL"] = pred_train[:, 1]
train["T_MWS"] = pred_train[:, 2]
test["T_EAP"] = pred_full_test[:, 0]
test["T_HPL"] = pred_full_test[:, 1]
test["T_MWS"] = pred_full_test[:, 2]



## === cell 11
svd = TruncatedSVD(n_components=20, algorithm="arpack", random_state=42)
svd.fit(full)
train_svd = pd.DataFrame(svd.transform(X_train_tv))
test_svd = pd.DataFrame(svd.transform(X_test_tv))

train_svd.columns = ["SVD_" + str(i) for i in range(20)]
test_svd.columns = ["SVD_" + str(i) for i in range(20)]
train = pd.concat([train, train_svd], axis=1)
test = pd.concat([test, test_svd], axis=1)



## === cell 12
train = train.drop(["id", "text", "author"], axis=1)
test = test.drop(["id", "text"], axis=1)



## === cell 13
NUM_FEATURES = 100
w2v_model = Word2Vec(
    sentences=[s.split() for s in (train_text + test_text)],
    min_count=2,
    vector_size=NUM_FEATURES,
    window=4,
    sg=1,
    alpha=1e-4,
    workers=4,
    seed=42,
)



## === cell 14
vocab_size = len(w2v_model.wv.key_to_index)
vocab_size



## === cell 15
if "raven" in w2v_model.wv:
    _ = w2v_model.wv.most_similar("raven", topn=5)




## === cell 16
def get_feature_vec(tokens, num_features, model):
    featureVec = np.zeros(shape=(num_features,), dtype="float32")
    missed = 0
    for word in tokens:
        if word in model.wv:
            featureVec += model.wv[word]
        else:
            missed += 1
    denom = len(tokens) - missed
    if denom == 0:
        return np.zeros(shape=(num_features,), dtype="float32")
    return (featureVec / denom).astype("float32")




## === cell 17
train_vectors = []
for sent in train_text:
    tokens = [
        lemmatizer.lemmatize(w.lower())
        for w in alpha_tokenizer.tokenize(sent)
        if w.lower() not in stop
    ]
    train_vectors.append(get_feature_vec(tokens, NUM_FEATURES, w2v_model))



## === cell 18
test_vectors = []
for sent in test_text:
    tokens = [
        lemmatizer.lemmatize(w.lower())
        for w in alpha_tokenizer.tokenize(sent)
        if w.lower() not in stop
    ]
    test_vectors.append(get_feature_vec(tokens, NUM_FEATURES, w2v_model))



## === cell 19
full_vectors = np.vstack([np.array(train_vectors), np.array(test_vectors)])



## === cell 20
svd = TruncatedSVD(n_components=30, algorithm="arpack", random_state=42)
svd.fit(full_vectors)
train_w2v = pd.DataFrame(svd.transform(np.array(train_vectors)))
test_w2v = pd.DataFrame(svd.transform(np.array(test_vectors)))

train_w2v.columns = ["W2V_" + str(i) for i in range(30)]
test_w2v.columns = ["W2V_" + str(i) for i in range(30)]

train = pd.concat([train, train_w2v], axis=1)
test = pd.concat([test, test_w2v], axis=1)



## === cell 21
pred_full_test = np.zeros((test.shape[0], 3), dtype=np.float64)

for dev_index, val_index in KFold(n_splits=5, shuffle=True, random_state=42).split(
    np.arange(train.shape[0])
):
    dev_X = train.iloc[dev_index]
    dev_y = y_train[dev_index]

    xgtrain = xgb.DMatrix(dev_X, label=dev_y)
    xgtest = xgb.DMatrix(test)

    booster = xgb.train(
        params=list(params.items()), dtrain=xgtrain, num_boost_round=1000
    )
    predictions = booster.predict(xgtest)  # shape: (n_test, 3)
    pred_full_test += predictions

pred_full_test /= 5.0



## === cell 22
author = pd.DataFrame(pred_full_test, columns=["EAP", "HPL", "MWS"])

final = pd.DataFrame()
final["id"] = test_id
final["EAP"] = author["EAP"].values
final["HPL"] = author["HPL"].values
final["MWS"] = author["MWS"].values

eps = 1e-15
for c in ["EAP", "HPL", "MWS"]:
    final[c] = final[c].clip(eps, 1 - eps)

final.to_csv("submission.csv", sep=",", index=False)
print("Wrote submission.csv with shape:", final.shape)
print(final.head())
