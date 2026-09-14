# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.9

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
seaborn==0.12.2
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

# 5. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import ensemble, metrics, model_selection, naive_bayes
from scipy import sparse

color = sns.color_palette()
nltk.download("stopwords", quiet=True)
eng_stopwords = set(stopwords.words("english"))
pd.options.mode.chained_assignment = None



## === cell 1
train_df = pd.read_csv("../input/spooky-author-identification/train.csv")
test_df = pd.read_csv("../input/spooky-author-identification/test.csv")
print(f"Number of rows in train dataset : {train_df.shape[0]}")
print(f"Number of rows in test dataset : {test_df.shape[0]}")



## === cell 2
train_df["num_words"] = train_df["text"].str.split().str.len()
test_df["num_words"] = test_df["text"].str.split().str.len()

train_df["num_unique_words"] = train_df["text"].str.split().apply(lambda x: len(set(x)))
test_df["num_unique_words"] = test_df["text"].str.split().apply(lambda x: len(set(x)))

train_df["num_chars"] = train_df["text"].str.len()
test_df["num_chars"] = test_df["text"].str.len()

train_df["num_stopwrods"] = (
    train_df["text"]
    .str.lower()
    .str.split()
    .apply(lambda words: sum(w in eng_stopwords for w in words))
)
test_df["num_stopwrods"] = (
    test_df["text"]
    .str.lower()
    .str.split()
    .apply(lambda words: sum(w in eng_stopwords for w in words))
)

train_df["num_punctuations"] = train_df["text"].apply(
    lambda x: sum(c in string.punctuation for c in x)
)
test_df["num_punctuations"] = test_df["text"].apply(
    lambda x: sum(c in string.punctuation for c in x)
)

train_df["num_words_upper"] = (
    train_df["text"].str.split().apply(lambda words: sum(w.isupper() for w in words))
)
test_df["num_words_upper"] = (
    test_df["text"].str.split().apply(lambda words: sum(w.isupper() for w in words))
)

train_df["mean_word_len"] = (
    train_df["text"]
    .str.split()
    .apply(lambda words: np.mean([len(w) for w in words]) if words else 0)
)
test_df["mean_word_len"] = (
    test_df["text"]
    .str.split()
    .apply(lambda words: np.mean([len(w) for w in words]) if words else 0)
)



## === cell 3
author_mapping_dict = {"EAP": 0, "HPL": 1, "MWS": 2}
train_y = train_df["author"].map(author_mapping_dict).values
train_id = train_df["id"].values
test_id = test_df["id"].values

cols_to_drop = ["id", "text"]
train_X = train_df.drop(cols_to_drop + ["author"], axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)




## === cell 4
def runXGB(
    train_X,
    train_y,
    test_X,
    test_y=None,
    test_X2=None,
    seed_val=0,
    child=1,
    colsample=0.7,
):
    param = {
        "objective": "multi:softprob",
        "eta": 0.05,
        "max_depth": 6,
        "silent": 1,
        "num_class": 3,
        "eval_metric": "mlogloss",
        "min_child_weight": child,
        "subsample": 0.8,
        "colsample_bytree": colsample,
        "seed": seed_val,
        "n_jobs": -1,  # use all CPUs
        "tree_method": "hist",  # histogram algorithm for speed
    }
    num_rounds = 4000
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [(xgtrain, "train"), (xgtest, "test")]
        model = xgb.train(
            param,
            xgtrain,
            num_rounds,
            watchlist,
            early_stopping_rounds=50,
            verbose_eval=False,
        )
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(param, xgtrain, num_rounds, verbose_eval=False)

    pred_test_y = model.predict(xgtest)

    pred_test_y2 = None
    if test_X2 is not None:
        xgtest2 = xgb.DMatrix(test_X2)
        pred_test_y2 = model.predict(xgtest2)

    return pred_test_y, pred_test_y2, model




## === cell 5
kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)
cv_scores = []
pred_full_test = np.zeros((test_X.shape[0], 3))
pred_train = np.zeros([train_df.shape[0], 3])

for dev_index, val_index in kf.split(train_X, train_y):
    dev_X, val_X = train_X.iloc[dev_index], train_X.iloc[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runXGB(
        dev_X,
        dev_y,
        val_X,
        val_y,
        test_X,
        seed_val=0,
        colsample=0.9,
    )
    pred_full_test += pred_test_y
    pred_train[val_index, :] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))

pred_full_test /= kf.get_n_splits()
print("cv scores : ", cv_scores)



## === cell 6
try:
    fig, ax = plt.subplots(figsize=(12, 12))
    xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
    plt.show()
except Exception as e:
    print("Feature importance plot skipped:", e)



## === cell 7
tfidf_vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 3))
full_tfidf = tfidf_vec.fit_transform(
    train_df["text"].tolist() + test_df["text"].tolist()
)
train_tfidf = tfidf_vec.transform(train_df["text"].tolist())
test_tfidf = tfidf_vec.transform(test_df["text"].tolist())




## === cell 8
def runMNB(train_X, train_y, test_X, test_y, test_X2):
    model = naive_bayes.MultinomialNB()
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)
    pred_test_y2 = model.predict_proba(test_X2)
    return pred_test_y, pred_test_y2, model




## === cell 9
cv_scores = []
pred_full_test = np.zeros((test_X.shape[0], 3))
pred_train = np.zeros([train_df.shape[0], 3])
kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)

for dev_index, val_index in kf.split(train_tfidf, train_y):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    pred_full_test += pred_test_y
    pred_train[val_index, :] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))

print(f"Mean cv score : {np.mean(cv_scores)}")
pred_full_test /= 5.0

train_df["nb_cvec_eap"] = pred_train[:, 0]
train_df["nb_cvec_hpl"] = pred_train[:, 1]
train_df["nb_cvec_mws"] = pred_train[:, 2]
test_df["nb_cvec_eap"] = pred_full_test[:, 0]
test_df["nb_cvec_hpl"] = pred_full_test[:, 1]
test_df["nb_cvec_mws"] = pred_full_test[:, 2]



## === cell 10
tfidf_vec = CountVectorizer(stop_words="english", ngram_range=(1, 3))
tfidf_vec.fit(train_df["text"].tolist() + test_df["text"].tolist())
train_tfidf = tfidf_vec.transform(train_df["text"].tolist())
test_tfidf = tfidf_vec.transform(test_df["text"].tolist())

cv_scores = []
pred_full_test = np.zeros((test_X.shape[0], 3))
pred_train = np.zeros([train_df.shape[0], 3])
kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)

for dev_index, val_index in kf.split(train_X, train_y):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    pred_full_test += pred_test_y
    pred_train[val_index, :] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))

print(f"Mean cv score : {np.mean(cv_scores)}")
pred_full_test /= 5.0

train_df["nb_cvec_char_eap"] = pred_train[:, 0]
train_df["nb_cvec_char_hpl"] = pred_train[:, 1]
train_df["nb_cvec_char_mws"] = pred_train[:, 2]
test_df["nb_cvec_char_eap"] = pred_full_test[:, 0]
test_df["nb_cvec_char_hpl"] = pred_full_test[:, 1]
test_df["nb_cvec_char_mws"] = pred_full_test[:, 2]



## === cell 11
tfidf_vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 3))
full_word_tfidf = tfidf_vec.fit_transform(
    train_df["text"].tolist() + test_df["text"].tolist()
)
train_word_tfidf = full_word_tfidf[: len(train_df)]
test_word_tfidf = full_word_tfidf[len(train_df) :]

cols_to_drop = ["id", "text"]
train_num = train_df.drop(cols_to_drop + ["author"], axis=1)
test_num = test_df.drop(cols_to_drop, axis=1)

train_X = sparse.hstack([sparse.csr_matrix(train_num.values), train_word_tfidf])
test_X = sparse.hstack([sparse.csr_matrix(test_num.values), test_word_tfidf])

kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)
cv_scores = []
pred_full_test = np.zeros((test_X.shape[0], 3))
pred_train = np.zeros([train_df.shape[0], 3])

for dev_index, val_index in kf.split(train_X, train_y):
    dev_X, val_X = train_X[dev_index], train_X[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runXGB(
        dev_X, dev_y, val_X, val_y, test_X, seed_val=0, colsample=0.9
    )
    pred_full_test += pred_test_y
    pred_train[val_index, :] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))

pred_full_test /= kf.get_n_splits()
print("cv scores : ", cv_scores)

out_df = pd.DataFrame(pred_full_test, columns=["EAP", "HPL", "MWS"])
out_df.insert(0, "id", test_id)
out_df.to_csv("sub_fe.csv", index=False)



## === cell 12
try:
    if pred_val_y.shape[0] == val_y.shape[0]:
        cnf_matrix = metrics.confusion_matrix(val_y, np.argmax(pred_val_y, axis=1))
        plt.figure(figsize=(8, 8))
        plt.title("Confusion matrix of XGB (last fold)")
        sns.heatmap(
            cnf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["EAP", "HPL", "MWS"],
            yticklabels=["EAP", "HPL", "MWS"],
        )
        plt.ylabel("True")
        plt.xlabel("Predicted")
        plt.show()
    else:
        print("Skipping confusion matrix: shape mismatch")
except Exception as e:
    print("Confusion matrix error:", e)
