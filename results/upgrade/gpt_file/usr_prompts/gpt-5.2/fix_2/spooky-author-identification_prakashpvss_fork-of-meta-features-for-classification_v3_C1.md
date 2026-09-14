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

3.6

# 3. Installed packages

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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
from nltk.corpus import stopwords
import string

try:
    _ = stopwords.words("english")
except LookupError:
    try:
        nltk.download("stopwords", quiet=True)
    except Exception:
        pass

try:
    eng_stopwords = set(stopwords.words("english"))
except Exception:
    eng_stopwords = set()



## === cell 1
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
sample = pd.read_csv("../input/sample_submission.csv")



## === cell 2
train_df["num_words"] = train_df["text"].apply(lambda x: len(str(x).split()))
test_df["num_words"] = test_df["text"].apply(lambda x: len(str(x).split()))



## === cell 3
train_df["num_unique_words"] = train_df["text"].apply(
    lambda x: len(set(str(x).split()))
)
test_df["num_unique_words"] = test_df["text"].apply(lambda x: len(set(str(x).split())))



## === cell 4
train_df["num_chars"] = train_df["text"].apply(lambda x: len(str(x)))
test_df["num_chars"] = test_df["text"].apply(lambda x: len(str(x)))

train_df["num_stopwords"] = train_df["text"].apply(
    lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords])
)
test_df["num_stopwords"] = test_df["text"].apply(
    lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords])
)

train_df["num_punctuations"] = train_df["text"].apply(
    lambda x: len([c for c in str(x) if c in string.punctuation])
)
test_df["num_punctuations"] = test_df["text"].apply(
    lambda x: len([c for c in str(x) if c in string.punctuation])
)

train_df["num_words_upper"] = train_df["text"].apply(
    lambda x: len([w for w in str(x).split() if w.isupper()])
)
test_df["num_words_upper"] = test_df["text"].apply(
    lambda x: len([w for w in str(x).split() if w.isupper()])
)

train_df["num_words_title"] = train_df["text"].apply(
    lambda x: len([w for w in str(x).split() if w.istitle()])
)
test_df["num_words_title"] = test_df["text"].apply(
    lambda x: len([w for w in str(x).split() if w.istitle()])
)

train_df["mean_word_len"] = train_df["text"].apply(
    lambda x: np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) else 0.0
)
test_df["mean_word_len"] = test_df["text"].apply(
    lambda x: np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) else 0.0
)



## === cell 5
author_mapping_dict = {"EAP": 0, "HPL": 1, "MWS": 2}
train_y = train_df["author"].map(author_mapping_dict).values
train_id = train_df["id"].values
test_id = test_df["id"].values



## === cell 6
cols_to_drop = ["id", "text"]
train_X = train_df.drop(cols_to_drop + ["author"], axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)



## === cell 7
train_X.head()



## === cell 8
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, model_selection, naive_bayes


def runLR(train_X, train_y, test_X, test_y=None, test_X2=None):
    model = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="auto")
    model.fit(train_X, train_y)
    return model.predict_proba(test_X), model.predict_proba(test_X2), model


def runMNB(train_X, train_y, test_X, test_y, test_X2):
    model = naive_bayes.MultinomialNB()
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)
    pred_test_y2 = model.predict_proba(test_X2)
    return pred_test_y, pred_test_y2, model




## === cell 9
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])

for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_X.iloc[dev_index], train_X.iloc[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runLR(dev_X, dev_y, val_X, val_y, test_X)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index, :] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))

print("cv scores : ", cv_scores)



## === cell 10
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD

tfidf_vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 3))
full_tfidf = tfidf_vec.fit_transform(
    train_df["text"].values.tolist() + test_df["text"].values.tolist()
)
train_tfidf = tfidf_vec.transform(train_df["text"].values.tolist())
test_tfidf = tfidf_vec.transform(test_df["text"].values.tolist())



## === cell 11
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])

for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index, :] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))

print("cv scores : ", cv_scores)



## === cell 12
n_comp = 20
svd_tf = TruncatedSVD(n_components=20, algorithm="arpack", random_state=2017)
svd_tf.fit(full_tfidf)

train_svd = pd.DataFrame(svd_tf.transform(train_tfidf))
test_svd = pd.DataFrame(svd_tf.transform(test_tfidf))

train_svd_cols = ["svd_word_" + str(i) for i in range(n_comp)]
test_svd_cols = ["svd_word_" + str(i) for i in range(n_comp)]
train_svd.columns = train_svd_cols
test_svd.columns = test_svd_cols

train_df = pd.concat([train_df, train_svd], axis=1)
test_df = pd.concat([test_df, test_svd], axis=1)
train_df.head()



## === cell 13
cols_to_drop = ["id", "text"]
train_X = train_df.drop(cols_to_drop + ["author"], axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)

train_X.columns = train_X.columns.astype(str)
test_X.columns = test_X.columns.astype(str)

kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])

for dev_index, val_index in kf.split(train_df):
    dev_X, val_X = train_X.iloc[dev_index], train_X.iloc[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runLR(dev_X, dev_y, val_X, val_y, test_X)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index, :] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))

print("cv scores : ", cv_scores)



## === cell 14
tfidf_vec = CountVectorizer(stop_words="english", ngram_range=(1, 3))
tfidf_vec.fit(train_df["text"].values.tolist() + test_df["text"].values.tolist())
train_tfidf = tfidf_vec.transform(train_df["text"].values.tolist())
test_tfidf = tfidf_vec.transform(test_df["text"].values.tolist())



## === cell 15
cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)

for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index, :] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))

print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 5.0
print("cv scores : ", cv_scores)

train_df["nb_cvec_eap"] = pred_train[:, 0]
train_df["nb_cvec_hpl"] = pred_train[:, 1]
train_df["nb_cvec_mws"] = pred_train[:, 2]
test_df["nb_cvec_eap"] = pred_full_test[:, 0]
test_df["nb_cvec_hpl"] = pred_full_test[:, 1]
test_df["nb_cvec_mws"] = pred_full_test[:, 2]



## === cell 16
cols_to_drop = ["id", "text"]
train_X = train_df.drop(cols_to_drop + ["author"], axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)

train_X.columns = train_X.columns.astype(str)
test_X.columns = test_X.columns.astype(str)

kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])

for dev_index, val_index in kf.split(train_df):
    dev_X, val_X = train_X.iloc[dev_index], train_X.iloc[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runLR(dev_X, dev_y, val_X, val_y, test_X)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index, :] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))

print("cv scores : ", cv_scores)



## === cell 17
p = pred_full_test / 5.0

result = pd.DataFrame(
    {
        "id": test_id,
        "EAP": p[:, 0],
        "HPL": p[:, 1],
        "MWS": p[:, 2],
    }
)

result = result[["id", "EAP", "HPL", "MWS"]]
result.to_csv("result.csv", index=False)

print(result.head())
print("Wrote submission:", "result.csv", "shape=", result.shape)
