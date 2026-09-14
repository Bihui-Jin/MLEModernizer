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

# 5. Target score

0.29678

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
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import ensemble, metrics, model_selection, naive_bayes

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
train_df.head()



## === cell 3
cnt_srs = train_df["author"].value_counts()
plt.figure(figsize=(8, 4))
sns.barplot(x=cnt_srs.index, y=cnt_srs.values, alpha=0.8)
plt.ylabel("Number of Occurences", fontsize=12)
plt.xlabel("Author Name", fontsize=12)
plt.show()



## === cell 4
train_df["num_words"] = train_df["text"].apply(lambda x: len(str(x).split()))
test_df["num_words"] = test_df["text"].apply(lambda x: len(str(x).split()))

train_df["num_unique_words"] = train_df["text"].apply(
    lambda x: len(set(str(x).split()))
)
test_df["num_unique_words"] = test_df["text"].apply(lambda x: len(set(str(x).split())))

train_df["num_chars"] = train_df["text"].apply(lambda x: len(str(x)))
test_df["num_chars"] = test_df["text"].apply(lambda x: len(str(x)))

train_df["num_stopwrods"] = train_df["text"].apply(
    lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords])
)
test_df["num_stopwrods"] = test_df["text"].apply(
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

train_df["mean_word_len"] = train_df["text"].apply(
    lambda x: np.mean([len(w) for w in str(x).split()])
)
test_df["mean_word_len"] = test_df["text"].apply(
    lambda x: np.mean([len(w) for w in str(x).split()])
)



## === cell 5
train_df.loc[train_df["num_words"] > 80, "num_words"] = 80
plt.figure(figsize=(12, 8))
sns.violinplot(x="author", y="num_words", data=train_df)
plt.xlabel("Author Name", fontsize=12)
plt.ylabel("Number of words in text", fontsize=12)
plt.title("Number of words by author", fontsize=15)
plt.show()



## === cell 6
train_df.loc[train_df["num_punctuations"] > 10, "num_punctuations"] = 10
plt.figure(figsize=(12, 8))
sns.violinplot(x="author", y="num_punctuations", data=train_df)
plt.xlabel("Author Name", fontsize=12)
plt.ylabel("Number of punctuations in text", fontsize=12)
plt.title("Number of punctuations by author", fontsize=15)
plt.show()



## === cell 7
author_mapping_dict = {"EAP": 0, "HPL": 1, "MWS": 2}
train_y = train_df["author"].map(author_mapping_dict).values
train_id = train_df["id"].values
test_id = test_df["id"].values

train_df["num_words"] = train_df["text"].apply(lambda x: len(str(x).split()))
test_df["num_words"] = test_df["text"].apply(lambda x: len(str(x).split()))
train_df["mean_word_len"] = train_df["text"].apply(
    lambda x: np.mean([len(w) for w in str(x).split()])
)
test_df["mean_word_len"] = test_df["text"].apply(
    lambda x: np.mean([len(w) for w in str(x).split()])
)

cols_to_drop = ["id", "text"]
train_X = train_df.drop(cols_to_drop + ["author"], axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)




## === cell 8
def runXGB(
    train_X,
    train_y,
    test_X,
    test_y=None,
    test_X2=None,
    seed_val=0,
    child=1,
    colsample=0.3,
):
    param = {
        "objective": "multi:softprob",
        "eta": 0.1,
        "max_depth": 3,
        "silent": 1,
        "num_class": 3,
        "eval_metric": "mlogloss",
        "min_child_weight": child,
        "subsample": 0.8,
        "colsample_bytree": colsample,
        "seed": seed_val,
    }
    num_rounds = 2000
    xgtrain = xgb.DMatrix(train_X, label=train_y)
    pred_test_y2 = None

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [(xgtrain, "train"), (xgtest, "test")]
        model = xgb.train(
            list(param.items()),
            xgtrain,
            num_rounds,
            watchlist,
            early_stopping_rounds=50,
            verbose_eval=False,
        )
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(list(param.items()), xgtrain, num_rounds, verbose_eval=False)

    pred_test_y = model.predict(xgtest, ntree_limit=model.best_ntree_limit)

    if test_X2 is not None:
        xgtest2 = xgb.DMatrix(test_X2)
        pred_test_y2 = model.predict(xgtest2, ntree_limit=model.best_ntree_limit)

    return pred_test_y, pred_test_y2, model




## === cell 9
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])

for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_X.loc[dev_index], train_X.loc[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runXGB(
        dev_X, dev_y, val_X, val_y, test_X, seed_val=0
    )
    pred_full_test += pred_test_y
    pred_train[val_index, :] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    break

print("cv scores : ", cv_scores)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3458019639.py in <cell line: 0>()
      7     dev_X, val_X = train_X.loc[dev_index], train_X.loc[val_index]
      8     dev_y, val_y = train_y[dev_index], train_y[val_index]
----> 9     pred_val_y, pred_test_y, model = runXGB(
     10         dev_X, dev_y, val_X, val_y, test_X, seed_val=0
     11     )

/tmp/ipykernel_11/441227824.py in runXGB(train_X, train_y, test_X, test_y, test_X2, seed_val, child, colsample)
     40         model = xgb.train(list(param.items()), xgtrain, num_rounds, verbose_eval=False)
     41 
---> 42     pred_test_y = model.predict(xgtest, ntree_limit=model.best_ntree_limit)
     43 
     44     if test_X2 is not None:

AttributeError: 'Booster' object has no attribute 'best_ntree_limit'

## === cell 10
fig, ax = plt.subplots(figsize=(12, 12))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1974152363.py in <cell line: 0>()
      1 fig, ax = plt.subplots(figsize=(12, 12))
----> 2 xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
      3 plt.show()
      4 

NameError: name 'model' is not defined

## === cell 11
tfidf_vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 3))
full_tfidf = tfidf_vec.fit_transform(
    train_df["text"].tolist() + test_df["text"].tolist()
)
train_tfidf = tfidf_vec.transform(train_df["text"].tolist())
test_tfidf = tfidf_vec.transform(test_df["text"].tolist())




## === cell 12
def runMNB(train_X, train_y, test_X, test_y, test_X2):
    model = naive_bayes.MultinomialNB()
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)
    pred_test_y2 = model.predict_proba(test_X2)
    return pred_test_y, pred_test_y2, model




## === cell 13
cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)

for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    pred_full_test += pred_test_y
    pred_train[val_index, :] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))

print(f"Mean cv score : {np.mean(cv_scores)}")
pred_full_test = pred_full_test / 5.0



## === cell 14
import itertools
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")




## === cell 15
cnf_matrix = confusion_matrix(val_y, np.argmax(pred_val_y, axis=1))
np.set_printoptions(precision=2)
plt.figure(figsize=(8, 8))
plot_confusion_matrix(
    cnf_matrix,
    classes=["EAP", "HPL", "MWS"],
    title="Confusion matrix, without normalization",
)
plt.show()



## === cell 16
n_comp = 20
svd_obj = TruncatedSVD(n_components=n_comp, algorithm="arpack")
svd_obj.fit(full_tfidf)
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
train_svd.columns = [f"svd_word_{i}" for i in range(n_comp)]
test_svd.columns = [f"svd_word_{i}" for i in range(n_comp)]
train_df = pd.concat([train_df, train_svd], axis=1)
test_df = pd.concat([test_df, test_svd], axis=1)

del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd



## === cell 17
tfidf_vec = CountVectorizer(stop_words="english", ngram_range=(1, 3))
tfidf_vec.fit(train_df["text"].tolist() + test_df["text"].tolist())
train_tfidf = tfidf_vec.transform(train_df["text"].tolist())
test_tfidf = tfidf_vec.transform(test_df["text"].tolist())



## === cell 18
cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)

for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    pred_full_test += pred_test_y
    pred_train[val_index, :] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))

print(f"Mean cv score : {np.mean(cv_scores)}")
pred_full_test = pred_full_test / 5.0

train_df["nb_cvec_eap"] = pred_train[:, 0]
train_df["nb_cvec_hpl"] = pred_train[:, 1]
train_df["nb_cvec_mws"] = pred_train[:, 2]
test_df["nb_cvec_eap"] = pred_full_test[:, 0]
test_df["nb_cvec_hpl"] = pred_full_test[:, 1]
test_df["nb_cvec_mws"] = pred_full_test[:, 2]



## === cell 19
cnf_matrix = confusion_matrix(val_y, np.argmax(pred_val_y, axis=1))
np.set_printoptions(precision=2)
plt.figure(figsize=(8, 8))
plot_confusion_matrix(
    cnf_matrix,
    classes=["EAP", "HPL", "MWS"],
    title="Confusion matrix of NB on word count, without normalization",
)
plt.show()



## === cell 20
tfidf_vec = CountVectorizer(ngram_range=(1, 7), analyzer="char")
tfidf_vec.fit(train_df["text"].tolist() + test_df["text"].tolist())
train_tfidf = tfidf_vec.transform(train_df["text"].tolist())
test_tfidf = tfidf_vec.transform(test_df["text"].tolist())

cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)

for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    pred_full_test += pred_test_y
    pred_train[val_index, :] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))

print(f"Mean cv score : {np.mean(cv_scores)}")
pred_full_test = pred_full_test / 5.0

train_df["nb_cvec_char_eap"] = pred_train[:, 0]
train_df["nb_cvec_char_hpl"] = pred_train[:, 1]
train_df["nb_cvec_char_mws"] = pred_train[:, 2]
test_df["nb_cvec_char_eap"] = pred_full_test[:, 0]
test_df["nb_cvec_char_hpl"] = pred_full_test[:, 1]
test_df["nb_cvec_char_mws"] = pred_full_test[:, 2]



## === cell 21
tfidf_vec = TfidfVectorizer(ngram_range=(1, 5), analyzer="char")
full_tfidf = tfidf_vec.fit_transform(
    train_df["text"].tolist() + test_df["text"].tolist()
)
train_tfidf = tfidf_vec.transform(train_df["text"].tolist())
test_tfidf = tfidf_vec.transform(test_df["text"].tolist())

cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)

for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    pred_full_test += pred_test_y
    pred_train[val_index, :] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))

print(f"Mean cv score : {np.mean(cv_scores)}")
pred_full_test = pred_full_test / 5.0

train_df["nb_tfidf_char_eap"] = pred_train[:, 0]
train_df["nb_tfidf_char_hpl"] = pred_train[:, 1]
train_df["nb_tfidf_char_mws"] = pred_train[:, 2]
test_df["nb_tfidf_char_eap"] = pred_full_test[:, 0]
test_df["nb_tfidf_char_hpl"] = pred_full_test[:, 1]
test_df["nb_tfidf_char_mws"] = pred_full_test[:, 2]



## === cell 22
n_comp = 20
svd_obj = TruncatedSVD(n_components=n_comp, algorithm="arpack")
svd_obj.fit(full_tfidf)
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
train_svd.columns = [f"svd_char_{i}" for i in range(n_comp)]
test_svd.columns = [f"svd_char_{i}" for i in range(n_comp)]
train_df = pd.concat([train_df, train_svd], axis=1)
test_df = pd.concat([test_df, test_svd], axis=1)

del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd



## === cell 23
cols_to_drop = ["id", "text"]
train_X = train_df.drop(cols_to_drop + ["author"], axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)

kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])

for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_X.loc[dev_index], train_X.loc[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runXGB(
        dev_X, dev_y, val_X, val_y, test_X, seed_val=0, colsample=0.7
    )
    pred_full_test += pred_test_y
    pred_train[val_index, :] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    break

print("cv scores : ", cv_scores)

out_df = pd.DataFrame(pred_full_test, columns=["EAP", "HPL", "MWS"])
out_df.insert(0, "id", test_id)
out_df.to_csv("sub_fe.csv", index=False)



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3253933172.py in <cell line: 0>()
     11     dev_X, val_X = train_X.loc[dev_index], train_X.loc[val_index]
     12     dev_y, val_y = train_y[dev_index], train_y[val_index]
---> 13     pred_val_y, pred_test_y, model = runXGB(
     14         dev_X, dev_y, val_X, val_y, test_X, seed_val=0, colsample=0.7
     15     )

/tmp/ipykernel_11/441227824.py in runXGB(train_X, train_y, test_X, test_y, test_X2, seed_val, child, colsample)
     40         model = xgb.train(list(param.items()), xgtrain, num_rounds, verbose_eval=False)
     41 
---> 42     pred_test_y = model.predict(xgtest, ntree_limit=model.best_ntree_limit)
     43 
     44     if test_X2 is not None:

AttributeError: 'Booster' object has no attribute 'best_ntree_limit'

## === cell 24
fig, ax = plt.subplots(figsize=(12, 12))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1974152363.py in <cell line: 0>()
      1 fig, ax = plt.subplots(figsize=(12, 12))
----> 2 xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
      3 plt.show()
      4 

/usr/local/lib/python3.11/dist-packages/xgboost/plotting.py in plot_importance(booster, ax, height, xlim, ylim, title, xlabel, ylabel, fmap, importance_type, max_num_features, grid, show_values, values_format, **kwargs)
     94         importance = booster
     95     else:
---> 96         raise ValueError("tree must be Booster, XGBModel or dict instance")
     97 
     98     if not importance:

ValueError: tree must be Booster, XGBModel or dict instance

## === cell 25
cnf_matrix = confusion_matrix(val_y, np.argmax(pred_val_y, axis=1))
np.set_printoptions(precision=2)
plt.figure(figsize=(8, 8))
plot_confusion_matrix(
    cnf_matrix,
    classes=["EAP", "HPL", "MWS"],
    title="Confusion matrix of XGB, without normalization",
)
plt.show()

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2168864447.py in <cell line: 0>()
----> 1 cnf_matrix = confusion_matrix(val_y, np.argmax(pred_val_y, axis=1))
      2 np.set_printoptions(precision=2)
      3 plt.figure(figsize=(8, 8))
      4 plot_confusion_matrix(
      5     cnf_matrix,

/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py in confusion_matrix(y_true, y_pred, labels, sample_weight, normalize)
    315     (0, 2, 1, 1)
    316     """
--> 317     y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    318     if y_type not in ("binary", "multiclass"):
    319         raise ValueError("%s is not supported" % y_type)

/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py in _check_targets(y_true, y_pred)
     84     y_pred : array or indicator matrix
     85     """
---> 86     check_consistent_length(y_true, y_pred)
     87     type_true = type_of_target(y_true, input_name="y_true")
     88     type_pred = type_of_target(y_pred, input_name="y_pred")

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_consistent_length(*arrays)
    395     uniques = np.unique(lengths)
    396     if len(uniques) > 1:
--> 397         raise ValueError(
    398             "Found input variables with inconsistent numbers of samples: %r"
    399             % [int(l) for l in lengths]

ValueError: Found input variables with inconsistent numbers of samples: [3525, 3524]
