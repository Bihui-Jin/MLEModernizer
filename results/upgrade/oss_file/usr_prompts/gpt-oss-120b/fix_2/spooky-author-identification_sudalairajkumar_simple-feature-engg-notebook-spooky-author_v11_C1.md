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

0.32073

# 6. Current score

0.38688

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 0.38688) has done: 'I fixed the plotting call, removed the Jupyter magic, and corrected the XGBoost prediction routine to avoid using the removed `best_ntree_limit` attribute. These changes let the notebook run end‑to‑end and generate a proper ‘sub_fe.csv’ submission file.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import ensemble, metrics, model_selection, naive_bayes


eng_stopwords = set(stopwords.words("english"))
pd.options.mode.chained_assignment = None
color = sns.color_palette()



## === cell 1
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Number of rows in train dataset : ", train_df.shape[0])
print("Number of rows in test dataset : ", test_df.shape[0])



## === cell 2
train_df.head()



## === cell 3
cnt_srs = train_df["author"].value_counts()

plt.figure(figsize=(8, 4))
sns.barplot(x=cnt_srs.index, y=cnt_srs.values, alpha=0.8)
plt.ylabel("Number of Occurrences", fontsize=12)
plt.xlabel("Author Name", fontsize=12)
plt.show()



## === cell 4
grouped_df = train_df.groupby("author")
for name, group in grouped_df:
    print("Author name : ", name)
    cnt = 0
    for ind, row in group.iterrows():
        print(row["text"])
        cnt += 1
        if cnt == 5:
            break
    print("\n")



## === cell 5
train_df["num_words"] = train_df["text"].apply(lambda x: len(str(x).split()))
test_df["num_words"] = test_df["text"].apply(lambda x: len(str(x).split()))

train_df["num_unique_words"] = train_df["text"].apply(
    lambda x: len(set(str(x).split()))
)
test_df["num_unique_words"] = test_df["text"].apply(lambda x: len(set(str(x).split())))

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
    lambda x: np.mean([len(w) for w in str(x).split()])
)
test_df["mean_word_len"] = test_df["text"].apply(
    lambda x: np.mean([len(w) for w in str(x).split()])
)



## === cell 6
train_df["num_words"].loc[
    train_df["num_words"] > 80
] = 80  # truncation for better visuals
plt.figure(figsize=(12, 8))
sns.violinplot(x="author", y="num_words", data=train_df)
plt.xlabel("Author Name", fontsize=12)
plt.ylabel("Number of words in text", fontsize=12)
plt.title("Number of words by author", fontsize=15)
plt.show()



## === cell 7
train_df["num_punctuations"].loc[
    train_df["num_punctuations"] > 10
] = 10  # truncation for better visuals
plt.figure(figsize=(12, 8))
sns.violinplot(x="author", y="num_punctuations", data=train_df)
plt.xlabel("Author Name", fontsize=12)
plt.ylabel("Number of puntuations in text", fontsize=12)
plt.title("Number of punctuations by author", fontsize=15)
plt.show()



## === cell 8
author_mapping_dict = {"EAP": 0, "HPL": 1, "MWS": 2}
train_y = train_df["author"].map(author_mapping_dict)
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




## === cell 9
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




## === cell 10
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
    break  # keep original early‑stop behaviour
print("cv scores : ", cv_scores)



## === cell 11
fig, ax = plt.subplots(figsize=(12, 12))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()



## === cell 12
tfidf_vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 3))
tfidf_vec.fit(train_df["text"].values.tolist() + test_df["text"].values.tolist())
train_tfidf = tfidf_vec.transform(train_df["text"].values.tolist())
test_tfidf = tfidf_vec.transform(test_df["text"].values.tolist())




## === cell 13
def runMNB(train_X, train_y, test_X, test_y, test_X2):
    model = naive_bayes.MultinomialNB()
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)
    pred_test_y2 = model.predict_proba(test_X2)
    return pred_test_y, pred_test_y2, model




## === cell 14
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

print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 5.0

train_df["nb_cvec_eap"] = pred_train[:, 0]
train_df["nb_cvec_hpl"] = pred_train[:, 1]
train_df["nb_cvec_mws"] = pred_train[:, 2]
test_df["nb_cvec_eap"] = pred_full_test[:, 0]
test_df["nb_cvec_hpl"] = pred_full_test[:, 1]
test_df["nb_cvec_mws"] = pred_full_test[:, 2]



## === cell 15
tfidf_vec = CountVectorizer(stop_words="english", ngram_range=(1, 3))
tfidf_vec.fit(train_df["text"].values.tolist() + test_df["text"].values.tolist())
train_tfidf = tfidf_vec.transform(train_df["text"].values.tolist())
test_tfidf = tfidf_vec.transform(test_df["text"].values.tolist())



## === cell 16
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

print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 5.0

train_df["nb_cvec_char_eap"] = pred_train[:, 0]
train_df["nb_cvec_char_hpl"] = pred_train[:, 1]
train_df["nb_cvec_char_mws"] = pred_train[:, 2]
test_df["nb_cvec_char_eap"] = pred_full_test[:, 0]
test_df["nb_cvec_char_hpl"] = pred_full_test[:, 1]
test_df["nb_cvec_char_mws"] = pred_full_test[:, 2]



## === cell 17
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



## === cell 18
fig, ax = plt.subplots(figsize=(12, 12))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()
