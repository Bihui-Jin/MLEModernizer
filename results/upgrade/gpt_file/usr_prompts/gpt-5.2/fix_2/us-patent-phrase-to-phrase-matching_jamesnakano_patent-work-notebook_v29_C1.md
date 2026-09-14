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
Given pairs of phrases (an `anchor` and a `target` phrase), build a model to rate how similar they are.  

## Metric
Pearson correlation coefficient.

## Submission Format
For each `id` (representing a pair of phrases) in the test set, you must predict the similarity `score`. The file should contain a header and have the following format:

```
id,score
4112d61851461f60,0
09e418c93a776564,0.25
36baf228038e314b,1
etc.

```

## Dataset
The scores are in the 0-1 range with increments of 0.25 with the following meanings:

- **1.0** - Very close match. This is typically an exact match except possibly for differences in conjugation, quantity (e.g. singular vs. plural), and addition or removal of stopwords (e.g. "the", "and", "or").
- **0.75** - Close synonym, e.g. "mobile phone" vs. "cellphone". This also includes abbreviations, e.g. "TCP" -> "transmission control protocol".
- **0.5** - Synonyms which don't have the same meaning (same function, same properties). This includes broad-narrow (hyponym) and narrow-broad (hypernym) matches.
- **0.25** - Somewhat related, e.g. the two phrases are in the same high level domain but are not synonyms. This also includes antonyms.
- **0.0** - Unrelated.

Files
-----

- **train.csv** - the training set, containing phrases, contexts, and their similarity scores
- **test.csv** - the test set set, identical in structure to the training set but without the score
- **sample_submission.csv** - a sample submission file in the correct format

Columns
-------

- `id` - a unique identifier for a pair of phrases
- `anchor` - the first phrase
- `target` - the second phrase
- `context` - the [CPC classification (version 2021.05)](https://en.wikipedia.org/wiki/Cooperative_Patent_Classification), which indicates the subject within which the similarity is to be scored
- `score` - the similarity. This is sourced from a combination of one or more manual expert ratings.

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
scipy==1.15.3
sklearn-pandas==2.2.0
tf_keras==2.18.0
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        input/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        working/
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
```

-> data/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> data/us-patent-phrase-to-phrase-matching/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/us-patent-phrase-to-phrase-matching/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/us-patent-phrase-to-phrase-matching/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> input/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> (stopped after 10 files for performance)

# 5. Target score

0.3841

# 6. Current score

0.63667

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.63667) has done: 'I fix the initial runtime crash by removing the unused Keras import that triggers a protobuf incompatibility in this environment, and I add robust NLTK resource downloads so tokenization/stopwords don’t fail. Then I fix the Keras training error by ensuring all feature matrices passed to Keras are purely numeric `float32` and by aligning train/test dummy columns (missing letters cause object dtypes and column mismatches). Finally, I ensure the script always writes a valid `submission.csv` with exactly `id,score`, converting predictions back from the log1p target and clipping to `[0, 1]` (score-neutral/format-correctness) so a submission is yielded.'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import LancasterStemmer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import pickle




## === cell 1
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

from nltk import word_tokenize



## === cell 2
train = pd.read_csv("../input/us-patent-phrase-to-phrase-matching/train.csv")
test = pd.read_csv("../input/us-patent-phrase-to-phrase-matching/test.csv")
train.shape, test.shape



## === cell 3
stop_words = stopwords.words("english")
ls = LancasterStemmer()



## === cell 4
train["letter"] = train["context"].apply(lambda x: x[0])
train["anchor_token"] = train["anchor"].apply(lambda x: word_tokenize(x))
train["anchor_len"] = train["anchor_token"].apply(lambda x: len(x))
train["anchor_nostop"] = train["anchor_token"].apply(
    lambda x: [word for word in x if word not in stop_words]
)
train["anchor_stem"] = train["anchor_nostop"].apply(
    lambda x: [ls.stem(word) for word in x]
)
train["new_anchor"] = train["anchor_stem"].apply(
    lambda x: " ".join([str(word) for word in x])
)
train["target_token"] = train["target"].apply(lambda x: word_tokenize(x))
train["target_len"] = train["target_token"].apply(lambda x: len(x))
train["target_nostop"] = train["target_token"].apply(
    lambda x: [word for word in x if word not in stop_words]
)
train["target_stem"] = train["target_nostop"].apply(
    lambda x: [ls.stem(word) for word in x]
)
train["new_target"] = train["target_stem"].apply(
    lambda x: " ".join([word for word in x])
)
train["final"] = train["new_anchor"] + " " + train["new_target"]

matchlist = []
for x, y in zip(train["anchor_stem"], train["target_stem"]):
    matches = 0
    for word in x:
        if word in y:
            matches += 1
    matchlist.append(matches)
train["matches"] = matchlist

train_dummies = pd.get_dummies(train["letter"])
train = pd.concat([train.drop("letter", axis=1), train_dummies], axis=1)

train["score"] = np.log1p(train["score"])



## === cell 5
test["letter"] = test["context"].apply(lambda x: x[0])
test["anchor_token"] = test["anchor"].apply(lambda x: word_tokenize(x))
test["anchor_len"] = test["anchor_token"].apply(lambda x: len(x))
test["anchor_nostop"] = test["anchor_token"].apply(
    lambda x: [word for word in x if word not in stop_words]
)
test["anchor_stem"] = test["anchor_nostop"].apply(
    lambda x: [ls.stem(word) for word in x]
)
test["new_anchor"] = test["anchor_stem"].apply(
    lambda x: " ".join([str(word) for word in x])
)
test["target_token"] = test["target"].apply(lambda x: word_tokenize(x))
test["target_len"] = test["target_token"].apply(lambda x: len(x))
test["target_nostop"] = test["target_token"].apply(
    lambda x: [word for word in x if word not in stop_words]
)
test["target_stem"] = test["target_nostop"].apply(
    lambda x: [ls.stem(word) for word in x]
)
test["new_target"] = test["target_stem"].apply(lambda x: " ".join([word for word in x]))
test["final"] = test["new_anchor"] + " " + test["new_target"]

matchlist = []
for x, y in zip(test["anchor_stem"], test["target_stem"]):
    matches = 0
    for word in x:
        if word in y:
            matches += 1
    matchlist.append(matches)
test["matches"] = matchlist

test_dummies = pd.get_dummies(test["letter"])
test = pd.concat([test.drop("letter", axis=1), test_dummies], axis=1)



## === cell 6
vectorizer = CountVectorizer(max_features=2000)
vectortrain = vectorizer.fit(train["final"])
pickle.dump(vectortrain, open("patent_vectorizer.pickle", "wb"))

matrixtrain = vectorizer.transform(train["final"])
trainvect = pd.DataFrame(
    matrixtrain.toarray(), columns=vectortrain.get_feature_names_out()
)

trainvect["score"] = train["score"].to_numpy()
trainvect["anchor_len"] = train["anchor_len"].to_numpy()
trainvect["target_len"] = train["target_len"].to_numpy()
trainvect["matches"] = train["matches"].to_numpy()
trainvect = pd.concat([trainvect, train_dummies.reset_index(drop=True)], axis=1)

trainvect = trainvect.apply(pd.to_numeric, errors="coerce").fillna(0.0)

trainx, testx, trainy, testy = train_test_split(
    trainvect.drop(["score"], axis=1),
    trainvect["score"],
    test_size=0.2,
    random_state=777,
)



## === cell 7
matrixtrain.shape



## === cell 8
pass



## === cell 9
rfr1 = RandomForestRegressor()
parameters = {
    "n_estimators": [200, 300, 400],
    "max_features": ["auto", 10, 20],
    "min_samples_split": [6],
    "min_samples_leaf": [1],
    "bootstrap": [True, False],
}



## === cell 10
pass



## === cell 11
xgb1 = XGBRegressor()
parameters = {
    "gamma": [0, 1, 2],
    "learning_rate": [0.2, 0.25, 0.1],
    "max_depth": [6, 8, 10, None],
    "min_child_weight": [5],
    "subsample": [1.0],
    "colsample_bytree": [0.6],
    "n_estimators": [400],
}



## === cell 12
pass



## === cell 13
testy = np.expm1(testy)



## === cell 14
xgbr = XGBRegressor(
    subsample=1.0,
    n_estimators=400,
    min_child_weight=5,
    max_depth=6,
    learning_rate=0.2,
    gamma=0,
    colsample_bytree=0.6,
    random_state=777,
)
xgbr_fit = xgbr.fit(trainx, trainy, eval_set=[(testx, np.log1p(testy))], verbose=False)

pred1 = np.expm1(xgbr_fit.predict(testx))
print(pearsonr(testy, pred1)[0])
pickle.dump(xgbr_fit, open("patent_xgb.pickle", "wb"))



## === cell 15
rfr = RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=1,
    min_samples_split=6,
    max_features=20,
    bootstrap=True,
    max_depth=150,
    max_samples=0.6,
    random_state=777,
    n_jobs=-1,
)
rfr_fit = rfr.fit(trainx, trainy)
pred = np.expm1(rfr_fit.predict(testx))
print(pearsonr(testy, pred)[0])
pickle.dump(rfr_fit, open("patent_rfr.pickle", "wb"))



## === cell 16
pearsonr(testy, (pred1 * 0.5 + pred * 0.5))[0]



## === cell 17
pred



## === cell 18
pred1



## === cell 19
testy



## === cell 20
matrixtest = vectorizer.transform(test["final"])
dftest = pd.DataFrame(matrixtest.toarray(), columns=vectortrain.get_feature_names_out())
dftest["anchor_len"] = test["anchor_len"].to_numpy()
dftest["target_len"] = test["target_len"].to_numpy()
dftest["matches"] = test["matches"].to_numpy()

test_dummies_aligned = test_dummies.reindex(
    columns=train_dummies.columns, fill_value=0
).reset_index(drop=True)
dftest = pd.concat([dftest, test_dummies_aligned], axis=1)

dftest = dftest.apply(pd.to_numeric, errors="coerce").fillna(0.0)

prediction0 = xgbr_fit.predict(dftest)
prediction1 = rfr_fit.predict(dftest)

pred_final = np.expm1(prediction0 * 0.5 + prediction1 * 0.5)

pred_final = np.clip(pred_final, 0.0, 1.0)

finaldf = pd.DataFrame({"id": test["id"], "score": pred_final})



## === cell 21
finaldf["score"]



## === cell 22
finaldf.to_csv("submission.csv", index=False)
print(finaldf.head())
print("Wrote submission.csv with shape:", finaldf.shape)
