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

0.63696

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.63696) has done: 'We remove the problematic keras imports and the failing neural‑network cells, keep the XGBoost + RandomForest pipeline, fix the test‑set prediction to convert back from the log‑scale, and finally write a proper `submission.csv` with the required columns. This resolves the import error, eliminates the ValueError in cell 27, and ensures a valid CSV is produced while preserving the original modeling logic.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import pickle

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)



## === cell 1
train_path = "../input/us-patent-phrase-to-phrase-matching/train.csv"
test_path = "../input/us-patent-phrase-to-phrase-matching/test.csv"
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)



## === cell 2
stop_words = set(stopwords.words("english"))
ls = LancasterStemmer()

train["letter"] = train["context"].str[0]
train["anchor_token"] = train["anchor"].apply(word_tokenize)
train["anchor_len"] = train["anchor_token"].apply(len)
train["anchor_nostop"] = train["anchor_token"].apply(
    lambda x: [w for w in x if w not in stop_words]
)
train["anchor_stem"] = train["anchor_nostop"].apply(lambda x: [ls.stem(w) for w in x])
train["new_anchor"] = train["anchor_stem"].apply(lambda x: " ".join(x))
train["target_token"] = train["target"].apply(word_tokenize)
train["target_len"] = train["target_token"].apply(len)
train["target_nostop"] = train["target_token"].apply(
    lambda x: [w for w in x if w not in stop_words]
)
train["target_stem"] = train["target_nostop"].apply(lambda x: [ls.stem(w) for w in x])
train["new_target"] = train["target_stem"].apply(lambda x: " ".join(x))
train["final"] = train["new_anchor"] + " " + train["new_target"]

matchlist = []
for a, t in zip(train["anchor_stem"], train["target_stem"]):
    matchlist.append(sum(1 for w in a if w in t))
train["matches"] = matchlist

train_dummies = pd.get_dummies(train["letter"])
train = pd.concat([train.drop("letter", axis=1), train_dummies], axis=1)

train["score"] = np.log1p(train["score"])

test["letter"] = test["context"].str[0]
test["anchor_token"] = test["anchor"].apply(word_tokenize)
test["anchor_len"] = test["anchor_token"].apply(len)
test["anchor_nostop"] = test["anchor_token"].apply(
    lambda x: [w for w in x if w not in stop_words]
)
test["anchor_stem"] = test["anchor_nostop"].apply(lambda x: [ls.stem(w) for w in x])
test["new_anchor"] = test["anchor_stem"].apply(lambda x: " ".join(x))
test["target_token"] = test["target"].apply(word_tokenize)
test["target_len"] = test["target_token"].apply(len)
test["target_nostop"] = test["target_token"].apply(
    lambda x: [w for w in x if w not in stop_words]
)
test["target_stem"] = test["target_nostop"].apply(lambda x: [ls.stem(w) for w in x])
test["new_target"] = test["target_stem"].apply(lambda x: " ".join(x))
test["final"] = test["new_anchor"] + " " + test["new_target"]

matchlist = []
for a, t in zip(test["anchor_stem"], test["target_stem"]):
    matchlist.append(sum(1 for w in a if w in t))
test["matches"] = matchlist

test_dummies = pd.get_dummies(test["letter"])
test = pd.concat([test.drop("letter", axis=1), test_dummies], axis=1)



## === cell 3
vectorizer = CountVectorizer(max_features=2000)
vectorizer.fit(train["final"])
pickle.dump(vectorizer, open("patent_vectorizer.pickle", "wb"))

matrixtrain = vectorizer.transform(train["final"])
trainvect = pd.DataFrame(
    matrixtrain.toarray(), columns=vectorizer.get_feature_names_out()
)
trainvect["score"] = train["score"].reset_index(drop=True)
trainvect["anchor_len"] = train["anchor_len"]
trainvect["target_len"] = train["target_len"]
trainvect["matches"] = train["matches"]
trainvect = pd.concat([trainvect, train_dummies.reset_index(drop=True)], axis=1)

trainx, valx, trainy, valy = train_test_split(
    trainvect.drop(["score"], axis=1),
    trainvect["score"],
    test_size=0.2,
    random_state=777,
)



## === cell 4
xgb_model = XGBRegressor(
    subsample=1.0,
    n_estimators=400,
    min_child_weight=5,
    max_depth=6,
    learning_rate=0.2,
    gamma=0,
    colsample_bytree=0.6,
    objective="reg:squarederror",
    verbosity=0,
)
xgb_model.fit(trainx, trainy)

rf_model = RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=1,
    min_samples_split=6,
    max_features=20,
    bootstrap=True,
    max_depth=150,
    max_samples=0.6,
    random_state=42,
    n_jobs=-1,
)
rf_model.fit(trainx, trainy)



## === cell 5
matrixtest = vectorizer.transform(test["final"])
dftest = pd.DataFrame(matrixtest.toarray(), columns=vectorizer.get_feature_names_out())
dftest["anchor_len"] = test["anchor_len"]
dftest["target_len"] = test["target_len"]
dftest["matches"] = test["matches"]
dftest = pd.concat([dftest, test_dummies.reset_index(drop=True)], axis=1)

pred_xgb_log = xgb_model.predict(dftest)
pred_rf_log = rf_model.predict(dftest)

combined_log = 0.5 * pred_xgb_log + 0.5 * pred_rf_log
test["score"] = np.expm1(combined_log)



## === cell 6
submission = test[["id", "score"]].copy()
submission.to_csv("submission.csv", index=False)
