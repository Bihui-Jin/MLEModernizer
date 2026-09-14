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

catboost==1.2.8
geopandas==0.14.4
lightgbm==4.6.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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

0.1455

# 6. Current score

0.22517

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.22517) has done: 'I make the notebook run end-to-end on Kaggle by removing the Jupyter-only magic and fixing the cell numbering so it’s valid as a script. To move the score upward (since you currently can’t submit), I fix a major feature-encoding bug: you’re using separate category codings for train vs test, which makes the numeric IDs inconsistent and harms predictions; we fit the categorical mapping on the combined train+test per column to keep codes aligned without changing the core model. I also fix a column selection mistake (you build `useful_features` from `train.columns` instead of the fold dataframe), and I switch the CV split to use `train[useful_features]` in `kfold.split(...)` to avoid any unintended dependency on non-feature columns. Finally, I keep the CatBoost training loop and hyperparameters intact, but ensure the submission aligns to `sample_submission.csv` by id order and writes a valid `submission.csv`.'
- What this solution (achieved 0.22202) has done: 'I make the notebook reliably produce a valid `submission.csv` (your current score is “Not yielded”, so the first priority is removing failure points). The biggest risk in your current pipeline is CatBoost not fitting due to `use_best_model=True` combined with `verbose=False` and no explicit `loss_function`, plus potential instability from scaling integer-coded categories; I keep your feature set and CV/training loop identical but add a fixed `loss_function='RMSE'`, explicit `thread_count`, and deterministic settings so it runs end-to-end. I also ensure the submission rows align exactly to `sample_submission.csv`’s `id` order by merging on `id` (rather than relying on identical row order), which prevents silent misalignment that can destroy correlation. Finally, I keep your 0.25 rounding (evaluation semantics) but guard against NaNs/infs in predictions.'
- What this solution (achieved 0.22517) has done: 'Your current “Not yielded” is most likely due to runtime/timeout (15k iterations × 5 folds is too slow under a 600s cap), not submission formatting (your merge-by-id looks correct). To get a valid submission while keeping the same CatBoost CV loop and features, I switch on `use_best_model=True` and stop passing `early_stopping_rounds` as a constructor arg (CatBoost uses `od_type/od_wait`), so training actually terminates early based on the validation set. This preserves your model family and evaluation semantics (still regression + no post-rounding), but makes the notebook reliably finish and write `submission.csv`. I also cap `thread_count` to a reasonable value for Kaggle stability while remaining deterministic.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import datetime as dt

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

matplotlib.rcParams["figure.figsize"] = (20, 10)



## === cell 1
start_time = dt.datetime.now()
print("started at", start_time)



## === cell 2
train = pd.read_csv("../input/us-patent-phrase-to-phrase-matching/train.csv")
test = pd.read_csv("../input/us-patent-phrase-to-phrase-matching/test.csv")
sample = pd.read_csv(
    "../input/us-patent-phrase-to-phrase-matching/sample_submission.csv"
)

print(
    f"train_shape: {train.shape}, test_shape: {test.shape}, sample_shape: {sample.shape}"
)
print(train.head())



## === cell 3
for col in ["anchor", "target", "context"]:
    comb = pd.concat([train[col], test[col]], axis=0).astype("category")
    train[col] = pd.Categorical(train[col], categories=comb.cat.categories).codes
    test[col] = pd.Categorical(test[col], categories=comb.cat.categories).codes



## === cell 4
from sklearn import model_selection

useful_features = [c for c in train.columns if c not in ("id", "score")]

train["kfold"] = -1
kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (_, valid_indices) in enumerate(kfold.split(X=train[useful_features])):
    train.loc[valid_indices, "kfold"] = fold

print(train.kfold.value_counts())
train.to_csv("trainfold_5.csv", index=False)



## === cell 5
df = pd.read_csv("./trainfold_5.csv")

useful_features = [c for c in df.columns if c not in ("id", "score", "kfold")]
test_model = test[useful_features].copy()



## === cell 6
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler

prediction = []

for fold in range(5):
    xtrain = df[df.kfold != fold].reset_index(drop=True)
    xvalid = df[df.kfold == fold].reset_index(drop=True)
    xtest = test_model.copy()

    ytrain = xtrain["score"]
    yvalid = xvalid["score"]

    xtrain = xtrain[useful_features].copy()
    xvalid = xvalid[useful_features].copy()

    scaler = StandardScaler()
    xtrain[useful_features] = scaler.fit_transform(xtrain[useful_features])
    xvalid[useful_features] = scaler.transform(xvalid[useful_features])
    xtest[useful_features] = scaler.transform(xtest[useful_features])

    c_para = {
        "loss_function": "RMSE",
        "iterations": 15000,
        "use_best_model": True,
        "od_type": "Iter",
        "od_wait": 300,
        "learning_rate": 0.00013589,
        "border_count": 32,
        "verbose": False,
        "random_state": 228,
        "subsample": 0.95312,
        "max_depth": 6,
        "min_data_in_leaf": 77,
        "l2_leaf_reg": 0.02247766515106271,
        "thread_count": 4,
        "allow_writing_files": False,
    }

    cat_boost = CatBoostRegressor(**c_para)
    cat_boost.fit(xtrain, ytrain, eval_set=[(xvalid, yvalid)])

    test_predict = cat_boost.predict(xtest)
    prediction.append(test_predict)
    print(f"complete fold:{fold}")



## === cell 7
final_predict = np.mean(np.column_stack(prediction), axis=1)

final_predict = np.nan_to_num(final_predict, nan=0.5, posinf=1.0, neginf=0.0)
final_predict = np.clip(final_predict, 0.0, 1.0)

pred_df = pd.DataFrame({"id": test["id"].values, "score": final_predict})
sub = sample[["id"]].merge(pred_df, on="id", how="left")

sub["score"] = sub["score"].fillna(0.5)

assert sub.shape[0] == sample.shape[0], "Row count mismatch vs sample_submission."
assert sub["score"].notna().all(), "Found NaN scores in submission."

sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("finished at", dt.datetime.now(), "elapsed", dt.datetime.now() - start_time)
