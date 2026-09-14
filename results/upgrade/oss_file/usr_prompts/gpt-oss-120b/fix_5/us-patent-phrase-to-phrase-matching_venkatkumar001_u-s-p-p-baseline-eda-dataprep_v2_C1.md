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

0.20578

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.22518) has done: 'I make the data‑loading paths robust so the script finds the CSV files regardless of the current working directory, and I drop the unnecessary StandardScaler step (CatBoost works well with raw integer encodings). These minimal changes keep the original modeling pipeline intact while likely improving the Pearson correlation enough to bring the score closer to the target. The script also explicitly writes the submission to the standard Kaggle working folder.'
- What this solution (achieved 0.20578) has done: 'I lower the number of boosting iterations from 2000 to 500 so the CatBoost models train less aggressively, which modestly reduces the validation Pearson correlation and moves the score closer to the target (without altering the core pipeline). The rest of the script stays unchanged, ensuring it still runs end‑to‑end and writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import datetime as dt
from scipy.stats import pearsonr
from pathlib import Path




## === cell 1
base_path = Path("/kaggle/input/us-patent-phrase-to-phrase-matching")
if not base_path.exists():
    base_path = Path("../input/us-patent-phrase-to-phrase-matching")
train_path = base_path / "train.csv"
test_path = base_path / "test.csv"
sample_path = base_path / "sample_submission.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample = pd.read_csv(sample_path)
print(
    f"train_shape: {train.shape}, test_shape: {test.shape}, sample_shape: {sample.shape}"
)




## === cell 2
cat_cols = ["anchor", "target", "context"]
combined = pd.concat([train[cat_cols], test[cat_cols]], axis=0)

for col in cat_cols:
    combined[col] = combined[col].astype("category")
    codes, _ = pd.factorize(combined[col], sort=True)
    train[col] = codes[: len(train)]
    test[col] = codes[len(train) :]




## === cell 3
from sklearn import model_selection

train["kfold"] = -1
kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(kfold.split(train)):
    train.loc[valid_idx, "kfold"] = fold

print(train.kfold.value_counts())
train.to_csv("trainfold_5.csv", index=False)




## === cell 4
df = pd.read_csv("./trainfold_5.csv")
useful_features = [c for c in train.columns if c not in ("id", "score", "kfold")]
test = test[useful_features]




## === cell 5
from catboost import CatBoostRegressor

prediction = []
validation_corrs = []

for fold in range(5):
    xtrain = df[df.kfold != fold].reset_index(drop=True)
    xvalid = df[df.kfold == fold].reset_index(drop=True)
    xtest = test.copy()

    ytrain = xtrain["score"]
    yvalid = xvalid["score"]

    xtrain = xtrain[useful_features]
    xvalid = xvalid[useful_features]

    c_para = {
        "iterations": 500,  # decreased from 2000
        "use_best_model": False,
        "early_stopping_rounds": 300,
        "learning_rate": 0.00013589,
        "border_count": 32,
        "verbose": False,
        "random_state": 228,
        "subsample": 0.95312,
        "max_depth": 6,
        "min_data_in_leaf": 77,
        "l2_leaf_reg": 0.02247766515106271,
    }
    cat_boost = CatBoostRegressor(**c_para)
    cat_boost.fit(xtrain, ytrain, eval_set=[(xvalid, yvalid)], verbose=False)

    preds_valid = cat_boost.predict(xvalid)
    corr, _ = pearsonr(yvalid, preds_valid)
    validation_corrs.append(corr)
    print(f"fold {fold} validation Pearson r: {corr:.5f}")

    test_predict = cat_boost.predict(xtest)
    prediction.append(test_predict)

print(f"Mean validation Pearson r across folds: {np.mean(validation_corrs):.5f}")




## === cell 6
final_predict = np.mean(np.column_stack(prediction), axis=1)
sample["score"] = final_predict
submission_path = Path("/kaggle/working/submission.csv")
sample.to_csv(submission_path, index=False)
print(f"Submission file '{submission_path}' written successfully.")




## === cell 7
sample.head(5)
