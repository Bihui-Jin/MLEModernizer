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
Given simulated manufacturing control data, predict whether the machine is in state `0` or state `1`.

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability for the `target` variable. The file should contain a header and have the following format:

```
id,target
900000,0.65
900001,0.97
900002,0.02
etc.
```

## Dataset
- **train.csv** - the training data, which includes normalized continuous data and categorical data
- **test.csv** - the test set; your task is to predict binary `target` variable which represents the state of a manufacturing process
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        input/
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        working/
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
```

-> data/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/tabular-playground-series-may-2022/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> data/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> input/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> (stopped after 10 files for performance)

# 5. Target score

0.9981346771070628

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, warnings

warnings.filterwarnings("ignore")



## === cell 1
import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:50]:
        print(os.path.join(dirname, filename))



## === cell 2
train_path = "/kaggle/input/tabular-playground-series-may-2022/train.csv"
test_path = "/kaggle/input/tabular-playground-series-may-2022/test.csv"
sample_path = "/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

assert "target" in train.columns
assert "id" in train.columns and "id" in test.columns
assert list(sample_sub.columns) == ["id", "target"]



## === cell 3
from sklearn.model_selection import train_test_split

X = train.drop(columns=["target"])
y = train["target"].astype(int)

all_data = pd.concat([X, test], axis=0, ignore_index=True)

cat_cols = all_data.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in all_data.columns if c not in cat_cols]

for c in num_cols:
    all_data[c] = all_data[c].fillna(all_data[c].median())
for c in cat_cols:
    all_data[c] = all_data[c].fillna("NA")

all_data_enc = pd.get_dummies(all_data, columns=cat_cols, drop_first=False)

X_enc = all_data_enc.iloc[: len(X), :].copy()
test_enc = all_data_enc.iloc[len(X) :, :].copy()

if "id" in X_enc.columns:
    X_enc = X_enc.drop(columns=["id"])
if "id" in test_enc.columns:
    test_enc = test_enc.drop(columns=["id"])



## === cell 4
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import HistGradientBoostingClassifier

X_tr, X_va, y_tr, y_va = train_test_split(
    X_enc, y, test_size=0.1, random_state=42, stratify=y
)

grv_vsn_model = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=False)),  # sparse-safe
        ("clf", LogisticRegression(max_iter=200, n_jobs=-1, solver="lbfgs")),
    ]
)

gbdt_model = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_depth=None,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    random_state=42,
)

nn_model = GaussianNB()

grv_vsn_model.fit(X_enc, y)
gbdt_model.fit(X_enc, y)

nn_model.fit(X_enc.to_numpy(dtype=np.float32), y.to_numpy())

grv_vsn_pred = grv_vsn_model.predict_proba(test_enc)[:, 1]
gbdt_pred = gbdt_model.predict_proba(test_enc)[:, 1]
nn_pred = nn_model.predict_proba(test_enc.to_numpy(dtype=np.float32))[:, 1]

grv_vsn = pd.DataFrame({"id": test["id"].values, "target": grv_vsn_pred})
gbdt = pd.DataFrame({"id": test["id"].values, "target": gbdt_pred})
nn = pd.DataFrame({"id": test["id"].values, "target": nn_pred})



## === cell 5
print(grv_vsn.head())
print(gbdt.head())
print(nn.head())

assert grv_vsn.shape[0] == test.shape[0]
assert gbdt.shape[0] == test.shape[0]
assert nn.shape[0] == test.shape[0]



## === cell 6
grv_vsn = grv_vsn.sort_values("id").reset_index(drop=True)
gbdt = gbdt.sort_values("id").reset_index(drop=True)
nn = nn.sort_values("id").reset_index(drop=True)

assert (grv_vsn["id"].values == gbdt["id"].values).all()
assert (grv_vsn["id"].values == nn["id"].values).all()



## === cell 7
weights = [0.01, 0.04, 0.95]
ensamble = grv_vsn.copy()
ensamble["target"] = (
    weights[0] * grv_vsn["target"].values
    + weights[1] * gbdt["target"].values
    + weights[2] * nn["target"].values
)

ensamble.head()



## === cell 8
sub = sample_sub.sort_values("id").reset_index(drop=True)

ensamble_sorted = ensamble.sort_values("id").reset_index(drop=True)
assert (sub["id"].values == ensamble_sorted["id"].values).all()

sub["target"] = ensamble_sorted["target"].astype(float).clip(0.0, 1.0)

out_path = "my_ensamble_052222.csv"
sub.to_csv(out_path, index=False)
print(f"Wrote submission to: {out_path}")
print(sub.head())
