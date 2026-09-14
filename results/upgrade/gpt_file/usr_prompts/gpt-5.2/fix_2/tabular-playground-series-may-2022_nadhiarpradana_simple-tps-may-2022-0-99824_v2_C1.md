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

0.9982021887045696

# 6. Current score

0.65823

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.65823) has done: 'The current notebook fails because it tries to read three external submission files that are not present in this Kaggle environment, so `subs`/`sub` never get created and all later cells error. To keep the “blend submissions” core logic while making it runnable end-to-end, I add a safe fallback: if those files don’t exist, train a simple scikit-learn logistic regression on `train.csv` (with one-hot encoding for categoricals) and use its predicted probabilities as a single “submission” to blend. This produces a valid `submission.csv` with the required `id,target` columns and should yield a strong AUC for TPS May 2022 without changing evaluation semantics. I also add minimal path resolution so it works with the provided `/kaggle/input/...` layout.'

# 9. Code solution

## === cell 0
import os
import pandas as pd



## === cell 1
candidate_paths = [
    "../input/tps-may-2022/submission-99822.csv",
    "../input/tps-may-2022/submission-99821.csv",
    "../input/tps-may-2022/submission-99814.csv",
]

subs = []
missing = []
for p in candidate_paths:
    if os.path.exists(p):
        df = pd.read_csv(p)
        subs.append(df)
    else:
        missing.append(p)

if len(subs) == 0:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    base_candidates = [
        "/kaggle/input/tabular-playground-series-may-2022",
        "/kaggle/data/tabular-playground-series-may-2022",
        "/kaggle/input",
        "/kaggle/data",
    ]

    train_path = None
    test_path = None
    for base in base_candidates:
        tp = os.path.join(base, "train.csv")
        tep = os.path.join(base, "test.csv")
        if os.path.exists(tp) and os.path.exists(tep):
            train_path, test_path = tp, tep
            break
        tp = os.path.join(base, "tabular-playground-series-may-2022", "train.csv")
        tep = os.path.join(base, "tabular-playground-series-may-2022", "test.csv")
        if os.path.exists(tp) and os.path.exists(tep):
            train_path, test_path = tp, tep
            break

    if train_path is None or test_path is None:
        raise FileNotFoundError(
            "Could not find train.csv/test.csv in expected Kaggle paths. "
            f"Missing external submissions: {missing}"
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    if "target" not in train.columns:
        raise ValueError("train.csv does not contain 'target' column.")

    y = train["target"].astype(int)
    X = train.drop(columns=["target"])
    X_test = test.copy()

    feature_cols = [c for c in X.columns if c != "id"]
    X = X[feature_cols]
    X_test = X_test[feature_cols]

    cat_cols = [c for c in feature_cols if X[c].dtype == "object"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=200,
        n_jobs=None,
        verbose=0,
    )

    model = Pipeline(steps=[("pre", pre), ("clf", clf)])
    model.fit(X, y)

    proba = model.predict_proba(X_test)[:, 1]

    sub = pd.DataFrame({"id": test["id"].values, "target": proba})
    subs = [sub]
else:
    sub = subs[0].copy()



## === cell 2
W = [0.8, 0.8, 0.6]

if len(subs) != len(W):
    W = W[: len(subs)]

total = sum(W)
W = [w / total for w in W]

state = 0
for i in range(len(W)):
    state += W[i] * subs[i]["target"]

sub = subs[0].copy()
sub["target"] = state



## === cell 3
if len(subs) >= 2:
    state_0 = subs[0]["target"]
    state_blend = sub["target"]
    diff = state_blend - state_0
    _diag = pd.DataFrame(
        data={"state_0": state_0, "state_blend": state_blend, "difference": diff}
    )
    _ = _diag.head()
else:
    _ = sub.head()



## === cell 4
sub = sub[["id", "target"]].copy()
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
