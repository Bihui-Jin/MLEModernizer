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

0.86814

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.65823) has done: 'The current notebook fails because it tries to read three external submission files that are not present in this Kaggle environment, so `subs`/`sub` never get created and all later cells error. To keep the “blend submissions” core logic while making it runnable end-to-end, I add a safe fallback: if those files don’t exist, train a simple scikit-learn logistic regression on `train.csv` (with one-hot encoding for categoricals) and use its predicted probabilities as a single “submission” to blend. This produces a valid `submission.csv` with the required `id,target` columns and should yield a strong AUC for TPS May 2022 without changing evaluation semantics. I also add minimal path resolution so it works with the provided `/kaggle/input/...` layout.'
- What this solution (achieved 0.67837) has done: 'You’re far below the target AUC, so we should improve the fallback model (used when the external submissions aren’t present) while keeping the same overall “blend submissions” logic intact. The biggest gain with minimal change is to make LogisticRegression strong on TPS May 2022 by using proper numeric scaling and more iterations/regularization settings that converge well, without changing the model family or the evaluation semantics. I also make sure the weight-blending code behaves sensibly when only one fallback submission exists (it already mostly does), and keep the submission schema (`id,target`) unchanged. These changes should substantially raise AUC toward the target without changing the core approach.'
- What this solution (achieved 0.86814) has done: 'Your current score (0.678) is far below the target (0.9982), so we should improve the fallback model (used when the external submission files are missing) while keeping the same “blend submissions” structure intact. The biggest issue is that plain one-hot + linear logistic regression underfits TPS May 2022; with minimal change, we can switch the fallback classifier to `HistGradientBoostingClassifier`, which handles non-linearities well and is typically near top-performing for this dataset while preserving the same train→predict_proba→blend semantics. We keep the blending logic unchanged (it just blend a single fallback submission), and we add only the necessary preprocessing: label-encoding the single categorical column and imputing numerics. This should move AUC substantially upward toward the target without altering submission format or I/O paths.'

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
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OrdinalEncoder

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

    y = train["target"].astype(int).values
    X = train.drop(columns=["target"])
    X_test = test.copy()

    feature_cols = [c for c in X.columns if c != "id"]
    X = X[feature_cols].copy()
    X_test = X_test[feature_cols].copy()

    cat_cols = [c for c in feature_cols if X[c].dtype == "object"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    if len(cat_cols) > 0:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_cat = enc.fit_transform(X[cat_cols])
        X_test_cat = enc.transform(X_test[cat_cols])
        X_cat = pd.DataFrame(X_cat, columns=cat_cols, index=X.index)
        X_test_cat = pd.DataFrame(X_test_cat, columns=cat_cols, index=X_test.index)
    else:
        X_cat = None
        X_test_cat = None

    imp = SimpleImputer(strategy="median")
    X_num = imp.fit_transform(X[num_cols]) if len(num_cols) > 0 else None
    X_test_num = imp.transform(X_test[num_cols]) if len(num_cols) > 0 else None

    if X_cat is not None and X_num is not None:
        X_proc = pd.concat(
            [pd.DataFrame(X_num, columns=num_cols, index=X.index), X_cat], axis=1
        )
        X_test_proc = pd.concat(
            [
                pd.DataFrame(X_test_num, columns=num_cols, index=X_test.index),
                X_test_cat,
            ],
            axis=1,
        )
    elif X_num is not None:
        X_proc = pd.DataFrame(X_num, columns=num_cols, index=X.index)
        X_test_proc = pd.DataFrame(X_test_num, columns=num_cols, index=X_test.index)
    else:
        X_proc = X_cat.copy()
        X_test_proc = X_test_cat.copy()

    clf = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_depth=6,
        max_leaf_nodes=64,
        min_samples_leaf=20,
        l2_regularization=0.0,
        max_bins=255,
        early_stopping=False,
        random_state=0,
    )
    clf.fit(X_proc, y)
    proba = clf.predict_proba(X_test_proc)[:, 1]

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
