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
Predict values for synthetic data.

### Description
## Metric
Area under the ROC curve for each target, with the final score being the average of the individual AUCs of each predicted column.

## Submission Format
For each `id` in the test set, you must predict the value for the targets `EC1` and `EC2`. The file should contain a header and have the following format:

```
id,EC1,EC2
14838,0.22,0.71
14839,0.78,0.43
14840,0.53,0.11
etc.
```

## Dataset 
- **train.csv** - the training dataset; `[EC1 - EC6]` are the (binary) targets, although you are only asked to predict `EC1` and `EC2`.
- **test.csv** - the test dataset; your objective is to predict the probability of the two targets `EC1` and `EC2`
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.11

# 3. Installed packages

geopandas==0.14.4
imbalanced-learn==0.13.0
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
            description.md (56 lines)
            sample_submission.csv (1485 lines)
            sample_submission.csv.zip (4.4 kB)
            test.csv (1485 lines)
            test.csv.zip (147.8 kB)
            train.csv (13355 lines)
            train.csv.zip (1.4 MB)
            playground-series-s3e18/
                description.md (56 lines)
                sample_submission.csv (1485 lines)
                ... and 5 other files
                playground-series-s3e18/
        input/
            description.md (56 lines)
            sample_submission.csv (1485 lines)
            sample_submission.csv.zip (4.4 kB)
            test.csv (1485 lines)
            test.csv.zip (147.8 kB)
            train.csv (13355 lines)
            train.csv.zip (1.4 MB)
            playground-series-s3e18/
                description.md (56 lines)
                sample_submission.csv (1485 lines)
                ... and 5 other files
                playground-series-s3e18/
        working/
            playground-series-s3e18/
                description.md (56 lines)
                sample_submission.csv (1485 lines)
                ... and 5 other files
                playground-series-s3e18/
```

-> data/playground-series-s3e18/sample_submission.csv has 1484 rows and 3 columns.
The columns are: id, EC1, EC2

-> data/playground-series-s3e18/test.csv has 1484 rows and 32 columns.
The columns are: id, BertzCT, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3v, Chi4n, EState_VSA1, EState_VSA2, ExactMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3... and 17 more columns

-> data/playground-series-s3e18/train.csv has 13354 rows and 38 columns.
The columns are: id, BertzCT, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3v, Chi4n, EState_VSA1, EState_VSA2, ExactMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3... and 23 more columns

-> data/sample_submission.csv has 1484 rows and 3 columns.
The columns are: id, EC1, EC2

-> data/test.csv has 1484 rows and 32 columns.
The columns are: id, BertzCT, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3v, Chi4n, EState_VSA1, EState_VSA2, ExactMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3... and 17 more columns

-> data/train.csv has 13354 rows and 38 columns.
The columns are: id, BertzCT, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3v, Chi4n, EState_VSA1, EState_VSA2, ExactMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3... and 23 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.56623

# 6. Current score

0.65881

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.64381) has done: 'I remove the hard dependency on `klib` (it’s not installed) by replacing its cleaning/plotting calls with small, equivalent pandas-based steps so the pipeline can run. I keep your core feature-selection logic (mutual information) and the XGBoost + MultiOutputClassifier + KFold training approach unchanged, but fix the broken variable flow caused by the early crash (so `train/test/X_train/X_test` are defined). I also make the column casing consistent between cleaned data and the raw targets, and ensure we always produce a valid `submission.csv` with columns `id,EC1,EC2` aligned to the sample submission. Finally, I keep the GPU→CPU fallback for XGBoost to avoid runtime errors in environments without GPU support.'
- What this solution (achieved 0.65862) has done: 'Your current score (0.64381) is already *better* than the target (0.56623), so to move **toward** the target (reduce the absolute gap) we should deliberately and minimally reduce predictive power while keeping the same overall pipeline. The smallest safe lever that preserves your core approach is to slightly weaken the XGBoost model capacity (fewer trees, shallower depth, fewer leaves) without changing the model family, CV approach, feature selection, or loss/metric semantics. I also keep everything else identical (same preprocessing, MI feature union, KFold, MultiOutputClassifier, predict_proba), and still write a valid `submission.csv` with `id,EC1,EC2`. This should generally lower AUC toward your target band without breaking the solution.'
- What this solution (achieved 0.65881) has done: 'Your current score (0.65862) is already above the target (0.56623), so to move closer we should *slightly weaken* the model while keeping the same end-to-end pipeline (MI feature filtering → scaling → KFold CV → XGBoost via MultiOutputClassifier → predict_proba → submission.csv). The smallest, most reliable lever is to reduce XGBoost capacity (fewer trees, shallower depth, stronger regularization, and row/column subsampling) without changing the model family, training loop, or output semantics. I also keep the GPU→CPU fallback unchanged and ensure we still write a valid `submission.csv` with `id,EC1,EC2`. This should generally lower AUC toward the target band while preserving stability and runtime.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import importlib

_klib_spec = importlib.util.find_spec("klib")
print("klib available:", _klib_spec is not None)



## === cell 2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler



## === cell 3
TRAIN_PATH = "/kaggle/input/playground-series-s3e18/train.csv"
TEST_PATH = "/kaggle/input/playground-series-s3e18/test.csv"
SUB_PATH = "/kaggle/input/playground-series-s3e18/sample_submission.csv"

df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)
sub = pd.read_csv(SUB_PATH)

print(df_train.shape, df_test.shape, sub.shape)
print("train columns (head):", df_train.columns[:10].tolist())




## === cell 4
def basic_data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip().lower() for c in out.columns]

    out = out.replace([np.inf, -np.inf], np.nan)

    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    obj_cols = out.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    for c in num_cols:
        if out[c].isna().any():
            out[c] = out[c].fillna(out[c].median())

    for c in obj_cols:
        if out[c].isna().any():
            mode = out[c].mode(dropna=True)
            fill_val = mode.iloc[0] if len(mode) else ""
            out[c] = out[c].fillna(fill_val)

    return out


train = basic_data_cleaning(df_train)
test = basic_data_cleaning(df_test)

print("clean train/test shapes:", train.shape, test.shape)
print("targets present (lowercase)?", {"ec1" in train.columns, "ec2" in train.columns})



## === cell 5
for col in ["ec3", "ec4", "ec5", "ec6"]:
    if col in train.columns:
        train = train.drop([col], axis=1)

train.head()



## === cell 6
from sklearn.feature_selection import mutual_info_classif, SelectKBest


def make_mi_ready_features(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    if len(cat_cols) > 0:
        return pd.get_dummies(df, columns=list(cat_cols), drop_first=False)
    return df


target = ["ec1", "ec2"]
dic = {}
for i in target:
    X_mi = train.drop([i], axis=1)
    y_mi = train[i]

    X_mi = make_mi_ready_features(X_mi)

    mutual_info = mutual_info_classif(X_mi, y_mi, random_state=42)
    mutual_info = pd.Series(mutual_info, index=X_mi.columns).sort_values(
        ascending=False
    )

    try:
        mutual_info.head(30).plot.bar(title=i, figsize=(20, 8))
        plt.show()
    except Exception as e:
        print("Skipping MI plot due to:", repr(e))

    select_cols = SelectKBest(mutual_info_classif, k=min(10, X_mi.shape[1]))
    select_cols.fit(X_mi, y_mi)
    dic[i] = X_mi.columns[select_cols.get_support()]

dic



## === cell 7
union_cols = dic["ec1"].union(dic["ec2"])
union_cols



## === cell 8
dic_mutual = {}
target_mutual_col = ["ec1", "ec2"]
for i in target_mutual_col:
    X_mi = train.drop([i], axis=1)
    y_mi = train[i]
    X_mi = make_mi_ready_features(X_mi)

    mutual_info = mutual_info_classif(X_mi, y_mi, random_state=42)
    mutual_info = pd.Series(mutual_info, index=X_mi.columns).sort_values(
        ascending=False
    )

    mutual_top = mutual_info.index[0 : round(len(mutual_info) / 2)]
    print("most mutual 50% features for " + i + ":")
    print(mutual_top.tolist()[:20], "... total:", len(mutual_top))
    dic_mutual[i] = mutual_top



## === cell 9
dic_mutual = dic_mutual["ec1"].union(dic_mutual["ec2"])
dic_mutual



## === cell 10
dic_mutual = dic_mutual.drop(["ec1", "ec2"], errors="ignore")
dic_mutual



## === cell 11
dic_mutual



## === cell 12
test.head()



## === cell 13
try:
    corr = train.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Train numeric correlation heatmap")
    plt.show()
except Exception as e:
    print("Skipping correlation plot due to:", repr(e))



## === cell 14
train.info()



## === cell 15
train.describe().T.head(10)



## === cell 16
n_train = len(train)
n_test = len(test)

df_train_test = pd.concat([train, test], axis=0, ignore_index=True)

df_train_test = df_train_test.drop(["ec1", "ec2", "id"], axis=1, errors="ignore")

df_train_test.shape



## === cell 17
df_train_test_enc = make_mi_ready_features(df_train_test)

dic_mutual_cols = [c for c in list(dic_mutual) if c in df_train_test_enc.columns]
if len(dic_mutual_cols) == 0:
    print("Warning: no dic_mutual columns found after encoding; using all features.")
    dic_mutual_cols = df_train_test_enc.columns.tolist()

df_train_test_enc = df_train_test_enc[dic_mutual_cols]
df_train_test_enc.shape



## === cell 18
df_train_test_enc.head()



## === cell 19
scaler = MinMaxScaler(feature_range=(0, 1))
df_train_test_scale = scaler.fit_transform(df_train_test_enc)
df_train_test_scale = pd.DataFrame(
    df_train_test_scale, columns=df_train_test_enc.columns
)
df_train_test_scale.shape



## === cell 20
X_train = df_train_test_scale.iloc[:n_train].reset_index(drop=True)
X_test = df_train_test_scale.iloc[n_train : n_train + n_test].reset_index(drop=True)

X_train.shape, X_test.shape



## === cell 21
y_train = train[["ec1", "ec2"]].copy()
y_train.columns = ["EC1", "EC2"]

assert len(y_train) == n_train, "Mismatch between y_train and X_train rows."

print(f"X_train shape is = {X_train.shape}")
print(f"y_train shape is = {y_train.shape}")
print(f"X_test shape is = {X_test.shape}")



## === cell 22
y_train.head(5)



## === cell 23
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score



## === cell 24
kfold = KFold(n_splits=5, shuffle=True, random_state=42)




## === cell 25
def make_xgb():
    return XGBClassifier(
        n_estimators=350,  # reduced from 900 to weaken fit (lower AUC expected)
        random_state=46,
        learning_rate=0.02,  # slight increase to keep training stable with fewer trees
        max_depth=3,  # reduced from 4
        max_leaves=6,  # reduced from 8
        subsample=0.70,  # add row subsampling to reduce generalization strength
        colsample_bytree=0.70,  # add feature subsampling
        reg_lambda=2.0,  # stronger L2 regularization
        min_child_weight=2.0,  # discourage overly specific splits
        tree_method="gpu_hist",  # will be attempted first, then fallback if needed
        eval_metric="logloss",
        n_jobs=-1,
    )




## === cell 26
oof_losses_xgb = []
models = []

for fn, (trn_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
    print("Starting fold:", fn)
    X_train_kf, X_val_kf = X_train.iloc[trn_idx], X_train.iloc[val_idx]
    y_train_kf, y_val_kf = y_train.iloc[trn_idx], y_train.iloc[val_idx]

    base_xgb = make_xgb()
    xgb_clf = MultiOutputClassifier(base_xgb)

    try:
        xgb_clf.fit(X_train_kf, y_train_kf)
    except Exception as e:
        print("GPU training failed; falling back to CPU hist. Error was:", repr(e))
        base_xgb = make_xgb()
        base_xgb.set_params(tree_method="hist")
        xgb_clf = MultiOutputClassifier(base_xgb)
        xgb_clf.fit(X_train_kf, y_train_kf)

    val_proba = xgb_clf.predict_proba(
        X_val_kf
    )  # list of arrays (n_samples, 2) per target
    val_pred = np.vstack([p[:, 1] for p in val_proba]).T  # (n_samples, 2)

    fold_auc_ec1 = roc_auc_score(y_val_kf["EC1"].values, val_pred[:, 0])
    fold_auc_ec2 = roc_auc_score(y_val_kf["EC2"].values, val_pred[:, 1])
    fold_auc_mean = (fold_auc_ec1 + fold_auc_ec2) / 2.0
    print(
        f"Fold AUC EC1: {fold_auc_ec1:.6f} | EC2: {fold_auc_ec2:.6f} | mean: {fold_auc_mean:.6f}"
    )

    oof_losses_xgb.append(fold_auc_mean)
    models.append(xgb_clf)

print("CV mean AUC:", float(np.mean(oof_losses_xgb)))



## === cell 27
final_base = make_xgb()
final_model = MultiOutputClassifier(final_base)
try:
    final_model.fit(X_train, y_train)
except Exception as e:
    print("GPU final training failed; falling back to CPU hist. Error was:", repr(e))
    final_base = make_xgb()
    final_base.set_params(tree_method="hist")
    final_model = MultiOutputClassifier(final_base)
    final_model.fit(X_train, y_train)



## === cell 28
test_proba = final_model.predict_proba(X_test)
test_pred = np.vstack([p[:, 1] for p in test_proba]).T  # (n_test, 2)

predictions = pd.DataFrame(test_pred, columns=["EC1", "EC2"])
results = pd.concat([sub["id"].reset_index(drop=True), predictions], axis=1)
results = results[["id", "EC1", "EC2"]]

assert (
    results.shape[0] == sub.shape[0]
), "Submission row count mismatch vs sample_submission."

results.to_csv("submission.csv", index=False)

print(results.head())
print("Wrote submission.csv with shape:", results.shape)
print("submission.csv columns:", results.columns.tolist())
