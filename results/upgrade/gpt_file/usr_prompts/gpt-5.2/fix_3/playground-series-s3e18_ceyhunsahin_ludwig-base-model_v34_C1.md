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

0.64649

# 6. Current score

0.51286

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.51286) has done: 'I remove the Ludwig dependency (it fails to import due to a torchtext/torch binary mismatch in this Kaggle image) and replace it with a lightweight scikit-learn pipeline that still produces probabilistic predictions for EC1 and EC2. I also fix a key logic bug where `id` was dropped and later needed for submission alignment; the updated code keeps `id` in `sample_submission` and aligns predictions by row order. To keep the spirit of your original preprocessing, I retain the Yeo–Johnson power transform, but apply it correctly only to numeric feature columns (not to targets). Finally, the script always write a valid `submission.csv` with columns `id,EC1,EC2`.'
- What this solution (achieved 0.51286) has done: 'To move your score up toward the 0.64649 target (from 0.51286) while keeping the same core scikit-learn pipeline approach, I make two minimal, high-impact adjustments that typically improve AUC for linear models: (1) use `StratifiedKFold` out-of-fold (OOF) probabilities to calibrate a simple blending weight between the logistic-regression model and a “feature-standardized” variant (same model family, same loss/semantics), and then (2) refit both on full training and blend their test probabilities with the learned weight. This stays within your existing approach (logistic regression on transformed numeric features), doesn’t change the dataset or targets, and produces the same `submission.csv` format. The OOF blending is designed to improve ranking quality (AUC) without changing labels or doing any leakage.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import pandas as pd

df_train = pd.read_csv("/kaggle/input/playground-series-s3e18/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s3e18/test.csv")
sample_submission = pd.read_csv(
    "/kaggle/input/playground-series-s3e18/sample_submission.csv"
)



## === cell 2
df_train.head()



## === cell 3
my_palette = sns.cubehelix_palette(
    n_colors=7, start=0.46, rot=-0.45, dark=0.2, hue=0.95
)
sns.palplot(my_palette)
plt.gcf().set_size_inches(13, 2)

for idx, values in enumerate(my_palette.as_hex()):
    plt.text(
        idx - 0.375,
        0,
        my_palette.as_hex()[idx],
        {"font": "Courier New", "size": 16, "weight": "bold", "color": "black"},
        alpha=0.7,
    )
plt.gcf().set_facecolor("white")
plt.show()



## === cell 4
df_train.describe().T



## === cell 5
df_train.info()



## === cell 6
df_train[["MaxAbsEStateIndex", "MinEStateIndex"]].head()



## === cell 7
target_col1 = "EC1"
target_col2 = "EC2"

num_cols = [
    "BertzCT",
    "Chi1",
    "Chi1n",
    "Chi1v",
    "Chi2n",
    "Chi2v",
    "Chi3v",
    "Chi4n",
    "EState_VSA1",
    "EState_VSA2",
    "ExactMolWt",
    "FpDensityMorgan1",
    "FpDensityMorgan2",
    "FpDensityMorgan3",
    "HallKierAlpha",
    "HeavyAtomMolWt",
    "Kappa3",
    "MaxAbsEStateIndex",
    "MinEStateIndex",
    "NumHeteroatoms",
    "PEOE_VSA10",
    "PEOE_VSA14",
    "PEOE_VSA6",
    "PEOE_VSA7",
    "PEOE_VSA8",
    "SMR_VSA10",
    "SMR_VSA5",
    "SlogP_VSA3",
    "VSA_EState9",
    "fr_COO",
    "fr_COO2",
]
binary_cols = ["EC1", "EC2", "EC3", "EC4", "EC5", "EC6"]

cat_cols = df_test.select_dtypes(include=["object"]).columns.tolist()

print(f"[INFO] Shapes:" f"\n train: {df_train.shape}" f"\n test: {df_test.shape}\n")

print(
    f"[INFO] Any missing values:"
    f"\n train: {df_train.isna().any().any()}"
    f"\n test: {df_test.isna().any().any()}"
)



## === cell 8
plt.figure(figsize=(14, 8))
sns.set_style("white")

colors = my_palette
plt.barh(
    df_train[target_col1].value_counts().index,
    df_train[target_col1].value_counts(),
    color=colors[1:3],
)
plt.title("EC1 Distribution in df_train", fontsize=14, fontweight="bold")
sns.despine()
plt.show()



## === cell 9
plt.figure(figsize=(14, 4))
plt.barh(
    df_train[target_col2].value_counts().index,
    df_train[target_col2].value_counts(),
    color=colors[1:3],
)
plt.title("EC2 Distribution in df_train", fontsize=14, fontweight="bold")
sns.despine()
plt.show()



## === cell 10
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

feature_cols = [c for c in df_train.columns if c not in (binary_cols + ["id"])]

X_train = df_train[feature_cols].copy()
X_test = df_test[feature_cols].copy()
y_ec1 = df_train["EC1"].astype(int).values
y_ec2 = df_train["EC2"].astype(int).values

numeric_feature_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
other_cols = [c for c in feature_cols if c not in numeric_feature_cols]
if len(other_cols) > 0:
    raise ValueError(f"Found non-numeric feature columns not handled: {other_cols}")

base_lr = LogisticRegression(
    solver="lbfgs",
    max_iter=4000,
    n_jobs=None,
    class_weight="balanced",
    random_state=42,
)

pipe_v1 = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("power", PowerTransformer(method="yeo-johnson", standardize=True)),
        ("clf", base_lr),
    ]
)

pipe_v2 = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("power", PowerTransformer(method="yeo-johnson", standardize=True)),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", base_lr),
    ]
)




## === cell 11
def oof_preds_for_target(X, y, model, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(X.shape[0], dtype=float)
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr = y[tr_idx]
        m = Pipeline(steps=model.steps)
        m.fit(X_tr, y_tr)
        oof[va_idx] = m.predict_proba(X_va)[:, 1]
    return np.clip(oof, 1e-6, 1 - 1e-6)


Xtr_num = X_train[numeric_feature_cols]
Xte_num = X_test[numeric_feature_cols]

oof1_v1 = oof_preds_for_target(Xtr_num, y_ec1, pipe_v1, n_splits=5, seed=42)
oof1_v2 = oof_preds_for_target(Xtr_num, y_ec1, pipe_v2, n_splits=5, seed=42)

oof2_v1 = oof_preds_for_target(Xtr_num, y_ec2, pipe_v1, n_splits=5, seed=42)
oof2_v2 = oof_preds_for_target(Xtr_num, y_ec2, pipe_v2, n_splits=5, seed=42)


def best_linear_blend_weight(y, p1, p2):
    alphas = np.linspace(0.0, 1.0, 21)
    best_a, best_auc = 0.5, -np.inf
    for a in alphas:
        p = np.clip(a * p1 + (1 - a) * p2, 1e-6, 1 - 1e-6)
        auc = roc_auc_score(y, p)
        if auc > best_auc:
            best_auc, best_a = auc, a
    return best_a, best_auc


a1, auc1 = best_linear_blend_weight(y_ec1, oof1_v1, oof1_v2)
a2, auc2 = best_linear_blend_weight(y_ec2, oof2_v1, oof2_v2)

print(
    f"[INFO] OOF AUC EC1: v1={roc_auc_score(y_ec1, oof1_v1):.5f}, v2={roc_auc_score(y_ec1, oof1_v2):.5f}, blend(alpha={a1:.2f})={auc1:.5f}"
)
print(
    f"[INFO] OOF AUC EC2: v1={roc_auc_score(y_ec2, oof2_v1):.5f}, v2={roc_auc_score(y_ec2, oof2_v2):.5f}, blend(alpha={a2:.2f})={auc2:.5f}"
)

m1_v1 = Pipeline(steps=pipe_v1.steps)
m1_v2 = Pipeline(steps=pipe_v2.steps)
m2_v1 = Pipeline(steps=pipe_v1.steps)
m2_v2 = Pipeline(steps=pipe_v2.steps)

m1_v1.fit(Xtr_num, y_ec1)
m1_v2.fit(Xtr_num, y_ec1)
m2_v1.fit(Xtr_num, y_ec2)
m2_v2.fit(Xtr_num, y_ec2)

p1_v1 = m1_v1.predict_proba(Xte_num)[:, 1]
p1_v2 = m1_v2.predict_proba(Xte_num)[:, 1]
p2_v1 = m2_v1.predict_proba(Xte_num)[:, 1]
p2_v2 = m2_v2.predict_proba(Xte_num)[:, 1]

pred_ec1 = np.clip(a1 * p1_v1 + (1 - a1) * p1_v2, 1e-6, 1 - 1e-6)
pred_ec2 = np.clip(a2 * p2_v1 + (1 - a2) * p2_v2, 1e-6, 1 - 1e-6)

print("[INFO] Predictions shapes:", pred_ec1.shape, pred_ec2.shape)



## === cell 12
submission = sample_submission.copy()
submission["EC1"] = pred_ec1
submission["EC2"] = pred_ec2

submission = submission[["id", "EC1", "EC2"]]
assert (
    submission.shape[0] == df_test.shape[0]
), "Submission row count must match test row count"

submission.to_csv("submission.csv", index=False)
submission.head()



## === cell 13
print("[INFO] Saved submission.csv with shape:", submission.shape)
print(submission.describe(include="all"))
