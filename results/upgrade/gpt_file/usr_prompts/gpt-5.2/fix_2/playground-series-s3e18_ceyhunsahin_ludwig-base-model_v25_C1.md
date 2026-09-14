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

0.64405

# 6. Current score

0.5127

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5127) has done: 'I remove the Ludwig dependency (it fails to import in this Kaggle environment due to a torchtext binary mismatch) and replace it with a minimal scikit-learn model that still produces probabilistic predictions for EC1/EC2 and writes a valid `submission.csv`. To keep the overall approach similar (tabular numeric features + engineered ratios/products), the patch uses the same feature set you built and trains one classifier per target. I also fix a major preprocessing bug: the earlier `power_transform` was applied to target columns too, corrupting labels; the fix applies transformations only to feature columns. Finally, I ensure the submission aligns to the `sample_submission` ids and has the exact required columns.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:50]:
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
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score



## === cell 11
from sklearn.preprocessing import PowerTransformer

df_tr_copy = df_train.copy()
df_tst_copy = df_test.copy()

feature_cols_base = [c for c in df_train.columns if c not in (["id"] + binary_cols)]
numeric_feature_cols = (
    df_train[feature_cols_base].select_dtypes(include=[np.number]).columns.tolist()
)

pt = PowerTransformer(method="yeo-johnson", standardize=True)
df_tr_copy[numeric_feature_cols] = pt.fit_transform(df_tr_copy[numeric_feature_cols])
df_tst_copy[numeric_feature_cols] = pt.transform(df_tst_copy[numeric_feature_cols])




## === cell 12
def out_iqr(df):
    columns = df.columns
    iqrs, lower_out, upper_out = [], [], []
    for column in columns:
        if not np.issubdtype(df[column].dtype, np.number):
            iqrs.append(np.nan)
            upper_out.append(0)
            lower_out.append(0)
            continue
        q25, q75 = np.quantile(df[column], 0.25), np.quantile(df[column], 0.75)
        iqr = q75 - q25
        cut_off = iqr * 1.5
        lower, upper = q25 - cut_off, q75 + cut_off
        df1 = df[df[column] > upper].shape[0]
        df2 = df[df[column] < lower].shape[0]
        iqrs.append(iqr)
        upper_out.append(df1)
        lower_out.append(df2)
    return pd.DataFrame(
        data={
            "columns": df.columns,
            "iqr": iqrs,
            "lower_outliers": lower_out,
            "upper_outliers": upper_out,
        }
    ).style.background_gradient(axis=0, cmap="Greens")




## === cell 13
out_iqr(df_tr_copy[numeric_feature_cols])



## === cell 14
numerical_columns = [c for c in num_cols if c in df_tr_copy.columns]

fig, axes = plt.subplots(
    len(numerical_columns), 2, figsize=(20, max(8, 2 * len(numerical_columns)))
)
if len(numerical_columns) == 1:
    axes = np.array([axes])  # ensure 2D indexing

for i, column in enumerate(numerical_columns):
    sns.histplot(
        df_tr_copy[column], bins=30, kde=True, ax=axes[i, 0], color=my_palette[2]
    )
    axes[i, 0].set_title(f"Distribution of {column} in df_train")
    axes[i, 0].set_xlabel("Value")
    axes[i, 0].set_ylabel("Frequency")

    sns.boxplot(x=df_train[column], ax=axes[i, 1], color=my_palette[1])
    axes[i, 1].set_title(f"Box plot of {column} in df_train")
    axes[i, 1].set_xlabel(column)
    axes[i, 1].set_ylabel("Value")

plt.tight_layout()
plt.show()




## === cell 15
def create_features(df):
    new_features = {
        "BertzCT_MaxAbsEStateIndex_Ratio": df["BertzCT"]
        / (df["MaxAbsEStateIndex"] + 1e-12),
        "BertzCT_ExactMolWt_Product": df["BertzCT"] * df["ExactMolWt"],
        "NumHeteroatoms_FpDensityMorgan1_Ratio": df["NumHeteroatoms"]
        / (df["FpDensityMorgan1"] + 1e-12),
        "VSA_EState9_EState_VSA1_Ratio": df["VSA_EState9"]
        / (df["EState_VSA1"] + 1e-12),
        "PEOE_VSA10_SMR_VSA5_Ratio": df["PEOE_VSA10"] / (df["SMR_VSA5"] + 1e-12),
        "Chi1v_ExactMolWt_Product": df["Chi1v"] * df["ExactMolWt"],
        "Chi2v_ExactMolWt_Product": df["Chi2v"] * df["ExactMolWt"],
        "Chi3v_ExactMolWt_Product": df["Chi3v"] * df["ExactMolWt"],
        "EState_VSA1_NumHeteroatoms_Product": df["EState_VSA1"] * df["NumHeteroatoms"],
        "PEOE_VSA10_Chi1_Ratio": df["PEOE_VSA10"] / (df["Chi1"] + 1e-12),
        "MaxAbsEStateIndex_NumHeteroatoms_Ratio": df["MaxAbsEStateIndex"]
        / (df["NumHeteroatoms"] + 1e-12),
        "BertzCT_Chi1_Ratio": df["BertzCT"] / (df["Chi1"] + 1e-12),
    }
    df = df.assign(**new_features)
    return df


df_tr_copy = create_features(df_tr_copy)
df_tst_copy = create_features(df_tst_copy)



## === cell 16
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

feature_cols = [c for c in df_tr_copy.columns if c not in (["id"] + binary_cols)]
X = df_tr_copy[feature_cols]
X_test = df_tst_copy[feature_cols]

y1 = df_train[target_col1].astype(int).values
y2 = df_train[target_col2].astype(int).values

base_clf = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        (
            "clf",
            LogisticRegression(
                solver="lbfgs",
                max_iter=2000,
                class_weight="balanced",
                n_jobs=None,
                random_state=42,
            ),
        ),
    ]
)



## === cell 17
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof1 = np.zeros(len(X), dtype=float)
for tr_idx, va_idx in skf.split(X, y1):
    m = base_clf
    m.fit(X.iloc[tr_idx], y1[tr_idx])
    oof1[va_idx] = m.predict_proba(X.iloc[va_idx])[:, 1]

auc1 = roc_auc_score(y1, oof1)
print(f"[INFO] EC1 CV AUC (5-fold): {auc1:.5f}")

model_ec1 = base_clf.fit(X, y1)



## === cell 18
oof2 = np.zeros(len(X), dtype=float)
for tr_idx, va_idx in skf.split(X, y2):
    m = base_clf
    m.fit(X.iloc[tr_idx], y2[tr_idx])
    oof2[va_idx] = m.predict_proba(X.iloc[va_idx])[:, 1]

auc2 = roc_auc_score(y2, oof2)
print(f"[INFO] EC2 CV AUC (5-fold): {auc2:.5f}")

model_ec2 = base_clf.fit(X, y2)



## === cell 19
pred_ec1 = model_ec1.predict_proba(X_test)[:, 1]
pred_ec2 = model_ec2.predict_proba(X_test)[:, 1]

print("[INFO] Predictions shapes:", pred_ec1.shape, pred_ec2.shape)



## === cell 20
sub = sample_submission[["id"]].copy()

test_ids = df_test[["id"]].copy()
pred_df = pd.DataFrame({"id": test_ids["id"].values, "EC1": pred_ec1, "EC2": pred_ec2})

sub = sub.merge(pred_df, on="id", how="left")

if sub[["EC1", "EC2"]].isna().any().any():
    print(
        "[WARN] NaNs detected after id-merge; falling back to direct assignment by row order."
    )
    sub["EC1"] = pred_ec1
    sub["EC2"] = pred_ec2

sub.to_csv("submission.csv", index=False)
sub.head()



## === cell 21
print("[INFO] submission.csv written with shape:", sub.shape)
print(sub.describe(include="all"))
