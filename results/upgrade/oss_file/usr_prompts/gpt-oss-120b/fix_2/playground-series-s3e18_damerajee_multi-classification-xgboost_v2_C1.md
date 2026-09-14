# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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
plotly==5.24.1
plotly-express==0.4.1
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

0.6375

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train_df = pd.read_csv("/kaggle/input/playground-series-s3e18/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s3e18/test.csv")



## === cell 2
train_df



## === cell 3
train_df.info()



## === cell 4
train_df.describe()



## === cell 5
missing_values = train_df.isnull().sum()
duplicated_value = train_df.duplicated().sum()
number_of_uniques = train_df.nunique().sum()

print("missing values:")
print(missing_values)
print("*****************************************")
print()
print("duplicated value:")
print(duplicated_value)
print("*****************************************")
print()
print("number of uniques:")
print(number_of_uniques)
print("*****************************************")
print()



## === cell 6
test_df



## === cell 7
test_df.info()



## === cell 8
test_df.describe()



## === cell 9
missing_values = train_df.isnull().sum()
duplicated_value = train_df.duplicated().sum()
number_of_uniques = train_df.nunique().sum()

print("missing values:")
print(missing_values)
print("*****************************************")
print()
print("duplicated value:")
print(duplicated_value)
print("*****************************************")
print()
print("number of uniques:")
print(number_of_uniques)
print("*****************************************")
print()



## === cell 10
for i in train_df.columns:
    values = train_df[i].value_counts()

    print("columns : ", i)
    print("values : ", values)
    print("*********************************************************")
    print()




## === cell 11
def histplot(df, title):
    for i, column in enumerate(df.columns):
        fig, axes = plt.subplots(3, 1, figsize=(11, 12))

        sns.histplot(df[column], ax=axes[0])
        axes[0].set_title("Histogram - Column: " + column)

        sns.kdeplot(df[column], ax=axes[1])
        axes[1].set_title("KDE Plot - Column: " + column)

        sns.boxplot(df[column], ax=axes[2])
        axes[2].set_title("Box Plot - Column: " + column)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()


histplot(train_df, title="Plots - Train data")
histplot(test_df, title="Plots - Test data")




## === cell 12
def heatmap(df, title):
    plt.figure(figsize=(15, 10))
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))
    sns.heatmap(
        df.corr(),
        mask=mask,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.show()


heatmap(train_df, title="Heatmap - Train data")
heatmap(test_df, title="Heatmap - Test data")



## === cell 13
positive = []
negative = []


def check_positive_negative(df, title):
    for i in df.columns:
        if df[i].corr(df["EC2"]) > 0 or df[i].corr(df["EC1"]) > 0:
            positive.append(i)
        elif df[i].corr(df["EC2"]) < 0 or df[i].corr(df["EC1"]) < 0:
            negative.append(i)

    print("Positive Correlated Columns:")
    print(positive)
    print()
    print("\nNegative Correlated Columns:")
    print(negative)


check_positive_negative(train_df, title="Correlation Check")



## === cell 14
import plotly.express as px

for i in train_df.columns:
    fig = px.density_heatmap(train_df, x="EC1", y=train_df[i])
    fig.show()



## === cell 15
for i in train_df.columns:
    fig = px.density_heatmap(train_df, x="EC2", y=train_df[i])
    fig.show()



## === cell 16
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

target_cols = ["EC1", "EC2"]
exclude_cols = ["id"] + target_cols
feature_cols = [c for c in train_df.columns if c not in exclude_cols]

X = train_df[feature_cols]
y1 = train_df["EC1"]
y2 = train_df["EC2"]

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

preprocess = ColumnTransformer(
    transformers=[("num", numeric_transformer, feature_cols)]
)

model_ec1 = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("clf", LogisticRegression(max_iter=1000, n_jobs=5, solver="lbfgs")),
    ]
)

model_ec2 = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("clf", LogisticRegression(max_iter=1000, n_jobs=5, solver="lbfgs")),
    ]
)

X_train, X_val, y1_train, y1_val, y2_train, y2_val = train_test_split(
    X, y1, y2, test_size=0.2, random_state=42, stratify=y1
)

model_ec1.fit(X_train, y1_train)
model_ec2.fit(X_train, y2_train)

val_pred_ec1 = model_ec1.predict_proba(X_val)[:, 1]
val_pred_ec2 = model_ec2.predict_proba(X_val)[:, 1]

auc_ec1 = roc_auc_score(y1_val, val_pred_ec1)
auc_ec2 = roc_auc_score(y2_val, val_pred_ec2)
print(
    f"Validation AUC – EC1: {auc_ec1:.4f}, EC2: {auc_ec2:.4f}, Mean: {(auc_ec1+auc_ec2)/2:.4f}"
)

model_ec1.fit(X, y1)
model_ec2.fit(X, y2)

test_X = test_df[feature_cols]
test_pred_ec1 = model_ec1.predict_proba(test_X)[:, 1]
test_pred_ec2 = model_ec2.predict_proba(test_X)[:, 1]

submission = pd.DataFrame(
    {"id": test_df["id"], "EC1": test_pred_ec1, "EC2": test_pred_ec2}
)

submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2662915786.py in <cell line: 0>()
     71 
     72 # Predict probabilities for the test set.
---> 73 test_X = test_df[feature_cols]
     74 test_pred_ec1 = model_ec1.predict_proba(test_X)[:, 1]
     75 test_pred_ec2 = model_ec2.predict_proba(test_X)[:, 1]

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['EC3', 'EC4', 'EC5', 'EC6'] not in index"
