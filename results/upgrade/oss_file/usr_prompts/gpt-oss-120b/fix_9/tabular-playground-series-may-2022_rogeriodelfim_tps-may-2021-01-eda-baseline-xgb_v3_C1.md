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
google-ai-generativelanguage==0.6.15
google-api-core==2.28.1
google-auth==2.38.0
google-auth-httplib2==0.2.0
google-auth-oauthlib==1.2.2
google-generativeai==0.8.5
googleapis-common-protos==1.70.0
joblib==1.5.2
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
pydata-google-auth==1.9.1
python-dateutil==2.9.0.post0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
shap==0.44.1
shapely==2.1.2
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
xgboost==2.0.3

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

0.93476

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import glob
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt


def find_file(fname: str) -> str:
    """
    Search common Kaggle data locations and recursively for a given filename.
    Returns the first existing path (including possible .zip files).
    """
    candidates = [
        os.path.join("kaggle", "data", fname),
        os.path.join("input", fname),
        os.path.join("data", fname),
        fname,
    ]
    subfolders = [
        os.path.join("kaggle", "data", "tabular-playground-series-may-2022", fname),
        os.path.join("input", "tabular-playground-series-may-2022", fname),
        os.path.join("data", "tabular-playground-series-may-2022", fname),
    ]
    candidates.extend(subfolders)

    for p in candidates:
        if os.path.isfile(p):
            return p

    for pattern in [fname, f"{fname}.zip"]:
        matches = glob.glob(f"**/{pattern}", recursive=True)
        if matches:
            return matches[0]

    raise FileNotFoundError(f"{fname} not found in searched paths.")


def read_csv_auto(fname: str) -> pd.DataFrame:
    """
    Load a CSV file, automatically handling a possible .zip wrapper.
    """
    path = find_file(fname)
    try:
        return pd.read_csv(path)
    except Exception:
        if path.lower().endswith(".zip"):
            return pd.read_csv(path, compression="zip")
        else:
            raise


train_path = "train.csv"
test_path = "test.csv"
sub_path = "sample_submission.csv"

df_train = read_csv_auto(train_path)
df_test = read_csv_auto(test_path)
df_sub = read_csv_auto(sub_path)

target = "target"
id_col = "id"




## === cell 1
def encode_df(df: pd.DataFrame, encoders=None):
    """
    Encode object columns using sklearn's LabelEncoder.
    When `encoders` is provided (e.g., for the test set), unseen categories are
    added to the existing encoder so that transformation never fails.
    """
    obj_cols = df.select_dtypes(include="object").columns
    if encoders is None:
        encoders = {}
        for col in obj_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        for col in obj_cols:
            le = encoders[col]
            try:
                df[col] = le.transform(df[col].astype(str))
            except ValueError:
                existing = list(le.classes_)
                new_vals = df[col].astype(str).unique().tolist()
                combined = pd.Index(existing + new_vals).unique()
                le_extended = LabelEncoder()
                le_extended.fit(combined)
                df[col] = le_extended.transform(df[col].astype(str))
                encoders[col] = le_extended
    return df, encoders


df_train, encoders = encode_df(df_train.copy())
df_test, _ = encode_df(df_test.copy(), encoders)

X = df_train.drop([target, id_col], axis=1)
y = df_train[target]
X_test = df_test.drop([id_col], axis=1)

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=123
)




## === cell 2
params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 1000,
    "random_state": 123,
    "n_jobs": -1,
}
model = xgb.XGBClassifier(**params)

model.fit(
    X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False
)

val_pred = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, val_pred)
print(f"Validation AUC: {auc:.5f}")




## === cell 3
submission_dir = os.path.join("submission")
os.makedirs(submission_dir, exist_ok=True)

test_pred = model.predict_proba(X_test)[:, 1]

df_sub[target] = test_pred
df_sub = df_sub[[id_col, target]]  # enforce column order

outfile = os.path.join(submission_dir, "submission.csv")
df_sub.to_csv(outfile, index=False)
print(f"Submission saved to {outfile}")




## === cell 4
from sklearn import metrics

fpr, tpr, _ = metrics.roc_curve(y_val, val_pred)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.5f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Validation ROC Curve")
plt.legend()
plt.show()
