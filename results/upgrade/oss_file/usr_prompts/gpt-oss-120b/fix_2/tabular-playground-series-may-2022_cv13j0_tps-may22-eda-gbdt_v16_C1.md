# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
import warnings

warnings.filterwarnings("ignore")




## === cell 2
DATA_ROWS = None
NROWS = 50
NCOLS = 15
BASE_PATH = "..."




## === cell 3
pd.options.display.float_format = "{:,.2f}".format
pd.set_option("display.max_columns", NCOLS)
pd.set_option("display.max_rows", NROWS)




## === cell 4
trn_data = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/train.csv")
tst_data = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/test.csv")

sub = pd.read_csv(
    "/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv"
)




## === cell 5
trn_data.shape




## === cell 6
trn_data.info()




## === cell 7
trn_data.head()




## === cell 8
trn_data.describe()




## === cell 9
trn_data.isnull().sum().sum()




## === cell 10
trn_data.isnull().sum()




## === cell 11
trn_data.nunique()




## === cell 12
trn_data.nunique().sort_values(ascending=True)




## === cell 13
categ_cols = [
    "f_29",
    "f_30",
    "f_13",
    "f_18",
    "f_17",
    "f_14",
    "f_11",
    "f_10",
    "f_09",
    "f_15",
    "f_07",
    "f_12",
    "f_16",
    "f_08",
    "f_27",
]
trn_data[categ_cols].sample(5)




## === cell 14
correlation = trn_data.select_dtypes(include=[np.number]).corr()




## === cell 15
correlation




## === cell 16
correlation["target"].sort_values(ascending=False)[:5]




## === cell 17
correlation["target"].sort_values(ascending=True)[:5]




## === cell 18
trn_data["target"].value_counts()




## === cell 19
trn_data["target"].describe()




## === cell 20
def count_sequence(df, field):
    """
    For each letter of the provided suquence it return new feature with the number of occurences.
    """
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for letter in alphabet:
        df[letter + "_count"] = df[field].str.count(letter)
    df["unique_characters"] = df["f_27"].apply(lambda s: len(set(s)))
    return df




## === cell 22
def count_chars(df, field):
    """
    Describes something...
    """
    for i in range(10):
        df[f"ch_{i}"] = df[field].str.get(i).apply(ord) - ord("A")
    df["unique_characters"] = df[field].apply(lambda s: len(set(s)))
    return df




## === cell 23
trn_data = count_chars(trn_data, "f_27")
tst_data = count_chars(tst_data, "f_27")




## === cell 24
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
trn_data["f_27_enc"] = encoder.fit_transform(trn_data["f_27"])
tst_data["f_27_enc"] = encoder.transform(tst_data["f_27"])




## === cell 25
trn_data.head()




## === cell 26
ignore = ["id", "target", "f_27", "f_27_enc"]  # f_27 has been label encoded...
features = [feat for feat in trn_data.columns if feat not in ignore]
target_feature = "target"




## === cell 27
from sklearn.model_selection import train_test_split

test_size_pct = 0.20
X_train, X_valid, y_train, y_valid = train_test_split(
    trn_data[features],
    trn_data[target_feature],
    test_size=test_size_pct,
    random_state=42,
)




## === cell 28
from xgboost import XGBClassifier

xgb_params = {
    "n_estimators": 8192,
    "min_child_weight": 96,
    "max_bin": 512,
    "random_state": 46,
    "objective": "binary:logistic",
    "tree_method": "hist",  # CPU mode
}




## === cell 29
xgb = XGBClassifier(**xgb_params)
xgb.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric=["auc"],
    early_stopping_rounds=256,
    verbose=250,
)




## === cell 30
from sklearn.metrics import roc_auc_score

val_preds = xgb.predict_proba(X_valid[features])[:, 1]
print("Validation AUC:", roc_auc_score(y_valid, val_preds))




## === cell 31
from lightgbm import LGBMClassifier




## === cell 32
lgb_params = {
    "n_estimators": 8192,
    "min_child_samples": 96,
    "max_bins": 512,
    "random_state": 46,
}




## === cell 33
lgb = LGBMClassifier(**lgb_params)
lgb.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric=["auc"],
    early_stopping_rounds=256,
    verbose=250,
)




## === cell 34
val_preds_lgb = lgb.predict_proba(X_valid[features])[:, 1]
print("LGBM Validation AUC:", roc_auc_score(y_valid, val_preds_lgb))




## === cell 35
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, roc_curve
import math




## === cell 36
lgb_params = {
    "n_estimators": 8192,
    "min_child_samples": 96,
    "max_bins": 512,
    "random_state": 46,
}

xgb_params = {
    "n_estimators": 8192,
    "min_child_weight": 96,
    "max_depth": 6,
    "learning_rate": 0.15,
    "subsample": 0.95,
    "colsample_bytree": 0.95,
    "reg_lambda": 1.50,
    "reg_alpha": 1.50,
    "gamma": 1.50,
    "max_bin": 512,
    "random_state": 46,
    "objective": "binary:logistic",
    "tree_method": "hist",  # CPU mode
}




## === cell 37
score_list = []
predictions = []
kf = KFold(n_splits=5)

for fold, (trn_idx, val_idx) in enumerate(kf.split(trn_data)):
    print(f"Training Fold {fold} ...")
    X_trn, X_val = trn_data.iloc[trn_idx][features], trn_data.iloc[val_idx][features]
    y_trn, y_val = (
        trn_data.iloc[trn_idx][target_feature],
        trn_data.iloc[val_idx][target_feature],
    )

    model = XGBClassifier(**xgb_params)
    model.fit(
        X_trn,
        y_trn,
        eval_set=[(X_val, y_val)],
        eval_metric=["auc"],
        early_stopping_rounds=256,
        verbose=0,
    )

    y_val_pred = model.predict_proba(X_val.values)[:, 1]
    score = roc_auc_score(y_val, y_val_pred)
    score_list.append(score)
    print(f"Fold {fold}, AUC = {score:.3f}\n")

    tst_pred = model.predict_proba(tst_data[features].values)[:, 1]
    predictions.append(tst_pred)

print(f"OOF AUC: {np.mean(score_list):.3f}")
print(".........")




## === cell 38
def plot_feature_importance(importance, names, model_type, max_features=10):
    import matplotlib.pyplot as plt
    import seaborn as sns

    feature_importance = np.array(importance)
    feature_names = np.array(names)

    data = {"feature_names": feature_names, "feature_importance": feature_importance}
    fi_df = pd.DataFrame(data)

    fi_df.sort_values(by=["feature_importance"], ascending=False, inplace=True)
    fi_df = fi_df.head(max_features)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=fi_df["feature_importance"], y=fi_df["feature_names"])
    plt.title(model_type + " FEATURE IMPORTANCE")
    plt.xlabel("FEATURE IMPORTANCE")
    plt.ylabel("FEATURE NAMES")
    plt.show()




## === cell 39
if "model" in globals():
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_feature_importance(
        model.feature_importances_, X_train.columns, "XGB ", max_features=25
    )
else:
    print("Model not available for plotting – skipping feature importance.")




## === cell 40
sub.head()




## === cell 41
sub["target"] = np.mean(predictions, axis=0)
sub.to_csv("my_submission_043022.csv", index=False)




## === cell 42
sub.head()
