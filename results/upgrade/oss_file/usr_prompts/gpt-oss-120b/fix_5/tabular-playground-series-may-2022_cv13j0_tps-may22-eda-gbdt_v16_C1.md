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

# 5. Target score

0.98462

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
pass



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



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/131749796.py in <cell line: 0>()
----> 1 pd.options.display.float_format = "{:,.2f}".format
      2 pd.set_option("display.max_columns", NCOLS)
      3 pd.set_option("display.max_rows", NROWS)
      4 

NameError: name 'pd' is not defined

## === cell 4
trn_data = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/train.csv")
tst_data = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/test.csv")

sub = pd.read_csv(
    "/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv"
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3879675917.py in <cell line: 0>()
----> 1 trn_data = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/train.csv")
      2 tst_data = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/test.csv")
      3 
      4 sub = pd.read_csv(
      5     "/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv"

NameError: name 'pd' is not defined

## === cell 5
trn_data.shape



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2265708964.py in <cell line: 0>()
----> 1 trn_data.shape
      2 

NameError: name 'trn_data' is not defined

## === cell 6
trn_data.info()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4021697419.py in <cell line: 0>()
----> 1 trn_data.info()
      2 

NameError: name 'trn_data' is not defined

## === cell 7
trn_data.head()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4154599176.py in <cell line: 0>()
----> 1 trn_data.head()
      2 

NameError: name 'trn_data' is not defined

## === cell 8
trn_data.describe()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1990105761.py in <cell line: 0>()
----> 1 trn_data.describe()
      2 

NameError: name 'trn_data' is not defined

## === cell 9
trn_data.isnull().sum().sum()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1662542954.py in <cell line: 0>()
----> 1 trn_data.isnull().sum().sum()
      2 

NameError: name 'trn_data' is not defined

## === cell 10
trn_data.isnull().sum()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/293612565.py in <cell line: 0>()
----> 1 trn_data.isnull().sum()
      2 

NameError: name 'trn_data' is not defined

## === cell 11
trn_data.nunique()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2279520620.py in <cell line: 0>()
----> 1 trn_data.nunique()
      2 

NameError: name 'trn_data' is not defined

## === cell 12
trn_data.nunique().sort_values(ascending=True)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1000955195.py in <cell line: 0>()
----> 1 trn_data.nunique().sort_values(ascending=True)
      2 

NameError: name 'trn_data' is not defined

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



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/675259904.py in <cell line: 0>()
     16     "f_27",
     17 ]
---> 18 trn_data[categ_cols].sample(5)
     19 

NameError: name 'trn_data' is not defined

## === cell 14
correlation = trn_data.select_dtypes(include=[np.number]).corr()



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3466435340.py in <cell line: 0>()
----> 1 correlation = trn_data.select_dtypes(include=[np.number]).corr()
      2 

NameError: name 'trn_data' is not defined

## === cell 15
correlation



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4086236556.py in <cell line: 0>()
----> 1 correlation
      2 

NameError: name 'correlation' is not defined

## === cell 16
correlation["target"].sort_values(ascending=False)[:5]



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2990616026.py in <cell line: 0>()
----> 1 correlation["target"].sort_values(ascending=False)[:5]
      2 

NameError: name 'correlation' is not defined

## === cell 17
correlation["target"].sort_values(ascending=True)[:5]



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/791778691.py in <cell line: 0>()
----> 1 correlation["target"].sort_values(ascending=True)[:5]
      2 

NameError: name 'correlation' is not defined

## === cell 18
trn_data["target"].value_counts()



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3659823563.py in <cell line: 0>()
----> 1 trn_data["target"].value_counts()
      2 

NameError: name 'trn_data' is not defined

## === cell 19
trn_data["target"].describe()




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1295016250.py in <cell line: 0>()
----> 1 trn_data["target"].describe()
      2 
      3 

NameError: name 'trn_data' is not defined

## === cell 20
def count_sequence(df, field):
    """
    For each letter of the provided sequence it returns a new feature with the number of occurrences.
    """
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for letter in alphabet:
        df[letter + "_count"] = df[field].str.count(letter)
    df["unique_characters"] = df["f_27"].apply(lambda s: len(set(s)))
    return df




## === cell 21
def count_chars(df, field):
    """
    Creates numeric character position features and a unique‑character count.
    """
    for i in range(10):
        chars = df[field].str.get(i).fillna("A")
        df[f"ch_{i}"] = chars.apply(lambda c: ord(c) - ord("A"))
    df["unique_characters"] = df[field].apply(lambda s: len(set(s)))
    return df




## === cell 22
trn_data = count_chars(trn_data, "f_27")
tst_data = count_chars(tst_data, "f_27")



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1706108373.py in <cell line: 0>()
----> 1 trn_data = count_chars(trn_data, "f_27")
      2 tst_data = count_chars(tst_data, "f_27")
      3 

NameError: name 'trn_data' is not defined

## === cell 23
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
trn_data["f_27_enc"] = encoder.fit_transform(trn_data["f_27"])
tst_data["f_27_enc"] = encoder.transform(tst_data["f_27"])



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1556338743.py in <cell line: 0>()
      2 
      3 encoder = LabelEncoder()
----> 4 trn_data["f_27_enc"] = encoder.fit_transform(trn_data["f_27"])
      5 tst_data["f_27_enc"] = encoder.transform(tst_data["f_27"])
      6 

NameError: name 'trn_data' is not defined

## === cell 24
trn_data.head()



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4154599176.py in <cell line: 0>()
----> 1 trn_data.head()
      2 

NameError: name 'trn_data' is not defined

## === cell 25
ignore = ["id", "target", "f_27", "f_27_enc"]  # f_27 has been label encoded...
features = [feat for feat in trn_data.columns if feat not in ignore]
target_feature = "target"



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1273065059.py in <cell line: 0>()
      1 ignore = ["id", "target", "f_27", "f_27_enc"]  # f_27 has been label encoded...
----> 2 features = [feat for feat in trn_data.columns if feat not in ignore]
      3 target_feature = "target"
      4 

NameError: name 'trn_data' is not defined

## === cell 26
from sklearn.model_selection import train_test_split

test_size_pct = 0.20
X_train, X_valid, y_train, y_valid = train_test_split(
    trn_data[features],
    trn_data[target_feature],
    test_size=test_size_pct,
    random_state=42,
)



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3331450524.py in <cell line: 0>()
      3 test_size_pct = 0.20
      4 X_train, X_valid, y_train, y_valid = train_test_split(
----> 5     trn_data[features],
      6     trn_data[target_feature],
      7     test_size=test_size_pct,

NameError: name 'trn_data' is not defined

## === cell 27
xgb = None
lgb = None



## === cell 28
pass



## === cell 29
pass



## === cell 30
pass



## === cell 31
lgb_params = {
    "n_estimators": 8192,
    "min_child_samples": 96,
    "max_bins": 512,
    "random_state": 46,
    "n_jobs": 1,
}



## === cell 32
pass



## === cell 33
pass



## === cell 34
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, roc_curve
import math



## === cell 35
lgb_params = {
    "n_estimators": 8192,
    "min_child_samples": 96,
    "max_bins": 512,
    "random_state": 46,
    "n_jobs": 5,  # use all CPUs per model
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
    "tree_method": "hist",
    "n_jobs": 5,  # enable multi‑threaded training
}



## === cell 36
from joblib import Parallel, delayed
from xgboost import XGBClassifier

X_all = trn_data[features].values
y_all = trn_data[target_feature].values
X_test = tst_data[features].values


def train_fold(fold, trn_idx, val_idx):
    X_trn, X_val = X_all[trn_idx], X_all[val_idx]
    y_trn, y_val = y_all[trn_idx], y_all[val_idx]

    model = XGBClassifier(**xgb_params)
    model.fit(
        X_trn,
        y_trn,
        eval_set=[(X_val, y_val)],
        eval_metric=["auc"],
        early_stopping_rounds=256,
        verbose=0,
    )

    y_val_pred = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, y_val_pred)

    tst_pred = model.predict_proba(X_test)[:, 1]
    return (fold, score, tst_pred)


kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = Parallel(n_jobs=1)(
    delayed(train_fold)(fold, trn_idx, val_idx)
    for fold, (trn_idx, val_idx) in enumerate(kf.split(X_all))
)

score_list = []
predictions = []
for fold, score, tst_pred in sorted(results, key=lambda x: x[0]):
    print(f"Fold {fold}, AUC = {score:.3f}\n")
    score_list.append(score)
    predictions.append(tst_pred)

print(f"OOF AUC: {np.mean(score_list):.3f}")
print(".........")




## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3210111360.py in <cell line: 0>()
      2 from xgboost import XGBClassifier
      3 
----> 4 X_all = trn_data[features].values
      5 y_all = trn_data[target_feature].values
      6 X_test = tst_data[features].values

NameError: name 'trn_data' is not defined

## === cell 37
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




## === cell 38
if "model" in globals():
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_feature_importance(
        model.feature_importances_, X_train.columns, "XGB ", max_features=25
    )
else:
    print("Model not available for plotting – skipping feature importance.")



## === cell 39
sub.head()



## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3518946013.py in <cell line: 0>()
----> 1 sub.head()
      2 

NameError: name 'sub' is not defined

## === cell 40
sub["target"] = np.mean(predictions, axis=0)
sub.to_csv("my_submission_043022.csv", index=False)



## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1788384864.py in <cell line: 0>()
----> 1 sub["target"] = np.mean(predictions, axis=0)
      2 sub.to_csv("my_submission_043022.csv", index=False)
      3 

NameError: name 'np' is not defined

## === cell 41
sub.head()

## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3832920140.py in <cell line: 0>()
----> 1 sub.head()

NameError: name 'sub' is not defined
