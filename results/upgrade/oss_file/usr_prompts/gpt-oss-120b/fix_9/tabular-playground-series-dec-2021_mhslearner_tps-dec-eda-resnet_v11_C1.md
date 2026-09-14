# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Predict the class of a given image from a synthetic dataset.

## MetricMulti-class classification accuracy.

## Submission FormatFor each `Id` in the test set, you must predict the `Cover_Type` class. The file should contain a header and have the following format:
```
Id,Cover_Type
4000000,2
4000001,1
4000001,3
etc.
```

## Dataset 
- train.csv - the training data with the target `Cover_Type` column
- test.csv - the test set; you will be predicting the `Cover_Type` for each row in this file (the target integer class)
- sample_submission.csv - a sample submission file in the correct format

# 2. Python version

3.10

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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

# 4. Data file paths

```
/
    kaggle/
        data/
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 2 other files
        input/
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 2 other files
        working/
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 2 other files
```

-> data/tabular-playground-series-dec-2021/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> data/tabular-playground-series-dec-2021/test.csv has 400000 rows and 55 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 40 more columns

-> data/tabular-playground-series-dec-2021/train.csv has 3600000 rows and 56 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 41 more columns

-> input/tabular-playground-series-dec-2021/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> input/tabular-playground-series-dec-2021/test.csv has 400000 rows and 55 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 40 more columns

-> input/tabular-playground-series-dec-2021/train.csv has 3600000 rows and 56 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 41 more columns

-> working/tabular-playground-series-dec-2021/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import numpy as np, pandas as pd, os, gc, warnings
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

try:
    from sklearnex import patch

    patch()
except Exception:
    pass

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier



## === cell 1
train_data = pd.read_csv("../input/tabular-playground-series-dec-2021/train.csv")
test_data = pd.read_csv("../input/tabular-playground-series-dec-2021/test.csv")
sample = pd.read_csv(
    "../input/tabular-playground-series-dec-2021/sample_submission.csv"
)
train = train_data.drop("Id", axis=1)
test = test_data.drop("Id", axis=1)




## === cell 2
def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type).startswith("int"):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if (c_min > np.finfo(np.float32).min) and (
                    c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


X_train = reduce_mem_usage(train.drop("Cover_Type", axis=1))
test = reduce_mem_usage(test)
y_target = train["Cover_Type"].copy()



## === cell 3
num_cols = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]



## === cell 4
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
test[num_cols] = scaler.transform(test[num_cols])



## === cell 5
label_enc = LabelEncoder()
y_encoded = label_enc.fit_transform(y_target)
n_classes = len(np.unique(y_encoded))
print(y_encoded.shape, X_train.shape, test.shape, "n_classes:", n_classes)



## === cell 6
X_np = X_train.values.astype(np.float32)
test_np = test.values.astype(np.float32)

N_F = 3  # number of folds
val_score = []
test_pred = np.zeros((test_np.shape[0], n_classes), dtype=np.float32)

SKF = StratifiedKFold(n_splits=N_F, shuffle=True, random_state=42)


def _train_fold(fold, idx_train, idx_valid):
    """Train a single RandomForest on the provided split and return validation score + test probabilities."""
    X_tr, y_tr = X_np[idx_train], y_encoded[idx_train]
    X_val, y_val = X_np[idx_valid], y_encoded[idx_valid]

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=1,  # keep trees single‑threaded; folds run in parallel threads
        random_state=fold,
        verbose=0,
    )
    clf.fit(X_tr, y_tr)

    val_pred = clf.predict(X_val)
    score = accuracy_score(y_val, val_pred)
    test_proba = clf.predict_proba(test_np)

    del X_tr, y_tr, X_val, y_val, clf
    gc.collect()

    return fold, score, test_proba


results = Parallel(n_jobs=min(N_F, os.cpu_count()), backend="threading")(
    delayed(_train_fold)(fold, idx_tr, idx_va)
    for fold, (idx_tr, idx_va) in enumerate(SKF.split(X_np, y_encoded))
)

for fold_id, score, test_proba in sorted(results, key=lambda x: x[0]):
    val_score.append(score)
    test_pred += test_proba
    print(f"FOLD {fold_id}: validation accuracy = {score:.6f}")

print("**************************************************")
print(f"Mean Validation Accuracy : {np.mean(val_score):.6f}")

test_pred /= N_F



## === cell 7
predictions = np.argmax(test_pred, axis=1)
predictions = label_enc.inverse_transform(predictions)



## === cell 8
sample["Cover_Type"] = predictions
sample.to_csv("resnet.csv", index=False)
print("Submission saved to resnet.csv")
sample.head()
