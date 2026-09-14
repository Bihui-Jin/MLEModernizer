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
            description.md (59 lines)
            sample_submission.csv (400001 lines)
            sample_submission.csv.zip (1.6 MB)
            test.csv (400001 lines)
            test.csv.zip (10.7 MB)
            train.csv (3600001 lines)
            train.csv.zip (97.9 MB)
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
        input/
            description.md (59 lines)
            sample_submission.csv (400001 lines)
            sample_submission.csv.zip (1.6 MB)
            test.csv (400001 lines)
            test.csv.zip (10.7 MB)
            train.csv (3600001 lines)
            train.csv.zip (97.9 MB)
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
        working/
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
```

-> data/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> data/tabular-playground-series-dec-2021/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> data/tabular-playground-series-dec-2021/test.csv has 400000 rows and 55 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 40 more columns

-> data/tabular-playground-series-dec-2021/train.csv has 3600000 rows and 56 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 41 more columns

-> data/test.csv has 400000 rows and 55 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 40 more columns

-> data/train.csv has 3600000 rows and 56 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 41 more columns

-> input/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

DATA_DIR_CANDIDATES = [
    "/kaggle/input/tabular-playground-series-dec-2021",
    "/kaggle/input",
    "/kaggle/data/tabular-playground-series-dec-2021",
    "/kaggle/data",
]


def _find_file(filename: str) -> str:
    for d in DATA_DIR_CANDIDATES:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Could not find {filename} in any of: {DATA_DIR_CANDIDATES}"
    )


train_path = _find_file("train.csv")
test_path = _find_file("test.csv")
sample_sub_path = _find_file("sample_submission.csv")

train_head = pd.read_csv(train_path, nrows=50)
test_head = pd.read_csv(test_path, nrows=50)

dtype_train = {}
for c in train_head.columns:
    if c == "Cover_Type":
        dtype_train[c] = np.int16
    elif c == "Id":
        dtype_train[c] = np.int64
    else:
        if pd.api.types.is_integer_dtype(train_head[c]):
            mn, mx = int(train_head[c].min()), int(train_head[c].max())
            if 0 <= mn and mx <= 1:
                dtype_train[c] = np.int8
            elif -32768 <= mn and mx <= 32767:
                dtype_train[c] = np.int16
            else:
                dtype_train[c] = np.int32
        else:
            dtype_train[c] = np.float32

dtype_test = {}
for c in test_head.columns:
    if c == "Id":
        dtype_test[c] = np.int64
    else:
        if pd.api.types.is_integer_dtype(test_head[c]):
            mn, mx = int(test_head[c].min()), int(test_head[c].max())
            if 0 <= mn and mx <= 1:
                dtype_test[c] = np.int8
            elif -32768 <= mn and mx <= 32767:
                dtype_test[c] = np.int16
            else:
                dtype_test[c] = np.int32
        else:
            dtype_test[c] = np.float32

read_kwargs = dict(low_memory=False)
try:
    train_df = pd.read_csv(
        train_path, dtype=dtype_train, engine="pyarrow", **read_kwargs
    )
    test_df = pd.read_csv(test_path, dtype=dtype_test, engine="pyarrow", **read_kwargs)
    sample_sub = pd.read_csv(sample_sub_path, engine="pyarrow")
except Exception:
    train_df = pd.read_csv(train_path, dtype=dtype_train, **read_kwargs)
    test_df = pd.read_csv(test_path, dtype=dtype_test, **read_kwargs)
    sample_sub = pd.read_csv(sample_sub_path)

print(train_df.shape, test_df.shape, sample_sub.shape)
print("train columns:", train_df.columns[:10].tolist(), "...")
print("test columns:", test_df.columns[:10].tolist(), "...")




## === cell 1
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

TARGET_COL = "Cover_Type"
ID_COL = "Id"

X_all = train_df.drop(columns=[TARGET_COL])
y_all = train_df[TARGET_COL].astype(int)

common_cols = [c for c in X_all.columns if c in test_df.columns]
X_all = X_all[common_cols]
X_test = test_df[common_cols]

cat_cols = [c for c in X_all.columns if X_all[c].dtype == "object"]
num_cols = [c for c in X_all.columns if c not in cat_cols]

transformers = [
    (
        "num",
        Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        ),
        num_cols,
    )
]
if len(cat_cols) > 0:
    transformers.append(
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                ]
            ),
            cat_cols,
        )
    )

preprocess = ColumnTransformer(
    transformers=transformers,
    remainder="drop",
    verbose_feature_names_out=False,
)

clf = LogisticRegression(
    multi_class="multinomial",
    solver="saga",
    C=3.0,
    max_iter=200,
    n_jobs=-1,
    random_state=42,
    warm_start=True,
    verbose=0,
)

y_counts = y_all.value_counts(dropna=False)
rare_classes = y_counts[y_counts < 2]
if len(rare_classes) > 0:
    print(
        "Warning: found classes with <2 samples; disabling stratify for split. Rare classes:",
        rare_classes.to_dict(),
    )
    stratify_arg = None
else:
    stratify_arg = y_all

X_tr, X_va, y_tr, y_va = train_test_split(
    X_all, y_all, test_size=0.02, random_state=42, stratify=stratify_arg
)

model = Pipeline(steps=[("prep", preprocess), ("clf", clf)])




## === cell 2
from scipy import sparse

X_all_t = preprocess.fit_transform(X_all)
X_va_t = preprocess.transform(X_va)
X_test_t = preprocess.transform(X_test)

if sparse.issparse(X_all_t) and not sparse.isspmatrix_csr(X_all_t):
    X_all_t = X_all_t.tocsr()
if sparse.issparse(X_va_t) and not sparse.isspmatrix_csr(X_va_t):
    X_va_t = X_va_t.tocsr()
if sparse.issparse(X_test_t) and not sparse.isspmatrix_csr(X_test_t):
    X_test_t = X_test_t.tocsr()

clf.fit(X_all_t, y_all)

va_pred = clf.predict(X_va_t)
print("Holdout accuracy:", accuracy_score(y_va, va_pred))

test_pred = clf.predict(X_test_t).astype(int)

sub = pd.DataFrame({ID_COL: test_df[ID_COL].values, TARGET_COL: test_pred})

if sample_sub.shape[0] == sub.shape[0] and ID_COL in sample_sub.columns:
    if not np.array_equal(sample_sub[ID_COL].values, sub[ID_COL].values):
        sub = sub.set_index(ID_COL).reindex(sample_sub[ID_COL].values).reset_index()

sub[ID_COL] = sub[ID_COL].astype(np.int64)
sub[TARGET_COL] = sub[TARGET_COL].astype(np.int64)

assert list(sub.columns) == [ID_COL, TARGET_COL]
assert sub.shape[0] == sample_sub.shape[0]
assert sub[TARGET_COL].between(1, 7).all(), "Cover_Type predictions should be in [1,7]"

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(sub.head())
