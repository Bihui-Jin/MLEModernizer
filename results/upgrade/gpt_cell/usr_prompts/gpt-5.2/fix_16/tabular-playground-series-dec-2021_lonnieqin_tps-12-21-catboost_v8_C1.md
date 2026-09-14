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

No external packages required in the script and installed.

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

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split




## === cell 1
class Config:
    is_kaggle_platform = os.path.exists("/kaggle/input")
    dataset_name = "tabular-playground-series-dec-2021"
    data_path = "/kaggle/input/%s/" % (dataset_name) if is_kaggle_platform else ""
    submit_filename = "submission.csv"
    label_name = "Cover_Type"
    id_field = "Id"


config = Config()



## === cell 2
if not config.is_kaggle_platform:
    raise RuntimeError(
        "This optimized script is intended for Kaggle where /kaggle/input exists. "
        "Non-Kaggle download cells were removed to avoid timeouts and notebook-magics in scripts."
    )



## === cell 3
train_path = config.data_path + "train.csv"
test_path = config.data_path + "test.csv"

train_cols = pd.read_csv(train_path, nrows=0).columns.tolist()
feature_cols = [c for c in train_cols if c not in (config.id_field, config.label_name)]

bin_cols = [
    c
    for c in feature_cols
    if c.startswith("Wilderness_Area") or c.startswith("Soil_Type")
]
cont_cols = [c for c in feature_cols if c not in set(bin_cols)]

feature_names = cont_cols + bin_cols

usecols_train = [config.id_field] + feature_names + [config.label_name]
usecols_test = [config.id_field] + feature_names



## === cell 4
dtype_map_train = {config.id_field: np.int32, config.label_name: np.uint8}
dtype_map_test = {config.id_field: np.int32}
for c in bin_cols:
    dtype_map_train[c] = np.uint8
    dtype_map_test[c] = np.uint8
for c in cont_cols:
    dtype_map_train[c] = np.float32
    dtype_map_test[c] = np.float32




## === cell 5
def _read_csv_fast(path, usecols, dtype_map):
    try:
        import pyarrow as pa  # noqa: F401
        import pyarrow.csv as pacsv

        conv = {}
        for k, v in dtype_map.items():
            if v == np.int32:
                conv[k] = pacsv.ConvertOptions(
                    column_types={k: pa.int32()}
                ).column_types[k]
            elif v == np.uint8:
                conv[k] = pacsv.ConvertOptions(
                    column_types={k: pa.uint8()}
                ).column_types[k]
            elif v == np.float32:
                conv[k] = pacsv.ConvertOptions(
                    column_types={k: pa.float32()}
                ).column_types[k]

        column_types = {}
        for col in usecols:
            v = dtype_map.get(col, None)
            if v == np.int32:
                column_types[col] = pa.int32()
            elif v == np.uint8:
                column_types[col] = pa.uint8()
            elif v == np.float32:
                column_types[col] = pa.float32()

        read_opts = pacsv.ReadOptions(use_threads=True, block_size=1 << 20)
        parse_opts = pacsv.ParseOptions(delimiter=",")
        conv_opts = pacsv.ConvertOptions(
            include_columns=usecols,
            column_types=column_types,
            strings_can_be_null=False,
        )
        table = pacsv.read_csv(
            path,
            read_options=read_opts,
            parse_options=parse_opts,
            convert_options=conv_opts,
        )
        return table.to_pandas(types_mapper=None, self_destruct=True)
    except Exception:
        return pd.read_csv(
            path,
            usecols=usecols,
            dtype=dtype_map,
            low_memory=False,
            memory_map=True,
            engine="c",
        )


def load_train_test_numpy_fast(
    train_csv: str,
    test_csv: str,
    cont_cols: list[str],
    bin_cols: list[str],
    usecols_train: list[str],
    usecols_test: list[str],
    dtype_map_train: dict,
    dtype_map_test: dict,
):
    train_df = _read_csv_fast(train_csv, usecols_train, dtype_map_train)
    test_df = _read_csv_fast(test_csv, usecols_test, dtype_map_test)

    y_np = train_df[config.label_name].to_numpy(copy=False)
    hit = np.flatnonzero(y_np == 5)
    if hit.size:
        mask = np.ones(train_df.shape[0], dtype=bool)
        mask[int(hit[0])] = False
        train_df = train_df.loc[mask]

    y = train_df[config.label_name].to_numpy(dtype=np.uint8, copy=False)
    X_cont = train_df[cont_cols].to_numpy(dtype=np.float32, copy=False)
    X_bin = train_df[bin_cols].to_numpy(dtype=np.uint8, copy=False)

    test_ids = test_df[config.id_field].to_numpy(dtype=np.int32, copy=False)
    X_test_cont = test_df[cont_cols].to_numpy(dtype=np.float32, copy=False)
    X_test_bin = test_df[bin_cols].to_numpy(dtype=np.uint8, copy=False)

    return X_cont, X_bin, y, X_test_cont, X_test_bin, test_ids


X_cont, X_bin, y_all, X_test_cont, X_test_bin, test_ids = load_train_test_numpy_fast(
    train_path,
    test_path,
    cont_cols,
    bin_cols,
    usecols_train,
    usecols_test,
    dtype_map_train,
    dtype_map_test,
)



## === cell 6
pass



## === cell 7
pass



## === cell 8
pass



## === cell 9
pass



## === cell 10
pass



## === cell 11
pass



## === cell 12
n = y_all.shape[0]
idx = np.arange(n, dtype=np.int32)
train_idx, val_idx = train_test_split(idx, test_size=0.15, random_state=42)

train_targets = np.ascontiguousarray(y_all[train_idx])
val_targets = np.ascontiguousarray(y_all[val_idx])

X_cont = np.ascontiguousarray(X_cont, dtype=np.float32)
X_bin = np.ascontiguousarray(X_bin, dtype=np.uint8)
X_test_cont = np.ascontiguousarray(X_test_cont, dtype=np.float32)
X_test_bin = np.ascontiguousarray(X_test_bin, dtype=np.uint8)




## === cell 13
def augment_with_row_stats(cont: np.ndarray, bin_: np.ndarray) -> np.ndarray:
    n_rows = cont.shape[0]
    n_cont = cont.shape[1]
    n_bin = bin_.shape[1]
    n_feat = n_cont + n_bin

    out = np.empty((n_rows, n_feat + 4), dtype=np.float32)
    out[:, :n_cont] = cont

    bin_f = bin_.astype(np.float32, copy=False)
    out[:, n_cont:n_feat] = bin_f

    sum_all = cont.sum(axis=1, dtype=np.float32)
    sum_all += bin_.sum(axis=1, dtype=np.float32)
    mean = sum_all / np.float32(n_feat)

    min_cont = cont.min(axis=1)
    max_cont = cont.max(axis=1)
    min_bin = bin_.min(axis=1).astype(np.float32, copy=False)
    max_bin = bin_.max(axis=1).astype(np.float32, copy=False)
    minv = np.minimum(min_cont, min_bin)
    maxv = np.maximum(max_cont, max_bin)

    cont_sq = np.empty_like(cont, dtype=np.float32)
    np.square(cont, out=cont_sq)
    ex2_cont = cont_sq.mean(axis=1, dtype=np.float32)
    del cont_sq

    bin_sq = np.empty_like(bin_f, dtype=np.float32)
    np.square(bin_f, out=bin_sq)
    ex2_bin = bin_sq.mean(axis=1, dtype=np.float32)
    del bin_sq

    ex2 = (ex2_cont * (n_cont / n_feat)) + (ex2_bin * (n_bin / n_feat))
    var_pop = ex2 - mean * mean
    np.maximum(var_pop, 0.0, out=var_pop)
    var_samp = var_pop * (n_feat / (n_feat - 1))
    std = np.sqrt(var_samp, dtype=np.float32)

    out[:, n_feat] = mean
    out[:, n_feat + 1] = minv
    out[:, n_feat + 2] = maxv
    out[:, n_feat + 3] = std
    return out


X_all_aug = augment_with_row_stats(X_cont, X_bin)
X_test_aug = augment_with_row_stats(X_test_cont, X_test_bin)
aug_feature_names = feature_names + ["mean", "min", "max", "std"]

X_train_aug = np.ascontiguousarray(X_all_aug[train_idx])
X_val_aug = np.ascontiguousarray(X_all_aug[val_idx])

del X_all_aug, X_cont, X_bin, X_test_cont, X_test_bin, y_all

train_pool = Pool(X_train_aug, train_targets, feature_names=aug_feature_names)
val_pool = Pool(X_val_aug, val_targets, feature_names=aug_feature_names)
test_pool = Pool(X_test_aug, feature_names=aug_feature_names)

cat_params = {
    "iterations": 15000,
    "learning_rate": 0.1,
    "od_wait": 1000,
    "depth": 7,
    "task_type": "CPU",
    "l2_leaf_reg": 3,
    "eval_metric": "Accuracy",
    "verbose": 1000,
    "random_seed": 42,
    "use_best_model": True,
    "thread_count": int(os.environ.get("OMP_NUM_THREADS", "8")),
}

cat = CatBoostClassifier(**cat_params)
cat.fit(train_pool, eval_set=val_pool)

y_pred = cat.predict(test_pool).reshape(-1)

submission = pd.DataFrame({config.id_field: test_ids, config.label_name: y_pred})
submission.to_csv(config.submit_filename, index=False)
