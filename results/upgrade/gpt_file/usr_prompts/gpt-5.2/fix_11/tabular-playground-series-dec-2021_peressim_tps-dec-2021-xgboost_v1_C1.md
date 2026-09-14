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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
xgboost==2.0.3

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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import random
import warnings
import gc

try:
    from sklearnex import patch_sklearn  # scikit-learn-intelex

    patch_sklearn()
except Exception:
    pass

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import xgboost as xgb

warnings.filterwarnings("ignore")




## === cell 1
seed = 47
random.seed(seed)
np.random.seed(seed)




## === cell 2
def evaluate_model(model, x, y):
    try:
        y_pred = model.inplace_predict(x)
    except Exception:
        y_pred = model.predict(x)
    acc = accuracy_score(y, y_pred)
    return {"accuracy": acc}




## === cell 3
def _build_dtypes_for_tps_dec2021():
    dtypes = {"Id": np.int32, "Cover_Type": np.int8}

    num_int16 = [
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
    for c in num_int16:
        dtypes[c] = np.int16

    for i in range(1, 5):
        dtypes[f"Wilderness_Area{i}"] = np.int8
    for i in range(1, 41):
        dtypes[f"Soil_Type{i}"] = np.int8

    return dtypes


TRAIN_PATH = "/kaggle/input/tabular-playground-series-dec-2021/train.csv"
TEST_PATH = "/kaggle/input/tabular-playground-series-dec-2021/test.csv"

DTYPES_TRAIN = _build_dtypes_for_tps_dec2021()
DTYPES_TEST = {k: v for k, v in DTYPES_TRAIN.items() if k != "Cover_Type"}

train_df = pd.read_csv(
    TRAIN_PATH, sep=",", engine="c", dtype=DTYPES_TRAIN, low_memory=False
)




## === cell 4
DROP_COLS = ["Id", "Soil_Type7", "Soil_Type15", "Cover_Type"]

_base_cols = [c for c in train_df.columns if c not in DROP_COLS]
X_base_df = train_df[_base_cols]

y_raw = train_df["Cover_Type"].astype(np.int8).to_numpy(copy=False)

X_base_np = np.ascontiguousarray(X_base_df.to_numpy(dtype=np.float32, copy=True))

col_to_i = {c: i for i, c in enumerate(_base_cols)}

i_elev = col_to_i["Elevation"]
i_aspect = col_to_i["Aspect"]
i_hdh = col_to_i["Horizontal_Distance_To_Hydrology"]
i_vdh = col_to_i["Vertical_Distance_To_Hydrology"]
i_hroad = col_to_i["Horizontal_Distance_To_Roadways"]
i_hfire = col_to_i["Horizontal_Distance_To_Fire_Points"]
i_hs9 = col_to_i["Hillshade_9am"]
i_hsn = col_to_i["Hillshade_Noon"]
i_hs3 = col_to_i["Hillshade_3pm"]

soil_features = [c for c in _base_cols if c.startswith("Soil_Type")]
wilderness_features = [c for c in _base_cols if c.startswith("Wilderness_Area")]
soil_idx = np.fromiter((col_to_i[c] for c in soil_features), dtype=np.int32)
wild_idx = np.fromiter((col_to_i[c] for c in wilderness_features), dtype=np.int32)

n = X_base_np.shape[0]
mean_feat = X_base_np.mean(axis=1, dtype=np.float32)

elev = X_base_np[:, i_elev]
aspect = X_base_np[:, i_aspect]
hdh = X_base_np[:, i_hdh]
vdh = X_base_np[:, i_vdh]
hroad = X_base_np[:, i_hroad]
hfire = X_base_np[:, i_hfire]

aspect2 = np.where(aspect + 180 > 360, aspect - 180, aspect + 180).astype(
    np.float32, copy=False
)
np.mod(aspect, 360, out=aspect)

np.clip(X_base_np[:, i_hs9], 0, 255, out=X_base_np[:, i_hs9])
np.clip(X_base_np[:, i_hsn], 0, 255, out=X_base_np[:, i_hsn])
np.clip(X_base_np[:, i_hs3], 0, 255, out=X_base_np[:, i_hs3])

EHiElv = (hroad * elev).astype(np.float32, copy=False)
EViElv = (vdh * elev).astype(np.float32, copy=False)
Highwater = (vdh < 0).astype(np.float32, copy=False)
EVDtH = (elev - vdh).astype(np.float32, copy=False)
EHDtH = (elev - hdh * 0.2).astype(np.float32, copy=False)
Euclid = np.sqrt(hdh * hdh + vdh * vdh, dtype=np.float32)
Manhattan = (hdh + vdh).astype(np.float32, copy=False)
Hydro_Fire_1 = (hdh + hfire).astype(np.float32, copy=False)
Hydro_Fire_2 = np.abs(hdh - hfire).astype(np.float32, copy=False)
Hydro_Road_1 = np.abs(hdh + hroad).astype(np.float32, copy=False)
Hydro_Road_2 = np.abs(hdh - hroad).astype(np.float32, copy=False)
Fire_Road_1 = np.abs(hfire + hroad).astype(np.float32, copy=False)
Fire_Road_2 = np.abs(hfire - hroad).astype(np.float32, copy=False)
Hillshade_3pm_is_zero = (X_base_np[:, i_hs3] == 0).astype(np.float32, copy=False)

soil_type_count = X_base_np[:, soil_idx].sum(axis=1, dtype=np.float32)
wilderness_area_count = X_base_np[:, wild_idx].sum(axis=1, dtype=np.float32)

X_full_np = np.column_stack(
    (
        X_base_np,
        mean_feat,
        EHiElv,
        EViElv,
        aspect2,
        Highwater,
        EVDtH,
        EHDtH,
        Euclid,
        Manhattan,
        Hydro_Fire_1,
        Hydro_Fire_2,
        Hydro_Road_1,
        Hydro_Road_2,
        Fire_Road_1,
        Fire_Road_2,
        Hillshade_3pm_is_zero,
        soil_type_count,
        wilderness_area_count,
    )
).astype(np.float32, copy=False)
X_full_np = np.ascontiguousarray(X_full_np)

idx = np.arange(n, dtype=np.int32)
idx_train, idx_valid, y_train_raw, y_valid_raw = train_test_split(
    idx, y_raw, test_size=0.2, random_state=seed, shuffle=True
)

X_train_np = X_full_np[idx_train]
X_valid_np = X_full_np[idx_valid]

y_train_np = (y_train_raw - 1).astype(np.int8, copy=False)
y_valid_np = (y_valid_raw - 1).astype(np.int8, copy=False)




## === cell 5
params = {
    "max_depth": 18,
    "subsample": 1.0,
    "eta": 0.3,
    "colsample_bytree": 1.0,
    "gamma": 0.0,
    "min_child_weight": 1,
    "reg_alpha": 1,
    "objective": "multi:softmax",
    "num_class": 7,
    "tree_method": "hist",
    "seed": seed,
}

device = "cpu"
try:
    _cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if _cuda_visible not in ("", "-1"):
        device = "cuda"
except Exception:
    device = "cpu"

n_jobs = os.cpu_count() or 1
params["nthread"] = n_jobs

if device == "cuda":
    params["device"] = "cuda"

_DMatrix = getattr(xgb, "QuantileDMatrix", xgb.DMatrix)
dtrain = _DMatrix(X_train_np, label=y_train_np)
dvalid = _DMatrix(X_valid_np, label=y_valid_np)

model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=300,
    evals=[(dvalid, "valid")],
    verbose_eval=False,
)

score = evaluate_model(model, X_valid_np, y_valid_np)
print(score)




## === cell 6
_to_del = [
    "train_df",
    "X_base_df",
    "X_base_np",
    "X_full_np",
    "X_train_np",
    "X_valid_np",
    "idx",
    "idx_train",
    "idx_valid",
    "y_raw",
    "y_train_raw",
    "y_valid_raw",
    "dtrain",
    "dvalid",
    "mean_feat",
    "EHiElv",
    "EViElv",
    "aspect2",
    "Highwater",
    "EVDtH",
    "EHDtH",
    "Euclid",
    "Manhattan",
    "Hydro_Fire_1",
    "Hydro_Fire_2",
    "Hydro_Road_1",
    "Hydro_Road_2",
    "Fire_Road_1",
    "Fire_Road_2",
    "Hillshade_3pm_is_zero",
    "soil_type_count",
    "wilderness_area_count",
]
for _name in _to_del:
    if _name in globals():
        del globals()[_name]
gc.collect()




## === cell 7
test_df = pd.read_csv(
    TEST_PATH, sep=",", engine="c", dtype=DTYPES_TEST, low_memory=False
)

_base_cols_test = [
    c for c in test_df.columns if c not in ["Id", "Soil_Type7", "Soil_Type15"]
]
X_test_base_df = test_df[_base_cols_test]

if _base_cols_test != _base_cols:
    X_test_base_df = X_test_base_df[_base_cols]
    _base_cols_test = _base_cols

X_test_base_np = np.ascontiguousarray(
    X_test_base_df.to_numpy(dtype=np.float32, copy=True)
)

mean_feat_t = X_test_base_np.mean(axis=1, dtype=np.float32)

elev_t = X_test_base_np[:, i_elev]
aspect_t = X_test_base_np[:, i_aspect]
hdh_t = X_test_base_np[:, i_hdh]
vdh_t = X_test_base_np[:, i_vdh]
hroad_t = X_test_base_np[:, i_hroad]
hfire_t = X_test_base_np[:, i_hfire]

aspect2_t = np.where(aspect_t + 180 > 360, aspect_t - 180, aspect_t + 180).astype(
    np.float32, copy=False
)
np.mod(aspect_t, 360, out=aspect_t)

np.clip(X_test_base_np[:, i_hs9], 0, 255, out=X_test_base_np[:, i_hs9])
np.clip(X_test_base_np[:, i_hsn], 0, 255, out=X_test_base_np[:, i_hsn])
np.clip(X_test_base_np[:, i_hs3], 0, 255, out=X_test_base_np[:, i_hs3])

EHiElv_t = (hroad_t * elev_t).astype(np.float32, copy=False)
EViElv_t = (vdh_t * elev_t).astype(np.float32, copy=False)
Highwater_t = (vdh_t < 0).astype(np.float32, copy=False)
EVDtH_t = (elev_t - vdh_t).astype(np.float32, copy=False)
EHDtH_t = (elev_t - hdh_t * 0.2).astype(np.float32, copy=False)
Euclid_t = np.sqrt(hdh_t * hdh_t + vdh_t * vdh_t, dtype=np.float32)
Manhattan_t = (hdh_t + vdh_t).astype(np.float32, copy=False)
Hydro_Fire_1_t = (hdh_t + hfire_t).astype(np.float32, copy=False)
Hydro_Fire_2_t = np.abs(hdh_t - hfire_t).astype(np.float32, copy=False)
Hydro_Road_1_t = np.abs(hdh_t + hroad_t).astype(np.float32, copy=False)
Hydro_Road_2_t = np.abs(hdh_t - hroad_t).astype(np.float32, copy=False)
Fire_Road_1_t = np.abs(hfire_t + hroad_t).astype(np.float32, copy=False)
Fire_Road_2_t = np.abs(hfire_t - hroad_t).astype(np.float32, copy=False)
Hillshade_3pm_is_zero_t = (X_test_base_np[:, i_hs3] == 0).astype(np.float32, copy=False)

soil_type_count_t = X_test_base_np[:, soil_idx].sum(axis=1, dtype=np.float32)
wilderness_area_count_t = X_test_base_np[:, wild_idx].sum(axis=1, dtype=np.float32)

X_test_np = np.column_stack(
    (
        X_test_base_np,
        mean_feat_t,
        EHiElv_t,
        EViElv_t,
        aspect2_t,
        Highwater_t,
        EVDtH_t,
        EHDtH_t,
        Euclid_t,
        Manhattan_t,
        Hydro_Fire_1_t,
        Hydro_Fire_2_t,
        Hydro_Road_1_t,
        Hydro_Road_2_t,
        Fire_Road_1_t,
        Fire_Road_2_t,
        Hillshade_3pm_is_zero_t,
        soil_type_count_t,
        wilderness_area_count_t,
    )
).astype(np.float32, copy=False)
X_test_np = np.ascontiguousarray(X_test_np)

try:
    target0 = model.inplace_predict(X_test_np).astype(int).squeeze()
except Exception:
    dtest = xgb.DMatrix(X_test_np)
    target0 = model.predict(dtest).astype(int).squeeze()

target = target0 + 1

ids = test_df["Id"].to_numpy(copy=False)
submission_xgboost = pd.DataFrame({"Id": ids, "Cover_Type": target.astype(np.int16)})

submission_xgboost.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_xgboost.shape)
print(submission_xgboost.head())
print(submission_xgboost.dtypes)
