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
imbalanced-learn==0.13.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
optuna==4.5.0
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
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
import xgboost as xgb

import optuna

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

_CPU_COUNT = os.cpu_count() or 4
NTHREAD = max(1, _CPU_COUNT - 1)
os.environ.setdefault("OMP_NUM_THREADS", str(NTHREAD))
os.environ.setdefault("MKL_NUM_THREADS", str(NTHREAD))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(NTHREAD))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(NTHREAD))

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

plt.ioff()




## === cell 1
train_path = "../input/tabular-playground-series-dec-2021/train.csv"
test_path = "../input/tabular-playground-series-dec-2021/test.csv"
sub_path = "../input/tabular-playground-series-dec-2021/sample_submission.csv"

df_train_og = pd.read_csv(train_path, engine="c", low_memory=False)
df_test_og = pd.read_csv(test_path, engine="c", low_memory=False)
submission = pd.read_csv(sub_path, engine="c", low_memory=False)




## === cell 2
pass




## === cell 3
pass




## === cell 4
def reduce_mem_usage(df, verbose=True):
    """
    Downcast numeric columns to reduce memory usage while preserving values within dtype bounds.
    """
    numerics = ("int8", "int16", "int32", "int64", "float16", "float32", "float64")
    start_mem = df.memory_usage(deep=False).sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype
        if col_type.name not in numerics:
            continue

        s = df[col]
        if col_type == np.int8 or col_type == np.float32:
            continue

        c_min = s.min()
        c_max = s.max()

        if np.issubdtype(col_type, np.integer):
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = s.astype(np.int8, copy=False)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = s.astype(np.int16, copy=False)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = s.astype(np.int32, copy=False)
            else:
                df[col] = s.astype(np.int64, copy=False)
        else:
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = s.astype(np.float32, copy=False)
            else:
                df[col] = s.astype(np.float64, copy=False)

    end_mem = df.memory_usage(deep=False).sum() / 1024**2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df




## === cell 5
df_train = reduce_mem_usage(df_train_og)
df_test = reduce_mem_usage(df_test_og)
del df_train_og, df_test_og




## === cell 6
cat_count = collections.Counter(df_train["Cover_Type"])
print(cat_count)




## === cell 7
df_train = df_train[(df_train["Cover_Type"] != 4) & (df_train["Cover_Type"] != 5)]




## === cell 8
drop_cols = ["Id", "Cover_Type", "Soil_Type7", "Soil_Type15"]

y = df_train["Cover_Type"]
class_counts = y.value_counts()
minority_class = class_counts.idxmin()
minority_n = int(class_counts.min())

rng = np.random.RandomState(RANDOM_STATE)

y_arr = y.to_numpy()
classes = np.unique(y_arr)
pos_idx_by_class = {int(cls): np.flatnonzero(y_arr == cls) for cls in classes}

chosen_indices = []
for cls, cnt in class_counts.items():
    cls = int(cls)
    idx = pos_idx_by_class[cls]
    if cls == int(minority_class):
        chosen = idx
    else:
        chosen = rng.choice(idx, size=minority_n, replace=False)
    chosen_indices.append(chosen)

chosen_indices = np.concatenate(chosen_indices)
chosen_indices = chosen_indices[rng.permutation(chosen_indices.shape[0])]

df_res = df_train.iloc[chosen_indices].reset_index(drop=True)
X_res = df_res.drop(columns=drop_cols)
y_res = df_res["Cover_Type"]

print("After undersampling (not minority -> equal to minority count):")
print(y_res.value_counts().sort_index())




## === cell 9
cat_count = collections.Counter(y_res)
print(cat_count)




## === cell 10
pass




## === cell 11
pass




## === cell 12
x_train, x_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=RANDOM_STATE, stratify=y_res
)




## === cell 13
classes_sorted = np.sort(y_res.unique())
label_to_index = {int(c): i for i, c in enumerate(classes_sorted)}
index_to_label = {i: int(c) for i, c in enumerate(classes_sorted)}

y_train_enc = y_train.map(label_to_index).astype(int)
y_test_enc = y_test.map(label_to_index).astype(int)
y_res_enc = y_res.map(label_to_index).astype(int)

num_class = int(len(classes_sorted))
print("Original classes kept:", classes_sorted.tolist())
print("Encoded num_class:", num_class)




## === cell 14
def _safe_tree_method(preferred="gpu_hist"):
    try:
        tmp_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        tmp_y = np.array([0, 1], dtype=np.int32)
        model = XGBClassifier(
            tree_method=preferred,
            n_estimators=1,
            max_depth=2,
            learning_rate=0.1,
            verbosity=0,
            objective="multi:softmax",
            num_class=2,
            n_jobs=1,
        )
        model.fit(tmp_X, tmp_y)
        return preferred
    except Exception:
        return "hist"


TREE_METHOD = _safe_tree_method("gpu_hist")
TREE_METHOD




## === cell 15
scaler_opt = StandardScaler()
x_train_scaled = scaler_opt.fit_transform(x_train)
x_test_scaled = scaler_opt.transform(x_test)

x_train_scaled = np.asarray(x_train_scaled, dtype=np.float32, order="C")
x_test_scaled = np.asarray(x_test_scaled, dtype=np.float32, order="C")

y_train_np = y_train_enc.to_numpy(dtype=np.int32, copy=False)
y_test_np = y_test_enc.to_numpy(dtype=np.int32, copy=False)

feature_names = list(X_res.columns)


def _make_dmatrix(X, y, feature_names):
    if TREE_METHOD in ("hist", "gpu_hist"):
        try:
            return xgb.QuantileDMatrix(
                X, label=y, feature_names=feature_names, nthread=NTHREAD
            )
        except Exception:
            pass
    return xgb.DMatrix(X, label=y, feature_names=feature_names, nthread=NTHREAD)


dtrain = _make_dmatrix(x_train_scaled, y_train_np, feature_names)
dvalid = _make_dmatrix(x_test_scaled, y_test_np, feature_names)




## === cell 16
try:
    from optuna.integration.xgboost import XGBoostPruningCallback  # type: ignore
except Exception:
    XGBoostPruningCallback = None


class OptunaPruningTrainingCallback(xgb.callback.TrainingCallback):
    def __init__(self, trial: optuna.Trial, metric_key: str = "valid-merror"):
        self.trial = trial
        self.metric_key = metric_key

    def after_iteration(self, model, epoch: int, evals_log) -> bool:
        try:
            if "-" in self.metric_key:
                data_name, metric_name = self.metric_key.split("-", 1)
                if data_name in evals_log and metric_name in evals_log[data_name]:
                    score = float(evals_log[data_name][metric_name][-1])
                    self.trial.report(score, step=epoch)
                    if self.trial.should_prune():
                        raise optuna.TrialPruned()
        except optuna.TrialPruned:
            raise
        except Exception:
            pass
        return False


_BASE_XGB_PARAMS = {
    "learning_rate": 0.01,
    "tree_method": TREE_METHOD,
    "booster": "gbtree",
    "verbosity": 0,
    "objective": "multi:softmax",
    "num_class": num_class,
    "random_state": RANDOM_STATE,
    "seed": RANDOM_STATE,
    "nthread": NTHREAD,
    "max_bin": 256,
    "eval_metric": "merror",
}

_REG_LAMBDA_DIST = optuna.distributions.IntDistribution(1, 100)
_REG_ALPHA_DIST = optuna.distributions.IntDistribution(1, 100)
_SUBSAMPLE_DIST = optuna.distributions.FloatDistribution(0.2, 1.0, step=0.1)
_COLSAMPLE_DIST = optuna.distributions.FloatDistribution(0.2, 1.0, step=0.1)
_MAX_DEPTH_DIST = optuna.distributions.IntDistribution(3, 10)
_MIN_CHILD_DIST = optuna.distributions.IntDistribution(2, 10)
_GAMMA_DIST = optuna.distributions.FloatDistribution(0.0, 20.0)
_N_ESTIMATORS_DIST = optuna.distributions.IntDistribution(500, 4000, step=100)


def objective_xgb(trial):
    xgb_params = dict(_BASE_XGB_PARAMS)
    xgb_params.update(
        {
            "reg_lambda": trial._suggest("reg_lambda", _REG_LAMBDA_DIST),
            "reg_alpha": trial._suggest("reg_alpha", _REG_ALPHA_DIST),
            "subsample": trial._suggest("subsample", _SUBSAMPLE_DIST),
            "colsample_bytree": trial._suggest("colsample_bytree", _COLSAMPLE_DIST),
            "max_depth": trial._suggest("max_depth", _MAX_DEPTH_DIST),
            "min_child_weight": trial._suggest("min_child_weight", _MIN_CHILD_DIST),
            "gamma": trial._suggest("gamma", _GAMMA_DIST),
        }
    )
    n_estimators = int(trial._suggest("n_estimators", _N_ESTIMATORS_DIST))

    if XGBoostPruningCallback is not None:
        callbacks = [XGBoostPruningCallback(trial, "valid-merror")]
    else:
        callbacks = [OptunaPruningTrainingCallback(trial, metric_key="valid-merror")]

    booster = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=n_estimators,
        evals=[(dvalid, "valid")],
        verbose_eval=False,
        callbacks=callbacks,
        early_stopping_rounds=100,
    )

    best_iter = booster.best_iteration
    if best_iter is None:
        best_iter = n_estimators - 1

    y_pred_enc = booster.predict(dvalid, iteration_range=(0, int(best_iter) + 1))
    y_pred_enc = y_pred_enc.astype(np.int32, copy=False)
    return float((y_pred_enc == y_test_np).mean())




## === cell 17
optuna.logging.set_verbosity(optuna.logging.WARNING)

sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE, multivariate=True)
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5, n_warmup_steps=50, interval_steps=10
)
study_xgb = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

study_xgb.optimize(objective_xgb, n_trials=50, n_jobs=1, show_progress_bar=False)

try:
    print("Best holdout accuracy (Optuna objective):", study_xgb.best_value)
    print("Best params:", study_xgb.best_params)
except Exception as e:
    print("Optuna did not produce a best trial:", repr(e))




## === cell 18
fallback_params = {
    "n_estimators": 2000,
    "max_depth": 6,
    "min_child_weight": 2,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.0,
    "reg_alpha": 1,
    "reg_lambda": 1,
}

if len(study_xgb.trials) > 0:
    completed = [
        t for t in study_xgb.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
else:
    completed = []

if len(completed) > 0:
    best_params_xgb = dict(study_xgb.best_params)
else:
    best_params_xgb = dict(fallback_params)

best_params_xgb.update(
    {
        "learning_rate": 0.01,
        "tree_method": TREE_METHOD,
        "booster": "gbtree",
        "random_state": RANDOM_STATE,
        "seed": RANDOM_STATE,
        "n_jobs": NTHREAD,
        "verbosity": 0,
        "objective": "multi:softmax",
        "num_class": num_class,
        "max_bin": 256,
        "eval_metric": "merror",
    }
)

pipe = Pipeline(
    steps=[("step1", StandardScaler()), ("step2", XGBClassifier(**best_params_xgb))]
)

pipe.fit(x_train, y_train_enc)
val_pred_enc = pipe.predict(x_test)
print("Holdout accuracy:", accuracy_score(y_test_enc, val_pred_enc))

pipe.fit(X_res, y_res_enc)

df_test_features = df_test.drop(columns=["Id", "Soil_Type7", "Soil_Type15"])
final_pred_enc = pipe.predict(df_test_features)

final_pred = pd.Series(final_pred_enc).map(index_to_label).astype(int).to_numpy()

submission["Cover_Type"] = final_pred.astype(int)
out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission.head())
