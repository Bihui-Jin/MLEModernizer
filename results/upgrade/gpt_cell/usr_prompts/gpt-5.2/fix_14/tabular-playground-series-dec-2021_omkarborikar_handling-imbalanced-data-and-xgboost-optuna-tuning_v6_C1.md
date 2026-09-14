# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.10

# 2. Installed packages

geopandas==0.14.4
imbalanced-learn==0.13.0
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
sklearn-pandas==2.2.0
xgboost==2.0.3

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import collections

try:
    from imblearn.under_sampling import RandomUnderSampler  # type: ignore
except Exception:

    class RandomUnderSampler:
        def __init__(self, sampling_strategy="auto", random_state=None):
            self.sampling_strategy = sampling_strategy
            self.random_state = random_state

        def fit_resample(self, X, y):
            if hasattr(X, "iloc"):
                X_df = X
                X_values = X.values
            else:
                X_df = None
                X_values = np.asarray(X)

            y_arr = np.asarray(y)
            rng = np.random.default_rng(self.random_state)

            classes, counts = np.unique(y_arr, return_counts=True)
            min_count = counts.min()
            keep_idx = []
            for c in classes:
                idx_c = np.flatnonzero(y_arr == c)
                if idx_c.size > min_count:
                    idx_c = rng.choice(idx_c, size=min_count, replace=False)
                keep_idx.append(idx_c)
            keep_idx = np.concatenate(keep_idx)
            keep_idx.sort()

            X_res = (
                X_df.iloc[keep_idx].copy() if X_df is not None else X_values[keep_idx]
            )
            y_res = y_arr[keep_idx]
            return X_res, y_res




## === cell 1
df_train_og = pd.read_csv("../input/tabular-playground-series-dec-2021/train.csv")
df_test_og = pd.read_csv("../input/tabular-playground-series-dec-2021/test.csv")
submission = pd.read_csv(
    "../input/tabular-playground-series-dec-2021/sample_submission.csv"
)



## === cell 2
df_train_og.shape



## === cell 3
df_train_og.head()



## === cell 4
df_train_og.nunique()




## === cell 5
def reduce_mem_usage(df, verbose=True):
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )

    return df




## === cell 6
df_train = reduce_mem_usage(df_train_og)
df_test = reduce_mem_usage(df_test_og)
del df_train_og
del df_test_og



## === cell 7
cat_count = collections.Counter(df_train["Cover_Type"])
cat_freq = cat_count.values()
cat = cat_count.keys()
plt.bar(cat, cat_freq)
print(cat_count)



## === cell 8
df_train = df_train



## === cell 9
y_full = df_train["Cover_Type"]
class_counts = y_full.value_counts()

target_per_class_cap = 600_000

sampling_strategy = {
    int(c): int(min(target_per_class_cap, class_counts.loc[c]))
    for c in class_counts.index
}

rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
X = df_train.drop(columns=["Id", "Cover_Type"])
y = y_full
X_res, y_res = rus.fit_resample(X, y)



## === cell 10
cat_count = collections.Counter(y_res)
cat_freq = cat_count.values()
cat = cat_count.keys()
plt.bar(cat, cat_freq)
print(cat_count)



## === cell 11
feature_cols = [
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
X = X_res[feature_cols]
y = y_res



## === cell 12
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k="all")
fitter = selector.fit(X, y)
scores_df = pd.DataFrame(fitter.scores_)
columns_df = pd.DataFrame(X.columns)
featurescores = pd.concat([scores_df, columns_df], axis=1)
featurescores.columns = ["score", "column name"]
plt.figure(figsize=(20, 5))
plt.bar(featurescores["column name"], featurescores["score"], width=0.4)
plt.xticks(rotation="vertical")
plt.plot()



## === cell 13
X = X
y = y



## === cell 14
_y_counts = pd.Series(y).value_counts()
_stratify_arg = y if _y_counts.min() >= 2 else None

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=_stratify_arg
)



## === cell 15
pipe = Pipeline(
    steps=[
        ("step1", StandardScaler()),
        (
            "step2",
            XGBClassifier(
                objective="multi:softmax",
                tree_method="gpu_hist",
                eval_metric="mlogloss",
                subsample=0.6,
                gamma=0.5,
                max_depth=6,
                alpha=0,
                learning_rate=0.03,
                n_estimators=1000,
                predictor="gpu_predictor",
                n_jobs=-1,
                random_state=42,
            ),
        ),
    ]
)



## === cell 16
classes_ = np.sort(pd.unique(np.asarray(y)))
label_encoder_ = {c: i for i, c in enumerate(classes_)}
label_encoder_inv_ = {i: c for c, i in label_encoder_.items()}

y_train_arr = np.asarray(y_train)
y_train_enc = np.vectorize(label_encoder_.get, otypes=[object])(y_train_arr)
if np.any(pd.isna(y_train_enc)):
    missing = pd.unique(y_train_arr[pd.isna(y_train_enc)])
    raise ValueError(f"Found unknown labels in y_train not in classes_: {missing}")
y_train_enc = y_train_enc.astype(np.int32)

y_test_arr = np.asarray(y_test)
y_test_enc = np.vectorize(label_encoder_.get, otypes=[object])(y_test_arr)
if np.any(pd.isna(y_test_enc)):
    missing = pd.unique(y_test_arr[pd.isna(y_test_enc)])
    raise ValueError(f"Found unknown labels in y_test not in classes_: {missing}")
y_test_enc = y_test_enc.astype(np.int32)

pipe.set_params(step2__num_class=int(len(classes_)))

try:
    from xgboost import config as xgb_config  # xgboost>=2.0

    has_gpu = bool(xgb_config.get_config().get("use_cuda", False))
except Exception:
    has_gpu = False

if not has_gpu:
    pipe.set_params(
        step2__tree_method="hist",
        step2__predictor="cpu_predictor",
    )

pipe.fit(x_train, y_train_enc)


## --- ERROR in cell 16, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1945410938.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     35[0m     )
[1;32m     36[0m [0;34m[0m[0m
[0;32m---> 37[0;31m [0mpipe[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mx_train[0m[0;34m,[0m [0my_train_enc[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py[0m in [0;36mfit[0;34m(self, X, y, **fit_params)[0m
[1;32m    403[0m             [0;32mif[0m [0mself[0m[0;34m.[0m[0m_final_estimator[0m [0;34m!=[0m [0;34m"passthrough"[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    404[0m                 [0mfit_params_last_step[0m [0;34m=[0m [0mfit_params_steps[0m[0;34m[[0m[0mself[0m[0;34m.[0m[0msteps[0m[0;34m[[0m[0;34m-[0m[0;36m1[0m[0;34m][0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 405[0;31m                 [0mself[0m[0;34m.[0m[0m_final_estimator[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mXt[0m[0;34m,[0m [0my[0m[0;34m,[0m [0;34m**[0m[0mfit_params_last_step[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    406[0m [0;34m[0m[0m
[1;32m    407[0m         [0;32mreturn[0m [0mself[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36minner_f[0;34m(*args, **kwargs)[0m
[1;32m    728[0m             [0;32mfor[0m [0mk[0m[0;34m,[0m [0marg[0m [0;32min[0m [0mzip[0m[0;34m([0m[0msig[0m[0;34m.[0m[0mparameters[0m[0;34m,[0m [0margs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    729[0m                 [0mkwargs[0m[0;34m[[0m[0mk[0m[0;34m][0m [0;34m=[0m [0marg[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 730[0;31m             [0;32mreturn[0m [0mfunc[0m[0;34m([0m[0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    731[0m [0;34m[0m[0m
[1;32m    732[0m         [0;32mreturn[0m [0minner_f[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py[0m in [0;36mfit[0;34m(self, X, y, sample_weight, base_margin, eval_set, eval_metric, early_stopping_rounds, verbose, xgb_model, sample_weight_eval_set, base_margin_eval_set, feature_weights, callbacks)[0m
[1;32m   1469[0m                 [0;32mor[0m [0;32mnot[0m [0;34m([0m[0mclasses[0m [0;34m==[0m [0mexpected_classes[0m[0;34m)[0m[0;34m.[0m[0mall[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1470[0m             ):
[0;32m-> 1471[0;31m                 raise ValueError(
[0m[1;32m   1472[0m                     [0;34mf"Invalid classes inferred from unique values of `y`.  "[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1473[0m                     [0;34mf"Expected: {expected_classes}, got {classes}"[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: Invalid classes inferred from unique values of `y`.  Expected: [0 1 2 3 4], got [1 2 3 5 6]

## === cell 17
if hasattr(x_test, "shape") and x_test.shape[0] == 0:
    y_pred_enc = np.asarray([], dtype=np.int32)
else:
    y_pred_enc = pipe.predict(x_test)
