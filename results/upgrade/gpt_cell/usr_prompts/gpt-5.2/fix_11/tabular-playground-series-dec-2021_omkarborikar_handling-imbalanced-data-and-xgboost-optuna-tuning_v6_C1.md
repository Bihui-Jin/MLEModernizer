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

# 5. Target score

0.64974

# 6. Current score

0.56458

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11281) has done: 'Diagnosis: The crash happens because `XGBClassifier` is configured to use GPU-only settings (`tree_method='gpu_hist'` and `predictor='gpu_predictor'`), but the runtime has no CUDA device, triggering `Must have at least one device`. This is an environment incompatibility rather than a data/shape issue. We should switch to CPU-compatible settings while keeping the same model type, objective, and hyperparameters.

Patch summary: In cell 16, detect whether XGBoost has an available GPU; if not, update the already-created pipeline’s XGBClassifier parameters to CPU-safe equivalents (`tree_method='hist'` and `predictor='cpu_predictor'`) before calling `fit`. This keeps the core logic intact and only changes hardware backend selection to prevent the crash.

Updated cells: cell 16 only.

Compatibility notes for cell k+1: `pipe` remains a fitted `Pipeline` and `pipe.predict(...)` in cell 17 continues to work unchanged, producing the same label-encoded class predictions as before.

Assumptions: The environment has no usable GPU/CUDA device; using CPU backend is acceptable and does not change the intended evaluation semantics (only runtime backend).'
- What this solution (achieved 0.56458) has done: 'The crash happens because `y_train` is a NumPy array (coming from the resampling step) and NumPy arrays don’t have the pandas `.map()` method. The minimal fix is to encode labels using a vectorized NumPy mapping instead of `.map()`, while keeping the same `label_encoder_` / `label_encoder_inv_` dictionaries for later decoding. This preserves the model training semantics (same labels, just encoded deterministically) and keeps `label_encoder_inv_` available for cell 17. No other logic (split, pipeline, model, GPU fallback) is changed.'
- What this solution (achieved 0.56458) has done: 'The crash happens because after filtering `y_test` to only “known” classes, `x_test`/`y_test_arr` can become empty, and `np.vectorize(label_encoder_.get)` raises a `ValueError` on empty inputs unless `otypes` is specified. The minimal fix is to add `otypes=[np.int32]` to both `np.vectorize(...)` calls so encoding works deterministically even for empty arrays. This preserves the same label mapping logic and training semantics, only preventing the empty-input crash. No other behavior is changed, and the variables used by cell 17 remain defined.'
- What this solution (achieved 0.56458) has done: 'Diagnosis: Cell 17 crashes because `x_test` can be empty after cell 16 filters out labels not present in `y_train` (`_known_mask`). When `x_test` has 0 rows, `StandardScaler.transform()` inside the pipeline raises `ValueError: Found array with 0 sample(s)`.  
Patch summary: Add a minimal guard in cell 17 to handle the empty-test case deterministically by skipping `pipe.predict` and returning an empty `y_pred_enc` array, preserving the expected variable name and type. This keeps downstream evaluation semantics consistent (cell 18 also operate on empty arrays without crashing, returning `nan`/raising depending on sklearn version, but it no longer fail in preprocessing).  
Updated cells: Only cell 17 is changed.  
Compatibility notes for cell k+1: `y_pred_enc` still exists and is a NumPy array; if `x_test` is non-empty, behavior is unchanged. If `x_test` is empty, `y_pred_enc` be an empty `int32` array matching `y_test_enc` length (0).  
Assumptions: An empty `x_test` is acceptable and should not abort the run; no changes are allowed to earlier filtering logic in cell 16.'

# 9. Code solution

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
rus = RandomUnderSampler(sampling_strategy="not minority", random_state=42)
X = df_train.drop(columns=["Id", "Cover_Type"])
y = df_train["Cover_Type"]
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
classes_ = np.sort(pd.unique(np.asarray(y_train)))
label_encoder_ = {c: i for i, c in enumerate(classes_)}
label_encoder_inv_ = {i: c for c, i in label_encoder_.items()}

y_train_arr = np.asarray(y_train)
y_train_enc = np.vectorize(label_encoder_.get, otypes=[np.int32])(y_train_arr).astype(
    np.int32
)

y_test_arr = np.asarray(y_test)
_known_mask = np.isin(y_test_arr, classes_)
x_test = x_test.loc[_known_mask]
y_test_arr = y_test_arr[_known_mask]
y_test_enc = np.vectorize(label_encoder_.get, otypes=[np.int32])(y_test_arr).astype(
    np.int32
)

pipe.set_params(step2__num_class=len(classes_))

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


## === cell 17
if hasattr(x_test, "shape") and x_test.shape[0] == 0:
    y_pred_enc = np.asarray([], dtype=np.int32)
else:
    y_pred_enc = pipe.predict(x_test)


## === cell 18
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test_enc, y_pred_enc))



## === cell 19
X_test = df_test[feature_cols]
final_pred_enc = pipe.predict(X_test)
final_pred = np.vectorize(label_encoder_inv_.get)(final_pred_enc).astype(np.int32)

submission["Cover_Type"] = final_pred
submission.to_csv("Submission.csv", index=False)
print(submission.head())
print("Wrote Submission.csv with shape:", submission.shape)
