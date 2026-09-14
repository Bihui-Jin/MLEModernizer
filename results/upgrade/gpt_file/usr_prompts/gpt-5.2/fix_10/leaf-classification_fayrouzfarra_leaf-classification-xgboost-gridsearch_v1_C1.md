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
Use binary leaf images and extracted features to identify the species of plant.

## Metric
Multi-class log loss. 

The submitted probabilities for a given device are not required to sum to one because they are rescaled prior to being scored (each row is divided by the row sum), but they need to be in the range of [0, 1]. In order to avoid the extremes of the log function, predicted probabilities are replaced with \\(max(min(p,1-10^{-15}),10^{-15})\\).

## Submission Format
You must submit a csv file with the image id, all candidate species names, and a probability for each species. The order of the rows does not matter. The file must have a header and should look like the following:

id,Acer_Capillipes,Acer_Circinatum,Acer_Mono,...
2,0.1,0.5,0,0.2,...
5,0,0.3,0,0.4,...
6,0,0,0,0.7,...
etc.

## Dataset
The dataset consists of images of leaf specimens which have been converted to binary black leaves against white backgrounds. 

Three sets of features are also provided per image: a shape contiguous descriptor, an interior texture histogram, and a ﬁne-scale margin histogram. 

For each feature, a 64-attribute vector is given per leaf sample.

### File descriptions
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format
- **images/** - the image files (each image is named with its corresponding id)

### Data fields
- **id** - an anonymous id unique to an image
- **margin_1, margin_2, margin_3, ..., margin_64** - each of the 64 attribute vectors for the margin feature
- **shape_1, shape_2, shape_3, ..., shape_64** - each of the 64 attribute vectors for the shape feature
- **texture_1, texture_2, texture_3, ..., texture_64** - each of the 64 attribute vectors for the texture feature

# 2. Python version

3.8

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
            description.md (70 lines)
            images.zip (22.0 MB)
            sample_submission.csv (100 lines)
            sample_submission.csv.zip (2.3 kB)
            test.csv (100 lines)
            test.csv.zip (39.3 kB)
            train.csv (892 lines)
            train.csv.zip (357.1 kB)
            images/
                42.jpg (32.6 kB)
                168.jpg (16.5 kB)
                ... and 988 other files
            leaf-classification/
                description.md (70 lines)
                images.zip (22.0 MB)
                ... and 6 other files
                images/
                    42.jpg (32.6 kB)
                    168.jpg (16.5 kB)
                    ... and 988 other files
                leaf-classification/
        input/
            description.md (70 lines)
            images.zip (22.0 MB)
            sample_submission.csv (100 lines)
            sample_submission.csv.zip (2.3 kB)
            test.csv (100 lines)
            test.csv.zip (39.3 kB)
            train.csv (892 lines)
            train.csv.zip (357.1 kB)
            images/
                42.jpg (32.6 kB)
                168.jpg (16.5 kB)
                ... and 988 other files
            leaf-classification/
                description.md (70 lines)
                images.zip (22.0 MB)
                ... and 6 other files
                images/
                    42.jpg (32.6 kB)
                    168.jpg (16.5 kB)
                    ... and 988 other files
                leaf-classification/
        working/
            leaf-classification/
                description.md (70 lines)
                images.zip (22.0 MB)
                ... and 6 other files
                images/
                    42.jpg (32.6 kB)
                    168.jpg (16.5 kB)
                    ... and 988 other files
                leaf-classification/
```

-> data/leaf-classification/sample_submission.csv has 99 rows and 100 columns.
The columns are: id, Acer_Capillipes, Acer_Circinatum, Acer_Mono, Acer_Opalus, Acer_Palmatum, Acer_Pictum, Acer_Platanoids, Acer_Rubrum, Acer_Rufinerve, Acer_Saccharinum, Alnus_Cordata, Alnus_Maximowiczii, Alnus_Rubra, Alnus_Sieboldiana... and 85 more columns

-> data/leaf-classification/test.csv has 99 rows and 193 columns.
The columns are: id, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13, margin14... and 178 more columns

-> data/leaf-classification/train.csv has 891 rows and 194 columns.
The columns are: id, species, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13... and 179 more columns

-> data/sample_submission.csv has 99 rows and 100 columns.
The columns are: id, Acer_Capillipes, Acer_Circinatum, Acer_Mono, Acer_Opalus, Acer_Palmatum, Acer_Pictum, Acer_Platanoids, Acer_Rubrum, Acer_Rufinerve, Acer_Saccharinum, Alnus_Cordata, Alnus_Maximowiczii, Alnus_Rubra, Alnus_Sieboldiana... and 85 more columns

-> data/test.csv has 99 rows and 193 columns.
The columns are: id, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13, margin14... and 178 more columns

-> data/train.csv has 891 rows and 194 columns.
The columns are: id, species, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13... and 179 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.70526

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.85065) has done: 'The timeout is dominated by `GridSearchCV` doing 5-fold CV over 8 hyperparameter combos (40 XGBoost fits), plus nested parallelism (`GridSearchCV(n_jobs=4)` and `XGBClassifier(n_jobs=4)`) causing CPU oversubscription. The fastest correctness-preserving fix is to keep the exact same search space, CV, scoring, and model, but eliminate nested parallelism by parallelizing at only one level and keeping XGBoost single-threaded during CV. We also enable Intel scikit-learn acceleration (already installed) and avoid the expensive directory walk/printing. All changes preserve the same evaluation semantics; only runtime improves.'
- What this solution (achieved 0.85065) has done: 'Your current score (0.85065, lower-is-better) is worse than the target (0.70526), so we should improve it with minimal changes that don’t alter the overall approach. The biggest, safe gain here is to add feature scaling (XGBoost can be sensitive on small tabular datasets with heterogeneous feature ranges) and to use stratified CV in the grid search so the fold class distributions match the full dataset, which typically improves logloss stability. We keep the same model family, objective, search space, CV=5, and logloss scoring; we just wrap the estimator in a `Pipeline(StandardScaler -> XGBClassifier)` and ensure consistent class/probability alignment in the submission. These changes are legitimate, minimal, and should move the score down toward your target.'
- What this solution (achieved 0.82722) has done: 'Your current logloss (0.85065; lower is better) is worse than the target (0.70526), so we should improve it with the smallest changes that keep the same overall XGBoost + GridSearchCV approach. The biggest low-risk gain is to tune a couple of regularization/row-sampling knobs that often reduce overfitting on this small dataset, while keeping the same CV setup and training loop. I’m also making the submission probability columns match `sample_submission.csv` exactly (by direct column assignment rather than a merge) to avoid any subtle alignment issues. Everything still trains via the same GridSearchCV over a small grid and writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

os.environ.setdefault("PYTHONHASHSEED", "42")
np.random.seed(42)



## === cell 1
df = pd.read_csv("../input/leaf-classification/train.csv.zip", index_col="id")
df



## === cell 2
for col in df.columns:
    if df[col].isna().sum() > 0:
        print(col, df[col].isna().sum() / len(df))



## === cell 3
df.species.value_counts()



## === cell 4
len(df.species.unique())



## === cell 5
y = df.species
y.head()



## === cell 6
X = df.drop(columns="species", axis=1)
X.head()



## === cell 7
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder().fit(y)
labeled_species = label_encoder.transform(y)



## === cell 8
classes = list(label_encoder.classes_)
classes



## === cell 9
parameters = {
    "n_estimators": list(range(100, 201, 100)),
    "learning_rate": [l / 100 for l in range(5, 15, 10)],
    "max_depth": list(range(6, 16, 10)),
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 5],
    "gamma": [0.0, 0.1],
    "reg_alpha": [0.0, 1e-3],
    "reg_lambda": [1.0, 2.0],
}
parameters



## === cell 10
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from itertools import product
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

scaler = StandardScaler(with_mean=True, with_std=True)
X_np = X.values.astype(np.float32, copy=False)
X_scaled = scaler.fit_transform(X_np)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_indices = list(cv.split(X_scaled, labeled_species))

cache_prefix = os.path.abspath("xgb_cache")
dtrain = xgb.QuantileDMatrix(X_scaled, label=labeled_species, max_bin=256, nthread=1)

grid = list(
    product(
        parameters["n_estimators"],
        parameters["learning_rate"],
        parameters["max_depth"],
        parameters["subsample"],
        parameters["colsample_bytree"],
        parameters["min_child_weight"],
        parameters["gamma"],
        parameters["reg_alpha"],
        parameters["reg_lambda"],
    )
)

best_score = np.inf
best_params = None

cache_path = "xgb_cv_cache.npy"
if os.path.exists(cache_path):
    try:
        cv_cache = np.load(cache_path, allow_pickle=True).item()
        if not isinstance(cv_cache, dict):
            cv_cache = {}
    except Exception:
        cv_cache = {}
else:
    cv_cache = {}


def _key_from_tuple(t):
    (
        n_estimators,
        learning_rate,
        max_depth,
        subsample,
        colsample_bytree,
        min_child_weight,
        gamma,
        reg_alpha,
        reg_lambda,
    ) = t
    return (
        int(n_estimators),
        float(learning_rate),
        int(max_depth),
        float(subsample),
        float(colsample_bytree),
        float(min_child_weight),
        float(gamma),
        float(reg_alpha),
        float(reg_lambda),
    )


_base_params = {
    "objective": "multi:softprob",
    "num_class": len(classes),
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "seed": 42,
    "verbosity": 0,
    "nthread": 1,  # single-threaded for reproducibility & to avoid oversubscription
    "cache_prefix": cache_prefix,
}

_cv_kwargs = dict(
    dtrain=dtrain,
    folds=fold_indices,
    shuffle=False,
    stratified=False,  # folds already stratified by our splitter
    metrics=("mlogloss",),
    verbose_eval=False,
    seed=42,
)


def _eval_one(param_tuple):
    key = _key_from_tuple(param_tuple)
    if key in cv_cache:
        return key, float(cv_cache[key])

    (
        n_estimators,
        learning_rate,
        max_depth,
        subsample,
        colsample_bytree,
        min_child_weight,
        gamma,
        reg_alpha,
        reg_lambda,
    ) = param_tuple

    params = dict(_base_params)
    params.update(
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        subsample=float(subsample),
        colsample_bytree=float(colsample_bytree),
        min_child_weight=float(min_child_weight),
        gamma=float(gamma),
        reg_alpha=float(reg_alpha),
        reg_lambda=float(reg_lambda),
    )

    cv_res = xgb.cv(
        params=params,
        num_boost_round=int(n_estimators),
        **_cv_kwargs,
    )
    mean_mlogloss = float(cv_res["test-mlogloss-mean"].iloc[-1])
    cv_cache[key] = mean_mlogloss
    return key, mean_mlogloss


cache_dirty = False
for t in grid:
    key, mean_mlogloss = _eval_one(t)
    cache_dirty = True

    if mean_mlogloss < best_score:
        (
            n_estimators,
            learning_rate,
            max_depth,
            subsample,
            colsample_bytree,
            min_child_weight,
            gamma,
            reg_alpha,
            reg_lambda,
        ) = t
        best_score = mean_mlogloss
        best_params = {
            "xgb__n_estimators": int(n_estimators),
            "xgb__learning_rate": float(learning_rate),
            "xgb__max_depth": int(max_depth),
            "xgb__subsample": float(subsample),
            "xgb__colsample_bytree": float(colsample_bytree),
            "xgb__min_child_weight": float(min_child_weight),
            "xgb__gamma": float(gamma),
            "xgb__reg_alpha": float(reg_alpha),
            "xgb__reg_lambda": float(reg_lambda),
        }

if cache_dirty:
    np.save(cache_path, cv_cache, allow_pickle=True)


class _GSearchShim:
    def __init__(self, best_params_):
        self.best_params_ = best_params_


gsearch = _GSearchShim(best_params)
gsearch.best_params_



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
XGBoostError                              Traceback (most recent call last)
/tmp/ipykernel_11/2745920466.py in <cell line: 0>()
    140 cache_dirty = False
    141 for t in grid:
--> 142     key, mean_mlogloss = _eval_one(t)
    143     cache_dirty = True
    144 

/tmp/ipykernel_11/2745920466.py in _eval_one(param_tuple)
    128     )
    129 
--> 130     cv_res = xgb.cv(
    131         params=params,
    132         num_boost_round=int(n_estimators),

/usr/local/lib/python3.11/dist-packages/xgboost/training.py in cv(params, dtrain, num_boost_round, nfold, stratified, folds, metrics, obj, feval, maximize, early_stopping_rounds, fpreproc, as_pandas, verbose_eval, show_stdv, seed, callbacks, shuffle, custom_metric)
    541 
    542     results: Dict[str, List[float]] = {}
--> 543     cvfolds = mknfold(
    544         dtrain, nfold, params, seed, metrics, fpreproc, stratified, folds, shuffle
    545     )

/usr/local/lib/python3.11/dist-packages/xgboost/training.py in mknfold(dall, nfold, param, seed, evals, fpreproc, stratified, folds, shuffle)
    396     for k in range(nfold):
    397         # perform the slicing using the indexes determined by the above methods
--> 398         dtrain = dall.slice(in_idset[k])
    399         dtest = dall.slice(out_idset[k])
    400         # run preprocessing on the data set if needed

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in slice(self, rindex, allow_groups)
   1256         res.handle = ctypes.c_void_p()
   1257         rindex = _maybe_np_slice(rindex, dtype=np.int32)
-> 1258         _check_call(
   1259             _LIB.XGDMatrixSliceDMatrixEx(
   1260                 self.handle,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in _check_call(ret)
    280     """
    281     if ret != 0:
--> 282         raise XGBoostError(py_str(_LIB.XGBGetLastError()))
    283 
    284 

XGBoostError: [00:26:53] /workspace/src/data/iterative_dmatrix.h:88: Slicing DMatrix is not supported for Quantile DMatrix.
Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x3effba) [0x7fff7aa24fba]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x3ff7ab) [0x7fff7aa347ab]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGDMatrixSliceDMatrixEx+0x146) [0x7fff7a795206]
  [bt] (3) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7ffff63ace2e]
  [bt] (4) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7ffff63a9493]
  [bt] (5) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0xa4d8) [0x7ffff63bc4d8]
  [bt] (6) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0x9c8e) [0x7ffff63bbc8e]
  [bt] (7) /usr/bin/python3(_PyObject_MakeTpCall+0x27c) [0x52f85c]
  [bt] (8) /usr/bin/python3(_PyEval_EvalFrameDefault+0x6bc) [0x53da0c]



## === cell 11
gsearch.best_params_



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3637443716.py in <cell line: 0>()
----> 1 gsearch.best_params_
      2 

NameError: name 'gsearch' is not defined

## === cell 12
best_n_estimators = gsearch.best_params_.get("xgb__n_estimators")
best_n_estimators



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/108908746.py in <cell line: 0>()
----> 1 best_n_estimators = gsearch.best_params_.get("xgb__n_estimators")
      2 best_n_estimators
      3 

NameError: name 'gsearch' is not defined

## === cell 13
best_learning_rate = gsearch.best_params_.get("xgb__learning_rate")
best_learning_rate



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2047786353.py in <cell line: 0>()
----> 1 best_learning_rate = gsearch.best_params_.get("xgb__learning_rate")
      2 best_learning_rate
      3 

NameError: name 'gsearch' is not defined

## === cell 14
best_max_depth = gsearch.best_params_.get("xgb__max_depth")
best_max_depth



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2281888336.py in <cell line: 0>()
----> 1 best_max_depth = gsearch.best_params_.get("xgb__max_depth")
      2 best_max_depth
      3 

NameError: name 'gsearch' is not defined

## === cell 15
best_subsample = gsearch.best_params_.get("xgb__subsample")
best_subsample



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/698389458.py in <cell line: 0>()
----> 1 best_subsample = gsearch.best_params_.get("xgb__subsample")
      2 best_subsample
      3 

NameError: name 'gsearch' is not defined

## === cell 16
best_colsample_bytree = gsearch.best_params_.get("xgb__colsample_bytree")
best_colsample_bytree



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2054979589.py in <cell line: 0>()
----> 1 best_colsample_bytree = gsearch.best_params_.get("xgb__colsample_bytree")
      2 best_colsample_bytree
      3 

NameError: name 'gsearch' is not defined

## === cell 17
best_min_child_weight = gsearch.best_params_.get("xgb__min_child_weight")
best_min_child_weight



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/616181047.py in <cell line: 0>()
----> 1 best_min_child_weight = gsearch.best_params_.get("xgb__min_child_weight")
      2 best_min_child_weight
      3 

NameError: name 'gsearch' is not defined

## === cell 18
best_gamma = gsearch.best_params_.get("xgb__gamma")
best_gamma



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/506902276.py in <cell line: 0>()
----> 1 best_gamma = gsearch.best_params_.get("xgb__gamma")
      2 best_gamma
      3 

NameError: name 'gsearch' is not defined

## === cell 19
best_reg_alpha = gsearch.best_params_.get("xgb__reg_alpha")
best_reg_alpha



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/964715463.py in <cell line: 0>()
----> 1 best_reg_alpha = gsearch.best_params_.get("xgb__reg_alpha")
      2 best_reg_alpha
      3 

NameError: name 'gsearch' is not defined

## === cell 20
best_reg_lambda = gsearch.best_params_.get("xgb__reg_lambda")
best_reg_lambda



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1145935854.py in <cell line: 0>()
----> 1 best_reg_lambda = gsearch.best_params_.get("xgb__reg_lambda")
      2 best_reg_lambda
      3 

NameError: name 'gsearch' is not defined

## === cell 21
from xgboost import XGBClassifier

final_xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=len(classes),
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42,
    n_jobs=max(1, min(4, os.cpu_count() or 1)),
    n_estimators=best_n_estimators,
    learning_rate=best_learning_rate,
    max_depth=best_max_depth,
    subsample=best_subsample,
    colsample_bytree=best_colsample_bytree,
    min_child_weight=best_min_child_weight,
    gamma=best_gamma,
    reg_alpha=best_reg_alpha,
    reg_lambda=best_reg_lambda,
    verbosity=0,
)
final_xgb.fit(X_scaled, labeled_species)



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2342746555.py in <cell line: 0>()
      8     random_state=42,
      9     n_jobs=max(1, min(4, os.cpu_count() or 1)),
---> 10     n_estimators=best_n_estimators,
     11     learning_rate=best_learning_rate,
     12     max_depth=best_max_depth,

NameError: name 'best_n_estimators' is not defined

## === cell 22
test = pd.read_csv("../input/leaf-classification/test.csv.zip", index_col="id")
test



## === cell 23
test_scaled = scaler.transform(test.values.astype(np.float32, copy=False))
pred_test = final_xgb.predict_proba(test_scaled)
pred_test.shape



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2489573547.py in <cell line: 0>()
      1 test_scaled = scaler.transform(test.values.astype(np.float32, copy=False))
----> 2 pred_test = final_xgb.predict_proba(test_scaled)
      3 pred_test.shape
      4 

NameError: name 'final_xgb' is not defined

## === cell 24
sample_sub = pd.read_csv("../input/leaf-classification/sample_submission.csv.zip")

est_classes = list(final_xgb.classes_)
proba_class_names = [classes[i] for i in est_classes]

submission = sample_sub.copy()
submission["id"] = test.index.values

for c in submission.columns:
    if c != "id":
        submission[c] = 0.0

submission.loc[:, proba_class_names] = pred_test

proba_cols = [c for c in submission.columns if c != "id"]
submission[proba_cols] = submission[proba_cols].clip(0.0, 1.0)

submission.to_csv("submission.csv", index=False)
print("wrote submission.csv with shape:", submission.shape)
submission.head()

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2273198586.py in <cell line: 0>()
      3 sample_sub = pd.read_csv("../input/leaf-classification/sample_submission.csv.zip")
      4 
----> 5 est_classes = list(final_xgb.classes_)
      6 proba_class_names = [classes[i] for i in est_classes]
      7 

NameError: name 'final_xgb' is not defined
