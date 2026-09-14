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
Predict the score of student essays.

## Metric
Quadratic weighted kappa.

## Submission Format
For each `essay_id` in the test set, you must predict the corresponding `score` (between 1-6, see [rubric](https://storage.googleapis.com/kaggle-forum-message-attachments/2733927/20538/Rubric_%20Holistic%20Essay%20Scoring.pdf) for more details). The file should contain a header and have the following format:

```
essay_id,score
000d118,3
000fe60,3
001ab80,4
...
```

## Dataset
- **train.csv** - Essays and scores to be used as training data.
    - `essay_id` - The unique ID of the essay
    - `full_text` - The full essay response
    - `score` - Holistic score of the essay on a 1-6 scale
- **test.csv** - The essays to be used as test data. Contains the same fields as `train.csv`, aside from exclusion of `score`.
- **sample_submission.csv** - A submission file in the correct format.
    - `essay_id` - The unique ID of the essay
    - `score` - The predicted holistic score of the essay on a 1-6 scale

# 2. Python version

3.12

# 3. Installed packages

cudf-polars-cu12==25.6.0
geopandas==0.14.4
joblib==1.5.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
polars==1.25.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
spacy==3.8.7
spacy-legacy==3.0.12
spacy-loggers==1.0.5
tqdm==4.67.1
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        input/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        working/
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
```

-> data/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/learning-agency-lab-automated-essay-scoring-2/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/learning-agency-lab-automated-essay-scoring-2/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> data/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> (stopped after 10 files for performance)

# 5. Target score

0.7789778763185116

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.exists(train_path), f"Missing: {train_path}"
assert os.path.exists(test_path), f"Missing: {test_path}"
assert os.path.exists(sample_sub_path), f"Missing: {sample_sub_path}"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_sub_path)

print(train_df.shape, test_df.shape, sample_sub.shape)
print(train_df.columns.tolist())
print(test_df.columns.tolist())
print(sample_sub.columns.tolist())




## === cell 1
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
    print("Applied sklearnex patch for acceleration.")
except Exception as e:
    print("sklearnex patch not applied:", repr(e))

import xgboost as xgb

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score

NTHREAD = os.cpu_count() or 4

for df, name in [(train_df, "train"), (test_df, "test")]:
    if "full_text" not in df.columns or "essay_id" not in df.columns:
        raise ValueError(f"{name} is missing required columns.")
    df["full_text"] = df["full_text"].fillna("").astype(str)
    df["essay_id"] = df["essay_id"].astype(str)

le = LabelEncoder()
y_raw = train_df["score"].astype(int).values
y = le.fit_transform(y_raw)  # classes correspond to scores 1..6

n_classes = len(le.classes_)
print("Encoded classes:", le.classes_, "n_classes:", n_classes)

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    max_features=60000,
    lowercase=True,
    sublinear_tf=True,
    token_pattern=r"(?u)\b\w+\b",
    strip_accents=None,
    dtype=np.float32,  # negligible FP diffs allowed
)

train_text = train_df["full_text"].values
test_text = test_df["full_text"].values

X_train = tfidf.fit_transform(train_text).tocsr(copy=False)
X_test = tfidf.transform(test_text).tocsr(copy=False)

print("TFIDF shapes:", X_train.shape, X_test.shape)

del train_text, test_text




## === cell 2
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)

models = []
oof_pred = np.zeros((X_train.shape[0],), dtype=np.float32)

cache_dir = "/kaggle/working/xgb_cache"
os.makedirs(cache_dir, exist_ok=True)
cache_prefix = os.path.join(cache_dir, "train_cache")

params = {
    "objective": "multi:softprob",
    "num_class": n_classes,
    "eval_metric": "mlogloss",
    "eta": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 1.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "tree_method": "hist",
    "seed": RANDOM_STATE,
    "nthread": NTHREAD,
}

num_boost_round = 1200
cls_idx = np.arange(n_classes, dtype=np.float32)

dtrain_full = xgb.QuantileDMatrix(
    xgb.DMatrix(X_train, label=y, nthread=NTHREAD, enable_categorical=False),
)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), 1):
    X_tr, X_va = X_train[tr_idx], X_train[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    dtr = xgb.QuantileDMatrix(X_tr, label=y_tr, nthread=NTHREAD)
    dva = xgb.QuantileDMatrix(X_va, label=y_va, nthread=NTHREAD)

    model = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=num_boost_round,
        evals=[(dva, "valid")],
        verbose_eval=200,
    )
    models.append(model)

    proba_va = model.predict(dva)  # (n, n_classes)
    oof_pred[va_idx] = proba_va @ cls_idx

oof_rounded = np.clip(np.rint(oof_pred), 0, n_classes - 1).astype(int)
oof_score = le.inverse_transform(oof_rounded)
qwk = cohen_kappa_score(y_raw, oof_score, weights="quadratic")
print("OOF QWK:", qwk)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2810217032.py in <cell line: 0>()
     33 # Build a cached DMatrix for the full training set once; then slice by index for folds.
     34 # XGBoost supports slicing DMatrix efficiently and it avoids repeating quantile sketching.
---> 35 dtrain_full = xgb.QuantileDMatrix(
     36     xgb.DMatrix(X_train, label=y, nthread=NTHREAD, enable_categorical=False),
     37 )

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in __init__(self, data, label, weight, base_margin, missing, silent, feature_names, feature_types, nthread, max_bin, ref, group, qid, label_lower_bound, label_upper_bound, feature_weights, enable_categorical, data_split_mode)
   1527                 )
   1528 
-> 1529         self._init(
   1530             data,
   1531             ref=ref,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in _init(self, data, ref, enable_categorical, **meta)
   1586             ctypes.byref(handle),
   1587         )
-> 1588         it.reraise()
   1589         # delay check_call to throw intermediate exception first
   1590         _check_call(ret)

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in reraise(self)
    574             exc = self._exception
    575             self._exception = None
--> 576             raise exc  # pylint: disable=raising-bad-type
    577 
    578     def __del__(self) -> None:

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in _handle_exception(self, fn, dft_ret)
    555 
    556         try:
--> 557             return fn()
    558         except Exception as e:  # pylint: disable=broad-except
    559             # Defer the exception in order to return 0 and stop the iteration.

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in <lambda>()
    639 
    640         # pylint: disable=not-callable
--> 641         return self._handle_exception(lambda: self.next(input_data), 0)
    642 
    643     @abstractmethod

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in next(self, input_data)
   1278             return 0
   1279         self.it += 1
-> 1280         input_data(**self.kwargs)
   1281         return 1
   1282 

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in input_data(data, feature_names, feature_types, **kwargs)
    622                 new, cat_codes, feature_names, feature_types = self._temporary_data
    623             else:
--> 624                 new, cat_codes, feature_names, feature_types = _proxy_transform(
    625                     data,
    626                     feature_names,

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in _proxy_transform(data, feature_names, feature_types, enable_categorical)
   1318         arr, _ = _ensure_np_dtype(arr, arr.dtype)
   1319         return arr, None, feature_names, feature_types
-> 1320     raise TypeError("Value type is not supported for data iterator:" + str(type(data)))
   1321 
   1322 

TypeError: Value type is not supported for data iterator:<class 'xgboost.core.DMatrix'>

## === cell 3
dtest = xgb.QuantileDMatrix(X_test, nthread=NTHREAD)

avg_pred = np.zeros((X_test.shape[0],), dtype=np.float32)
for m in models:
    proba = m.predict(dtest)
    avg_pred += (proba @ cls_idx).astype(np.float32)
avg_pred /= len(models)

pred_cls = np.clip(np.rint(avg_pred), 0, n_classes - 1).astype(int)
pred_score = le.inverse_transform(pred_cls).astype(int)
pred_score = np.clip(pred_score, 1, 6)

submission = pd.DataFrame({"essay_id": test_df["essay_id"].values, "score": pred_score})
print(submission.head())
print(submission.shape)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3098830262.py in <cell line: 0>()
     10 
     11 pred_cls = np.clip(np.rint(avg_pred), 0, n_classes - 1).astype(int)
---> 12 pred_score = le.inverse_transform(pred_cls).astype(int)
     13 pred_score = np.clip(pred_score, 1, 6)
     14 

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_label.py in inverse_transform(self, y)
    160         diff = np.setdiff1d(y, np.arange(len(self.classes_)))
    161         if len(diff):
--> 162             raise ValueError("y contains previously unseen labels: %s" % str(diff))
    163         y = np.asarray(y)
    164         return self.classes_[y]

ValueError: y contains previously unseen labels: [-9223372036854775808]

## === cell 4
submission = sample_sub[["essay_id"]].merge(submission, on="essay_id", how="left")

if submission["score"].isna().any():
    fill_val = int(np.median(train_df["score"].values))
    submission["score"] = submission["score"].fillna(fill_val).astype(int)

submission["score"] = submission["score"].astype(int)
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv")
print(submission.head())
print(submission["score"].value_counts().sort_index())

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/301401133.py in <cell line: 0>()
----> 1 submission = sample_sub[["essay_id"]].merge(submission, on="essay_id", how="left")
      2 
      3 if submission["score"].isna().any():
      4     fill_val = int(np.median(train_df["score"].values))
      5     submission["score"] = submission["score"].fillna(fill_val).astype(int)

NameError: name 'submission' is not defined
