# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

np.random.seed(0)



## === cell 1
rename_map = {
    f"{prefix}{i}": f"{prefix}_{i}"
    for prefix in ("margin", "shape", "texture")
    for i in range(1, 65)
}

df = pd.read_csv("../input/leaf-classification/train.csv.zip", index_col="id")
df = df.rename(columns=rename_map)



## === cell 2
from sklearnex import patch_sklearn

patch_sklearn()

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder().fit(df["species"])
classes = list(label_encoder.classes_)

y = label_encoder.transform(df["species"])
X = df.drop("species", axis=1)



## === cell 3
parameters = {
    "n_estimators": [400, 700],
    "learning_rate": [0.03, 0.05],
    "max_depth": [4, 6],
    "min_child_weight": [1, 5],
    "subsample": [0.7, 1.0],
    "colsample_bytree": [0.7, 1.0],
}



## === cell 4
my_randome_state = 1384



## === cell 5
X_np = np.ascontiguousarray(X.to_numpy(dtype=np.float32, copy=False))
y_np = np.ascontiguousarray(y, dtype=np.int32)



## === cell 6
import hashlib
import joblib
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

cpu_count = os.cpu_count() or 2
xgb_threads = max(1, min(cpu_count, 8))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=my_randome_state)
folds = list(skf.split(X_np, y_np))

dtrain = xgb.DMatrix(X_np, label=y_np)

param_grid = []
for n_estimators in parameters["n_estimators"]:
    for learning_rate in parameters["learning_rate"]:
        for max_depth in parameters["max_depth"]:
            for min_child_weight in parameters["min_child_weight"]:
                for subsample in parameters["subsample"]:
                    for colsample_bytree in parameters["colsample_bytree"]:
                        param_grid.append(
                            {
                                "n_estimators": n_estimators,
                                "learning_rate": learning_rate,
                                "max_depth": max_depth,
                                "min_child_weight": min_child_weight,
                                "subsample": subsample,
                                "colsample_bytree": colsample_bytree,
                            }
                        )

data_sig = hashlib.md5(
    (
        f"Xshape={X_np.shape}|Xcols={tuple(X.columns)}|classes={tuple(classes)}|seed={my_randome_state}|"
        f"folds={str([(tuple(tr[:5]), tuple(te[:5]), len(tr), len(te)) for tr, te in folds])}"
    ).encode("utf-8")
).hexdigest()

param_sig = hashlib.md5(
    (
        str(param_grid)
        + "|xgb_cv|tree_method=hist|metric=mlogloss|objective=multi:softprob"
        + f"|num_class={len(classes)}|nthread={xgb_threads}"
    ).encode("utf-8")
).hexdigest()

cache_path = os.path.join(".", f"xgbcv_grid_cache_{data_sig}_{param_sig}.joblib")

if os.path.exists(cache_path):
    grid_result = joblib.load(cache_path)
    best_params = grid_result["best_params"]
else:
    best_score = float("inf")  # minimize mlogloss
    best_params = None

    base_params = {
        "objective": "multi:softprob",
        "num_class": len(classes),
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "seed": my_randome_state,
        "verbosity": 0,
        "nthread": xgb_threads,
    }

    for cand in param_grid:
        params = dict(base_params)
        params.update(
            {
                "eta": cand["learning_rate"],
                "max_depth": cand["max_depth"],
                "min_child_weight": cand["min_child_weight"],
                "subsample": cand["subsample"],
                "colsample_bytree": cand["colsample_bytree"],
            }
        )

        cv_hist = xgb.cv(
            params=params,
            dtrain=dtrain,
            num_boost_round=int(cand["n_estimators"]),
            folds=folds,
            metrics=("mlogloss",),
            seed=my_randome_state,
            shuffle=False,  # folds are precomputed; keep deterministic
            stratified=False,  # already stratified by construction
            verbose_eval=False,
        )

        mean_loss = float(cv_hist["test-mlogloss-mean"].iloc[-1])
        if mean_loss < best_score:
            best_score = mean_loss
            best_params = cand

    joblib.dump(
        {"best_params": best_params, "best_score": best_score}, cache_path, compress=3
    )



## === cell 7
best_n_estimators = best_params.get("n_estimators")
best_learning_rate = best_params.get("learning_rate")
best_max_depth = best_params.get("max_depth")
best_min_child_weight = best_params.get("min_child_weight")
best_subsample = best_params.get("subsample")
best_colsample_bytree = best_params.get("colsample_bytree")



## === cell 8
from xgboost import XGBClassifier

final_jobs = max(1, min(cpu_count, 8))

final_model = XGBClassifier(
    n_estimators=best_n_estimators,
    learning_rate=best_learning_rate,
    max_depth=best_max_depth,
    min_child_weight=best_min_child_weight,
    subsample=best_subsample,
    colsample_bytree=best_colsample_bytree,
    random_state=my_randome_state,
    objective="multi:softprob",
    num_class=len(classes),
    eval_metric="mlogloss",
    tree_method="hist",
    n_jobs=final_jobs,
    use_label_encoder=False,
    max_cached_hist_node=2**18,
)
final_model.fit(X_np, y_np)



## === cell 9
test = pd.read_csv("../input/leaf-classification/test.csv.zip", index_col="id")
test = test.rename(columns=rename_map)



## === cell 10
test_np = np.ascontiguousarray(test.to_numpy(dtype=np.float32, copy=False))
preds_test = final_model.predict_proba(test_np)



## === cell 11
sample_sub = pd.read_csv("../input/leaf-classification/sample_submission.csv.zip")
sub_cols = list(sample_sub.columns)
assert sub_cols[0] == "id"
expected_classes = sub_cols[1:]

submission = pd.DataFrame(preds_test, columns=classes)
submission.insert(0, "id", test.index)

for c in expected_classes:
    if c not in submission.columns:
        submission[c] = 0.0

submission = submission[["id"] + expected_classes]
submission[expected_classes] = submission[expected_classes].clip(0.0, 1.0)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
