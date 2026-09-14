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
np.random.seed(42)



## === cell 1
df = pd.read_csv("../input/leaf-classification/train.csv.zip", index_col="id")



## === cell 2
na_frac = df.isna().mean(axis=0)
na_cols = na_frac[na_frac > 0]
if len(na_cols):
    for col, frac in na_cols.items():
        print(col, float(frac))



## === cell 3
_ = df["species"].nunique()



## === cell 4
y = df["species"]



## === cell 5
X = df.drop(columns="species", axis=1)



## === cell 6
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder().fit(y)
labeled_species = label_encoder.transform(y)



## === cell 7
classes = list(label_encoder.classes_)



## === cell 8
from sklearn.preprocessing import StandardScaler

parameters = {
    "clf__n_estimators": list(range(100, 201, 100)),
    "clf__learning_rate": [l / 100 for l in range(5, 15, 10)],
    "clf__max_depth": list(range(6, 16, 10)),
    "clf__subsample": [0.8, 1.0],
    "clf__min_child_weight": [1, 5],
    "clf__colsample_bytree": [0.8, 1.0],
    "clf__reg_lambda": [1.0, 3.0],
    "clf__reg_alpha": [0.0, 0.1],
}



## === cell 9
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

test_for_scaler = pd.read_csv(
    "../input/leaf-classification/test.csv.zip", index_col="id"
)
X_all = pd.concat([X, test_for_scaler], axis=0)

scaler = StandardScaler(with_mean=True, with_std=True)
X_all_np = X_all.to_numpy(dtype=np.float32, copy=False)
X_scaled = scaler.fit_transform(X_all_np)[: len(X)]
y_arr = np.asarray(labeled_species, dtype=np.int32)

dtrain = xgb.DMatrix(X_scaled, label=y_arr)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(X_scaled, y_arr))

try:
    import os as _os

    _nthread = max(1, len(_os.sched_getaffinity(0)))
except Exception:
    _nthread = 4

base_params = dict(
    objective="multi:softprob",
    eval_metric="mlogloss",
    num_class=len(classes),
    tree_method="hist",
    seed=42,
    nthread=_nthread,
)

grid_params = {
    "num_boost_round": [int(v) for v in parameters["clf__n_estimators"]],
    "eta": [float(v) for v in parameters["clf__learning_rate"]],
    "max_depth": [int(v) for v in parameters["clf__max_depth"]],
    "subsample": [float(v) for v in parameters["clf__subsample"]],
    "min_child_weight": [float(v) for v in parameters["clf__min_child_weight"]],
    "colsample_bytree": [float(v) for v in parameters["clf__colsample_bytree"]],
    "reg_lambda": [float(v) for v in parameters["clf__reg_lambda"]],
    "reg_alpha": [float(v) for v in parameters["clf__reg_alpha"]],
}

from itertools import product

keys = list(grid_params.keys())
values = [grid_params[k] for k in keys]

best_mlogloss = float("inf")
best_combo = None

for combo in product(*values):
    params = dict(base_params)
    num_boost_round = None
    for k, v in zip(keys, combo):
        if k == "num_boost_round":
            num_boost_round = int(v)
        else:
            params[k] = v

    cv_res = xgb.cv(
        params=params,
        dtrain=dtrain,
        folds=folds,
        metrics=("mlogloss",),
        stratified=False,  # we already provide stratified folds
        seed=42,
        shuffle=False,
        as_pandas=True,
        verbose_eval=False,
        num_boost_round=num_boost_round,
    )

    mlogloss = float(cv_res["test-mlogloss-mean"].iloc[-1])
    if mlogloss < best_mlogloss:
        best_mlogloss = mlogloss
        best_combo = dict(zip(keys, combo))

best_params_ = {
    "n_estimators": int(best_combo["num_boost_round"]),
    "learning_rate": float(best_combo["eta"]),
    "max_depth": int(best_combo["max_depth"]),
    "subsample": float(best_combo["subsample"]),
    "min_child_weight": float(best_combo["min_child_weight"]),
    "colsample_bytree": float(best_combo["colsample_bytree"]),
    "reg_lambda": float(best_combo["reg_lambda"]),
    "reg_alpha": float(best_combo["reg_alpha"]),
}

best_score = -float(best_mlogloss)  # neg_log_loss (higher is better)


class _GSearchLike:
    def __init__(self, best_params_, best_score_):
        self.best_params_ = best_params_
        self.best_score_ = best_score_


gsearch = _GSearchLike(best_params_=best_params_, best_score_=best_score)

print(f"[grid_search] best neg_log_loss={gsearch.best_score_:.6f}")


## === cell 10
_ = gsearch.best_params_
_ is not None



## === cell 11
best_n_estimators = gsearch.best_params_.get("n_estimators")
best_n_estimators



## === cell 12
best_learning_rate = gsearch.best_params_.get("learning_rate")
best_learning_rate



## === cell 13
best_max_depth = gsearch.best_params_.get("max_depth")
best_max_depth



## === cell 14
best_subsample = gsearch.best_params_.get("subsample")
best_subsample



## === cell 15
best_min_child_weight = gsearch.best_params_.get("min_child_weight")
best_min_child_weight



## === cell 16
best_colsample_bytree = gsearch.best_params_.get("colsample_bytree")
best_colsample_bytree



## === cell 17
best_reg_lambda = gsearch.best_params_.get("reg_lambda")
best_reg_lambda



## === cell 18
best_reg_alpha = gsearch.best_params_.get("reg_alpha")
best_reg_alpha



## === cell 19
final_params = dict(
    objective="multi:softprob",
    eval_metric="mlogloss",
    num_class=len(classes),
    tree_method="hist",
    seed=42,
    nthread=_nthread,
    eta=float(best_learning_rate),
    max_depth=int(best_max_depth),
    subsample=float(best_subsample),
    min_child_weight=float(best_min_child_weight),
    colsample_bytree=float(best_colsample_bytree),
    reg_lambda=float(best_reg_lambda),
    reg_alpha=float(best_reg_alpha),
)

final_booster = xgb.train(
    params=final_params,
    dtrain=dtrain,
    num_boost_round=int(best_n_estimators),
)



## === cell 20
test = test_for_scaler
test_np = test.to_numpy(dtype=np.float32, copy=False)
test_scaled = scaler.transform(test_np)
dtest = xgb.DMatrix(test_scaled)

pred_test = final_booster.predict(dtest)

eps = 1e-15
pred_test = np.clip(pred_test, eps, 1.0 - eps)
pred_test = pred_test / pred_test.sum(axis=1, keepdims=True)



## === cell 21
_ = pred_test.shape



## === cell 22
sample_sub = pd.read_csv("../input/leaf-classification/sample_submission.csv.zip")
submission_cols = [c for c in sample_sub.columns if c != "id"]

output = pd.DataFrame(pred_test, columns=classes)
output.insert(0, "id", test.index)

output = output.reindex(columns=["id"] + submission_cols, fill_value=0.0)

output.to_csv("submission.csv", index=False)
print("done")
