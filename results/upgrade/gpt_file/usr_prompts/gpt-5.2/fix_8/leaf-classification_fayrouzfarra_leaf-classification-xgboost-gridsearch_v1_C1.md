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

dtrain = xgb.DMatrix(X_scaled, label=labeled_species)

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

xgb_nthread = max(1, min(8, os.cpu_count() or 1))

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

cache_dirty = False

for (
    n_estimators,
    learning_rate,
    max_depth,
    subsample,
    colsample_bytree,
    min_child_weight,
    gamma,
    reg_alpha,
    reg_lambda,
) in grid:
    key = (
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

    if key in cv_cache:
        mean_mlogloss = float(cv_cache[key])
    else:
        params = {
            "objective": "multi:softprob",
            "num_class": len(classes),
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "seed": 42,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "nthread": xgb_nthread,
            "verbosity": 0,
        }

        cv_res = xgb.cv(
            params=params,
            dtrain=dtrain,
            num_boost_round=n_estimators,
            folds=fold_indices,  # deterministic stratified folds
            shuffle=False,  # folds already fixed
            stratified=False,  # provided folds already stratified
            metrics=("mlogloss",),
            verbose_eval=False,
            seed=42,
        )

        mean_mlogloss = float(cv_res["test-mlogloss-mean"].iloc[-1])
        cv_cache[key] = mean_mlogloss
        cache_dirty = True

    if mean_mlogloss < best_score:
        best_score = mean_mlogloss
        best_params = {
            "xgb__n_estimators": n_estimators,
            "xgb__learning_rate": learning_rate,
            "xgb__max_depth": max_depth,
            "xgb__subsample": subsample,
            "xgb__colsample_bytree": colsample_bytree,
            "xgb__min_child_weight": min_child_weight,
            "xgb__gamma": gamma,
            "xgb__reg_alpha": reg_alpha,
            "xgb__reg_lambda": reg_lambda,
        }

if cache_dirty:
    np.save(cache_path, cv_cache, allow_pickle=True)


class _GSearchShim:
    def __init__(self, best_params_):
        self.best_params_ = best_params_


gsearch = _GSearchShim(best_params)
gsearch.best_params_



## === cell 11
gsearch.best_params_



## === cell 12
best_n_estimators = gsearch.best_params_.get("xgb__n_estimators")
best_n_estimators



## === cell 13
best_learning_rate = gsearch.best_params_.get("xgb__learning_rate")
best_learning_rate



## === cell 14
best_max_depth = gsearch.best_params_.get("xgb__max_depth")
best_max_depth



## === cell 15
best_subsample = gsearch.best_params_.get("xgb__subsample")
best_subsample



## === cell 16
best_colsample_bytree = gsearch.best_params_.get("xgb__colsample_bytree")
best_colsample_bytree



## === cell 17
best_min_child_weight = gsearch.best_params_.get("xgb__min_child_weight")
best_min_child_weight



## === cell 18
best_gamma = gsearch.best_params_.get("xgb__gamma")
best_gamma



## === cell 19
best_reg_alpha = gsearch.best_params_.get("xgb__reg_alpha")
best_reg_alpha



## === cell 20
best_reg_lambda = gsearch.best_params_.get("xgb__reg_lambda")
best_reg_lambda



## === cell 21
from xgboost import XGBClassifier

final_xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=len(classes),
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42,
    n_jobs=4,
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



## === cell 22
test = pd.read_csv("../input/leaf-classification/test.csv.zip", index_col="id")
test



## === cell 23
test_scaled = scaler.transform(test.values.astype(np.float32, copy=False))
pred_test = final_xgb.predict_proba(test_scaled)
pred_test.shape



## === cell 24
sample_sub = pd.read_csv("../input/leaf-classification/sample_submission.csv.zip")

est_classes = list(final_xgb.classes_)
proba_class_names = [classes[i] for i in est_classes]

submission = sample_sub.copy()
submission.iloc[:, 1:] = 0.0  # fill all class columns with 0 first
submission["id"] = test.index.values  # test ids

submission.loc[:, proba_class_names] = pred_test

proba_cols = [c for c in submission.columns if c != "id"]
submission[proba_cols] = submission[proba_cols].clip(0.0, 1.0)

submission.to_csv("submission.csv", index=False)
print("done")
submission
