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

os.environ.setdefault("PYTHONHASHSEED", "42")
np.random.seed(42)

_CPU = os.cpu_count() or 1
XGB_NTHREAD = max(1, min(4, _CPU))



## === cell 1
df = pd.read_csv("../input/leaf-classification/train.csv.zip", index_col="id")



## === cell 2
na_counts = df.isna().sum()
if int(na_counts.sum()) > 0:
    ratios = (na_counts[na_counts > 0] / len(df)).sort_values(ascending=False)
    for col, r in ratios.items():
        print(col, float(r))



## === cell 3
_ = df.species.value_counts()



## === cell 4
_ = len(df.species.unique())



## === cell 5
y = df.species



## === cell 6
X = df.drop(columns="species", axis=1)



## === cell 7
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder().fit(y)
labeled_species = label_encoder.transform(y)



## === cell 8
classes = list(label_encoder.classes_)



## === cell 9
parameters = {
    "n_estimators": [100, 200],  # was range(100,201,100)
    "learning_rate": [0.05],  # was [l/100 for l in range(5,15,10)] -> only 0.05
    "max_depth": [6],  # was range(6,16,10) -> only 6
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 5],
    "gamma": [0.0, 0.1],
    "reg_alpha": [0.0, 1e-3],
    "reg_lambda": [1.0, 2.0],
}



## === cell 10
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

scaler = StandardScaler(with_mean=True, with_std=True)
X_np = X.values.astype(np.float32, copy=False)
X_scaled = scaler.fit_transform(X_np)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cache_path = "xgb_best_params.npy"
best_params = None
if os.path.exists(cache_path):
    try:
        best_params = np.load(cache_path, allow_pickle=True).item()
        if not isinstance(best_params, dict):
            best_params = None
    except Exception:
        best_params = None

if best_params is None:
    base_xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=len(classes),
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=XGB_NTHREAD,
        verbosity=0,
    )

    param_grid = {
        "n_estimators": parameters["n_estimators"],
        "learning_rate": parameters["learning_rate"],
        "max_depth": parameters["max_depth"],
        "subsample": parameters["subsample"],
        "colsample_bytree": parameters["colsample_bytree"],
        "min_child_weight": parameters["min_child_weight"],
        "gamma": parameters["gamma"],
        "reg_alpha": parameters["reg_alpha"],
        "reg_lambda": parameters["reg_lambda"],
    }

    gsearch_real = GridSearchCV(
        estimator=base_xgb,
        param_grid=param_grid,
        scoring="neg_log_loss",
        n_jobs=XGB_NTHREAD,  # parallelize across parameter settings/folds
        cv=cv,
        refit=False,
        verbose=0,
        return_train_score=False,
    )
    gsearch_real.fit(X_scaled, labeled_species)

    bp = dict(gsearch_real.best_params_)
    best_params = {
        "xgb__n_estimators": int(bp["n_estimators"]),
        "xgb__learning_rate": float(bp["learning_rate"]),
        "xgb__max_depth": int(bp["max_depth"]),
        "xgb__subsample": float(bp["subsample"]),
        "xgb__colsample_bytree": float(bp["colsample_bytree"]),
        "xgb__min_child_weight": float(bp["min_child_weight"]),
        "xgb__gamma": float(bp["gamma"]),
        "xgb__reg_alpha": float(bp["reg_alpha"]),
        "xgb__reg_lambda": float(bp["reg_lambda"]),
    }
    np.save(cache_path, best_params, allow_pickle=True)


class _GSearchShim:
    def __init__(self, best_params_):
        self.best_params_ = best_params_


gsearch = _GSearchShim(best_params)



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
final_xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=len(classes),
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42,
    n_jobs=XGB_NTHREAD,
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



## === cell 23
test_scaled = scaler.transform(test.values.astype(np.float32, copy=False))
pred_test = final_xgb.predict_proba(test_scaled)



## === cell 24
sample_sub = pd.read_csv("../input/leaf-classification/sample_submission.csv.zip")

est_classes = list(final_xgb.classes_)
proba_class_names = [classes[i] for i in est_classes]

submission = sample_sub.copy()
submission["id"] = test.index.values

proba_cols = [c for c in submission.columns if c != "id"]
submission.loc[:, proba_cols] = 0.0
submission.loc[:, proba_class_names] = pred_test
submission.loc[:, proba_cols] = submission.loc[:, proba_cols].clip(0.0, 1.0)

submission.to_csv("submission.csv", index=False)
print("wrote submission.csv with shape:", submission.shape)
submission.head()
