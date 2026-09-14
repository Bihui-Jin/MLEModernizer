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

3.5

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

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

try:
    from sklearnex import patch_sklearn  # scikit-learn-intelex

    patch_sklearn()
except Exception:
    pass

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingGridSearchCV

np.random.seed(42)

DATA_DIR = "/kaggle/input/leaf-classification"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

X = train.drop(["id", "species"], axis=1).to_numpy(dtype=np.float64, copy=False)
le = LabelEncoder().fit(train["species"])
y = le.transform(train["species"].to_numpy())

test_ids = test["id"].to_numpy()
X_test = test.drop(["id"], axis=1).to_numpy(dtype=np.float64, copy=False)

n_jobs = int(os.environ.get("SKLEARN_N_JOBS", "-1"))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scaler = StandardScaler(copy=True)
X_s = np.ascontiguousarray(scaler.fit_transform(X), dtype=np.float64)
X_test_s = np.ascontiguousarray(scaler.transform(X_test), dtype=np.float64)

base = LogisticRegression(
    max_iter=4000,  # halving uses this as the resource to allocate
    n_jobs=1,  # prevent per-fit oversubscription
    multi_class="multinomial",
    random_state=42,  # saga is stochastic; keep fixed for determinism
    warm_start=True,
)

C_lbfgs = [
    0.03,
    0.05,
    0.08,
    0.1,
    0.15,
    0.2,
    0.3,
    0.5,
    0.8,
    1.0,
    1.5,
    2.0,
    3.0,
    5.0,
    8.0,
    10.0,
    20.0,
    50.0,
    100.0,
]
C_saga = [
    0.03,
    0.05,
    0.08,
    0.1,
    0.15,
    0.2,
    0.3,
    0.5,
    0.8,
    1.0,
    1.5,
    2.0,
    3.0,
    5.0,
    8.0,
    10.0,
]

params_lbfgs = {
    "solver": ["lbfgs"],
    "penalty": ["l2"],
    "C": C_lbfgs,
    "tol": [0.001, 0.0005, 0.0001],
    "class_weight": [None, "balanced"],
}

params_saga = {
    "solver": ["saga"],
    "penalty": ["elasticnet"],
    "C": C_saga,
    "l1_ratio": [0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95],
    "tol": [0.001, 0.0005, 0.0001],
    "class_weight": [None, "balanced"],
}

param_grid = [params_lbfgs, params_saga]

gs = HalvingGridSearchCV(
    estimator=base,
    param_grid=param_grid,
    scoring="neg_log_loss",
    cv=cv,
    refit=True,
    n_jobs=n_jobs,
    verbose=0,
    factor=3,
    resource="max_iter",
    max_resources=4000,
    min_resources=200,
    return_train_score=False,
)

gs.fit(X_s, y)

best_solver = gs.best_params_.get("solver", "unknown")
chosen = "saga_elasticnet" if best_solver == "saga" else "lbfgs_l2"

print("chosen search:", chosen)
print("best params:", gs.best_params_)
print("best CV score (neg_log_loss):", gs.best_score_)

y_test = gs.predict_proba(X_test_s)

submission = pd.DataFrame(y_test, columns=le.classes_)
submission.insert(0, "id", test_ids)

submission = submission.reindex(columns=sample_sub.columns).fillna(1e-15)

prob_cols = submission.columns[1:]
submission[prob_cols] = submission[prob_cols].clip(1e-15, 1 - 1e-15)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)



## === cell 1
print(train.shape, test.shape)
print("Train species classes:", len(le.classes_))
train.describe(include="all")
