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

3.6

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
seaborn==0.12.2
sklearn-pandas==2.2.0
tf_keras==2.18.0

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
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(1337)
os.environ["PYTHONHASHSEED"] = "1337"
random.seed(1337)

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass



## === cell 1
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (
    train_test_split,
)  # kept for compatibility with original intent (unused)
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.decomposition import PCA



## === cell 2
from pylab import rcParams

rcParams["figure.figsize"] = 10, 10



## === cell 3
BASE_CANDIDATES = [
    "/kaggle/input/leaf-classification",
    "/kaggle/data/leaf-classification",
    "/kaggle/input",
    "/kaggle/data",
]


def first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of these paths exist: {paths}")


base = first_existing(*BASE_CANDIDATES)
train_path = first_existing(os.path.join(base, "train.csv"))
test_path = first_existing(os.path.join(base, "test.csv"))
sample_path = first_existing(os.path.join(base, "sample_submission.csv"))

train_df = pd.read_csv(train_path)
parent_data = train_df.copy()  # keep original copy
train_ids = train_df.pop("id")



## === cell 4
print("train_df shape:", train_df.shape)
print("num features:", train_df.shape[1] - 1)



## === cell 5
y = train_df.pop("species")
le = LabelEncoder()
y_enc = le.fit_transform(y)
n_classes = len(le.classes_)
print("y_enc shape:", y_enc.shape, "num_classes:", n_classes)



## === cell 6
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

X_raw = train_df.values

pca_grid = [0.95, 0.97, 0.98, 0.99]
C_grid = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1337)

best = None
best_score = np.inf

splits = list(skf.split(X_raw, y_enc))

fold_cache = []
for tr_idx, va_idx in splits:
    X_tr, X_va = X_raw[tr_idx], X_raw[va_idx]
    y_tr, y_va = y_enc[tr_idx], y_enc[va_idx]

    scaler_cv = StandardScaler()
    X_tr_s = scaler_cv.fit_transform(X_tr)
    X_va_s = scaler_cv.transform(X_va)
    fold_cache.append((X_tr_s, X_va_s, y_tr, y_va))

for pca_ret in pca_grid:
    transformed_folds = []
    for X_tr_s, X_va_s, y_tr, y_va in fold_cache:
        pca_cv = PCA(
            n_components=pca_ret,
            svd_solver="full",
            random_state=1337,
        )
        X_tr_p = pca_cv.fit_transform(X_tr_s)
        X_va_p = pca_cv.transform(X_va_s)
        transformed_folds.append((X_tr_p, X_va_p, y_tr, y_va))

    for C in C_grid:
        scores = []
        for X_tr_p, X_va_p, y_tr, y_va in transformed_folds:
            clf_cv = LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                max_iter=4000,
                C=C,
                tol=1e-5,
                class_weight=None,
                n_jobs=None,
                random_state=1337,
            )
            clf_cv.fit(X_tr_p, y_tr)
            p_va = clf_cv.predict_proba(X_va_p)
            scores.append(float(log_loss(y_va, p_va, labels=np.arange(n_classes))))
        mean_score = float(np.mean(scores))
        if mean_score < best_score:
            best_score = mean_score
            best = (pca_ret, C)

print("CV best (pca_ret, C) grid-search:", best, "mean_log_loss:", best_score)

if best is None:
    best = (0.98, 10.0)
    print("WARNING: CV did not set best params; falling back to:", best)

best_pca_ret, best_C = best



## === cell 7
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

pca = PCA(n_components=best_pca_ret, svd_solver="full", random_state=1337)
X = pca.fit_transform(X_scaled)

print("X_scaled shape:", X_scaled.shape)
print("X (after PCA) shape:", X.shape)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1337)

try:
    clf_cvC = LogisticRegressionCV(
        Cs=np.array(C_grid, dtype=float),
        cv=cv,
        scoring="neg_log_loss",
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=4000,
        tol=1e-5,
        class_weight=None,
        n_jobs=None,
        refit=True,
        random_state=1337,
    )
    clf_cvC.fit(X, y_enc)
    clf = clf_cvC
    chosen_C = float(np.median(clf_cvC.C_)) if hasattr(clf_cvC, "C_") else float(best_C)
    print("LogisticRegressionCV chosen C (median across classes):", chosen_C)
except Exception as e:
    print(
        "WARNING: LogisticRegressionCV failed; falling back to LogisticRegression. Reason:",
        repr(e),
    )
    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=4000,  # keep stable convergence
        C=best_C,
        tol=1e-5,
        class_weight=None,
        n_jobs=None,  # lbfgs ignores n_jobs; kept explicit
        random_state=1337,
    )
    clf.fit(X, y_enc)



## === cell 8
try:
    p_train = clf.predict_proba(X)
    print(
        "train log_loss:",
        float(log_loss(y_enc, p_train, labels=np.arange(n_classes))),
    )
except Exception as e:
    print("Skipping train log_loss diagnostic due to:", repr(e))



## === cell 9
history = {"val_accuracy": []}



## === cell 10
print("best val accuracy: (not available for sklearn model)")



## === cell 11
plt.figure()
plt.title("Validation Accuracy vs Epochs (not available for sklearn model)")
plt.xlabel("Number of Epochs")
plt.ylabel("Validation Accuracy")
plt.show()



## === cell 12
test_df = pd.read_csv(test_path)
test_ids = test_df.pop("id").values



## === cell 13
X_test_scaled = scaler.transform(test_df.values)
X_test = pca.transform(X_test_scaled)



## === cell 14
y_pred = clf.predict_proba(X_test)

alpha = 0.002
y_pred = (1.0 - alpha) * y_pred + alpha * (1.0 / n_classes)

eps = 1e-15
y_pred = np.clip(y_pred, eps, 1.0 - eps)

sample_sub = pd.read_csv(sample_path)
class_cols = [c for c in sample_sub.columns if c != "id"]

pred_df = pd.DataFrame(y_pred, columns=list(le.classes_))
pred_df = pred_df.reindex(columns=class_cols, fill_value=0.0)

sub = pd.concat([pd.DataFrame({"id": test_ids}), pred_df], axis=1)

for c in class_cols:
    sub[c] = sub[c].clip(0.0, 1.0)

print("submission shape:", sub.shape)
print(sub.head(2))

out_path = "submission_nn_kernel.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path)
