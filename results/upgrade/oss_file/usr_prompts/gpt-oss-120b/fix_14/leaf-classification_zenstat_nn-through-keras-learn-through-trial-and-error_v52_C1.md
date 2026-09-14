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

No external packages required in the script and installed.

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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import log_loss
import concurrent.futures

np.random.seed(42)




## === cell 1
train_path_candidates = [
    "../input/train.csv",
    "./train.csv",
    "/kaggle/input/leaf-classification/train.csv",
    "/kaggle/input/train.csv",
]
for p in train_path_candidates:
    if os.path.exists(p):
        train_path = p
        break
else:
    raise FileNotFoundError("train.csv not found in any expected location")

data = pd.read_csv(train_path)

ids = data.pop("id")
y_raw = data.pop("species")

le = LabelEncoder()
y_int = le.fit_transform(y_raw)

scaler = StandardScaler()
X = scaler.fit_transform(data.values).astype(np.float32)




## === cell 2
X_train, X_val, y_train_int, y_val_int = train_test_split(
    X,
    y_int,
    test_size=0.20,
    random_state=42,
    stratify=y_int,
)


def train_lr(X_tr, y_tr):
    model = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        solver="lbfgs",
        n_jobs=1,
        class_weight="balanced",
        verbose=0,
    )
    model.fit(X_tr, y_tr)
    return model


def train_gb(X_tr, y_tr):
    model = GradientBoostingClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        max_features=0.8,
        random_state=42,
    )
    model.fit(X_tr, y_tr)
    return model


def train_rf(X_tr, y_tr):
    model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=None,
        max_features="sqrt",
        n_jobs=-1,  # use all available cores now that we run sequentially
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_tr, y_tr)
    return model


model_lr = train_lr(X_train, y_train_int)
model_gb = train_gb(X_train, y_train_int)
model_rf = train_rf(X_train, y_train_int)




## === cell 3
val_pred_lr = model_lr.predict_proba(X_val)
val_pred_gb = model_gb.predict_proba(X_val)
val_pred_rf = model_rf.predict_proba(X_val)

val_logloss_lr = log_loss(y_val_int, val_pred_lr)
val_logloss_gb = log_loss(y_val_int, val_pred_gb)
val_logloss_rf = log_loss(y_val_int, val_pred_rf)

print(
    f"Validation log‑loss – LR: {val_logloss_lr:.5f}, GB: {val_logloss_gb:.5f}, RF: {val_logloss_rf:.5f}"
)

best_loss = np.inf
best_weights = (0.0, 0.0, 0.0)  # (w_lr, w_gb, w_rf)

weight_steps = np.arange(0.0, 1.01, 0.05)
for w_lr in weight_steps:
    for w_gb in weight_steps:
        if w_lr + w_gb > 1.0:
            continue
        w_rf = 1.0 - w_lr - w_gb
        blended = w_lr * val_pred_lr + w_gb * val_pred_gb + w_rf * val_pred_rf
        loss = log_loss(y_val_int, blended)
        if loss < best_loss:
            best_loss = loss
            best_weights = (w_lr, w_gb, w_rf)

w_lr_opt, w_gb_opt, w_rf_opt = best_weights
print(
    f"Best blend weights – LR: {w_lr_opt:.2f}, GB: {w_gb_opt:.2f}, RF: {w_rf_opt:.2f} giving log‑loss {best_loss:.5f}"
)




## === cell 4
test_path_candidates = [
    "../input/test.csv",
    "./test.csv",
    "/kaggle/input/leaf-classification/test.csv",
    "/kaggle/input/test.csv",
]
for p in test_path_candidates:
    if os.path.exists(p):
        test_path = p
        break
else:
    raise FileNotFoundError("test.csv not found in any expected location")

test_df = pd.read_csv(test_path)
test_ids = test_df.pop("id")
X_test = scaler.transform(test_df.values).astype(np.float32)

test_pred_lr = model_lr.predict_proba(X_test)
test_pred_gb = model_gb.predict_proba(X_test)
test_pred_rf = model_rf.predict_proba(X_test)

test_pred = w_lr_opt * test_pred_lr + w_gb_opt * test_pred_gb + w_rf_opt * test_pred_rf
test_pred = np.clip(test_pred, 1e-15, 1 - 1e-15)

submission = pd.DataFrame(test_pred, columns=le.classes_)
submission.insert(0, "id", test_ids.values)

submission_path = "submission_nn_kernel.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
