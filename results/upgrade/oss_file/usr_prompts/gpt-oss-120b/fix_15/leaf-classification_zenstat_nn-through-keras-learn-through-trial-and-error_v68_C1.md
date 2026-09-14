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

os.environ["OMP_NUM_THREADS"] = (
    "1"  # limit OpenMP to a single thread for lower overhead
)

import numpy as np

np.random.seed(42)  # ensure deterministic behaviour
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss, accuracy_score




## === cell 1
from pylab import rcParams

rcParams["figure.figsize"] = 10, 10




## === cell 2
train_path = "../input/train.csv"
train_df = pd.read_csv(train_path, low_memory=False)
train_ids = train_df.pop("id")  # keep ids separate (not used for training)




## === cell 3
print("Train shape (features + label):", train_df.shape)




## === cell 4
y = train_df.pop("species")
y_enc, le = pd.factorize(
    y
)  # le contains the original class names in order of appearance




## === cell 5
X = train_df.values.astype(np.float32, copy=False)
X = np.ascontiguousarray(X)  # ensure contiguous memory layout for speed
print("Feature matrix shape:", X.shape)




## === cell 6
X_train, X_val, y_train, y_val = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)




## === cell 7
base_model = GradientBoostingClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=9,
    random_state=42,
)




## === cell 8
base_model.fit(X_train, y_train)

val_losses = []
val_probas_list = []
test_probas_list = []

val_gen = base_model.staged_predict_proba(X_val)
test_gen = base_model.staged_predict_proba(X_test)

for val_proba, test_proba in zip(val_gen, test_gen):
    loss = log_loss(y_val, val_proba)
    val_losses.append(loss)
    val_probas_list.append(val_proba)
    test_probas_list.append(test_proba)

best_iter = int(np.argmin(val_losses)) + 1
best_val_proba = val_probas_list[best_iter - 1]
test_pred_proba = test_probas_list[best_iter - 1]

print(f"Best iteration based on val log‑loss: {best_iter}")




## === cell 9
val_acc = accuracy_score(y_val, np.argmax(best_val_proba, axis=1))
val_logloss = log_loss(y_val, best_val_proba)
print(f"Validation accuracy: {val_acc:.4f}")
print(f"Validation log‑loss: {val_logloss:.6f}")




## === cell 10
test_path = "../input/test.csv"
test_df = pd.read_csv(test_path, low_memory=False)
test_ids = test_df.pop("id")  # keep ids for submission
X_test = test_df.values.astype(np.float32, copy=False)
X_test = np.ascontiguousarray(X_test)
print("Test feature matrix shape:", X_test.shape)




## === cell 11
if test_pred_proba is None:
    raise RuntimeError("Best iteration exceeds number of staged predictions.")




## === cell 12
sample_sub_path = "../input/sample_submission.csv"
sample_sub = pd.read_csv(sample_sub_path)
class_columns = list(sample_sub.columns[1:])  # ordered class names




## === cell 13
prob_df = pd.DataFrame(test_pred_proba, columns=class_columns)
prob_df = prob_df.fillna(0.0)
prob_df = prob_df.div(prob_df.sum(axis=1), axis=0)  # ensure rows sum to 1




## === cell 14
submission = pd.concat(
    [test_ids.reset_index(drop=True), prob_df.reset_index(drop_index=True)], axis=1
)




## === cell 15
submission_path = "submission_nn_kernel.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
