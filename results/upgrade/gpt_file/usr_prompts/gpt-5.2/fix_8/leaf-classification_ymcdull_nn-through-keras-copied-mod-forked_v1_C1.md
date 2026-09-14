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
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder

np.random.seed(42)

DATA_DIR = "/kaggle/input/leaf-classification"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

print("Using paths:")
print(TRAIN_PATH)
print(TEST_PATH)
print(SAMPLE_SUB_PATH)



## === cell 1
from sklearn.linear_model import LogisticRegressionCV



## === cell 2
from pylab import rcParams

rcParams["figure.figsize"] = 10, 10



## === cell 3
data = pd.read_csv(TRAIN_PATH)
parent_data = data.copy()  # keep original
ID = data.pop("id")

print("Train shape:", parent_data.shape)



## === cell 4
data.shape



## === cell 5
y = data.pop("species")
le = LabelEncoder()
y = le.fit_transform(y)
print("y shape:", y.shape, "num_classes:", len(le.classes_))



## === cell 6
X_all = data.values.astype(np.float32)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_all)

print("X_train shape:", X_train.shape)



## === cell 7
model = LogisticRegressionCV(
    Cs=np.logspace(-2, 2, 9),  # keep same sweep: [0.01 ... 100]
    cv=5,
    scoring="neg_log_loss",
    multi_class="multinomial",
    solver="lbfgs",
    penalty="l2",
    max_iter=4000,
    n_jobs=1,
    verbose=0,
    refit=True,
    random_state=42,
)
print(model)



## === cell 8
model.fit(X_train, y)

scores = model.scores_[model.classes_[0]]  # shape: (n_folds, n_Cs)
mean = scores.mean(axis=0)
std = scores.std(axis=0, ddof=1)
best_idx = int(np.argmax(mean))
threshold = (
    mean[best_idx] - std[best_idx]
)  # within 1 std of the best (higher neg_log_loss is better)

Cs_grid = np.asarray(model.Cs_, dtype=float)
candidate_idxs = np.where(mean >= threshold)[0]
chosen_idx = int(candidate_idxs[0]) if len(candidate_idxs) else best_idx
chosen_C = float(Cs_grid[chosen_idx])

final_model = LogisticRegressionCV(
    Cs=[chosen_C],
    cv=5,
    scoring="neg_log_loss",
    multi_class="multinomial",
    solver="lbfgs",
    penalty="l2",
    max_iter=4000,
    n_jobs=1,
    verbose=0,
    refit=True,
    random_state=42,
)
final_model.fit(X_train, y)

print("Model trained. Classes:", len(final_model.classes_))
print("Best C selected by CV (original):", float(np.atleast_1d(model.C_)[0]))
print("Chosen C by 1-SE rule (refit):", chosen_C)



## === cell 9
train_acc = float(final_model.score(X_train, y))
print("Training accuracy (full train):", train_acc)



## === cell 10
history = None
print("Keras history not available (using scikit-learn model).")



## === cell 11
plt.figure()
plt.title("No training curve (scikit-learn model)")
plt.axis("off")
plt.show()



## === cell 12
test_df = pd.read_csv(TEST_PATH)
index = test_df.pop("id").values

X_test = scaler.transform(test_df.values.astype(np.float32))
print("Test shape:", X_test.shape)



## === cell 13
yPred = final_model.predict_proba(X_test).astype(np.float64)
print("Pred shape:", yPred.shape, "min/max:", float(yPred.min()), float(yPred.max()))

eps = 1e-15
yPred = np.clip(yPred, eps, 1.0 - eps)

alpha = 0.002  # small uniform mixing weight; intended to improve log-loss vs overconfident predictions
K = yPred.shape[1]
yPred = (1.0 - alpha) * yPred + alpha * (1.0 / K)

yPred = np.clip(yPred, eps, 1.0 - eps)
yPred = yPred / yPred.sum(axis=1, keepdims=True)

print("Post-processed pred min/max:", float(yPred.min()), float(yPred.max()))
print("Row sums (first 5):", yPred[:5].sum(axis=1))



## === cell 14
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
class_cols = [c for c in sample_sub.columns if c != "id"]

pred_aligned = np.zeros((yPred.shape[0], len(class_cols)), dtype=np.float64)
missing = []

train_class_names = list(le.classes_)  # index i corresponds to column in yPred
train_name_to_idx = {name: i for i, name in enumerate(train_class_names)}

for j, cls in enumerate(class_cols):
    if cls in train_name_to_idx:
        pred_aligned[:, j] = yPred[:, train_name_to_idx[cls]]
    else:
        missing.append(cls)

if missing:
    print(
        "Warning: classes present in submission but missing from training encoder:",
        missing,
    )

pred_aligned = np.clip(pred_aligned, eps, 1.0 - eps)
pred_aligned = pred_aligned / pred_aligned.sum(axis=1, keepdims=True)

sub = pd.DataFrame(pred_aligned.astype(np.float32), columns=class_cols)
sub.insert(0, "id", index)

print(sub.head())
print("Submission shape:", sub.shape)



## === cell 15
out_path = "submission_nn_kernel.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path)
print("Columns:", len(sub.columns), "First columns:", list(sub.columns[:5]))
