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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



## === cell 1
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split



## === cell 2
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV



## === cell 3
from pylab import rcParams

rcParams["figure.figsize"] = 10, 10



## === cell 4
BASE_CANDIDATES = [
    "/kaggle/input/leaf-classification",
    "/kaggle/input/leaf-classification/leaf-classification",
]


def _pick_path(filename):
    for base in BASE_CANDIDATES:
        p = os.path.join(base, filename)
        if os.path.exists(p):
            return p
    return os.path.join("/kaggle/input/leaf-classification", filename)


TRAIN_PATH = _pick_path("train.csv")
TEST_PATH = _pick_path("test.csv")
SAMPLE_SUB_PATH = _pick_path("sample_submission.csv")

data = pd.read_csv(TRAIN_PATH)
parent_data = data.copy()  # keep original labels for class name ordering (unchanged)
ID = data.pop("id")



## === cell 5
data.shape



## === cell 6
y_raw = data.pop("species")
le = LabelEncoder()
y = le.fit_transform(y_raw)
print(y.shape)



## === cell 7
scaler = StandardScaler()
X = scaler.fit_transform(data.values)
print(X.shape)



## === cell 8
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_model = LogisticRegressionCV(
    Cs=np.array([0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]),
    cv=cv,
    scoring="neg_log_loss",
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=4000,
    n_jobs=None,
    refit=True,
    verbose=0,
    fit_intercept=True,  # explicit default (stability/consistency)
    class_weight=None,  # explicit default (stability/consistency)
)
cv_model.fit(X, y)

best_C = float(np.atleast_1d(cv_model.C_)[0])
print("Selected C (CV, neg_log_loss):", best_C)

from sklearn.calibration import CalibratedClassifierCV

cal_model = CalibratedClassifierCV(
    estimator=cv_model,
    method="isotonic",
    cv=cv,  # cross-validated calibration to avoid overfitting
)
cal_model.fit(X, y)

model = cal_model



## === cell 9
pass



## === cell 10
n_classes = int(np.unique(y).shape[0])
n_samples = int(X.shape[0])

min_test_frac = (n_classes / n_samples) + 1e-9
test_size = max(0.1, min_test_frac)

X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

_diag_model = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=4000,
    n_jobs=None,
    verbose=0,
    C=best_C,
    fit_intercept=True,  # explicit default (consistency)
    class_weight=None,  # explicit default (consistency)
)
_diag_model.fit(X_tr, y_tr)
val_acc = _diag_model.score(X_va, y_va)
print("Validation accuracy (diagnostic only):", val_acc, "| test_size:", test_size)



## === cell 11
val_acc



## === cell 12
plt.plot([val_acc], "o-")
plt.xlabel("Number of Iterations")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy (diagnostic)")
plt.show()



## === cell 13
test = pd.read_csv(TEST_PATH)



## === cell 14
index = test.pop("id")

test_scaled = scaler.transform(test.values)

yPred = model.predict_proba(test_scaled)

eps = 1e-15
yPred = np.clip(yPred, eps, 1.0 - eps)

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
class_cols = [c for c in sample_sub.columns if c != "id"]

pred_df = pd.DataFrame(yPred, columns=le.classes_)
pred_df["id"] = index.values

pred_df = (
    pred_df.set_index("id").reindex(columns=class_cols, fill_value=eps).reset_index()
)

pred_df[class_cols] = pred_df[class_cols].clip(eps, 1.0 - eps)
row_sum = pred_df[class_cols].sum(axis=1).replace(0.0, 1.0)
pred_df[class_cols] = pred_df[class_cols].div(row_sum, axis=0).clip(eps, 1.0 - eps)

SUB_PATH = "submission_nn_kernel.csv"
pred_df.to_csv(SUB_PATH, index=False)
print("Wrote submission:", SUB_PATH)
print(pred_df.head())
