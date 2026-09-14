# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Predict the class of a given image from a synthetic dataset.

## MetricMulti-class classification accuracy.

## Submission FormatFor each `Id` in the test set, you must predict the `Cover_Type` class. The file should contain a header and have the following format:
```
Id,Cover_Type
4000000,2
4000001,1
4000001,3
etc.
```

## Dataset 
- train.csv - the training data with the target `Cover_Type` column
- test.csv - the test set; you will be predicting the `Cover_Type` for each row in this file (the target integer class)
- sample_submission.csv - a sample submission file in the correct format

# 2. Python version

3.10

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
            description.md (59 lines)
            sample_submission.csv (400001 lines)
            sample_submission.csv.zip (1.6 MB)
            test.csv (400001 lines)
            test.csv.zip (10.7 MB)
            train.csv (3600001 lines)
            train.csv.zip (97.9 MB)
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
        input/
            description.md (59 lines)
            sample_submission.csv (400001 lines)
            sample_submission.csv.zip (1.6 MB)
            test.csv (400001 lines)
            test.csv.zip (10.7 MB)
            train.csv (3600001 lines)
            train.csv.zip (97.9 MB)
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
        working/
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
```

-> data/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> data/tabular-playground-series-dec-2021/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> data/tabular-playground-series-dec-2021/test.csv has 400000 rows and 55 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 40 more columns

-> data/tabular-playground-series-dec-2021/train.csv has 3600000 rows and 56 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 41 more columns

-> data/test.csv has 400000 rows and 55 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 40 more columns

-> data/train.csv has 3600000 rows and 56 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 41 more columns

-> input/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O

np.random.seed(42)

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier




## === cell 2
import warnings

warnings.filterwarnings("ignore")




## === cell 3
train = pd.read_csv("../input/tabular-playground-series-dec-2021/train.csv")
test = pd.read_csv("../input/tabular-playground-series-dec-2021/test.csv")

x_data = train.drop(["Id", "Cover_Type"], axis=1)
x_test = test.drop("Id", axis=1)
y_data = train["Cover_Type"]  # original class labels (1‑7)




## === cell 4
for df in (x_data, x_test):
    df["mean"] = df.mean(axis=1)
    df["std"] = df.std(axis=1)
    df["max"] = df.max(axis=1)
    df["min"] = df.min(axis=1)
    df["f1"] = df.iloc[:, 0] * df.iloc[:, 1] / (df.iloc[:, 2] + 1)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)




## === cell 5
le = LabelEncoder()
y_encoded = le.fit_transform(y_data)  # now 0‑6

x_train, x_val, y_train_enc, y_val_enc = train_test_split(
    x_data,
    y_encoded,
    test_size=0.2,
    random_state=42,
    shuffle=True,
)

train_classes = set(y_train_enc)
all_classes = set(range(len(le.classes_)))  # expected 0‑6
missing_in_train = all_classes - train_classes
if missing_in_train:
    dummy_X = pd.DataFrame(
        np.zeros((len(missing_in_train), x_train.shape[1])),
        columns=x_train.columns,
    )
    dummy_y = np.array(list(missing_in_train))
    x_train = pd.concat([x_train, dummy_X], ignore_index=True)
    y_train_enc = np.concatenate([y_train_enc, dummy_y])

y_val_original = le.inverse_transform(y_val_enc)

x_train_np = x_train.astype(np.float32).values
x_val_np = x_val.astype(np.float32).values
x_test_np = x_test.astype(np.float32).values




## === cell 6
model = XGBClassifier(
    objective="multi:softprob",
    tree_method="hist",
    n_estimators=500,  # reduced from 1000
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=42,
    num_class=len(le.classes_),  # dynamic number of classes (7)
    n_jobs=-1,  # use all CPU cores
)

model.fit(
    x_train_np,
    y_train_enc,
    eval_set=[(x_val_np, y_val_enc)],
    early_stopping_rounds=30,
    verbose=False,
)

y_val_pred_enc = model.predict(x_val_np)
y_val_pred = le.inverse_transform(y_val_pred_enc)
acc = accuracy_score(y_val_original, y_val_pred)
print(f"validation acc {acc:.5f}")




## === cell 7
y_test_pred_enc = model.predict(x_test_np)
y_test_pred = le.inverse_transform(y_test_pred_enc)




## === cell 8
submission = pd.read_csv(
    "../input/tabular-playground-series-dec-2021/sample_submission.csv"
)
submission["Cover_Type"] = y_test_pred
submission.to_csv("submission.csv", index=False)
print("Submission written to submission.csv")
