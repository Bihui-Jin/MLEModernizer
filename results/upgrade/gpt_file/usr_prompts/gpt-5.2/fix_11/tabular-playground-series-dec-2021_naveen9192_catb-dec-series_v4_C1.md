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

catboost==1.2.8
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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
s_data = pd.read_csv(
    "/kaggle/input/tabular-playground-series-dec-2021/sample_submission.csv"
)
s_data.head()



## === cell 2
train_data = pd.read_csv("/kaggle/input/tabular-playground-series-dec-2021/train.csv")
train_data.set_index("Id", inplace=True)
train_data.head()



## === cell 3
test_data = pd.read_csv("/kaggle/input/tabular-playground-series-dec-2021/test.csv")
test_data.set_index("Id", inplace=True)
test_data.head()



## === cell 4
train_data.describe()



## === cell 5
train_data.info()



## === cell 6
print("Shape of Train DF -", train_data.shape)
print("Shape of Test DF :", test_data.shape)
print("NA values in Train DF :", train_data.isna().sum().sum())
print("NA values in Test DF :", test_data.isna().sum().sum())



## === cell 7
target = "Cover_Type"
features = [col for col in train_data.columns if col != target]
features



## === cell 8
X = train_data[features]
y = train_data[target]

y = pd.to_numeric(y, errors="coerce")
if y.isna().any():
    nan_ct = int(y.isna().sum())
    raise ValueError(
        f"Found {nan_ct} NaNs in Cover_Type after coercion; cannot train reliably."
    )
y = y.astype(np.int64)

unique_classes = np.sort(y.unique())
if not np.array_equal(unique_classes, np.arange(1, 8)):
    raise ValueError(
        f"Unexpected classes in Cover_Type. Expected 1..7, got: {unique_classes}"
    )

y_trainable = (y - 1).to_numpy(dtype=np.int64)



## === cell 9
print(f"Shape of data X - {X.shape}, y - {y_trainable.shape} ")
print("Cover_Type value counts (head):")
vc = pd.Series(y_trainable).value_counts().sort_index()
print(vc.head(10))
print("Min class count:", int(vc.min()))
print("Trainable label range:", int(y_trainable.min()), int(y_trainable.max()))



## === cell 10
from sklearn.model_selection import StratifiedShuffleSplit

y_series = pd.Series(y_trainable)
class_counts = y_series.value_counts()
if int(class_counts.min()) < 2:
    from sklearn.model_selection import train_test_split

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y_trainable,
        random_state=42,
        test_size=0.2,
        shuffle=True,
        stratify=None,
    )
else:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, valid_idx = next(splitter.split(X, y_trainable))
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y_trainable[train_idx], y_trainable[valid_idx]

print(
    "Train class counts min/max:",
    int(pd.Series(y_train).value_counts().min()),
    int(pd.Series(y_train).value_counts().max()),
)
print(
    "Valid class counts min/max:",
    int(pd.Series(y_valid).value_counts().min()),
    int(pd.Series(y_valid).value_counts().max()),
)
print("Unique classes in train:", np.unique(y_train))
print("Unique classes in valid:", np.unique(y_valid))



## === cell 11
catb_params = {
    "loss_function": "MultiClass",
    "task_type": "CPU",
    "random_seed": 42,
    "classes_count": 7,
}



## === cell 12
from catboost import CatBoostClassifier

model = CatBoostClassifier(**catb_params)

model.fit(
    X_train,
    y_train,
    eval_set=(X_valid, y_valid),
    verbose=0,
)



## === cell 13
preds = model.predict(X_valid)
preds = np.asarray(preds).reshape(-1).astype(int)



## === cell 14
from sklearn import metrics

acc = metrics.accuracy_score(y_valid, preds)
print("Accuracy of this Model - ", acc)



## === cell 15
test_X = test_data[features]
predict = model.predict(test_X)
predict = np.asarray(predict).reshape(-1).astype(int)

predict = predict + 1
predict = np.clip(predict, 1, 7)



## === cell 16
predictions = pd.DataFrame({"Id": test_data.index.values, "Cover_Type": predict})

if list(predictions.columns) != ["Id", "Cover_Type"]:
    raise ValueError("Submission columns are incorrect.")
if len(predictions) != len(test_data):
    raise ValueError("Submission row count does not match test set.")
if predictions["Cover_Type"].isna().any():
    raise ValueError("Submission contains NaN predictions.")
if predictions["Cover_Type"].min() < 1 or predictions["Cover_Type"].max() > 7:
    raise ValueError("Submission predictions out of valid class range 1..7.")

predictions.to_csv("submission.csv", index=False)
predictions.head()
