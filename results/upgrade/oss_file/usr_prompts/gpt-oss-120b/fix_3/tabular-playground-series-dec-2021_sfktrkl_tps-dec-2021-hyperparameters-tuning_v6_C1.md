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
scipy==1.15.3
seaborn==0.12.2
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
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from scipy.stats import mode

from xgboost import XGBClassifier




## === cell 1
train = pd.read_csv("../input/tabular-playground-series-dec-2021/train.csv")
test = pd.read_csv("../input/tabular-playground-series-dec-2021/test.csv")




## === cell 2
def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


train = reduce_mem_usage(train)
test = reduce_mem_usage(test)




## === cell 3
train.head()




## === cell 4
train.describe()




## === cell 5
print("Columns: \n{0}".format(list(train.columns)))




## === cell 6
print("Train data shape:", train.shape)
print("Test data shape:", test.shape)




## === cell 7
missing_values_train = train.isna().any().sum()
print("Missing values in train data: {0}".format(missing_values_train))

missing_values_test = test.isna().any().sum()
print("Missing values in test data: {0}".format(missing_values_test))




## === cell 8
duplicates_train = train.duplicated().sum()
print("Duplicates in train data: {0}".format(duplicates_train))

duplicates_test = test.duplicated().sum()
print("Duplicates in test data: {0}".format(duplicates_test))




## === cell 9
categorical_features = train.columns[11:-1]
print("Categorical Columns: \n{0}".format(list(categorical_features)))




## === cell 10
numerical_features = train.columns[1:11]
print("Numerical Columns: \n{0}".format(list(numerical_features)))
train[numerical_features].describe()




## === cell 11
plt.figure(figsize=(10, 6))
plt.title("Target distribution")
sns.countplot(x=train["Cover_Type"], data=train)




## === cell 12
cType5 = train[train["Cover_Type"] == 5].index
print("Number of rows with Cover_Type = 5: {0}".format(len(cType5)))




## === cell 13
print(
    "Unique values in Soil_Type7 column train data: {0}".format(
        train["Soil_Type7"].unique()
    )
)
print(
    "Unique values in Soil_Type15 column train data: {0}".format(
        train["Soil_Type15"].unique()
    )
)
print(
    "Unique values in Soil_Type7 column test data: {0}".format(
        test["Soil_Type7"].unique()
    )
)
print(
    "Unique values in Soil_Type15 column test data: {0}".format(
        test["Soil_Type15"].unique()
    )
)




## === cell 14
train.drop(cType5, axis=0, inplace=True)
train.drop(["Soil_Type7", "Soil_Type15"], axis=1, inplace=True)
test.drop(["Soil_Type7", "Soil_Type15"], axis=1, inplace=True)




## === cell 15
X = train.iloc[:, 1:-1].copy()
le = LabelEncoder()
y = le.fit_transform(train["Cover_Type"].values)

X_np = X.values
y_np = y  # already a NumPy array




## === cell 16
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1, stratify=y)


def run_model(model):
    model.fit(
        train_X,
        train_y,
        eval_set=[(val_X, val_y)],
        early_stopping_rounds=40,
        eval_metric="mlogloss",
        verbose=False,
    )
    predictions = model.predict(val_X)
    score = accuracy_score(val_y, predictions)
    return score, "Accuracy score:  {:.6f}".format(score)


def evaluate_model(model):
    print("Accuracy score:", accuracy_score(train_y, model.predict(train_X)))




## === cell 17
def run_xgboost_model(c, max_score):
    try:
        model = XGBClassifier(
            seed=1,
            tree_method="hist",  # CPU‑friendly
            predictor="auto",
            learning_rate=float(c[0]),
            gamma=float(c[1]),
            max_depth=int(c[2]),
            reg_alpha=float(c[3]),
            reg_lambda=float(c[4]),
            n_estimators=int(c[5]),
            eval_metric="mlogloss",
            use_label_encoder=False,
            n_jobs=-1,  # utilize all cores for speed
        )
        value = run_model(model)
        if value[0] > max_score[0]:
            max_score[0] = value[0]
            max_score[1] = c
        print(
            "Combination: lr={0}, gamma={1}, depth={2}, alpha={3}, lambda={4}, n_est={5} -> {6}".format(
                c[0], c[1], c[2], c[3], c[4], c[5], value[1]
            )
        )
    except Exception as e:
        print(f"Invalid combination {c}: {e}")




## === cell 18
learning_rate = [0.5]
gamma = [1.0]
max_depth = [8]
reg_alpha = [0]
reg_lambda = [0, 0.1, 0.2]
n_estimators = [100]




## === cell 19
max_score = [0.0, []]  # [best_score, best_params]
combinations = np.array(
    np.meshgrid(learning_rate, gamma, max_depth, reg_alpha, reg_lambda, n_estimators)
).T.reshape(-1, 6)

for combination in combinations:
    run_xgboost_model(combination, max_score)

if len(max_score[1]) == 0:
    print("No valid hyper‑parameter combination was found.")
else:
    print(
        "Best score: learning_rate: {0}, gamma: {1}, max_depth: {2}, reg_alpha: {3}, reg_lambda: {4}, n_estimators: {5}, Accuracy score: {6:.6f}".format(
            max_score[1][0],
            max_score[1][1],
            max_score[1][2],
            max_score[1][3],
            max_score[1][4],
            max_score[1][5],
            max_score[0],
        )
    )




## === cell 20
test_X = test.iloc[:, 1:]
test_X_np = test_X.values

model = XGBClassifier(
    seed=1,
    tree_method="hist",
    predictor="auto",
    eval_metric="mlogloss",
    learning_rate=0.3,
    gamma=1.6,
    max_depth=10,
    use_label_encoder=False,
    n_jobs=-1,  # speed up training with all cores
)

fold = 1
accuracy_scores = []
test_predictions = []

skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
for train_idx, val_idx in skf.split(X_np, y_np):
    train_X_fold, val_X_fold = X_np[train_idx], X_np[val_idx]
    train_y_fold, val_y_fold = y_np[train_idx], y_np[val_idx]

    model.fit(
        train_X_fold,
        train_y_fold,
        early_stopping_rounds=40,
        eval_set=[(val_X_fold, val_y_fold)],
        verbose=False,
    )

    preds = model.predict(val_X_fold)
    score = accuracy_score(val_y_fold, preds)
    print(f"Fold: {fold}\tAccuracy score: {score:.6f}")
    accuracy_scores.append(score)

    test_pred_fold = model.predict(test_X_np)
    test_predictions.append(test_pred_fold)
    fold += 1

test_predictions = np.squeeze(mode(np.column_stack(test_predictions), axis=1)[0])
test_predictions = le.inverse_transform(test_predictions.astype(int))

print("Mean accuracy score across folds: {0:.6f}".format(np.mean(accuracy_scores)))




## === cell 21
output = pd.DataFrame({"Id": test["Id"], "Cover_Type": test_predictions})
output.to_csv("submission.csv", index=False)
output
