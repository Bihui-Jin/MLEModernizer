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
scipy==1.15.3
seaborn==0.12.2
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
import numpy as np
import pandas as pd
import os
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

sns.set_style("darkgrid")




## === cell 1
candidate_paths = [
    "../input/tps-12-nn-tpu-pseudolabeling-0-95690/tps12-pseudeo-submission.csv",
    "../input/k/yamqwe/pseudolabeling-features-engineering/submission.csv",
    "../input/tps-12-g-res-variable-selection-nn-keras/baseline_nn.csv",
    "../input/tps202112-reasonable-xgboost-model/submission.csv",
    "../input/tps-dec-21-nn-feature-engg-tf/submission.csv",
    "../input/tps202112-reasonable-xgboost-model/submission.csv",
    "../input/tps-12-simple-nn-fe-pseudolabels-keras/baseline_nn.csv",
    "../input/tps-12-nn-tpu-pseudolabeling-0-95690/tps12-pseudeo-submission.csv",
    "../input/tps-12-nn-tpu-pseudolabeling-0-95690/tps12-pseudeo-submission.csv",
    "../input/k/yamqwe/pseudolabeling-features-engineering/submission.csv",
    "../input/tps202112-reasonable-xgboost-model/submission.csv",
]

predictions = []
for path in candidate_paths:
    try:
        df = pd.read_csv(path)
        if "Cover_Type" in df.columns:
            predictions.append(df)
    except Exception:
        pass

train_path = "../input/tabular-playground-series-dec-2021/train.csv"
test_path = "../input/tabular-playground-series-dec-2021/test.csv"

if os.path.exists(train_path) and os.path.exists(test_path):
    train_df = pd.read_csv(train_path)
    target_col = "Cover_Type"
    feature_cols = [c for c in train_df.columns if c not in ["Id", target_col]]

    X = train_df[feature_cols].fillna(-1).values  # NumPy array for faster fit
    y = train_df[target_col].values
    most_common_class = pd.Series(y).mode().iloc[0]

    model = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=5, max_features="sqrt"
    )
    model.fit(X, y)

    test_df = pd.read_csv(test_path)
    test_X = test_df[feature_cols].fillna(-1).values
    test_pred = model.predict(test_X)

    df_model = pd.DataFrame({"Id": test_df["Id"], "Cover_Type": test_pred})
    predictions.append(df_model)

submission = pd.read_csv(
    "../input/tabular-playground-series-dec-2021/sample_submission.csv"
)




## === cell 2
pred_arrays = [df["Cover_Type"].to_numpy() for df in predictions]



## === cell 3
if pred_arrays:
    pred_matrix = np.column_stack(pred_arrays)  # shape: (n_samples, n_models)
    ensemble = stats.mode(pred_matrix, axis=1, keepdims=False)[0]
    dif = (np.diff(np.sort(pred_matrix, axis=1), axis=1) != 0).sum(axis=1) + 1 - 1
else:
    n_samples = test_df.shape[0] if "test_df" in globals() else 0
    ensemble = np.full(n_samples, most_common_class, dtype=int)
    dif = np.zeros(n_samples, dtype=int)




## === cell 4
def nunique(a, axis):
    return (np.diff(np.sort(a, axis=axis), axis=axis) != 0).sum(axis=axis) + 1




## === cell 5
submission["Cover_Type"] = ensemble
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv with", submission.shape[0], "rows.")




## === cell 6
if submission["Cover_Type"].notnull().any():
    plt.figure(figsize=(10, 5))
    ax = sns.countplot(data=submission, x="Cover_Type")
    plt.title("Predicted Cover_Type Distribution")
    plt.xlabel("Cover Type")
    ax.bar_label(ax.containers[0])
    plt.show()
else:
    print("No valid predictions to plot.")
