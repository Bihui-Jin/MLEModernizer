# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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
imbalanced-learn==0.13.0
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

# 5. Target score

0.90073

# 6. Current score

0.03851

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.03851) has done: 'Diagnosis: The crash happens in cell 16 during `pipe.fit(...)` because the `XGBClassifier` in the pipeline was configured with `tree_method='gpu_hist'` and `predictor='gpu_predictor'`, but the runtime has no available GPU device (`ctx_->gpu_id >= 0 ... Must have at least one device`). This is an environment/device mismatch rather than a data or model-shape issue. The fix is to switch the XGBoost model to CPU execution so training can proceed deterministically on this machine.  

Patch summary: In cell 16, before fitting, override the already-created pipeline’s XGBClassifier parameters to CPU (`tree_method='hist'`, `predictor='cpu_predictor'`) and keep everything else unchanged. Then perform the same label-encoding and `pipe.fit` call as before.  

Updated cells: cell 16 only.  

Compatibility notes for cell k+1: `pipe` remains the same fitted `Pipeline` object, so `pipe.predict(x_test)` in cell 17 work unchanged and return the same label-space (encoded class indices) as before.  

Assumptions: This environment has no GPU available to XGBoost; using CPU is acceptable and does not change the intended training/evaluation semantics beyond device choice.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

try:
    from imblearn.under_sampling import (
        RandomUnderSampler,
    )  # Used for under sampling. explained further in notebook.
except ModuleNotFoundError:
    RandomUnderSampler = None

import collections
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier


## === cell 1
df_train_og = pd.read_csv('../input/tabular-playground-series-dec-2021/train.csv')
df_test_og  = pd.read_csv('../input/tabular-playground-series-dec-2021/test.csv')
submission  = pd.read_csv('../input/tabular-playground-series-dec-2021/sample_submission.csv')


## === cell 2
df_train_og.shape


## === cell 3
df_train_og.head()


## === cell 4
df_train_og.nunique()


## === cell 5
def reduce_mem_usage(df, verbose=True):
    numerics = ['int8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
 
    return df


## === cell 6
df_train = reduce_mem_usage(df_train_og)
df_test = reduce_mem_usage(df_test_og)
del df_train_og
del df_test_og


## === cell 7
cat_count = collections.Counter(df_train['Cover_Type'])
cat_freq = cat_count.values()
cat = cat_count.keys()
plt.bar(cat , cat_freq)

print(cat_count)


## === cell 8
df_train = df_train[(df_train['Cover_Type'] != 4) & (df_train['Cover_Type'] != 5)]


## === cell 9

X = df_train.drop(columns=["Id", "Cover_Type"])
y = df_train["Cover_Type"]


def _fallback_random_undersample_not_minority(X_df, y_ser, random_state=42):
    df = X_df.copy()
    df["_target_"] = y_ser.values

    vc = df["_target_"].value_counts()
    minority_class = vc.idxmin()
    n_min = int(vc.min())

    parts = []
    for cls, grp in df.groupby("_target_", sort=False):
        if cls == minority_class:
            parts.append(grp)
        else:
            parts.append(grp.sample(n=n_min, replace=False, random_state=random_state))

    df_res = (
        pd.concat(parts, axis=0)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )
    y_res_local = df_res.pop("_target_")
    X_res_local = df_res
    return X_res_local, y_res_local


try:
    if RandomUnderSampler is None:
        from imblearn.under_sampling import RandomUnderSampler as _RandomUnderSampler  # type: ignore

        RandomUnderSampler = _RandomUnderSampler

    rus = RandomUnderSampler(sampling_strategy="not minority")
    X_res, y_res = rus.fit_resample(X, y)
except Exception:
    X_res, y_res = _fallback_random_undersample_not_minority(X, y, random_state=42)


## === cell 10
cat_count = collections.Counter(y_res)
cat_freq = cat_count.values()
cat = cat_count.keys()
plt.bar(cat , cat_freq)

print(cat_count)


## === cell 11
X = X_res.drop(columns = ['Soil_Type7' , 'Soil_Type15'])
y = y_res


## === cell 12
from sklearn.feature_selection import SelectKBest,f_classif
selector = SelectKBest(f_classif,k="all")
fitter = selector.fit(X,y)
scores_df = pd.DataFrame(fitter.scores_) 
columns_df = pd.DataFrame(X.columns)
featurescores = pd.concat([scores_df ,columns_df] , axis=1)
featurescores.columns=['score','column name']
plt.figure(figsize=(20,5))
plt.bar(featurescores['column name'] , featurescores['score'],width=0.4)
plt.xticks(rotation = 'vertical')
plt.plot()


## === cell 14
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)


## === cell 15
pipe = Pipeline(steps = [
    
    ('step1' , StandardScaler()),
    ('step2' , XGBClassifier(objective = 'multi:softmax', tree_method = 'gpu_hist', eval_metric = 'mlogloss', 
                    subsample = 0.6,gamma = 0.5,max_depth = 6,alpha = 0,learning_rate = .03,
                    n_estimators = 1000,predictor = 'gpu_predictor'))
])


## === cell 16
pipe.set_params(
    step2__tree_method="hist",
    step2__predictor="cpu_predictor",
)

_classes_sorted = np.sort(pd.unique(y_train))
_class_to_index = {c: i for i, c in enumerate(_classes_sorted)}
_index_to_class = {i: c for c, i in _class_to_index.items()}

y_train_enc = y_train.map(_class_to_index).astype(np.int32)

pipe.fit(x_train, y_train_enc)


## === cell 17
y_pred = pipe.predict(x_test)


## === cell 18
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))


## === cell 19
df_test = df_test.drop(columns = ['Id' , 'Soil_Type7' , 'Soil_Type15'])
Final_pred = pipe.predict(df_test)


## === cell 20
submission['Cover_Type'] = Final_pred
submission.to_csv('Submission.csv' , index=False)
