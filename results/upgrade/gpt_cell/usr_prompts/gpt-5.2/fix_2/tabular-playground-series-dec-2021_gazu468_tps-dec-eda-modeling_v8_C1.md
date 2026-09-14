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

category_encoders==2.7.0
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
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from  datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import datetime as datetime
from sklearn.metrics import accuracy_score
from category_encoders.target_encoder import TargetEncoder
warnings.filterwarnings('ignore')


## === cell 1
train = pd.read_csv("../input/tabular-playground-series-dec-2021/train.csv")
test = pd.read_csv("../input/tabular-playground-series-dec-2021/test.csv")


## === cell 2
train.head(100)


## === cell 3
train.tail()


## === cell 4
train.columns


## === cell 5
train.describe()


## === cell 7
train['Cover_Type'].describe()


## === cell 8
train.groupby('Cover_Type').size()


## === cell 9
plt.hist(train['Cover_Type'])


## === cell 10
print("Skewness: %f" % train['Cover_Type'].skew())
print("Kurtosis: %f" % train['Cover_Type'].kurt())


## === cell 11
var = 'Hillshade_Noon'
data = pd.concat([train['Cover_Type'], train[var]], axis=1)
data.plot.scatter(x=var, y='Cover_Type', ylim=(0,800000))


## === cell 12
corr=train.corr()
v=10
colmn = corr.nlargest(v, 'Cover_Type')
colmn


## === cell 13
colmn = corr.nlargest(v, 'Cover_Type')['Cover_Type'].index 
xm = np.corrcoef(train[colmn].values.T)
sns.set(font_scale=1.25)
plt.figure(figsize=(18, 18))
hm = sns.heatmap(xm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                 yticklabels=colmn.values, xticklabels=colmn.values)
plt.show()


## === cell 14
total= train.isnull().sum().sort_values(ascending=False)
total


## === cell 15
train_X = train.drop('Cover_Type', axis=1)
train_y = train['Cover_Type']


## === cell 16
train_y.head()


## === cell 17
train_X


## === cell 18
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.22, random_state=2021 )


## === cell 19
y_test


## === cell 20
nums_cols = [col for col in X_train.columns if X_train[col].dtype in ["float16","float32","float64"]]
catgo_cols = [col for col in X_train.columns if X_train[col].dtype not in ["float16","float32","float64"]]


## === cell 21
d_test = test

for cols in catgo_cols:
    enc = TargetEncoder(cols=[cols])
    X_train = enc.fit_transform(X_train, y_train)
    X_test= enc.transform(X_test)
    d_test = enc.transform(d_test)


## === cell 22
del train, test, train_X, train_y


## === cell 23
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
scaler.fit(X_train)
train_X = pd.DataFrame(scaler.transform(X_train))
test_X = pd.DataFrame(scaler.transform(X_test))
test = pd.DataFrame(scaler.transform(d_test))


## === cell 24
train_X


## === cell 25
train_X = train_X.to_numpy().astype(np.float32)
y_train =y_train.to_numpy().astype(np.float32)
y_test =y_test.to_numpy().astype(np.float32)
test_X =test_X.to_numpy().astype(np.float32)
test =test.to_numpy().astype(np.float32)


## === cell 27
from xgboost import XGBClassifier

y_train = (y_train - 1).astype(np.int32)
y_test = (y_test - 1).astype(np.int32)

params = {
    "objective": "multi:softmax",
    "tree_method": "gpu_hist",
    "eval_metric": "mlogloss",
    "booster": "gbtree",
    "gamma": 0.75,
    "max_depth": 7,
    "alpha": 10,
    "learning_rate": 0.007,
    "n_estimators": 2000,
    "predictor": "gpu_predictor",
}


xgb = XGBClassifier(**params)
xgb.fit(
    train_X,
    y_train,
    early_stopping_rounds=200,
    eval_set=[(test_X, y_test)],
    verbose=True,
)


## === cell 28
preds_valid = xgb.predict(test_X).astype('int')
acc = accuracy_score(y_test,  preds_valid)
print("accuracy score:", acc)


## === cell 29
preds_valid


## === cell 30
sub = pd.read_csv('../input/tabular-playground-series-dec-2021/sample_submission.csv')
sub['Cover_Type'] =xgb.predict(test).astype('int')
sub.to_csv("submission.csv",index=False)
sub.head()
