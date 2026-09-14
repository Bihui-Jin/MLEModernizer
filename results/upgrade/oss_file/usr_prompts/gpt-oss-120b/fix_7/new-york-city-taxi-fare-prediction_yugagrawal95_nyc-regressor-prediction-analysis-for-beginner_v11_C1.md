# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Predict the fare amount for a taxi ride given the pickup and dropoff locations.

## Metric
Root mean-squared error.

## Submission Format
For each `key` in the test set, you must predict a value for the `fare_amount` variable. The file should contain a header and have the following format:

```
key,fare_amount
2015-01-27 13:08:24.0000002,11.00
2015-02-27 13:08:24.0000002,12.05
2015-03-27 13:08:24.0000002,11.23
2015-04-27 13:08:24.0000002,14.17
2015-05-27 13:08:24.0000002,15.12
etc
```

## Dataset
- **train.csv** - Input features and target `fare_amount` values for the training set (about 55M rows).
- **test.csv** - Input features for the test set (about 10K rows). Your goal is to predict `fare_amount` for each row.
- **sample_submission.csv** - a sample submission file in the correct format (columns `key` and `fare_amount`). This file 'predicts' `fare_amount` to be $`11.35` for all rows, which is the mean `fare_amount` from the training set.

### Data fields
**ID**

- **key** - Unique `string` identifying each row in both the training and test sets. Comprised of **pickup_datetime** plus a unique integer, but this doesn't matter, it should just be used as a unique ID field.Required in your submission CSV. Not necessarily needed in the training set, but could be useful to simulate a 'submission file' while doing cross-validation within the training set.

**Features**

- **pickup_datetime** - `timestamp` value indicating when the taxi ride started.
- **pickup_longitude** - `float` for longitude coordinate of where the taxi ride started.
- **pickup_latitude** - `float` for latitude coordinate of where the taxi ride started.
- **dropoff_longitude** - `float` for longitude coordinate of where the taxi ride ended.
- **dropoff_latitude** - `float` for latitude coordinate of where the taxi ride ended.
- **passenger_count** - `integer` indicating the number of passengers in the taxi ride.

**Target**

- **fare_amount** - `float` dollar amount of the cost of the taxi ride. This value is only in the training set; this is what you are predicting in the test set and it is required in your submission CSV.

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
lightgbm==4.6.0
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
            GCP-Coupons-Instructions.rtf (486 Bytes)
            description.md (100 lines)
            labels.csv (55413943 lines)
            labels.csv.zip (1.6 GB)
            sample_submission.csv (9915 lines)
            sample_submission.csv.zip (76.2 kB)
            test.csv (9915 lines)
            test.csv.zip (273.0 kB)
            train.csv (55423857 lines)
            train.csv.zip (1.6 GB)
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
        input/
            GCP-Coupons-Instructions.rtf (486 Bytes)
            description.md (100 lines)
            labels.csv (55413943 lines)
            labels.csv.zip (1.6 GB)
            sample_submission.csv (9915 lines)
            sample_submission.csv.zip (76.2 kB)
            test.csv (9915 lines)
            test.csv.zip (273.0 kB)
            train.csv (55423857 lines)
            train.csv.zip (1.6 GB)
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
        working/
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
```

-> data/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> data/new-york-city-taxi-fare-prediction/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/train.csv has 55423856 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> data/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/train.csv has 55423856 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



## === cell 1
train_data = pd.read_csv("../input/train.csv", nrows=200000)
test_data = pd.read_csv("../input/test.csv")
train_data.head()
test_data.head()



## === cell 2
train_data.describe()




## === cell 3
def changeDataType(dataset):
    dataset["passenger_count"] = dataset.passenger_count.astype("uint8")
    dataset["pickup_longitude"] = dataset.pickup_longitude.astype("float32")
    dataset["pickup_latitude"] = dataset.pickup_latitude.astype("float32")
    dataset["dropoff_longitude"] = dataset.dropoff_longitude.astype("float32")
    dataset["dropoff_latitude"] = dataset.dropoff_latitude.astype("float32")
    dataset["pickup_datetime"] = pd.to_datetime(
        arg=dataset["pickup_datetime"], format="%Y-%m-%d %H:%M:%S UTC"
    )
    dataset.info()


changeDataType(train_data)
print("--" * 40)
changeDataType(test_data)

train_data["fare_amount"] = train_data.fare_amount.astype("float32")



## === cell 4
train_data["pickup_datetime"].head()



## === cell 5
train_data.describe()



## === cell 6
train_data.isnull().sum()
train_data = train_data.dropna(axis=0)
train_data.isnull().sum()



## === cell 7
pd.set_option("float_format", "{:f}".format)
train_data.describe()



## === cell 8
plt.figure(figsize=(8, 5), dpi=80)
sns.distplot(train_data["fare_amount"], color="red", kde=False)
train_data = train_data.loc[train_data["fare_amount"] > 0]
train_data.describe()



## === cell 9
sns.distplot(a=train_data.fare_amount, kde=False)



## === cell 10
p = pd.cut(train_data.fare_amount, 3)
p.value_counts()



## === cell 11
train_data = train_data[train_data.fare_amount < 400]



## === cell 12
sns.kdeplot(data=train_data.fare_amount)



## === cell 13
sns.countplot(x=train_data.passenger_count)



## === cell 14
train_data.passenger_count.describe()
train_data = train_data[train_data.passenger_count <= 6]



## === cell 15
sns.distplot(a=train_data.passenger_count, kde=False)



## === cell 16
train_data.describe()



## === cell 17
cond = (
    (train_data["pickup_latitude"] >= -90)
    & (train_data["pickup_latitude"] <= 90)
    & (train_data["pickup_longitude"] >= -180)
    & (train_data["pickup_longitude"] <= 180)
    & (train_data["dropoff_latitude"] >= -90)
    & (train_data["dropoff_latitude"] <= 90)
    & (train_data["dropoff_longitude"] >= -180)
    & (train_data["dropoff_longitude"] <= 180)
)
train_data = train_data[cond].reset_index(drop=True)



## === cell 18
train_data = train_data[
    train_data.pickup_latitude.between(
        test_data.pickup_latitude.min(), test_data.pickup_latitude.max()
    )
]
train_data = train_data[
    train_data.pickup_longitude.between(
        test_data.pickup_longitude.min(), test_data.pickup_longitude.max()
    )
]
train_data = train_data[
    train_data.dropoff_latitude.between(
        test_data.dropoff_latitude.min(), test_data.dropoff_latitude.max()
    )
]
train_data = train_data[
    train_data.dropoff_longitude.between(
        test_data.dropoff_longitude.min(), test_data.dropoff_longitude.max()
    )
]



## === cell 19
sns.scatterplot(x=train_data.pickup_latitude, y=train_data.pickup_longitude)
sns.scatterplot(x=train_data.dropoff_latitude, y=train_data.dropoff_longitude)




## === cell 20
def degree_to_radion(degree):
    return degree * (np.pi / 180)


def calculate_distance(
    pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude
):
    from_lat = degree_to_radion(pickup_latitude)
    from_long = degree_to_radion(pickup_longitude)
    to_lat = degree_to_radion(dropoff_latitude)
    to_long = degree_to_radion(dropoff_longitude)
    radius = 6371.01  # Earth radius in km
    lat_diff = to_lat - from_lat
    long_diff = to_long - from_long
    a = (
        np.sin(lat_diff / 2) ** 2
        + np.cos(from_lat) * np.cos(to_lat) * np.sin(long_diff / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return radius * c




## === cell 21
train_data["distance"] = calculate_distance(
    train_data.pickup_latitude,
    train_data.pickup_longitude,
    train_data.dropoff_latitude,
    train_data.dropoff_longitude,
)



## === cell 22
train_data.describe()



## === cell 23
test_data["distance"] = calculate_distance(
    test_data.pickup_latitude,
    test_data.pickup_longitude,
    test_data.dropoff_latitude,
    test_data.dropoff_longitude,
)



## === cell 24
test_data.describe()



## === cell 25
p = pd.cut(train_data.distance, 10)
p.value_counts()



## === cell 26
train_data = train_data.loc[train_data.distance < 200]



## === cell 27
train_data.describe()



## === cell 28
sns.distplot(train_data.distance, kde=False)



## === cell 29
train_data = train_data.drop(columns="key")



## === cell 30
train_data.describe()



## === cell 31
test_data_key = test_data["key"]
test_data = test_data.drop(columns="key")



## === cell 32
test_data.head()



## === cell 33
data = [train_data, test_data]
for i in data:
    i["Year"] = i["pickup_datetime"].dt.year
    i["Month"] = i["pickup_datetime"].dt.month
    i["Date"] = i["pickup_datetime"].dt.day
    i["Day of Week"] = i["pickup_datetime"].dt.dayofweek
    i["Hour"] = i["pickup_datetime"].dt.hour



## === cell 34
train_data.head()



## === cell 35
test_data.head()



## === cell 36
sns.scatterplot(x=train_data["passenger_count"], y=train_data["fare_amount"])



## === cell 37
sns.scatterplot(x=train_data["distance"], y=train_data["fare_amount"])



## === cell 38
g = sns.FacetGrid(train_data, col="Year")
g.map(sns.scatterplot, "distance", "fare_amount")



## === cell 39
train_data.describe()



## === cell 40
train_data[(train_data.distance > 100) & (train_data.fare_amount < 50)]



## === cell 41
sns.scatterplot(x=train_data["Year"], y=train_data["fare_amount"])



## === cell 42
train_data.groupby(["Month", "Year"]).count()["fare_amount"]



## === cell 43
sns.scatterplot(
    x=train_data["Month"], y=train_data["fare_amount"], hue=train_data["Year"]
)



## === cell 44
sns.scatterplot(x=train_data["Month"], y=train_data["fare_amount"])



## === cell 45
w = sns.FacetGrid(train_data, col="Year")
w.map(sns.scatterplot, "Month", "fare_amount")



## === cell 46
sns.barplot(x=train_data["Day of Week"], y=train_data["fare_amount"])



## === cell 47
plt.figure(figsize=(10, 10), dpi=150)
w = sns.FacetGrid(train_data, col="Month")
w.map(sns.barplot, "Day of Week", "fare_amount")



## === cell 48
sns.barplot(x=train_data["Hour"], y=train_data["fare_amount"])



## === cell 49
train_data = train_data.loc[train_data.pickup_latitude != 0]
train_data = train_data.loc[train_data.pickup_longitude != 0]
train_data = train_data.loc[train_data.dropoff_latitude != 0]
train_data = train_data.loc[train_data.dropoff_longitude != 0]



## === cell 50
test_data = test_data.loc[test_data.pickup_latitude != 0]
test_data = test_data.loc[test_data.pickup_longitude != 0]
test_data = test_data.loc[test_data.dropoff_latitude != 0]
test_data = test_data.loc[test_data.dropoff_longitude != 0]
test_data_key = test_data_key.loc[test_data.index]



## === cell 51
train_data = train_data.drop(columns="pickup_datetime", axis=1)
test_data = test_data.drop(columns="pickup_datetime", axis=1)



## === cell 52
X = train_data.loc[:, train_data.columns != "fare_amount"]
y = np.log1p(train_data["fare_amount"])



## === cell 53
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



## === cell 54
X_train.describe()



## === cell 55
ran_for_reg = RandomForestRegressor(
    max_depth=400, n_estimators=200, n_jobs=-1, random_state=0
)
ran_for_reg.fit(X_train, y_train)
y_ranfor_pred_log = ran_for_reg.predict(X_test)
y_ranfor_pred = np.expm1(y_ranfor_pred_log)
y_test_orig = np.expm1(y_test)
error = np.sqrt(mean_squared_error(y_test_orig, y_ranfor_pred))
error



## === cell 56
sns.barplot(x=ran_for_reg.feature_importances_, y=X_test.columns)



## === cell 57
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

bagreg = BaggingRegressor(
    base_estimator=DecisionTreeRegressor(),
    n_estimators=10,
    bootstrap=True,
    random_state=0,
    n_jobs=-1,
)
bagreg.fit(X_train, y_train)
y_bagg_pred_log = bagreg.predict(X_test)
y_bagg_pred = np.expm1(y_bagg_pred_log)
error = np.sqrt(mean_squared_error(y_test_orig, y_bagg_pred))
error



## === cell 58
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

adareg = AdaBoostRegressor(DecisionTreeRegressor())
adareg.fit(X_train, y_train)
y_adareg_pred_log = adareg.predict(X_test)
y_adareg_pred = np.expm1(y_adareg_pred_log)
error = np.sqrt(mean_squared_error(y_test_orig, y_adareg_pred))
error



## === cell 59
from sklearn.ensemble import GradientBoostingRegressor

gradient_reg = GradientBoostingRegressor()
gradient_reg.fit(X_train, y_train)
y_gradient_pred_log = gradient_reg.predict(X_test)
y_gradient_pred = np.expm1(y_gradient_pred_log)
error = np.sqrt(mean_squared_error(y_test_orig, y_gradient_pred))
error



## === cell 60
from xgboost import XGBRegressor

xgreg = XGBRegressor(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    n_jobs=-1,
    random_state=7,
)
xgreg.fit(X_train, y_train)
y_xgreg_pred_log = xgreg.predict(X_test)
y_xgreg_pred = np.expm1(y_xgreg_pred_log)
error = np.sqrt(mean_squared_error(y_test_orig, y_xgreg_pred))
error



## === cell 61
import lightgbm as lgb

model_lgb = lgb.LGBMRegressor(
    n_estimators=800,
    learning_rate=0.05,
    num_leaves=31,
    random_state=0,
    n_jobs=-1,
)
model_lgb.fit(X_train, y_train)
y_lgb_pred_log = model_lgb.predict(X_test)
y_lgb_pred = np.expm1(y_lgb_pred_log)
error = np.sqrt(mean_squared_error(y_test_orig, y_lgb_pred))
error



## === cell 62
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold



## === cell 63
n_folds = 5


def rmsle_cv(model):
    """Return RMSE from cross‑validation on the original scale."""
    kf = KFold(n_folds, shuffle=True, random_state=42)
    rmses = []
    for train_idx, valid_idx in kf.split(X_train):
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        pred_log = model.predict(X_train.iloc[valid_idx])
        pred = np.expm1(pred_log)
        true = np.expm1(y_train.iloc[valid_idx])
        rmses.append(np.sqrt(mean_squared_error(true, pred)))
    return np.array(rmses)




## === cell 64
from sklearn.ensemble import GradientBoostingRegressor

GBoost = GradientBoostingRegressor(
    n_estimators=3000,
    learning_rate=0.05,
    max_depth=4,
    max_features="sqrt",
    min_samples_leaf=15,
    min_samples_split=10,
    loss="huber",
    random_state=5,
)



## === cell 65
from xgboost import XGBRegressor

model_xgb = XGBRegressor(
    colsample_bytree=0.4603,
    gamma=0.0468,
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=1.7817,
    n_estimators=2200,
    reg_alpha=0.4640,
    reg_lambda=0.8571,
    subsample=0.5213,
    objective="reg:squarederror",
    nthread=-1,
    random_state=7,
    silent=1,
)



## === cell 66
import lightgbm as lgb

model_lgb1 = lgb.LGBMRegressor(
    objective="regression",
    num_leaves=5,
    learning_rate=0.05,
    n_estimators=720,
    max_bin=55,
    bagging_fraction=0.8,
    bagging_freq=5,
    feature_fraction=0.2319,
    feature_fraction_seed=9,
    bagging_seed=9,
    min_data_in_leaf=6,
    min_sum_hessian_in_leaf=11,
    n_jobs=-1,
    random_state=0,
)



## === cell 67
score = rmsle_cv(GBoost)
print("\nGradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



## === cell 68
score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



## === cell 69
score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



## === cell 70
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone


class AverageModel(BaseEstimator):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=1)




## === cell 71
average_model = AverageModel(models=(GBoost, model_xgb, model_lgb))
score = rmsle_cv(average_model)
print("Average model CV RMSE: {:.4f}".format(score.mean()))



## === cell 72
score.mean()




## === cell 73
class stackingModel(BaseEstimator):
    def __init__(self, base_model, meta_model, k_fold=5):
        self.base_model = base_model
        self.meta_model = meta_model
        self.k_fold = k_fold

    def fit(self, X, y):
        kfold = KFold(n_splits=self.k_fold, shuffle=True, random_state=156)
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_model)))
        for i, model in enumerate(self.base_model):
            for train_index, holdout_index in kfold.split(X, y):
                model.fit(X[train_index], y[train_index])
                y_pred = model.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
        self.meta_features_ = out_of_fold_predictions
        return self

    def predict(self, X):
        meta_feature = np.column_stack([model.predict(X) for model in self.base_model])
        return self.meta_model.predict(meta_feature)




## === cell 74
base_model = [GBoost, model_xgb, model_lgb1]


def test1(X, y):
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)
    out_of_fold_predictions = np.zeros((X.shape[0], len(base_model)))
    for i, model in enumerate(base_model):
        for train_index, holdout_index in kfold.split(X, y):
            model.fit(X.iloc[train_index], y.iloc[train_index])
            y_pred = model.predict(X.iloc[holdout_index])
            out_of_fold_predictions[holdout_index, i] = y_pred
    return out_of_fold_predictions


out_of_fold_predictions = test1(X_train, y_train)



## === cell 75
meta_model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    random_state=0,
    n_jobs=-1,
)
meta_model.fit(out_of_fold_predictions, y_train)



## === cell 76
for model in base_model:
    model.fit(X, y)

feature_data = np.column_stack([model.predict(test_data) for model in base_model])



## === cell 77
avg_pred = np.mean(feature_data, axis=1)
meta_pred = meta_model.predict(feature_data)

meta_y_log = 0.6 * meta_pred + 0.4 * avg_pred
meta_y = np.expm1(meta_y_log)
meta_y = np.clip(meta_y, a_min=0, a_max=None)



## === cell 78
submission = pd.DataFrame(
    {"key": test_data_key.reset_index(drop=True), "fare_amount": meta_y},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission.csv", index=False)
