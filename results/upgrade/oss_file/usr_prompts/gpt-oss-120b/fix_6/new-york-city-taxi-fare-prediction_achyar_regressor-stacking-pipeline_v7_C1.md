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

3.9

# 3. Installed packages

catboost==1.2.8
geopandas==0.14.4
joblib==1.5.2
lightgbm==4.6.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
mlxtend==0.23.4
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
import numpy as np, pandas as pd, os, matplotlib.pyplot as plt, seaborn as sns
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype

pd.set_option("display.float_format", lambda x: "%.3f" % x)
RSEED = 2020
plt.style.use("fivethirtyeight")
plt.rcParams["font.size"] = 12
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso, LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn.cluster import KMeans
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import re
from sklearn.base import BaseEstimator, TransformerMixin



## === cell 1
data = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/train.csv",
    nrows=500_000,
    parse_dates=["pickup_datetime"],
).drop(columns="key")
data = data.dropna()




## === cell 2
def minkowski_distance(x1, x2, y1, y2, p):
    return ((abs(x2 - x1) ** p) + (abs(y2 - y1)) ** p) ** (1 / p)


R = 6378.0


def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


place = pd.DataFrame(
    {
        "loc": ["jfk", "nyc", "ewr", "lgr"],
        "long": [-73.7822222222, -74.0063889, -74.175, -73.87],
        "lat": [40.6441666667, 40.7141667, 40.69, 40.77],
    }
)


def distance_to_place(df, location, source_long, source_lat):
    sel = place[place["loc"] == location].reset_index()
    return haversine_np(df[source_long], df[source_lat], sel["long"][0], sel["lat"][0])


def calculate_direction(df):
    d_lon = df["pickup_longitude"] - df["dropoff_longitude"]
    d_lat = df["pickup_latitude"] - df["dropoff_latitude"]
    result = np.zeros(len(d_lon))
    l = np.sqrt(d_lon**2 + d_lat**2)
    mask = d_lon > 0
    result[mask] = (180 / np.pi) * np.arcsin(d_lat[mask] / l[mask])
    mask = (d_lon < 0) & (d_lat > 0)
    result[mask] = 180 - (180 / np.pi) * np.arcsin(d_lat[mask] / l[mask])
    mask = (d_lon < 0) & (d_lat < 0)
    result[mask] = -180 - (180 / np.pi) * np.arcsin(d_lat[mask] / l[mask])
    return result




## === cell 3
def extract_dateinfo(
    df,
    date_col,
    drop=True,
    time=False,
    start_ref=pd.Timestamp("1900-01-01"),
    extra_attr=False,
):
    """
    Extract a rich set of datetime features from a column.
    Handles both naive and timezone‑aware datetime dtypes.
    """
    df = df.copy()
    fld = df[date_col]

    if not (is_datetime64_any_dtype(fld) or is_datetime64tz_dtype(fld)):
        df[date_col] = fld = pd.to_datetime(fld, infer_datetime_format=True)

    pre = re.sub("[Dd]ate", "", date_col)
    pre = re.sub("[Tt]ime", "", pre)

    attr = [
        "Year",
        "Month",
        "Week",
        "Day",
        "Dayofweek",
        "Dayofyear",
        "Days_in_month",
        "is_leap_year",
    ]
    if extra_attr:
        attr += [
            "Is_month_end",
            "Is_month_start",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
        ]
    if time:
        attr += ["Hour", "Minute", "Second"]
    for n in attr:
        if n == "Week":
            df[pre + n] = fld.dt.isocalendar().week
        else:
            df[pre + n] = getattr(fld.dt, n.lower())
    df[pre + "Days_in_year"] = df[pre + "is_leap_year"] + 365
    if time:
        df[pre + "frac_day"] = (
            (df[pre + "Hour"]) + (df[pre + "Minute"] / 60) + (df[pre + "Second"] / 3600)
        ) / 24
        df[pre + "frac_week"] = (df[pre + "Dayofweek"] + df[pre + "frac_day"]) / 7
        df[pre + "frac_month"] = (df[pre + "Day"] + df[pre + "frac_day"]) / (
            df[pre + "Days_in_month"] + 1
        )
        df[pre + "frac_year"] = (df[pre + "Dayofyear"] + df[pre + "frac_day"]) / (
            df[pre + "Days_in_year"] + 1
        )
    df[pre + "Elapsed"] = (fld - start_ref).dt.total_seconds()
    if drop:
        df = df.drop(date_col, axis=1)
    return df




## === cell 4
class FeatureAdder(BaseEstimator, TransformerMixin):
    """
    Performs all heavy feature engineering (clipping, clustering, distances,
    datetime expansions, holiday flag) once and returns the full dataframe.
    """

    def __init__(self, num_cols, cat_cols):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.kmeans = KMeans(n_clusters=5, random_state=RSEED)

    def fit(self, X, y=None):
        coords = np.vstack(
            [
                X[["pickup_longitude", "pickup_latitude"]].values,
                X[["dropoff_longitude", "dropoff_latitude"]].values,
            ]
        )
        self.kmeans.fit(coords)
        return self

    def transform(self, X):
        df = X.copy()
        df["passenger_count"] = np.where(
            df["passenger_count"] < 1, 1, df["passenger_count"]
        )
        df["passenger_count"] = np.where(
            df["passenger_count"] > 6, 5, df["passenger_count"]
        )
        df["pickup_latitude"] = np.clip(df["pickup_latitude"], 40, 42)
        df["dropoff_latitude"] = np.clip(df["dropoff_latitude"], 40, 42)
        df["pickup_longitude"] = np.clip(df["pickup_longitude"], -75, -72)
        df["dropoff_longitude"] = np.clip(df["dropoff_longitude"], -75, -72)

        df["cluster_pickup"] = self.kmeans.predict(
            df[["pickup_longitude", "pickup_latitude"]]
        )
        df["cluster_dropoff"] = self.kmeans.predict(
            df[["dropoff_longitude", "dropoff_latitude"]]
        )

        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
        df["abs_lat_diff"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()
        df["abs_lon_diff"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()
        df["manhattan"] = minkowski_distance(
            df["pickup_longitude"],
            df["dropoff_longitude"],
            df["pickup_latitude"],
            df["dropoff_latitude"],
            1,
        )
        df["euclidean"] = minkowski_distance(
            df["pickup_longitude"],
            df["dropoff_longitude"],
            df["pickup_latitude"],
            df["dropoff_latitude"],
            2,
        )
        df["haversine"] = haversine_np(
            df["pickup_longitude"],
            df["pickup_latitude"],
            df["dropoff_longitude"],
            df["dropoff_latitude"],
        )
        for loc in place["loc"]:
            for side in ["pickup", "dropoff"]:
                col_name = f"{side}_distance_to{loc}"
                df[col_name] = distance_to_place(
                    df, loc, f"{side}_longitude", f"{side}_latitude"
                )
        df["direction"] = calculate_direction(df)

        df = extract_dateinfo(
            df,
            "pickup_datetime",
            drop=False,
            time=True,
            start_ref=df["pickup_datetime"].min(),
        )

        holidays = calendar().holidays()
        df["usFedHoliday"] = df["pickup_datetime"].dt.normalize().isin(holidays)

        df[self.cat_cols] = df[self.cat_cols].astype(str)

        return df




## === cell 5
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))




## === cell 6
ori_cols = [
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
num_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "abs_lat_diff",
    "abs_lon_diff",
    "manhattan",
    "euclidean",
    "haversine",
    "pickup_distance_tojfk",
    "dropoff_distance_tojfk",
    "pickup_distance_tonyc",
    "dropoff_distance_tonyc",
    "pickup_distance_toewr",
    "dropoff_distance_toewr",
    "pickup_distance_tolgr",
    "dropoff_distance_tolgr",
    "direction",
    "pickup_Year",
    "pickup_Month",
    "pickup_Week",
    "pickup_Day",
    "pickup_Dayofweek",
    "pickup_Dayofyear",
    "pickup_Days_in_month",
    "pickup_Hour",
    "pickup_Minute",
    "pickup_Second",
    "pickup_Days_in_year",
    "pickup_frac_day",
    "pickup_frac_week",
    "pickup_frac_month",
    "pickup_frac_year",
    "pickup_Elapsed",
]
cat_cols = ["pickup_is_leap_year", "usFedHoliday"]
target = "fare_amount"



## === cell 7
train_df, val_df = train_test_split(data, test_size=0.1, random_state=RSEED)



## === cell 8
feature_adder = FeatureAdder(num_cols=num_cols, cat_cols=cat_cols)

preprocess_numeric = SimpleImputer(strategy="mean")
preprocess_categorical = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", preprocess_numeric, num_cols),
        ("cat", preprocess_categorical, cat_cols),
    ]
)

knn = KNeighborsRegressor(n_neighbors=3)
dt = DecisionTreeRegressor(random_state=123)
rf = DecisionTreeRegressor(random_state=123)  # placeholder
eln = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=2020)
rg = Ridge(alpha=1.0, random_state=2020)
ls = Lasso(alpha=1.0, random_state=2020)
xgb = XGBRegressor(
    random_state=2020, booster="gbtree", n_estimators=20, tree_method="hist"
)
lgb = LGBMRegressor(
    objective="regression",
    random_state=2020,
    metric="rmse",
    num_leaves=31,
    boosting_type="gbdt",
    max_depth=5,
    learning_rate=0.034,
)
catb = CatBoostRegressor(iterations=2, learning_rate=0.5, depth=3, silent=True)
lr = LinearRegression()

stack = StackingCVRegressor(
    regressors=(knn, lgb, xgb, catb, dt, rf, eln, rg, ls),
    meta_regressor=lr,
    cv=3,
)

pipeline = Pipeline(
    steps=[
        ("features", feature_adder),
        ("preprocess", preprocess),
        ("stack", stack),
    ]
)

model = TransformedTargetRegressor(regressor=pipeline, transformer=PowerTransformer())

grid = GridSearchCV(
    estimator=model,
    param_grid={},
    cv=2,  # reduced folds for faster fitting
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    error_score="raise",
)

X_train = train_df.drop(columns=target)
y_train = train_df[target]
grid.fit(X_train, y_train)



## === cell 9
X_val = val_df.drop(columns=target)
y_val = val_df[target]
val_pred = grid.predict(X_val)
print(f"Validation RMSE: {rmse(y_val, val_pred):.4f}")



## === cell 10
test = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/test.csv",
    parse_dates=["pickup_datetime"],
)
test_X = test[ori_cols]
test_pred = grid.predict(test_X)

submission = pd.DataFrame({"key": test["key"], "fare_amount": test_pred})
submission_path = "submission_v1a.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
