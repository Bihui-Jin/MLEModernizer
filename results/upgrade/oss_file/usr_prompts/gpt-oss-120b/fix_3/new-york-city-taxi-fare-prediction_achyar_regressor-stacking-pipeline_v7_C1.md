# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

3.56643

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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
class data_transform:
    def __init__(self, num, cat, is_cat):
        self.num = num
        self.cat = cat
        self.is_cat = is_cat

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
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

        kmeans = KMeans(n_clusters=5, random_state=RSEED)
        df["cluster_pickup"] = kmeans.fit_predict(
            df[["pickup_longitude", "pickup_latitude"]]
        )
        df["cluster_dropoff"] = kmeans.fit_predict(
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

        df[self.cat] = df[self.cat].astype(str)

        if self.is_cat == 1:
            return df[self.cat]
        elif self.is_cat == 0:
            return df[self.num]
        else:
            return df

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)




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
num_transformer = Pipeline(
    steps=[
        ("prep", data_transform(num_cols, cat_cols, is_cat=0)),
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", PowerTransformer()),
    ]
)

cat_transformer = Pipeline(
    steps=[
        ("prep", data_transform(num_cols, cat_cols, is_cat=1)),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", num_transformer, ori_cols),
        ("cat", cat_transformer, ori_cols),
    ]
)

knn = KNeighborsRegressor(n_neighbors=3)
dt = DecisionTreeRegressor(random_state=123)
rf = DecisionTreeRegressor(random_state=123)  # kept as placeholder
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
    regressors=(knn, lgb, xgb, catb, dt, rf, eln, rg, ls), meta_regressor=lr, cv=3
)

stack_pipe = Pipeline(steps=[("preprocess", preprocess), ("stack", stack)])

model = TransformedTargetRegressor(regressor=stack_pipe, transformer=PowerTransformer())

grid = GridSearchCV(
    estimator=model,
    param_grid={},
    cv=3,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    error_score="raise",
)

X_train = train_df.drop(columns=target)
y_train = train_df[target]
grid.fit(X_train, y_train)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
_RemoteTraceback                          Traceback (most recent call last)
_RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/joblib/externals/loky/process_executor.py", line 490, in _process_worker
    r = call_item()
        ^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/externals/loky/process_executor.py", line 291, in __call__
    return self.fn(*self.args, **self.kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/parallel.py", line 607, in __call__
    return [func(*args, **kwargs) for func, args, kwargs in self.items]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/parallel.py", line 607, in <listcomp>
    return [func(*args, **kwargs) for func, args, kwargs in self.items]
            ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/utils/parallel.py", line 123, in __call__
    return self.function(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_validation.py", line 686, in _fit_and_score
    estimator.fit(X_train, y_train, **fit_params)
  File "/usr/local/lib/python3.11/dist-packages/sklearn/compose/_target.py", line 262, in fit
    self.regressor_.fit(X, y_trans, **fit_params)
  File "/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py", line 401, in fit
    Xt = self._fit(X, y, **fit_params_steps)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py", line 359, in _fit
    X, fitted_transformer = fit_transform_one_cached(
                            ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/memory.py", line 326, in __call__
    return self.func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py", line 893, in _fit_transform_one
    res = transformer.fit_transform(X, y, **fit_params)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py", line 140, in wrapped
    data_to_wrap = f(self, X, *args, **kwargs)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/compose/_column_transformer.py", line 727, in fit_transform
    result = self._fit_transform(X, y, _fit_transform_one)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/compose/_column_transformer.py", line 658, in _fit_transform
    return Parallel(n_jobs=self.n_jobs)(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/utils/parallel.py", line 63, in __call__
    return super().__call__(iterable_with_config)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/parallel.py", line 1986, in __call__
    return output if self.return_generator else list(output)
                                                ^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/parallel.py", line 1914, in _get_sequential_output
    res = func(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/utils/parallel.py", line 123, in __call__
    return self.function(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py", line 893, in _fit_transform_one
    res = transformer.fit_transform(X, y, **fit_params)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py", line 437, in fit_transform
    Xt = self._fit(X, y, **fit_params_steps)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py", line 359, in _fit
    X, fitted_transformer = fit_transform_one_cached(
                            ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/memory.py", line 326, in __call__
    return self.func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py", line 893, in _fit_transform_one
    res = transformer.fit_transform(X, y, **fit_params)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/2824070080.py", line 84, in fit_transform
  File "/tmp/ipykernel_11/2824070080.py", line 62, in transform
  File "/tmp/ipykernel_11/639919718.py", line 45, in extract_dateinfo
AttributeError: 'DatetimeProperties' object has no attribute 'week'
"""

The above exception was the direct cause of the following exception:

AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2699988741.py in <cell line: 0>()
     62 X_train = train_df.drop(columns=target)
     63 y_train = train_df[target]
---> 64 grid.fit(X_train, y_train)
     65 

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_search.py in fit(self, X, y, groups, **fit_params)
    872                 return results
    873 
--> 874             self._run_search(evaluate_candidates)
    875 
    876             # multimetric is determined here because in the case of a callable

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_search.py in _run_search(self, evaluate_candidates)
   1386     def _run_search(self, evaluate_candidates):
   1387         """Search all candidates in param_grid"""
-> 1388         evaluate_candidates(ParameterGrid(self.param_grid))
   1389 
   1390 

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_search.py in evaluate_candidates(candidate_params, cv, more_results)
    819                     )
    820 
--> 821                 out = parallel(
    822                     delayed(_fit_and_score)(
    823                         clone(base_estimator),

/usr/local/lib/python3.11/dist-packages/sklearn/utils/parallel.py in __call__(self, iterable)
     61             for delayed_func, args, kwargs in iterable
     62         )
---> 63         return super().__call__(iterable_with_config)
     64 
     65 

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in __call__(self, iterable)
   2070         next(output)
   2071 
-> 2072         return output if self.return_generator else list(output)
   2073 
   2074     def __repr__(self):

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _get_outputs(self, iterator, pre_dispatch)
   1680 
   1681             with self._backend.retrieval_context():
-> 1682                 yield from self._retrieve()
   1683 
   1684         except GeneratorExit:

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _retrieve(self)
   1782             # worker traceback.
   1783             if self._aborting:
-> 1784                 self._raise_error_fast()
   1785                 break
   1786 

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _raise_error_fast(self)
   1857         # called directly or if the generator is gc'ed.
   1858         if error_job is not None:
-> 1859             error_job.get_result(self.timeout)
   1860 
   1861     def _warn_exit_early(self):

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in get_result(self, timeout)
    756             # callback thread, and is stored internally. It's just waiting to
    757             # be returned.
--> 758             return self._return_or_raise()
    759 
    760         # For other backends, the main thread needs to run the retrieval step.

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _return_or_raise(self)
    771         try:
    772             if self.status == TASK_ERROR:
--> 773                 raise self._result
    774             return self._result
    775         finally:

AttributeError: 'DatetimeProperties' object has no attribute 'week'

## === cell 9
X_val = val_df.drop(columns=target)
y_val = val_df[target]
val_pred = grid.predict(X_val)
print(f"Validation RMSE: {rmse(y_val, val_pred):.4f}")



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/1958067529.py in <cell line: 0>()
      1 X_val = val_df.drop(columns=target)
      2 y_val = val_df[target]
----> 3 val_pred = grid.predict(X_val)
      4 print(f"Validation RMSE: {rmse(y_val, val_pred):.4f}")
      5 

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_search.py in predict(self, X)
    496             the best found parameters.
    497         """
--> 498         check_is_fitted(self)
    499         return self.best_estimator_.predict(X)
    500 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1388 
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 
   1392 

NotFittedError: This GridSearchCV instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.

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

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/3125590364.py in <cell line: 0>()
      4 )
      5 test_X = test[ori_cols]
----> 6 test_pred = grid.predict(test_X)
      7 
      8 submission = pd.DataFrame({"key": test["key"], "fare_amount": test_pred})

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_search.py in predict(self, X)
    496             the best found parameters.
    497         """
--> 498         check_is_fitted(self)
    499         return self.best_estimator_.predict(X)
    500 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1388 
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 
   1392 

NotFittedError: This GridSearchCV instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.
