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

3.7

# 3. Installed packages

fastai==2.8.5
geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

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

4.28023

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 12.80367) has done: 'I fix the `IndexError: index out of range in self` during test prediction by ensuring train/test categorical columns share the same category mapping (the current code creates separate `category` codes, causing unseen codes at inference). I do this with a minimal change: explicitly align `test_df` categories to `train_df` categories before building the FastAI `TabularPandas`. I also remove the device-switch hack and run `get_preds` normally (FastAI handles device), which avoids subtle dataloader/model device mismatches. Finally, I make sure the submission is always written to `submission.csv` with the required columns.'
- What this solution (achieved 26.89503) has done: 'Your score (12.80 RMSE) is far worse than the target (4.28), so we should make small, legitimate changes that improve generalization without changing the model architecture or training loop semantics. The biggest issue in your current preprocessing is that you never filter out obviously bad/erroneous training labels (negative fares, extreme fares) and you also don’t remove clearly invalid NYC coordinates; both are known to dominate error for this competition and can be fixed with minimal, standard filtering. I add conservative training-only filters for (a) fare range and (b) NYC bounding box + non-zero trip distance, while keeping your exact FastAI tabular setup, layers, optimizer behavior, epochs, and prediction pipeline the same. These changes typically move RMSE substantially toward the 4–6 range on this competition with the same simple model, without introducing any new modeling logic.'
- What this solution (achieved 16.16045) has done: 'Your current RMSE (26.90) is much worse than the target (4.28), so we need a small but high-impact fix that keeps your FastAI tabular model and training loop intact. The biggest remaining issue is that you train on only 100k rows, which is usually too little for this competition and leads to a very weak model; increasing the training sample size is a minimal change that preserves core logic but typically drops RMSE a lot. To stay within runtime, I increase `nrows` to a still-manageable size and keep your existing cleaning/feature engineering, splits, learner, layers, and training schedule unchanged. I also add one conservative additional training-only filter (drop rows with 0/invalid coordinates) to reduce label noise without altering the modeling approach.'

# 9. Code solution

## === cell 0
import os, math, random, warnings
import numpy as np
import pandas as pd
import torch

from fastai.tabular.all import *

warnings.filterwarnings("ignore")

PATH = "/kaggle/input"

if os.path.exists(os.path.join(PATH, "train.csv")):
    DATA_PATH = PATH
elif os.path.exists(
    os.path.join(PATH, "new-york-city-taxi-fare-prediction", "train.csv")
):
    DATA_PATH = os.path.join(PATH, "new-york-city-taxi-fare-prediction")
else:
    DATA_PATH = "../input"

DATA_PATH



## === cell 1
os.listdir(DATA_PATH)[:20]



## === cell 2
manual_seed = 555
random.seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.set_num_threads(max(1, min(8, os.cpu_count() or 2)))



## === cell 3
TRAIN_NROWS = 3_000_000  # keep identical row cap

TRAIN_USECOLS = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
TRAIN_DTYPES = {
    "key": "string",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}

_read_csv_kwargs = dict(
    nrows=TRAIN_NROWS,
    usecols=TRAIN_USECOLS,
    dtype=TRAIN_DTYPES,
)
try:
    train_df = pd.read_csv(
        f"{DATA_PATH}/train.csv", engine="pyarrow", **_read_csv_kwargs
    )
except Exception:
    train_df = pd.read_csv(f"{DATA_PATH}/train.csv", **_read_csv_kwargs)

train_df.shape, train_df.columns.tolist()



## === cell 4
TEST_USECOLS = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
TEST_DTYPES = {
    "key": "string",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}

_read_csv_kwargs = dict(
    usecols=TEST_USECOLS,
    dtype=TEST_DTYPES,
)
try:
    test_df = pd.read_csv(f"{DATA_PATH}/test.csv", engine="pyarrow", **_read_csv_kwargs)
except Exception:
    test_df = pd.read_csv(f"{DATA_PATH}/test.csv", **_read_csv_kwargs)

test_df.shape, test_df.columns.tolist()



## === cell 5
print(train_df.isnull().sum())




## === cell 6
def add_travel_vector_features(df):
    plon = df["pickup_longitude"].to_numpy()
    dlon = df["dropoff_longitude"].to_numpy()
    plat = df["pickup_latitude"].to_numpy()
    dlat = df["dropoff_latitude"].to_numpy()
    df["abs_diff_longitude"] = np.abs(dlon - plon).astype("float32", copy=False)
    df["abs_diff_latitude"] = np.abs(dlat - plat).astype("float32", copy=False)
    return df




## === cell 7
train_df.head().T



## === cell 8
test_df.head().T




## === cell 9
def data_preprocessing(df, is_train=True):
    df = df.copy(deep=False)

    if is_train:
        df = df.dropna(how="any", axis="rows")

    df = add_travel_vector_features(df)

    if is_train:
        df = df[(df.fare_amount > 0) & (df.fare_amount <= 250)]
        df = df[(df.passenger_count > 0) & (df.passenger_count <= 6)]

        nyc_lon_min, nyc_lon_max = -74.3, -73.7
        nyc_lat_min, nyc_lat_max = 40.5, 41.0
        df = df[
            (df.pickup_longitude.between(nyc_lon_min, nyc_lon_max))
            & (df.dropoff_longitude.between(nyc_lon_min, nyc_lon_max))
            & (df.pickup_latitude.between(nyc_lat_min, nyc_lat_max))
            & (df.dropoff_latitude.between(nyc_lat_min, nyc_lat_max))
        ]

        df = df[(df.abs_diff_longitude + df.abs_diff_latitude) > 1e-6]
        df = df[(df.abs_diff_longitude < 0.5) & (df.abs_diff_latitude < 0.5)]

        df = df[
            (df.pickup_longitude != 0)
            & (df.pickup_latitude != 0)
            & (df.dropoff_longitude != 0)
            & (df.dropoff_latitude != 0)
        ]

    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], errors="coerce", utc=False
    )
    if is_train:
        df = df.dropna(subset=["pickup_datetime"])

    dt = df["pickup_datetime"]

    df["hour"] = dt.dt.hour.fillna(0).astype("int16")
    df["minute"] = dt.dt.minute.fillna(0).astype("int16")
    df["second"] = dt.dt.second.fillna(0).astype("int16")

    key_arr = df["key"].astype("string").to_numpy(dtype="U", copy=False)
    tail = np.char.rpartition(key_arr, ".")[:, 2]
    order_no = pd.to_numeric(tail, errors="coerce").fillna(0).astype("int64")
    df["order_no"] = order_no

    df["Year"] = dt.dt.year.fillna(0).astype("int16")
    df["Month"] = dt.dt.month.fillna(0).astype("int16")
    df["Day"] = dt.dt.day.fillna(0).astype("int16")
    df["Dayofweek"] = dt.dt.dayofweek.fillna(0).astype("int16")
    df["Dayofyear"] = dt.dt.dayofyear.fillna(0).astype("int16")

    try:
        df["Week"] = dt.dt.isocalendar().week.astype("int16")
    except Exception:
        df["Week"] = dt.dt.week.fillna(0).astype("int16")

    df["Is_month_end"] = dt.dt.is_month_end.astype("int8")
    df["Is_month_start"] = dt.dt.is_month_start.astype("int8")
    df["Is_quarter_end"] = dt.dt.is_quarter_end.astype("int8")
    df["Is_quarter_start"] = dt.dt.is_quarter_start.astype("int8")
    df["Is_year_end"] = dt.dt.is_year_end.astype("int8")
    df["Is_year_start"] = dt.dt.is_year_start.astype("int8")

    df = df.drop(["pickup_datetime"], axis=1, errors="ignore")
    return df




## === cell 10
train_df = data_preprocessing(train_df, is_train=True)
train_df.shape, train_df.columns.tolist()[:30]



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/227684313.py in <cell line: 0>()
----> 1 train_df = data_preprocessing(train_df, is_train=True)
      2 train_df.shape, train_df.columns.tolist()[:30]
      3 

/tmp/ipykernel_11/446896975.py in data_preprocessing(df, is_train)
     51     # np.char.rpartition is vectorized; returns (head, sep, tail)
     52     tail = np.char.rpartition(key_arr, ".")[:, 2]
---> 53     order_no = pd.to_numeric(tail, errors="coerce").fillna(0).astype("int64")
     54     df["order_no"] = order_no
     55 

AttributeError: 'numpy.ndarray' object has no attribute 'fillna'

## === cell 11
test_df = data_preprocessing(test_df, is_train=False)
test_df.shape, test_df.columns.tolist()[:30]



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/954707079.py in <cell line: 0>()
----> 1 test_df = data_preprocessing(test_df, is_train=False)
      2 test_df.shape, test_df.columns.tolist()[:30]
      3 

/tmp/ipykernel_11/446896975.py in data_preprocessing(df, is_train)
     51     # np.char.rpartition is vectorized; returns (head, sep, tail)
     52     tail = np.char.rpartition(key_arr, ".")[:, 2]
---> 53     order_no = pd.to_numeric(tail, errors="coerce").fillna(0).astype("int64")
     54     df["order_no"] = order_no
     55 

AttributeError: 'numpy.ndarray' object has no attribute 'fillna'

## === cell 12
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)



## === cell 13
train_df.columns



## === cell 14
cat_vars = [
    "passenger_count",
    "Year",
    "Month",
    "Week",
    "Day",
    "Dayofweek",
    "Dayofyear",
    "Is_month_end",
    "Is_month_start",
    "Is_quarter_end",
    "Is_quarter_start",
    "Is_year_end",
    "Is_year_start",
    "hour",
    "minute",
    "second",
    "order_no",
]
contin_vars = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "abs_diff_longitude",
    "abs_diff_latitude",
]
dep = "fare_amount"

n = len(train_df)
n



## === cell 15
missing_cat = [c for c in cat_vars if c not in train_df.columns]
missing_cont = [c for c in contin_vars if c not in train_df.columns]
missing_cat, missing_cont



## === cell 16
if "Week" not in train_df.columns:
    if "Weekofyear" in train_df.columns:
        train_df["Week"] = train_df["Weekofyear"]
        test_df["Week"] = test_df["Weekofyear"]
    elif all(c in train_df.columns for c in ["Year", "Month", "Day"]):
        trn_dt = pd.to_datetime(
            dict(year=train_df["Year"], month=train_df["Month"], day=train_df["Day"]),
            errors="coerce",
        )
        tst_dt = pd.to_datetime(
            dict(year=test_df["Year"], month=test_df["Month"], day=test_df["Day"]),
            errors="coerce",
        )
        train_df["Week"] = (
            trn_dt.dt.isocalendar().week.astype("Int64").fillna(0).astype(int)
        )
        test_df["Week"] = (
            tst_dt.dt.isocalendar().week.astype("Int64").fillna(0).astype(int)
        )
    else:
        train_df["Week"] = 0
        test_df["Week"] = 0

missing_cat = [c for c in cat_vars if c not in train_df.columns]
missing_cont = [c for c in contin_vars if c not in train_df.columns]
assert (
    len(missing_cat) == 0 and len(missing_cont) == 0
), f"Missing columns: {missing_cat}, {missing_cont}"



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/3469226544.py in <cell line: 0>()
     25 missing_cont = [c for c in contin_vars if c not in train_df.columns]
     26 assert (
---> 27     len(missing_cat) == 0 and len(missing_cont) == 0
     28 ), f"Missing columns: {missing_cat}, {missing_cont}"
     29 

AssertionError: Missing columns: ['Year', 'Month', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start', 'hour', 'minute', 'second', 'order_no'], ['abs_diff_longitude', 'abs_diff_latitude']

## === cell 17
train_df = train_df[cat_vars + contin_vars + [dep, "key"]].copy()
test_df[dep] = 0.0
test_df = test_df[cat_vars + contin_vars + [dep, "key"]].copy()

train_df.shape, test_df.shape



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3745504587.py in <cell line: 0>()
----> 1 train_df = train_df[cat_vars + contin_vars + [dep, "key"]].copy()
      2 test_df[dep] = 0.0
      3 test_df = test_df[cat_vars + contin_vars + [dep, "key"]].copy()
      4 
      5 train_df.shape, test_df.shape

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['Year', 'Month', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start', 'hour', 'minute', 'second', 'order_no', 'abs_diff_longitude', 'abs_diff_latitude'] not in index"

## === cell 18
for v in cat_vars:
    train_df[v] = train_df[v].astype("category")
    test_df[v] = test_df[v].astype("category")
    test_df[v] = test_df[v].cat.set_categories(train_df[v].cat.categories)

for v in contin_vars:
    train_df[v] = train_df[v].fillna(0).astype("float32")
    test_df[v] = test_df[v].fillna(0).astype("float32")

train_df.shape, test_df.shape



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'Year'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2597173483.py in <cell line: 0>()
      1 # Speed: build categoricals in one pass; ensure shared categories once.
      2 for v in cat_vars:
----> 3     train_df[v] = train_df[v].astype("category")
      4     test_df[v] = test_df[v].astype("category")
      5     test_df[v] = test_df[v].cat.set_categories(train_df[v].cat.categories)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'Year'

## === cell 19
train_df = train_df.set_index("key")
test_df = test_df.set_index("key")



## === cell 20
splits = RandomSplitter(valid_pct=0.2, seed=manual_seed)(range_of(train_df))

procs = [Categorify, FillMissing, Normalize]
to = TabularPandas(
    train_df.reset_index(),
    procs=procs,
    cat_names=cat_vars,
    cont_names=contin_vars,
    y_names=dep,
    splits=splits,
)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3360683956.py in <cell line: 0>()
      2 
      3 procs = [Categorify, FillMissing, Normalize]
----> 4 to = TabularPandas(
      5     train_df.reset_index(),
      6     procs=procs,

/usr/local/lib/python3.11/dist-packages/fastai/tabular/core.py in __init__(self, df, procs, cat_names, cont_names, y_names, y_block, splits, do_setup, device, inplace, reduce_memory)
    168         self.cat_names,self.cont_names,self.procs = L(cat_names),L(cont_names),Pipeline(procs)
    169         self.split = len(df) if splits is None else len(splits[0])
--> 170         if do_setup: self.setup()
    171 
    172     def new(self, df, inplace=False):

/usr/local/lib/python3.11/dist-packages/fastai/tabular/core.py in setup(self)
    179     def decode_row(self, row): return self.new(pd.DataFrame(row).T).decode().items.iloc[0]
    180     def show(self, max_n=10, **kwargs): display_df(self.new(self.all_cols[:max_n]).decode().items)
--> 181     def setup(self): self.procs.setup(self)
    182     def process(self): self.procs(self)
    183     def loc(self): return self.items.loc

/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py in setup(self, items, train_setup)
    238         tfms = self.fs[:]
    239         self.fs.clear()
--> 240         for t in tfms: self.add(t,items, train_setup)
    241 
    242     def add(self,ts, items=None, train_setup=False):

/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py in add(self, ts, items, train_setup)
    242     def add(self,ts, items=None, train_setup=False):
    243         if not is_listy(ts): ts=[ts]
--> 244         for t in ts: t.setup(items, train_setup)
    245         self.fs+=ts
    246         self.fs = self.fs.sorted(key='order')

/usr/local/lib/python3.11/dist-packages/fastai/tabular/core.py in setup(self, items, train_setup)
    224     "Base class to write a non-lazy tabular processor for dataframes"
    225     def setup(self, items=None, train_setup=False): #TODO: properly deal with train_setup
--> 226         super().setup(getattr(items,'train',items), train_setup=False)
    227         # Procs are called as soon as data is available
    228         return self(items.items if isinstance(items,Datasets) else items)

/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py in setup(self, items, train_setup)
    117         train_setup = train_setup if self.train_setup is None else self.train_setup
    118         items = getattr(items, 'train', items) if train_setup else items
--> 119         try: return self.setups(items)
    120         except (AttributeError, NotFoundLookupError): return None
    121 

/usr/local/lib/python3.11/dist-packages/plum/function.py in __call__(self, _, *args, **kw_args)
    507 
    508     def __call__(self, _, *args, **kw_args):
--> 509         return self._f(self._instance, *args, **kw_args)
    510 
    511     def invoke(self, *types):

    [... skipping hidden 1 frame]

/usr/local/lib/python3.11/dist-packages/fastai/tabular/core.py in setups(self, to)
    302 
    303     def setups(self, to):
--> 304         missing = pd.isnull(to.conts).any()
    305         store_attr(but='to', na_dict={n:self.fill_strategy(to[n], self.fill_vals[n])
    306                             for n in missing[missing].keys()})

/usr/local/lib/python3.11/dist-packages/fastai/tabular/core.py in f(o)
    208 def _add_prop(cls, nm):
    209     @property
--> 210     def f(o): return o[list(getattr(o,nm+'_names'))]
    211     @f.setter
    212     def fset(o, v): o[getattr(o,nm+'_names')] = v

/usr/local/lib/python3.11/dist-packages/fastcore/foundation.py in __getitem__(self, k)
     93     def __init__(self, items): self.items = items
     94     def __len__(self): return len(self.items)
---> 95     def __getitem__(self, k): return self.items[list(k) if isinstance(k,CollBase) else k]
     96     def __setitem__(self, k, v): self.items[list(k) if isinstance(k,CollBase) else k] = v
     97     def __delitem__(self, i): del(self.items[i])

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['abs_diff_longitude', 'abs_diff_latitude'] not in index"

## === cell 21
num_workers = min(4, (os.cpu_count() or 2))
pin_mem = torch.cuda.is_available()
dls = to.dataloaders(
    bs=128,
    num_workers=num_workers,
    pin_memory=pin_mem,
    persistent_workers=(num_workers > 0),
)




## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2182342942.py in <cell line: 0>()
      3 num_workers = min(4, (os.cpu_count() or 2))
      4 pin_mem = torch.cuda.is_available()
----> 5 dls = to.dataloaders(
      6     bs=128,
      7     num_workers=num_workers,

NameError: name 'to' is not defined

## === cell 22
def rmse_fastai(inp, targ):
    return torch.sqrt(F.mse_loss(inp, targ))




## === cell 23
max_y = float(train_df[dep].max())
y_range = (0.0, max_y * 1.2)
max_y, y_range



## === cell 24
learn = tabular_learner(
    dls, layers=[1000, 500, 100], y_range=y_range, metrics=rmse_fastai
)



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1587646704.py in <cell line: 0>()
      1 learn = tabular_learner(
----> 2     dls, layers=[1000, 500, 100], y_range=y_range, metrics=rmse_fastai
      3 )
      4 

NameError: name 'dls' is not defined

## === cell 25
lr = 1e-3



## === cell 26
learn.fit(9, lr=lr)



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3013435637.py in <cell line: 0>()
----> 1 learn.fit(9, lr=lr)
      2 

NameError: name 'learn' is not defined

## === cell 27
to_test = to.new(test_df.reset_index())
test_dl = learn.dls.test_dl(
    to_test.items,
    bs=1024,  # Speed: larger test batch reduces overhead; does not change predictions.
)

len(to_test.items), to_test.items.columns.tolist()[:10]



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3511569776.py in <cell line: 0>()
----> 1 to_test = to.new(test_df.reset_index())
      2 test_dl = learn.dls.test_dl(
      3     to_test.items,
      4     bs=1024,  # Speed: larger test batch reduces overhead; does not change predictions.
      5 )

NameError: name 'to' is not defined

## === cell 28
preds, _ = learn.get_preds(dl=test_dl)
y_test = preds.squeeze().detach().cpu().numpy().astype(np.float64).reshape(-1)
y_test = np.clip(y_test, 0.0, None)

assert (
    len(y_test) == test_df.shape[0]
), f"Pred length {len(y_test)} != test rows {test_df.shape[0]}"
len(y_test), y_test[:5]



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1823456741.py in <cell line: 0>()
----> 1 preds, _ = learn.get_preds(dl=test_dl)
      2 y_test = preds.squeeze().detach().cpu().numpy().astype(np.float64).reshape(-1)
      3 y_test = np.clip(y_test, 0.0, None)
      4 
      5 assert (

NameError: name 'learn' is not defined

## === cell 29
submission = pd.DataFrame(
    {"key": test_df.index.values, "fare_amount": y_test},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission.csv", index=False)
submission.head()



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2438568250.py in <cell line: 0>()
      1 submission = pd.DataFrame(
----> 2     {"key": test_df.index.values, "fare_amount": y_test},
      3     columns=["key", "fare_amount"],
      4 )
      5 submission.to_csv("submission.csv", index=False)

NameError: name 'y_test' is not defined

## === cell 30
assert (
    submission.shape[0] == test_df.shape[0]
), "Submission rows do not match test rows."
assert list(submission.columns) == [
    "key",
    "fare_amount",
], "Submission columns incorrect."
os.path.getsize("submission.csv"), submission.shape

## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/84615114.py in <cell line: 0>()
      1 assert (
----> 2     submission.shape[0] == test_df.shape[0]
      3 ), "Submission rows do not match test rows."
      4 assert list(submission.columns) == [
      5     "key",

NameError: name 'submission' is not defined
