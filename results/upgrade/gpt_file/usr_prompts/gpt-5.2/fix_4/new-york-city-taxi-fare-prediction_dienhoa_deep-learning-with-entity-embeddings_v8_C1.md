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

# 8. Previous improvement plan

N/A

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



## === cell 3
train_df = pd.read_csv(f"{DATA_PATH}/train.csv", nrows=100000)
train_df.shape, train_df.columns.tolist()



## === cell 4
test_df = pd.read_csv(f"{DATA_PATH}/test.csv")
test_df.shape, test_df.columns.tolist()



## === cell 5
print(train_df.isnull().sum())




## === cell 6
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()




## === cell 7
train_df.head().T



## === cell 8
test_df.head().T




## === cell 9
def data_preprocessing(df, is_train=True):
    df = df.copy()

    if is_train:
        df = df.dropna(how="any", axis="rows")

    add_travel_vector_features(df)

    if is_train:
        df = df[(df.abs_diff_longitude < 5) & (df.abs_diff_latitude < 5)]
        df = df[(df.passenger_count > 0) & (df.passenger_count <= 6)]

    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], errors="coerce", utc=False
    )
    if is_train:
        df = df.dropna(subset=["pickup_datetime"])

    df["date"] = df["pickup_datetime"].dt.date.astype(str)
    df["time"] = df["pickup_datetime"].dt.time.astype(str)

    tparts = df["time"].str.split(":", expand=True)
    df["hour"] = pd.to_numeric(tparts[0], errors="coerce").fillna(0).astype("int64")
    df["minute"] = pd.to_numeric(tparts[1], errors="coerce").fillna(0).astype("int64")
    sec = tparts[2].str.split(".", expand=True)[0] if tparts.shape[1] > 2 else "0"
    df["second"] = pd.to_numeric(sec, errors="coerce").fillna(0).astype("int64")

    kparts = df["key"].astype(str).str.split(".", expand=True)
    if kparts.shape[1] > 1:
        df["order_no"] = (
            pd.to_numeric(kparts[1], errors="coerce").fillna(0).astype("int64")
        )
    else:
        df["order_no"] = 0

    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")

    df = add_datepart(df, "date_dt", drop=False)

    df = df.drop(
        ["pickup_datetime", "date", "time", "date_dt"], axis=1, errors="ignore"
    )
    return df




## === cell 10
train_df = data_preprocessing(train_df, is_train=True)
train_df.shape, train_df.columns.tolist()[:30]



## === cell 11
test_df = data_preprocessing(test_df, is_train=False)
test_df.shape, test_df.columns.tolist()[:30]



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
    else:
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

missing_cat = [c for c in cat_vars if c not in train_df.columns]
missing_cont = [c for c in contin_vars if c not in train_df.columns]
assert (
    len(missing_cat) == 0 and len(missing_cont) == 0
), f"Missing columns: {missing_cat}, {missing_cont}"



## --- ERROR in cell 16, traceback:
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
/tmp/ipykernel_11/609501818.py in <cell line: 0>()
      9         # Reconstruct a datetime from available date parts (these exist because add_datepart now worked)
     10         trn_dt = pd.to_datetime(
---> 11             dict(year=train_df["Year"], month=train_df["Month"], day=train_df["Day"]),
     12             errors="coerce",
     13         )

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

KeyError: "['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'] not in index"

## === cell 18
for v in cat_vars:
    train_df[v] = train_df[v].astype("category")
    test_df[v] = test_df[v].astype("category")

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
/tmp/ipykernel_11/3937465532.py in <cell line: 0>()
      1 for v in cat_vars:
----> 2     train_df[v] = train_df[v].astype("category")
      3     test_df[v] = test_df[v].astype("category")
      4 
      5 for v in contin_vars:

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
    243     order = 1
    244     def setups(self, to):
--> 245         store_attr(classes={n:CategoryMap(to.iloc[:,n].items, add_na=(n in to.cat_names)) for n in to.cat_names}, but='to')
    246 
    247     def encodes(self, to): to.transform(to.cat_names, partial(_apply_cats, self.classes, 1))

/usr/local/lib/python3.11/dist-packages/fastai/tabular/core.py in <dictcomp>(.0)
    243     order = 1
    244     def setups(self, to):
--> 245         store_attr(classes={n:CategoryMap(to.iloc[:,n].items, add_na=(n in to.cat_names)) for n in to.cat_names}, but='to')
    246 
    247     def encodes(self, to): to.transform(to.cat_names, partial(_apply_cats, self.classes, 1))

/usr/local/lib/python3.11/dist-packages/fastai/tabular/core.py in __getitem__(self, idxs)
    139         if isinstance(idxs,tuple):
    140             rows,cols = idxs
--> 141             cols = df.columns.isin(cols) if is_listy(cols) else df.columns.get_loc(cols)
    142         else: rows,cols = idxs,slice(None)
    143         return self.to.new(df.iloc[rows, cols])

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'Year'

## === cell 21
dls = to.dataloaders(bs=128)




## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/490966224.py in <cell line: 0>()
----> 1 dls = to.dataloaders(bs=128)
      2 
      3 

NameError: name 'to' is not defined

## === cell 22
def rmse_fastai(inp, targ):
    return torch.sqrt(F.mse_loss(inp, targ))




## === cell 23
learn = tabular_learner(dls, layers=[1000, 500, 100], metrics=rmse_fastai)



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3033388612.py in <cell line: 0>()
----> 1 learn = tabular_learner(dls, layers=[1000, 500, 100], metrics=rmse_fastai)
      2 

NameError: name 'dls' is not defined

## === cell 24
max_y = float(train_df[dep].max())
y_range = (0.0, max_y * 1.2)
max_y, y_range



## === cell 25
lr = 1e-3



## === cell 26
learn.fit(3, lr=lr)



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1021770392.py in <cell line: 0>()
----> 1 learn.fit(3, lr=lr)
      2 

NameError: name 'learn' is not defined

## === cell 27
learn.fit(3, lr=lr)



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1021770392.py in <cell line: 0>()
----> 1 learn.fit(3, lr=lr)
      2 

NameError: name 'learn' is not defined

## === cell 28
learn.fit(3, lr=lr)



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1021770392.py in <cell line: 0>()
----> 1 learn.fit(3, lr=lr)
      2 

NameError: name 'learn' is not defined

## === cell 29
to_test = TabularPandas(
    test_df.reset_index(),
    procs=to.procs,
    cat_names=cat_vars,
    cont_names=contin_vars,
    y_names=dep,
    splits=None,
)
test_dl = learn.dls.test_dl(to_test.items)



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3914511922.py in <cell line: 0>()
      1 to_test = TabularPandas(
      2     test_df.reset_index(),
----> 3     procs=to.procs,
      4     cat_names=cat_vars,
      5     cont_names=contin_vars,

NameError: name 'to' is not defined

## === cell 30
preds, _ = learn.get_preds(dl=test_dl)
y_test = preds.squeeze().cpu().numpy().reshape(-1)
len(y_test), y_test[:5]



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/515165039.py in <cell line: 0>()
----> 1 preds, _ = learn.get_preds(dl=test_dl)
      2 y_test = preds.squeeze().cpu().numpy().reshape(-1)
      3 len(y_test), y_test[:5]
      4 

NameError: name 'learn' is not defined

## === cell 31
submission = pd.DataFrame(
    {"key": test_df.index.values, "fare_amount": y_test},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission.csv", index=False)
submission.head()



## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2438568250.py in <cell line: 0>()
      1 submission = pd.DataFrame(
----> 2     {"key": test_df.index.values, "fare_amount": y_test},
      3     columns=["key", "fare_amount"],
      4 )
      5 submission.to_csv("submission.csv", index=False)

NameError: name 'y_test' is not defined

## === cell 32
assert (
    submission.shape[0] == test_df.shape[0]
), "Submission rows do not match test rows."
assert list(submission.columns) == [
    "key",
    "fare_amount",
], "Submission columns incorrect."
os.path.getsize("submission.csv"), submission.shape

## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/84615114.py in <cell line: 0>()
      1 assert (
----> 2     submission.shape[0] == test_df.shape[0]
      3 ), "Submission rows do not match test rows."
      4 assert list(submission.columns) == [
      5     "key",

NameError: name 'submission' is not defined
