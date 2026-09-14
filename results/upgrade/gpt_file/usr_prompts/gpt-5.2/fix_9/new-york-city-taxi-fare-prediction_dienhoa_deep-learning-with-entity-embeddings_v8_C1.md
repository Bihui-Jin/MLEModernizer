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
    return df




## === cell 7
train_df.head().T



## === cell 8
test_df.head().T




## === cell 9
def data_preprocessing(df, is_train=True):
    df = df.copy()

    if is_train:
        df = df.dropna(how="any", axis="rows")

    df = add_travel_vector_features(df)

    if is_train:
        df = df[(df.abs_diff_longitude < 5) & (df.abs_diff_latitude < 5)]
        df = df[(df.passenger_count > 0) & (df.passenger_count <= 6)]

    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], errors="coerce", utc=False
    )
    if is_train:
        df = df.dropna(subset=["pickup_datetime"])

    dt = df["pickup_datetime"]
    df["hour"] = dt.dt.hour.fillna(0).astype("int64")
    df["minute"] = dt.dt.minute.fillna(0).astype("int64")
    df["second"] = dt.dt.second.fillna(0).astype("int64")

    kparts = df["key"].astype(str).str.split(".", expand=True)
    if kparts.shape[1] > 1:
        df["order_no"] = (
            pd.to_numeric(kparts[1], errors="coerce").fillna(0).astype("int64")
        )
    else:
        df["order_no"] = 0

    df["Year"] = dt.dt.year.fillna(0).astype("int64")
    df["Month"] = dt.dt.month.fillna(0).astype("int64")
    df["Day"] = dt.dt.day.fillna(0).astype("int64")
    df["Dayofweek"] = dt.dt.dayofweek.fillna(0).astype("int64")
    df["Dayofyear"] = dt.dt.dayofyear.fillna(0).astype("int64")

    try:
        df["Week"] = dt.dt.isocalendar().week.astype("int64")
    except Exception:
        df["Week"] = dt.dt.week.fillna(0).astype("int64")

    df["Is_month_end"] = dt.dt.is_month_end.astype("int64")
    df["Is_month_start"] = dt.dt.is_month_start.astype("int64")
    df["Is_quarter_end"] = dt.dt.is_quarter_end.astype("int64")
    df["Is_quarter_start"] = dt.dt.is_quarter_start.astype("int64")
    df["Is_year_end"] = dt.dt.is_year_end.astype("int64")
    df["Is_year_start"] = dt.dt.is_year_start.astype("int64")

    df = df.drop(["pickup_datetime"], axis=1, errors="ignore")
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



## === cell 17
train_df = train_df[cat_vars + contin_vars + [dep, "key"]].copy()
test_df[dep] = 0.0
test_df = test_df[cat_vars + contin_vars + [dep, "key"]].copy()

train_df.shape, test_df.shape



## === cell 18
for v in cat_vars:
    train_df[v] = train_df[v].astype("category")
    test_df[v] = test_df[v].astype("category")

for v in contin_vars:
    train_df[v] = train_df[v].fillna(0).astype("float32")
    test_df[v] = test_df[v].fillna(0).astype("float32")

train_df.shape, test_df.shape



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



## === cell 21
dls = to.dataloaders(bs=128)




## === cell 22
def rmse_fastai(inp, targ):
    return torch.sqrt(F.mse_loss(inp, targ))




## === cell 23
learn = tabular_learner(dls, layers=[1000, 500, 100], metrics=rmse_fastai)



## === cell 24
max_y = float(train_df[dep].max())
y_range = (0.0, max_y * 1.2)
max_y, y_range



## === cell 25
lr = 1e-3



## === cell 26
learn.fit(3, lr=lr)



## === cell 27
learn.fit(3, lr=lr)



## === cell 28
learn.fit(3, lr=lr)



## === cell 29
to_test = to.new(test_df.reset_index())
test_dl = learn.dls.test_dl(to_test.items)

len(to_test.items), to_test.items.columns.tolist()[:10]



## === cell 30
preds, _ = learn.get_preds(dl=test_dl)
y_test = preds.squeeze().detach().cpu().numpy().astype(np.float64).reshape(-1)

y_test = np.clip(y_test, 0.0, None)

assert (
    len(y_test) == test_df.shape[0]
), f"Pred length {len(y_test)} != test rows {test_df.shape[0]}"

len(y_test), y_test[:5]



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/640321911.py in <cell line: 0>()
----> 1 preds, _ = learn.get_preds(dl=test_dl)
      2 y_test = preds.squeeze().detach().cpu().numpy().astype(np.float64).reshape(-1)
      3 
      4 # Keep fares non-negative (sanity constraint; typically score-neutral or helpful)
      5 y_test = np.clip(y_test, 0.0, None)

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in get_preds(self, ds_idx, dl, with_input, with_decoded, with_loss, act, inner, reorder, cbs, **kwargs)
    314         if with_loss: ctx_mgrs.append(self.loss_not_reduced())
    315         with ContextManagers(ctx_mgrs):
--> 316             self._do_epoch_validate(dl=dl)
    317             if act is None: act = getcallable(self.loss_func, 'activation')
    318             res = cb.all_tensors()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _do_epoch_validate(self, ds_idx, dl)
    250         if dl is None: dl = self.dls[ds_idx]
    251         self.dl = dl
--> 252         with torch.no_grad(): self._with_events(self.all_batches, 'validate', CancelValidException)
    253 
    254     def _do_epoch(self):

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _with_events(self, f, event_type, ex, final)
    205 
    206     def _with_events(self, f, event_type, ex, final=noop):
--> 207         try: self(f'before_{event_type}');  f()
    208         except ex: self(f'after_cancel_{event_type}')
    209         self(f'after_{event_type}');  final()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in all_batches(self)
    211     def all_batches(self):
    212         self.n_iter = len(self.dl)
--> 213         for o in enumerate(self.dl): self.one_batch(*o)
    214 
    215     def _backward(self): self.loss_grad.backward()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in one_batch(self, i, b)
    241         b = self._set_device(b)
    242         self._split(b)
--> 243         self._with_events(self._do_one_batch, 'batch', CancelBatchException)
    244 
    245     def _do_epoch_train(self):

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _with_events(self, f, event_type, ex, final)
    207         try: self(f'before_{event_type}');  f()
    208         except ex: self(f'after_cancel_{event_type}')
--> 209         self(f'after_{event_type}');  final()
    210 
    211     def all_batches(self):

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in __call__(self, event_name)
    178 
    179     def ordered_cbs(self, event): return [cb for cb in self.cbs.sorted('order') if hasattr(cb, event)]
--> 180     def __call__(self, event_name): L(event_name).map(self._call_one)
    181 
    182     def _call_one(self, event_name):

/usr/local/lib/python3.11/dist-packages/fastcore/foundation.py in map(self, f, *args, **kwargs)
    166     def range(cls, a, b=None, step=None): return cls(range_of(a, b=b, step=step))
    167 
--> 168     def map(self, f, *args, **kwargs): return self._new(map_ex(self, f, *args, gen=False, **kwargs))
    169     def argwhere(self, f, negate=False, **kwargs): return self._new(argwhere(self, f, negate, **kwargs))
    170     def argfirst(self, f, negate=False):

/usr/local/lib/python3.11/dist-packages/fastcore/basics.py in map_ex(iterable, f, gen, *args, **kwargs)
    949     res = map(g, iterable)
    950     if gen: return res
--> 951     return list(res)
    952 
    953 # %% ../nbs/01_basics.ipynb

/usr/local/lib/python3.11/dist-packages/fastcore/basics.py in __call__(self, *args, **kwargs)
    934             if isinstance(v,_Arg): kwargs[k] = args.pop(v.i)
    935         fargs = [args[x.i] if isinstance(x, _Arg) else x for x in self.pargs] + args[self.maxi+1:]
--> 936         return self.func(*fargs, **kwargs)
    937 
    938 # %% ../nbs/01_basics.ipynb

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _call_one(self, event_name)
    182     def _call_one(self, event_name):
    183         if not hasattr(event, event_name): raise Exception(f'missing {event_name}')
--> 184         for cb in self.cbs.sorted('order'): cb(event_name)
    185 
    186     def _bn_bias_state(self, with_bias): return norm_bias_params(self.model, with_bias).map(self.opt.state)

/usr/local/lib/python3.11/dist-packages/fastai/callback/core.py in __call__(self, event_name)
     62             try: res = getcallable(self, event_name)()
     63             except (CancelBatchException, CancelBackwardException, CancelEpochException, CancelFitException, CancelStepException, CancelTrainException, CancelValidException): raise
---> 64             except Exception as e: raise modify_exception(e, f'Exception occured in `{self.__class__.__name__}` when calling event `{event_name}`:\n\t{e.args[0]}', replace=True)
     65         if event_name=='after_fit': self.run=True #Reset self.run to True at each end of fit
     66         return res

/usr/local/lib/python3.11/dist-packages/fastai/callback/core.py in __call__(self, event_name)
     60         res = None
     61         if self.run and _run:
---> 62             try: res = getcallable(self, event_name)()
     63             except (CancelBatchException, CancelBackwardException, CancelEpochException, CancelFitException, CancelStepException, CancelTrainException, CancelValidException): raise
     64             except Exception as e: raise modify_exception(e, f'Exception occured in `{self.__class__.__name__}` when calling event `{event_name}`:\n\t{e.args[0]}', replace=True)

/usr/local/lib/python3.11/dist-packages/fastai/callback/core.py in after_batch(self)
    140         "Save predictions, targets and potentially losses"
    141         if not hasattr(self, 'pred'): return
--> 142         preds,targs = self.learn.to_detach(self.pred),self.learn.to_detach(self.yb)
    143         if self.with_preds: self.preds.append(preds)
    144         if self.with_targs: self.targets.append(targs)

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in to_detach(self, b, cpu, gather)
    360 
    361     def to_detach(self,b,cpu=True,gather=True):
--> 362         return self.dl.to_detach(b,cpu,gather) if hasattr(getattr(self,'dl',None),'to_detach') else to_detach(b,cpu,gather)
    363 
    364     def __getstate__(self): return {k:v for k,v in self.__dict__.items() if k!='lock'}

/usr/local/lib/python3.11/dist-packages/fastai/torch_core.py in to_detach(b, cpu, gather)
    244         if gather: x = maybe_gather(x)
    245         return x.cpu() if cpu else x
--> 246     return apply(_inner, b, cpu=cpu, gather=gather)
    247 
    248 # %% ../nbs/00_torch_core.ipynb 68

/usr/local/lib/python3.11/dist-packages/fastai/torch_core.py in apply(func, x, *args, **kwargs)
    224     if is_listy(x): return type(x)([apply(func, o, *args, **kwargs) for o in x])
    225     if isinstance(x,(dict,MutableMapping)): return {k: apply(func, v, *args, **kwargs) for k,v in x.items()}
--> 226     res = func(x, *args, **kwargs)
    227     return res if x is None else retain_type(res, x)
    228 

/usr/local/lib/python3.11/dist-packages/fastai/torch_core.py in _inner(x, cpu, gather)
    243         x = x.detach()
    244         if gather: x = maybe_gather(x)
--> 245         return x.cpu() if cpu else x
    246     return apply(_inner, b, cpu=cpu, gather=gather)
    247 

RuntimeError: Exception occured in `GatherPredsCallback` when calling event `after_batch`:
	CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.


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
