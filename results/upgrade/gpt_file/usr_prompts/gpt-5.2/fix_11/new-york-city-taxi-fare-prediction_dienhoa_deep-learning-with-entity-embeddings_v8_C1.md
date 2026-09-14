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

12.80367

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 12.80367) has done: 'I fix the `IndexError: index out of range in self` during test prediction by ensuring train/test categorical columns share the same category mapping (the current code creates separate `category` codes, causing unseen codes at inference). I do this with a minimal change: explicitly align `test_df` categories to `train_df` categories before building the FastAI `TabularPandas`. I also remove the device-switch hack and run `get_preds` normally (FastAI handles device), which avoids subtle dataloader/model device mismatches. Finally, I make sure the submission is always written to `submission.csv` with the required columns.'

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
    test_df[v] = test_df[v].cat.set_categories(train_df[v].cat.categories)

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
max_y = float(train_df[dep].max())
y_range = (0.0, max_y * 1.2)
max_y, y_range



## === cell 24
learn = tabular_learner(
    dls, layers=[1000, 500, 100], y_range=y_range, metrics=rmse_fastai
)



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



## === cell 31
submission = pd.DataFrame(
    {"key": test_df.index.values, "fare_amount": y_test},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission.csv", index=False)
submission.head()



## === cell 32
assert (
    submission.shape[0] == test_df.shape[0]
), "Submission rows do not match test rows."
assert list(submission.columns) == [
    "key",
    "fare_amount",
], "Submission columns incorrect."
os.path.getsize("submission.csv"), submission.shape
