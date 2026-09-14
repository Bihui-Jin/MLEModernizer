# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.7

# 2. Installed packages

fastai==2.8.5
geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 3. Data file paths

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

# 4. Code solution

## === cell 0
%matplotlib inline


## === cell 1
import os

PATH = "/kaggle/data"


## === cell 2
os.listdir(PATH)


## === cell 3
import random
import numpy as np
import torch

manual_seed = 555
random.seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)
torch.backends.cudnn.deterministic = True


## === cell 4
import pandas as pd

train_df = pd.read_csv(f"{PATH}/train.csv", nrows=100000)


## === cell 5
test_df = pd.read_csv(f'{PATH}/test.csv')


## === cell 6
print(train_df.isnull().sum())


## === cell 7
def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()


## === cell 8
train_df.head().T


## === cell 9
test_df.head().T


## === cell 10
def data_preprocessing(df):
    df = df.dropna(how='any',axis='rows')
    add_travel_vector_features(df)
    df = df[(df.abs_diff_longitude<5) & (df.abs_diff_latitude<5)]
    df = df[(df.passenger_count > 0) & (df.passenger_count <= 6)]
    df[['date','time','timezone']] = df['pickup_datetime'].str.split(expand=True)
    add_datepart(df, "date", drop=False)

    df[['hour','minute','second']] = df['time'].str.split(':',expand=True).astype('int64')
    df[['trash', 'order_no']] = df['key'].str.split('.',expand=True)
    df['order_no'] = df['order_no'].astype('int64')
    df = df.drop(['timezone','time', 'pickup_datetime','trash','date'], axis = 1)
    return df


## === cell 11
from fastai.tabular.core import add_datepart

train_df = data_preprocessing(train_df)


## === cell 12
test_df = data_preprocessing(test_df)


## === cell 13
train_df = train_df.reset_index()
test_df = test_df.reset_index()


## === cell 14
train_df.columns


## === cell 15
cat_vars = ['passenger_count', 'Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
    'Is_month_end','Is_month_start','Is_quarter_end','Is_quarter_start','Is_year_end','Is_year_start','hour','minute','second','order_no']

contin_vars = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
   'abs_diff_longitude', 'abs_diff_latitude']

dep = 'fare_amount'
n = len(train_df); n


## === cell 16
train_df = train_df[cat_vars+contin_vars+ [dep,'key']].copy()
test_df[dep] = 0
test_df = test_df[cat_vars+contin_vars+ [dep,'key']].copy()


## === cell 17
for v in cat_vars: train_df[v] = train_df[v].astype('category').cat.as_ordered()


## === cell 18
import pandas as pd


def apply_cats(to_df, from_df):
    for c in from_df.columns:
        if c in to_df.columns and pd.api.types.is_categorical_dtype(from_df[c]):
            to_df[c] = pd.Categorical(
                to_df[c],
                categories=from_df[c].cat.categories,
                ordered=from_df[c].cat.ordered,
            )


apply_cats(test_df, train_df)


## === cell 19
for v in contin_vars:
    train_df[v] = train_df[v].fillna(0).astype('float32')
    test_df[v] = test_df[v].fillna(0).astype('float32')


## === cell 20
train_df = train_df.set_index("key")


## === cell 21
import numpy as np
import pandas as pd


def proc_df(df, y_fld, do_scale=False):
    df = df.copy()

    y = None
    if y_fld is not None and y_fld in df.columns:
        y = df[y_fld].values
        df = df.drop(columns=[y_fld])

    nas = {}

    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].isna().any():
            fill_val = df[c].median()
            df[c] = df[c].fillna(fill_val)
            nas[c] = fill_val

    cat_cols = [
        c
        for c in df.columns
        if pd.api.types.is_categorical_dtype(df[c]) or df[c].dtype == object
    ]
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols, dummy_na=False)

    mapper = None
    if do_scale:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        means = df[num_cols].mean()
        stds = df[num_cols].std(ddof=0).replace(0, 1.0)
        df[num_cols] = (df[num_cols] - means) / stds
        mapper = {"means": means, "stds": stds, "num_cols": num_cols}

    return df, y, nas, mapper


df, y, nas, mapper = proc_df(train_df, "fare_amount", do_scale=True)


## === cell 22
test_df = test_df.set_index("key")


## === cell 23
import numpy as np
import pandas as pd


def proc_df(df, y_fld, do_scale=False, na_dict=None, mapper=None):
    df = df.copy()

    y = None
    if y_fld is not None and y_fld in df.columns:
        y = df[y_fld].values
        df = df.drop(columns=[y_fld])

    nas = {} if na_dict is None else dict(na_dict)
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].isna().any():
            fill_val = nas.get(c, df[c].median())
            df[c] = df[c].fillna(fill_val)
            nas[c] = fill_val

    cat_cols = [
        c
        for c in df.columns
        if pd.api.types.is_categorical_dtype(df[c]) or df[c].dtype == object
    ]
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols, dummy_na=False)

    if do_scale:
        if mapper is None:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            means = df[num_cols].mean()
            stds = df[num_cols].std(ddof=0).replace(0, 1.0)
            df[num_cols] = (df[num_cols] - means) / stds
            mapper = {"means": means, "stds": stds, "num_cols": num_cols}
        else:
            num_cols = list(mapper.get("num_cols", []))
            means = mapper["means"]
            stds = mapper["stds"]
            for c in num_cols:
                if c in df.columns:
                    df[c] = (df[c] - means[c]) / stds[c]

    if mapper is not None and "cols" in mapper:
        ref_cols = list(mapper["cols"])
        for c in ref_cols:
            if c not in df.columns:
                df[c] = 0
        extra = [c for c in df.columns if c not in ref_cols]
        if extra:
            df = df.drop(columns=extra)
        df = df[ref_cols]
    elif mapper is not None and "cols" not in mapper:
        mapper = dict(mapper)
        mapper["cols"] = df.columns

    return df, y, nas, mapper


df_test, _, nas, mapper = proc_df(
    test_df, "fare_amount", do_scale=True, mapper=mapper, na_dict=nas
)


## === cell 24
train_ratio = 0.8
train_size = int(n * train_ratio); train_size
val_idx = list(range(train_size, len(df)))


## === cell 25
y


## === cell 26
def rmse(y_pred, targ):
    pct_var = (targ - y_pred)
    return math.sqrt((pct_var**2).mean())


## === cell 27
class _ColumnarModelData:
    def __init__(
        self,
        path,
        trn_df,
        val_df,
        trn_y,
        val_y,
        cat_flds,
        bs,
        is_reg=True,
        test_df=None,
    ):
        self.path = path
        self.trn_df = trn_df
        self.val_df = val_df
        self.trn_y = trn_y
        self.val_y = val_y
        self.cat_flds = cat_flds
        self.bs = bs
        self.is_reg = is_reg
        self.test_df = test_df

    @classmethod
    def from_data_frame(
        cls, path, val_idx, df, y, cat_flds, bs, test_df=None, is_reg=True
    ):
        val_idx = np.array(list(val_idx), dtype=int)
        all_idx = np.arange(len(df), dtype=int)
        trn_mask = np.ones(len(df), dtype=bool)
        trn_mask[val_idx] = False
        trn_idx = all_idx[trn_mask]

        trn_df = df.iloc[trn_idx].copy()
        val_df = df.iloc[val_idx].copy()
        trn_y = np.asarray(y)[trn_idx]
        val_y = np.asarray(y)[val_idx]

        return cls(
            path,
            trn_df,
            val_df,
            trn_y,
            val_y,
            cat_flds=cat_flds,
            bs=bs,
            is_reg=is_reg,
            test_df=test_df,
        )

    @classmethod
    def from_data_frames(
        cls, path, trn_df, val_df, trn_y, val_y, cat_flds, bs, is_reg=True, test_df=None
    ):
        return cls(
            path,
            trn_df.copy(),
            val_df.copy(),
            np.asarray(trn_y),
            np.asarray(val_y),
            cat_flds=cat_flds,
            bs=bs,
            is_reg=is_reg,
            test_df=test_df,
        )


## === cell 28
md = _ColumnarModelData.from_data_frame(PATH, val_idx, df, y.astype(np.float32), cat_flds=cat_vars, bs=128,test_df=df_test)


## === cell 29
cat_vars


## === cell 30
cat_sz = [(c, len(train_df[c].cat.categories)+1) for c in cat_vars]


## === cell 31
y


## === cell 32
emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz];emb_szs


## === cell 33
max_y = np.max(y)
y_range = (0, max_y*1.2)


## === cell 34
TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"


## === cell 35
!ls ../input


## === cell 36
import math
import numpy as np
import torch
import torch.nn as nn
from fastai.data.core import DataLoaders
from fastai.learner import Learner
from fastai.metrics import rmse as fastai_rmse


class _ColumnarModelData:
    def __init__(
        self,
        path,
        trn_df,
        val_df,
        trn_y,
        val_y,
        cat_flds,
        bs,
        is_reg=True,
        test_df=None,
    ):
        self.path = path
        self.trn_df = trn_df
        self.val_df = val_df
        self.trn_y = trn_y
        self.val_y = val_y
        self.cat_flds = cat_flds
        self.bs = bs
        self.is_reg = is_reg
        self.test_df = test_df

    @classmethod
    def from_data_frame(
        cls, path, val_idx, df, y, cat_flds, bs, test_df=None, is_reg=True
    ):
        val_idx = np.array(list(val_idx), dtype=int)
        all_idx = np.arange(len(df), dtype=int)
        trn_mask = np.ones(len(df), dtype=bool)
        trn_mask[val_idx] = False
        trn_idx = all_idx[trn_mask]

        trn_df = df.iloc[trn_idx].copy()
        val_df = df.iloc[val_idx].copy()
        trn_y = np.asarray(y)[trn_idx]
        val_y = np.asarray(y)[val_idx]

        return cls(
            path,
            trn_df,
            val_df,
            trn_y,
            val_y,
            cat_flds=cat_flds,
            bs=bs,
            is_reg=is_reg,
            test_df=test_df,
        )

    @classmethod
    def from_data_frames(
        cls, path, trn_df, val_df, trn_y, val_y, cat_flds, bs, is_reg=True, test_df=None
    ):
        return cls(
            path,
            trn_df.copy(),
            val_df.copy(),
            np.asarray(trn_y),
            np.asarray(val_y),
            cat_flds=cat_flds,
            bs=bs,
            is_reg=is_reg,
            test_df=test_df,
        )

    def get_learner(
        self,
        emb_szs,
        n_cont,
        emb_drop,
        out_sz,
        layers,
        drops,
        y_range=None,
        tmp_name=None,
        models_name=None,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        Xtr = self.trn_df.to_numpy(dtype=np.float32)
        Xva = self.val_df.to_numpy(dtype=np.float32)
        ytr = np.asarray(self.trn_y, dtype=np.float32).reshape(-1, 1)
        yva = np.asarray(self.val_y, dtype=np.float32).reshape(-1, 1)

        tr_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(Xtr), torch.from_numpy(ytr)
        )
        va_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(Xva), torch.from_numpy(yva)
        )

        tr_dl = torch.utils.data.DataLoader(
            tr_ds, batch_size=self.bs, shuffle=True, drop_last=False
        )
        va_dl = torch.utils.data.DataLoader(
            va_ds, batch_size=self.bs, shuffle=False, drop_last=False
        )
        dls = DataLoaders(tr_dl, va_dl, device=device)

        in_sz = Xtr.shape[1]

        def _make_mlp(in_features, layer_sizes, ps, out_features, yrng):
            mods = []
            n_in = in_features
            for i, n_out in enumerate(layer_sizes):
                mods.append(nn.Linear(n_in, n_out))
                mods.append(nn.ReLU(inplace=True))
                p = ps[i] if i < len(ps) else 0.0
                if p and p > 0:
                    mods.append(nn.Dropout(p))
                n_in = n_out
            mods.append(nn.Linear(n_in, out_features))

            if yrng is None:
                return nn.Sequential(*mods)

            y_low, y_high = float(yrng[0]), float(yrng[1])

            class _RangeModel(nn.Module):
                def __init__(self, base, lo, hi):
                    super().__init__()
                    self.base = base
                    self.lo = lo
                    self.hi = hi

                def forward(self, x):
                    x = self.base(x)
                    x = torch.sigmoid(x)
                    return x * (self.hi - self.lo) + self.lo

            return _RangeModel(nn.Sequential(*mods), y_low, y_high)

        model = _make_mlp(in_sz, layers, drops, out_sz, y_range).to(device)

        learn = Learner(
            dls=dls,
            model=model,
            loss_func=nn.MSELoss(),
            metrics=[fastai_rmse],
        )
        return learn


## === cell 37
from fastai.callback.schedule import lr_find

md = _ColumnarModelData.from_data_frame(
    PATH, val_idx, df, y.astype(np.float32), cat_flds=cat_vars, bs=128, test_df=df_test
)

m = md.get_learner(
    emb_szs=emb_szs,
    n_cont=len(contin_vars),
    emb_drop=0.0,
    out_sz=1,
    layers=[200, 100],
    drops=[0.1, 0.1],
    y_range=y_range,
    tmp_name=TMP_PATH,
    models_name=MODEL_PATH,
)

lr_find(m)


## --- ERROR in cell 37, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1115755295.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     19[0m )
[1;32m     20[0m [0;34m[0m[0m
[0;32m---> 21[0;31m [0mlr_find[0m[0;34m([0m[0mm[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;31mTypeError[0m: 'NoneType' object is not callable

## === cell 38
m.sched.plot(200)
