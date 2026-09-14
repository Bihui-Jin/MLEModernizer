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
%matplotlib inline


## === cell 1
from fastai.structured import *
from fastai.column_data import *
PATH = '../input'


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/601751806.py in <cell line: 0>()
----> 1 from fastai.structured import *
      2 from fastai.column_data import *
      3 # np.set_printoptions(threshold=50, edgeitems=20)
      4 PATH = '../input'

ModuleNotFoundError: No module named 'fastai.structured'

## === cell 2
os.listdir(PATH)


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/406056175.py in <cell line: 0>()
----> 1 os.listdir(PATH)

NameError: name 'os' is not defined

## === cell 3
manual_seed = 555
random.seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)
torch.backends.cudnn.deterministic = True


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1808669564.py in <cell line: 0>()
      1 manual_seed = 555
----> 2 random.seed(manual_seed)
      3 np.random.seed(manual_seed)
      4 torch.manual_seed(manual_seed)
      5 torch.cuda.manual_seed_all(manual_seed)

NameError: name 'random' is not defined

## === cell 4
train_df = pd.read_csv(f'{PATH}/train.csv', nrows=100000)


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2108707868.py in <cell line: 0>()
----> 1 train_df = pd.read_csv(f'{PATH}/train.csv', nrows=100000)

NameError: name 'pd' is not defined

## === cell 5
test_df = pd.read_csv(f'{PATH}/test.csv')


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1224432971.py in <cell line: 0>()
----> 1 test_df = pd.read_csv(f'{PATH}/test.csv')

NameError: name 'pd' is not defined

## === cell 6
print(train_df.isnull().sum())


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3174598079.py in <cell line: 0>()
----> 1 print(train_df.isnull().sum())

NameError: name 'train_df' is not defined

## === cell 7
def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()


## === cell 8
train_df.head().T


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/505331435.py in <cell line: 0>()
----> 1 train_df.head().T

NameError: name 'train_df' is not defined

## === cell 9
test_df.head().T


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3688550299.py in <cell line: 0>()
----> 1 test_df.head().T

NameError: name 'test_df' is not defined

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
train_df = data_preprocessing(train_df)


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/13996680.py in <cell line: 0>()
----> 1 train_df = data_preprocessing(train_df)

NameError: name 'train_df' is not defined

## === cell 12
test_df = data_preprocessing(test_df)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4115518886.py in <cell line: 0>()
----> 1 test_df = data_preprocessing(test_df)

NameError: name 'test_df' is not defined

## === cell 13
train_df = train_df.reset_index()
test_df = test_df.reset_index()


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2667808009.py in <cell line: 0>()
----> 1 train_df = train_df.reset_index()
      2 test_df = test_df.reset_index()

NameError: name 'train_df' is not defined

## === cell 14
train_df.columns


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1723399785.py in <cell line: 0>()
----> 1 train_df.columns

NameError: name 'train_df' is not defined

## === cell 15
cat_vars = ['passenger_count', 'Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
    'Is_month_end','Is_month_start','Is_quarter_end','Is_quarter_start','Is_year_end','Is_year_start','hour','minute','second','order_no']

contin_vars = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
   'abs_diff_longitude', 'abs_diff_latitude']

dep = 'fare_amount'
n = len(train_df); n


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2311643485.py in <cell line: 0>()
      6 
      7 dep = 'fare_amount'
----> 8 n = len(train_df); n

NameError: name 'train_df' is not defined

## === cell 16
train_df = train_df[cat_vars+contin_vars+ [dep,'key']].copy()
test_df[dep] = 0
test_df = test_df[cat_vars+contin_vars+ [dep,'key']].copy()


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/287855881.py in <cell line: 0>()
----> 1 train_df = train_df[cat_vars+contin_vars+ [dep,'key']].copy()
      2 test_df[dep] = 0
      3 test_df = test_df[cat_vars+contin_vars+ [dep,'key']].copy()

NameError: name 'train_df' is not defined

## === cell 17
for v in cat_vars: train_df[v] = train_df[v].astype('category').cat.as_ordered()


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/992632115.py in <cell line: 0>()
----> 1 for v in cat_vars: train_df[v] = train_df[v].astype('category').cat.as_ordered()

NameError: name 'train_df' is not defined

## === cell 18
apply_cats(test_df, train_df)


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/442486857.py in <cell line: 0>()
      1 # for v in cat_vars: test_df[v] = test_df[v].astype('category').cat.as_ordered()
----> 2 apply_cats(test_df, train_df)

NameError: name 'apply_cats' is not defined

## === cell 19
for v in contin_vars:
    train_df[v] = train_df[v].fillna(0).astype('float32')
    test_df[v] = test_df[v].fillna(0).astype('float32')


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/832043971.py in <cell line: 0>()
      1 for v in contin_vars:
----> 2     train_df[v] = train_df[v].fillna(0).astype('float32')
      3     test_df[v] = test_df[v].fillna(0).astype('float32')

NameError: name 'train_df' is not defined

## === cell 20
train_df = train_df.set_index("key")


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1975308616.py in <cell line: 0>()
----> 1 train_df = train_df.set_index("key")

NameError: name 'train_df' is not defined

## === cell 21
df, y, nas, mapper = proc_df(train_df, 'fare_amount', do_scale=True)


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3704674369.py in <cell line: 0>()
----> 1 df, y, nas, mapper = proc_df(train_df, 'fare_amount', do_scale=True)

NameError: name 'proc_df' is not defined

## === cell 22
test_df = test_df.set_index("key")


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3660799496.py in <cell line: 0>()
----> 1 test_df = test_df.set_index("key")

NameError: name 'test_df' is not defined

## === cell 23
df_test, _, nas, mapper = proc_df(test_df, 'fare_amount', do_scale=True,
                                  mapper=mapper, na_dict=nas)


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2504821635.py in <cell line: 0>()
----> 1 df_test, _, nas, mapper = proc_df(test_df, 'fare_amount', do_scale=True,
      2                                   mapper=mapper, na_dict=nas)

NameError: name 'proc_df' is not defined

## === cell 24
train_ratio = 0.8
train_size = int(n * train_ratio); train_size
val_idx = list(range(train_size, len(df)))


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2511726415.py in <cell line: 0>()
      1 # train_ratio = 0.75
      2 train_ratio = 0.8
----> 3 train_size = int(n * train_ratio); train_size
      4 val_idx = list(range(train_size, len(df)))

NameError: name 'n' is not defined

## === cell 25
y


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/859018229.py in <cell line: 0>()
----> 1 y

NameError: name 'y' is not defined

## === cell 26
def rmse(y_pred, targ):
    pct_var = (targ - y_pred)
    return math.sqrt((pct_var**2).mean())


## === cell 27
class _ColumnarModelData(ColumnarModelData):
    @classmethod
    def from_data_frames(cls, path, trn_df, val_df, trn_y, val_y, cat_flds, bs, is_reg, test_df=None):
        test_ds = ColumnarDataset.from_data_frame(test_df, cat_flds, None, is_reg) if test_df is not None else None
        return cls(path, ColumnarDataset.from_data_frame(trn_df, cat_flds, trn_y, is_reg),
                    ColumnarDataset.from_data_frame(val_df, cat_flds, val_y, is_reg), bs, test_ds=test_ds)


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2758501431.py in <cell line: 0>()
----> 1 class _ColumnarModelData(ColumnarModelData):
      2     @classmethod
      3     def from_data_frames(cls, path, trn_df, val_df, trn_y, val_y, cat_flds, bs, is_reg, test_df=None):
      4         test_ds = ColumnarDataset.from_data_frame(test_df, cat_flds, None, is_reg) if test_df is not None else None
      5         return cls(path, ColumnarDataset.from_data_frame(trn_df, cat_flds, trn_y, is_reg),

NameError: name 'ColumnarModelData' is not defined

## === cell 28
md = _ColumnarModelData.from_data_frame(PATH, val_idx, df, y.astype(np.float32), cat_flds=cat_vars, bs=128,test_df=df_test)


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2854352585.py in <cell line: 0>()
----> 1 md = _ColumnarModelData.from_data_frame(PATH, val_idx, df, y.astype(np.float32), cat_flds=cat_vars, bs=128,test_df=df_test)

NameError: name '_ColumnarModelData' is not defined

## === cell 29
cat_vars


## === cell 30
cat_sz = [(c, len(train_df[c].cat.categories)+1) for c in cat_vars]


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2777861354.py in <cell line: 0>()
----> 1 cat_sz = [(c, len(train_df[c].cat.categories)+1) for c in cat_vars]

/tmp/ipykernel_11/2777861354.py in <listcomp>(.0)
----> 1 cat_sz = [(c, len(train_df[c].cat.categories)+1) for c in cat_vars]

NameError: name 'train_df' is not defined

## === cell 31
y


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/859018229.py in <cell line: 0>()
----> 1 y

NameError: name 'y' is not defined

## === cell 32
emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz];emb_szs


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4284066683.py in <cell line: 0>()
----> 1 emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz];emb_szs

NameError: name 'cat_sz' is not defined

## === cell 33
max_y = np.max(y)
y_range = (0, max_y*1.2)


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1178014383.py in <cell line: 0>()
----> 1 max_y = np.max(y)
      2 y_range = (0, max_y*1.2)

NameError: name 'np' is not defined

## === cell 34
TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"


## === cell 35
!ls ../input


## === cell 36
m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars),
                   0.04, 1, [1000,500,100], [0.008,0.08, 0.01], y_range=y_range,tmp_name=TMP_PATH,models_name=MODEL_PATH)
m.summary()


## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3960899629.py in <cell line: 0>()
----> 1 m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars),
      2                    0.04, 1, [1000,500,100], [0.008,0.08, 0.01], y_range=y_range,tmp_name=TMP_PATH,models_name=MODEL_PATH)
      3 # criterion = nn.MSELoss()
      4 # m.crit = criterion
      5 m.summary()

NameError: name 'md' is not defined

## === cell 37
m.lr_find()


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/333574313.py in <cell line: 0>()
----> 1 m.lr_find()

NameError: name 'm' is not defined

## === cell 38
m.sched.plot(200)


## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2631736155.py in <cell line: 0>()
----> 1 m.sched.plot(200)

NameError: name 'm' is not defined

## === cell 39
m.sched.plot_lr()


## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1099990443.py in <cell line: 0>()
----> 1 m.sched.plot_lr()

NameError: name 'm' is not defined

## === cell 40
lr = 1e-3


## === cell 41
m.fit(lr, 3, metrics=[rmse])


## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/613681584.py in <cell line: 0>()
----> 1 m.fit(lr, 3, metrics=[rmse])

NameError: name 'm' is not defined

## === cell 42
m.fit(lr, 3, cycle_len=1, metrics=[rmse])


## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1665199188.py in <cell line: 0>()
----> 1 m.fit(lr, 3, cycle_len=1, metrics=[rmse])

NameError: name 'm' is not defined

## === cell 43
m.fit(lr, 3, cycle_len=1, cycle_mult=2)


## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2263219429.py in <cell line: 0>()
----> 1 m.fit(lr, 3, cycle_len=1, cycle_mult=2)

NameError: name 'm' is not defined

## === cell 44
pred_test=m.predict()


## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3556118254.py in <cell line: 0>()
----> 1 pred_test=m.predict()

NameError: name 'm' is not defined

## === cell 45
len(pred_test)


## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3120790120.py in <cell line: 0>()
----> 1 len(pred_test)

NameError: name 'pred_test' is not defined

## === cell 46
len(y[val_idx])


## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3584948629.py in <cell line: 0>()
----> 1 len(y[val_idx])

NameError: name 'y' is not defined

## === cell 47
y[:20]


## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4180743950.py in <cell line: 0>()
----> 1 y[:20]

NameError: name 'y' is not defined

## === cell 48
y_test = m.predict(True)


## --- ERROR in cell 48, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2584256468.py in <cell line: 0>()
----> 1 y_test = m.predict(True)

NameError: name 'm' is not defined

## === cell 49
y_test = y_test.reshape(-1)


## --- ERROR in cell 49, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3333553687.py in <cell line: 0>()
----> 1 y_test = y_test.reshape(-1)

NameError: name 'y_test' is not defined

## === cell 50
submission = pd.DataFrame(
    {'key': test_df.index, 'fare_amount': y_test},
    columns = ['key', 'fare_amount'])
submission.to_csv('submission.csv', index = False)


## --- ERROR in cell 50, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2112526989.py in <cell line: 0>()
----> 1 submission = pd.DataFrame(
      2     {'key': test_df.index, 'fare_amount': y_test},
      3     columns = ['key', 'fare_amount'])
      4 submission.to_csv('submission.csv', index = False)

NameError: name 'pd' is not defined

## === cell 51
test_df.index


## --- ERROR in cell 51, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1176341717.py in <cell line: 0>()
----> 1 test_df.index

NameError: name 'test_df' is not defined
