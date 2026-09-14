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
Given time series of breaths, predict the airway pressure in the respiratory circuit during the breath, given the time series of control inputs.

The best submissions will take lung attributes compliance and resistance into account.

## Metric
Mean absolute error between the predicted and actual pressures during the inspiratory phase of each breath. The expiratory phase is not scored.

## Submission Format
For each `id` in the test set, you must predict a value for the `pressure` variable. The file should contain a header and have the following format:

```
id,pressure
1,20
2,23
3,24
etc.
```

## Dataset
The ventilator data used in this competition was produced using a modified [open-source ventilator](https://pvp.readthedocs.io/) connected to an [artificial bellows test lung](https://www.ingmarmed.com/product/quicklung/) via a respiratory circuit. The diagram below illustrates the setup, with the two control inputs highlighted in green and the state variable (airway pressure) to predict in blue. The first control input is a continuous variable from 0 to 100 representing the percentage the inspiratory solenoid valve is open to let air into the lung (i.e., 0 is completely closed and no air is let in and 100 is completely open). The second control input is a binary variable representing whether the exploratory valve is open (1) or closed (0) to let air out.

![Ventilator diagram](https://raw.githubusercontent.com/google/deluca-lung/main/assets/2020-10-02%20Ventilator%20diagram.svg)

Each time series represents an approximately 3-second breath. The files are organized such that each row is a time step in a breath and gives the two control signals, the resulting airway pressure, and relevant attributes of the lung, described below.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `id` - globally-unique time step identifier across an entire file
- `breath_id` - globally-unique time step for breaths
- `R` - lung attribute indicating how restricted the airway is (in cmH2O/L/S). Physically, this is the change in pressure per change in flow (air volume per time). Intuitively, one can imagine blowing up a balloon through a straw. We can change `R` by changing the diameter of the straw, with higher `R` being harder to blow.
- `C` - lung attribute indicating how compliant the lung is (in mL/cmH2O). Physically, this is the change in volume per change in pressure. Intuitively, one can imagine the same balloon example. We can change `C` by changing the thickness of the balloon’s latex, with higher `C` having thinner latex and easier to blow.
- `time_step` - the actual time stamp.
- `u_in` - the control input for the inspiratory solenoid valve. Ranges from 0 to 100.
- `u_out` - the control input for the exploratory solenoid valve. Either 0 or 1.
- `pressure` - the airway pressure measured in the respiratory circuit, measured in cmH2O.

# 2. Python version

3.10

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
tqdm==4.67.1
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (82 lines)
            sample_submission.csv (603601 lines)
            sample_submission.csv.zip (1.3 MB)
            test.csv (603601 lines)
            test.csv.zip (8.5 MB)
            train.csv (5432401 lines)
            train.csv.zip (93.8 MB)
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
        input/
            description.md (82 lines)
            sample_submission.csv (603601 lines)
            sample_submission.csv.zip (1.3 MB)
            test.csv (603601 lines)
            test.csv.zip (8.5 MB)
            train.csv (5432401 lines)
            train.csv.zip (93.8 MB)
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
        working/
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
```

-> data/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> data/test.csv has 603600 rows and 7 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [20, 50, 10]
R (int64) has 3 unique values: [50, 5, 20]
breath_id (int64) has range: 2436.00 - 124535.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/train.csv has 5432400 rows and 8 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [10, 20, 50]
R (int64) has 3 unique values: [5, 50, 20]
breath_id (int64) has range: 1194.00 - 124492.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (float64) has range: 3.52 - 41.48, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/ventilator-pressure-prediction/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> data/ventilator-pressure-prediction/test.csv has 603600 rows and 7 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [20, 50, 10]
R (int64) has 3 unique values: [50, 5, 20]
breath_id (int64) has range: 2436.00 - 124535.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/ventilator-pressure-prediction/train.csv has 5432400 rows and 8 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [10, 20, 50]
R (int64) has 3 unique values: [5, 50, 20]
breath_id (int64) has range: 1194.00 - 124492.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (float64) has range: 3.52 - 41.48, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> input/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> (stopped after 10 files for performance)

# 5. Target score

1.1526

# 6. Current score

4.26769

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 4.26769) has done: 'The crash happens because `pd.read_csv(..., dtype={4:...,5:...,6:...,7:...})` uses integer keys, which pandas interprets as positional column indices; after `index_col=0`, the test file has only 6 remaining columns (0–5), so key `7` is out of range and triggers `IndexError`. The minimal fix is to specify dtypes by column *names* instead of positions, keeping the same float32 intent for the numeric columns. This preserves the same loaded DataFrames (`train_data`, `test_data`, `sample`) expected by later cells and does not change any downstream modeling logic. No other cells need modification.'

# 9. Code solution

## === cell 0
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


## === cell 1
train_data = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv",
    index_col=0,
    dtype={
        "time_step": np.float32,
        "u_in": np.float32,
        "u_out": np.float32,
        "pressure": np.float32,
    },
)
test_data = pd.read_csv(
    "../input/ventilator-pressure-prediction/test.csv",
    index_col=0,
    dtype={"time_step": np.float32, "u_in": np.float32, "u_out": np.float32},
)
sample = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")


## === cell 2
train_data.head()


## === cell 3
train_data.info()


## === cell 4
test_data.head()


## === cell 5
test_data.info()


## === cell 6
train_data.isnull().sum().to_frame()


## === cell 7
test_data.isnull().sum().to_frame()


## === cell 8
breath_one = train_data[train_data['breath_id'] == 3928].reset_index(drop=True)
breath_one


## === cell 9
breath_one.nunique().to_frame()


## === cell 10
fig,axes = plt.subplots(3,1,figsize=(12,15))
sns.lineplot(x='time_step',y='u_in',data=breath_one,ax=axes[0])
axes[0].set_title("u_in")
sns.lineplot(x='time_step',y='u_out',data=breath_one,ax=axes[1])
axes[1].set_title("u_out")
sns.lineplot(x='time_step',y='pressure',data=breath_one,ax=axes[2])
axes[2].set_title("pressure")


## === cell 11
breath_one.describe()


## === cell 12
train_data.R.value_counts().to_frame()


## === cell 13
train_data.C.value_counts().to_frame()


## === cell 14
train_data.describe()


## === cell 15
fig,axes = plt.subplots(1,1,figsize=(10,5))
sns.histplot(data=train_data,x="pressure",ax=axes)


## === cell 16
idxmax_time_step = train_data.groupby('breath_id')['time_step'].idxmax()
last_value_u_in = train_data.loc[idxmax_time_step, ['breath_id','u_in']]
last_value_u_in.columns = ['breath_id','last_value_u_in']

train_data = train_data.merge(last_value_u_in, on='breath_id')
train_data


## === cell 17
idxmax_time_step = test_data.groupby('breath_id')['time_step'].idxmax()
last_value_u_in = test_data.loc[idxmax_time_step, ['breath_id','u_in']]
last_value_u_in.columns = ['breath_id','last_value_u_in']

test_data = test_data.merge(last_value_u_in, on='breath_id')
test_data


## === cell 18
mean_u_in = train_data.groupby('breath_id')['u_in'].mean().to_frame()
mean_u_in.columns = ['mean_value_u_in']
train_data = train_data.merge(mean_u_in,on='breath_id')


## === cell 19
train_data


## === cell 20
mean_u_in = test_data.groupby('breath_id')['u_in'].mean().to_frame()
mean_u_in.columns = ['mean_value_u_in']
test_data = test_data.merge(mean_u_in,on='breath_id')
test_data


## === cell 21
train_data['diff_u_in'] = train_data.groupby('breath_id')['u_in'].diff()


## === cell 22
train_data = train_data.fillna(0)
train_data


## === cell 23
test_data['diff_u_in'] = test_data.groupby('breath_id')['u_in'].diff()
test_data = test_data.fillna(0)
test_data


## === cell 24
train_data['diff_diff_u_in'] = train_data.groupby('breath_id')['diff_u_in'].diff()
train_data = train_data.fillna(0)
train_data


## === cell 25
test_data['diff_diff_u_in'] = test_data.groupby('breath_id')['diff_u_in'].diff()
test_data = test_data.fillna(0)
test_data


## === cell 26
train_data['u_in_cumsum'] = (train_data['u_in']).groupby(train_data['breath_id']).cumsum()
test_data['u_in_cumsum'] = (test_data['u_in']).groupby(test_data['breath_id']).cumsum()


## === cell 27
sum_u_in = train_data.groupby('breath_id')['u_in'].sum().to_frame()
sum_u_in.columns = ['sum_value_u_in']
train_data = train_data.merge(sum_u_in,on='breath_id')


## === cell 28
sum_u_in = test_data.groupby('breath_id')['u_in'].sum().to_frame()
sum_u_in.columns = ['sum_value_u_in']
test_data = test_data.merge(sum_u_in,on='breath_id')


## === cell 29
train_data["u_in_cumsum_rate"] = train_data["u_in_cumsum"] / train_data["sum_value_u_in"]
test_data["u_in_cumsum_rate"] = test_data["u_in_cumsum"] / test_data["sum_value_u_in"]


## === cell 30
train_data[train_data["sum_value_u_in"] == 0]


## === cell 31
test_data[test_data["sum_value_u_in"] == 0]


## === cell 32
train_data[train_data["breath_id"] == 3928]


## === cell 33
train_data = train_data.fillna(0)
test_data = test_data.fillna(0)


## === cell 34
train_data['lag_u_in'] = train_data.groupby('breath_id')['u_in'].shift(1)
train_data = train_data.fillna(0)

test_data['lag_u_in'] = test_data.groupby('breath_id')['u_in'].shift(1)
test_data = test_data.fillna(0)

train_data['lag_2_u_in'] = train_data.groupby('breath_id')['u_in'].shift(2)
train_data = train_data.fillna(0)
test_data['lag_2_u_in'] = test_data.groupby('breath_id')['u_in'].shift(2)
test_data = test_data.fillna(0)


## === cell 35
train_data['lag_-1_u_in'] = train_data.groupby('breath_id')['u_in'].shift(-1)
train_data = train_data.fillna(0)
test_data['lag_-1_u_in'] = test_data.groupby('breath_id')['u_in'].shift(-1)
test_data = test_data.fillna(0)

train_data['lag_-2_u_in'] = train_data.groupby('breath_id')['u_in'].shift(-2)
train_data = train_data.fillna(0)
test_data['lag_-2_u_in'] = test_data.groupby('breath_id')['u_in'].shift(-2)
test_data = test_data.fillna(0)


## === cell 36
train_data['lag_-3_u_in'] = train_data.groupby('breath_id')['u_in'].shift(-3)
train_data = train_data.fillna(0)
test_data['lag_-3_u_in'] = test_data.groupby('breath_id')['u_in'].shift(-3)
test_data = test_data.fillna(0)

train_data['lag_3_u_in'] = train_data.groupby('breath_id')['u_in'].shift(3)
train_data = train_data.fillna(0)
test_data['lag_3_u_in'] = test_data.groupby('breath_id')['u_in'].shift(3)
test_data = test_data.fillna(0)


## === cell 37
train_data["max_u_in_breathid"] = train_data.groupby("breath_id")["u_in"].transform("max")
test_data["max_u_in_breathid"] = test_data.groupby("breath_id")["u_in"].transform("max")

train_data["R*C"] = train_data['R'] * train_data['C']
test_data['R*C'] = test_data['R'] * test_data['C']

train_data['breath_id__u_in__min'] = train_data.groupby(['breath_id'])['u_in'].transform('min')
test_data['breath_id__u_in__min'] = test_data.groupby(['breath_id'])['u_in'].transform('min')

train_data['breath_id__u_in__diffmax'] = train_data.groupby(['breath_id'])['u_in'].transform('max') - train_data['u_in']
train_data['breath_id__u_in__diffmean'] = train_data.groupby(['breath_id'])['u_in'].transform('mean') - train_data['u_in']

test_data['breath_id__u_in__diffmax'] = test_data.groupby(['breath_id'])['u_in'].transform('max') - test_data['u_in']
test_data['breath_id__u_in__diffmean'] = test_data.groupby(['breath_id'])['u_in'].transform('mean') - test_data['u_in']

train_data['u_in_partition_out_sum'] = train_data.groupby(['breath_id',"u_out"])['u_in'].transform("sum")
test_data['u_in_partition_out_sum'] = test_data.groupby(['breath_id',"u_out"])['u_in'].transform("sum")

train_data['area'] = train_data['time_step'] * train_data['u_in']
train_data['area'] = train_data.groupby('breath_id')['area'].cumsum()
test_data['area'] = test_data['time_step'] * test_data['u_in']
test_data['area'] = test_data.groupby('breath_id')['area'].cumsum()


## === cell 38
GRAPH = True
if(GRAPH):
    sample_train = train_data.sample(frac=0.001)
    sample_train = sample_train[sample_train["u_out"] == 0]
    fig,axes = plt.subplots(3,7,figsize=(25,15))
    sns.scatterplot(data=sample_train,x='last_value_u_in',y='pressure',ax=axes[0][0])
    sns.scatterplot(data=sample_train,x='mean_value_u_in',y='pressure',ax=axes[0][1])
    sns.scatterplot(data=sample_train,x='diff_u_in',y='pressure',ax=axes[0][2])
    sns.scatterplot(data=sample_train,x='u_in_cumsum',y='pressure',ax=axes[0][3])
    sns.scatterplot(data=sample_train,x='time_step',y='pressure',ax=axes[0][4])
    sns.scatterplot(data=sample_train,x='diff_diff_u_in',y='pressure',ax=axes[0][5])
    sns.scatterplot(data=sample_train,x='sum_value_u_in',y='pressure',ax=axes[0][6])
    sns.scatterplot(data=sample_train,x='u_in_cumsum_rate',y='pressure',ax=axes[1][0])
    sns.scatterplot(data=sample_train,x='lag_u_in',y='pressure',ax=axes[1][1])
    sns.scatterplot(data=sample_train,x='lag_2_u_in',y='pressure',ax=axes[1][2])
    sns.scatterplot(data=sample_train,x='max_u_in_breathid',y='pressure',ax=axes[1][3])
    sns.scatterplot(data=sample_train,x='R*C',y='pressure',ax=axes[1][4])
    sns.scatterplot(data=sample_train,x='lag_-3_u_in',y='pressure',ax=axes[1][5])
    sns.scatterplot(data=sample_train,x='lag_3_u_in',y='pressure',ax=axes[1][6])
    sns.scatterplot(data=sample_train,x='breath_id__u_in__min',y='pressure',ax=axes[2][0])
    sns.scatterplot(data=sample_train,x='breath_id__u_in__diffmax',y='pressure',ax=axes[2][1])
    sns.scatterplot(data=sample_train,x='breath_id__u_in__diffmean',y='pressure',ax=axes[2][2])
    sns.scatterplot(data=sample_train,x='u_in_partition_out_sum',y='pressure',ax=axes[2][3])
    sns.scatterplot(data=sample_train,x='area',y='pressure',ax=axes[2][4])


## === cell 39
GRAPH = True
if(GRAPH):
    sample_train = train_data.sample(frac=0.001)
    sample_train = sample_train[sample_train["u_out"] == 1]
    fig,axes = plt.subplots(3,7,figsize=(25,15))
    sns.scatterplot(data=sample_train,x='last_value_u_in',y='pressure',ax=axes[0][0])
    sns.scatterplot(data=sample_train,x='mean_value_u_in',y='pressure',ax=axes[0][1])
    sns.scatterplot(data=sample_train,x='diff_u_in',y='pressure',ax=axes[0][2])
    sns.scatterplot(data=sample_train,x='u_in_cumsum',y='pressure',ax=axes[0][3])
    sns.scatterplot(data=sample_train,x='time_step',y='pressure',ax=axes[0][4])
    sns.scatterplot(data=sample_train,x='diff_diff_u_in',y='pressure',ax=axes[0][5])
    sns.scatterplot(data=sample_train,x='sum_value_u_in',y='pressure',ax=axes[0][6])
    sns.scatterplot(data=sample_train,x='u_in_cumsum_rate',y='pressure',ax=axes[1][0])
    sns.scatterplot(data=sample_train,x='lag_u_in',y='pressure',ax=axes[1][1])
    sns.scatterplot(data=sample_train,x='lag_2_u_in',y='pressure',ax=axes[1][2])
    sns.scatterplot(data=sample_train,x='max_u_in_breathid',y='pressure',ax=axes[1][3])
    sns.scatterplot(data=sample_train,x='R*C',y='pressure',ax=axes[1][4])
    sns.scatterplot(data=sample_train,x='lag_-3_u_in',y='pressure',ax=axes[1][5])
    sns.scatterplot(data=sample_train,x='lag_3_u_in',y='pressure',ax=axes[1][6])
    sns.scatterplot(data=sample_train,x='breath_id__u_in__min',y='pressure',ax=axes[2][0])
    sns.scatterplot(data=sample_train,x='breath_id__u_in__diffmax',y='pressure',ax=axes[2][1])
    sns.scatterplot(data=sample_train,x='breath_id__u_in__diffmean',y='pressure',ax=axes[2][2])
    sns.scatterplot(data=sample_train,x='u_in_partition_out_sum',y='pressure',ax=axes[2][3])
    sns.scatterplot(data=sample_train,x='area',y='pressure',ax=axes[2][4])


## === cell 40
del fig
del axes
del sample_train


## === cell 41
import gc
gc.collect()


## === cell 42
train_data["train_test"] = "train"
test_data["train_test"] = "test"

train_test_all = pd.concat([train_data,test_data],axis=0)

del train_data
del test_data
gc.collect()


## === cell 43
train_test_all


## === cell 44
train_test_all['R_C'] = [f'{r}_{c}' for r, c in zip(train_test_all['R'], train_test_all['C'])]


## === cell 45
train_test_all.info()


## === cell 46
train_test_all = pd.get_dummies(train_test_all,columns=["R_C"])
train_test_all.columns


## === cell 47
train_test_all['time_diff']=train_test_all.time_step.diff().fillna(0)


## === cell 48
train_data = train_test_all[train_test_all["train_test"] == "train"]
test_data = train_test_all[train_test_all["train_test"] == "test"]


## === cell 49
del train_test_all
gc.collect()


## === cell 50
LM = True
u_out_zero_only = False ## if train from only u_out=0 data


## === cell 51
if(u_out_zero_only):
    train_data = train_data[train_data["u_out"] == 0]
    train_data = train_data.reset_index(drop=True)
X_train = train_data.drop(["pressure","breath_id","train_test"],axis=1)
y_train = train_data['pressure']
X_test = test_data.drop(["pressure","breath_id","train_test"],axis=1)

if(LM):
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_std = scaler.transform(X_train)


    lm = LinearRegression().fit(X_train_std, y_train)
    print("coefficient of determination = ",lm.score(X_train_std, y_train))


    
    X_test_std = scaler.transform(X_test)
    sample['pressure'] = lm.predict(X_test_std)

    sample.to_csv("submission_lm.csv",index=False)


## === cell 52
if(LM):
    insample_result = pd.DataFrame()
    insample_result['correct'] = y_train
    insample_result['result'] = lm.predict(X_train_std)

    fig,axes = plt.subplots(1,1,figsize=(10,10))
    sns.scatterplot(data=insample_result,x='correct',y='result',ax=axes)

    x = np.linspace(0, 60, 10)
    y = x
    axes.plot(x, y, color = "r")


## === cell 53
if(LM):
    insample_MSE = mean_absolute_error(insample_result['correct'],insample_result['result'])
    print(insample_MSE)


## === cell 54
if(LM):
    del insample_result
    del fig
    del axes
    del X_train_std
    del X_test_std

del test_data


## === cell 55
NEW_GBM = False


## === cell 56
!pip install lightgbm


## === cell 57
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import os
import numpy as np
import time
import lightgbm as lgb

from sklearn.model_selection import GroupKFold 
from sklearn.model_selection import  KFold
from sklearn import metrics


## === cell 58
y_train


## === cell 59
gbm_val_result = pd.DataFrame()
gbm_val_result['correct'] = y_train


## === cell 60
if(NEW_GBM):
    scores = []
    feature_importance = pd.DataFrame()
    columns = [col for col in train_data.columns if col not in ['id', 'breath_id', 'pressure',"train_test"]]

    models = []
    X = X_train
    y = y_train

    del X_train
    del y_train

    params = {'objective': 'regression',
              'learning_rate': 0.25,
              "boosting_type": "gbdt",
              'min_data_in_leaf':600,
              'max_bin': 196,
              'feature_fraction':0.4,
              'lambda_l1':36, 'lambda_l2':80,
              'max_depth':16,
              'num_leaves':1000,
              "metric": 'mae',
              'n_jobs': -1
             }
    folds = GroupKFold(n_splits=5)
    for fold_n, (train_index, valid_index) in enumerate(folds.split(train_data, y, groups=train_data['breath_id'])):
        print(f'Fold {fold_n} started at {time.ctime()}')
        X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        model = lgb.LGBMRegressor(**params, n_estimators=8000)
        model.fit(X_train, y_train, 
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                verbose=100, early_stopping_rounds=10)
        score = metrics.mean_absolute_error(y_valid, model.predict(X_valid))

        models.append(model)
        scores.append(score)

        y_pred = model.predict(X_valid)

        gbm_val_result.loc[valid_index,["result"]] = y_pred #for scatterplot



        fold_importance = pd.DataFrame()
        fold_importance["feature"] = columns
        fold_importance["importance"] = model.feature_importances_
        fold_importance["fold"] = fold_n + 1
        feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))


## === cell 61
if(NEW_GBM):
    for model in models:
        sample['pressure'] += model.predict(X_test)
    sample['pressure'] /= 5

    sample.to_csv('submission.csv', index=False)
if(NEW_GBM):
    fig,axes = plt.subplots(1,1,figsize=(10,10))
    sns.scatterplot(data=gbm_val_result,x='correct',y='result',ax=axes)

    x = np.linspace(0, 60, 10)
    y = x
    axes.plot(x, y, color = "r")


## === cell 62
from sklearn.model_selection import GridSearchCV, StratifiedKFold, GroupKFold, KFold, train_test_split
from tqdm import tqdm_notebook as tqdm
import lightgbm as lgb
groups = train_data["breath_id"]


## === cell 63
OLD_GBM = True
if(OLD_GBM):
    scores = []
    importance = []
    y_pred_test = np.zeros(len(X_test)) #array for predict value
    gkf = GroupKFold(n_splits=5)

    for i, (train_ix, test_ix) in tqdm(enumerate(gkf.split(X_train, y_train, groups))):

        X_train_, y_train_, groups_train_ = X_train.iloc[train_ix], y_train.iloc[train_ix], groups[train_ix]
        X_val, y_val, groups_val = X_train.iloc[test_ix], y_train.iloc[test_ix], groups[test_ix]

        print('Train Groups', np.unique(groups_train_))
        print('Val Groups', np.unique(groups_val))
        print(X_train_.shape, X_val.shape)

        model = lgb.LGBMRegressor(random_state=71, importance_type='gain')

        model.fit(X_train_, y_train_)
        y_pred = model.predict(X_val)

        gbm_val_result.loc[test_ix,["result"]] = y_pred #for scatterplot

        y_pred_test += model.predict(X_test) # add predict value

        score =  mean_absolute_error(y_val, y_pred)
        scores.append(score) 

        importance_df = pd.DataFrame(model.feature_importances_, index = X_test.columns, columns=['importance'])
        importance.append(importance_df)

        print('CV Score of Fold_%d is %f' % (i, score))


## === cell 64
if(OLD_GBM):
    print(scores)
    print(np.mean(scores))


## === cell 65
if(OLD_GBM):
    for df in importance:
        display(df.sort_values('importance',ascending=False))


## === cell 66
if(OLD_GBM):
    y_pred_test_submit = y_pred_test/5 #n_splits=5
    sample['pressure'] = y_pred_test_submit
    sample.to_csv("submission.csv",index=False)


## === cell 67
if(OLD_GBM):
    fig,axes = plt.subplots(1,1,figsize=(10,10))
    sns.scatterplot(data=gbm_val_result,x='correct',y='result',ax=axes)

    x = np.linspace(0, 60, 10)
    y = x
    axes.plot(x, y, color = "r")
