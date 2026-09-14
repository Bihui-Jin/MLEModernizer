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

3.9

# 3. Installed packages

catboost==1.2.8
geopandas==0.14.4
lightgbm==4.6.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
optuna==4.5.0
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
plotly==5.24.1
plotly-express==0.4.1
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
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

0.193

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

os.environ.setdefault("PYTHONHASHSEED", "228")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np
import pandas as pd

import warnings

warnings.filterwarnings("ignore")

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers.schedules import ExponentialDecay

pd.set_option("display.max_columns", None)

np.random.seed(228)
tf.keras.utils.set_random_seed(228)

DATA_DIR = "../input/ventilator-pressure-prediction"
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
ss = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
print(f"Length of TRAIN dataset: {len(train)}")
print(f"Length of TEST dataset: {len(test)}\n")

print("Missing values in TRAIN dataset")
print(train.iloc[:, 0:-1].isna().sum().to_string())
print("\nMissing values in TEST dataset")
print(test.isna().sum().to_string())
print("")
print(f'Number of breaths in train dataset: {train["breath_id"].nunique()}')
print(f'Number of breaths in test dataset: {test["breath_id"].nunique()}')
obs_per_breath = train["breath_id"].value_counts().iat[0]
print(f"The number of observations for each breath: {obs_per_breath}")



## === cell 2
pass



## === cell 3
pass



## === cell 4
pass




## === cell 5
def features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["area"] = (
        (df["time_step"] * df["u_in"]).groupby(df["breath_id"], sort=False).cumsum()
    )
    df["u_in_cumsum"] = df["u_in"].groupby(df["breath_id"], sort=False).cumsum()

    df["u_in_lag2"] = df["u_in"].shift(2).fillna(0)
    df["u_in_lag4"] = df["u_in"].shift(4).fillna(0)

    df["R"] = df["R"].astype(str)
    df["C"] = df["C"].astype(str)

    g = df.groupby("breath_id", sort=False)["u_in"]

    ewm_g = g.ewm(halflife=10)
    df["ewm_u_in_mean"] = ewm_g.mean().reset_index(level=0, drop=True)
    df["ewm_u_in_std"] = ewm_g.std().reset_index(level=0, drop=True)
    df["ewm_u_in_corr"] = ewm_g.corr().reset_index(level=0, drop=True)

    roll_g = g.rolling(window=10, min_periods=1)
    df["rolling_10_mean"] = roll_g.mean().reset_index(level=0, drop=True)
    df["rolling_10_max"] = roll_g.max().reset_index(level=0, drop=True)
    df["rolling_10_std"] = roll_g.std().reset_index(level=0, drop=True)

    exp_g = g.expanding(2)
    df["expand_mean"] = exp_g.mean().reset_index(level=0, drop=True)
    df["expand_max"] = exp_g.max().reset_index(level=0, drop=True)
    df["expand_std"] = exp_g.std().reset_index(level=0, drop=True)

    return df


train_feat = features(train)
test_feat = features(test)

train_feat = pd.get_dummies(train_feat, columns=["R", "C"])
test_feat = pd.get_dummies(test_feat, columns=["R", "C"])

train_feat, test_feat = train_feat.align(test_feat, join="left", axis=1, fill_value=0)

train = train_feat
test = test_feat
del train_feat, test_feat



## === cell 6
train = train.fillna(0)
test = test.fillna(0)



## === cell 7
targets = train[["pressure"]].to_numpy().reshape(-1, 80).astype(np.float32, copy=False)

train.drop(["pressure", "id", "breath_id"], axis=1, inplace=True)
test_ids = test["id"].to_numpy()
test.drop(["id", "breath_id"], axis=1, inplace=True)



## === cell 8
RS = RobustScaler()
train = RS.fit_transform(train).astype(np.float32, copy=False)
test = RS.transform(test).astype(np.float32, copy=False)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1040488276.py in <cell line: 0>()
      2 RS = RobustScaler()
      3 train = RS.fit_transform(train).astype(np.float32, copy=False)
----> 4 test = RS.transform(test).astype(np.float32, copy=False)
      5 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_data.py in transform(self, X)
   1575         """
   1576         check_is_fitted(self)
-> 1577         X = self._validate_data(
   1578             X,
   1579             accept_sparse=("csr", "csc"),

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    546             validated.
    547         """
--> 548         self._check_feature_names(X, reset=reset)
    549 
    550         if y is None and self._get_tags()["requires_y"]:

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _check_feature_names(self, X, reset)
    479                 )
    480 
--> 481             raise ValueError(message)
    482 
    483     def _validate_data(

ValueError: The feature names should match those that were passed during fit.
Feature names unseen at fit time:
- pressure


## === cell 9
n_features = train.shape[-1]
train = np.ascontiguousarray(train.reshape(-1, 80, n_features))
test = np.ascontiguousarray(test.reshape(-1, 80, n_features))
targets = np.ascontiguousarray(targets)

print(
    "Train shape:",
    train.shape,
    "Targets shape:",
    targets.shape,
    "Test shape:",
    test.shape,
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3301442756.py in <cell line: 0>()
      1 n_features = train.shape[-1]
      2 train = np.ascontiguousarray(train.reshape(-1, 80, n_features))
----> 3 test = np.ascontiguousarray(test.reshape(-1, 80, n_features))
      4 targets = np.ascontiguousarray(targets)
      5 

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'reshape'

## === cell 10
EPOCH = 300
BATCH_SIZE = 1024

try:
    tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu_resolver)
    print("Using TPU strategy.")
except Exception:
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using MirroredStrategy on {len(gpus)} GPUs.")
    else:
        strategy = tf.distribute.get_strategy()
        print("Using default strategy (CPU or single GPU).")

AUTOTUNE = tf.data.AUTOTUNE

with strategy.scope():
    kf = KFold(n_splits=5, shuffle=True, random_state=228)
    test_preds = []

    test_ds = (
        tf.data.Dataset.from_tensor_slices(test)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train, targets)):
        print("-" * 15, ">", f"Fold {fold+1}", "<", "-" * 15)

        X_train, X_valid = train[train_idx], train[valid_idx]
        y_train, y_valid = targets[train_idx], targets[valid_idx]

        train_ds = (
            tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .batch(BATCH_SIZE, drop_remainder=False)
            .prefetch(AUTOTUNE)
        )
        valid_ds = (
            tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
            .batch(BATCH_SIZE, drop_remainder=False)
            .prefetch(AUTOTUNE)
        )

        model = keras.models.Sequential(
            [
                keras.layers.Input(shape=train.shape[-2:]),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(400, return_sequences=True)
                ),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(300, return_sequences=True)
                ),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(200, return_sequences=True)
                ),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(100, return_sequences=True)
                ),
                keras.layers.Dense(50, activation="selu"),
                keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mae")

        steps_per_epoch = int(np.ceil(len(X_train) / BATCH_SIZE))
        decay_schedule = ExponentialDecay(
            1e-3, decay_steps=400 * max(1, steps_per_epoch), decay_rate=1e-5
        )

        def lr_fn(epoch, lr):
            return float(decay_schedule(epoch * steps_per_epoch).numpy())

        lr_cb = LearningRateScheduler(lr_fn, verbose=1)

        model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=EPOCH,
            callbacks=[lr_cb],
            verbose=2,
        )

        preds = model.predict(test_ds, verbose=0).reshape(-1)
        test_preds.append(preds)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/structure.py in normalize_element(element, element_signature)
    104         if spec is None:
--> 105           spec = type_spec_from_value(t, use_fallback=False)
    106       except TypeError:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/structure.py in type_spec_from_value(element, use_fallback)
    513 
--> 514   raise TypeError("Could not build a `TypeSpec` for {} with type {}".format(
    515       element,

TypeError: Could not build a `TypeSpec` for         time_step       u_in  u_out  pressure        area  u_in_cumsum  \
0        0.000000   0.133805      0         0    0.000000     0.133805   
1        0.031955   2.411108      0         0    0.077048     2.544914   
2        0.063910   4.484026      0         0    0.363623     7.028940   
3        0.095838   9.191747      0         0    1.244544    16.220687   
4        0.128655  14.005526      0         0    3.046428    30.226213   
...           ...        ...    ...       ...         ...          ...   
603595   2.394955   4.943961      1         0  332.104214   366.143303   
603596   2.427024   4.952263      1         0  344.123477   371.095566   
603597   2.458956   4.959308      1         0  356.318194   376.054874   
603598   2.490949   4.965323      1         0  368.686561   381.020197   
603599   2.522907   4.970444      1         0  381.226531   385.990641   

        u_in_lag2  u_in_lag4  ewm_u_in_mean  ewm_u_in_std  ewm_u_in_corr  \
0        0.000000   0.000000       0.133805      0.000000            0.0   
1        0.000000   0.000000       1.311904      1.610296            1.0   
2        0.133805   0.000000       2.443356      2.172681            1.0   
3        2.411108   0.000000       4.309700      3.899985            1.0   
4        4.484026   0.133805       6.526550      5.637190            1.0   
...           ...        ...            ...           ...            ...   
603595   4.922880   4.893849       4.129197      1.741815            1.0   
603596   4.934249   4.909532       4.184582      1.695215            1.0   
603597   4.943961   4.922880       4.236697      1.649060            1.0   
603598   4.952263   4.934249       4.285696      1.603426            1.0   
603599   4.959308   4.943961       4.331731      1.558380            1.0   

        rolling_10_mean  rolling_10_max  rolling_10_std  expand_mean  \
0              0.133805        0.133805        0.000000     0.000000   
1              1.272457        2.411108        1.610296     1.272457   
2              2.342980        4.484026        2.175910     2.342980   
3              4.055172        9.191747        3.857823     4.055172   
4              6.045243       14.005526        5.564531     6.045243   
...                 ...             ...             ...          ...   
603595         4.872566        4.943961        0.060275     4.817675   
603596         4.891371        4.952263        0.051396     4.819423   
603597         4.907405        4.959308        0.043816     4.821216   
603598         4.921076        4.965323        0.037353     4.823040   
603599         4.932728        4.970444        0.031844     4.824883   

        expand_max  expand_std   R_20    R_5  R_50   C_10  C_20   C_50  
0         0.000000    0.000000  False  False  True  False  True  False  
1         2.411108    1.610296  False  False  True  False  True  False  
2         4.484026    2.175910  False  False  True  False  True  False  
3         9.191747    3.857823  False  False  True  False  True  False  
4        14.005526    5.564531  False  False  True  False  True  False  
...            ...         ...    ...    ...   ...    ...   ...    ...  
603595   24.865530    4.207989  False  False  True  False  True  False  
603596   24.865530    4.180241  False  False  True  False  True  False  
603597   24.865530    4.153038  False  False  True  False  True  False  
603598   24.865530    4.126362  False  False  True  False  True  False  
603599   24.865530    4.100196  False  False  True  False  True  False  

[603600 rows x 23 columns] with type DataFrame

During handling of the above exception, another exception occurred:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/540520161.py in <cell line: 0>()
     22 
     23     test_ds = (
---> 24         tf.data.Dataset.from_tensor_slices(test)
     25         .batch(BATCH_SIZE, drop_remainder=False)
     26         .prefetch(AUTOTUNE)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in from_tensor_slices(tensors, name)
    825     # pylint: disable=g-import-not-at-top,protected-access
    826     from tensorflow.python.data.ops import from_tensor_slices_op
--> 827     return from_tensor_slices_op._from_tensor_slices(tensors, name)
    828     # pylint: enable=g-import-not-at-top,protected-access
    829 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/from_tensor_slices_op.py in _from_tensor_slices(tensors, name)
     23 
     24 def _from_tensor_slices(tensors, name=None):
---> 25   return _TensorSliceDataset(tensors, name=name)
     26 
     27 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/from_tensor_slices_op.py in __init__(self, element, is_files, name)
     31   def __init__(self, element, is_files=False, name=None):
     32     """See `Dataset.from_tensor_slices` for details."""
---> 33     element = structure.normalize_element(element)
     34     batched_spec = structure.type_spec_from_value(element)
     35     self._tensors = structure.to_batched_tensor_list(batched_spec, element)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/structure.py in normalize_element(element, element_signature)
    108         # the value. As a fallback try converting the value to a tensor.
    109         normalized_components.append(
--> 110             ops.convert_to_tensor(t, name="component_%d" % i))
    111       else:
    112         # To avoid a circular dependency between dataset_ops and structure,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/profiler/trace.py in wrapped(*args, **kwargs)
    181         with Trace(trace_name, **trace_kwargs):
    182           return func(*args, **kwargs)
--> 183       return func(*args, **kwargs)
    184 
    185     return wrapped

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in convert_to_tensor(value, dtype, name, as_ref, preferred_dtype, dtype_hint, ctx, accepted_result_types)
    730   # TODO(b/142518781): Fix all call-sites and remove redundant arg
    731   preferred_dtype = preferred_dtype or dtype_hint
--> 732   return tensor_conversion_registry.convert(
    733       value, dtype, name, as_ref, preferred_dtype, accepted_result_types
    734   )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor_conversion_registry.py in convert(value, dtype, name, as_ref, preferred_dtype, accepted_result_types)
    232 
    233     if ret is None:
--> 234       ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
    235 
    236     if ret is NotImplemented:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_tensor_conversion.py in _constant_tensor_conversion_function(v, dtype, name, as_ref)
     27 
     28   _ = as_ref
---> 29   return constant_op.constant(v, dtype=dtype, name=name)
     30 
     31 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/weak_tensor_ops.py in wrapper(*args, **kwargs)
    140   def wrapper(*args, **kwargs):
    141     if not ops.is_auto_dtype_conversion_enabled():
--> 142       return op(*args, **kwargs)
    143     bound_arguments = signature.bind(*args, **kwargs)
    144     bound_arguments.apply_defaults()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in constant(value, dtype, shape, name)
    274     ValueError: if called on a symbolic tensor.
    275   """
--> 276   return _constant_impl(value, dtype, shape, name, verify_shape=False,
    277                         allow_broadcast=True)
    278 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in _constant_impl(value, dtype, shape, name, verify_shape, allow_broadcast)
    287       with trace.Trace("tf.constant"):
    288         return _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
--> 289     return _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
    290 
    291   const_tensor = ops._create_graph_constant(  # pylint: disable=protected-access

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
    299 ) -> ops._EagerTensorBase:
    300   """Creates a constant on the current device."""
--> 301   t = convert_to_eager_tensor(value, ctx, dtype)
    302   if shape is None:
    303     return t

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in convert_to_eager_tensor(value, ctx, dtype)
    106       dtype = dtypes.as_dtype(dtype).as_datatype_enum
    107   ctx.ensure_initialized()
--> 108   return ops.EagerTensor(value, ctx.device_name, dtype)
    109 
    110 

ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type float).

## === cell 11
if len(test_preds) == 0:
    raise RuntimeError(
        "No fold predictions were produced; training likely failed earlier."
    )

pred = sum(test_preds) / len(test_preds)
pred = pred.reshape(-1)

if len(pred) != len(ss):
    raise ValueError(
        f"Prediction length {len(pred)} does not match submission length {len(ss)}"
    )

ss["pressure"] = pred.astype(np.float32)
ss.to_csv("submission.csv", index=False)

print("Wrote submission to submission.csv with shape:", ss.shape)
print(ss.head())

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/651044910.py in <cell line: 0>()
      1 # Write a valid submission. Ensure length matches sample_submission rows.
      2 if len(test_preds) == 0:
----> 3     raise RuntimeError(
      4         "No fold predictions were produced; training likely failed earlier."
      5     )

RuntimeError: No fold predictions were produced; training likely failed earlier.
