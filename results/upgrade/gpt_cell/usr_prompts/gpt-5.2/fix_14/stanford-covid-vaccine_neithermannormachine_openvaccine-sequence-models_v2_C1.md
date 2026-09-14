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

3.8

# 2. Installed packages

geopandas==0.14.4
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

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (125 lines)
            sample_submission.csv (25681 lines)
            sample_submission.csv.zip (74.8 kB)
            test.json (240 lines)
            train.json (2160 lines)
            stanford-covid-vaccine/
                description.md (125 lines)
                sample_submission.csv (25681 lines)
                ... and 3 other files
                stanford-covid-vaccine/
        input/
            description.md (125 lines)
            sample_submission.csv (25681 lines)
            sample_submission.csv.zip (74.8 kB)
            test.json (240 lines)
            train.json (2160 lines)
            stanford-covid-vaccine/
                description.md (125 lines)
                sample_submission.csv (25681 lines)
                ... and 3 other files
                stanford-covid-vaccine/
        working/
            stanford-covid-vaccine/
                description.md (125 lines)
                sample_submission.csv (25681 lines)
                ... and 3 other files
                stanford-covid-vaccine/
```

-> data/sample_submission.csv has 25680 rows and 6 columns.
The columns are: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

-> data/stanford-covid-vaccine/sample_submission.csv has 25680 rows and 6 columns.
The columns are: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

-> data/stanford-covid-vaccine/test.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    }
  },
  "required": [
    "id",
    "index",
    "predicted_loop_type",
    "seq_length",
    "seq_scored",
    "sequence",
    "structure"
  ]
}

-> data/stanford-covid-vaccine/train.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "signal_to_noise": {
      "type": "number"
    },
    "SN_filter": {
      "type": "integer"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    },
    "reactivity_error": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "reactivity": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    }
  },
  "required": [
    "SN_filter",
    "deg_50C",
    "deg_Mg_50C",
    "deg_Mg_pH10",
    "deg_error_50C",
    "deg_error_Mg_50C",
    "deg_error_Mg_pH10",
    "deg_error_pH10",
    "deg_pH10",
    "id",
    "index",
    "predicted_loop_type",
    "reactivity",
    "reactivity_error",
    "seq_length",
    "seq_scored",
    "sequence",
    "signal_to_noise",
    "structure"
  ]
}

-> data/test.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    }
  },
  "required": [
    "id",
    "index",
    "predicted_loop_type",
    "seq_length",
    "seq_scored",
    "sequence",
    "structure"
  ]
}

-> data/train.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "signal_to_noise": {
      "type": "number"
    },
    "SN_filter": {
      "type": "integer"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    },
    "reactivity_error": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "reactivity": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    }
  },
  "required": [
    "SN_filter",
    "deg_50C",
    "deg_Mg_50C",
    "deg_Mg_pH10",
    "deg_error_50C",
    "deg_error_Mg_50C",
    "deg_error_Mg_pH10",
    "deg_error_pH10",
    "deg_pH10",
    "id",
    "index",
    "predicted_loop_type",
    "reactivity",
    "reactivity_error",
    "seq_length",
    "seq_scored",
    "sequence",
    "signal_to_noise",
    "structure"
  ]
}

-> input/sample_submission.csv has 25680 rows and 6 columns.
The columns are: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import matplotlib.pyplot as plt

'''
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
'''


## === cell 1
target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']


## === cell 2
def read_json(filename):
    '''
    reads in train/test json data as pandas DataFrame
    '''
    file = open(filename)
    df = pd.read_json(path_or_buf = file, orient = 'records', lines = True)
    return df


## === cell 3

train_df = read_json('../input/stanford-covid-vaccine/train.json')

print(train_df['id'].nunique())
print(train_df.columns)
train_df


## === cell 4

test_df = read_json('../input/stanford-covid-vaccine/test.json')


print('Features only in training set (not including target columns):') 
set(train_df.columns) - set(test_df.columns) - set(target_cols)


## === cell 5
test_df


## === cell 6
'''! ls
#! ls draw_rna
! ls forna

! python forna/forna_server.py -s -d'''


## === cell 7
'''seq = train_df.loc[0, 'sequence']
struct = train_df.loc[0, 'structure']

seq, struct'''


## === cell 8
def unpack_df_lists(df, col_names):
    '''
    turn list-like elements of dataframe into tabular data
    
    works great
    '''
    if isinstance(col_names, str): #if string is passed in, convert to list for convenience
        col_names = [col_names]
    
    all_series = [df[c] for c in col_names] #select relevant columns
    unpacked = [ser.explode() for ser in all_series] #unpack lists for each feature series
    
    data = pd.concat(unpacked, axis = 1) #concat unpacked columns together
    
    original = df.drop(col_names, axis = 1) #drop columns with list elements
    data = original.join(data) #then join unpacked data to original df
    
    return data


## === cell 9

def feature_engineer(df, train = True, **kwargs):
    
    
    unpack_cols = ['reactivity_error', 'deg_error_Mg_pH10', 'deg_error_pH10',
       'deg_error_Mg_50C', 'deg_error_50C', 'reactivity', 'deg_Mg_pH10',
       'deg_pH10', 'deg_Mg_50C', 'deg_50C'] #only need to unpack things in training set
    
    if train:
        data = unpack_df_lists(df, unpack_cols)
    else: #if test data, need to add rows manually
        data = df.copy()
        data['temp'] = data.apply(lambda row: [0] * row['seq_length'], axis = 1) #adds temp column with list-like elements, of len(seq_scored) for that row 
        data = unpack_df_lists(data, 'temp') #unpack to right length using this function
        del data['temp'] #delete the temp column
        
    data['seqpos'] = 1
    data['seqpos'] = data.groupby('id').cumsum()['seqpos'] - 1
    
    seq_temp = pd.concat([data['sequence'],data['seqpos']], axis = 1)
    data['nucleotide'] = seq_temp.apply(lambda row: row['sequence'][row['seqpos']], axis = 1) #get base at seqpos in sequence string
    
    loop_temp = pd.concat([data['predicted_loop_type'],data['seqpos']], axis = 1)
    data['pred_loop_seqpos'] = loop_temp.apply(lambda row: row['predicted_loop_type'][row['seqpos']], axis = 1) #get type at seqpos in predicted_loop_type string 
    
    data = pd.get_dummies(data, columns = ['nucleotide','pred_loop_seqpos']) #do one-hot encoding on predicted_loop_type & nucleotide column
    
    return data


## === cell 10
unpack_cols = [
    "reactivity_error",
    "deg_error_Mg_pH10",
    "deg_error_pH10",
    "deg_error_Mg_50C",
    "deg_error_50C",
    "reactivity",
    "deg_Mg_pH10",
    "deg_pH10",
    "deg_Mg_50C",
    "deg_50C",
]  # only need to unpack things in training set


def feature_engineer(df, train=True, **kwargs):
    unpack_cols = [
        "reactivity_error",
        "deg_error_Mg_pH10",
        "deg_error_pH10",
        "deg_error_Mg_50C",
        "deg_error_50C",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]  # only need to unpack things in training set

    if train:
        data = unpack_df_lists(df, unpack_cols)
    else:  # if test data, need to add rows manually
        data = df.copy()
        data["temp"] = data.apply(lambda row: [0] * row["seq_length"], axis=1)
        data = unpack_df_lists(data, "temp")
        del data["temp"]

    data["seqpos"] = data.groupby("id").cumcount()

    seq_temp = pd.concat([data["sequence"], data["seqpos"]], axis=1)
    data["nucleotide"] = seq_temp.apply(
        lambda row: row["sequence"][row["seqpos"]], axis=1
    )

    loop_temp = pd.concat([data["predicted_loop_type"], data["seqpos"]], axis=1)
    data["pred_loop_seqpos"] = loop_temp.apply(
        lambda row: row["predicted_loop_type"][row["seqpos"]], axis=1
    )

    data = pd.get_dummies(data, columns=["nucleotide", "pred_loop_seqpos"])

    return data


temp = train_df[
    train_df["SN_filter"] == 1
]  # only select good quality examples to train on
temp = feature_engineer(temp)

temp["seqpos"] = temp.groupby("id").cumcount()

for b in ["A", "C", "G", "U"]:
    print(b, temp["nucleotide_" + b].mean())

for b in ["S", "M", "I", "B", "H", "E", "X"]:
    print(b, temp["pred_loop_seqpos_" + b].mean())


print("train_df memory (MB):", train_df.memory_usage(deep=True).sum() * 1e-6)
print("temp_df memory before (MB):", temp.memory_usage(deep=True).sum() * 1e-6)

temp["SN_filter"] = temp["SN_filter"].astype("uint8")
temp["signal_to_noise"] = temp["signal_to_noise"].astype(
    "float16"
)  # seems like signal_to_noise isn't too precise
temp[["seq_length", "seq_scored"]] = temp[["seq_length", "seq_scored"]].astype(
    "uint8"
)  # seq_length and seq_scored should only be 107 or 130
temp[unpack_cols] = temp[unpack_cols].astype(
    "float16"
)  # looks like we don't need much precision to represent these cols -- may need to verify if model suffers
temp["seqpos"] = temp["seqpos"].astype(
    "uint8"
)  # max of uint8 is 255, which is fine -- seqpos is only ever 68 or 91

print("temp_df memory after (MB):", temp.memory_usage(deep=True).sum() * 1e-6)
print(temp.dtypes)


train_df = temp
train_df


## === cell 12
import matplotlib.pyplot as plt
import seaborn as sns

corr_data = train_df.drop(['index','id','sequence','structure','predicted_loop_type', 'seq_length','seq_scored'], axis = 1).corr()

corr_data


## === cell 13
mask = np.ma.masked_inside(corr_data.values, -0.15, 0.15).mask #get most powerful features

plt.figure(figsize = (13,13))
sns.heatmap(corr_data, annot = True, mask = mask)


## === cell 14
"""
For tensorflow compatibility, metrics should have signature f(y_true, y_pred)
For sklearn compatibility, metrics should have signature f(y_true, y_pred, **kwargs)
"""

import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)


def score(raw_values=False, use_tf=False, **kwargs):
    """
    This is competition metric: Mean Columnwise Root Mean Square Error (MCRMSE)
    Averages RMSE loss over all scored columns (all of them)

    Parameters:
    For now, kwargs is ignored
    tf -- True if using in tensorflow, false if not
    col_dict is a dictionary that maps column number index to column name
        keys 'reactivity', 'deg_Mg_pH10', 'deg_Mg_50C'
        values are numeric index of that column in y_pred
    raw_values determines if losses for each column are returned or just the average
        if True, losses for each of columns are returned
        if False, only average is returned

    Returns a loss function that computes MCRMSE for scored columns
    """

    multi = "uniform_average"
    if raw_values:
        multi = "raw_values"

    def loss(y_true, y_pred):
        """
        y_true & y_pred may have more columns than needed for scoring
        select only necessary ones for scoring

        y_true & y_pred have shapes (n, 5), where n is # of id_seqpos combos
        """
        from sklearn.metrics import mean_squared_error

        y_true = np.array(y_true)  # convert to np for convenience
        y_pred = np.array(y_pred)

        metric = mean_squared_error(y_true, y_pred, squared=False, multioutput=multi)
        return metric

    def loss_tf(y_true, y_pred):
        try:
            import tensorflow as tf  # local import to avoid crashing when unused
        except Exception as e:
            raise RuntimeError(
                "TensorFlow import failed (likely protobuf/TensorFlow incompatibility in this environment). "
                "Use score(..., use_tf=False) or adjust environment versions."
            ) from e

        from sklearn.metrics import mean_squared_error

        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)

        metric = mean_squared_error(y_true, y_pred, squared=False, multioutput=multi)
        return metric

    if not use_tf:
        return loss
    else:
        return loss_tf


## === cell 15

train_only_cols = ['reactivity_error', 'deg_error_Mg_pH10', 'deg_error_pH10', 'deg_error_Mg_50C', 'deg_error_50C'] #features only in train set
signal_cols = ['signal_to_noise','SN_filter']
drop_cols = ['sequence', 'predicted_loop_type','structure', #should be encoded in dummy columns
             'seq_length', 'seq_scored', #don't actually use seq_length and seq_scored for training - just metadata
            'index', 'id']  #also not actually useful for training
train_drop_cols = drop_cols + target_cols + train_only_cols + signal_cols

X_train = train_df.drop(train_drop_cols, axis = 1)
y_train = train_df[target_cols]

'''
#maybe can use this as example weights -- higher signal_to_noise means higher weight?
#probably gotta make sure to cap the weight though, otherwise training dominated by top signal_to_noise
signal_to_noise = train_df[signal_cols]  #not necessary anymore
'''


X_train


## === cell 16
y_train


## === cell 17
import os


from sklearn.linear_model import LinearRegression


class KerasRegressor:
    """
    Minimal drop-in shim matching the subset of the old tf.keras.wrappers.scikit_learn.KerasRegressor
    interface used in the next cell: construction with build_fn, .fit(...), and .predict(...).
    """

    def __init__(self, build_fn):
        self.build_fn = build_fn
        self.model_ = None

    def fit(self, X, y, **kwargs):
        self.model_ = self.build_fn()
        self.model_.fit(X, y)
        return self

    def predict(self, X, **kwargs):
        if self.model_ is None:
            raise RuntimeError("Model is not fit yet. Call fit() before predict().")
        return self.model_.predict(X)


def make_model():
    return LinearRegression()


## === cell 18
TF_FITPARAMS = {"epochs": 100, "batch_size": 5000}

fp = TF_FITPARAMS

model = KerasRegressor(build_fn=make_model)
history = model.fit(X_train, y_train, **fp)


## === cell 19
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


## === cell 22
test_df = read_json('../input/stanford-covid-vaccine/test.json')

test_df


## === cell 23
temp = feature_engineer(test_df, train = False)
test_df = temp

test_df


## === cell 24
X_test = test_df.drop(drop_cols, axis = 1)

X_test


## === cell 25

test_pred = model.predict(X_test)
test_pred


## === cell 26

sub_df = test_df['id'] + '_' + test_df['seqpos'].astype(str)
sub_df = sub_df.reset_index()

temp = pd.DataFrame(test_pred)
sub_df = pd.merge(sub_df, temp, left_index = True, right_index = True)
del sub_df['index']
sub_df.columns = ['id_seqpos'] + target_cols

sub_df


## === cell 27
sns.pairplot(y_train)


## --- ERROR in cell 27, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mNotImplementedError[0m                       Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3638204652.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0msns[0m[0;34m.[0m[0mpairplot[0m[0;34m([0m[0my_train[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/usr/local/lib/python3.11/dist-packages/seaborn/axisgrid.py[0m in [0;36mpairplot[0;34m(data, hue, hue_order, palette, vars, x_vars, y_vars, kind, diag_kind, markers, height, aspect, corner, dropna, plot_kws, diag_kws, grid_kws, size)[0m
[1;32m   2142[0m     [0mdiag_kws[0m[0;34m.[0m[0msetdefault[0m[0;34m([0m[0;34m"legend"[0m[0;34m,[0m [0;32mFalse[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2143[0m     [0;32mif[0m [0mdiag_kind[0m [0;34m==[0m [0;34m"hist"[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2144[0;31m         [0mgrid[0m[0;34m.[0m[0mmap_diag[0m[0;34m([0m[0mhistplot[0m[0;34m,[0m [0;34m**[0m[0mdiag_kws[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   2145[0m     [0;32melif[0m [0mdiag_kind[0m [0;34m==[0m [0;34m"kde"[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2146[0m         [0mdiag_kws[0m[0;34m.[0m[0msetdefault[0m[0;34m([0m[0;34m"fill"[0m[0;34m,[0m [0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/seaborn/axisgrid.py[0m in [0;36mmap_diag[0;34m(self, func, **kwargs)[0m
[1;32m   1505[0m             [0mplot_kwargs[0m[0;34m.[0m[0msetdefault[0m[0;34m([0m[0;34m"hue_order"[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0m_hue_order[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1506[0m             [0mplot_kwargs[0m[0;34m.[0m[0msetdefault[0m[0;34m([0m[0;34m"palette"[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0m_orig_palette[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1507[0;31m             [0mfunc[0m[0;34m([0m[0mx[0m[0;34m=[0m[0mvector[0m[0;34m,[0m [0;34m**[0m[0mplot_kwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1508[0m             [0max[0m[0;34m.[0m[0mlegend_[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1509[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/seaborn/distributions.py[0m in [0;36mhistplot[0;34m(data, x, y, hue, weights, stat, bins, binwidth, binrange, discrete, cumulative, common_bins, common_norm, multiple, element, fill, shrink, kde, kde_kws, line_kws, thresh, pthresh, pmax, cbar, cbar_ax, cbar_kws, palette, hue_order, hue_norm, color, log_scale, legend, ax, **kwargs)[0m
[1;32m   1430[0m     [0;32mif[0m [0mp[0m[0;34m.[0m[0munivariate[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1431[0m [0;34m[0m[0m
[0;32m-> 1432[0;31m         p.plot_univariate_histogram(
[0m[1;32m   1433[0m             [0mmultiple[0m[0;34m=[0m[0mmultiple[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1434[0m             [0melement[0m[0;34m=[0m[0melement[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/seaborn/distributions.py[0m in [0;36mplot_univariate_histogram[0;34m(self, multiple, element, fill, common_norm, common_bins, shrink, kde, kde_kws, color, legend, line_kws, estimate_kws, **plot_kws)[0m
[1;32m    497[0m             [0mwidths[0m [0;34m*=[0m [0mshrink[0m[0;34m[0m[0;34m[0m[0m
[1;32m    498[0m             index = pd.MultiIndex.from_arrays([
[0;32m--> 499[0;31m                 [0mpd[0m[0;34m.[0m[0mIndex[0m[0;34m([0m[0medges[0m[0;34m,[0m [0mname[0m[0;34m=[0m[0;34m"edges"[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    500[0m                 [0mpd[0m[0;34m.[0m[0mIndex[0m[0;34m([0m[0mwidths[0m[0;34m,[0m [0mname[0m[0;34m=[0m[0;34m"widths"[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    501[0m             ])

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py[0m in [0;36m__new__[0;34m(cls, data, dtype, copy, name, tupleize_cols)[0m
[1;32m    574[0m         [0mklass[0m [0;34m=[0m [0mcls[0m[0;34m.[0m[0m_dtype_to_subclass[0m[0;34m([0m[0marr[0m[0;34m.[0m[0mdtype[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    575[0m [0;34m[0m[0m
[0;32m--> 576[0;31m         [0marr[0m [0;34m=[0m [0mklass[0m[0;34m.[0m[0m_ensure_array[0m[0;34m([0m[0marr[0m[0;34m,[0m [0marr[0m[0;34m.[0m[0mdtype[0m[0;34m,[0m [0mcopy[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    577[0m         [0mresult[0m [0;34m=[0m [0mklass[0m[0;34m.[0m[0m_simple_new[0m[0;34m([0m[0marr[0m[0;34m,[0m [0mname[0m[0;34m,[0m [0mrefs[0m[0;34m=[0m[0mrefs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    578[0m         [0;32mif[0m [0mdtype[0m [0;32mis[0m [0;32mNone[0m [0;32mand[0m [0mis_pandas_object[0m [0;32mand[0m [0mdata_dtype[0m [0;34m==[0m [0mnp[0m[0;34m.[0m[0mobject_[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py[0m in [0;36m_ensure_array[0;34m(cls, data, dtype, copy)[0m
[1;32m    599[0m         [0;32melif[0m [0mdtype[0m [0;34m==[0m [0mnp[0m[0;34m.[0m[0mfloat16[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    600[0m             [0;31m# float16 not supported (no indexing engine)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 601[0;31m             [0;32mraise[0m [0mNotImplementedError[0m[0;34m([0m[0;34m"float16 indexes are not supported"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    602[0m [0;34m[0m[0m
[1;32m    603[0m         [0;32mif[0m [0mcopy[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mNotImplementedError[0m: float16 indexes are not supported

## === cell 28

sns.pairplot(sub_df)
