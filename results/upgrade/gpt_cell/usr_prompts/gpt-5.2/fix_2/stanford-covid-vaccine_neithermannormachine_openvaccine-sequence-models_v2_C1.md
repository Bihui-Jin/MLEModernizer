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


## --- ERROR in cell 10, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mNotImplementedError[0m                       Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1926346979.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     16[0m     [0mtrain_df[0m[0;34m[[0m[0;34m"SN_filter"[0m[0;34m][0m [0;34m==[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[1;32m     17[0m ]  # only select good quality examples to train on
[0;32m---> 18[0;31m [0mtemp[0m [0;34m=[0m [0mfeature_engineer[0m[0;34m([0m[0mtemp[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     19[0m [0;34m[0m[0m
[1;32m     20[0m [0;31m# BUGFIX: pandas>=2.x groupby().cumsum() on a mixed-dtype DataFrame can raise[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/20449227.py[0m in [0;36mfeature_engineer[0;34m(df, train, **kwargs)[0m
[1;32m     20[0m     [0;31m#adds seqpos column to record position of each row in each id's individual sequence[0m[0;34m[0m[0;34m[0m[0m
[1;32m     21[0m     [0mdata[0m[0;34m[[0m[0;34m'seqpos'[0m[0;34m][0m [0;34m=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 22[0;31m     [0mdata[0m[0;34m[[0m[0;34m'seqpos'[0m[0;34m][0m [0;34m=[0m [0mdata[0m[0;34m.[0m[0mgroupby[0m[0;34m([0m[0;34m'id'[0m[0;34m)[0m[0;34m.[0m[0mcumsum[0m[0;34m([0m[0;34m)[0m[0;34m[[0m[0;34m'seqpos'[0m[0;34m][0m [0;34m-[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     23[0m [0;34m[0m[0m
[1;32m     24[0m     [0;31m#adds nucleotide column to record base (A,C,G,U) at position seqpos[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py[0m in [0;36mcumsum[0;34m(self, axis, *args, **kwargs)[0m
[1;32m   4934[0m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_python_apply_general[0m[0;34m([0m[0mf[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0m_selected_obj[0m[0;34m,[0m [0mis_transform[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   4935[0m [0;34m[0m[0m
[0;32m-> 4936[0;31m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_cython_transform[0m[0;34m([0m[0;34m"cumsum"[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   4937[0m [0;34m[0m[0m
[1;32m   4938[0m     [0;34m@[0m[0mfinal[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/generic.py[0m in [0;36m_cython_transform[0;34m(self, how, numeric_only, axis, **kwargs)[0m
[1;32m   1700[0m         [0;31m# We could use `mgr.apply` here and not have to set_axis, but[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1701[0m         [0;31m#  we would have to do shape gymnastics for ArrayManager compat[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1702[0;31m         [0mres_mgr[0m [0;34m=[0m [0mmgr[0m[0;34m.[0m[0mgrouped_reduce[0m[0;34m([0m[0marr_func[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1703[0m         [0mres_mgr[0m[0;34m.[0m[0mset_axis[0m[0;34m([0m[0;36m1[0m[0;34m,[0m [0mmgr[0m[0;34m.[0m[0maxes[0m[0;34m[[0m[0;36m1[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1704[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py[0m in [0;36mgrouped_reduce[0;34m(self, func)[0m
[1;32m   1467[0m                 [0;31m#  while others do not.[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1468[0m                 [0;32mfor[0m [0msb[0m [0;32min[0m [0mblk[0m[0;34m.[0m[0m_split[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1469[0;31m                     [0mapplied[0m [0;34m=[0m [0msb[0m[0;34m.[0m[0mapply[0m[0;34m([0m[0mfunc[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1470[0m                     [0mresult_blocks[0m [0;34m=[0m [0mextend_blocks[0m[0;34m([0m[0mapplied[0m[0;34m,[0m [0mresult_blocks[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1471[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py[0m in [0;36mapply[0;34m(self, func, **kwargs)[0m
[1;32m    391[0m         [0mone[0m[0;34m[0m[0;34m[0m[0m
[1;32m    392[0m         """
[0;32m--> 393[0;31m         [0mresult[0m [0;34m=[0m [0mfunc[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mvalues[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    394[0m [0;34m[0m[0m
[1;32m    395[0m         [0mresult[0m [0;34m=[0m [0mmaybe_coerce_values[0m[0;34m([0m[0mresult[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/generic.py[0m in [0;36marr_func[0;34m(bvalues)[0m
[1;32m   1694[0m [0;34m[0m[0m
[1;32m   1695[0m         [0;32mdef[0m [0marr_func[0m[0;34m([0m[0mbvalues[0m[0;34m:[0m [0mArrayLike[0m[0;34m)[0m [0;34m->[0m [0mArrayLike[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1696[0;31m             return self._grouper._cython_operation(
[0m[1;32m   1697[0m                 [0;34m"transform"[0m[0;34m,[0m [0mbvalues[0m[0;34m,[0m [0mhow[0m[0;34m,[0m [0;36m1[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1698[0m             )

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py[0m in [0;36m_cython_operation[0;34m(self, kind, values, how, axis, min_count, **kwargs)[0m
[1;32m    829[0m         [0mids[0m[0;34m,[0m [0m_[0m[0;34m,[0m [0m_[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mgroup_info[0m[0;34m[0m[0;34m[0m[0m
[1;32m    830[0m         [0mngroups[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mngroups[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 831[0;31m         return cy_op.cython_operation(
[0m[1;32m    832[0m             [0mvalues[0m[0;34m=[0m[0mvalues[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    833[0m             [0maxis[0m[0;34m=[0m[0maxis[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py[0m in [0;36mcython_operation[0;34m(self, values, axis, min_count, comp_ids, ngroups, **kwargs)[0m
[1;32m    548[0m             )
[1;32m    549[0m [0;34m[0m[0m
[0;32m--> 550[0;31m         return self._cython_op_ndim_compat(
[0m[1;32m    551[0m             [0mvalues[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    552[0m             [0mmin_count[0m[0;34m=[0m[0mmin_count[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py[0m in [0;36m_cython_op_ndim_compat[0;34m(self, values, min_count, ngroups, comp_ids, mask, result_mask, **kwargs)[0m
[1;32m    342[0m             [0;32mreturn[0m [0mres[0m[0;34m.[0m[0mT[0m[0;34m[0m[0;34m[0m[0m
[1;32m    343[0m [0;34m[0m[0m
[0;32m--> 344[0;31m         return self._call_cython_op(
[0m[1;32m    345[0m             [0mvalues[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    346[0m             [0mmin_count[0m[0;34m=[0m[0mmin_count[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py[0m in [0;36m_call_cython_op[0;34m(self, values, min_count, ngroups, comp_ids, mask, result_mask, **kwargs)[0m
[1;32m    399[0m [0;34m[0m[0m
[1;32m    400[0m         [0mout_shape[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_get_output_shape[0m[0;34m([0m[0mngroups[0m[0;34m,[0m [0mvalues[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 401[0;31m         [0mfunc[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_get_cython_function[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mkind[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mhow[0m[0;34m,[0m [0mvalues[0m[0;34m.[0m[0mdtype[0m[0;34m,[0m [0mis_numeric[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    402[0m         [0mvalues[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_get_cython_vals[0m[0;34m([0m[0mvalues[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    403[0m         [0mout_dtype[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_get_out_dtype[0m[0;34m([0m[0mvalues[0m[0;34m.[0m[0mdtype[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py[0m in [0;36m_get_cython_function[0;34m(cls, kind, how, dtype, is_numeric)[0m
[1;32m    198[0m             [0;32melif[0m [0;34m"object"[0m [0;32mnot[0m [0;32min[0m [0mf[0m[0;34m.[0m[0m__signatures__[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    199[0m                 [0;31m# raise NotImplementedError here rather than TypeError later[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 200[0;31m                 raise NotImplementedError(
[0m[1;32m    201[0m                     [0;34mf"function is not implemented for this dtype: "[0m[0;34m[0m[0;34m[0m[0m
[1;32m    202[0m                     [0;34mf"[how->{how},dtype->{dtype_str}]"[0m[0;34m[0m[0;34m[0m[0m

[0;31mNotImplementedError[0m: function is not implemented for this dtype: [how->cumsum,dtype->object]

## === cell 12
import matplotlib.pyplot as plt
import seaborn as sns

corr_data = train_df.drop(['index','id','sequence','structure','predicted_loop_type', 'seq_length','seq_scored'], axis = 1).corr()

corr_data
