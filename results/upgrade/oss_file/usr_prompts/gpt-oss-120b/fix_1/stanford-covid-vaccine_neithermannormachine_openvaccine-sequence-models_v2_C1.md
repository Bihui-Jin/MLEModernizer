# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Overview
Predict likely degradation rates at each base of an RNA molecule.

## Metric
Mean columnwise root mean squared error:

$\textrm{MCRMSE} = \frac{1}{N_{t}}\sum_{j=1}^{N_{t}}\sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_{ij} - \hat{y}_{ij})^2}$

where $N_{t}$ is the number of scored ground truth target columns, and $y$ and $\hat{y}$ are the actual and predicted values, respectively.

There are multiple ground truth values provided in the training data. While the submission format requires all 5 to be predicted, only the following are scored: reactivity, deg_Mg_pH10, and deg_Mg_50C.

## Submission Formats
For each sample `id` in the test set, you must predict targets for *each* sequence position (`seqpos`), one per row. If the length of the `sequence` of an `id` is, e.g., 107, then you should make 107 predictions. Positions greater than the `seq_scored` value of a sample are not scored, but still need a value in the solution file.

```csv
id_seqpos,reactivity,deg_Mg_pH10,deg_pH10,deg_Mg_50C,deg_50C
id_d190610e8_0,0.1,0.3,0.2,0.5,0.4
id_d190610e8_1,0.3,0.2,0.5,0.4,0.2
id_d190610e8_2,0.5,0.4,0.2,0.1,0.2
etc.
```

## Dataset 
- **train.json** - the training data
- **test.json** - the test set, without any columns associated with the ground truth.
- **sample_submission.csv** - a sample submission file in the correct format

#### Columns
- `id` - An arbitrary identifier for each sample.
- `seq_scored` - (68 in Train and Public Test, 68 in Private Test) Integer value denoting the number of positions used in scoring with predicted values. This should match the length of `reactivity`, `deg_*` and `*_error_*` columns.
- `seq_length` - (107 in Train and Public Test, 107 in Private Test) Integer values, denotes the length of `sequence`.
- `sequence` - (1x107 string in Train and Public Test, 107 in Private Test) Describes the RNA sequence, a combination of `A`, `G`, `U`, and `C` for each sample. Should be 107 characters long, and the first 68 bases should correspond to the 68 positions specified in `seq_scored` (note: indexed starting at 0).
- `structure` - (1x107 string in Train and Public Test, 107 in Private Test) An array of `(`, `)`, and `.` characters that describe whether a base is estimated to be paired or unpaired. Paired bases are denoted by opening and closing parentheses e.g. (....) means that base 0 is paired to base 5, and bases 1-4 are unpaired.
- `reactivity` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likely secondary structure of the RNA sample.
- `deg_pH10` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating without magnesium at high pH (pH 10).
- `deg_Mg_pH10` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating with magnesium in high pH (pH 10).
- `deg_50C` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating without magnesium at high temperature (50 degrees Celsius).
- `deg_Mg_50C` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating with magnesium at high temperature (50 degrees Celsius).
- `*_error_*` - An array of floating point numbers, should have the same length as the corresponding `reactivity` or `deg_*` columns, calculated errors in experimental values obtained in `reactivity` and `deg_*` columns.
- `predicted_loop_type` - (1x107 string) Describes the structural context (also referred to as 'loop type')of each character in `sequence`. Loop types assigned by bpRNA from Vienna RNAfold 2 structure. From the bpRNA_documentation: S: paired "Stem" M: Multiloop I: Internal loop B: Bulge H: Hairpin loop E: dangling End X: eXternal loop
    - `S/N filter` Indicates if the sample passed filters described below in `Additional Notes`.

#### Additional Notes
At the beginning of the competition, Stanford scientists have data on 2400 RNA sequences of length 107. For technical reasons, measurements cannot be carried out on the final bases of these RNA sequences, so we have experimental data (ground truth) in 5 conditions for the first 68 bases.

We have split out 240 of these 2400 sequences for a public test set to allow for continuous evaluation through the competition, on the public leaderboard. These sequences, in `test.json`, have been additionally filtered based on three criteria detailed below to ensure that this subset is not dominated by any large cluster of RNA molecules with poor data, which might bias the public leaderboard. The remaining 2160 sequences for which we have data are in `train.json`.

For our final and most important scoring (the Private Leaderbooard), Stanford scientists are carrying out measurements on 240 new RNAs. For these data, we expect to have measurements for the first 68 bases, again missing the ends of the RNA. These sequences constitute the 240 sequences in `test.json`.

For those interested in how the sequences in `test.json` were filtered, here were the steps to ensure a diverse and high quality test set for public leaderboard scoring:

1. Minimum value across all 5 conditions must be greater than -0.5.
2. Mean signal/noise across all 5 conditions must be greater than 1.0. [Signal/noise is defined as mean( measurement value over 68 nts )/mean( statistical error in measurement value over 68 nts)]
3. To help ensure sequence diversity, the resulting sequences were clustered into clusters with less than 50% sequence similarity, and the 240 test set sequences were chosen from clusters with 3 or fewer members. That is, any sequence in the test set should be sequence similar to at most 2 other sequences.

Note that these filters have not been applied to the 2160 RNAs in the public training data `train.json` -- some of those measurements have negative values or poor signal-to-noise, or some RNA sequences have near-identical sequences in that set. But we are providing all those data in case competitors can squeeze out more signal.

# 2. Python version

3.8

# 3. Installed packages

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

# 4. Data file paths

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

# 5. Target score

0.48857

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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
unpack_cols = ['reactivity_error', 'deg_error_Mg_pH10', 'deg_error_pH10',
       'deg_error_Mg_50C', 'deg_error_50C', 'reactivity', 'deg_Mg_pH10',
       'deg_pH10', 'deg_Mg_50C', 'deg_50C'] #only need to unpack things in training set


temp = train_df[train_df['SN_filter'] == 1] #only select good quality examples to train on
temp = feature_engineer(temp)

for b in ['A','C','G','U']:
    print(b, temp['nucleotide_'+b].mean())
    
for b in ['S','M','I','B','H','E','X']:
    print(b, temp['pred_loop_seqpos_'+b].mean())


print('train_df memory (MB):', train_df.memory_usage(deep = True).sum() * 1e-6)
print('temp_df memory before (MB):', temp.memory_usage(deep = True).sum() * 1e-6)

temp['SN_filter'] = temp['SN_filter'].astype('uint8')
temp['signal_to_noise'] = temp['signal_to_noise'].astype('float16') #seems like signal_to_noise isn't too precise
temp[['seq_length','seq_scored']] = temp[['seq_length','seq_scored']].astype('uint8') #seq_length and seq_scored should only be 107 or 130
temp[unpack_cols] = temp[unpack_cols].astype('float16') #looks like we don't need much precision to represent these cols -- may need to verify if model suffers
temp['seqpos'] = temp['seqpos'].astype('uint8') #max of uint8 is 255, which is fine -- seqpos is only ever 68 or 91

print('temp_df memory after (MB):', temp.memory_usage(deep = True).sum() * 1e-6)
print(temp.dtypes)


train_df = temp
train_df


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NotImplementedError                       Traceback (most recent call last)
/tmp/ipykernel_11/1221092434.py in <cell line: 0>()
      7 
      8 temp = train_df[train_df['SN_filter'] == 1] #only select good quality examples to train on
----> 9 temp = feature_engineer(temp)
     10 
     11 #check proportions of nucleotides and loop types

/tmp/ipykernel_11/20449227.py in feature_engineer(df, train, **kwargs)
     20     #adds seqpos column to record position of each row in each id's individual sequence
     21     data['seqpos'] = 1
---> 22     data['seqpos'] = data.groupby('id').cumsum()['seqpos'] - 1
     23 
     24     #adds nucleotide column to record base (A,C,G,U) at position seqpos

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in cumsum(self, axis, *args, **kwargs)
   4934             return self._python_apply_general(f, self._selected_obj, is_transform=True)
   4935 
-> 4936         return self._cython_transform("cumsum", **kwargs)
   4937 
   4938     @final

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/generic.py in _cython_transform(self, how, numeric_only, axis, **kwargs)
   1700         # We could use `mgr.apply` here and not have to set_axis, but
   1701         #  we would have to do shape gymnastics for ArrayManager compat
-> 1702         res_mgr = mgr.grouped_reduce(arr_func)
   1703         res_mgr.set_axis(1, mgr.axes[1])
   1704 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in grouped_reduce(self, func)
   1467                 #  while others do not.
   1468                 for sb in blk._split():
-> 1469                     applied = sb.apply(func)
   1470                     result_blocks = extend_blocks(applied, result_blocks)
   1471             else:

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in apply(self, func, **kwargs)
    391         one
    392         """
--> 393         result = func(self.values, **kwargs)
    394 
    395         result = maybe_coerce_values(result)

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/generic.py in arr_func(bvalues)
   1694 
   1695         def arr_func(bvalues: ArrayLike) -> ArrayLike:
-> 1696             return self._grouper._cython_operation(
   1697                 "transform", bvalues, how, 1, **kwargs
   1698             )

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py in _cython_operation(self, kind, values, how, axis, min_count, **kwargs)
    829         ids, _, _ = self.group_info
    830         ngroups = self.ngroups
--> 831         return cy_op.cython_operation(
    832             values=values,
    833             axis=axis,

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py in cython_operation(self, values, axis, min_count, comp_ids, ngroups, **kwargs)
    548             )
    549 
--> 550         return self._cython_op_ndim_compat(
    551             values,
    552             min_count=min_count,

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py in _cython_op_ndim_compat(self, values, min_count, ngroups, comp_ids, mask, result_mask, **kwargs)
    342             return res.T
    343 
--> 344         return self._call_cython_op(
    345             values,
    346             min_count=min_count,

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py in _call_cython_op(self, values, min_count, ngroups, comp_ids, mask, result_mask, **kwargs)
    399 
    400         out_shape = self._get_output_shape(ngroups, values)
--> 401         func = self._get_cython_function(self.kind, self.how, values.dtype, is_numeric)
    402         values = self._get_cython_vals(values)
    403         out_dtype = self._get_out_dtype(values.dtype)

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py in _get_cython_function(cls, kind, how, dtype, is_numeric)
    198             elif "object" not in f.__signatures__:
    199                 # raise NotImplementedError here rather than TypeError later
--> 200                 raise NotImplementedError(
    201                     f"function is not implemented for this dtype: "
    202                     f"[how->{how},dtype->{dtype_str}]"

NotImplementedError: function is not implemented for this dtype: [how->cumsum,dtype->object]

## === cell 12
import matplotlib.pyplot as plt
import seaborn as sns

corr_data = train_df.drop(['index','id','sequence','structure','predicted_loop_type', 'seq_length','seq_scored'], axis = 1).corr()

corr_data


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
TypeError: float() argument must be a string or a real number, not 'list'

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1451525039.py in <cell line: 0>()
      2 import seaborn as sns
      3 
----> 4 corr_data = train_df.drop(['index','id','sequence','structure','predicted_loop_type', 'seq_length','seq_scored'], axis = 1).corr()
      5 
      6 corr_data

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in corr(self, method, min_periods, numeric_only)
  11047         cols = data.columns
  11048         idx = cols.copy()
> 11049         mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)
  11050 
  11051         if method == "pearson":

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in to_numpy(self, dtype, copy, na_value)
   1991         if dtype is not None:
   1992             dtype = np.dtype(dtype)
-> 1993         result = self._mgr.as_array(dtype=dtype, copy=copy, na_value=na_value)
   1994         if result.dtype is not dtype:
   1995             result = np.asarray(result, dtype=dtype)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in as_array(self, dtype, copy, na_value)
   1692                 arr.flags.writeable = False
   1693         else:
-> 1694             arr = self._interleave(dtype=dtype, na_value=na_value)
   1695             # The underlying data was copied within _interleave, so no need
   1696             # to further copy if copy=True or setting na_value

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in _interleave(self, dtype, na_value)
   1751             else:
   1752                 arr = blk.get_values(dtype)
-> 1753             result[rl.indexer] = arr
   1754             itemmask[rl.indexer] = 1
   1755 

ValueError: setting an array element with a sequence.

## === cell 13
mask = np.ma.masked_inside(corr_data.values, -0.15, 0.15).mask #get most powerful features

plt.figure(figsize = (13,13))
sns.heatmap(corr_data, annot = True, mask = mask)


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4045757733.py in <cell line: 0>()
----> 1 mask = np.ma.masked_inside(corr_data.values, -0.15, 0.15).mask #get most powerful features
      2 #print(masked)
      3 
      4 plt.figure(figsize = (13,13))
      5 sns.heatmap(corr_data, annot = True, mask = mask)

NameError: name 'corr_data' is not defined

## === cell 14
'''
For tensorflow compatibility, metrics should have signature f(y_true, y_pred)
For sklearn compatibility, metrics should have signature f(y_true, y_pred, **kwargs)
'''

import tensorflow as tf

def score(raw_values = False, use_tf = False, **kwargs):
    '''
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
    '''
    
    
    
    
    '''
    col_dict = {
        'reactivity':0,
        'deg_Mg_pH10':1,
        'deg_Mg_50C':3
    }'''
    
    
    multi = 'uniform_average'
    if raw_values:
        multi = 'raw_values'
    
    def loss(y_true, y_pred):
        '''
        y_true & y_pred may have more columns than needed for scoring
        select only necessary ones for scoring

        y_true & y_pred have shapes (n, 5), where n is # of id_seqpos combos
        '''
        from sklearn.metrics import mean_squared_error
        y_true = np.array(y_true) #convert to np for convenience
        y_pred = np.array(y_pred)

        '''
        #only select scored columns
        y_true = y_true[:, list(col_dict.values())]
        y_pred = y_pred[:, list(col_dict.values())]
        '''

        metric = mean_squared_error(y_true, y_pred, squared = False, multioutput = multi)
        return metric
    
    def loss_tf(y_true, y_pred):
        from sklearn.metrics import mean_squared_error
        
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)
        
        '''
        #set values in y_pred to be same as y_true for unscored columns
        #this ensures that these columns do not contribute to loss
        for c in unscored:
            y_pred[:, c] = y_true[:, c]
        '''
        
        metric = mean_squared_error(y_true, y_pred, squared = False, multioutput = multi)
        return metric
        
    if not use_tf:
        return loss
    else:
        return loss_tf


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.linear_model import LinearRegression


def make_model():
    shape = X_train.shape[1:] #this seems like bad functional programming, please change it
    
    inputs = tf.keras.Input(shape = shape)
    x = layers.Dense(100, activation = 'relu')(inputs)
    x = layers.Dense(30, activation = 'relu')(x)
    x = layers.Dense(5, activation = 'linear')(x) #need output layer of 5
    
    model = tf.keras.Model(inputs = inputs, outputs = x)
    
    
    
    optimizer = 'adam'
    model.compile(optimizer = optimizer, loss = 'mse', metrics = [])
    
    return model


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/297277797.py in <cell line: 0>()
      1 import tensorflow as tf
      2 import tensorflow.keras.layers as layers
----> 3 from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
      4 from sklearn.linear_model import LinearRegression
      5 

ModuleNotFoundError: No module named 'tensorflow.keras.wrappers.scikit_learn'

## === cell 18
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

TF_FITPARAMS = {'epochs': 100,
               'batch_size':5000}


fp = TF_FITPARAMS

model = KerasRegressor(build_fn = make_model)
history = model.fit(X_train, y_train, **fp)


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/952863846.py in <cell line: 0>()
----> 1 from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
      2 
      3 TF_FITPARAMS = {'epochs': 100,
      4                'batch_size':5000}
      5 

ModuleNotFoundError: No module named 'tensorflow.keras.wrappers.scikit_learn'

## === cell 19
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


## === cell 22
test_df = read_json('../input/stanford-covid-vaccine/test.json')

test_df


## === cell 23
temp = feature_engineer(test_df, train = False)
test_df = temp

test_df


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NotImplementedError                       Traceback (most recent call last)
/tmp/ipykernel_11/1530070190.py in <cell line: 0>()
----> 1 temp = feature_engineer(test_df, train = False)
      2 test_df = temp
      3 
      4 test_df

/tmp/ipykernel_11/20449227.py in feature_engineer(df, train, **kwargs)
     20     #adds seqpos column to record position of each row in each id's individual sequence
     21     data['seqpos'] = 1
---> 22     data['seqpos'] = data.groupby('id').cumsum()['seqpos'] - 1
     23 
     24     #adds nucleotide column to record base (A,C,G,U) at position seqpos

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in cumsum(self, axis, *args, **kwargs)
   4934             return self._python_apply_general(f, self._selected_obj, is_transform=True)
   4935 
-> 4936         return self._cython_transform("cumsum", **kwargs)
   4937 
   4938     @final

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/generic.py in _cython_transform(self, how, numeric_only, axis, **kwargs)
   1700         # We could use `mgr.apply` here and not have to set_axis, but
   1701         #  we would have to do shape gymnastics for ArrayManager compat
-> 1702         res_mgr = mgr.grouped_reduce(arr_func)
   1703         res_mgr.set_axis(1, mgr.axes[1])
   1704 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in grouped_reduce(self, func)
   1467                 #  while others do not.
   1468                 for sb in blk._split():
-> 1469                     applied = sb.apply(func)
   1470                     result_blocks = extend_blocks(applied, result_blocks)
   1471             else:

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in apply(self, func, **kwargs)
    391         one
    392         """
--> 393         result = func(self.values, **kwargs)
    394 
    395         result = maybe_coerce_values(result)

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/generic.py in arr_func(bvalues)
   1694 
   1695         def arr_func(bvalues: ArrayLike) -> ArrayLike:
-> 1696             return self._grouper._cython_operation(
   1697                 "transform", bvalues, how, 1, **kwargs
   1698             )

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py in _cython_operation(self, kind, values, how, axis, min_count, **kwargs)
    829         ids, _, _ = self.group_info
    830         ngroups = self.ngroups
--> 831         return cy_op.cython_operation(
    832             values=values,
    833             axis=axis,

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py in cython_operation(self, values, axis, min_count, comp_ids, ngroups, **kwargs)
    548             )
    549 
--> 550         return self._cython_op_ndim_compat(
    551             values,
    552             min_count=min_count,

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py in _cython_op_ndim_compat(self, values, min_count, ngroups, comp_ids, mask, result_mask, **kwargs)
    342             return res.T
    343 
--> 344         return self._call_cython_op(
    345             values,
    346             min_count=min_count,

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py in _call_cython_op(self, values, min_count, ngroups, comp_ids, mask, result_mask, **kwargs)
    399 
    400         out_shape = self._get_output_shape(ngroups, values)
--> 401         func = self._get_cython_function(self.kind, self.how, values.dtype, is_numeric)
    402         values = self._get_cython_vals(values)
    403         out_dtype = self._get_out_dtype(values.dtype)

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py in _get_cython_function(cls, kind, how, dtype, is_numeric)
    198             elif "object" not in f.__signatures__:
    199                 # raise NotImplementedError here rather than TypeError later
--> 200                 raise NotImplementedError(
    201                     f"function is not implemented for this dtype: "
    202                     f"[how->{how},dtype->{dtype_str}]"

NotImplementedError: function is not implemented for this dtype: [how->cumsum,dtype->object]

## === cell 24
X_test = test_df.drop(drop_cols, axis = 1)

X_test


## === cell 25

test_pred = model.predict(X_test)
test_pred


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1604162581.py in <cell line: 0>()
      1 #do actual prediction
      2 
----> 3 test_pred = model.predict(X_test)
      4 test_pred

NameError: name 'model' is not defined

## === cell 26

sub_df = test_df['id'] + '_' + test_df['seqpos'].astype(str)
sub_df = sub_df.reset_index()

temp = pd.DataFrame(test_pred)
sub_df = pd.merge(sub_df, temp, left_index = True, right_index = True)
del sub_df['index']
sub_df.columns = ['id_seqpos'] + target_cols

sub_df


## --- ERROR in cell 26, traceback:
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

KeyError: 'seqpos'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3840177988.py in <cell line: 0>()
      2 
      3 #create id_seqpos column
----> 4 sub_df = test_df['id'] + '_' + test_df['seqpos'].astype(str)
      5 sub_df = sub_df.reset_index()
      6 

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

KeyError: 'seqpos'

## === cell 27
sns.pairplot(y_train)


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3638204652.py in <cell line: 0>()
----> 1 sns.pairplot(y_train)

/usr/local/lib/python3.11/dist-packages/seaborn/axisgrid.py in pairplot(data, hue, hue_order, palette, vars, x_vars, y_vars, kind, diag_kind, markers, height, aspect, corner, dropna, plot_kws, diag_kws, grid_kws, size)
   2112     # Set up the PairGrid
   2113     grid_kws.setdefault("diag_sharey", diag_kind == "hist")
-> 2114     grid = PairGrid(data, vars=vars, x_vars=x_vars, y_vars=y_vars, hue=hue,
   2115                     hue_order=hue_order, palette=palette, corner=corner,
   2116                     height=height, aspect=aspect, dropna=dropna, **grid_kws)

/usr/local/lib/python3.11/dist-packages/seaborn/axisgrid.py in __init__(self, data, hue, vars, x_vars, y_vars, hue_order, palette, hue_kws, corner, diag_sharey, height, aspect, layout_pad, despine, dropna)
   1264 
   1265         if not x_vars:
-> 1266             raise ValueError("No variables found for grid columns.")
   1267         if not y_vars:
   1268             raise ValueError("No variables found for grid rows.")

ValueError: No variables found for grid columns.

## === cell 28

sns.pairplot(sub_df)


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1201416118.py in <cell line: 0>()
      2 #ax = plt.subplot(1,1,1)
      3 
----> 4 sns.pairplot(sub_df)
      5 #sub_df.plot(y='deg_Mg_pH10', ax = ax)

NameError: name 'sub_df' is not defined

## === cell 29
sub_df.to_csv('submission.csv', index = False)


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1903632625.py in <cell line: 0>()
----> 1 sub_df.to_csv('submission.csv', index = False)

NameError: name 'sub_df' is not defined
