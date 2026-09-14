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

No external packages required in the script and installed.

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

0.3756351841458719

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 2
import json,os, math

import subprocess

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.keras.backend as K
import plotly.express as px
import tensorflow.keras.layers as L
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')

import tensorflow_addons as tfa

from itertools import combinations_with_replacement
from sklearn.model_selection import train_test_split, KFold,  StratifiedKFold,GroupKFold
from keras.utils import plot_model
from colorama import Fore, Back, Style

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
SEED = 53

n_folds=5

debug=True

Window_features = True



model_name="GG"                ## MODEL NAME (Files will save according to this )
epochs=100              ## NUMBER OF EPOCHS MODEL TRAIN IN EACH FOLD. USE 3, 5, 7,... 
BATCH_SIZE = 32                 ## NUMBER OF BATCH_SIZE USE 16, 32, 64, 128,...
n_layers = 2                  ## Number of Layers Present in model # ex. 3 Layer of GRU Model
layers = ["GRU","GRU"]   ## Stacking sequence of GRU and LSTM (list of length == n_layers)
hidden_dim = [128, 128]    ## Hidden Dimension in Model (Default : [128,128]) (list of length == n_layers)
dropout = [0.5, 0.5]       ## 1.0 means no dropout, and 0.0 means no outputs from the layer.
sp_dropout = 0.2                ## SpatialDropout1D (Fraction of the input units to drop) [https://stackoverflow.com/a/55244985]
embed_dim = 250                  ## Output Dimention of Embedding Layer (Default : 75)
num_hidden_units = 8      ## Number of GRU units after num_input layer



Cosine_Schedule = True         ## cosine_schedule Rate
Rampup_decy_lr = False           ## Rampup decy lr Schedule

## === cell 6
def seed_everything(seed=1234):   
    np.random.seed(seed)   
    tf.random.set_seed(seed)   
    os.environ['PYTHONHASHSEED'] = str(seed)   
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
seed_everything(SEED)

## === cell 8
target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C', 'deg_pH10', 'deg_50C']
window_columns = ['sequence','structure','predicted_loop_type']

categorical_features = ['sequence', 'structure', 'predicted_loop_type',]

cat_feature = len(categorical_features)
if Window_features:
    cat_feature += len(window_columns)

numerical_features = ['BPPS_Max','BPPS_nb', 'BPPS_sum',
                      'positional_entropy',
                      'stems', 'interior_loops', 'multiloops',#'hairpin loops', 'fiveprimes', 'threeprimes', 
                      'A_percent', 'G_percent','C_percent', 'U_percent', 
                      'U-G', 'C-G', 'U-A', 'G-C', 'A-U', 'G-U', 
                      'pair_map', 'pair_distance', ]
    

num_features = len(numerical_features)  ## ( Numerical Features Only)

feature_cols = categorical_features + numerical_features
pred_col_names = ["pred_"+c_name for c_name in target_cols]

target_eval_col = ['reactivity','deg_Mg_pH10','deg_Mg_50C']
pred_eval_col = ["pred_"+c_name for c_name in target_eval_col]

## === cell 10
data_dir = '/kaggle/input/stanford-covid-vaccine/'
fearure_data_path = '../input/openvaccine/'


train  = pd.read_json(fearure_data_path+'train.json')
test = pd.read_json(fearure_data_path+'test.json')

sample_sub = pd.read_csv(data_dir + 'sample_submission.csv')

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1166239412.py in <cell line: 0>()
      5 # test = pd.read_csv(fearure_data_path+'test.csv')
      6 
----> 7 train  = pd.read_json(fearure_data_path+'train.json')
      8 test = pd.read_json(fearure_data_path+'test.json')
      9 

/usr/local/lib/python3.11/dist-packages/pandas/io/json/_json.py in read_json(path_or_buf, orient, typ, dtype, convert_axes, convert_dates, keep_default_dates, precise_float, date_unit, encoding, encoding_errors, lines, chunksize, compression, nrows, storage_options, dtype_backend, engine)
    789         convert_axes = True
    790 
--> 791     json_reader = JsonReader(
    792         path_or_buf,
    793         orient=orient,

/usr/local/lib/python3.11/dist-packages/pandas/io/json/_json.py in __init__(self, filepath_or_buffer, orient, typ, dtype, convert_axes, convert_dates, keep_default_dates, precise_float, date_unit, encoding, lines, chunksize, compression, nrows, storage_options, encoding_errors, dtype_backend, engine)
    902             self.data = filepath_or_buffer
    903         elif self.engine == "ujson":
--> 904             data = self._get_data_from_filepath(filepath_or_buffer)
    905             self.data = self._preprocess_data(data)
    906 

/usr/local/lib/python3.11/dist-packages/pandas/io/json/_json.py in _get_data_from_filepath(self, filepath_or_buffer)
    958             and not file_exists(filepath_or_buffer)
    959         ):
--> 960             raise FileNotFoundError(f"File {filepath_or_buffer} does not exist")
    961         else:
    962             warnings.warn(

FileNotFoundError: File ../input/openvaccine/train.json does not exist

## === cell 11
train = train[train['signal_to_noise'] >= 0.5]

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1539379604.py in <cell line: 0>()
      1 # train = train[train['SN_filter'] == 1]
----> 2 train = train[train['signal_to_noise'] >= 0.5]

NameError: name 'train' is not defined

## === cell 12
def pair_feature(row):
    arr = list(row)
    its = [iter(['_']+arr[:]) ,iter(arr[1:]+['_'])]
    list_touple = list(zip(*its))
    return list(map("".join,list_touple))

## === cell 13
def preprocess_categorical_inputs(df, cols=categorical_features,Window_features=Window_features):
    
    if Window_features:
        for c in window_columns:
            df["pair_"+c] = df[c].apply(pair_feature)
            cols.append("pair_"+c)
    cols = list(set(cols))
    
    return np.transpose(
        np.array(
            df[cols]
            .applymap(lambda seq: [token2int[x] for x in seq])
            .values
            .tolist()
        ),
        (0, 2, 1)
    )

## === cell 14
def preprocess_numerical_inputs(df, cols=numerical_features):
    
    return np.transpose(
        np.array(
            df[cols].values.tolist()
        ),
        (0, 2, 1)
    )

## === cell 15
token_list = list("().ACGUBshftim")
if Window_features:
    comb = combinations_with_replacement(list('_().ACGUBshftim'*2), 2) 
    token_list += list(set(list(map("".join,comb))))

token2int = {x:i for i, x in enumerate(list(set(token_list)))}
print("token_list Size :",len(token_list))
    
train_inputs_all_cat = preprocess_categorical_inputs(train,cols=categorical_features)
train_inputs_all_num = preprocess_numerical_inputs(train,cols=numerical_features)
train_labels_all = np.array(train[target_cols].values.tolist(),dtype =np.float32).transpose((0, 2, 1))

print("Train categorical Features Shape : ",train_inputs_all_cat.shape)
print("Train numerical Features Shape : ",train_inputs_all_num.shape)
print("Train labels Shape : ",train_labels_all.shape)

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/523902203.py in <cell line: 0>()
      4 token_list = list("().ACGUBshftim")
      5 if Window_features:
----> 6     comb = combinations_with_replacement(list('_().ACGUBshftim'*2), 2)
      7     token_list += list(set(list(map("".join,comb))))
      8 

NameError: name 'combinations_with_replacement' is not defined

## === cell 19
public_df = test.query("seq_length == 107")
private_df = test.query("seq_length == 130")
print("public_df : ",public_df.shape)
print("private_df : ",private_df.shape)

public_inputs_cat = preprocess_categorical_inputs(public_df)
private_inputs_cat = preprocess_categorical_inputs(private_df)

public_inputs_num = preprocess_numerical_inputs(public_df,cols=numerical_features)
private_inputs_num = preprocess_numerical_inputs(private_df,cols=numerical_features)

print("Public categorical Features Shape : ",public_inputs_cat.shape)
print("Public numerical Features Shape : ",public_inputs_num.shape)

print("Private categorical Features Shape : ",private_inputs_cat.shape)
print("Private numerical Features Shape : ",private_inputs_num.shape)

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2851450592.py in <cell line: 0>()
----> 1 public_df = test.query("seq_length == 107")
      2 private_df = test.query("seq_length == 130")
      3 print("public_df : ",public_df.shape)
      4 print("private_df : ",private_df.shape)
      5 

NameError: name 'test' is not defined

## === cell 21



def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true[:,:,:3] - y_pred[:,:,:3]), axis=1)
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)

## === cell 23
def get_lr_callback(batch_size=8):
    lr_start   = 0.00001
    lr_max     = 0.004
    lr_min     = 0.00005
    lr_ramp_ep = 45
    lr_sus_ep  = 2
    lr_decay   = 0.8
   
    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
            
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
            
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
            
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    return lr_callback

## === cell 25
def get_cosine_schedule_with_warmup(lr,num_warmup_steps, num_training_steps, num_cycles=3.5):
    """
    Modified version of the get_cosine_schedule_with_warmup from huggingface.
    (https://huggingface.co/transformers/_modules/transformers/optimization.html#get_cosine_schedule_with_warmup)

    Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lrfn(epoch):
        if epoch < num_warmup_steps:
            return (float(epoch) / float(max(1, num_warmup_steps))) * lr
        progress = float(epoch - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr

    return tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)

## === cell 27
def lstm_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(
                                tf.keras.layers.LSTM(hidden_dim,
                                dropout=dropout,
                                return_sequences=True,
                                kernel_initializer = 'orthogonal'))

## === cell 28
def gru_layer(hidden_dim, dropout):
    return L.Bidirectional(
        L.GRU(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer='orthogonal')
    )

## === cell 31
def build_model(embed_size, 
                seq_len = 107, 
                pred_len = 68, 
                dropout = dropout, 
                sp_dropout = sp_dropout, 
                num_features = num_features,
                num_hidden_units = num_hidden_units,
                embed_dim = embed_dim,
                layers = layers, 
                hidden_dim = hidden_dim, 
                n_layers = n_layers,
                cat_feature = cat_feature):
    
    inputs = L.Input(shape=(seq_len, cat_feature),name='category_input')
    embed = L.Embedding(input_dim=embed_size, output_dim=embed_dim)(inputs)
    reshaped = tf.reshape(embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))
    reshaped = L.SpatialDropout1D(sp_dropout)(reshaped)
    reshaped_conv = tf.keras.layers.Conv1D(filters=512, kernel_size=3,strides=1, padding='same', activation='elu')(reshaped)
    
    numerical_input = L.Input(shape=(seq_len, num_features), name='numeric_input')
    
    hidden = L.concatenate([reshaped_conv, numerical_input])
    hidden_1 = tf.keras.layers.Conv1D(filters=256, kernel_size=4,strides=1, padding='same', activation='elu')(hidden)
    hidden = gru_layer(128, 0.5)(hidden_1)
    hidden = L.concatenate([hidden, hidden_1])
    
    for x in range(n_layers):
        if layers[x] == "GRU":
            hidden = gru_layer(hidden_dim[x], dropout[x])(hidden)
        else:
            hidden = lstm_layer(hidden_dim[x], dropout[x])(hidden)
        
        hidden = L.concatenate([hidden, hidden_1])
            
    truncated = hidden[:, :pred_len]
    
    out = L.Dense(5)(truncated)
    
    model = tf.keras.Model(inputs=[inputs] + [numerical_input], outputs=out)
    
    adam = tf.optimizers.Adam()
    radam = tfa.optimizers.RectifiedAdam()
    lookahead = tfa.optimizers.Lookahead(adam, sync_period=6)
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6)
    
    model.compile(optimizer=radam, loss=MCRMSE)
    
    return model

## === cell 33
model = build_model(embed_size=len(token_list))
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2818588091.py in <cell line: 0>()
----> 1 model = build_model(embed_size=len(token_list))
      2 plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

/tmp/ipykernel_11/3456967479.py in build_model(embed_size, seq_len, pred_len, dropout, sp_dropout, num_features, num_hidden_units, embed_dim, layers, hidden_dim, n_layers, cat_feature)
     14     inputs = L.Input(shape=(seq_len, cat_feature),name='category_input')
     15     embed = L.Embedding(input_dim=embed_size, output_dim=embed_dim)(inputs)
---> 16     reshaped = tf.reshape(embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))
     17     reshaped = L.SpatialDropout1D(sp_dropout)(reshaped)
     18     reshaped_conv = tf.keras.layers.Conv1D(filters=512, kernel_size=3,strides=1, padding='same', activation='elu')(reshaped)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/weak_tensor_ops.py in wrapper(*args, **kwargs)
     86   def wrapper(*args, **kwargs):
     87     if not ops.is_auto_dtype_conversion_enabled():
---> 88       return op(*args, **kwargs)
     89     bound_arguments = signature.bind(*args, **kwargs)
     90     bound_arguments.apply_defaults()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/keras_tensor.py in __tf_tensor__(self, dtype, name)
    136 
    137     def __tf_tensor__(self, dtype=None, name=None):
--> 138         raise ValueError(
    139             "A KerasTensor cannot be used as input to a TensorFlow function. "
    140             "A KerasTensor is a symbolic placeholder for a shape and dtype, "

ValueError: A KerasTensor cannot be used as input to a TensorFlow function. A KerasTensor is a symbolic placeholder for a shape and dtype, used when constructing Keras Functional models or Keras Functions. You can only use it as input to a Keras layer or a Keras operation (from the namespaces `keras.layers` and `keras.operations`). You are likely doing something like:

```
x = Input(...)
...
tf_fn(x)  # Invalid.
```

What you should do instead is wrap `tf_fn` in a layer:

```
class MyLayer(Layer):
    def call(self, x):
        return tf_fn(x)

x = MyLayer()(x)
```


## === cell 36
def get_stratify_group(row):
    snf = row['SN_filter']
    snr = row['signal_to_noise']
    cnt = row['cnt']
    id_ = row['id']
    structure = row['structure']
    if snf == 0:
        if snr<0:
            snr_c = 0
        elif 0<= snr < 2:
            snr_c = 1
        elif 2<= snr < 4:
            snr_c = 2
        elif 4<= snr < 5.5:
            snr_c = 3
        elif 5.5<= snr < 10:
            snr_c = 4
        elif snr >= 10:
            snr_c = 5
            
    else: # snf == 1
        if snr<0:
            snr_c = 6
        elif 0<= snr < 1:
            snr_c = 7
        elif 1<= snr < 2:
            snr_c = 8
        elif 2<= snr < 3:
            snr_c = 9
        elif 3<= snr < 4:
            snr_c = 10
        elif 4<= snr < 5:
            snr_c = 11
        elif 5<= snr < 6:
            snr_c = 12
        elif 6<= snr < 7:
            snr_c = 13
        elif 7<= snr < 8:
            snr_c = 14
        elif 8<= snr < 9:
            snr_c = 15
        elif 9<= snr < 10:
            snr_c = 15
        elif snr >= 10:
            snr_c = 16
        
    return '{}_{}'.format(id_,snr_c)

train['stratify_group'] = train.apply(get_stratify_group, axis=1)
train['stratify_group'] = train['stratify_group'].astype('category').cat.codes

skf = StratifiedKFold(n_folds, shuffle=True, random_state=SEED)
gkf = GroupKFold(n_splits=n_folds)
fig, ax = plt.subplots(n_folds,3,figsize=(20,5*n_folds))

for Fold, (train_index, val_index) in enumerate(gkf.split(train_inputs_all_cat, groups=train['stratify_group'])):
    print(Fore.YELLOW);print('#'*45);print("###  Fold : ", str(Fold+1));print('#'*45);print(Style.RESET_ALL)
    
    train_data = train.iloc[train_index]
    val_data = train.iloc[val_index]
    print("Augmented data Present in Val Data : ",len(val_data[val_data['cnt'] != 1]))
    print("Augmented data Present in Train Data : ",len(train_data[train_data['cnt'] != 1]))
    val_data = val_data[val_data['cnt'] == 1]
    print("Data Lekage : ",len(val_data[val_data['id'].isin(train_data['id'])]))
    
    print("number of Train Data points : ",len(train_data))
    print("number of val_data Data points : ",len(val_data))
    print("number of unique Structure in Train data : ", len(train_data.structure.unique()))
    print("number of unique Structure in val data : ",len(val_data.structure.unique()), val_data.structure.value_counts()[:5].values)
    
    print("Train SN_Filter == 1 : ", len(train_data[train_data['SN_filter']==1]))
    print("val_data SN_Filter == 1 : ", len(val_data[val_data['SN_filter']==1]))
    print("Train SN_Filter == 0 : ", len(train_data[train_data['SN_filter']==0]))
    print("val_data SN_Filter == 0 : ", len(val_data[val_data['SN_filter']==0]))
    
    print("Unique ID :",len(train_data.id.unique()))
    sns.kdeplot(train[train['SN_filter']==0]['signal_to_noise'],ax=ax[Fold][0],color="Red",label='Train All')
    sns.kdeplot(train_data[train_data['SN_filter']==0]['signal_to_noise'],ax=ax[Fold][0],color="Blue",label='Train')
    sns.kdeplot(val_data[val_data['SN_filter']==0]['signal_to_noise'],ax=ax[Fold][0],color="Green",label='Validation')            
    ax[Fold][0].set_title(f'Fold : {Fold+1} Signal/Noise & SN_filter == 0')
    
    sns.kdeplot(train[train['SN_filter']==1]['signal_to_noise'],ax=ax[Fold][1],color="Red",label='Train All')
    sns.kdeplot(train_data[train_data['SN_filter']==1]['signal_to_noise'],ax=ax[Fold][1],color="Blue",label='Train')
    sns.kdeplot(val_data[val_data['SN_filter']==1]['signal_to_noise'],ax=ax[Fold][1],color="Green",label='Validation')            
    ax[Fold][1].set_title(f'Fold : {Fold+1} Signal/Noise & SN_filter == 1')
    
    sns.kdeplot(train['signal_to_noise'],ax=ax[Fold][2],color="Red",label='Train All')
    sns.kdeplot(train_data['signal_to_noise'],ax=ax[Fold][2],color="Blue",label='Train')
    sns.kdeplot(val_data['signal_to_noise'],ax=ax[Fold][2],color="Green",label='Validation')            
    ax[Fold][2].set_title(f'Fold : {Fold+1} Signal/Noise')
            
plt.show()


## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3570735127.py in <cell line: 0>()
     47     return '{}_{}'.format(id_,snr_c)
     48 
---> 49 train['stratify_group'] = train.apply(get_stratify_group, axis=1)
     50 train['stratify_group'] = train['stratify_group'].astype('category').cat.codes
     51 

NameError: name 'train' is not defined

## === cell 37
submission = pd.DataFrame(index=sample_sub.index, columns=target_cols).fillna(0) # test dataframe with 0 values

val_losses = []
historys = []
oof_preds_all = []
stacking_pred_all = []

kf = KFold(n_folds, shuffle=True, random_state=SEED)
skf = StratifiedKFold(n_folds, shuffle=True, random_state=SEED)
gkf = GroupKFold(n_splits=n_folds)

for Fold, (train_index, val_index) in enumerate(gkf.split(train_inputs_all_cat, groups=train['stratify_group'])):
    print(Fore.YELLOW);print('#'*45);print("###  Fold : ", str(Fold+1));print('#'*45);print(Style.RESET_ALL)
    print(f"|| Batch_size: {BATCH_SIZE} \n|| n_layers: {n_layers} \n|| embed_dim: {embed_dim}")
    print(f"|| cat_feature: {cat_feature} \n|| num_features: {num_features}")
    print(f"|| layers : {layers} \n|| hidden_dim: {hidden_dim} \n|| dropout: {dropout} \n|| sp_dropout: {sp_dropout}")
    
    
    train_data = train.iloc[train_index]
    val_data = train.iloc[val_index]
    
    print("|| number Augmented data Present in Val Data : ",len(val_data[val_data['cnt'] != 1]))
    print("|| number Augmented data Present in Train Data : ",len(train_data[train_data['cnt'] != 1]))
    print("|| Data Lekage : ",len(val_data[val_data['id'].isin(train_data['id'])]))
    
    val_data = val_data[val_data['cnt'] == 1]
    model_train = build_model(embed_size=len(token_list))
    model_short = build_model(embed_size=len(token_list),seq_len=107, pred_len=107)
    model_long = build_model(embed_size=len(token_list),seq_len=130, pred_len=130)

    train_inputs_cat = preprocess_categorical_inputs(train_data,cols=categorical_features)
    train_inputs_num = preprocess_numerical_inputs(train_data,cols=numerical_features)
    train_labels = np.array(train_data[target_cols].values.tolist(),dtype =np.float32).transpose((0, 2, 1))
    
    val_inputs_cat = preprocess_categorical_inputs(val_data,cols=categorical_features)
    val_inputs_num = preprocess_numerical_inputs(val_data,cols=numerical_features)
    val_labels = np.array(val_data[target_cols].values.tolist(),dtype =np.float32).transpose((0, 2, 1))
    
    
    csv_logger = tf.keras.callbacks.CSVLogger(f'Fold_{Fold}_log.csv', separator=',', append=False)
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(f'{model_name}_Fold_{Fold}.h5', 
                                                    monitor='val_loss', 
                                                    verbose=0, 
                                                    mode='min', 
                                                    save_freq='epoch')
    
    if Cosine_Schedule:
        lr_schedule= get_cosine_schedule_with_warmup(lr=0.001, num_warmup_steps=20, num_training_steps=epochs)
    elif Rampup_decy_lr : 
        lr_schedule = get_lr_callback(BATCH_SIZE)
    else:
        lr_schedule = tf.keras.callbacks.ReduceLROnPlateau()
        
    history = model_train.fit(
        {'numeric_input': train_inputs_num,
           'category_input': train_inputs_cat} , train_labels, 
        validation_data=({'numeric_input': val_inputs_num,
                          'category_input': val_inputs_cat}
                         ,val_labels),
        batch_size=BATCH_SIZE,
        epochs=epochs, 
        callbacks=[lr_schedule, checkpoint, csv_logger,lr_schedule],
        verbose=1 if debug else 0
    )
    
    print("Min Validation Loss : ", min(history.history['val_loss']))
    print("Min Validation Epoch : ",np.argmin( history.history['val_loss'] )+1)
    val_losses.append(min(history.history['val_loss']))
    historys.append(history)

    model_short.load_weights(f'{model_name}_Fold_{Fold}.h5')
    model_long.load_weights(f'{model_name}_Fold_{Fold}.h5')
    
    public_preds = model_short.predict({'numeric_input': public_inputs_num,
                                        'category_input': public_inputs_cat})
    
    private_preds = model_long.predict({'numeric_input': private_inputs_num,
                                        'category_input': private_inputs_cat})
    
    oof_preds = model_train.predict({'numeric_input': val_inputs_num,
                                        'category_input': val_inputs_cat})
    
    stacking_pred = model_short.predict({'numeric_input': val_inputs_num,
                                        'category_input': val_inputs_cat})
    
    preds_model = []
    for df, preds in [(public_df, public_preds), (private_df, private_preds)]:
        for i, uid in enumerate(df.id):
            single_pred = preds[i]

            single_df = pd.DataFrame(single_pred, columns=target_cols)
            single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]
            
            preds_model.append(single_df)
    
    preds_model_df = pd.concat(preds_model)
    preds_model_df = preds_model_df.groupby(['id_seqpos'],as_index=True).mean()
    submission[target_cols] += preds_model_df[target_cols].values / n_folds
    
    for df, preds in [(val_data, oof_preds)]:
        for i, uid in enumerate(df.id):
            single_pred = preds[i]
            single_label = val_labels[i]
            single_label_df = pd.DataFrame(single_label, columns=target_cols)
            single_label_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_label_df.shape[0])]
            single_label_df['id'] = [f'{uid}' for x in range(single_label_df.shape[0])]
            single_label_df['s_id'] = [x for x in range(single_label_df.shape[0])]
            single_df = pd.DataFrame(single_pred, columns=pred_col_names)
            single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]
            
            single_df = pd.merge(single_label_df,single_df, on="id_seqpos", how="left")
            
            oof_preds_all.append(single_df)
    
    for df, preds in [(val_data, stacking_pred)]:
        for i, uid in enumerate(df.id):
            single_pred = preds[i]
            single_df = pd.DataFrame(single_pred, columns=pred_col_names)
            single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]
            single_df['id'] = [uid for x in range(single_df.shape[0])]
            stacking_pred_all.append(single_df)
        
    history_data = pd.read_csv(f'Fold_{Fold}_log.csv')
    EPOCHS = len(history_data['epoch'])
    history = pd.DataFrame({'history':history_data.to_dict('list')})
    fig = plt.figure(figsize=(15,5))
    plt.plot(np.arange(EPOCHS),history.history['lr'],'-',label='Learning Rate',color='#ff7f0e')
    x = np.argmax( history.history['lr'] ); y = np.max( history.history['lr'] )
    xdist = plt.xlim()[1] - plt.xlim()[0]; ydist = plt.ylim()[1] - plt.ylim()[0]
    plt.scatter(x,y,s=200,color='#1f77b4'); plt.text(x-0.03*xdist,y-0.13*ydist,f'Max Learning Rate : {y}' ,size=12)
    plt.ylabel('Learning Rate',size=14); plt.xlabel('Epoch',size=14)
    plt.legend(loc=1)
    plt2 = plt.gca().twinx()
    plt2.plot(np.arange(EPOCHS),history.history['loss'],'-o',label='Train Loss',color='#2ca02c')
    plt2.plot(np.arange(EPOCHS),history.history['val_loss'],'-o',label='Val Loss',color='#d62728')
    x = np.argmin( history.history['val_loss'] ); y = np.min( history.history['val_loss'] )
    ydist = plt.ylim()[1] - plt.ylim()[0]
    plt.scatter(x,y,s=200,color='#d62728'); plt.text(x-0.03*xdist,y+0.05*ydist,'min loss',size=14)
    plt.ylabel('Loss',size=14)
    fig.text(s=f"Model Name : {model_name}" , x=0.5, y=1.08, fontsize=18, ha='center', va='center',color="green")
    fig.text(s=f"|| Fold : {Fold+1} | Batch_size: {BATCH_SIZE} | num_features: {num_features} | cat_feature: {cat_feature} |n_layers: {n_layers} | embed_dim: {embed_dim} ||", x=0.5, y=1.0, fontsize=15, ha='center', va='center',color="red")
    fig.text(s=f"|| layers : {layers} | hidden_dim: {hidden_dim} | dropout: {dropout} | sp_dropout: {sp_dropout} ||", x=0.5, y=0.92, fontsize=15, ha='center', va='center',color="blue")
    plt.legend(loc=3)
    plt.savefig(f'Fold_{Fold+1}.png', bbox_inches='tight')
    plt.show()
    
submission["id_seqpos"] = preds_model_df.index
submission = pd.merge(sample_sub["id_seqpos"], submission, on="id_seqpos", how="left")
OOF = pd.concat(oof_preds_all)
stacking_df = pd.concat(stacking_pred_all)

## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/813688065.py in <cell line: 0>()
----> 1 submission = pd.DataFrame(index=sample_sub.index, columns=target_cols).fillna(0) # test dataframe with 0 values
      2 
      3 val_losses = []
      4 historys = []
      5 oof_preds_all = []

NameError: name 'sample_sub' is not defined

## === cell 41
OOF = OOF.groupby(['id_seqpos','id','s_id'],as_index=False).mean()
OOF = OOF.sort_values(['id','s_id'],ascending=[True, True])
OOF_score = MCRMSE(np.expand_dims(OOF[target_eval_col].values, axis=0), np.expand_dims(OOF[pred_eval_col].values, axis=0)).numpy()[0]
print("Overall OOF Score :",OOF_score)
OOF.to_csv('OOf.csv',index=True)
OOF.head()

## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2453898284.py in <cell line: 0>()
----> 1 OOF = OOF.groupby(['id_seqpos','id','s_id'],as_index=False).mean()
      2 OOF = OOF.sort_values(['id','s_id'],ascending=[True, True])
      3 OOF_score = MCRMSE(np.expand_dims(OOF[target_eval_col].values, axis=0), np.expand_dims(OOF[pred_eval_col].values, axis=0)).numpy()[0]
      4 print("Overall OOF Score :",OOF_score)
      5 OOF.to_csv('OOf.csv',index=True)

NameError: name 'OOF' is not defined

## === cell 43
OOF_filter_1 = pd.merge(train[['SN_filter','id']],OOF,on='id')
OOF_filter_1 = OOF_filter_1.groupby(['id_seqpos','id'],as_index=False).mean()
OOF_filter_1 = OOF_filter_1.sort_values(['id','s_id'],ascending=[True, True])
OOF_filter_1 = OOF_filter_1[OOF_filter_1['SN_filter'] == 1]
OOF_filter_1_score = MCRMSE(np.expand_dims(OOF_filter_1[target_eval_col].values, axis=0), np.expand_dims(OOF_filter_1[pred_eval_col].values, axis=0)).numpy()[0]
print("OOF_public Score :",OOF_filter_1_score)
OOF_filter_1.to_csv('OOF_filter_1.csv',index=False)
OOF_filter_1.head()

## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3804886607.py in <cell line: 0>()
----> 1 OOF_filter_1 = pd.merge(train[['SN_filter','id']],OOF,on='id')
      2 OOF_filter_1 = OOF_filter_1.groupby(['id_seqpos','id'],as_index=False).mean()
      3 OOF_filter_1 = OOF_filter_1.sort_values(['id','s_id'],ascending=[True, True])
      4 OOF_filter_1 = OOF_filter_1[OOF_filter_1['SN_filter'] == 1]
      5 OOF_filter_1_score = MCRMSE(np.expand_dims(OOF_filter_1[target_eval_col].values, axis=0), np.expand_dims(OOF_filter_1[pred_eval_col].values, axis=0)).numpy()[0]

NameError: name 'train' is not defined

## === cell 45
OOF_filter_0 = pd.merge(train[['SN_filter','id']],OOF,on='id')
OOF_filter_0 = OOF_filter_0.sort_values(['id','s_id'],ascending=[True, True])
OOF_filter_0 = OOF_filter_0[OOF_filter_0['SN_filter'] == 0]
OOF_filter_0_score = MCRMSE(np.expand_dims(OOF_filter_0[target_eval_col].values, axis=0), np.expand_dims(OOF_filter_0[pred_eval_col].values, axis=0)).numpy()[0]
print("OOF_public Score :",OOF_filter_0_score)
OOF_filter_0.to_csv('OOF_filter_0.csv',index=False)
OOF_filter_0.head()

## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/986570644.py in <cell line: 0>()
----> 1 OOF_filter_0 = pd.merge(train[['SN_filter','id']],OOF,on='id')
      2 OOF_filter_0 = OOF_filter_0.sort_values(['id','s_id'],ascending=[True, True])
      3 OOF_filter_0 = OOF_filter_0[OOF_filter_0['SN_filter'] == 0]
      4 OOF_filter_0_score = MCRMSE(np.expand_dims(OOF_filter_0[target_eval_col].values, axis=0), np.expand_dims(OOF_filter_0[pred_eval_col].values, axis=0)).numpy()[0]
      5 print("OOF_public Score :",OOF_filter_0_score)

NameError: name 'train' is not defined

## === cell 47
stacking_df.to_csv('stacking.csv', index=False)
stacking_df.head()

## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1635640884.py in <cell line: 0>()
----> 1 stacking_df.to_csv('stacking.csv', index=False)
      2 stacking_df.head()

NameError: name 'stacking_df' is not defined

## === cell 49
submission.to_csv('submission.csv', index=False)
submission.head()

## --- ERROR in cell 49, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2177586279.py in <cell line: 0>()
----> 1 submission.to_csv('submission.csv', index=False)
      2 submission.head()

NameError: name 'submission' is not defined

## === cell 51
print("|No|OOF_score|OOF_filter_1 |OOF_filter_0|LB|n_folds|Window_features|cat_feature|num_features |epochs|BATCH_SIZE|")
print("|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|")
print(f"|-|{OOF_score}|{OOF_filter_1_score}|{OOF_filter_0_score}|-|{n_folds}|{Window_features}|{cat_feature}|{num_features}|{epochs}|{BATCH_SIZE}|")

## --- ERROR in cell 51, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/362224246.py in <cell line: 0>()
      1 print("|No|OOF_score|OOF_filter_1 |OOF_filter_0|LB|n_folds|Window_features|cat_feature|num_features |epochs|BATCH_SIZE|")
      2 print("|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|")
----> 3 print(f"|-|{OOF_score}|{OOF_filter_1_score}|{OOF_filter_0_score}|-|{n_folds}|{Window_features}|{cat_feature}|{num_features}|{epochs}|{BATCH_SIZE}|")

NameError: name 'OOF_score' is not defined
