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

0.63824

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 0.63824) has done: 'I fix the environment-crashing TensorFlow import issue by avoiding the optional `tensorflow_addons` import path that triggers the `MessageFactory.GetPrototype` protobuf incompatibility, while keeping the model and training loop unchanged. I also fix the JSON loading error (“Trailing data”) by reading the competition JSON files with `lines=True`, which matches the actual file format in this dataset. Finally, I make the public/private test split robust (this dataset uses length 107 for both), and ensure the submission is always aligned to `sample_submission.csv` and written as `submission.csv` with the correct columns.'

# 9. Code solution

## === cell 0
import os, math, json, warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as L

from itertools import combinations_with_replacement
from sklearn.model_selection import GroupKFold

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    import seaborn as sns
except Exception:
    sns = None
try:
    from colorama import Fore, Style
except Exception:

    class _Dummy:
        YELLOW = ""
        RESET_ALL = ""

    Fore = _Dummy()
    Style = _Dummy()

HAS_TFA = False
tfa = None

try:
    from keras.utils import plot_model
except Exception:
    plot_model = None

print("TensorFlow:", tf.__version__, "| TFA:", HAS_TFA)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
SEED = 53
n_folds = 5
debug = True
Window_features = True

model_name = "GG"
epochs = 100
BATCH_SIZE = 32

n_layers = 2
layers = ["GRU", "GRU"]
hidden_dim = [128, 128]
dropout = [0.5, 0.5]
sp_dropout = 0.2
embed_dim = 250
num_hidden_units = 8  # kept for config compatibility (not used in model definition)

Cosine_Schedule = True
Rampup_decy_lr = False




## === cell 2
def seed_everything(seed=1234):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


seed_everything(SEED)



## === cell 3
target_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C", "deg_pH10", "deg_50C"]
window_columns = ["sequence", "structure", "predicted_loop_type"]

categorical_features = ["sequence", "structure", "predicted_loop_type"]

cat_feature = len(categorical_features)
if Window_features:
    cat_feature += len(window_columns)

numerical_features = [
    "BPPS_Max",
    "BPPS_nb",
    "BPPS_sum",
    "positional_entropy",
    "stems",
    "interior_loops",
    "multiloops",
    "A_percent",
    "G_percent",
    "C_percent",
    "U_percent",
    "U-G",
    "C-G",
    "U-A",
    "G-C",
    "A-U",
    "G-U",
    "pair_map",
    "pair_distance",
]
num_features = len(numerical_features)

feature_cols = categorical_features + numerical_features
pred_col_names = ["pred_" + c_name for c_name in target_cols]

target_eval_col = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
pred_eval_col = ["pred_" + c_name for c_name in target_eval_col]



## === cell 4
data_dir = "/kaggle/input/stanford-covid-vaccine/"

train = pd.read_json(os.path.join(data_dir, "train.json"), lines=True)
test = pd.read_json(os.path.join(data_dir, "test.json"), lines=True)
sample_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

print("train:", train.shape, "| test:", test.shape, "| sample_sub:", sample_sub.shape)
print("train cols:", list(train.columns)[:10], "...")
print("test cols:", list(test.columns))



## === cell 5
train = train[train["signal_to_noise"] >= 0.5].reset_index(drop=True)
print("train after SNR filter:", train.shape)

if "cnt" not in train.columns:
    train["cnt"] = 1




## === cell 6
def _make_engineered_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    seqs = out["sequence"].astype(str).values
    Ls = out["seq_length"].astype(int).values

    def frac(ch):
        return np.array([s.count(ch) / max(1, len(s)) for s in seqs], dtype=np.float32)

    out["A_percent"] = frac("A")
    out["C_percent"] = frac("C")
    out["G_percent"] = frac("G")
    out["U_percent"] = frac("U")

    structures = out["structure"].astype(str).values
    out["stems"] = np.array(
        [s.count("(") + s.count(")") for s in structures], dtype=np.float32
    ) / np.maximum(1, Ls)
    out["interior_loops"] = np.array(
        [s.count(".") for s in structures], dtype=np.float32
    ) / np.maximum(1, Ls)

    loops = out["predicted_loop_type"].astype(str).values
    out["multiloops"] = np.array(
        [lt.count("M") for lt in loops], dtype=np.float32
    ) / np.maximum(1, Ls)

    out["pair_map"] = 0.0
    out["pair_distance"] = 0.0
    out["BPPS_Max"] = 0.0
    out["BPPS_nb"] = 0.0
    out["BPPS_sum"] = 0.0
    out["positional_entropy"] = 0.0

    def dinuc_frac(dn):
        return np.array(
            [s.count(dn) / max(1, len(s) - 1) for s in seqs], dtype=np.float32
        )

    for dn in ["UG", "CG", "UA", "GC", "AU", "GU"]:
        out[dn[0] + "-" + dn[1]] = dinuc_frac(dn)

    for c in numerical_features:
        if c not in out.columns:
            out[c] = 0.0

    for c in numerical_features:
        if not isinstance(out.iloc[0][c], (list, np.ndarray)):
            out[c] = [
                ([float(v)] * int(sl))
                for v, sl in zip(
                    out[c].astype(float).values, out["seq_length"].astype(int).values
                )
            ]
        else:
            fixed = []
            for v, sl in zip(out[c].values, out["seq_length"].astype(int).values):
                arr = list(v)
                if len(arr) < sl:
                    arr = arr + [float(arr[-1]) if len(arr) else 0.0] * (sl - len(arr))
                elif len(arr) > sl:
                    arr = arr[:sl]
                fixed.append(arr)
            out[c] = fixed

    return out


train = _make_engineered_numeric_features(train)
test = _make_engineered_numeric_features(test)

print("Engineered numeric features ready.")




## === cell 7
def pair_feature(row):
    arr = list(row)
    its = [iter(["_"] + arr[:]), iter(arr[1:] + ["_"])]
    list_touple = list(zip(*its))
    return list(map("".join, list_touple))


def preprocess_categorical_inputs(df, cols=None, Window_features=Window_features):
    if cols is None:
        cols = list(categorical_features)
    else:
        cols = list(cols)

    df = df.copy()
    if Window_features:
        for c in window_columns:
            df["pair_" + c] = df[c].apply(pair_feature)
            cols.append("pair_" + c)

    cols = list(dict.fromkeys(cols))  # preserve order, unique
    arr = df[cols].applymap(lambda seq: [token2int[x] for x in seq]).values.tolist()
    return np.transpose(np.array(arr), (0, 2, 1))


def preprocess_numerical_inputs(df, cols=numerical_features):
    arr = np.array(df[cols].values.tolist())
    return np.transpose(arr, (0, 2, 1)).astype(np.float32)




## === cell 8
token_list = list("().ACGUBshftim")
if Window_features:
    comb = combinations_with_replacement(list("_().ACGUBshftim"), 2)
    token_list += list(set(list(map("".join, comb))))

token_list = list(set(token_list))
token2int = {x: i for i, x in enumerate(token_list)}
print("token_list size:", len(token_list))

train_inputs_all_cat = preprocess_categorical_inputs(train, cols=categorical_features)
train_inputs_all_num = preprocess_numerical_inputs(train, cols=numerical_features)
train_labels_all = np.array(
    train[target_cols].values.tolist(), dtype=np.float32
).transpose((0, 2, 1))

print("Train categorical:", train_inputs_all_cat.shape)
print("Train numerical:", train_inputs_all_num.shape)
print("Train labels:", train_labels_all.shape)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1943506572.py in <cell line: 0>()
      8 print("token_list size:", len(token_list))
      9 
---> 10 train_inputs_all_cat = preprocess_categorical_inputs(train, cols=categorical_features)
     11 train_inputs_all_num = preprocess_numerical_inputs(train, cols=numerical_features)
     12 train_labels_all = np.array(

/tmp/ipykernel_11/3675965678.py in preprocess_categorical_inputs(df, cols, Window_features)
     19 
     20     cols = list(dict.fromkeys(cols))  # preserve order, unique
---> 21     arr = df[cols].applymap(lambda seq: [token2int[x] for x in seq]).values.tolist()
     22     return np.transpose(np.array(arr), (0, 2, 1))
     23 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in applymap(self, func, na_action, **kwargs)
  10520             stacklevel=find_stack_level(),
  10521         )
> 10522         return self.map(func, na_action=na_action, **kwargs)
  10523 
  10524     # ----------------------------------------------------------------------

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in map(self, func, na_action, **kwargs)
  10466             return x._map_values(func, na_action=na_action)
  10467 
> 10468         return self.apply(infer).__finalize__(self, "map")
  10469 
  10470     def applymap(

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in apply(self, func, axis, raw, result_type, args, by_row, engine, engine_kwargs, **kwargs)
  10372             kwargs=kwargs,
  10373         )
> 10374         return op.apply().__finalize__(self, method="apply")
  10375 
  10376     def map(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply(self)
    914             return self.apply_raw(engine=self.engine, engine_kwargs=self.engine_kwargs)
    915 
--> 916         return self.apply_standard()
    917 
    918     def agg(self):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_standard(self)
   1061     def apply_standard(self):
   1062         if self.engine == "python":
-> 1063             results, res_index = self.apply_series_generator()
   1064         else:
   1065             results, res_index = self.apply_series_numba()

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_series_generator(self)
   1079             for i, v in enumerate(series_gen):
   1080                 # ignore SettingWithCopy here in case the user mutates
-> 1081                 results[i] = self.func(v, *self.args, **self.kwargs)
   1082                 if isinstance(results[i], ABCSeries):
   1083                     # If we have a view on v, we need to make a copy because

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in infer(x)
  10464 
  10465         def infer(x):
> 10466             return x._map_values(func, na_action=na_action)
  10467 
  10468         return self.apply(infer).__finalize__(self, "map")

/usr/local/lib/python3.11/dist-packages/pandas/core/base.py in _map_values(self, mapper, na_action, convert)
    919             return arr.map(mapper, na_action=na_action)
    920 
--> 921         return algorithms.map_array(arr, mapper, na_action=na_action, convert=convert)
    922 
    923     @final

/usr/local/lib/python3.11/dist-packages/pandas/core/algorithms.py in map_array(arr, mapper, na_action, convert)
   1741     values = arr.astype(object, copy=False)
   1742     if na_action is None:
-> 1743         return lib.map_infer(values, mapper, convert=convert)
   1744     else:
   1745         return lib.map_infer_mask(

lib.pyx in pandas._libs.lib.map_infer()

/tmp/ipykernel_11/3675965678.py in <lambda>(seq)
     19 
     20     cols = list(dict.fromkeys(cols))  # preserve order, unique
---> 21     arr = df[cols].applymap(lambda seq: [token2int[x] for x in seq]).values.tolist()
     22     return np.transpose(np.array(arr), (0, 2, 1))
     23 

/tmp/ipykernel_11/3675965678.py in <listcomp>(.0)
     19 
     20     cols = list(dict.fromkeys(cols))  # preserve order, unique
---> 21     arr = df[cols].applymap(lambda seq: [token2int[x] for x in seq]).values.tolist()
     22     return np.transpose(np.array(arr), (0, 2, 1))
     23 

KeyError: 'E'

## === cell 9
public_df = test.query("seq_length == 107").reset_index(drop=True)
private_df = test.query("seq_length != 107").reset_index(drop=True)
print("public_df:", public_df.shape)
print("private_df:", private_df.shape)

public_inputs_cat = preprocess_categorical_inputs(public_df)
public_inputs_num = preprocess_numerical_inputs(public_df, cols=numerical_features)

if len(private_df):
    private_inputs_cat = preprocess_categorical_inputs(private_df)
    private_inputs_num = preprocess_numerical_inputs(
        private_df, cols=numerical_features
    )
else:
    private_inputs_cat = None
    private_inputs_num = None

print("Public cat/num:", public_inputs_cat.shape, public_inputs_num.shape)
if len(private_df):
    print("Private cat/num:", private_inputs_cat.shape, private_inputs_num.shape)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2077616952.py in <cell line: 0>()
      6 print("private_df:", private_df.shape)
      7 
----> 8 public_inputs_cat = preprocess_categorical_inputs(public_df)
      9 public_inputs_num = preprocess_numerical_inputs(public_df, cols=numerical_features)
     10 

/tmp/ipykernel_11/3675965678.py in preprocess_categorical_inputs(df, cols, Window_features)
     19 
     20     cols = list(dict.fromkeys(cols))  # preserve order, unique
---> 21     arr = df[cols].applymap(lambda seq: [token2int[x] for x in seq]).values.tolist()
     22     return np.transpose(np.array(arr), (0, 2, 1))
     23 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in applymap(self, func, na_action, **kwargs)
  10520             stacklevel=find_stack_level(),
  10521         )
> 10522         return self.map(func, na_action=na_action, **kwargs)
  10523 
  10524     # ----------------------------------------------------------------------

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in map(self, func, na_action, **kwargs)
  10466             return x._map_values(func, na_action=na_action)
  10467 
> 10468         return self.apply(infer).__finalize__(self, "map")
  10469 
  10470     def applymap(

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in apply(self, func, axis, raw, result_type, args, by_row, engine, engine_kwargs, **kwargs)
  10372             kwargs=kwargs,
  10373         )
> 10374         return op.apply().__finalize__(self, method="apply")
  10375 
  10376     def map(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply(self)
    914             return self.apply_raw(engine=self.engine, engine_kwargs=self.engine_kwargs)
    915 
--> 916         return self.apply_standard()
    917 
    918     def agg(self):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_standard(self)
   1061     def apply_standard(self):
   1062         if self.engine == "python":
-> 1063             results, res_index = self.apply_series_generator()
   1064         else:
   1065             results, res_index = self.apply_series_numba()

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_series_generator(self)
   1079             for i, v in enumerate(series_gen):
   1080                 # ignore SettingWithCopy here in case the user mutates
-> 1081                 results[i] = self.func(v, *self.args, **self.kwargs)
   1082                 if isinstance(results[i], ABCSeries):
   1083                     # If we have a view on v, we need to make a copy because

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in infer(x)
  10464 
  10465         def infer(x):
> 10466             return x._map_values(func, na_action=na_action)
  10467 
  10468         return self.apply(infer).__finalize__(self, "map")

/usr/local/lib/python3.11/dist-packages/pandas/core/base.py in _map_values(self, mapper, na_action, convert)
    919             return arr.map(mapper, na_action=na_action)
    920 
--> 921         return algorithms.map_array(arr, mapper, na_action=na_action, convert=convert)
    922 
    923     @final

/usr/local/lib/python3.11/dist-packages/pandas/core/algorithms.py in map_array(arr, mapper, na_action, convert)
   1741     values = arr.astype(object, copy=False)
   1742     if na_action is None:
-> 1743         return lib.map_infer(values, mapper, convert=convert)
   1744     else:
   1745         return lib.map_infer_mask(

lib.pyx in pandas._libs.lib.map_infer()

/tmp/ipykernel_11/3675965678.py in <lambda>(seq)
     19 
     20     cols = list(dict.fromkeys(cols))  # preserve order, unique
---> 21     arr = df[cols].applymap(lambda seq: [token2int[x] for x in seq]).values.tolist()
     22     return np.transpose(np.array(arr), (0, 2, 1))
     23 

/tmp/ipykernel_11/3675965678.py in <listcomp>(.0)
     19 
     20     cols = list(dict.fromkeys(cols))  # preserve order, unique
---> 21     arr = df[cols].applymap(lambda seq: [token2int[x] for x in seq]).values.tolist()
     22     return np.transpose(np.array(arr), (0, 2, 1))
     23 

KeyError: 'E'

## === cell 10
def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true[:, :, :3] - y_pred[:, :, :3]), axis=1)
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)




## === cell 11
def get_lr_callback(batch_size=8):
    lr_start = 0.00001
    lr_max = 0.004
    lr_min = 0.00005
    lr_ramp_ep = 45
    lr_sus_ep = 2
    lr_decay = 0.8

    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_decay ** (
                epoch - lr_ramp_ep - lr_sus_ep
            ) + lr_min
        return lr

    return tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)


def get_cosine_schedule_with_warmup(
    lr, num_warmup_steps, num_training_steps, num_cycles=3.5
):
    def lrfn(epoch):
        if epoch < num_warmup_steps:
            return (float(epoch) / float(max(1, num_warmup_steps))) * lr
        progress = float(epoch - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)) * lr,
        )

    return tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)




## === cell 12
def lstm_layer(hidden_dim, dropout):
    return L.Bidirectional(
        L.LSTM(
            hidden_dim,
            dropout=dropout,
            return_sequences=True,
            kernel_initializer="orthogonal",
        )
    )


def gru_layer(hidden_dim, dropout):
    return L.Bidirectional(
        L.GRU(
            hidden_dim,
            dropout=dropout,
            return_sequences=True,
            kernel_initializer="orthogonal",
        )
    )




## === cell 13
def build_model(
    embed_size,
    seq_len=107,
    pred_len=68,
    dropout=dropout,
    sp_dropout=sp_dropout,
    num_features=num_features,
    num_hidden_units=num_hidden_units,  # kept (unused) to preserve signature
    embed_dim=embed_dim,
    layers=layers,
    hidden_dim=hidden_dim,
    n_layers=n_layers,
    cat_feature=cat_feature,
):
    inputs = L.Input(shape=(seq_len, cat_feature), name="category_input")
    embed = L.Embedding(input_dim=embed_size, output_dim=embed_dim)(inputs)

    reshaped = L.Reshape((seq_len, cat_feature * embed_dim))(embed)
    reshaped = L.SpatialDropout1D(sp_dropout)(reshaped)
    reshaped_conv = L.Conv1D(
        filters=512, kernel_size=3, strides=1, padding="same", activation="elu"
    )(reshaped)

    numerical_input = L.Input(shape=(seq_len, num_features), name="numeric_input")
    hidden = L.Concatenate()([reshaped_conv, numerical_input])

    hidden_1 = L.Conv1D(
        filters=256, kernel_size=4, strides=1, padding="same", activation="elu"
    )(hidden)
    hidden = gru_layer(128, 0.5)(hidden_1)
    hidden = L.Concatenate()([hidden, hidden_1])

    for x in range(n_layers):
        if layers[x] == "GRU":
            hidden = gru_layer(hidden_dim[x], dropout[x])(hidden)
        else:
            hidden = lstm_layer(hidden_dim[x], dropout[x])(hidden)
        hidden = L.Concatenate()([hidden, hidden_1])

    truncated = hidden[:, :pred_len]
    out = L.Dense(5)(truncated)

    model = tf.keras.Model(inputs=[inputs, numerical_input], outputs=out)

    opt = tf.keras.optimizers.Adam()

    model.compile(optimizer=opt, loss=MCRMSE)
    return model




## === cell 14
model = build_model(embed_size=len(token_list))
print(model.count_params())
if plot_model is not None:
    try:
        plot_model(
            model, to_file="model_plot.png", show_shapes=True, show_layer_names=True
        )
        print("Saved model_plot.png")
    except Exception as e:
        print("plot_model failed (non-fatal):", repr(e))




## === cell 15
def get_stratify_group(row):
    snf = row["SN_filter"]
    snr = row["signal_to_noise"]
    id_ = row["id"]

    if snf == 0:
        if snr < 0:
            snr_c = 0
        elif 0 <= snr < 2:
            snr_c = 1
        elif 2 <= snr < 4:
            snr_c = 2
        elif 4 <= snr < 5.5:
            snr_c = 3
        elif 5.5 <= snr < 10:
            snr_c = 4
        else:
            snr_c = 5
    else:
        if snr < 0:
            snr_c = 6
        elif 0 <= snr < 1:
            snr_c = 7
        elif 1 <= snr < 2:
            snr_c = 8
        elif 2 <= snr < 3:
            snr_c = 9
        elif 3 <= snr < 4:
            snr_c = 10
        elif 4 <= snr < 5:
            snr_c = 11
        elif 5 <= snr < 6:
            snr_c = 12
        elif 6 <= snr < 7:
            snr_c = 13
        elif 7 <= snr < 8:
            snr_c = 14
        elif 8 <= snr < 10:
            snr_c = 15
        else:
            snr_c = 16

    return f"{id_}_{snr_c}"


train["stratify_group"] = train.apply(get_stratify_group, axis=1)
train["stratify_group"] = train["stratify_group"].astype("category").cat.codes

gkf = GroupKFold(n_splits=n_folds)



## === cell 16
submission = pd.DataFrame(index=sample_sub.index, columns=target_cols).fillna(0.0)

val_losses = []
oof_preds_all = []
stacking_pred_all = []

for Fold, (train_index, val_index) in enumerate(
    gkf.split(train_inputs_all_cat, groups=train["stratify_group"])
):
    print(Fore.YELLOW + "#" * 45)
    print(f"###  Fold : {Fold+1}")
    print("#" * 45 + Style.RESET_ALL)

    train_data = train.iloc[train_index].reset_index(drop=True)
    val_data = train.iloc[val_index].reset_index(drop=True)

    val_data = val_data[val_data["cnt"] == 1].reset_index(drop=True)

    model_train = build_model(embed_size=len(token_list))
    model_short = build_model(embed_size=len(token_list), seq_len=107, pred_len=107)

    model_long = None
    if len(private_df):
        model_long = build_model(
            embed_size=len(token_list),
            seq_len=int(private_df["seq_length"].iloc[0]),
            pred_len=int(private_df["seq_length"].iloc[0]),
        )

    train_inputs_cat = preprocess_categorical_inputs(
        train_data, cols=categorical_features
    )
    train_inputs_num = preprocess_numerical_inputs(train_data, cols=numerical_features)
    train_labels = np.array(
        train_data[target_cols].values.tolist(), dtype=np.float32
    ).transpose((0, 2, 1))

    val_inputs_cat = preprocess_categorical_inputs(val_data, cols=categorical_features)
    val_inputs_num = preprocess_numerical_inputs(val_data, cols=numerical_features)
    val_labels = np.array(
        val_data[target_cols].values.tolist(), dtype=np.float32
    ).transpose((0, 2, 1))

    csv_logger = tf.keras.callbacks.CSVLogger(
        f"Fold_{Fold}_log.csv", separator=",", append=False
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f"{model_name}_Fold_{Fold}.weights.h5",
        monitor="val_loss",
        verbose=0,
        mode="min",
        save_best_only=True,
        save_weights_only=True,
    )

    if Cosine_Schedule:
        lr_schedule = get_cosine_schedule_with_warmup(
            lr=0.001, num_warmup_steps=20, num_training_steps=epochs
        )
    elif Rampup_decy_lr:
        lr_schedule = get_lr_callback(BATCH_SIZE)
    else:
        lr_schedule = tf.keras.callbacks.ReduceLROnPlateau()

    history = model_train.fit(
        {"numeric_input": train_inputs_num, "category_input": train_inputs_cat},
        train_labels,
        validation_data=(
            {"numeric_input": val_inputs_num, "category_input": val_inputs_cat},
            val_labels,
        ),
        batch_size=BATCH_SIZE,
        epochs=epochs,
        callbacks=[lr_schedule, checkpoint, csv_logger],
        verbose=1 if debug else 0,
    )

    min_v = float(np.min(history.history["val_loss"]))
    val_losses.append(min_v)
    print(
        "Min Validation Loss:",
        min_v,
        "| at epoch",
        int(np.argmin(history.history["val_loss"]) + 1),
    )

    model_short.load_weights(f"{model_name}_Fold_{Fold}.weights.h5")
    if model_long is not None:
        model_long.load_weights(f"{model_name}_Fold_{Fold}.weights.h5")

    public_preds = model_short.predict(
        {"numeric_input": public_inputs_num, "category_input": public_inputs_cat},
        verbose=0,
    )
    private_preds = None
    if model_long is not None and private_inputs_num is not None:
        private_preds = model_long.predict(
            {"numeric_input": private_inputs_num, "category_input": private_inputs_cat},
            verbose=0,
        )

    oof_preds = model_train.predict(
        {"numeric_input": val_inputs_num, "category_input": val_inputs_cat}, verbose=0
    )
    stacking_pred = model_short.predict(
        {"numeric_input": val_inputs_num, "category_input": val_inputs_cat}, verbose=0
    )

    preds_model = []
    for i, uid in enumerate(public_df.id.values):
        single_pred = public_preds[i]
        single_df = pd.DataFrame(single_pred, columns=target_cols)
        single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
        preds_model.append(single_df)

    if private_preds is not None:
        for i, uid in enumerate(private_df.id.values):
            single_pred = private_preds[i]
            single_df = pd.DataFrame(single_pred, columns=target_cols)
            single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
            preds_model.append(single_df)

    preds_model_df = pd.concat(preds_model, axis=0)
    preds_model_df = preds_model_df.groupby(["id_seqpos"], as_index=True).mean()

    fold_sub = pd.merge(
        sample_sub[["id_seqpos"]],
        preds_model_df.reset_index(),
        on="id_seqpos",
        how="left",
    )
    fold_sub[target_cols] = fold_sub[target_cols].fillna(0.0)
    submission[target_cols] += fold_sub[target_cols].values / n_folds

    for i, uid in enumerate(val_data.id.values):
        single_pred = oof_preds[i]
        single_label = val_labels[i]

        single_label_df = pd.DataFrame(single_label, columns=target_cols)
        single_label_df["id_seqpos"] = [
            f"{uid}_{x}" for x in range(single_label_df.shape[0])
        ]
        single_label_df["id"] = uid
        single_label_df["s_id"] = list(range(single_label_df.shape[0]))

        single_df = pd.DataFrame(single_pred, columns=pred_col_names)
        single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]

        single_df = pd.merge(single_label_df, single_df, on="id_seqpos", how="left")
        oof_preds_all.append(single_df)

    for i, uid in enumerate(val_data.id.values):
        single_pred = stacking_pred[i]
        single_df = pd.DataFrame(single_pred, columns=pred_col_names)
        single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
        single_df["id"] = uid
        stacking_pred_all.append(single_df)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1752300177.py in <cell line: 0>()
      6 
      7 for Fold, (train_index, val_index) in enumerate(
----> 8     gkf.split(train_inputs_all_cat, groups=train["stratify_group"])
      9 ):
     10     print(Fore.YELLOW + "#" * 45)

NameError: name 'train_inputs_all_cat' is not defined

## === cell 17
submission["id_seqpos"] = sample_sub["id_seqpos"].values
submission = submission[["id_seqpos"] + target_cols]
submission[target_cols] = submission[target_cols].fillna(0.0)

OOF = pd.concat(oof_preds_all, axis=0) if len(oof_preds_all) else pd.DataFrame()
stacking_df = (
    pd.concat(stacking_pred_all, axis=0) if len(stacking_pred_all) else pd.DataFrame()
)

print("submission shape:", submission.shape)
print(submission.head())



## === cell 18
if len(OOF):
    OOF = OOF.groupby(["id_seqpos", "id", "s_id"], as_index=False).mean()
    OOF = OOF.sort_values(["id", "s_id"], ascending=[True, True])

    OOF_score = MCRMSE(
        np.expand_dims(OOF[target_eval_col].values, axis=0),
        np.expand_dims(OOF[pred_eval_col].values, axis=0),
    ).numpy()[0]
    print("Overall OOF Score:", float(OOF_score))
    OOF.to_csv("OOf.csv", index=False)

    OOF_filter_1 = pd.merge(train[["SN_filter", "id"]], OOF, on="id")
    OOF_filter_1 = OOF_filter_1[OOF_filter_1["SN_filter"] == 1].sort_values(
        ["id", "s_id"]
    )
    OOF_filter_1_score = MCRMSE(
        np.expand_dims(OOF_filter_1[target_eval_col].values, axis=0),
        np.expand_dims(OOF_filter_1[pred_eval_col].values, axis=0),
    ).numpy()[0]
    print("OOF SN_filter==1 Score:", float(OOF_filter_1_score))
    OOF_filter_1.to_csv("OOF_filter_1.csv", index=False)

    OOF_filter_0 = pd.merge(train[["SN_filter", "id"]], OOF, on="id")
    OOF_filter_0 = OOF_filter_0[OOF_filter_0["SN_filter"] == 0].sort_values(
        ["id", "s_id"]
    )
    OOF_filter_0_score = MCRMSE(
        np.expand_dims(OOF_filter_0[target_eval_col].values, axis=0),
        np.expand_dims(OOF_filter_0[pred_eval_col].values, axis=0),
    ).numpy()[0]
    print("OOF SN_filter==0 Score:", float(OOF_filter_0_score))
    OOF_filter_0.to_csv("OOF_filter_0.csv", index=False)

    if len(stacking_df):
        stacking_df.to_csv("stacking.csv", index=False)



## === cell 19
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
