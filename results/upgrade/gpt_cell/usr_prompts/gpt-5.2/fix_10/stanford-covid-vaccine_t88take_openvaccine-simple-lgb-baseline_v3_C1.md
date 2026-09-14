# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import gc
import os
import random
import itertools

import lightgbm as lgb
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

try:
    from IPython.display import display  # type: ignore
except Exception:

    def display(x):
        print(x)


sns.set(style="darkgrid")
SEEDS = 42

os.environ["PYTHONHASHSEED"] = str(SEEDS)
random.seed(SEEDS)
np.random.seed(SEEDS)




## === cell 1
def rmse(y_true, y_pred):
    return (mean_squared_error(y_true, y_pred)) ** 0.5




## === cell 2
class TreeModel:
    def __init__(self, model_type):
        self.model_type = model_type
        self.tr_data = None
        self.vl_data = None
        self.model = None

    def train(
        self,
        params,
        train_x,
        train_y,
        valid_x=None,
        valid_y=None,
        num_round=None,
        early_stopping=None,
        verbose=None,
    ):
        if self.model_type == "lgb":
            self.tr_data = lgb.Dataset(train_x, label=train_y)
            self.vl_data = lgb.Dataset(valid_x, label=valid_y)
            self.model = lgb.train(
                params,
                self.tr_data,
                valid_sets=[self.tr_data, self.vl_data],
                num_boost_round=num_round,
                early_stopping_rounds=early_stopping,
                verbose_eval=verbose,
            )

    def predict(self, X):
        if self.model_type == "lgb":
            return self.model.predict(X, num_iteration=self.model.best_iteration)

    @property
    def feature_names_(self):
        if self.model_type == "lgb":
            return self.model.feature_name()

    @property
    def feature_importances_(self):
        if self.model_type == "lgb":
            return self.model.feature_importance(importance_type="gain")




## === cell 3
train = pd.read_json("../input/stanford-covid-vaccine/train.json", lines=True)
test = pd.read_json("../input/stanford-covid-vaccine/test.json", lines=True)
submission = pd.read_csv("/kaggle/input/stanford-covid-vaccine/sample_submission.csv")




## === cell 4
def build_positional_df(df, is_train: bool):
    ids = df["id"].astype(str).to_numpy()
    seqs = df["sequence"].astype(str).to_numpy()
    structs = df["structure"].astype(str).to_numpy()
    loops = df["predicted_loop_type"].astype(str).to_numpy()
    seq_lengths = df["seq_length"].to_numpy(dtype=np.int32)

    n_pos_list = np.full(len(df), 68, dtype=np.int32) if is_train else seq_lengths
    total_rows = int(n_pos_list.sum())

    id_col = np.empty(total_rows, dtype=object)
    id_seqpos_col = np.empty(total_rows, dtype=object)

    seq_base = np.empty(total_rows, dtype=object)
    st_base = np.empty(total_rows, dtype=object)
    lp_base = np.empty(total_rows, dtype=object)

    shifts = (1, 2, 3, 4, 5)
    b_seq = {sh: np.empty(total_rows, dtype=object) for sh in shifts}
    b_st = {sh: np.empty(total_rows, dtype=object) for sh in shifts}
    b_lp = {sh: np.empty(total_rows, dtype=object) for sh in shifts}
    a_seq = {sh: np.empty(total_rows, dtype=object) for sh in shifts}
    a_st = {sh: np.empty(total_rows, dtype=object) for sh in shifts}
    a_lp = {sh: np.empty(total_rows, dtype=object) for sh in shifts}

    if is_train:
        tgt_cols = [
            "reactivity",
            "reactivity_error",
            "deg_Mg_pH10",
            "deg_error_Mg_pH10",
            "deg_pH10",
            "deg_error_pH10",
            "deg_Mg_50C",
            "deg_error_Mg_50C",
            "deg_50C",
            "deg_error_50C",
        ]
        tgt_mat = {}
        for c in tgt_cols:
            tgt_mat[c] = np.stack(df[c].to_numpy(), axis=0).astype(
                np.float32, copy=False
            )
        tgt_out = {c: np.empty(total_rows, dtype=object) for c in tgt_cols}

    out_i = 0
    for row in range(len(df)):
        npos = int(n_pos_list[row])
        mol_id = ids[row]
        seq = seqs[row]
        st = structs[row]
        lp = loops[row]

        sl = slice(out_i, out_i + npos)

        id_col[sl] = mol_id

        id_seqpos_col[sl] = [f"{mol_id}_{p}" for p in range(npos)]

        seq_b = np.frombuffer(seq.encode("ascii"), dtype="S1")
        st_b = np.frombuffer(st.encode("ascii"), dtype="S1")
        lp_b = np.frombuffer(lp.encode("ascii"), dtype="S1")

        seq_base[sl] = seq_b[:npos].astype(str)
        st_base[sl] = st_b[:npos].astype(str)
        lp_base[sl] = lp_b[:npos].astype(str)

        pos = np.arange(npos, dtype=np.int32)
        Lseq, Lst, Llp = len(seq_b), len(st_b), len(lp_b)

        for sh in shifts:
            bpos = pos - sh
            apos = pos + sh

            bm = bpos >= 0
            am_seq = apos < Lseq
            am_st = apos < Lst
            am_lp = apos < Llp

            tmp = np.empty(npos, dtype=object)
            tmp[~bm] = -1
            if bm.any():
                tmp[bm] = seq_b[bpos[bm]].astype(str)
            b_seq[sh][sl] = tmp

            tmp = np.empty(npos, dtype=object)
            tmp[~bm] = -1
            if bm.any():
                tmp[bm] = st_b[bpos[bm]].astype(str)
            b_st[sh][sl] = tmp

            tmp = np.empty(npos, dtype=object)
            tmp[~bm] = -1
            if bm.any():
                tmp[bm] = lp_b[bpos[bm]].astype(str)
            b_lp[sh][sl] = tmp

            tmp = np.empty(npos, dtype=object)
            tmp[~am_seq] = -1
            if am_seq.any():
                tmp[am_seq] = seq_b[apos[am_seq]].astype(str)
            a_seq[sh][sl] = tmp

            tmp = np.empty(npos, dtype=object)
            tmp[~am_st] = -1
            if am_st.any():
                tmp[am_st] = st_b[apos[am_st]].astype(str)
            a_st[sh][sl] = tmp

            tmp = np.empty(npos, dtype=object)
            tmp[~am_lp] = -1
            if am_lp.any():
                tmp[am_lp] = lp_b[apos[am_lp]].astype(str)
            a_lp[sh][sl] = tmp

        if is_train:
            for c in tgt_cols:
                arr = tgt_mat[c][row]  # shape (68,)
                tgt_out[c][sl] = [arr] * npos

        out_i += npos

    data = {
        "id": id_col,
        "id_seqpos": id_seqpos_col,
        "sequence": seq_base,
        "structure": st_base,
        "predicted_loop_type": lp_base,
    }
    for sh in shifts:
        data[f"b{sh}_sequence"] = b_seq[sh]
        data[f"b{sh}_structure"] = b_st[sh]
        data[f"b{sh}_predicted_loop_type"] = b_lp[sh]
        data[f"a{sh}_sequence"] = a_seq[sh]
        data[f"a{sh}_structure"] = a_st[sh]
        data[f"a{sh}_predicted_loop_type"] = a_lp[sh]

    if is_train:
        for c in tgt_cols:
            data[c] = tgt_out[c]

    out = pd.DataFrame(data)

    return out


train_data = build_positional_df(train, is_train=True)
test_data = build_positional_df(test, is_train=False)
train_data.head()



## === cell 5
sequence_encmap = {"A": 0, "G": 1, "C": 2, "U": 3}
structure_encmap = {".": 0, "(": 1, ")": 2}
looptype_encmap = {"S": 0, "E": 1, "H": 2, "I": 3, "X": 4, "M": 5, "B": 6}

enc_targets = ["sequence", "structure", "predicted_loop_type"]
enc_maps = [sequence_encmap, structure_encmap, looptype_encmap]

for t, m in zip(enc_targets, enc_maps):
    cols = [c for c in train_data.columns if t in c]
    for c in cols:
        train_data[c] = train_data[c].map(m).fillna(train_data[c]).astype(np.int16)
        test_data[c] = test_data[c].map(m).fillna(test_data[c]).astype(np.int16)



## === cell 6
not_use_cols = ["id", "id_seqpos"]
features = [c for c in test_data.columns if c not in not_use_cols]
targets = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]



## === cell 7
FOLD_N = 5
gkf = GroupKFold(n_splits=FOLD_N)



## === cell 8
params = {
    "objective": "regression",
    "boosting": "gbdt",
    "metric": "rmse",
    "learning_rate": 0.1,
    "seed": SEEDS,
    "num_threads": max(1, os.cpu_count() or 1),
    "verbosity": -1,
}



## === cell 9
early_stopping = None
verbose = None

feature_importances = pd.DataFrame()
result = {}
oof_df = pd.DataFrame(train_data.id_seqpos)

X_all = np.ascontiguousarray(
    train_data[features].to_numpy(dtype=np.float32, copy=False)
)
X_test = np.ascontiguousarray(
    test_data[features].to_numpy(dtype=np.float32, copy=False)
)
groups = train_data["id"].to_numpy()

cv_splits = list(gkf.split(X_all, train_data["reactivity"], groups))

sub_index = pd.Index(submission["id_seqpos"])
test_index = pd.Index(test_data["id_seqpos"])
sub_pos = sub_index.get_indexer(test_index)
if (sub_pos < 0).any():
    raise RuntimeError(
        "submission id_seqpos and test id_seqpos do not align as expected."
    )

pos_idx = (
    train_data["id_seqpos"]
    .astype(str)
    .str.rsplit("_", n=1)
    .str[-1]
    .astype(np.int16)
    .to_numpy()
)

NUM_BOOST_ROUND = 2000

for target in targets:
    preds = np.zeros(len(test_data), dtype=np.float32)
    scores = 0.0

    y_series = train_data[target]
    y_all = np.fromiter(
        (np.asarray(arr, dtype=np.float32)[i] for arr, i in zip(y_series, pos_idx)),
        dtype=np.float32,
        count=len(train_data),
    )

    oof_parts = []
    fi_parts = []

    for n, (tr_idx, vl_idx) in enumerate(cv_splits):
        tr_x, tr_y = X_all[tr_idx], y_all[tr_idx]
        vl_x, vl_y = X_all[vl_idx], y_all[vl_idx]
        vl_id = train_data["id_seqpos"].iloc[vl_idx].to_numpy()

        model = TreeModel(model_type="lgb")

        callbacks = []
        if early_stopping is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=100))
        if verbose is not None:
            callbacks.append(lgb.log_evaluation(period=1000))

        model.tr_data = lgb.Dataset(
            tr_x, label=tr_y, free_raw_data=True, feature_name=features
        )
        model.vl_data = lgb.Dataset(
            vl_x,
            label=vl_y,
            reference=model.tr_data,
            free_raw_data=True,
            feature_name=features,
        )
        model.model = lgb.train(
            params,
            model.tr_data,
            valid_sets=[model.tr_data, model.vl_data],
            num_boost_round=NUM_BOOST_ROUND,
            callbacks=callbacks,
        )

        fi_parts.append(
            pd.DataFrame(
                {
                    "feature": model.feature_names_,
                    "importance": model.feature_importances_,
                    "fold": n,
                    "target": target,
                }
            )
        )

        vl_pred = model.predict(vl_x).astype(np.float32, copy=False)
        score = rmse(vl_y, vl_pred)
        scores += score / FOLD_N
        print(f"score : {score}")

        oof_parts.append(pd.DataFrame({"id_seqpos": vl_id, target: vl_pred}))

        preds += model.predict(X_test).astype(np.float32, copy=False) / FOLD_N

        del model
        gc.collect()

    feature_importances = pd.concat(
        [feature_importances, pd.concat(fi_parts, ignore_index=True)], ignore_index=True
    )

    oof = pd.concat(oof_parts, ignore_index=True)
    oof_df = oof_df.merge(oof, on="id_seqpos", how="inner")

    submission.loc[sub_pos, target] = preds

    print(f"{target}_rmse : {scores}")
    result[target] = scores


## === cell 10
display(result)
display(f"total : {np.mean(list(result.values()))}")



## === cell 11
pass



## === cell 12
oof_df.head()



## === cell 13
submission.head()



## === cell 14
display(oof_df.shape)
display(submission.shape)



## === cell 15
for c in ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]:
    if submission[c].isna().any():
        submission[c] = submission[c].fillna(0.0)

oof_df.to_csv("oof_df.csv", index=False)
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
