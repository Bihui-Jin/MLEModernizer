# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.3733857541732854

# 6. Current score

0.63824

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 0.63824) has done: 'The crash is because the notebook expects external pretrained weight files that are not present in your provided `/kaggle/input/...` tree, so inference never runs and no submission is written. I make the weight directory discovery robust (search common Kaggle input locations) and, if no weights are found, fall back to a deterministic zero-prediction submission that still matches the required `sample_submission.csv` row order and columns (so you always get a valid `.csv`). This keeps the core model/inference logic unchanged when weights exist, and only activates the fallback when they don’t. I also fix the relative paths to use `/kaggle/input/stanford-covid-vaccine` explicitly to avoid `../input` issues.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import h5py


def seed_everything(seed: int = 110):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(110)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




## === cell 1
token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}
pred_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

DATA_DIR = "/kaggle/input/stanford-covid-vaccine"
TRAIN_JSON = os.path.join(DATA_DIR, "train.json")
TEST_JSON = os.path.join(DATA_DIR, "test.json")

train = pd.read_json(TRAIN_JSON, lines=True)
test = pd.read_json(TEST_JSON, lines=True)


def _attach_dummy_bpps(df: pd.DataFrame) -> pd.DataFrame:
    if "bpps_sum" not in df.columns:
        df = df.copy()
        df["bpps_sum"] = df["seq_length"].apply(lambda L: np.zeros(L, dtype=np.float32))
        df["bpps_max"] = df["seq_length"].apply(lambda L: np.zeros(L, dtype=np.float32))
        df["bpps_nb"] = df["seq_length"].apply(lambda L: np.zeros(L, dtype=np.float32))
    return df


train = _attach_dummy_bpps(train)
test = _attach_dummy_bpps(test)


def preprocess_inputs(df, cols=("sequence", "structure", "predicted_loop_type")):
    base_fea = np.transpose(
        np.array(
            df[list(cols)]
            .applymap(lambda seq: [token2int[x] for x in seq])
            .values.tolist()
        ),
        (0, 2, 1),
    ).astype(np.int64)

    bpps_sum_fea = np.array(df["bpps_sum"].to_list(), dtype=np.float32)[
        :, :, np.newaxis
    ]
    bpps_max_fea = np.array(df["bpps_max"].to_list(), dtype=np.float32)[
        :, :, np.newaxis
    ]
    bpps_nb_fea = np.array(df["bpps_nb"].to_list(), dtype=np.float32)[:, :, np.newaxis]

    return np.concatenate([base_fea, bpps_sum_fea, bpps_max_fea, bpps_nb_fea], axis=2)


def find_weight_template():
    candidates = [
        "/kaggle/input/gru-lstm-with-feature-engineering-and-augmentation/modelGRU_LSTM1_cv{fold}.h5",
        "/kaggle/input/gru-lstm-with-feature-engineering-and-augmentation/modelGRU_LSTM1_cv{fold}.hdf5",
    ]
    input_root = "/kaggle/input"
    try:
        for ds in os.listdir(input_root):
            dspath = os.path.join(input_root, ds)
            if not os.path.isdir(dspath):
                continue
            for fn in ("modelGRU_LSTM1_cv0.h5", "modelGRU_LSTM1_cv0.hdf5"):
                if os.path.exists(os.path.join(dspath, fn)):
                    ext = ".h5" if fn.endswith(".h5") else ".hdf5"
                    candidates.insert(
                        0, os.path.join(dspath, f"modelGRU_LSTM1_cv{{fold}}{ext}")
                    )
                    break
    except Exception:
        pass

    for tmpl in candidates:
        if os.path.exists(tmpl.format(fold=0)):
            return tmpl
    return None


WEIGHT_TMPL = find_weight_template()
print("Weight template:", WEIGHT_TMPL)




## === cell 2
def _h5_list_datasets(h5obj, prefix=""):
    items = []

    def _rec(g, pfx):
        for k in g.keys():
            obj = g[k]
            if isinstance(obj, h5py.Dataset):
                items.append(pfx + k)
            else:
                _rec(obj, pfx + k + "/")

    _rec(h5obj, prefix)
    return items


def load_weights_for_fold(weight_path: str):
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Missing weight file: {weight_path}")

    with h5py.File(weight_path, "r") as f:
        ds_paths = _h5_list_datasets(f)
        arrays = []
        for p in ds_paths:
            arr = np.array(f[p])
            if arr.size == 0:
                continue
            arrays.append(arr)

    emb = [a for a in arrays if a.ndim == 2 and a.shape == (len(token2int), 100)]
    if len(emb) != 1:
        raise RuntimeError(
            f"Could not uniquely identify embedding weights in {weight_path}. Found {len(emb)} candidates."
        )
    emb_w = emb[0].astype(np.float32)

    k_303 = [a for a in arrays if a.ndim == 2 and a.shape == (303, 768)]
    k_512 = [a for a in arrays if a.ndim == 2 and a.shape == (512, 768)]
    k_256 = [a for a in arrays if a.ndim == 2 and a.shape == (256, 768)]

    b_2_768 = [a for a in arrays if a.ndim == 2 and a.shape == (2, 768)]
    b_768 = [a for a in arrays if a.ndim == 1 and a.shape == (768,)]

    dense_k = [a for a in arrays if a.ndim == 2 and a.shape == (512, 5)]
    dense_b = [a for a in arrays if a.ndim == 1 and a.shape == (5,)]

    if len(k_303) != 2 or len(k_512) != 2 or len(k_256) < 4:
        raise RuntimeError(
            "Could not identify required GRU kernels uniquely. "
            f"(303,768):{len(k_303)} (512,768):{len(k_512)} (256,768):{len(k_256)}"
        )
    if len(dense_k) != 1 or len(dense_b) != 1:
        raise RuntimeError(
            f"Could not identify dense weights uniquely. dense_k:{len(dense_k)} dense_b:{len(dense_b)}"
        )

    if len(b_2_768) >= 4:
        biases = [b.astype(np.float32) for b in b_2_768[:4]]
    elif len(b_768) >= 4:
        biases = [
            np.stack([b.astype(np.float32), b.astype(np.float32)], axis=0)
            for b in b_768[:4]
        ]
    else:
        raise RuntimeError(f"Could not identify enough GRU biases in {weight_path}.")

    rec_kernels = [a.astype(np.float32) for a in k_256[:4]]

    in_kernels_303 = [a.astype(np.float32) for a in k_303[:2]]
    in_kernels_512 = [a.astype(np.float32) for a in k_512[:2]]

    weights = {
        "emb_w": emb_w,
        "gru0_in_w_f": in_kernels_303[0],
        "gru0_in_w_b": in_kernels_303[1],
        "gru1_in_w_f": in_kernels_512[0],
        "gru1_in_w_b": in_kernels_512[1],
        "gru0_rec_w_f": rec_kernels[0],
        "gru0_rec_w_b": rec_kernels[1],
        "gru1_rec_w_f": rec_kernels[2],
        "gru1_rec_w_b": rec_kernels[3],
        "gru0_bias_f": biases[0],
        "gru0_bias_b": biases[1],
        "gru1_bias_f": biases[2],
        "gru1_bias_b": biases[3],
        "dense_w": dense_k[0].astype(np.float32),
        "dense_b": dense_b[0].astype(np.float32),
    }
    return weights




## === cell 3
def Init_params(shape, w=None, b=None):
    if w is None:
        w = torch.nn.Parameter(torch.empty(*shape))
        nn.init.xavier_uniform_(w)
    else:
        w = torch.nn.Parameter(w)
    if b is None:
        b = torch.nn.Parameter(torch.zeros(shape[1]))
    else:
        b = torch.nn.Parameter(b)
    return w, b


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, w_i=None, b_i=None, w_h=None, b_h=None):
        super().__init__()
        self.w_i, self.b_i = Init_params([input_dim, 3 * hidden_dim], w_i, b_i)
        self.w_h, self.b_h = Init_params([hidden_dim, 3 * hidden_dim], w_h, b_h)
        self.hd = hidden_dim

    def forward(self, x):
        hidden = torch.zeros((x.shape[0], self.hd), device=x.device, dtype=x.dtype)
        output = []
        for i in range(x.shape[1]):
            x_z = torch.matmul(x[:, i, :], self.w_i[:, : self.hd]) + self.b_i[: self.hd]
            x_r = (
                torch.matmul(x[:, i, :], self.w_i[:, self.hd : 2 * self.hd])
                + self.b_i[self.hd : 2 * self.hd]
            )
            x_n = (
                torch.matmul(x[:, i, :], self.w_i[:, 2 * self.hd :])
                + self.b_i[2 * self.hd :]
            )

            h_z = torch.matmul(hidden, self.w_h[:, : self.hd]) + self.b_h[: self.hd]
            h_r = (
                torch.matmul(hidden, self.w_h[:, self.hd : 2 * self.hd])
                + self.b_h[self.hd : 2 * self.hd]
            )
            h_n = (
                torch.matmul(hidden, self.w_h[:, 2 * self.hd :])
                + self.b_h[2 * self.hd :]
            )

            z = torch.sigmoid(x_z + h_z)
            r = torch.sigmoid(x_r + h_r)
            n = torch.tanh(x_n + r * h_n)
            hidden = (1 - z) * n + z * hidden
            output.append(hidden.unsqueeze(1))
        return torch.cat(output, 1)


class Net(nn.Module):
    def __init__(self, wdict):
        super().__init__()
        num_target = 5

        emb_w = torch.tensor(wdict["emb_w"], dtype=torch.float32)
        self.cate_emb = nn.Embedding.from_pretrained(emb_w, freeze=False)

        def _split_bias(b2_768):
            b2_768 = torch.tensor(b2_768, dtype=torch.float32)
            return b2_768[0].contiguous(), b2_768[1].contiguous()

        b0f_i, b0f_h = _split_bias(wdict["gru0_bias_f"])
        b0b_i, b0b_h = _split_bias(wdict["gru0_bias_b"])
        b1f_i, b1f_h = _split_bias(wdict["gru1_bias_f"])
        b1b_i, b1b_h = _split_bias(wdict["gru1_bias_b"])

        self.gru = GRU(
            100 * 3 + 3,
            256,
            torch.tensor(wdict["gru0_in_w_f"]).contiguous(),
            b0f_i,
            torch.tensor(wdict["gru0_rec_w_f"]).contiguous(),
            b0f_h,
        )
        self.reverse_gru = GRU(
            100 * 3 + 3,
            256,
            torch.tensor(wdict["gru0_in_w_b"]).contiguous(),
            b0b_i,
            torch.tensor(wdict["gru0_rec_w_b"]).contiguous(),
            b0b_h,
        )
        self.gru1 = GRU(
            512,
            256,
            torch.tensor(wdict["gru1_in_w_f"]).contiguous(),
            b1f_i,
            torch.tensor(wdict["gru1_rec_w_f"]).contiguous(),
            b1f_h,
        )
        self.reverse_gru1 = GRU(
            512,
            256,
            torch.tensor(wdict["gru1_in_w_b"]).contiguous(),
            b1b_i,
            torch.tensor(wdict["gru1_rec_w_b"]).contiguous(),
            b1b_h,
        )

        self.predict = nn.Linear(512, num_target)
        with torch.no_grad():
            self.predict.weight.copy_(
                torch.tensor(wdict["dense_w"].T, dtype=torch.float32)
            )
            self.predict.bias.copy_(torch.tensor(wdict["dense_b"], dtype=torch.float32))

    def forward(self, cateX, contX):
        cate_x = self.cate_emb(cateX).view(cateX.shape[0], cateX.shape[1], -1)
        sequence = torch.cat([cate_x, contX], -1)
        x = self.gru(sequence)
        reverse_x = torch.flip(self.reverse_gru(torch.flip(sequence, [1])), [1])
        sequence = torch.cat([x, reverse_x], -1)
        x = self.gru1(sequence)
        reverse_x = torch.flip(self.reverse_gru1(torch.flip(sequence, [1])), [1])
        x = torch.cat([x, reverse_x], -1)
        x = F.dropout(x, 0.5, training=self.training)
        return self.predict(x)




## === cell 4
def Pred(df: pd.DataFrame):
    test_x = preprocess_inputs(df)
    test_cate_x = torch.LongTensor(test_x[:, :, :3])
    test_cont_x = torch.Tensor(test_x[:, :, 3:])

    test_data = TensorDataset(test_cate_x, test_cont_x)
    test_data_loader = DataLoader(
        dataset=test_data, shuffle=False, batch_size=64, num_workers=0
    )

    all_id = []
    for _, row in df.iterrows():
        for j in range(int(row["seq_length"])):
            all_id.append(f"{row['id']}_{j}")
    all_id = np.array(all_id).reshape(-1, 1)

    all_pred = np.zeros((len(all_id), 5), dtype=np.float32)

    for fold in range(5):
        wdict = load_weights_for_fold(WEIGHT_TMPL.format(fold=fold))
        model = Net(wdict).to(device)
        model.eval()

        t_all_pred = []
        with torch.no_grad():
            for cate_x, cont_x in test_data_loader:
                cate_x = cate_x.to(device)
                cont_x = cont_x.to(device)
                outputs = model(cate_x, cont_x)  # (B, 107, 5)
                t_all_pred.append(outputs.detach().cpu().numpy())
        t_all_pred = np.concatenate(t_all_pred, 0).reshape(-1, 5)
        all_pred += t_all_pred

    all_pred /= 5.0

    sub = pd.DataFrame(all_pred, columns=pred_cols)
    sub.insert(0, "id_seqpos", all_id.reshape(-1))
    return sub


sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)

if WEIGHT_TMPL is None:
    pytorch_sub = sample_sub.copy()
    for c in pred_cols:
        pytorch_sub[c] = 0.0
else:
    pytorch_sub = Pred(test)

    pytorch_sub = pytorch_sub.sort_values("id_seqpos").reset_index(drop=True)
    sample_sub = sample_sub.sort_values("id_seqpos").reset_index(drop=True)

    pytorch_sub = (
        pytorch_sub.set_index("id_seqpos")
        .reindex(sample_sub["id_seqpos"])
        .reset_index()
    )
    for c in pred_cols:
        pytorch_sub[c] = pytorch_sub[c].astype(np.float32).fillna(0.0)
    pytorch_sub = pytorch_sub[["id_seqpos"] + pred_cols]

out_path = "./submission.csv"
pytorch_sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", pytorch_sub.shape)
print(pytorch_sub.head())
