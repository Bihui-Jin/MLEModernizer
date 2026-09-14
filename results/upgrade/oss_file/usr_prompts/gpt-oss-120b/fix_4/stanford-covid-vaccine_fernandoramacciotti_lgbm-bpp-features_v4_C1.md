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

# 5. Target score

0.42625

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import itertools
import lightgbm as lgb



## === cell 1
train = pd.read_json("../input/stanford-covid-vaccine/train.json", lines=True)
test = pd.read_json("../input/stanford-covid-vaccine/test.json", lines=True)




## === cell 2
def _load_bpp(mol_id, seq_len):
    """Load base‑pair probability matrix; if missing, return zeros."""
    path = f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy"
    try:
        bpp = np.load(path)
        if bpp.shape[0] < seq_len or bpp.shape[1] < seq_len:
            bpp = np.zeros((seq_len, seq_len))
    except FileNotFoundError:
        bpp = np.zeros((seq_len, seq_len))
    return bpp


def preprocess(df, train_df=True):
    df_data = []
    for mol_id in df["id"].unique():
        sample_data = df.loc[df["id"] == mol_id]
        sample_seq_length = int(sample_data.seq_length.values[0])

        bpp = _load_bpp(mol_id, sample_seq_length)

        rng = 68 if train_df else sample_seq_length
        for i in range(rng):
            bpp_i = bpp[:, i]
            sum_bpp = np.sum(bpp_i)

            top_3_ix = bpp_i.argsort()[-3:][::-1]

            base_dict = {
                "id": sample_data["id"].values[0],
                "id_seqpos": f"{sample_data['id'].values[0]}_{i}",
                "sequence": sample_data["sequence"].values[0][i],
                "structure": sample_data["structure"].values[0][i],
                "predicted_loop_type": sample_data["predicted_loop_type"].values[0][i],
                "sum_bpp": sum_bpp,
                "sum_bpp_top1": np.sum(
                    [
                        bi
                        for ix, bi in enumerate(bpp[:, top_3_ix[-1]])
                        if ix != top_3_ix[-1]
                    ]
                ),
                "sum_bpp_top2": np.sum(
                    [
                        bi
                        for ix, bi in enumerate(bpp[:, top_3_ix[-2]])
                        if ix != top_3_ix[-2]
                    ]
                ),
                "sum_bpp_top3": np.sum(
                    [
                        bi
                        for ix, bi in enumerate(bpp[:, top_3_ix[-3]])
                        if ix != top_3_ix[-3]
                    ]
                ),
                "sequence_top1": sample_data["sequence"].values[0][top_3_ix[-1]],
                "sequence_top2": sample_data["sequence"].values[0][top_3_ix[-2]],
                "sequence_top3": sample_data["sequence"].values[0][top_3_ix[-3]],
            }

            if train_df:
                base_dict.update(
                    {
                        "reactivity": sample_data["reactivity"].values[0][i],
                        "deg_Mg_pH10": sample_data["deg_Mg_pH10"].values[0][i],
                        "deg_pH10": sample_data["deg_pH10"].values[0][i],
                        "deg_Mg_50C": sample_data["deg_Mg_50C"].values[0][i],
                        "deg_50C": sample_data["deg_50C"].values[0][i],
                    }
                )

            shifts = [1, 2, 3]
            shift_cols = ["sequence", "structure", "predicted_loop_type"]
            for shift, col in itertools.product(shifts, shift_cols):
                if i - shift >= 0:
                    base_dict[f"b{shift}_{col}"] = sample_data[col].values[0][i - shift]
                else:
                    base_dict[f"b{shift}_{col}"] = -1

                if i + shift <= sample_seq_length - 1:
                    base_dict[f"a{shift}_{col}"] = sample_data[col].values[0][i + shift]
                else:
                    base_dict[f"a{shift}_{col}"] = -1

            df_data.append(base_dict)

    return pd.DataFrame(df_data)




## === cell 3
train_data = preprocess(train, train_df=True)
test_data = preprocess(test, train_df=False)



## === cell 4
targets = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
num_feats = ["sum_bpp", "sum_bpp_top1", "sum_bpp_top2", "sum_bpp_top3"]
not_use_cols = ["id", "id_seqpos"]
features = [f for f in train_data.columns if f not in not_use_cols + targets]
cat_feats = [f for f in features if f not in num_feats]



## === cell 5
sequence_encmap = {"A": 0, "G": 1, "C": 2, "U": 3}
structure_encmap = {".": 0, "(": 1, ")": 1}
looptype_encmap = {"S": 0, "E": 1, "H": 2, "I": 3, "X": 4, "M": 5, "B": 6}

enc_targets = [
    "sequence",
    "a1_sequence",
    "a2_sequence",
    "a3_sequence",
    "b1_sequence",
    "b2_sequence",
    "b3_sequence",
    "sequence_top1",
    "sequence_top2",
    "sequence_top3",
    "structure",
    "a1_structure",
    "a2_structure",
    "a3_structure",
    "b1_structure",
    "b2_structure",
    "b3_structure",
    "predicted_loop_type",
    "a1_predicted_loop_type",
    "a2_predicted_loop_type",
    "a3_predicted_loop_type",
    "b1_predicted_loop_type",
    "b2_predicted_loop_type",
    "b3_predicted_loop_type",
]

enc_maps = [
    sequence_encmap,
    sequence_encmap,
    sequence_encmap,
    sequence_encmap,
    sequence_encmap,
    sequence_encmap,
    sequence_encmap,
    sequence_encmap,
    sequence_encmap,
    sequence_encmap,
    structure_encmap,
    structure_encmap,
    structure_encmap,
    structure_encmap,
    structure_encmap,
    structure_encmap,
    structure_encmap,
    looptype_encmap,
    looptype_encmap,
    looptype_encmap,
    looptype_encmap,
    looptype_encmap,
    looptype_encmap,
    looptype_encmap,
]

for t, m in zip(enc_targets, enc_maps):
    train_data[t] = train_data[t].apply(lambda x: m[x] if x in m else -1)
    test_data[t] = test_data[t].apply(lambda x: m[x] if x in m else -1)



## === cell 6
seed = 2020
params = {
    "objective": "regression",
    "boosting": "gbdt",
    "metric": "rmse",
    "num_leaves": 32,
    "max_bin": 512,
    "reg_lambda": 0.5,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "learning_rate": 0.08,
    "min_data_in_leaf": 200,
    "seed": seed,
    "n_jobs": -1,
}

cv_results = {}
models = {}
for tgt in targets:
    print("-" * 30, tgt, "-" * 30)
    dtrain = lgb.Dataset(
        train_data[features], label=train_data[tgt], categorical_feature=cat_feats
    )
    cv = lgb.cv(
        params,
        dtrain,
        num_boost_round=300,
        nfold=10,
        stratified=False,
    )
    best_iter = int(np.argmin(cv["rmse-mean"])) + 1
    cv_results[tgt] = cv["rmse-mean"][best_iter - 1]

    model = lgb.train(params, dtrain, num_boost_round=best_iter)
    test_data[tgt] = model.predict(test_data[features])
    models[tgt] = model

print("CV results:", cv_results)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1284734423.py in <cell line: 0>()
     30     )
     31     # best iteration = index of minimal mean rmse + 1 (since boost rounds start at 1)
---> 32     best_iter = int(np.argmin(cv["rmse-mean"])) + 1
     33     cv_results[tgt] = cv["rmse-mean"][best_iter - 1]
     34 

KeyError: 'rmse-mean'

## === cell 7
submission_path = "../input/stanford-covid-vaccine/sample_submission.csv"
submission = pd.read_csv(submission_path, usecols=["id_seqpos"])
submission = submission.merge(
    test_data[["id_seqpos"] + targets], on="id_seqpos", how="left"
)
submission = submission[["id_seqpos"] + targets]
submission.head()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/437337666.py in <cell line: 0>()
      2 submission = pd.read_csv(submission_path, usecols=["id_seqpos"])
      3 submission = submission.merge(
----> 4     test_data[["id_seqpos"] + targets], on="id_seqpos", how="left"
      5 )
      6 submission = submission[["id_seqpos"] + targets]

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C'] not in index"

## === cell 8
submission.to_csv("submission.csv", index=False)

## --- ERROR in outputing the csv:
Invalid submission: Expected the submission to have columns ['id_seqpos', 'reactivity', 'deg_Mg_pH10', 'deg_Mg_50C', 'deg_pH10', 'deg_50C'], but instead it has columns Index(['id_seqpos'], dtype='object').
