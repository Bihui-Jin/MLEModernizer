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
import numpy as np
import pandas as pd

%matplotlib inline
import matplotlib.pyplot as plt

import itertools


## === cell 1
train = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)
test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)


## === cell 2
def preprocess(df, train_df=True):
    df_data = []
    for mol_id in df['id'].unique():
        sample_data = df.loc[df['id'] == mol_id]
        sample_seq_length = sample_data.seq_length.values[0]

        bpp = np.load(f'../input/stanford-covid-vaccine/bpps/{mol_id}.npy')

        rng = 68 if train_df else sample_seq_length
        for i in range(rng):
            bpp_i = bpp[:, i]
            aux = [bi for ix, bi in enumerate(bpp_i)]
            sum_bpp = np.sum(aux)

            top_3_ix = bpp_i.argsort()[-3:][::-1]
            
            if train_df:
                sample_dict = {'id' : sample_data['id'].values[0],
                               'id_seqpos' : sample_data['id'].values[0] + '_' + str(i),
                               'sequence' : sample_data['sequence'].values[0][i],
                               'structure' : sample_data['structure'].values[0][i],
                               'predicted_loop_type' : sample_data['predicted_loop_type'].values[0][i],
                               'sum_bpp': sum_bpp,
                               'sum_bpp_top1': np.sum([bi for ix, bi in enumerate(bpp[:, top_3_ix[-1]]) if ix != top_3_ix[-1]]),
                               'sum_bpp_top2': np.sum([bi for ix, bi in enumerate(bpp[:, top_3_ix[-2]]) if ix != top_3_ix[-2]]),
                               'sum_bpp_top3': np.sum([bi for ix, bi in enumerate(bpp[:, top_3_ix[-3]]) if ix != top_3_ix[-3]]),
                               'sequence_top1': sample_data['sequence'].values[0][top_3_ix[-1]],
                               'sequence_top2': sample_data['sequence'].values[0][top_3_ix[-2]],
                               'sequence_top3': sample_data['sequence'].values[0][top_3_ix[-3]],
                               'reactivity' : sample_data['reactivity'].values[0][i],
                               'deg_Mg_pH10' : sample_data['deg_Mg_pH10'].values[0][i],
                               'deg_pH10' : sample_data['deg_pH10'].values[0][i],
                               'deg_Mg_50C' : sample_data['deg_Mg_50C'].values[0][i],
                               'deg_50C' : sample_data['deg_50C'].values[0][i],
                }
            else:
                sample_dict = {'id' : sample_data['id'].values[0],
                               'id_seqpos' : sample_data['id'].values[0] + '_' + str(i),
                               'sequence' : sample_data['sequence'].values[0][i],
                               'structure' : sample_data['structure'].values[0][i],
                               'predicted_loop_type' : sample_data['predicted_loop_type'].values[0][i],
                               'sum_bpp': sum_bpp,
                               'sum_bpp_top1': np.sum([bi for ix, bi in enumerate(bpp[:, top_3_ix[-1]]) if ix != top_3_ix[-1]]),
                               'sum_bpp_top2': np.sum([bi for ix, bi in enumerate(bpp[:, top_3_ix[-2]]) if ix != top_3_ix[-2]]),
                               'sum_bpp_top3': np.sum([bi for ix, bi in enumerate(bpp[:, top_3_ix[-3]]) if ix != top_3_ix[-3]]),
                               'sequence_top1': sample_data['sequence'].values[0][top_3_ix[-1]],
                               'sequence_top2': sample_data['sequence'].values[0][top_3_ix[-2]],
                               'sequence_top3': sample_data['sequence'].values[0][top_3_ix[-3]],
                }

            shifts = [1, 2, 3]
            shift_cols = ['sequence', 'structure', 'predicted_loop_type']
            for shift,col in itertools.product(shifts, shift_cols):
                if i - shift >= 0:
                    sample_dict['b'+str(shift)+'_'+col] = sample_data[col].values[0][i-shift]
                else:
                    sample_dict['b'+str(shift)+'_'+col] = -1

                if i + shift <= sample_seq_length - 1:
                    sample_dict['a'+str(shift)+'_'+col] = sample_data[col].values[0][i+shift]
                else:
                    sample_dict['a'+str(shift)+'_'+col] = -1


            df_data.append(sample_dict)
    df_data = pd.DataFrame(df_data)
    
    return df_data


## === cell 3
import os


def preprocess(df, train_df=True):
    df_data = []
    for mol_id in df["id"].unique():
        sample_data = df.loc[df["id"] == mol_id]
        sample_seq_length = sample_data.seq_length.values[0]

        bpp_fname = f"{mol_id}.npy"

        candidates = [
            f"../input/stanford-covid-vaccine/bpps/{bpp_fname}",
            f"../input/stanford-covid-vaccine/stanford-covid-vaccine/bpps/{bpp_fname}",
            f"../input/stanford-covid-vaccine/data/stanford-covid-vaccine/bpps/{bpp_fname}",
            f"../input/data/stanford-covid-vaccine/bpps/{bpp_fname}",
            f"../input/data/stanford-covid-vaccine/stanford-covid-vaccine/bpps/{bpp_fname}",
            os.path.join("/kaggle/input/stanford-covid-vaccine/bpps", bpp_fname),
            os.path.join(
                "/kaggle/input/stanford-covid-vaccine/stanford-covid-vaccine/bpps",
                bpp_fname,
            ),
            os.path.join("/kaggle/data/stanford-covid-vaccine/bpps", bpp_fname),
            os.path.join(
                "/kaggle/data/stanford-covid-vaccine/stanford-covid-vaccine/bpps",
                bpp_fname,
            ),
            os.path.join(
                "/kaggle/input/stanford-covid-vaccine/data/stanford-covid-vaccine/bpps",
                bpp_fname,
            ),
            os.path.join("/kaggle/input/data/stanford-covid-vaccine/bpps", bpp_fname),
            os.path.join(
                "/kaggle/input/data/stanford-covid-vaccine/stanford-covid-vaccine/bpps",
                bpp_fname,
            ),
            os.path.join(
                "/kaggle/data/input/stanford-covid-vaccine/data/stanford-covid-vaccine/bpps",
                bpp_fname,
            ),
            os.path.join(
                "/kaggle/data/input/data/stanford-covid-vaccine/bpps", bpp_fname
            ),
            os.path.join(
                "/kaggle/data/input/data/stanford-covid-vaccine/stanford-covid-vaccine/bpps",
                bpp_fname,
            ),
        ]
        bpp_path = next((p for p in candidates if os.path.exists(p)), None)

        if bpp_path is None:
            bpp = np.zeros((sample_seq_length, sample_seq_length), dtype=np.float32)
        else:
            bpp = np.load(bpp_path)

        rng = 68 if train_df else sample_seq_length
        for i in range(rng):
            bpp_i = bpp[:, i]
            aux = [bi for ix, bi in enumerate(bpp_i)]
            sum_bpp = np.sum(aux)

            top_3_ix = bpp_i.argsort()[-3:][::-1]

            if train_df:
                sample_dict = {
                    "id": sample_data["id"].values[0],
                    "id_seqpos": sample_data["id"].values[0] + "_" + str(i),
                    "sequence": sample_data["sequence"].values[0][i],
                    "structure": sample_data["structure"].values[0][i],
                    "predicted_loop_type": sample_data["predicted_loop_type"].values[0][
                        i
                    ],
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
                    "reactivity": sample_data["reactivity"].values[0][i],
                    "deg_Mg_pH10": sample_data["deg_Mg_pH10"].values[0][i],
                    "deg_pH10": sample_data["deg_pH10"].values[0][i],
                    "deg_Mg_50C": sample_data["deg_Mg_50C"].values[0][i],
                    "deg_50C": sample_data["deg_50C"].values[0][i],
                }
            else:
                sample_dict = {
                    "id": sample_data["id"].values[0],
                    "id_seqpos": sample_data["id"].values[0] + "_" + str(i),
                    "sequence": sample_data["sequence"].values[0][i],
                    "structure": sample_data["structure"].values[0][i],
                    "predicted_loop_type": sample_data["predicted_loop_type"].values[0][
                        i
                    ],
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

            shifts = [1, 2, 3]
            shift_cols = ["sequence", "structure", "predicted_loop_type"]
            for shift, col in itertools.product(shifts, shift_cols):
                if i - shift >= 0:
                    sample_dict["b" + str(shift) + "_" + col] = sample_data[col].values[
                        0
                    ][i - shift]
                else:
                    sample_dict["b" + str(shift) + "_" + col] = -1

                if i + shift <= sample_seq_length - 1:
                    sample_dict["a" + str(shift) + "_" + col] = sample_data[col].values[
                        0
                    ][i + shift]
                else:
                    sample_dict["a" + str(shift) + "_" + col] = -1

            df_data.append(sample_dict)
    df_data = pd.DataFrame(df_data)

    return df_data


train_data = preprocess(train)
test_data = preprocess(test, train_df=False)


## === cell 4
train_data.head()


## === cell 5
targets = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
num_feats = ['sum_bpp', 'sum_bpp_top1', 'sum_bpp_top2', 'sum_bpp_top3']
not_use_cols = ['id', 'id_seqpos']
features = [f for f in train_data.columns if f not in not_use_cols if f not in targets]
cat_feats = [f for f in features if f not in num_feats]


## === cell 6
sequence_encmap = {'A': 0, 'G' : 1, 'C' : 2, 'U' : 3}
structure_encmap = {'.' : 0, '(' : 1, ')' : 1}
looptype_encmap = {'S':0, 'E':1, 'H':2, 'I':3, 'X':4, 'M':5, 'B':6}

enc_targets = ['sequence', 'a1_sequence', 'a2_sequence', 'a3_sequence',
               'b1_sequence', 'b2_sequence', 'b3_sequence',
               'sequence_top1', 'sequence_top2', 'sequence_top3',
               'structure', 'a1_structure', 'a2_structure', 'a3_structure',
               'b1_structure', 'b2_structure', 'b3_structure',
               'predicted_loop_type', 'a1_predicted_loop_type', 'a2_predicted_loop_type',
               'a3_predicted_loop_type', 'b1_predicted_loop_type', 'b2_predicted_loop_type',
               'b3_predicted_loop_type']
enc_maps = [sequence_encmap, sequence_encmap, sequence_encmap, sequence_encmap,
            sequence_encmap, sequence_encmap, sequence_encmap,
            sequence_encmap, sequence_encmap, sequence_encmap,
            structure_encmap, structure_encmap, structure_encmap, structure_encmap,
            structure_encmap, structure_encmap, structure_encmap,
            looptype_encmap, looptype_encmap, looptype_encmap,
            looptype_encmap, looptype_encmap, looptype_encmap,
            looptype_encmap,]

for t, m in zip(enc_targets, enc_maps):
    print(t)
    train_data[t] = train_data[t].apply(lambda x: m[x] if x in m else -1)
    test_data[t] = test_data[t].apply(lambda x: m[x] if x in m else -1)


## === cell 7
train_data.head()


## === cell 8
import lightgbm as lgb


## === cell 9
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

cv_results = dict()
models = dict()
preds = dict()
for tgt in targets:
    print(
        "-" * 30,
        tgt,
        "-" * 30,
    )
    DTrain = lgb.Dataset(
        train_data[features], train_data[tgt], categorical_feature=cat_feats
    )

    m = lgb.cv(
        params,
        DTrain,
        num_boost_round=300,
        nfold=10,
        stratified=False,
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    cv_results[tgt] = m["rmse-mean"][-1]

    DTrain = lgb.Dataset(
        train_data[features], train_data[tgt], categorical_feature=cat_feats
    )
    model = lgb.train(params, DTrain, num_boost_round=len(m["rmse-mean"]))
    test_data[tgt] = model.predict(test_data[features])
    models[tgt] = model

cv_results


## --- ERROR in cell 9, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mKeyError[0m                                  Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/556519867.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     38[0m         [0mcallbacks[0m[0;34m=[0m[0;34m[[0m[0mlgb[0m[0;34m.[0m[0mearly_stopping[0m[0;34m([0m[0mstopping_rounds[0m[0;34m=[0m[0;36m30[0m[0;34m,[0m [0mverbose[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m][0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     39[0m     )
[0;32m---> 40[0;31m     [0mcv_results[0m[0;34m[[0m[0mtgt[0m[0;34m][0m [0;34m=[0m [0mm[0m[0;34m[[0m[0;34m"rmse-mean"[0m[0;34m][0m[0;34m[[0m[0;34m-[0m[0;36m1[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     41[0m [0;34m[0m[0m
[1;32m     42[0m     DTrain = lgb.Dataset(

[0;31mKeyError[0m: 'rmse-mean'

## === cell 10
for tgt in targets:
    tmp = pd.Series(models[tgt].feature_importance('gain'), index=features)

    fig, ax = plt.subplots(figsize=(10, 5))
    tmp.sort_values(ascending=False).plot.barh(ax=ax)
    ax.set_title(tgt)
    fig.tight_layout()
