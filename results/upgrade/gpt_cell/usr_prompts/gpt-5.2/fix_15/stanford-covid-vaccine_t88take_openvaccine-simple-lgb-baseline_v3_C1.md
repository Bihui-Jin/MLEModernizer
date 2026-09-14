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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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

os.environ.setdefault("OMP_NUM_THREADS", str(max(1, os.cpu_count() or 1)))
os.environ.setdefault("MKL_NUM_THREADS", os.environ["OMP_NUM_THREADS"])
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ["OMP_NUM_THREADS"])
os.environ.setdefault("NUMEXPR_NUM_THREADS", os.environ["OMP_NUM_THREADS"])




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
    seq_lengths = df["seq_length"].to_numpy(dtype=np.int32, copy=False)

    n_pos_list = np.full(len(df), 68, dtype=np.int32) if is_train else seq_lengths
    total_rows = int(n_pos_list.sum())

    row_starts = np.empty(len(df), dtype=np.int64)
    np.cumsum(n_pos_list, out=row_starts)
    row_starts -= n_pos_list

    mol_idx_per_pos = np.repeat(np.arange(len(df), dtype=np.int32), n_pos_list)
    pos_in_mol = np.concatenate(
        [np.arange(n, dtype=np.int16) for n in n_pos_list], axis=0
    )

    id_col = ids[mol_idx_per_pos].astype(object, copy=False)
    id_seqpos_col = (
        pd.Series(ids[mol_idx_per_pos]) + "_" + pd.Series(pos_in_mol.astype(str))
    ).to_numpy(dtype=object)

    seq_base = np.full(total_rows, "-", dtype="U1")
    st_base = np.full(total_rows, "-", dtype="U1")
    lp_base = np.full(total_rows, "-", dtype="U1")

    shifts = (1, 2, 3, 4, 5)
    b_seq = {sh: np.full(total_rows, "-", dtype="U1") for sh in shifts}
    b_st = {sh: np.full(total_rows, "-", dtype="U1") for sh in shifts}
    b_lp = {sh: np.full(total_rows, "-", dtype="U1") for sh in shifts}
    a_seq = {sh: np.full(total_rows, "-", dtype="U1") for sh in shifts}
    a_st = {sh: np.full(total_rows, "-", dtype="U1") for sh in shifts}
    a_lp = {sh: np.full(total_rows, "-", dtype="U1") for sh in shifts}

    for r in range(len(df)):
        npos = int(n_pos_list[r])
        start = int(row_starts[r])
        end = start + npos

        seq_arr = np.frombuffer(seqs[r].encode("ascii"), dtype="S1").astype(
            "U1", copy=False
        )
        st_arr = np.frombuffer(structs[r].encode("ascii"), dtype="S1").astype(
            "U1", copy=False
        )
        lp_arr = np.frombuffer(loops[r].encode("ascii"), dtype="S1").astype(
            "U1", copy=False
        )

        seq_base[start:end] = seq_arr[:npos]
        st_base[start:end] = st_arr[:npos]
        lp_base[start:end] = lp_arr[:npos]

        for sh in shifts:
            if npos > sh:
                b_seq[sh][start + sh : end] = seq_arr[: npos - sh]
                b_st[sh][start + sh : end] = st_arr[: npos - sh]
                b_lp[sh][start + sh : end] = lp_arr[: npos - sh]
                a_seq[sh][start : end - sh] = seq_arr[sh:npos]
                a_st[sh][start : end - sh] = st_arr[sh:npos]
                a_lp[sh][start : end - sh] = lp_arr[sh:npos]

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

    return pd.DataFrame(data)


train_data = build_positional_df(train, is_train=True)
test_data = build_positional_df(test, is_train=False)
train_data.head()




## === cell 5
sequence_encmap = {"A": 0, "G": 1, "C": 2, "U": 3}
structure_encmap = {".": 0, "(": 1, ")": 2}
looptype_encmap = {"S": 0, "E": 1, "H": 2, "I": 3, "X": 4, "M": 5, "B": 6}

PAD_CODE = -1
for _m in (sequence_encmap, structure_encmap, looptype_encmap):
    _m.setdefault("-", PAD_CODE)
    _m.setdefault("N", PAD_CODE)  # defensive for potential unknown base
    _m.setdefault("?", PAD_CODE)  # defensive for any unexpected token

enc_targets = ["sequence", "structure", "predicted_loop_type"]
enc_maps = [sequence_encmap, structure_encmap, looptype_encmap]

for t, m in zip(enc_targets, enc_maps):
    cols = [c for c in train_data.columns if t in c]
    for c in cols:
        train_data[c] = train_data[c].replace(m).astype(np.int16, copy=False)
        test_data[c] = test_data[c].replace(m).astype(np.int16, copy=False)


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

result = {}
oof_df = pd.DataFrame(train_data.id_seqpos)

X_all = np.ascontiguousarray(
    train_data[features].to_numpy(dtype=np.float32, copy=False)
)
X_test = np.ascontiguousarray(
    test_data[features].to_numpy(dtype=np.float32, copy=False)
)
groups = train_data["id"].to_numpy()

cv_splits = list(gkf.split(X_all, train["reactivity"], groups))

sub_index = pd.Index(submission["id_seqpos"])
test_index = pd.Index(test_data["id_seqpos"])
sub_pos = sub_index.get_indexer(test_index)
if (sub_pos < 0).any():
    raise RuntimeError(
        "submission id_seqpos and test id_seqpos do not align as expected."
    )

pos_idx_int = (
    pd.Series(train_data["id_seqpos"], copy=False)
    .astype(str)
    .str.rsplit("_", n=1)
    .str[-1]
    .astype(np.int16)
    .to_numpy(dtype=np.int64, copy=False)
)

NUM_BOOST_ROUND = 2000

callbacks = []
if early_stopping is not None:
    callbacks.append(lgb.early_stopping(stopping_rounds=100))
if verbose is not None:
    callbacks.append(lgb.log_evaluation(period=1000))

target_order = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
y_tensor = np.stack(
    [np.stack(train[t].to_numpy(), axis=0) for t in target_order], axis=-1
).astype(
    np.float32, copy=False
)  # (n_mols, 68, 5)

mol_idx_per_pos = np.repeat(np.arange(len(train), dtype=np.int32), 68)

fi_records = []
oof_pred_cols = {t: np.empty(len(train_data), dtype=np.float32) for t in targets}

fold_datasets = []
for n, (tr_idx, vl_idx) in enumerate(cv_splits):
    tr_data = lgb.Dataset(
        X_all[tr_idx],
        label=np.zeros(len(tr_idx), dtype=np.float32),
        free_raw_data=True,
        feature_name=features,
    )
    vl_data = lgb.Dataset(
        X_all[vl_idx],
        label=np.zeros(len(vl_idx), dtype=np.float32),
        reference=tr_data,
        free_raw_data=True,
        feature_name=features,
    )
    fold_datasets.append((tr_idx, vl_idx, tr_data, vl_data))

for ti, target in enumerate(targets):
    preds = np.zeros(len(test_data), dtype=np.float32)
    scores = 0.0

    y_all = y_tensor[mol_idx_per_pos, pos_idx_int, ti]

    for n, (tr_idx, vl_idx, tr_data, vl_data) in enumerate(fold_datasets):
        tr_data.set_label(y_all[tr_idx])
        vl_data.set_label(y_all[vl_idx])

        model = TreeModel(model_type="lgb")
        model.tr_data = tr_data
        model.vl_data = vl_data

        model.model = lgb.train(
            params,
            model.tr_data,
            valid_sets=[model.vl_data],
            num_boost_round=NUM_BOOST_ROUND,
            callbacks=callbacks,
        )

        imp = model.feature_importances_
        fn = model.feature_names_
        fi_records.extend((f, float(im), int(n), target) for f, im in zip(fn, imp))

        vl_x = X_all[vl_idx]
        vl_y = y_all[vl_idx]
        vl_pred = model.predict(vl_x).astype(np.float32, copy=False)
        score = rmse(vl_y, vl_pred)
        scores += score / FOLD_N
        print(f"score : {score}")

        oof_pred_cols[target][vl_idx] = vl_pred

        preds += model.predict(X_test).astype(np.float32, copy=False) / FOLD_N

        del model
        gc.collect()

    submission.loc[sub_pos, target] = preds
    print(f"{target}_rmse : {scores}")
    result[target] = scores

feature_importances = pd.DataFrame(
    fi_records, columns=["feature", "importance", "fold", "target"]
)
for t in targets:
    oof_df[t] = oof_pred_cols[t]




## --- ERROR in cell 9, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/224271722.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     13[0m [0mgroups[0m [0;34m=[0m [0mtrain_data[0m[0;34m[[0m[0;34m"id"[0m[0;34m][0m[0;34m.[0m[0mto_numpy[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     14[0m [0;34m[0m[0m
[0;32m---> 15[0;31m [0mcv_splits[0m [0;34m=[0m [0mlist[0m[0;34m([0m[0mgkf[0m[0;34m.[0m[0msplit[0m[0;34m([0m[0mX_all[0m[0;34m,[0m [0mtrain[0m[0;34m[[0m[0;34m"reactivity"[0m[0;34m][0m[0;34m,[0m [0mgroups[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     16[0m [0;34m[0m[0m
[1;32m     17[0m [0msub_index[0m [0;34m=[0m [0mpd[0m[0;34m.[0m[0mIndex[0m[0;34m([0m[0msubmission[0m[0;34m[[0m[0;34m"id_seqpos"[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py[0m in [0;36msplit[0;34m(self, X, y, groups)[0m
[1;32m    340[0m             [0mThe[0m [0mtesting[0m [0mset[0m [0mindices[0m [0;32mfor[0m [0mthat[0m [0msplit[0m[0;34m.[0m[0;34m[0m[0;34m[0m[0m
[1;32m    341[0m         """
[0;32m--> 342[0;31m         [0mX[0m[0;34m,[0m [0my[0m[0;34m,[0m [0mgroups[0m [0;34m=[0m [0mindexable[0m[0;34m([0m[0mX[0m[0;34m,[0m [0my[0m[0;34m,[0m [0mgroups[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    343[0m         [0mn_samples[0m [0;34m=[0m [0m_num_samples[0m[0;34m([0m[0mX[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    344[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0mn_splits[0m [0;34m>[0m [0mn_samples[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py[0m in [0;36mindexable[0;34m(*iterables)[0m
[1;32m    441[0m [0;34m[0m[0m
[1;32m    442[0m     [0mresult[0m [0;34m=[0m [0;34m[[0m[0m_make_indexable[0m[0;34m([0m[0mX[0m[0;34m)[0m [0;32mfor[0m [0mX[0m [0;32min[0m [0miterables[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 443[0;31m     [0mcheck_consistent_length[0m[0;34m([0m[0;34m*[0m[0mresult[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    444[0m     [0;32mreturn[0m [0mresult[0m[0;34m[0m[0;34m[0m[0m
[1;32m    445[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py[0m in [0;36mcheck_consistent_length[0;34m(*arrays)[0m
[1;32m    395[0m     [0muniques[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0munique[0m[0;34m([0m[0mlengths[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    396[0m     [0;32mif[0m [0mlen[0m[0;34m([0m[0muniques[0m[0;34m)[0m [0;34m>[0m [0;36m1[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 397[0;31m         raise ValueError(
[0m[1;32m    398[0m             [0;34m"Found input variables with inconsistent numbers of samples: %r"[0m[0;34m[0m[0;34m[0m[0m
[1;32m    399[0m             [0;34m%[0m [0;34m[[0m[0mint[0m[0;34m([0m[0ml[0m[0;34m)[0m [0;32mfor[0m [0ml[0m [0;32min[0m [0mlengths[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: Found input variables with inconsistent numbers of samples: [146880, 2160, 146880]

## === cell 10
display(result)
display(f"total : {np.mean(list(result.values()))}")
