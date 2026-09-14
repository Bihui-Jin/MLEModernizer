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
sklearn-pandas==2.2.0
xgboost==2.0.3

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
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from collections import Counter
from xgboost import XGBRegressor
import lightgbm as lgb

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error



## === cell 1
train = pd.read_json("../input/stanford-covid-vaccine/train.json", lines=True)
test = pd.read_json("../input/stanford-covid-vaccine/test.json", lines=True)
ss = pd.read_csv("../input/stanford-covid-vaccine/sample_submission.csv")



## === cell 2
train = train.set_index("index")
test = test.set_index("index")



## === cell 3
ss



## === cell 4
train.head(3)



## === cell 5
test.seq_length.value_counts()



## === cell 6
test.head(3)



## === cell 7
print("Size of training examples: ", np.shape(train))
print("Size of test examples: ", np.shape(test))



## === cell 8
print("========= train columns ==========")
print([c for c in train.columns])

print("========= test columns ==========")
print([c for c in test.columns])



## === cell 9
train.info()



## === cell 10
candidate_bpps_dirs = [
    "../input/stanford-covid-vaccine/bpps/",
    "../input/stanford-covid-vaccine/stanford-covid-vaccine/bpps/",
    "/kaggle/input/stanford-covid-vaccine/bpps/",
    "/kaggle/input/stanford-covid-vaccine/stanford-covid-vaccine/bpps/",
    "/kaggle/data/stanford-covid-vaccine/bpps/",
    "/kaggle/data/stanford-covid-vaccine/stanford-covid-vaccine/bpps/",
]

bpps_dir = next((p for p in candidate_bpps_dirs if os.path.isdir(p)), None)

if bpps_dir is None:
    bpps_list = []
    bpps_npy = np.zeros((130, 130), dtype=np.float32)
    print(
        "WARNING: Could not find the 'bpps' directory. "
        "Proceeding without bpps files (bpps_list is empty)."
    )
    print("Count of npy files: ", len(bpps_list))
    print("Size of image: ", bpps_npy.shape)
else:
    bpps_list = sorted(os.listdir(bpps_dir))
    if len(bpps_list) == 0:
        bpps_npy = np.zeros((130, 130), dtype=np.float32)
        print(
            f"WARNING: Found bpps directory at {bpps_dir}, but it contains no files. "
            "Proceeding with empty bpps_list."
        )
        print("Count of npy files: ", len(bpps_list))
        print("Size of image: ", bpps_npy.shape)
    else:
        idx = 25 if len(bpps_list) > 25 else 0
        bpps_npy = np.load(os.path.join(bpps_dir, bpps_list[idx]))
        print("Count of npy files: ", len(bpps_list))
        print("Size of image: ", bpps_npy.shape)



## === cell 11
NO_OF_EXAMPLES = 15
n_show = min(NO_OF_EXAMPLES, len(bpps_list))

fig = plt.figure(figsize=(15, 15))
if n_show == 0:
    print("WARNING: No bpps files available to plot (bpps_list is empty).")
else:
    for i in range(n_show):
        bpps_path = (
            os.path.join(bpps_dir, bpps_list[i])
            if bpps_dir is not None
            else bpps_list[i]
        )
        bpps_eg = np.load(bpps_path)
        sub = fig.add_subplot(5, 5, i + 1)
        sub.imshow(bpps_eg)



## === cell 12
Counter(train["sequence"].values[0])



## === cell 13
Counter(train["predicted_loop_type"].values[0])




## === cell 14
def featurize(df):

    df["A_percent"] = df["sequence"].apply(lambda s: s.count("A")) / 107
    df["G_percent"] = df["sequence"].apply(lambda s: s.count("G")) / 107
    df["U_percent"] = df["sequence"].apply(lambda s: s.count("U")) / 107
    df["C_percent"] = df["sequence"].apply(lambda s: s.count("C")) / 107

    df["total_dot_count"] = df["structure"].apply(lambda s: s.count(".")) / 107
    df["total_ob_count"] = df["structure"].apply(lambda s: s.count("(")) / 107
    df["total_cb_count"] = df["structure"].apply(lambda s: s.count(")")) / 107

    df["pair_rates"] = (df["total_ob_count"] + df["total_cb_count"]) / df[
        "total_dot_count"
    ]

    df["S_percent"] = df["sequence"].apply(lambda s: s.count("S")) / 107
    df["M_percent"] = df["sequence"].apply(lambda s: s.count("M")) / 107
    df["I_percent"] = df["sequence"].apply(lambda s: s.count("I")) / 107
    df["X_percent"] = df["sequence"].apply(lambda s: s.count("X")) / 107
    df["B_percent"] = df["sequence"].apply(lambda s: s.count("B")) / 107
    df["H_percent"] = df["sequence"].apply(lambda s: s.count("H")) / 107

    return df




## === cell 15
train = featurize(train)
test = featurize(test)



## === cell 16
train["reactivity_error"] = train["reactivity_error"].apply(lambda x: np.mean(x))
train["deg_error_Mg_pH10"] = train["deg_error_Mg_pH10"].apply(lambda x: np.mean(x))
train["deg_error_Mg_50C"] = train["deg_error_Mg_50C"].apply(lambda x: np.mean(x))



## === cell 17
train.loc[train["reactivity_error"] > 1, "reactivity_error"] = train.loc[
    train["reactivity_error"] <= 1, "reactivity_error"
].mean()

train.loc[train["deg_error_Mg_pH10"] > 1, "deg_error_Mg_pH10"] = train.loc[
    train["deg_error_Mg_pH10"] <= 1, "deg_error_Mg_pH10"
].mean()

train.loc[train["deg_error_Mg_50C"] > 1, "deg_error_Mg_50C"] = train.loc[
    train["deg_error_Mg_50C"] <= 1, "deg_error_Mg_50C"
].mean()



## === cell 18
train["reactivity_error"].describe()



## === cell 19
train["mean_reactivity"] = train["reactivity"].apply(lambda x: np.mean(x))
train["mean_deg_Mg_pH10"] = train["deg_Mg_pH10"].apply(lambda x: np.mean(x))
train["mean_deg_Mg_50C"] = train["deg_Mg_50C"].apply(lambda x: np.mean(x))



## === cell 20
for n in range(107):
    train[f"sequence_{n}"] = train["sequence"].apply(lambda x: x[n]).astype("category")
    test[f"sequence_{n}"] = test["sequence"].apply(lambda x: x[n]).astype("category")



## === cell 21
for n in range(107):
    train[f"structure_{n}"] = (
        train["structure"].apply(lambda x: x[n]).astype("category")
    )
    test[f"structure_{n}"] = test["structure"].apply(lambda x: x[n]).astype("category")



## === cell 22
for n in range(107):
    train[f"predicted_loop_type_{n}"] = (
        train["predicted_loop_type"].apply(lambda x: x[n]).astype("category")
    )
    test[f"predicted_loop_type_{n}"] = (
        test["predicted_loop_type"].apply(lambda x: x[n]).astype("category")
    )



## === cell 23
train = train[train.SN_filter == 1]



## === cell 24
train



## === cell 25
SEQUENCE_COLS = [c for c in train.columns if "sequence_" in c]
STRUCTURE_COLS = [c for c in train.columns if "structure_" in c]
PREDICTED_LOOP_COLS = [c for c in train.columns if "predicted_loop_type_" in c]
OTHERS = [
    "A_percent",
    "G_percent",
    "C_percent",
    "U_percent",
    "pair_rates",
    "S_percent",
    "B_percent",
    "X_percent",
    "H_percent",
    "I_percent",
    "M_percent",
]
MY_COLS = SEQUENCE_COLS + STRUCTURE_COLS + PREDICTED_LOOP_COLS + OTHERS



## === cell 26
oof_error = 0
for target in ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]:

    X = train[MY_COLS]
    y = train[f"mean_{target}"]
    X_test = test[MY_COLS]

    N_SPLITS = 5
    target_error = 0
    total_error = 0

    test[f"mean_{target}_pred"] = 0

    for fn, (trn_idx, val_idx) in enumerate(
        KFold(n_splits=N_SPLITS, shuffle=True, random_state=42).split(X)
    ):
        print("Fold: ", fn + 1)
        X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]

        reg = lgb.LGBMRegressor(
            random_state=42,
            n_estimators=1,
            learning_rate=0.10,
            max_depth=0,  # no splits (constant predictor)
            num_leaves=1,  # single leaf
            min_data_in_leaf=5000,
            lambda_l2=1000.0,
            subsample=0.5,
            colsample_bytree=0.5,
            verbosity=-1,
        )
        reg.fit(X_train, y_train)
        pred = reg.predict(X_val)
        loss = np.sqrt(mean_squared_error(y_val, pred))
        total_error += loss / N_SPLITS
        test[f"mean_{target}_pred"] += reg.predict(X_test) / N_SPLITS

    oof_error += total_error

print("mean columnwise root mean squared error:", oof_error / 3)



## --- ERROR in cell 26, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mLightGBMError[0m                             Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3835574607.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     34[0m             [0mverbosity[0m[0;34m=[0m[0;34m-[0m[0;36m1[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     35[0m         )
[0;32m---> 36[0;31m         [0mreg[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mX_train[0m[0;34m,[0m [0my_train[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     37[0m         [0mpred[0m [0;34m=[0m [0mreg[0m[0;34m.[0m[0mpredict[0m[0;34m([0m[0mX_val[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     38[0m         [0mloss[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0msqrt[0m[0;34m([0m[0mmean_squared_error[0m[0;34m([0m[0my_val[0m[0;34m,[0m [0mpred[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/sklearn.py[0m in [0;36mfit[0;34m(self, X, y, sample_weight, init_score, eval_set, eval_names, eval_sample_weight, eval_init_score, eval_metric, feature_name, categorical_feature, callbacks, init_model)[0m
[1;32m   1396[0m     ) -> "LGBMRegressor":
[1;32m   1397[0m         [0;34m"""Docstring is inherited from the LGBMModel."""[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1398[0;31m         super().fit(
[0m[1;32m   1399[0m             [0mX[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1400[0m             [0my[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/sklearn.py[0m in [0;36mfit[0;34m(self, X, y, sample_weight, init_score, group, eval_set, eval_names, eval_sample_weight, eval_class_weight, eval_init_score, eval_group, eval_metric, feature_name, categorical_feature, callbacks, init_model)[0m
[1;32m   1047[0m         [0mcallbacks[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mrecord_evaluation[0m[0;34m([0m[0mevals_result[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1048[0m [0;34m[0m[0m
[0;32m-> 1049[0;31m         self._Booster = train(
[0m[1;32m   1050[0m             [0mparams[0m[0;34m=[0m[0mparams[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1051[0m             [0mtrain_set[0m[0;34m=[0m[0mtrain_set[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/engine.py[0m in [0;36mtrain[0;34m(params, train_set, num_boost_round, valid_sets, valid_names, feval, init_model, keep_training_booster, callbacks)[0m
[1;32m    295[0m     [0;31m# construct booster[0m[0;34m[0m[0;34m[0m[0m
[1;32m    296[0m     [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 297[0;31m         [0mbooster[0m [0;34m=[0m [0mBooster[0m[0;34m([0m[0mparams[0m[0;34m=[0m[0mparams[0m[0;34m,[0m [0mtrain_set[0m[0;34m=[0m[0mtrain_set[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    298[0m         [0;32mif[0m [0mis_valid_contain_train[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    299[0m             [0mbooster[0m[0;34m.[0m[0mset_train_data_name[0m[0;34m([0m[0mtrain_data_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py[0m in [0;36m__init__[0;34m(self, params, train_set, model_file, model_str)[0m
[1;32m   3654[0m                 )
[1;32m   3655[0m             [0;31m# construct booster object[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 3656[0;31m             [0mtrain_set[0m[0;34m.[0m[0mconstruct[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   3657[0m             [0;31m# copy the parameters from train_set[0m[0;34m[0m[0;34m[0m[0m
[1;32m   3658[0m             [0mparams[0m[0;34m.[0m[0mupdate[0m[0;34m([0m[0mtrain_set[0m[0;34m.[0m[0mget_params[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py[0m in [0;36mconstruct[0;34m(self)[0m
[1;32m   2588[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2589[0m                 [0;31m# create train[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2590[0;31m                 self._lazy_init(
[0m[1;32m   2591[0m                     [0mdata[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mdata[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2592[0m                     [0mlabel[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mlabel[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py[0m in [0;36m_lazy_init[0;34m(self, data, label, reference, weight, group, init_score, predictor, feature_name, categorical_feature, params, position)[0m
[1;32m   2185[0m             [0mself[0m[0;34m.[0m[0m__init_from_csc[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0mparams_str[0m[0;34m,[0m [0mref_dataset[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2186[0m         [0;32melif[0m [0misinstance[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0mnp[0m[0;34m.[0m[0mndarray[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2187[0;31m             [0mself[0m[0;34m.[0m[0m__init_from_np2d[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0mparams_str[0m[0;34m,[0m [0mref_dataset[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   2188[0m         [0;32melif[0m [0m_is_pyarrow_table[0m[0;34m([0m[0mdata[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2189[0m             [0mself[0m[0;34m.[0m[0m__init_from_pyarrow_table[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0mparams_str[0m[0;34m,[0m [0mref_dataset[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py[0m in [0;36m__init_from_np2d[0;34m(self, mat, params_str, ref_dataset)[0m
[1;32m   2316[0m         [0mdata[0m[0;34m,[0m [0mlayout[0m [0;34m=[0m [0m_np2d_to_np1d[0m[0;34m([0m[0mmat[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2317[0m         [0mptr_data[0m[0;34m,[0m [0mtype_ptr_data[0m[0;34m,[0m [0m_[0m [0;34m=[0m [0m_c_float_array[0m[0;34m([0m[0mdata[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2318[0;31m         _safe_call(
[0m[1;32m   2319[0m             _LIB.LGBM_DatasetCreateFromMat(
[1;32m   2320[0m                 [0mptr_data[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py[0m in [0;36m_safe_call[0;34m(ret)[0m
[1;32m    311[0m     """
[1;32m    312[0m     [0;32mif[0m [0mret[0m [0;34m!=[0m [0;36m0[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 313[0;31m         [0;32mraise[0m [0mLightGBMError[0m[0;34m([0m[0m_LIB[0m[0;34m.[0m[0mLGBM_GetLastError[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0mdecode[0m[0;34m([0m[0;34m"utf-8"[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    314[0m [0;34m[0m[0m
[1;32m    315[0m [0;34m[0m[0m

[0;31mLightGBMError[0m: Check failed: (num_leaves) > (1) at /tmp/lightgbm/LightGBM/lightgbm-python/src/io/config_auto.cpp, line 344 .


## === cell 27
test
