# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Predict the `scalar_coupling_constant` between atom pairs in molecules, given the two atom types (e.g., C and H), the coupling type (e.g., `2JHC`), and any features you are able to create from the molecule structure (`xyz`) files.

## Metric
Log of the Mean Absolute Error, calculated for each scalar coupling type, and then averaged across types.

## Submission Format
```
id,scalar_coupling_constant
2324604,0.0
2324605,0.0
2324606,0.0
etc.
```

## Dataset
The training and test splits are by *molecule*, so that no molecule in the training data is found in the test data.

- **train.csv** - the training set, where the first column (`molecule_name`) is the name of the molecule where the coupling constant originates (the corresponding XYZ file is located at ./structures/.xyz), the second (`atom_index_0`) and third column (`atom_index_1`) is the atom indices of the atom-pair creating the coupling and the fourth column (`scalar_coupling_constant`) is the scalar coupling constant that we want to be able to predict
- **test.csv** - the test set; same info as train, without the target variable
- **sample_submission.csv** - a sample submission file in the correct format
- **structures.zip** - folder containing molecular structure (xyz) files, where the first line is the number of atoms in the molecule, followed by a blank line, and then a line for every atom, where the first column contains the atomic element (H for hydrogen, C for carbon etc.) and the remaining columns contain the X, Y and Z cartesian coordinates (a standard format for chemists and molecular visualization programs)
- **structures.csv** - this file contains the **same** information as the individual xyz structure files, but in a single file
- **dipole_moments.csv** - contains the molecular electric dipole moments. These are three dimensional vectors that indicate the charge distribution in the molecule. The first column (`molecule_name`) are the names of the molecule, the second to fourth column are the `X`, `Y` and `Z` components respectively of the dipole moment.
- **magnetic_shielding_tensors.csv** - contains the magnetic shielding tensors for all atoms in the molecules. The first column (`molecule_name`) contains the molecule name, the second column (`atom_index`) contains the index of the atom in the molecule, the third to eleventh columns contain the `XX`, `YX`, `ZX`, `XY`, `YY`, `ZY`, `XZ`, `YZ` and `ZZ` elements of the tensor/matrix respectively.
- **mulliken_charges.csv** - contains the mulliken charges for all atoms in the molecules. The first column (`molecule_name`) contains the name of the molecule, the second column (`atom_index`) contains the index of the atom in the molecule, the third column (`mulliken_charge`) contains the mulliken charge of the atom.
- **potential_energy.csv** - contains the potential energy of the molecules. The first column (`molecule_name`) contains the name of the molecule, the second column (`potential_energy`) contains the potential energy of the molecule.
- **scalar_coupling_contributions.csv** - The scalar coupling constants in `train.csv` (or corresponding files) are a sum of four terms. `scalar_coupling_contributions.csv` contain all these terms. The first column (`molecule_name`) are the name of the molecule, the second (`atom_index_0`) and third column (`atom_index_1`) are the atom indices of the atom-pair, the fourth column indicates the type of coupling, the fifth column (`fc`) is the Fermi Contact contribution, the sixth column (`sd`) is the Spin-dipolar contribution, the seventh column (`pso`) is the Paramagnetic spin-orbit contribution and the eighth column (`dso`) is the Diamagnetic spin-orbit contribution.

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
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
            description.md (111 lines)
            dipole_moments.csv (76511 lines)
            dipole_moments.csv.zip (892.4 kB)
            magnetic_shielding_tensors.csv (1379965 lines)
            magnetic_shielding_tensors.csv.zip (47.9 MB)
            mulliken_charges.csv (1379965 lines)
            mulliken_charges.csv.zip (9.5 MB)
            potential_energy.csv (76511 lines)
            potential_energy.csv.zip (641.9 kB)
            sample_submission.csv (467814 lines)
            sample_submission.csv.zip (846.9 kB)
            scalar_coupling_contributions.csv (4191264 lines)
            scalar_coupling_contributions.csv.zip (90.0 MB)
            structures.csv (1379965 lines)
            structures.csv.zip (33.0 MB)
            structures.zip (44.3 MB)
            test.csv (467814 lines)
            test.csv.zip (2.6 MB)
            train.csv (4191264 lines)
            train.csv.zip (43.6 MB)
            champs-scalar-coupling/
                description.md (111 lines)
                dipole_moments.csv (76511 lines)
                ... and 18 other files
                champs-scalar-coupling/
                structures/
                    dsgdb9nsd_000001.xyz (212 Bytes)
                    dsgdb9nsd_000002.xyz (171 Bytes)
                    ... and 76508 other files
            structures/
                dsgdb9nsd_000001.xyz (212 Bytes)
                dsgdb9nsd_000002.xyz (171 Bytes)
                ... and 76508 other files
        input/
            description.md (111 lines)
            dipole_moments.csv (76511 lines)
            dipole_moments.csv.zip (892.4 kB)
            magnetic_shielding_tensors.csv (1379965 lines)
            magnetic_shielding_tensors.csv.zip (47.9 MB)
            mulliken_charges.csv (1379965 lines)
            mulliken_charges.csv.zip (9.5 MB)
            potential_energy.csv (76511 lines)
            potential_energy.csv.zip (641.9 kB)
            sample_submission.csv (467814 lines)
            sample_submission.csv.zip (846.9 kB)
            scalar_coupling_contributions.csv (4191264 lines)
            scalar_coupling_contributions.csv.zip (90.0 MB)
            structures.csv (1379965 lines)
            structures.csv.zip (33.0 MB)
            structures.zip (44.3 MB)
            test.csv (467814 lines)
            test.csv.zip (2.6 MB)
            train.csv (4191264 lines)
            train.csv.zip (43.6 MB)
            champs-scalar-coupling/
                description.md (111 lines)
                dipole_moments.csv (76511 lines)
                ... and 18 other files
                champs-scalar-coupling/
                structures/
                    dsgdb9nsd_000001.xyz (212 Bytes)
                    dsgdb9nsd_000002.xyz (171 Bytes)
                    ... and 76508 other files
            structures/
                dsgdb9nsd_000001.xyz (212 Bytes)
                dsgdb9nsd_000002.xyz (171 Bytes)
                ... and 76508 other files
        working/
            champs-scalar-coupling/
                description.md (111 lines)
                dipole_moments.csv (76511 lines)
                ... and 18 other files
                champs-scalar-coupling/
                structures/
                    dsgdb9nsd_000001.xyz (212 Bytes)
                    dsgdb9nsd_000002.xyz (171 Bytes)
                    ... and 76508 other files
```

-> data/champs-scalar-coupling/dipole_moments.csv has 76510 rows and 4 columns.
The columns are: molecule_name, X, Y, Z

-> data/champs-scalar-coupling/magnetic_shielding_tensors.csv has 1379964 rows and 11 columns.
The columns are: molecule_name, atom_index, XX, YX, ZX, XY, YY, ZY, XZ, YZ, ZZ

-> data/champs-scalar-coupling/mulliken_charges.csv has 1379964 rows and 3 columns.
The columns are: molecule_name, atom_index, mulliken_charge

-> data/champs-scalar-coupling/potential_energy.csv has 76510 rows and 2 columns.
The columns are: molecule_name, potential_energy

-> data/champs-scalar-coupling/sample_submission.csv has 467813 rows and 2 columns.
The columns are: id, scalar_coupling_constant

-> data/champs-scalar-coupling/scalar_coupling_contributions.csv has 4191263 rows and 8 columns.
The columns are: molecule_name, atom_index_0, atom_index_1, type, fc, sd, pso, dso

-> data/champs-scalar-coupling/structures.csv has 1379964 rows and 6 columns.
The columns are: molecule_name, atom_index, atom, x, y, z

-> data/champs-scalar-coupling/test.csv has 467813 rows and 5 columns.
The columns are: id, molecule_name, atom_index_0, atom_index_1, type

-> data/champs-scalar-coupling/train.csv has 4191263 rows and 6 columns.
The columns are: id, molecule_name, atom_index_0, atom_index_1, type, scalar_coupling_constant

-> (stopped after 10 files for performance)

# 5. Target score

0.4801548222909683

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import sys
import hashlib
import time
import numpy as np
import pandas as pd



## === cell 1
SEED = 31
TRIALS = 200
TARGET = "scalar_coupling_constant"
PREDICTION = "pred"




## === cell 2
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


seed_everything(SEED)




## === cell 3
def group_mean_log_mae_np(y_true, y_pred, types, floor=1e-9):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    types = np.asarray(types)

    type_ids, _ = pd.factorize(types, sort=False)
    abs_err = np.abs(y_true - y_pred)

    sums = np.bincount(type_ids, weights=abs_err)
    cnts = np.bincount(type_ids)
    maes = sums / cnts
    maes = np.maximum(maes, floor)
    return float(np.mean(np.log(maes)))


def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    return group_mean_log_mae_np(
        np.asarray(y_true), np.asarray(y_pred), np.asarray(types), floor=floor
    )




## === cell 4
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor

DATA_DIR_CANDIDATES = [
    "/kaggle/input/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling",
    "/kaggle/data/input/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling/champs-scalar-coupling",
    "/kaggle/data/input/champs-scalar-coupling/champs-scalar-coupling",
]


def find_data_dir():
    for d in DATA_DIR_CANDIDATES:
        if os.path.exists(os.path.join(d, "train.csv")):
            return d
    d = "/kaggle/data/champs-scalar-coupling"
    if os.path.exists(os.path.join(d, "train.csv")):
        return d
    raise FileNotFoundError(
        "Could not locate champs-scalar-coupling data directory with train.csv"
    )


DATA_DIR = find_data_dir()
print("Using DATA_DIR:", DATA_DIR)

train = pd.read_csv(
    os.path.join(DATA_DIR, "train.csv"),
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
        TARGET: np.float32,
    },
)
test = pd.read_csv(
    os.path.join(DATA_DIR, "test.csv"),
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
    },
)
structures = pd.read_csv(
    os.path.join(DATA_DIR, "structures.csv"),
    dtype={
        "molecule_name": "category",
        "atom_index": np.int16,
        "atom": "category",
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
    },
)

print("train:", train.shape, "test:", test.shape, "structures:", structures.shape)

structures = structures.sort_values(["molecule_name", "atom_index"], kind="mergesort")

s0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "x_0",
        "y": "y_0",
        "z": "z_0",
    }
)
s1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "x": "x_1",
        "y": "y_1",
        "z": "z_1",
    }
)


def add_pair_features(df):
    df = df.merge(
        s0[["molecule_name", "atom_index_0", "atom_0", "x_0", "y_0", "z_0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
        sort=False,
        copy=False,
    )
    df = df.merge(
        s1[["molecule_name", "atom_index_1", "atom_1", "x_1", "y_1", "z_1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
        sort=False,
        copy=False,
    )
    dx = df["x_0"].to_numpy(np.float32) - df["x_1"].to_numpy(np.float32)
    dy = df["y_0"].to_numpy(np.float32) - df["y_1"].to_numpy(np.float32)
    dz = df["z_0"].to_numpy(np.float32) - df["z_1"].to_numpy(np.float32)
    df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)
    return df


train_f = add_pair_features(train)
test_f = add_pair_features(test)

all_types = pd.Categorical(
    pd.concat([train_f["type"], test_f["type"]], axis=0), ordered=False
)
type_map = {k: i for i, k in enumerate(all_types.categories)}

atom0_all = pd.Categorical(
    pd.concat([train_f["atom_0"], test_f["atom_0"]], axis=0), ordered=False
)
atom1_all = pd.Categorical(
    pd.concat([train_f["atom_1"], test_f["atom_1"]], axis=0), ordered=False
)
atom0_map = {k: i for i, k in enumerate(atom0_all.categories)}
atom1_map = {k: i for i, k in enumerate(atom1_all.categories)}

for df in (train_f, test_f):
    df["type_id"] = df["type"].map(type_map).astype(np.int16)
    df["atom0_id"] = df["atom_0"].map(atom0_map).fillna(-1).astype(np.int16)
    df["atom1_id"] = df["atom_1"].map(atom1_map).fillna(-1).astype(np.int16)

feature_cols = [
    "dist",
    "x_0",
    "y_0",
    "z_0",
    "x_1",
    "y_1",
    "z_1",
    "type_id",
    "atom0_id",
    "atom1_id",
]
for c in feature_cols:
    train_f[c] = train_f[c].fillna(0)
    test_f[c] = test_f[c].fillna(0)

X = train_f[feature_cols].to_numpy(dtype=np.float32, copy=False)
y = train_f[TARGET].to_numpy(dtype=np.float32, copy=False)
X_test = test_f[feature_cols].to_numpy(dtype=np.float32, copy=False)
groups = train_f["molecule_name"].to_numpy()
types = train_f["type"].to_numpy()

models = [
    (
        "lasso",
        Pipeline(
            [
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("model", Lasso(alpha=1e-4, random_state=SEED, max_iter=5000)),
            ]
        ),
    ),
    (
        "rf",
        RandomForestRegressor(
            n_estimators=80,
            random_state=SEED,
            n_jobs=-1,
            max_depth=18,
            min_samples_leaf=2,
        ),
    ),
    (
        "xgb",
        ExtraTreesRegressor(
            n_estimators=250,
            random_state=SEED,
            n_jobs=-1,
            max_depth=22,
            min_samples_leaf=2,
        ),
    ),
    (
        "lgb",
        Pipeline(
            [
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("model", Ridge(alpha=1.0, random_state=SEED)),
            ]
        ),
    ),
    (
        "keras",
        Pipeline(
            [
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                (
                    "model",
                    KNeighborsRegressor(n_neighbors=25, weights="distance", n_jobs=-1),
                ),
            ]
        ),
    ),
]

gkf = GroupKFold(n_splits=3)

CACHE_DIR = "./model_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def _fingerprint_array(a, n=2048):
    a = np.asarray(a)
    if a.ndim == 1:
        view = a[:n]
    else:
        view = a[: min(len(a), n), : min(a.shape[1], 32)]
    h = hashlib.md5(view.tobytes()).hexdigest()
    return h


data_fp = hashlib.md5(
    (
        "|".join(feature_cols) + f"|seed={SEED}|X={X.shape}|Xt={X_test.shape}|"
        f"xfp={_fingerprint_array(X)}|yfp={_fingerprint_array(y)}|tstfp={_fingerprint_array(X_test)}"
    ).encode("utf-8")
).hexdigest()


def cache_path(model_name, model_obj):
    params = str(getattr(model_obj, "get_params", lambda: {})())
    key = hashlib.md5(
        (data_fp + "|" + model_name + "|" + params).encode("utf-8")
    ).hexdigest()
    return os.path.join(CACHE_DIR, f"{model_name}_{key}.npz")


train_sets = []
test_sets = []

for name, model in models:
    cpath = cache_path(name, model)
    if os.path.exists(cpath):
        loaded = np.load(cpath)
        oof = loaded["oof"].astype(np.float32, copy=False)
        test_pred = loaded["test_pred"].astype(np.float32, copy=False)
        print(f"{name}: loaded cached predictions from {cpath}")
    else:
        oof = np.zeros(len(train_f), dtype=np.float32)
        test_pred = np.zeros(len(test_f), dtype=np.float32)

        for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_va = X[va_idx]

            model.fit(X_tr, y_tr)
            oof[va_idx] = model.predict(X_va).astype(np.float32, copy=False)
            test_pred += (
                model.predict(X_test).astype(np.float32, copy=False) / gkf.n_splits
            )

        np.savez_compressed(cpath, oof=oof, test_pred=test_pred)
        print(f"{name}: saved cached predictions to {cpath}")

    fold_score = group_mean_log_mae(y, oof, types)
    print(f"{name} OOF score: {fold_score:.6f}")

    df_tr = train_f[["id", "type", TARGET]].copy()
    df_tr[PREDICTION] = oof
    train_sets.append(df_tr)

    df_te = test_f[["id"]].copy()
    df_te[TARGET] = test_pred
    test_sets.append(df_te)

print("Built train_sets and test_sets:", len(train_sets), len(test_sets))
print("Shapes:", [t.shape for t in train_sets], [t.shape for t in test_sets])



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3662638074.py in <cell line: 0>()
    131 for df in (train_f, test_f):
    132     df["type_id"] = df["type"].map(type_map).astype(np.int16)
--> 133     df["atom0_id"] = df["atom_0"].map(atom0_map).fillna(-1).astype(np.int16)
    134     df["atom1_id"] = df["atom_1"].map(atom1_map).fillna(-1).astype(np.int16)
    135 

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in fillna(self, value, method, axis, inplace, limit, downcast)
   7347                     )
   7348 
-> 7349                 new_data = self._mgr.fillna(
   7350                     value=value, limit=limit, inplace=inplace, downcast=downcast
   7351                 )

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/base.py in fillna(self, value, limit, inplace, downcast)
    184             limit = libalgos.validate_limit(None, limit=limit)
    185 
--> 186         return self.apply_with_block(
    187             "fillna",
    188             value=value,

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in apply(self, f, align_keys, **kwargs)
    361                 applied = b.apply(f, **kwargs)
    362             else:
--> 363                 applied = getattr(b, f)(**kwargs)
    364             result_blocks = extend_blocks(applied, result_blocks)
    365 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in fillna(self, value, limit, inplace, downcast, using_cow, already_warned)
   2332                 # 3rd party EA that has not implemented copy keyword yet
   2333                 refs = None
-> 2334                 new_values = self.values.fillna(value=value, method=None, limit=limit)
   2335                 # issue the warning *after* retrying, in case the TypeError
   2336                 #  was caused by an invalid fill_value

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/_mixins.py in fillna(self, value, method, limit, copy)
    374             # We validate the fill_value even if there is nothing to fill
    375             if value is not None:
--> 376                 self._validate_setitem_value(value)
    377 
    378             if not copy:

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/categorical.py in _validate_setitem_value(self, value)
   1587             return self._validate_listlike(value)
   1588         else:
-> 1589             return self._validate_scalar(value)
   1590 
   1591     def _validate_scalar(self, fill_value):

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/categorical.py in _validate_scalar(self, fill_value)
   1612             fill_value = self._unbox_scalar(fill_value)
   1613         else:
-> 1614             raise TypeError(
   1615                 "Cannot setitem on a Categorical with a new "
   1616                 f"category ({fill_value}), set the categories first"

TypeError: Cannot setitem on a Categorical with a new category (-1), set the categories first

## === cell 5
print(f"Train sets ready: {[df.shape for df in train_sets]}")



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3204823644.py in <cell line: 0>()
----> 1 print(f"Train sets ready: {[df.shape for df in train_sets]}")
      2 

NameError: name 'train_sets' is not defined

## === cell 6
print(train_sets[0].head())
print(test_sets[0].head())




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/614568636.py in <cell line: 0>()
----> 1 print(train_sets[0].head())
      2 print(test_sets[0].head())
      3 
      4 

NameError: name 'train_sets' is not defined

## === cell 7
def weights(n, min_weight=0.01, max_weight=0.99):
    if n < 1:
        raise ValueError("n must not be less than 1")
    res = []
    remainder = 1.0
    for _ in range(n - 1):
        w = random.uniform(min_weight, max_weight) * remainder
        res.append(w)
        remainder -= w
    res.append(remainder)
    return res


pred_mat = np.vstack(
    [df[PREDICTION].to_numpy(np.float64, copy=False) for df in train_sets]
)  # (n_models, n_rows)
y_true = train_sets[0][TARGET].to_numpy(np.float64, copy=False)
types_arr = train_sets[0]["type"].to_numpy()


def trial(pred_mat, y_true, types_arr):
    ws = np.asarray(weights(pred_mat.shape[0]), dtype=np.float64)
    y_pred = (ws[:, None] * pred_mat).sum(axis=0)
    score = group_mean_log_mae_np(y_true, y_pred, types_arr)
    return score, ws.tolist()


t0 = time.time()
best = sys.maxsize
best_weights = []
for i in range(TRIALS):
    score, ws = trial(pred_mat=pred_mat, y_true=y_true, types_arr=types_arr)
    if score < best:
        best = score
        best_weights = ws

print(f"best={best:.6f} (searched {TRIALS} trials in {time.time()-t0:.1f}s)")
print(
    f"""best weights (sum={sum(best_weights)})
  m1={best_weights[0]:.4f}
  m2={best_weights[1]:.4f}
  m3={best_weights[2]:.4f}
  m4={best_weights[3]:.4f}
  m5={best_weights[4]:.4f}
"""
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3640895670.py in <cell line: 0>()
     16 # Pre-extract arrays once to avoid per-trial DataFrame operations.
     17 pred_mat = np.vstack(
---> 18     [df[PREDICTION].to_numpy(np.float64, copy=False) for df in train_sets]
     19 )  # (n_models, n_rows)
     20 y_true = train_sets[0][TARGET].to_numpy(np.float64, copy=False)

NameError: name 'train_sets' is not defined

## === cell 8
test_pred_mat = np.vstack(
    [df[TARGET].to_numpy(np.float64, copy=False) for df in test_sets]
)  # (n_models, n_test)
best_w = np.asarray(best_weights, dtype=np.float64)
blended = (best_w[:, None] * test_pred_mat).sum(axis=0)

submission = test_sets[0][["id"]].copy()
submission[TARGET] = blended.astype(np.float64, copy=False)

submission = submission[["id", TARGET]].copy()
submission["id"] = submission["id"].astype(np.int64)
submission[TARGET] = submission[TARGET].astype(np.float64)

print(submission.head())
print(submission.shape)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print("Files in working dir:", os.listdir("."))

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2548992153.py in <cell line: 0>()
      1 # CHANGE (timeout): Vectorize test blending similarly; no semantic change.
      2 test_pred_mat = np.vstack(
----> 3     [df[TARGET].to_numpy(np.float64, copy=False) for df in test_sets]
      4 )  # (n_models, n_test)
      5 best_w = np.asarray(best_weights, dtype=np.float64)

NameError: name 'test_sets' is not defined
