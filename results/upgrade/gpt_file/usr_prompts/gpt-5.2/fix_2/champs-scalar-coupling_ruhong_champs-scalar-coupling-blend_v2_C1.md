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
def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    """
    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling
    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric
    """
    maes = (y_true - y_pred).abs().groupby(types).mean()
    maes = np.log(maes.map(lambda x: max(x, floor)))
    return maes.mean()




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

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
structures = pd.read_csv(os.path.join(DATA_DIR, "structures.csv"))

print("train:", train.shape, "test:", test.shape, "structures:", structures.shape)

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
    )
    df = df.merge(
        s1[["molecule_name", "atom_index_1", "atom_1", "x_1", "y_1", "z_1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
    )
    dx = df["x_0"] - df["x_1"]
    dy = df["y_0"] - df["y_1"]
    dz = df["z_0"] - df["z_1"]
    df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz)

    return df


train_f = add_pair_features(train)
test_f = add_pair_features(test)

all_types = pd.concat([train_f["type"], test_f["type"]], axis=0).astype("category")
type_map = {k: i for i, k in enumerate(all_types.cat.categories)}
atom0_all = pd.concat([train_f["atom_0"], test_f["atom_0"]], axis=0).astype("category")
atom1_all = pd.concat([train_f["atom_1"], test_f["atom_1"]], axis=0).astype("category")
atom0_map = {k: i for i, k in enumerate(atom0_all.cat.categories)}
atom1_map = {k: i for i, k in enumerate(atom1_all.cat.categories)}

for df in (train_f, test_f):
    df["type_id"] = df["type"].map(type_map).astype(np.int16)
    df["atom0_id"] = df["atom_0"].map(atom0_map).astype(np.int8)
    df["atom1_id"] = df["atom_1"].map(atom1_map).astype(np.int8)

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

X = train_f[feature_cols].to_numpy(dtype=np.float32)
y = train_f[TARGET].to_numpy(dtype=np.float32)
X_test = test_f[feature_cols].to_numpy(dtype=np.float32)
groups = train_f["molecule_name"].values
types = train_f["type"].values

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

train_sets = []
test_sets = []

for name, model in models:
    oof = np.zeros(len(train_f), dtype=np.float32)
    test_pred = np.zeros(len(test_f), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        model.fit(X_tr, y_tr)
        oof[va_idx] = model.predict(X_va).astype(np.float32)
        test_pred += model.predict(X_test).astype(np.float32) / gkf.n_splits

    fold_score = group_mean_log_mae(pd.Series(y), pd.Series(oof), pd.Series(types))
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
IntCastingNaNError                        Traceback (most recent call last)
/tmp/ipykernel_11/1675749825.py in <cell line: 0>()
     96 for df in (train_f, test_f):
     97     df["type_id"] = df["type"].map(type_map).astype(np.int16)
---> 98     df["atom0_id"] = df["atom_0"].map(atom0_map).astype(np.int8)
     99     df["atom1_id"] = df["atom_1"].map(atom1_map).astype(np.int8)
    100 

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in astype(self, dtype, copy, errors)
   6641         else:
   6642             # else, only a single dtype is given
-> 6643             new_data = self._mgr.astype(dtype=dtype, copy=copy, errors=errors)
   6644             res = self._constructor_from_mgr(new_data, axes=new_data.axes)
   6645             return res.__finalize__(self, method="astype")

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in astype(self, dtype, copy, errors)
    428             copy = False
    429 
--> 430         return self.apply(
    431             "astype",
    432             dtype=dtype,

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in apply(self, f, align_keys, **kwargs)
    361                 applied = b.apply(f, **kwargs)
    362             else:
--> 363                 applied = getattr(b, f)(**kwargs)
    364             result_blocks = extend_blocks(applied, result_blocks)
    365 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in astype(self, dtype, copy, errors, using_cow, squeeze)
    756             values = values[0, :]  # type: ignore[call-overload]
    757 
--> 758         new_values = astype_array_safe(values, dtype, copy=copy, errors=errors)
    759 
    760         new_values = maybe_coerce_values(new_values)

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array_safe(values, dtype, copy, errors)
    235 
    236     try:
--> 237         new_values = astype_array(values, dtype, copy=copy)
    238     except (ValueError, TypeError):
    239         # e.g. _astype_nansafe can fail on object-dtype of strings

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array(values, dtype, copy)
    180 
    181     else:
--> 182         values = _astype_nansafe(values, dtype, copy=copy)
    183 
    184     # in pandas we don't store numpy str dtypes, so convert to object

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in _astype_nansafe(arr, dtype, copy, skipna)
     99 
    100     elif np.issubdtype(arr.dtype, np.floating) and dtype.kind in "iu":
--> 101         return _astype_float_to_int_nansafe(arr, dtype, copy)
    102 
    103     elif arr.dtype == object:

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in _astype_float_to_int_nansafe(values, dtype, copy)
    143     """
    144     if not np.isfinite(values).all():
--> 145         raise IntCastingNaNError(
    146             "Cannot convert non-finite values (NA or inf) to integer"
    147         )

IntCastingNaNError: Cannot convert non-finite values (NA or inf) to integer

## === cell 5
print(f"Train sets ready: {[df.shape for df in train_sets]}")



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4035745354.py in <cell line: 0>()
      1 # Keep this cell to preserve original structure; train_sets already built above.
      2 # (No-op, but ensures train_sets exists even if user executes cells sequentially.)
----> 3 print(f"Train sets ready: {[df.shape for df in train_sets]}")
      4 

NameError: name 'train_sets' is not defined

## === cell 6
print(train_sets[0].head())
print(test_sets[0].head())



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/13641062.py in <cell line: 0>()
      1 # (Optional) Show a small preview to confirm expected columns exist.
----> 2 print(train_sets[0].head())
      3 print(test_sets[0].head())
      4 

NameError: name 'train_sets' is not defined

## === cell 7
import time


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


def trial(train_sets, prediction_column, target_column):
    ws = weights(len(train_sets))
    df = train_sets[0].copy()
    df[prediction_column] = 0.0
    for i, t in enumerate(train_sets):
        df[prediction_column] += t[prediction_column].astype(np.float64) * ws[i]
    score = group_mean_log_mae(df[target_column], df[prediction_column], df["type"])
    return score, ws


t0 = time.time()
best = sys.maxsize
best_weights = []
for i in range(TRIALS):
    score, ws = trial(
        train_sets=train_sets, prediction_column=PREDICTION, target_column=TARGET
    )
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
/tmp/ipykernel_11/1671269749.py in <cell line: 0>()
     30 for i in range(TRIALS):
     31     score, ws = trial(
---> 32         train_sets=train_sets, prediction_column=PREDICTION, target_column=TARGET
     33     )
     34     if score < best:

NameError: name 'train_sets' is not defined

## === cell 8
submission = test_sets[0].copy()
submission[TARGET] = 0.0
for i, t in enumerate(test_sets):
    submission[TARGET] += t[TARGET].astype(np.float64) * best_weights[i]

submission = submission[["id", TARGET]].copy()
submission["id"] = submission["id"].astype(np.int64)
submission[TARGET] = submission[TARGET].astype(np.float64)

print(submission.head())
print(submission.shape)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/993641818.py in <cell line: 0>()
----> 1 submission = test_sets[0].copy()
      2 submission[TARGET] = 0.0
      3 for i, t in enumerate(test_sets):
      4     submission[TARGET] += t[TARGET].astype(np.float64) * best_weights[i]
      5 

NameError: name 'test_sets' is not defined

## === cell 9
out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print("Files in working dir:", os.listdir("."))

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2708350118.py in <cell line: 0>()
      1 # Write a valid Kaggle submission with .csv suffix.
      2 out_path = "submission.csv"
----> 3 submission.to_csv(out_path, index=False)
      4 print("Wrote:", out_path)
      5 print("Files in working dir:", os.listdir("."))

NameError: name 'submission' is not defined
