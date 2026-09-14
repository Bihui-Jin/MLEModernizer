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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
seaborn==0.12.2
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

-2.4226869287246378

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

DATA_DIR_CANDIDATES = [
    "/kaggle/input/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling",
    "../input/champs-scalar-coupling",
    "/kaggle/input/champs-scalar-coupling/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling/champs-scalar-coupling",
    "../input/champs-scalar-coupling/champs-scalar-coupling",
]


def pick_data_dir(cands):
    for d in cands:
        if os.path.exists(os.path.join(d, "train.csv")) and os.path.exists(
            os.path.join(d, "test.csv")
        ):
            return d
    raise FileNotFoundError(
        "Could not locate champs-scalar-coupling data directory in known candidates."
    )


DATA_DIR = pick_data_dir(DATA_DIR_CANDIDATES)
DATA_DIR



## === cell 1
train = pd.read_csv(
    os.path.join(DATA_DIR, "train.csv"),
    usecols=[
        "id",
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "scalar_coupling_constant",
    ],
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
        "scalar_coupling_constant": np.float32,
    },
)
test = pd.read_csv(
    os.path.join(DATA_DIR, "test.csv"),
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
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
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
    dtype={
        "molecule_name": "category",
        "atom_index": np.int16,
        "atom": "category",
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
    },
)


def safe_read_csv(path, **kwargs):
    return pd.read_csv(path, **kwargs) if os.path.exists(path) else None


dipole = safe_read_csv(
    os.path.join(DATA_DIR, "dipole_moments.csv"),
    usecols=["molecule_name", "X", "Y", "Z"],
    dtype={
        "molecule_name": "category",
        "X": np.float32,
        "Y": np.float32,
        "Z": np.float32,
    },
)
pot = safe_read_csv(
    os.path.join(DATA_DIR, "potential_energy.csv"),
    usecols=["molecule_name", "potential_energy"],
    dtype={"molecule_name": "category", "potential_energy": np.float32},
)
mull = safe_read_csv(
    os.path.join(DATA_DIR, "mulliken_charges.csv"),
    usecols=["molecule_name", "atom_index", "mulliken_charge"],
    dtype={
        "molecule_name": "category",
        "atom_index": np.int16,
        "mulliken_charge": np.float32,
    },
)

(
    train.shape,
    test.shape,
    structures.shape,
    None if dipole is None else dipole.shape,
    None if pot is None else pot.shape,
    None if mull is None else mull.shape,
)



## === cell 2
ATOM_MAP = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}

structures = structures.copy()
structures["atom_num"] = structures["atom"].map(ATOM_MAP).fillna(0).astype(np.int16)

mol_agg = (
    structures.groupby("molecule_name", sort=False)
    .agg(
        n_atoms=("atom_index", "count"),
        x_mean=("x", "mean"),
        y_mean=("y", "mean"),
        z_mean=("z", "mean"),
        x_std=("x", "std"),
        y_std=("y", "std"),
        z_std=("z", "std"),
        atom_num_mean=("atom_num", "mean"),
        atom_num_std=("atom_num", "std"),
    )
    .reset_index()
)

for c in ["x_std", "y_std", "z_std", "atom_num_std"]:
    mol_agg[c] = mol_agg[c].fillna(0.0).astype(np.float32)
mol_agg["n_atoms"] = mol_agg["n_atoms"].astype(np.int16)
for c in ["x_mean", "y_mean", "z_mean", "atom_num_mean"]:
    mol_agg[c] = mol_agg[c].astype(np.float32)

if dipole is not None:
    dip = dipole.copy()
    dip["dipole_norm"] = np.sqrt(
        dip["X"] * dip["X"] + dip["Y"] * dip["Y"] + dip["Z"] * dip["Z"]
    ).astype(np.float32)
    mol_agg = mol_agg.merge(
        dip[["molecule_name", "X", "Y", "Z", "dipole_norm"]],
        on="molecule_name",
        how="left",
        copy=False,
    )
else:
    mol_agg["X"] = np.nan
    mol_agg["Y"] = np.nan
    mol_agg["Z"] = np.nan
    mol_agg["dipole_norm"] = np.nan

if pot is not None:
    mol_agg = mol_agg.merge(pot, on="molecule_name", how="left", copy=False)
else:
    mol_agg["potential_energy"] = np.nan

if mull is not None:
    mull_agg = (
        mull.groupby("molecule_name", sort=False)
        .agg(
            mull_mean=("mulliken_charge", "mean"),
            mull_std=("mulliken_charge", "std"),
            mull_min=("mulliken_charge", "min"),
            mull_max=("mulliken_charge", "max"),
        )
        .reset_index()
    )
    mull_agg["mull_std"] = mull_agg["mull_std"].fillna(0.0).astype(np.float32)
    for c in ["mull_mean", "mull_min", "mull_max"]:
        mull_agg[c] = mull_agg[c].astype(np.float32)
    mol_agg = mol_agg.merge(mull_agg, on="molecule_name", how="left", copy=False)
else:
    mol_agg["mull_mean"] = np.nan
    mol_agg["mull_std"] = np.nan
    mol_agg["mull_min"] = np.nan
    mol_agg["mull_max"] = np.nan


def add_pair_geometry(df, s_indexed):
    df = df.copy()
    a0 = (
        df[["molecule_name", "atom_index_0"]]
        .join(s_indexed, on=["molecule_name", "atom_index_0"])
        .rename(columns={"atom_num": "atom0_num", "x": "x0", "y": "y0", "z": "z0"})
    )
    a1 = (
        df[["molecule_name", "atom_index_1"]]
        .join(s_indexed, on=["molecule_name", "atom_index_1"])
        .rename(columns={"atom_num": "atom1_num", "x": "x1", "y": "y1", "z": "z1"})
    )

    df["atom0_num"] = a0["atom0_num"].astype(np.int16)
    df["atom1_num"] = a1["atom1_num"].astype(np.int16)

    dx = a0["x0"].to_numpy(dtype=np.float32, copy=False) - a1["x1"].to_numpy(
        dtype=np.float32, copy=False
    )
    dy = a0["y0"].to_numpy(dtype=np.float32, copy=False) - a1["y1"].to_numpy(
        dtype=np.float32, copy=False
    )
    dz = a0["z0"].to_numpy(dtype=np.float32, copy=False) - a1["z1"].to_numpy(
        dtype=np.float32, copy=False
    )

    dist2 = dx * dx + dy * dy + dz * dz
    df["dist2"] = dist2.astype(np.float32)
    df["dist"] = np.sqrt(dist2, dtype=np.float32).astype(np.float32)
    df["atom_num_sum"] = (
        df["atom0_num"].astype(np.int32) + df["atom1_num"].astype(np.int32)
    ).astype(np.float32)
    df["atom_num_diff"] = (
        df["atom0_num"].astype(np.int32) - df["atom1_num"].astype(np.int32)
    ).astype(np.float32)
    return df


s_indexed = (
    structures[["molecule_name", "atom_index", "atom_num", "x", "y", "z"]]
    .set_index(["molecule_name", "atom_index"])
    .sort_index()
)

train_full = train.merge(mol_agg, on="molecule_name", how="left", copy=False)
test_full = test.merge(mol_agg, on="molecule_name", how="left", copy=False)

train_full = add_pair_geometry(train_full, s_indexed)
test_full = add_pair_geometry(test_full, s_indexed)

type_union = pd.Categorical(
    pd.concat(
        [train_full["type"].astype(str), test_full["type"].astype(str)],
        ignore_index=True,
    )
)
type_categories = type_union.categories
train_full["type"] = pd.Categorical(
    train_full["type"].astype(str), categories=type_categories
)
test_full["type"] = pd.Categorical(
    test_full["type"].astype(str), categories=type_categories
)

train_full["type_code"] = train_full["type"].cat.codes.astype(np.int16)
test_full["type_code"] = test_full["type"].cat.codes.astype(np.int16)

FEATURES = [
    "type_code",
    "atom_index_0",
    "atom_index_1",
    "atom0_num",
    "atom1_num",
    "atom_num_sum",
    "atom_num_diff",
    "dist",
    "dist2",
    "n_atoms",
    "x_mean",
    "y_mean",
    "z_mean",
    "x_std",
    "y_std",
    "z_std",
    "atom_num_mean",
    "atom_num_std",
    "X",
    "Y",
    "Z",
    "dipole_norm",
    "potential_energy",
    "mull_mean",
    "mull_std",
    "mull_min",
    "mull_max",
]

train_medians = train_full[FEATURES].median(numeric_only=True)
train_full[FEATURES] = train_full[FEATURES].fillna(train_medians)
test_full[FEATURES] = test_full[FEATURES].fillna(train_medians)

n_nan_train = int(train_full[FEATURES].isna().sum().sum())
n_nan_test = int(test_full[FEATURES].isna().sum().sum())
assert n_nan_train == 0, f"NaNs remain in train features: {n_nan_train}"
assert n_nan_test == 0, f"NaNs remain in test features: {n_nan_test}"

train_full[FEATURES].head()



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1770189654.py in <cell line: 0>()
      2 
      3 structures = structures.copy()
----> 4 structures["atom_num"] = structures["atom"].map(ATOM_MAP).fillna(0).astype(np.int16)
      5 
      6 # Speed: avoid Python lambda in groupby aggregation; n_atoms = count per molecule is equivalent to max+1

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

TypeError: Cannot setitem on a Categorical with a new category (0), set the categories first

## === cell 3
from sklearn.ensemble import RandomForestRegressor

TARGET = "scalar_coupling_constant"

X_train_np = np.ascontiguousarray(
    train_full[FEATURES].to_numpy(dtype=np.float32, copy=False)
)
y_train_np = np.ascontiguousarray(
    train_full[TARGET].to_numpy(dtype=np.float32, copy=False)
)
X_test_np = np.ascontiguousarray(
    test_full[FEATURES].to_numpy(dtype=np.float32, copy=False)
)

train_type = train_full["type_code"].to_numpy(dtype=np.int16, copy=False)
test_type = test_full["type_code"].to_numpy(dtype=np.int16, copy=False)

pred_test = np.zeros(len(test_full), dtype=np.float32)

rf_params = dict(
    n_estimators=80,
    random_state=42,
    n_jobs=-1,
    max_depth=None,
    min_samples_leaf=2,
)

train_idx_by_type = {}
for tcode in np.unique(train_type):
    train_idx_by_type[int(tcode)] = np.flatnonzero(train_type == tcode)

test_idx_by_type = {}
for tcode in np.unique(test_type):
    test_idx_by_type[int(tcode)] = np.flatnonzero(test_type == tcode)

for tcode in sorted(train_idx_by_type.keys()):
    te_idx = test_idx_by_type.get(int(tcode))
    if te_idx is None or te_idx.size == 0:
        continue
    tr_idx = train_idx_by_type[int(tcode)]

    model = RandomForestRegressor(**rf_params)
    model.fit(X_train_np[tr_idx], y_train_np[tr_idx])
    pred_test[te_idx] = model.predict(X_test_np[te_idx]).astype(np.float32)

test_full["final_preds"] = pred_test
test_full[["id", "final_preds"]].head()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1607711116.py in <cell line: 0>()
      6 # Correctness: model sees identical numeric values/rows as before.
      7 X_train_np = np.ascontiguousarray(
----> 8     train_full[FEATURES].to_numpy(dtype=np.float32, copy=False)
      9 )
     10 y_train_np = np.ascontiguousarray(

NameError: name 'train_full' is not defined

## === cell 4
submission = pd.DataFrame(
    {
        "id": test_full["id"].astype(np.int64).to_numpy(copy=False),
        "scalar_coupling_constant": test_full["final_preds"]
        .astype(np.float32)
        .to_numpy(copy=False),
    }
)

submission = submission.sort_values("id").reset_index(drop=True)

submission.to_csv("ensemble_sub.csv", index=False)

assert submission.shape[0] == test.shape[0]
assert list(submission.columns) == ["id", "scalar_coupling_constant"]
submission.head()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3882616355.py in <cell line: 0>()
      1 submission = pd.DataFrame(
      2     {
----> 3         "id": test_full["id"].astype(np.int64).to_numpy(copy=False),
      4         "scalar_coupling_constant": test_full["final_preds"]
      5         .astype(np.float32)

NameError: name 'test_full' is not defined

## === cell 5
submission["scalar_coupling_constant"].describe()

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1761324384.py in <cell line: 0>()
----> 1 submission["scalar_coupling_constant"].describe()

NameError: name 'submission' is not defined
