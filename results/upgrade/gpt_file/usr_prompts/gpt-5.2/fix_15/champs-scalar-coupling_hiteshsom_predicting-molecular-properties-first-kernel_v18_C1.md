# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

category_encoders==2.7.0
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

# 5. Code solution

## === cell 0
import numpy as np
import pandas as pd
import gc
import math
import category_encoders as ce
import lightgbm as lgbm
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_absolute_error as mae
import os

np.random.seed(10)

print(os.listdir("../input"))



## === cell 1
gc.collect()



## === cell 2
DATA_DIR = "../input/champs-scalar-coupling"

train = pd.read_csv(
    f"{DATA_DIR}/train.csv",
    dtype={
        "id": "int32",
        "molecule_name": "object",
        "atom_index_0": "int16",
        "atom_index_1": "int16",
        "type": "object",
        "scalar_coupling_constant": "float32",
    },
)
test = pd.read_csv(
    f"{DATA_DIR}/test.csv",
    dtype={
        "id": "int32",
        "molecule_name": "object",
        "atom_index_0": "int16",
        "atom_index_1": "int16",
        "type": "object",
    },
)
sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv", dtype={"id": "int32"})
structures = pd.read_csv(
    f"{DATA_DIR}/structures.csv",
    dtype={
        "molecule_name": "object",
        "atom_index": "int16",
        "atom": "object",
        "x": "float32",
        "y": "float32",
        "z": "float32",
    },
)

print(f"train.shape: {train.shape}")
print(f"test.shape: {test.shape}")
print(f"structures.shape: {structures.shape}")



## === cell 3
X_train = train.drop(columns=["scalar_coupling_constant"])
y_train = train["scalar_coupling_constant"]
X_test = test

print(f"X_train.shape: {X_train.shape}")
print(f"X_test.shape: {X_test.shape}")



## === cell 4
train_id = X_train["id"].copy()
test_id = X_test["id"].copy()

X_train = X_train.drop(columns=["id"])
X_test = X_test.drop(columns=["id"])



## === cell 5
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
X_train.insert(0, "index", np.arange(X_train.shape[0], dtype=np.int32))
X_test.insert(0, "index", np.arange(X_test.shape[0], dtype=np.int32))




## === cell 6
def convert_object_to_categories_fit_on_concat(X_train_in, X_test_in):
    X_train = X_train_in.copy()
    X_test = X_test_in.copy()
    obj_cols = [
        c
        for c in X_train.columns
        if (X_train[c].dtype == "O" or str(X_train[c].dtype) == "object")
    ]
    for col in obj_cols:
        combined = pd.concat(
            [X_train[col].astype("string"), X_test[col].astype("string")],
            axis=0,
            ignore_index=True,
        )
        cats = pd.Categorical(combined).categories
        X_train[col] = pd.Categorical(X_train[col].astype("string"), categories=cats)
        X_test[col] = pd.Categorical(X_test[col].astype("string"), categories=cats)
    return X_train, X_test




## === cell 7
print(f"{X_train['type'].unique()}")
print(f"{X_test['type'].unique()}")


def calc_score(X_fold, y_true, y_pred):
    types = X_fold["type"].astype("string").to_numpy()
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = np.abs(y_true - y_pred)

    df = pd.DataFrame({"type": types, "err": err})
    g = df.groupby("type")["err"].mean()
    return float(np.log(g).mean())




## === cell 8
def cross_val_grouped_by_molecule(X, y, groups):
    print(X.shape)
    gkf = GroupKFold(n_splits=5)
    fold = 0
    for train_index, val_index in gkf.split(X, y, groups=groups):
        fold += 1
        lgbm_model = lgbm.LGBMRegressor(
            n_estimators=600,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.2,
            min_child_samples=50,
            random_state=10,
            n_jobs=-1,
            objective="regression_l1",
        )
        lgbm_model.fit(X.loc[train_index, :], y.iloc[train_index])
        y_val = lgbm_model.predict(X.loc[val_index, :])
        print(
            f"fold{fold} score: {calc_score(X.loc[val_index, :], y.iloc[val_index], y_val)}"
        )




## === cell 9
num_atoms_df = (
    structures.groupby("molecule_name")["atom_index"]
    .max()
    .add(1)
    .rename("num_atoms")
    .reset_index()
)



## === cell 10
structures_small = structures[
    ["molecule_name", "atom_index", "atom", "x", "y", "z"]
].copy()
structures_small = structures_small.set_index(
    ["molecule_name", "atom_index"], drop=False
)


def join_structure_features_merge(df, atom_col, prefix_atom):
    s = structures_small.rename(
        columns={
            "atom_index": atom_col,
            "atom": f"atom_{prefix_atom}",
            "x": f"atom_index_{prefix_atom}_x",
            "y": f"atom_index_{prefix_atom}_y",
            "z": f"atom_index_{prefix_atom}_z",
        }
    )
    out = df.join(
        s.set_index(["molecule_name", atom_col])[
            [
                f"atom_{prefix_atom}",
                f"atom_index_{prefix_atom}_x",
                f"atom_index_{prefix_atom}_y",
                f"atom_index_{prefix_atom}_z",
            ]
        ],
        on=["molecule_name", atom_col],
        how="left",
    )
    return out


X_train = join_structure_features_merge(X_train, "atom_index_0", "0")
X_test = join_structure_features_merge(X_test, "atom_index_0", "0")

assert (
    X_train.shape[0] == train.shape[0]
), "Row count changed for train after atom_0 join."
assert X_test.shape[0] == test.shape[0], "Row count changed for test after atom_0 join."



## === cell 11
X_train = join_structure_features_merge(X_train, "atom_index_1", "1")
X_test = join_structure_features_merge(X_test, "atom_index_1", "1")

assert (
    X_train.shape[0] == train.shape[0]
), "Row count changed for train after atom_1 join."
assert X_test.shape[0] == test.shape[0], "Row count changed for test after atom_1 join."



## === cell 12
if "atom_index_0_0" in X_train.columns:
    X_train = X_train.drop(columns=["atom_index_0_0"])
if "atom_index_1_1" in X_train.columns:
    X_train = X_train.drop(columns=["atom_index_1_1"])
if "atom_index_0_0" in X_test.columns:
    X_test = X_test.drop(columns=["atom_index_0_0"])
if "atom_index_1_1" in X_test.columns:
    X_test = X_test.drop(columns=["atom_index_1_1"])



## === cell 13
dx_tr = X_train["atom_index_0_x"].to_numpy(dtype=np.float32) - X_train[
    "atom_index_1_x"
].to_numpy(dtype=np.float32)
dy_tr = X_train["atom_index_0_y"].to_numpy(dtype=np.float32) - X_train[
    "atom_index_1_y"
].to_numpy(dtype=np.float32)
dz_tr = X_train["atom_index_0_z"].to_numpy(dtype=np.float32) - X_train[
    "atom_index_1_z"
].to_numpy(dtype=np.float32)
X_train["distance"] = np.sqrt(dx_tr * dx_tr + dy_tr * dy_tr + dz_tr * dz_tr).astype(
    np.float32
)

dx_te = X_test["atom_index_0_x"].to_numpy(dtype=np.float32) - X_test[
    "atom_index_1_x"
].to_numpy(dtype=np.float32)
dy_te = X_test["atom_index_0_y"].to_numpy(dtype=np.float32) - X_test[
    "atom_index_1_y"
].to_numpy(dtype=np.float32)
dz_te = X_test["atom_index_0_z"].to_numpy(dtype=np.float32) - X_test[
    "atom_index_1_z"
].to_numpy(dtype=np.float32)
X_test["distance"] = np.sqrt(dx_te * dx_te + dy_te * dy_te + dz_te * dz_te).astype(
    np.float32
)

eps = np.float32(1e-9)
X_train["inv_distance"] = (
    1.0 / (X_train["distance"].to_numpy(dtype=np.float32) + eps)
).astype(np.float32)
X_test["inv_distance"] = (
    1.0 / (X_test["distance"].to_numpy(dtype=np.float32) + eps)
).astype(np.float32)



## === cell 14
X_train["join_type"] = X_train["type"].astype("string").str.slice(0, 2)
X_test["join_type"] = X_test["type"].astype("string").str.slice(0, 2)



## === cell 15
print(X_train["atom_0"].unique())
print(X_test["atom_0"].unique())



## === cell 16
num_atoms_df_idx = num_atoms_df.set_index("molecule_name")
X_train = X_train.join(num_atoms_df_idx, on="molecule_name", how="left")
X_test = X_test.join(num_atoms_df_idx, on="molecule_name", how="left")



## === cell 17
contrib = pd.read_csv(
    f"{DATA_DIR}/scalar_coupling_contributions.csv",
    usecols=[
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "fc",
        "sd",
        "pso",
        "dso",
    ],
    dtype={
        "molecule_name": "object",
        "atom_index_0": "int16",
        "atom_index_1": "int16",
        "type": "object",
        "fc": "float32",
        "sd": "float32",
        "pso": "float32",
        "dso": "float32",
    },
)
contrib_idx = contrib.set_index(
    ["molecule_name", "atom_index_0", "atom_index_1", "type"]
)

X_train = X_train.join(
    contrib_idx,
    on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
    how="left",
)
X_test = X_test.join(
    contrib_idx,
    on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
    how="left",
)

for c in ["fc", "sd", "pso", "dso"]:
    X_train[c] = X_train[c].fillna(np.float32(0.0))
    X_test[c] = X_test[c].fillna(np.float32(0.0))

assert (
    X_train.shape[0] == train.shape[0]
), "Row count changed for train after contributions merge."
assert (
    X_test.shape[0] == test.shape[0]
), "Row count changed for test after contributions merge."

del contrib, contrib_idx
gc.collect()



## === cell 18
mulliken = pd.read_csv(
    f"{DATA_DIR}/mulliken_charges.csv",
    usecols=["molecule_name", "atom_index", "mulliken_charge"],
    dtype={
        "molecule_name": "object",
        "atom_index": "int16",
        "mulliken_charge": "float32",
    },
)
mst = pd.read_csv(
    f"{DATA_DIR}/magnetic_shielding_tensors.csv",
    usecols=[
        "molecule_name",
        "atom_index",
        "XX",
        "YX",
        "ZX",
        "XY",
        "YY",
        "ZY",
        "XZ",
        "YZ",
        "ZZ",
    ],
    dtype={
        "molecule_name": "object",
        "atom_index": "int16",
        "XX": "float32",
        "YX": "float32",
        "ZX": "float32",
        "XY": "float32",
        "YY": "float32",
        "ZY": "float32",
        "XZ": "float32",
        "YZ": "float32",
        "ZZ": "float32",
    },
)
dipole = pd.read_csv(
    f"{DATA_DIR}/dipole_moments.csv",
    dtype={"molecule_name": "object", "X": "float32", "Y": "float32", "Z": "float32"},
)
pot = pd.read_csv(
    f"{DATA_DIR}/potential_energy.csv",
    dtype={"molecule_name": "object", "potential_energy": "float32"},
)

mull_idx = mulliken.set_index(["molecule_name", "atom_index"])["mulliken_charge"]
mst_idx = mst.set_index(["molecule_name", "atom_index"])

X_train["mulliken_charge_0"] = mull_idx.reindex(
    pd.MultiIndex.from_arrays(
        [X_train["molecule_name"].values, X_train["atom_index_0"].values]
    )
).to_numpy(dtype=np.float32)
X_train["mulliken_charge_1"] = mull_idx.reindex(
    pd.MultiIndex.from_arrays(
        [X_train["molecule_name"].values, X_train["atom_index_1"].values]
    )
).to_numpy(dtype=np.float32)

X_test["mulliken_charge_0"] = mull_idx.reindex(
    pd.MultiIndex.from_arrays(
        [X_test["molecule_name"].values, X_test["atom_index_0"].values]
    )
).to_numpy(dtype=np.float32)
X_test["mulliken_charge_1"] = mull_idx.reindex(
    pd.MultiIndex.from_arrays(
        [X_test["molecule_name"].values, X_test["atom_index_1"].values]
    )
).to_numpy(dtype=np.float32)

mst_cols = ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]


def _reindex_mst(df, atom_index_col, prefix):
    mi = pd.MultiIndex.from_arrays(
        [df["molecule_name"].values, df[atom_index_col].values]
    )
    vals = mst_idx.reindex(mi)[mst_cols]
    vals = vals.rename(columns={c: f"{prefix}_{c}" for c in mst_cols})
    return vals.reset_index(drop=True)


X_train = pd.concat(
    [
        X_train.reset_index(drop=True),
        _reindex_mst(X_train, "atom_index_0", "mst0"),
        _reindex_mst(X_train, "atom_index_1", "mst1"),
    ],
    axis=1,
)
X_test = pd.concat(
    [
        X_test.reset_index(drop=True),
        _reindex_mst(X_test, "atom_index_0", "mst0"),
        _reindex_mst(X_test, "atom_index_1", "mst1"),
    ],
    axis=1,
)

dipole_idx = dipole.set_index("molecule_name")
pot_idx = pot.set_index("molecule_name")

X_train = X_train.join(dipole_idx, on="molecule_name", how="left")
X_test = X_test.join(dipole_idx, on="molecule_name", how="left")

X_train = X_train.join(pot_idx, on="molecule_name", how="left")
X_test = X_test.join(pot_idx, on="molecule_name", how="left")

num_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
meds = X_train[num_cols].median(numeric_only=True)
meds = meds.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
X_train[num_cols] = X_train[num_cols].fillna(meds)
common_num_cols = [c for c in num_cols if c in X_test.columns]
X_test[common_num_cols] = X_test[common_num_cols].fillna(meds[common_num_cols])

assert (
    X_train.shape[0] == train.shape[0]
), "Row count changed for train after aux merges."
assert X_test.shape[0] == test.shape[0], "Row count changed for test after aux merges."

del mulliken, mst, dipole, pot, mull_idx, mst_idx, dipole_idx, pot_idx
gc.collect()



## === cell 19
X_train, X_test = convert_object_to_categories_fit_on_concat(X_train, X_test)



## === cell 20
X_train_new = X_train.set_index(keys="index", drop=True)
X_test_new = X_test.set_index(keys="index", drop=True)

assert (
    X_train_new.shape[0] == train.shape[0]
), "X_train_new row mismatch after indexing."
assert X_test_new.shape[0] == test.shape[0], "X_test_new row mismatch after indexing."



## === cell 21
print("Unique molecules in train:", X_train_new["molecule_name"].nunique())



## === cell 22
X_train_new["num_bonds"] = X_train_new["join_type"].astype("string").str.slice(0, 1)
X_test_new["num_bonds"] = X_test_new["join_type"].astype("string").str.slice(0, 1)

X_train_new["num_bonds"] = X_train_new["num_bonds"].astype("int")
X_test_new["num_bonds"] = X_test_new["num_bonds"].astype("int")




## === cell 23
def add_bond_direction_features_inplace(df: pd.DataFrame) -> pd.DataFrame:
    dx = df["atom_index_0_x"] - df["atom_index_1_x"]
    dy = df["atom_index_0_y"] - df["atom_index_1_y"]
    dz = df["atom_index_0_z"] - df["atom_index_1_z"]
    dist = df["distance"].replace(0, np.nan)
    df["dir_x"] = (dx / dist).fillna(0.0).astype(np.float32)
    df["dir_y"] = (dy / dist).fillna(0.0).astype(np.float32)
    df["dir_z"] = (dz / dist).fillna(0.0).astype(np.float32)
    return df


X_train_new = add_bond_direction_features_inplace(X_train_new)
X_test_new = add_bond_direction_features_inplace(X_test_new)

X_train_new["atom_index_diff"] = (
    (X_train_new["atom_index_0"] - X_train_new["atom_index_1"]).abs().astype("int16")
)
X_test_new["atom_index_diff"] = (
    (X_test_new["atom_index_0"] - X_test_new["atom_index_1"]).abs().astype("int16")
)




## === cell 24
def make_lgbm_compatible_concat(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train = train_df.copy()
    test = test_df.copy()

    cat_cols = []
    for col in train.columns:
        if (
            (str(train[col].dtype) == "string")
            or pd.api.types.is_categorical_dtype(train[col])
            or train[col].dtype == "O"
        ):
            cat_cols.append(col)

    if len(cat_cols):
        all_cat = pd.concat(
            [train[cat_cols], test[cat_cols]], axis=0, ignore_index=True
        )
        for c in cat_cols:
            if str(all_cat[c].dtype) == "string":
                all_cat[c] = all_cat[c].astype("category")
            elif not pd.api.types.is_categorical_dtype(all_cat[c]):
                all_cat[c] = all_cat[c].astype("category")
        ntr = train.shape[0]
        for c in cat_cols:
            codes = all_cat[c].cat.codes.astype("int32").to_numpy()
            train[c] = codes[:ntr]
            test[c] = codes[ntr:]

    return train, test, cat_cols


X_train_model = X_train_new.drop(columns=["molecule_name"])
X_test_model = X_test_new.drop(columns=["molecule_name"])

X_train_model = X_train_model.drop(columns=["atom_index_0", "atom_index_1"])
X_test_model = X_test_model.drop(columns=["atom_index_0", "atom_index_1"])

X_train_lgb, X_test_lgb, cat_cols = make_lgbm_compatible_concat(
    X_train_model, X_test_model
)

missing_in_test = [c for c in X_train_lgb.columns if c not in X_test_lgb.columns]
missing_in_train = [c for c in X_test_lgb.columns if c not in X_train_lgb.columns]
for c in missing_in_test:
    X_test_lgb[c] = 0
for c in missing_in_train:
    X_train_lgb[c] = 0
X_test_lgb = X_test_lgb[X_train_lgb.columns]

bad_cols = [
    c
    for c in X_train_lgb.columns
    if not (
        pd.api.types.is_numeric_dtype(X_train_lgb[c])
        or pd.api.types.is_bool_dtype(X_train_lgb[c])
    )
]
assert len(bad_cols) == 0, f"Non-numeric columns still present for LightGBM: {bad_cols}"

categorical_feature_names = [c for c in cat_cols if c in X_train_lgb.columns]




## === cell 25
def compute_bias_and_fold_models_per_type(
    X_lgb: pd.DataFrame,
    y: pd.Series,
    X_meta: pd.DataFrame,
    categorical_feature_names,
    n_splits=5,
    random_state=10,
):
    train_types = X_meta["type"].astype("string").to_numpy()
    molecules = X_meta["molecule_name"].astype("string").to_numpy()

    bias_by_type = {}
    fold_models_by_type = {}

    unique_types = sorted(pd.unique(train_types).tolist())
    for t in unique_types:
        pos = np.flatnonzero(train_types == t)
        if pos.size < n_splits:
            bias_by_type[t] = 0.0
            fold_models_by_type[t] = []
            continue

        X_t = X_lgb.iloc[pos, :]
        y_t = y.iloc[pos]
        g_t = molecules[pos]

        gkf = GroupKFold(n_splits=n_splits)
        oof_pred = np.zeros(X_t.shape[0], dtype=np.float64)
        models = []

        for tr_i, va_i in gkf.split(X_t, y_t, groups=g_t):
            model = lgbm.LGBMRegressor(
                n_estimators=600,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.2,
                min_child_samples=50,
                random_state=random_state,
                n_jobs=-1,
                objective="regression_l1",
            )
            model.fit(
                X_t.iloc[tr_i, :],
                y_t.iloc[tr_i],
                categorical_feature=(
                    categorical_feature_names
                    if len(categorical_feature_names)
                    else "auto"
                ),
            )
            oof_pred[va_i] = model.predict(X_t.iloc[va_i, :])
            models.append(model)

        residual = y_t.values.astype(np.float64) - oof_pred
        bias_by_type[t] = float(np.median(residual))
        fold_models_by_type[t] = models

    return bias_by_type, fold_models_by_type


bias_by_type, fold_models_by_type = compute_bias_and_fold_models_per_type(
    X_train_lgb,
    y_train,
    X_train_new,
    categorical_feature_names=categorical_feature_names,
    n_splits=5,
    random_state=10,
)
print("Computed per-type bias correction for types:", sorted(bias_by_type.keys()))



## === cell 26
y_predict = np.zeros(X_test_lgb.shape[0], dtype=np.float64)

train_types = X_train_new["type"].astype("string")
test_types_arr = X_test_new["type"].astype("string").to_numpy()

unique_types = sorted(train_types.unique().tolist())
print("Predicting per type (averaging fold models) for types:", unique_types)

for t in unique_types:
    te_pos = np.flatnonzero(test_types_arr == t)
    if te_pos.size == 0:
        continue

    models = fold_models_by_type.get(t, [])
    if len(models) == 0:
        y_predict[te_pos] = float(bias_by_type.get(t, 0.0))
        continue

    X_te_t = X_test_lgb.iloc[te_pos, :]
    pred_sum = np.zeros(X_te_t.shape[0], dtype=np.float64)
    for m in models:
        pred_sum += m.predict(X_te_t)
    preds = pred_sum / float(len(models))
    preds = preds + float(bias_by_type.get(t, 0.0))
    y_predict[te_pos] = preds

assert np.isfinite(y_predict).all(), "Non-finite predictions produced."



## === cell 27
submission = pd.DataFrame({"id": test_id.values, "scalar_coupling_constant": y_predict})

assert (
    submission.shape[0] == sample_sub.shape[0]
), "Submission row count mismatch vs sample_submission."
assert (
    submission["id"].iloc[0] == sample_sub["id"].iloc[0]
), "Submission id order mismatch vs sample_submission."

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
