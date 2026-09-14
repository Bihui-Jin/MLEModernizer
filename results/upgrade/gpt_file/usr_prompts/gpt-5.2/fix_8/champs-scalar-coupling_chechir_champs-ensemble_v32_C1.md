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

# 5. Code solution

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
atom_as_str = structures["atom"].astype(str)
structures["atom_num"] = atom_as_str.map(ATOM_MAP).fillna(0).astype(np.int16)

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

    idx0 = pd.MultiIndex.from_arrays(
        [
            df["molecule_name"].to_numpy(copy=False),
            df["atom_index_0"].to_numpy(copy=False),
        ],
        names=["molecule_name", "atom_index"],
    )
    idx1 = pd.MultiIndex.from_arrays(
        [
            df["molecule_name"].to_numpy(copy=False),
            df["atom_index_1"].to_numpy(copy=False),
        ],
        names=["molecule_name", "atom_index"],
    )

    a0 = s_indexed.reindex(idx0)
    a1 = s_indexed.reindex(idx1)

    atom0_num = a0["atom_num"].to_numpy(dtype=np.float32, copy=False)
    atom1_num = a1["atom_num"].to_numpy(dtype=np.float32, copy=False)
    df["atom0_num"] = np.nan_to_num(atom0_num, nan=0.0).astype(np.int16, copy=False)
    df["atom1_num"] = np.nan_to_num(atom1_num, nan=0.0).astype(np.int16, copy=False)

    x0 = a0["x"].to_numpy(dtype=np.float32, copy=False)
    y0 = a0["y"].to_numpy(dtype=np.float32, copy=False)
    z0 = a0["z"].to_numpy(dtype=np.float32, copy=False)
    x1 = a1["x"].to_numpy(dtype=np.float32, copy=False)
    y1 = a1["y"].to_numpy(dtype=np.float32, copy=False)
    z1 = a1["z"].to_numpy(dtype=np.float32, copy=False)

    dx = np.nan_to_num(x0 - x1, nan=0.0, posinf=0.0, neginf=0.0)
    dy = np.nan_to_num(y0 - y1, nan=0.0, posinf=0.0, neginf=0.0)
    dz = np.nan_to_num(z0 - z1, nan=0.0, posinf=0.0, neginf=0.0)

    dist2 = dx * dx + dy * dy + dz * dz
    df["dist2"] = dist2.astype(np.float32, copy=False)
    df["dist"] = np.sqrt(dist2).astype(np.float32, copy=False)

    a0i = df["atom0_num"].to_numpy(dtype=np.int32, copy=False)
    a1i = df["atom1_num"].to_numpy(dtype=np.int32, copy=False)
    df["atom_num_sum"] = (a0i + a1i).astype(np.float32, copy=False)
    df["atom_num_diff"] = (a0i - a1i).astype(np.float32, copy=False)
    return df


s_indexed = structures[
    ["molecule_name", "atom_index", "atom_num", "x", "y", "z"]
].set_index(["molecule_name", "atom_index"])

train_full = train.merge(mol_agg, on="molecule_name", how="left", copy=False)
test_full = test.merge(mol_agg, on="molecule_name", how="left", copy=False)

train_full = add_pair_geometry(train_full, s_indexed)
test_full = add_pair_geometry(test_full, s_indexed)

train_types = train_full["type"].astype(str)
test_types = test_full["type"].astype(str)
train_cat = pd.Categorical(train_types)
train_categories = list(train_cat.categories)
missing = pd.Index(pd.unique(test_types)).difference(train_categories)
type_categories = pd.Index(train_categories).append(missing)

train_full["type"] = pd.Categorical(train_types, categories=type_categories)
test_full["type"] = pd.Categorical(test_types, categories=type_categories)

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


def indices_by_code(codes: np.ndarray):
    order = np.argsort(codes, kind="mergesort")  # deterministic
    sorted_codes = codes[order]
    uniq, starts = np.unique(sorted_codes, return_index=True)
    idx = {}
    for i, code in enumerate(uniq):
        s = starts[i]
        e = starts[i + 1] if i + 1 < len(starts) else len(order)
        idx[int(code)] = order[s:e]
    return idx


train_idx_by_type = indices_by_code(train_type)
test_idx_by_type = indices_by_code(test_type)

for tcode in sorted(train_idx_by_type.keys()):
    te_idx = test_idx_by_type.get(int(tcode))
    if te_idx is None or te_idx.size == 0:
        continue
    tr_idx = train_idx_by_type[int(tcode)]

    model = RandomForestRegressor(**rf_params)
    model.fit(X_train_np[tr_idx], y_train_np[tr_idx])
    pred_test[te_idx] = model.predict(X_test_np[te_idx]).astype(np.float32, copy=False)

test_full["final_preds"] = pred_test
test_full[["id", "final_preds"]].head()



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



## === cell 5
submission["scalar_coupling_constant"].describe()
