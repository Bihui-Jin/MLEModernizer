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

_DEFAULT_THREADS = str(min(8, (os.cpu_count() or 2)))
os.environ.setdefault("OMP_NUM_THREADS", _DEFAULT_THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _DEFAULT_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _DEFAULT_THREADS)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", _DEFAULT_THREADS)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _DEFAULT_THREADS)

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
    for d in [
        "/kaggle/data/champs-scalar-coupling",
        "/kaggle/data/champs-scalar-coupling/champs-scalar-coupling",
    ]:
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

atom_str = structures["atom"].astype("string")
structures["atom_num"] = (
    atom_str.map(ATOM_MAP).astype("float32").fillna(0.0).astype(np.int16)
)

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
    dx = dipole["X"].to_numpy(np.float32, copy=False)
    dy = dipole["Y"].to_numpy(np.float32, copy=False)
    dz = dipole["Z"].to_numpy(np.float32, copy=False)
    dip2 = dipole[["molecule_name", "X", "Y", "Z"]].copy()
    dip2["dipole_norm"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(
        np.float32, copy=False
    )
    mol_agg = mol_agg.merge(dip2, on="molecule_name", how="left", copy=False)
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


def build_struct_row_map(structures_df):
    mol_cats = structures_df["molecule_name"].cat.categories
    mol_codes = structures_df["molecule_name"].cat.codes.to_numpy(np.int32, copy=False)
    atom_idx = (
        structures_df["atom_index"]
        .to_numpy(np.int16, copy=False)
        .astype(np.int32, copy=False)
    )

    n_mols = len(mol_cats)
    max_atom = int(atom_idx.max(initial=0))
    row_map = np.full((n_mols, max_atom + 1), -1, dtype=np.int32)

    rows = np.arange(len(structures_df), dtype=np.int32)
    row_map[mol_codes, atom_idx] = rows
    return mol_cats, row_map


def add_pair_geometry_via_rowmap(df, mol_categories, row_map, arrs):
    if not pd.api.types.is_categorical_dtype(df["molecule_name"]):
        df["molecule_name"] = pd.Categorical(
            df["molecule_name"], categories=mol_categories
        )

    mol_codes = (
        df["molecule_name"]
        .cat.set_categories(mol_categories)
        .cat.codes.to_numpy(np.int32, copy=False)
    )
    ai0 = df["atom_index_0"].to_numpy(np.int16, copy=False).astype(np.int32, copy=False)
    ai1 = df["atom_index_1"].to_numpy(np.int16, copy=False).astype(np.int32, copy=False)

    max_atom = row_map.shape[1] - 1
    ai0c = np.clip(ai0, 0, max_atom)
    ai1c = np.clip(ai1, 0, max_atom)

    r0 = row_map[mol_codes, ai0c]
    r1 = row_map[mol_codes, ai1c]

    def take_with_fill(a, idx, fill_value, dtype):
        out = np.empty(idx.shape[0], dtype=dtype)
        m = idx >= 0
        out[~m] = fill_value
        out[m] = a[idx[m]]
        return out

    atom0_num = take_with_fill(arrs["atom_num"], r0, 0, np.int16)
    atom1_num = take_with_fill(arrs["atom_num"], r1, 0, np.int16)

    x0 = take_with_fill(arrs["x"], r0, np.nan, np.float32)
    y0 = take_with_fill(arrs["y"], r0, np.nan, np.float32)
    z0 = take_with_fill(arrs["z"], r0, np.nan, np.float32)
    x1 = take_with_fill(arrs["x"], r1, np.nan, np.float32)
    y1 = take_with_fill(arrs["y"], r1, np.nan, np.float32)
    z1 = take_with_fill(arrs["z"], r1, np.nan, np.float32)

    dx = np.nan_to_num(x0 - x1, nan=0.0, posinf=0.0, neginf=0.0)
    dy = np.nan_to_num(y0 - y1, nan=0.0, posinf=0.0, neginf=0.0)
    dz = np.nan_to_num(z0 - z1, nan=0.0, posinf=0.0, neginf=0.0)

    dist2 = dx * dx + dy * dy + dz * dz
    dist = np.sqrt(dist2, dtype=np.float32)

    atom_num_sum = (atom0_num.astype(np.int32) + atom1_num.astype(np.int32)).astype(
        np.float32
    )
    atom_num_diff = (atom0_num.astype(np.int32) - atom1_num.astype(np.int32)).astype(
        np.float32
    )

    return (
        atom0_num,
        atom1_num,
        dist.astype(np.float32, copy=False),
        dist2.astype(np.float32, copy=False),
        atom_num_sum,
        atom_num_diff,
    )


mol_categories, row_map = build_struct_row_map(structures)

arrs = {
    "atom_num": structures["atom_num"].to_numpy(np.int16, copy=False),
    "x": structures["x"].to_numpy(np.float32, copy=False),
    "y": structures["y"].to_numpy(np.float32, copy=False),
    "z": structures["z"].to_numpy(np.float32, copy=False),
}

mol_agg = mol_agg.set_index("molecule_name", drop=True)

train["molecule_name"] = pd.Categorical(
    train["molecule_name"], categories=mol_categories
)
test["molecule_name"] = pd.Categorical(test["molecule_name"], categories=mol_categories)

train_type_cat = train["type"]
test_type_cat = test["type"]
type_categories = train_type_cat.cat.categories.union(test_type_cat.cat.categories)
train["type"] = train_type_cat.cat.set_categories(type_categories)
test["type"] = test_type_cat.cat.set_categories(type_categories)

train_type_code = train["type"].cat.codes.astype(np.int16)
test_type_code = test["type"].cat.codes.astype(np.int16)

mol_train = mol_agg.reindex(train["molecule_name"]).to_numpy(
    dtype=np.float32, copy=True
)
mol_test = mol_agg.reindex(test["molecule_name"]).to_numpy(dtype=np.float32, copy=True)

(
    atom0_num_tr,
    atom1_num_tr,
    dist_tr,
    dist2_tr,
    atom_num_sum_tr,
    atom_num_diff_tr,
) = add_pair_geometry_via_rowmap(train, mol_categories, row_map, arrs)

(
    atom0_num_te,
    atom1_num_te,
    dist_te,
    dist2_te,
    atom_num_sum_te,
    atom_num_diff_te,
) = add_pair_geometry_via_rowmap(test, mol_categories, row_map, arrs)

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


def build_X(
    df,
    type_code,
    atom0_num,
    atom1_num,
    atom_num_sum,
    atom_num_diff,
    dist,
    dist2,
    mol_block,
):
    n = len(df)
    X = np.empty((n, 27), dtype=np.float32)
    X[:, 0] = type_code.astype(np.float32, copy=False)
    X[:, 1] = df["atom_index_0"].to_numpy(np.float32, copy=False)
    X[:, 2] = df["atom_index_1"].to_numpy(np.float32, copy=False)
    X[:, 3] = atom0_num.astype(np.float32, copy=False)
    X[:, 4] = atom1_num.astype(np.float32, copy=False)
    X[:, 5] = atom_num_sum
    X[:, 6] = atom_num_diff
    X[:, 7] = dist
    X[:, 8] = dist2
    X[:, 9:] = mol_block  # remaining 18 cols
    return X


X_train_np = build_X(
    train,
    train_type_code.to_numpy(copy=False),
    atom0_num_tr,
    atom1_num_tr,
    atom_num_sum_tr,
    atom_num_diff_tr,
    dist_tr,
    dist2_tr,
    mol_train,
)
X_test_np = build_X(
    test,
    test_type_code.to_numpy(copy=False),
    atom0_num_te,
    atom1_num_te,
    atom_num_sum_te,
    atom_num_diff_te,
    dist_te,
    dist2_te,
    mol_test,
)

col_medians = np.nanmedian(X_train_np, axis=0).astype(np.float32, copy=False)


def fill_nan_inplace(X, medians):
    m = ~np.isfinite(X)
    if m.any():
        X[m] = np.take(medians, np.nonzero(m)[1])


fill_nan_inplace(X_train_np, col_medians)
fill_nan_inplace(X_test_np, col_medians)

assert not np.isnan(X_train_np).any()
assert not np.isnan(X_test_np).any()

X_train_np[:2], X_test_np[:2]



## === cell 3
from sklearn.ensemble import RandomForestRegressor

TARGET = "scalar_coupling_constant"

X_train_np = np.ascontiguousarray(X_train_np)
y_train_np = np.ascontiguousarray(train[TARGET].to_numpy(dtype=np.float32, copy=False))
X_test_np = np.ascontiguousarray(X_test_np)

train_type = train_type_code.to_numpy(dtype=np.int16, copy=False)
test_type = test_type_code.to_numpy(dtype=np.int16, copy=False)

pred_test = np.zeros(len(test), dtype=np.float32)

rf_params = dict(
    n_estimators=80,
    random_state=42,
    n_jobs=-1,
    max_depth=None,
    min_samples_leaf=2,
)

tr_order = np.argsort(train_type, kind="mergesort")  # deterministic
te_order = np.argsort(test_type, kind="mergesort")  # deterministic

train_type_sorted = train_type[tr_order]
test_type_sorted = test_type[te_order]

tr_uniq, tr_starts = np.unique(train_type_sorted, return_index=True)
te_uniq, te_starts = np.unique(test_type_sorted, return_index=True)

tr_ends = np.empty_like(tr_starts)
tr_ends[:-1] = tr_starts[1:]
tr_ends[-1] = len(tr_order)

te_ends = np.empty_like(te_starts)
te_ends[:-1] = te_starts[1:]
te_ends[-1] = len(te_order)

te_start_map = {
    int(code): (int(s), int(e)) for code, s, e in zip(te_uniq, te_starts, te_ends)
}

Xtr_sorted = X_train_np[tr_order]
ytr_sorted = y_train_np[tr_order]
Xte_sorted = X_test_np[te_order]

for i, tcode in enumerate(tr_uniq.tolist()):
    te_se = te_start_map.get(int(tcode))
    if te_se is None:
        continue

    tr_s = int(tr_starts[i])
    tr_e = int(tr_ends[i])
    te_s, te_e = te_se

    model = RandomForestRegressor(**rf_params)
    model.fit(Xtr_sorted[tr_s:tr_e], ytr_sorted[tr_s:tr_e])
    pred_test_sorted = model.predict(Xte_sorted[te_s:te_e]).astype(
        np.float32, copy=False
    )
    pred_test[te_order[te_s:te_e]] = pred_test_sorted

pred_test[:5]



## === cell 4
submission = pd.DataFrame(
    {
        "id": test["id"].astype(np.int64).to_numpy(copy=False),
        "scalar_coupling_constant": pred_test.astype(np.float32, copy=False),
    }
)

submission = submission.sort_values("id").reset_index(drop=True)
submission.to_csv("ensemble_sub.csv", index=False)

assert submission.shape[0] == test.shape[0]
assert list(submission.columns) == ["id", "scalar_coupling_constant"]
submission.head()



## === cell 5
submission["scalar_coupling_constant"].describe()
