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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

INPUT_DIR = "/kaggle/input/champs-scalar-coupling"
WORKING_DIR = "/kaggle/working"

print("Listing /kaggle/input (top):", os.listdir("/kaggle/input")[:10])
print("Using INPUT_DIR:", INPUT_DIR)
print("Files in INPUT_DIR (sample):", os.listdir(INPUT_DIR)[:10])

os.environ.setdefault("PYTHONHASHSEED", "42")
np.random.seed(42)

train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")
structures_path = os.path.join(INPUT_DIR, "structures.csv")
sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")

_read_csv_kwargs = {}
try:
    import pyarrow  # noqa: F401

    _read_csv_kwargs["engine"] = "pyarrow"
except Exception:
    pass

train = pd.read_csv(
    train_path,
    dtype={
        "id": np.int32,
        "molecule_name": "string",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "string",
        "scalar_coupling_constant": np.float32,
    },
    **_read_csv_kwargs,
)
test = pd.read_csv(
    test_path,
    dtype={
        "id": np.int32,
        "molecule_name": "string",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "string",
    },
    **_read_csv_kwargs,
)
structures = pd.read_csv(
    structures_path,
    dtype={
        "molecule_name": "string",
        "atom_index": np.int16,
        "atom": "string",
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
    },
    **_read_csv_kwargs,
)
sample_sub = pd.read_csv(
    sample_sub_path,
    dtype={"id": np.int32, "scalar_coupling_constant": np.float32},
    **_read_csv_kwargs,
)

print(train.shape, test.shape, structures.shape, sample_sub.shape)
print(train.columns.tolist())
print(test.columns.tolist())
print(structures.columns.tolist())



## === cell 1
all_types = pd.Index(
    pd.concat([train["type"], test["type"]], axis=0).unique()
).sort_values()
all_atoms = pd.Index(structures["atom"].unique()).sort_values()

type_dtype = pd.CategoricalDtype(categories=list(all_types), ordered=False)
atom_dtype = pd.CategoricalDtype(categories=list(all_atoms), ordered=False)

train["type"] = train["type"].astype(type_dtype)
test["type"] = test["type"].astype(type_dtype)
structures["atom"] = structures["atom"].astype(atom_dtype)

atom_cats = list(atom_dtype.categories)
if "UNK" not in atom_cats:
    atom_cats.append("UNK")
new_atom_dtype = pd.CategoricalDtype(categories=atom_cats, ordered=False)

structures_key = (
    structures[["molecule_name", "atom_index", "atom", "x", "y", "z"]]
    .drop_duplicates(subset=["molecule_name", "atom_index"], keep="first")
    .sort_values(["molecule_name", "atom_index"], kind="mergesort")
)

mol_cat = pd.Categorical(structures_key["molecule_name"])
structures_key = structures_key.copy()
structures_key["mol_code"] = mol_cat.codes.astype(np.int32, copy=False)

struct_key_u = (structures_key["mol_code"].to_numpy(np.uint64) << np.uint64(16)) | (
    structures_key["atom_index"].to_numpy(np.uint64) & np.uint64(0xFFFF)
)

ord_struct = np.argsort(struct_key_u, kind="mergesort")
struct_key_u = struct_key_u[ord_struct]

_struct_x = structures_key["x"].to_numpy(np.float32, copy=False)[ord_struct]
_struct_y = structures_key["y"].to_numpy(np.float32, copy=False)[ord_struct]
_struct_z = structures_key["z"].to_numpy(np.float32, copy=False)[ord_struct]
_struct_atom_codes = structures_key["atom"].cat.codes.to_numpy(np.int16, copy=False)[
    ord_struct
]

unk_code = new_atom_dtype.categories.get_loc("UNK")

_mol_categories = pd.Index(mol_cat.categories)
_mol_code_by_name = pd.Series(
    np.arange(_mol_categories.size, dtype=np.int32), index=_mol_categories
)


def _molname_to_code(mol_name_arr) -> np.ndarray:
    codes = _mol_code_by_name.reindex(pd.Index(mol_name_arr)).to_numpy(
        dtype=np.float32, copy=False
    )
    out = np.empty(codes.shape[0], dtype=np.int32)
    mask = np.isfinite(codes)
    out[mask] = codes[mask].astype(np.int32, copy=False)
    out[~mask] = np.int32(-1)
    return out


def _lookup_atoms_xyz(mol_name_arr, atom_index_arr):
    mol_codes = _molname_to_code(mol_name_arr)
    key = (mol_codes.astype(np.uint64, copy=False) << np.uint64(16)) | (
        atom_index_arr.astype(np.uint64, copy=False) & np.uint64(0xFFFF)
    )

    pos = np.searchsorted(struct_key_u, key, side="left")

    found = np.zeros(key.shape[0], dtype=bool)
    in_range = pos < struct_key_u.size
    if in_range.any():
        pos_ir = pos[in_range]
        found[in_range] = struct_key_u[pos_ir] == key[in_range]

    x = np.zeros(key.shape[0], dtype=np.float32)
    y = np.zeros(key.shape[0], dtype=np.float32)
    z = np.zeros(key.shape[0], dtype=np.float32)
    atom_codes = np.full(key.shape[0], unk_code, dtype=np.int16)

    if found.any():
        idx = pos[found]
        x[found] = _struct_x[idx]
        y[found] = _struct_y[idx]
        z[found] = _struct_z[idx]
        atom_codes[found] = _struct_atom_codes[idx]

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return atom_codes, x, y, z


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=False)

    atom0_codes, x0, y0, z0 = _lookup_atoms_xyz(
        out["molecule_name"], out["atom_index_0"]
    )
    atom1_codes, x1, y1, z1 = _lookup_atoms_xyz(
        out["molecule_name"], out["atom_index_1"]
    )

    dx = (x0 - x1).astype(np.float32, copy=False)
    dy = (y0 - y1).astype(np.float32, copy=False)
    dz = (z0 - z1).astype(np.float32, copy=False)
    dist = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32, copy=False)

    out = out.copy()
    out["atom_0"] = pd.Categorical.from_codes(
        atom0_codes, categories=new_atom_dtype.categories
    )
    out["atom_1"] = pd.Categorical.from_codes(
        atom1_codes, categories=new_atom_dtype.categories
    )
    out["x0"] = x0
    out["y0"] = y0
    out["z0"] = z0
    out["x1"] = x1
    out["y1"] = y1
    out["z1"] = z1
    out["dx"] = dx
    out["dy"] = dy
    out["dz"] = dz
    out["dist"] = dist
    out["type"] = out["type"].astype(type_dtype)
    return out


train_fe = add_features(train)
test_fe = add_features(test)

check_cols = [
    "atom_0",
    "atom_1",
    "x0",
    "y0",
    "z0",
    "x1",
    "y1",
    "z1",
    "dx",
    "dy",
    "dz",
    "dist",
    "type",
]
print(
    "Missing values after feature build - train:",
    int(train_fe[check_cols].isna().sum().sum()),
    "test:",
    int(test_fe[check_cols].isna().sum().sum()),
)
print(
    "Dtypes:",
    {
        "atom_0": str(train_fe["atom_0"].dtype),
        "atom_1": str(train_fe["atom_1"].dtype),
        "type": str(train_fe["type"].dtype),
    },
)



## === cell 2
from sklearn.ensemble import RandomForestRegressor

FEATURE_NUM = ["dist", "dx", "dy", "dz"]

X_num_train = np.ascontiguousarray(
    train_fe[FEATURE_NUM].to_numpy(dtype=np.float32, copy=False)
)
X_num_test = np.ascontiguousarray(
    test_fe[FEATURE_NUM].to_numpy(dtype=np.float32, copy=False)
)
X_num_train = np.nan_to_num(X_num_train, nan=0.0, posinf=0.0, neginf=0.0).astype(
    np.float32, copy=False
)
X_num_test = np.nan_to_num(X_num_test, nan=0.0, posinf=0.0, neginf=0.0).astype(
    np.float32, copy=False
)

type_tr = train_fe["type"].cat.codes.to_numpy(dtype=np.int16, copy=False)
type_te = test_fe["type"].cat.codes.to_numpy(dtype=np.int16, copy=False)
a0_tr = train_fe["atom_0"].cat.codes.to_numpy(dtype=np.int16, copy=False)
a1_tr = train_fe["atom_1"].cat.codes.to_numpy(dtype=np.int16, copy=False)
a0_te = test_fe["atom_0"].cat.codes.to_numpy(dtype=np.int16, copy=False)
a1_te = test_fe["atom_1"].cat.codes.to_numpy(dtype=np.int16, copy=False)

X_train_all = np.empty((X_num_train.shape[0], 7), dtype=np.float32)
X_test_all = np.empty((X_num_test.shape[0], 7), dtype=np.float32)

X_train_all[:, 0:4] = X_num_train
X_test_all[:, 0:4] = X_num_test
X_train_all[:, 4] = type_tr.astype(np.float32, copy=False)
X_test_all[:, 4] = type_te.astype(np.float32, copy=False)
X_train_all[:, 5] = a0_tr.astype(np.float32, copy=False)
X_test_all[:, 5] = a0_te.astype(np.float32, copy=False)
X_train_all[:, 6] = a1_tr.astype(np.float32, copy=False)
X_test_all[:, 6] = a1_te.astype(np.float32, copy=False)

X_train_all = np.ascontiguousarray(X_train_all)
X_test_all = np.ascontiguousarray(X_test_all)

y_train_all = train_fe["scalar_coupling_constant"].to_numpy(
    dtype=np.float32, copy=False
)
pred_test = np.zeros(len(test_fe), dtype=np.float32)

types = list(all_types)
print("Types:", types)

rf_params = dict(
    n_estimators=120,
    random_state=42,
    n_jobs=-1,
    max_depth=None,
    min_samples_leaf=1,
    max_samples=1.0,  # explicit = use all samples per tree (no approximation)
)

type_categories = train_fe["type"].cat.categories.astype(str).tolist()

order_tr = np.argsort(type_tr, kind="mergesort")
sorted_tr_codes = type_tr[order_tr]
tr_bounds = np.flatnonzero(
    np.r_[True, sorted_tr_codes[1:] != sorted_tr_codes[:-1], True]
)
tr_codes_unique = sorted_tr_codes[tr_bounds[:-1]]

order_te = np.argsort(type_te, kind="mergesort")
sorted_te_codes = type_te[order_te]
te_bounds = np.flatnonzero(
    np.r_[True, sorted_te_codes[1:] != sorted_te_codes[:-1], True]
)
te_codes_unique = sorted_te_codes[te_bounds[:-1]]

te_seg = {
    int(code): (te_bounds[i], te_bounds[i + 1])
    for i, code in enumerate(te_codes_unique)
}

global_median = float(np.median(y_train_all))

for i, code in enumerate(tr_codes_unique):
    code_int = int(code)
    seg = te_seg.get(code_int)
    if seg is None:
        continue
    tr_start, tr_end = tr_bounds[i], tr_bounds[i + 1]
    te_start, te_end = seg

    tr_idx = order_tr[tr_start:tr_end]
    te_idx = order_te[te_start:te_end]

    Xtr = X_train_all[tr_idx]
    ytr = y_train_all[tr_idx]
    Xte = X_test_all[te_idx]

    model = RandomForestRegressor(**rf_params)
    model.fit(Xtr, ytr)
    pred_test[te_idx] = model.predict(Xte).astype(np.float32, copy=False)

    t = (
        type_categories[code_int]
        if 0 <= code_int < len(type_categories)
        else str(code_int)
    )
    print(f"type={t:4s} train_rows={tr_idx.size:7d} test_rows={te_idx.size:7d}")

missing_idx = np.flatnonzero(~np.isfinite(pred_test))
if missing_idx.size:
    pred_test[missing_idx] = np.float32(global_median)

trained_types = set(int(c) for c in tr_codes_unique.tolist())
for code_int, (te_start, te_end) in te_seg.items():
    if code_int in trained_types:
        continue
    te_idx = order_te[te_start:te_end]
    pred_test[te_idx] = np.float32(global_median)
    t = (
        type_categories[code_int]
        if 0 <= code_int < len(type_categories)
        else str(code_int)
    )
    print(
        f"type={t:4s} train_rows={0:7d} test_rows={te_idx.size:7d} (fallback=global_median)"
    )



## === cell 3
submission = pd.DataFrame(
    {
        "id": test["id"].to_numpy(copy=False),
        "scalar_coupling_constant": pred_test.astype(np.float32, copy=False),
    }
)

submission = sample_sub[["id"]].merge(submission, on="id", how="left")
assert submission.shape[0] == sample_sub.shape[0]
if submission["scalar_coupling_constant"].isna().any():
    submission["scalar_coupling_constant"] = submission[
        "scalar_coupling_constant"
    ].fillna(np.float32(0.0))

out_path = os.path.join(WORKING_DIR, "submission.csv")
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission.head())
print(
    "Submission rows:",
    submission.shape[0],
    "NaNs:",
    int(submission["scalar_coupling_constant"].isna().sum()),
)
