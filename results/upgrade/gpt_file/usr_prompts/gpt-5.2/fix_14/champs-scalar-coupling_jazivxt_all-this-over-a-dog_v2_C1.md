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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.90666

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.34178) has done: 'The crash happens because the merge with `structures.csv` leaves missing coordinates for some rows, and `ExtraTreesRegressor` cannot predict with NaNs. I keep your model and feature logic the same, but add a minimal missing-value handling step (median imputation fitted on train, applied to both train/test) so training and inference run end-to-end. I also make the file reading robust to your environment’s actual paths (prefer `/kaggle/input/champs-scalar-coupling/`, fallback to `../input/`). Finally, I ensure the submission is written as `submission.csv` with the exact required columns.'
- What this solution (achieved 1.30329) has done: 'You’re currently far from the target (1.34178 vs 0.90666; lower is better), so we should legitimately improve the model while keeping the same overall pipeline (same feature set + ExtraTreesRegressor). The biggest win with minimal semantic change is to align training/validation and model capacity with the competition metric being averaged per `type`: we (1) train a separate ExtraTrees model per coupling `type` (same regressor, same features, same loss), and (2) make the quick local validation split by `molecule_name` (to match the competition’s molecule split) instead of a random row split on the tail. I also keep your existing missing-value handling and data paths, and ensure we still write `submission.csv` with the required columns.'
- What this solution (achieved 1.17332) has done: 'The timeout is dominated by repeatedly fitting ExtraTrees many times (once per coupling `type`) on large DataFrame slices, plus expensive pandas merges and per-column fill loops on multi‑million rows. I keep the exact model/training logic, but make it faster by (1) reading only needed columns with smaller dtypes, (2) performing the structures joins via indexed `reindex` instead of large merges, (3) doing categorical encoding and missing-value filling in a vectorized way, and (4) converting features/targets once to contiguous NumPy arrays so each per-type fit/predict uses cheap integer indexing rather than repeated `.loc` slicing. These changes are provably equivalent in semantics and preserve determinism while cutting overhead drastically.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from sklearn import ensemble, metrics

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

np.random.seed(99)
pd.options.mode.copy_on_write = True  # pandas 2.x: reduces unnecessary copies

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

BASE_CANDIDATES = [
    "/kaggle/input/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling",
    "../input/champs-scalar-coupling",
    "../input",
]
BASE = None
for b in BASE_CANDIDATES:
    if os.path.exists(b):
        BASE = b
        break
if BASE is None:
    raise FileNotFoundError(
        "Could not locate input data directory among: " + str(BASE_CANDIDATES)
    )

train_dtypes = {
    "id": np.int32,
    "molecule_name": "category",
    "atom_index_0": np.int16,
    "atom_index_1": np.int16,
    "type": "category",
    "scalar_coupling_constant": np.float32,
}
test_dtypes = {
    "id": np.int32,
    "molecule_name": "category",
    "atom_index_0": np.int16,
    "atom_index_1": np.int16,
    "type": "category",
}

train = pd.read_csv(
    os.path.join(BASE, "train.csv"),
    usecols=list(train_dtypes.keys()),
    dtype=train_dtypes,
)
test = pd.read_csv(
    os.path.join(BASE, "test.csv"),
    usecols=list(test_dtypes.keys()),
    dtype=test_dtypes,
)
sub = pd.read_csv(
    os.path.join(BASE, "sample_submission.csv"),
    usecols=["id", "scalar_coupling_constant"],
)
print(train.shape, test.shape, sub.shape)

type_str_train = train["type"].astype("string")
type_str_test = test["type"].astype("string")
train["atom"] = type_str_train.str[3].astype("string")
test["atom"] = type_str_test.str[3].astype("string")

for i in range(4):
    tr = type_str_train.str[i]
    te = type_str_test.str[i]
    tr_cat = pd.Categorical(tr)
    te_cat = pd.Categorical(te)
    union = tr_cat.categories.union(te_cat.categories)
    train["type" + str(i)] = pd.Categorical(tr, categories=union).codes.astype(np.int32)
    test["type" + str(i)] = pd.Categorical(te, categories=union).codes.astype(np.int32)

structures = pd.read_csv(
    os.path.join(BASE, "structures.csv"),
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
    dtype={
        "molecule_name": "category",
        "atom_index": np.int16,
        "atom": "category",
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
    },
).drop_duplicates(subset=["molecule_name", "atom_index"], keep="first")

mol_union = (
    train["molecule_name"]
    .cat.categories.union(test["molecule_name"].cat.categories)
    .union(structures["molecule_name"].cat.categories)
)
train["molecule_name"] = train["molecule_name"].cat.set_categories(mol_union)
test["molecule_name"] = test["molecule_name"].cat.set_categories(mol_union)
structures["molecule_name"] = structures["molecule_name"].cat.set_categories(mol_union)

train["mol_code"] = train["molecule_name"].cat.codes.astype(np.int32)
test["mol_code"] = test["molecule_name"].cat.codes.astype(np.int32)
structures["mol_code"] = structures["molecule_name"].cat.codes.astype(np.int32)

structures_small = structures[["mol_code", "atom_index", "atom", "x", "y", "z"]].copy()

_struct_key = (structures_small["mol_code"].to_numpy(np.int64) << 16) | (
    structures_small["atom_index"].to_numpy(np.int64) & 0xFFFF
)

_struct_order = np.argsort(_struct_key, kind="mergesort")
_struct_key_sorted = _struct_key[_struct_order]

_struct_atom = (
    structures_small["atom"].astype("string").to_numpy(copy=False)[_struct_order]
)
_struct_x = structures_small["x"].to_numpy(np.float32, copy=False)[_struct_order]
_struct_y = structures_small["y"].to_numpy(np.float32, copy=False)[_struct_order]
_struct_z = structures_small["z"].to_numpy(np.float32, copy=False)[_struct_order]


def _lookup_atom_features_fast(df, idx_col, out_prefix):
    k = (df["mol_code"].to_numpy(np.int64) << 16) | (
        df[idx_col].to_numpy(np.int64) & 0xFFFF
    )
    pos = np.searchsorted(_struct_key_sorted, k)
    ok = (pos < _struct_key_sorted.size) & (_struct_key_sorted[pos] == k)
    atom_out = np.empty(len(df), dtype=_struct_atom.dtype)
    x_out = np.empty(len(df), dtype=np.float32)
    y_out = np.empty(len(df), dtype=np.float32)
    z_out = np.empty(len(df), dtype=np.float32)

    atom_out[ok] = _struct_atom[pos[ok]]
    x_out[ok] = _struct_x[pos[ok]]
    y_out[ok] = _struct_y[pos[ok]]
    z_out[ok] = _struct_z[pos[ok]]

    if (~ok).any():
        atom_out[~ok] = pd.array([pd.NA] * int((~ok).sum()), dtype="string")
        x_out[~ok] = np.nan
        y_out[~ok] = np.nan
        z_out[~ok] = np.nan

    df[out_prefix + "atom"] = atom_out
    df[out_prefix + "x"] = x_out
    df[out_prefix + "y"] = y_out
    df[out_prefix + "z"] = z_out


_lookup_atom_features_fast(train, "atom_index_0", "atom0_")
_lookup_atom_features_fast(test, "atom_index_0", "atom0_")
_lookup_atom_features_fast(train, "atom_index_1", "atom1_")
_lookup_atom_features_fast(test, "atom_index_1", "atom1_")

train.rename(
    columns={
        "atom0_atom": "atom0",
        "atom0_x": "x0",
        "atom0_y": "y0",
        "atom0_z": "z0",
        "atom1_atom": "atom1",
        "atom1_x": "x1",
        "atom1_y": "y1",
        "atom1_z": "z1",
    },
    inplace=True,
)
test.rename(
    columns={
        "atom0_atom": "atom0",
        "atom0_x": "x0",
        "atom0_y": "y0",
        "atom0_z": "z0",
        "atom1_atom": "atom1",
        "atom1_x": "x1",
        "atom1_y": "y1",
        "atom1_z": "z1",
    },
    inplace=True,
)

train.drop(columns=["mol_code"], inplace=True)
test.drop(columns=["mol_code"], inplace=True)

del structures, structures_small, type_str_train, type_str_test
del (
    _struct_key,
    _struct_order,
    _struct_key_sorted,
    _struct_atom,
    _struct_x,
    _struct_y,
    _struct_z,
)
print(train.shape, test.shape, sub.shape)



## === cell 1
pass



## === cell 2
col = [
    c
    for c in train.columns
    if c not in ["id", "molecule_name", "scalar_coupling_constant", "type", "atom"]
]

X_train = train[col].copy()
X_test = test[col].copy()

obj_cols = [
    c
    for c in col
    if (not pd.api.types.is_numeric_dtype(X_train[c]))
    or (not pd.api.types.is_numeric_dtype(X_test[c]))
]

for c in obj_cols:
    tr = X_train[c].astype("string")
    te = X_test[c].astype("string")
    cats = pd.Index(tr.unique()).union(pd.Index(te.unique()))
    X_train[c] = pd.Categorical(tr, categories=cats).codes.astype(np.int32)
    X_test[c] = pd.Categorical(te, categories=cats).codes.astype(np.int32)

for c in col:
    if pd.api.types.is_numeric_dtype(X_train[c]):
        s = X_train[c]
        fill = float(s.median(skipna=True)) if s.notna().any() else 0.0
        if np.isnan(fill):
            fill = 0.0
        X_train[c] = s.fillna(fill)
        X_test[c] = X_test[c].fillna(fill)
    else:
        mode = X_train[c].mode(dropna=True)
        fill = int(mode.iloc[0]) if len(mode) else -1
        X_train[c] = X_train[c].fillna(fill)
        X_test[c] = X_test[c].fillna(fill)

rng = np.random.RandomState(99)
mol_codes = train["molecule_name"].cat.codes.to_numpy(np.int32, copy=False)
unique_mols = np.unique(mol_codes)
rng.shuffle(unique_mols)
n_val = int(0.2 * len(unique_mols))
val_mols = set(unique_mols[:n_val])
is_val = np.isin(mol_codes, np.fromiter(val_mols, dtype=np.int32))

reg_params = dict(
    n_estimators=200,  # unchanged for initial fit
    n_jobs=-1,
    random_state=4,
    min_samples_leaf=1,
    min_samples_split=2,
    warm_start=True,
)

type_codes_train = train["type"].cat.codes.to_numpy(np.int16, copy=False)
type_codes_test = test["type"].cat.codes.to_numpy(np.int16, copy=False)
types = np.unique(type_codes_train)

Xtr_np = np.ascontiguousarray(X_train.to_numpy(dtype=np.float64, copy=False))
Xte_np = np.ascontiguousarray(X_test.to_numpy(dtype=np.float64, copy=False))
y_np = train["scalar_coupling_constant"].to_numpy(np.float64, copy=False)

oof = np.full(len(train), np.nan, dtype=np.float64)

idx_by_type_all_tr = {}
idx_by_type_tr = {}
idx_by_type_va = {}
is_val_bool = is_val.astype(bool, copy=False)
not_val_bool = ~is_val_bool

for t in types:
    m_all = type_codes_train == t
    if m_all.any():
        idx_all = np.flatnonzero(m_all)
        idx_by_type_all_tr[t] = idx_all
        m_tr = not_val_bool & m_all
        m_va = is_val_bool & m_all
        if m_tr.any() and m_va.any():
            idx_by_type_tr[t] = np.flatnonzero(m_tr)
            idx_by_type_va[t] = np.flatnonzero(m_va)

reg_by_type = {}
for t in types:
    if t not in idx_by_type_tr:
        continue
    idx_tr = idx_by_type_tr[t]
    idx_va = idx_by_type_va[t]

    reg = ensemble.ExtraTreesRegressor(**reg_params)
    reg.fit(Xtr_np[idx_tr], y_np[idx_tr])
    oof[idx_va] = reg.predict(Xtr_np[idx_va])
    reg_by_type[t] = reg  # reuse later

val_scores = []
for t in types:
    m = is_val_bool & (type_codes_train == t)
    if not m.any():
        continue
    pred = oof[m]
    if np.isnan(pred).any():
        continue
    mae = metrics.mean_absolute_error(y_np[m], pred)
    mae = max(float(mae), 1e-12)
    val_scores.append(np.log(mae))
print("Validation mean log(MAE) across types:", float(np.mean(val_scores)))

test_pred = np.full(len(test), np.nan, dtype=np.float64)

idx_test_by_type = {}
for t in types:
    te_mask = type_codes_test == t
    if te_mask.any():
        idx_test_by_type[t] = np.flatnonzero(te_mask)

for t in types:
    idx_te = idx_test_by_type.get(t)
    if idx_te is None:
        continue
    idx_tr_all = idx_by_type_all_tr.get(t)
    if idx_tr_all is None or len(idx_tr_all) == 0:
        continue

    reg = reg_by_type.get(t)
    if reg is None:
        reg = ensemble.ExtraTreesRegressor(**reg_params)
        reg.fit(Xtr_np[idx_tr_all], y_np[idx_tr_all])
    else:
        reg.set_params(n_estimators=reg.n_estimators + 200)
        reg.fit(Xtr_np[idx_tr_all], y_np[idx_tr_all])

    test_pred[idx_te] = reg.predict(Xte_np[idx_te])

if np.isnan(test_pred).any():
    reg_global = ensemble.ExtraTreesRegressor(**reg_params)
    reg_global.fit(Xtr_np, y_np)
    miss = np.isnan(test_pred)
    test_pred[miss] = reg_global.predict(Xte_np[miss])

test["scalar_coupling_constant"] = test_pred

submission = test[["id", "scalar_coupling_constant"]].copy()
submission.to_csv("submission.csv", float_format="%.9f", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/350817036.py in <cell line: 0>()
     22     # union of unique values is equivalent to categories of concatenation for encoding purposes
     23     cats = pd.Index(tr.unique()).union(pd.Index(te.unique()))
---> 24     X_train[c] = pd.Categorical(tr, categories=cats).codes.astype(np.int32)
     25     X_test[c] = pd.Categorical(te, categories=cats).codes.astype(np.int32)
     26 

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/categorical.py in __init__(self, values, categories, ordered, dtype, fastpath, copy)
    389             fastpath = False
    390 
--> 391         dtype = CategoricalDtype._from_values_or_dtype(
    392             values, categories, ordered, dtype
    393         )

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/dtypes.py in _from_values_or_dtype(cls, values, categories, ordered, dtype)
    336             # Note: This could potentially have categories=None and
    337             # ordered=None.
--> 338             dtype = CategoricalDtype(categories, ordered)
    339 
    340         return cast(CategoricalDtype, dtype)

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/dtypes.py in __init__(self, categories, ordered)
    219 
    220     def __init__(self, categories=None, ordered: Ordered = False) -> None:
--> 221         self._finalize(categories, ordered, fastpath=False)
    222 
    223     @classmethod

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/dtypes.py in _finalize(self, categories, ordered, fastpath)
    376 
    377         if categories is not None:
--> 378             categories = self.validate_categories(categories, fastpath=fastpath)
    379 
    380         self._categories = categories

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/dtypes.py in validate_categories(categories, fastpath)
    574         if not fastpath:
    575             if categories.hasnans:
--> 576                 raise ValueError("Categorical categories cannot be null")
    577 
    578             if not categories.is_unique:

ValueError: Categorical categories cannot be null
