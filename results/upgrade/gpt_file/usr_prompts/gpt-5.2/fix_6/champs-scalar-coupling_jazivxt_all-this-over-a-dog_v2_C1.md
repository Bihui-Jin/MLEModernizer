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

1.17332

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
from sklearn import preprocessing, ensemble, metrics

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
    os.path.join(BASE, "test.csv"), usecols=list(test_dtypes.keys()), dtype=test_dtypes
)
sub = pd.read_csv(
    os.path.join(BASE, "sample_submission.csv"),
    usecols=["id", "scalar_coupling_constant"],
)
print(train.shape, test.shape, sub.shape)

train["atom"] = train["type"].astype("string").str[3]
test["atom"] = test["type"].astype("string").str[3]

lbl = preprocessing.LabelEncoder()
type_str_train = train["type"].astype("string")
type_str_test = test["type"].astype("string")
for i in range(4):
    train["type" + str(i)] = lbl.fit_transform(type_str_train.str[i])
    test["type" + str(i)] = lbl.transform(type_str_test.str[i])

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
)

st = structures.set_index(["molecule_name", "atom_index"])[["atom", "x", "y", "z"]]

idx1_tr = pd.MultiIndex.from_arrays([train["molecule_name"], train["atom_index_1"]])
idx1_te = pd.MultiIndex.from_arrays([test["molecule_name"], test["atom_index_1"]])
v1_tr = st.reindex(idx1_tr)
v1_te = st.reindex(idx1_te)
train["atom1"] = v1_tr["atom"].astype("string")
train["x1"] = v1_tr["x"].to_numpy(dtype=np.float32, copy=False)
train["y1"] = v1_tr["y"].to_numpy(dtype=np.float32, copy=False)
train["z1"] = v1_tr["z"].to_numpy(dtype=np.float32, copy=False)
test["atom1"] = v1_te["atom"].astype("string")
test["x1"] = v1_te["x"].to_numpy(dtype=np.float32, copy=False)
test["y1"] = v1_te["y"].to_numpy(dtype=np.float32, copy=False)
test["z1"] = v1_te["z"].to_numpy(dtype=np.float32, copy=False)

idx0_tr = pd.MultiIndex.from_arrays([train["molecule_name"], train["atom_index_0"]])
idx0_te = pd.MultiIndex.from_arrays([test["molecule_name"], test["atom_index_0"]])
v0_tr = st.reindex(idx0_tr)
v0_te = st.reindex(idx0_te)
train["atom0"] = v0_tr["atom"].astype("string")
train["x0"] = v0_tr["x"].to_numpy(dtype=np.float32, copy=False)
train["y0"] = v0_tr["y"].to_numpy(dtype=np.float32, copy=False)
train["z0"] = v0_tr["z"].to_numpy(dtype=np.float32, copy=False)
test["atom0"] = v0_te["atom"].astype("string")
test["x0"] = v0_te["x"].to_numpy(dtype=np.float32, copy=False)
test["y0"] = v0_te["y"].to_numpy(dtype=np.float32, copy=False)
test["z0"] = v0_te["z"].to_numpy(dtype=np.float32, copy=False)

del structures, st, v0_tr, v0_te, v1_tr, v1_te, idx0_tr, idx0_te, idx1_tr, idx1_te

print(train.shape, test.shape, sub.shape)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2467332805.py in <cell line: 0>()
     92 v1_tr = st.reindex(idx1_tr)
     93 v1_te = st.reindex(idx1_te)
---> 94 train["atom1"] = v1_tr["atom"].astype("string")
     95 train["x1"] = v1_tr["x"].to_numpy(dtype=np.float32, copy=False)
     96 train["y1"] = v1_tr["y"].to_numpy(dtype=np.float32, copy=False)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __setitem__(self, key, value)
   4309         else:
   4310             # set column
-> 4311             self._set_item(key, value)
   4312 
   4313     def _setitem_slice(self, key: slice, value) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _set_item(self, key, value)
   4522         ensure homogeneity.
   4523         """
-> 4524         value, refs = self._sanitize_column(value)
   4525 
   4526         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _sanitize_column(self, value)
   5261             if not isinstance(value, Series):
   5262                 value = Series(value)
-> 5263             return _reindex_for_setitem(value, self.index)
   5264 
   5265         if is_list_like(value):

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _reindex_for_setitem(value, index)
  12690         if not value.index.is_unique:
  12691             # duplicate axis
> 12692             raise err
  12693 
  12694         raise TypeError(

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _reindex_for_setitem(value, index)
  12685     # GH#4107
  12686     try:
> 12687         reindexed_value = value.reindex(index)._values
  12688     except ValueError as err:
  12689         # raised in MultiIndex.from_tuples, see test_insert_error_msmgs

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in reindex(self, index, axis, method, copy, level, fill_value, limit, tolerance)
   5151         tolerance=None,
   5152     ) -> Series:
-> 5153         return super().reindex(
   5154             index=index,
   5155             method=method,

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in reindex(self, labels, index, columns, axis, method, copy, level, fill_value, limit, tolerance)
   5608 
   5609         # perform the reindex on the axes
-> 5610         return self._reindex_axes(
   5611             axes, level, limit, tolerance, method, fill_value, copy
   5612         ).__finalize__(self, method="reindex")

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _reindex_axes(self, axes, level, limit, tolerance, method, fill_value, copy)
   5631 
   5632             ax = self._get_axis(a)
-> 5633             new_index, indexer = ax.reindex(
   5634                 labels, level=level, limit=limit, tolerance=tolerance, method=method
   5635             )

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in reindex(self, target, method, level, limit, tolerance)
   4424                     )
   4425                 elif self._is_multi:
-> 4426                     raise ValueError("cannot handle a non-unique multi-index!")
   4427                 elif not self.is_unique:
   4428                     # GH#42568

ValueError: cannot handle a non-unique multi-index!

## === cell 1
train.head()



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
    if not pd.api.types.is_numeric_dtype(X_train[c])
    or not pd.api.types.is_numeric_dtype(X_test[c])
]

for c in obj_cols:
    combined = pd.concat([X_train[c], X_test[c]], axis=0, ignore_index=True).astype(
        "string"
    )
    cats = pd.Categorical(combined)
    codes = cats.codes.astype(np.int32)  # -1 indicates missing
    X_train[c] = codes[: len(X_train)]
    X_test[c] = codes[len(X_train) :]

for c in col:
    s_tr = X_train[c]
    if pd.api.types.is_numeric_dtype(s_tr):
        fill = s_tr.median()
        if pd.isna(fill):
            fill = 0.0
        X_train[c] = s_tr.fillna(fill)
        X_test[c] = X_test[c].fillna(fill)
    else:
        mode = s_tr.mode(dropna=True)
        fill = mode.iloc[0] if len(mode) else -1
        X_train[c] = s_tr.fillna(fill)
        X_test[c] = X_test[c].fillna(fill)

rng = np.random.RandomState(99)
unique_mols = train["molecule_name"].unique()
rng.shuffle(unique_mols)
n_val = int(0.2 * len(unique_mols))
val_mols = set(unique_mols[:n_val])
is_val = train["molecule_name"].isin(val_mols).to_numpy()

reg_params = dict(
    n_estimators=200,  # unchanged
    n_jobs=-1,
    random_state=4,
    min_samples_leaf=1,
    min_samples_split=2,
)

types = train["type"].unique()

Xtr_np = np.ascontiguousarray(X_train[col].to_numpy())
Xte_np = np.ascontiguousarray(X_test[col].to_numpy())
y_np = train["scalar_coupling_constant"].to_numpy()

type_train = train["type"].to_numpy()
type_test = test["type"].to_numpy()

oof = np.full(len(train), np.nan, dtype=np.float64)

idx_by_type_tr = {}
idx_by_type_va = {}
for t in types:
    m_tr = (~is_val) & (type_train == t)
    m_va = (is_val) & (type_train == t)
    if m_tr.any() and m_va.any():
        idx_by_type_tr[t] = np.flatnonzero(m_tr)
        idx_by_type_va[t] = np.flatnonzero(m_va)

for t in types:
    if t not in idx_by_type_tr:
        continue
    idx_tr = idx_by_type_tr[t]
    idx_va = idx_by_type_va[t]

    reg = ensemble.ExtraTreesRegressor(**reg_params)
    reg.fit(Xtr_np[idx_tr], y_np[idx_tr])
    oof[idx_va] = reg.predict(Xtr_np[idx_va])

val_scores = []
for t in types:
    m = (is_val) & (type_train == t)
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

idx_train_by_type = {}
idx_test_by_type = {}
for t in types:
    tr_mask = type_train == t
    te_mask = type_test == t
    if te_mask.any():
        idx_train_by_type[t] = np.flatnonzero(tr_mask)
        idx_test_by_type[t] = np.flatnonzero(te_mask)

for t in types:
    if t not in idx_test_by_type:
        continue
    idx_tr = idx_train_by_type[t]
    idx_te = idx_test_by_type[t]

    reg = ensemble.ExtraTreesRegressor(**reg_params)
    reg.fit(Xtr_np[idx_tr], y_np[idx_tr])
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
