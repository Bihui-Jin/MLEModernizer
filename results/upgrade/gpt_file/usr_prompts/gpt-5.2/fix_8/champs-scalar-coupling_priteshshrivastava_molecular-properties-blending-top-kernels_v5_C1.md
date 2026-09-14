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

-1.5801604792430517

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 1.99777) has done: 'I remove the broken dependency on external “blending” input files that aren’t present in your environment and replace it with a minimal, self-contained baseline that trains and predicts from the provided CHAMPS data. To keep changes minimal while ensuring a valid end-to-end run, the new code build a small set of safe geometric features from `structures.csv` (pairwise distance and atom types) and train a separate lightweight model per coupling `type`. This preserves the original intent (predict `scalar_coupling_constant` from engineered structure features) and produces a correctly formatted `submission.csv`. The approach is deterministic and should yield a meaningful (though not SOTA) score instead of failing before submission creation.'

# 9. Code solution

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

structures_key = structures[
    ["molecule_name", "atom_index", "atom", "x", "y", "z"]
].drop_duplicates(subset=["molecule_name", "atom_index"], keep="first")
_struct_idx = structures_key.set_index(["molecule_name", "atom_index"], drop=True)[
    ["atom", "x", "y", "z"]
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=False)

    key0 = pd.MultiIndex.from_frame(
        out[["molecule_name", "atom_index_0"]], names=["molecule_name", "atom_index"]
    )
    key1 = pd.MultiIndex.from_frame(
        out[["molecule_name", "atom_index_1"]], names=["molecule_name", "atom_index"]
    )

    a0 = _struct_idx.reindex(key0)
    a1 = _struct_idx.reindex(key1)

    x0 = a0["x"].to_numpy(dtype=np.float32, copy=False)
    y0 = a0["y"].to_numpy(dtype=np.float32, copy=False)
    z0 = a0["z"].to_numpy(dtype=np.float32, copy=False)
    x1 = a1["x"].to_numpy(dtype=np.float32, copy=False)
    y1 = a1["y"].to_numpy(dtype=np.float32, copy=False)
    z1 = a1["z"].to_numpy(dtype=np.float32, copy=False)

    x0 = np.nan_to_num(x0, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )
    y0 = np.nan_to_num(y0, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )
    z0 = np.nan_to_num(z0, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )
    x1 = np.nan_to_num(x1, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )
    y1 = np.nan_to_num(y1, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )
    z1 = np.nan_to_num(z1, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )

    atom0 = a0["atom"]
    atom1 = a1["atom"]
    atom0 = atom0.astype("string").fillna("UNK").astype(new_atom_dtype)
    atom1 = atom1.astype("string").fillna("UNK").astype(new_atom_dtype)

    dx = (x0 - x1).astype(np.float32, copy=False)
    dy = (y0 - y1).astype(np.float32, copy=False)
    dz = (z0 - z1).astype(np.float32, copy=False)
    dist = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32, copy=False)

    out = out.copy()  # materialize once with new columns
    out["atom_0"] = atom0
    out["atom_1"] = atom1
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



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3882928937.py in <cell line: 0>()
    101 
    102 
--> 103 train_fe = add_features(train)
    104 test_fe = add_features(test)
    105 

/tmp/ipykernel_11/3882928937.py in add_features(df)
     85 
     86     out = out.copy()  # materialize once with new columns
---> 87     out["atom_0"] = atom0
     88     out["atom_1"] = atom1
     89     out["x0"] = x0

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

## === cell 2

from sklearn.ensemble import RandomForestRegressor

FEATURE_NUM = ["dist", "dx", "dy", "dz"]

X_num_train = train_fe[FEATURE_NUM].to_numpy(dtype=np.float32, copy=False)
X_num_test = test_fe[FEATURE_NUM].to_numpy(dtype=np.float32, copy=False)
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

y_train_all = train_fe["scalar_coupling_constant"].to_numpy(
    dtype=np.float32, copy=False
)
pred_test = np.zeros(len(test_fe), dtype=np.float32)

types = list(all_types)
print("Types:", types)

rf_params = dict(
    n_estimators=120, random_state=42, n_jobs=-1, max_depth=None, min_samples_leaf=1
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

for i, code in enumerate(tr_codes_unique):
    code_int = int(code)
    if code_int not in te_seg:
        continue
    tr_start, tr_end = tr_bounds[i], tr_bounds[i + 1]
    te_start, te_end = te_seg[code_int]

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
    global_median = float(np.median(y_train_all))
    pred_test[missing_idx] = np.float32(global_median)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1900255378.py in <cell line: 0>()
     10 
     11 # Build dense feature matrices: [dist, dx, dy, dz, type_code, atom0_code, atom1_code]
---> 12 X_num_train = train_fe[FEATURE_NUM].to_numpy(dtype=np.float32, copy=False)
     13 X_num_test = test_fe[FEATURE_NUM].to_numpy(dtype=np.float32, copy=False)
     14 X_num_train = np.nan_to_num(X_num_train, nan=0.0, posinf=0.0, neginf=0.0).astype(

NameError: name 'train_fe' is not defined

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

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/596636507.py in <cell line: 0>()
      2     {
      3         "id": test["id"].to_numpy(copy=False),
----> 4         "scalar_coupling_constant": pred_test.astype(np.float32, copy=False),
      5     }
      6 )

NameError: name 'pred_test' is not defined
