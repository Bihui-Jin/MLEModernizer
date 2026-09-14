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

# 5. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd
from sklearn import preprocessing, ensemble, model_selection, metrics

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

SEED = 99
np.random.seed(SEED)

DATA_DIR = "../input"

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

train_dtypes = {
    "id": "int32",
    "molecule_name": "category",
    "atom_index_0": "int16",
    "atom_index_1": "int16",
    "type": "category",
    "scalar_coupling_constant": "float32",
}
test_dtypes = {
    "id": "int32",
    "molecule_name": "category",
    "atom_index_0": "int16",
    "atom_index_1": "int16",
    "type": "category",
}
usecols_train = [
    "id",
    "molecule_name",
    "atom_index_0",
    "atom_index_1",
    "type",
    "scalar_coupling_constant",
]
usecols_test = ["id", "molecule_name", "atom_index_0", "atom_index_1", "type"]

train = pd.read_csv(f"{DATA_DIR}/train.csv", usecols=usecols_train, dtype=train_dtypes)
test = pd.read_csv(f"{DATA_DIR}/test.csv", usecols=usecols_test, dtype=test_dtypes)
sub = pd.read_csv(
    f"{DATA_DIR}/sample_submission.csv", usecols=["id", "scalar_coupling_constant"]
)
print(train.shape, test.shape, sub.shape)

train_type_str = train["type"].astype(str)
test_type_str = test["type"].astype(str)

train["atom"] = train_type_str.str[3]
test["atom"] = test_type_str.str[3]

lbl = preprocessing.LabelEncoder()
for i in range(4):
    col = f"type{i}"
    train[col] = lbl.fit_transform(train_type_str.str[i].to_numpy())
    test[col] = lbl.transform(test_type_str.str[i].to_numpy())

train["_is_train"] = np.int8(1)
test["_is_train"] = np.int8(0)

df_all = pd.concat([train, test], axis=0, ignore_index=True, copy=False)
del train, test
gc.collect()

df_all.sort_values(
    ["molecule_name", "atom_index_0", "atom_index_1"], inplace=True, kind="mergesort"
)
df_all.reset_index(drop=True, inplace=True)

structures = pd.read_csv(
    f"{DATA_DIR}/structures.csv",
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
    dtype={
        "molecule_name": "category",
        "atom_index": "int16",
        "atom": "category",
        "x": "float32",
        "y": "float32",
        "z": "float32",
    },
)

structures.sort_values(["molecule_name", "atom_index"], inplace=True, kind="mergesort")
structures.reset_index(drop=True, inplace=True)

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
    )
    .reset_index()
)
df_all = pd.merge(
    df_all, mol_agg, how="left", on="molecule_name", copy=False, sort=False
)
del mol_agg
gc.collect()

structures_idx = structures.set_index(["molecule_name", "atom_index"], drop=True)
scols = ["atom", "x", "y", "z"]

s1 = structures_idx[scols].rename(
    columns={"atom": "atom_1", "x": "x_1", "y": "y_1", "z": "z_1"}
)
df_all = df_all.join(s1, on=["molecule_name", "atom_index_1"], how="left", sort=False)

s0 = structures_idx[scols].rename(
    columns={"atom": "atom_0", "x": "x_0", "y": "y_0", "z": "z_0"}
)
df_all = df_all.join(s0, on=["molecule_name", "atom_index_0"], how="left", sort=False)

del structures, structures_idx, s0, s1
gc.collect()

x0 = np.ascontiguousarray(df_all["x_0"].to_numpy(dtype=np.float64, copy=False))
y0 = np.ascontiguousarray(df_all["y_0"].to_numpy(dtype=np.float64, copy=False))
z0 = np.ascontiguousarray(df_all["z_0"].to_numpy(dtype=np.float64, copy=False))
x1 = np.ascontiguousarray(df_all["x_1"].to_numpy(dtype=np.float64, copy=False))
y1 = np.ascontiguousarray(df_all["y_1"].to_numpy(dtype=np.float64, copy=False))
z1 = np.ascontiguousarray(df_all["z_1"].to_numpy(dtype=np.float64, copy=False))

dx = x0 - x1
dy = y0 - y1
dz = z0 - z1

dist2 = dx * dx + dy * dy + dz * dz
dist = np.sqrt(dist2)

df_all["dist"] = dist
df_all["dist2"] = dist2
df_all["abs_dx"] = np.abs(dx)
df_all["abs_dy"] = np.abs(dy)
df_all["abs_dz"] = np.abs(dz)

eps = 1e-6
df_all["inv_dist"] = 1.0 / (dist + eps)
df_all["dist3"] = dist * dist2

x_mean = np.ascontiguousarray(df_all["x_mean"].to_numpy(dtype=np.float64, copy=False))
y_mean = np.ascontiguousarray(df_all["y_mean"].to_numpy(dtype=np.float64, copy=False))
z_mean = np.ascontiguousarray(df_all["z_mean"].to_numpy(dtype=np.float64, copy=False))

x0_c = x0 - x_mean
y0_c = y0 - y_mean
z0_c = z0 - z_mean
x1_c = x1 - x_mean
y1_c = y1 - y_mean
z1_c = z1 - z_mean

df_all["x0_c"] = x0_c
df_all["y0_c"] = y0_c
df_all["z0_c"] = z0_c
df_all["x1_c"] = x1_c
df_all["y1_c"] = y1_c
df_all["z1_c"] = z1_c

r0_c = np.sqrt(x0_c * x0_c + y0_c * y0_c + z0_c * z0_c)
r1_c = np.sqrt(x1_c * x1_c + y1_c * y1_c + z1_c * z1_c)
df_all["r0_c"] = r0_c
df_all["r1_c"] = r1_c
df_all["abs_dr_c"] = np.abs(r0_c - r1_c)

pe = pd.read_csv(
    f"{DATA_DIR}/potential_energy.csv",
    usecols=["molecule_name", "potential_energy"],
    dtype={"molecule_name": "category", "potential_energy": "float32"},
).set_index("molecule_name")
df_all = df_all.join(pe, on="molecule_name", how="left", sort=False)
del pe
gc.collect()

mc = pd.read_csv(
    f"{DATA_DIR}/mulliken_charges.csv",
    usecols=["molecule_name", "atom_index", "mulliken_charge"],
    dtype={
        "molecule_name": "category",
        "atom_index": "int16",
        "mulliken_charge": "float32",
    },
).set_index(["molecule_name", "atom_index"])
mc0 = mc.rename(columns={"mulliken_charge": "mulliken_charge_0"})
mc1 = mc.rename(columns={"mulliken_charge": "mulliken_charge_1"})
df_all = df_all.join(mc0, on=["molecule_name", "atom_index_0"], how="left", sort=False)
df_all = df_all.join(mc1, on=["molecule_name", "atom_index_1"], how="left", sort=False)
del mc, mc0, mc1
gc.collect()

dm = pd.read_csv(
    f"{DATA_DIR}/dipole_moments.csv",
    usecols=["molecule_name", "X", "Y", "Z"],
    dtype={"molecule_name": "category", "X": "float32", "Y": "float32", "Z": "float32"},
).set_index("molecule_name")
df_all = df_all.join(dm, on="molecule_name", how="left", sort=False)
del dm
gc.collect()

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
        "molecule_name": "category",
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
).set_index(["molecule_name", "atom_index"])

mst0 = mst.rename(
    columns={
        "XX": "XX_0",
        "YX": "YX_0",
        "ZX": "ZX_0",
        "XY": "XY_0",
        "YY": "YY_0",
        "ZY": "ZY_0",
        "XZ": "XZ_0",
        "YZ": "YZ_0",
        "ZZ": "ZZ_0",
    }
)
mst1 = mst.rename(
    columns={
        "XX": "XX_1",
        "YX": "YX_1",
        "ZX": "ZX_1",
        "XY": "XY_1",
        "YY": "YY_1",
        "ZY": "ZY_1",
        "XZ": "XZ_1",
        "YZ": "YZ_1",
        "ZZ": "ZZ_1",
    }
)

df_all = df_all.join(mst0, on=["molecule_name", "atom_index_0"], how="left", sort=False)
df_all = df_all.join(mst1, on=["molecule_name", "atom_index_1"], how="left", sort=False)
del mst, mst0, mst1
gc.collect()

train = (
    df_all[df_all["_is_train"] == 1].drop(columns=["_is_train"]).reset_index(drop=True)
)
test = (
    df_all[df_all["_is_train"] == 0]
    .drop(columns=["_is_train", "scalar_coupling_constant"])
    .reset_index(drop=True)
)
del df_all
gc.collect()

print(train.shape, test.shape, sub.shape)



## === cell 1
_ = None



## === cell 2
base_col = [
    c
    for c in train.columns
    if c not in ["id", "molecule_name", "scalar_coupling_constant", "type", "atom"]
]
base_col = [c for c in base_col if pd.api.types.is_numeric_dtype(train[c])]

X_train_all = train[base_col]
y_train_all = train["scalar_coupling_constant"].to_numpy()
X_test_all = test[base_col]

train_type_cat = train["type"].cat.codes.to_numpy()
test_type_cat = test["type"].cat.codes.to_numpy()

test_pred = np.empty(len(test), dtype=np.float64)
test_pred[:] = np.nan

rng_state = 99
val_scores = {}

types = sorted(train["type"].unique())
print("Num types:", len(types), "Types:", types)

X_train_np = np.ascontiguousarray(X_train_all.to_numpy())
X_test_np = np.ascontiguousarray(X_test_all.to_numpy())
del X_train_all, X_test_all
gc.collect()


def build_group_indices(type_codes, n_types):
    order = np.argsort(type_codes, kind="mergesort")
    sorted_codes = type_codes[order]
    bounds = np.flatnonzero(np.r_[True, sorted_codes[1:] != sorted_codes[:-1], True])
    starts = bounds[:-1]
    ends = bounds[1:]
    idx_by_type = [np.empty(0, dtype=np.int64) for _ in range(n_types)]
    for s, e in zip(starts, ends):
        code = int(sorted_codes[s])
        idx_by_type[code] = order[s:e]
    return idx_by_type


n_types_total = int(train["type"].cat.categories.size)
tr_idx_by_code = build_group_indices(train_type_cat, n_types_total)
te_idx_by_code = build_group_indices(test_type_cat, n_types_total)

fill_values_by_code = [None] * n_types_total
for code in range(n_types_total):
    tr_idx = tr_idx_by_code[code]
    if tr_idx.size == 0:
        continue
    fill_values_by_code[code] = np.nanmedian(X_train_np[tr_idx], axis=0)

for t in types:
    t_str = str(t)
    t_code = train["type"].cat.categories.get_loc(t)

    tr_idx = tr_idx_by_code[t_code]
    te_idx = te_idx_by_code[t_code]

    if tr_idx.size == 0:
        continue

    fill_vals = fill_values_by_code[t_code]

    X_tr_view = X_train_np[tr_idx]
    nan_mask_tr = np.isnan(X_tr_view)
    if nan_mask_tr.any():
        X_tr = X_tr_view.copy()
        rr, cc = np.nonzero(nan_mask_tr)
        X_tr[rr, cc] = fill_vals[cc]
    else:
        X_tr = X_tr_view

    y_tr = y_train_all[tr_idx]

    X_te_view = X_test_np[te_idx]
    if X_te_view.size:
        nan_mask_te = np.isnan(X_te_view)
        if nan_mask_te.any():
            X_te = X_te_view.copy()
            rr, cc = np.nonzero(nan_mask_te)
            X_te[rr, cc] = fill_vals[cc]
        else:
            X_te = X_te_view
    else:
        X_te = X_te_view

    reg = ensemble.ExtraTreesRegressor(
        n_jobs=-1,
        random_state=4,
        n_estimators=80,
        bootstrap=True,
        min_samples_leaf=2,
    )

    n_val_cap = 250_000
    if tr_idx.size > 10_000:
        n_small = min(n_val_cap, tr_idx.size)
        X_small = X_tr[-n_small:]
        y_small = y_tr[-n_small:]
        x1, x2, y1, y2 = model_selection.train_test_split(
            X_small,
            y_small,
            test_size=0.2,
            random_state=rng_state,
        )
        reg.fit(x1, y1)
        mae = metrics.mean_absolute_error(y2, reg.predict(x2))
        val_scores[t_str] = float(np.log(mae))
        print(f"type={t_str:>4s} log(MAE)={val_scores[t_str]:.5f} (diagnostic)")

    reg.fit(X_tr, y_tr)
    if te_idx.size:
        test_pred[te_idx] = reg.predict(X_te)

n_missing = int(np.isnan(test_pred).sum())
print("Missing test predictions:", n_missing)
if n_missing != 0:
    global_med = float(train["scalar_coupling_constant"].median())
    test_pred[np.isnan(test_pred)] = global_med

test_out = test[["id"]].copy()
test_out["scalar_coupling_constant"] = test_pred

test_out.to_csv("submission.csv", float_format="%.9f", index=False)
print("Wrote submission.csv with shape:", test_out.shape)
print(test_out.head())
