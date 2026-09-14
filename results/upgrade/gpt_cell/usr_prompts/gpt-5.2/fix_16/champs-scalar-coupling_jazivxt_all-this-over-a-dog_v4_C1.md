# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.7

# 2. Installed packages

geopandas==0.14.4
lightgbm==4.6.0
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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
import lightgbm as lgb

np.random.seed(99)

BASE = "../input"
if not os.path.exists(BASE):
    if os.path.exists("/kaggle/data/champs-scalar-coupling"):
        BASE = "/kaggle/data/champs-scalar-coupling"
    elif os.path.exists("/kaggle/input/champs-scalar-coupling"):
        BASE = "/kaggle/input/champs-scalar-coupling"
    elif os.path.exists("/kaggle/input"):
        BASE = "/kaggle/input"

train = pd.read_csv(
    os.path.join(BASE, "train.csv"),
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
    os.path.join(BASE, "test.csv"),
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
    },
)
sub = pd.read_csv(os.path.join(BASE, "sample_submission.csv"), dtype={"id": np.int32})
print(train.shape, test.shape, sub.shape)

train_type = train["type"].astype(str)
test_type = test["type"].astype(str)
train["atom1"] = train_type.str[2]
train["atom2"] = train_type.str[3]
test["atom1"] = test_type.str[2]
test["atom2"] = test_type.str[3]

for i in range(4):
    lbl = preprocessing.LabelEncoder()
    all_vals = (
        pd.concat([train_type.str[i], test_type.str[i]], axis=0).astype(str).values
    )
    lbl.fit(all_vals)
    train[f"type{i}"] = lbl.transform(train_type.str[i].astype(str).values).astype(
        np.int8, copy=False
    )
    test[f"type{i}"] = lbl.transform(test_type.str[i].astype(str).values).astype(
        np.int8, copy=False
    )

structures_small = pd.read_csv(
    os.path.join(BASE, "structures.csv"),
    usecols=["molecule_name", "atom_index", "x", "y", "z"],
    dtype={
        "molecule_name": "category",
        "atom_index": np.int16,
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
    },
)

all_mols = pd.Categorical(
    pd.concat(
        [train["molecule_name"].astype(str), test["molecule_name"].astype(str)],
        axis=0,
        ignore_index=True,
    )
)
mol_dtype = pd.CategoricalDtype(categories=all_mols.categories, ordered=False)

train["molecule_name"] = train["molecule_name"].astype(str).astype(mol_dtype)
test["molecule_name"] = test["molecule_name"].astype(str).astype(mol_dtype)
structures_small["molecule_name"] = (
    structures_small["molecule_name"].astype(str).astype(mol_dtype)
)

mol_codes_struct = structures_small["molecule_name"].cat.codes.to_numpy(
    np.int32, copy=False
)
atom_idx_struct = structures_small["atom_index"].to_numpy(np.int16, copy=False)
xyz_struct = structures_small[["x", "y", "z"]].to_numpy(np.float32, copy=False)

ATOM_KEY_BASE = np.int32(128)

struct_keys = mol_codes_struct.astype(np.int64) * int(
    ATOM_KEY_BASE
) + atom_idx_struct.astype(np.int64)
order = np.argsort(struct_keys, kind="mergesort")
struct_keys_sorted = struct_keys[order]
xyz_sorted = xyz_struct[order]


def attach_coords_fast(df, atom_index_col, prefix, mol_code_series):
    keys = mol_code_series.to_numpy(np.int32, copy=False).astype(np.int64) * int(
        ATOM_KEY_BASE
    ) + df[atom_index_col].to_numpy(np.int16, copy=False).astype(np.int64)

    pos = np.searchsorted(struct_keys_sorted, keys)

    in_range = pos < struct_keys_sorted.size
    ok = np.zeros(keys.shape[0], dtype=bool)
    ok[in_range] = struct_keys_sorted[pos[in_range]] == keys[in_range]
    if not np.all(ok):
        bad = np.flatnonzero(~ok)[:5]
        raise KeyError(
            f"Missing structure coordinates for {bad.size} rows (showing up to 5 indices): {bad.tolist()} "
            f"for prefix={prefix}, atom_index_col={atom_index_col}. This would create invalid features."
        )

    xyz = xyz_sorted[pos]
    df[f"x{prefix}"] = xyz[:, 0]
    df[f"y{prefix}"] = xyz[:, 1]
    df[f"z{prefix}"] = xyz[:, 2]
    return df


train_mol_codes_cat = train["molecule_name"].cat.codes
test_mol_codes_cat = test["molecule_name"].cat.codes

train = attach_coords_fast(train, "atom_index_0", "0", train_mol_codes_cat)
test = attach_coords_fast(test, "atom_index_0", "0", test_mol_codes_cat)
train = attach_coords_fast(train, "atom_index_1", "1", train_mol_codes_cat)
test = attach_coords_fast(test, "atom_index_1", "1", test_mol_codes_cat)

del structures_small, mol_codes_struct, atom_idx_struct, xyz_struct, struct_keys, order
print(train.shape, test.shape, sub.shape)



## --- ERROR in cell 0, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mKeyError[0m                                  Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2779366577.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m    135[0m [0;34m[0m[0m
[1;32m    136[0m [0mtrain[0m [0;34m=[0m [0mattach_coords_fast[0m[0;34m([0m[0mtrain[0m[0;34m,[0m [0;34m"atom_index_0"[0m[0;34m,[0m [0;34m"0"[0m[0;34m,[0m [0mtrain_mol_codes_cat[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 137[0;31m [0mtest[0m [0;34m=[0m [0mattach_coords_fast[0m[0;34m([0m[0mtest[0m[0;34m,[0m [0;34m"atom_index_0"[0m[0;34m,[0m [0;34m"0"[0m[0;34m,[0m [0mtest_mol_codes_cat[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    138[0m [0mtrain[0m [0;34m=[0m [0mattach_coords_fast[0m[0;34m([0m[0mtrain[0m[0;34m,[0m [0;34m"atom_index_1"[0m[0;34m,[0m [0;34m"1"[0m[0;34m,[0m [0mtrain_mol_codes_cat[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    139[0m [0mtest[0m [0;34m=[0m [0mattach_coords_fast[0m[0;34m([0m[0mtest[0m[0;34m,[0m [0;34m"atom_index_1"[0m[0;34m,[0m [0;34m"1"[0m[0;34m,[0m [0mtest_mol_codes_cat[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/2779366577.py[0m in [0;36mattach_coords_fast[0;34m(df, atom_index_col, prefix, mol_code_series)[0m
[1;32m    119[0m     [0;32mif[0m [0;32mnot[0m [0mnp[0m[0;34m.[0m[0mall[0m[0;34m([0m[0mok[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    120[0m         [0mbad[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mflatnonzero[0m[0;34m([0m[0;34m~[0m[0mok[0m[0;34m)[0m[0;34m[[0m[0;34m:[0m[0;36m5[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 121[0;31m         raise KeyError(
[0m[1;32m    122[0m             [0;34mf"Missing structure coordinates for {bad.size} rows (showing up to 5 indices): {bad.tolist()} "[0m[0;34m[0m[0;34m[0m[0m
[1;32m    123[0m             [0;34mf"for prefix={prefix}, atom_index_col={atom_index_col}. This would create invalid features."[0m[0;34m[0m[0;34m[0m[0m

[0;31mKeyError[0m: 'Missing structure coordinates for 5 rows (showing up to 5 indices): [0, 1, 2, 3, 4] for prefix=0, atom_index_col=atom_index_0. This would create invalid features.'

## === cell 1
train_p0 = train[["x0", "y0", "z0"]].to_numpy(dtype=np.float32, copy=False)
train_p1 = train[["x1", "y1", "z1"]].to_numpy(dtype=np.float32, copy=False)
test_p0 = test[["x0", "y0", "z0"]].to_numpy(dtype=np.float32, copy=False)
test_p1 = test[["x1", "y1", "z1"]].to_numpy(dtype=np.float32, copy=False)

train_dx = (train_p0[:, 0] - train_p1[:, 0]).astype(np.float32, copy=False)
train_dy = (train_p0[:, 1] - train_p1[:, 1]).astype(np.float32, copy=False)
train_dz = (train_p0[:, 2] - train_p1[:, 2]).astype(np.float32, copy=False)
test_dx = (test_p0[:, 0] - test_p1[:, 0]).astype(np.float32, copy=False)
test_dy = (test_p0[:, 1] - test_p1[:, 1]).astype(np.float32, copy=False)
test_dz = (test_p0[:, 2] - test_p1[:, 2]).astype(np.float32, copy=False)

train["dx"] = train_dx
train["dy"] = train_dy
train["dz"] = train_dz
test["dx"] = test_dx
test["dy"] = test_dy
test["dz"] = test_dz

train_dist2 = (train_dx * train_dx + train_dy * train_dy + train_dz * train_dz).astype(
    np.float32, copy=False
)
test_dist2 = (test_dx * test_dx + test_dy * test_dy + test_dz * test_dz).astype(
    np.float32, copy=False
)

train_dist = np.sqrt(train_dist2, dtype=np.float32)
test_dist = np.sqrt(test_dist2, dtype=np.float32)

train["dist2"] = train_dist2
test["dist2"] = test_dist2
train["dist"] = train_dist
test["dist"] = test_dist

train_type_codes = train["type"].cat.codes.to_numpy(dtype=np.int16, copy=False)
test_type_codes = test["type"].cat.codes.to_numpy(dtype=np.int16, copy=False)

type_mean_dist = train.groupby("type", sort=False)["dist"].mean()
type_std_dist = train.groupby("type", sort=False)["dist"].std().replace(0.0, np.nan)

mean_by_code = type_mean_dist.reindex(train["type"].cat.categories).to_numpy(
    np.float32, copy=False
)
std_by_code = type_std_dist.reindex(train["type"].cat.categories).to_numpy(
    np.float32, copy=False
)

train_mean = mean_by_code[train_type_codes]
test_mean = mean_by_code[test_type_codes]
train_std = std_by_code[train_type_codes]
test_std = std_by_code[test_type_codes]

train["dist_to_type_mean"] = (train_dist / train_mean).astype(np.float32, copy=False)
test["dist_to_type_mean"] = (test_dist / test_mean).astype(np.float32, copy=False)

train["dist_type_z"] = ((train_dist - train_mean) / train_std).astype(
    np.float32, copy=False
)
test["dist_type_z"] = ((test_dist - test_mean) / test_std).astype(
    np.float32, copy=False
)

train["dist_type_z"] = train["dist_type_z"].fillna(0.0).astype(np.float32, copy=False)
test["dist_type_z"] = test["dist_type_z"].fillna(0.0).astype(np.float32, copy=False)
