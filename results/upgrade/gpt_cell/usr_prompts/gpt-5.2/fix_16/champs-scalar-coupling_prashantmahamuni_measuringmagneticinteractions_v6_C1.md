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
import numpy as np
import pandas as pd

import gc
import os

import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics

SEED = 420
np.random.seed(SEED)

pd.options.mode.chained_assignment = None

pd.options.mode.copy_on_write = False



## === cell 1
print(os.listdir("../input"))



## === cell 2
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
structures_dtypes = {
    "molecule_name": "category",
    "atom_index": "int16",
    "atom": "category",
    "x": "float32",
    "y": "float32",
    "z": "float32",
}

train_original = pd.read_csv(
    "../input/train.csv",
    usecols=[
        "id",
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "scalar_coupling_constant",
    ],
    dtype=train_dtypes,
)
structures_original = pd.read_csv(
    "../input/structures.csv",
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
    dtype=structures_dtypes,
)
test_original = pd.read_csv(
    "../input/test.csv",
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
    dtype=test_dtypes,
)
sample_sub = pd.read_csv(
    "../input/sample_submission.csv", usecols=["id"], dtype={"id": "int32"}
)



## === cell 3
_ = train_original.shape
_ = structures_original.shape
_ = test_original.shape



## === cell 4
pass



## === cell 5
moleculeCount = (
    structures_original.groupby(["molecule_name", "atom"], sort=False)
    .size()
    .unstack(fill_value=0)
)

for el in ["C", "F", "H", "N", "O"]:
    if el not in moleculeCount.columns:
        moleculeCount[el] = 0

moleculeCount = moleculeCount[["C", "F", "H", "N", "O"]].astype(np.int16).reset_index()
moleculeCount.head()



## === cell 6
pass



## === cell 7
structures_feat = structures_original.merge(
    moleculeCount, how="left", on="molecule_name", copy=False
)[["molecule_name", "atom_index", "atom", "x", "y", "z", "C", "F", "H", "N", "O"]]

_struct_mol_codes, _struct_mol_uniques = pd.factorize(
    structures_feat["molecule_name"], sort=False
)
_struct_atom_idx = structures_feat["atom_index"].to_numpy(np.int32, copy=False)
_struct_key = (_struct_mol_codes.astype(np.int64) << 20) + _struct_atom_idx.astype(
    np.int64
)
_struct_key_index = pd.Index(_struct_key, name="key")

_struct_atom = structures_feat["atom"].to_numpy(copy=False)
_struct_x = structures_feat["x"].to_numpy(np.float32, copy=False)
_struct_y = structures_feat["y"].to_numpy(np.float32, copy=False)
_struct_z = structures_feat["z"].to_numpy(np.float32, copy=False)
_struct_C = structures_feat["C"].to_numpy(np.int16, copy=False)
_struct_F = structures_feat["F"].to_numpy(np.int16, copy=False)
_struct_H = structures_feat["H"].to_numpy(np.int16, copy=False)
_struct_N = structures_feat["N"].to_numpy(np.int16, copy=False)
_struct_O = structures_feat["O"].to_numpy(np.int16, copy=False)

_struct_mol_map = pd.Series(
    np.arange(len(_struct_mol_uniques), dtype=np.int32), index=_struct_mol_uniques
)


def _attach_atom_side(df_base, atom_col, prefix, need_counts):
    mol_codes = _struct_mol_map.loc[df_base["molecule_name"]].to_numpy(
        np.int32, copy=False
    )
    atom_idx = df_base[atom_col].to_numpy(np.int32, copy=False)
    keys = (mol_codes.astype(np.int64) << 20) + atom_idx.astype(np.int64)

    pos = _struct_key_index.get_indexer(keys)
    if (pos < 0).any():
        raise KeyError(
            "Some (molecule_name, atom_index) pairs were not found in structures."
        )

    out = {
        f"atom_nm_{prefix}": _struct_atom.take(pos),
        f"x_{prefix}": _struct_x.take(pos),
        f"y_{prefix}": _struct_y.take(pos),
        f"z_{prefix}": _struct_z.take(pos),
    }
    if need_counts:
        out["C"] = _struct_C.take(pos)
        out["F"] = _struct_F.take(pos)
        out["H"] = _struct_H.take(pos)
        out["N"] = _struct_N.take(pos)
        out["O"] = _struct_O.take(pos)
    return pd.DataFrame(out)


side0 = _attach_atom_side(train_original, "atom_index_0", "0", need_counts=True)
side1 = _attach_atom_side(train_original, "atom_index_1", "1", need_counts=False)

train = pd.concat([train_original.reset_index(drop=True), side0, side1], axis=1)

train = train[
    [
        "id",
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "atom_nm_0",
        "x_0",
        "y_0",
        "z_0",
        "atom_nm_1",
        "x_1",
        "y_1",
        "z_1",
        "C",
        "F",
        "H",
        "N",
        "O",
        "scalar_coupling_constant",
    ]
].rename(columns={"atom_index_0": "atom_0", "atom_index_1": "atom_1"})

train.reset_index(inplace=True, drop=True)

del side0, side1
gc.collect()



## === cell 8

try:
    _missing = ~test_original["molecule_name"].isin(_struct_mol_map.index)
    if bool(_missing.any()):
        structures_original = pd.read_csv(
            "../input/champs-scalar-coupling/structures.csv",
            usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
            dtype=structures_dtypes,
        )

        moleculeCount = (
            structures_original.groupby(["molecule_name", "atom"], sort=False)
            .size()
            .unstack(fill_value=0)
        )

        for el in ["C", "F", "H", "N", "O"]:
            if el not in moleculeCount.columns:
                moleculeCount[el] = 0

        moleculeCount = (
            moleculeCount[["C", "F", "H", "N", "O"]].astype(np.int16).reset_index()
        )

        structures_feat = structures_original.merge(
            moleculeCount, how="left", on="molecule_name", copy=False
        )[
            [
                "molecule_name",
                "atom_index",
                "atom",
                "x",
                "y",
                "z",
                "C",
                "F",
                "H",
                "N",
                "O",
            ]
        ]

        _struct_mol_codes, _struct_mol_uniques = pd.factorize(
            structures_feat["molecule_name"], sort=False
        )
        _struct_atom_idx = structures_feat["atom_index"].to_numpy(np.int32, copy=False)
        _struct_key = (
            _struct_mol_codes.astype(np.int64) << 20
        ) + _struct_atom_idx.astype(np.int64)
        _struct_key_index = pd.Index(_struct_key, name="key")

        _struct_atom = structures_feat["atom"].to_numpy(copy=False)
        _struct_x = structures_feat["x"].to_numpy(np.float32, copy=False)
        _struct_y = structures_feat["y"].to_numpy(np.float32, copy=False)
        _struct_z = structures_feat["z"].to_numpy(np.float32, copy=False)
        _struct_C = structures_feat["C"].to_numpy(np.int16, copy=False)
        _struct_F = structures_feat["F"].to_numpy(np.int16, copy=False)
        _struct_H = structures_feat["H"].to_numpy(np.int16, copy=False)
        _struct_N = structures_feat["N"].to_numpy(np.int16, copy=False)
        _struct_O = structures_feat["O"].to_numpy(np.int16, copy=False)

        _struct_mol_map = pd.Series(
            np.arange(len(_struct_mol_uniques), dtype=np.int32),
            index=_struct_mol_uniques,
        )
finally:
    pass


def _attach_atom_side(df_base, atom_col, prefix, need_counts):
    mol_names = df_base["molecule_name"].astype("object").to_numpy(copy=False)
    mol_codes = _struct_mol_map.reindex(mol_names).to_numpy(np.float32, copy=False)
    if np.isnan(mol_codes).any():
        missing = pd.unique(mol_names[np.isnan(mol_codes)])
        raise KeyError(
            f"Some molecule_name values were not found in structures mapping: {missing[:10]}"
        )
    mol_codes = mol_codes.astype(np.int32, copy=False)

    atom_idx = df_base[atom_col].to_numpy(np.int32, copy=False)
    keys = (mol_codes.astype(np.int64) << 20) + atom_idx.astype(np.int64)

    pos = _struct_key_index.get_indexer(keys)
    if (pos < 0).any():
        raise KeyError(
            "Some (molecule_name, atom_index) pairs were not found in structures."
        )

    out = {
        f"atom_nm_{prefix}": _struct_atom.take(pos),
        f"x_{prefix}": _struct_x.take(pos),
        f"y_{prefix}": _struct_y.take(pos),
        f"z_{prefix}": _struct_z.take(pos),
    }
    if need_counts:
        out["C"] = _struct_C.take(pos)
        out["F"] = _struct_F.take(pos)
        out["H"] = _struct_H.take(pos)
        out["N"] = _struct_N.take(pos)
        out["O"] = _struct_O.take(pos)
    return pd.DataFrame(out)


side0_t = _attach_atom_side(test_original, "atom_index_0", "0", need_counts=True)
side1_t = _attach_atom_side(test_original, "atom_index_1", "1", need_counts=False)

test = pd.concat([test_original.reset_index(drop=True), side0_t, side1_t], axis=1)

test = test[
    [
        "id",
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "atom_nm_0",
        "x_0",
        "y_0",
        "z_0",
        "atom_nm_1",
        "x_1",
        "y_1",
        "z_1",
        "C",
        "F",
        "H",
        "N",
        "O",
    ]
].rename(columns={"atom_index_0": "atom_0", "atom_index_1": "atom_1"})

test.reset_index(inplace=True, drop=True)

del side0_t, side1_t
gc.collect()


## --- ERROR in cell 8, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mKeyError[0m                                  Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/837941271.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m    103[0m [0;34m[0m[0m
[1;32m    104[0m [0;34m[0m[0m
[0;32m--> 105[0;31m [0mside0_t[0m [0;34m=[0m [0m_attach_atom_side[0m[0;34m([0m[0mtest_original[0m[0;34m,[0m [0;34m"atom_index_0"[0m[0;34m,[0m [0;34m"0"[0m[0;34m,[0m [0mneed_counts[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    106[0m [0mside1_t[0m [0;34m=[0m [0m_attach_atom_side[0m[0;34m([0m[0mtest_original[0m[0;34m,[0m [0;34m"atom_index_1"[0m[0;34m,[0m [0;34m"1"[0m[0;34m,[0m [0mneed_counts[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    107[0m [0;34m[0m[0m

[0;32m/tmp/ipykernel_11/837941271.py[0m in [0;36m_attach_atom_side[0;34m(df_base, atom_col, prefix, need_counts)[0m
[1;32m     74[0m     [0;32mif[0m [0mnp[0m[0;34m.[0m[0misnan[0m[0;34m([0m[0mmol_codes[0m[0;34m)[0m[0;34m.[0m[0many[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     75[0m         [0mmissing[0m [0;34m=[0m [0mpd[0m[0;34m.[0m[0munique[0m[0;34m([0m[0mmol_names[0m[0;34m[[0m[0mnp[0m[0;34m.[0m[0misnan[0m[0;34m([0m[0mmol_codes[0m[0;34m)[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 76[0;31m         raise KeyError(
[0m[1;32m     77[0m             [0;34mf"Some molecule_name values were not found in structures mapping: {missing[:10]}"[0m[0;34m[0m[0;34m[0m[0m
[1;32m     78[0m         )

[0;31mKeyError[0m: "Some molecule_name values were not found in structures mapping: ['dsgdb9nsd_071451' 'dsgdb9nsd_004012' 'dsgdb9nsd_108451'\n 'dsgdb9nsd_094734' 'dsgdb9nsd_014342' 'dsgdb9nsd_019598'\n 'dsgdb9nsd_026320' 'dsgdb9nsd_065187' 'dsgdb9nsd_017640'\n 'dsgdb9nsd_098435']"

## === cell 9
train_original = None
del train_original
structures_original = None
del structures_original
test_original = None
del test_original
moleculeCount = None
del moleculeCount

structures_feat = None
del structures_feat

_struct_key_index = None
del _struct_key_index
_struct_atom = None
_struct_x = None
_struct_y = None
_struct_z = None
_struct_C = None
_struct_F = None
_struct_H = None
_struct_N = None
_struct_O = None
_struct_mol_map = None
_struct_mol_codes = None
_struct_mol_uniques = None
_struct_atom_idx = None
_struct_key = None
del (
    _struct_atom,
    _struct_x,
    _struct_y,
    _struct_z,
    _struct_C,
    _struct_F,
    _struct_H,
    _struct_N,
    _struct_O,
)
del (
    _struct_mol_map,
    _struct_mol_codes,
    _struct_mol_uniques,
    _struct_atom_idx,
    _struct_key,
)

gc.collect()
