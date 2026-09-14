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
import warnings
import os

warnings.filterwarnings("ignore")

INPUT_DIR = "../input"



## === cell 1
train_df = pd.read_csv(
    f"{INPUT_DIR}/train.csv",
    usecols=[
        "id",
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "scalar_coupling_constant",
    ],
    dtype={
        "id": "int32",
        "molecule_name": "string",
        "atom_index_0": "int16",
        "atom_index_1": "int16",
        "type": "category",
        "scalar_coupling_constant": "float32",
    },
)
test_df = pd.read_csv(
    f"{INPUT_DIR}/test.csv",
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
    dtype={
        "id": "int32",
        "molecule_name": "string",
        "atom_index_0": "int16",
        "atom_index_1": "int16",
        "type": "category",
    },
)
structures = pd.read_csv(
    f"{INPUT_DIR}/structures.csv",
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
    dtype={
        "molecule_name": "string",
        "atom_index": "int16",
        "atom": "category",
        "x": "float32",
        "y": "float32",
        "z": "float32",
    },
)



## === cell 2
print("Shape of train dataset:", train_df.shape)
print("Shape of test dataset:", test_df.shape)
print("Shape of structures dataset:", structures.shape)



## === cell 3
structures_idx = structures.set_index(["molecule_name", "atom_index"]).sort_index()


def map_atom_data(df, atom_idx):
    key = pd.MultiIndex.from_arrays(
        [df["molecule_name"].to_numpy(), df[f"atom_index_{atom_idx}"].to_numpy()],
        names=["molecule_name", "atom_index"],
    )
    joined = structures_idx.loc[key, ["atom", "x", "y", "z"]].reset_index(drop=True)
    joined = joined.rename(
        columns={
            "atom": f"atom_{atom_idx}",
            "x": f"x_{atom_idx}",
            "y": f"y_{atom_idx}",
            "z": f"z_{atom_idx}",
        }
    )
    return pd.concat([df.reset_index(drop=True), joined], axis=1)


train_df = map_atom_data(train_df, 0)
train_df = map_atom_data(train_df, 1)
test_df = map_atom_data(test_df, 0)
test_df = map_atom_data(test_df, 1)



## --- ERROR in cell 3, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mKeyError[0m                                  Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1622169991.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     23[0m [0mtrain_df[0m [0;34m=[0m [0mmap_atom_data[0m[0;34m([0m[0mtrain_df[0m[0;34m,[0m [0;36m0[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     24[0m [0mtrain_df[0m [0;34m=[0m [0mmap_atom_data[0m[0;34m([0m[0mtrain_df[0m[0;34m,[0m [0;36m1[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 25[0;31m [0mtest_df[0m [0;34m=[0m [0mmap_atom_data[0m[0;34m([0m[0mtest_df[0m[0;34m,[0m [0;36m0[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     26[0m [0mtest_df[0m [0;34m=[0m [0mmap_atom_data[0m[0;34m([0m[0mtest_df[0m[0;34m,[0m [0;36m1[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     27[0m [0;34m[0m[0m

[0;32m/tmp/ipykernel_11/1622169991.py[0m in [0;36mmap_atom_data[0;34m(df, atom_idx)[0m
[1;32m      9[0m         [0mnames[0m[0;34m=[0m[0;34m[[0m[0;34m"molecule_name"[0m[0;34m,[0m [0;34m"atom_index"[0m[0;34m][0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     10[0m     )
[0;32m---> 11[0;31m     [0mjoined[0m [0;34m=[0m [0mstructures_idx[0m[0;34m.[0m[0mloc[0m[0;34m[[0m[0mkey[0m[0;34m,[0m [0;34m[[0m[0;34m"atom"[0m[0;34m,[0m [0;34m"x"[0m[0;34m,[0m [0;34m"y"[0m[0;34m,[0m [0;34m"z"[0m[0;34m][0m[0;34m][0m[0;34m.[0m[0mreset_index[0m[0;34m([0m[0mdrop[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     12[0m     joined = joined.rename(
[1;32m     13[0m         columns={

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py[0m in [0;36m__getitem__[0;34m(self, key)[0m
[1;32m   1182[0m             [0;32mif[0m [0mself[0m[0;34m.[0m[0m_is_scalar_access[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1183[0m                 [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mobj[0m[0;34m.[0m[0m_get_value[0m[0;34m([0m[0;34m*[0m[0mkey[0m[0;34m,[0m [0mtakeable[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0m_takeable[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1184[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_getitem_tuple[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1185[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1186[0m             [0;31m# we by definition only have the 0th axis[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py[0m in [0;36m_getitem_tuple[0;34m(self, tup)[0m
[1;32m   1366[0m         [0;32mwith[0m [0msuppress[0m[0;34m([0m[0mIndexingError[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1367[0m             [0mtup[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_expand_ellipsis[0m[0;34m([0m[0mtup[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1368[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_getitem_lowerdim[0m[0;34m([0m[0mtup[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1369[0m [0;34m[0m[0m
[1;32m   1370[0m         [0;31m# no multi-index, so validate all of the indexers[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py[0m in [0;36m_getitem_lowerdim[0;34m(self, tup)[0m
[1;32m   1039[0m         [0;31m# we may have a nested tuples indexer here[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1040[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0m_is_nested_tuple_indexer[0m[0;34m([0m[0mtup[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1041[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_getitem_nested_tuple[0m[0;34m([0m[0mtup[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1042[0m [0;34m[0m[0m
[1;32m   1043[0m         [0;31m# we maybe be using a tuple to represent multiple dimensions here[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py[0m in [0;36m_getitem_nested_tuple[0;34m(self, tup)[0m
[1;32m   1151[0m                 [0;32mcontinue[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1152[0m [0;34m[0m[0m
[0;32m-> 1153[0;31m             [0mobj[0m [0;34m=[0m [0mgetattr[0m[0;34m([0m[0mobj[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mname[0m[0;34m)[0m[0;34m.[0m[0m_getitem_axis[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0maxis[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1154[0m             [0maxis[0m [0;34m-=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1155[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py[0m in [0;36m_getitem_axis[0;34m(self, key, axis)[0m
[1;32m   1418[0m                     [0;32mraise[0m [0mValueError[0m[0;34m([0m[0;34m"Cannot index with multidimensional key"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1419[0m [0;34m[0m[0m
[0;32m-> 1420[0;31m                 [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_getitem_iterable[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0maxis[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1421[0m [0;34m[0m[0m
[1;32m   1422[0m             [0;31m# nested tuple slicing[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py[0m in [0;36m_getitem_iterable[0;34m(self, key, axis)[0m
[1;32m   1358[0m [0;34m[0m[0m
[1;32m   1359[0m         [0;31m# A collection of keys[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1360[0;31m         [0mkeyarr[0m[0;34m,[0m [0mindexer[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_get_listlike_indexer[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0maxis[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1361[0m         return self.obj._reindex_with_indexers(
[1;32m   1362[0m             [0;34m{[0m[0maxis[0m[0;34m:[0m [0;34m[[0m[0mkeyarr[0m[0;34m,[0m [0mindexer[0m[0;34m][0m[0;34m}[0m[0;34m,[0m [0mcopy[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m [0mallow_dups[0m[0;34m=[0m[0;32mTrue[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py[0m in [0;36m_get_listlike_indexer[0;34m(self, key, axis)[0m
[1;32m   1556[0m         [0maxis_name[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mobj[0m[0;34m.[0m[0m_get_axis_name[0m[0;34m([0m[0maxis[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1557[0m [0;34m[0m[0m
[0;32m-> 1558[0;31m         [0mkeyarr[0m[0;34m,[0m [0mindexer[0m [0;34m=[0m [0max[0m[0;34m.[0m[0m_get_indexer_strict[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0maxis_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1559[0m [0;34m[0m[0m
[1;32m   1560[0m         [0;32mreturn[0m [0mkeyarr[0m[0;34m,[0m [0mindexer[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/multi.py[0m in [0;36m_get_indexer_strict[0;34m(self, key, axis_name)[0m
[1;32m   2764[0m             [0;32mreturn[0m [0mself[0m[0;34m[[0m[0mindexer[0m[0;34m][0m[0;34m,[0m [0mindexer[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2765[0m [0;34m[0m[0m
[0;32m-> 2766[0;31m         [0;32mreturn[0m [0msuper[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0m_get_indexer_strict[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0maxis_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   2767[0m [0;34m[0m[0m
[1;32m   2768[0m     [0;32mdef[0m [0m_raise_if_missing[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mkey[0m[0;34m,[0m [0mindexer[0m[0;34m,[0m [0maxis_name[0m[0;34m:[0m [0mstr[0m[0;34m)[0m [0;34m->[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py[0m in [0;36m_get_indexer_strict[0;34m(self, key, axis_name)[0m
[1;32m   6198[0m             [0mkeyarr[0m[0;34m,[0m [0mindexer[0m[0;34m,[0m [0mnew_indexer[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_reindex_non_unique[0m[0;34m([0m[0mkeyarr[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   6199[0m [0;34m[0m[0m
[0;32m-> 6200[0;31m         [0mself[0m[0;34m.[0m[0m_raise_if_missing[0m[0;34m([0m[0mkeyarr[0m[0;34m,[0m [0mindexer[0m[0;34m,[0m [0maxis_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   6201[0m [0;34m[0m[0m
[1;32m   6202[0m         [0mkeyarr[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mtake[0m[0;34m([0m[0mindexer[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/multi.py[0m in [0;36m_raise_if_missing[0;34m(self, key, indexer, axis_name)[0m
[1;32m   2784[0m                 [0;32mraise[0m [0mKeyError[0m[0;34m([0m[0;34mf"{keyarr} not in index"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2785[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2786[0;31m             [0;32mreturn[0m [0msuper[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0m_raise_if_missing[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0mindexer[0m[0;34m,[0m [0maxis_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   2787[0m [0;34m[0m[0m
[1;32m   2788[0m     [0;32mdef[0m [0m_get_indexer_level_0[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mtarget[0m[0;34m)[0m [0;34m->[0m [0mnpt[0m[0;34m.[0m[0mNDArray[0m[0;34m[[0m[0mnp[0m[0;34m.[0m[0mintp[0m[0;34m][0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py[0m in [0;36m_raise_if_missing[0;34m(self, key, indexer, axis_name)[0m
[1;32m   6247[0m         [0;32mif[0m [0mnmissing[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   6248[0m             [0;32mif[0m [0mnmissing[0m [0;34m==[0m [0mlen[0m[0;34m([0m[0mindexer[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 6249[0;31m                 [0;32mraise[0m [0mKeyError[0m[0;34m([0m[0;34mf"None of [{key}] are in the [{axis_name}]"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   6250[0m [0;34m[0m[0m
[1;32m   6251[0m             [0mnot_found[0m [0;34m=[0m [0mlist[0m[0;34m([0m[0mensure_index[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m[[0m[0mmissing_mask[0m[0;34m.[0m[0mnonzero[0m[0;34m([0m[0;34m)[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m][0m[0;34m.[0m[0munique[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mKeyError[0m: "None of [MultiIndex([('dsgdb9nsd_071451',  9),\n            ('dsgdb9nsd_071451',  9),\n            ('dsgdb9nsd_071451',  9),\n            ('dsgdb9nsd_071451',  9),\n            ('dsgdb9nsd_071451',  9),\n            ('dsgdb9nsd_071451',  9),\n            ('dsgdb9nsd_071451', 10),\n            ('dsgdb9nsd_071451', 10),\n            ('dsgdb9nsd_071451', 10),\n            ('dsgdb9nsd_071451', 10),\n            ...\n            ('dsgdb9nsd_118777', 23),\n            ('dsgdb9nsd_118777', 23),\n            ('dsgdb9nsd_118777', 23),\n            ('dsgdb9nsd_118777', 23),\n            ('dsgdb9nsd_118777', 23),\n            ('dsgdb9nsd_118777', 24),\n            ('dsgdb9nsd_118777', 24),\n            ('dsgdb9nsd_118777', 24),\n            ('dsgdb9nsd_118777', 24),\n            ('dsgdb9nsd_118777', 24)],\n           names=['molecule_name', 'atom_index'], length=467813)] are in the [index]"

## === cell 4
train_m_0 = train_df[["x_0", "y_0", "z_0"]].to_numpy(dtype=np.float32, copy=False)
train_m_1 = train_df[["x_1", "y_1", "z_1"]].to_numpy(dtype=np.float32, copy=False)
test_m_0 = test_df[["x_0", "y_0", "z_0"]].to_numpy(dtype=np.float32, copy=False)
test_m_1 = test_df[["x_1", "y_1", "z_1"]].to_numpy(dtype=np.float32, copy=False)

train_df["dist_vector"] = np.linalg.norm(train_m_0 - train_m_1, axis=1).astype(
    np.float32
)
test_df["dist_vector"] = np.linalg.norm(test_m_0 - test_m_1, axis=1).astype(np.float32)
