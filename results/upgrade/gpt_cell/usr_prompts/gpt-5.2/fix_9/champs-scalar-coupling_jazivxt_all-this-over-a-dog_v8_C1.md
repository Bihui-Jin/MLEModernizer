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
import numpy as np
import pandas as pd
from sklearn import preprocessing, ensemble

np.random.seed(4)

TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SUB_PATH = "../input/sample_submission.csv"
STRUCT_PATH = "../input/structures.csv"

train = pd.read_csv(
    TRAIN_PATH,
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
        "scalar_coupling_constant": np.float64,
    },
)
test = pd.read_csv(
    TEST_PATH,
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
    },
)
sub = pd.read_csv(
    SUB_PATH, dtype={"id": np.int32, "scalar_coupling_constant": np.float64}
)
print(train.shape, test.shape, sub.shape)

train_type_str = train["type"].astype(str)
test_type_str = test["type"].astype(str)

train["atom1"] = train_type_str.str[2]
train["atom2"] = train_type_str.str[3]
test["atom1"] = test_type_str.str[2]
test["atom2"] = test_type_str.str[3]

lbl = preprocessing.LabelEncoder()
for i in range(4):
    trn_ch = train_type_str.str[i]
    tst_ch = test_type_str.str[i]
    train[f"type{i}"] = lbl.fit_transform(trn_ch)
    test[f"type{i}"] = lbl.transform(tst_ch)

structures = pd.read_csv(
    STRUCT_PATH,
    dtype={
        "molecule_name": "category",
        "atom_index": np.int16,
        "atom": "category",
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
    },
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
)

all_molecules = (
    pd.Index(train["molecule_name"].astype(str).to_numpy())
    .append(pd.Index(test["molecule_name"].astype(str).to_numpy()))
    .append(pd.Index(structures["molecule_name"].astype(str).to_numpy()))
    .unique()
)
shared_mol_dtype = pd.api.types.CategoricalDtype(
    categories=all_molecules, ordered=False
)
train["molecule_name"] = train["molecule_name"].astype(str).astype(shared_mol_dtype)
test["molecule_name"] = test["molecule_name"].astype(str).astype(shared_mol_dtype)
structures["molecule_name"] = (
    structures["molecule_name"].astype(str).astype(shared_mol_dtype)
)


def _attach_coords_fast(
    df: pd.DataFrame, idx_col: str, atom_col: str, suffix: str
) -> pd.DataFrame:
    mol_code = df["molecule_name"].cat.codes.to_numpy(np.int32, copy=False)
    atom_idx = df[idx_col].to_numpy(np.int32, copy=False)

    stride = 2048
    key = mol_code.astype(np.int64) * stride + atom_idx.astype(np.int64)

    cache = _attach_coords_fast.__dict__.setdefault("_cache", {})
    if not cache:
        smol = structures["molecule_name"].cat.codes.to_numpy(np.int32, copy=False)
        sidx = structures["atom_index"].to_numpy(np.int32, copy=False)
        skey = smol.astype(np.int64) * stride + sidx.astype(np.int64)

        order = np.argsort(skey, kind="mergesort")
        cache["skey_sorted"] = skey[order]
        cache["atom_sorted"] = structures["atom"].astype(str).to_numpy()[order]
        cache["x_sorted"] = structures["x"].to_numpy(copy=False)[order]
        cache["y_sorted"] = structures["y"].to_numpy(copy=False)[order]
        cache["z_sorted"] = structures["z"].to_numpy(copy=False)[order]

    skey_sorted = cache["skey_sorted"]
    pos = np.searchsorted(skey_sorted, key)

    if len(pos) > 0:
        sample = pos[:: max(1, len(pos) // 16)]
        if not np.all(skey_sorted[sample] == key[:: max(1, len(key) // 16)]):
            raise RuntimeError(
                "Structure lookup key mismatch; cannot attach coords deterministically."
            )

    df[atom_col] = cache["atom_sorted"][pos]
    df[f"x{suffix}"] = cache["x_sorted"][pos]
    df[f"y{suffix}"] = cache["y_sorted"][pos]
    df[f"z{suffix}"] = cache["z_sorted"][pos]
    return df


train = _attach_coords_fast(train, "atom_index_0", "atom1", "0")
test = _attach_coords_fast(test, "atom_index_0", "atom1", "0")
train = _attach_coords_fast(train, "atom_index_1", "atom2", "1")
test = _attach_coords_fast(test, "atom_index_1", "atom2", "1")

del structures

print(train.shape, test.shape, sub.shape)


## --- ERROR in cell 0, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mIndexError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2356725701.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m    123[0m [0;34m[0m[0m
[1;32m    124[0m [0mtrain[0m [0;34m=[0m [0m_attach_coords_fast[0m[0;34m([0m[0mtrain[0m[0;34m,[0m [0;34m"atom_index_0"[0m[0;34m,[0m [0;34m"atom1"[0m[0;34m,[0m [0;34m"0"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 125[0;31m [0mtest[0m [0;34m=[0m [0m_attach_coords_fast[0m[0;34m([0m[0mtest[0m[0;34m,[0m [0;34m"atom_index_0"[0m[0;34m,[0m [0;34m"atom1"[0m[0;34m,[0m [0;34m"0"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    126[0m [0mtrain[0m [0;34m=[0m [0m_attach_coords_fast[0m[0;34m([0m[0mtrain[0m[0;34m,[0m [0;34m"atom_index_1"[0m[0;34m,[0m [0;34m"atom2"[0m[0;34m,[0m [0;34m"1"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    127[0m [0mtest[0m [0;34m=[0m [0m_attach_coords_fast[0m[0;34m([0m[0mtest[0m[0;34m,[0m [0;34m"atom_index_1"[0m[0;34m,[0m [0;34m"atom2"[0m[0;34m,[0m [0;34m"1"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/2356725701.py[0m in [0;36m_attach_coords_fast[0;34m(df, idx_col, atom_col, suffix)[0m
[1;32m    110[0m     [0;32mif[0m [0mlen[0m[0;34m([0m[0mpos[0m[0;34m)[0m [0;34m>[0m [0;36m0[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    111[0m         [0msample[0m [0;34m=[0m [0mpos[0m[0;34m[[0m[0;34m:[0m[0;34m:[0m [0mmax[0m[0;34m([0m[0;36m1[0m[0;34m,[0m [0mlen[0m[0;34m([0m[0mpos[0m[0;34m)[0m [0;34m//[0m [0;36m16[0m[0;34m)[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 112[0;31m         [0;32mif[0m [0;32mnot[0m [0mnp[0m[0;34m.[0m[0mall[0m[0;34m([0m[0mskey_sorted[0m[0;34m[[0m[0msample[0m[0;34m][0m [0;34m==[0m [0mkey[0m[0;34m[[0m[0;34m:[0m[0;34m:[0m [0mmax[0m[0;34m([0m[0;36m1[0m[0;34m,[0m [0mlen[0m[0;34m([0m[0mkey[0m[0;34m)[0m [0;34m//[0m [0;36m16[0m[0;34m)[0m[0;34m][0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    113[0m             raise RuntimeError(
[1;32m    114[0m                 [0;34m"Structure lookup key mismatch; cannot attach coords deterministically."[0m[0;34m[0m[0;34m[0m[0m

[0;31mIndexError[0m: index 1379964 is out of bounds for axis 0 with size 1379964

## === cell 1
train_p0 = train[["x0", "y0", "z0"]].to_numpy(copy=False)
train_p1 = train[["x1", "y1", "z1"]].to_numpy(copy=False)
test_p0 = test[["x0", "y0", "z0"]].to_numpy(copy=False)
test_p1 = test[["x1", "y1", "z1"]].to_numpy(copy=False)

train["dist"] = np.linalg.norm(train_p0 - train_p1, axis=1)
test["dist"] = np.linalg.norm(test_p0 - test_p1, axis=1)

type_mean_train = train.groupby("type")["dist"].mean()
global_mean_train = float(train["dist"].mean())

train["dist_to_type_mean"] = (
    train["dist"] / train["type"].map(type_mean_train).to_numpy()
)

type_mean_train_str = type_mean_train.copy()
type_mean_train_str.index = type_mean_train_str.index.astype(str)
test_type_mean = (
    test["type"].astype(str).map(type_mean_train_str).fillna(global_mean_train)
)
test["dist_to_type_mean"] = test["dist"] / test_type_mean
