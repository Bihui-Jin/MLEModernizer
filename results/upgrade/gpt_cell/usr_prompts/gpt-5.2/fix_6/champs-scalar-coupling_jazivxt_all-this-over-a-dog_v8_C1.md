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

sidx = pd.MultiIndex.from_frame(structures[["molecule_name", "atom_index"]])
struct_atom = pd.Series(
    structures["atom"].astype(str).to_numpy(), index=sidx, name="atom"
)
struct_xyz = structures[["x", "y", "z"]].to_numpy()
struct_x = pd.Series(struct_xyz[:, 0], index=sidx, name="x")
struct_y = pd.Series(struct_xyz[:, 1], index=sidx, name="y")
struct_z = pd.Series(struct_xyz[:, 2], index=sidx, name="z")
del structures, struct_xyz


def attach_atom_coords(
    df: pd.DataFrame, idx_col: str, atom_col: str, suffix: str
) -> pd.DataFrame:
    key = pd.MultiIndex.from_frame(df[["molecule_name", idx_col]])
    df[atom_col] = struct_atom.reindex(key).to_numpy()
    df[f"x{suffix}"] = struct_x.reindex(key).to_numpy()
    df[f"y{suffix}"] = struct_y.reindex(key).to_numpy()
    df[f"z{suffix}"] = struct_z.reindex(key).to_numpy()
    return df


train = attach_atom_coords(train, "atom_index_0", "atom1", "0")
test = attach_atom_coords(test, "atom_index_0", "atom1", "0")
train = attach_atom_coords(train, "atom_index_1", "atom2", "1")
test = attach_atom_coords(test, "atom_index_1", "atom2", "1")

print(train.shape, test.shape, sub.shape)



## === cell 1
train_p0 = train[["x0", "y0", "z0"]].to_numpy(copy=False)
train_p1 = train[["x1", "y1", "z1"]].to_numpy(copy=False)
test_p0 = test[["x0", "y0", "z0"]].to_numpy(copy=False)
test_p1 = test[["x1", "y1", "z1"]].to_numpy(copy=False)

train["dist"] = np.linalg.norm(train_p0 - train_p1, axis=1)
test["dist"] = np.linalg.norm(test_p0 - test_p1, axis=1)

train["dist_to_type_mean"] = train["dist"] / train.groupby("type")["dist"].transform(
    "mean"
)

type_mean_train = train.groupby("type")["dist"].mean()
global_mean_train = train["dist"].mean()
test_type_mean = test["type"].map(type_mean_train).fillna(global_mean_train)
test["dist_to_type_mean"] = test["dist"] / test_type_mean



## --- ERROR in cell 1, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1994924520.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     14[0m [0mtype_mean_train[0m [0;34m=[0m [0mtrain[0m[0;34m.[0m[0mgroupby[0m[0;34m([0m[0;34m"type"[0m[0;34m)[0m[0;34m[[0m[0;34m"dist"[0m[0;34m][0m[0;34m.[0m[0mmean[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     15[0m [0mglobal_mean_train[0m [0;34m=[0m [0mtrain[0m[0;34m[[0m[0;34m"dist"[0m[0;34m][0m[0;34m.[0m[0mmean[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 16[0;31m [0mtest_type_mean[0m [0;34m=[0m [0mtest[0m[0;34m[[0m[0;34m"type"[0m[0;34m][0m[0;34m.[0m[0mmap[0m[0;34m([0m[0mtype_mean_train[0m[0;34m)[0m[0;34m.[0m[0mfillna[0m[0;34m([0m[0mglobal_mean_train[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     17[0m [0mtest[0m[0;34m[[0m[0;34m"dist_to_type_mean"[0m[0;34m][0m [0;34m=[0m [0mtest[0m[0;34m[[0m[0;34m"dist"[0m[0;34m][0m [0;34m/[0m [0mtest_type_mean[0m[0;34m[0m[0;34m[0m[0m
[1;32m     18[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py[0m in [0;36mfillna[0;34m(self, value, method, axis, inplace, limit, downcast)[0m
[1;32m   7347[0m                     )
[1;32m   7348[0m [0;34m[0m[0m
[0;32m-> 7349[0;31m                 new_data = self._mgr.fillna(
[0m[1;32m   7350[0m                     [0mvalue[0m[0;34m=[0m[0mvalue[0m[0;34m,[0m [0mlimit[0m[0;34m=[0m[0mlimit[0m[0;34m,[0m [0minplace[0m[0;34m=[0m[0minplace[0m[0;34m,[0m [0mdowncast[0m[0;34m=[0m[0mdowncast[0m[0;34m[0m[0;34m[0m[0m
[1;32m   7351[0m                 )

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/internals/base.py[0m in [0;36mfillna[0;34m(self, value, limit, inplace, downcast)[0m
[1;32m    184[0m             [0mlimit[0m [0;34m=[0m [0mlibalgos[0m[0;34m.[0m[0mvalidate_limit[0m[0;34m([0m[0;32mNone[0m[0;34m,[0m [0mlimit[0m[0;34m=[0m[0mlimit[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    185[0m [0;34m[0m[0m
[0;32m--> 186[0;31m         return self.apply_with_block(
[0m[1;32m    187[0m             [0;34m"fillna"[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    188[0m             [0mvalue[0m[0;34m=[0m[0mvalue[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py[0m in [0;36mapply[0;34m(self, f, align_keys, **kwargs)[0m
[1;32m    361[0m                 [0mapplied[0m [0;34m=[0m [0mb[0m[0;34m.[0m[0mapply[0m[0;34m([0m[0mf[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    362[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 363[0;31m                 [0mapplied[0m [0;34m=[0m [0mgetattr[0m[0;34m([0m[0mb[0m[0;34m,[0m [0mf[0m[0;34m)[0m[0;34m([0m[0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    364[0m             [0mresult_blocks[0m [0;34m=[0m [0mextend_blocks[0m[0;34m([0m[0mapplied[0m[0;34m,[0m [0mresult_blocks[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    365[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py[0m in [0;36mfillna[0;34m(self, value, limit, inplace, downcast, using_cow, already_warned)[0m
[1;32m   2332[0m                 [0;31m# 3rd party EA that has not implemented copy keyword yet[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2333[0m                 [0mrefs[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2334[0;31m                 [0mnew_values[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mvalues[0m[0;34m.[0m[0mfillna[0m[0;34m([0m[0mvalue[0m[0;34m=[0m[0mvalue[0m[0;34m,[0m [0mmethod[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m [0mlimit[0m[0;34m=[0m[0mlimit[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   2335[0m                 [0;31m# issue the warning *after* retrying, in case the TypeError[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2336[0m                 [0;31m#  was caused by an invalid fill_value[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/_mixins.py[0m in [0;36mfillna[0;34m(self, value, method, limit, copy)[0m
[1;32m    374[0m             [0;31m# We validate the fill_value even if there is nothing to fill[0m[0;34m[0m[0;34m[0m[0m
[1;32m    375[0m             [0;32mif[0m [0mvalue[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 376[0;31m                 [0mself[0m[0;34m.[0m[0m_validate_setitem_value[0m[0;34m([0m[0mvalue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    377[0m [0;34m[0m[0m
[1;32m    378[0m             [0;32mif[0m [0;32mnot[0m [0mcopy[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/categorical.py[0m in [0;36m_validate_setitem_value[0;34m(self, value)[0m
[1;32m   1587[0m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_validate_listlike[0m[0;34m([0m[0mvalue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1588[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1589[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_validate_scalar[0m[0;34m([0m[0mvalue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1590[0m [0;34m[0m[0m
[1;32m   1591[0m     [0;32mdef[0m [0m_validate_scalar[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mfill_value[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/categorical.py[0m in [0;36m_validate_scalar[0;34m(self, fill_value)[0m
[1;32m   1612[0m             [0mfill_value[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_unbox_scalar[0m[0;34m([0m[0mfill_value[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1613[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1614[0;31m             raise TypeError(
[0m[1;32m   1615[0m                 [0;34m"Cannot setitem on a Categorical with a new "[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1616[0m                 [0;34mf"category ({fill_value}), set the categories first"[0m[0;34m[0m[0;34m[0m[0m

[0;31mTypeError[0m: Cannot setitem on a Categorical with a new category (2.360938549041748), set the categories first

## === cell 2
col = [
    c
    for c in train.columns
    if c
    not in [
        "id",
        "molecule_name",
        "scalar_coupling_constant",
        "type",
        "atom1",
        "atom2",
        "atom_index_0",
        "atom_index_1",
    ]
]

X_train_all = train[col].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
y_train_all = train["scalar_coupling_constant"].to_numpy()
X_test_all = test[col].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()

rng = np.random.RandomState(4)
mols = train["molecule_name"].unique()
rng.shuffle(mols)
val_mols = set(mols[: max(1, int(0.05 * len(mols)))])
is_val = train["molecule_name"].isin(val_mols).to_numpy()
is_trn = ~is_val

test_pred = np.empty(len(test), dtype=np.float64)

n_estimators_grid = [50, 100]

train_type_arr = train["type"].astype(str).to_numpy()
test_type_arr = test["type"].astype(str).to_numpy()

for t in pd.unique(train_type_arr):
    trn_mask = (train_type_arr == t) & is_trn
    val_mask = (train_type_arr == t) & is_val
    te_mask = test_type_arr == t

    X_trn = X_train_all[trn_mask]
    y_trn = y_train_all[trn_mask]

    best_n = n_estimators_grid[-1]
    if val_mask.sum() > 0 and trn_mask.sum() > 0:
        X_val = X_train_all[val_mask]
        y_val = y_train_all[val_mask]

        best_mae = np.inf
        for n_est in n_estimators_grid:
            reg = ensemble.ExtraTreesRegressor(
                n_jobs=-1, n_estimators=n_est, random_state=4
            )
            reg.fit(X_trn, y_trn)
            pred_val = reg.predict(X_val)
            mae = np.mean(np.abs(pred_val - y_val))
            if mae < best_mae:
                best_mae = mae
                best_n = n_est

    full_mask = train_type_arr == t
    X_full = X_train_all[full_mask]
    y_full = y_train_all[full_mask]
    X_te = X_test_all[te_mask]

    reg = ensemble.ExtraTreesRegressor(n_jobs=-1, n_estimators=best_n, random_state=4)
    reg.fit(X_full, y_full)
    test_pred[te_mask] = reg.predict(X_te)

test["scalar_coupling_constant"] = test_pred
test[["id", "scalar_coupling_constant"]].to_csv("submission.csv", index=False)
print(
    "Wrote submission.csv with shape:", test[["id", "scalar_coupling_constant"]].shape
)
