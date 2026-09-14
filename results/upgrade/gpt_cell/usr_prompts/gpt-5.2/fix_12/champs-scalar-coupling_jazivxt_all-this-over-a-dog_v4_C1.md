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

mol_codes_struct = structures_small["molecule_name"].cat.codes.to_numpy(
    np.int32, copy=False
)
atom_idx_struct = structures_small["atom_index"].to_numpy(np.int16, copy=False)
xyz_struct = structures_small[["x", "y", "z"]].to_numpy(np.float32, copy=False)

coord_index = pd.MultiIndex.from_arrays(
    [mol_codes_struct, atom_idx_struct], names=["mol_code", "atom_index"]
)
coord_table = pd.DataFrame(xyz_struct, index=coord_index, columns=["x", "y", "z"])
coord_table.sort_index(inplace=True)


def attach_coords_fast(df, atom_index_col, prefix, mol_code_series):
    idx = pd.MultiIndex.from_arrays(
        [
            mol_code_series.to_numpy(np.int32, copy=False),
            df[atom_index_col].to_numpy(np.int16, copy=False),
        ],
        names=["mol_code", "atom_index"],
    )
    xyz = coord_table.reindex(idx).to_numpy(dtype=np.float32, copy=False)
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

del (
    structures_small,
    coord_table,
    mol_codes_struct,
    atom_idx_struct,
    xyz_struct,
    coord_index,
)
print(train.shape, test.shape, sub.shape)



## === cell 1
train_p0 = train[["x0", "y0", "z0"]].to_numpy(dtype=np.float32, copy=False)
train_p1 = train[["x1", "y1", "z1"]].to_numpy(dtype=np.float32, copy=False)
test_p0 = test[["x0", "y0", "z0"]].to_numpy(dtype=np.float32, copy=False)
test_p1 = test[["x1", "y1", "z1"]].to_numpy(dtype=np.float32, copy=False)

train_dist = np.linalg.norm(train_p0 - train_p1, axis=1)
test_dist = np.linalg.norm(test_p0 - test_p1, axis=1)
train["dist"] = train_dist
test["dist"] = test_dist

type_mean_dist = train.groupby("type", sort=False)["dist"].mean()
train["dist_to_type_mean"] = (
    train_dist / train["type"].map(type_mean_dist).to_numpy()
).astype(np.float32, copy=False)
test["dist_to_type_mean"] = (
    test_dist / test["type"].map(type_mean_dist).to_numpy()
).astype(np.float32, copy=False)



## === cell 2
col = [
    c
    for c in train.columns
    if c
    not in ["id", "molecule_name", "scalar_coupling_constant", "type", "atom1", "atom2"]
]


def lgb_lmae(preds, dtrain):
    labels = dtrain.get_label()
    score = np.log(metrics.mean_absolute_error(labels, preds))
    return "lmae", score, False


try:
    import multiprocessing as mp

    _cpu = mp.cpu_count()
except Exception:
    _cpu = 4

params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.2,
    "num_leaves": 64,
    "num_threads": int(
        max(1, min(_cpu, 8))
    ),  # practical cap; avoids slowdown on shared CPUs
    "seed": 99,
    "feature_fraction_seed": 99,
    "bagging_seed": 99,
    "data_random_seed": 99,
    "force_col_wise": True,
    "verbosity": -1,
    "histogram_pool_size": 2048,
}

X_train_full = np.ascontiguousarray(train[col].to_numpy(dtype=np.float32, copy=False))
y_train_full = np.ascontiguousarray(
    train["scalar_coupling_constant"].to_numpy(dtype=np.float32, copy=False)
)
X_test_full = np.ascontiguousarray(test[col].to_numpy(dtype=np.float32, copy=False))

pred = np.zeros(len(test), dtype=np.float64)

train_type_codes = train["type"].cat.codes.to_numpy(dtype=np.int16, copy=False)
test_type_codes = test["type"].cat.codes.to_numpy(dtype=np.int16, copy=False)
mol_codes = train_mol_codes_cat.to_numpy(dtype=np.int32, copy=False)

rng = np.random.RandomState(99)

unique_type_codes = np.unique(train_type_codes)

order_tr = np.argsort(train_type_codes, kind="mergesort")
sorted_tc_tr = train_type_codes[order_tr]
bounds_tr = np.flatnonzero(np.r_[True, sorted_tc_tr[1:] != sorted_tc_tr[:-1], True])
type_to_rows_train = {
    int(sorted_tc_tr[bounds_tr[i]]): order_tr[bounds_tr[i] : bounds_tr[i + 1]]
    for i in range(bounds_tr.size - 1)
}

order_te = np.argsort(test_type_codes, kind="mergesort")
sorted_tc_te = test_type_codes[order_te]
bounds_te = np.flatnonzero(np.r_[True, sorted_tc_te[1:] != sorted_tc_te[:-1], True])
type_to_rows_test = {
    int(sorted_tc_te[bounds_te[i]]): order_te[bounds_te[i] : bounds_te[i + 1]]
    for i in range(bounds_te.size - 1)
}

for tc in unique_type_codes:
    trn_rows = type_to_rows_train.get(int(tc))
    tst_rows = type_to_rows_test.get(int(tc))
    if trn_rows is None or tst_rows is None or tst_rows.size == 0:
        continue

    X_all = X_train_full[trn_rows]
    y_all = y_train_full[trn_rows]
    groups = mol_codes[trn_rows]

    uniq_g = np.unique(groups)
    perm = rng.permutation(uniq_g.shape[0])
    n_valid_g = int(np.floor(0.2 * uniq_g.shape[0]))
    valid_g = np.sort(uniq_g[perm[:n_valid_g]])  # sort for fast search

    pos = np.searchsorted(valid_g, groups)
    is_valid = (pos < valid_g.size) & (valid_g[pos] == groups)

    train_idx = np.flatnonzero(~is_valid)
    valid_idx = np.flatnonzero(is_valid)

    x1 = X_all[train_idx]
    y1 = y_all[train_idx]
    x2 = X_all[valid_idx]
    y2 = y_all[valid_idx]

    dtrain = lgb.Dataset(x1, label=y1, free_raw_data=False)
    dvalid = lgb.Dataset(x2, label=y2, reference=dtrain, free_raw_data=False)

    model = lgb.train(
        params,
        dtrain,
        20000,
        valid_sets=[dvalid],
        feval=lgb_lmae,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=200, first_metric_only=False, verbose=True
            ),
            lgb.log_evaluation(period=1000),
        ],
        keep_training_booster=True,
    )

    pred[tst_rows] = model.predict(
        X_test_full[tst_rows], num_iteration=model.best_iteration
    )

test["scalar_coupling_constant"] = pred
out = sub[["id"]].merge(test[["id", "scalar_coupling_constant"]], on="id", how="left")
out[["id", "scalar_coupling_constant"]].to_csv(
    "submission.csv", float_format="%.9f", index=False
)
print(out.shape, out.head())

## --- ERROR in cell 2, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mIndexError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/741409441.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     94[0m [0;34m[0m[0m
[1;32m     95[0m     [0mpos[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0msearchsorted[0m[0;34m([0m[0mvalid_g[0m[0;34m,[0m [0mgroups[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 96[0;31m     [0mis_valid[0m [0;34m=[0m [0;34m([0m[0mpos[0m [0;34m<[0m [0mvalid_g[0m[0;34m.[0m[0msize[0m[0;34m)[0m [0;34m&[0m [0;34m([0m[0mvalid_g[0m[0;34m[[0m[0mpos[0m[0;34m][0m [0;34m==[0m [0mgroups[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     97[0m [0;34m[0m[0m
[1;32m     98[0m     [0mtrain_idx[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mflatnonzero[0m[0;34m([0m[0;34m~[0m[0mis_valid[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mIndexError[0m: index 15248 is out of bounds for axis 0 with size 15248
