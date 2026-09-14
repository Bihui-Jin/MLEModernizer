# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.62156

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 2.92781) has done: 'Diagnosis: The crash happens because LightGBM 4.6.0 removed the `early_stopping_rounds` and `verbose_eval` keyword arguments from `lgb.train()`, so passing them now raises `TypeError`. The intended behavior (early stopping + periodic logging) must be expressed via callbacks in current LightGBM.  
Patch summary: In cell 2 only, replace `early_stopping_rounds` and `verbose_eval` arguments with `callbacks=[lgb.early_stopping(...), lgb.log_evaluation(...)]` while keeping the same parameters, datasets, num_boost_round, and custom `feval`. This preserves training semantics and keeps `model.best_iteration` available for the prediction in the next lines.  
Updated cells: Only cell 2 is changed.  
Compatibility notes for cell k+1: No interface changes—`model` remains a `Booster` with `best_iteration` set by early stopping, and `test['scalar_coupling_constant']` is still produced the same way.  
Assumptions: LightGBM’s callback API (`lgb.early_stopping`, `lgb.log_evaluation`) is available in the installed `lightgbm==4.6.0` (it is).'
- What this solution (achieved 1.17616) has done: 'The crash happens because `np.searchsorted(valid_g, groups)` can return an index equal to `valid_g.size` for values larger than the last element, and then `valid_g[pos]` raises an `IndexError`. The intended logic is to mark rows as validation when `groups` is in `valid_g`, but it must avoid indexing `valid_g` with out-of-range positions. I compute `is_valid` in two safe steps: first create a mask of in-range positions, then only index `valid_g` for those positions. This keeps the exact split semantics and only fixes the out-of-bounds access. No other training, parameters, or data processing logic is changed.'

# 9. Code solution

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
    "num_threads": int(max(1, min(_cpu, 8))),
    "seed": 99,
    "feature_fraction_seed": 99,
    "bagging_seed": 99,
    "data_random_seed": 99,
    "force_col_wise": True,
    "verbosity": -1,
    "histogram_pool_size": 2048,
    "feature_pre_filter": True,
}

X_train_full = np.ascontiguousarray(train[col].to_numpy(dtype=np.float32, copy=False))
y_train_full = np.ascontiguousarray(
    train["scalar_coupling_constant"].to_numpy(dtype=np.float32, copy=False)
)
X_test_full = np.ascontiguousarray(test[col].to_numpy(dtype=np.float32, copy=False))

pred = np.zeros(len(test), dtype=np.float64)

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
    valid_g = np.sort(uniq_g[perm[:n_valid_g]])

    pos = np.searchsorted(valid_g, groups)
    in_range = pos < valid_g.size
    is_valid = np.zeros(groups.shape[0], dtype=bool)
    is_valid[in_range] = valid_g[pos[in_range]] == groups[in_range]

    train_idx = np.flatnonzero(~is_valid)
    valid_idx = np.flatnonzero(is_valid)

    x1 = X_all[train_idx]
    y1 = y_all[train_idx]
    x2 = X_all[valid_idx]
    y2 = y_all[valid_idx]

    dtrain = lgb.Dataset(
        x1,
        label=y1,
        free_raw_data=False,
        feature_name=col,
    )
    dvalid = lgb.Dataset(
        x2,
        label=y2,
        reference=dtrain,
        free_raw_data=False,
        feature_name=col,
    )

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
