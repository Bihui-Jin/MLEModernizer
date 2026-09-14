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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn import metrics
import lightgbm as lgb

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

np.random.seed(99)


def _resolve_input_path(filename: str) -> str:
    candidates = [
        os.path.join("..", "input", filename),
        os.path.join("..", "input", "champs-scalar-coupling", filename),
        os.path.join("/kaggle", "input", "champs-scalar-coupling", filename),
        os.path.join("/kaggle", "data", "champs-scalar-coupling", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return os.path.join("..", "input", "champs-scalar-coupling", filename)


train_path = _resolve_input_path("train.csv")
test_path = _resolve_input_path("test.csv")
sub_path = _resolve_input_path("sample_submission.csv")
structures_path = _resolve_input_path("structures.csv")

train = pd.read_csv(
    train_path,
    usecols=[
        "id",
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "scalar_coupling_constant",
    ],
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
    test_path,
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
    },
)
sub = pd.read_csv(sub_path)
print(train.shape, test.shape, sub.shape)

train_type = train["type"].astype("string")
test_type = test["type"].astype("string")

train["atom1"] = train_type.str.slice(2, 3)
train["atom2"] = train_type.str.slice(3, 4)
test["atom1"] = test_type.str.slice(2, 3)
test["atom2"] = test_type.str.slice(3, 4)

type_all = pd.concat([train_type, test_type], axis=0, ignore_index=True)

for i in range(4):
    codes, _ = pd.factorize(type_all.str.slice(i, i + 1), sort=False)
    codes = codes.astype(np.int8, copy=False)
    train[f"type{i}"] = codes[: len(train)]
    test[f"type{i}"] = codes[len(train) :]

structures_base = pd.read_csv(
    structures_path,
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

mol_cat = pd.api.types.union_categoricals(
    [
        train["molecule_name"].astype("category"),
        test["molecule_name"].astype("category"),
        structures_base["molecule_name"].astype("category"),
    ],
    sort_categories=False,
)
train["molecule_name"] = train["molecule_name"].cat.set_categories(mol_cat.categories)
test["molecule_name"] = test["molecule_name"].cat.set_categories(mol_cat.categories)
structures_base["molecule_name"] = structures_base["molecule_name"].cat.set_categories(
    mol_cat.categories
)

train["mol_id"] = train["molecule_name"].cat.codes.astype(np.int32, copy=False)
test["mol_id"] = test["molecule_name"].cat.codes.astype(np.int32, copy=False)
structures_base["mol_id"] = structures_base["molecule_name"].cat.codes.astype(
    np.int32, copy=False
)

atom_cat = pd.api.types.union_categoricals(
    [
        train["atom1"].astype("string").astype("category"),
        train["atom2"].astype("string").astype("category"),
        structures_base["atom"].astype("string").astype("category"),
    ],
    sort_categories=False,
)

train["atom1"] = (
    train["atom1"]
    .astype("string")
    .astype("category")
    .cat.set_categories(atom_cat.categories)
)
train["atom2"] = (
    train["atom2"]
    .astype("string")
    .astype("category")
    .cat.set_categories(atom_cat.categories)
)
test["atom1"] = (
    test["atom1"]
    .astype("string")
    .astype("category")
    .cat.set_categories(atom_cat.categories)
)
test["atom2"] = (
    test["atom2"]
    .astype("string")
    .astype("category")
    .cat.set_categories(atom_cat.categories)
)
structures_base["atom"] = (
    structures_base["atom"]
    .astype("string")
    .astype("category")
    .cat.set_categories(atom_cat.categories)
)

train["atom1_id"] = train["atom1"].cat.codes.astype(np.int8, copy=False)
train["atom2_id"] = train["atom2"].cat.codes.astype(np.int8, copy=False)
test["atom1_id"] = test["atom1"].cat.codes.astype(np.int8, copy=False)
test["atom2_id"] = test["atom2"].cat.codes.astype(np.int8, copy=False)
structures_base["atom_id"] = structures_base["atom"].cat.codes.astype(
    np.int8, copy=False
)

structures_base = structures_base.sort_values(
    ["mol_id", "atom_index"], kind="mergesort"
)
coord_index = structures_base.set_index(["mol_id", "atom_index"], drop=False)[
    ["atom_id", "x", "y", "z"]
]


def _attach_coords(
    df: pd.DataFrame, atom_col_prefix: str, atom_index_col: str, atom_id_col: str
) -> None:
    key = pd.MultiIndex.from_arrays(
        [df["mol_id"].to_numpy(copy=False), df[atom_index_col].to_numpy(copy=False)],
        names=["mol_id", "atom_index"],
    )
    got = coord_index.reindex(key)
    ok = got["atom_id"].to_numpy(copy=False) == df[atom_id_col].to_numpy(copy=False)
    x = got["x"].to_numpy(copy=False)
    y = got["y"].to_numpy(copy=False)
    z = got["z"].to_numpy(copy=False)
    df[f"x{atom_col_prefix}"] = np.where(ok, x, np.nan).astype(np.float32, copy=False)
    df[f"y{atom_col_prefix}"] = np.where(ok, y, np.nan).astype(np.float32, copy=False)
    df[f"z{atom_col_prefix}"] = np.where(ok, z, np.nan).astype(np.float32, copy=False)


_attach_coords(train, "0", "atom_index_0", "atom1_id")
_attach_coords(test, "0", "atom_index_0", "atom1_id")
_attach_coords(train, "1", "atom_index_1", "atom2_id")
_attach_coords(test, "1", "atom_index_1", "atom2_id")

del structures_base, coord_index, mol_cat, atom_cat
print(train.shape, test.shape, sub.shape)

missing_coords = train[["x0", "y0", "z0", "x1", "y1", "z1"]].isna().any(axis=1).mean()
print("Train fraction with any missing coords:", float(missing_coords))




## === cell 1
train_p0 = train[["x0", "y0", "z0"]].to_numpy(dtype=np.float32, copy=False)
train_p1 = train[["x1", "y1", "z1"]].to_numpy(dtype=np.float32, copy=False)
test_p0 = test[["x0", "y0", "z0"]].to_numpy(dtype=np.float32, copy=False)
test_p1 = test[["x1", "y1", "z1"]].to_numpy(dtype=np.float32, copy=False)

d = train_p0 - train_p1
train["dist"] = np.sqrt((d * d).sum(axis=1)).astype(np.float32, copy=False)
d = test_p0 - test_p1
test["dist"] = np.sqrt((d * d).sum(axis=1)).astype(np.float32, copy=False)

type_mean_dist = train.groupby("type", sort=False)["dist"].mean()
train["dist_to_type_mean"] = (train["dist"] / train["type"].map(type_mean_dist)).astype(
    np.float32
)
test["dist_to_type_mean"] = (test["dist"] / test["type"].map(type_mean_dist)).astype(
    np.float32
)
test["dist_to_type_mean"] = (
    test["dist_to_type_mean"]
    .fillna(test["dist"] / train["dist"].mean())
    .astype(np.float32)
)




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


cpu_threads = os.cpu_count() or 1
num_threads = int(min(8, max(1, cpu_threads)))

categorical_features = [
    c
    for c in col
    if c in ("mol_id", "atom1_id", "atom2_id", "type0", "type1", "type2", "type3")
]
cat_feature_indices = [col.index(c) for c in categorical_features]

params = {
    "boosting_type": "gbdt",
    "objective": "regression_l1",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "min_data_in_leaf": 64,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "verbosity": -1,
    "num_threads": num_threads,
    "feature_pre_filter": False,
    "seed": 99,
    "feature_fraction_seed": 99,
    "bagging_seed": 99,
    "data_random_seed": 99,
    "deterministic": True,
}

test["scalar_coupling_constant"] = np.nan

callbacks = [
    lgb.early_stopping(stopping_rounds=200),
    lgb.log_evaluation(period=200),
]

all_types = sorted(train["type"].unique().tolist())
print("Training per-type models for", len(all_types), "types:", all_types)

train_type_codes = train["type"].cat.codes.to_numpy(np.int16, copy=False)
test_type_codes = test["type"].cat.codes.to_numpy(np.int16, copy=False)
code_to_type = dict(enumerate(train["type"].cat.categories.tolist()))
type_to_code = {t: i for i, t in code_to_type.items()}


def _build_type_slices(type_codes: np.ndarray):
    order = np.argsort(type_codes, kind="mergesort")
    sorted_codes = type_codes[order]
    uniq, starts = np.unique(sorted_codes, return_index=True)
    ends = np.r_[starts[1:], sorted_codes.size]
    slices = {int(c): order[s:e] for c, s, e in zip(uniq, starts, ends)}
    return slices


train_code_to_idx = _build_type_slices(train_type_codes)
test_code_to_idx = _build_type_slices(test_type_codes)

X_train_all = np.ascontiguousarray(train[col].to_numpy(copy=False))
y_train_all = train["scalar_coupling_constant"].to_numpy(dtype=np.float32, copy=False)
X_test_all = np.ascontiguousarray(test[col].to_numpy(copy=False))
mol_codes_all = train["mol_id"].to_numpy(dtype=np.int32, copy=False)

preds_all = np.full((len(test),), np.nan, dtype=np.float32)

type_train_mean = train.groupby("type", sort=False)["scalar_coupling_constant"].mean()
global_mean = float(train["scalar_coupling_constant"].mean())

rng = np.random.RandomState(99)

row_is_clean = ~np.isnan(X_train_all).any(axis=1)

for t in all_types:
    tcode = type_to_code[t]
    tr_idx = train_code_to_idx.get(tcode, None)
    te_idx = test_code_to_idx.get(tcode, None)
    if te_idx is None or te_idx.size == 0:
        continue
    if tr_idx is None or tr_idx.size == 0:
        fill_val = (
            float(type_train_mean.loc[t]) if t in type_train_mean.index else global_mean
        )
        preds_all[te_idx] = np.float32(fill_val)
        continue

    tr_idx_good = tr_idx[row_is_clean[tr_idx]]
    if tr_idx_good.size < 100:
        fill_val = (
            float(type_train_mean.loc[t]) if t in type_train_mean.index else global_mean
        )
        preds_all[te_idx] = np.float32(fill_val)
        print(
            f"Type={t}: too few clean rows ({tr_idx_good.size}); filled test with type mean."
        )
        continue

    groups = mol_codes_all[tr_idx_good]
    uniq_groups = np.unique(groups)
    perm = rng.permutation(uniq_groups.size)
    n_val_groups = max(1, int(np.ceil(0.2 * uniq_groups.size)))
    val_groups = uniq_groups[perm[:n_val_groups]]

    is_val = np.isin(groups, val_groups, assume_unique=False)

    x_full = X_train_all[tr_idx_good]
    y_full = y_train_all[tr_idx_good]

    dfull = lgb.Dataset(
        x_full,
        label=y_full,
        free_raw_data=False,
        categorical_feature=cat_feature_indices,
    )

    rel_va = np.flatnonzero(is_val).astype(np.int32, copy=False)
    rel_tr = np.flatnonzero(~is_val).astype(np.int32, copy=False)

    dtrain = dfull.subset(rel_tr)
    dvalid = dfull.subset(rel_va)

    model = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=20000,
        valid_sets=[dvalid],
        valid_names=["valid"],
        feval=lgb_lmae,
        callbacks=callbacks,
    )

    preds = model.predict(
        X_test_all[te_idx], num_iteration=model.best_iteration
    ).astype(np.float32, copy=False)
    preds_all[te_idx] = preds
    print(
        f"Finished type={t}: train_rows={tr_idx.size}, train_clean={tr_idx_good.size}, "
        f"test_rows={te_idx.size}, best_iter={model.best_iteration}"
    )

if np.isnan(preds_all).any():
    for t in all_types:
        tcode = type_to_code[t]
        te_idx = test_code_to_idx.get(tcode, None)
        if te_idx is None or te_idx.size == 0:
            continue
        nan_mask = np.isnan(preds_all[te_idx])
        if nan_mask.any():
            fill_val = (
                float(type_train_mean.loc[t])
                if t in type_train_mean.index
                else global_mean
            )
            preds_all[te_idx[nan_mask]] = np.float32(fill_val)

test["scalar_coupling_constant"] = preds_all

submission = test[["id", "scalar_coupling_constant"]].copy()
submission = submission.sort_values("id")
submission.to_csv("submission.csv", float_format="%.9f", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("NaNs in predictions:", int(submission["scalar_coupling_constant"].isna().sum()))
