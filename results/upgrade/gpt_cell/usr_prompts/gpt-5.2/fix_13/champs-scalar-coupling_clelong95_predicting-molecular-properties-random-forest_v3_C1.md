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

1.24022

# 6. Current score

2.40564

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 2.40588) has done: 'Diagnosis: The crash in cell 20 occurs because the feature matrix `test` still contains missing values (NaNs), coming from the left-join in `map_atom_info` when atom coordinates/types are not found for some rows. `Ridge` cannot predict with NaNs, so `model.predict(test)` raises `ValueError: Input X contains NaN`. The core model/training logic is fine; the fix is to ensure the post-alignment `test` feature matrix has no NaNs before prediction.

Patch summary: In cell 20, after `test.reindex(columns=X.columns, fill_value=0)`, fill any remaining NaNs with 0 (safe because these represent missing merged features and 0 is already used for absent dummy columns). This keeps the same model and feature schema and only adds minimal preprocessing to unblock prediction.

Updated cells: Only cell 20 is changed.

Compatibility notes for cell k+1: `pred` remains a 1D numpy array aligned with `id_test`, so cell 21 work unchanged.

Assumptions: Replacing NaNs with 0 is acceptable here because NaNs originate from missing merged structure info, and the notebook already uses 0-filling for missing dummy columns; no other missingness handling is intended in the original approach.'
- What this solution (achieved 3.74227) has done: 'You’re currently far from the target (2.40588 vs 1.24022; lower is better), so we need a legitimate accuracy lift without changing the overall approach. The biggest win available while preserving your core pipeline is to evaluate/train in a way that matches the competition metric: the metric is MAE grouped by `type` and then log-averaged, but your CV/tuning is optimizing RMSE on raw targets and ignoring `type`. I keep the same feature engineering (distance + one-hot) and same model family (Ridge), but (1) tune `alpha` using a grouped-by-type MAE scorer, and (2) fit separate Ridge models per coupling `type` (still Ridge; same loss), which usually reduces error a lot because each type has very different target scale. The submission writing be kept compatible (id-aligned, correct columns, `.csv` output).'
- What this solution (achieved 1.99777) has done: 'Main bottlenecks are (1) one-hot encoding `molecule_name` (tens of thousands of unique values) which explodes memory/CPU, and (2) doing many full DataFrame `.iloc` slices and refits across 11 alphas × 3 folds (and again per type) using pandas objects. To keep the exact same modeling logic (Ridge on one-hot encoded categorical features + distance) while making it fast, we switch to sparse one-hot via `OneHotEncoder(handle_unknown="ignore")` and feed Ridge with sparse CSR matrices (provably equivalent to `get_dummies`), and we reuse fold splits and the encoded `type` mask matrix to avoid repeated pandas work. We also avoid re-creating large merged DataFrames repeatedly by merging only needed columns and using `float32` for coordinates/dist (negligible FP diffs) while keeping targets/predictions in float64. Finally, we remove plotting/EDA cells (they cost time and are not part of training/prediction semantics).'
- What this solution (achieved 1.99777) has done: 'The crash happens inside `Ridge.fit()` when it chooses the default sparse solver (`sparse_cg`), which calls SciPy’s `cg()` with a `tol` keyword that isn’t accepted by the SciPy version in this runtime. To make the code deterministic and compatible across SciPy versions, we force Ridge to use a dense/closed-form solver that doesn’t rely on `cg()` when fitting on sparse input. This keeps the same Ridge regression objective and predictions, only changing the numerical solver backend to avoid the incompatible call. The patch is localized to cell 17 and keeps all variable names/outputs used later unchanged.'
- What this solution (achieved 1.99777) has done: 'The crash comes from fitting `Ridge(solver="svd")` on a CSR sparse matrix while keeping the default `fit_intercept=True`; scikit-learn disallows `svd` with intercept on sparse input. The smallest deterministic fix is to keep the exact same model class and CV logic, but switch to a solver that supports sparse + intercept (e.g., `"auto"`). This preserves the training/evaluation semantics and keeps `best_alpha` and `L` computed the same way for downstream cell 18. No other cells need changes.'
- What this solution (achieved 1.99777) has done: 'Diagnosis: The crash happens inside `Ridge.fit` when `solver="auto"` selects the sparse conjugate-gradient path, which calls `scipy.sparse.linalg.cg` with a `tol` keyword that is not accepted by the SciPy version available in this runtime. This is an environment/API incompatibility between scikit-learn’s sparse_cg Ridge solver and SciPy’s `cg` signature. The simplest deterministic fix is to force Ridge to use a solver that does not rely on SciPy’s `cg` for sparse input. We keep the same model family and training/evaluation logic, only changing the solver choice.

Patch summary: In cell 17, change the Ridge instantiation to use `solver="lsqr"` (stable for sparse CSR inputs) and keep all other logic identical.

Updated cells: Only cell 17 is modified.

Compatibility notes for cell k+1: `best_alpha` remains a float computed the same way, so cell 18 can use it unchanged. No variable names, shapes, or downstream interfaces are altered.

Assumptions: Using `solver="lsqr"` is acceptable as it is still Ridge regression with the same objective, and it avoids the SciPy `cg` API mismatch while remaining deterministic under the same random seed.'
- What this solution (achieved 1.99776) has done: 'Diagnosis: The crash happens inside `sklearn.linear_model.Ridge.fit` because the default solver chosen for sparse input ends up calling SciPy’s `sparse.linalg.cg` with a `tol=` keyword that is not accepted by the installed SciPy version in this runtime. This is an upstream compatibility issue between scikit-learn’s `sparse_cg` path and SciPy’s `cg` signature. We can avoid the incompatible code path by explicitly selecting a Ridge solver that does not rely on `cg` for sparse matrices (e.g., `lsqr`), keeping the same Ridge model and training semantics.

Patch summary: In cell 18 only, set `solver="lsqr"` for every `Ridge(...)` instantiation used for fitting on sparse matrices. This keeps the model identical (Ridge regression with the same alpha and random_state) while bypassing the SciPy `cg` call that triggers the TypeError.

Updated cells: (cell 18 only)

Compatibility notes for cell k+1: The variable `pred` is still produced as a NumPy float64 array of length `T_sparse.shape[0]`, so cell 19 continues to work unchanged.

Assumptions: `solver="lsqr"` is supported by the installed scikit-learn (>=1.2.2) and is compatible with CSR sparse matrices.'
- What this solution (achieved 2.40564) has done: 'We’re still far from the target (1.99776 vs 1.24022, lower is better), so the smallest legitimate improvement without changing the core Ridge + OHE approach is to stop leaving test rows unpredicted. Right now, if a coupling `type` appears in test but not in train, those rows remain at the default prediction 0, which badly hurts MAE for those types. I add a simple fallback: also train one global Ridge model on all training data (same features, same best_alpha), and use it to fill predictions for any test rows that didn’t get a per-type model prediction. This preserves the same feature extraction, model family, and training approach while reducing large avoidable errors.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

np.random.seed(42)

INPUT_DIR = "../input"
print(os.listdir(INPUT_DIR))



## === cell 1
structures = pd.read_csv(
    f"{INPUT_DIR}/structures.csv",
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
    dtype={
        "atom_index": np.int16,
        "atom": "category",
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
    },
)
train = pd.read_csv(
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
        "id": np.int32,
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
    },
)
test = pd.read_csv(
    f"{INPUT_DIR}/test.csv",
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
    dtype={
        "id": np.int32,
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
    },
)




## === cell 2
def map_atom_info(df, atom_idx):
    df = pd.merge(
        df,
        structures,
        how="left",
        left_on=["molecule_name", f"atom_index_{atom_idx}"],
        right_on=["molecule_name", "atom_index"],
        copy=False,
    )
    df = df.drop("atom_index", axis=1)
    df = df.rename(
        columns={
            "atom": f"atom_{atom_idx}",
            "x": f"x_{atom_idx}",
            "y": f"y_{atom_idx}",
            "z": f"z_{atom_idx}",
        }
    )
    return df


train = map_atom_info(train, 0)
train = map_atom_info(train, 1)

test = map_atom_info(test, 0)
test = map_atom_info(test, 1)



## === cell 3
dx = (train["x_1"].to_numpy() - train["x_0"].to_numpy()).astype(np.float32, copy=False)
dy = (train["y_1"].to_numpy() - train["y_0"].to_numpy()).astype(np.float32, copy=False)
dz = (train["z_1"].to_numpy() - train["z_0"].to_numpy()).astype(np.float32, copy=False)
train["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)

dx = (test["x_1"].to_numpy() - test["x_0"].to_numpy()).astype(np.float32, copy=False)
dy = (test["y_1"].to_numpy() - test["y_0"].to_numpy()).astype(np.float32, copy=False)
dz = (test["z_1"].to_numpy() - test["z_0"].to_numpy()).astype(np.float32, copy=False)
test["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)



## === cell 4
train = train.drop(["atom_0", "atom_index_1", "atom_index_0"], axis=1)
test = test.drop(["atom_0", "atom_index_1", "atom_index_0"], axis=1)



## === cell 5
pass



## === cell 6
pass



## === cell 7
pass



## === cell 8
pass



## === cell 9
pass



## === cell 10
pass



## === cell 11
pass



## === cell 12
pass



## === cell 13
pass



## === cell 14
pass



## === cell 15
from sklearn.preprocessing import OneHotEncoder

feature_cols = [c for c in train.columns if c not in ["scalar_coupling_constant"]]
X_df = train[feature_cols].copy()
test_df = test.copy()

for c in ["molecule_name", "type", "atom_1"]:
    if c in X_df.columns:
        X_df[c] = X_df[c].astype("category")
    if c in test_df.columns:
        test_df[c] = test_df[c].astype("category")

cat_cols = [c for c in ["molecule_name", "type", "atom_1"] if c in X_df.columns]
num_cols = [c for c in X_df.columns if c not in cat_cols and c != "id"]

X_num = X_df[num_cols].fillna(0.0).to_numpy(dtype=np.float64, copy=False)
T_num = test_df[num_cols].fillna(0.0).to_numpy(dtype=np.float64, copy=False)

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64)
X_cat = ohe.fit_transform(X_df[cat_cols])
T_cat = ohe.transform(test_df[cat_cols])

from scipy import sparse

X_full_sparse = sparse.hstack([sparse.csr_matrix(X_num), X_cat], format="csr")
T_full_sparse = sparse.hstack([sparse.csr_matrix(T_num), T_cat], format="csr")

type_categories = None
if "type" in cat_cols:
    type_categories = list(ohe.categories_[cat_cols.index("type")])
    cat_sizes = [len(cats) for cats in ohe.categories_]
    type_start = int(sum(cat_sizes[: cat_cols.index("type")]))
    type_len = int(cat_sizes[cat_cols.index("type")])
    X_type_ohe = X_cat[:, type_start : type_start + type_len].tocsr()
    T_type_ohe = T_cat[:, type_start : type_start + type_len].tocsr()
else:
    X_type_ohe = None
    T_type_ohe = None

Y = train["scalar_coupling_constant"].to_numpy(dtype=np.float64, copy=False)
id_test = test_df["id"].to_numpy(copy=False)



## === cell 16
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge


def champs_metric_from_arrays(y_true, y_pred, type_onehot_csr, n_types):
    maes = []
    if type_onehot_csr is None or n_types == 0:
        return float(np.log(np.mean(np.abs(y_true - y_pred)) + 1e-9))
    for j in range(n_types):
        col = type_onehot_csr[:, j]
        mask_idx = col.indices
        if mask_idx.size == 0:
            continue
        mae = np.mean(np.abs(y_true[mask_idx] - y_pred[mask_idx]))
        maes.append(mae)
    if len(maes) == 0:
        return float(np.log(np.mean(np.abs(y_true - y_pred)) + 1e-9))
    return float(np.mean(np.log(np.array(maes) + 1e-9)))


def cv_champs_score_grouped_sparse(
    estimator, X_all_csr, y_all, groups, type_onehot_csr, n_types, n_splits=3
):
    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    for tr_idx, va_idx in gkf.split(np.zeros_like(groups), y_all, groups=groups):
        X_tr = X_all_csr[tr_idx]
        y_tr = y_all[tr_idx]
        X_va = X_all_csr[va_idx]
        y_va = y_all[va_idx]

        m = estimator
        m.fit(X_tr, y_tr)
        pred_va = m.predict(X_va)

        type_va = type_onehot_csr[va_idx] if type_onehot_csr is not None else None
        score_va = champs_metric_from_arrays(y_va, pred_va, type_va, n_types=n_types)
        scores.append(score_va)
    return np.array(scores, dtype=np.float64)




## === cell 17
mol_codes = (
    train["molecule_name"]
    .astype("category")
    .cat.codes.to_numpy(dtype=np.int32, copy=False)
)
groups = mol_codes

alpha_list = np.concatenate(
    [
        np.array([1e-4, 3e-4, 1e-3, 3e-3]),
        np.array([1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0]),
    ]
)

n_types = 0 if X_type_ohe is None else X_type_ohe.shape[1]

L = []
for alpha_val in alpha_list:
    model = Ridge(alpha=float(alpha_val), random_state=42, solver="lsqr")
    L.append(
        cv_champs_score_grouped_sparse(
            model,
            X_full_sparse,
            Y,
            groups=groups,
            type_onehot_csr=X_type_ohe,
            n_types=n_types,
            n_splits=3,
        ).mean()
    )

best_alpha = float(alpha_list[int(np.argmin(L))])
print("best_cv_score(logMAE_by_type, GroupKFold by molecule):", float(np.min(L)))
print("best_alpha:", best_alpha)



## === cell 18
cat_cols_nomol = [c for c in ["type", "atom_1"] if c in X_df.columns]
num_cols_nomol = num_cols

ohe2 = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64)
X_cat2 = ohe2.fit_transform(X_df[cat_cols_nomol])
T_cat2 = ohe2.transform(test_df[cat_cols_nomol])

X_sparse = sparse.hstack([sparse.csr_matrix(X_num), X_cat2], format="csr")
T_sparse = sparse.hstack([sparse.csr_matrix(T_num), T_cat2], format="csr")

if "type" in cat_cols_nomol:
    type_categories2 = list(ohe2.categories_[cat_cols_nomol.index("type")])
    cat_sizes2 = [len(cats) for cats in ohe2.categories_]
    type_start2 = int(sum(cat_sizes2[: cat_cols_nomol.index("type")]))
    type_len2 = int(cat_sizes2[cat_cols_nomol.index("type")])
    X_type2 = X_cat2[:, type_start2 : type_start2 + type_len2].tocsr()
    T_type2 = T_cat2[:, type_start2 : type_start2 + type_len2].tocsr()
else:
    type_categories2 = []
    X_type2 = None
    T_type2 = None

global_model = Ridge(alpha=best_alpha, random_state=42, solver="lsqr")
global_model.fit(X_sparse, Y)
global_pred_all = global_model.predict(T_sparse)

if X_type2 is None or X_type2.shape[1] == 0:
    pred = global_pred_all
else:
    pred = np.full(T_sparse.shape[0], np.nan, dtype=np.float64)

    for j in range(X_type2.shape[1]):
        tr_idx = X_type2[:, j].indices
        if tr_idx.size == 0:
            continue
        model = Ridge(alpha=best_alpha, random_state=42, solver="lsqr")
        model.fit(X_sparse[tr_idx], Y[tr_idx])

        te_idx = T_type2[:, j].indices
        if te_idx.size:
            pred[te_idx] = model.predict(T_sparse[te_idx])

    nan_mask = np.isnan(pred)
    if nan_mask.any():
        pred[nan_mask] = global_pred_all[nan_mask]



## === cell 19
test_output = pd.DataFrame({"id": id_test, "scalar_coupling_constant": pred})
test_output.set_index("id", inplace=True)
test_output.to_csv("prediction1.csv")
print("Wrote prediction1.csv with shape:", test_output.shape)
print("Any NaNs in prediction?", bool(np.isnan(pred).any()))
