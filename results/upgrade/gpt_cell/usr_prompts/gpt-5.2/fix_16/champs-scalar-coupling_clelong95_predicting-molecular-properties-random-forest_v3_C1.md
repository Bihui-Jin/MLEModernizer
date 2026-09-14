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

# 5. Code solution

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
eps = np.float32(1e-6)

train["dist2"] = (train["dist"].to_numpy(dtype=np.float32, copy=False) ** 2).astype(
    np.float32, copy=False
)
test["dist2"] = (test["dist"].to_numpy(dtype=np.float32, copy=False) ** 2).astype(
    np.float32, copy=False
)

train["inv_dist"] = (
    1.0 / (train["dist"].to_numpy(dtype=np.float32, copy=False) + eps)
).astype(np.float32, copy=False)
test["inv_dist"] = (
    1.0 / (test["dist"].to_numpy(dtype=np.float32, copy=False) + eps)
).astype(np.float32, copy=False)



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
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse

feature_cols = [c for c in train.columns if c not in ["scalar_coupling_constant"]]
X_df = train[feature_cols].copy()
test_df = test.copy()

for c in ["molecule_name", "type", "atom_0", "atom_1", "atom_index_0", "atom_index_1"]:
    if c in X_df.columns:
        X_df[c] = X_df[c].astype("category")
    if c in test_df.columns:
        test_df[c] = test_df[c].astype("category")

cat_cols = [
    c
    for c in [
        "molecule_name",
        "type",
        "atom_0",
        "atom_1",
        "atom_index_0",
        "atom_index_1",
    ]
    if c in X_df.columns
]
num_cols = [c for c in X_df.columns if c not in cat_cols and c != "id"]

X_num = X_df[num_cols].fillna(0.0).to_numpy(dtype=np.float64, copy=False)
T_num = test_df[num_cols].fillna(0.0).to_numpy(dtype=np.float64, copy=False)

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64)
X_cat = ohe.fit_transform(X_df[cat_cols])
T_cat = ohe.transform(test_df[cat_cols])

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



## === cell 15
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


def cv_champs_score_grouped_sparse_precomputed(
    estimator, X_all_csr, y_all, splits, type_onehot_csr, n_types
):
    scores = []
    for tr_idx, va_idx in splits:
        X_tr = X_all_csr[tr_idx]
        y_tr = y_all[tr_idx]
        X_va = X_all_csr[va_idx]
        y_va = y_all[va_idx]

        estimator.fit(X_tr, y_tr)
        pred_va = estimator.predict(X_va)

        type_va = type_onehot_csr[va_idx] if type_onehot_csr is not None else None
        score_va = champs_metric_from_arrays(y_va, pred_va, type_va, n_types=n_types)
        scores.append(score_va)
    return np.array(scores, dtype=np.float64)




## === cell 16
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

n_types = 0 if X_type_ohe is None else int(X_type_ohe.shape[1])

gkf = GroupKFold(n_splits=3)
splits = list(gkf.split(np.zeros_like(groups), Y, groups=groups))

model = Ridge(alpha=float(alpha_list[0]), random_state=42, solver="lsqr")

L = []
for alpha_val in alpha_list:
    model.set_params(alpha=float(alpha_val))
    L.append(
        cv_champs_score_grouped_sparse_precomputed(
            model,
            X_full_sparse,
            Y,
            splits=splits,
            type_onehot_csr=X_type_ohe,
            n_types=n_types,
        ).mean()
    )

best_alpha = float(alpha_list[int(np.argmin(L))])
print("best_cv_score(logMAE_by_type, GroupKFold by molecule):", float(np.min(L)))
print("best_alpha:", best_alpha)



## === cell 17
cat_cols_nomol = [
    c
    for c in ["type", "atom_0", "atom_1", "atom_index_0", "atom_index_1"]
    if c in X_df.columns
]

ohe2 = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64)
X_cat2 = ohe2.fit_transform(X_df[cat_cols_nomol])
T_cat2 = ohe2.transform(test_df[cat_cols_nomol])

X_num_csr = sparse.csr_matrix(X_num)
T_num_csr = sparse.csr_matrix(T_num)
X_sparse = sparse.hstack([X_num_csr, X_cat2], format="csr")
T_sparse = sparse.hstack([T_num_csr, T_cat2], format="csr")

if "type" in cat_cols_nomol:
    cat_sizes2 = [len(cats) for cats in ohe2.categories_]
    type_start2 = int(sum(cat_sizes2[: cat_cols_nomol.index("type")]))
    type_len2 = int(cat_sizes2[cat_cols_nomol.index("type")])
    X_type2 = X_cat2[:, type_start2 : type_start2 + type_len2].tocsr()
    T_type2 = T_cat2[:, type_start2 : type_start2 + type_len2].tocsr()
else:
    X_type2 = None
    T_type2 = None

global_model = Ridge(alpha=best_alpha, random_state=42, solver="lsqr")
global_model.fit(X_sparse, Y)
global_pred_all = global_model.predict(T_sparse)

if X_type2 is None or X_type2.shape[1] == 0:
    pred = global_pred_all
else:
    pred = np.full(T_sparse.shape[0], np.nan, dtype=np.float64)

    type_model = Ridge(alpha=best_alpha, random_state=42, solver="lsqr")

    for j in range(X_type2.shape[1]):
        tr_idx = X_type2[:, j].indices
        if tr_idx.size == 0:
            continue
        type_model.fit(X_sparse[tr_idx], Y[tr_idx])

        te_idx = T_type2[:, j].indices
        if te_idx.size:
            pred[te_idx] = type_model.predict(T_sparse[te_idx])

    nan_mask = np.isnan(pred)
    if nan_mask.any():
        pred[nan_mask] = global_pred_all[nan_mask]

test_output = pd.DataFrame({"id": id_test, "scalar_coupling_constant": pred})
test_output.set_index("id", inplace=True)
test_output.to_csv("prediction1.csv")
print("Wrote prediction1.csv with shape:", test_output.shape)
print("Any NaNs in prediction?", bool(np.isnan(pred).any()))
