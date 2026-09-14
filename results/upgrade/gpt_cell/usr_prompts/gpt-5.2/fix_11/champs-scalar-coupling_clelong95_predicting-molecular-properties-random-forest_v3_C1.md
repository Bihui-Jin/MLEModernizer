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
        mask_idx = (
            col.indices
        )  # CSR column slice is CSR; indices correspond to rows with nonzero
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

if X_type2 is None or X_type2.shape[1] == 0:
    model = Ridge(alpha=best_alpha, random_state=42)
    model.fit(X_sparse, Y)
    pred = model.predict(T_sparse)
else:
    pred = np.zeros(T_sparse.shape[0], dtype=np.float64)
    for j in range(X_type2.shape[1]):
        tr_idx = X_type2[:, j].indices
        if tr_idx.size == 0:
            continue
        model = Ridge(alpha=best_alpha, random_state=42)
        model.fit(X_sparse[tr_idx], Y[tr_idx])

        te_idx = T_type2[:, j].indices
        if te_idx.size:
            pred[te_idx] = model.predict(T_sparse[te_idx])



## --- ERROR in cell 18, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py[0m in [0;36m_solve_sparse_cg[0;34m(X, y, alpha, max_iter, tol, verbose, X_offset, X_scale, sample_weight_sqrt)[0m
[1;32m    128[0m             [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 129[0;31m                 coefs[i], info = sp_linalg.cg(
[0m[1;32m    130[0m                     [0mC[0m[0;34m,[0m [0my_column[0m[0;34m,[0m [0mmaxiter[0m[0;34m=[0m[0mmax_iter[0m[0;34m,[0m [0mtol[0m[0;34m=[0m[0mtol[0m[0;34m,[0m [0matol[0m[0;34m=[0m[0;34m"legacy"[0m[0;34m[0m[0;34m[0m[0m

[0;31mTypeError[0m: cg() got an unexpected keyword argument 'tol'

During handling of the above exception, another exception occurred:

[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1907596416.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     39[0m             [0;32mcontinue[0m[0;34m[0m[0;34m[0m[0m
[1;32m     40[0m         [0mmodel[0m [0;34m=[0m [0mRidge[0m[0;34m([0m[0malpha[0m[0;34m=[0m[0mbest_alpha[0m[0;34m,[0m [0mrandom_state[0m[0;34m=[0m[0;36m42[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 41[0;31m         [0mmodel[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mX_sparse[0m[0;34m[[0m[0mtr_idx[0m[0;34m][0m[0;34m,[0m [0mY[0m[0;34m[[0m[0mtr_idx[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     42[0m [0;34m[0m[0m
[1;32m     43[0m         [0mte_idx[0m [0;34m=[0m [0mT_type2[0m[0;34m[[0m[0;34m:[0m[0;34m,[0m [0mj[0m[0;34m][0m[0;34m.[0m[0mindices[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py[0m in [0;36mfit[0;34m(self, X, y, sample_weight)[0m
[1;32m   1132[0m             [0my_numeric[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1133[0m         )
[0;32m-> 1134[0;31m         [0;32mreturn[0m [0msuper[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mX[0m[0;34m,[0m [0my[0m[0;34m,[0m [0msample_weight[0m[0;34m=[0m[0msample_weight[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1135[0m [0;34m[0m[0m
[1;32m   1136[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py[0m in [0;36mfit[0;34m(self, X, y, sample_weight)[0m
[1;32m    898[0m                 [0mparams[0m [0;34m=[0m [0;34m{[0m[0;34m}[0m[0;34m[0m[0;34m[0m[0m
[1;32m    899[0m [0;34m[0m[0m
[0;32m--> 900[0;31m             self.coef_, self.n_iter_ = _ridge_regression(
[0m[1;32m    901[0m                 [0mX[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    902[0m                 [0my[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py[0m in [0;36m_ridge_regression[0;34m(X, y, alpha, sample_weight, solver, max_iter, tol, verbose, positive, random_state, return_n_iter, return_intercept, X_scale, X_offset, check_input, fit_intercept)[0m
[1;32m    669[0m     [0mn_iter[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[1;32m    670[0m     [0;32mif[0m [0msolver[0m [0;34m==[0m [0;34m"sparse_cg"[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 671[0;31m         coef = _solve_sparse_cg(
[0m[1;32m    672[0m             [0mX[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    673[0m             [0my[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py[0m in [0;36m_solve_sparse_cg[0;34m(X, y, alpha, max_iter, tol, verbose, X_offset, X_scale, sample_weight_sqrt)[0m
[1;32m    132[0m             [0;32mexcept[0m [0mTypeError[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    133[0m                 [0;31m# old scipy[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 134[0;31m                 [0mcoefs[0m[0;34m[[0m[0mi[0m[0;34m][0m[0;34m,[0m [0minfo[0m [0;34m=[0m [0msp_linalg[0m[0;34m.[0m[0mcg[0m[0;34m([0m[0mC[0m[0;34m,[0m [0my_column[0m[0;34m,[0m [0mmaxiter[0m[0;34m=[0m[0mmax_iter[0m[0;34m,[0m [0mtol[0m[0;34m=[0m[0mtol[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    135[0m [0;34m[0m[0m
[1;32m    136[0m         [0;32mif[0m [0minfo[0m [0;34m<[0m [0;36m0[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mTypeError[0m: cg() got an unexpected keyword argument 'tol'

## === cell 19
test_output = pd.DataFrame({"id": id_test, "scalar_coupling_constant": pred})
test_output.set_index("id", inplace=True)
test_output.to_csv("prediction1.csv")
print("Wrote prediction1.csv with shape:", test_output.shape)
