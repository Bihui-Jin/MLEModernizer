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
from sklearn import *

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sub = pd.read_csv("../input/sample_submission.csv")
print(train.shape, test.shape, sub.shape)

train["atom1"] = train["type"].map(lambda x: str(x)[2])
train["atom2"] = train["type"].map(lambda x: str(x)[3])
test["atom1"] = test["type"].map(lambda x: str(x)[2])
test["atom2"] = test["type"].map(lambda x: str(x)[3])

for i in range(4):
    lbl_i = preprocessing.LabelEncoder()
    all_vals = pd.concat(
        [
            train["type"].map(lambda x: str(x)[i]),
            test["type"].map(lambda x: str(x)[i]),
        ],
        axis=0,
        ignore_index=True,
    )
    lbl_i.fit(all_vals)
    train["type" + str(i)] = lbl_i.transform(train["type"].map(lambda x: str(x)[i]))
    test["type" + str(i)] = lbl_i.transform(test["type"].map(lambda x: str(x)[i]))

structures = pd.read_csv("../input/structures.csv").rename(
    columns={
        "atom_index": "atom_index_0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
        "atom": "atom1",
    }
)
train = pd.merge(
    train, structures, how="left", on=["molecule_name", "atom_index_0", "atom1"]
)
test = pd.merge(
    test, structures, how="left", on=["molecule_name", "atom_index_0", "atom1"]
)
del structures

structures = pd.read_csv("../input/structures.csv").rename(
    columns={
        "atom_index": "atom_index_1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
        "atom": "atom2",
    }
)
train = pd.merge(
    train, structures, how="left", on=["molecule_name", "atom_index_1", "atom2"]
)
test = pd.merge(
    test, structures, how="left", on=["molecule_name", "atom_index_1", "atom2"]
)
del structures
print(train.shape, test.shape, sub.shape)



## === cell 1
train_p0 = train[["x0", "y0", "z0"]].values
train_p1 = train[["x1", "y1", "z1"]].values
test_p0 = test[["x0", "y0", "z0"]].values
test_p1 = test[["x1", "y1", "z1"]].values

train["dx"] = train["x0"] - train["x1"]
train["dy"] = train["y0"] - train["y1"]
train["dz"] = train["z0"] - train["z1"]
test["dx"] = test["x0"] - test["x1"]
test["dy"] = test["y0"] - test["y1"]
test["dz"] = test["z0"] - test["z1"]

train["dist2"] = (
    train["dx"] * train["dx"] + train["dy"] * train["dy"] + train["dz"] * train["dz"]
)
test["dist2"] = (
    test["dx"] * test["dx"] + test["dy"] * test["dy"] + test["dz"] * test["dz"]
)

train["dist"] = np.sqrt(train["dist2"])
test["dist"] = np.sqrt(test["dist2"])

train_type_mean = train.groupby("type")["dist"].mean()
global_mean = train["dist"].mean()
eps = 1e-12

train_mean_for_row = train["type"].map(train_type_mean).fillna(global_mean)
test_mean_for_row = test["type"].map(train_type_mean).fillna(global_mean)

train["dist_to_type_mean"] = train["dist"] / (train_mean_for_row + eps)
test["dist_to_type_mean"] = test["dist"] / (test_mean_for_row + eps)

train["inv_dist"] = 1.0 / (train["dist"] + eps)
test["inv_dist"] = 1.0 / (test["dist"] + eps)

train.replace([np.inf, -np.inf], np.nan, inplace=True)
test.replace([np.inf, -np.inf], np.nan, inplace=True)



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
    ]
]

reg = ensemble.ExtraTreesRegressor(n_jobs=-1, n_estimators=20, random_state=4)

x1, x2, y1, y2 = model_selection.train_test_split(
    train[col].tail(2_000_000),
    train["scalar_coupling_constant"].tail(2_000_000),
    test_size=0.2,
    random_state=99,
)

y1_t = np.log1p(y1.values)
y2_t = np.log1p(y2.values)

num_cols = x1.columns[x1.dtypes.apply(lambda dt: np.issubdtype(dt, np.number))]
fill_values = x1[num_cols].median()

x1[num_cols] = x1[num_cols].fillna(fill_values)
x2[num_cols] = x2[num_cols].fillna(fill_values)

test_col = test[col].copy()
test_col[num_cols] = test_col[num_cols].fillna(fill_values)

reg.fit(x1, y1_t)

pred_valid = np.expm1(reg.predict(x2))
valid_df = pd.DataFrame(
    {
        "type": train.loc[x2.index, "type"].values,
        "y_true": y2.values,
        "y_pred": pred_valid,
    }
)
mae_by_type = valid_df.groupby("type").apply(
    lambda g: np.mean(np.abs(g["y_true"] - g["y_pred"]))
)
cv_metric = float(np.mean(np.log(mae_by_type + 1e-12)))
print(cv_metric)

pred_test = np.expm1(reg.predict(test_col))

pred_test = np.where(np.isfinite(pred_test), pred_test, np.nan)
pred_test = np.nan_to_num(pred_test, nan=np.nanmedian(pred_test))
test["scalar_coupling_constant"] = pred_test

test[["id", "scalar_coupling_constant"]].to_csv("submission.csv", index=False)

## --- ERROR in cell 2, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_10/2291079693.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     35[0m [0mtest_col[0m[0;34m[[0m[0mnum_cols[0m[0;34m][0m [0;34m=[0m [0mtest_col[0m[0;34m[[0m[0mnum_cols[0m[0;34m][0m[0;34m.[0m[0mfillna[0m[0;34m([0m[0mfill_values[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     36[0m [0;34m[0m[0m
[0;32m---> 37[0;31m [0mreg[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mx1[0m[0;34m,[0m [0my1_t[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     38[0m [0;34m[0m[0m
[1;32m     39[0m [0;31m# Change (monitoring only): report official-style metric proxy = mean over types of log(MAE).[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_forest.py[0m in [0;36mfit[0;34m(self, X, y, sample_weight)[0m
[1;32m    343[0m         [0;32mif[0m [0missparse[0m[0;34m([0m[0my[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    344[0m             [0;32mraise[0m [0mValueError[0m[0;34m([0m[0;34m"sparse multilabel-indicator for y is not supported."[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 345[0;31m         X, y = self._validate_data(
[0m[1;32m    346[0m             [0mX[0m[0;34m,[0m [0my[0m[0;34m,[0m [0mmulti_output[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m [0maccept_sparse[0m[0;34m=[0m[0;34m"csc"[0m[0;34m,[0m [0mdtype[0m[0;34m=[0m[0mDTYPE[0m[0;34m[0m[0;34m[0m[0m
[1;32m    347[0m         )

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/base.py[0m in [0;36m_validate_data[0;34m(self, X, y, reset, validate_separately, **check_params)[0m
[1;32m    582[0m                 [0my[0m [0;34m=[0m [0mcheck_array[0m[0;34m([0m[0my[0m[0;34m,[0m [0minput_name[0m[0;34m=[0m[0;34m"y"[0m[0;34m,[0m [0;34m**[0m[0mcheck_y_params[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    583[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 584[0;31m                 [0mX[0m[0;34m,[0m [0my[0m [0;34m=[0m [0mcheck_X_y[0m[0;34m([0m[0mX[0m[0;34m,[0m [0my[0m[0;34m,[0m [0;34m**[0m[0mcheck_params[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    585[0m             [0mout[0m [0;34m=[0m [0mX[0m[0;34m,[0m [0my[0m[0;34m[0m[0;34m[0m[0m
[1;32m    586[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py[0m in [0;36mcheck_X_y[0;34m(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)[0m
[1;32m   1120[0m     )
[1;32m   1121[0m [0;34m[0m[0m
[0;32m-> 1122[0;31m     [0my[0m [0;34m=[0m [0m_check_y[0m[0;34m([0m[0my[0m[0;34m,[0m [0mmulti_output[0m[0;34m=[0m[0mmulti_output[0m[0;34m,[0m [0my_numeric[0m[0;34m=[0m[0my_numeric[0m[0;34m,[0m [0mestimator[0m[0;34m=[0m[0mestimator[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1123[0m [0;34m[0m[0m
[1;32m   1124[0m     [0mcheck_consistent_length[0m[0;34m([0m[0mX[0m[0;34m,[0m [0my[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py[0m in [0;36m_check_y[0;34m(y, multi_output, y_numeric, estimator)[0m
[1;32m   1130[0m     [0;34m"""Isolated part of check_X_y dedicated to y validation"""[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1131[0m     [0;32mif[0m [0mmulti_output[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1132[0;31m         y = check_array(
[0m[1;32m   1133[0m             [0my[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1134[0m             [0maccept_sparse[0m[0;34m=[0m[0;34m"csr"[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py[0m in [0;36mcheck_array[0;34m(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)[0m
[1;32m    919[0m [0;34m[0m[0m
[1;32m    920[0m         [0;32mif[0m [0mforce_all_finite[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 921[0;31m             _assert_all_finite(
[0m[1;32m    922[0m                 [0marray[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    923[0m                 [0minput_name[0m[0;34m=[0m[0minput_name[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py[0m in [0;36m_assert_all_finite[0;34m(X, allow_nan, msg_dtype, estimator_name, input_name)[0m
[1;32m    159[0m                 [0;34m"#estimators-that-handle-nan-values"[0m[0;34m[0m[0;34m[0m[0m
[1;32m    160[0m             )
[0;32m--> 161[0;31m         [0;32mraise[0m [0mValueError[0m[0;34m([0m[0mmsg_err[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    162[0m [0;34m[0m[0m
[1;32m    163[0m [0;34m[0m[0m

[0;31mValueError[0m: Input y contains NaN.
