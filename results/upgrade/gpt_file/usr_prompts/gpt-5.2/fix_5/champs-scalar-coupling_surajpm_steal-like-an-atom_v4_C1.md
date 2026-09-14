# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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

-1.3233581718076366

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 2.21789) has done: 'I fix the NaN crash by ensuring all features used by the Ridge pipeline are non-missing: `mol_n_atoms` was computed as `max(atom_index)` (which yields NaN for molecules with a single atom) and the various molecule-level means can also become NaN if a merge fails; I correct `mol_n_atoms` to be `max+1`, fill all molecule stats NaNs, and add a minimal `SimpleImputer` for numeric columns inside the pipeline as a final safety net. This is score-neutral in intent (it just makes the model runnable/stable), and it preserves the same model family, CV scheme, and blending logic. I also renumber the notebook cells to start at 1 (your current script starts at cell 0) so it matches the required execution format. Finally, I ensure `submission.csv` is always written with the correct columns and aligned IDs.'
- What this solution (achieved 4.05102) has done: 'Your current public score (2.21789, lower-is-better) is far from the target (-1.323...), so we should improve the model legitimately with minimal disruption to the existing Ridge+OHE+GroupKFold approach. The biggest likely issue vs the competition metric is that you train a single global model but the evaluation is an average of log(MAE) computed *per coupling type*, so we match that by fitting the same Ridge pipeline separately for each `type` and predicting test rows by their type. We also switch the local validation printout from global MAE to a per-type log-MAE (so you can see progress aligned with Kaggle), while keeping everything else (features, Ridge, folds, blending) essentially the same. This tends to materially reduce score for this competition without changing the “core logic” (still Ridge on engineered distance/position stats with OHE and GroupKFold).'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

BASE_PATH_CANDIDATES = [
    "/kaggle/input/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling",
    "/kaggle/data/input/champs-scalar-coupling",
    "../input/champs-scalar-coupling",
]
DATA_PATH = None
for p in BASE_PATH_CANDIDATES:
    if os.path.exists(p) and os.path.exists(os.path.join(p, "train.csv")):
        DATA_PATH = p
        break

if DATA_PATH is None:
    raise FileNotFoundError(
        "Could not locate champs-scalar-coupling data folder. Checked: "
        + ", ".join(BASE_PATH_CANDIDATES)
    )


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast="integer")
        elif pd.api.types.is_float_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast="float")
    return df


print("DATA_PATH:", DATA_PATH)
print("Files:", sorted([f for f in os.listdir(DATA_PATH) if f.endswith(".csv")])[:10])



## === cell 1
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
test = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
structures = pd.read_csv(os.path.join(DATA_PATH, "structures.csv"))

train = reduce_mem_usage(train)
test = reduce_mem_usage(test)
structures = reduce_mem_usage(structures)

s = structures.copy()

mol_stats = (
    s.groupby("molecule_name")
    .agg(
        mol_n_atoms=("atom_index", "max"),
        mol_x_mean=("x", "mean"),
        mol_y_mean=("y", "mean"),
        mol_z_mean=("z", "mean"),
        mol_x_std=("x", "std"),
        mol_y_std=("y", "std"),
        mol_z_std=("z", "std"),
    )
    .reset_index()
)

mol_stats["mol_n_atoms"] = mol_stats["mol_n_atoms"].fillna(0).astype("float32") + 1.0
for c in [
    "mol_x_mean",
    "mol_y_mean",
    "mol_z_mean",
    "mol_x_std",
    "mol_y_std",
    "mol_z_std",
]:
    mol_stats[c] = mol_stats[c].fillna(0.0)

atom_tbl = s[["molecule_name", "atom_index", "atom", "x", "y", "z"]].copy()


def add_atom_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(mol_stats, on="molecule_name", how="left")

    a0 = atom_tbl.rename(
        columns={
            "atom_index": "atom_index_0",
            "atom": "atom_0",
            "x": "x0",
            "y": "y0",
            "z": "z0",
        }
    )
    a1 = atom_tbl.rename(
        columns={
            "atom_index": "atom_index_1",
            "atom": "atom_1",
            "x": "x1",
            "y": "y1",
            "z": "z1",
        }
    )
    df = df.merge(a0, on=["molecule_name", "atom_index_0"], how="left")
    df = df.merge(a1, on=["molecule_name", "atom_index_1"], how="left")

    dx = df["x0"] - df["x1"]
    dy = df["y0"] - df["y1"]
    dz = df["z0"] - df["z1"]
    df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz)

    df["x0_c"] = df["x0"] - df["mol_x_mean"]
    df["y0_c"] = df["y0"] - df["mol_y_mean"]
    df["z0_c"] = df["z0"] - df["mol_z_mean"]
    df["x1_c"] = df["x1"] - df["mol_x_mean"]
    df["y1_c"] = df["y1"] - df["mol_y_mean"]
    df["z1_c"] = df["z1"] - df["mol_z_mean"]

    fill_cols = [
        "dist",
        "mol_n_atoms",
        "mol_x_mean",
        "mol_y_mean",
        "mol_z_mean",
        "mol_x_std",
        "mol_y_std",
        "mol_z_std",
        "x0",
        "y0",
        "z0",
        "x1",
        "y1",
        "z1",
        "x0_c",
        "y0_c",
        "z0_c",
        "x1_c",
        "y1_c",
        "z1_c",
    ]
    for c in fill_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    for c in ["atom_0", "atom_1", "type"]:
        if c in df.columns:
            df[c] = df[c].fillna("X")

    return df


train_f = add_atom_features(train)
test_f = add_atom_features(test)

y = train_f["scalar_coupling_constant"].astype("float32")

base_num = ["dist", "mol_n_atoms", "mol_x_std", "mol_y_std", "mol_z_std"]
pos_num = base_num + ["x0_c", "y0_c", "z0_c", "x1_c", "y1_c", "z1_c"]
raw_num = base_num + ["x0", "y0", "z0", "x1", "y1", "z1"]

cat_cols = ["type", "atom_0", "atom_1"]


def champs_metric(y_true, y_pred, types, eps=1e-9):
    dfm = pd.DataFrame({"y": y_true, "p": y_pred, "t": types})
    maes = dfm.groupby("t").apply(
        lambda g: np.mean(np.abs(g["y"].values - g["p"].values))
    )
    return float(np.mean(np.log(maes.values + eps)))


def fit_predict_variant(num_cols, alpha=1.0, seed=0):
    gkf = GroupKFold(n_splits=3)

    preds_test = np.zeros(len(test_f), dtype=np.float32)
    oof_all = np.zeros(len(train_f), dtype=np.float32)

    cat_cols_local = ["type", "atom_0", "atom_1"]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_local),
            (
                "num",
                Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=0.0))]),
                num_cols,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    model = Ridge(alpha=alpha, random_state=seed)
    pipe = Pipeline([("pre", pre), ("model", model)])

    y_log = np.log1p(y.astype(np.float64)).astype(np.float32)

    for t in sorted(train_f["type"].unique()):
        tr_mask = train_f["type"].values == t
        te_mask = test_f["type"].values == t

        Xtr = train_f.loc[tr_mask]
        ytr = y_log.loc[tr_mask]
        groups = Xtr["molecule_name"]

        oof_t = np.zeros(len(Xtr), dtype=np.float32)
        preds_t = np.zeros(int(te_mask.sum()), dtype=np.float32)

        for tr_idx, va_idx in gkf.split(Xtr, ytr, groups=groups):
            pipe.fit(Xtr.iloc[tr_idx], ytr.iloc[tr_idx])

            oof_t[va_idx] = np.expm1(pipe.predict(Xtr.iloc[va_idx])).astype(np.float32)
            if te_mask.any():
                preds_t += (
                    np.expm1(pipe.predict(test_f.loc[te_mask])).astype(np.float32)
                    / gkf.n_splits
                )

        oof_all[tr_mask] = oof_t
        if te_mask.any():
            preds_test[te_mask] = preds_t

    mae = mean_absolute_error(y, oof_all)
    cmet = champs_metric(y.values, oof_all, train_f["type"].values)
    print(
        f"Variant num_cols={len(num_cols)}, alpha={alpha}: OOF MAE={mae:.6f}, CHAMPS logMAE={cmet:.6f}"
    )
    return preds_test


pred1 = fit_predict_variant(pos_num, alpha=1.0, seed=0)
pred2 = fit_predict_variant(raw_num, alpha=2.0, seed=1)
pred3 = fit_predict_variant(base_num, alpha=0.5, seed=2)
pred4 = fit_predict_variant(
    pos_num + ["x0", "y0", "z0", "x1", "y1", "z1"], alpha=1.5, seed=3
)

sub1 = pd.DataFrame({"id": test["id"].values, "scalar_coupling_constant": pred1})
sub2 = pd.DataFrame({"id": test["id"].values, "scalar_coupling_constant": pred2})
sub3 = pd.DataFrame({"id": test["id"].values, "scalar_coupling_constant": pred3})
sub4 = pd.DataFrame({"id": test["id"].values, "scalar_coupling_constant": pred4})

print(sub1["scalar_coupling_constant"].describe())
print(sub2["scalar_coupling_constant"].describe())
print(sub3["scalar_coupling_constant"].describe())
print(sub4["scalar_coupling_constant"].describe())



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3682899518.py in <cell line: 0>()
    199 
    200 
--> 201 pred1 = fit_predict_variant(pos_num, alpha=1.0, seed=0)
    202 pred2 = fit_predict_variant(raw_num, alpha=2.0, seed=1)
    203 pred3 = fit_predict_variant(base_num, alpha=0.5, seed=2)

/tmp/ipykernel_11/3682899518.py in fit_predict_variant(num_cols, alpha, seed)
    177 
    178         for tr_idx, va_idx in gkf.split(Xtr, ytr, groups=groups):
--> 179             pipe.fit(Xtr.iloc[tr_idx], ytr.iloc[tr_idx])
    180 
    181             # Predict in log-space then invert

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in fit(self, X, y, **fit_params)
    403             if self._final_estimator != "passthrough":
    404                 fit_params_last_step = fit_params_steps[self.steps[-1][0]]
--> 405                 self._final_estimator.fit(Xt, y, **fit_params_last_step)
    406 
    407         return self

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in fit(self, X, y, sample_weight)
   1124 
   1125         _accept_sparse = _get_valid_accept_sparse(sparse.issparse(X), self.solver)
-> 1126         X, y = self._validate_data(
   1127             X,
   1128             y,

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    582                 y = check_array(y, input_name="y", **check_y_params)
    583             else:
--> 584                 X, y = check_X_y(X, y, **check_params)
    585             out = X, y
    586 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_X_y(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)
   1120     )
   1121 
-> 1122     y = _check_y(y, multi_output=multi_output, y_numeric=y_numeric, estimator=estimator)
   1123 
   1124     check_consistent_length(X, y)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in _check_y(y, multi_output, y_numeric, estimator)
   1130     """Isolated part of check_X_y dedicated to y validation"""
   1131     if multi_output:
-> 1132         y = check_array(
   1133             y,
   1134             accept_sparse="csr",

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    919 
    920         if force_all_finite:
--> 921             _assert_all_finite(
    922                 array,
    923                 input_name=input_name,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in _assert_all_finite(X, allow_nan, msg_dtype, estimator_name, input_name)
    159                 "#estimators-that-handle-nan-values"
    160             )
--> 161         raise ValueError(msg_err)
    162 
    163 

ValueError: Input y contains NaN.

## === cell 2
sub1 = sub1.sort_values("id").reset_index(drop=True)
sub2 = sub2.sort_values("id").reset_index(drop=True)
sub3 = sub3.sort_values("id").reset_index(drop=True)
sub4 = sub4.sort_values("id").reset_index(drop=True)

assert np.array_equal(sub1["id"].values, sub2["id"].values)
assert np.array_equal(sub1["id"].values, sub3["id"].values)
assert np.array_equal(sub1["id"].values, sub4["id"].values)

sub1["scalar_coupling_constant"] = (
    0.3 * sub1["scalar_coupling_constant"]
    + 0.3 * sub2["scalar_coupling_constant"]
    + 0.15 * sub3["scalar_coupling_constant"]
    + 0.25 * sub4["scalar_coupling_constant"]
).astype(np.float32)

sub1.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub1.shape)
print(sub1.head())
print(sub1["scalar_coupling_constant"].describe())

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3905307409.py in <cell line: 0>()
----> 1 sub1 = sub1.sort_values("id").reset_index(drop=True)
      2 sub2 = sub2.sort_values("id").reset_index(drop=True)
      3 sub3 = sub3.sort_values("id").reset_index(drop=True)
      4 sub4 = sub4.sort_values("id").reset_index(drop=True)
      5 

NameError: name 'sub1' is not defined
