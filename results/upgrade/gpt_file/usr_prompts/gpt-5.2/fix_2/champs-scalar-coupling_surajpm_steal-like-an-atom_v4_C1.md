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

# 8. Previous improvement plan

N/A

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




## === cell 1
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print("DATA_PATH:", DATA_PATH)
print("Files:", sorted([f for f in os.listdir(DATA_PATH) if f.endswith(".csv")])[:10])



## === cell 2

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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
for c in ["mol_x_std", "mol_y_std", "mol_z_std"]:
    mol_stats[c] = mol_stats[c].fillna(0.0)

atom_tbl = s.rename(columns={"atom_index": "atom_index"}).copy()
atom_tbl = atom_tbl[["molecule_name", "atom_index", "atom", "x", "y", "z"]]


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
    for c in ["atom_0", "atom_1"]:
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


def fit_predict_variant(num_cols, alpha=1.0, seed=0):
    gkf = GroupKFold(n_splits=3)
    groups = train_f["molecule_name"]
    oof = np.zeros(len(train_f), dtype=np.float32)
    preds_test = np.zeros(len(test_f), dtype=np.float32)

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    model = Ridge(alpha=alpha, random_state=seed)

    pipe = Pipeline([("pre", pre), ("model", model)])

    for tr_idx, va_idx in gkf.split(train_f, y, groups=groups):
        pipe.fit(train_f.iloc[tr_idx], y.iloc[tr_idx])
        oof[va_idx] = pipe.predict(train_f.iloc[va_idx]).astype(np.float32)
        preds_test += pipe.predict(test_f).astype(np.float32) / gkf.n_splits

    mae = mean_absolute_error(y, oof)
    print(f"Variant num_cols={len(num_cols)}, alpha={alpha}: OOF MAE={mae:.6f}")
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



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/20390958.py in <cell line: 0>()
    152 
    153 # Create four prediction sources (sub1..sub4) consistent with original blending downstream
--> 154 pred1 = fit_predict_variant(pos_num, alpha=1.0, seed=0)
    155 pred2 = fit_predict_variant(raw_num, alpha=2.0, seed=1)
    156 pred3 = fit_predict_variant(base_num, alpha=0.5, seed=2)

/tmp/ipykernel_11/20390958.py in fit_predict_variant(num_cols, alpha, seed)
    144         pipe.fit(train_f.iloc[tr_idx], y.iloc[tr_idx])
    145         oof[va_idx] = pipe.predict(train_f.iloc[va_idx]).astype(np.float32)
--> 146         preds_test += pipe.predict(test_f).astype(np.float32) / gkf.n_splits
    147 
    148     mae = mean_absolute_error(y, oof)

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in predict(self, X, **predict_params)
    479         for _, name, transform in self._iter(with_final=False):
    480             Xt = transform.transform(Xt)
--> 481         return self.steps[-1][1].predict(Xt, **predict_params)
    482 
    483     @available_if(_final_estimator_has("fit_predict"))

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py in predict(self, X)
    352             Returns predicted values.
    353         """
--> 354         return self._decision_function(X)
    355 
    356     def _set_intercept(self, X_offset, y_offset, X_scale):

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py in _decision_function(self, X)
    335         check_is_fitted(self)
    336 
--> 337         X = self._validate_data(X, accept_sparse=["csr", "csc", "coo"], reset=False)
    338         return safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
    339 

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    563             raise ValueError("Validation should be done on X, y or both.")
    564         elif not no_val_X and no_val_y:
--> 565             X = check_array(X, input_name="X", **check_params)
    566             out = X
    567         elif no_val_X and not no_val_y:

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

ValueError: Input X contains NaN.
Ridge does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values

## === cell 3
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
)

sub1.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub1.shape)
sub1["scalar_coupling_constant"].describe()

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1279722263.py in <cell line: 0>()
      1 # Fix: ensure blending aligns by id (robust to any row order changes) and always write a valid CSV.
----> 2 sub1 = sub1.sort_values("id").reset_index(drop=True)
      3 sub2 = sub2.sort_values("id").reset_index(drop=True)
      4 sub3 = sub3.sort_values("id").reset_index(drop=True)
      5 sub4 = sub4.sort_values("id").reset_index(drop=True)

NameError: name 'sub1' is not defined
