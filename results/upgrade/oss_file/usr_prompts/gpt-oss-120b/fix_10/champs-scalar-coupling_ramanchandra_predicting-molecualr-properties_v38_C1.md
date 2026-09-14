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

1.29765

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 1.99777) has done: 'I fill missing values in the feature matrices (and target) with zeros (or the mean for the target) before training, so the Lasso model receives only finite numbers and can produce a valid prediction file.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path

import numpy as np
import pandas as pd

base_path = Path("/kaggle/input/champs-scalar-coupling")
if not base_path.is_dir():
    base_path = Path("../input")
assert base_path.is_dir(), "Input directory not found."

pot_energy = pd.read_csv(base_path / "potential_energy.csv")
mulliken_charges = pd.read_csv(base_path / "mulliken_charges.csv")
train_df = pd.read_csv(base_path / "train.csv")
scalar_coupling_cont = pd.read_csv(base_path / "scalar_coupling_contributions.csv")
test_df = pd.read_csv(base_path / "test.csv")
magnetic_shield_tensor = pd.read_csv(base_path / "magnetic_shielding_tensors.csv")
dipole_moment = pd.read_csv(base_path / "dipole_moments.csv")
structures = pd.read_csv(base_path / "structures.csv")




## === cell 1
def map_atom_data(df: pd.DataFrame, atom_idx: int) -> pd.DataFrame:
    """Merge atom‑level structural information for a given atom index."""
    merged = pd.merge(
        df,
        structures,
        how="left",
        left_on=["molecule_name", f"atom_index_{atom_idx}"],
        right_on=["molecule_name", "atom_index"],
    )
    merged = merged.drop(columns=["atom_index"])
    merged = merged.rename(
        columns={
            "atom": f"atom_{atom_idx}",
            "x": f"x_{atom_idx}",
            "y": f"y_{atom_idx}",
            "z": f"z_{atom_idx}",
        }
    )
    return merged


train_df = map_atom_data(train_df, 0)
train_df = map_atom_data(train_df, 1)
test_df = map_atom_data(test_df, 0)
test_df = map_atom_data(test_df, 1)



## === cell 2
train_df = pd.merge(
    train_df,
    scalar_coupling_cont,
    on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
    how="left",
)
test_df = pd.merge(
    test_df,
    scalar_coupling_cont,
    on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
    how="left",
)

train_df = train_df.merge(pot_energy, on="molecule_name", how="left")
test_df = test_df.merge(pot_energy, on="molecule_name", how="left")

dipole_renamed = dipole_moment.rename(
    columns={"X": "dipole_X", "Y": "dipole_Y", "Z": "dipole_Z"}
)
train_df = train_df.merge(dipole_renamed, on="molecule_name", how="left")
test_df = test_df.merge(dipole_renamed, on="molecule_name", how="left")

mulliken_0 = mulliken_charges.rename(
    columns={"atom_index": "atom_index_0", "mulliken_charge": "mulliken_0"}
)
mulliken_1 = mulliken_charges.rename(
    columns={"atom_index": "atom_index_1", "mulliken_charge": "mulliken_1"}
)
train_df = train_df.merge(mulliken_0, on=["molecule_name", "atom_index_0"], how="left")
train_df = train_df.merge(mulliken_1, on=["molecule_name", "atom_index_1"], how="left")
test_df = test_df.merge(mulliken_0, on=["molecule_name", "atom_index_0"], how="left")
test_df = test_df.merge(mulliken_1, on=["molecule_name", "atom_index_1"], how="left")



## === cell 3
train_m_0 = train_df[["x_0", "y_0", "z_0"]].values
train_m_1 = train_df[["x_1", "y_1", "z_1"]].values
test_m_0 = test_df[["x_0", "y_0", "z_0"]].values
test_m_1 = test_df[["x_1", "y_1", "z_1"]].values

train_df["dist_vector"] = np.linalg.norm(train_m_0 - train_m_1, axis=1)
train_df["dist_X"] = (train_df["x_0"] - train_df["x_1"]) ** 2
train_df["dist_Y"] = (train_df["y_0"] - train_df["y_1"]) ** 2
train_df["dist_Z"] = (train_df["z_0"] - train_df["z_1"]) ** 2

test_df["dist_vector"] = np.linalg.norm(test_m_0 - test_m_1, axis=1)
test_df["dist_X"] = (test_df["x_0"] - test_df["x_1"]) ** 2
test_df["dist_Y"] = (test_df["y_0"] - test_df["y_1"]) ** 2
test_df["dist_Z"] = (test_df["z_0"] - test_df["z_1"]) ** 2



## === cell 4
train_df = train_df.drop(columns=["molecule_name"], errors="ignore")
test_df = test_df.drop(columns=["molecule_name"], errors="ignore")



## === cell 5
cat_cols = ["type", "atom_0", "atom_1"]
for col in cat_cols:
    train_df[col] = train_df[col].astype("category")
    test_df[col] = test_df[col].astype("category")

Attributes = [
    "id",
    "atom_index_0",
    "atom_index_1",
    "type",
    "x_0",
    "y_0",
    "z_0",
    "atom_0",
    "atom_1",
    "x_1",
    "y_1",
    "z_1",
    "dist_vector",
    "dist_X",
    "dist_Y",
    "dist_Z",
    "fc",
    "sd",
    "pso",
    "dso",
    "potential_energy",
    "dipole_X",
    "dipole_Y",
    "dipole_Z",
    "mulliken_0",
    "mulliken_1",
]
cat_attributes = ["type", "atom_0", "atom_1"]
target_label = ["scalar_coupling_constant"]

X_train = train_df[Attributes].copy()
X_test = test_df[Attributes].copy()
y_target = train_df[target_label].copy()

y_series = y_target.squeeze()
if y_series.isnull().any():
    y_series = y_series.fillna(y_series.mean())
y_series = y_series.replace([np.inf, -np.inf], np.nan).fillna(0)
y_log = np.log1p(y_series)

X_train = pd.get_dummies(X_train, columns=cat_attributes)
X_test = pd.get_dummies(X_test, columns=cat_attributes)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

from sklearn import linear_model

lasso = linear_model.Lasso(alpha=0.08, max_iter=10000)
lasso.fit(X_train, y_log.values.ravel())

train_score = np.round(lasso.score(X_train, y_log), 3)
print("Training R^2 score (log target):", train_score)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1607139.py in <cell line: 0>()
     63 
     64 lasso = linear_model.Lasso(alpha=0.08, max_iter=10000)
---> 65 lasso.fit(X_train, y_log.values.ravel())
     66 
     67 train_score = np.round(lasso.score(X_train, y_log), 3)

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_coordinate_descent.py in fit(self, X, y, sample_weight, check_input)
    906         if check_input:
    907             X_copied = self.copy_X and self.fit_intercept
--> 908             X, y = self._validate_data(
    909                 X,
    910                 y,

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

## === cell 6
y_pred_log = lasso.predict(X_test)
y_pred = np.expm1(y_pred_log)

submission = pd.read_csv(base_path / "sample_submission.csv")
submission = submission.sort_values("id").reset_index(drop=True)
submission["scalar_coupling_constant"] = y_pred
submission_path = "Lasso_Regression_model.csv"
submission.to_csv(submission_path, index=False)
print("Submission saved to", submission_path)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3118144993.py in <cell line: 0>()
----> 1 y_pred_log = lasso.predict(X_test)
      2 y_pred = np.expm1(y_pred_log)
      3 
      4 submission = pd.read_csv(base_path / "sample_submission.csv")
      5 submission = submission.sort_values("id").reset_index(drop=True)

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py in predict(self, X)
    352             Returns predicted values.
    353         """
--> 354         return self._decision_function(X)
    355 
    356     def _set_intercept(self, X_offset, y_offset, X_scale):

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_coordinate_descent.py in _decision_function(self, X)
   1072             return safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
   1073         else:
-> 1074             return super()._decision_function(X)
   1075 
   1076 

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py in _decision_function(self, X)
    336 
    337         X = self._validate_data(X, accept_sparse=["csr", "csc", "coo"], reset=False)
--> 338         return safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
    339 
    340     def predict(self, X):

AttributeError: 'Lasso' object has no attribute 'coef_'
