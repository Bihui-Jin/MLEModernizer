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
import random
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn import metrics
from sklearn.preprocessing import (
    StandardScaler,
)  # added scaler for better Lasso performance



## === cell 1
SEED = 31
FOLDS = 5  # increased folds for more stable CV selection
TARGET = "scalar_coupling_constant"
PREDICTORS = [
    "molecule_atom_index_0_dist_mean_div",
    "molecule_atom_index_0_dist_max_div",
    "molecule_atom_index_1_dist_max_div",
    "molecule_atom_index_0_dist_std_div",
    "molecule_atom_index_0_dist_min_div",
    "molecule_atom_index_1_dist_mean_div",
    "molecule_atom_index_1_dist_std_div",
    "molecule_atom_1_dist_std_diff",
    "molecule_atom_index_0_dist_std_diff",
    "molecule_atom_index_0_dist_mean_diff",
    "molecule_atom_index_1_dist_max_diff",
    "molecule_atom_index_0_dist_max_diff",
    "molecule_type_0_dist_std_diff",
    "molecule_atom_index_1_dist_mean_diff",
    "molecule_atom_index_1_dist_std_diff",
    "molecule_atom_1_dist_min_div",
    "molecule_atom_1_dist_min_diff",
    "type_0",
    "type_1",
    "molecule_type_dist_min",
    "molecule_type_dist_mean",
    "molecule_type_0_dist_std",
    "dist_to_type_1_mean",
    "dist",
    "molecule_type_dist_max",
    "dist_x",
    "dist_y",
    "dist_z",
    "type_code",  # new numeric encoding of coupling type
]




## === cell 2
def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


seed_everything(SEED)



## === cell 3
DATA_ROOT = "/kaggle/input/champs-scalar-coupling"
train_path = os.path.join(DATA_ROOT, "train.csv")
test_path = os.path.join(DATA_ROOT, "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print(f"train shape: {train.shape}, test shape: {test.shape}")

type_combined = pd.concat([train["type"], test["type"]], ignore_index=True)
type_dummies = pd.get_dummies(type_combined, prefix="type")
train_type_dummies = type_dummies.iloc[: len(train)].reset_index(drop=True)
test_type_dummies = type_dummies.iloc[len(train) :].reset_index(drop=True)

train = pd.concat([train.reset_index(drop=True), train_type_dummies], axis=1)
test = pd.concat([test.reset_index(drop=True), test_type_dummies], axis=1)

train["type_code"] = train["type"].astype("category").cat.codes
test["type_code"] = test["type"].astype("category").cat.codes

dummy_cols = [c for c in train.columns if c.startswith("type_")]
PREDICTORS.extend(dummy_cols)



## === cell 4
available_predictors = [col for col in PREDICTORS if col in train.columns]
if not available_predictors:
    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    fallback = [c for c in numeric_cols if c not in [TARGET, "id"]]
    available_predictors = fallback
    print("Warning: No engineered predictors found – using fallback numeric columns.")
else:
    print(f"Using {len(available_predictors)} engineered predictors.")




## === cell 5
def group_mean_log_mae(
    y_true: np.ndarray, y_pred: np.ndarray, types: pd.Series, floor: float = 1e-9
) -> float:
    """
    Computes the competition metric: log of MAE per coupling type, then averaged.
    """
    abs_errors = pd.Series(np.abs(y_true - y_pred), index=types)
    maes = abs_errors.groupby(types).mean()
    log_maes = np.log(maes.map(lambda x: max(x, floor)))
    return log_maes.mean()




## === cell 6
y_train = train[TARGET]
X_train = train[available_predictors]

type_series = train["type"]  # global series for extracting types in scorer
group_groups = train["molecule_name"]

model = Lasso(random_state=SEED)  # max_iter will be set via the grid
pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])

param_grid = {
    "model__max_iter": [200, 500, 1000, 2000, 5000, 10000, 20000],
    "model__alpha": [
        1e-12,  # even smaller regularisation
        1e-9,
        1e-8,
        5e-8,
        1e-7,
        5e-7,
        1e-6,
        5e-6,
        1e-5,
        5e-5,
        1e-4,
        5e-4,
        1e-3,
        5e-3,
        1e-2,
        5e-2,
        1e-1,
        1.0,
    ],
}


def log_mae_scorer(estimator, X, y):
    preds = estimator.predict(X)
    types_val = type_series.loc[X.index]
    return -group_mean_log_mae(y, preds, types_val)


cv = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=GroupKFold(n_splits=FOLDS),  # split by molecule_name to match evaluation
    scoring=log_mae_scorer,
    n_jobs=-1,
)

cv.fit(X_train, y_train, groups=group_groups)
print("best_params_ =", cv.best_params_)
print("best_score_  =", cv.best_score_)



## === cell 7
y_pred_train = cv.predict(X_train)
gmlm = group_mean_log_mae(y_train.values, y_pred_train, train["type"])
print(f"group_mean_log_mae = {gmlm:.6f}")



## === cell 8
X_test = test[available_predictors]
preds = cv.predict(X_test)
print(f"Predictions shape: {preds.shape}")



## === cell 9
submission = pd.DataFrame({"id": test["id"], TARGET: preds})
submission.head()



## === cell 10
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print("Current directory contents:", os.listdir("."))
