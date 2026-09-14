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

2.187699377893074

# 6. Current score

1.23565

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 2.88712) has done: 'I correct the data path, make the predictor list robust to missing columns, and ensure variables are defined before use so the notebook runs end‑to‑end and writes a proper `submission.csv`. This fixes the FileNotFoundError and subsequent NameErrors while keeping the original modelling approach unchanged.'
- What this solution (achieved 2.88712) has done: 'I fixed the `group_mean_log_mae` function to correctly handle NumPy arrays by using `np.abs` and creating a pandas Series for grouping, and expanded the Lasso `max_iter` grid to include larger values so the optimizer can converge better, which should modestly lower the validation error toward the target score. The rest of the workflow remains unchanged, and the script now writes a proper `submission.csv` file.'
- What this solution (achieved 2.88437) has done: 'I add a StandardScaler to the pipeline and expand the hyper‑parameter grid to also search over Lasso’s alpha values. Scaling the features usually helps Lasso converge to a better solution, and a modest α sweep can lower the MAE (and thus the log‑MAE metric) without changing the core modelling approach. These tweaks are minimal, keep the original logic intact, and are expected to move the score closer to the target.'
- What this solution (achieved 2.88437) has done: 'I expand the hyper‑parameter grid to allow longer convergence (larger max_iter) and finer regularisation (include a smaller alpha). I also switch the cross‑validation to a GroupKFold that groups by coupling type, which aligns the validation more closely with the competition’s per‑type metric and should help select a model that reduces the log‑MAE toward the target. The rest of the pipeline and output logic stay the same.'
- What this solution (achieved 2.88869) has done: 'I add a custom scorer that directly optimizes the competition’s log‑MAE metric (using the global `type` series) and expand the hyper‑parameter grid slightly with a smaller α and a larger `max_iter`. This aligns the GridSearchCV objective with the final evaluation, allowing the model to target a lower score without altering the core Lasso pipeline.'
- What this solution (achieved 2.88869) has done: 'I increase the number of cross‑validation folds to give a more reliable hyper‑parameter search and expand the Lasso α grid with a few intermediate values. These minimal adjustments keep the original Lasso pipeline unchanged while aiming to lower the log‑MAE score toward the target.'
- What this solution (achieved 2.88869) has done: 'I tighten the validation so it mirrors the competition split: use `molecule_name` for GroupKFold grouping and compute the custom log‑MAE scorer on the actual validation rows (by extracting the `type` values via the index). This corrects a leakage in the original scorer and selects hyper‑parameters that generalise better, which should reduce the log‑MAE toward the target score while keeping the Lasso pipeline unchanged.'
- What this solution (achieved 2.88869) has done: 'I added simple one‑hot encoding for the coupling `type` column so that meaningful categorical information is fed to the Lasso model (the original code fell back to just numeric columns like `id`, giving a weak model). The dummy variables are created before selecting predictors, allowing them to be automatically picked up by the fallback numeric‑column logic. This minor feature engineering should lower the log‑MAE toward the target without altering the core modelling pipeline.'
- What this solution (achieved 2.88869) has done: 'I extend the Lasso α grid to include a few smaller values (down to 1e‑8) while keeping the rest of the pipeline unchanged. Smaller regularisation can let the model capture more signal from the engineered features, which is expected to lower the log‑MAE and move the score toward the target 2.1877 without altering the core logic.'
- What this solution (achieved 2.77709) has done: 'The changes add a numeric encoding of the coupling `type` ( `type_code` ) so Lasso can directly use this information, expand the α grid slightly with a smaller value, and switch the GroupKFold grouping to the coupling `type` (which aligns the CV objective with the competition metric). These minimal adjustments keep the original pipeline intact while nudging the model toward a lower log‑MAE, moving the score nearer to the target.'
- What this solution (achieved 1.23565) has done: 'The update speeds up the exhaustive GridSearchCV by parallelising the folds (using all CPUs) and drastically reducing the hyper‑parameter space to just a few representative settings, which cuts the number of model fits from dozens to a handful while keeping the same Lasso model and evaluation metric.  The custom scorer and the group‑wise log‑MAE calculation are rewritten with pure NumPy operations to avoid costly Pandas groupby overhead, preserving identical results.  No changes are made to feature engineering, data handling, or the final prediction pipeline, so the model’s logic and accuracy remain unchanged.'

# 9. Code solution

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
    Vectorised version of the competition metric:
    log of MAE per coupling type, then averaged.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    types_arr = types.values

    abs_err = np.abs(y_true - y_pred)

    uniq, inv = np.unique(types_arr, return_inverse=True)
    sums = np.bincount(inv, weights=abs_err)
    counts = np.bincount(inv)
    maes = sums / counts

    log_maes = np.log(np.maximum(maes, floor))

    return float(log_maes.mean())




## === cell 6
y_train = train[TARGET]
X_train = train[available_predictors].astype(np.float32)

type_series = train["type"]  # global series for extracting types in scorer
group_groups = train["molecule_name"]

scaler = StandardScaler()
X_train_scaled_np = scaler.fit_transform(X_train)
X_train_scaled = pd.DataFrame(
    X_train_scaled_np, columns=available_predictors, index=X_train.index
)

model = Lasso(random_state=SEED)  # max_iter will be set via the grid

param_grid = {
    "max_iter": [2000],  # keep a single, reasonable iteration cap
    "alpha": [1e-4, 1e-2],  # two alphas to explore regularisation strength
}


def log_mae_scorer(estimator, X, y):
    preds = estimator.predict(X)
    types_val = type_series.loc[X.index]
    return -group_mean_log_mae(y, preds, types_val)


cv = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=GroupKFold(n_splits=FOLDS),  # split by molecule_name to match evaluation
    scoring=log_mae_scorer,
    n_jobs=-1,  # parallelise over all CPUs
)

cv.fit(X_train_scaled, y_train, groups=group_groups)
print("best_params_ =", cv.best_params_)
print("best_score_  =", cv.best_score_)




## === cell 7
y_pred_train = cv.predict(X_train_scaled)
gmlm = group_mean_log_mae(y_train.values, y_pred_train, train["type"])
print(f"group_mean_log_mae = {gmlm:.6f}")




## === cell 8
X_test = test[available_predictors].astype(np.float32)
X_test_scaled_np = scaler.transform(X_test)
X_test_scaled = pd.DataFrame(
    X_test_scaled_np, columns=available_predictors, index=X_test.index
)

preds = cv.predict(X_test_scaled)
print(f"Predictions shape: {preds.shape}")




## === cell 9
submission = pd.DataFrame({"id": test["id"], TARGET: preds})
submission.head()




## === cell 10
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print("Current directory contents:", os.listdir("."))
