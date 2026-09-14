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

- What this solution (achieved 1.23566) has done: 'I replace the failing external‑submission loading with a self‑contained baseline: read the competition train and test files, compute the mean scalar coupling constant for each coupling type (and fall back to the global mean when needed), apply these means as predictions for the test set, and write a correctly formatted `submission.csv`. This removes the FileNotFoundError, eliminates the undefined‑variable error, and guarantees a valid submission file.'
- What this solution (achieved 1.25615) has done: 'I replace the simple type‑mean predictions with a more robust type‑median baseline and add a tiny data‑driven scaling factor estimated on a held‑out validation split (by molecule name). This keeps the overall “mean‑like” logic while likely lowering the log‑MAE, moving the score toward the negative target.'
- What this solution (achieved 1.23498) has done: 'I replace the single global scaling factor with per‑coupling‑type scaling derived from the validation split. By adjusting each type’s prediction with its own ratio of true to baseline mean, the predictions become better calibrated, which should lower the log‑MAE and move the score toward the negative target while keeping the overall simple median‑baseline approach.'
- What this solution (achieved 1.235) has done: 'I replace the simple median baseline with a mean‑based baseline and fit a tiny per‑type linear calibration (slope + intercept) on a held‑out validation split. This keeps the overall “type‑mean + scaling” logic but adds a more expressive correction that should lower the log‑MAE and move the score closer to the negative target, while preserving the rest of the pipeline and ensuring a valid CSV is written.'
- What this solution (achieved 1.235) has done: 'I replace the mean‑based baseline with a median‑based one (more robust to outliers) and keep the same lightweight per‑type linear calibration. This small change is expected to lower the absolute errors and thus move the log‑MAE score toward the negative target while preserving the overall pipeline and output format.'
- What this solution (achieved 1.23532) has done: 'I replace the median‑based baseline with a mean‑based one (which usually gives lower absolute error) and use a slightly larger validation split (10 %) to obtain more stable per‑type calibration coefficients. The rest of the pipeline stays unchanged, so the script still writes a correct `submission.csv` while moving the log‑MAE closer to the negative target.'
- What this solution (achieved 2.98085) has done: 'The fix adds simple NaN handling: rows with missing feature values are removed from the training set, and missing values in the test set are imputed with the median of each feature from the cleaned training data. This prevents the Ridge model from crashing on NaNs and ensures the `submission` DataFrame is always created, allowing the script to write a valid `submission.csv` file.'
- What this solution (achieved 3.28064) has done: 'I add a small validation split, fit the Ridge model on the training portion, then learn a per‑coupling‑type linear calibration (slope + intercept) on the held‑out validation set. The calibrated coefficients are applied to the test predictions, which should lower the log‑MAE and move the score nearer the negative target while keeping the original feature set and model unchanged.'
- What this solution (achieved 2.91589) has done: 'I add a few inexpensive, physically‑motivated features (sum and difference of atomic numbers, squared distance, molecule‑level dipole magnitude and potential energy) and merge the corresponding tables. These extra numeric columns give the Ridge model more signal without changing its core logic, and should lower the log‑MAE toward the negative target while still producing a valid `submission.csv`.'
- What this solution (achieved 1.84108) has done: 'I lower the regularisation strength (increase alpha) to make the Ridge model more stable, then clip each test prediction to the observed minimum‑maximum range of its coupling type from the training split. This simple post‑processing reduces extreme overshoots, which should lower the log‑MAE and move the score closer to the negative target while keeping the overall pipeline unchanged.'
- What this solution (achieved 1.84445) has done: 'I loosen the Ridge regularisation (α = 1.0 instead of 5.0) and use a larger validation split (20 % instead of 10 %). This gives the model more flexibility and a more stable per‑type calibration, which should lower the log‑MAE and move the score toward the negative target while keeping the overall pipeline unchanged.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

print("Input directories:", os.listdir("../input"))

TRAIN_PATH = "../input/champs-scalar-coupling/train.csv"
TEST_PATH = "../input/champs-scalar-coupling/test.csv"
STRUCTURES_PATH = "../input/champs-scalar-coupling/structures.csv"
DIPOLE_PATH = "../input/champs-scalar-coupling/dipole_moments.csv"
ENERGY_PATH = "../input/champs-scalar-coupling/potential_energy.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
structures_df = pd.read_csv(STRUCTURES_PATH)
dipole_df = pd.read_csv(DIPOLE_PATH)
energy_df = pd.read_csv(ENERGY_PATH)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Structures shape:", structures_df.shape)
print("Dipole shape:", dipole_df.shape)
print("Energy shape:", energy_df.shape)




## === cell 1
ELEMENT_TO_Z = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
}


def element_to_z(el):
    return ELEMENT_TO_Z.get(el, 0)


def add_atom_features(df, idx_col, suffix):
    """
    Merge atom information from structures_df and rename the coordinate columns
    with the given suffix (e.g. "_0" or "_1").
    """
    merged = df.merge(
        structures_df,
        left_on=["molecule_name", idx_col],
        right_on=["molecule_name", "atom_index"],
        how="left",
    )
    merged.rename(
        columns={
            "atom": f"atom_{suffix}",
            "x": f"x_{suffix}",
            "y": f"y_{suffix}",
            "z": f"z_{suffix}",
        },
        inplace=True,
    )
    merged.drop(columns=["atom_index"], inplace=True)
    return merged


train_feat = add_atom_features(train_df, "atom_index_0", "0")
train_feat = add_atom_features(train_feat, "atom_index_1", "1")
test_feat = add_atom_features(test_df, "atom_index_0", "0")
test_feat = add_atom_features(test_feat, "atom_index_1", "1")


def compute_distance(df):
    return np.sqrt(
        (df["x_0"] - df["x_1"]) ** 2
        + (df["y_0"] - df["y_1"]) ** 2
        + (df["z_0"] - df["z_1"]) ** 2
    )


train_feat["distance"] = compute_distance(train_feat)
test_feat["distance"] = compute_distance(test_feat)

train_feat["Z_0"] = train_feat["atom_0"].apply(element_to_z)
train_feat["Z_1"] = train_feat["atom_1"].apply(element_to_z)
test_feat["Z_0"] = test_feat["atom_0"].apply(element_to_z)
test_feat["Z_1"] = test_feat["atom_1"].apply(element_to_z)

train_feat["Z_sum"] = train_feat["Z_0"] + train_feat["Z_1"]
train_feat["Z_diff"] = np.abs(train_feat["Z_0"] - train_feat["Z_1"])
train_feat["distance_sq"] = train_feat["distance"] ** 2

test_feat["Z_sum"] = test_feat["Z_0"] + test_feat["Z_1"]
test_feat["Z_diff"] = np.abs(test_feat["Z_0"] - test_feat["Z_1"])
test_feat["distance_sq"] = test_feat["distance"] ** 2

train_feat["type_code"], type_mapping = pd.factorize(train_feat["type"])
test_feat["type_code"] = (
    test_feat["type"]
    .map({t: i for i, t in enumerate(type_mapping)})
    .fillna(-1)
    .astype(int)
)

dipole_df["dipole_mag"] = np.sqrt(
    dipole_df["X"] ** 2 + dipole_df["Y"] ** 2 + dipole_df["Z"] ** 2
)
train_feat = train_feat.merge(
    dipole_df[["molecule_name", "dipole_mag"]], on="molecule_name", how="left"
)
test_feat = test_feat.merge(
    dipole_df[["molecule_name", "dipole_mag"]], on="molecule_name", how="left"
)
train_feat = train_feat.merge(energy_df, on="molecule_name", how="left")
test_feat = test_feat.merge(energy_df, on="molecule_name", how="left")

train_feat = train_feat.dropna(subset=["scalar_coupling_constant"]).reset_index(
    drop=True
)

FEATURE_COLUMNS = [
    "type_code",
    "Z_0",
    "Z_1",
    "distance",
    "Z_sum",
    "Z_diff",
    "distance_sq",
    "dipole_mag",
    "potential_energy",
]

train_idx, val_idx = train_test_split(
    np.arange(len(train_feat)),
    test_size=0.2,
    random_state=42,
    stratify=train_feat["type_code"],
)
train_split = train_feat.iloc[train_idx].reset_index(drop=True)
val_split = train_feat.iloc[val_idx].reset_index(drop=True)

train_split["log_target"] = np.log1p(train_split["scalar_coupling_constant"])
val_split["log_target"] = np.log1p(val_split["scalar_coupling_constant"])

X_train = train_split[FEATURE_COLUMNS].values
y_train = train_split["log_target"].values

train_mask = ~np.isnan(X_train).any(axis=1)
X_train = X_train[train_mask]
y_train = y_train[train_mask]

col_medians = np.nanmedian(X_train, axis=0)




## === cell 2
ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train, y_train)

X_val = val_split[FEATURE_COLUMNS].values
X_val = np.where(np.isnan(X_val), col_medians, X_val)
val_pred_log = ridge.predict(X_val)

calibration = {}
for t in val_split["type_code"].unique():
    mask = val_split["type_code"] == t
    if mask.sum() < 2:
        calibration[t] = (1.0, 0.0)
        continue
    a, b = np.polyfit(val_pred_log[mask], val_split.loc[mask, "log_target"], 1)
    calibration[t] = (a, b)

X_test = test_feat[FEATURE_COLUMNS].values
X_test = np.where(np.isnan(X_test), col_medians, X_test)
test_pred_log = ridge.predict(X_test)

calibrated_log = []
for pred, t in zip(test_pred_log, test_feat["type_code"]):
    a, b = calibration.get(t, (1.0, 0.0))
    calibrated_log.append(a * pred + b)
calibrated_log = np.array(calibrated_log)

test_pred = np.expm1(calibrated_log)

type_min = train_split.groupby("type_code")["scalar_coupling_constant"].min()
type_max = train_split.groupby("type_code")["scalar_coupling_constant"].max()
clipped_preds = []
for pred, t in zip(test_pred, test_feat["type_code"]):
    lo = type_min.get(t, -np.inf)
    hi = type_max.get(t, np.inf)
    clipped_preds.append(np.clip(pred, lo, hi))
test_pred = np.array(clipped_preds)

submission = pd.DataFrame(
    {"id": test_feat["id"], "scalar_coupling_constant": test_pred}
)

print("Submission preview:")
print(submission.head())




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2366004746.py in <cell line: 0>()
      1 ridge = Ridge(alpha=1.0, random_state=42)
----> 2 ridge.fit(X_train, y_train)
      3 
      4 X_val = val_split[FEATURE_COLUMNS].values
      5 X_val = np.where(np.isnan(X_val), col_medians, X_val)

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

## === cell 3
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3714236941.py in <cell line: 0>()
      1 submission_path = "submission.csv"
----> 2 submission.to_csv(submission_path, index=False)
      3 print(f"Submission file written to {submission_path}")

NameError: name 'submission' is not defined
