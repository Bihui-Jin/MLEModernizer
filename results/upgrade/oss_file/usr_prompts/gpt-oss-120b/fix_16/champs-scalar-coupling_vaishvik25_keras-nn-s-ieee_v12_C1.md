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

-1.476322251698409

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'The script was trying to read non‑existent directories and external prediction files, which caused all subsequent cells to fail. I replaced those parts with a simple, self‑contained baseline: load the competition’s train and test data from the correct input folder, compute the mean coupling constant per `type`, predict those means for the test set, and write a valid `submission.csv`. This fixes the path errors, removes missing‑file dependencies, and guarantees a proper CSV output while keeping the overall workflow intact.'
- What this solution (achieved 1.23566) has done: 'I add atom‑type information from structures.csv and use the mean coupling constant for each (type, atom 0, atom 1) combo (falling back to the per‑type mean and finally the global mean). This richer “mean‑by‑combo” mapping lowers the MAE, so the logged metric moves toward the lower target score while keeping the original simple baseline structure.'
- What this solution (achieved 1.60418) has done: 'Implemented NaN‑handling for feature columns by imputing median values, ensuring the RandomForest receives a clean matrix. Added a modest adjustment to the ensemble weighting (favoring the robust combo‑mean fallback) to improve the validation log‑MAE while preserving the original modeling pipeline. Fixed the submission write‑out by guaranteeing the `submission` dataframe is correctly created before saving.'
- What this solution (achieved 1.89808) has done: 'I keep the overall pipeline unchanged but tweak the model and blending to move the validation log‑MAE lower. Specifically, I increase the RandomForest capacity (more trees and deeper trees) and shift the blending weight toward the learned model (70 % model, 30 % combo fallback). These minimal edits preserve the core logic while aiming to reduce the error toward the target score.'
- What this solution (achieved 1.23566) has done: 'I lower the blending weight of the RandomForest model to zero and rely entirely on the combo‑based fallback predictions, which have shown better validation performance in earlier experiments. This change keeps the existing data processing and model training intact while adjusting only the final prediction aggregation to move the log‑MAE toward the lower target score.'
- What this solution (achieved 1.71091) has done: 'I blend the RandomForest predictions with the existing combo‑fallback predictions instead of using the fallback alone. This simple averaging keeps the original pipeline and feature set, but leverages the model’s learned signal, which should lower the log‑MAE and move the metric toward the negative target. The same blending is applied to the test set so the submitted values reflect the improved prediction.'
- What this solution (achieved 1.23566) has done: 'The changes keep the same feature engineering and RandomForest model but speed up the heavy steps: merges are performed via indexed joins, data are cast to compact NumPy arrays before fitting/predicting, and the forest uses fewer trees (100) while retaining the same depth and other settings, which cuts training time roughly in half without altering the model’s structure. All other logic, including fallback means and blending, remains unchanged, preserving the original predictions.'
- What this solution (achieved 1.70331) has done: 'I add a simple engineered feature (the sum of the atom numbers) to give the RandomForest a bit more signal, increase the forest size modestly, and set a non‑zero blending weight so the model’s predictions contribute alongside the combo‑fallback means. These lightweight tweaks keep the original pipeline intact while expectedly lowering the validation log‑MAE toward the negative target.'
- What this solution (achieved 1.23566) has done: 'I lower the blending weight to 0 so the final predictions rely entirely on the combo‑fallback means, which have shown better validation performance in earlier experiments. This minimal change keeps the whole pipeline unchanged while moving the log‑MAE toward the lower target score.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

base_path = "../input/champs-scalar-coupling"

train_path = os.path.join(base_path, "train.csv")
test_path = os.path.join(base_path, "test.csv")
structures_path = os.path.join(base_path, "structures.csv")

train_df = pd.read_csv(
    train_path,
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
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
        "scalar_coupling_constant": np.float32,
    },
)

test_df = pd.read_csv(
    test_path,
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
    },
)

structures = pd.read_csv(
    structures_path,
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
    dtype={
        "molecule_name": "category",
        "atom_index": np.int16,
        "atom": "category",
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
    },
)

structures0 = structures.rename(
    columns={"atom": "atom_0", "x": "x0", "y": "y0", "z": "z0"}
).set_index(["molecule_name", "atom_index"])
structures1 = structures.rename(
    columns={"atom": "atom_1", "x": "x1", "y": "y1", "z": "z1"}
).set_index(["molecule_name", "atom_index"])


def add_atom_info(df):
    df = df.merge(
        structures0,
        left_on=["molecule_name", "atom_index_0"],
        right_index=True,
        how="left",
    )
    df = df.merge(
        structures1,
        left_on=["molecule_name", "atom_index_1"],
        right_index=True,
        how="left",
    )
    return df


train_full = add_atom_info(train_df)
test_full = add_atom_info(test_df)

atom_number = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Cl": 17,
    "Br": 35,
    "I": 53,
    "S": 16,
    "P": 15,
}


def get_num(sym):
    return atom_number.get(sym, 0)


for col in ["atom_0", "atom_1"]:
    train_full[f"{col}_num"] = train_full[col].map(get_num).astype(np.int8)
    test_full[f"{col}_num"] = test_full[col].map(get_num).astype(np.int8)

train_full["atom_sum_num"] = train_full["atom_0_num"] + train_full["atom_1_num"]
test_full["atom_sum_num"] = test_full["atom_0_num"] + test_full["atom_1_num"]

train_full["distance"] = np.sqrt(
    (train_full["x0"] - train_full["x1"]) ** 2
    + (train_full["y0"] - train_full["y1"]) ** 2
    + (train_full["z0"] - train_full["z1"]) ** 2
).astype(np.float32)
test_full["distance"] = np.sqrt(
    (test_full["x0"] - test_full["x1"]) ** 2
    + (test_full["y0"] - test_full["y1"]) ** 2
    + (test_full["z0"] - test_full["z1"]) ** 2
).astype(np.float32)

train_full["type_code"] = train_full["type"].cat.codes
type_uniques = train_full["type"].cat.categories
test_full["type_code"] = test_full["type"].cat.codes

feature_cols = ["distance", "atom_0_num", "atom_1_num", "atom_sum_num", "type_code"]
imputer = SimpleImputer(strategy="median")
imputer.fit(train_full[feature_cols])
train_full[feature_cols] = imputer.transform(train_full[feature_cols])
test_full[feature_cols] = imputer.transform(test_full[feature_cols])


def make_combo_key(df):
    a0 = df["atom_0"].astype(str).values
    a1 = df["atom_1"].astype(str).values
    min_atom = np.where(a0 < a1, a0, a1)
    max_atom = np.where(a0 < a1, a1, a0)
    return df["type"].astype(str).values + "_" + min_atom + "_" + max_atom


train_full["combo_key"] = make_combo_key(train_full).astype("category")
test_full["combo_key"] = make_combo_key(test_full).astype("category")

type_means_full = train_full.groupby("type")["scalar_coupling_constant"].mean()
global_mean_full = train_full["scalar_coupling_constant"].mean()
combo_means_full = train_full.groupby("combo_key")["scalar_coupling_constant"].mean()

train_split, val_split = train_test_split(
    train_full,
    test_size=0.2,
    random_state=42,
    stratify=train_full["type"],
)

X_train = train_split[feature_cols].values.astype(np.float32)
y_train = train_split["scalar_coupling_constant"].values.astype(np.float32)

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,  # fully parallelise
    random_state=42,
    min_samples_leaf=1,
    max_features="sqrt",
)
rf.fit(X_train, y_train)

combo_fallback = (
    val_split["combo_key"]
    .map(combo_means_full)
    .fillna(val_split["type"].map(type_means_full))
    .fillna(global_mean_full)
)

X_val = val_split[feature_cols].values.astype(np.float32)
model_pred = rf.predict(X_val)


def log_mae_series(true, pred, types):
    mae_per_type = true.groupby(types).apply(
        lambda g: mean_absolute_error(g["scalar_coupling_constant"], pred.loc[g.index])
    )
    return np.log(mae_per_type).mean()


fallback_series = pd.Series(combo_fallback.values, index=val_split.index)
fallback_log_mae = log_mae_series(val_split, fallback_series, "type")

model_series = pd.Series(model_pred, index=val_split.index)
model_log_mae = log_mae_series(val_split, model_series, "type")

blend_weight = 0.0
if model_log_mae < fallback_log_mae:
    blend_weight = 0.2

val_pred_series = pd.Series(
    blend_weight * model_pred + (1 - blend_weight) * combo_fallback.values,
    index=val_split.index,
)

final_log_mae = log_mae_series(val_split, val_pred_series, "type")
print(f"Validation log‑MAE (lower is better): {final_log_mae:.5f}")

test_combo_fallback = (
    test_full["combo_key"]
    .map(combo_means_full)
    .fillna(test_full["type"].map(type_means_full))
    .fillna(global_mean_full)
)

X_test = test_full[feature_cols].values.astype(np.float32)
test_model_pred = rf.predict(X_test)

test_pred = (
    blend_weight * test_model_pred + (1 - blend_weight) * test_combo_fallback.values
)

submission = pd.DataFrame(
    {"id": test_full["id"], "scalar_coupling_constant": test_pred}
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3024123631.py in <cell line: 0>()
    141 
    142 # Create combo_key once and store as categorical to speed up later mapping
--> 143 train_full["combo_key"] = make_combo_key(train_full).astype("category")
    144 test_full["combo_key"] = make_combo_key(test_full).astype("category")
    145 

TypeError: data type 'category' not understood

## === cell 1
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False, float_format="%.6f")
print(f"Submission written to {submission_path}")

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1532181227.py in <cell line: 0>()
      1 submission_path = "submission.csv"
----> 2 submission.to_csv(submission_path, index=False, float_format="%.6f")
      3 print(f"Submission written to {submission_path}")

NameError: name 'submission' is not defined
