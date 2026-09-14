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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

DATA_ROOT = "/kaggle/input/champs-scalar-coupling"
if not os.path.isdir(DATA_ROOT):
    DATA_ROOT = os.path.join(os.getcwd(), "data", "champs-scalar-coupling")

train_path = os.path.join(DATA_ROOT, "train.csv")
test_path = os.path.join(DATA_ROOT, "test.csv")
structures_path = os.path.join(DATA_ROOT, "structures.csv")

train_df = pd.read_csv(
    train_path,
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
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
    },
)
structures_df = pd.read_csv(
    structures_path,
    dtype={
        "molecule_name": "category",
        "atom_index": np.int16,
        "atom": "category",
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
    },
)

structures0 = structures_df.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
    }
)
structures1 = structures_df.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
    }
)

global_mean = train_df["scalar_coupling_constant"].mean()
type_means = (
    train_df.groupby("type")["scalar_coupling_constant"]
    .mean()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "type_mean"})
)

train_idxs, val_idxs = train_test_split(train_df.index, test_size=0.2, random_state=42)
val_df = train_df.loc[val_idxs].copy()
val_df = val_df.merge(type_means, how="left", on="type")
val_df["type_mean"].fillna(global_mean, inplace=True)


def log_mae(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    return np.log(mae)


best_w = 1.0
best_score = np.inf
for w in np.arange(0.0, 1.001, 0.01):
    blended_pred = w * val_df["type_mean"] + (1 - w) * global_mean
    score = log_mae(val_df["scalar_coupling_constant"], blended_pred)
    if score < best_score:
        best_score = score
        best_w = w

print(
    f"Optimal blend weight for type mean: {best_w:.2f}, validation log‑MAE: {best_score:.5f}"
)

train_full = train_df.copy()
train_full = train_full.merge(type_means, how="left", on="type")
train_full["type_mean"].fillna(global_mean, inplace=True)
train_full["blended_pred"] = (
    best_w * train_full["type_mean"] + (1 - best_w) * global_mean
)
train_full["residual"] = (
    train_full["scalar_coupling_constant"] - train_full["blended_pred"]
)

type_correction = (
    train_full.groupby("type")["residual"]
    .mean()
    .reset_index()
    .rename(columns={"residual": "type_corr"})
)

val_df = val_df.merge(type_correction, how="left", on="type")
val_df["type_corr"].fillna(0.0, inplace=True)
val_df["blended_pred"] = best_w * val_df["type_mean"] + (1 - best_w) * global_mean

best_alpha = 1.0
best_score_alpha = np.inf
for a in np.arange(0.0, 2.01, 0.05):
    pred = val_df["blended_pred"] + a * val_df["type_corr"]
    score = log_mae(val_df["scalar_coupling_constant"], pred)
    if score < best_score_alpha:
        best_score_alpha = score
        best_alpha = a

print(
    f"Optimal residual scaling (alpha): {best_alpha:.2f}, validation log‑MAE after scaling: {best_score_alpha:.5f}"
)

elem_to_z = {
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


def add_geometry(df):
    df = df.merge(structures0, how="left", on=["molecule_name", "atom_index_0"])
    df = df.merge(structures1, how="left", on=["molecule_name", "atom_index_1"])
    df["distance"] = np.sqrt(
        (df["x0"] - df["x1"]) ** 2
        + (df["y0"] - df["y1"]) ** 2
        + (df["z0"] - df["z1"]) ** 2
    )
    df["z0"] = df["atom_0"].astype(str).map(elem_to_z).fillna(0).astype(np.int8)
    df["z1"] = df["atom_1"].astype(str).map(elem_to_z).fillna(0).astype(np.int8)
    df["atom_num_diff"] = np.abs(df["z0"] - df["z1"])
    return df


train_full = add_geometry(train_full)
train_full = train_full.merge(type_correction, how="left", on="type")
train_full["type_corr"].fillna(0.0, inplace=True)

train_full["final_residual"] = train_full["scalar_coupling_constant"] - (
    train_full["blended_pred"] + best_alpha * train_full["type_corr"]
)

feature_cols = ["distance", "atom_num_diff"]
gbr = GradientBoostingRegressor(
    n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
)
gbr.fit(train_full[feature_cols].values, train_full["final_residual"].values)

del train_full
import gc

gc.collect()




## === cell 1
submission = test_df[
    ["id", "type", "molecule_name", "atom_index_0", "atom_index_1"]
].copy()
submission = submission.merge(type_means, how="left", on="type")
submission["type_mean"].fillna(global_mean, inplace=True)

submission["blended_pred"] = (
    best_w * submission["type_mean"] + (1 - best_w) * global_mean
)
submission = submission.merge(type_correction, how="left", on="type")
submission["type_corr"].fillna(0.0, inplace=True)

submission = add_geometry(submission)

submission["gbr_residual"] = gbr.predict(submission[feature_cols].values)

submission["scalar_coupling_constant"] = (
    submission["blended_pred"]
    + best_alpha * submission["type_corr"]
    + submission["gbr_residual"]
)

submission = submission.drop(
    columns=[
        "type",
        "blended_pred",
        "type_corr",
        "distance",
        "atom_num_diff",
        "atom_0",
        "atom_1",
        "x0",
        "y0",
        "z0",
        "x1",
        "y1",
        "z1",
        "gbr_residual",
        "atom_index_0",
        "atom_index_1",
    ]
)

output_path = "my_blend_2.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}, shape: {submission.shape}")
