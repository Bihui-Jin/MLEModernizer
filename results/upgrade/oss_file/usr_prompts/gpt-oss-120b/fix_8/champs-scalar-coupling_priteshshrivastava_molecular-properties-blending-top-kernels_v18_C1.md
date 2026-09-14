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

-1.6838160788283738

# 6. Current score

1.26419

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'The script failed because it tried to merge on the “type” column after dropping it from the submission DataFrame. We keep the “type” column for the merge, fill missing predictions with the global mean, then drop the auxiliary column and rename the prediction column to match the required submission format.'
- What this solution (achieved 1.23566) has done: 'I add a quick validation step that searches for a simple blend between the per‑type mean and the overall global mean. By picking the blend weight that gives the lowest *log‑MAE* on a held‑out split, we can modestly lower the score without changing the core modeling approach. The script now computes this optimal weight and uses it to produce the final submission.'
- What this solution (achieved 1.23551) has done: 'I added the missing imports, defined the data root, loaded the training data, and computed the required `global_mean` and per‑type means (`type_means`). The script now performs the train/validation split, finds the optimal blending weight, builds the type‑wise residual correction, and finally creates a valid submission CSV using the test set. All previously undefined variables are now defined, and the output file has the required `.csv` suffix and columns.'
- What this solution (achieved 1.23566) has done: 'I keep the overall blending approach but improve the residual correction by computing it on the full training set (instead of only the validation split). This uses the same core logic but gives a more accurate per‑type adjustment, which should lower the log‑MAE and move the score toward the target. The changes are limited to additional merges and calculations while preserving the original workflow.'
- What this solution (achieved 1.26419) has done: 'I add a light post‑processing step that scales the per‑type residual correction. After finding the optimal blend weight `best_w`, I search for a small scaling factor `alpha` on the validation split that minimizes the log‑MAE, then apply that same `alpha` to the test predictions. This keeps the original modelling flow unchanged while giving a modest improvement toward the target score.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_ROOT = "/kaggle/input/champs-scalar-coupling"
if not os.path.isdir(DATA_ROOT):
    DATA_ROOT = os.path.join(os.getcwd(), "data", "champs-scalar-coupling")

train_path = os.path.join(DATA_ROOT, "train.csv")
train_df = pd.read_csv(train_path)

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



## === cell 1
test_path = os.path.join(DATA_ROOT, "test.csv")
test_df = pd.read_csv(test_path)

submission = test_df[["id", "type"]].copy()
submission = submission.merge(type_means, how="left", on="type")
submission["type_mean"].fillna(global_mean, inplace=True)

submission["blended_pred"] = (
    best_w * submission["type_mean"] + (1 - best_w) * global_mean
)

submission = submission.merge(type_correction, how="left", on="type")
submission["type_corr"].fillna(0.0, inplace=True)

submission["scalar_coupling_constant"] = (
    submission["blended_pred"] + best_alpha * submission["type_corr"]
)

submission = submission.drop(columns=["type", "blended_pred", "type_corr"])

output_path = "my_blend_1.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}, shape: {submission.shape}")
