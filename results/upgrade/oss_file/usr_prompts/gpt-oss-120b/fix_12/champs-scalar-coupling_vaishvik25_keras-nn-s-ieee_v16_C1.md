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

-1.3517984281028244

# 6. Current score

1.23566

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'I replace the failing stacking code with a simple baseline that computes the mean scalar coupling constant for each coupling type from the training set and uses these means as predictions for the test set. This ensures the script runs end‑to‑end, reads the correct data paths, and writes a valid `submission.csv` file with the required columns.'
- What this solution (achieved 1.23566) has done: 'I enrich the simple type‑mean baseline with atom‑type information: each coupling pair’s two atom elements are taken from structures.csv, and a mean target is computed for each (type, atom0, atom1) combination. Predictions first use this more specific mean, fall back to the type‑mean, and finally to the overall global mean. This adds only a lightweight feature while preserving the original logic and should lower the log‑MAE toward the target.'
- What this solution (achieved 3.00563) has done: 'I keep the original data loading and atom‑mapping logic, add reading of the coupling‑contributions file, and compute a small weighted blend between the exact contribution sum and the previous type‑mean baseline. By searching a simple weight on a held‑out validation split, the blend is calibrated so that the resulting log‑MAE moves toward the target ‑1.3518 without overshooting it. The script then uses this calibrated blend for the test set and writes a proper `submission.csv`.'
- What this solution (achieved 1.23566) has done: 'I tighten the blending weight search by using a finer grid (step 0.01) and handle missing contribution values by falling back to the type‑mean before blending. This yields a weight that gives a validation MAE ≈ exp(‑1.3518) ≈ 0.26, moving the log‑MAE much closer to the target without altering the core modelling logic.'
- What this solution (achieved 1.23566) has done: 'I add a lightweight per‑(type, atom0, atom1) mean (“combo_mean”) and use it in place of the broader type‑mean when blending with the contribution‑sum. This keeps the original blending logic but provides a more specific baseline, which should lower the log‑MAE and move the score closer to the target while preserving the core workflow.'
- What this solution (achieved 1.23566) has done: 'I keep the overall workflow but add a per‑type blending weight: after finding the global best weight `best_w`, I compute a small optimal weight for each coupling `type` on the validation split (using the same grid search). During test‑time prediction I look up the weight for the row’s type (falling back to the global weight) and blend the contribution‑sum with the combo‑mean accordingly. This adds only a lightweight, targeted improvement that should lower the log‑MAE toward the target without altering the core model logic.'
- What this solution (achieved 1.23566) has done: 'I adjust the weight‑selection logic so that it directly minimizes the log‑MAE (the true objective) instead of minimizing the absolute gap to the target value. This change keeps the overall workflow unchanged while encouraging a blend that reduces the error, moving the score closer to the desired negative log‑MAE. The per‑type weight search is updated in the same way.'
- What this solution (achieved 1.23566) has done: 'I simplify the blending logic so the prediction relies directly on the summed contributions, which are the exact target values. By fixing the blend weight to 1 (and per‑type weights to 1) we avoid any unnecessary averaging that hurts the log‑MAE, moving the score toward the negative target. The changes keep the original data loading and merging untouched and only adjust the weighting step and the test‑time prediction.'
- What this solution (achieved 1.23566) has done: 'I add a lightweight grid‑search on the validation split to find the blend weight w that minimizes the true log‑MAE (average of log‑MAE per coupling type). The optimal w is then used to combine the summed contributions with the more specific combo‑mean when producing test predictions, which should lower the overall log‑MAE and move the score toward the negative target.'
- What this solution (achieved 1.23566) has done: 'I set the blend weight to 1 so the predictions rely on the exact contribution sum (the true target) and adjust the fallback for missing contribution values to use the more specific `combo_mean` instead of the generic `type_mean`. This keeps the overall workflow unchanged but should lower the log‑MAE toward the negative target.'
- What this solution (achieved 1.23566) has done: 'I keep the overall workflow unchanged but fix the blending weight handling: the script currently overwrites the optimal weight found by the grid‑search with a hard‑coded `best_w = 1.0`, which prevents the best blending of the contribution sum and the combo‑mean from being used. By removing that override and using the weight discovered during validation, the predictions become better calibrated, moving the log‑MAE toward the negative target. I also update the printed message to reflect the actual selected weight.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

BASE_INPUT = os.path.abspath("../input/champs-scalar-coupling")



## === cell 1
train_path = os.path.join(BASE_INPUT, "train.csv")
test_path = os.path.join(BASE_INPUT, "test.csv")
structures_path = os.path.join(BASE_INPUT, "structures.csv")
contrib_path = os.path.join(BASE_INPUT, "scalar_coupling_contributions.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
structures_df = pd.read_csv(structures_path)
contrib_df = pd.read_csv(contrib_path)



## === cell 2
atom_lookup = (
    structures_df[["molecule_name", "atom_index", "atom"]]
    .drop_duplicates()
    .set_index(["molecule_name", "atom_index"])["atom"]
)


def map_atoms(df):
    df = df.copy()
    df["atom0"] = df.apply(
        lambda row: atom_lookup.get((row["molecule_name"], row["atom_index_0"])), axis=1
    )
    df["atom1"] = df.apply(
        lambda row: atom_lookup.get((row["molecule_name"], row["atom_index_1"])), axis=1
    )
    return df


train_ext = map_atoms(train_df)
test_ext = map_atoms(test_df)



## === cell 3
global_mean = train_df["scalar_coupling_constant"].mean()

type_means = (
    train_df.groupby("type")["scalar_coupling_constant"]
    .mean()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "type_mean"})
)

combo_means = (
    train_ext.groupby(["type", "atom0", "atom1"])["scalar_coupling_constant"]
    .mean()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "combo_mean"})
)

contrib_cols = ["fc", "sd", "pso", "dso"]
contrib_df["contrib_sum"] = contrib_df[contrib_cols].sum(axis=1)

train_full = train_ext.merge(
    contrib_df, on=["molecule_name", "atom_index_0", "atom_index_1", "type"], how="left"
)

train_full = train_full.merge(type_means, on="type", how="left")
train_full = train_full.merge(combo_means, on=["type", "atom0", "atom1"], how="left")

train_full["type_mean"].fillna(global_mean, inplace=True)
train_full["combo_mean"].fillna(train_full["type_mean"], inplace=True)
train_full["contrib_sum"].fillna(train_full["combo_mean"], inplace=True)

train_split, val_split = train_test_split(
    train_full, test_size=0.2, random_state=42, stratify=train_full["type"]
)


def log_mae(df, pred):
    mae_per_type = df.groupby("type").apply(
        lambda sub: mean_absolute_error(
            sub["scalar_coupling_constant"], pred.loc[sub.index]
        )
    )
    return np.mean(np.log(mae_per_type + 1e-12))


best_w = 1.0
best_score = np.inf
for w in np.arange(0.0, 2.01, 0.01):
    blended = w * train_split["contrib_sum"] + (1 - w) * train_split["combo_mean"]
    score = log_mae(train_split, blended)
    if score < best_score:
        best_score = score
        best_w = w

print(
    f"Selected blend weight w={best_w:.2f} (log‑MAE on validation ≈ {best_score:.5f})"
)



## === cell 4
test_full = test_ext.merge(
    contrib_df, on=["molecule_name", "atom_index_0", "atom_index_1", "type"], how="left"
)
test_full = test_full.merge(type_means, on="type", how="left")
test_full = test_full.merge(combo_means, on=["type", "atom0", "atom1"], how="left")

test_full["type_mean"].fillna(global_mean, inplace=True)
test_full["combo_mean"].fillna(test_full["type_mean"], inplace=True)
test_full["contrib_sum"].fillna(test_full["combo_mean"], inplace=True)

test_full["scalar_coupling_constant"] = (
    best_w * test_full["contrib_sum"] + (1 - best_w) * test_full["combo_mean"]
)
test_full["scalar_coupling_constant"].fillna(global_mean, inplace=True)



## === cell 5
submission = test_full[["id", "scalar_coupling_constant"]]
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False, float_format="%.6f")
print(f"Submission written to {submission_path}")
