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

0.4801548222909683

# 6. Current score

1.23566

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'The script failed because it tried to access a non‑existent **type** column in the `submission` DataFrame (which only contained `id`). The fix uses the original `test` DataFrame to map coupling types to their mean target values, then fills missing types with the overall mean. This resolves the `KeyError` and allows a proper CSV submission to be written.'
- What this solution (achieved 3.00563) has done: 'The fix corrects how predictions are looked up: it now reindexes the pre‑computed type‑atom means onto the train and test rows, then fills missing values with type‑level and global means. This resolves the index‑alignment errors and the missing target column in the test set, allowing a valid `submission.csv` to be written and the metric to be computed without crashes.'
- What this solution (achieved 3.00563) has done: 'Implemented a fix to the metric calculation by passing pandas Series instead of NumPy arrays, restoring the `.abs()` method and correct grouping. Re‑ordered cells to start at 1 as required and kept the original modeling logic unchanged. The script now runs end‑to‑end, computes a valid training log‑MAE, and writes a proper `submission.csv` file.'
- What this solution (achieved 1.23566) has done: 'The fix aligns prediction series with the original row order by mapping the multi‑index means back to a simple RangeIndex, preventing the join error during metric calculation. It also updates the test‑set prediction using the same mapping logic and renumbers the notebook cells to start at 1 as required. No core modeling logic is changed, so the score move toward the target without altering the algorithm.'
- What this solution (achieved 1.23566) has done: 'Implemented a lightweight blend of the original type/atom mean predictions with mean scalar‑coupling contributions per coupling type. This adds a modest amount of signal (from `scalar_coupling_contributions.csv`) without altering the core mean‑encoding logic, aiming to lower the log‑MAE toward the target. The script now reads the contributions, computes per‑type average sums, and combines them with the existing predictions using a 70/30 weighting. Cell numbering has been renumbered to start at 1 as required.'
- What this solution (achieved 1.23566) has done: 'I lower the blending weight for the simple type‑atom mean (from 0.7 to 0.3) so that the prediction relies more on the scalar‑coupling contributions, which are a closer physical approximation of the target. This small change keeps the overall logic unchanged while expected to reduce the log‑MAE toward the target value.'
- What this solution (achieved 1.23566) has done: 'I keep the overall feature engineering and blending approach but replace the original type‑atom mean prediction with a simpler hierarchy: first use the per‑type mean (fallback to the global mean) and then blend it with the physics‑based contribution estimate. Increasing the contribution weight (to 0.8) gives the model more signal from the domain‑specific features while still preserving the original mean‑encoding logic, which should lower the log‑MAE toward the target. I also renumber the cells to start at 1 as required.'
- What this solution (achieved 1.23566) has done: 'I add a more specific “type‑atom” mean prediction (using the type + atom0 + atom1 hierarchy) and blend it with the physics‑based contribution estimate. The blend weight for the contribution is reduced to give the richer hierarchical mean more influence, which should lower the log‑MAE toward the target while keeping the original logic intact.'

# 9. Code solution

## === cell 0
import os, random, sys
import numpy as np
import pandas as pd




## === cell 1
SEED = 31
TARGET = "scalar_coupling_constant"
PREDICTION = TARGET


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


seed_everything(SEED)




## === cell 2
def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    """
    Fast metric computation for this competition:
    https://www.kaggle.com/c/champs-scalar-coupling
    """
    maes = (y_true - y_pred).abs().groupby(types).mean()
    maes = np.log(maes.map(lambda x: max(x, floor)))
    return maes.mean()




## === cell 3
train_path = "../input/champs-scalar-coupling/train.csv"
test_path = "../input/champs-scalar-coupling/test.csv"
structures_path = "../input/champs-scalar-coupling/structures.csv"
contrib_path = "../input/champs-scalar-coupling/scalar_coupling_contributions.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
structures = pd.read_csv(structures_path)
contrib = pd.read_csv(contrib_path)

atom_lookup = structures.set_index(["molecule_name", "atom_index"])["atom"]


def add_atom_elements(df):
    """Add element symbols for the two atoms of each coupling."""
    df = df.copy()
    df["atom0"] = df.apply(
        lambda row: atom_lookup.get((row["molecule_name"], row["atom_index_0"])), axis=1
    )
    df["atom1"] = df.apply(
        lambda row: atom_lookup.get((row["molecule_name"], row["atom_index_1"])), axis=1
    )
    return df


train_ext = add_atom_elements(train)
test_ext = add_atom_elements(test)

group_key = ["type", "atom0", "atom1"]
type_atom_means = train_ext.groupby(group_key)[TARGET].mean()
type_means = train.groupby("type")[TARGET].mean()
global_mean = train[TARGET].mean()

type_contrib_sum = (
    contrib.groupby("type")[["fc", "sd", "pso", "dso"]].mean().sum(axis=1)
)
overall_contrib_mean = contrib[["fc", "sd", "pso", "dso"]].mean().sum()
type_contrib_sum = type_contrib_sum.fillna(overall_contrib_mean)




## === cell 4
base_pred_train = train_ext.apply(
    lambda row: type_atom_means.get((row["type"], row["atom0"], row["atom1"])), axis=1
)
base_pred_train = base_pred_train.fillna(train_ext["type"].map(type_means)).fillna(
    global_mean
)

base_pred_test = test_ext.apply(
    lambda row: type_atom_means.get((row["type"], row["atom0"], row["atom1"])), axis=1
)
base_pred_test = base_pred_test.fillna(test_ext["type"].map(type_means)).fillna(
    global_mean
)

contrib_est_train = train_ext["type"].map(type_contrib_sum).fillna(overall_contrib_mean)
contrib_est_test = test_ext["type"].map(type_contrib_sum).fillna(overall_contrib_mean)

weight_contrib = 0.6
train_pred = (1 - weight_contrib) * base_pred_train + weight_contrib * contrib_est_train
test_pred = (1 - weight_contrib) * base_pred_test + weight_contrib * contrib_est_test

train_score = group_mean_log_mae(train[TARGET], train_pred, train["type"])
print(f"Train metric (log‑MAE): {train_score:.5f}")




## === cell 5
submission = test[["id"]].copy()
submission[TARGET] = test_pred.values

submission.to_csv("submission.csv", index=False)
print("Submission saved. Current directory contents:")
print(os.listdir("."))
