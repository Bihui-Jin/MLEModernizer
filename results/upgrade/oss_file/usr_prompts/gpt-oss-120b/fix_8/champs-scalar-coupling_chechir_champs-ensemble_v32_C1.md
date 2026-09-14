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

-2.4226869287246378

# 6. Current score

1.90752

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'I replace the failing ensemble‑loading logic with a simple, self‑contained baseline: read the competition’s train and test files, compute the mean scalar coupling constant for each coupling type in the training set, and use those means as predictions for the test set (filling any missing types with the overall mean). This eliminates all missing‑file errors, ensures the required columns exist, and writes a valid `ensemble_sub.csv` submission file. The core logic is unchanged apart from the prediction method, keeping the solution minimal and functional.'
- What this solution (achieved 3.00563) has done: 'I fixed the mismatch that caused pandas to raise a “cannot join with no overlapping index names” error when building the submission DataFrame. The predictions Series now has its index reset (or converted to a NumPy array) so it aligns correctly with the `id` column. This change ensures a valid CSV is written without altering the core modelling logic.'
- What this solution (achieved 3.00628) has done: 'I fix the mismatch between the prediction Series index and the test DataFrame index by converting the predictions to a plain NumPy array before building the submission. This resolves the “cannot join with no overlapping index names” error and ensures a valid CSV is written. No other logic is altered, preserving the original model and scoring approach.'
- What this solution (achieved 1.90752) has done: 'I fixed the indexing error when fitting per‑type linear adjustments: instead of trying to locate validation predictions by the original row indices (which don’t exist in the series indexed by `(type, atom_0, atom_1)`), I store the validation predictions as a plain NumPy array and select the relevant values using a boolean mask on the `type` column. This resolves the KeyError, allows the coefficient fitting loop to run, and produces a valid `submission` DataFrame that can be printed.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

DATA_ROOT = "../input/champs-scalar-coupling"
TRAIN_PATH = os.path.join(DATA_ROOT, "train.csv")
TEST_PATH = os.path.join(DATA_ROOT, "test.csv")
STRUCTURES_PATH = os.path.join(DATA_ROOT, "structures.csv")
SUBMISSION_PATH = "ensemble_sub.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

required_train_cols = {"id", "type", "scalar_coupling_constant"}
required_test_cols = {"id", "type"}
assert required_train_cols.issubset(
    train.columns
), "Train file missing required columns"
assert required_test_cols.issubset(test.columns), "Test file missing required columns"

structures = pd.read_csv(STRUCTURES_PATH)  # molecule_name, atom_index, atom, x, y, z
atom_lookup = structures.set_index(["molecule_name", "atom_index"])["atom"]


def attach_atom_elements(df):
    df["atom_0"] = df.apply(
        lambda row: atom_lookup.get((row["molecule_name"], row["atom_index_0"])), axis=1
    )
    df["atom_1"] = df.apply(
        lambda row: atom_lookup.get((row["molecule_name"], row["atom_index_1"])), axis=1
    )
    return df


train = attach_atom_elements(train)
test = attach_atom_elements(test)

type_atom_medians = train.groupby(["type", "atom_0", "atom_1"])[
    "scalar_coupling_constant"
].median()
type_medians = train.groupby("type")["scalar_coupling_constant"].median()
overall_median = train["scalar_coupling_constant"].median()

test_preds_series = (
    test.set_index(["type", "atom_0", "atom_1"])
    .index.to_series()
    .map(type_atom_medians)
    .fillna(test["type"].map(type_medians))
    .fillna(overall_median)
)
test_preds = test_preds_series.values  # aligned with test["id"]

val_mask = np.random.RandomState(42).rand(len(train)) < 0.1
train_split = train[~val_mask]
val_split = train[val_mask]

type_atom_medians_split = train_split.groupby(["type", "atom_0", "atom_1"])[
    "scalar_coupling_constant"
].median()
type_medians_split = train_split.groupby("type")["scalar_coupling_constant"].median()
overall_median_split = train_split["scalar_coupling_constant"].median()

val_preds_series = (
    val_split.set_index(["type", "atom_0", "atom_1"])
    .index.to_series()
    .map(type_atom_medians_split)
    .fillna(val_split["type"].map(type_medians_split))
    .fillna(overall_median_split)
)
val_preds = val_preds_series.values  # aligned with val_split rows

A_global = np.vstack([val_preds, np.ones_like(val_preds)]).T
global_a, global_b = np.linalg.lstsq(
    A_global, val_split["scalar_coupling_constant"].values, rcond=None
)[0]

coeffs = {}
for typ, grp in val_split.groupby("type"):
    mask = (val_split["type"] == typ).values
    preds_grp = val_preds[mask]
    targets_grp = grp["scalar_coupling_constant"].values
    if len(preds_grp) > 1:
        A = np.vstack([preds_grp, np.ones_like(preds_grp)]).T
        a, b = np.linalg.lstsq(A, targets_grp, rcond=None)[0]
    else:
        a, b = global_a, global_b
    coeffs[typ] = (a, b)

adjusted_test_preds = []
for pred, typ in zip(test_preds, test["type"]):
    a, b = coeffs.get(typ, (global_a, global_b))
    adjusted_test_preds.append(a * pred + b)
test_preds = np.array(adjusted_test_preds)

submission = pd.DataFrame({"id": test["id"], "scalar_coupling_constant": test_preds})
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission written to {SUBMISSION_PATH}, shape: {submission.shape}")



## === cell 1
print(submission.head())
