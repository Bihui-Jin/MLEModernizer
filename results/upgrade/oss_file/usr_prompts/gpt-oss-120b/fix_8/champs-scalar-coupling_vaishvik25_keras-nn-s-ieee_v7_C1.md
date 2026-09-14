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

-1.5192329259486128

# 6. Current score

1.23628

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'The script is updated to load the actual competition data (train and test) from the correct input directory, compute a simple but effective baseline prediction (the mean scalar coupling constant for each coupling type, falling back to the overall mean), and write a valid `submission.csv` file with the required columns. All previous path errors and undefined variables are removed, and the code now runs end‑to‑end producing a usable submission that should bring the score closer to the target.'
- What this solution (achieved 1.23566) has done: 'I add a lightweight feature‑engineering step that joins atom element information from structures.csv to the train and test data, then compute mean coupling constants for each (type, atom0, atom1) combination. Predictions first use this finer‑grained mean, falling back to the type mean and finally the global mean. This small change keeps the original workflow but should lower the log‑MAE toward the target score.'
- What this solution (achieved 1.23572) has done: 'I add a simple shrinkage (regularisation) to the per‑combo and per‑type means so that groups with few samples are pulled toward the global mean. This modest adjustment keeps the original workflow but should reduce over‑fitting and thus lower the log‑MAE, moving the score closer to the target. The changes are limited to the statistics computation and prediction logic.'
- What this solution (achieved 1.23584) has done: 'I make two modest adjustments that keep the original mean‑shrinkage approach while giving it more data to work with and reducing over‑fitting. First, I treat each atom pair symmetrically (so (C,H) and (H,C) are combined) which enlarges the groups used for the per‑combo statistics. Second, I increase the shrinkage strength from 10 to 30 so that small groups are pulled nearer the global mean. These changes are small, preserve the core logic, and are expected to lower the log‑MAE toward the target score.'
- What this solution (achieved 1.23569) has done: 'I reduce the regularisation strength by changing `shrink_lambda` from 30 to 5. A smaller λ lets the per‑combo and per‑type statistics retain more of their original values instead of being pulled toward the global mean, which should lower the log‑MAE and move the score closer to the negative target while preserving the original workflow.'
- What this solution (achieved 1.23878) has done: 'I add a lightweight dipole‑moment bias: compute the magnitude of each molecule’s dipole vector, fit a simple linear correction on the training residuals, and apply it to the test predictions. This keeps the original mean‑shrinkage logic while giving a modest performance gain that moves the log‑MAE closer to the negative target.'
- What this solution (achieved 1.23628) has done: 'I add a per‑coupling‑type dipole correction instead of a single global linear adjustment. The code now fits a separate slope `a` and intercept `b` for each coupling `type` using the training residuals, stores them in a dictionary, and applies the appropriate correction when predicting the test set. This small change keeps the original mean‑shrinkage logic while giving the model a more tailored bias correction, which should lower the log‑MAE toward the target score.'

# 9. Code solution

## === cell 0
import os, glob
import pandas as pd
import numpy as np

train_path = glob.glob(os.path.join("..", "input", "**", "train.csv"), recursive=True)[
    0
]
test_path = glob.glob(os.path.join("..", "input", "**", "test.csv"), recursive=True)[0]

print(f"Found train: {train_path}")
print(f"Found test : {test_path}")



## === cell 1
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)
print(train_df.head())



## === cell 2
structures_path = glob.glob(
    os.path.join("..", "input", "**", "structures.csv"), recursive=True
)[0]
structures_df = pd.read_csv(structures_path)

atom0 = structures_df.rename(columns={"atom_index": "atom_index_0", "atom": "atom0"})[
    ["molecule_name", "atom_index_0", "atom0"]
]

atom1 = structures_df.rename(columns={"atom_index": "atom_index_1", "atom": "atom1"})[
    ["molecule_name", "atom_index_1", "atom1"]
]

train_df = train_df.merge(atom0, on=["molecule_name", "atom_index_0"], how="left")
train_df = train_df.merge(atom1, on=["molecule_name", "atom_index_1"], how="left")

test_df = test_df.merge(atom0, on=["molecule_name", "atom_index_0"], how="left")
test_df = test_df.merge(atom1, on=["molecule_name", "atom_index_1"], how="left")

train_df["atom_min"] = np.where(
    train_df["atom0"] <= train_df["atom1"], train_df["atom0"], train_df["atom1"]
)
train_df["atom_max"] = np.where(
    train_df["atom0"] > train_df["atom1"], train_df["atom0"], train_df["atom1"]
)

test_df["atom_min"] = np.where(
    test_df["atom0"] <= test_df["atom1"], test_df["atom0"], test_df["atom1"]
)
test_df["atom_max"] = np.where(
    test_df["atom0"] > test_df["atom1"], test_df["atom0"], test_df["atom1"]
)

print("After merging atom types and creating symmetric keys:")
print(train_df[["type", "atom0", "atom1", "atom_min", "atom_max"]].head())



## === cell 3
global_mean = train_df["scalar_coupling_constant"].mean()
shrink_lambda = 5.0

combo_stats = train_df.groupby(["type", "atom_min", "atom_max"])[
    "scalar_coupling_constant"
].agg(["sum", "count"])
combo_shrink = (combo_stats["sum"] + shrink_lambda * global_mean) / (
    combo_stats["count"] + shrink_lambda
)

type_stats = train_df.groupby("type")["scalar_coupling_constant"].agg(["sum", "count"])
type_shrink = (type_stats["sum"] + shrink_lambda * global_mean) / (
    type_stats["count"] + shrink_lambda
)

print("Combo groups:", combo_shrink.shape[0])
print("Type groups :", type_shrink.shape[0])
print("Global mean :", global_mean)



## === cell 4
dipole_path = glob.glob(
    os.path.join("..", "input", "**", "dipole_moments.csv"), recursive=True
)[0]
dipole_df = pd.read_csv(dipole_path)
dipole_df["dipole_mag"] = np.sqrt(
    dipole_df["X"] ** 2 + dipole_df["Y"] ** 2 + dipole_df["Z"] ** 2
)

train_df = train_df.merge(
    dipole_df[["molecule_name", "dipole_mag"]], on="molecule_name", how="left"
)
test_df = test_df.merge(
    dipole_df[["molecule_name", "dipole_mag"]], on="molecule_name", how="left"
)



## === cell 5
combo_shrink_dict = combo_shrink.to_dict()
type_shrink_dict = type_shrink.to_dict()


def base_predict(row):
    key = (row["type"], row["atom_min"], row["atom_max"])
    if key in combo_shrink_dict:
        return combo_shrink_dict[key]
    if row["type"] in type_shrink_dict:
        return type_shrink_dict[row["type"]]
    return global_mean


train_base_pred = train_df.apply(base_predict, axis=1)

residual = train_df["scalar_coupling_constant"] - train_base_pred

valid_mask = ~np.isnan(residual) & ~np.isnan(train_df["dipole_mag"])
type_groups = train_df.loc[valid_mask].groupby("type")

dipole_corrections = {}  # type -> (a, b)
for t, grp in type_groups:
    if len(grp) >= 2:  # need at least two points for polyfit
        a, b = np.polyfit(grp["dipole_mag"], residual.loc[grp.index], 1)
    else:
        a, b = 0.0, 0.0
    dipole_corrections[t] = (a, b)

global_a, global_b = np.polyfit(
    train_df.loc[valid_mask, "dipole_mag"], residual.loc[valid_mask], 1
)

print("Per‑type dipole corrections fitted (example):")
example_type = next(iter(dipole_corrections))
print(
    f"Type {example_type}: a={dipole_corrections[example_type][0]:.6f}, b={dipole_corrections[example_type][1]:.6f}"
)



## === cell 6
test_base_pred = test_df.apply(base_predict, axis=1)

global_dipole_mag = dipole_df["dipole_mag"].mean()
test_dipole = test_df["dipole_mag"].fillna(global_dipole_mag)


def apply_dipole_correction(row):
    a, b = dipole_corrections.get(row["type"], (global_a, global_b))
    return a * row["dipole_mag_filled"] + b


test_df["dipole_mag_filled"] = test_dipole

test_pred = test_base_pred + test_df.apply(apply_dipole_correction, axis=1)

submission = pd.DataFrame({"id": test_df["id"], "scalar_coupling_constant": test_pred})

print(submission.head())



## === cell 7
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False, float_format="%.6f")
print(f"Submission written to {submission_path}")
