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

-1.572091784711615

# 6. Current score

1.99777

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 3.00563) has done: 'I make the file‑reading robust by using the correct Kaggle input path and falling back to a simple baseline (global mean of the training target) when the original blend files are missing. This guarantees the script runs end‑to‑end and writes a valid `my_blend_1.csv` submission with the required columns.'
- What this solution (achieved 1.23566) has done: 'I replace the simple global‑mean fallback with a per‑coupling‑type mean prediction, which is still a trivial model but usually lowers the log‑MAE substantially. If a type is unseen in training we keep the global mean as a safety net. This change only touches the fallback branch and leaves the blend logic untouched, ensuring the script still runs end‑to‑end and writes a valid CSV while moving the score toward the target.'
- What this solution (achieved 3.00563) has done: 'I replace the simple per‑type mean fallback with a lightweight Ridge regression that uses the scalar‑coupling contributions and molecule‑level features (potential energy and dipole moments) together with one‑hot encoded coupling types. This keeps the overall script structure unchanged, guarantees a valid CSV is written, and introduces just enough additional signal to lower the log‑MAE toward the target without overhauling the core logic.'
- What this solution (achieved 1.23566) has done: 'I replace the fallback‑model part (used when none of the blend files exist) with a simple per‑coupling‑type mean predictor. This keeps the overall script structure, removes the Ridge regression, and uses a baseline that is known to lower the log‑MAE, moving the score toward the target without altering any core blending logic.'
- What this solution (achieved 1.99777) has done: 'I improve the fallback prediction by using the scalar‑coupling contribution terms (fc, sd, pso, dso) when they are available, because their sum equals the true coupling constant. This generally yields a much lower error than the simple per‑type mean, moving the log‑MAE toward the negative target while keeping the overall script structure unchanged. If any contribution columns are missing, the code falls back to the per‑type mean as before.'
- What this solution (achieved 1.99777) has done: 'I enhance the fallback prediction so that when any scalar‑coupling contribution column is missing we do not fall back to a crude per‑type mean. Instead we compute the sum of the available contributions and add a learned residual (the average difference between the true constant and the sum of available contributions for that coupling type from the training data). This keeps the original blend‑logic untouched, adds only a lightweight correction, and is expected to lower the log‑MAE toward the negative target score.'

# 9. Code solution

## === cell 0
import os
import pandas as pd

input_dir = os.path.abspath(os.path.join("..", "input"))
print("Input directory:", input_dir)
print("Available top‑level entries:", os.listdir(input_dir))




## === cell 1
def safe_read(path):
    try:
        df = pd.read_csv(path)
        print(f"Loaded {os.path.basename(path)} with shape {df.shape}")
        return df
    except FileNotFoundError:
        print(f"File not found, skipping: {path}")
        return None


base_path = os.path.join(input_dir, "champs-scalar-coupling")

one = safe_read(os.path.join(base_path, "1.csv"))  # distance
two = safe_read(os.path.join(base_path, "2.csv"))  # LGB + features
three = safe_read(os.path.join(base_path, "3.csv"))  # MPNN
four = safe_read(os.path.join(base_path, "submission-2.csv"))  # GIBA
five = safe_read(os.path.join(base_path, "submission-giba-1.csv"))  # GIBA
six = safe_read(os.path.join(base_path, "workingsubmission-test.csv"))  # unknown
seven = safe_read(os.path.join(base_path, "LGB_2019-07-18_-1.2243.csv"))  # LGB

if any(df is not None for df in [one, two, three, four, five, six, seven]):
    submission = pd.DataFrame()
    ref = next(df for df in [one, two, three, four, five, six, seven] if df is not None)
    submission["id"] = ref["id"]

    submission["scalar_coupling_constant"] = 0.0

    weights = {
        "one": 0.17,
        "two": 0.17,
        "three": 0.17,
        "four": 0.16,
        "five": 0.16,
        "six": 0.17,
    }

    if one is not None:
        submission["scalar_coupling_constant"] += (
            weights["one"] * one["scalar_coupling_constant"]
        )
    if two is not None:
        submission["scalar_coupling_constant"] += (
            weights["two"] * two["scalar_coupling_constant"]
        )
    if three is not None:
        submission["scalar_coupling_constant"] += (
            weights["three"] * three["scalar_coupling_constant"]
        )
    if four is not None:
        submission["scalar_coupling_constant"] += (
            weights["four"] * four["scalar_coupling_constant"]
        )
    if five is not None:
        submission["scalar_coupling_constant"] += (
            weights["five"] * five["scalar_coupling_constant"]
        )
    if six is not None:
        submission["scalar_coupling_constant"] += (
            weights["six"] * six["scalar_coupling_constant"]
        )

else:
    import numpy as np

    train_path = os.path.join(base_path, "train.csv")
    test_path = os.path.join(base_path, "test.csv")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    contrib = safe_read(os.path.join(base_path, "scalar_coupling_contributions.csv"))
    potential = safe_read(os.path.join(base_path, "potential_energy.csv"))
    dipole = safe_read(os.path.join(base_path, "dipole_moments.csv"))

    if contrib is not None:
        train = train.merge(
            contrib,
            on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
            how="left",
        )
        test = test.merge(
            contrib,
            on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
            how="left",
        )
    if potential is not None:
        train = train.merge(potential, on="molecule_name", how="left")
        test = test.merge(potential, on="molecule_name", how="left")
    if dipole is not None:
        train = train.merge(dipole, on="molecule_name", how="left")
        test = test.merge(dipole, on="molecule_name", how="left")

    contribution_cols = ["fc", "sd", "pso", "dso"]

    if all(col in test.columns for col in contribution_cols):
        train_sum = train[contribution_cols].sum(axis=1, skipna=True)
        test_sum = test[contribution_cols].sum(axis=1, skipna=True)

        train_residual = train["scalar_coupling_constant"] - train_sum
        type_residual_mean = train_residual.groupby(train["type"]).mean()

        global_residual_mean = train_residual.mean()

        missing_mask = test[contribution_cols].isnull().any(axis=1)

        pred = test_sum.copy()

        if missing_mask.any():
            residual_estimate = (
                test.loc[missing_mask, "type"]
                .map(type_residual_mean)
                .fillna(global_residual_mean)
            )
            pred.loc[missing_mask] = pred.loc[missing_mask] + residual_estimate

        if pred.isnull().any():
            global_mean = train["scalar_coupling_constant"].mean()
            type_means = train.groupby("type")["scalar_coupling_constant"].mean()
            fallback = test["type"].map(type_means).fillna(global_mean)
            pred = pred.fillna(fallback)

    else:
        global_mean = train["scalar_coupling_constant"].mean()
        type_means = train.groupby("type")["scalar_coupling_constant"].mean()
        pred = test["type"].map(type_means).fillna(global_mean).values

    submission = pd.DataFrame({"id": test["id"], "scalar_coupling_constant": pred})

output_path = "my_blend_1.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path} with shape {submission.shape}")
