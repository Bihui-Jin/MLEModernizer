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

1.23566

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'I replace the missing‑file blending code with a simple, reliable baseline: compute the average `scalar_coupling_constant` for each coupling `type` from the training set and use those averages to predict the test set (fallback to the overall mean when a type is unseen). This removes the file‑not‑found error, guarantees a valid CSV submission, and provides a reasonable score that moves toward the target without altering any core modeling logic.'
- What this solution (achieved 1.23566) has done: 'I replace the simple per‑type mean prediction with a prediction built from the average coupling contributions (fc, sd, pso, dso) for each coupling type. These contributions sum to the target constant, so using their averaged sum should lower the MAE and move the score toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 1.23566) has done: 'I keep the original per‑type contribution‑based prediction but also compute a per‑type average of the target itself from the training data. By blending the two (each captures slightly different information) and still falling back to the global mean when a type is unseen, the predictions become a bit more calibrated, which should lower the MAE and move the log‑MAE score closer to the negative target without altering the overall pipeline or model architecture.'
- What this solution (achieved 1.23566) has done: 'I replace the blended prediction with a simpler per‑type mean prediction, which is more directly aligned with the target variable and expected to lower the log‑MAE (moving the score toward the negative target). Missing coupling types still fall back to the global mean, preserving the original fallback logic.'
- What this solution (achieved 1.18497) has done: 'I replace the per‑type mean prediction with a per‑type median prediction (and use the global median as fallback). Medians are more robust to outliers, so this small change should lower the MAE and move the log‑MAE score closer to the negative target while keeping the overall pipeline unchanged.'
- What this solution (achieved 1.19011) has done: 'I keep the original data loading and median‑based baseline, but add a lightweight per‑type contribution prediction (average fc + sd + pso + dso) and blend it with the median prediction. This small calibration step is expected to lower the MAE (and thus the log‑MAE) without altering the core pipeline, moving the score closer to the negative target.'
- What this solution (achieved 1.23566) has done: 'The plan is to replace the per‑type median `scalar_coupling_constant` with the per‑type mean (which aligns directly with the target), use the global mean as a fallback, and blend the two predictions equally (0.5 × mean + 0.5 × contribution‑sum). This small change keeps the original pipeline but should lower the MAE, moving the log‑MAE score toward the negative target.'
- What this solution (achieved 1.23566) has done: 'I keep the overall pipeline unchanged but adjust the blending of predictions to rely more on the physics‑based contribution sums, which are typically more accurate than the simple per‑type scalar mean. By weighting the contribution‑based prediction higher (70 % contribution + 30 % scalar mean) we expect a lower MAE and thus a log‑MAE closer to the negative target. The rest of the code, including data loading and CSV output, remains the same.'

# 9. Code solution

## === cell 0
import os
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join("..", "input", "champs-scalar-coupling"))

print("Available files in BASE_DIR:")
print(os.listdir(BASE_DIR)[:10])



## === cell 1
train_path = os.path.join(BASE_DIR, "train.csv")
test_path = os.path.join(BASE_DIR, "test.csv")
contrib_path = os.path.join(BASE_DIR, "scalar_coupling_contributions.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
contrib = pd.read_csv(contrib_path)

print(
    f"Train shape: {train.shape}, Test shape: {test.shape}, Contributions shape: {contrib.shape}"
)



## === cell 2
global_mean = train["scalar_coupling_constant"].mean()

type_scalar_means = (
    train.groupby("type")["scalar_coupling_constant"]
    .mean()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "pred_scalar_mean"})
)

type_contrib_means = (
    contrib.groupby("type")[["fc", "sd", "pso", "dso"]]
    .mean()
    .sum(axis=1)
    .reset_index()
    .rename(columns={0: "pred_contrib_sum"})
)

print("Number of coupling types with scalar means:", type_scalar_means.shape[0])
print("Number of coupling types with contribution means:", type_contrib_means.shape[0])
print("Global mean scalar coupling constant:", global_mean)



## === cell 3
test = test.merge(type_scalar_means, on="type", how="left")
test = test.merge(type_contrib_means, on="type", how="left")

test["pred_scalar_mean"].fillna(global_mean, inplace=True)
test["pred_contrib_sum"].fillna(global_mean, inplace=True)

test["pred_scalar"] = 0.3 * test["pred_scalar_mean"] + 0.7 * test["pred_contrib_sum"]

submission = (
    test[["id", "pred_scalar"]]
    .rename(columns={"pred_scalar": "scalar_coupling_constant"})
    .copy()
)



## === cell 4
output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}, rows: {submission.shape[0]}")
