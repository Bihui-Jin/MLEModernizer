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

-1.5801604792430517

# 6. Current score

1.17339

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.17332) has done: 'I replace the missing‑file blending code with a self‑contained baseline that reads the provided training data, computes simple mean targets per coupling type (and optionally per atom pair), merges these averages onto the test set, and writes a correctly‑named CSV submission. This fixes the `FileNotFoundError`, ensures a valid `my_blend_1.csv` is produced, and gives a reasonable starting score without altering any core modeling logic.'
- What this solution (achieved 1.17379) has done: 'I keep the same overall structure but improve the simple‑mean baseline by smoothing the pair‑wise averages. For each (type, atom_index_0, atom_index_1) I also store the count of training examples and blend the pair mean with the type mean, giving less weight to noisy pair means that appear only a few times. This small statistical tweak is expected to reduce the log‑MAE and move the score closer to the negative target without altering any core modeling logic.'
- What this solution (achieved 1.17337) has done: 'I keep the same baseline logic but tighten the smoothing factor that blends pair‑level means with type‑level means. Using a smaller `SMOOTH` (e.g., 1.0 instead of 10.0) lets well‑observed atom‑pair statistics influence the prediction more strongly while still shrinking noisy pairs toward the type average, which should lower the MAE and move the log‑MAE score closer to the negative target.'
- What this solution (achieved 1.17864) has done: 'I add a symmetric pair identifier so that (atom 0, atom 1) and (atom 1, atom 0) share the same statistics, and I increase the smoothing toward the type mean (and a small pull toward the global mean) by using larger smoothing constants. This modest change keeps the original mean‑blending logic while giving more robust estimates and is expected to lower the log‑MAE toward the negative target.'
- What this solution (achieved 1.17332) has done: 'I lower the smoothing constants so the prediction relies directly on the observed pair means (when available) and falls back to the type or global mean otherwise. This removes the extra pull toward higher‑level averages, which should reduce the MAE and thus move the log‑MAE score closer to the negative target while keeping the core logic unchanged.'
- What this solution (achieved 1.17345) has done: 'I add a small count‑threshold rule so that noisy atom‑pair averages fall back to the more reliable coupling‑type mean (and finally to the global mean). This modest change keeps the original mean‑blending approach but should reduce error for rare pairs, moving the log‑MAE lower toward the negative target. The script now imports numpy, defines `MIN_COUNT`, and applies the conditional logic before writing the submission.'
- What this solution (achieved 1.17379) has done: 'I replace the hard `MIN_COUNT` threshold with a smooth blending of pair‑level and type‑level means, using a small smoothing constant. This keeps the same overall mean‑based logic but lets well‑observed pairs influence the prediction more while still shrinking noisy pairs toward the more reliable type mean, which should lower the log‑MAE and move the score toward the negative target.'
- What this solution (achieved 1.17339) has done: 'I lower the smoothing factor and add a small count‑threshold so that only atom‑pair statistics with at least a few observations are used; rarer pairs fall back to the more reliable coupling‑type mean. This reduces over‑smoothing and should lower the log‑MAE, moving the score closer to the negative target while keeping the overall mean‑blending logic unchanged.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

INPUT_ROOT = "../input/champs-scalar-coupling"
assert os.path.isdir(INPUT_ROOT), f"Input directory {INPUT_ROOT} not found"



## === cell 1
train_path = os.path.join(INPUT_ROOT, "train.csv")
test_path = os.path.join(INPUT_ROOT, "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_df["pair_0"] = train_df[["atom_index_0", "atom_index_1"]].min(axis=1)
train_df["pair_1"] = train_df[["atom_index_0", "atom_index_1"]].max(axis=1)

test_df["pair_0"] = test_df[["atom_index_0", "atom_index_1"]].min(axis=1)
test_df["pair_1"] = test_df[["atom_index_0", "atom_index_1"]].max(axis=1)

type_means = (
    train_df.groupby("type")["scalar_coupling_constant"]
    .mean()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "type_mean"})
)

pair_stats = (
    train_df.groupby(["type", "pair_0", "pair_1"])["scalar_coupling_constant"]
    .agg(["mean", "count"])
    .reset_index()
    .rename(columns={"mean": "pair_mean", "count": "pair_count"})
)

test_pred = test_df.merge(type_means, on="type", how="left")
test_pred = test_pred.merge(pair_stats, on=["type", "pair_0", "pair_1"], how="left")

global_mean = train_df["scalar_coupling_constant"].mean()

SMOOTH = 1.0
MIN_COUNT = 3

pair_present = test_pred["pair_count"].notna()
pair_enough = pair_present & (test_pred["pair_count"] >= MIN_COUNT)

blended = (
    test_pred["pair_mean"] * test_pred["pair_count"] + test_pred["type_mean"] * SMOOTH
) / (test_pred["pair_count"] + SMOOTH)

test_pred["prediction"] = np.where(pair_enough, blended, test_pred["type_mean"])
test_pred["prediction"] = test_pred["prediction"].fillna(global_mean)

submission = test_pred[["id", "prediction"]].rename(
    columns={"prediction": "scalar_coupling_constant"}
)

submission_path = "my_blend_1.csv"
submission.to_csv(submission_path, index=False)

print(f"Submission written to {submission_path} with {submission.shape[0]:,} rows.")
