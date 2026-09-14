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

-1.3300563636911908

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'I replace the failing file‑reading and blending logic with a simple, robust baseline: load the training data, compute the average `scalar_coupling_constant` for each coupling `type`, and use those averages as predictions for the test set (falling back to the overall mean when a type is missing). The script writes a correctly‑named CSV submission and includes a quick histogram of the predictions.'
- What this solution (achieved 1.23566) has done: 'I add atom‑type information from structures.csv and use the average scalar_coupling_constant grouped by (type, atom_0, atom_1) as the prediction, falling back to the per‑type mean and finally the overall mean. This keeps the simple averaging logic while providing more granular statistics, which should lower the log‑MAE toward the target score.'
- What this solution (achieved 1.23566) has done: 'I keep the data‑loading and merging steps, add a Euclidean distance feature, fit a tiny per‑type linear regression (slope + intercept) on that distance, and use the regression output as the prediction, falling back to the per‑type mean and finally the overall mean when needed. This small model adds only a few lines, preserves the original averaging logic as a safety net, and is expected to move the log‑MAE closer to the target (lower score). The script also correctly writes a “submission.csv” file.'
- What this solution (achieved 1.23566) has done: 'I add a more granular mean‑based prediction that first looks up the average scalar coupling for each exact `(type, atom_0, atom_1)` combination. If that specific combo is absent, the code falls back to the per‑type mean, and finally to the overall mean. This small change keeps the original averaging logic and safety nets while giving a finer‑grained estimate, which should move the log‑MAE lower toward the target score.'
- What this solution (achieved 1.23566) has done: 'I add a small blending step to the fallback prediction: when a specific atom‑type combo is missing we now combine the per‑type linear‑distance regression with the per‑type mean (70 % regression, 30 % mean). I also clip each prediction to the observed min‑max range for its coupling type, preventing extreme out‑of‑distribution values that can hurt log‑MAE. These lightweight tweaks keep the original averaging logic while tightening predictions, moving the score toward the lower target.'
- What this solution (achieved 1.18497) has done: 'I fix the bug where the variable `t` was undefined in the fallback logic, replace the averaging statistics with more robust medians (combo‑level and type‑level) and keep the existing simple regression blend. Using medians should reduce absolute errors, moving the log‑MAE closer to the negative target while preserving the overall model structure.'
- What this solution (achieved 1.18497) has done: 'I tighten the fallback logic so that when a specific `(type, atom_0, atom_1)` combo is missing we rely directly on the per‑type distance regression (instead of a 0.7/0.3 blend with the median). I also make the regression fallback use the type‑level median as the intercept when there are too few distance points, which is more robust for MAE. These minimal adjustments keep the overall averaging framework intact while giving clearer, less noisy predictions, helping to lower the log‑MAE toward the target.'
- What this solution (achieved 1.18497) has done: 'I blend the distance‑based regression with the per‑type median instead of using the regression alone. This keeps the original simple averaging framework but makes predictions less extreme, which should lower the log‑MAE and move the score toward the negative target. The change is limited to the prediction logic and does not alter the overall model architecture or data handling.'
- What this solution (achieved 1.23566) has done: 'I replace the blended regression‑median fallback with a simpler, more stable per‑type mean (or median) fallback. This removes the noisy linear‑regression component for unseen atom‑type combos, keeping the overall averaging framework unchanged while likely reducing large errors and moving the log‑MAE from the current positive value toward the negative target. The rest of the pipeline—including distance calculation, combo‑level medians, clipping, and CSV writing—remains identical.'
- What this solution (achieved 2.56707) has done: 'I replace the fallback logic with a more robust median‑based per‑type baseline and add a light linear‑distance regression blend when a specific atom‑type combo is missing. This keeps the overall averaging framework, only tweaks the predict_row function, and should lower the log‑MAE toward the negative target while still writing a correct submission.csv.'
- What this solution (achieved 2.90939) has done: 'I tighten the fallback logic so it relies on the per‑type linear distance regression (or the per‑type mean when a regression cannot be built) instead of blending with a median. This removes the noisy 0.5 + 0.5 blend, uses the already‑computed per‑type means for defaults, and keeps all existing steps (feature creation, clipping, CSV writing) unchanged, which should lower the MAE and move the log‑MAE score closer to the negative target.'
- What this solution (achieved 1.18497) has done: 'I improve the prediction logic by (1) increasing combo‑median coverage: when a (type, atom_0, atom_1) lookup fails we also try the reversed atom order, and (2) simplifying the fallback to use the per‑type median (or overall median) instead of the distance‑based linear regression, which is noisy and hurts the log‑MAE. These minimal tweaks keep the overall architecture unchanged while expected to lower the score toward the negative target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_PATH = os.path.abspath(os.path.join("..", "input", "champs-scalar-coupling"))
print("Data directory:", BASE_PATH)
print("Available files (sample):", os.listdir(BASE_PATH)[:5])

train_path = os.path.join(BASE_PATH, "train.csv")
test_path = os.path.join(BASE_PATH, "test.csv")
sample_sub_path = os.path.join(BASE_PATH, "sample_submission.csv")
structures_path = os.path.join(BASE_PATH, "structures.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_sub_path)
structures_df = pd.read_csv(structures_path)

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)
print("Structures shape:", structures_df.shape)

train_df = train_df.merge(
    structures_df.rename(columns={"atom_index": "atom_index_0", "atom": "atom_0"}),
    on=["molecule_name", "atom_index_0"],
    how="left",
)
train_df = train_df.merge(
    structures_df.rename(columns={"atom_index": "atom_index_1", "atom": "atom_1"}),
    on=["molecule_name", "atom_index_1"],
    how="left",
)

test_df = test_df.merge(
    structures_df.rename(columns={"atom_index": "atom_index_0", "atom": "atom_0"}),
    on=["molecule_name", "atom_index_0"],
    how="left",
)
test_df = test_df.merge(
    structures_df.rename(columns={"atom_index": "atom_index_1", "atom": "atom_1"}),
    on=["molecule_name", "atom_index_1"],
    how="left",
)


def euclidean_distance(df):
    return np.sqrt(
        (df["x_x"] - df["x_y"]) ** 2
        + (df["y_x"] - df["y_y"]) ** 2
        + (df["z_x"] - df["z_y"]) ** 2
    )


train_df["distance"] = euclidean_distance(train_df)
test_df["distance"] = euclidean_distance(test_df)

overall_median = train_df["scalar_coupling_constant"].median()
type_medians = train_df.groupby("type")["scalar_coupling_constant"].median()
type_means = train_df.groupby("type")["scalar_coupling_constant"].mean()
type_medians_dict = type_medians.to_dict()
type_means_dict = type_means.to_dict()

type_minmax = {}
for t, grp in train_df.groupby("type"):
    type_minmax[t] = (
        grp["scalar_coupling_constant"].min(),
        grp["scalar_coupling_constant"].max(),
    )

type_params = {}
for t, grp in train_df.groupby("type"):
    if grp["distance"].notna().sum() >= 2:
        slope, intercept = np.polyfit(
            grp["distance"], grp["scalar_coupling_constant"], 1
        )
        type_params[t] = (slope, intercept)
    else:
        type_params[t] = (0.0, type_means_dict.get(t, overall_median))

combo_medians = (
    train_df.groupby(["type", "atom_0", "atom_1"])["scalar_coupling_constant"]
    .median()
    .to_dict()
)


def predict_row(row):
    """
    Predict a single row using:
    1) exact combo median (or its reversed‑atom version) if available,
    2) otherwise blend the per‑type median with the per‑type distance regression
       (70 % median, 30 % regression),
    3) finally clipping to the observed range for that type.
    """
    t = row["type"]
    atom0 = row["atom_0"]
    atom1 = row["atom_1"]
    combo_key = (t, atom0, atom1)
    rev_key = (t, atom1, atom0)

    if combo_key in combo_medians:
        base_pred = combo_medians[combo_key]
    elif rev_key in combo_medians:
        base_pred = combo_medians[rev_key]
    else:
        median_pred = type_medians_dict.get(t, overall_median)
        slope, intercept = type_params.get(t, (0.0, overall_median))
        reg_pred = slope * row["distance"] + intercept
        base_pred = 0.7 * median_pred + 0.3 * reg_pred

    lo, hi = type_minmax.get(t, (overall_median, overall_median))
    return max(min(base_pred, hi), lo)


test_df["scalar_coupling_constant_pred"] = test_df.apply(predict_row, axis=1)



## === cell 1
submission = pd.DataFrame(
    {
        "id": test_df["id"],
        "scalar_coupling_constant": test_df["scalar_coupling_constant_pred"],
    }
)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print("First rows of submission:")
print(submission.head())



## === cell 2
submission["scalar_coupling_constant"].plot.hist(bins=100, edgecolor="black")
plt.title("Histogram of Predicted scalar_coupling_constant")
plt.xlabel("scalar_coupling_constant")
plt.ylabel("Frequency")
plt.show()
