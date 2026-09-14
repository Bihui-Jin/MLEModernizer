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

-1.6630743227479918

# 6. Current score

1.23566

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'The script failed because it tried to load non‑existent blending CSV files, causing a `FileNotFoundError`. I replaced the blending step with a simple, reliable baseline: compute the mean scalar coupling constant for each coupling type from the provided training data and use those means to predict the test set. Missing types fall back to the overall training mean. The resulting predictions are saved as a correctly formatted CSV submission.'
- What this solution (achieved 1.23566) has done: 'I enrich the baseline by merging the scalar coupling contributions (fc, sd, pso, dso) into the test set and use their sum as a more informed prediction; when the contributions are missing I fall back to the original type‑wise mean (and finally the global mean). This small addition keeps the core logic intact while providing features that are highly correlated with the target, moving the score toward the lower‑than‑target range.'
- What this solution (achieved 1.23566) has done: 'I add a small fallback that uses the average of the summed contribution terms (`fc+sd+pso+dso`) for each coupling type when the contribution data is missing. This keeps the original logic (use summed contributions first) but provides a more directly related estimate than the overall scalar‑coupling mean, so the predictions should become a bit closer to the target (lower) score. The rest of the pipeline and the submission format stay unchanged.'
- What this solution (achieved 1.23566) has done: 'I add a lightweight per‑type linear adjustment that maps the summed contributions (`fc+sd+pso+dso`) to the true coupling constant using a simple regression on the training data. This keeps the original fallback hierarchy (type‑wise contribution mean → type mean → global mean) but replaces the raw contribution sum with a calibrated estimate, which should reduce the MAE and move the score closer to the lower target. Only minimal imports and a few lines are added, preserving the core logic and output format.'
- What this solution (achieved 1.23566) has done: 'I add a lightweight residual‑correction step that learns the average error of the current calibrated predictions on the training set (grouped by coupling type) and applies this bias to the test predictions. This keeps the original calibration logic intact, only adds a small function for reuse, and shifts the predictions toward the true values, moving the log‑MAE lower and closer to the target score. The script still writes a correctly formatted CSV submission.'
- What this solution (achieved 1.23566) has done: 'I keep the existing pipeline but add a lightweight global linear calibration after the per‑type residual correction. By fitting a simple slope‑intercept on the corrected training predictions and applying the same transformation to the test predictions, we modestly shift the outputs toward the true values, which should lower the log‑MAE and move the score closer to the target without altering the core logic.'
- What this solution (achieved 1.23566) has done: 'I add two lightweight cells that bring in molecule‑level dipole‑moment magnitude and potential‑energy features, fit a simple multivariate linear correction (using the already‑calibrated predictions plus these new features) on the training set, and apply it to the test predictions. This keeps the original calibration pipeline intact while providing extra signal that should reduce the log‑MAE and move the score closer to the target. Finally, the submission file is written as before.'
- What this solution (achieved 1.23566) has done: 'I fixed the shape mismatch in the linear‑correction step by correctly extracting the coefficient vector and intercept from the least‑squares solution, ensuring the matrix dimensions align for the dot product. The rest of the pipeline is unchanged, preserving the original modeling logic while allowing the corrected calibration to improve the score.'
- What this solution (achieved 1.23566) has done: 'I simplify the post‑processing step that adds dipole‑moment and potential‑energy corrections. These extra linear adjustments can introduce noise and raise the log‑MAE, so I keep the earlier calibrated predictions and skip the least‑squares fit that modifies them. The rest of the pipeline, including the type‑wise calibrations and residual corrections, remains unchanged, ensuring the script still writes a valid CSV while moving the score closer to the lower target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

train_path = "../input/champs-scalar-coupling/train.csv"
test_path = "../input/champs-scalar-coupling/test.csv"
contrib_path = "../input/champs-scalar-coupling/scalar_coupling_contributions.csv"




## === cell 1
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
contrib = pd.read_csv(contrib_path)




## === cell 2
type_means = train.groupby("type")["scalar_coupling_constant"].mean()
global_mean = train["scalar_coupling_constant"].mean()

contrib["sum_contrib"] = contrib[["fc", "sd", "pso", "dso"]].sum(axis=1)
type_contrib_means = contrib.groupby("type")["sum_contrib"].mean()

train_with_contrib = train.merge(
    contrib[["molecule_name", "atom_index_0", "atom_index_1", "type", "sum_contrib"]],
    on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
    how="left",
)

type_slope = {}
type_intercept = {}

for tp, grp in train_with_contrib.groupby("type"):
    mask = grp["sum_contrib"].notna()
    if mask.sum() < 2:
        type_slope[tp] = 1.0
        type_intercept[tp] = 0.0
        continue
    x = grp.loc[mask, "sum_contrib"].values
    y = grp.loc[mask, "scalar_coupling_constant"].values
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    type_slope[tp] = a
    type_intercept[tp] = b




## === cell 3
contrib_keep = contrib[
    ["molecule_name", "atom_index_0", "atom_index_1", "type", "sum_contrib"]
]




## === cell 4
test = test.merge(
    contrib_keep,
    on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
    how="left",
)




## === cell 5
def calibrated_pred(row):
    if pd.isna(row["sum_contrib"]):
        return np.nan
    tp = row["type"]
    a = type_slope.get(tp, 1.0)
    b = type_intercept.get(tp, 0.0)
    return a * row["sum_contrib"] + b


test["scalar_coupling_constant"] = test.apply(calibrated_pred, axis=1)

missing_mask = test["scalar_coupling_constant"].isna()
test.loc[missing_mask, "scalar_coupling_constant"] = test.loc[missing_mask, "type"].map(
    type_contrib_means
)

missing_mask = test["scalar_coupling_constant"].isna()
test.loc[missing_mask, "scalar_coupling_constant"] = test.loc[missing_mask, "type"].map(
    type_means
)

missing_mask = test["scalar_coupling_constant"].isna()
test.loc[missing_mask, "scalar_coupling_constant"] = global_mean


def get_predictions(df):
    df = df.merge(
        contrib_keep,
        on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
        how="left",
    )
    df["pred"] = df.apply(calibrated_pred, axis=1)
    miss = df["pred"].isna()
    df.loc[miss, "pred"] = df.loc[miss, "type"].map(type_contrib_means)
    miss = df["pred"].isna()
    df.loc[miss, "pred"] = df.loc[miss, "type"].map(type_means)
    miss = df["pred"].isna()
    df.loc[miss, "pred"] = global_mean
    return df["pred"]


train_pred = get_predictions(train)
train_residual = train["scalar_coupling_constant"] - train_pred
type_residual_mean = train_residual.groupby(train["type"]).mean()

test["scalar_coupling_constant"] += test["type"].map(type_residual_mean).fillna(0)

train_pred_corrected = train_pred + train["type"].map(type_residual_mean).fillna(0)

A = np.vstack([train_pred_corrected, np.ones_like(train_pred_corrected)]).T
global_a, global_b = np.linalg.lstsq(A, train["scalar_coupling_constant"], rcond=None)[
    0
]

test["scalar_coupling_constant"] = (
    test["scalar_coupling_constant"] * global_a + global_b
)




## === cell 6
dipole_path = "../input/champs-scalar-coupling/dipole_moments.csv"
potential_path = "../input/champs-scalar-coupling/potential_energy.csv"

dipole = pd.read_csv(dipole_path)
dipole["dipole_mag"] = np.sqrt(dipole["X"] ** 2 + dipole["Y"] ** 2 + dipole["Z"] ** 2)
dipole = dipole[["molecule_name", "dipole_mag"]]

potential = pd.read_csv(potential_path)  # columns: molecule_name, potential_energy

train_feat = train.merge(dipole, on="molecule_name", how="left")
train_feat = train_feat.merge(potential, on="molecule_name", how="left")

test_feat = test.merge(dipole, on="molecule_name", how="left")
test_feat = test_feat.merge(potential, on="molecule_name", how="left")

dipole_mean = dipole["dipole_mag"].mean()
pot_mean = potential["potential_energy"].mean()
train_feat["dipole_mag"].fillna(dipole_mean, inplace=True)
train_feat["potential_energy"].fillna(pot_mean, inplace=True)
test_feat["dipole_mag"].fillna(dipole_mean, inplace=True)
test_feat["potential_energy"].fillna(pot_mean, inplace=True)





## === cell 7
submission = test[["id", "scalar_coupling_constant"]]
submission.to_csv("my_blend_1.csv", index=False)
