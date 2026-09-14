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

-1.30873

# 6. Current score

1.18497

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'I replace the failing blending code with a simple baseline that predicts the mean `scalar_coupling_constant` for each coupling type using the training data. This removes the missing‑file errors, defines all variables, and writes a properly formatted `submission.csv` so the notebook finishes without runtime errors while keeping the core logic minimal.'
- What this solution (achieved 1.18487) has done: 'I add a lightweight validation split to compute a per‑type calibration factor (median ratio of true to predicted) and apply that factor to the test predictions. This keeps the original “type‑mean” baseline while correcting systematic bias, which should lower the log‑MAE and move the score closer to the negative target.'
- What this solution (achieved 1.23554) has done: 'I replace the simple per‑type mean baseline with the physics‑based contribution sum, which is provided in `scalar_coupling_contributions.csv`. By merging these contributions into the train/validation split and the test set, predictions become the direct sum `fc+sd+pso+dso`. Missing contributions fall back to the original type‑mean baseline, keeping the script robust while dramatically lowering the log‑MAE toward the negative target.'
- What this solution (achieved 1.23554) has done: 'I keep the physics‑based contribution sum as the primary prediction but add a lightweight per‑type calibration using the validation split. By computing the median ratio of true to predicted values for each coupling type (and a global fallback), we can scale both the contribution‑based and fallback predictions, which should lower the log‑MAE and move the score closer to the negative target without altering the core model logic. The changes are limited to cell 0 where the calibration factors are computed and applied to the final test predictions.'
- What this solution (achieved 1.23566) has done: 'I replace the 20 % validation split with a calibration that uses the entire training set (where the physics‑based contribution is available).  By computing per‑type and global multiplicative factors on all usable rows, the predictions are scaled more accurately, which should lower the log‑MAE and move the score nearer the negative target while keeping the original baseline logic unchanged.'
- What this solution (achieved 1.18497) has done: 'I keep the physics‑based contribution baseline but make two modest tweaks that should lower the log‑MAE: (1) use per‑type *median* (and a global median) as the fallback when a contribution is missing, which is more robust than the mean; (2) cap the calibration multipliers to a reasonable range (0.5 – 2.0) to avoid extreme scaling from outlier ratios. These changes preserve the core logic while nudging the score toward the negative target.'
- What this solution (achieved 1.18497) has done: 'I replace the simple multiplicative calibration with a per‑type linear calibration (slope + intercept) fitted on the rows where the physics‑based contribution is available.  The slope is clipped to [0.5, 2.0] to avoid extreme scaling, and a global fallback is used when a type has too few points.  This keeps the original contribution‑based baseline while adding a modest but more expressive correction expected to lower the log‑MAE toward the negative target.'
- What this solution (achieved 1.18497) has done: 'I replace the linear‐calibration step with a simpler, more robust per‑type multiplicative calibration based on the median ratio of true to physics‑based predictions (and a global fallback). This keeps the original contribution‑based baseline while applying a modest scaling that is less prone to over‑fitting, and the scaling factors are clipped to a safe range (0.5‑2.0) to avoid extreme adjustments. The rest of the pipeline and submission code remain unchanged.'
- What this solution (achieved 1.18497) has done: 'I add a small per‑type additive correction (median residual) to the baseline predictions before applying the existing multiplicative calibration. This keeps the original physics‑based and ratio‑based logic while giving the model a modest ability to fix systematic under‑/over‑predictions, which should lower the log‑MAE and move the score closer to the negative target.'
- What this solution (achieved 1.18497) has done: 'I replace the separate additive‑offset and multiplicative‑ratio calibration with a single per‑type linear calibration (slope + intercept) fitted on rows where the physics‑based contribution is available. The slopes and intercepts are clipped to safe ranges (0.5‑2.0 for slopes, ‑5‑5 for intercepts) and fallback to global values when a type has too few points. This keeps the original contribution‑based prediction core while giving a modest, more expressive correction that should reduce the log‑MAE and move the score toward the negative target.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

train_path = "../input/champs-scalar-coupling/train.csv"
test_path = "../input/champs-scalar-coupling/test.csv"
contrib_path = "../input/champs-scalar-coupling/scalar_coupling_contributions.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
contrib_df = pd.read_csv(contrib_path)

contrib_df["pred"] = contrib_df[["fc", "sd", "pso", "dso"]].sum(axis=1)

key_cols = ["molecule_name", "atom_index_0", "atom_index_1", "type"]

train_df = train_df.merge(contrib_df[key_cols + ["pred"]], on=key_cols, how="left")
test_df = test_df.merge(contrib_df[key_cols + ["pred"]], on=key_cols, how="left")

type_median = train_df.groupby("type")["scalar_coupling_constant"].median()
global_median = train_df["scalar_coupling_constant"].median()

test_pred_raw = test_df["pred"].fillna(
    test_df["type"].map(type_median).fillna(global_median)
)

calib_df = train_df.dropna(subset=["pred"]).copy()


def fit_linear(g):
    if len(g) < 5:
        return pd.Series({"slope": np.nan, "intercept": np.nan})
    X = g["pred"].values[:, None]
    y = g["scalar_coupling_constant"].values
    A = np.hstack([X, np.ones_like(X)])
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return pd.Series({"slope": a, "intercept": b})


type_fit = calib_df.groupby("type").apply(fit_linear)

global_fit_vals = fit_linear(calib_df)
global_slope = global_fit_vals["slope"]
global_intercept = global_fit_vals["intercept"]

type_fit["slope"] = type_fit["slope"].clip(lower=0.5, upper=2.0)
type_fit["intercept"] = type_fit["intercept"].clip(lower=-5.0, upper=5.0)
global_slope = np.clip(global_slope, 0.5, 2.0)
global_intercept = np.clip(global_intercept, -5.0, 5.0)

slope_map = test_df["type"].map(type_fit["slope"]).fillna(global_slope)
intercept_map = test_df["type"].map(type_fit["intercept"]).fillna(global_intercept)

test_pred = test_pred_raw * slope_map + intercept_map



## === cell 1
submission = test_df[["id"]].copy()
submission["scalar_coupling_constant"] = test_pred

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print(f"Submission written to {submission_path}")
print(submission.head())



## === cell 2
submission["scalar_coupling_constant"].plot(
    kind="hist", bins=100, title="Prediction Histogram"
)
