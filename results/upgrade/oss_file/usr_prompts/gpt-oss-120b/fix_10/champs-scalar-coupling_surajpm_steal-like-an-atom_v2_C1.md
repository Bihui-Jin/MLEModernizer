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

-1.31695555864586

# 6. Current score

1.23566

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'I replace the failing loading of external submissions with a simple baseline that reads the competition train and test files, computes the mean scalar coupling constant for each coupling type, and uses these means to generate predictions. The script now creates a valid `submission.csv` containing the required `id` and `scalar_coupling_constant` columns, ensuring the notebook runs end‑to‑end without missing‑file errors.'
- What this solution (achieved 1.23566) has done: 'I remove the erroneous markdown cell, fix the notebook indexing, and replace the simple type‑mean baseline with a deterministic predictor that sums the four coupling contributions (fc, sd, pso, dso) provided in `scalar_coupling_contributions.csv`. This uses exact physical information, so predictions are much closer to the true values, moving the log‑MAE toward the target (lower is better). Any rows missing contributions fall back to the type‑mean baseline to keep predictions valid.'
- What this solution (achieved 1.23566) has done: 'I keep the original workflow but add a simple residual correction: after merging the physical contributions, I compute the average difference between the true coupling constant and the summed contributions for each coupling type in the training set. This per‑type offset is then added to the test predictions (or to the type‑mean fallback). The change is tiny, preserves the core logic, and should lower the log‑MAE, moving the score toward the target.'
- What this solution (achieved 1.23566) has done: 'I keep the overall workflow but add a small per‑type scaling factor for the summed physical contributions. This factor (the mean ratio of true value to summed contribution in the training set) corrects systematic under‑/over‑estimation of the contributions. I also keep the existing per‑type residual correction and the original fallback, ensuring a valid CSV is still written.'
- What this solution (achieved 1.23566) has done: 'I replace the mean‑based per‑type scaling and residual corrections with their more robust median counterparts. Using the median reduces the influence of outliers in the training data, which should lower the log‑MAE and move the score closer to the target while keeping the overall workflow unchanged.'
- What this solution (achieved 1.23566) has done: 'I replace the median‑based scaling + offset with a tiny per‑type linear regression (slope α and intercept β) fitted on the training rows that have a summed contribution. This keeps the original “use physical contributions, fall back to type means” workflow while giving a better calibrated prediction, which should lower the log‑MAE and move the score closer to the target. The rest of the script (loading, merging, fallback handling, CSV output) stays unchanged.'
- What this solution (achieved 1.23566) has done: 'I add a small post‑processing step that clips each predicted value to the range (min, max) observed for its coupling type in the training set. This keeps the core “linear‑regression‑per‑type with fallback to type means” logic unchanged while preventing extreme outliers that hurt the log‑MAE, moving the score closer to the target.'
- What this solution (achieved 1.23566) has done: 'I replace the per‑type linear‑regression calibration with a more robust median‑based scaling + offset, which better handles outliers while preserving the overall workflow (use summed physical contributions, fall back to type means, and clip to observed ranges). This small change is expected to lower the log‑MAE, moving the score closer to the target.'
- What this solution (achieved 1.23566) has done: 'I replace the per‑type median‑based scaling + offset with a simple ordinary‑least‑squares linear regression (slope α and intercept β) fitted on the rows that have summed physical contributions.  This keeps the overall workflow (use contributions, fallback to type means, clipping) unchanged while giving a more accurate calibration, which should lower the log‑MAE toward the target.  The rest of the code and file paths stay the same.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

print("Input folders:", os.listdir("../input"))




## === cell 1
train_path = "../input/champs-scalar-coupling/train.csv"
test_path = "../input/champs-scalar-coupling/test.csv"
contrib_path = "../input/champs-scalar-coupling/scalar_coupling_contributions.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
contrib_df = pd.read_csv(contrib_path)

type_means = train_df.groupby("type")["scalar_coupling_constant"].mean()
global_mean = train_df["scalar_coupling_constant"].mean()

contrib_df["total_contrib"] = contrib_df[["fc", "sd", "pso", "dso"]].sum(axis=1)

merge_cols = ["molecule_name", "atom_index_0", "atom_index_1", "type"]
train_merged = train_df.merge(
    contrib_df[merge_cols + ["total_contrib"]], on=merge_cols, how="left"
)

valid_train = train_merged[
    train_merged["total_contrib"].notna() & (train_merged["total_contrib"] != 0)
]

type_factor = {}  # slope (α)
type_offset = {}  # intercept (β)

for t, grp in valid_train.groupby("type"):
    x = grp["total_contrib"].values.reshape(-1, 1)
    y = grp["scalar_coupling_constant"].values
    if len(x) >= 2:
        A = np.hstack([x, np.ones_like(x)])
        coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
        factor, offset = coeffs[0], coeffs[1]
    else:
        factor, offset = np.nan, np.nan
    type_factor[t] = factor
    type_offset[t] = offset

test_merged = test_df.merge(
    contrib_df[merge_cols + ["total_contrib"]], on=merge_cols, how="left"
)


def predict_row(row):
    factor = type_factor.get(row["type"], np.nan)
    offset = type_offset.get(row["type"], np.nan)
    if pd.isna(factor) or pd.isna(offset) or pd.isna(row["total_contrib"]):
        return np.nan
    return factor * row["total_contrib"] + offset


test_pred = test_merged.apply(predict_row, axis=1)

fallback_pred = test_df["type"].map(type_means)
test_pred = test_pred.fillna(fallback_pred).fillna(global_mean)

type_min = train_df.groupby("type")["scalar_coupling_constant"].min()
type_max = train_df.groupby("type")["scalar_coupling_constant"].max()

clip_df = pd.DataFrame({"pred": test_pred, "type": test_df["type"]})
clip_df["pred"] = clip_df.apply(
    lambda r: np.clip(
        r["pred"],
        type_min.get(r["type"], -np.inf),
        type_max.get(r["type"], np.inf),
    ),
    axis=1,
)
test_pred = clip_df["pred"]

submission = pd.DataFrame({"id": test_df["id"], "scalar_coupling_constant": test_pred})




## === cell 2
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Saved submission to {submission_path}")
print(submission.head())
