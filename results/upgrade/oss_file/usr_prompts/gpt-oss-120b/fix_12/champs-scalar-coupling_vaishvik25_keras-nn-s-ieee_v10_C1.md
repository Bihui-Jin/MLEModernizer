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

-1.572352650827347

# 6. Current score

3.00563

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'I replace the failing exploratory code with a short, robust pipeline that loads the official data, computes the mean scalar coupling constant for each coupling type from the training set, and applies these means to the test set to create a valid `submission.csv`. The script also includes a quick group‑aware validation split to report the log‑MAE score, ensuring the process runs end‑to‑end without path errors or undefined variables. All unnecessary imports and broken references are removed, and the submission file is written with the required columns and a `.csv` extension.'
- What this solution (achieved 1.23566) has done: 'I load the scalar coupling contributions and use the mean total contribution (fc + sd + pso + dso) per coupling type instead of the raw target mean. This simple adjustment keeps the overall “per‑type mean” logic while providing a tighter estimate of each type’s value, which should lower the log‑MAE toward the negative target. I also apply the same computation in the validation split so the reported score reflects the new prediction method.'
- What this solution (achieved 1.23626) has done: 'I smooth the per‑type contribution means with a global‑mean prior (to avoid noisy estimates for rare types) and then apply a simple per‑type scaling factor that aligns the summed contributions to the actual scalar coupling constants observed in the training data. These lightweight adjustments keep the original “type‑mean” logic but should lower the MAE enough to move the log‑MAE toward the negative target while still producing a correct submission CSV.'
- What this solution (achieved 1.23626) has done: 'I replace the contribution‑based per‑type prediction with a simpler per‑type target mean, applying the same smoothing logic that was already used. This keeps the overall “per‑type smoothed mean” approach while removing the noisy ratio scaling, which should lower the log‑MAE and move the score closer to the negative target. The same change is applied to the validation split so the reported metric reflects the new prediction method.'
- What this solution (achieved 1.23626) has done: 'I incorporate the per‑type mean of the summed coupling contributions (the “total_contrib” column) and blend it with the original per‑type target mean, using a small weight for the contributions. This adds useful physics‑based information while keeping the original smoothed‑mean logic, and should lower the log‑MAE toward the negative target. The changes are limited to data merging, computing the new blended prediction, and using it for both validation and the final submission.'
- What this solution (achieved 1.18492) has done: 'Implemented robust path handling to locate the dataset, fixed variable scope issues, and ensured the pipeline runs end‑to‑end producing a valid `submission.csv`. Added safe fallbacks for missing per‑type predictions and retained the original smoothing‑based per‑type prediction logic, which modestly improves the log‑MAE while keeping core methodology unchanged.'
- What this solution (achieved 1.18496) has done: 'I slightly increase the blending weight so the per‑type target median contributes to the prediction, and lower the smoothing factor `m` to reduce the shrinkage toward the global mean. These modest adjustments keep the original “type‑median” logic but should give per‑type predictions that are closer to the true values, decreasing the log‑MAE and moving the score toward the negative target.'
- What this solution (achieved 1.18494) has done: 'I increase the smoothing factor and give more weight to the actual target median, which should pull the per‑type predictions closer to the true values and thus lower the log‑MAE toward the negative target. The changes are limited to the parameter definitions in cell 2 (and therefore affect the validation split as well).'
- What this solution (achieved 3.00563) has done: 'I replace the per‑type median prediction with the more detailed per‑row total contribution (which is available from `scalar_coupling_contributions.csv`). Using the actual summed physics‑based contribution for each atom pair should lower the MAE, moving the log‑MAE toward the negative target while keeping the overall pipeline unchanged. The validation split now predicts with the row‑wise contribution (falling back to the global mean when missing), and the final submission merges the same contribution column for the test set.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

possible_dirs = [
    "./data/champs-scalar-coupling",
    "./input/champs-scalar-coupling",
    "/kaggle/input/champs-scalar-coupling",
    "./working/champs-scalar-coupling",
]
DATA_DIR = next((d for d in possible_dirs if os.path.isdir(d)), None)
if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not locate the 'champs-scalar-coupling' data directory."
    )

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
contrib = pd.read_csv(os.path.join(DATA_DIR, "scalar_coupling_contributions.csv"))

contrib["total_contrib"] = contrib[["fc", "sd", "pso", "dso"]].sum(axis=1)

train_merged = train.merge(
    contrib[["molecule_name", "atom_index_0", "atom_index_1", "type", "total_contrib"]],
    on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
    how="left",
)



## === cell 1
m = 50
blend_weight = 0.6

global_target_mean = train["scalar_coupling_constant"].mean()

type_stats = (
    train_merged.groupby("type")
    .agg(
        type_target_median=("scalar_coupling_constant", "median"),
        type_contrib_median=("total_contrib", "median"),
        count=("scalar_coupling_constant", "size"),
    )
    .reset_index()
)

type_stats["blended_median"] = (
    blend_weight * type_stats["type_target_median"]
    + (1 - blend_weight) * type_stats["type_contrib_median"]
)

type_stats["smoothed_median"] = (
    type_stats["blended_median"] * type_stats["count"] + global_target_mean * m
) / (type_stats["count"] + m)

type_stats["final_pred"] = type_stats["smoothed_median"]

print("Per‑type fallback predictions (preview):")
print(type_stats[["type", "final_pred"]].head())



## === cell 2
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
groups = train["molecule_name"]
train_idx, val_idx = next(gss.split(train, groups=groups))

train_split = train.iloc[train_idx].reset_index(drop=True)
val_split = train.iloc[val_idx].reset_index(drop=True)

train_split_merged = train_split.merge(
    contrib[["molecule_name", "atom_index_0", "atom_index_1", "type", "total_contrib"]],
    on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
    how="left",
)

global_target_mean_split = train_split["scalar_coupling_constant"].mean()

val_pred = val_split.merge(
    contrib[["molecule_name", "atom_index_0", "atom_index_1", "type", "total_contrib"]],
    on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
    how="left",
)["total_contrib"]
val_pred.fillna(global_target_mean_split, inplace=True)


def log_mae_per_type(y_true, y_pred, types):
    mae = np.abs(y_true - y_pred)
    df = pd.DataFrame({"type": types, "mae": mae})
    df["mae"] = df["mae"].clip(lower=1e-9)  # avoid log(0)
    log_mae = np.log(df.groupby("type")["mae"].mean())
    return log_mae.mean()


score = log_mae_per_type(
    val_split["scalar_coupling_constant"], val_pred, val_split["type"]
)
print(f"Estimated log‑MAE (lower is better): {score:.6f}")



## === cell 3
submission = test.merge(
    contrib[["molecule_name", "atom_index_0", "atom_index_1", "type", "total_contrib"]],
    on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
    how="left",
)
submission["total_contrib"].fillna(global_target_mean, inplace=True)

submission_file = "submission.csv"
submission[["id", "total_contrib"]].rename(
    columns={"total_contrib": "scalar_coupling_constant"}
).to_csv(submission_file, index=False, float_format="%.6f")
print(f"Submission written to {submission_file}")
