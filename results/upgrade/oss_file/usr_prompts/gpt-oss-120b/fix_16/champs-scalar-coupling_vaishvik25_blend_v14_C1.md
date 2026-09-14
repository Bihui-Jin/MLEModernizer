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

-1.348743156545858

# 6. Current score

1.4258

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'The fix replaces the missing‑file blending logic with a simple, reproducible baseline: it reads the training data, computes the mean scalar coupling constant for each coupling type, applies those means to the test set, and writes a correctly‑formatted `submission1236.csv`. This eliminates the FileNotFound errors, ensures a valid CSV is produced, and provides a reasonable starting score without altering any core modeling logic.'
- What this solution (achieved 1.23566) has done: 'I enhance the baseline by incorporating atom element information. By merging the atom types from structures.csv into both train and test sets and computing means for each (type, atom 0, atom 1) combination, predictions become more specific. Missing combinations fall back to the original type‑level mean and then to the global mean, keeping the original simple‑mean logic while improving accuracy toward the target score.'
- What this solution (achieved 1.23566) has done: 'Implemented a hierarchical back‑off strategy for predictions: after trying the detailed (type + atom 0 + atom 1) mean, the code now falls back to means conditioned on (type + atom 0) and (type + atom 1) before the broader type‑level and global means. This adds only lightweight aggregation steps, preserving the original simple‑mean logic while giving more tailored estimates for many unseen atom‑pair combos, which should lower the Log‑MAE toward the target score.'
- What this solution (achieved 1.23566) has done: 'I add atom‑coordinate features and a binned inter‑atomic distance, then compute means conditioned on (type, atom_0, atom_1, distance_bin). This keeps the original hierarchical mean‑based logic but makes predictions more specific, which should lower the Log‑MAE toward the target while preserving the overall pipeline.'
- What this solution (achieved 1.23566) has done: 'I add a lightweight bias‑correction step that uses the same hierarchical means on the training data to compute the average residual (actual − prediction). This average residual is then added to every test prediction, keeping the original simple‑mean pipeline unchanged while nudging the predictions closer to the true values, which should lower the Log‑MAE toward the target. No core modeling logic is altered, and the script still writes a correctly‑formatted submission CSV.'
- What this solution (achieved 1.23566) has done: 'I replace the single global bias correction with a per‑type bias adjustment: after the hierarchical mean predictions are built, I compute the average residual for each coupling type on the training data and add the corresponding type‑specific bias to the test predictions (fallback to the overall mean bias if a type is unseen). This small change keeps the original mean‑based pipeline intact while providing a finer correction that should lower the Log‑MAE toward the target score.'
- What this solution (achieved 1.23566) has done: 'I add a lightweight per‑type‑and‑distance bias correction that uses the average residual for each (type, distance_bin) pair computed on the training data. This bias is merged into the test predictions and added to the existing predictions, which should reduce the Log‑MAE and move the score closer to the negative target without altering the overall mean‑based pipeline. The change is confined to the prediction‑generation cell and keeps the original hierarchical means and type‑level bias intact.'
- What this solution (achieved 1.23566) has done: 'I add a lightweight calibration step that fits a simple linear relationship between the hierarchical‑mean predictions and the true coupling constants on the training set, then apply this correction to the test predictions. This small post‑processing tweak preserves the original mean‑based logic while reducing systematic bias, helping move the Log‑MAE score closer to the negative target. The rest of the pipeline—including feature creation, hierarchical back‑off, and bias adjustments—remains unchanged.'
- What this solution (achieved 1.23566) has done: 'I make two minimal adjustments that keep the original mean‑based pipeline intact while providing a finer distance binning (0.01 Å instead of 0.1 Å) and a per‑type residual correction after the global linear calibration. The finer bins give more specific means, and the type‑specific residual offsets correct systematic bias left after the global calibration, both of which should lower the Log‑MAE and move the score toward the negative target.'
- What this solution (achieved 1.23566) has done: 'I add the missing bias corrections to the training predictions before fitting the linear calibration, so the calibration uses the same enriched features as the test side. This small reorder keeps the overall mean‑based pipeline unchanged while improving the fitted coefficient and thus moving the Log‑MAE score closer to the negative target.'
- What this solution (achieved 1.23566) has done: 'I add a lightweight per‑type linear calibration step.  
After the existing bias corrections, I fit a separate slope / intercept for each coupling type (using the training predictions and true values).  
These type‑specific coefficients are merged back into the test predictions, and each prediction is scaled by its type’s slope and intercept (falling back to the original global calibration if a type is unseen).  
This small adjustment keeps the overall mean‑based pipeline untouched while giving a finer correction that should lower the Log‑MAE toward the negative target.'
- What this solution (achieved 1.23566) has done: 'I keep the overall hierarchical‑mean + bias + linear‑calibration pipeline unchanged but make the distance bins a bit coarser (0.1 Å instead of 0.01 Å). Fewer, better‑populated bins reduce noise in the detailed means and should lower the Log‑MAE, moving the score toward the negative target. The modification is confined to the `add_distance` helper, preserving all other logic.'
- What this solution (achieved 1.31901) has done: 'I import matplotlib to avoid a runtime error in the histogram cell, and I add a lightweight shrink‑toward‑global‑mean step for the final predictions. This small regularisation tends to reduce extreme errors, which should lower the Log‑MAE and move the score closer to the negative target while preserving all existing hierarchical‑mean, bias, and calibration logic.'
- What this solution (achieved 1.23566) has done: 'I keep the full prediction pipeline but remove the shrinkage toward the global mean, setting shrinkage to 1.0 so the calibrated predictions are used unchanged. This small adjustment preserves the hierarchical‑mean, bias, and calibration steps while allowing the model to express its full learned signal, which should lower the Log‑MAE toward the negative target. The script is otherwise unchanged and still writes a valid CSV submission.'
- What this solution (achieved 1.4258) has done: 'I lower the shrinkage factor from 1.0 to 0.9 so the final predictions are slightly pulled toward the global mean, which usually reduces extreme errors and improves the Log‑MAE (moving the score lower toward the negative target). This change is minimal, keeps the original hierarchical‑mean‑plus‑bias‑plus‑calibration pipeline intact, and simply adjusts the final blending step.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

print("Root input contents:", os.listdir("../input"))

train_path = "../input/champs-scalar-coupling/train.csv"
test_path = "../input/champs-scalar-coupling/test.csv"
structures_path = "../input/champs-scalar-coupling/structures.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
structures = pd.read_csv(
    structures_path, usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"]
)

train_tmp = train_df.merge(
    structures.rename(columns={"atom": "atom_0", "x": "x_0", "y": "y_0", "z": "z_0"}),
    left_on=["molecule_name", "atom_index_0"],
    right_on=["molecule_name", "atom_index"],
    how="left",
).drop(columns="atom_index")

train_tmp = train_tmp.merge(
    structures.rename(columns={"atom": "atom_1", "x": "x_1", "y": "y_1", "z": "z_1"}),
    left_on=["molecule_name", "atom_index_1"],
    right_on=["molecule_name", "atom_index"],
    how="left",
).drop(columns="atom_index")

test_tmp = test_df.merge(
    structures.rename(columns={"atom": "atom_0", "x": "x_0", "y": "y_0", "z": "z_0"}),
    left_on=["molecule_name", "atom_index_0"],
    right_on=["molecule_name", "atom_index"],
    how="left",
).drop(columns="atom_index")

test_tmp = test_tmp.merge(
    structures.rename(columns={"atom": "atom_1", "x": "x_1", "y": "y_1", "z": "z_1"}),
    left_on=["molecule_name", "atom_index_1"],
    right_on=["molecule_name", "atom_index"],
    how="left",
).drop(columns="atom_index")


def add_distance(df):
    """
    Compute Euclidean distance between the two atoms and bin it.
    The bin width is set to 0.1 Å to obtain robust statistics for the detailed means.
    """
    df["distance"] = np.sqrt(
        (df["x_0"] - df["x_1"]) ** 2
        + (df["y_0"] - df["y_1"]) ** 2
        + (df["z_0"] - df["z_1"]) ** 2
    )
    df["distance_bin"] = (df["distance"] * 10).round() / 10.0
    return df


train_tmp = add_distance(train_tmp)
test_tmp = add_distance(test_tmp)

global_mean = train_df["scalar_coupling_constant"].mean()

type_means = train_df.groupby("type")["scalar_coupling_constant"].mean().reset_index()
type_means.rename(columns={"scalar_coupling_constant": "type_mean"}, inplace=True)

detailed_means = (
    train_tmp.groupby(["type", "atom_0", "atom_1", "distance_bin"])[
        "scalar_coupling_constant"
    ]
    .mean()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "detailed_mean"})
)

type_atom0_means = (
    train_tmp.groupby(["type", "atom_0"])["scalar_coupling_constant"]
    .mean()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "type_atom0_mean"})
)

type_atom1_means = (
    train_tmp.groupby(["type", "atom_1"])["scalar_coupling_constant"]
    .mean()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "type_atom1_mean"})
)

test_pred = test_tmp.merge(
    detailed_means, on=["type", "atom_0", "atom_1", "distance_bin"], how="left"
)
test_pred = test_pred.merge(type_atom0_means, on=["type", "atom_0"], how="left")
test_pred = test_pred.merge(type_atom1_means, on=["type", "atom_1"], how="left")
test_pred = test_pred.merge(type_means, on="type", how="left")

test_pred["pred"] = test_pred["detailed_mean"]
test_pred["pred"].fillna(test_pred["type_atom0_mean"], inplace=True)
test_pred["pred"].fillna(test_pred["type_atom1_mean"], inplace=True)
test_pred["pred"].fillna(test_pred["type_mean"], inplace=True)
test_pred["pred"].fillna(global_mean, inplace=True)

train_pred = train_tmp.merge(
    detailed_means, on=["type", "atom_0", "atom_1", "distance_bin"], how="left"
)
train_pred = train_pred.merge(type_atom0_means, on=["type", "atom_0"], how="left")
train_pred = train_pred.merge(type_atom1_means, on=["type", "atom_1"], how="left")
train_pred = train_pred.merge(type_means, on="type", how="left")

train_pred["pred"] = train_pred["detailed_mean"]
train_pred["pred"].fillna(train_pred["type_atom0_mean"], inplace=True)
train_pred["pred"].fillna(train_pred["type_atom1_mean"], inplace=True)
train_pred["pred"].fillna(train_pred["type_mean"], inplace=True)
train_pred["pred"].fillna(global_mean, inplace=True)

train_pred["resid"] = train_pred["scalar_coupling_constant"] - train_pred["pred"]

type_bias = (
    train_pred.groupby("type")["resid"]
    .mean()
    .reset_index()
    .rename(columns={"resid": "type_bias"})
)
global_bias = type_bias["type_bias"].mean()  # fallback bias

type_dist_bias = (
    train_pred.groupby(["type", "distance_bin"])["resid"]
    .mean()
    .reset_index()
    .rename(columns={"resid": "type_distance_bias"})
)

train_pred = train_pred.merge(type_bias, on="type", how="left")
train_pred["pred"] += train_pred["type_bias"].fillna(global_bias)

train_pred = train_pred.merge(type_dist_bias, on=["type", "distance_bin"], how="left")
train_pred["pred"] += train_pred["type_distance_bias"].fillna(0.0)

test_pred = test_pred.merge(type_bias, on="type", how="left")
test_pred["pred"] += test_pred["type_bias"].fillna(global_bias)

test_pred = test_pred.merge(type_dist_bias, on=["type", "distance_bin"], how="left")
test_pred["pred"] += test_pred["type_distance_bias"].fillna(0.0)  # no bias if unseen

coeff = np.polyfit(train_pred["pred"], train_pred["scalar_coupling_constant"], 1)

type_coeff = (
    train_pred.groupby("type")
    .apply(
        lambda df: pd.Series(
            np.polyfit(df["pred"], df["scalar_coupling_constant"], 1),
            index=["slope", "intercept"],
        )
    )
    .reset_index()
)

test_pred = test_pred.merge(type_coeff, on="type", how="left")
test_pred["pred"] = test_pred["pred"] * test_pred["slope"].fillna(coeff[0]) + test_pred[
    "intercept"
].fillna(coeff[1])

train_pred["pred_calib"] = coeff[0] * train_pred["pred"] + coeff[1]

type_resid_corr = (
    train_pred.groupby("type")
    .apply(lambda df: (df["scalar_coupling_constant"] - df["pred_calib"]).mean())
    .reset_index(name="type_resid_corr")
)

test_pred = test_pred.merge(type_resid_corr, on="type", how="left")
test_pred["pred"] += test_pred["type_resid_corr"].fillna(0.0)  # fallback 0

shrinkage = 0.9
test_pred["pred"] = shrinkage * test_pred["pred"] + (1 - shrinkage) * global_mean

submission = test_pred[["id"]].copy()
submission["scalar_coupling_constant"] = test_pred["pred"]
submission_path = "submission1236.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")



## === cell 1
submission["scalar_coupling_constant"].plot.hist(bins=100, title="Prediction Histogram")
plt.xlabel("scalar_coupling_constant")
plt.show()
