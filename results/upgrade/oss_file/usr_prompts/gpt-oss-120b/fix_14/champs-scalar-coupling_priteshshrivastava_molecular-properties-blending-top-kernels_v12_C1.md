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

-1.6821991287997458

# 6. Current score

1.18497

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'The original script crashes because it tries to read non‑existent blending files. I replaced those reads with a simple, self‑contained baseline: compute the mean `scalar_coupling_constant` for each coupling `type` from the training set and use those means as predictions for the test set (fallback to the overall mean). This guarantees a valid `submission.csv` with the required columns and provides a reasonable score without altering any core modeling logic.'
- What this solution (achieved 1.23566) has done: 'I add a lightweight feature: the element types of the two atoms in each pair. By merging the structures file I can obtain the atom symbols, then compute a mean target for each (type, atom_0, atom_1) combination. Predictions use these specific means when available, falling back to the per‑type mean and finally the global mean. This small enrichment should lower the log‑MAE toward the target without altering the overall simple baseline logic.'
- What this solution (achieved 1.23566) has done: 'I keep the overall simple mean‑based baseline but add a few lightweight hierarchical fallback averages (type‑+‑atom0 and type‑+‑atom1) so that more predictions use relevant statistics, which should modestly reduce the log‑MAE and move the score nearer the target. The core logic and file handling stay unchanged, only the prediction filling order is extended.'
- What this solution (achieved 1.23566) has done: 'I added robust handling for missing distance values so the binning step no longer throws an integer‑casting error, and ensured the fallback predictions always fill any remaining NaNs. This lets the script run through to the end and reliably write a valid `submission.csv` file.'
- What this solution (achieved 1.23566) has done: 'I add a lightweight fallback that uses the average `scalar_coupling_constant` for each coupling **type + distance‑bin** combination.  
If a prediction is still missing after the existing atom‑based fallbacks, this step supplies a more distance‑aware estimate, which should reduce the log‑MAE and move the score closer to the target without altering the overall baseline logic.'
- What this solution (achieved 1.23566) has done: 'I add a lightweight linear‑regression fallback that predicts the coupling constant from the inter‑atomic distance for each (type, atom_0, atom_1) group that has enough samples. This regression is merged into the test set and used as the first prediction; remaining NaNs are then filled by the existing hierarchical means (distance‑bin, atom‑specific, type‑specific, global). This small model‑based step should lower the log‑MAE toward the target without changing the overall baseline logic.'
- What this solution (achieved 1.23566) has done: 'The changes lower the sample threshold for the per‑group distance‑based linear regression (so more groups get a fitted line) and add a lightweight fallback regression at the **type** level. After the existing combo‑distance average fallback, any still‑missing predictions are now filled using the type‑specific regression before finally falling back to the simpler means. These adjustments keep the overall baseline logic intact while providing more calibrated estimates, which should reduce the log‑MAE and move the score closer to the target.'
- What this solution (achieved 1.18497) has done: 'I replace the group‑by averages with medians (which better match the MAE metric) and lower the minimum sample count for the per‑group linear regression from 3 to 2 so more groups get a fitted line. These small adjustments keep the overall baseline logic intact while expectedly reducing the log‑MAE toward the target score.'
- What this solution (achieved 1.18497) has done: 'I lower the evaluation score by making the type‑level linear‑regression fallback apply to any group with at least the same `min_samples` (2) used elsewhere, and I rewrite that regression so it always returns explicit “slope” and “intercept” fields. This expands coverage of the regression fallback, reducing reliance on coarse median values and moving the log‑MAE closer to the target value.'
- What this solution (achieved 1.23566) has done: 'I replace all median‑based fallback statistics with mean‑based ones (global, per‑type, atom‑specific, distance‑specific, etc.) because the evaluation uses MAE, and means are optimal for that metric. This small statistical change keeps the overall pipeline and model unchanged while expected to lower the log‑MAE toward the negative target score.'
- What this solution (achieved 1.18497) has done: 'I replace the mean‑based aggregations with median‑based ones (global, per‑type, per‑atom, per‑combo and distance‑bin statistics). Using medians aligns better with the MAE metric, so the predictions should become slightly more accurate and the log‑MAE move lower toward the target without altering the overall pipeline.'
- What this solution (achieved 1.18497) has done: 'The changes lower the minimum sample requirement to 1 so that a linear‑regression fallback is available for every (type, atom0, atom1) group, and we apply the type‑level regression earlier in the fallback chain (before distance‑bin and combo medians). This adds more calibrated predictions while keeping the original simple‑baseline structure, which should reduce the log‑MAE and move the score closer to the negative target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

base_path = "../input/champs-scalar-coupling/"

train_path = os.path.join(base_path, "train.csv")
test_path = os.path.join(base_path, "test.csv")
structures_path = os.path.join(base_path, "structures.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

structures = pd.read_csv(
    structures_path, usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"]
)

train = train.merge(
    structures.rename(
        columns={
            "atom_index": "atom_index_0",
            "atom": "atom_0",
            "x": "x_0",
            "y": "y_0",
            "z": "z_0",
        }
    ),
    on=["molecule_name", "atom_index_0"],
    how="left",
).merge(
    structures.rename(
        columns={
            "atom_index": "atom_index_1",
            "atom": "atom_1",
            "x": "x_1",
            "y": "y_1",
            "z": "z_1",
        }
    ),
    on=["molecule_name", "atom_index_1"],
    how="left",
)

test = test.merge(
    structures.rename(
        columns={
            "atom_index": "atom_index_0",
            "atom": "atom_0",
            "x": "x_0",
            "y": "y_0",
            "z": "z_0",
        }
    ),
    on=["molecule_name", "atom_index_0"],
    how="left",
).merge(
    structures.rename(
        columns={
            "atom_index": "atom_index_1",
            "atom": "atom_1",
            "x": "x_1",
            "y": "y_1",
            "z": "z_1",
        }
    ),
    on=["molecule_name", "atom_index_1"],
    how="left",
)


def compute_dist(df):
    return np.sqrt(
        (df["x_0"] - df["x_1"]) ** 2
        + (df["y_0"] - df["y_1"]) ** 2
        + (df["z_0"] - df["z_1"]) ** 2
    )


train["distance"] = compute_dist(train)
test["distance"] = compute_dist(test)

train["dist_bin"] = (train["distance"] * 10).round().fillna(-1).astype(int)
test["dist_bin"] = (test["distance"] * 10).round().fillna(-1).astype(int)

min_samples = 1

grouped = train.groupby(["type", "atom_0", "atom_1"])


def fit_group(g):
    if len(g) >= min_samples:
        coeffs = np.polyfit(g["distance"], g["scalar_coupling_constant"], 1)
        return pd.Series({"slope": coeffs[0], "intercept": coeffs[1]})
    else:
        return pd.Series({"slope": np.nan, "intercept": np.nan})


reg_params = grouped.apply(fit_group).reset_index()
test = test.merge(reg_params, on=["type", "atom_0", "atom_1"], how="left")

type_reg = (
    train.groupby("type")
    .apply(
        lambda g: pd.Series(
            {
                "slope_type": (
                    np.polyfit(g["distance"], g["scalar_coupling_constant"], 1)[0]
                    if len(g) >= min_samples
                    else np.nan
                ),
                "intercept_type": (
                    np.polyfit(g["distance"], g["scalar_coupling_constant"], 1)[1]
                    if len(g) >= min_samples
                    else np.nan
                ),
            }
        )
    )
    .reset_index()
)
test = test.merge(type_reg, on="type", how="left")

pred = (test["slope"] * test["distance"] + test["intercept"]).values

mask_nan = np.isnan(pred)
if mask_nan.any():
    pred_type = (
        test.loc[mask_nan, "slope_type"] * test.loc[mask_nan, "distance"]
        + test.loc[mask_nan, "intercept_type"]
    ).values
    pred[mask_nan] = pred_type

mask_nan = np.isnan(pred)
if mask_nan.any():
    combo_dist_medians = train.groupby(["type", "atom_0", "atom_1", "dist_bin"])[
        "scalar_coupling_constant"
    ].median()
    idx_dist = pd.MultiIndex.from_arrays(
        [
            test.loc[mask_nan, "type"],
            test.loc[mask_nan, "atom_0"],
            test.loc[mask_nan, "atom_1"],
            test.loc[mask_nan, "dist_bin"],
        ]
    )
    pred[mask_nan] = combo_dist_medians.reindex(idx_dist).values

mask_nan = np.isnan(pred)
if mask_nan.any():
    combo_medians = train.groupby(["type", "atom_0", "atom_1"])[
        "scalar_coupling_constant"
    ].median()
    idx_combo = pd.MultiIndex.from_arrays(
        [
            test.loc[mask_nan, "type"],
            test.loc[mask_nan, "atom_0"],
            test.loc[mask_nan, "atom_1"],
        ]
    )
    pred[mask_nan] = combo_medians.reindex(idx_combo).values

mask_nan = np.isnan(pred)
if mask_nan.any():
    atom0_medians = train.groupby(["type", "atom_0"])[
        "scalar_coupling_constant"
    ].median()
    idx_atom0 = pd.MultiIndex.from_arrays(
        [test.loc[mask_nan, "type"], test.loc[mask_nan, "atom_0"]]
    )
    pred[mask_nan] = atom0_medians.reindex(idx_atom0).values

mask_nan = np.isnan(pred)
if mask_nan.any():
    atom1_medians = train.groupby(["type", "atom_1"])[
        "scalar_coupling_constant"
    ].median()
    idx_atom1 = pd.MultiIndex.from_arrays(
        [test.loc[mask_nan, "type"], test.loc[mask_nan, "atom_1"]]
    )
    pred[mask_nan] = atom1_medians.reindex(idx_atom1).values

mask_nan = np.isnan(pred)
if mask_nan.any():
    type_dist_medians = train.groupby(["type", "dist_bin"])[
        "scalar_coupling_constant"
    ].median()
    idx_type_dist = pd.MultiIndex.from_arrays(
        [test.loc[mask_nan, "type"], test.loc[mask_nan, "dist_bin"]]
    )
    pred[mask_nan] = type_dist_medians.reindex(idx_type_dist).values

mask_nan = np.isnan(pred)
if mask_nan.any():
    type_medians = train.groupby("type")["scalar_coupling_constant"].median()
    pred[mask_nan] = test.loc[mask_nan, "type"].map(type_medians).values

mask_nan = np.isnan(pred)
if mask_nan.any():
    global_median = train["scalar_coupling_constant"].median()
    pred[mask_nan] = global_median

submission = pd.DataFrame({"id": test["id"], "scalar_coupling_constant": pred})



## === cell 1
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
