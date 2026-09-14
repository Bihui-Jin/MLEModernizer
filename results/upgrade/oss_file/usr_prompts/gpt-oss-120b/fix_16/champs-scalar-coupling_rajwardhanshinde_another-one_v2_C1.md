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

-1.5209593019916507

# 6. Current score

1.13021

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'I replace the missing external submission files with a simple baseline that predicts the mean coupling constant for each coupling type derived from the training data. This removes the FileNotFoundError, ensures the variables exist, and creates a valid `stackers_blend.csv` submission with the required columns. The new cells load the competition data, compute per‑type means, merge them to the test set, fill any missing values with the overall mean, and write the output file.'
- What this solution (achieved 1.23566) has done: 'I add loading of the atom element information from **structures.csv** and compute a more specific mean prediction for each coupling type together with the two atom element types. This richer grouping usually lowers the MAE a bit compared to only using the coupling type mean, moving the score closer to the target while keeping the original baseline approach unchanged.'
- What this solution (achieved 3.00563) has done: 'I add a simple distance‑based correction to the existing per‑type/atom‑pair mean baseline. By merging the atomic coordinates from `structures.csv` we can compute the inter‑atomic Euclidean distance, fit a tiny linear regression (scalar ≈ a · distance + b) on the training data, and adjust each group‑mean prediction by the distance residual. This keeps the original grouping logic while providing an extra signal that should lower the MAE and move the log‑MAE toward the target value.'
- What this solution (achieved 3.00563) has done: 'I keep the overall baseline (type‑ and atom‑pair means) but replace the single global distance correction with per‑type distance regressions. This adds a small amount of extra fitting that better captures how distance affects each coupling type, and it is done without changing the core grouping logic. The script is renumbered to start at cell 1, and the new per‑type coefficients are merged back to the test set so that predictions are adjusted more accurately, moving the log‑MAE toward the target value.'
- What this solution (achieved 3.00563) has done: 'The update adjusts the fallback distance correction: instead of fitting a global model on the raw target, it now fits the global linear regression on the residuals (target − baseline). This aligns the correction with the intended residual‑based approach, reducing systematic bias for types lacking per‑type parameters and moving the log‑MAE closer to the target score while keeping the overall baseline logic unchanged.'
- What this solution (achieved 1.23566) has done: 'I simplify the distance correction by using the globally‑trained linear model for all coupling types instead of per‑type regressions, which were over‑fitting and worsening the log‑MAE. This keeps the baseline grouping logic intact while applying a single, more stable correction, bringing the score closer to the target.'
- What this solution (achieved 1.23566) has done: 'Implemented a robust data‑loading routine that automatically finds the correct dataset directory, fixing the FileNotFoundError and ensuring all required DataFrames (`train`, `test`, `sample_submission`, `structures`) are defined. The rest of the notebook logic (feature enrichment, baseline means, distance correction, and submission creation) is unchanged, preserving the original modeling approach while now producing a valid `stackers_blend.csv` submission file.'
- What this solution (achieved 1.23566) has done: 'I lower the minimum sample threshold so that the per‑type distance regressions are applied whenever they exist, and simplify the logic that decides whether to use a type‑specific or the global correction. This lets more tailored corrections be used, which should reduce the log‑MAE and move the score closer to the target.'
- What this solution (achieved 1.17332) has done: 'Implemented robust data‑loading that searches common Kaggle input locations, fixing the FileNotFoundError. Consolidated the workflow into sequential cells, preserving the original baseline logic (type/atom‑pair means with a global distance correction) while ensuring all variables are defined before use. The script now creates a valid `stackers_blend.csv` submission file with the required columns.'
- What this solution (achieved 1.17332) has done: 'I extend the distance‑adding routine to also bring in each atom’s element symbol, then create a new baseline that uses the mean coupling for each (type + sorted element‑pair) group. The prediction hierarchy becomes: element‑pair mean → atom‑index pair mean → type mean → global mean, after which the existing distance‑based linear correction is applied. This small enrichment keeps the original modelling steps while giving the model more specific information, which should move the log‑MAE closer to the target.'
- What this solution (achieved 1.13021) has done: 'Implemented a targeted improvement: switched baseline aggregations from **mean** to **median** for type, atom‑pair, and element‑pair groups. Median better aligns with the MAE‑based metric, so predictions become more robust and the log‑MAE should move closer to the target while preserving the original workflow and distance‑based correction.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np


def find_data_dir():
    """Return the first existing data directory among common Kaggle locations."""
    candidates = [
        "./data/champs-scalar-coupling",
        "./input/champs-scalar-coupling",
        "/kaggle/input/champs-scalar-coupling",
        "/kaggle/input/data/champs-scalar-coupling",
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(
        "Unable to locate the 'champs-scalar-coupling' data directory."
    )


DATA_DIR = find_data_dir()




## === cell 1
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_submission = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
structures = pd.read_csv(os.path.join(DATA_DIR, "structures.csv"))

train = train.rename(columns={"atom_index_0": "atom0", "atom_index_1": "atom1"})
test = test.rename(columns={"atom_index_0": "atom0", "atom_index_1": "atom1"})




## === cell 2
def add_distance_and_elements(df):
    """Merge atomic coordinates, element symbols and compute Euclidean distance."""
    df = df.merge(
        structures[["molecule_name", "atom_index", "x", "y", "z", "atom"]],
        left_on=["molecule_name", "atom0"],
        right_on=["molecule_name", "atom_index"],
        how="left",
    ).rename(columns={"x": "x0", "y": "y0", "z": "z0", "atom": "atom0_elem"})
    df = df.drop(columns=["atom_index"], errors="ignore")

    df = df.merge(
        structures[["molecule_name", "atom_index", "x", "y", "z", "atom"]],
        left_on=["molecule_name", "atom1"],
        right_on=["molecule_name", "atom_index"],
        how="left",
    ).rename(columns={"x": "x1", "y": "y1", "z": "z1", "atom": "atom1_elem"})
    df = df.drop(columns=["atom_index"], errors="ignore")

    df["distance"] = np.sqrt(
        (df["x0"] - df["x1"]) ** 2
        + (df["y0"] - df["y1"]) ** 2
        + (df["z0"] - df["z1"]) ** 2
    )

    df = df.drop(
        columns=[
            "x0",
            "y0",
            "z0",
            "x1",
            "y1",
            "z1",
        ],
        errors="ignore",
    )
    return df


train = add_distance_and_elements(train)
test = add_distance_and_elements(test)




## === cell 3
type_medians = (
    train.groupby("type")["scalar_coupling_constant"]
    .median()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "pred_type"})
)

train["atom_a"] = train[["atom0", "atom1"]].min(axis=1)
train["atom_b"] = train[["atom0", "atom1"]].max(axis=1)

group_medians = (
    train.groupby(["type", "atom_a", "atom_b"])["scalar_coupling_constant"]
    .median()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "pred_group"})
)

train["elem_a"] = train[["atom0_elem", "atom1_elem"]].min(axis=1)
train["elem_b"] = train[["atom0_elem", "atom1_elem"]].max(axis=1)

elem_group_medians = (
    train.groupby(["type", "elem_a", "elem_b"])["scalar_coupling_constant"]
    .median()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "pred_elem"})
)

global_mean = train["scalar_coupling_constant"].mean()

train = train.merge(group_medians, on=["type", "atom_a", "atom_b"], how="left")
train = train.merge(elem_group_medians, on=["type", "elem_a", "elem_b"], how="left")

train["base_pred"] = train["pred_elem"]
train["base_pred"].fillna(train["pred_group"], inplace=True)
train["base_pred"].fillna(
    train["type"].map(type_medians.set_index("type")["pred_type"]), inplace=True
)
train["base_pred"].fillna(global_mean, inplace=True)

train["residual"] = train["scalar_coupling_constant"] - train["base_pred"]

type_params = {}
type_counts = {}

valid = train.dropna(subset=["residual", "distance"])

for t, sub in valid.groupby("type"):
    A = np.vstack([sub["distance"].values, np.ones_like(sub["distance"].values)]).T
    y = sub["residual"].values
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    type_params[t] = (a, b)
    type_counts[t] = len(sub)

A_glob = np.vstack([train["distance"].values, np.ones_like(train["distance"].values)]).T
y_glob = train["residual"].values
global_a, global_b = np.linalg.lstsq(A_glob, y_glob, rcond=None)[0]

type_params_df = (
    pd.DataFrame.from_dict(type_params, orient="index", columns=["coef", "intercept"])
    .reset_index()
    .rename(columns={"index": "type"})
)
type_params_df["count"] = type_params_df["type"].map(type_counts)




## === cell 4
test["atom_a"] = test[["atom0", "atom1"]].min(axis=1)
test["atom_b"] = test[["atom0", "atom1"]].max(axis=1)

test["elem_a"] = test[["atom0_elem", "atom1_elem"]].min(axis=1)
test["elem_b"] = test[["atom0_elem", "atom1_elem"]].max(axis=1)

test_pred = test.merge(group_medians, on=["type", "atom_a", "atom_b"], how="left")
test_pred = test_pred.merge(
    elem_group_medians, on=["type", "elem_a", "elem_b"], how="left"
)
test_pred = test_pred.merge(type_medians, on="type", how="left")

test_pred["pred"] = test_pred["pred_elem"]
test_pred["pred"].fillna(test_pred["pred_group"], inplace=True)
test_pred["pred"].fillna(test_pred["pred_type"], inplace=True)
test_pred["pred"].fillna(global_mean, inplace=True)

test_pred = test_pred.merge(
    type_params_df[["type", "coef", "intercept", "count"]],
    on="type",
    how="left",
)

use_type = test_pred["coef"].notna()
corr = np.where(
    use_type,
    test_pred["coef"] * test_pred["distance"] + test_pred["intercept"],
    global_a * test_pred["distance"] + global_b,
)
test_pred["pred"] = test_pred["pred"] + pd.Series(corr).fillna(0)
test_pred["pred"].fillna(global_mean, inplace=True)




## === cell 5
submission = sample_submission.copy()
submission["scalar_coupling_constant"] = test_pred["pred"].values

submission_path = "stackers_blend.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, shape: {submission.shape}")
