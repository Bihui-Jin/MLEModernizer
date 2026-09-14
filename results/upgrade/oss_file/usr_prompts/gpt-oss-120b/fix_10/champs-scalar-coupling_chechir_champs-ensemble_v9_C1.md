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

-0.8555725082919478

# 6. Current score

1.23566

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'Ireplace the missing external prediction files with a simple baseline that uses the mean `scalar_coupling_constant` for each coupling `type` from the training data (falling back to the global mean when a type is absent). This removes the file‑not‑found errors, creates the required columns (`final_preds`), and writes a valid `sub_ensemble.csv` submission. The core logic stays unchanged apart from the prediction generation, which is needed to produce a runnable pipeline and a score that can be evaluated.'
- What this solution (achieved 1.23566) has done: 'I keep the same overall workflow but improve the prediction by adding a per‑molecule offset to the type‑wise mean. For each molecule we compute its average coupling constant in the training data; the test prediction becomes type_mean + (molecule_mean − global_mean). This small adjustment uses only existing columns, preserves the original logic, and should move the log‑MAE closer to the target lower value.'
- What this solution (achieved 1.23566) has done: 'I add atom‑type information from the structures file and compute a mean coupling for each combination of coupling type and the two atom elements. The prediction first try this more specific mean, fall back to the type‑wise mean, and then apply the same per‑molecule offset as before. This keeps the original baseline logic while using extra relevant data to reduce the error and move the score closer to the target.'
- What this solution (achieved 1.23566) has done: 'Implemented a robust per‑molecule‑type offset calculation that avoids the non‑unique index alignment error and restored the downstream workflow.  
- Re‑computed `mol_type_offset_dict` by directly subtracting the global type mean from each molecule‑type mean using index tuple mapping.  
- No other logic was altered, preserving the original baseline while ensuring a valid `final_preds` column and successful CSV output.'
- What this solution (achieved 1.23566) has done: 'I extend the baseline by (1) adding atom coordinates to compute inter‑atomic distances, (2) using unordered atom‑pair means so the prediction works regardless of atom order, and (3) applying a small distance‑based residual correction. These tweaks keep the original mean‑based logic while giving extra useful information, which should lower the log‑MAE toward the target without changing the overall workflow.'
- What this solution (achieved 1.23566) has done: 'I slightly smooth the distance‑based correction by rounding distance bins to one decimal place (giving more samples per bin) and add a small weighted overall‑molecule offset (based on each molecule’s mean deviation from the global mean). These minor tweaks keep the original mean‑based baseline while gently regularising predictions, which should lower the log‑MAE and move the score closer to the target.'
- What this solution (achieved 1.23566) has done: 'I keep the overall pipeline but soften the per‑molecule and distance corrections, which tend to over‑fit and increase the log‑MAE. By applying a smaller weight (0.5) to the molecule‑type offset and a modest overall‑molecule offset (0.1) while ignoring the distance‑based correction, the predictions stay close to the reliable type‑wise means yet still capture some systematic bias, moving the score toward the lower target value. I also clip predictions to the training target range to avoid extreme outliers.'
- What this solution (achieved 1.23566) has done: 'I increase the influence of the molecule‑type offset and the overall‑molecule offset, and also add a modest contribution from the distance‑based correction. This keeps the original baseline logic while giving the model more calibrated adjustments, which should lower the log‑MAE and move the score closer to the target.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import os

TRAIN_PATH = "../input/champs-scalar-coupling/train.csv"
TEST_PATH = "../input/champs-scalar-coupling/test.csv"
STRUCTURES_PATH = "../input/champs-scalar-coupling/structures.csv"
SUBMISSION_PATH = "sub_ensemble.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

structures = pd.read_csv(STRUCTURES_PATH)[
    ["molecule_name", "atom_index", "atom", "x", "y", "z"]
]

train = (
    train.merge(
        structures,
        left_on=["molecule_name", "atom_index_0"],
        right_on=["molecule_name", "atom_index"],
        how="left",
    )
    .rename(columns={"atom": "atom_0", "x": "x_0", "y": "y_0", "z": "z_0"})
    .drop(columns=["atom_index"])
)

test = (
    test.merge(
        structures,
        left_on=["molecule_name", "atom_index_0"],
        right_on=["molecule_name", "atom_index"],
        how="left",
    )
    .rename(columns={"atom": "atom_0", "x": "x_0", "y": "y_0", "z": "z_0"})
    .drop(columns=["atom_index"])
)

train = (
    train.merge(
        structures,
        left_on=["molecule_name", "atom_index_1"],
        right_on=["molecule_name", "atom_index"],
        how="left",
    )
    .rename(columns={"atom": "atom_1", "x": "x_1", "y": "y_1", "z": "z_1"})
    .drop(columns=["atom_index"])
)

test = (
    test.merge(
        structures,
        left_on=["molecule_name", "atom_index_1"],
        right_on=["molecule_name", "atom_index"],
        how="left",
    )
    .rename(columns={"atom": "atom_1", "x": "x_1", "y": "y_1", "z": "z_1"})
    .drop(columns=["atom_index"])
)



## === cell 1
type_means = train.groupby("type")["scalar_coupling_constant"].mean()
global_mean = train["scalar_coupling_constant"].mean()
molecule_means = train.groupby("molecule_name")["scalar_coupling_constant"].mean()

type_atom_means = train.groupby(["type", "atom_0", "atom_1"])[
    "scalar_coupling_constant"
].mean()
type_atom_means_dict = type_atom_means.to_dict()

train["atom_pair"] = train.apply(
    lambda r: tuple(sorted([r["atom_0"], r["atom_1"]])), axis=1
)
type_atom_pair_means = train.groupby(["type", "atom_pair"])[
    "scalar_coupling_constant"
].mean()
type_atom_pair_means_dict = type_atom_pair_means.to_dict()

molecule_type_means = train.groupby(["molecule_name", "type"])[
    "scalar_coupling_constant"
].mean()
offset_series = molecule_type_means - molecule_type_means.index.map(
    lambda idx: type_means[idx[1]]
)
mol_type_offset_dict = offset_series.to_dict()




## === cell 2
def get_base_pred(row):
    key_unordered = (row["type"], tuple(sorted([row["atom_0"], row["atom_1"]])))
    if key_unordered in type_atom_pair_means_dict:
        return type_atom_pair_means_dict[key_unordered]
    key_ordered = (row["type"], row["atom_0"], row["atom_1"])
    if key_ordered in type_atom_means_dict:
        return type_atom_means_dict[key_ordered]
    return type_means.get(row["type"], global_mean)


test["type_pred"] = test.apply(get_base_pred, axis=1)
test["offset"] = test.apply(
    lambda r: mol_type_offset_dict.get((r["molecule_name"], r["type"]), 0.0), axis=1
)

train["dist"] = np.sqrt(
    (train["x_0"] - train["x_1"]) ** 2
    + (train["y_0"] - train["y_1"]) ** 2
    + (train["z_0"] - train["z_1"]) ** 2
)
test["dist"] = np.sqrt(
    (test["x_0"] - test["x_1"]) ** 2
    + (test["y_0"] - test["y_1"]) ** 2
    + (test["z_0"] - test["z_1"]) ** 2
)

train["dist_bin"] = train["dist"].round(1)
test["dist_bin"] = test["dist"].round(1)

train["type_pred"] = train.apply(get_base_pred, axis=1)
train["offset"] = train.apply(
    lambda r: mol_type_offset_dict.get((r["molecule_name"], r["type"]), 0.0), axis=1
)
train["base_pred"] = train["type_pred"] + train["offset"]

train["residual"] = train["scalar_coupling_constant"] - train["base_pred"]
dist_residual_means = train.groupby("dist_bin")["residual"].mean()
dist_residual_dict = dist_residual_means.to_dict()

test["dist_correction"] = test["dist_bin"].map(dist_residual_dict).fillna(0.0)

mol_overall_offset = (molecule_means - global_mean).to_dict()
test["overall_offset"] = test["molecule_name"].map(mol_overall_offset).fillna(0.0)

test["final_preds"] = (
    test["type_pred"]
    + 0.8 * test["offset"]
    + 0.1 * test["dist_correction"]
    + 0.3 * test["overall_offset"]
)

min_target = train["scalar_coupling_constant"].min()
max_target = train["scalar_coupling_constant"].max()
test["final_preds"] = test["final_preds"].clip(lower=min_target, upper=max_target)



## === cell 3
submission = pd.DataFrame(
    {"id": test["id"], "scalar_coupling_constant": test["final_preds"]}
)



## === cell 4
submission.to_csv(SUBMISSION_PATH, index=False)



## === cell 5
print(f"Submission written to {os.path.abspath(SUBMISSION_PATH)}")
print(submission.head(10))
