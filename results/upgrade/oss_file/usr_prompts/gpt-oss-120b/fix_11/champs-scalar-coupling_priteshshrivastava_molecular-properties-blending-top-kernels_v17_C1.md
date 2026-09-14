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

- What this solution (achieved 1.23566) has done: 'The blend script referenced nonexistent CSV files, causing a FileNotFoundError. I replaced the blending step with a simple, reliable baseline: read the official training and test data, compute the average coupling constant for each coupling type, and use those averages as predictions. This ensures the script runs to completion, writes a correctly‑named `submission.csv` with the required columns, and yields a valid submission without altering any core modeling logic beyond replacing the missing files.'
- What this solution (achieved 1.23566) has done: 'I enrich the baseline by adding atom‑type information for each pair: after loading the structures file I merge the atom symbols for both atoms into the train and test tables, compute mean coupling constants for each (type, atom0, atom1) combination, and use these more detailed averages for predictions, falling back to the original type‑wise mean and finally the global mean. This small feature addition should lower the log‑MAE toward the target while keeping the original workflow intact.'
- What this solution (achieved 1.23566) has done: 'I add a simple geometric feature – the Euclidean distance between the two atoms – and use the mean coupling constant for each (type, distance bucket) as an additional prediction level. The new distance‑based means are tried after the atom‑pair means and before falling back to the type‑wise and global means, which should lower the log‑MAE toward the negative target while keeping the original baseline logic intact.'
- What this solution (achieved 1.23566) has done: 'I add a more specific grouping that also considers the distance bin together with the atom‑pair and coupling type. The prediction hierarchy now tries (type, atom₀, atom₁, distance‑bin) → (type, atom₀, atom₁) → (type, distance‑bin) → type → global mean, which should reduce the log‑MAE and move the score closer to the negative target while keeping the original workflow unchanged.'
- What this solution (achieved 1.23566) has done: 'I fix the NaN‑to‑int conversion error by filling missing distances before binning, so `dist_bin` is always a valid integer column in both train and test. This resolves the `IntCastingNaNError` and the subsequent `KeyError` for `dist_bin`. The rest of the logic remains unchanged, preserving the hierarchical mean‑based predictions while ensuring a valid `submission.csv` is written.'
- What this solution (achieved 1.23566) has done: 'I slightly refine the distance binning (using a finer 0.1 Å per bin) to let the hierarchical means capture more geometric detail, and I drop the restrictive clipping of predictions so they can naturally fall outside the training range when appropriate. These minimal tweaks keep the original mean‑based logic intact while aiming to lower the log‑MAE toward the target.'
- What this solution (achieved 1.18497) has done: 'I replace the group‑wise means with medians, which are less sensitive to outliers and often lower the MAE (and thus the log‑MAE) while keeping the same hierarchical fallback logic. This minimal tweak preserves the overall workflow and still writes a valid `submission.csv`.'
- What this solution (achieved 1.23566) has done: 'I replace all median aggregations with means (type, global, pair, distance, and combo) to provide predictions that better match the underlying distribution and typically reduce MAE, moving the log‑MAE toward the lower target while keeping the original workflow unchanged.'
- What this solution (achieved 1.18497) has done: 'I replace the mean‑based aggregations with median‑based ones for all hierarchical groups (type, atom pair, distance bin, and their combinations). Using medians is less sensitive to outliers and typically lowers the MAE, which in turn reduces the log‑MAE toward the negative target while keeping the overall workflow unchanged.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

print(os.listdir("../input"))

train_path = "../input/champs-scalar-coupling/train.csv"
test_path = "../input/champs-scalar-coupling/test.csv"
struct_path = "../input/champs-scalar-coupling/structures.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
structures = pd.read_csv(struct_path)

struct0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
    }
)
struct1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
    }
)

train = train.merge(
    struct0[["molecule_name", "atom_index_0", "atom_0"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
)
train = train.merge(
    struct1[["molecule_name", "atom_index_1", "atom_1"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
)
test = test.merge(
    struct0[["molecule_name", "atom_index_0", "atom_0"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
)
test = test.merge(
    struct1[["molecule_name", "atom_index_1", "atom_1"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
)

train = train.merge(
    struct0[["molecule_name", "atom_index_0", "x0", "y0", "z0"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
)
train = train.merge(
    struct1[["molecule_name", "atom_index_1", "x1", "y1", "z1"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
)
test = test.merge(
    struct0[["molecule_name", "atom_index_0", "x0", "y0", "z0"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
)
test = test.merge(
    struct1[["molecule_name", "atom_index_1", "x1", "y1", "z1"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
)

train["dist"] = np.sqrt(
    (train["x0"] - train["x1"]) ** 2
    + (train["y0"] - train["y1"]) ** 2
    + (train["z0"] - train["z1"]) ** 2
)
test["dist"] = np.sqrt(
    (test["x0"] - test["x1"]) ** 2
    + (test["y0"] - test["y1"]) ** 2
    + (test["z0"] - test["z1"]) ** 2
)

train["dist_bin"] = (train["dist"] * 10).fillna(-1).astype(int)
test["dist_bin"] = (test["dist"] * 10).fillna(-1).astype(int)



## === cell 1
type_medians = train.groupby("type")["scalar_coupling_constant"].median()
global_median = train["scalar_coupling_constant"].median()

pair_medians = train.groupby(["type", "atom_0", "atom_1"])[
    "scalar_coupling_constant"
].median()
dist_medians = train.groupby(["type", "dist_bin"])["scalar_coupling_constant"].median()
combo_medians = train.groupby(["type", "atom_0", "atom_1", "dist_bin"])[
    "scalar_coupling_constant"
].median()

combo_key = list(zip(test["type"], test["atom_0"], test["atom_1"], test["dist_bin"]))
pair_key = list(zip(test["type"], test["atom_0"], test["atom_1"]))
dist_key = list(zip(test["type"], test["dist_bin"]))

pred_combo = combo_medians.reindex(combo_key).values
pred_pair = pair_medians.reindex(pair_key).values
pred_dist = dist_medians.reindex(dist_key).values
pred_type = test["type"].map(type_medians).values

predictions = pd.Series(pred_combo)
predictions = predictions.fillna(pd.Series(pred_pair))
predictions = predictions.fillna(pd.Series(pred_dist))
predictions = predictions.fillna(pd.Series(pred_type))
predictions = predictions.fillna(global_median)

submission = pd.DataFrame({"id": test["id"], "scalar_coupling_constant": predictions})

submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv, shape:", submission.shape)
