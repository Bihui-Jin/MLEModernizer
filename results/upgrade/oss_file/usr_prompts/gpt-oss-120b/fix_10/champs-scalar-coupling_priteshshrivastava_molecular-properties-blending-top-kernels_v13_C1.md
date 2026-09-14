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

-1.666828640654786

# 6. Current score

1.23566

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'The original script fails because the referenced blend CSV files do not exist in the provided environment. I replaced the missing‑file logic with a simple, reproducible baseline: read the official training set, compute the average `scalar_coupling_constant` for each coupling `type`, and use these type‑wise means as predictions for the test set. The result is saved as a correctly formatted CSV submission (`my_blend_1.csv`). This fix restores end‑to‑end execution and produces a valid submission file.'
- What this solution (achieved 3.00563) has done: 'I keep the original data loading and basic checks, then enrich the training and test sets with the atomic element of each coupled atom by merging the `structures.csv` file. Using these atom types together with the coupling type, I compute a three‑way group mean (`type`, `atom_0`, `atom_1`) and use it as the prediction, falling back to the overall global mean when a combination is unseen. This adds useful chemical information with only a small change to the pipeline and is expected to lower the log‑MAE toward the target value.'
- What this solution (achieved 1.23566) has done: 'I fixed the error caused by calling `fillna` on an `Index` object. After mapping the pair‑wise means, the result is now explicitly turned into a `Series` so that `fillna` can accept another `Series` (the type‑wise means). The rest of the logic is unchanged, preserving the original feature engineering and prediction strategy while ensuring a valid `.csv` submission is written.'
- What this solution (achieved 1.23566) has done: 'I replace the unordered atom‑pair key with an ordered combination of the two atom types, grouping by `type`, `atom_0` and `atom_1`. This keeps the overall workflow identical while providing a finer‑grained mean for each specific ordered pair, which should reduce the log‑MAE and move the score closer to the target. The rest of the script (data loading, merges, fall‑backs) remains unchanged.'
- What this solution (achieved 1.23566) has done: 'The fix corrects the `fillna` usage that was incorrectly passing an `Index` object; we now build a proper `Series` for the fallback predictions based on pair means. This resolves the runtime error and ensures a valid CSV submission is written. No other logic is changed, preserving the original model approach.'
- What this solution (achieved 1.23566) has done: 'The changes add the scalar‑coupling contribution data and use the mean total contribution for each `(type, atom_0, atom_1)` group as an additional fallback prediction. This keeps the original grouping logic unchanged while providing a more chemically‑informed estimate, which should lower the log‑MAE and move the score closer to the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd

print("Root input contents:", os.listdir("../input"))
print("Champs folder contents:", os.listdir("../input/champs-scalar-coupling")[:5])



## === cell 1
import numpy as np

train_path = "../input/champs-scalar-coupling/train.csv"
test_path = "../input/champs-scalar-coupling/test.csv"
structures_path = "../input/champs-scalar-coupling/structures.csv"
contrib_path = (
    "../input/champs-scalar-coupling/scalar_coupling_contributions.csv"  # new
)

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

structures_df = pd.read_csv(structures_path)[
    ["molecule_name", "atom_index", "atom", "x", "y", "z"]
]

train_df = train_df.merge(
    structures_df.rename(
        columns={
            "atom_index": "atom_index_0",
            "atom": "atom_0",
            "x": "x0",
            "y": "y0",
            "z": "z0",
        }
    ),
    on=["molecule_name", "atom_index_0"],
    how="left",
)
test_df = test_df.merge(
    structures_df.rename(
        columns={
            "atom_index": "atom_index_0",
            "atom": "atom_0",
            "x": "x0",
            "y": "y0",
            "z": "z0",
        }
    ),
    on=["molecule_name", "atom_index_0"],
    how="left",
)

train_df = train_df.merge(
    structures_df.rename(
        columns={
            "atom_index": "atom_index_1",
            "atom": "atom_1",
            "x": "x1",
            "y": "y1",
            "z": "z1",
        }
    ),
    on=["molecule_name", "atom_index_1"],
    how="left",
)
test_df = test_df.merge(
    structures_df.rename(
        columns={
            "atom_index": "atom_index_1",
            "atom": "atom_1",
            "x": "x1",
            "y": "y1",
            "z": "z1",
        }
    ),
    on=["molecule_name", "atom_index_1"],
    how="left",
)


def euclidean(row):
    return np.sqrt(
        (row["x0"] - row["x1"]) ** 2
        + (row["y0"] - row["y1"]) ** 2
        + (row["z0"] - row["z1"]) ** 2
    )


train_df["distance"] = train_df.apply(euclidean, axis=1)
test_df["distance"] = test_df.apply(euclidean, axis=1)

train_df["dist_bin"] = (train_df["distance"] * 10).round() / 10
test_df["dist_bin"] = (test_df["distance"] * 10).round() / 10

pair_means = train_df.groupby(["type", "atom_0", "atom_1"])[
    "scalar_coupling_constant"
].mean()
pair_dist_means = train_df.groupby(["type", "atom_0", "atom_1", "dist_bin"])[
    "scalar_coupling_constant"
].mean()
type_means = train_df.groupby("type")["scalar_coupling_constant"].mean()
global_mean = train_df["scalar_coupling_constant"].mean()

contrib_df = pd.read_csv(contrib_path)[
    ["type", "atom_index_0", "atom_index_1", "fc", "sd", "pso", "dso"]
]
contrib_df = contrib_df.rename(
    columns={"atom_index_0": "atom_0", "atom_index_1": "atom_1"}
)
contrib_means = contrib_df.groupby(["type", "atom_0", "atom_1"])[
    ["fc", "sd", "pso", "dso"]
].mean()
contrib_means["total"] = contrib_means.sum(axis=1)
contrib_means_series = contrib_means["total"]

test_idx = test_df.set_index(["type", "atom_0", "atom_1", "dist_bin"]).index
pred_series = pd.Series(
    test_idx.map(lambda k: pair_dist_means.get(k, np.nan)), index=test_df.index
)

fallback_idx = test_df.set_index(["type", "atom_0", "atom_1"]).index
fallback_series = pd.Series(
    fallback_idx.map(lambda k: pair_means.get(k, np.nan)), index=test_df.index
)
pred_series = pred_series.fillna(fallback_series)

contrib_idx = test_df.set_index(["type", "atom_0", "atom_1"]).index
contrib_series = pd.Series(
    contrib_idx.map(lambda k: contrib_means_series.get(k, np.nan)), index=test_df.index
)
pred_series = pred_series.fillna(contrib_series)

pred_series = pred_series.fillna(test_df["type"].map(type_means))
pred_series = pred_series.fillna(global_mean)

submission = pd.DataFrame(
    {"id": test_df["id"], "scalar_coupling_constant": pred_series}
)

submission_path = "my_blend_3.csv"
submission.to_csv(submission_path, index=False)

print(f"Submission file written to {submission_path} with {len(submission)} rows.")
