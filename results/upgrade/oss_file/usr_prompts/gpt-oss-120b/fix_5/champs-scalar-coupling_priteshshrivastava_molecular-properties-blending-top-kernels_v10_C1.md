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

-1.6712010456498954

# 6. Current score

1.18497

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'The script failed because it tried to load non‑existent blending CSVs. I replaced that logic with a simple, reliable baseline: compute the average coupling constant (overall and per coupling type) from the provided training data and use these means to fill the submission for the test set. This ensures the code runs end‑to‑end, writes a correctly‑formatted `my_blend_1.csv`, and avoids any file‑not‑found errors.'
- What this solution (achieved 1.23566) has done: 'I enrich the simple type‑based averages with atom‑type information: for each coupling I add the element symbols of the two atoms (using `structures.csv`) and compute means for each `(type, atom_0, atom_1)` combination. Predictions first try this more specific mean, then fall back to the type mean, and finally to the overall mean. This adds informative granularity while keeping the original averaging approach and ensures a valid CSV submission.'
- What this solution (achieved 1.18497) has done: 'I replace the averaging logic with medians, which are the optimal point estimate for minimizing MAE. By using `median()` instead of `mean()` for the overall, per‑type, and (type,atom0,atom1) statistics, the predictions should become closer to the true constants, lowering the MAE and thus moving the log‑MAE score toward the negative target. No other parts of the pipeline are altered, so the script still runs end‑to‑end and writes a valid CSV.'
- What this solution (achieved 1.18497) has done: 'I keep the original median‑based blending approach but add two finer‑grained fallback statistics: medians per `(type, atom_0)` and per `(type, atom_1)`. When a specific `(type, atom_0, atom_1)` combination is missing in the training data, the script now tries to fill the prediction with the `(type, atom_0)` median, then the `(type, atom_1)` median, before falling back to the overall type median and finally the global median. This small extension adds useful atom‑type information without altering the core modeling logic, and is expected to reduce MAE and move the log‑MAE score closer to the negative target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd



## === cell 1
base_path = "../input/champs-scalar-coupling"

train_path = os.path.join(base_path, "train.csv")
train_df = pd.read_csv(train_path)

test_path = os.path.join(base_path, "test.csv")
test_df = pd.read_csv(test_path)

structures_path = os.path.join(base_path, "structures.csv")
struct_df = pd.read_csv(
    structures_path, usecols=["molecule_name", "atom_index", "atom"]
)

struct_idx = struct_df.set_index(["molecule_name", "atom_index"])["atom"]

train_df["atom_0"] = struct_idx.reindex(
    pd.MultiIndex.from_arrays([train_df["molecule_name"], train_df["atom_index_0"]])
).values
train_df["atom_1"] = struct_idx.reindex(
    pd.MultiIndex.from_arrays([train_df["molecule_name"], train_df["atom_index_1"]])
).values

test_df["atom_0"] = struct_idx.reindex(
    pd.MultiIndex.from_arrays([test_df["molecule_name"], test_df["atom_index_0"]])
).values
test_df["atom_1"] = struct_idx.reindex(
    pd.MultiIndex.from_arrays([test_df["molecule_name"], test_df["atom_index_1"]])
).values



## === cell 2
overall_median = train_df["scalar_coupling_constant"].median()
type_medians = train_df.groupby("type")["scalar_coupling_constant"].median()
combo_medians = train_df.groupby(["type", "atom_0", "atom_1"])[
    "scalar_coupling_constant"
].median()

type_atom0_medians = train_df.groupby(["type", "atom_0"])[
    "scalar_coupling_constant"
].median()
type_atom1_medians = train_df.groupby(["type", "atom_1"])[
    "scalar_coupling_constant"
].median()



## === cell 3
test_idx = pd.MultiIndex.from_arrays(
    [test_df["type"], test_df["atom_0"], test_df["atom_1"]],
    names=["type", "atom_0", "atom_1"],
)
test_df["scalar_coupling_constant"] = combo_medians.reindex(test_idx).values

missing_mask = test_df["scalar_coupling_constant"].isna()

if missing_mask.any():
    idx_atom0 = pd.MultiIndex.from_arrays(
        [test_df.loc[missing_mask, "type"], test_df.loc[missing_mask, "atom_0"]],
        names=["type", "atom_0"],
    )
    fill_atom0 = type_atom0_medians.reindex(idx_atom0).values
    test_df.loc[missing_mask, "scalar_coupling_constant"] = fill_atom0
    missing_mask = test_df["scalar_coupling_constant"].isna()

if missing_mask.any():
    idx_atom1 = pd.MultiIndex.from_arrays(
        [test_df.loc[missing_mask, "type"], test_df.loc[missing_mask, "atom_1"]],
        names=["type", "atom_1"],
    )
    fill_atom1 = type_atom1_medians.reindex(idx_atom1).values
    test_df.loc[missing_mask, "scalar_coupling_constant"] = fill_atom1
    missing_mask = test_df["scalar_coupling_constant"].isna()

if missing_mask.any():
    test_df.loc[missing_mask, "scalar_coupling_constant"] = test_df.loc[
        missing_mask, "type"
    ].map(type_medians)
    missing_mask = test_df["scalar_coupling_constant"].isna()

test_df["scalar_coupling_constant"].fillna(overall_median, inplace=True)



## === cell 4
submission = test_df[["id", "scalar_coupling_constant"]].copy()
submission.to_csv("my_blend_1.csv", index=False)
