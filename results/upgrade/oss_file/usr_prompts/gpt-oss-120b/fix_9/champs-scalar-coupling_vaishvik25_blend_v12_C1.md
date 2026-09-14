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

-1.3534795878235684

# 6. Current score

1.18497

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'I replace the failing code that tries to load non‑existent previous submissions with a simple, reproducible baseline: compute the average scalar coupling constant for each coupling type from the training set and use those averages as predictions for the test set. This ensures the script runs end‑to‑end, creates a correctly‑formatted `submission.csv` with the required columns, and avoids the earlier FileNotFound and NameError issues.'
- What this solution (achieved 1.18497) has done: 'I replace the per‑type mean baseline with a per‑type median baseline (median is the optimal constant for minimizing MAE, which directly improves the log‑MAE metric). Missing types are filled with the overall median. This small adjustment keeps the original workflow unchanged while moving the score lower (closer to the target).'
- What this solution (achieved 1.41353) has done: 'I replace the simple per‑type median prediction with a per‑type median of the log‑transformed target (using `np.log1p`), then exponentiate back with `np.expm1`. This better matches the log‑MAE evaluation metric and should modestly lower the score, moving it toward the target while keeping the overall workflow unchanged. The fallback global prediction is also adjusted to the same transformation.'
- What this solution (achieved 1.18497) has done: 'I replace the log‑median baseline with the direct median of the raw scalar coupling constants for each coupling type (the optimal constant for minimizing MAE). This keeps the overall workflow unchanged while providing predictions that better align with the competition’s MAE‑based metric, moving the score closer to the target. The fallback global median is also updated accordingly.'
- What this solution (achieved 1.44597) has done: 'I replace the simple per‑type median baseline with a per‑type geometric‑mean baseline ( exp of the mean of log targets ) and use the overall geometric mean as a fallback. This aligns the constant predictions with the log‑MAE evaluation metric and should lower the score toward the target while keeping the overall workflow unchanged.'
- What this solution (achieved 1.18497) has done: 'I replace the geometric‑mean baseline with a per‑type median (and a global median fallback). The median is the optimal constant for minimizing MAE, which directly reduces the log‑MAE metric used in the competition, so the validation score should move lower toward the target. All other parts of the script remain unchanged.'
- What this solution (achieved 1.18497) has done: 'I enrich the baseline by using atom element information from the structures file: for each coupling I create a key that combines the coupling type and the two atom types, compute the median target for each such key, and use it for predictions (falling back to the per‑type median and finally the overall median). This adds only lightweight joins and should lower the MAE, moving the log‑MAE score closer to the negative target while keeping the original workflow intact.'
- What this solution (achieved 1.18497) has done: 'We keep the original workflow but add two extra, more specific fallback predictions: the median scalar coupling for each `(type, atom_0)` pair and for each `(type, atom_1)` pair. After trying the detailed `type_atom_key` median, we fill missing values first with the `(type, atom_0)` median, then the `(type, atom_1)` median, then the per‑type median, and finally the global median. This adds only lightweight joins and should reduce the MAE (and thus the log‑MAE) moving the score closer to the negative target without changing the core model logic.'

# 9. Code solution

## === cell 0
import os, pandas as pd, numpy as np

print("Root input directory contents:", os.listdir("../input"))



## === cell 1
train_path = os.path.join("..", "input", "champs-scalar-coupling", "train.csv")
test_path = os.path.join("..", "input", "champs-scalar-coupling", "test.csv")
structures_path = os.path.join(
    "..", "input", "champs-scalar-coupling", "structures.csv"
)

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
structures = pd.read_csv(structures_path)



## === cell 2
train = train.merge(
    structures[["molecule_name", "atom_index", "atom"]].rename(
        columns={"atom_index": "atom_index_0", "atom": "atom_0"}
    ),
    on=["molecule_name", "atom_index_0"],
    how="left",
)
train = train.merge(
    structures[["molecule_name", "atom_index", "atom"]].rename(
        columns={"atom_index": "atom_index_1", "atom": "atom_1"}
    ),
    on=["molecule_name", "atom_index_1"],
    how="left",
)

test = test.merge(
    structures[["molecule_name", "atom_index", "atom"]].rename(
        columns={"atom_index": "atom_index_0", "atom": "atom_0"}
    ),
    on=["molecule_name", "atom_index_0"],
    how="left",
)
test = test.merge(
    structures[["molecule_name", "atom_index", "atom"]].rename(
        columns={"atom_index": "atom_index_1", "atom": "atom_1"}
    ),
    on=["molecule_name", "atom_index_1"],
    how="left",
)


def make_key(df):
    atoms = np.where(
        df["atom_0"] <= df["atom_1"],
        df["atom_0"] + "_" + df["atom_1"],
        df["atom_1"] + "_" + df["atom_0"],
    )
    return df["type"] + "_" + atoms


train["type_atom_key"] = make_key(train)
test["type_atom_key"] = make_key(test)



## === cell 3
key_median = (
    train.groupby("type_atom_key")["scalar_coupling_constant"]
    .median()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "pred_key"})
)

test_pred = test.merge(key_median, on="type_atom_key", how="left")

type_atom0_median = (
    train.groupby(["type", "atom_0"])["scalar_coupling_constant"]
    .median()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "pred_type_atom0"})
)
test_pred = test_pred.merge(type_atom0_median, on=["type", "atom_0"], how="left")
test_pred["pred_key"].fillna(test_pred["pred_type_atom0"], inplace=True)

type_atom1_median = (
    train.groupby(["type", "atom_1"])["scalar_coupling_constant"]
    .median()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "pred_type_atom1"})
)
test_pred = test_pred.merge(type_atom1_median, on=["type", "atom_1"], how="left")
test_pred["pred_key"].fillna(test_pred["pred_type_atom1"], inplace=True)

type_median = (
    train.groupby("type")["scalar_coupling_constant"]
    .median()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "pred_type"})
)
test_pred = test_pred.merge(type_median, on="type", how="left")
test_pred["pred_key"].fillna(test_pred["pred_type"], inplace=True)

global_median = train["scalar_coupling_constant"].median()
test_pred["pred_key"].fillna(global_median, inplace=True)



## === cell 4
submission = test_pred[["id", "pred_key"]].copy()
submission.rename(columns={"pred_key": "scalar_coupling_constant"}, inplace=True)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path} with {len(submission)} rows.")
