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

-1.6777209112242684

# 6. Current score

1.18523

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.18497) has done: 'I remove the dependency on missing external Kaggle datasets (`keras-neural-net-and-distance-features` and `keras-nn-with-multi-output`) and instead build a self-contained baseline that always produces predictions for every test `id`. To keep core logic minimal and stable, I use a simple per-`type` median target computed from `train.csv` and apply it to `test.csv`, with a global fallback for any unseen types. I also ensure the submission contains all required ids exactly once, sorted by `id`, matching the sample submission format. This fix the runtime errors and the “Missing required ids” submission error while yielding a reasonable non-null score.'
- What this solution (achieved 1.18523) has done: 'Your current score (1.18497, lower-is-better) is far worse than the target (-1.6777), so we should improve legitimately while keeping the same “per-type constant prediction” core logic. The smallest strong gain is to use the known decomposition of the target: predict per-type medians for the four contribution terms (`fc`,`sd`,`pso`,`dso`) from `scalar_coupling_contributions.csv` and sum them to form the final prediction; for test, merge contributions by `(molecule_name, atom_index_0, atom_index_1, type)` and fall back to per-type median constants when missing. This preserves the “median-per-type baseline” approach but aligns it with a much more predictive signal that is available for both train and test. We also keep all the submission integrity checks (ids, sorting, no duplicates) unchanged to ensure a valid submission CSV.'
- What this solution (achieved 1.18523) has done: 'Your current approach already uses the best “available-for-test” signal (the contributions file) and then falls back to per-type medians, so the main realistic gain with minimal logic change is to (1) make the contributions join robust to reversed atom order, and (2) align training/test aggregates by symmetrizing the contributions table the same way. This keeps the same “merge contributions if present, else per-type median of terms, else per-type target median” semantics, but should reduce missing merges and improve MAE without changing the model concept. I also avoid the slow per-row `map(lambda ...)` and replace it with a vectorized reindex+fill, which is deterministic and reduces runtime risk. Output path, submission schema, id alignment checks, and the overall baseline logic remain unchanged.'

# 9. Code solution

## === cell 0
import warnings

warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd

BASE_PATH = "../input/champs-scalar-coupling"
print("Listing ../input:", os.listdir("../input")[:10])
print("Using BASE_PATH:", BASE_PATH)



## === cell 1
train = pd.read_csv(
    f"{BASE_PATH}/train.csv",
    usecols=["type", "scalar_coupling_constant"],
)
test = pd.read_csv(
    f"{BASE_PATH}/test.csv",
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
)

sample_sub = pd.read_csv(
    f"{BASE_PATH}/sample_submission.csv", usecols=["id", "scalar_coupling_constant"]
)

print("train:", train.shape, "test:", test.shape, "sample_sub:", sample_sub.shape)
print(
    "Unique types in train:",
    train["type"].nunique(),
    "in test:",
    test["type"].nunique(),
)



## === cell 2
contrib = pd.read_csv(
    f"{BASE_PATH}/scalar_coupling_contributions.csv",
    usecols=[
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "fc",
        "sd",
        "pso",
        "dso",
    ],
)

term_cols = ["fc", "sd", "pso", "dso"]
test_key = ["molecule_name", "atom_index_0", "atom_index_1", "type"]

contrib_base = contrib.copy()
contrib_base["contrib_sum"] = contrib_base[term_cols].sum(axis=1).astype(float)
contrib_base = contrib_base[test_key + term_cols + ["contrib_sum"]]

contrib_swapped = contrib_base.rename(
    columns={"atom_index_0": "atom_index_1", "atom_index_1": "atom_index_0"}
)

contrib_sym = pd.concat([contrib_base, contrib_swapped], ignore_index=True)

contrib_sym = contrib_sym.groupby(test_key, as_index=False)[
    term_cols + ["contrib_sum"]
].median()

type_term_median = contrib_sym.groupby("type")[term_cols].median()
global_term_median = contrib_sym[term_cols].median()

type_median_target = train.groupby("type")["scalar_coupling_constant"].median()
global_median_target = float(train["scalar_coupling_constant"].median())

pred_terms_df = type_term_median.reindex(test["type"]).reset_index(drop=True)
pred_terms_df = pred_terms_df.fillna(global_term_median)
pred_from_terms_sum = pred_terms_df[term_cols].sum(axis=1).astype(float)

test_with = test.merge(
    contrib_sym[test_key + ["contrib_sum"]],
    on=test_key,
    how="left",
    copy=False,
)

pred = test_with["contrib_sum"].astype(float)
pred = pred.fillna(pred_from_terms_sum)

pred = (
    pred.fillna(test["type"].map(type_median_target))
    .fillna(global_median_target)
    .astype(float)
)

sub = pd.DataFrame(
    {"id": test["id"].astype(np.int64), "scalar_coupling_constant": pred}
)

sub = sub.sort_values("id").reset_index(drop=True)

missing = set(sample_sub["id"]) - set(sub["id"])
extra = set(sub["id"]) - set(sample_sub["id"])
dupes = sub["id"].duplicated().sum()

print("Missing ids vs sample:", len(missing))
print("Extra ids vs sample:", len(extra))
print("Duplicate ids:", dupes)

assert len(missing) == 0, "Submission is missing required ids."
assert len(extra) == 0, "Submission has extra ids not in sample submission."
assert dupes == 0, "Submission contains duplicate ids."
assert sub.shape[0] == sample_sub.shape[0], "Row count mismatch with sample submission."

sub.head()



## === cell 3
out_path = "/kaggle/working/submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(sub.describe(include="all"))
