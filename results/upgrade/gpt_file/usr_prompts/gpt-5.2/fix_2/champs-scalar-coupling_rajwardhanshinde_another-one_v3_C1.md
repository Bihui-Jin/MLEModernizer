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

-1.5209593019916507

# 6. Current score

1.18497

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 1.18497) has done: 'Your notebook fails because it tries to read two external submissions from `../input/...` folders that are not present in your environment, so `sub1/sub2` never get created and all downstream cells error. I keep the same “blend two submissions into `sample_submission` and write a CSV” core logic, but make it robust by (1) loading those files only if they exist, otherwise falling back to valid baseline predictions built from the provided competition data. This guarantees the notebook runs end-to-end and always writes `stackers_blend.csv` with the required columns. The fallback uses only train statistics per coupling `type` (median) which is score-reasonable and should move you toward the target much more than a constant-0 submission.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path


def find_existing_path(candidates):
    for p in candidates:
        if p is None:
            continue
        p = Path(p)
        if p.exists():
            return str(p)
    return None


BASE_INPUT = find_existing_path(
    [
        "/kaggle/data/champs-scalar-coupling",
        "/kaggle/input/champs-scalar-coupling",
        "../input/champs-scalar-coupling",
        "/kaggle/data/input/champs-scalar-coupling",
        "/kaggle/input",
        "../input",
    ]
)

if BASE_INPUT is None:
    raise FileNotFoundError(
        "Could not locate the input directory containing champs-scalar-coupling data."
    )

print("Resolved BASE_INPUT:", BASE_INPUT)



## === cell 1
import numpy as np
import pandas as pd
import seaborn as sns

import os

try:
    print("Contents of BASE_INPUT:", os.listdir(BASE_INPUT)[:20])
except Exception as e:
    print("Could not list BASE_INPUT due to:", repr(e))



## === cell 2

sample_path = os.path.join(BASE_INPUT, "sample_submission.csv")
train_path = os.path.join(BASE_INPUT, "train.csv")
test_path = os.path.join(BASE_INPUT, "test.csv")

sample = pd.read_csv(sample_path)
test = pd.read_csv(test_path)

sub1_path = find_existing_path(
    [
        "../input/lgb-public-kernels-plus-more-features/sub_lgb_model_individual.csv",
        "/kaggle/input/lgb-public-kernels-plus-more-features/sub_lgb_model_individual.csv",
    ]
)
sub2_path = find_existing_path(
    [
        "../input/staking-and-stealing-like-a-molecule/submission.csv",
        "/kaggle/input/staking-and-stealing-like-a-molecule/submission.csv",
    ]
)

sub1 = None
sub2 = None


def validate_submission(df, name):
    if df is None:
        return False
    ok = ("id" in df.columns) and ("scalar_coupling_constant" in df.columns)
    if not ok:
        print(f"{name} invalid columns: {df.columns.tolist()}")
        return False
    if df["id"].nunique() != len(df):
        print(f"{name} has duplicate ids; will re-aggregate by id with mean.")
        df = df.groupby("id", as_index=False)["scalar_coupling_constant"].mean()
    return True


if sub1_path is not None:
    sub1 = pd.read_csv(sub1_path)
    if not validate_submission(sub1, "sub1"):
        sub1 = None
else:
    print("sub1 file not found; will use fallback predictions for sub1.")

if sub2_path is not None:
    sub2 = pd.read_csv(sub2_path)
    if not validate_submission(sub2, "sub2"):
        sub2 = None
else:
    print("sub2 file not found; will use fallback predictions for sub2.")


def make_type_median_baseline(train_csv_path, test_df):
    train = pd.read_csv(train_csv_path, usecols=["type", "scalar_coupling_constant"])
    med = train.groupby("type")["scalar_coupling_constant"].median()
    global_median = float(train["scalar_coupling_constant"].median())
    pred = test_df["type"].map(med).fillna(global_median).astype(np.float32)
    out = pd.DataFrame(
        {"id": test_df["id"].values, "scalar_coupling_constant": pred.values}
    )
    return out


if sub1 is None:
    sub1 = make_type_median_baseline(train_path, test)
if sub2 is None:
    sub2 = make_type_median_baseline(train_path, test)


def align_to_sample(sub, sample_ids):
    if sub["id"].nunique() != len(sub):
        sub = sub.groupby("id", as_index=False)["scalar_coupling_constant"].mean()
    sub = sub.set_index("id").reindex(sample_ids)
    if sub["scalar_coupling_constant"].isna().any():
        sub["scalar_coupling_constant"] = sub["scalar_coupling_constant"].fillna(0.0)
    return sub["scalar_coupling_constant"].values


ids = sample["id"].values
pred1 = align_to_sample(sub1, ids)
pred2 = align_to_sample(sub2, ids)

print("Loaded/created predictions:", pred1.shape, pred2.shape)
print(
    "Prediction stats sub1:",
    float(np.min(pred1)),
    float(np.max(pred1)),
    float(np.mean(pred1)),
)
print(
    "Prediction stats sub2:",
    float(np.min(pred2)),
    float(np.max(pred2)),
    float(np.mean(pred2)),
)



## === cell 3
sub1_aligned_series = pd.Series(pred1, name="scalar_coupling_constant")
print(sub1_aligned_series.describe())



## === cell 4
sub2_aligned_series = pd.Series(pred2, name="scalar_coupling_constant")
print(sub2_aligned_series.describe())



## === cell 5
sample["scalar_coupling_constant"] = (0.6 * pred2 + 0.4 * pred1).astype(np.float32)

assert list(sample.columns) == [
    "id",
    "scalar_coupling_constant",
], f"Unexpected submission columns: {sample.columns.tolist()}"
assert len(sample) == len(ids)
assert sample["scalar_coupling_constant"].notna().all()

sample.to_csv("stackers_blend.csv", index=False)
print("Wrote submission:", "stackers_blend.csv", "rows:", len(sample))
print(sample.head())



## === cell 6
sns.histplot(sample["scalar_coupling_constant"], bins=100)
