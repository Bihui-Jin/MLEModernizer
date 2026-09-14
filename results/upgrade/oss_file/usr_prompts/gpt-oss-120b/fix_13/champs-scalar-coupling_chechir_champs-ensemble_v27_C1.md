# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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

# 5. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

BASE_PATH = "../input/champs-scalar-coupling"
if not os.path.isdir(BASE_PATH):
    BASE_PATH = "/kaggle/input/champs-scalar-coupling"

TRAIN_PATH = os.path.join(BASE_PATH, "train.csv")
TEST_PATH = os.path.join(BASE_PATH, "test.csv")
STRUCTURES_PATH = os.path.join(BASE_PATH, "structures.csv")
DIPOLE_PATH = os.path.join(BASE_PATH, "dipole_moments.csv")
ENERGY_PATH = os.path.join(BASE_PATH, "potential_energy.csv")

print("Paths resolved:")
print("TRAIN:", TRAIN_PATH)
print("TEST :", TEST_PATH)
print("STRUCTURES:", STRUCTURES_PATH)
print("DIPOLE:", DIPOLE_PATH)
print("ENERGY:", ENERGY_PATH)

train_dtypes = {
    "id": "int64",
    "molecule_name": "category",
    "atom_index_0": "int16",
    "atom_index_1": "int16",
    "type": "category",
    "scalar_coupling_constant": "float32",
}
test_dtypes = {
    "id": "int64",
    "molecule_name": "category",
    "atom_index_0": "int16",
    "atom_index_1": "int16",
    "type": "category",
}
struct_dtypes = {
    "molecule_name": "category",
    "atom_index": "int16",
    "atom": "category",
    "x": "float32",
    "y": "float32",
    "z": "float32",
}
dipole_dtypes = {
    "molecule_name": "category",
    "X": "float32",
    "Y": "float32",
    "Z": "float32",
}
energy_dtypes = {
    "molecule_name": "category",
    "potential_energy": "float32",
}

train_df = pd.read_csv(TRAIN_PATH, dtype=train_dtypes)
test_df = pd.read_csv(TEST_PATH, dtype=test_dtypes)
print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)

struct_df = pd.read_csv(
    STRUCTURES_PATH,
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
    dtype=struct_dtypes,
)
print("Structures shape:", struct_df.shape)

dipole_df = pd.read_csv(DIPOLE_PATH, dtype=dipole_dtypes)
energy_df = pd.read_csv(ENERGY_PATH, dtype=energy_dtypes)

struct_idx = struct_df.set_index(["molecule_name", "atom_index"])


def attach_atom_features(df, atom_idx_col, suffix):
    df = df.join(struct_idx, on=["molecule_name", atom_idx_col])
    df = df.rename(
        columns={
            "atom": f"atom{suffix}",
            "x": f"x{suffix}",
            "y": f"y{suffix}",
            "z": f"z{suffix}",
        }
    )
    df = df.drop(columns=[atom_idx_col])
    return df


train_merged = attach_atom_features(train_df, "atom_index_0", "_0")
train_merged = attach_atom_features(train_merged, "atom_index_1", "_1")

test_merged = attach_atom_features(test_df, "atom_index_0", "_0")
test_merged = attach_atom_features(test_merged, "atom_index_1", "_1")

train_merged["distance"] = np.sqrt(
    (train_merged["x_0"] - train_merged["x_1"]) ** 2
    + (train_merged["y_0"] - train_merged["y_1"]) ** 2
    + (train_merged["z_0"] - train_merged["z_1"]) ** 2
)
test_merged["distance"] = np.sqrt(
    (test_merged["x_0"] - test_merged["x_1"]) ** 2
    + (test_merged["y_0"] - test_merged["y_1"]) ** 2
    + (test_merged["z_0"] - test_merged["z_1"]) ** 2
)

atom_number = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
}
for suf in ["_0", "_1"]:
    train_merged[f"atom{suf}_num"] = (
        train_merged[f"atom{suf}"].astype(str).map(atom_number).fillna(0).astype("int8")
    )
    test_merged[f"atom{suf}_num"] = (
        test_merged[f"atom{suf}"].astype(str).map(atom_number).fillna(0).astype("int8")
    )

type_mean = train_df.groupby("type")["scalar_coupling_constant"].mean()
train_merged["type_mean"] = train_merged["type"].astype(str).map(type_mean)
test_merged["type_mean"] = (
    test_merged["type"].astype(str).map(type_mean).fillna(type_mean.mean())
)

train_merged = train_merged.merge(dipole_df, on="molecule_name", how="left")
test_merged = test_merged.merge(dipole_df, on="molecule_name", how="left")
train_merged = train_merged.merge(energy_df, on="molecule_name", how="left")
test_merged = test_merged.merge(energy_df, on="molecule_name", how="left")

cat_cols = ["type", "atom_0", "atom_1"]
train_cat = pd.get_dummies(train_merged[cat_cols], drop_first=True, dtype=np.uint8)
test_cat = pd.get_dummies(test_merged[cat_cols], drop_first=True, dtype=np.uint8)
train_cat, test_cat = train_cat.align(test_cat, join="left", axis=1, fill_value=0)

numeric_feats = [
    "distance",
    "atom_0_num",
    "atom_1_num",
    "type_mean",
    "X",
    "Y",
    "Z",
    "potential_energy",
]
X_train = pd.concat([train_cat, train_merged[numeric_feats]], axis=1).fillna(0)
y_train = train_merged["scalar_coupling_constant"]
X_test = pd.concat([test_cat, test_merged[numeric_feats]], axis=1).fillna(0)

print("Feature matrix shapes – train:", X_train.shape, "test:", X_test.shape)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=1,
    max_samples=0.5,
    n_jobs=5,
    random_state=42,
)

model.fit(X_tr, y_tr)

test_predictions = model.predict(X_test)
test_df["scalar_coupling_constant"] = test_predictions

print("Test predictions preview:")
print(test_df[["id", "scalar_coupling_constant"]].head())



## === cell 1
submission = pd.DataFrame(
    {
        "id": test_df["id"],
        "scalar_coupling_constant": test_df["scalar_coupling_constant"],
    }
)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")



## === cell 2
print("Submission head:")
print(submission.head())
