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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import warnings

warnings.filterwarnings("ignore")
import random

random.seed(42)
np.random.seed(42)

import os

BASE_INPUT = "../input"
if not os.path.exists(BASE_INPUT):
    if os.path.exists("/kaggle/data/champs-scalar-coupling"):
        BASE_INPUT = "/kaggle/data/champs-scalar-coupling"
    elif os.path.exists("/kaggle/input/champs-scalar-coupling"):
        BASE_INPUT = "/kaggle/input/champs-scalar-coupling"
    elif os.path.exists("/kaggle/data/input"):
        BASE_INPUT = "/kaggle/data/input"
    else:
        BASE_INPUT = "/kaggle/data/champs-scalar-coupling"

print("Using BASE_INPUT:", BASE_INPUT)
if os.path.exists("../input"):
    print(os.listdir("../input")[:20])



## === cell 1
usecols_train = [
    "id",
    "molecule_name",
    "atom_index_0",
    "atom_index_1",
    "type",
    "scalar_coupling_constant",
]
usecols_test = ["id", "molecule_name", "atom_index_0", "atom_index_1", "type"]
usecols_struct = ["molecule_name", "atom_index", "atom", "x", "y", "z"]

train_df = pd.read_csv(f"{BASE_INPUT}/train.csv", usecols=usecols_train)
test_df = pd.read_csv(f"{BASE_INPUT}/test.csv", usecols=usecols_test)
structures = pd.read_csv(f"{BASE_INPUT}/structures.csv", usecols=usecols_struct)

SCC_sample_path = f"{BASE_INPUT}/sample_submission.csv"




## === cell 2
print("Shape of train dataset:", train_df.shape)
print("Shape of test dataset:", test_df.shape)
print("Shape of structures dataset:", structures.shape)



## === cell 3
print("Train dtypes:\n", train_df.dtypes)
print("Test dtypes:\n", test_df.dtypes)
print("Structures dtypes:\n", structures.dtypes)



## === cell 4
structures_idx = structures.set_index(
    ["molecule_name", "atom_index"], drop=False
).sort_index()


def map_atom_data(df, atom_idx):
    keys = pd.MultiIndex.from_frame(df[["molecule_name", f"atom_index_{atom_idx}"]])
    looked = structures_idx.reindex(keys)[["atom", "x", "y", "z"]]
    looked.columns = [
        f"atom_{atom_idx}",
        f"x_{atom_idx}",
        f"y_{atom_idx}",
        f"z_{atom_idx}",
    ]
    return pd.concat([df.reset_index(drop=True), looked.reset_index(drop=True)], axis=1)


train_df = map_atom_data(train_df, 0)
train_df = map_atom_data(train_df, 1)
test_df = map_atom_data(test_df, 0)
test_df = map_atom_data(test_df, 1)



## === cell 5
train_m_0 = train_df[["x_0", "y_0", "z_0"]].to_numpy()
train_m_1 = train_df[["x_1", "y_1", "z_1"]].to_numpy()
test_m_0 = test_df[["x_0", "y_0", "z_0"]].to_numpy()
test_m_1 = test_df[["x_1", "y_1", "z_1"]].to_numpy()

train_df["dist_vector"] = np.linalg.norm(train_m_0 - train_m_1, axis=1)
train_df["dist_X"] = (train_df["x_0"] - train_df["x_1"]) ** 2
train_df["dist_Y"] = (train_df["y_0"] - train_df["y_1"]) ** 2
train_df["dist_Z"] = (train_df["z_0"] - train_df["z_1"]) ** 2

test_df["dist_vector"] = np.linalg.norm(test_m_0 - test_m_1, axis=1)
test_df["dist_X"] = (test_df["x_0"] - test_df["x_1"]) ** 2
test_df["dist_Y"] = (test_df["y_0"] - test_df["y_1"]) ** 2
test_df["dist_Z"] = (test_df["z_0"] - test_df["z_1"]) ** 2



## === cell 6
train_df = train_df.drop(columns=["molecule_name"], axis=1)
train_df.head(6)



## === cell 7
test_df = test_df.drop(columns=["molecule_name"], axis=1)
test_df.head(10)



## === cell 8
train_df["type"] = train_df.type.astype("category")
train_df["atom_0"] = train_df.atom_0.astype("category")
train_df["atom_1"] = train_df.atom_1.astype("category")

test_df["type"] = test_df.type.astype("category")
test_df["atom_0"] = test_df.atom_0.astype("category")
test_df["atom_1"] = test_df.atom_1.astype("category")



## === cell 9
threshold = 0.95

candidate_numeric = [
    "atom_index_0",
    "atom_index_1",
    "x_0",
    "y_0",
    "z_0",
    "x_1",
    "y_1",
    "z_1",
    "dist_vector",
    "dist_X",
    "dist_Y",
    "dist_Z",
]
candidate_numeric = [c for c in candidate_numeric if c in train_df.columns]

corr_matrix = train_df[candidate_numeric].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))



## === cell 10
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
print("There are are %d columns to remove." % (len(to_drop)))



## === cell 11
if len(to_drop) > 0:
    train_df = train_df.drop(columns=to_drop)
    test_df = test_df.drop(columns=[c for c in to_drop if c in test_df.columns])



## === cell 12
Attributes = [
    "id",
    "atom_index_0",
    "atom_index_1",
    "type",
    "x_0",
    "y_0",
    "z_0",
    "atom_0",
    "atom_1",
    "x_1",
    "y_1",
    "z_1",
    "dist_vector",
    "dist_X",
    "dist_Y",
    "dist_Z",
]
cat_attributes = ["type", "atom_0", "atom_1"]
target_label = ["scalar_coupling_constant"]

Attributes = [
    c
    for c in Attributes
    if c in train_df.columns or c in test_df.columns or c in target_label
]
X_train = train_df[Attributes].copy()
X_test = test_df[Attributes].copy()
y_target = train_df[target_label].copy()



## === cell 13
X_train = pd.get_dummies(X_train, columns=cat_attributes)
X_test = pd.get_dummies(X_test, columns=cat_attributes)

X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)



## === cell 14
print(X_train.shape, X_test.shape)



## === cell 15
print(y_target.shape)



## === cell 16
X_train.head(6)



## === cell 17
X_test.head(6)



## === cell 18
from sklearn import linear_model

linear_reg = linear_model.Lasso(alpha=0.0001, random_state=42, max_iter=10000)
lasso_model = linear_reg.fit(X_train, y_target.values.ravel())
score = np.round(lasso_model.score(X_train, y_target.values.ravel()), 3)
print("Accuracy of trained model:", score)



## === cell 19
y_pred = lasso_model.predict(X_test)

SCC = pd.read_csv(SCC_sample_path)
SCC["scalar_coupling_constant"] = np.asarray(y_pred).reshape(-1)

out_path = "Lasso_Regression_model.csv"
SCC.to_csv(out_path, index=False)
print("Wrote submission:", out_path, "shape:", SCC.shape)
print(SCC.head())
