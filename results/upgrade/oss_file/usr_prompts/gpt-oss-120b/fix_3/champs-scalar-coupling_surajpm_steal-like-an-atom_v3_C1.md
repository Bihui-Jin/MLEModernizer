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

# 5. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

input_dir = "/kaggle/input"
comp_dir = os.path.join(input_dir, "champs-scalar-coupling")
if not os.path.isdir(comp_dir):
    comp_dir = os.path.join(input_dir, "data", "champs-scalar-coupling")
print("Using competition directory:", comp_dir)

ATOM_NUM = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
}


def atom_to_num(sym):
    return ATOM_NUM.get(sym, 0)




## === cell 1
train_path = os.path.join(comp_dir, "train.csv")
test_path = os.path.join(comp_dir, "test.csv")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)



## === cell 2
structures = pd.read_csv(os.path.join(comp_dir, "structures.csv"))
potential = pd.read_csv(os.path.join(comp_dir, "potential_energy.csv"))
dipole = pd.read_csv(os.path.join(comp_dir, "dipole_moments.csv"))
mulliken = pd.read_csv(os.path.join(comp_dir, "mulliken_charges.csv"))

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


def merge_atoms(df):
    df = df.merge(
        struct0[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    df = df.merge(
        struct1[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
    )
    return df


train_df = merge_atoms(train_df)
test_df = merge_atoms(test_df)


def compute_distance(df):
    return np.sqrt(
        (df["x0"] - df["x1"]) ** 2
        + (df["y0"] - df["y1"]) ** 2
        + (df["z0"] - df["z1"]) ** 2
    )


train_df["distance"] = compute_distance(train_df)
test_df["distance"] = compute_distance(test_df)

train_df["atom0_num"] = train_df["atom_0"].map(ATOM_NUM).fillna(0).astype(int)
train_df["atom1_num"] = train_df["atom_1"].map(ATOM_NUM).fillna(0).astype(int)
test_df["atom0_num"] = test_df["atom_0"].map(ATOM_NUM).fillna(0).astype(int)
test_df["atom1_num"] = test_df["atom_1"].map(ATOM_NUM).fillna(0).astype(int)

potential = potential.rename(columns={"potential_energy": "pot_energy"})
train_df = train_df.merge(potential, on="molecule_name", how="left")
test_df = test_df.merge(potential, on="molecule_name", how="left")

dipole["dipole_mag"] = np.sqrt(dipole["X"] ** 2 + dipole["Y"] ** 2 + dipole["Z"] ** 2)
dipole = dipole.rename(columns={"dipole_mag": "dipole_mag"})
train_df = train_df.merge(
    dipole[["molecule_name", "dipole_mag"]], on="molecule_name", how="left"
)
test_df = test_df.merge(
    dipole[["molecule_name", "dipole_mag"]], on="molecule_name", how="left"
)

mulliken = mulliken.rename(columns={"mulliken_charge": "mulliken_charge"})
charge0 = mulliken.rename(
    columns={"atom_index": "atom_index_0", "mulliken_charge": "charge0"}
)
charge1 = mulliken.rename(
    columns={"atom_index": "atom_index_1", "mulliken_charge": "charge1"}
)
train_df = train_df.merge(
    charge0[["molecule_name", "atom_index_0", "charge0"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
)
train_df = train_df.merge(
    charge1[["molecule_name", "atom_index_1", "charge1"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
)
test_df = test_df.merge(
    charge0[["molecule_name", "atom_index_0", "charge0"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
)
test_df = test_df.merge(
    charge1[["molecule_name", "atom_index_1", "charge1"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
)

train_df["type_code"], type_mapping = pd.factorize(train_df["type"])
test_df["type_code"] = test_df["type"].map(
    {lbl: idx for idx, lbl in enumerate(type_mapping)}
)

feature_cols = [
    "distance",
    "atom0_num",
    "atom1_num",
    "pot_energy",
    "dipole_mag",
    "charge0",
    "charge1",
    "type_code",
]
train_X = train_df[feature_cols].fillna(-1)
test_X = test_df[feature_cols].fillna(-1)
train_y = train_df["scalar_coupling_constant"]



## === cell 3
X_tr, X_val, y_tr, y_val = train_test_split(
    train_X, train_y, test_size=0.2, random_state=42
)
model = GradientBoostingRegressor(
    n_estimators=250, learning_rate=0.05, max_depth=4, random_state=42
)
model.fit(X_tr, y_tr)

val_pred = model.predict(X_val)
val_mae = mean_absolute_error(y_val, val_pred)
print(f"Validation MAE: {val_mae:.5f}, log‑MAE: {np.log(val_mae):.5f}")



## === cell 4
model.fit(train_X, train_y)
test_pred = model.predict(test_X)

submission = pd.DataFrame({"id": test_df["id"], "scalar_coupling_constant": test_pred})
output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}, shape {submission.shape}")
