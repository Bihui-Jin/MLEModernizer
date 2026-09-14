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

# 5. Target score

-1.7766920641782642

# 6. Current score

1.47643

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.14343) has done: 'I replace the missing‑file ensemble code with a self‑contained baseline that reads the original training and test CSVs, encodes the categorical columns, trains a lightweight GradientBoostingRegressor on a sampled subset of the data, and writes the predictions to a proper `submission.csv`. This fixes the FileNotFound and KeyError failures, guarantees a correctly‑named output file, and provides a reasonable baseline that moves the score toward the target without altering any core competition logic.'
- What this solution (achieved 1.1364) has done: 'Implemented faster data handling and switched to the histogram‑based Gradient Boosting model, which is orders of magnitude quicker on large tabular data while keeping the same overall modeling approach and deterministic behavior.'
- What this solution (achieved 1.40414) has done: 'I add a simple geometric feature – the Euclidean distance between the two atoms of each coupling – by merging the structures data into the train and test frames. This distance often correlates with coupling strength, so it should lower the MAE (and thus the log‑MAE score) toward the target. I also slightly increase the number of boosting iterations to give the model more capacity while keeping the original algorithm unchanged.'
- What this solution (achieved 1.39858) has done: 'I add additional atomic element features (the element symbol of each atom) and molecule‑level descriptors (dipole moment components and potential energy) to give the model more useful information, encode the new categorical columns, and modestly increase the number of boosting iterations. These changes keep the same modeling pipeline while providing richer features that should lower the log‑MAE toward the target score.'
- What this solution (achieved 1.42853) has done: 'I add two small feature enhancements that are inexpensive but should improve the model’s predictive power, moving the log‑MAE closer to the target. First, I create an inverse‑distance feature (1 / distance) which often captures coupling strength better than raw distance. Second, I merge per‑atom Mulliken charges and magnetic shielding tensor components (both atom‑level descriptors) into the train and test tables, providing richer physical information without changing the overall modeling pipeline. These additions keep the existing architecture and training unchanged while giving the model more relevant signals to lower the error.'
- What this solution (achieved 1.44639) has done: 'I add a simple squared‑distance feature (which often captures interaction strength better than raw distance) and slightly increase the model capacity by raising the number of boosting iterations and lowering the learning rate. These minimal changes keep the overall pipeline unchanged while giving the regressor a bit more expressive power, which should reduce the log‑MAE and move the score closer to the target (lower is better).'
- What this solution (achieved 1.47257) has done: 'I add a few inexpensive numeric features (atomic numbers, charge sums/differences, and a distance‑charge interaction) that give the model more physical information, remove the down‑sampling so the model trains on the full dataset, and slightly increase the boosting iterations. These changes keep the existing pipeline and model type while expectedly lowering the log‑MAE toward the target score.'
- What this solution (achieved 1.47643) has done: 'I move the charge‑based feature creation to after the Mulliken charge tables are merged (so the required columns exist) and remove those references from the earlier cell that caused the KeyError. This fixes the runtime error while keeping the modeling pipeline unchanged, ensuring a valid submission CSV is produced.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

DATA_ROOT = "../input/champs-scalar-coupling"
TRAIN_PATH = f"{DATA_ROOT}/train.csv"
TEST_PATH = f"{DATA_ROOT}/test.csv"
SAMPLE_SUB_PATH = f"{DATA_ROOT}/sample_submission.csv"




## === cell 1
train_cols = [
    "id",
    "molecule_name",
    "atom_index_0",
    "atom_index_1",
    "type",
    "scalar_coupling_constant",
]
test_cols = ["id", "molecule_name", "atom_index_0", "atom_index_1", "type"]

dtype_train = {
    "id": "int32",
    "molecule_name": "object",
    "atom_index_0": "int16",
    "atom_index_1": "int16",
    "type": "object",
    "scalar_coupling_constant": "float32",
}
dtype_test = {
    "id": "int32",
    "molecule_name": "object",
    "atom_index_0": "int16",
    "atom_index_1": "int16",
    "type": "object",
}

train = pd.read_csv(TRAIN_PATH, usecols=train_cols, dtype=dtype_train)
test = pd.read_csv(TEST_PATH, usecols=test_cols, dtype=dtype_test)

test_ids = test["id"].copy()




## === cell 2
struct_path = f"{DATA_ROOT}/structures.csv"
struct_cols = ["molecule_name", "atom_index", "atom", "x", "y", "z"]
dtype_struct = {
    "molecule_name": "object",
    "atom_index": "int16",
    "atom": "object",
    "x": "float32",
    "y": "float32",
    "z": "float32",
}
structures = pd.read_csv(struct_path, usecols=struct_cols, dtype=dtype_struct)

s0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
    }
)
s1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
    }
)

train = train.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
train = train.merge(s1, on=["molecule_name", "atom_index_1"], how="left")
test = test.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
test = test.merge(s1, on=["molecule_name", "atom_index_1"], how="left")

train["distance"] = np.sqrt(
    (train["x0"] - train["x1"]) ** 2
    + (train["y0"] - train["y1"]) ** 2
    + (train["z0"] - train["z1"]) ** 2
)
test["distance"] = np.sqrt(
    (test["x0"] - test["x1"]) ** 2
    + (test["y0"] - test["y1"]) ** 2
    + (test["z0"] - test["z1"]) ** 2
)

train["distance_sq"] = train["distance"] ** 2
test["distance_sq"] = test["distance"] ** 2

train["inv_distance"] = 1.0 / (train["distance"] + 1e-6)
test["inv_distance"] = 1.0 / (test["distance"] + 1e-6)

atom_num_map = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
}
train["atom0_num"] = train["atom0"].map(atom_num_map).fillna(0).astype("int8")
test["atom0_num"] = test["atom0"].map(atom_num_map).fillna(0).astype("int8")
train["atom1_num"] = train["atom1"].map(atom_num_map).fillna(0).astype("int8")
test["atom1_num"] = test["atom1"].map(atom_num_map).fillna(0).astype("int8")

coord_cols = ["x0", "y0", "z0", "x1", "y1", "z1"]
train.drop(columns=coord_cols, inplace=True)
test.drop(columns=coord_cols, inplace=True)




## === cell 3
dipole_path = f"{DATA_ROOT}/dipole_moments.csv"
pot_path = f"{DATA_ROOT}/potential_energy.csv"
mulliken_path = f"{DATA_ROOT}/mulliken_charges.csv"
shield_path = f"{DATA_ROOT}/magnetic_shielding_tensors.csv"

dipole = pd.read_csv(
    dipole_path,
    dtype={"molecule_name": "object", "X": "float32", "Y": "float32", "Z": "float32"},
)
pot = pd.read_csv(
    pot_path, dtype={"molecule_name": "object", "potential_energy": "float32"}
)

mulliken = pd.read_csv(
    mulliken_path,
    dtype={
        "molecule_name": "object",
        "atom_index": "int16",
        "mulliken_charge": "float32",
    },
)

shield = pd.read_csv(
    shield_path,
    dtype={
        "molecule_name": "object",
        "atom_index": "int16",
        "XX": "float32",
        "YX": "float32",
        "ZX": "float32",
        "XY": "float32",
        "YY": "float32",
        "ZY": "float32",
        "XZ": "float32",
        "YZ": "float32",
        "ZZ": "float32",
    },
)

train = train.merge(dipole, on="molecule_name", how="left")
test = test.merge(dipole, on="molecule_name", how="left")
train = train.merge(pot, on="molecule_name", how="left")
test = test.merge(pot, on="molecule_name", how="left")

m0 = mulliken.rename(
    columns={"atom_index": "atom_index_0", "mulliken_charge": "mulliken_charge_0"}
)
m1 = mulliken.rename(
    columns={"atom_index": "atom_index_1", "mulliken_charge": "mulliken_charge_1"}
)
train = train.merge(m0, on=["molecule_name", "atom_index_0"], how="left")
train = train.merge(m1, on=["molecule_name", "atom_index_1"], how="left")
test = test.merge(m0, on=["molecule_name", "atom_index_0"], how="left")
test = test.merge(m1, on=["molecule_name", "atom_index_1"], how="left")

train["charge_sum"] = train["mulliken_charge_0"] + train["mulliken_charge_1"]
test["charge_sum"] = test["mulliken_charge_0"] + test["mulliken_charge_1"]
train["charge_diff"] = train["mulliken_charge_0"] - train["mulliken_charge_1"]
test["charge_diff"] = test["mulliken_charge_0"] - test["mulliken_charge_1"]
train["dist_charge_sum"] = train["distance"] * train["charge_sum"]
test["dist_charge_sum"] = test["distance"] * test["charge_sum"]

s0t = shield.rename(
    columns={
        c: f"{c}_0" for c in shield.columns if c not in ["molecule_name", "atom_index"]
    }
)
s0t = s0t.rename(columns={"atom_index": "atom_index_0"})
s1t = shield.rename(
    columns={
        c: f"{c}_1" for c in shield.columns if c not in ["molecule_name", "atom_index"]
    }
)
s1t = s1t.rename(columns={"atom_index": "atom_index_1"})

train = train.merge(s0t, on=["molecule_name", "atom_index_0"], how="left")
train = train.merge(s1t, on=["molecule_name", "atom_index_1"], how="left")
test = test.merge(s0t, on=["molecule_name", "atom_index_0"], how="left")
test = test.merge(s1t, on=["molecule_name", "atom_index_1"], how="left")




## === cell 4
cat_cols = ["type", "molecule_name", "atom0", "atom1"]
for col in cat_cols:
    combined = pd.concat([train[col], test[col]], ignore_index=True)
    codes, _ = pd.factorize(combined, sort=True)
    train[col] = codes[: len(train)]
    test[col] = codes[len(train) :]

train[cat_cols] = train[cat_cols].astype("int32")
test[cat_cols] = test[cat_cols].astype("int32")

target_col = "scalar_coupling_constant"
feature_cols = [c for c in train.columns if c not in ["id", target_col]]

X = train[feature_cols].values
y = train[target_col].values




## === cell 5
gbr = HistGradientBoostingRegressor(
    max_iter=2500,  # more boosting rounds
    learning_rate=0.035,  # slightly smaller step
    max_depth=6,
    random_state=42,
)

gbr.fit(X, y)




## === cell 6
test_features = test[feature_cols].values
test_pred = gbr.predict(test_features)

submission = pd.DataFrame({"id": test_ids, "scalar_coupling_constant": test_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")




## === cell 7
submission.head()
