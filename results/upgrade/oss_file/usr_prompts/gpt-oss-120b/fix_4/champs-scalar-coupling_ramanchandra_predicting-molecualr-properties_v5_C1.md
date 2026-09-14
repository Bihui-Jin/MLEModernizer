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

# 5. Target score

2.20425

# 6. Current score

2.55537

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 2.50959) has done: 'I add a simple imputation step to replace any NaN values that appear after merging and one‑hot encoding, because LinearRegression cannot handle missing values. By filling NaNs with 0 in both the training and test feature matrices (right after dropping the unused `molecule_name` column), the model can fit and predict without errors, allowing a valid CSV submission to be generated.'
- What this solution (achieved 2.55537) has done: 'I add a few physically‑relevant numeric features (potential energy, dipole moment components, and per‑atom Mulliken charges) to the training and test sets before one‑hot encoding. These extra columns are merged directly after the atom‑coordinate mapping, then included in the attribute list used for the LinearRegression model. The added information should help the linear model explain more variance, decreasing the validation MAE log and moving the score from 2.50959 closer to the target 2.20425 while keeping the overall pipeline unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import warnings, os

warnings.filterwarnings("ignore")
print(os.listdir("../input"))




## === cell 1
pot_energy = pd.read_csv("../input/potential_energy.csv")
mulliken_charges = pd.read_csv("../input/mulliken_charges.csv")
train_df = pd.read_csv("../input/train.csv")
scalar_coupling_cont = pd.read_csv("../input/scalar_coupling_contributions.csv")
test_df = pd.read_csv("../input/test.csv")
magnetic_shield_tensor = pd.read_csv("../input/magnetic_shielding_tensors.csv")
dipole_moment = pd.read_csv("../input/dipole_moments.csv")
structures = pd.read_csv("../input/structures.csv")




## === cell 2
def map_atom_data(df, atom_idx):
    df = pd.merge(
        df,
        structures,
        how="left",
        left_on=["molecule_name", f"atom_index_{atom_idx}"],
        right_on=["molecule_name", "atom_index"],
    )
    df = df.drop("atom_index", axis=1)
    df = df.rename(
        columns={
            "atom": f"atom_{atom_idx}",
            "x": f"x_{atom_idx}",
            "y": f"y_{atom_idx}",
            "z": f"z_{atom_idx}",
        }
    )
    return df


train_df = map_atom_data(train_df, 0)
train_df = map_atom_data(train_df, 1)
test_df = map_atom_data(test_df, 0)
test_df = map_atom_data(test_df, 1)

train_df = train_df.merge(pot_energy, on="molecule_name", how="left")
test_df = test_df.merge(pot_energy, on="molecule_name", how="left")

dipole_renamed = dipole_moment.rename(
    columns={"X": "dipole_X", "Y": "dipole_Y", "Z": "dipole_Z"}
)
train_df = train_df.merge(dipole_renamed, on="molecule_name", how="left")
test_df = test_df.merge(dipole_renamed, on="molecule_name", how="left")

charges_0 = mulliken_charges.rename(
    columns={"atom_index": "atom_index_0", "mulliken_charge": "charge_0"}
)
charges_1 = mulliken_charges.rename(
    columns={"atom_index": "atom_index_1", "mulliken_charge": "charge_1"}
)
train_df = train_df.merge(
    charges_0[["molecule_name", "atom_index_0", "charge_0"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
)
train_df = train_df.merge(
    charges_1[["molecule_name", "atom_index_1", "charge_1"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
)
test_df = test_df.merge(
    charges_0[["molecule_name", "atom_index_0", "charge_0"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
)
test_df = test_df.merge(
    charges_1[["molecule_name", "atom_index_1", "charge_1"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
)




## === cell 3
train_m_0 = train_df[["x_0", "y_0", "z_0"]].values
train_m_1 = train_df[["x_1", "y_1", "z_1"]].values
test_m_0 = test_df[["x_0", "y_0", "z_0"]].values
test_m_1 = test_df[["x_1", "y_1", "z_1"]].values  # corrected to use atom 1 coordinates

train_df["dist_vector"] = np.linalg.norm(train_m_0 - train_m_1, axis=1)
test_df["dist_vector"] = np.linalg.norm(test_m_0 - test_m_1, axis=1)




## === cell 4
train_df["type"] = train_df.type.astype("category")
train_df["atom_0"] = train_df.atom_0.astype("category")
train_df["atom_1"] = train_df.atom_1.astype("category")

test_df["type"] = test_df.type.astype("category")
test_df["atom_0"] = test_df.atom_0.astype("category")
test_df["atom_1"] = test_df.atom_1.astype("category")




## === cell 5
Attributes = [
    "id",
    "molecule_name",
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
    "potential_energy",
    "dipole_X",
    "dipole_Y",
    "dipole_Z",
    "charge_0",
    "charge_1",
]
cat_attributes = ["type", "atom_0", "atom_1"]
target_label = ["scalar_coupling_constant"]

X_train = train_df[Attributes]
X_test = test_df[Attributes]
y_target = train_df[target_label]




## === cell 6
print("Train / test shapes before encoding:", X_train.shape, X_test.shape)




## === cell 7
X_train = pd.get_dummies(X_train, columns=cat_attributes)
X_test = pd.get_dummies(X_test, columns=cat_attributes)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print("Shape after one‑hot encoding and alignment:")
print("X_train:", X_train.shape, "X_test:", X_test.shape)




## === cell 8
X_train = X_train.drop("molecule_name", axis=1).fillna(0)
X_test = X_test.drop("molecule_name", axis=1).fillna(0)




## === cell 9
from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()
lr_model = linear_reg.fit(X_train, y_target.values.ravel())
train_score = np.round(lr_model.score(X_train, y_target), 3)
print("R² on training data:", train_score)




## === cell 10
y_pred = lr_model.predict(X_test)

submission = pd.read_csv("../input/sample_submission.csv")
submission["scalar_coupling_constant"] = y_pred
submission_path = "Linear_Regression_model.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
