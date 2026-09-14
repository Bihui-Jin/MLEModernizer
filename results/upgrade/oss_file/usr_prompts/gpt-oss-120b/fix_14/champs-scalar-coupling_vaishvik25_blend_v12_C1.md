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
import os, pandas as pd, numpy as np

print("Root input directory contents:", os.listdir("../input"))

train_path = os.path.join("..", "input", "champs-scalar-coupling", "train.csv")
test_path = os.path.join("..", "input", "champs-scalar-coupling", "test.csv")
structures_path = os.path.join(
    "..", "input", "champs-scalar-coupling", "structures.csv"
)

train = pd.read_csv(
    train_path,
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
        "scalar_coupling_constant": np.float32,
    },
)
test = pd.read_csv(
    test_path,
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
    },
)
structures = pd.read_csv(
    structures_path,
    usecols=["molecule_name", "atom_index", "atom"],
    dtype={"molecule_name": "category", "atom_index": np.int16, "atom": "category"},
)



## === cell 1
train = train.merge(
    structures.rename(columns={"atom_index": "atom_index_0", "atom": "atom_0"}),
    on=["molecule_name", "atom_index_0"],
    how="left",
)
train = train.merge(
    structures.rename(columns={"atom_index": "atom_index_1", "atom": "atom_1"}),
    on=["molecule_name", "atom_index_1"],
    how="left",
)

test = test.merge(
    structures.rename(columns={"atom_index": "atom_index_0", "atom": "atom_0"}),
    on=["molecule_name", "atom_index_0"],
    how="left",
)
test = test.merge(
    structures.rename(columns={"atom_index": "atom_index_1", "atom": "atom_1"}),
    on=["molecule_name", "atom_index_1"],
    how="left",
)


def make_key(df):
    a0 = df["atom_0"].astype(str)
    a1 = df["atom_1"].astype(str)
    atoms = np.where(a0 <= a1, a0 + "_" + a1, a1 + "_" + a0)
    return df["type"].astype(str) + "_" + atoms


train["type_atom_key"] = make_key(train)
test["type_atom_key"] = make_key(test)



## === cell 2
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
type_atom1_median = (
    train.groupby(["type", "atom_1"])["scalar_coupling_constant"]
    .median()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "pred_type_atom1"})
)

test_pred = test_pred.merge(type_atom0_median, on=["type", "atom_0"], how="left")
test_pred = test_pred.merge(type_atom1_median, on=["type", "atom_1"], how="left")

atom_median_cols = ["pred_type_atom0", "pred_type_atom1"]
test_pred["pred_atom_median"] = test_pred[atom_median_cols].mean(axis=1)

type_median = (
    train.groupby("type")["scalar_coupling_constant"]
    .median()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "pred_type"})
)

test_pred = test_pred.merge(type_median, on="type", how="left")

global_median = train["scalar_coupling_constant"].median()

test_pred["pred_key"] = test_pred["pred_key"].fillna(test_pred["pred_atom_median"])
test_pred["pred_key"] = test_pred["pred_key"].fillna(test_pred["pred_type"])
test_pred["pred_key"] = test_pred["pred_key"].fillna(global_median)



## === cell 3
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

cat_cols = ["type", "atom_0", "atom_1", "type_atom_key"]
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

X_all = train[cat_cols].astype(str)
enc.fit(X_all)
X_enc = enc.transform(X_all)

y = train["scalar_coupling_constant"].values

gbr = GradientBoostingRegressor(
    n_estimators=800,  # a bit stronger model
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
)
gbr.fit(X_enc, y)

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_enc, y, test_size=0.001, random_state=42
)
val_pred = gbr.predict(X_val)
val_mae = mean_absolute_error(y_val, val_pred)
val_log_mae = np.log(val_mae)
print(f"Quick validation MAE: {val_mae:.5f}, Log‑MAE: {val_log_mae:.5f}")

X_test = test[cat_cols].astype(str)
X_test_enc = enc.transform(X_test)
model_pred = gbr.predict(X_test_enc)

final_pred = 0.9 * model_pred + 0.1 * test_pred["pred_key"].values

submission = pd.DataFrame({"id": test["id"], "scalar_coupling_constant": final_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path} with {len(submission)} rows.")
