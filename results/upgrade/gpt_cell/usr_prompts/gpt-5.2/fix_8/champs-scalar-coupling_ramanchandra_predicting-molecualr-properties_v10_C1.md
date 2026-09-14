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
import warnings
import os

warnings.filterwarnings("ignore")

INPUT_DIR = "../input"

np.random.seed(42)



## === cell 1
train_df = pd.read_csv(
    f"{INPUT_DIR}/train.csv",
    usecols=[
        "id",
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "scalar_coupling_constant",
    ],
    dtype={
        "id": "int32",
        "molecule_name": "string",
        "atom_index_0": "int16",
        "atom_index_1": "int16",
        "type": "category",
        "scalar_coupling_constant": "float32",
    },
)
test_df = pd.read_csv(
    f"{INPUT_DIR}/test.csv",
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
    dtype={
        "id": "int32",
        "molecule_name": "string",
        "atom_index_0": "int16",
        "atom_index_1": "int16",
        "type": "category",
    },
)
structures = pd.read_csv(
    f"{INPUT_DIR}/structures.csv",
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
    dtype={
        "molecule_name": "string",
        "atom_index": "int16",
        "atom": "category",
        "x": "float32",
        "y": "float32",
        "z": "float32",
    },
)



## === cell 2
print("Shape of train dataset:", train_df.shape)
print("Shape of test dataset:", test_df.shape)
print("Shape of structures dataset:", structures.shape)



## === cell 3
structures_idx = structures.set_index(["molecule_name", "atom_index"]).sort_index()


def map_atom_data(df, atom_idx):
    key = pd.MultiIndex.from_arrays(
        [df["molecule_name"].to_numpy(), df[f"atom_index_{atom_idx}"].to_numpy()],
        names=["molecule_name", "atom_index"],
    )

    joined = (
        structures_idx.reindex(key)[["atom", "x", "y", "z"]]
        .reset_index(drop=True)
        .rename(
            columns={
                "atom": f"atom_{atom_idx}",
                "x": f"x_{atom_idx}",
                "y": f"y_{atom_idx}",
                "z": f"z_{atom_idx}",
            }
        )
    )
    return pd.concat([df.reset_index(drop=True), joined], axis=1)


train_df = map_atom_data(train_df, 0)
train_df = map_atom_data(train_df, 1)
test_df = map_atom_data(test_df, 0)
test_df = map_atom_data(test_df, 1)



## === cell 4
train_m_0 = train_df[["x_0", "y_0", "z_0"]].to_numpy(dtype=np.float32, copy=False)
train_m_1 = train_df[["x_1", "y_1", "z_1"]].to_numpy(dtype=np.float32, copy=False)
test_m_0 = test_df[["x_0", "y_0", "z_0"]].to_numpy(dtype=np.float32, copy=False)
test_m_1 = test_df[["x_1", "y_1", "z_1"]].to_numpy(dtype=np.float32, copy=False)

train_df["dist_vector"] = np.linalg.norm(train_m_0 - train_m_1, axis=1).astype(
    np.float32
)
test_df["dist_vector"] = np.linalg.norm(test_m_0 - test_m_1, axis=1).astype(np.float32)



## === cell 5
train_df["atom_0"] = train_df["atom_0"].astype("category")
train_df["atom_1"] = train_df["atom_1"].astype("category")
test_df["atom_0"] = test_df["atom_0"].astype("category")
test_df["atom_1"] = test_df["atom_1"].astype("category")



## === cell 6
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
]
cat_attributes = ["type", "atom_0", "atom_1"]
target_label = ["scalar_coupling_constant"]

X_train = train_df[Attributes]
X_test = test_df[Attributes]
y_target = train_df[target_label]

print(X_train.shape, X_test.shape)
print(y_target.shape)



## === cell 7
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error


def build_features_fast(X_tr_df: pd.DataFrame, X_te_df: pd.DataFrame, cat_cols):
    X_all = pd.concat([X_tr_df, X_te_df], axis=0, ignore_index=True, copy=False)

    base_cols = [
        "id",
        "atom_index_0",
        "atom_index_1",
        "x_0",
        "y_0",
        "z_0",
        "x_1",
        "y_1",
        "z_1",
        "dist_vector",
    ]
    base = X_all[base_cols].to_numpy(dtype=np.float32, copy=False)

    one_hots = []
    for c in cat_cols:
        if not pd.api.types.is_categorical_dtype(X_all[c]):
            X_all[c] = X_all[c].astype("category")
        codes = X_all[c].cat.codes.to_numpy(copy=False)  # -1 shouldn't happen here
        n_cat = len(X_all[c].cat.categories)
        oh = np.zeros((len(X_all), n_cat), dtype=np.float32)
        oh[np.arange(len(X_all)), codes] = 1.0
        one_hots.append(oh)

    X_np = np.concatenate([base] + one_hots, axis=1)
    n_train = len(X_tr_df)
    return X_np[:n_train], X_np[n_train:]


X_train_np, X_test_np = build_features_fast(X_train, X_test, cat_attributes)
y_np = y_target.values.ravel().astype(np.float32, copy=False)

print("shape of transformed train matrix:", X_train_np.shape)
print("shape of transformed test matrix:", X_test_np.shape)

groups = train_df["molecule_name"].to_numpy()
gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
tr_idx, va_idx = next(gss.split(X_train_np, y_np, groups=groups))


def clean_and_cast_np(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32, order="C")
    np.nan_to_num(a, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return a


X_tr = clean_and_cast_np(X_train_np[tr_idx])
y_tr = y_np[tr_idx]
X_va = clean_and_cast_np(X_train_np[va_idx])
y_va = y_np[va_idx]
X_test_clean = clean_and_cast_np(X_test_np)

rf = RandomForestRegressor(
    n_estimators=40,  # unchanged from provided code
    bootstrap=True,
    max_depth=30,
    max_features=1.0,
    min_samples_leaf=3,
    min_samples_split=6,
    random_state=42,
    n_jobs=-1,
)

rf.fit(X_tr, y_tr)

va_pred = rf.predict(X_va)
va_mae = mean_absolute_error(y_va, va_pred)
print("Holdout (by molecule) MAE:", np.round(va_mae, 6))

y_pred = rf.predict(X_test_clean)

SCC = pd.read_csv(f"{INPUT_DIR}/sample_submission.csv")
SCC["scalar_coupling_constant"] = y_pred
SCC.to_csv("Random_Forest_Regression_model.csv", index=False)
print(
    "Wrote submission:", "Random_Forest_Regression_model.csv", "with shape", SCC.shape
)
