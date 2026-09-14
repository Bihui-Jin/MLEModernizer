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

category_encoders==2.7.0
geopandas==0.14.4
lightgbm==4.6.0
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
import gc, os
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgbm

train = pd.read_csv(
    "../input/train.csv",
    dtype={
        "id": np.int32,
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "scalar_coupling_constant": np.float32,
    },
    usecols=[
        "id",
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "scalar_coupling_constant",
    ],
)
test = pd.read_csv(
    "../input/test.csv",
    dtype={"id": np.int32, "atom_index_0": np.int16, "atom_index_1": np.int16},
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
)
sample_sub = pd.read_csv("../input/sample_submission.csv")
structures = pd.read_csv(
    "../input/structures.csv",
    dtype={
        "molecule_name": "category",
        "atom_index": np.int16,
        "atom": "category",
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
    },
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
)

print(f"train.shape: {train.shape}")
print(f"test.shape: {test.shape}")



## === cell 1
y_train = train["scalar_coupling_constant"].copy()
X_train = train.drop(columns=["scalar_coupling_constant", "id"]).copy()
X_test = test.drop(columns=["id"]).copy()



## === cell 2
structures_idx = structures.set_index(["molecule_name", "atom_index"])
X_train = X_train.join(
    structures_idx, on=["molecule_name", "atom_index_0"], rsuffix="_0", how="left"
)
X_test = X_test.join(
    structures_idx, on=["molecule_name", "atom_index_0"], rsuffix="_0", how="left"
)
X_train = X_train.rename(
    columns={
        "atom_index": "atom_index_0_tmp",
        "x": "atom_index_0_x",
        "y": "atom_index_0_y",
        "z": "atom_index_0_z",
        "atom": "atom_0",
    }
)
X_test = X_test.rename(
    columns={
        "atom_index": "atom_index_0_tmp",
        "x": "atom_index_0_x",
        "y": "atom_index_0_y",
        "z": "atom_index_0_z",
        "atom": "atom_0",
    }
)

X_train = X_train.join(
    structures_idx, on=["molecule_name", "atom_index_1"], rsuffix="_1", how="left"
)
X_test = X_test.join(
    structures_idx, on=["molecule_name", "atom_index_1"], rsuffix="_1", how="left"
)
X_train = X_train.rename(
    columns={
        "atom_index": "atom_index_1_tmp",
        "x": "atom_index_1_x",
        "y": "atom_index_1_y",
        "z": "atom_index_1_z",
        "atom": "atom_1",
    }
)
X_test = X_test.rename(
    columns={
        "atom_index": "atom_index_1_tmp",
        "x": "atom_index_1_x",
        "y": "atom_index_1_y",
        "z": "atom_index_1_z",
        "atom": "atom_1",
    }
)

del structures_idx
gc.collect()



## === cell 3
X_train = X_train.drop(
    columns=["atom_index_0_tmp", "atom_index_1_tmp"], errors="ignore"
)
X_test = X_test.drop(columns=["atom_index_0_tmp", "atom_index_1_tmp"], errors="ignore")



## === cell 4
coord_cols = [
    "atom_index_0_x",
    "atom_index_0_y",
    "atom_index_0_z",
    "atom_index_1_x",
    "atom_index_1_y",
    "atom_index_1_z",
]
for col in coord_cols:
    X_train[col] = X_train[col].astype(np.float32)
    X_test[col] = X_test[col].astype(np.float32)

X_train["distance"] = np.sqrt(
    (X_train["atom_index_0_x"] - X_train["atom_index_1_x"]) ** 2
    + (X_train["atom_index_0_y"] - X_train["atom_index_1_y"]) ** 2
    + (X_train["atom_index_0_z"] - X_train["atom_index_1_z"]) ** 2
)
X_test["distance"] = np.sqrt(
    (X_test["atom_index_0_x"] - X_test["atom_index_1_x"]) ** 2
    + (X_test["atom_index_0_y"] - X_test["atom_index_1_y"]) ** 2
    + (X_test["atom_index_0_z"] - X_test["atom_index_1_z"]) ** 2
)

dot = (
    X_train["atom_index_0_x"] * X_train["atom_index_1_x"]
    + X_train["atom_index_0_y"] * X_train["atom_index_1_y"]
    + X_train["atom_index_0_z"] * X_train["atom_index_1_z"]
)
mag0 = np.sqrt(
    X_train["atom_index_0_x"] ** 2
    + X_train["atom_index_0_y"] ** 2
    + X_train["atom_index_0_z"] ** 2
)
mag1 = np.sqrt(
    X_train["atom_index_1_x"] ** 2
    + X_train["atom_index_1_y"] ** 2
    + X_train["atom_index_1_z"] ** 2
)
cos_angle = np.clip(dot / (mag0 * mag1 + 1e-9), -1.0, 1.0)
X_train["angle"] = np.arccos(cos_angle).astype(np.float32)

dot = (
    X_test["atom_index_0_x"] * X_test["atom_index_1_x"]
    + X_test["atom_index_0_y"] * X_test["atom_index_1_y"]
    + X_test["atom_index_0_z"] * X_test["atom_index_1_z"]
)
mag0 = np.sqrt(
    X_test["atom_index_0_x"] ** 2
    + X_test["atom_index_0_y"] ** 2
    + X_test["atom_index_0_z"] ** 2
)
mag1 = np.sqrt(
    X_test["atom_index_1_x"] ** 2
    + X_test["atom_index_1_y"] ** 2
    + X_test["atom_index_1_z"] ** 2
)
cos_angle = np.clip(dot / (mag0 * mag1 + 1e-9), -1.0, 1.0)
X_test["angle"] = np.arccos(cos_angle).astype(np.float32)

X_train["join_type"] = X_train["type"].str.slice(0, 2)
X_test["join_type"] = X_test["type"].str.slice(0, 2)
X_train["num_bonds"] = X_train["type"].str.slice(0, 1).astype(np.int8)
X_test["num_bonds"] = X_test["type"].str.slice(0, 1).astype(np.int8)



## === cell 5
num_atoms_series = X_train.groupby("molecule_name")["atom_index_0"].max() + 1
X_train["num_atoms"] = X_train["molecule_name"].map(num_atoms_series).astype(np.int16)
X_test["num_atoms"] = (
    X_test["molecule_name"].map(num_atoms_series).fillna(-1).astype(np.int16)
)




## === cell 6
def convert_object_to_categories(df_train, df_test):
    for col in df_train.columns:
        if df_train[col].dtype == "O":
            df_train[col] = df_train[col].astype("category")
            df_test[col] = df_test[col].astype("category")
    return df_train, df_test


X_train, X_test = convert_object_to_categories(X_train, X_test)

numeric_cols = X_train.select_dtypes(
    include=["int16", "int32", "int64", "float32", "float64"]
).columns
X_train[numeric_cols] = X_train[numeric_cols].fillna(-999)
X_test[numeric_cols] = X_test[numeric_cols].fillna(-999)

categorical_cols = X_train.select_dtypes(include=["category"]).columns
for col in categorical_cols:
    if not pd.api.types.is_categorical_dtype(X_test[col]):
        X_test[col] = X_test[col].astype("category")
    if "missing" not in X_train[col].cat.categories:
        X_train[col] = X_train[col].cat.add_categories("missing")
        X_test[col] = X_test[col].cat.add_categories("missing")
    X_train[col] = X_train[col].fillna("missing")
    X_test[col] = X_test[col].fillna("missing")
    X_train[col] = X_train[col].cat.codes.astype(np.int16)
    X_test[col] = X_test[col].cat.codes.astype(np.int16)

cat_features = list(categorical_cols)
cat_features_idx = [X_train.columns.get_loc(c) for c in categorical_cols]

X_train_np = X_train.values.astype(np.float32)
X_test_np = X_test.values.astype(np.float32)

del X_train, X_test, train
gc.collect()



## === cell 7
kf = KFold(n_splits=5, shuffle=True, random_state=42)
preds = np.zeros(len(X_test_np))
lgb_params = {
    "n_estimators": 2500,
    "learning_rate": 0.02,
    "num_leaves": 511,
    "max_bin": 255,
    "objective": "regression",
    "random_state": 42,
    "n_jobs": -1,
    "metric": "mae",
    "bagging_fraction": 0.8,
    "feature_fraction": 0.8,
    "verbosity": -1,
}
for train_idx, val_idx in kf.split(X_train_np):
    X_tr, X_val = X_train_np[train_idx], X_train_np[val_idx]
    y_tr, y_val = y_train.iloc[train_idx].values, y_train.iloc[val_idx].values
    model = lgbm.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        categorical_feature=cat_features_idx,
        callbacks=[lgbm.early_stopping(stopping_rounds=100, verbose=False)],
    )
    preds += model.predict(X_test_np) / kf.n_splits



## === cell 8
sample_sub["scalar_coupling_constant"] = preds
sample_sub.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
