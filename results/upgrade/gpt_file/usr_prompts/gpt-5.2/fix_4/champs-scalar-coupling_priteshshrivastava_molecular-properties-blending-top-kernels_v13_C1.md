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
import numpy as np
import pandas as pd

DATA_DIR = "/kaggle/data/champs-scalar-coupling"

print("DATA_DIR exists:", os.path.isdir(DATA_DIR))
print("Files:", sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])[:10])



## === cell 1
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
structures_path = os.path.join(DATA_DIR, "structures.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
structures = pd.read_csv(structures_path)

print(train.shape, test.shape, structures.shape)
print(train.columns)
print(test.columns)
print(structures.columns)



## === cell 2
s0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
    }
)[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]]

s1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
    }
)[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]]


def add_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
    df = df.merge(s1, on=["molecule_name", "atom_index_1"], how="left")

    dx = df["x0"] - df["x1"]
    dy = df["y0"] - df["y1"]
    dz = df["z0"] - df["z1"]

    df["dx"] = dx
    df["dy"] = dy
    df["dz"] = dz
    df["abs_dx"] = np.abs(dx)
    df["abs_dy"] = np.abs(dy)
    df["abs_dz"] = np.abs(dz)

    df["distance"] = np.sqrt(dx * dx + dy * dy + dz * dz)
    df["inv_distance"] = 1.0 / (df["distance"] + 1e-6)
    df["distance2"] = df["distance"] * df["distance"]
    return df


train_f = add_structure_features(train)
test_f = add_structure_features(test)

feature_cols_num = [
    "distance",
    "inv_distance",
    "distance2",
    "dx",
    "dy",
    "dz",
    "abs_dx",
    "abs_dy",
    "abs_dz",
]
feature_cols_cat = ["atom_0", "atom_1", "type"]
target_col = "scalar_coupling_constant"

missing0 = train_f["atom_0"].isna().mean()
missing1 = train_f["atom_1"].isna().mean()
print("Missing atom_0 rate:", missing0, "Missing atom_1 rate:", missing1)
print(
    "Unique coupling types (train):",
    train_f["type"].nunique(),
    sorted(train_f["type"].unique())[:10],
)



## === cell 3
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, feature_cols_num),
        ("cat", categorical_transformer, feature_cols_cat),
    ],
    remainder="drop",
)

base_model = HistGradientBoostingRegressor(
    loss="absolute_error",
    max_depth=10,
    learning_rate=0.06,
    max_iter=350,
    min_samples_leaf=20,
    l2_regularization=0.0,
    random_state=42,
)


def make_pipeline():
    return Pipeline(steps=[("preprocess", preprocess), ("model", base_model)])


gss = GroupShuffleSplit(n_splits=1, test_size=0.02, random_state=42)

X_global = train_f[feature_cols_num + feature_cols_cat]
y_global = train_f[target_col].astype(np.float32)
global_pipe = make_pipeline()
global_pipe.fit(X_global, y_global)

val_mae_by_type = {}
val_weighted_mae = 0.0
val_n = 0

RARE_MIN_ROWS = 5000

for t in sorted(train_f["type"].unique()):
    df_t = train_f[train_f["type"] == t]
    X_t = df_t[feature_cols_num + feature_cols_cat]
    y_t = df_t[target_col].astype(np.float32)
    groups = df_t["molecule_name"].values

    if len(df_t) < RARE_MIN_ROWS or df_t["molecule_name"].nunique() < 3:
        pred_t = global_pipe.predict(X_t)
        mae_t = mean_absolute_error(y_t, pred_t)
        val_mae_by_type[t] = mae_t
        val_weighted_mae += mae_t * len(df_t)
        val_n += len(df_t)
        continue

    tr_idx, va_idx = next(gss.split(X_t, y_t, groups=groups))
    X_tr, X_va = X_t.iloc[tr_idx], X_t.iloc[va_idx]
    y_tr, y_va = y_t.iloc[tr_idx], y_t.iloc[va_idx]

    pipe = make_pipeline()
    pipe.fit(X_tr, y_tr)
    va_pred = pipe.predict(X_va)
    mae_t = mean_absolute_error(y_va, va_pred)
    val_mae_by_type[t] = mae_t
    val_weighted_mae += mae_t * len(y_va)
    val_n += len(y_va)

print("Validation MAE by type (first 8):", dict(list(val_mae_by_type.items())[:8]))
print("Validation weighted MAE (informational):", val_weighted_mae / max(val_n, 1))



## === cell 4
test_pred = np.zeros(len(test_f), dtype=np.float32)

types = sorted(train_f["type"].unique())
type_counts = train_f["type"].value_counts().to_dict()

for t in types:
    test_t_idx = test_f.index[test_f["type"] == t].values
    if len(test_t_idx) == 0:
        continue

    X_test_t = test_f.loc[test_t_idx, feature_cols_num + feature_cols_cat]

    if type_counts.get(t, 0) < 5000:
        test_pred[test_t_idx] = global_pipe.predict(X_test_t).astype(np.float32)
        continue

    train_t = train_f[train_f["type"] == t]
    X_train_t = train_t[feature_cols_num + feature_cols_cat]
    y_train_t = train_t[target_col].astype(np.float32)

    pipe = make_pipeline()
    pipe.fit(X_train_t, y_train_t)
    test_pred[test_t_idx] = pipe.predict(X_test_t).astype(np.float32)

submission = test[["id"]].copy()
submission["scalar_coupling_constant"] = test_pred

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path, "shape:", submission.shape)
print(submission.head())
print(
    "Pred stats:",
    float(np.min(test_pred)),
    float(np.mean(test_pred)),
    float(np.max(test_pred)),
)

sample = pd.read_csv(sample_path)
print(
    "Sample submission shape:", sample.shape, "Our submission shape:", submission.shape
)
print("Columns match:", list(sample.columns) == list(submission.columns))
