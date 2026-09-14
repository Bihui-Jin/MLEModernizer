# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

# 5. Target score

-1.666828640654786

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.95983) has done: 'I remove the dependency on missing external blend files (which cause the `FileNotFoundError`) and replace it with a self-contained baseline that uses only the provided competition data. To keep the solution minimal and robust, the script build a simple, legitimate feature set from `structures.csv` (pairwise distance + atom types + coupling type) and train a regression model. This run end-to-end within the Kaggle environment and write a valid `submission.csv` with the required `id` and `scalar_coupling_constant` columns. The approach is score-improving versus an invalid submission, while staying simple and stable.'
- What this solution (achieved 1.2362) has done: 'Your current score (1.95983, lower-is-better) is far worse than the target (-1.6668), so we should legitimately improve the model while keeping the same overall approach (pairwise geometry + atom/type categoricals + tree-based regression). The biggest issue is that the competition metric averages MAE per coupling `type`, but your single global model implicitly overweights frequent types; training and predicting per-type with the same features/model usually improves the metric a lot with minimal logic change. I also switch the validation split to be by `molecule_name` (as the competition split is by molecule) to better match generalization and avoid leakage-like optimism; this doesn’t change submission semantics, but it guides safer choices. Finally, I keep the same estimator family and preprocessing, but use a slightly stronger (still fast) set of HGB hyperparameters and train separate models per type, then write a standard `submission.csv`.'

# 9. Code solution

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

train = pd.read_csv(
    train_path,
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
        "molecule_name": "category",
        "atom_index_0": "int16",
        "atom_index_1": "int16",
        "type": "category",
        "scalar_coupling_constant": "float32",
    },
)
test = pd.read_csv(
    test_path,
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
    dtype={
        "id": "int32",
        "molecule_name": "category",
        "atom_index_0": "int16",
        "atom_index_1": "int16",
        "type": "category",
    },
)
structures = pd.read_csv(
    structures_path,
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
    dtype={
        "molecule_name": "category",
        "atom_index": "int16",
        "atom": "category",
        "x": "float32",
        "y": "float32",
        "z": "float32",
    },
)

print(train.shape, test.shape, structures.shape)
print(train.columns)
print(test.columns)
print(structures.columns)



## === cell 2
structures_idx = structures.set_index(["molecule_name", "atom_index"])[
    ["atom", "x", "y", "z"]
]


def add_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    a0 = structures_idx.reindex(
        pd.MultiIndex.from_frame(
            df[["molecule_name", "atom_index_0"]].rename(
                columns={"atom_index_0": "atom_index"}
            )
        )
    )
    a0 = a0.rename(
        columns={"atom": "atom_0", "x": "x0", "y": "y0", "z": "z0"}
    ).reset_index(drop=True)

    a1 = structures_idx.reindex(
        pd.MultiIndex.from_frame(
            df[["molecule_name", "atom_index_1"]].rename(
                columns={"atom_index_1": "atom_index"}
            )
        )
    )
    a1 = a1.rename(
        columns={"atom": "atom_1", "x": "x1", "y": "y1", "z": "z1"}
    ).reset_index(drop=True)

    df = pd.concat([df.reset_index(drop=True), a0, a1], axis=1)

    dx = df["x0"].to_numpy(dtype=np.float32) - df["x1"].to_numpy(dtype=np.float32)
    dy = df["y0"].to_numpy(dtype=np.float32) - df["y1"].to_numpy(dtype=np.float32)
    dz = df["z0"].to_numpy(dtype=np.float32) - df["z1"].to_numpy(dtype=np.float32)

    df["dx"] = dx
    df["dy"] = dy
    df["dz"] = dz
    df["abs_dx"] = np.abs(dx)
    df["abs_dy"] = np.abs(dy)
    df["abs_dz"] = np.abs(dz)

    dist = np.sqrt(dx * dx + dy * dy + dz * dz, dtype=np.float32)
    df["distance"] = dist
    df["inv_distance"] = 1.0 / (dist + np.float32(1e-6))
    df["distance2"] = dist * dist
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
    sorted(train_f["type"].astype(str).unique())[:10],
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
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
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

X_global_df = train_f[feature_cols_num + feature_cols_cat]
y_global = train_f[target_col].astype(np.float32)

global_pipe = make_pipeline()
global_pipe.fit(X_global_df, y_global)

X_global_tr = global_pipe.named_steps["preprocess"].transform(X_global_df)
X_test_tr = global_pipe.named_steps["preprocess"].transform(
    test_f[feature_cols_num + feature_cols_cat]
)

train_type = train_f["type"].astype(str).to_numpy()
test_type = test_f["type"].astype(str).to_numpy()
train_mol = train_f["molecule_name"].astype(str).to_numpy()

val_mae_by_type = {}
val_weighted_mae = 0.0
val_n = 0

RARE_MIN_ROWS = 5000

types = np.unique(train_type)
types.sort()

for t in types:
    mask_t = train_type == t
    idx_t = np.flatnonzero(mask_t)
    n_t = idx_t.size

    if n_t == 0:
        continue

    if n_t < RARE_MIN_ROWS or len(np.unique(train_mol[mask_t])) < 3:
        pred_t = global_pipe.named_steps["model"].predict(X_global_tr[idx_t].toarray())
        mae_t = mean_absolute_error(y_global.iloc[idx_t], pred_t)
        val_mae_by_type[t] = mae_t
        val_weighted_mae += mae_t * n_t
        val_n += n_t
        continue

    groups = train_mol[mask_t]
    tr_rel, va_rel = next(
        gss.split(np.zeros(n_t), y_global.iloc[idx_t].to_numpy(), groups=groups)
    )
    tr_idx = idx_t[tr_rel]
    va_idx = idx_t[va_rel]

    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_depth=10,
        learning_rate=0.06,
        max_iter=350,
        min_samples_leaf=20,
        l2_regularization=0.0,
        random_state=42,
    )
    model.fit(
        X_global_tr[tr_idx].toarray(), y_global.iloc[tr_idx].to_numpy(dtype=np.float32)
    )
    va_pred = model.predict(X_global_tr[va_idx].toarray())
    mae_t = mean_absolute_error(y_global.iloc[va_idx], va_pred)

    val_mae_by_type[t] = mae_t
    val_weighted_mae += mae_t * va_idx.size
    val_n += va_idx.size

print("Validation MAE by type (first 8):", dict(list(val_mae_by_type.items())[:8]))
print("Validation weighted MAE (informational):", val_weighted_mae / max(val_n, 1))



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/236970056.py in <cell line: 0>()
    105     )
    106     model.fit(
--> 107         X_global_tr[tr_idx].toarray(), y_global.iloc[tr_idx].to_numpy(dtype=np.float32)
    108     )
    109     va_pred = model.predict(X_global_tr[va_idx].toarray())

AttributeError: 'numpy.ndarray' object has no attribute 'toarray'

## === cell 4
test_pred = np.zeros(len(test_f), dtype=np.float32)

type_counts = pd.Series(train_type).value_counts().to_dict()

for t in types:
    test_idx_t = np.flatnonzero(test_type == t)
    if test_idx_t.size == 0:
        continue

    if type_counts.get(t, 0) < 5000:
        test_pred[test_idx_t] = (
            global_pipe.named_steps["model"]
            .predict(X_test_tr[test_idx_t].toarray())
            .astype(np.float32)
        )
        continue

    train_idx_t = np.flatnonzero(train_type == t)

    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_depth=10,
        learning_rate=0.06,
        max_iter=350,
        min_samples_leaf=20,
        l2_regularization=0.0,
        random_state=42,
    )
    model.fit(
        X_global_tr[train_idx_t].toarray(),
        y_global.iloc[train_idx_t].to_numpy(dtype=np.float32),
    )
    test_pred[test_idx_t] = model.predict(X_test_tr[test_idx_t].toarray()).astype(
        np.float32
    )

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

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/19988374.py in <cell line: 0>()
     30     )
     31     model.fit(
---> 32         X_global_tr[train_idx_t].toarray(),
     33         y_global.iloc[train_idx_t].to_numpy(dtype=np.float32),
     34     )

AttributeError: 'numpy.ndarray' object has no attribute 'toarray'
