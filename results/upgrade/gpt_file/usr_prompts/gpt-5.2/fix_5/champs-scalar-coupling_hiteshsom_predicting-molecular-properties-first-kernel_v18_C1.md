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

# 5. Target score

0.76707

# 6. Current score

1.72794

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 2.32775) has done: 'I fix the LightGBM training crash by converting the remaining pandas `string` columns (e.g., `join_type`) into proper categorical codes (int) while keeping the same features and default `LGBMRegressor` core logic. I also ensure train/test get identical transformations and that no object/string dtypes reach LightGBM. Finally, I make submission creation robust by guaranteeing `y_predict` is defined only after successful training/prediction and writing a valid `submission.csv` with the required columns and order.'
- What this solution (achieved 1.7226) has done: 'Your current score is far from the target (lower is better), so we need a real accuracy improvement while keeping the same overall LightGBM regressor approach. The biggest issue is that `molecule_name` is being used as a numeric ID feature after coding, which won’t generalize because train/test molecules are disjoint; dropping it is a minimal change that typically improves generalization a lot for this competition. Also, training one global model across all coupling `type` forces the model to learn multiple very different targets at once; a standard minimal adjustment (still LightGBM regression) is to train one model per `type` and predict test rows of that `type`. Finally, the “angle” feature is currently computed as an angle between absolute position vectors (from origin), which is not physically meaningful here; replacing it with a safe, minimal geometric feature (direction cosines of the bond vector) improves signal without changing the overall pipeline.'
- What this solution (achieved 1.72794) has done: 'To move your score down toward the 0.767 target while preserving the same LightGBM-per-type core approach, I make three minimal, high-impact adjustments: (1) change the CV/training split to be by `molecule_name` (GroupKFold) to match the competition’s molecule-disjoint generalization and reduce overfitting; (2) add a tiny amount of regularization + more trees (same model family, same training loop) which typically improves MAE stability without changing the approach; and (3) calibrate each type’s predictions by applying an out-of-fold mean residual “bias correction” per type (a minimal post-processing step consistent with the metric and often reduces log-MAE). The rest of your feature logic and per-type training remains intact, and the script still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import gc
import math
import category_encoders as ce
import lightgbm as lgbm
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_absolute_error as mae
import os

print(os.listdir("../input"))



## === cell 1
gc.collect()



## === cell 2
DATA_DIR = "../input/champs-scalar-coupling"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
structures = pd.read_csv(f"{DATA_DIR}/structures.csv")

print(f"train.shape: {train.shape}")
print(f"test.shape: {test.shape}")
print(f"structures.shape: {structures.shape}")



## === cell 3
X_train = train.drop(columns=["scalar_coupling_constant"]).copy()
y_train = train["scalar_coupling_constant"].copy()
X_test = test.copy()

print(f"X_train.shape: {X_train.shape}")
print(f"X_test.shape: {X_test.shape}")



## === cell 4
train_id = X_train["id"].copy()
test_id = X_test["id"].copy()

X_train = X_train.drop(columns=["id"])
X_test = X_test.drop(columns=["id"])



## === cell 5
X_train = X_train.reset_index(drop=True).reset_index()  # adds column "index"
X_test = X_test.reset_index(drop=True).reset_index()




## === cell 6
def convert_object_to_categories(X_train_in, X_test_in):
    X_train = X_train_in.copy()
    X_test = X_test_in.copy()
    for col in X_train.columns:
        if X_train[col].dtype == "O" or str(X_train[col].dtype) == "object":
            combined = pd.concat(
                [X_train[col].astype("string"), X_test[col].astype("string")], axis=0
            )
            cats = pd.Categorical(combined).categories
            X_train[col] = pd.Categorical(
                X_train[col].astype("string"), categories=cats
            )
            X_test[col] = pd.Categorical(X_test[col].astype("string"), categories=cats)
    return X_train, X_test


X_train, X_test = convert_object_to_categories(X_train, X_test)



## === cell 7
print(f"{X_train['type'].unique()}")
print(f"{X_test['type'].unique()}")


def calc_score(X_fold, y_true, y_pred):
    X_new = X_fold.copy()
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    X_new = X_new.reset_index(drop=True)
    X_new = X_new.merge(
        pd.DataFrame(y_true, columns=["scalar_coupling_constant"]),
        left_index=True,
        right_index=True,
    )
    X_new = X_new.merge(
        pd.DataFrame(y_pred, columns=["y_val"]), left_index=True, right_index=True
    )
    X_new["error"] = (X_new["scalar_coupling_constant"] - X_new["y_val"]).abs()
    X_new["count"] = 1
    score_df = X_new.groupby(by=["type"]).agg({"count": "count", "error": "sum"})
    score_df["error"] = (score_df["error"] / score_df["count"]).apply(
        np.log, dtype=float
    )
    score = (1 / score_df.shape[0]) * (score_df["error"].sum())
    return score




## === cell 8
def cross_val_grouped_by_molecule(X, y, groups):
    print(X.shape)
    gkf = GroupKFold(n_splits=5)
    fold = 0
    for train_index, val_index in gkf.split(X, y, groups=groups):
        fold += 1
        lgbm_model = lgbm.LGBMRegressor(
            n_estimators=600,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.2,
            random_state=10,
            n_jobs=-1,
        )
        lgbm_model.fit(X.loc[train_index, :], y.iloc[train_index])
        y_val = lgbm_model.predict(X.loc[val_index, :])
        print(
            f"fold{fold} score: {calc_score(X.loc[val_index, :], y.iloc[val_index], y_val)}"
        )




## === cell 9
num_atoms_df = (
    structures.groupby("molecule_name")["atom_index"]
    .max()
    .add(1)
    .rename("num_atoms")
    .reset_index()
)



## === cell 10
X_train = X_train.merge(
    structures,
    left_on=["molecule_name", "atom_index_0"],
    right_on=["molecule_name", "atom_index"],
    how="left",
    sort=False,
)
X_train = X_train.rename(
    columns={
        "atom_index": "atom_index_0_0",
        "x": "atom_index_0_x",
        "y": "atom_index_0_y",
        "z": "atom_index_0_z",
        "atom": "atom_0",
    }
)

X_test = X_test.merge(
    structures,
    left_on=["molecule_name", "atom_index_0"],
    right_on=["molecule_name", "atom_index"],
    how="left",
    sort=False,
)
X_test = X_test.rename(
    columns={
        "atom_index": "atom_index_0_0",
        "x": "atom_index_0_x",
        "y": "atom_index_0_y",
        "z": "atom_index_0_z",
        "atom": "atom_0",
    }
)

assert (
    X_train.shape[0] == train.shape[0]
), "Row count changed for train after atom_0 merge."
assert (
    X_test.shape[0] == test.shape[0]
), "Row count changed for test after atom_0 merge."



## === cell 11
X_train = X_train.merge(
    structures,
    left_on=["molecule_name", "atom_index_1"],
    right_on=["molecule_name", "atom_index"],
    how="left",
    sort=False,
)
X_train = X_train.rename(
    columns={
        "atom_index": "atom_index_1_1",
        "x": "atom_index_1_x",
        "y": "atom_index_1_y",
        "z": "atom_index_1_z",
        "atom": "atom_1",
    }
)

X_test = X_test.merge(
    structures,
    left_on=["molecule_name", "atom_index_1"],
    right_on=["molecule_name", "atom_index"],
    how="left",
    sort=False,
)
X_test = X_test.rename(
    columns={
        "atom_index": "atom_index_1_1",
        "x": "atom_index_1_x",
        "y": "atom_index_1_y",
        "z": "atom_index_1_z",
        "atom": "atom_1",
    }
)

assert (
    X_train.shape[0] == train.shape[0]
), "Row count changed for train after atom_1 merge."
assert (
    X_test.shape[0] == test.shape[0]
), "Row count changed for test after atom_1 merge."



## === cell 12
X_train = X_train.drop(columns=["atom_index_0_0", "atom_index_1_1"])
X_test = X_test.drop(columns=["atom_index_0_0", "atom_index_1_1"])



## === cell 13
X_train["distance"] = (
    (X_train["atom_index_0_x"] - X_train["atom_index_1_x"]) ** 2
    + (X_train["atom_index_0_y"] - X_train["atom_index_1_y"]) ** 2
    + (X_train["atom_index_0_z"] - X_train["atom_index_1_z"]) ** 2
) ** 0.5

X_test["distance"] = (
    (X_test["atom_index_0_x"] - X_test["atom_index_1_x"]) ** 2
    + (X_test["atom_index_0_y"] - X_test["atom_index_1_y"]) ** 2
    + (X_test["atom_index_0_z"] - X_test["atom_index_1_z"]) ** 2
) ** 0.5



## === cell 14
X_train["join_type"] = X_train["type"].astype("string").str.slice(0, 2)
X_test["join_type"] = X_test["type"].astype("string").str.slice(0, 2)



## === cell 15
print(X_train["atom_0"].unique())
print(X_test["atom_0"].unique())



## === cell 16
X_train = X_train.merge(num_atoms_df, on="molecule_name", how="left", sort=False)
X_test = X_test.merge(num_atoms_df, on="molecule_name", how="left", sort=False)



## === cell 17
X_train, X_test = convert_object_to_categories(X_train, X_test)



## === cell 18
X_train_new = X_train.set_index(keys="index")
X_test_new = X_test.set_index(keys="index")

X_train_new = X_train_new.sort_index(axis=0)
X_test_new = X_test_new.sort_index(axis=0)

assert (
    X_train_new.shape[0] == train.shape[0]
), "X_train_new row mismatch after indexing/sorting."
assert (
    X_test_new.shape[0] == test.shape[0]
), "X_test_new row mismatch after indexing/sorting."



## === cell 19
print("Unique molecules in train:", X_train_new["molecule_name"].nunique())



## === cell 20
X_train_new["num_bonds"] = X_train_new["join_type"].astype("string").str.slice(0, 1)
X_test_new["num_bonds"] = X_test_new["join_type"].astype("string").str.slice(0, 1)

X_train_new["num_bonds"] = X_train_new["num_bonds"].astype("int")
X_test_new["num_bonds"] = X_test_new["num_bonds"].astype("int")




## === cell 21
def add_bond_direction_features(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    dx = df["atom_index_0_x"] - df["atom_index_1_x"]
    dy = df["atom_index_0_y"] - df["atom_index_1_y"]
    dz = df["atom_index_0_z"] - df["atom_index_1_z"]
    dist = df["distance"].replace(0, np.nan)
    df["dir_x"] = (dx / dist).fillna(0.0)
    df["dir_y"] = (dy / dist).fillna(0.0)
    df["dir_z"] = (dz / dist).fillna(0.0)
    return df


X_train_new = add_bond_direction_features(X_train_new)
X_test_new = add_bond_direction_features(X_test_new)




## === cell 22
def make_lgbm_compatible(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    for col in df.columns:
        dt = df[col].dtype
        if str(dt) == "string":
            df[col] = df[col].astype("category")
        if pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].cat.codes.astype("int32")
        if df[col].dtype == "O":
            df[col] = df[col].astype("category").cat.codes.astype("int32")
    return df


X_train_model = X_train_new.drop(columns=["molecule_name"])
X_test_model = X_test_new.drop(columns=["molecule_name"])

X_train_lgb = make_lgbm_compatible(X_train_model)
X_test_lgb = make_lgbm_compatible(X_test_model)

bad_cols = [
    c
    for c in X_train_lgb.columns
    if not (
        pd.api.types.is_numeric_dtype(X_train_lgb[c])
        or pd.api.types.is_bool_dtype(X_train_lgb[c])
    )
]
assert len(bad_cols) == 0, f"Non-numeric columns still present for LightGBM: {bad_cols}"




## === cell 23
def compute_type_bias_correction(X_lgb, y, X_meta, n_splits=5, random_state=10):
    train_types = X_meta["type"].astype("string")
    groups = X_meta["molecule_name"].astype("string")

    gkf = GroupKFold(n_splits=n_splits)
    oof_pred = np.zeros(len(X_lgb), dtype=np.float64)

    for fold, (tr_idx, va_idx) in enumerate(
        gkf.split(X_lgb, y, groups=groups), start=1
    ):
        model = lgbm.LGBMRegressor(
            n_estimators=600,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.2,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_lgb.iloc[tr_idx, :], y.iloc[tr_idx])
        oof_pred[va_idx] = model.predict(X_lgb.iloc[va_idx, :])

    resid = y.values - oof_pred
    bias_by_type = (
        pd.DataFrame({"type": train_types.values, "resid": resid})
        .groupby("type")["resid"]
        .mean()
        .to_dict()
    )
    return bias_by_type


bias_by_type = compute_type_bias_correction(
    X_train_lgb, y_train, X_train_new, n_splits=5, random_state=10
)
print("Computed bias correction for types:", sorted(bias_by_type.keys()))



## === cell 24
y_predict = np.zeros(X_test_lgb.shape[0], dtype=np.float64)

train_types = X_train_new["type"].astype("string")
test_types = X_test_new["type"].astype("string")

unique_types = sorted(train_types.unique().tolist())
print("Training per-type models for types:", unique_types)

for t in unique_types:
    tr_idx = train_types == t
    te_idx = test_types == t

    if te_idx.sum() == 0:
        continue

    lgbm_model = lgbm.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.2,
        random_state=10,
        n_jobs=-1,
    )
    lgbm_model.fit(X_train_lgb.loc[tr_idx, :], y_train.loc[tr_idx])
    preds = lgbm_model.predict(X_test_lgb.loc[te_idx, :])

    preds = preds + float(bias_by_type.get(t, 0.0))

    y_predict[te_idx.values] = preds

assert np.isfinite(y_predict).all(), "Non-finite predictions produced."



## === cell 25
submission = pd.DataFrame({"id": test_id.values, "scalar_coupling_constant": y_predict})

assert (
    submission.shape[0] == sample_sub.shape[0]
), "Submission row count mismatch vs sample_submission."
assert (
    submission["id"].iloc[0] == sample_sub["id"].iloc[0]
), "Submission id order mismatch vs sample_submission."

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
