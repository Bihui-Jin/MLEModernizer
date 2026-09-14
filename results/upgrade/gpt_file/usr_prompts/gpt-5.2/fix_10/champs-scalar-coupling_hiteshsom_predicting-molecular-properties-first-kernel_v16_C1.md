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
import numpy as np  # linear algebra
import pandas as pd  # data processing
import gc
import lightgbm as lgbm
from sklearn.model_selection import GroupKFold
import os

print(os.listdir("../input"))



## === cell 1
gc.collect()



## === cell 2
BASE_PATH = "../input"
if os.path.exists(os.path.join(BASE_PATH, "champs-scalar-coupling", "train.csv")):
    DATA_PATH = os.path.join(BASE_PATH, "champs-scalar-coupling")
else:
    DATA_PATH = BASE_PATH

train_dtypes = {
    "id": np.int32,
    "molecule_name": "category",
    "atom_index_0": np.int16,
    "atom_index_1": np.int16,
    "type": "category",
    "scalar_coupling_constant": np.float32,
}
test_dtypes = {
    "id": np.int32,
    "molecule_name": "category",
    "atom_index_0": np.int16,
    "atom_index_1": np.int16,
    "type": "category",
}
structures_dtypes = {
    "molecule_name": "category",
    "atom_index": np.int16,
    "atom": "category",
    "x": np.float32,
    "y": np.float32,
    "z": np.float32,
}
mulliken_dtypes = {
    "molecule_name": "category",
    "atom_index": np.int16,
    "mulliken_charge": np.float32,
}
shield_dtypes = {
    "molecule_name": "category",
    "atom_index": np.int16,
    "XX": np.float32,
    "YX": np.float32,
    "ZX": np.float32,
    "XY": np.float32,
    "YY": np.float32,
    "ZY": np.float32,
    "XZ": np.float32,
    "YZ": np.float32,
    "ZZ": np.float32,
}

train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"), dtype=train_dtypes)
test = pd.read_csv(os.path.join(DATA_PATH, "test.csv"), dtype=test_dtypes)
sample_sub = pd.read_csv(
    os.path.join(DATA_PATH, "sample_submission.csv"),
    usecols=["id", "scalar_coupling_constant"],
)

structures = pd.read_csv(
    os.path.join(DATA_PATH, "structures.csv"),
    dtype=structures_dtypes,
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
)
mulliken = pd.read_csv(
    os.path.join(DATA_PATH, "mulliken_charges.csv"),
    dtype=mulliken_dtypes,
    usecols=["molecule_name", "atom_index", "mulliken_charge"],
)
shield = pd.read_csv(
    os.path.join(DATA_PATH, "magnetic_shielding_tensors.csv"), dtype=shield_dtypes
)

print(f"train.shape: {train.shape}")
print(f"test.shape: {test.shape}")
print(f"structures.shape: {structures.shape}")
print(f"mulliken.shape: {mulliken.shape}")
print(f"shield.shape: {shield.shape}")



## === cell 3
X_train = train.drop(columns=["scalar_coupling_constant"]).copy()
y_train = train["scalar_coupling_constant"].copy()
X_test = test.copy()

print(f"X_train.shape: {X_train.shape}")
print(f"X_test.shape: {X_test.shape}")



## === cell 4
train_id = X_train["id"].copy()
test_id = X_test["id"].copy()



## === cell 5
X_train = X_train.copy()
X_test = X_test.copy()
X_train["row_id"] = np.arange(len(X_train), dtype=np.int32)
X_test["row_id"] = np.arange(len(X_test), dtype=np.int32)




## === cell 6
def convert_object_to_categories(X_train, X_test):
    for col in X_train.columns:
        if X_train[col].dtype == "O":
            X_train[col] = X_train[col].astype("category")
            if col in X_test.columns:
                X_test[col] = X_test[col].astype("category")
    for col in X_test.columns:
        if X_test[col].dtype == "O":
            X_test[col] = X_test[col].astype("category")
    return X_train, X_test


X_train, X_test = convert_object_to_categories(X_train, X_test)




## === cell 7
def calc_score_by_type(df_type, y_true, y_pred):
    err = np.mean(np.abs(y_true - y_pred))
    return float(np.log(err + 1e-12))


def calc_overall_score(types, y_true, y_pred):
    tmp = pd.DataFrame({"type": types.astype(str).values, "y": y_true, "p": y_pred})
    per_type = tmp.groupby("type").apply(
        lambda g: calc_score_by_type(g["type"], g["y"].values, g["p"].values)
    )
    return float(per_type.mean())


print(f"{X_train['type'].unique()}")
print(f"{X_test['type'].unique()}")




## === cell 8
def cross_val_grouped_by_molecule(X, y, groups, cat_cols, n_splits=3, lgb_params=None):
    if lgb_params is None:
        lgb_params = {}
    gkf = GroupKFold(n_splits=n_splits)

    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        Xtr = X.iloc[tr_idx]
        ytr = y.iloc[tr_idx]
        Xva = X.iloc[va_idx]
        yva = y.iloc[va_idx]

        model = lgbm.LGBMRegressor(**lgb_params)
        model.fit(Xtr, ytr, categorical_feature=cat_cols)
        pva = model.predict(Xva)

        score = calc_overall_score(Xva["type"], yva.values, pva)
        fold_scores.append(score)
        print(f"GroupKFold fold {fold} logMAE-by-type score: {score:.5f}")

    print(f"Mean CV score (GroupKFold): {np.mean(fold_scores):.5f}")
    return float(np.mean(fold_scores))




## === cell 9
structures_join = structures.copy()
structures_join = structures_join.sort_values(
    ["molecule_name", "atom_index"], kind="mergesort"
)
structures_join = structures_join.set_index(["molecule_name", "atom_index"], drop=False)


def add_atom_features(df, atom_col, prefix):
    tmp = df[["row_id", "molecule_name", atom_col]].copy()
    tmp = tmp.rename(columns={atom_col: "atom_index"})
    looked = tmp.join(
        structures_join[["x", "y", "z", "atom"]],
        on=["molecule_name", "atom_index"],
        how="left",
    )
    df[f"{prefix}x"] = looked["x"].to_numpy()
    df[f"{prefix}y"] = looked["y"].to_numpy()
    df[f"{prefix}z"] = looked["z"].to_numpy()
    df[f"atom{prefix[:-1]}"] = looked["atom"].to_numpy()
    return df


X_train = add_atom_features(X_train, "atom_index_0", "atom_index_0_")
X_test = add_atom_features(X_test, "atom_index_0", "atom_index_0_")
X_train = X_train.rename(columns={"atom0": "atom_0"})
X_test = X_test.rename(columns={"atom0": "atom_0"})



## === cell 10
X_train = add_atom_features(X_train, "atom_index_1", "atom_index_1_")
X_test = add_atom_features(X_test, "atom_index_1", "atom_index_1_")
X_train = X_train.rename(columns={"atom1": "atom_1"})
X_test = X_test.rename(columns={"atom1": "atom_1"})



## === cell 11
pass



## === cell 12
X_train.head()



## === cell 13
dx = X_train["atom_index_0_x"].to_numpy() - X_train["atom_index_1_x"].to_numpy()
dy = X_train["atom_index_0_y"].to_numpy() - X_train["atom_index_1_y"].to_numpy()
dz = X_train["atom_index_0_z"].to_numpy() - X_train["atom_index_1_z"].to_numpy()
X_train["distance"] = np.sqrt(dx * dx + dy * dy + dz * dz)

dx = X_test["atom_index_0_x"].to_numpy() - X_test["atom_index_1_x"].to_numpy()
dy = X_test["atom_index_0_y"].to_numpy() - X_test["atom_index_1_y"].to_numpy()
dz = X_test["atom_index_0_z"].to_numpy() - X_test["atom_index_1_z"].to_numpy()
X_test["distance"] = np.sqrt(dx * dx + dy * dy + dz * dz)



## === cell 14
X_train["join_type"] = X_train["type"].astype(str).str.slice(0, 2)
X_test["join_type"] = X_test["type"].astype(str).str.slice(0, 2)



## === cell 15
print(X_train["atom_0"].unique())
print(X_test["atom_0"].unique())



## === cell 16
mol_num_atoms = (
    structures.groupby("molecule_name", sort=False)["atom_index"]
    .max()
    .add(1)
    .astype(np.int32)
    .rename("num_atoms")
).reset_index()

X_train = X_train.merge(mol_num_atoms, on="molecule_name", how="left")
X_test = X_test.merge(mol_num_atoms, on="molecule_name", how="left")



## === cell 17
mulliken_join = mulliken.copy()
mulliken_join = mulliken_join.sort_values(
    ["molecule_name", "atom_index"], kind="mergesort"
)
mulliken_join = mulliken_join.set_index(["molecule_name", "atom_index"], drop=False)[
    ["mulliken_charge"]
]


def add_mulliken(df, atom_col, out_col):
    tmp = df[["row_id", "molecule_name", atom_col]].copy()
    tmp = tmp.rename(columns={atom_col: "atom_index"})
    looked = tmp.join(mulliken_join, on=["molecule_name", "atom_index"], how="left")
    df[out_col] = looked["mulliken_charge"].to_numpy()
    return df


X_train = add_mulliken(X_train, "atom_index_0", "mulliken_charge_0")
X_train = add_mulliken(X_train, "atom_index_1", "mulliken_charge_1")
X_test = add_mulliken(X_test, "atom_index_0", "mulliken_charge_0")
X_test = add_mulliken(X_test, "atom_index_1", "mulliken_charge_1")



## === cell 18
shield_join = shield.copy()
shield_join = shield_join.sort_values(["molecule_name", "atom_index"], kind="mergesort")
shield_join = shield_join.set_index(["molecule_name", "atom_index"], drop=False)
shield_cols = ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]


def add_shield(df, atom_col, suffix):
    tmp = df[["row_id", "molecule_name", atom_col]].copy()
    tmp = tmp.rename(columns={atom_col: "atom_index"})
    looked = tmp.join(
        shield_join[shield_cols], on=["molecule_name", "atom_index"], how="left"
    )
    for c in shield_cols:
        df[f"{c}_{suffix}"] = looked[c].to_numpy()
    return df


X_train = add_shield(X_train, "atom_index_0", "0")
X_train = add_shield(X_train, "atom_index_1", "1")
X_test = add_shield(X_test, "atom_index_0", "0")
X_test = add_shield(X_test, "atom_index_1", "1")



## === cell 19
X_train, X_test = convert_object_to_categories(X_train, X_test)



## === cell 20
for c in X_train.columns:
    if (
        str(X_train[c].dtype) == "category"
        and c in X_test.columns
        and str(X_test[c].dtype) == "category"
    ):
        X_test[c] = X_test[c].cat.set_categories(X_train[c].cat.categories)



## === cell 21
X_train_new = X_train.set_index("row_id", drop=True)
X_test_new = X_test.set_index("row_id", drop=True)



## === cell 22
num_cols_train = X_train_new.select_dtypes(include=[np.number]).columns
num_cols_test = X_test_new.select_dtypes(include=[np.number]).columns
X_train_new[num_cols_train] = X_train_new[num_cols_train].fillna(0.0)
X_test_new[num_cols_test] = X_test_new[num_cols_test].fillna(0.0)

if "scalar_coupling_constant" in X_train_new.columns:
    X_train_new = X_train_new.drop(columns=["scalar_coupling_constant"])
if "scalar_coupling_constant" in X_test_new.columns:
    X_test_new = X_test_new.drop(columns=["scalar_coupling_constant"])

if "id" in X_train_new.columns:
    X_train_new = X_train_new.drop(columns=["id"])
if "id" in X_test_new.columns:
    X_test_new = X_test_new.drop(columns=["id"])

X_test_new = X_test_new.reindex(columns=X_train_new.columns)

assert X_train_new.shape[0] == y_train.shape[0], "Train features/target row mismatch."
assert X_test_new.shape[0] == test.shape[0], "Test row mismatch after processing."
assert X_test_new.shape[1] > 0, "No feature columns available after alignment."



## === cell 23
cat_cols = [c for c in X_train_new.columns if str(X_train_new[c].dtype) == "category"]

lgb_params = dict(
    objective="regression_l1",
    n_estimators=1200,
    learning_rate=0.05,
    num_leaves=128,
    max_depth=-1,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
)

try:
    cv_sample_n = 300000
    if X_train_new.shape[0] > cv_sample_n:
        idx = np.random.RandomState(42).choice(
            X_train_new.index.values, size=cv_sample_n, replace=False
        )
        X_cv = X_train_new.loc[idx]
        y_cv = y_train.loc[idx]
        groups_cv = X_cv["molecule_name"].astype(str)
    else:
        X_cv = X_train_new
        y_cv = y_train
        groups_cv = X_train_new["molecule_name"].astype(str)

    gkf = GroupKFold(n_splits=3)
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(
        gkf.split(X_cv, y_cv, groups=groups_cv), start=1
    ):
        Xtr = X_cv.iloc[tr_idx]
        ytr = y_cv.iloc[tr_idx]
        Xva = X_cv.iloc[va_idx]
        yva = y_cv.iloc[va_idx]
        model = lgbm.LGBMRegressor(**lgb_params)
        model.fit(
            Xtr,
            ytr,
            categorical_feature=[c for c in cat_cols if c in Xtr.columns],
            eval_set=[(Xva, yva)],
            eval_metric="l1",
            callbacks=[lgbm.early_stopping(stopping_rounds=50, verbose=False)],
        )
        pva = model.predict(Xva)
        score = calc_overall_score(Xva["type"], yva.values, pva)
        fold_scores.append(score)
        print(f"GroupKFold fold {fold} logMAE-by-type score: {score:.5f}")
    print(f"Mean CV score (GroupKFold): {np.mean(fold_scores):.5f}")
except Exception as e:
    print("CV skipped due to:", repr(e))



## === cell 24
y_pred = np.zeros(X_test_new.shape[0], dtype=np.float64)

type_medians = train.groupby("type")["scalar_coupling_constant"].median().to_dict()
global_median = float(train["scalar_coupling_constant"].median())

train_types = X_train_new["type"].astype(str)
test_types = X_test_new["type"].astype(str)

train_type_to_idx = train_types.groupby(train_types, sort=False).indices
test_type_to_idx = test_types.groupby(test_types, sort=False).indices

feature_cols_no_type = [c for c in X_train_new.columns if c != "type"]
cat_cols_no_type = [c for c in cat_cols if c != "type"]

for t in sorted(test_type_to_idx.keys()):
    test_idx = np.fromiter(test_type_to_idx[t], dtype=np.int64)
    tr_list = train_type_to_idx.get(t, None)

    median_t = float(type_medians.get(t, global_median))

    if tr_list is None or len(tr_list) == 0:
        y_pred[test_idx] = median_t
        continue

    train_idx = np.fromiter(tr_list, dtype=np.int64)

    Xtr = X_train_new.iloc[train_idx][feature_cols_no_type]
    ytr = y_train.iloc[train_idx].astype(np.float64) - median_t
    Xte = X_test_new.iloc[test_idx][feature_cols_no_type]

    model = lgbm.LGBMRegressor(**lgb_params)
    model.fit(Xtr, ytr, categorical_feature=cat_cols_no_type)
    y_pred[test_idx] = model.predict(Xte) + median_t

y_predict = y_pred



## === cell 25
sub = pd.DataFrame({"id": test_id.values, "scalar_coupling_constant": y_predict})
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))
print("NaNs in predictions:", np.isnan(sub["scalar_coupling_constant"]).sum())
assert (
    sub.shape[0] == sample_sub.shape[0]
), "Submission row count mismatch vs sample_submission."
assert list(sub.columns) == [
    "id",
    "scalar_coupling_constant",
], "Submission columns incorrect."
