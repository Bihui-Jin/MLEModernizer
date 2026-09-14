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

train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
test = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
sample_sub = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))

structures = pd.read_csv(os.path.join(DATA_PATH, "structures.csv"))

mulliken = pd.read_csv(os.path.join(DATA_PATH, "mulliken_charges.csv"))
shield = pd.read_csv(os.path.join(DATA_PATH, "magnetic_shielding_tensors.csv"))

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
X_train = X_train.reset_index().rename(columns={"index": "row_id"})
X_test = X_test.reset_index().rename(columns={"index": "row_id"})




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
X_train = X_train.merge(
    structures,
    how="left",
    left_on=["molecule_name", "atom_index_0"],
    right_on=["molecule_name", "atom_index"],
    suffixes=("", "_s0"),
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
    how="left",
    left_on=["molecule_name", "atom_index_0"],
    right_on=["molecule_name", "atom_index"],
    suffixes=("", "_s0"),
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

X_train.head()



## === cell 10
X_train = X_train.merge(
    structures,
    how="left",
    left_on=["molecule_name", "atom_index_1"],
    right_on=["molecule_name", "atom_index"],
    suffixes=("", "_s1"),
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
    how="left",
    left_on=["molecule_name", "atom_index_1"],
    right_on=["molecule_name", "atom_index"],
    suffixes=("", "_s1"),
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

X_train.head()



## === cell 11
X_train = X_train.drop(columns=["atom_index_0_0", "atom_index_1_1"])
X_test = X_test.drop(columns=["atom_index_0_0", "atom_index_1_1"])



## === cell 12
X_train.head()



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
X_train["join_type"] = X_train["type"].astype(str).str.slice(0, 2)
X_test["join_type"] = X_test["type"].astype(str).str.slice(0, 2)



## === cell 15
print(X_train["atom_0"].unique())
print(X_test["atom_0"].unique())



## === cell 16
mol_num_atoms = (
    structures.groupby("molecule_name")["atom_index"]
    .max()
    .add(1)
    .astype(np.int32)
    .rename("num_atoms")
).reset_index()

X_train = X_train.merge(mol_num_atoms, on="molecule_name", how="left")
X_test = X_test.merge(mol_num_atoms, on="molecule_name", how="left")



## === cell 17
m0 = mulliken.rename(
    columns={"atom_index": "atom_index_0", "mulliken_charge": "mulliken_charge_0"}
)
m1 = mulliken.rename(
    columns={"atom_index": "atom_index_1", "mulliken_charge": "mulliken_charge_1"}
)
X_train = X_train.merge(m0, on=["molecule_name", "atom_index_0"], how="left")
X_train = X_train.merge(m1, on=["molecule_name", "atom_index_1"], how="left")
X_test = X_test.merge(m0, on=["molecule_name", "atom_index_0"], how="left")
X_test = X_test.merge(m1, on=["molecule_name", "atom_index_1"], how="left")



## === cell 18
s0 = shield.rename(columns={"atom_index": "atom_index_0"}).add_suffix("_0")
s0 = s0.rename(
    columns={"molecule_name_0": "molecule_name", "atom_index_0_0": "atom_index_0"}
)
s1 = shield.rename(columns={"atom_index": "atom_index_1"}).add_suffix("_1")
s1 = s1.rename(
    columns={"molecule_name_1": "molecule_name", "atom_index_1_1": "atom_index_1"}
)

X_train = X_train.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
X_train = X_train.merge(s1, on=["molecule_name", "atom_index_1"], how="left")
X_test = X_test.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
X_test = X_test.merge(s1, on=["molecule_name", "atom_index_1"], how="left")



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
X_train_new = X_train.set_index(keys="row_id")
X_test_new = X_test.set_index(keys="row_id")



## === cell 22
X_train_new = X_train_new.sort_index(axis=0)
X_test_new = X_test_new.sort_index(axis=0)



## === cell 23
expected_train_index = pd.RangeIndex(
    start=0, stop=train.shape[0], step=1, name="row_id"
)
expected_test_index = pd.RangeIndex(start=0, stop=test.shape[0], step=1, name="row_id")

X_train_new = X_train_new.reindex(expected_train_index)
X_test_new = X_test_new.reindex(expected_test_index)

num_cols_train = X_train_new.select_dtypes(include=[np.number]).columns
num_cols_test = X_test_new.select_dtypes(include=[np.number]).columns
X_train_new[num_cols_train] = X_train_new[num_cols_train].fillna(0.0)
X_test_new[num_cols_test] = X_test_new[num_cols_test].fillna(0.0)

if "scalar_coupling_constant" in X_train_new.columns:
    X_train_new = X_train_new.drop(columns=["scalar_coupling_constant"])
if "scalar_coupling_constant" in X_test_new.columns:
    X_test_new = X_test_new.drop(columns=["scalar_coupling_constant"])

if X_train_new.columns.duplicated().any():
    X_train_new = X_train_new.loc[:, ~X_train_new.columns.duplicated()]
if X_test_new.columns.duplicated().any():
    X_test_new = X_test_new.loc[:, ~X_test_new.columns.duplicated()]

if "id" in X_train_new.columns:
    X_train_new = X_train_new.drop(columns=["id"])
if "id" in X_test_new.columns:
    X_test_new = X_test_new.drop(columns=["id"])

X_test_new = X_test_new.reindex(columns=X_train_new.columns)

assert X_train_new.shape[0] == y_train.shape[0], "Train features/target row mismatch."
assert X_test_new.shape[0] == test.shape[0], "Test row mismatch after processing."
assert X_test_new.shape[1] > 0, "No feature columns available after alignment."



## === cell 24
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

    cross_val_grouped_by_molecule(
        X_cv,
        y_cv,
        groups=groups_cv,
        cat_cols=[c for c in cat_cols if c in X_cv.columns],
        n_splits=3,
        lgb_params=lgb_params,
    )
except Exception as e:
    print("CV skipped due to:", repr(e))



## === cell 25
y_pred = np.zeros(X_test_new.shape[0], dtype=np.float64)

type_medians = train.groupby("type")["scalar_coupling_constant"].median().to_dict()
global_median = float(train["scalar_coupling_constant"].median())

train_types = X_train_new["type"].astype(str)
test_types = X_test_new["type"].astype(str)

for t in sorted(test_types.unique()):
    test_mask = (test_types == t).values
    train_mask = (train_types == t).values

    median_t = float(type_medians.get(t, global_median))

    if train_mask.sum() == 0:
        y_pred[test_mask] = median_t
        continue

    Xtr = X_train_new.loc[train_mask].drop(columns=["type"])
    ytr = y_train.loc[train_mask].astype(np.float64) - median_t
    Xte = X_test_new.loc[test_mask].drop(columns=["type"])

    cat_cols_sub = [c for c in cat_cols if c in Xtr.columns]

    model = lgbm.LGBMRegressor(**lgb_params)
    model.fit(Xtr, ytr, categorical_feature=cat_cols_sub)
    y_pred[test_mask] = model.predict(Xte) + median_t

y_predict = y_pred



## === cell 26
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
