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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.7858

# 6. Current score

2.01286

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.88801) has done: 'I fix the categorical conversion bug by ensuring category lists never include null values, which currently causes the `ValueError: Categorical categories cannot be null`. Then I make LightGBM accept the categorical columns by explicitly casting the intended categorical features to pandas `category` dtype (instead of leaving them as `object`), which resolves the “bad pandas dtypes” error during `.fit()`. Finally, I ensure training completes and `y_predict` is defined so the pipeline always writes a valid `submission.csv` with the correct columns and row count.'
- What this solution (achieved 2.01286) has done: 'Your current model is trained on raw `molecule_name` as a categorical feature, which encourages memorization and hurts generalization because train/test are split by molecule; dropping that single column typically moves the score substantially toward your target without changing the overall approach. I also make the train/valid split in your (optional) CV routine molecule-grouped (to match the competition split) so you can sanity-check improvements locally without leakage. Finally, I keep everything else (LightGBM regressor, features, training flow, submission writing) the same, only adding a couple of stable training parameters so categorical handling is consistent.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import gc
import lightgbm as lgbm
from sklearn.model_selection import KFold, GroupKFold
import os

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_INPUT = "../input"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "/kaggle/input/champs-scalar-coupling"

print("Using BASE_INPUT =", BASE_INPUT)
print(os.listdir(BASE_INPUT)[:20])



## === cell 1
gc.collect()



## === cell 2
train = pd.read_csv(f"{BASE_INPUT}/train.csv")
test = pd.read_csv(f"{BASE_INPUT}/test.csv")
sample_sub = pd.read_csv(f"{BASE_INPUT}/sample_submission.csv")
structures = pd.read_csv(f"{BASE_INPUT}/structures.csv")

print(f"train.shape: {train.shape}")
print(f"test.shape: {test.shape}")
print(f"structures.shape: {structures.shape}")



## === cell 3
train_id = train["id"].copy()
test_id = test["id"].copy()

X_train = train.drop(columns=["scalar_coupling_constant"]).copy()
y_train = train["scalar_coupling_constant"].copy()
X_test = test.copy()

print(f"X_train.shape: {X_train.shape}")
print(f"X_test.shape: {X_test.shape}")



## === cell 4
X_train = X_train.drop(columns=["id"])
X_test = X_test.drop(columns=["id"])



## === cell 5
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
X_train["orig_idx"] = np.arange(len(X_train), dtype=np.int64)
X_test["orig_idx"] = np.arange(len(X_test), dtype=np.int64)




## === cell 6
def convert_object_to_categories(X_train_df, X_test_df):
    for col in X_train_df.columns:
        if X_train_df[col].dtype == "O" or str(X_train_df[col].dtype) == "object":
            combined = pd.concat(
                [X_train_df[col], X_test_df[col]], axis=0, ignore_index=True
            )
            cats = pd.Index(combined.dropna().unique())
            X_train_df[col] = pd.Categorical(X_train_df[col], categories=cats)
            X_test_df[col] = pd.Categorical(X_test_df[col], categories=cats)
    return X_train_df, X_test_df


X_train, X_test = convert_object_to_categories(X_train, X_test)



## === cell 7
import math

print(f"{X_train['type'].unique()}")
print(f"{X_test['type'].unique()}")


def calc_score(X_val, y_true, y_pred):
    tmp = X_val[["type"]].copy()
    tmp["y_true"] = np.asarray(y_true)
    tmp["y_pred"] = np.asarray(y_pred)
    tmp["error"] = (tmp["y_true"] - tmp["y_pred"]).abs()

    g = tmp.groupby("type")["error"].mean()
    g = np.log(np.maximum(g.values, 1e-12))
    return float(np.mean(g))




## === cell 8
def cross_val_grouped_by_molecule(X, y, groups):
    gkf = GroupKFold(n_splits=5)
    fold = 0
    for train_index, val_index in gkf.split(X, y, groups=groups):
        fold += 1
        lgbm_model = lgbm.LGBMRegressor(
            random_state=RANDOM_STATE,
            n_estimators=200,
            objective="regression",
        )
        lgbm_model.fit(X.iloc[train_index, :], y.iloc[train_index])
        y_val = lgbm_model.predict(X.iloc[val_index, :])
        print(
            f"fold {fold} score: {calc_score(X.iloc[val_index, :], y.iloc[val_index], y_val)}"
        )




## === cell 9
assert len(X_train) == len(y_train)
assert len(X_test) > 0
assert (
    "molecule_name" in X_train.columns
    and "atom_index_0" in X_train.columns
    and "atom_index_1" in X_train.columns
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
X_train["join_type"] = X_train["type"].astype(str).str.slice(0, 2)
X_test["join_type"] = X_test["type"].astype(str).str.slice(0, 2)



## === cell 15
X_train["num_atoms"] = X_train.groupby(["molecule_name", "atom_1"])["atom_1"].transform(
    "count"
)
X_test["num_atoms"] = X_test.groupby(["molecule_name", "atom_1"])["atom_1"].transform(
    "count"
)



## === cell 16
X_train["num_atoms"] = X_train["num_atoms"].astype("str") + X_train["atom_1"].astype(
    str
)
X_test["num_atoms"] = X_test["num_atoms"].astype("str") + X_test["atom_1"].astype(str)



## === cell 17
X_train, X_test = convert_object_to_categories(X_train, X_test)

for col in X_train.columns:
    if X_train[col].dtype == "O" or str(X_train[col].dtype) == "object":
        combined = pd.concat([X_train[col], X_test[col]], axis=0, ignore_index=True)
        cats = pd.Index(combined.dropna().unique())
        X_train[col] = pd.Categorical(X_train[col], categories=cats)
        X_test[col] = pd.Categorical(X_test[col], categories=cats)



## === cell 18
df = X_train.merge(
    pd.DataFrame(
        {
            "scalar_coupling_constant": y_train.values,
            "orig_idx": X_train["orig_idx"].values,
        }
    ),
    on="orig_idx",
    how="left",
)
numeric_corr = df.select_dtypes(include=[np.number]).corr()
print("numeric_corr computed with shape:", numeric_corr.shape)



## === cell 19
X_train_new = X_train.set_index("orig_idx").sort_index()
X_test_new = X_test.set_index("orig_idx").sort_index()

assert X_train_new.shape[0] == len(y_train)
assert X_test_new.shape[0] == len(test)



## === cell 20
X_train_new = X_train_new.drop(columns=["molecule_name"])
X_test_new = X_test_new.drop(columns=["molecule_name"])

cat_cols = [c for c in X_train_new.columns if str(X_train_new[c].dtype) == "category"]
print("Categorical columns:", cat_cols)

lgbm_model = lgbm.LGBMRegressor(
    random_state=RANDOM_STATE,
    n_estimators=200,
    objective="regression",
)
lgbm_model.fit(X_train_new, y_train, categorical_feature=cat_cols)

y_predict = lgbm_model.predict(X_test_new)

print(
    "Predictions:",
    y_predict.shape,
    "min/max:",
    float(np.min(y_predict)),
    float(np.max(y_predict)),
)



## === cell 21
submission = pd.DataFrame({"id": test_id.values, "scalar_coupling_constant": y_predict})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
assert submission.shape[0] == sample_sub.shape[0]
assert list(submission.columns) == ["id", "scalar_coupling_constant"]
assert submission["id"].is_unique
