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

# 5. Code solution

## === cell 0
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
import os

INPUT_CANDIDATES = [
    "../input",
    "/kaggle/input",
    "/kaggle/data/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling/champs-scalar-coupling",
]


def resolve_path(filename: str) -> str:
    for base in INPUT_CANDIDATES:
        p = os.path.join(base, filename)
        if os.path.exists(p):
            return p
    return os.path.join("../input", filename)


print("Listing ../input (if exists):")
try:
    print(os.listdir("../input"))
except Exception as e:
    print("Could not list ../input:", e)



## === cell 1
train_path = resolve_path("train.csv")
trainSet = pd.read_csv(train_path)
print(trainSet.head())



## === cell 2
test_path = resolve_path("test.csv")
testSet = pd.read_csv(test_path)
print(testSet.head())



## === cell 3
structures_path = resolve_path("structures.csv")
structures = pd.read_csv(structures_path)
print(structures.head())




## === cell 4
def map_atom_info(df, atom_idx):
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


trainSet = map_atom_info(trainSet, 0)
trainSet = map_atom_info(trainSet, 1)

testSet = map_atom_info(testSet, 0)
testSet = map_atom_info(testSet, 1)



## === cell 5
print(trainSet.head())
print(testSet.head())



## === cell 6
coord_cols = ["x_0", "y_0", "z_0", "x_1", "y_1", "z_1"]
for df_name, df in [("trainSet", trainSet), ("testSet", testSet)]:
    missing_coords = df[coord_cols].isna().any(axis=1).sum()
    if missing_coords > 0:
        print(
            f"Warning: {df_name} has rows with missing coordinates:",
            int(missing_coords),
        )
        df[coord_cols] = df[coord_cols].fillna(0.0)

train_p0 = trainSet[["x_0", "y_0", "z_0"]].values
train_p1 = trainSet[["x_1", "y_1", "z_1"]].values
test_p0 = testSet[["x_0", "y_0", "z_0"]].values
test_p1 = testSet[["x_1", "y_1", "z_1"]].values

trainSet["dx"] = trainSet["x_0"] - trainSet["x_1"]
trainSet["dy"] = trainSet["y_0"] - trainSet["y_1"]
trainSet["dz"] = trainSet["z_0"] - trainSet["z_1"]

testSet["dx"] = testSet["x_0"] - testSet["x_1"]
testSet["dy"] = testSet["y_0"] - testSet["y_1"]
testSet["dz"] = testSet["z_0"] - testSet["z_1"]

trainSet["dist"] = np.linalg.norm(train_p0 - train_p1, axis=1)
testSet["dist"] = np.linalg.norm(test_p0 - test_p1, axis=1)

eps = 1e-3
trainSet["inv_dist"] = 1.0 / (trainSet["dist"] + eps)
testSet["inv_dist"] = 1.0 / (testSet["dist"] + eps)

type_mean_dist = trainSet.groupby("type")["dist"].mean()
global_mean_dist = float(trainSet["dist"].mean())

train_den = trainSet["type"].map(type_mean_dist)
test_den = testSet["type"].map(type_mean_dist).fillna(global_mean_dist)

trainSet["dist_to_type_mean"] = trainSet["dist"] / train_den
testSet["dist_to_type_mean"] = testSet["dist"] / test_den

for df in (trainSet, testSet):
    df["dist_to_type_mean"] = df["dist_to_type_mean"].replace([np.inf, -np.inf], np.nan)
    df["dist_to_type_mean"] = df["dist_to_type_mean"].fillna(1.0)
    df["inv_dist"] = df["inv_dist"].replace([np.inf, -np.inf], np.nan).fillna(0.0)



## === cell 7
train_atom0_unique = trainSet["atom_0"].dropna().unique()
test_atom0_unique = testSet["atom_0"].dropna().unique()

if not (len(train_atom0_unique) == 1 and train_atom0_unique[0] == "H"):
    print("Warning: train atom_0 is not always H. Unique values:", train_atom0_unique)
if not (len(test_atom0_unique) == 1 and test_atom0_unique[0] == "H"):
    print("Warning: test atom_0 is not always H. Unique values:", test_atom0_unique)



## === cell 8
print(trainSet["atom_1"].astype("category").cat.categories)
print(testSet["atom_1"].astype("category").cat.categories)



## === cell 9
print(testSet["type"].astype("category").cat.categories)
print(trainSet["type"].astype("category").cat.categories)



## === cell 10
type_categories = trainSet["type"].astype("category").cat.categories.values
for i in type_categories:
    col = "type_" + str(i)
    trainSet[col] = (trainSet["type"] == i).astype(np.int8)
    testSet[col] = (testSet["type"] == i).astype(np.int8)

extra_test_types = set(testSet["type"].unique()) - set(type_categories)
if len(extra_test_types) > 0:
    print(
        "Warning: unseen types in test not present in train:",
        sorted(list(extra_test_types)),
    )



## === cell 11
base_model_params = dict(
    fit_intercept=True,
    max_iter=1000,
    tol=1e-5,
    warm_start=False,
)



## === cell 12
atom0_cats = trainSet["atom_0"].astype("category").cat.categories.values
atom1_cats = trainSet["atom_1"].astype("category").cat.categories.values

for a in atom0_cats:
    col = f"atom0_{a}"
    trainSet[col] = (trainSet["atom_0"] == a).astype(np.int8)
    testSet[col] = (testSet["atom_0"] == a).astype(np.int8)

for a in atom1_cats:
    col = f"atom1_{a}"
    trainSet[col] = (trainSet["atom_1"] == a).astype(np.int8)
    testSet[col] = (testSet["atom_1"] == a).astype(np.int8)




## === cell 13
def add_external_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    dipole_path = resolve_path("dipole_moments.csv")
    pe_path = resolve_path("potential_energy.csv")

    dipole = pd.read_csv(dipole_path)  # molecule_name, X, Y, Z
    dipole = dipole.rename(columns={"X": "dipole_X", "Y": "dipole_Y", "Z": "dipole_Z"})
    df = df.merge(dipole, on="molecule_name", how="left")

    pe = pd.read_csv(pe_path)  # molecule_name, potential_energy
    df = df.merge(pe, on="molecule_name", how="left")

    mull_path = resolve_path("mulliken_charges.csv")
    mull = pd.read_csv(mull_path)  # molecule_name, atom_index, mulliken_charge

    mull0 = mull.rename(
        columns={
            "atom_index": "atom_index_0",
            "mulliken_charge": "mulliken_charge_0",
        }
    )
    mull1 = mull.rename(
        columns={
            "atom_index": "atom_index_1",
            "mulliken_charge": "mulliken_charge_1",
        }
    )

    df = df.merge(mull0, on=["molecule_name", "atom_index_0"], how="left")
    df = df.merge(mull1, on=["molecule_name", "atom_index_1"], how="left")

    mst_path = resolve_path("magnetic_shielding_tensors.csv")
    mst = pd.read_csv(mst_path)

    tensor_cols = ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]
    mst["mst_trace"] = mst["XX"] + mst["YY"] + mst["ZZ"]
    mst["mst_mean"] = mst[tensor_cols].mean(axis=1)
    mst["mst_std"] = mst[tensor_cols].std(axis=1)
    mst_small = mst[["molecule_name", "atom_index", "mst_trace", "mst_mean", "mst_std"]]

    mst0 = mst_small.rename(
        columns={
            "atom_index": "atom_index_0",
            "mst_trace": "mst_trace_0",
            "mst_mean": "mst_mean_0",
            "mst_std": "mst_std_0",
        }
    )
    mst1 = mst_small.rename(
        columns={
            "atom_index": "atom_index_1",
            "mst_trace": "mst_trace_1",
            "mst_mean": "mst_mean_1",
            "mst_std": "mst_std_1",
        }
    )

    df = df.merge(mst0, on=["molecule_name", "atom_index_0"], how="left")
    df = df.merge(mst1, on=["molecule_name", "atom_index_1"], how="left")

    df["mulliken_charge_diff"] = df["mulliken_charge_0"] - df["mulliken_charge_1"]
    df["mulliken_charge_sum"] = df["mulliken_charge_0"] + df["mulliken_charge_1"]
    df["mst_trace_diff"] = df["mst_trace_0"] - df["mst_trace_1"]
    df["mst_trace_sum"] = df["mst_trace_0"] + df["mst_trace_1"]

    return df


trainSet = add_external_features(trainSet)
testSet = add_external_features(testSet)

ext_num_cols = [
    "dipole_X",
    "dipole_Y",
    "dipole_Z",
    "potential_energy",
    "mulliken_charge_0",
    "mulliken_charge_1",
    "mulliken_charge_diff",
    "mulliken_charge_sum",
    "mst_trace_0",
    "mst_mean_0",
    "mst_std_0",
    "mst_trace_1",
    "mst_mean_1",
    "mst_std_1",
    "mst_trace_diff",
    "mst_trace_sum",
]
for df_name, df in [("trainSet", trainSet), ("testSet", testSet)]:
    for c in ext_num_cols:
        if c not in df.columns:
            df[c] = 0.0
    nonfinite = (~np.isfinite(df[ext_num_cols].to_numpy(dtype=np.float64))).sum()
    if nonfinite > 0:
        print(
            f"Warning: {df_name} has non-finite external feature values; filling with 0. Count:",
            int(nonfinite),
        )
        df[ext_num_cols] = (
            df[ext_num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )



## === cell 14
type_feature_cols = ["type_" + str(i) for i in type_categories]
atom_feature_cols = [f"atom0_{a}" for a in atom0_cats] + [
    f"atom1_{a}" for a in atom1_cats
]

feature_cols = (
    type_feature_cols
    + atom_feature_cols
    + [
        "dist",
        "dist_to_type_mean",
        "inv_dist",
        "dx",
        "dy",
        "dz",
    ]
    + ext_num_cols
)

for c in feature_cols:
    if c not in trainSet.columns:
        trainSet[c] = 0
    if c not in testSet.columns:
        testSet[c] = 0

numeric_feature_cols = [
    "dist",
    "dist_to_type_mean",
    "inv_dist",
    "dx",
    "dy",
    "dz",
] + ext_num_cols

train_num = trainSet[numeric_feature_cols].to_numpy(dtype=np.float64)
train_num = np.nan_to_num(train_num, nan=0.0, posinf=0.0, neginf=0.0)

num_mean = train_num.mean(axis=0)
num_std = train_num.std(axis=0)
num_std = np.where(num_std == 0.0, 1.0, num_std)


def apply_standardization(df: pd.DataFrame) -> None:
    arr = df[numeric_feature_cols].to_numpy(dtype=np.float64, copy=True)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = (arr - num_mean) / num_std
    df.loc[:, numeric_feature_cols] = arr


apply_standardization(trainSet)
apply_standardization(testSet)

X_train_all = trainSet[feature_cols].to_numpy(dtype=np.float64)
y_train_all = trainSet["scalar_coupling_constant"].to_numpy(dtype=np.float64)

if not np.isfinite(X_train_all).all():
    bad = ~np.isfinite(X_train_all)
    print(
        "Warning: non-finite values in training features; sanitizing to 0.0. Count:",
        int(bad.sum()),
    )
    X_train_all = np.nan_to_num(X_train_all, nan=0.0, posinf=0.0, neginf=0.0)




## === cell 15
def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)
    types = pd.Series(types)
    maes = (y_true - y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()




## === cell 16
global_model = HuberRegressor(**base_model_params).fit(X_train_all, y_train_all)

type_models = {}
for t in sorted(trainSet["type"].unique()):
    mask = trainSet["type"].values == t
    X_t = trainSet.loc[mask, feature_cols].to_numpy(dtype=np.float64)
    y_t = trainSet.loc[mask, "scalar_coupling_constant"].to_numpy(dtype=np.float64)
    if not np.isfinite(X_t).all():
        X_t = np.nan_to_num(X_t, nan=0.0, posinf=0.0, neginf=0.0)

    type_models[t] = HuberRegressor(**base_model_params).fit(X_t, y_t)

print("Trained per-type models:", len(type_models), "Global fallback model: yes")



## === cell 17
train_pred_global = global_model.predict(X_train_all)

train_pred_per_type = np.empty(len(trainSet), dtype=np.float64)
for t, model in type_models.items():
    m = trainSet["type"].values == t
    X_m = trainSet.loc[m, feature_cols].to_numpy(dtype=np.float64)
    if not np.isfinite(X_m).all():
        X_m = np.nan_to_num(X_m, nan=0.0, posinf=0.0, neginf=0.0)
    train_pred_per_type[m] = model.predict(X_m)

print(
    "In-sample group logMAE (global model):",
    group_mean_log_mae(y_train_all, train_pred_global, trainSet["type"]),
)
print(
    "In-sample group logMAE (per-type models):",
    group_mean_log_mae(y_train_all, train_pred_per_type, trainSet["type"]),
)



## === cell 18
X_test_all = testSet[feature_cols].to_numpy(dtype=np.float64)

if not np.isfinite(X_test_all).all():
    col_nonfinite = (~np.isfinite(testSet[feature_cols])).sum()
    col_nonfinite = col_nonfinite[col_nonfinite > 0]
    print("Non-finite counts in test features (by column):\n", col_nonfinite)
    X_test_all = np.nan_to_num(X_test_all, nan=0.0, posinf=0.0, neginf=0.0)

test_pred = np.empty(len(testSet), dtype=np.float64)
test_types = testSet["type"].values

unseen_mask_total = np.zeros(len(testSet), dtype=bool)
for t in np.unique(test_types):
    m = test_types == t
    X_m = testSet.loc[m, feature_cols].to_numpy(dtype=np.float64)
    if not np.isfinite(X_m).all():
        X_m = np.nan_to_num(X_m, nan=0.0, posinf=0.0, neginf=0.0)
    if t in type_models:
        test_pred[m] = type_models[t].predict(X_m)
    else:
        unseen_mask_total[m] = True

if unseen_mask_total.any():
    X_u = testSet.loc[unseen_mask_total, feature_cols].to_numpy(dtype=np.float64)
    if not np.isfinite(X_u).all():
        X_u = np.nan_to_num(X_u, nan=0.0, posinf=0.0, neginf=0.0)
    test_pred[unseen_mask_total] = global_model.predict(X_u)
    print(
        "Used global fallback for unseen test types count:",
        int(unseen_mask_total.sum()),
    )

resultSet = pd.DataFrame(
    {
        "id": testSet["id"].values,
        "scalar_coupling_constant": test_pred,
    }
)



## === cell 19
out_path = "submission.csv"
resultSet.to_csv(out_path, index=False)

with open(out_path, "r") as f:
    for i, line in enumerate(f):
        print(line.strip())
        if i > 5:
            break
print("Wrote:", out_path, "rows:", len(resultSet))
