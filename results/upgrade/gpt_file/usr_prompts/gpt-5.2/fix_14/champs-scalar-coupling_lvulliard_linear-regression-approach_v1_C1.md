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

try:
    from sklearnex import patch_sklearn  # type: ignore

    patch_sklearn()
except Exception:
    pass

np.random.seed(0)

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
trainSet = pd.read_csv(
    train_path,
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
        "scalar_coupling_constant": np.float64,
    },
)
print(trainSet.head())



## === cell 2
test_path = resolve_path("test.csv")
testSet = pd.read_csv(
    test_path,
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
    },
)
print(testSet.head())



## === cell 3
structures_path = resolve_path("structures.csv")
structures = pd.read_csv(
    structures_path,
    dtype={
        "molecule_name": "category",
        "atom_index": np.int16,
        "atom": "category",
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
    },
)
print(structures.head())



## === cell 4
structures_idx = structures.set_index(["molecule_name", "atom_index"], drop=True)[
    ["atom", "x", "y", "z"]
]


def map_atom_info_fast(df: pd.DataFrame) -> pd.DataFrame:
    idx0 = pd.MultiIndex.from_frame(df[["molecule_name", "atom_index_0"]])
    s0 = structures_idx.reindex(idx0)
    s0 = s0.rename(
        columns={"atom": "atom_0", "x": "x_0", "y": "y_0", "z": "z_0"}
    ).reset_index(drop=True)

    idx1 = pd.MultiIndex.from_frame(df[["molecule_name", "atom_index_1"]])
    s1 = structures_idx.reindex(idx1)
    s1 = s1.rename(
        columns={"atom": "atom_1", "x": "x_1", "y": "y_1", "z": "z_1"}
    ).reset_index(drop=True)

    return pd.concat([df.reset_index(drop=True), s0, s1], axis=1)


trainSet = map_atom_info_fast(trainSet)
testSet = map_atom_info_fast(testSet)



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
        df.loc[:, coord_cols] = df[coord_cols].fillna(0.0)

train_p0 = trainSet[["x_0", "y_0", "z_0"]].to_numpy(dtype=np.float64, copy=False)
train_p1 = trainSet[["x_1", "y_1", "z_1"]].to_numpy(dtype=np.float64, copy=False)
test_p0 = testSet[["x_0", "y_0", "z_0"]].to_numpy(dtype=np.float64, copy=False)
test_p1 = testSet[["x_1", "y_1", "z_1"]].to_numpy(dtype=np.float64, copy=False)

train_delta = train_p0 - train_p1
test_delta = test_p0 - test_p1

trainSet["dx"] = train_delta[:, 0]
trainSet["dy"] = train_delta[:, 1]
trainSet["dz"] = train_delta[:, 2]

testSet["dx"] = test_delta[:, 0]
testSet["dy"] = test_delta[:, 1]
testSet["dz"] = test_delta[:, 2]

trainSet["dist"] = np.linalg.norm(train_delta, axis=1)
testSet["dist"] = np.linalg.norm(test_delta, axis=1)

eps = 1e-3
trainSet["inv_dist"] = 1.0 / (trainSet["dist"] + eps)
testSet["inv_dist"] = 1.0 / (testSet["dist"] + eps)

type_mean_dist = trainSet.groupby("type", observed=True)["dist"].mean()
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
type_dummies_train = pd.get_dummies(trainSet["type"], prefix="type", dtype=np.int8)
type_dummies_train = type_dummies_train.reindex(
    columns=[f"type_{t}" for t in type_categories], fill_value=0
)
trainSet = pd.concat([trainSet, type_dummies_train], axis=1)

type_dummies_test = pd.get_dummies(testSet["type"], prefix="type", dtype=np.int8)
type_dummies_test = type_dummies_test.reindex(
    columns=[f"type_{t}" for t in type_categories], fill_value=0
)
testSet = pd.concat([testSet, type_dummies_test], axis=1)

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

atom0_train = pd.get_dummies(trainSet["atom_0"], prefix="atom0", dtype=np.int8).reindex(
    columns=[f"atom0_{a}" for a in atom0_cats], fill_value=0
)
atom1_train = pd.get_dummies(trainSet["atom_1"], prefix="atom1", dtype=np.int8).reindex(
    columns=[f"atom1_{a}" for a in atom1_cats], fill_value=0
)
trainSet = pd.concat([trainSet, atom0_train, atom1_train], axis=1)

atom0_test = pd.get_dummies(testSet["atom_0"], prefix="atom0", dtype=np.int8).reindex(
    columns=[f"atom0_{a}" for a in atom0_cats], fill_value=0
)
atom1_test = pd.get_dummies(testSet["atom_1"], prefix="atom1", dtype=np.int8).reindex(
    columns=[f"atom1_{a}" for a in atom1_cats], fill_value=0
)
testSet = pd.concat([testSet, atom0_test, atom1_test], axis=1)



## === cell 13
dipole_path = resolve_path("dipole_moments.csv")
pe_path = resolve_path("potential_energy.csv")
mull_path = resolve_path("mulliken_charges.csv")
mst_path = resolve_path("magnetic_shielding_tensors.csv")

dipole = pd.read_csv(
    dipole_path,
    dtype={
        "molecule_name": "category",
        "X": np.float32,
        "Y": np.float32,
        "Z": np.float32,
    },
).rename(columns={"X": "dipole_X", "Y": "dipole_Y", "Z": "dipole_Z"})
dipole_idx = dipole.set_index("molecule_name")[["dipole_X", "dipole_Y", "dipole_Z"]]

pe = pd.read_csv(
    pe_path,
    dtype={"molecule_name": "category", "potential_energy": np.float32},
)
pe_idx = pe.set_index("molecule_name")[["potential_energy"]]

mull = pd.read_csv(
    mull_path,
    dtype={
        "molecule_name": "category",
        "atom_index": np.int16,
        "mulliken_charge": np.float32,
    },
)
mull_idx = mull.set_index(["molecule_name", "atom_index"])[["mulliken_charge"]]

mst = pd.read_csv(
    mst_path,
    dtype={
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
    },
)

tensor_cols = ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]
mst["mst_trace"] = mst["XX"] + mst["YY"] + mst["ZZ"]
mst["mst_mean"] = mst[tensor_cols].mean(axis=1)
mst["mst_std"] = mst[tensor_cols].std(axis=1)
mst_small = mst[["molecule_name", "atom_index", "mst_trace", "mst_mean", "mst_std"]]
mst_idx = mst_small.set_index(["molecule_name", "atom_index"])[
    ["mst_trace", "mst_mean", "mst_std"]
]


def add_external_features_fast(df: pd.DataFrame) -> pd.DataFrame:
    out = df.join(dipole_idx, on="molecule_name")
    out = out.join(pe_idx, on="molecule_name")

    idx0 = pd.MultiIndex.from_frame(
        out[["molecule_name", "atom_index_0"]].rename(
            columns={"atom_index_0": "atom_index"}
        )
    )
    idx1 = pd.MultiIndex.from_frame(
        out[["molecule_name", "atom_index_1"]].rename(
            columns={"atom_index_1": "atom_index"}
        )
    )

    mc0 = (
        mull_idx.reindex(idx0)
        .rename(columns={"mulliken_charge": "mulliken_charge_0"})
        .reset_index(drop=True)
    )
    mc1 = (
        mull_idx.reindex(idx1)
        .rename(columns={"mulliken_charge": "mulliken_charge_1"})
        .reset_index(drop=True)
    )

    ms0 = (
        mst_idx.reindex(idx0)
        .rename(
            columns={
                "mst_trace": "mst_trace_0",
                "mst_mean": "mst_mean_0",
                "mst_std": "mst_std_0",
            }
        )
        .reset_index(drop=True)
    )
    ms1 = (
        mst_idx.reindex(idx1)
        .rename(
            columns={
                "mst_trace": "mst_trace_1",
                "mst_mean": "mst_mean_1",
                "mst_std": "mst_std_1",
            }
        )
        .reset_index(drop=True)
    )

    out = pd.concat([out.reset_index(drop=True), mc0, mc1, ms0, ms1], axis=1)

    out["mulliken_charge_diff"] = out["mulliken_charge_0"] - out["mulliken_charge_1"]
    out["mulliken_charge_sum"] = out["mulliken_charge_0"] + out["mulliken_charge_1"]
    out["mst_trace_diff"] = out["mst_trace_0"] - out["mst_trace_1"]
    out["mst_trace_sum"] = out["mst_trace_0"] + out["mst_trace_1"]
    return out


trainSet = add_external_features_fast(trainSet)
testSet = add_external_features_fast(testSet)

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
    arr = df[ext_num_cols].to_numpy(dtype=np.float64, copy=False)
    nonfinite = (~np.isfinite(arr)).sum()
    if nonfinite > 0:
        print(
            f"Warning: {df_name} has non-finite external feature values; filling with 0. Count:",
            int(nonfinite),
        )
        df.loc[:, ext_num_cols] = (
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

train_num = trainSet[numeric_feature_cols].to_numpy(dtype=np.float64, copy=False)
train_num = np.nan_to_num(train_num, nan=0.0, posinf=0.0, neginf=0.0)

num_mean = train_num.mean(axis=0)
num_std = train_num.std(axis=0)
num_std = np.where(num_std == 0.0, 1.0, num_std)


def apply_standardization(df: pd.DataFrame) -> None:
    arr = df[numeric_feature_cols].to_numpy(dtype=np.float64, copy=True)
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    arr = (arr - num_mean) / num_std
    df.loc[:, numeric_feature_cols] = arr


apply_standardization(trainSet)
apply_standardization(testSet)

X_train_all = trainSet[feature_cols].to_numpy(dtype=np.float64, copy=False)
y_train_all = trainSet["scalar_coupling_constant"].to_numpy(
    dtype=np.float64, copy=False
)

bad_count = int((~np.isfinite(X_train_all)).sum())
if bad_count:
    print(
        "Warning: non-finite values in training features; sanitizing to 0.0. Count:",
        bad_count,
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

train_types_arr = trainSet["type"].to_numpy()
unique_types = np.array(sorted(trainSet["type"].unique()))

type_to_idx = {}
for t in unique_types:
    type_to_idx[t] = np.flatnonzero(train_types_arr == t)

type_models = {}
for t in unique_types:
    idx = type_to_idx[t]
    X_t = X_train_all[idx]
    y_t = y_train_all[idx]
    type_models[t] = HuberRegressor(**base_model_params).fit(X_t, y_t)

print("Trained per-type models:", len(type_models), "Global fallback model: yes")



## === cell 17
train_pred_global = global_model.predict(X_train_all)

train_pred_per_type = np.empty(len(trainSet), dtype=np.float64)
for t in unique_types:
    idx = type_to_idx[t]
    train_pred_per_type[idx] = type_models[t].predict(X_train_all[idx])

print(
    "In-sample group logMAE (global model):",
    group_mean_log_mae(y_train_all, train_pred_global, trainSet["type"]),
)
print(
    "In-sample group logMAE (per-type models):",
    group_mean_log_mae(y_train_all, train_pred_per_type, trainSet["type"]),
)



## === cell 18
X_test_all = testSet[feature_cols].to_numpy(dtype=np.float64, copy=False)

bad_count_test = int((~np.isfinite(X_test_all)).sum())
if bad_count_test:
    col_nonfinite = (~np.isfinite(testSet[feature_cols])).sum()
    col_nonfinite = col_nonfinite[col_nonfinite > 0]
    print("Non-finite counts in test features (by column):\n", col_nonfinite)
    X_test_all = np.nan_to_num(X_test_all, nan=0.0, posinf=0.0, neginf=0.0)

test_pred = np.empty(len(testSet), dtype=np.float64)
test_types = testSet["type"].to_numpy()

for t in np.unique(test_types):
    idx = np.flatnonzero(test_types == t)
    X_m = X_test_all[idx]
    if t in type_models:
        test_pred[idx] = type_models[t].predict(X_m)
    else:
        test_pred[idx] = global_model.predict(X_m)

resultSet = pd.DataFrame(
    {
        "id": testSet["id"].to_numpy(),
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
