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

1.19404

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 2.9981) has done: 'I (1) fix the broken hydrogen assertion by checking actual values rather than comparing category lists (which can be empty or differently shaped), (2) eliminate NaNs in test features by computing the `dist_to_type_mean` normalization using **train** type means and falling back safely for any unseen types, and (3) ensure train/test one-hot type columns are aligned (missing type columns get filled with 0). These changes are minimal, keep the same model and features, unblock inference, and should also improve score stability by preventing NaN-driven failures. Finally, the script write a valid Kaggle submission CSV with the required filename suffix and columns.'

# 9. Code solution

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
train_p0 = trainSet[["x_0", "y_0", "z_0"]].values
train_p1 = trainSet[["x_1", "y_1", "z_1"]].values
test_p0 = testSet[["x_0", "y_0", "z_0"]].values
test_p1 = testSet[["x_1", "y_1", "z_1"]].values

trainSet["dist"] = np.linalg.norm(train_p0 - train_p1, axis=1)
testSet["dist"] = np.linalg.norm(test_p0 - test_p1, axis=1)

type_mean_dist = trainSet.groupby("type")["dist"].mean()
global_mean_dist = float(trainSet["dist"].mean())

train_den = trainSet["type"].map(type_mean_dist)
test_den = testSet["type"].map(type_mean_dist).fillna(global_mean_dist)

trainSet["dist_to_type_mean"] = trainSet["dist"] / train_den
testSet["dist_to_type_mean"] = testSet["dist"] / test_den

for df in (trainSet, testSet):
    df["dist_to_type_mean"] = df["dist_to_type_mean"].replace([np.inf, -np.inf], np.nan)
    df["dist_to_type_mean"] = df["dist_to_type_mean"].fillna(1.0)



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
base_model_params = (
    dict()
)  # keep defaults to preserve core behavior as much as possible



## === cell 12
feature_cols = [
    "type_1JHC",
    "type_1JHN",
    "type_2JHC",
    "type_2JHH",
    "type_2JHN",
    "type_3JHC",
    "type_3JHH",
    "dist",
    "dist_to_type_mean",
]

for c in feature_cols:
    if c not in trainSet.columns:
        trainSet[c] = 0
    if c not in testSet.columns:
        testSet[c] = 0

X_train_all = trainSet[feature_cols].to_numpy(dtype=np.float64)
y_train_all = trainSet["scalar_coupling_constant"].to_numpy(dtype=np.float64)




## === cell 13
def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)
    types = pd.Series(types)
    maes = (y_true - y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()




## === cell 14
global_model = HuberRegressor(**base_model_params).fit(X_train_all, y_train_all)

type_models = {}
for t in sorted(trainSet["type"].unique()):
    mask = trainSet["type"].values == t
    X_t = trainSet.loc[mask, feature_cols].to_numpy(dtype=np.float64)
    y_t = trainSet.loc[mask, "scalar_coupling_constant"].to_numpy(dtype=np.float64)
    type_models[t] = HuberRegressor(**base_model_params).fit(X_t, y_t)

print("Trained per-type models:", len(type_models), "Global fallback model: yes")



## === cell 15
train_pred_global = global_model.predict(X_train_all)

train_pred_per_type = np.empty(len(trainSet), dtype=np.float64)
for t, model in type_models.items():
    m = trainSet["type"].values == t
    train_pred_per_type[m] = model.predict(
        trainSet.loc[m, feature_cols].to_numpy(dtype=np.float64)
    )

print(
    "In-sample group logMAE (global model):",
    group_mean_log_mae(y_train_all, train_pred_global, trainSet["type"]),
)
print(
    "In-sample group logMAE (per-type models):",
    group_mean_log_mae(y_train_all, train_pred_per_type, trainSet["type"]),
)



## === cell 16
X_test_all = testSet[feature_cols].to_numpy(dtype=np.float64)

if np.isnan(X_test_all).any():
    col_nan = pd.isna(testSet[feature_cols]).sum()
    print("NaN counts in test features:\n", col_nan[col_nan > 0])
    X_test_all = np.nan_to_num(X_test_all, nan=0.0, posinf=0.0, neginf=0.0)

test_pred = np.empty(len(testSet), dtype=np.float64)
test_types = testSet["type"].values

unseen_mask_total = np.zeros(len(testSet), dtype=bool)
for t in np.unique(test_types):
    m = test_types == t
    if t in type_models:
        test_pred[m] = type_models[t].predict(
            testSet.loc[m, feature_cols].to_numpy(dtype=np.float64)
        )
    else:
        unseen_mask_total[m] = True

if unseen_mask_total.any():
    test_pred[unseen_mask_total] = global_model.predict(
        testSet.loc[unseen_mask_total, feature_cols].to_numpy(dtype=np.float64)
    )
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



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3820064391.py in <cell line: 0>()
     14     m = test_types == t
     15     if t in type_models:
---> 16         test_pred[m] = type_models[t].predict(
     17             testSet.loc[m, feature_cols].to_numpy(dtype=np.float64)
     18         )

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py in predict(self, X)
    352             Returns predicted values.
    353         """
--> 354         return self._decision_function(X)
    355 
    356     def _set_intercept(self, X_offset, y_offset, X_scale):

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py in _decision_function(self, X)
    335         check_is_fitted(self)
    336 
--> 337         X = self._validate_data(X, accept_sparse=["csr", "csc", "coo"], reset=False)
    338         return safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
    339 

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    563             raise ValueError("Validation should be done on X, y or both.")
    564         elif not no_val_X and no_val_y:
--> 565             X = check_array(X, input_name="X", **check_params)
    566             out = X
    567         elif no_val_X and not no_val_y:

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    919 
    920         if force_all_finite:
--> 921             _assert_all_finite(
    922                 array,
    923                 input_name=input_name,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in _assert_all_finite(X, allow_nan, msg_dtype, estimator_name, input_name)
    159                 "#estimators-that-handle-nan-values"
    160             )
--> 161         raise ValueError(msg_err)
    162 
    163 

ValueError: Input X contains NaN.
HuberRegressor does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values

## === cell 17
out_path = "submission.csv"
resultSet.to_csv(out_path, index=False)

with open(out_path, "r") as f:
    for i, line in enumerate(f):
        print(line.strip())
        if i > 5:
            break
print("Wrote:", out_path, "rows:", len(resultSet))

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2372780226.py in <cell line: 0>()
      1 out_path = "submission.csv"
----> 2 resultSet.to_csv(out_path, index=False)
      3 
      4 with open(out_path, "r") as f:
      5     for i, line in enumerate(f):

NameError: name 'resultSet' is not defined
