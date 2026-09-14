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

1.27404

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 3.00294) has done: 'I fix the failing assertions by replacing the incorrect category-comparison logic with a robust check that all `atom_0` values are `"H"` (or skip the check if missing), so the notebook doesn’t crash. Then I fix the NaNs in test features: `dist_to_type_mean` is currently computed using the test set’s own group means, which can be missing/NaN for rare types; instead I compute type means from train and map them onto both train and test (with a safe fallback). Finally, I ensure the feature columns exist for both train and test (adding any missing dummy columns as zeros), and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 32.33616) has done: 'You’re far from the target (3.00 vs 1.27, lower is better), so we should improve accuracy while keeping your linear-regression core intact. The biggest gain with minimal disruption is to fit *separate LinearRegression models per coupling `type`* (same model, same features) because the metric is averaged per-type and the relationships differ strongly by type. To support this, we keep your existing features but also add a tiny, type-specific distance normalization (`dist_centered = dist - type_mean_dist`) which is still the same feature family you already use (distance + type mean distance) and doesn’t change the approach. Finally, we ensure robust handling for any types that might appear only in test (fallback to the global model).'
- What this solution (achieved 3.77723) has done: 'Your current score is far worse than the target (32.34 vs 1.274, lower is better), so we should improve accuracy while keeping your same LinearRegression-per-type core. The biggest issue is that the model is missing the dominant signal: atom identities and key per-atom physics tables (mulliken charges, shielding tensors), which can be added as simple numeric features without changing the training approach. I minimally extend your existing `map_atom_info` merge pattern to also merge Mulliken and shielding for atom_0/atom_1, add safe one-hot encoding for `atom_1` (and `atom_0` if present), and include these new columns in `FEATURES` while keeping the same per-type model fitting/prediction and submission writing. These changes should substantially reduce error toward the target while staying within your linear-model framework.'
- What this solution (achieved 9.2925) has done: 'We keep your per-type LinearRegression approach and existing features, but fix a key mismatch with the competition metric by training the model to better match MAE behavior via a simple target transform: fit on `sign(y)*log1p(|y|)` per type, then invert with `sign*expm1(|.|)` at prediction time. This is a minimal change (same model class, same training loops, same features) and typically reduces large-error sensitivity that hurts log(MAE) without changing evaluation semantics. We also add one more physics table (`dipole_moments.csv`) and one molecule-level numeric feature (`potential_energy.csv`) via the same merge pattern you already use; these are strong signals and should move your score down toward the 1.274 target. Finally, we keep all column/NaN safety and ensure a valid `submission.csv` is written.'
- What this solution (achieved 18.78985) has done: 'We move the score down toward the 1.274 target by fixing the main regression mismatch introduced by the current `log1p` target transform: it compresses large couplings too aggressively and is likely driving the public score up (worse). Keeping the exact same per-type `LinearRegression` core and the same features/loops, we instead standardize the target **per type** (z-score) during training and invert at prediction time; this keeps the regression linear while improving calibration for each coupling type (which the metric averages). We also make the per-type dummy columns robust by building them from the union of train+test types (so no missing type indicators), and we keep all NaN/inf safety and the submission writing unchanged.'
- What this solution (achieved 18.78985) has done: 'Your current score (18.78985, lower is better) is far worse than the target (1.27404), and the biggest likely reason is that the model is unintentionally conflicting with itself: you train separate per-type models while also feeding a full `type_*` one-hot block into every per-type model, which makes the type dummies constant within each per-type subset and can destabilize/degenerate the fit. I keep the exact same per-type `LinearRegression` approach and all your existing feature engineering, but remove `type_*` one-hot columns from the per-type and global regressors (since you already split by type), which should improve generalization without changing the core logic. I also make the atom one-hot columns use the union of train+test categories (like you already do for types) to avoid missing indicators for rare atoms in test. Finally, I keep the per-type z-score target transform/inversion exactly as-is and still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import os

BASE = "../input/champs-scalar-coupling"
if not os.path.exists(BASE):
    BASE = "../input"

print("Using BASE:", BASE)
print("Files in BASE:", os.listdir(BASE)[:20])



## === cell 1
train_path = os.path.join(BASE, "train.csv")
trainSet = pd.read_csv(train_path)
print(trainSet.shape)
print(trainSet.head())



## === cell 2
test_path = os.path.join(BASE, "test.csv")
testSet = pd.read_csv(test_path)
print(testSet.shape)
print(testSet.head())



## === cell 3
structures_path = os.path.join(BASE, "structures.csv")
structures = pd.read_csv(structures_path)
print(structures.shape)
print(structures.head())



## === cell 4
mulliken_path = os.path.join(BASE, "mulliken_charges.csv")
shielding_path = os.path.join(BASE, "magnetic_shielding_tensors.csv")
dipole_path = os.path.join(BASE, "dipole_moments.csv")
potential_path = os.path.join(BASE, "potential_energy.csv")

mulliken = pd.read_csv(mulliken_path)
shielding = pd.read_csv(shielding_path)

dipole = pd.read_csv(dipole_path)
potential = pd.read_csv(potential_path)

print("mulliken:", mulliken.shape, "shielding:", shielding.shape)
print("dipole:", dipole.shape, "potential:", potential.shape)
print(mulliken.head())
print(shielding.head())
print(dipole.head())
print(potential.head())




## === cell 5
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


def map_mulliken(df, atom_idx):
    df = pd.merge(
        df,
        mulliken,
        how="left",
        left_on=["molecule_name", f"atom_index_{atom_idx}"],
        right_on=["molecule_name", "atom_index"],
    )
    df = df.drop("atom_index", axis=1)
    df = df.rename(columns={"mulliken_charge": f"mulliken_charge_{atom_idx}"})
    return df


def map_shielding(df, atom_idx):
    df = pd.merge(
        df,
        shielding,
        how="left",
        left_on=["molecule_name", f"atom_index_{atom_idx}"],
        right_on=["molecule_name", "atom_index"],
    )
    df = df.drop("atom_index", axis=1)
    rename_cols = {
        c: f"shield_{c}_{atom_idx}"
        for c in ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]
    }
    df = df.rename(columns=rename_cols)
    return df


def map_dipole(df):
    return pd.merge(df, dipole, how="left", on="molecule_name")


def map_potential(df):
    return pd.merge(df, potential, how="left", on="molecule_name")


trainSet = map_atom_info(trainSet, 0)
trainSet = map_atom_info(trainSet, 1)
testSet = map_atom_info(testSet, 0)
testSet = map_atom_info(testSet, 1)

trainSet = map_mulliken(trainSet, 0)
trainSet = map_mulliken(trainSet, 1)
testSet = map_mulliken(testSet, 0)
testSet = map_mulliken(testSet, 1)

trainSet = map_shielding(trainSet, 0)
trainSet = map_shielding(trainSet, 1)
testSet = map_shielding(testSet, 0)
testSet = map_shielding(testSet, 1)

trainSet = map_dipole(trainSet)
testSet = map_dipole(testSet)

trainSet = map_potential(trainSet)
testSet = map_potential(testSet)

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

train_type_mean = trainSet["type"].map(type_mean_dist).fillna(global_mean_dist)
test_type_mean = testSet["type"].map(type_mean_dist).fillna(global_mean_dist)

trainSet["dist_to_type_mean"] = trainSet["dist"] / train_type_mean
testSet["dist_to_type_mean"] = testSet["dist"] / test_type_mean

trainSet["dist_centered"] = trainSet["dist"] - train_type_mean
testSet["dist_centered"] = testSet["dist"] - test_type_mean

trainSet[["dist", "dist_to_type_mean", "dist_centered"]] = (
    trainSet[["dist", "dist_to_type_mean", "dist_centered"]]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)
testSet[["dist", "dist_to_type_mean", "dist_centered"]] = (
    testSet[["dist", "dist_to_type_mean", "dist_centered"]]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)



## === cell 7
if "atom_0" in trainSet.columns and trainSet["atom_0"].notna().any():
    if not (trainSet["atom_0"].dropna() == "H").all():
        print("Warning: non-H atom_0 found in train; continuing.")
if "atom_0" in testSet.columns and testSet["atom_0"].notna().any():
    if not (testSet["atom_0"].dropna() == "H").all():
        print("Warning: non-H atom_0 found in test; continuing.")



## === cell 8
print("Train atom_1 categories:", trainSet["atom_1"].astype("category").cat.categories)
print("Test atom_1 categories:", testSet["atom_1"].astype("category").cat.categories)



## === cell 9
print("Test type categories:", testSet["type"].astype("category").cat.categories)
print("Train type categories:", trainSet["type"].astype("category").cat.categories)



## === cell 10
all_types = pd.Index(
    pd.concat([trainSet["type"], testSet["type"]], axis=0).astype(str).unique()
).sort_values()

for t in all_types:
    col = "type_" + str(t)
    trainSet[col] = (trainSet["type"].astype(str) == str(t)).astype(np.int8)
    testSet[col] = (testSet["type"].astype(str) == str(t)).astype(np.int8)

atom1_all = pd.Index(
    pd.concat([trainSet["atom_1"], testSet["atom_1"]], axis=0).astype(str).unique()
).sort_values()
for a in atom1_all:
    col = "atom1_" + str(a)
    trainSet[col] = (trainSet["atom_1"].astype(str) == str(a)).astype(np.int8)
    testSet[col] = (testSet["atom_1"].astype(str) == str(a)).astype(np.int8)

atom0_all = []
if "atom_0" in trainSet.columns and "atom_0" in testSet.columns:
    atom0_all = pd.Index(
        pd.concat([trainSet["atom_0"], testSet["atom_0"]], axis=0).astype(str).unique()
    ).sort_values()
    for a in atom0_all:
        col = "atom0_" + str(a)
        trainSet[col] = (trainSet["atom_0"].astype(str) == str(a)).astype(np.int8)
        testSet[col] = (testSet["atom_0"].astype(str) == str(a)).astype(np.int8)



## === cell 11
model = LinearRegression(n_jobs=-1)



## === cell 12
shield_cols_0 = [
    f"shield_{c}_0" for c in ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]
]
shield_cols_1 = [
    f"shield_{c}_1" for c in ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]
]

atom1_ohe = ["atom1_" + str(a) for a in atom1_all]
atom0_ohe = ["atom0_" + str(a) for a in atom0_all] if len(atom0_all) else []

base_features = [
    "dist",
    "dist_to_type_mean",
    "dist_centered",
    "mulliken_charge_0",
    "mulliken_charge_1",
    "X",
    "Y",
    "Z",
    "potential_energy",
]

FEATURES = base_features + shield_cols_0 + shield_cols_1 + atom1_ohe + atom0_ohe

for c in FEATURES:
    if c not in trainSet.columns:
        trainSet[c] = 0.0
    if c not in testSet.columns:
        testSet[c] = 0.0

trainSet[FEATURES] = trainSet[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
testSet[FEATURES] = testSet[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)




## === cell 13
def y_transform_type(y, t=None):
    y = np.asarray(y, dtype=np.float64)
    return np.sign(y) * np.log1p(np.abs(y))


def y_inverse_type(yt, t=None):
    yt = np.asarray(yt, dtype=np.float64)
    return np.sign(yt) * np.expm1(np.abs(yt))




## === cell 14
global_model = LinearRegression(n_jobs=-1)
global_model.fit(
    trainSet[FEATURES].values,
    y_transform_type(trainSet["scalar_coupling_constant"].values),
)

type_models = {}
for t, idx in trainSet.groupby("type").groups.items():
    m = LinearRegression(n_jobs=-1)
    X_t = trainSet.loc[idx, FEATURES].values
    y_t = y_transform_type(trainSet.loc[idx, "scalar_coupling_constant"].values, t)
    m.fit(X_t, y_t)
    type_models[t] = m

print(
    "Fitted global model and",
    len(type_models),
    "type-specific models (per-type signed log1p target).",
)



## === cell 15
some_t = next(iter(type_models.keys()))
print("Example type:", some_t)
print("Coef shape:", type_models[some_t].coef_.shape)




## === cell 16
def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred, index=y_true.index)
    types = pd.Series(types, index=y_true.index)
    maes = (y_true - y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()




## === cell 17
train_pred = np.empty(len(trainSet), dtype=np.float64)
for t, idx in trainSet.groupby("type").groups.items():
    m = type_models.get(t, global_model)
    pred_t = m.predict(trainSet.loc[idx, FEATURES].values)
    train_pred[idx] = y_inverse_type(pred_t, t)

print(
    "Train group_mean_log_mae:",
    group_mean_log_mae(
        trainSet["scalar_coupling_constant"].values, train_pred, trainSet["type"].values
    ),
)



## === cell 18
print(
    "Baseline median:",
    group_mean_log_mae(
        trainSet["scalar_coupling_constant"],
        trainSet["scalar_coupling_constant"].median(),
        trainSet["type"],
    ),
)
print(
    "Baseline 0.85:",
    group_mean_log_mae(trainSet["scalar_coupling_constant"], 0.85, trainSet["type"]),
)



## === cell 19
test_pred = np.empty(len(testSet), dtype=np.float64)
for t, idx in testSet.groupby("type").groups.items():
    m = type_models.get(t, global_model)
    pred_t = m.predict(testSet.loc[idx, FEATURES].values)
    test_pred[idx] = y_inverse_type(pred_t, t)

resultSet = pd.DataFrame(
    {
        "id": testSet["id"].values,
        "scalar_coupling_constant": test_pred.astype(np.float64),
    }
)

print(resultSet.head())
print(resultSet.shape)



## === cell 20
submission_path = "submission.csv"
resultSet.to_csv(submission_path, index=False, header=True)
print("Wrote:", submission_path)



## === cell 21
with open(submission_path, "r") as f:
    for i, line in enumerate(f):
        print(line.strip())
        if i > 5:
            break
