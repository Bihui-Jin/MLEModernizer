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

4.53708

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 3.00294) has done: 'Diagnosis: The crash happens in cell 18 because the feature matrix passed to `model.predict()` contains NaNs. These NaNs are most likely coming from `dist_to_type_mean` for test rows whose `type` group mean is missing or undefined (e.g., a type not present in the groupby result, or a zero/NaN mean). `LinearRegression` cannot handle NaNs at prediction time, so we must ensure the exact same feature columns are fully finite before calling `predict`.

Patch summary: In cell 18 only, compute the test feature matrix once, coerce it to numeric, and replace any NaN/inf values with safe defaults (0.0). This keeps the existing model and feature set unchanged while preventing `predict()` from crashing.

Updated cells: Only cell 18 is modified.

Compatibility notes for cell k+1: `resultSet` remains a pandas DataFrame with columns `id` and `scalar_coupling_constant`, so cell 19 (`resultSet.to_csv(...)`) work unchanged.

Assumptions: Replacing missing feature values with 0.0 is acceptable as a minimal, deterministic preprocessing step to unblock prediction without altering the model/training logic or adding new modeling components.'
- What this solution (achieved 3.00294) has done: 'Your current score (3.00294, lower-is-better) is far from the target (1.27404), so we should make a small, legitimate improvement that doesn’t change the modeling approach: fix a key feature leakage/consistency bug. Right now, `dist_to_type_mean` in test is computed using *test* type means (and can be NaN), while it should be computed using *train* type means to match training semantics and reduce distribution shift. I compute and apply the train-type mean distances to both train and test (same feature definition, just consistent), and keep the existing linear regression and feature set unchanged. This should improve generalization materially while remaining minimal and still producing a valid `results.csv`.'
- What this solution (achieved 4.10965) has done: 'To move your score down toward the 1.27 target without changing the modeling approach, I keep the exact same LinearRegression and feature set but fix a key generalization issue: the training is currently evaluated and fit with raw `dist`, even though `dist` is very type-dependent. A minimal, metric-aligned improvement is to train/predict on `log1p(dist)` (monotonic transform) to reduce scale differences across coupling types while preserving the same core distance feature semantics. I apply the same transform consistently to both train and test, keep `dist_to_type_mean` computed from train-type means (as you already fixed), and keep the submission writing unchanged.'
- What this solution (achieved 4.53716) has done: 'Your current score (4.10965, lower-is-better) is far above the target (1.27404), so we should make a small, metric-aligned improvement without changing the model: keep LinearRegression but fit separate models per coupling `type`, because the metric averages MAE per type and the relationship is type-specific. This preserves the same core features (`dist`, `dist_to_type_mean`, and type info) and training approach (linear regression), but removes the need for one global set of coefficients to cover all types. To keep the submission valid and robust, predictions are generated type-by-type and any missing/invalid rows fall back to a global model. This should materially reduce the per-type MAE and move the score down toward the target while remaining a minimal, legitimate change.'
- What this solution (achieved 4.53749) has done: 'Your current score (4.53716, lower-is-better) is far worse than the target (1.27404), so we need a small but meaningful improvement without changing the model family or training loop. The biggest issue is that `dist_to_type_mean` is undefined (and then filled with 0) for coupling types that exist in test but not in train (notably `3JHN`), which makes those rows systematically wrong and hurts the per-type averaged metric. I keep the same LinearRegression setup (global + per-type models) and the same core distance features, but add a tiny fallback so `dist_to_type_mean` uses a global mean when a type mean is missing. I also ensure we create type dummy columns for the union of train+test types so `3JHN` is represented consistently.'
- What this solution (achieved 4.53678) has done: 'Your current score (4.53749, lower-is-better) is still far from the target (1.27404), so we make a minimal, metric-aligned improvement without changing the model family (still `LinearRegression`) or adding new learning algorithms. The main issue is that including one-hot `type_*` columns inside each per-type model is redundant and can destabilize coefficients (nearly constant columns within a type), which hurts per-type MAE and therefore the competition metric. I keep the same per-type training/prediction approach, but for per-type models I train only on the two distance features (`dist`, `dist_to_type_mean`), while keeping the global fallback model unchanged (still uses the full feature set including type dummies). This is a small, safe change that usually improves per-type fits and should move the score down toward the target.'
- What this solution (achieved 4.53718) has done: 'Your current score (4.53678, lower-is-better) is still far from the target (1.27404), so we need a small but meaningful improvement without changing the model family or overall approach. The biggest remaining issue is that the “global” model is trying to learn one set of coefficients across very different coupling types; since the metric averages MAE per type, a minimal metric-aligned fix is to keep the per-type LinearRegression models but also normalize the target within each type (fit on residuals after subtracting the per-type mean, then add it back at prediction). This preserves the same features and linear regression core logic while dramatically reducing per-type bias. I also keep your existing global fallback (with type dummies) for any unseen types, and ensure all feature matrices are numeric and finite before prediction.'
- What this solution (achieved 4.53708) has done: 'Your score is much worse than the target (lower-is-better), so the smallest meaningful improvement is to align training with the metric more closely while preserving your linear-regression/per-type approach. I keep your exact feature set and per-type residual modeling, but instead of fitting each per-type model on raw residuals, I fit it with inverse-frequency sample weights so each coupling `type` contributes roughly equally (matching the “average over types” metric). I also remove the now-redundant earlier global fit (`model`/`fitDist`) usage from affecting anything by leaving it intact but ensuring the final submission uses the weighted per-type models only. This should reduce the per-type averaged MAE and move the score down toward the target without changing the core logic.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import os

print(os.listdir("../input"))



## === cell 1
trainSet = pd.read_csv("../input/train.csv")
display(trainSet.head())



## === cell 2
testSet = pd.read_csv("../input/test.csv")
display(testSet.head())



## === cell 3
structures = pd.read_csv("../input/structures.csv")
display(structures.head())




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
display(trainSet.head())
display(testSet.head())



## === cell 6
train_p0 = trainSet[["x_0", "y_0", "z_0"]].values
train_p1 = trainSet[["x_1", "y_1", "z_1"]].values
test_p0 = testSet[["x_0", "y_0", "z_0"]].values
test_p1 = testSet[["x_1", "y_1", "z_1"]].values

trainSet["dist"] = np.linalg.norm(train_p0 - train_p1, axis=1)
testSet["dist"] = np.linalg.norm(test_p0 - test_p1, axis=1)

trainSet["dist"] = np.log1p(trainSet["dist"].astype(np.float64))
testSet["dist"] = np.log1p(testSet["dist"].astype(np.float64))

trainSet["dist_to_type_mean"] = trainSet["dist"] / trainSet.groupby("type")[
    "dist"
].transform("mean")
testSet["dist_to_type_mean"] = testSet["dist"] / testSet.groupby("type")[
    "dist"
].transform("mean")



## === cell 7
DATA_DIR = "../input/champs-scalar-coupling"

trainSet = pd.read_csv(f"{DATA_DIR}/train.csv")
testSet = pd.read_csv(f"{DATA_DIR}/test.csv")
structures = pd.read_csv(f"{DATA_DIR}/structures.csv")


def map_atom_info(df, atom_idx):
    left_key = f"atom_index_{atom_idx}"
    if left_key in df.columns:
        df[left_key] = pd.to_numeric(df[left_key], errors="coerce").astype("Int64")
    structures["atom_index"] = pd.to_numeric(
        structures["atom_index"], errors="coerce"
    ).astype("Int64")

    df = pd.merge(
        df,
        structures,
        how="left",
        left_on=["molecule_name", left_key],
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

train_p0 = trainSet[["x_0", "y_0", "z_0"]].values
train_p1 = trainSet[["x_1", "y_1", "z_1"]].values
test_p0 = testSet[["x_0", "y_0", "z_0"]].values
test_p1 = testSet[["x_1", "y_1", "z_1"]].values

trainSet["dist"] = np.linalg.norm(train_p0 - train_p1, axis=1)
testSet["dist"] = np.linalg.norm(test_p0 - test_p1, axis=1)

trainSet["dist"] = np.log1p(trainSet["dist"].astype(np.float64))
testSet["dist"] = np.log1p(testSet["dist"].astype(np.float64))

train_type_mean_dist = trainSet.groupby("type")["dist"].mean()
global_mean_dist = float(trainSet["dist"].mean())

train_den = trainSet["type"].map(train_type_mean_dist).fillna(global_mean_dist)
test_den = testSet["type"].map(train_type_mean_dist).fillna(global_mean_dist)

trainSet["dist_to_type_mean"] = trainSet["dist"] / train_den
testSet["dist_to_type_mean"] = testSet["dist"] / test_den



## === cell 8
print(trainSet["atom_1"].astype("category").cat.categories)
print(testSet["atom_1"].astype("category").cat.categories)



## === cell 9
print(testSet["type"].astype("category").cat.categories)
print(trainSet["type"].astype("category").cat.categories)



## === cell 10
all_types = pd.Index(trainSet["type"].unique()).union(
    pd.Index(testSet["type"].unique())
)
for i in all_types.values:
    col = "type_" + str(i)
    trainSet[col] = trainSet["type"] == i
    testSet[col] = testSet["type"] == i



## === cell 11
model = LinearRegression(n_jobs=-1)



## === cell 12
from sklearn.linear_model import LinearRegression



## === cell 13
fitDist = model.fit(
    np.array(
        trainSet[
            [
                "type_1JHC",
                "type_1JHN",
                "type_2JHC",
                "type_2JHH",
                "type_2JHN",
                "type_3JHC",
                "type_3JHH",
                "type_3JHN",
                "dist",
                "dist_to_type_mean",
            ]
        ]
    ),
    trainSet["scalar_coupling_constant"],
)



## === cell 14
fitDist.coef_




## === cell 15
def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    maes = (y_true - y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()




## === cell 16
group_mean_log_mae(
    trainSet["scalar_coupling_constant"],
    model.predict(
        np.array(
            trainSet[
                [
                    "type_1JHC",
                    "type_1JHN",
                    "type_2JHC",
                    "type_2JHH",
                    "type_2JHN",
                    "type_3JHC",
                    "type_3JHH",
                    "type_3JHN",
                    "dist",
                    "dist_to_type_mean",
                ]
            ]
        )
    ),
    trainSet["type"],
)



## === cell 17
print(
    group_mean_log_mae(
        trainSet["scalar_coupling_constant"],
        trainSet["scalar_coupling_constant"].median(),
        trainSet["type"],
    )
)
print(group_mean_log_mae(trainSet["scalar_coupling_constant"], 0.85, trainSet["type"]))



## === cell 18
global_feature_cols = [
    "type_1JHC",
    "type_1JHN",
    "type_2JHC",
    "type_2JHH",
    "type_2JHN",
    "type_3JHC",
    "type_3JHH",
    "type_3JHN",
    "dist",
    "dist_to_type_mean",
]
per_type_feature_cols = ["dist", "dist_to_type_mean"]


def make_X(df, cols):
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.to_numpy()


X_train_all = make_X(trainSet, global_feature_cols)
y_train_all = trainSet["scalar_coupling_constant"].to_numpy()

global_model = LinearRegression(n_jobs=-1)
global_model.fit(X_train_all, y_train_all)

type_target_mean = trainSet.groupby("type")["scalar_coupling_constant"].mean()
global_target_mean = float(trainSet["scalar_coupling_constant"].mean())

type_counts = trainSet["type"].value_counts()
inv_freq_w = trainSet["type"].map(lambda t: 1.0 / float(type_counts.loc[t])).to_numpy()

type_models = {}
for t, grp in trainSet.groupby("type", sort=False):
    X_t = make_X(grp, per_type_feature_cols)
    y_t = grp["scalar_coupling_constant"].to_numpy()
    mu_t = (
        float(type_target_mean.loc[t])
        if t in type_target_mean.index
        else global_target_mean
    )
    y_res = y_t - mu_t

    w_t = inv_freq_w[grp.index.to_numpy()]

    m = LinearRegression(n_jobs=-1)
    m.fit(X_t, y_res, sample_weight=w_t)
    type_models[t] = m

pred = np.empty(len(testSet), dtype=np.float64)
pred[:] = np.nan

for t, idx in testSet.groupby("type", sort=False).groups.items():
    m = type_models.get(t, None)
    if m is None:
        pred[idx] = global_model.predict(make_X(testSet.loc[idx], global_feature_cols))
    else:
        mu_t = (
            float(type_target_mean.loc[t])
            if t in type_target_mean.index
            else global_target_mean
        )
        pred[idx] = m.predict(make_X(testSet.loc[idx], per_type_feature_cols)) + mu_t

if np.isnan(pred).any():
    missing_mask = np.isnan(pred)
    pred[missing_mask] = global_model.predict(
        make_X(testSet.loc[missing_mask], global_feature_cols)
    )

resultSet = pd.DataFrame(
    {"id": testSet["id"].to_numpy(), "scalar_coupling_constant": pred}
)



## === cell 19
resultSet.to_csv("results.csv", index=False, header=True)



## === cell 20
with open("results.csv", "r") as f:
    for i, line in enumerate(f):
        print(line)
        if i > 5:
            break
