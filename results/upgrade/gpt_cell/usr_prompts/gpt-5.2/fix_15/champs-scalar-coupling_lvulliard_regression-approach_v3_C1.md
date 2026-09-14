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

2.85872

# 6. Current score

4.05087

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.09682) has done: 'Diagnosis: The crash happens because `X_test` is built from `testSet` after cell 6 may have dropped rows with missing atom info from the merge, resulting in `testSet` becoming empty (0 rows). `LinearRegression.predict` requires at least one sample, so it raises `ValueError: Found array with 0 sample(s)`. Even if prediction succeeded, the current assertion expects the submission to have exactly the same number of rows as the original `test.csv`, so we must preserve that row count and `id` coverage.

Patch summary: In cell 14 only, rebuild the feature matrix by re-reading the original `test.csv` ids and left-merging the computed features (`dist`, `atom_C`, `atom_H`) from the post-merge `testSet`. For any ids missing features (e.g., dropped earlier due to missing atom info), fill with safe defaults (`dist=0.0`, booleans `False`) so prediction can run and the output has the correct length and order. Keep the same model and feature set; only ensure non-empty, aligned input and a correctly-sized result.

Updated cells: (cell 14 only)

Compatibility notes for cell k+1: `resultSet` remains a DataFrame with columns `id` and `scalar_coupling_constant`, so cell 15 (`resultSet.to_csv(...)`) works unchanged. The output row count matches the original `test.csv` exactly, satisfying the existing assertion.

Assumptions: It is acceptable to provide default feature values for rows that could not be merged (since earlier cells already dropped them and no ground-truth exists for test); the goal here is to avoid crashing and maintain required submission shape/order without changing the model or training logic.'
- What this solution (achieved 4.05087) has done: 'Your current score (4.09682, lower-is-better) is far from the target (2.85872), so we need a small, legitimate improvement without changing the overall approach (still linear regression on simple engineered features). The biggest gain with minimal disruption is to fit separate LinearRegression models per coupling `type`, because the metric averages MAE per type and each type has very different target scales—one global model is systematically miscalibrated. We keep the same features (`dist`, `atom_C`, `atom_H`) and still use LinearRegression, but train/predict by `type` and then stitch predictions back to the original test `id` order to guarantee a valid submission. This should move the score substantially toward the target while preserving your existing feature engineering and evaluation semantics.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

print("Listing ../input:")
print(os.listdir("../input") if os.path.exists("../input") else "../input not found")




## === cell 1
def _pick_existing_base_dir():
    candidates = [
        "../input/champs-scalar-coupling",
        "/kaggle/input/champs-scalar-coupling",
        "/kaggle/data/input/champs-scalar-coupling",
        "/kaggle/data/champs-scalar-coupling",
        "../kaggle/data/input/champs-scalar-coupling",
        "../kaggle/data/champs-scalar-coupling",
        "/kaggle/data/champs-scalar-coupling/champs-scalar-coupling",
        "/kaggle/data/input/champs-scalar-coupling/champs-scalar-coupling",
        "/kaggle/data/champs-scalar-coupling",
    ]

    required_struct_cols = {"molecule_name", "atom_index", "atom", "x", "y", "z"}

    def _is_valid_dataset_dir(base):
        train_path = os.path.join(base, "train.csv")
        test_path = os.path.join(base, "test.csv")
        struct_path = os.path.join(base, "structures.csv")
        if not (
            os.path.exists(train_path)
            and os.path.exists(test_path)
            and os.path.exists(struct_path)
        ):
            return False
        try:
            cols = set(pd.read_csv(struct_path, nrows=0).columns)
        except Exception:
            return False
        return required_struct_cols.issubset(cols)

    for base in candidates:
        if _is_valid_dataset_dir(base):
            return base

    search_roots = [
        "../input",
        "/kaggle/input",
        "/kaggle/data/input",
        "/kaggle/data",
        "/",
    ]
    for root in search_roots:
        if not os.path.exists(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if os.path.basename(
                dirpath
            ) == "champs-scalar-coupling" and _is_valid_dataset_dir(dirpath):
                return dirpath

    return "../input/champs-scalar-coupling"


base_dir = _pick_existing_base_dir()
print("Using base_dir:", base_dir)



## === cell 2
trainSet = pd.read_csv(os.path.join(base_dir, "train.csv"))
testSet = pd.read_csv(os.path.join(base_dir, "test.csv"))
structures = pd.read_csv(os.path.join(base_dir, "structures.csv"))

print(trainSet.head())
print(testSet.head())
print(structures.head())




## === cell 3
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


for df in (trainSet, testSet):
    df["atom_index_0"] = pd.to_numeric(df["atom_index_0"], errors="raise").astype(
        np.int32
    )
    df["atom_index_1"] = pd.to_numeric(df["atom_index_1"], errors="raise").astype(
        np.int32
    )
structures["atom_index"] = pd.to_numeric(
    structures["atom_index"], errors="raise"
).astype(np.int32)

trainSet = map_atom_info(trainSet, 0)
trainSet = map_atom_info(trainSet, 1)
testSet = map_atom_info(testSet, 0)
testSet = map_atom_info(testSet, 1)

print(trainSet.head())
print(testSet.head())



## === cell 4
dx_tr = trainSet["x_1"].to_numpy() - trainSet["x_0"].to_numpy()
dy_tr = trainSet["y_1"].to_numpy() - trainSet["y_0"].to_numpy()
dz_tr = trainSet["z_1"].to_numpy() - trainSet["z_0"].to_numpy()
trainSet["dist"] = np.sqrt(dx_tr * dx_tr + dy_tr * dy_tr + dz_tr * dz_tr)

dx_te = testSet["x_1"].to_numpy() - testSet["x_0"].to_numpy()
dy_te = testSet["y_1"].to_numpy() - testSet["y_0"].to_numpy()
dz_te = testSet["z_1"].to_numpy() - testSet["z_0"].to_numpy()
testSet["dist"] = np.sqrt(dx_te * dx_te + dy_te * dy_te + dz_te * dz_te)



## === cell 5
for _df_name, _df in (("trainSet", trainSet), ("testSet", testSet)):
    _na_mask = _df["atom_0"].isna() | _df["atom_1"].isna()
    if _na_mask.any():
        print(
            f"{_df_name}: dropping {_na_mask.sum()} rows with missing atom info after merge"
        )
        if _df_name == "trainSet":
            trainSet = _df.loc[~_na_mask].copy()
        else:
            testSet = _df.loc[~_na_mask].copy()

assert trainSet["atom_0"].notna().all()
assert testSet["atom_0"].notna().all()



## === cell 6
print(trainSet["atom_1"].astype("category").cat.categories)
print(testSet["atom_1"].astype("category").cat.categories)



## === cell 7
trainSet["atom_C"] = trainSet["atom_1"] == "C"
trainSet["atom_H"] = trainSet["atom_1"] == "H"
trainSet["atom_N"] = trainSet["atom_1"] == "N"



## === cell 8
testSet["atom_C"] = testSet["atom_1"] == "C"
testSet["atom_H"] = testSet["atom_1"] == "H"
testSet["atom_N"] = testSet["atom_1"] == "N"



## === cell 9
model = LinearRegression(n_jobs=-1)



## === cell 10
X_train = np.array(trainSet[["dist", "atom_C", "atom_H"]])
y_train = trainSet["scalar_coupling_constant"].to_numpy()
fitDist = model.fit(X_train, y_train)



## === cell 11
print("Coefficients:", fitDist.coef_)



## === cell 12
r_sq = model.score(X_train, y_train)
print("coefficient of determination:", r_sq)



## === cell 13
orig_test = pd.read_csv(os.path.join(base_dir, "test.csv"), usecols=["id", "type"])
orig_test_ids = orig_test[["id"]].copy()

test_features = testSet[["id", "type", "dist", "atom_C", "atom_H"]].copy()
aligned_test = orig_test.merge(test_features, on=["id", "type"], how="left")

aligned_test["dist"] = aligned_test["dist"].fillna(0.0)
aligned_test["atom_C"] = aligned_test["atom_C"].fillna(False).astype(bool)
aligned_test["atom_H"] = aligned_test["atom_H"].fillna(False).astype(bool)

models_by_type = {}
global_fallback = LinearRegression(n_jobs=-1).fit(
    trainSet[["dist", "atom_C", "atom_H"]].to_numpy(),
    trainSet["scalar_coupling_constant"].to_numpy(),
)

pred = np.empty(aligned_test.shape[0], dtype=np.float64)

for t, idx in aligned_test.groupby("type").groups.items():
    tr_t = trainSet.loc[trainSet["type"] == t]
    X_te_t = aligned_test.loc[idx, ["dist", "atom_C", "atom_H"]].to_numpy()

    if tr_t.shape[0] == 0:
        pred[idx] = global_fallback.predict(X_te_t)
        continue

    X_tr_t = tr_t[["dist", "atom_C", "atom_H"]].to_numpy()
    y_tr_t = tr_t["scalar_coupling_constant"].to_numpy()

    m = LinearRegression(n_jobs=-1)
    m.fit(X_tr_t, y_tr_t)
    models_by_type[t] = m
    pred[idx] = m.predict(X_te_t)

resultSet = pd.DataFrame(
    {"id": aligned_test["id"].to_numpy(), "scalar_coupling_constant": pred}
)
resultSet = resultSet.merge(
    orig_test_ids, on="id", how="right"
)  # ensure full id coverage
resultSet = resultSet.sort_values("id", kind="mergesort")  # deterministic ordering
resultSet = resultSet[["id", "scalar_coupling_constant"]]

resultSet = orig_test_ids.merge(resultSet, on="id", how="left")
assert resultSet.shape[0] == orig_test_ids.shape[0]
assert resultSet["scalar_coupling_constant"].notna().all()



## === cell 14
resultSet.to_csv("results.csv", index=False, header=True)
print("Wrote results.csv with shape:", resultSet.shape)



## === cell 15
with open("results.csv", "r") as f:
    for i, line in enumerate(f):
        print(line.rstrip("\n"))
        if i > 5:
            break
