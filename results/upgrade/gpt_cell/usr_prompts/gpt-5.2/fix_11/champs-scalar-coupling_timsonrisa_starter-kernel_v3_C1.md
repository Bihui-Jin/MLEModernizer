# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.7

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = "/kaggle/input/champs-scalar-coupling"
NESTED_DIR = os.path.join(BASE_DIR, "champs-scalar-coupling")
DATA_DIR = (
    NESTED_DIR if os.path.exists(os.path.join(NESTED_DIR, "train.csv")) else BASE_DIR
)

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
STRUCT_PATH = os.path.join(DATA_DIR, "structures.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

print("Using DATA_DIR =", DATA_DIR)
print(
    "Train exists:",
    os.path.exists(TRAIN_PATH),
    "Test exists:",
    os.path.exists(TEST_PATH),
    "Structures exists:",
    os.path.exists(STRUCT_PATH),
)


## === cell 1
train_data = pd.read_csv(TRAIN_PATH)
y_label = train_data.pop("scalar_coupling_constant")
print(f"Training data is of shape: {train_data.shape}")
train_data.head(3)


## === cell 2
test_data = pd.read_csv(TEST_PATH)
print(f"Test data is of shape: {test_data.shape}")
test_data.head(3)


## === cell 3
print(
    f"Training Data has {train_data.molecule_name.nunique()} unique molecules with {train_data.type.nunique()} unique types"
)
print(
    f"Test Data has {test_data.molecule_name.nunique()} unique molecules with {test_data.type.nunique()} unique types"
)
print(
    f"Coupling Constant Dist.: mean={round(float(y_label.mean()),2)} ± std={round(float(y_label.std()),2)}"
)


## === cell 4
structures = pd.read_csv(STRUCT_PATH)
structures.head(3)




## === cell 5
def MergeData(data, structures):
    data = pd.merge(
        data,
        structures,
        how="inner",
        left_on=["molecule_name", "atom_index_0"],
        right_on=["molecule_name", "atom_index"],
    )
    data.drop(columns=["atom_index"], inplace=True)
    data.rename(
        index=str,
        columns={"atom": "atom_0", "x": "x_0", "y": "y_0", "z": "z_0"},
        inplace=True,
    )

    data = pd.merge(
        data,
        structures,
        how="inner",
        left_on=["molecule_name", "atom_index_1"],
        right_on=["molecule_name", "atom_index"],
    )
    data.drop(columns=["atom_index"], inplace=True)
    data.rename(
        index=str,
        columns={"atom": "atom_1", "x": "x_1", "y": "y_1", "z": "z_1"},
        inplace=True,
    )

    keep_cols = [
        "id",
        "molecule_name",
        "type",
        "atom_index_0",
        "atom_0",
        "x_0",
        "y_0",
        "z_0",
        "atom_index_1",
        "atom_1",
        "x_1",
        "y_1",
        "z_1",
    ]
    data = data.reindex(columns=keep_cols)
    return data


train_data = MergeData(train_data, structures)
test_data = MergeData(test_data, structures)

print("After merge - train shape:", train_data.shape, "test shape:", test_data.shape)
print(
    "Any missing ids after merge? train:",
    train_data["id"].isna().any(),
    "test:",
    test_data["id"].isna().any(),
)


## === cell 6
for f in ["type", "atom_0", "atom_1"]:
    lbl = LabelEncoder()
    lbl.fit(list(train_data[f].values) + list(test_data[f].values))
    train_data[f] = lbl.transform(list(train_data[f].values))
    test_data[f] = lbl.transform(list(test_data[f].values))


def add_distance_feature(df):
    dx = df["x_0"].values - df["x_1"].values
    dy = df["y_0"].values - df["y_1"].values
    dz = df["z_0"].values - df["z_1"].values
    df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz)
    return df


train_data = add_distance_feature(train_data)
test_data = add_distance_feature(test_data)


## === cell 7
feature_cols = ["type", "atom_0", "atom_1", "dist"]

train = train_data[feature_cols].values
test = test_data[feature_cols].values

reg = RandomForestRegressor(
    n_estimators=60,
    max_depth=9,
    min_samples_leaf=3,
    n_jobs=-1,
    random_state=42,
    oob_score=True,
    bootstrap=True,
)
reg.fit(train, y_label)

print("OOB R^2 (sanity check):", float(reg.oob_score_))

if test.shape[0] == 0:
    yhat = np.array([], dtype=float)
else:
    yhat = reg.predict(test)

if yhat.size == 0:
    print("Pred stats: empty predictions (test set has 0 rows)")
else:
    print("Pred stats:", float(np.min(yhat)), float(np.mean(yhat)), float(np.max(yhat)))


## === cell 8

pred_df = pd.DataFrame({"id": test_data["id"].values, "scalar_coupling_constant": yhat})

sample_submission = pd.read_csv(SAMPLE_SUB_PATH)

pred_df["id"] = pd.to_numeric(pred_df["id"], errors="raise").astype(np.int64)
sample_submission["id"] = pd.to_numeric(sample_submission["id"], errors="raise").astype(
    np.int64
)

if pred_df["id"].duplicated().any():
    pred_df = pred_df.drop_duplicates(subset=["id"], keep="first")

if "scalar_coupling_constant" in sample_submission.columns:
    sample_submission = sample_submission.drop(columns=["scalar_coupling_constant"])

submission = sample_submission.merge(
    pred_df, on="id", how="left", validate="one_to_one"
)

if submission["scalar_coupling_constant"].isna().any():
    missing = int(submission["scalar_coupling_constant"].isna().sum())
    raise ValueError(
        f"Submission has {missing} NaN predictions after alignment. "
        f"This indicates merge/id mismatch; cannot produce a valid submission."
    )

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
print("Any NaNs in submission:", submission["scalar_coupling_constant"].isna().any())


## --- ERROR in cell 8, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3493228695.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     25[0m [0;32mif[0m [0msubmission[0m[0;34m[[0m[0;34m"scalar_coupling_constant"[0m[0;34m][0m[0;34m.[0m[0misna[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0many[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     26[0m     [0mmissing[0m [0;34m=[0m [0mint[0m[0;34m([0m[0msubmission[0m[0;34m[[0m[0;34m"scalar_coupling_constant"[0m[0;34m][0m[0;34m.[0m[0misna[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0msum[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 27[0;31m     raise ValueError(
[0m[1;32m     28[0m         [0;34mf"Submission has {missing} NaN predictions after alignment. "[0m[0;34m[0m[0;34m[0m[0m
[1;32m     29[0m         [0;34mf"This indicates merge/id mismatch; cannot produce a valid submission."[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: Submission has 467813 NaN predictions after alignment. This indicates merge/id mismatch; cannot produce a valid submission.
