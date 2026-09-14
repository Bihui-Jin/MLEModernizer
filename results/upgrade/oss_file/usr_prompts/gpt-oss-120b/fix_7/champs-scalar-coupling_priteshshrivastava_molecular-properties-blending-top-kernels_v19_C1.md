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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

print("Available input directories:", os.listdir("../input"))



## === cell 1
train_path = "../input/champs-scalar-coupling/train.csv"
test_path = "../input/champs-scalar-coupling/test.csv"
structures_path = "../input/champs-scalar-coupling/structures.csv"
dipole_path = "../input/champs-scalar-coupling/dipole_moments.csv"
potential_path = "../input/champs-scalar-coupling/potential_energy.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
struct_df = pd.read_csv(structures_path)

struct_df = struct_df.rename(columns={"atom": "element"})[
    ["molecule_name", "atom_index", "element", "x", "y", "z"]
]

dipole_df = pd.read_csv(dipole_path)
potential_df = pd.read_csv(potential_path)

dipole_df["dipole_mag"] = np.sqrt(
    dipole_df["X"] ** 2 + dipole_df["Y"] ** 2 + dipole_df["Z"] ** 2
)



## === cell 2
train_merged = (
    train_df.merge(
        struct_df,
        left_on=["molecule_name", "atom_index_0"],
        right_on=["molecule_name", "atom_index"],
        how="left",
    )
    .rename(columns=lambda c: c + "_0" if c in ["element", "x", "y", "z"] else c)
    .drop(columns=["atom_index"])
)

train_merged = (
    train_merged.merge(
        struct_df,
        left_on=["molecule_name", "atom_index_1"],
        right_on=["molecule_name", "atom_index"],
        how="left",
    )
    .rename(columns=lambda c: c + "_1" if c in ["element", "x", "y", "z"] else c)
    .drop(columns=["atom_index"])
)

train_merged = train_merged.merge(
    dipole_df[["molecule_name", "dipole_mag"]], on="molecule_name", how="left"
)
train_merged = train_merged.merge(potential_df, on="molecule_name", how="left")

test_merged = (
    test_df.merge(
        struct_df,
        left_on=["molecule_name", "atom_index_0"],
        right_on=["molecule_name", "atom_index"],
        how="left",
    )
    .rename(columns=lambda c: c + "_0" if c in ["element", "x", "y", "z"] else c)
    .drop(columns=["atom_index"])
)
test_merged = (
    test_merged.merge(
        struct_df,
        left_on=["molecule_name", "atom_index_1"],
        right_on=["molecule_name", "atom_index"],
        how="left",
    )
    .rename(columns=lambda c: c + "_1" if c in ["element", "x", "y", "z"] else c)
    .drop(columns=["atom_index"])
)
test_merged = test_merged.merge(
    dipole_df[["molecule_name", "dipole_mag"]], on="molecule_name", how="left"
)
test_merged = test_merged.merge(potential_df, on="molecule_name", how="left")



## === cell 3
atomic_number = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Cl": 17,
    "Br": 35,
    "I": 53,
    "S": 16,
    "P": 15,
}
train_merged["atomic_num_0"] = train_merged["element_0"].map(atomic_number).fillna(0)
train_merged["atomic_num_1"] = train_merged["element_1"].map(atomic_number).fillna(0)
test_merged["atomic_num_0"] = test_merged["element_0"].map(atomic_number).fillna(0)
test_merged["atomic_num_1"] = test_merged["element_1"].map(atomic_number).fillna(0)

train_merged["distance"] = np.sqrt(
    (train_merged["x_0"] - train_merged["x_1"]) ** 2
    + (train_merged["y_0"] - train_merged["y_1"]) ** 2
    + (train_merged["z_0"] - train_merged["z_1"]) ** 2
)
test_merged["distance"] = np.sqrt(
    (test_merged["x_0"] - test_merged["x_1"]) ** 2
    + (test_merged["y_0"] - test_merged["y_1"]) ** 2
    + (test_merged["z_0"] - test_merged["z_1"]) ** 2
)

train_merged["type_code"], type_uniques = pd.factorize(train_merged["type"])
test_merged["type_code"] = type_uniques.get_indexer(test_merged["type"])
unseen = test_merged["type_code"] == -1
if unseen.any():
    new_code = len(type_uniques)
    test_merged.loc[unseen, "type_code"] = new_code



## === cell 4
combo_means = train_merged.groupby(["type", "element_0", "element_1"])[
    "scalar_coupling_constant"
].mean()
pair_means = train_merged.groupby(["element_0", "element_1"])[
    "scalar_coupling_constant"
].mean()
type_means = train_df.groupby("type")["scalar_coupling_constant"].mean()
global_mean = train_df["scalar_coupling_constant"].mean()


def baseline_predict(df, combo_means, pair_means, type_means, global_mean):
    key_combo = df.set_index(["type", "element_0", "element_1"]).index
    pred = combo_means.reindex(key_combo).reset_index(drop=True)

    missing = pred.isna()
    if missing.any():
        key_pair = (
            df.loc[missing, ["element_0", "element_1"]]
            .set_index(["element_0", "element_1"])
            .index
        )
        pred_pair = pair_means.reindex(key_pair).values
        pred[missing] = pred_pair

    missing = pd.isna(pred)
    if missing.any():
        pred[missing] = df.loc[missing, "type"].map(type_means)

    return pred.fillna(global_mean).values


train_baseline = baseline_predict(
    train_merged, combo_means, pair_means, type_means, global_mean
)
train_residual = train_merged["scalar_coupling_constant"].values - train_baseline

test_baseline = baseline_predict(
    test_merged, combo_means, pair_means, type_means, global_mean
)



## === cell 5
train_merged["atomic_num_sum"] = (
    train_merged["atomic_num_0"] + train_merged["atomic_num_1"]
)
train_merged["atomic_num_diff"] = (
    train_merged["atomic_num_0"] - train_merged["atomic_num_1"]
).abs()
test_merged["atomic_num_sum"] = (
    test_merged["atomic_num_0"] + test_merged["atomic_num_1"]
)
test_merged["atomic_num_diff"] = (
    test_merged["atomic_num_0"] - test_merged["atomic_num_1"]
).abs()

feature_cols = [
    "distance",
    "dipole_mag",
    "potential_energy",
    "atomic_num_0",
    "atomic_num_1",
    "type_code",
    "atomic_num_sum",
    "atomic_num_diff",
]

X = train_merged[feature_cols].fillna(-1)
y = train_residual

gbr = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
)

gbr.fit(X, y)

test_features = test_merged[feature_cols].fillna(-1)
test_residual_pred = gbr.predict(test_features)

final_pred = test_baseline + test_residual_pred

final_pred = np.nan_to_num(final_pred, nan=global_mean)



## === cell 6
test_df["scalar_coupling_constant"] = final_pred

submission = test_df[["id", "scalar_coupling_constant"]].copy()
submission_path = "my_blend_2.csv"
submission.to_csv(submission_path, index=False)

print(f"Submission written to {submission_path}, shape: {submission.shape}")
