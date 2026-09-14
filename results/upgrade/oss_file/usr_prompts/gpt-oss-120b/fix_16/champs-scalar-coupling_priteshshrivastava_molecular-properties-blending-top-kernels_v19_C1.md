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

# 5. Target score

-1.6819062841656574

# 6. Current score

3.47576

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'I replace the failing blending code with a simple, reliable baseline that reads the available training data, computes the average scalar_coupling_constant for each coupling type, and uses these averages to predict the test set (fall‑back to the overall mean when a type is unseen). The script writes a valid CSV (`my_blend_1.csv`) containing the required `id` and `scalar_coupling_constant` columns, eliminating the FileNotFoundError and producing a usable submission.'
- What this solution (achieved 1.23566) has done: 'The fix addresses the KeyError caused by trying to select a non‑existent `scalar_coupling_constant` column from the test set. Instead of indexing that column, we build a MultiIndex from the test rows (`type`, `element_0`, `element_1`) and reindex the pre‑computed combo means to obtain predictions, then fall back to type‑level and global means. This corrects the prediction generation while preserving the original blending logic and ensures a valid CSV submission is written.'
- What this solution (achieved 1.23566) has done: 'I keep the original data loading and merging steps, but add a second‑level fallback based on the mean coupling for each element‑pair (ignoring the coupling type). The prediction hierarchy now tries: (type + elements) → (elements only) → (type) → global mean. This modest extension uses only simple aggregations, preserving the baseline logic while expected to lower the log‑MAE and move the score toward the target.'
- What this solution (achieved 4.00165) has done: 'I keep the original data‑loading and merging steps, add a few cheap numeric features (inter‑atomic distance, dipole magnitude, potential energy, atomic numbers and a numeric encoding of the coupling type) and train a lightweight GradientBoostingRegressor on the residuals of the existing hierarchical mean‑baseline. By predicting a correction to the baseline and adding it back, the model can capture simple patterns missed by the pure averaging scheme, which should lower the log‑MAE toward the target while preserving the overall workflow and still producing a valid `my_blend_2.csv` submission.'
- What this solution (achieved 3.97274) has done: 'I correct the `HistGradientBoostingRegressor` initialization by using a valid loss name (`"squared_error"` instead of the invalid `"least_squares"`). This fixes the `InvalidParameterError` that stopped training, allowing the model to fit, produce residual predictions, and define `final_pred`. With the model running, the subsequent cell can write a proper CSV submission, restoring the intended end‑to‑end workflow without altering any core feature engineering or blending logic.'
- What this solution (achieved 3.47576) has done: 'I switch the gradient‑boosting loss to **absolute_error** (more aligned with the MAE metric) and give the model a few more trees (max_iter = 1200, max_depth = 7) to capture additional patterns. After adding the residual predictions I clip the final values to the range observed in the training targets, which reduces extreme outliers and typically lowers the log‑MAE while keeping the original workflow unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor

BASE_PATH = "/kaggle/input/champs-scalar-coupling"

train_path = os.path.join(BASE_PATH, "train.csv")
test_path = os.path.join(BASE_PATH, "test.csv")
structures_path = os.path.join(BASE_PATH, "structures.csv")
dipole_path = os.path.join(BASE_PATH, "dipole_moments.csv")
potential_path = os.path.join(BASE_PATH, "potential_energy.csv")

train_df = pd.read_csv(
    train_path,
    dtype={
        "id": "int32",
        "molecule_name": "category",
        "atom_index_0": "int16",
        "atom_index_1": "int16",
        "type": "category",
        "scalar_coupling_constant": "float32",
    },
)
test_df = pd.read_csv(
    test_path,
    dtype={
        "id": "int32",
        "molecule_name": "category",
        "atom_index_0": "int16",
        "atom_index_1": "int16",
        "type": "category",
    },
)

struct_df = pd.read_csv(
    structures_path,
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
    dtype={
        "molecule_name": "category",
        "atom_index": "int16",
        "atom": "category",
        "x": "float32",
        "y": "float32",
        "z": "float32",
    },
)

struct_df = struct_df.rename(columns={"atom": "element"})[
    ["molecule_name", "atom_index", "element", "x", "y", "z"]
]

dipole_df = pd.read_csv(dipole_path)
potential_df = pd.read_csv(potential_path)

dipole_df["dipole_mag"] = np.sqrt(
    dipole_df["X"] ** 2 + dipole_df["Y"] ** 2 + dipole_df["Z"] ** 2
)

struct_idx = struct_df.set_index(["molecule_name", "atom_index"])


def enrich_with_struct(df):
    """Add element and coordinates for both atoms via fast reindex."""
    df = df.copy()
    idx0 = pd.MultiIndex.from_arrays([df["molecule_name"], df["atom_index_0"]])
    df["element_0"] = struct_idx["element"].reindex(idx0).values
    df["x_0"] = struct_idx["x"].reindex(idx0).values
    df["y_0"] = struct_idx["y"].reindex(idx0).values
    df["z_0"] = struct_idx["z"].reindex(idx0).values

    idx1 = pd.MultiIndex.from_arrays([df["molecule_name"], df["atom_index_1"]])
    df["element_1"] = struct_idx["element"].reindex(idx1).values
    df["x_1"] = struct_idx["x"].reindex(idx1).values
    df["y_1"] = struct_idx["y"].reindex(idx1).values
    df["z_1"] = struct_idx["z"].reindex(idx1).values
    return df


train_merged = enrich_with_struct(train_df)
test_merged = enrich_with_struct(test_df)


def sort_pairwise(df):
    """Ensure element_0 <= element_1 by swapping when needed."""
    mask = df["element_0"].astype(str) > df["element_1"].astype(str)
    if mask.any():
        df.loc[mask, ["element_0", "element_1"]] = df.loc[
            mask, ["element_1", "element_0"]
        ].values
        coord0 = ["x_0", "y_0", "z_0"]
        coord1 = ["x_1", "y_1", "z_1"]
        tmp = df.loc[mask, coord0].values.copy()
        df.loc[mask, coord0] = df.loc[mask, coord1].values
        df.loc[mask, coord1] = tmp
        df.loc[mask, ["atom_index_0", "atom_index_1"]] = df.loc[
            mask, ["atom_index_1", "atom_index_0"]
        ].values
    return df


train_merged = sort_pairwise(train_merged)
test_merged = sort_pairwise(test_merged)

train_merged = train_merged.merge(
    dipole_df[["molecule_name", "dipole_mag"]], on="molecule_name", how="left"
)
train_merged = train_merged.merge(potential_df, on="molecule_name", how="left")

test_merged = test_merged.merge(
    dipole_df[["molecule_name", "dipole_mag"]], on="molecule_name", how="left"
)
test_merged = test_merged.merge(potential_df, on="molecule_name", how="left")



## === cell 1
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

for df in (train_merged, test_merged):
    df["atomic_num_0"] = (
        df["element_0"].astype(str).map(atomic_number).fillna(0).astype(int)
    )
    df["atomic_num_1"] = (
        df["element_1"].astype(str).map(atomic_number).fillna(0).astype(int)
    )

for df in (train_merged, test_merged):
    df["distance"] = np.sqrt(
        (df["x_0"] - df["x_1"]) ** 2
        + (df["y_0"] - df["y_1"]) ** 2
        + (df["z_0"] - df["z_1"]) ** 2
    )

train_merged["type_code"], type_uniques = pd.factorize(train_merged["type"])
test_merged["type_code"] = type_uniques.get_indexer(test_merged["type"])
unseen = test_merged["type_code"] == -1
if unseen.any():
    new_code = len(type_uniques)
    test_merged.loc[unseen, "type_code"] = new_code



## === cell 2
combo_means = train_merged.groupby(["type", "element_0", "element_1"])[
    "scalar_coupling_constant"
].mean()
pair_means = train_merged.groupby(["element_0", "element_1"])[
    "scalar_coupling_constant"
].mean()
type_means = train_df.groupby("type")["scalar_coupling_constant"].mean()
global_mean = train_df["scalar_coupling_constant"].mean()


def baseline_predict(df):
    """
    Hierarchical mean baseline:
    1) type + element pair
    2) element pair
    3) type
    4) global mean
    Returns a NumPy array of predictions.
    """
    key_combo = pd.MultiIndex.from_arrays(
        [df["type"], df["element_0"], df["element_1"]]
    )
    pred = combo_means.reindex(key_combo).values

    missing = np.isnan(pred)
    if missing.any():
        key_pair = pd.MultiIndex.from_arrays(
            [df.loc[missing, "element_0"], df.loc[missing, "element_1"]]
        )
        pred_pair = pair_means.reindex(key_pair).values
        pred[missing] = pred_pair

    missing = np.isnan(pred)
    if missing.any():
        pred_type = df.loc[missing, "type"].map(type_means).values
        pred[missing] = pred_type

    pred = np.where(np.isnan(pred), global_mean, pred)
    return pred


train_baseline = baseline_predict(train_merged)
train_residual = train_merged["scalar_coupling_constant"].values - train_baseline

test_baseline = baseline_predict(test_merged)



## === cell 3
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

X = train_merged[feature_cols].fillna(-1).astype(np.float32).values
y = train_residual.astype(np.float32)

gbr = HistGradientBoostingRegressor(
    loss="absolute_error",
    learning_rate=0.05,
    max_iter=1200,
    max_depth=7,
    random_state=42,
)

gbr.fit(X, y)

test_features = test_merged[feature_cols].fillna(-1).astype(np.float32).values
test_residual_pred = gbr.predict(test_features)

final_pred = test_baseline + test_residual_pred

train_target_min = train_df["scalar_coupling_constant"].min()
train_target_max = train_df["scalar_coupling_constant"].max()
final_pred = np.clip(final_pred, train_target_min, train_target_max)

final_pred = np.nan_to_num(final_pred, nan=global_mean)



## === cell 4
test_df["scalar_coupling_constant"] = final_pred
submission = test_df[["id", "scalar_coupling_constant"]].copy()
submission_path = "my_blend_2.csv"
submission.to_csv(submission_path, index=False)

print(f"Submission written to {submission_path}, shape: {submission.shape}")
