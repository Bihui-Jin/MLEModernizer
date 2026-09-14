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

0.2447767932130148

# 6. Current score

1.23566

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'I replace the missing external prediction files with a simple baseline that uses the mean scalar coupling constant per coupling type from the training data. This fixes the file‑not‑found errors, ensures a valid `submission.csv` is written, and adds a quick validation step to show the resulting Log‑MAE score, moving the solution toward the target metric without altering the core modeling approach.'
- What this solution (achieved 1.23566) has done: 'I add a lightweight feature that incorporates the atom element types for each pair, merging this information from `structures.csv` and creating a combined “type‑atom0‑atom1” key. The model first try to predict using the mean target for this detailed key; if a key is missing it falls back to the original per‑type mean. This small enrichment typically lowers the Log‑MAE toward the target without changing the core baseline logic.'
- What this solution (achieved 2.11738) has done: 'Implemented fixes to resolve missing “baseline” column errors by assigning baseline values to the training split as done for validation and test sets. Added a quadratic distance feature to the residual model for modest predictive improvement. Ensured `test_pred` is correctly computed before creating the submission file, and streamlined column handling to guarantee a valid `submission.csv` is written.'
- What this solution (achieved 5.68849) has done: 'The fix adds a simple imputation step for the distance‑based features to remove NaNs that caused LinearRegression to fail on the test set, and ensures the imputed value is applied consistently to train, validation, and test features. After this change the pipeline runs end‑to‑end and writes a valid `submission.csv`.'
- What this solution (achieved 4.10756) has done: 'I enrich the data with additional numeric features (atom type encodings, dipole moments, and potential energy) and include them in the per‑type linear‑regression residual model. This adds useful signal without changing the overall baseline‑plus‑residual architecture, and should decrease the Log‑MAE → bringing the validation score much closer to the target.'
- What this solution (achieved 1.23566) has done: 'I simplify the prediction step by discarding the noisy residual model and using only the baseline mean (type + atom‑pair) as the final prediction. This removes the over‑fitting linear‑regression residuals, which caused an extremely high Log‑MAE (≈4.1), and should pull the validation score much closer to the target (lower is better). The core data‑processing and feature‑engineering remain unchanged, and the script still writes a proper `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression




## === cell 1
SEED = 31
TARGET = "scalar_coupling_constant"


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


seed_everything(SEED)




## === cell 2
def group_mean_log_mae(
    y_true: pd.Series, y_pred: pd.Series, types: pd.Series, floor: float = 1e-9
) -> float:
    maes = (y_true - y_pred).abs().groupby(types).mean()
    maes = np.log(maes.map(lambda x: max(x, floor)))
    return maes.mean()




## === cell 3
train_path = "/kaggle/input/champs-scalar-coupling/train.csv"
test_path = "/kaggle/input/champs-scalar-coupling/test.csv"
structures_path = "/kaggle/input/champs-scalar-coupling/structures.csv"
dipole_path = "/kaggle/input/champs-scalar-coupling/dipole_moments.csv"
potential_path = "/kaggle/input/champs-scalar-coupling/potential_energy.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
structures_df = pd.read_csv(structures_path)
dipole_df = pd.read_csv(dipole_path).rename(
    columns={"X": "dipole_X", "Y": "dipole_Y", "Z": "dipole_Z"}
)
potential_df = pd.read_csv(potential_path).rename(
    columns={"potential_energy": "potential_energy"}
)

print(
    f"train shape: {train_df.shape}, test shape: {test_df.shape}, "
    f"structures shape: {structures_df.shape}"
)


def add_atom_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(
        structures_df[["molecule_name", "atom_index", "atom"]].rename(
            columns={"atom_index": "atom_index_0", "atom": "atom_0"}
        ),
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    df = df.merge(
        structures_df[["molecule_name", "atom_index", "atom"]].rename(
            columns={"atom_index": "atom_index_1", "atom": "atom_1"}
        ),
        on=["molecule_name", "atom_index_1"],
        how="left",
    )
    return df


def add_distance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(
        structures_df[["molecule_name", "atom_index", "x", "y", "z"]].rename(
            columns={
                "atom_index": "atom_index_0",
                "x": "x0",
                "y": "y0",
                "z": "z0",
            }
        ),
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    df = df.merge(
        structures_df[["molecule_name", "atom_index", "x", "y", "z"]].rename(
            columns={
                "atom_index": "atom_index_1",
                "x": "x1",
                "y": "y1",
                "z": "z1",
            }
        ),
        on=["molecule_name", "atom_index_1"],
        how="left",
    )
    df["distance"] = np.sqrt(
        (df["x0"] - df["x1"]) ** 2
        + (df["y0"] - df["y1"]) ** 2
        + (df["z0"] - df["z1"]) ** 2
    )
    return df


def add_global_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(dipole_df, on="molecule_name", how="left")
    df = df.merge(potential_df, on="molecule_name", how="left")
    return df


atom_encoder = {a: i for i, a in enumerate(structures_df["atom"].unique())}


def encode_atom_types(df: pd.DataFrame) -> pd.DataFrame:
    df["atom0_enc"] = df["atom_0"].map(atom_encoder).fillna(-1).astype(int)
    df["atom1_enc"] = df["atom_1"].map(atom_encoder).fillna(-1).astype(int)
    return df


train_df = add_atom_types(train_df)
test_df = add_atom_types(test_df)

train_df = add_distance(train_df)
test_df = add_distance(test_df)

train_df = add_global_features(train_df)
test_df = add_global_features(test_df)

train_df = encode_atom_types(train_df)
test_df = encode_atom_types(test_df)




## === cell 4
train_split, val_split = train_test_split(
    train_df, test_size=0.2, random_state=SEED, stratify=train_df["type"]
)


def make_key(df: pd.DataFrame) -> pd.Series:
    a0 = df["atom_0"].fillna("X")
    a1 = df["atom_1"].fillna("X")
    sorted_atoms = np.where(a0 <= a1, a0 + "_" + a1, a1 + "_" + a0)
    return df["type"] + "_" + sorted_atoms


full_key_means = train_df.groupby(make_key(train_df))[TARGET].mean()
type_means = train_df.groupby("type")[TARGET].mean()

for split in (train_df, val_split, train_split, test_df):
    split["baseline"] = make_key(split).map(full_key_means)
    split.loc[split["baseline"].isna(), "baseline"] = split.loc[
        split["baseline"].isna(), "type"
    ].map(type_means)

train_split["residual"] = train_split[TARGET] - train_split["baseline"]




## === cell 5
def prep_features(df):
    X = df[["distance"]].copy()
    X["distance_sq"] = X["distance"] ** 2
    X["atom0_enc"] = df["atom0_enc"]
    X["atom1_enc"] = df["atom1_enc"]
    X["dipole_X"] = df["dipole_X"]
    X["dipole_Y"] = df["dipole_Y"]
    X["dipole_Z"] = df["dipole_Z"]
    X["potential_energy"] = df["potential_energy"]
    return X


X_train_base = prep_features(train_split)
X_val_base = prep_features(val_split)
X_test_base = prep_features(test_df)

impute_vals = X_train_base.median()
X_train_base = X_train_base.fillna(impute_vals)
X_val_base = X_val_base.fillna(impute_vals)
X_test_base = X_test_base.fillna(impute_vals)

type_models = {}
for typ in train_split["type"].unique():
    mask = train_split["type"] == typ
    lr = LinearRegression()
    lr.fit(X_train_base[mask], train_split.loc[mask, "residual"])
    type_models[typ] = lr

val_res_pred = np.zeros(len(val_split))  # no residual contribution
val_pred = val_split["baseline"] + val_res_pred

val_score = group_mean_log_mae(
    y_true=val_split[TARGET], y_pred=val_pred, types=val_split["type"]
)
print(f"Validation Log‑MAE (baseline only): {val_score:.5f}")

test_res_pred = np.zeros(len(test_df))  # no residual contribution
test_pred = test_df["baseline"] + test_res_pred




## === cell 6
submission = pd.DataFrame({"id": test_df["id"], TARGET: test_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print(f"Submission written to {submission_path}")
print("Current directory listing:", os.listdir("."))
