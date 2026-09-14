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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
seaborn==0.12.2
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

-1.222736449496261

# 6. Current score

1.22987

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'I replace the broken stacking logic with a simple, robust baseline that reads the training data, computes the mean scalar coupling constant for each coupling type, and uses these means to predict the test set. This fixes the FileNotFound errors, ensures a valid `submission.csv` is written with the correct columns, and provides a reasonable starting score without altering any core modeling philosophy.'
- What this solution (achieved 1.23566) has done: 'I keep the original per‑type mean baseline but add a tiny validation step that finds a shrinkage factor α to blend each type’s mean with the overall global mean. This modest adjustment usually lowers the MAE (hence the log‑MAE metric) and moves the score closer to the target while preserving the core logic and still writing a correct `submission.csv`.'
- What this solution (achieved 1.23566) has done: 'I keep the original per‑type mean baseline but add a very light linear‑regression model that uses the coupling type and the two atom element symbols as features. The regression’s predictions are blended with the baseline using a small grid search for the optimal weight, which should reduce the MAE (and thus the log‑MAE) and move the score closer to the target while preserving the overall simple‑baseline logic.'
- What this solution (achieved 1.23566) has done: 'I add a simple numeric feature – the per‑type mean scalar coupling constant – to the one‑hot encoded regression model. This gives the linear model a calibrated baseline value for each coupling type, which usually reduces MAE and therefore moves the log‑MAE score closer to the target while keeping the original baseline logic unchanged. The changes are limited to feature creation and merging, and the script still writes a correct `submission.csv`.'
- What this solution (achieved 1.40944) has done: 'I fix the NaN issue that stops the LinearRegression prediction on the test set by filling missing numeric values (distance and type_mean) with the median from the training data. This allows the model to run end‑to‑end, produces a valid `submission.csv`, and keeps the original baseline‑plus‑regression logic unchanged, preserving the intended score‑direction. The changes are minimal and only affect data‑preprocessing before prediction.'
- What this solution (achieved 1.40944) has done: 'I keep the overall baseline‑plus‑regression approach but replace the plain LinearRegression with a regularized Ridge model (which still respects the original linear‑blend logic) and fill any missing distance values in the training/validation sets with the median distance. These tiny adjustments are expected to improve generalisation and lower the MAE, moving the log‑MAE score closer to the target while preserving the core pipeline.'
- What this solution (achieved 1.23244) has done: 'I make the data‑loading robust by searching for the required CSV files in any sub‑folder (handling both `data/champs‑scalar‑coupling` and the typical Kaggle `/kaggle/input` layout). This fixes the FileNotFoundError, restores the definitions of `best_alpha`, `type_means`, etc., and lets the later cells run unchanged, producing a valid `submission.csv` with the correct columns.'
- What this solution (achieved 1.22987) has done: 'We fine‑tune the shrinkage α and regression blend γ with a tighter 0.001 step size (instead of 0.01) while keeping the same search windows, so the validation MAE can improve slightly without altering the core pipeline. This minor refinement is expected to lower the overall MAE and move the score closer to the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge


def locate_file(relative_path: str) -> str:
    """
    Return an existing absolute path for a file that may be located under
    several possible data roots (e.g., ./data/... or /kaggle/input/...).
    Raises FileNotFoundError if the file cannot be found.
    """
    possible_roots = [
        Path("."),
        Path("/kaggle/input"),
        Path("/kaggle/working"),
    ]
    for root in possible_roots:
        candidate = root / relative_path
        if candidate.is_file():
            return str(candidate.resolve())
    filename = Path(relative_path).name
    for path in Path(".").rglob(filename):
        if path.is_file():
            return str(path.resolve())
    raise FileNotFoundError(f"Unable to locate {relative_path}")


data_root = "data/champs-scalar-coupling"
train_path = locate_file(os.path.join(data_root, "train.csv"))
test_path = locate_file(os.path.join(data_root, "test.csv"))
sample_sub_path = locate_file(os.path.join(data_root, "sample_submission.csv"))
structures_path = locate_file(os.path.join(data_root, "structures.csv"))

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_sub_path)
structures_df = pd.read_csv(structures_path)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Structures shape:", structures_df.shape)




## === cell 1
train_split, val_split = train_test_split(
    train_df, test_size=0.20, random_state=42, stratify=train_df["type"]
)

type_means_split = train_split.groupby("type")["scalar_coupling_constant"].mean()
global_mean_split = train_split["scalar_coupling_constant"].mean()

best_alpha = 1.0
best_mae = mean_absolute_error(
    val_split["scalar_coupling_constant"],
    val_split["type"].map(type_means_split).fillna(global_mean_split),
)
for alpha in np.linspace(0.0, 1.0, 11):
    blended = (
        alpha * val_split["type"].map(type_means_split).fillna(global_mean_split)
        + (1 - alpha) * global_mean_split
    )
    mae = mean_absolute_error(val_split["scalar_coupling_constant"], blended)
    if mae < best_mae:
        best_mae = mae
        best_alpha = alpha

refined_alphas = np.arange(
    max(0.0, best_alpha - 0.1), min(1.0, best_alpha + 0.1) + 1e-9, 0.001
)
for alpha in refined_alphas:
    blended = (
        alpha * val_split["type"].map(type_means_split).fillna(global_mean_split)
        + (1 - alpha) * global_mean_split
    )
    mae = mean_absolute_error(val_split["scalar_coupling_constant"], blended)
    if mae < best_mae:
        best_mae = mae
        best_alpha = alpha

print(f"Selected shrinkage α = {best_alpha:.4f}, validation MAE = {best_mae:.5f}")

type_means = train_df.groupby("type")["scalar_coupling_constant"].mean()
global_mean = train_df["scalar_coupling_constant"].mean()


def add_atom_features(df, structures):
    """Merge atomic coordinates, compute Euclidean distance and its square."""
    df = df.merge(
        structures.rename(
            columns={
                "atom_index": "atom_index_0",
                "atom": "atom_0",
                "x": "x0",
                "y": "y0",
                "z": "z0",
            }
        ),
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    df = df.merge(
        structures.rename(
            columns={
                "atom_index": "atom_index_1",
                "atom": "atom_1",
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
    df["distance_sq"] = df["distance"] ** 2
    return df


train_feat = add_atom_features(train_split, structures_df)
val_feat = add_atom_features(val_split, structures_df)

median_distance = train_feat["distance"].median()
train_feat["distance"] = train_feat["distance"].fillna(median_distance)
val_feat["distance"] = val_feat["distance"].fillna(median_distance)
train_feat["distance_sq"] = train_feat["distance_sq"].fillna(median_distance**2)
val_feat["distance_sq"] = val_feat["distance_sq"].fillna(median_distance**2)

train_feat["type_mean"] = (
    train_feat["type"].map(type_means_split).fillna(global_mean_split)
)
val_feat["type_mean"] = val_feat["type"].map(type_means_split).fillna(global_mean_split)

cat_cols = ["type", "atom_0", "atom_1"]
train_onehot = pd.get_dummies(train_feat[cat_cols], drop_first=False)
val_onehot = pd.get_dummies(val_feat[cat_cols], drop_first=False)
val_onehot = val_onehot.reindex(columns=train_onehot.columns, fill_value=0)

numeric_cols = ["type_mean", "distance", "distance_sq"]
train_X = pd.concat(
    [
        train_onehot.reset_index(drop=True),
        train_feat[numeric_cols].reset_index(drop=True),
    ],
    axis=1,
)
val_X = pd.concat(
    [val_onehot.reset_index(drop=True), val_feat[numeric_cols].reset_index(drop=True)],
    axis=1,
)

candidate_ridge_alphas = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

best_total_mae = best_mae
best_ridge_alpha = 1.0
best_gamma = 0.0
best_model = Ridge(alpha=best_ridge_alpha, random_state=42)
best_model.fit(train_X, train_feat["scalar_coupling_constant"])

for ridge_alpha in candidate_ridge_alphas:
    model = Ridge(alpha=ridge_alpha, random_state=42)
    model.fit(train_X, train_feat["scalar_coupling_constant"])
    ridge_pred = model.predict(val_X)

    baseline_pred = (
        best_alpha * val_split["type"].map(type_means_split).fillna(global_mean_split)
        + (1 - best_alpha) * global_mean_split
    )

    best_gamma_local = 0.0
    best_mae_local = best_total_mae
    for gamma in np.linspace(0.0, 1.0, 11):
        blended = gamma * ridge_pred + (1 - gamma) * baseline_pred
        mae = mean_absolute_error(val_split["scalar_coupling_constant"], blended)
        if mae < best_mae_local:
            best_mae_local = mae
            best_gamma_local = gamma

    refined_gammas = np.arange(
        max(0.0, best_gamma_local - 0.1),
        min(1.0, best_gamma_local + 0.1) + 1e-9,
        0.001,
    )
    for gamma in refined_gammas:
        blended = gamma * ridge_pred + (1 - gamma) * baseline_pred
        mae = mean_absolute_error(val_split["scalar_coupling_constant"], blended)
        if mae < best_mae_local:
            best_mae_local = mae
            best_gamma_local = gamma

    if best_mae_local < best_total_mae:
        best_total_mae = best_mae_local
        best_ridge_alpha = ridge_alpha
        best_gamma = best_gamma_local
        best_model = model

print(
    f"Selected Ridge α = {best_ridge_alpha:.4f}, regression blend γ = {best_gamma:.4f}, blended MAE = {best_total_mae:.5f}"
)




## === cell 2
test_pred_base = (
    best_alpha * test_df["type"].map(type_means).fillna(global_mean)
    + (1 - best_alpha) * global_mean
)

test_feat = add_atom_features(test_df, structures_df)
test_feat["type_mean"] = test_feat["type"].map(type_means).fillna(global_mean)

test_onehot = pd.get_dummies(test_feat[cat_cols], drop_first=False)
test_onehot = test_onehot.reindex(columns=train_onehot.columns, fill_value=0)

numeric_test = test_feat[["type_mean", "distance", "distance_sq"]]
test_X = pd.concat(
    [test_onehot.reset_index(drop=True), numeric_test.reset_index(drop=True)],
    axis=1,
)

test_X = test_X.fillna(
    {
        "distance": median_distance,
        "distance_sq": median_distance**2,
        "type_mean": global_mean,
    }
)

lr_test_pred = best_model.predict(test_X)

final_test_pred = best_gamma * lr_test_pred + (1 - best_gamma) * test_pred_base

submission = pd.DataFrame(
    {"id": test_df["id"], "scalar_coupling_constant": final_test_pred}
)
print("Submission preview:")
print(submission.head())




## === cell 3
output_path = "submission.csv"
submission.to_csv(output_path, index=False, float_format="%.6f")
print(f"Submission written to {output_path}")
