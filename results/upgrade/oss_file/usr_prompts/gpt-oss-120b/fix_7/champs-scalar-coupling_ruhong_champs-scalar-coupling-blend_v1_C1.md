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

0.3023773260272986

# 6. Current score

1.23566

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'I replace the missing external prediction files with a simple baseline that uses the average scalar coupling constant for each coupling type from the training data. This ensures the script runs without file‑not‑found errors, creates a valid `submission.csv` with the correct columns, and provides reasonable predictions that move the score toward the target.'
- What this solution (achieved 1.23566) has done: 'I add a small feature‑engineered baseline that uses not only the coupling type but also the two atom elements (and their distance) to compute a more specific mean target for each “type‑atom0‑atom1” combination. This keeps the simple mean‑prediction logic, adds only inexpensive merging and arithmetic, and is expected to lower the log‑MAE toward the target without changing the overall workflow.'
- What this solution (achieved 1.23566) has done: 'I add a lightweight distance‑binned mean feature and use it as an additional fallback when the more specific “type‑atom‑pair” mean is missing. This keeps the overall mean‑based approach while giving slightly better calibration, especially for rare combinations, and should lower the log‑MAE toward the target without altering the core modeling logic.'
- What this solution (achieved 1.23566) has done: 'I add a simple distance‑adjusted mean model: for each `type_atom_pair` we compute the mean target, the mean distance and a linear slope (target vs distance). Predictions use `mean_target + slope*(dist‑mean_dist)` when the pair exists, otherwise they fall back to the existing hierarchy (distance‑bin, type, global). This keeps the original baseline logic while giving a finer‑grained estimate, which should move the log‑MAE closer to the target.'
- What this solution (achieved 1.23566) has done: 'I add a symmetric‑pair fallback to the prediction logic: if the “type_atom_pair” key (type + atom0 + atom1) is not found, the code try the reversed order (type + atom1 + atom0) using the same mean‑target, distance‑adjustment and slope maps before falling back to the broader averages. This small change keeps the core model untouched while giving the model extra information that should lower the log‑MAE and move the score toward the target.'
- What this solution (achieved 1.23566) has done: 'I tweak the prediction function to blend the distance‑bin and type‑level averages instead of using a pure fallback hierarchy. By averaging these two sensible baselines before falling back to the global mean, the model gets a slightly better calibrated estimate for cases where the detailed “type‑atom‑pair” statistics are missing, which should lower the log‑MAE toward the target without altering the overall architecture.'

# 9. Code solution

## === cell 0
import os
import random
import sys
import numpy as np
import pandas as pd




## === cell 1
SEED = 31
TARGET = "scalar_coupling_constant"
PREDICTION = "pred"


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


seed_everything(SEED)




## === cell 2
train_path = "../input/champs-scalar-coupling/train.csv"
test_path = "../input/champs-scalar-coupling/test.csv"
structures_path = "../input/champs-scalar-coupling/structures.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
structures_df = pd.read_csv(structures_path)

print(
    f"train shape: {train_df.shape}, test shape: {test_df.shape}, structures shape: {structures_df.shape}"
)


def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    """
    Fast metric computation for this competition:
    log(mean absolute error) per coupling type, then averaged.
    """
    maes = (y_true - y_pred).abs().groupby(types).mean()
    maes = np.log(maes.map(lambda x: max(x, floor)))
    return maes.mean()


atom0 = structures_df.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
    }
)[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]]

atom1 = structures_df.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
    }
)[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]]


def enrich(df):
    df = df.merge(atom0, on=["molecule_name", "atom_index_0"], how="left")
    df = df.merge(atom1, on=["molecule_name", "atom_index_1"], how="left")
    df["dist"] = np.sqrt(
        (df["x0"] - df["x1"]) ** 2
        + (df["y0"] - df["y1"]) ** 2
        + (df["z0"] - df["z1"]) ** 2
    )
    df["type_atom_pair"] = df["type"] + "_" + df["atom_0"] + "_" + df["atom_1"]
    df["dist_bin"] = pd.qcut(df["dist"], q=10, duplicates="drop")
    df["type_dist_bin"] = df["type"].astype(str) + "_" + df["dist_bin"].astype(str)
    return df


train_enriched = enrich(train_df)
test_enriched = enrich(test_df)

key_means = train_enriched.groupby("type_atom_pair")[TARGET].mean()
type_means = train_enriched.groupby("type")[TARGET].mean()
global_mean = train_enriched[TARGET].mean()
dist_means = train_enriched.groupby("type_dist_bin")[TARGET].mean()


def compute_group_stats(g):
    if len(g) < 2:
        slope = 0.0
    else:
        cov = np.cov(g["dist"], g[TARGET], bias=True)[0, 1]
        var = np.var(g["dist"])
        slope = cov / (var + 1e-9)
    return pd.Series(
        {
            "mean_target": g[TARGET].mean(),
            "mean_dist": g["dist"].mean(),
            "slope": slope,
        }
    )


group_stats = train_enriched.groupby("type_atom_pair").apply(compute_group_stats)

mean_target_map = group_stats["mean_target"]
mean_dist_map = group_stats["mean_dist"]
slope_map = group_stats["slope"]

shuffled = train_enriched.sample(frac=1, random_state=SEED)
split_idx = int(0.8 * len(shuffled))
val_df = shuffled.iloc[split_idx:]


def predict_series(df):
    pred = df["type_atom_pair"].map(mean_target_map)
    adj = (df["dist"] - df["type_atom_pair"].map(mean_dist_map).fillna(0)) * df[
        "type_atom_pair"
    ].map(slope_map).fillna(0)
    pred = pred + adj

    rev_key = df["type"] + "_" + df["atom_1"] + "_" + df["atom_0"]
    rev_pred = rev_key.map(mean_target_map)
    rev_adj = (df["dist"] - rev_key.map(mean_dist_map).fillna(0)) * rev_key.map(
        slope_map
    ).fillna(0)
    rev_pred = rev_pred + rev_adj

    pred = pred.fillna(rev_pred)

    fb1 = df["type_dist_bin"].map(dist_means)
    fb2 = df["type"].map(type_means)
    blended = (fb1 + fb2) / 2.0
    pred = pred.fillna(blended)

    pred = pred.fillna(fb1)
    pred = pred.fillna(fb2)

    pred = pred.fillna(global_mean)
    return pred


val_pred = predict_series(val_df)

baseline_score = group_mean_log_mae(val_df[TARGET], val_pred, val_df["type"])
print(f"Improved validation score (log‑MAE): {baseline_score:.5f}")




## === cell 3
test_pred = predict_series(test_enriched)

submission = test_enriched[["id"]].copy()
submission[TARGET] = test_pred

print("First few predictions:")
print(submission.head())




## === cell 4
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print("Current directory contents:", os.listdir("."))
