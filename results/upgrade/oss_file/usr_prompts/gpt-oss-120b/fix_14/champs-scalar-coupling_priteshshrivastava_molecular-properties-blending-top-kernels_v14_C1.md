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

-1.6708838544682014

# 6. Current score

1.23566

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'I replace the missing blend files with a simple baseline that reads the official training data, computes the average scalar coupling constant for each coupling type, and uses these averages to predict the test set. This fixes the FileNotFoundError, ensures a valid `submission.csv` is written with the required columns, and keeps the core logic minimal and deterministic.'
- What this solution (achieved 1.23566) has done: 'I add a lightweight atom‑type feature to the baseline: load the structures file, attach the element symbols for each atom in a pair, and compute average coupling constants for each (type, atom 0, atom 1) combination. Predictions first use these more specific averages, then fall back to the per‑type average and finally to the global mean, keeping the original simple averaging logic while aiming to lower the error toward the target score.'
- What this solution (achieved 1.23566) has done: 'I add a distance‑based feature using the atom coordinates and create a finer‑grained average (type + atom 0 + atom 1 + distance‑bucket). This keeps the simple averaging core while giving the model more specificity, which should lower the MAE toward the target. The fallback hierarchy (bucket mean → pair mean → type mean → global mean) remains unchanged, and a valid `submission.csv` is still written.'
- What this solution (achieved 1.23566) has done: 'I keep the overall averaging‑based approach but add a new fallback that uses the mean coupling for each *(type, distance bucket)* combination. This adds a bit more specificity without changing the core logic. I also make the distance buckets slightly finer (round to two decimals) to capture more detail. The prediction now falls back in this order: bucket (type + atom + distance), pair (type + atom pair), type‑distance, type, and finally the global mean.'
- What this solution (achieved 1.23566) has done: 'I add a small count‑based smoothing step: averages that are based on fewer than 5 samples are discarded so the model falls back to a higher‑level mean, which reduces noisy predictions and moves the log‑MAE lower. I also adjust the fallback order to use the type‑distance mean before the pair mean, keeping the overall averaging logic unchanged.'
- What this solution (achieved 1.23566) has done: 'I lower the minimum sample threshold to use more of the available data and create a blended prediction that combines bucket‑level and pair‑level averages when both exist (weighted by their counts). This keeps the overall averaging‑based strategy while giving each prediction a more informed estimate, which should reduce the log‑MAE and move the score closer to the target.'
- What this solution (achieved 1.23566) has done: 'I lower the minimum sample threshold to 1 so that every observed combination can contribute a mean estimate, and I make the distance buckets a bit finer by rounding distances to three decimals instead of two. These tiny adjustments keep the original averaging‑based workflow unchanged while allowing the model to use more specific statistics, which should reduce the log‑MAE and move the score closer to the target.'
- What this solution (achieved 1.23566) has done: 'I add a lightweight use of the scalar coupling contribution components (fc, sd, pso, dso) to the existing averaging pipeline. By merging the contributions file, computing per‑type and per‑pair average sums of these components, and blending this contribution‑based estimate with the current blended mean, the prediction becomes a simple average of two reasonable estimates. This keeps the core averaging logic intact while providing extra signal that should lower the log‑MAE and move the score toward the target.'
- What this solution (achieved 1.23566) has done: 'I increase the minimum sample threshold to 5 so that averages are computed from more reliable groups, and I simplify the final prediction to rely solely on the physics‑based contribution sums (pair → type) which directly model the target constant. This keeps the overall averaging framework unchanged while giving the model a stronger, more relevant signal and is expected to move the log‑MAE closer to the target negative value.'
- What this solution (achieved 1.23566) has done: 'I lowered the minimum sample threshold to 1 so that more specific group statistics are kept, and I changed the final prediction to favour the blended statistical mean (bucket/pair) and fall back to the physics‑based contribution sum only when the blended estimate is unavailable. This keeps the overall averaging workflow while providing richer, more specific predictions, which should move the log‑MAE closer to the negative target value.'
- What this solution (achieved 1.23566) has done: 'The fix adds robust dataset path detection so the CSV files load correctly, preventing the FileNotFoundError that halted execution. The rest of the script remains unchanged, preserving the original averaging‑based prediction logic while now producing a valid `submission.csv` file. This minimal change restores end‑to‑end functionality without altering core modeling behavior.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

MIN_COUNT = 1

possible_paths = [
    "data/champs-scalar-coupling",
    "/kaggle/input/champs-scalar-coupling",
    "/kaggle/working/champs-scalar-coupling",
    "/kaggle/input",
    "/kaggle/working",
    ".",
]
BASE_PATH = None
for p in possible_paths:
    if os.path.isdir(p) and os.path.isfile(os.path.join(p, "train.csv")):
        BASE_PATH = p
        break
if BASE_PATH is None:
    raise FileNotFoundError(
        "Could not locate the dataset directory containing train.csv"
    )

train_df = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
test_df = pd.read_csv(os.path.join(BASE_PATH, "test.csv"))
structures = pd.read_csv(os.path.join(BASE_PATH, "structures.csv"))
contrib_df = pd.read_csv(os.path.join(BASE_PATH, "scalar_coupling_contributions.csv"))


## === cell 1
type_means = train_df.groupby("type")["scalar_coupling_constant"].mean().reset_index()
type_means.rename(columns={"scalar_coupling_constant": "type_mean"}, inplace=True)
global_mean = train_df["scalar_coupling_constant"].mean()

atom0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
    }
)[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]]

atom1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
    }
)[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]]

train_aug = train_df.merge(atom0, on=["molecule_name", "atom_index_0"], how="left")
train_aug = train_aug.merge(atom1, on=["molecule_name", "atom_index_1"], how="left")
train_aug["distance"] = np.sqrt(
    (train_aug["x0"] - train_aug["x1"]) ** 2
    + (train_aug["y0"] - train_aug["y1"]) ** 2
    + (train_aug["z0"] - train_aug["z1"]) ** 2
)
train_aug["dist_bucket"] = train_aug["distance"].round(3)

train_aug = train_aug.merge(
    contrib_df,
    on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
    how="left",
)

pair_stats = (
    train_aug.groupby(["type", "atom_0", "atom_1"])["scalar_coupling_constant"]
    .agg(["mean", "size"])
    .reset_index()
)
pair_stats["pair_mean"] = pair_stats["mean"].where(
    pair_stats["size"] >= MIN_COUNT, np.nan
)
pair_stats.rename(columns={"size": "pair_size"}, inplace=True)
pair_means = pair_stats[["type", "atom_0", "atom_1", "pair_mean", "pair_size"]]

bucket_stats = (
    train_aug.groupby(["type", "atom_0", "atom_1", "dist_bucket"])[
        "scalar_coupling_constant"
    ]
    .agg(["mean", "size"])
    .reset_index()
)
bucket_stats["bucket_mean"] = bucket_stats["mean"].where(
    bucket_stats["size"] >= MIN_COUNT, np.nan
)
bucket_stats.rename(columns={"size": "bucket_size"}, inplace=True)
bucket_means = bucket_stats[
    ["type", "atom_0", "atom_1", "dist_bucket", "bucket_mean", "bucket_size"]
]

type_dist_means = (
    train_aug.groupby(["type", "dist_bucket"])["scalar_coupling_constant"]
    .mean()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "type_dist_mean"})
)

type_contrib_means = (
    train_aug.groupby("type")[["fc", "sd", "pso", "dso"]].mean().reset_index()
)
type_contrib_means["type_contrib_sum"] = type_contrib_means[
    ["fc", "sd", "pso", "dso"]
].sum(axis=1)

pair_contrib_means = (
    train_aug.groupby(["type", "atom_0", "atom_1"])[["fc", "sd", "pso", "dso"]]
    .mean()
    .reset_index()
)
pair_contrib_means["pair_contrib_sum"] = pair_contrib_means[
    ["fc", "sd", "pso", "dso"]
].sum(axis=1)

train_aug["contrib_sum"] = train_aug[["fc", "sd", "pso", "dso"]].sum(axis=1)
valid_mask = train_aug["contrib_sum"].notna()
X = train_aug.loc[valid_mask, "contrib_sum"].values.reshape(-1, 1)
y = train_aug.loc[valid_mask, "scalar_coupling_constant"].values
A = np.hstack([X, np.ones_like(X)])
coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
coef_a, coef_b = coeffs[0], coeffs[1]


## === cell 2
test_aug = test_df.merge(atom0, on=["molecule_name", "atom_index_0"], how="left")
test_aug = test_aug.merge(atom1, on=["molecule_name", "atom_index_1"], how="left")
test_aug["distance"] = np.sqrt(
    (test_aug["x0"] - test_aug["x1"]) ** 2
    + (test_aug["y0"] - test_aug["y1"]) ** 2
    + (test_aug["z0"] - test_aug["z1"]) ** 2
)
test_aug["dist_bucket"] = test_aug["distance"].round(3)

test_pred = test_aug.merge(
    bucket_means, on=["type", "atom_0", "atom_1", "dist_bucket"], how="left"
)
test_pred = test_pred.merge(pair_means, on=["type", "atom_0", "atom_1"], how="left")
test_pred = test_pred.merge(type_dist_means, on=["type", "dist_bucket"], how="left")
test_pred = test_pred.merge(type_means, on="type", how="left")
test_pred = test_pred.merge(
    pair_contrib_means[["type", "atom_0", "atom_1", "pair_contrib_sum"]],
    on=["type", "atom_0", "atom_1"],
    how="left",
)
test_pred = test_pred.merge(
    type_contrib_means[["type", "type_contrib_sum"]],
    on="type",
    how="left",
)


def blended_row(row):
    b_mean, b_sz = row["bucket_mean"], row["bucket_size"]
    p_mean, p_sz = row["pair_mean"], row["pair_size"]
    if pd.notna(b_mean) and pd.notna(p_mean):
        return (b_mean * b_sz + p_mean * p_sz) / (b_sz + p_sz)
    elif pd.notna(b_mean):
        return b_mean
    elif pd.notna(p_mean):
        return p_mean
    else:
        return np.nan


test_pred["blended_mean"] = test_pred.apply(blended_row, axis=1)

primary_pred = (
    test_pred["blended_mean"]
    .fillna(test_pred["bucket_mean"])
    .fillna(test_pred["pair_mean"])
    .fillna(test_pred["type_dist_mean"])
    .fillna(test_pred["type_mean"])
    .fillna(global_mean)
)

test_pred["best_contrib_sum"] = test_pred["pair_contrib_sum"].where(
    pd.notna(test_pred["pair_contrib_sum"]), test_pred["type_contrib_sum"]
)
test_pred["contrib_pred"] = test_pred["best_contrib_sum"] * coef_a + coef_b

final_pred = ((primary_pred + test_pred["contrib_pred"]) / 2).fillna(primary_pred)

test_pred["scalar_coupling_constant"] = final_pred

submission = test_pred[["id", "scalar_coupling_constant"]].copy()
output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}, shape: {submission.shape}")
