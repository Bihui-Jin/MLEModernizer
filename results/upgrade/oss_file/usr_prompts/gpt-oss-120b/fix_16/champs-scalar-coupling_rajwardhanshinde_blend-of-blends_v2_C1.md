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

-1.3684302901167014

# 6. Current score

2.14731

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'I replace the missing blend file reads with a simple baseline that predicts the mean scalar coupling constant for each coupling type using the training data. This fixes the FileNotFoundError and NameError, ensures a valid `submission.csv` is written, and provides a reasonable score without altering the core modeling approach.'
- What this solution (achieved 1.23566) has done: 'I keep the original data loading and simple mean‑by‑type baseline, but add a finer‑grained mean that also conditions on the molecule name. For each (type, molecule) pair that appears in the training set we use its specific mean; otherwise we fall back to the per‑type mean and finally the overall mean. This small feature‑aware adjustment requires only a few extra pandas operations and is expected to lower the log‑MAE toward the target without changing the overall modeling approach.'
- What this solution (achieved 1.23566) has done: 'I add atom‑type information from the structures file and use a mean prediction conditioned on the coupling type together with the two atom elements (atom_0, atom_1). This finer‑grained baseline should reduce the log‑MAE, moving the score closer to the target while keeping the original simple‑mean logic unchanged. The script now loads structures, merges atom symbols into train and test, computes the new conditional means, and falls back to the per‑type and overall means if needed, finally writing a valid submission CSV.'
- What this solution (achieved 1.23566) has done: 'I add two finer‑grained conditional mean tables – one for (type, atom_0) and one for (type, atom_1) – and use them as additional fall‑back steps before the coarse per‑type mean. This small extension keeps the original simple‑mean logic while giving the model more relevant information, which should lower the log‑MAE and move the score closer to the target (‑1.3684).'
- What this solution (achieved 1.23569) has done: 'I replace the plain group‑by means with a lightly smoothed version (adding a small prior toward the overall mean) for each conditional table. This keeps the same hierarchical‑fallback logic but reduces the impact of noisy, low‑count groups, which should lower the log‑MAE and move the score closer to the target while preserving the original workflow.'
- What this solution (achieved 1.23569) has done: 'I fixed the NaN‑handling when creating the distance bin for the test set (so the integer cast no longer crashes) and ensured the data‑processing cell runs completely, which restores all later variables (e.g., atom_pair_means). The rest of the hierarchy‑fallback prediction logic is kept unchanged, and the final cell now safely writes a valid `submission.csv`.'
- What this solution (achieved 1.23569) has done: 'I add a lightweight linear‑distance correction per coupling type: compute a simple slope + intercept model on the training data (only for types with enough samples) and use it to predict test values before the existing hierarchical mean fallback. This extra feature should lower the log‑MAE, moving the score toward the target while keeping the original baseline logic unchanged.'
- What this solution (achieved 1.23578) has done: 'I increase the smoothing prior from 5 to 20 so low‑count group means are pulled more toward the overall mean, and lower the `min_samples` threshold for the per‑type distance linear model from 200 to 50 so more coupling types benefit from a distance‑based correction. These small adjustments keep the hierarchical‑fallback logic unchanged while expectedly reducing the log‑MAE, moving the score closer to the target.'
- What this solution (achieved 1.23569) has done: 'I adjust the smoothing strength and the distance‑based linear model threshold to make the hierarchical‑fallback predictions a bit less biased and less noisy.  
- Reduce the smoothing `prior` from 20 to 5 so low‑count group means stay closer to their observed values.  
- Raise the `min_samples` for fitting a distance‑based linear correction from 50 to 200 so only well‑supported coupling types use this model, reducing the chance of over‑fitting.  
These minimal changes keep the overall logic unchanged while moving the log‑MAE toward the target lower score.'
- What this solution (achieved 1.23578) has done: 'I increase the smoothing prior from 5 to 20 so low‑count group means are pulled more toward the overall mean, and raise the distance‑model sample threshold from 200 to 500 to apply the linear correction only for well‑supported coupling types. I also clip any predicted values to the training target range to avoid extreme outliers. These tiny adjustments keep the original hierarchical‑fallback logic intact while reducing variance and expected error, moving the log‑MAE closer to the target score.'
- What this solution (achieved 1.23569) has done: 'I lower the smoothing prior from 20 to 5 so low‑count groups rely more on their observed values, and reduce the distance‑model sample threshold from 500 to 200 to let more coupling types benefit from a linear distance correction. These minimal tweaks keep the hierarchical‑fallback logic unchanged while expectedly lowering the log‑MAE toward the target score.'
- What this solution (achieved 1.23578) has done: 'I increase the smoothing prior to 20 (so low‑count groups are pulled more toward the overall mean) and lower the `min_samples` threshold to 100 so more coupling types receive a distance‑based linear correction. I also remove the aggressive clipping of predictions to the training range, allowing the hierarchical fall‑backs to provide values for entries that would otherwise be forced into an overly narrow interval. These minimal adjustments keep the original hierarchical‑fallback logic while aiming to lower the log‑MAE toward the target score.'
- What this solution (achieved 1.23569) has done: 'I lower the smoothing `prior` from 20 to 5 so low‑count groups keep more of their observed mean, and reduce the `min_samples` threshold for fitting a distance‑based linear correction from 100 to 50 so more coupling types benefit from that correction. After the hierarchical fallback I also clip predictions to the training target range to avoid extreme outliers; these minimal tweaks keep the original workflow intact while moving the log‑MAE toward the target lower score.'
- What this solution (achieved 2.14731) has done: 'I increased the smoothing prior to pull sparse group means toward the overall mean, raised the minimum sample threshold so only well‑supported coupling types receive a distance‑based linear correction, and blended the final hierarchical prediction with the overall mean to temper extreme values. These small, targeted tweaks keep the original hierarchical‑fallback logic while aiming to lower the log‑MAE toward the negative target score.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

print(os.listdir("../input"))




## === cell 1
train_path = "../input/champs-scalar-coupling/train.csv"
test_path = "../input/champs-scalar-coupling/test.csv"
structures_path = "../input/champs-scalar-coupling/structures.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
structures = pd.read_csv(structures_path)

struct0 = structures.rename(columns={"atom": "atom_0", "atom_index": "atom_index_0"})
struct1 = structures.rename(columns={"atom": "atom_1", "atom_index": "atom_index_1"})

train = train.merge(
    struct0[["molecule_name", "atom_index_0", "atom_0"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
)
train = train.merge(
    struct1[["molecule_name", "atom_index_1", "atom_1"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
)

test = test.merge(
    struct0[["molecule_name", "atom_index_0", "atom_0"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
)
test = test.merge(
    struct1[["molecule_name", "atom_index_1", "atom_1"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
)

coords0 = structures.rename(
    columns={"atom_index": "atom_index_0", "x": "x0", "y": "y0", "z": "z0"}
)[["molecule_name", "atom_index_0", "x0", "y0", "z0"]]

coords1 = structures.rename(
    columns={"atom_index": "atom_index_1", "x": "x1", "y": "y1", "z": "z1"}
)[["molecule_name", "atom_index_1", "x1", "y1", "z1"]]

train = train.merge(coords0, on=["molecule_name", "atom_index_0"], how="left")
train = train.merge(coords1, on=["molecule_name", "atom_index_1"], how="left")
test = test.merge(coords0, on=["molecule_name", "atom_index_0"], how="left")
test = test.merge(coords1, on=["molecule_name", "atom_index_1"], how="left")

train["distance"] = np.sqrt(
    (train["x0"] - train["x1"]) ** 2
    + (train["y0"] - train["y1"]) ** 2
    + (train["z0"] - train["z1"]) ** 2
)
test["distance"] = np.sqrt(
    (test["x0"] - test["x1"]) ** 2
    + (test["y0"] - test["y1"]) ** 2
    + (test["z0"] - test["z1"]) ** 2
)

test["distance_bin"] = (test["distance"].fillna(-1) * 10).round().astype(int)
train["distance_bin"] = (train["distance"] * 10).round().astype(int)

overall_mean = train["scalar_coupling_constant"].mean()
prior = 20


def smoothed_mean(series):
    """Return (sum + prior*overall_mean) / (count + prior)."""
    return (series.sum() + prior * overall_mean) / (len(series) + prior)


type_means = train.groupby("type")["scalar_coupling_constant"].apply(smoothed_mean)

pair_means = train.groupby(["type", "molecule_name"])["scalar_coupling_constant"].apply(
    smoothed_mean
)

atom_pair_means = train.groupby(["type", "atom_0", "atom_1"])[
    "scalar_coupling_constant"
].apply(smoothed_mean)

type_atom0_means = train.groupby(["type", "atom_0"])["scalar_coupling_constant"].apply(
    smoothed_mean
)

type_atom1_means = train.groupby(["type", "atom_1"])["scalar_coupling_constant"].apply(
    smoothed_mean
)

type_dist_means = train.groupby(["type", "distance_bin"])[
    "scalar_coupling_constant"
].apply(smoothed_mean)

type_distance_models = {}
min_samples = 200
for t, grp in train.groupby("type"):
    if len(grp) >= min_samples:
        x = grp["distance"].fillna(grp["distance"].median())
        y = grp["scalar_coupling_constant"]
        slope, intercept = np.polyfit(x, y, 1)
        type_distance_models[t] = (slope, intercept)




## === cell 2
linear_pred = pd.Series(np.nan, index=test.index)
for t, (slope, intercept) in type_distance_models.items():
    mask = test["type"] == t
    linear_pred.loc[mask] = slope * test.loc[mask, "distance"] + intercept
test["scalar_coupling_constant"] = linear_pred

test_key_atom = list(zip(test["type"], test["atom_0"], test["atom_1"]))
test["scalar_coupling_constant"] = test["scalar_coupling_constant"].fillna(
    pd.Series(test_key_atom).map(atom_pair_means)
)

test_key_type_mol = list(zip(test["type"], test["molecule_name"]))
test["scalar_coupling_constant"] = test["scalar_coupling_constant"].fillna(
    pd.Series(test_key_type_mol).map(pair_means)
)

test_key_type_atom0 = list(zip(test["type"], test["atom_0"]))
test["scalar_coupling_constant"] = test["scalar_coupling_constant"].fillna(
    pd.Series(test_key_type_atom0).map(type_atom0_means)
)

test_key_type_atom1 = list(zip(test["type"], test["atom_1"]))
test["scalar_coupling_constant"] = test["scalar_coupling_constant"].fillna(
    pd.Series(test_key_type_atom1).map(type_atom1_means)
)

test_key_type_dist = list(zip(test["type"], test["distance_bin"]))
test["scalar_coupling_constant"] = test["scalar_coupling_constant"].fillna(
    pd.Series(test_key_type_dist).map(type_dist_means)
)

test["scalar_coupling_constant"] = test["scalar_coupling_constant"].fillna(
    test["type"].map(type_means)
)

test["scalar_coupling_constant"] = test["scalar_coupling_constant"].fillna(overall_mean)

train_min = train["scalar_coupling_constant"].min()
train_max = train["scalar_coupling_constant"].max()
test["scalar_coupling_constant"] = test["scalar_coupling_constant"].clip(
    lower=train_min, upper=train_max
)

blend_weight = 0.6
test["scalar_coupling_constant"] = (
    blend_weight * test["scalar_coupling_constant"] + (1 - blend_weight) * overall_mean
)

submission = test[["id", "scalar_coupling_constant"]]




## === cell 3
submission.to_csv("submission.csv", index=False)
print(
    "Submission file written to submission.csv with shape:",
    submission.shape,
)
