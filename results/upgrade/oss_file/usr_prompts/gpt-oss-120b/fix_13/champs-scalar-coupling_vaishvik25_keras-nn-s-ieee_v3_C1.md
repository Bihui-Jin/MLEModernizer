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

-1.5059779559326905

# 6. Current score

1.64963

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'I replace the failing cells with a simple, robust pipeline that loads the training and test data from the correct Kaggle input directory, computes the mean scalar coupling constant for each coupling type, applies these means as predictions for the test set, and writes a properly formatted `submission.csv`. This fixes the file‑not‑found errors and ensures a valid submission file is produced, while using a reasonable baseline that moves the score toward the target without altering any core modeling logic.'
- What this solution (achieved 1.18497) has done: 'I replace the per‑type aggregation from a simple mean to a median, which is more robust to outliers and usually lowers the MAE (hence the log‑MAE score). The rest of the pipeline stays unchanged, but the predictions become slightly better and move the score closer to the negative target.'
- What this solution (achieved 1.18497) has done: 'I add a lightweight feature‑engineered baseline that groups by coupling type together with the two atom element symbols (derived from the structures file). This yields more specific medians than using only the type, while still keeping the simple median‑based prediction logic. Missing groups fall back to the per‑type median and finally to the global median, ensuring a valid prediction for every test row and moving the MAE (and thus log‑MAE) toward the negative target.'
- What this solution (achieved 1.18497) has done: 'I add two additional median look‑ups: one per coupling type + atom0 and another per coupling type + atom1. During prediction the missing combo‑median values be filled with the average of these two more specific medians before falling back to the per‑type median and finally the global median. This keeps the original simple median‑based approach while giving more granular estimates, which should lower the MAE and move the log‑MAE score closer to the negative target.'
- What this solution (achieved 1.18497) has done: 'I add a lightweight distance‑based feature to the existing median‑lookup pipeline. After merging atom symbols I also merge the atomic coordinates, compute the Euclidean distance between the two atoms and bucket it to 0.5 Å. A new median table is built for each (type, atom0, atom1, distance‑bucket) group and used as the first prediction source, falling back to the previous combo‑median, atom‑specific medians, type‑median and finally the global median. This small, deterministic enhancement keeps the overall median‑based logic while aiming to lower the log‑MAE toward the negative target.'
- What this solution (achieved 1.18497) has done: 'I reduce the distance‑bin size from 0.5 Å to 0.25 Å so the “type‑atom‑atom‑distance” groups become more specific, which usually improves the median‑based predictions and lowers the MAE (and thus the log‑MAE) without altering the overall modelling logic. The rest of the pipeline and fallback hierarchy stay unchanged, ensuring a valid `submission.csv` is still produced.'
- What this solution (achieved 1.18497) has done: 'The fix reorders the notebook so that data loading and feature creation happen before any aggregation or prediction steps. All variables (`train_df`, `test_df`, `global_median`, etc.) are defined before they are used, eliminating the NameError crashes. The core median‑based prediction logic and hierarchical fallback remain unchanged, ensuring the same modeling approach while now producing a valid `submission.csv` file.'
- What this solution (achieved 1.18497) has done: 'I lower the distance‑bin size to 0.05 Å for finer grouping and, when both a distance‑based median and a plain (type‑atom‑atom) median exist, replace the distance‑based value with their average. This keeps the original hierarchical median‑lookup logic while giving a slightly more calibrated prediction, which should reduce the log‑MAE and move the score closer to the negative target.'
- What this solution (achieved 1.18497) has done: 'I add a “mean‑based” fallback for the most specific grouping (type + atom 0 + atom 1 + distance bin). This keeps the original median hierarchy but uses the mean where it is available, which often yields a tighter estimate and should lower the MAE, moving the log‑MAE score closer to the negative target while preserving the overall pipeline.'
- What this solution (achieved 1.3023) has done: 'I add a modest shrink‑toward‑the‑global‑median step after the hierarchical fallback finishes. By blending each prediction slightly with the overall median (e.g., 85 % prediction + 15 % global median) we reduce extreme errors without changing the core grouping logic, which should lower the MAE and thus move the log‑MAE closer to the negative target.'
- What this solution (achieved 1.64963) has done: 'I increase the distance‑bin width to 0.1 Å so each distance bucket contains more samples, making the median/mean estimates more stable. Then I raise the shrink‑toward‑global‑median factor from 0.15 to 0.5, blending predictions more heavily with the overall median to reduce extreme errors. Both changes are tiny adjustments to the existing pipeline and are expected to lower the log‑MAE, moving the score toward the negative target while keeping the core logic intact.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

BASE_PATH = "/kaggle/input/champs-scalar-coupling"
TRAIN_PATH = os.path.join(BASE_PATH, "train.csv")
TEST_PATH = os.path.join(BASE_PATH, "test.csv")
STRUCTURES_PATH = os.path.join(BASE_PATH, "structures.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
structures_df = pd.read_csv(STRUCTURES_PATH)


def add_atom_info(df, idx_col, suffix):
    merged = df.merge(
        structures_df[["molecule_name", "atom_index", "atom", "x", "y", "z"]],
        left_on=["molecule_name", idx_col],
        right_on=["molecule_name", "atom_index"],
        how="left",
    )
    merged = merged.rename(
        columns={
            "atom": f"atom{suffix}",
            "x": f"x{suffix}",
            "y": f"y{suffix}",
            "z": f"z{suffix}",
        }
    )
    merged = merged.drop(columns=["atom_index"])
    return merged


train_df = add_atom_info(train_df, "atom_index_0", "0")
train_df = add_atom_info(train_df, "atom_index_1", "1")
test_df = add_atom_info(test_df, "atom_index_0", "0")
test_df = add_atom_info(test_df, "atom_index_1", "1")


def compute_distance(df, bin_width=0.1):
    df["distance"] = np.sqrt(
        (df["x0"] - df["x1"]) ** 2
        + (df["y0"] - df["y1"]) ** 2
        + (df["z0"] - df["z1"]) ** 2
    )
    df["dist_bin"] = (df["distance"] / bin_width).round() * bin_width
    return df


train_df = compute_distance(train_df, bin_width=0.1)
test_df = compute_distance(test_df, bin_width=0.1)




## === cell 1
combo_dist_medians = (
    train_df.groupby(["type", "atom0", "atom1", "dist_bin"])["scalar_coupling_constant"]
    .median()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "predicted_combo_dist"})
)

combo_dist_means = (
    train_df.groupby(["type", "atom0", "atom1", "dist_bin"])["scalar_coupling_constant"]
    .mean()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "predicted_combo_dist_mean"})
)

combo_dist_atom0_medians = (
    train_df.groupby(["type", "atom0", "dist_bin"])["scalar_coupling_constant"]
    .median()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "predicted_combo_dist_atom0"})
)

combo_dist_atom1_medians = (
    train_df.groupby(["type", "atom1", "dist_bin"])["scalar_coupling_constant"]
    .median()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "predicted_combo_dist_atom1"})
)

combo_medians = (
    train_df.groupby(["type", "atom0", "atom1"])["scalar_coupling_constant"]
    .median()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "predicted_combo"})
)

type_medians = (
    train_df.groupby("type")["scalar_coupling_constant"]
    .median()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "predicted_type"})
)

type_atom0_medians = (
    train_df.groupby(["type", "atom0"])["scalar_coupling_constant"]
    .median()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "pred_type_atom0"})
)

type_atom1_medians = (
    train_df.groupby(["type", "atom1"])["scalar_coupling_constant"]
    .median()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "pred_type_atom1"})
)

global_median = train_df["scalar_coupling_constant"].median()




## === cell 2
pred_df = test_df.merge(
    combo_dist_medians, on=["type", "atom0", "atom1", "dist_bin"], how="left"
)
pred_df = pred_df.merge(
    combo_dist_means, on=["type", "atom0", "atom1", "dist_bin"], how="left"
)

pred_df = pred_df.merge(combo_medians, on=["type", "atom0", "atom1"], how="left")
pred_df = pred_df.merge(
    combo_dist_atom0_medians, on=["type", "atom0", "dist_bin"], how="left"
)
pred_df = pred_df.merge(
    combo_dist_atom1_medians, on=["type", "atom1", "dist_bin"], how="left"
)
pred_df = pred_df.merge(type_medians, on="type", how="left")
pred_df = pred_df.merge(type_atom0_medians, on=["type", "atom0"], how="left")
pred_df = pred_df.merge(type_atom1_medians, on=["type", "atom1"], how="left")

pred_df["predicted_combo_dist"] = pred_df["predicted_combo_dist_mean"].where(
    pred_df["predicted_combo_dist_mean"].notna(),
    pred_df["predicted_combo_dist"],
)

both_present = (
    pred_df["predicted_combo_dist"].notna() & pred_df["predicted_combo"].notna()
)
pred_df.loc[both_present, "predicted_combo_dist"] = (
    pred_df.loc[both_present, "predicted_combo_dist"]
    + pred_df.loc[both_present, "predicted_combo"]
) / 2

pred_df["predicted"] = pred_df["predicted_combo_dist"]

mask = pred_df["predicted"].isna()
pred_df.loc[mask, "predicted"] = pred_df.loc[mask, "predicted_combo_dist_atom0"]

mask = pred_df["predicted"].isna()
pred_df.loc[mask, "predicted"] = pred_df.loc[mask, "predicted_combo_dist_atom1"]

mask = pred_df["predicted"].isna()
pred_df.loc[mask, "predicted"] = pred_df.loc[mask, "predicted_combo"]

mask = pred_df["predicted"].isna()
pred_df.loc[mask, "predicted"] = (
    pred_df.loc[mask, "pred_type_atom0"] + pred_df.loc[mask, "pred_type_atom1"]
) / 2

mask = pred_df["predicted"].isna()
pred_df.loc[mask, "predicted"] = pred_df.loc[mask, "predicted_type"]

pred_df["predicted"].fillna(global_median, inplace=True)

shrink_factor = 0.5  # increased blending with global median to damp extreme errors
pred_df["predicted"] = (1 - shrink_factor) * pred_df[
    "predicted"
] + shrink_factor * global_median




## === cell 3
submission = pred_df[["id", "predicted"]].rename(
    columns={"predicted": "scalar_coupling_constant"}
)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False, float_format="%.6f")
print(f"Submission saved to {submission_path}")
