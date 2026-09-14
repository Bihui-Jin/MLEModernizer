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

-1.32544

# 6. Current score

1.23566

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 3.00563) has done: 'The fix replaces the missing “blender” submission files with a simple baseline: it loads the official train and test data, computes the overall mean scalar coupling constant from the training set, and writes a valid submission containing that mean for every test row. Paths are made robust with existence checks, and the script now always produces `submission1236.csv` with the correct columns, eliminating the FileNotFound and NameError failures while keeping the original blending idea minimal. This ensures the notebook runs end‑to‑end and generates a proper CSV submission.'
- What this solution (achieved 1.23566) has done: 'The update replaces the naïve global‑mean baseline with a per‑coupling‑type mean prediction, which is a minimal yet effective change that better respects the variation across coupling types and is expected to lower the log‑MAE toward the target. The blending branch remains unchanged, and the submission file is still written as before.'
- What this solution (achieved 1.24184) has done: 'I keep the existing blending logic for when external submissions exist, but improve the fallback branch by training a very lightweight linear regression on the training rows (using atom indices and one‑hot encoded coupling type) and blending its predictions with the per‑type mean baseline. This adds only a fast, interpretable model while preserving the overall structure, and should lower the MAE (and thus the log‑MAE) toward the target score.'
- What this solution (achieved 1.23566) has done: 'The fallback branch was blending a simple per‑type mean with a very lightweight linear regression, which was slightly worsening the validation score. By removing the regression and using only the per‑type mean (with overall mean as a fallback), the predictions become more stable and move the log‑MAE closer to the target lower value.'
- What this solution (achieved 1.23566) has done: 'I add lightweight feature engineering by merging atom element information from structures.csv into the train and test frames and then use the mean scalar coupling for each (type, element0, element1) triple as a more specific baseline. Missing groups fall back to the original per‑type mean, preserving the existing logic while improving prediction accuracy and moving the log‑MAE toward the target lower score. The rest of the pipeline (blending fallback, CSV output) stays unchanged.'
- What this solution (achieved 1.23566) has done: 'I enhance the fallback prediction by adding hierarchical group‑mean look‑ups: first use the mean for the exact (type, element0, element1) triple, then fall back to means for (type, element0) and (type, element1), before finally using the per‑type and overall means. This keeps the original logic but provides more specific baselines where data exist, moving the log‑MAE lower toward the target.'

# 9. Code solution

## === cell 0
import os, numpy as np, pandas as pd

print("Input dirs:", os.listdir("../input"))




## === cell 1
def locate(path):
    if os.path.exists(path):
        return path
    alt = os.path.abspath(os.path.join("../..", path))
    return alt if os.path.exists(alt) else path


def safe_read(csv_path):
    try:
        return pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"File not found: {csv_path}. Using empty placeholder.")
        return pd.DataFrame()


blender_dir = locate("../input/blender")
sub1 = safe_read(os.path.join(blender_dir, "LGB_2019-07-11_-1.4378.csv"))
sub2 = safe_read(os.path.join(blender_dir, "submission-2.csv"))
sub3 = safe_read(os.path.join(blender_dir, "stack_minmax_median.csv"))
sub4 = safe_read(os.path.join(blender_dir, "stack_mean.csv"))
sub5 = safe_read(os.path.join(blender_dir, "stack_median.csv"))
sub6 = safe_read(os.path.join(blender_dir, "workingsubmission-test.csv"))

train_path = locate("../input/champs-scalar-coupling/train.csv")
test_path = locate("../input/champs-scalar-coupling/test.csv")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

structures_path = locate("../input/champs-scalar-coupling/structures.csv")
structures_df = safe_read(structures_path)

if not structures_df.empty:
    elem0 = structures_df.rename(
        columns={"atom": "element0", "atom_index": "atom_index_0"}
    )
    train_df = train_df.merge(
        elem0[["molecule_name", "atom_index_0", "element0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    test_df = test_df.merge(
        elem0[["molecule_name", "atom_index_0", "element0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    elem1 = structures_df.rename(
        columns={"atom": "element1", "atom_index": "atom_index_1"}
    )
    train_df = train_df.merge(
        elem1[["molecule_name", "atom_index_1", "element1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
    )
    test_df = test_df.merge(
        elem1[["molecule_name", "atom_index_1", "element1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
    )
else:
    train_df["element0"] = np.nan
    train_df["element1"] = np.nan
    test_df["element0"] = np.nan
    test_df["element1"] = np.nan




## === cell 2
if not sub1.empty and not sub2.empty and not sub3.empty and not sub6.empty:
    cols = ["id", "scalar_coupling_constant"]
    sub1 = sub1[cols].set_index("id")
    sub2 = sub2[cols].set_index("id")
    sub3 = sub3[cols].set_index("id")
    sub6 = sub6[cols].set_index("id")
    blended = (
        0.3 * sub1["scalar_coupling_constant"]
        + 0.3 * sub2["scalar_coupling_constant"]
        + 0.15 * sub3["scalar_coupling_constant"]
        + 0.25 * sub6["scalar_coupling_constant"]
    )
    submission = blended.reset_index()
else:
    type_means = train_df.groupby("type")["scalar_coupling_constant"].mean()
    overall_mean = train_df["scalar_coupling_constant"].mean()

    triple_means = (
        train_df.groupby(["type", "element0", "element1"])["scalar_coupling_constant"]
        .mean()
        .reset_index()
        .rename(columns={"scalar_coupling_constant": "triple_mean"})
    )
    type_elem0_means = (
        train_df.groupby(["type", "element0"])["scalar_coupling_constant"]
        .mean()
        .reset_index()
        .rename(columns={"scalar_coupling_constant": "elem0_mean"})
    )
    type_elem1_means = (
        train_df.groupby(["type", "element1"])["scalar_coupling_constant"]
        .mean()
        .reset_index()
        .rename(columns={"scalar_coupling_constant": "elem1_mean"})
    )

    baseline = test_df[["id", "type", "element0", "element1"]].copy()
    baseline = baseline.join(type_means, on="type", rsuffix="_type_mean")
    baseline["scalar_coupling_constant"] = baseline["scalar_coupling_constant"].fillna(
        overall_mean
    )

    baseline = baseline.merge(
        triple_means, on=["type", "element0", "element1"], how="left"
    )
    baseline["scalar_coupling_constant"] = baseline["triple_mean"].fillna(
        baseline["scalar_coupling_constant"]
    )
    baseline = baseline.merge(type_elem0_means, on=["type", "element0"], how="left")
    baseline["scalar_coupling_constant"] = baseline["elem0_mean"].fillna(
        baseline["scalar_coupling_constant"]
    )
    baseline = baseline.merge(type_elem1_means, on=["type", "element1"], how="left")
    baseline["scalar_coupling_constant"] = baseline["elem1_mean"].fillna(
        baseline["scalar_coupling_constant"]
    )
    submission = baseline[["id", "scalar_coupling_constant"]]




## === cell 3
out_path = "submission1236.csv"
submission.to_csv(out_path, index=False)
print(f"Submission written to {out_path}, shape: {submission.shape}")




## === cell 4
submission["scalar_coupling_constant"].plot(
    kind="hist", bins=100, title="Prediction Histogram"
)
