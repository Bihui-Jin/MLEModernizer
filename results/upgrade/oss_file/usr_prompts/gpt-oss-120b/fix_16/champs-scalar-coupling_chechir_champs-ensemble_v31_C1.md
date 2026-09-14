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

-2.426993439250257

# 6. Current score

1.99777

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'I correct the data path so the CSV files are found, keep the simple type‑mean baseline, and ensure the script writes a proper `.csv` submission file in the working directory.'
- What this solution (achieved 1.23566) has done: 'I extend the simple type‑mean baseline by adding atomic element information from ``structures.csv`` and compute means for each combination of coupling type and the two atom elements. This adds only a small, inexpensive feature and keeps the overall approach unchanged, while expected to lower the log‑MAE toward the target negative score.'
- What this solution (achieved 1.43112) has done: 'I add loading of the scalar‑coupling contribution data and merge it with the train and test frames, then use the summed contributions as an additional inexpensive signal. The final prediction be a simple weighted blend of the original type/atom‑combo mean and the contribution‑based estimate, which should lower the MAE (and thus the log‑MAE) toward the target without altering the overall baseline architecture.'
- What this solution (achieved 1.99777) has done: 'I replace the weighted blend with a direct use of the summed coupling contributions wherever they are available, falling back to the type‑atom‑combo means only when the contribution data is missing. This leverages the fact that the target constant equals the sum of the four contributions, so it should substantially lower the log‑MAE toward the negative target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 1.99777) has done: 'I add a lightweight residual correction: after building the baseline predictions (type‑atom‑combo means with fall‑backs), I compute the average error per coupling type on the training data and add this correction to the test predictions for rows where the contribution sum isn’t available. This small calibration shifts the predictions toward the true values, decreasing the log‑MAE and moving the score closer to the negative target without changing the overall model logic.'
- What this solution (achieved 1.99777) has done: 'I add a lightweight per‑type‑and‑atom‑combo residual correction.  
First the residuals are computed on the training set, then a mean residual for each `(type, atom0, atom1)` combo is derived.  
During prediction the combo mean is shifted by this combo‑residual; if the combo is unknown we fall back to the type mean plus its type‑residual, and finally to the global mean.  
Rows where the exact contribution sum is available still use that exact value. This small calibration should lower the log‑MAE toward the negative target without altering the overall baseline logic.'
- What this solution (achieved 1.99777) has done: 'I add a cheap calibration that uses the average contribution sum for each coupling type when the exact contributions are unavailable in the test set, and then apply an overall bias correction based on the mean residual observed on the training data. These small adjustments keep the original baseline logic untouched while giving the model a better fallback estimate, which should move the log‑MAE lower toward the negative target.'
- What this solution (achieved 1.99777) has done: 'I replace the complex combo‑based fallback with a simpler strategy that uses the exact contribution sum when it exists and otherwise falls back to the average contribution sum for the coupling type. This reduces noise from sparse combo means and keeps the core baseline logic while adding only an overall bias correction derived from training residuals, which should lower the MAE and move the log‑MAE toward the negative target.'
- What this solution (achieved 1.99777) has done: 'I replace the simple type‑average fallback with a richer hierarchy: use the exact contribution sum when present, otherwise fall back to the mean scalar coupling for the specific `(type, atom0, atom1)` combo, then to the per‑type mean, and finally to the global mean. The same hierarchy is applied to the training data so the residual bias is computed on comparable predictions. This small calibration keeps the baseline‑plus‑bias structure while providing more accurate estimates, moving the log‑MAE toward the negative target.'
- What this solution (achieved 1.99777) has done: 'I add a lightweight calibration that first applies a per‑type‑atom‑combo residual correction, then falls back to a per‑type residual, and finally to the overall bias. This keeps the original baseline logic intact while providing finer adjustments that should lower the log‑MAE toward the negative target.'
- What this solution (achieved 1.99777) has done: 'I add a more reliable combo‑residual lookup (row‑wise `.apply` instead of the index‑map that can miss many values) and then apply a lightweight per‑type scaling factor derived from the training data to the final predictions. This keeps the original hierarchy and residual corrections while slightly better calibrating predictions, which should reduce the log‑MAE and move the score toward the negative target.'
- What this solution (achieved 1.99777) has done: 'I add fallback predictions based on the average contribution sum for each `(type, atom0, atom1)` combo and for each type, then use the overall average contribution when needed. This provides a more accurate estimate than the previous scalar‑mean fall‑backs and should lower the log‑MAE toward the negative target. I also remove the final per‑type scaling factor, which can unintentionally inflate errors, keeping the core baseline‑plus‑residual logic intact.'
- What this solution (achieved 1.99777) has done: 'I add a lightweight per‑type multiplicative scaling factor derived from the training data and apply it to the baseline predictions before the residual corrections. This small calibration keeps the original hierarchy untouched while nudging predictions closer to the true values, which should lower the log‑MAE and move the score toward the negative target.'
- What this solution (achieved 1.99777) has done: 'I simplify the prediction step to avoid the possibly destabilising scaling factor and excessive residual blending.  
The baseline predictions already use contribution‑based fall‑backs; adding a modest per‑type residual (or overall bias) is enough to improve calibration while keeping the core hierarchy unchanged. This should lower the MAE and move the log‑MAE closer to the negative target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

DATA_ROOT = os.path.join("/kaggle", "input", "champs-scalar-coupling")



## === cell 1
train_path = os.path.join(DATA_ROOT, "train.csv")
test_path = os.path.join(DATA_ROOT, "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)



## === cell 2
structures_path = os.path.join(DATA_ROOT, "structures.csv")
structures = pd.read_csv(structures_path)[["molecule_name", "atom_index", "atom"]]

struct0 = structures.rename(
    columns={"atom_index": "atom_index_0", "atom": "atom0_elem"}
)
struct1 = structures.rename(
    columns={"atom_index": "atom_index_1", "atom": "atom1_elem"}
)

train = train.merge(struct0, on=["molecule_name", "atom_index_0"], how="left")
train = train.merge(struct1, on=["molecule_name", "atom_index_1"], how="left")
test = test.merge(struct0, on=["molecule_name", "atom_index_0"], how="left")
test = test.merge(struct1, on=["molecule_name", "atom_index_1"], how="left")

contrib_path = os.path.join(DATA_ROOT, "scalar_coupling_contributions.csv")
contrib = pd.read_csv(contrib_path)[
    ["molecule_name", "atom_index_0", "atom_index_1", "type", "fc", "sd", "pso", "dso"]
]

train = train.merge(
    contrib,
    on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
    how="left",
    suffixes=("", "_c"),
)
test = test.merge(
    contrib,
    on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
    how="left",
    suffixes=("", "_c"),
)

train["contrib_sum"] = train[["fc", "sd", "pso", "dso"]].sum(axis=1)
test["contrib_sum"] = test[["fc", "sd", "pso", "dso"]].sum(axis=1)

combo_mean = train.groupby(["type", "atom0_elem", "atom1_elem"])[
    "scalar_coupling_constant"
].mean()
type_mean = train.groupby("type")["scalar_coupling_constant"].mean()
global_mean = train["scalar_coupling_constant"].mean()

combo_contrib_mean = train.groupby(["type", "atom0_elem", "atom1_elem"])[
    "contrib_sum"
].mean()
type_contrib_mean = train.groupby("type")["contrib_sum"].mean()
global_contrib_mean = train["contrib_sum"].mean()


def fill_missing(df, source_col, target_col):
    """Fill target_col using source_col first, then increasingly coarse contribution‑based fall‑backs,
    finally falling back to the original scalar‑based statistics."""
    df[target_col] = df[source_col]

    mask = df[target_col].isna()
    if mask.any():
        df.loc[mask, target_col] = df.loc[mask].apply(
            lambda r: combo_contrib_mean.get(
                (r["type"], r["atom0_elem"], r["atom1_elem"]), np.nan
            ),
            axis=1,
        )

    mask = df[target_col].isna()
    if mask.any():
        df.loc[mask, target_col] = df.loc[mask, "type"].map(type_contrib_mean)

    mask = df[target_col].isna()
    if mask.any():
        df.loc[mask, target_col] = global_contrib_mean

    mask = df[target_col].isna()
    if mask.any():
        df.loc[mask, target_col] = df.loc[mask].apply(
            lambda r: combo_mean.get(
                (r["type"], r["atom0_elem"], r["atom1_elem"]), np.nan
            ),
            axis=1,
        )

    mask = df[target_col].isna()
    if mask.any():
        df.loc[mask, target_col] = df.loc[mask, "type"].map(type_mean)

    mask = df[target_col].isna()
    if mask.any():
        df.loc[mask, target_col] = global_mean


fill_missing(train, "contrib_sum", "baseline_pred")
fill_missing(test, "contrib_sum", "baseline_pred")

train["residual"] = train["scalar_coupling_constant"] - train["baseline_pred"]
overall_bias = train["residual"].mean()

type_residual_map = train.groupby("type")["residual"].mean()
test["type_residual"] = test["type"].map(type_residual_map)

combo_residual_map = train.groupby(["type", "atom0_elem", "atom1_elem"])[
    "residual"
].mean()
test["combo_residual"] = test.apply(
    lambda r: combo_residual_map.get(
        (r["type"], r["atom0_elem"], r["atom1_elem"]), np.nan
    ),
    axis=1,
)

test["final_preds"] = test["baseline_pred"] + test["type_residual"].fillna(overall_bias)



## === cell 3
submission = pd.DataFrame(
    {"id": test["id"], "scalar_coupling_constant": test["final_preds"]}
)



## === cell 4
submission_path = os.path.join("/kaggle", "working", "ensemble_sub.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, shape: {submission.shape}")
