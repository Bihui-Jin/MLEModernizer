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

-1.36251

# 6. Current score

1.23548

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'I replace the failing stacking code with a simple but valid baseline: load the training data, compute the average scalar coupling constant for each coupling type (and a global mean for fallback), apply these averages to the test set, and write the predictions to a proper `submission.csv`. This fixes the file‑path errors, removes undefined variables, and guarantees a correctly formatted CSV so the notebook runs end‑to‑end.'
- What this solution (achieved 1.23548) has done: 'I add a lightweight bias‑correction step: split the training data into a small validation set, compute per‑type mean prediction errors on that set, and adjust the test‑set predictions by those mean residuals. This keeps the original mean‑by‑type baseline while nudging predictions toward the true values, which should lower the log‑MAE toward the target score. The rest of the pipeline and submission format remain unchanged.'
- What this solution (achieved 1.23548) has done: 'I enrich the simple type‑mean baseline by incorporating atom element information from structures.csv. By merging the atom types for each pair and computing means per (type, atom0, atom1) combination (with fallback to type‑mean then global mean), predictions become more specific, which should lower the log‑MAE and move the score toward the target. The overall pipeline and file handling remain unchanged.'
- What this solution (achieved 1.23548) has done: 'I keep the original mean‑by‑type baseline but add a finer‑grained bias correction: compute residual means per coupling pair (type + atom0 + atom1) on a validation split and add those to the test predictions together with the per‑type residuals already used. This small adjustment preserves the overall logic while giving the model more specific correction, which should lower the log‑MAE toward the target score.'
- What this solution (achieved 1.23548) has done: 'The fix updates the data loading path to correctly locate the competition files (using `/kaggle/input/champs-scalar-coupling` when available) and ensures the submission CSV is written to the working directory. No other logic is altered, preserving the existing mean‑by‑type baseline with residual corrections while making the notebook run end‑to‑end and output a valid `submission.csv`.'
- What this solution (achieved 1.23548) has done: 'I add two lightweight residual corrections based on the atom element of each partner (atom0 and atom1). After computing validation residuals I calculate the mean residual per atom type, fill missing values with 0, and add these adjustments to the test predictions together with the existing pair, type, and molecule residuals. This keeps the original mean‑by‑type baseline and only refines it with a small, interpretable bias term, which is expected to lower the log‑MAE toward the target score.'

# 9. Code solution

## === cell 0
import os
import pandas as pd

possible_base = "/kaggle/input/champs-scalar-coupling"
if os.path.isdir(possible_base):
    BASE_DIR = possible_base
else:
    BASE_DIR = os.path.abspath(os.path.join(".", "data", "champs-scalar-coupling"))
    if not os.path.isdir(BASE_DIR):
        BASE_DIR = os.getcwd()

train_path = os.path.join(BASE_DIR, "train.csv")
test_path = os.path.join(BASE_DIR, "test.csv")
structures_path = os.path.join(BASE_DIR, "structures.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
structures_df = pd.read_csv(structures_path)

print("train shape:", train_df.shape)
print("test shape:", test_df.shape)
print("structures shape:", structures_df.shape)




## === cell 1
def add_atom_elements(df):
    df = df.merge(
        structures_df[["molecule_name", "atom_index", "atom"]].rename(
            columns={"atom_index": "atom_index_0", "atom": "atom0"}
        ),
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    df = df.merge(
        structures_df[["molecule_name", "atom_index", "atom"]].rename(
            columns={"atom_index": "atom_index_1", "atom": "atom1"}
        ),
        on=["molecule_name", "atom_index_1"],
        how="left",
    )
    df["atom0"] = df["atom0"].fillna("NA")
    df["atom1"] = df["atom1"].fillna("NA")
    return df


train_df = add_atom_elements(train_df)
test_df = add_atom_elements(test_df)

train_df["pair_key"] = (
    train_df["type"]
    + "_"
    + train_df["atom0"].astype(str)
    + "_"
    + train_df["atom1"].astype(str)
)
test_df["pair_key"] = (
    test_df["type"]
    + "_"
    + test_df["atom0"].astype(str)
    + "_"
    + test_df["atom1"].astype(str)
)

pair_means_full = train_df.groupby("pair_key")["scalar_coupling_constant"].mean()
type_means_full = train_df.groupby("type")["scalar_coupling_constant"].mean()
global_mean_full = train_df["scalar_coupling_constant"].mean()

print("Number of pair keys (full):", pair_means_full.shape[0])
print("Number of coupling types:", type_means_full.shape[0])
print("Global mean (full data):", global_mean_full)



## === cell 2
val_frac = 0.20
val_df = train_df.sample(frac=val_frac, random_state=42)
train_sub = train_df.drop(val_df.index)

pair_means_sub = train_sub.groupby("pair_key")["scalar_coupling_constant"].mean()
type_means_sub = train_sub.groupby("type")["scalar_coupling_constant"].mean()
global_mean_sub = train_sub["scalar_coupling_constant"].mean()

val_pred = (
    val_df["pair_key"]
    .map(pair_means_sub)
    .fillna(val_df["type"].map(type_means_sub))
    .fillna(global_mean_sub)
)

val_residuals = val_df["scalar_coupling_constant"] - val_pred

type_residual_means = (
    val_residuals.groupby(val_df["type"])
    .mean()
    .reindex(type_means_full.index)
    .fillna(0.0)
)

pair_residual_means = (
    val_residuals.groupby(val_df["pair_key"])
    .mean()
    .reindex(pair_means_full.index)
    .fillna(0.0)
)

atom0_residual_means = (
    val_residuals.groupby(val_df["atom0"])
    .mean()
    .reindex(train_df["atom0"].unique())
    .fillna(0.0)
)

atom1_residual_means = (
    val_residuals.groupby(val_df["atom1"])
    .mean()
    .reindex(train_df["atom1"].unique())
    .fillna(0.0)
)

molecule_residual_means = val_residuals.groupby(val_df["molecule_name"]).mean()



## === cell 3
test_pred_base = (
    test_df["pair_key"]
    .map(pair_means_full)
    .fillna(test_df["type"].map(type_means_full))
    .fillna(global_mean_full)
)

test_pred_adjusted = (
    test_pred_base
    + test_df["pair_key"].map(pair_residual_means).fillna(0.0)
    + test_df["type"].map(type_residual_means).fillna(0.0)
    + test_df["molecule_name"].map(molecule_residual_means).fillna(0.0)
    + test_df["atom0"].map(atom0_residual_means).fillna(0.0)
    + test_df["atom1"].map(atom1_residual_means).fillna(0.0)
)

test_pred_adjusted = test_pred_adjusted.fillna(global_mean_full)

submission = test_df[["id"]].copy()
submission["scalar_coupling_constant"] = test_pred_adjusted.astype(float)

print("Submission preview:")
print(submission.head())



## === cell 4
output_path = "/kaggle/working/submission.csv"
submission.to_csv(output_path, index=False, float_format="%.6f")
print(f"Submission written to {output_path}, rows:", submission.shape[0])
