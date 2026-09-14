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

-1.3628733700694355

# 6. Current score

1.23566

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 3.00563) has done: 'I replaced the failing imports of external submissions with a self‑contained baseline: read the competition’s `train.csv` and `test.csv`, compute the overall mean of `scalar_coupling_constant` from the training data, and use that constant as the prediction for every test row. The script now creates a valid `submission.csv` with the required columns and runs end‑to‑end without missing‑file errors.'
- What this solution (achieved 1.23566) has done: 'I replace the single global‑mean prediction with a per‑coupling‑type mean. Using the average target value for each coupling “type” gives predictions that are closer to the true distribution, which should lower the MAE and thus move the log‑MAE score toward the target (lower is better). The rest of the pipeline and file handling stay unchanged.'
- What this solution (achieved 1.23566) has done: 'I add a lightweight feature‑engineering step that uses the atom element types of each pair. By merging the `structures.csv` file we can compute a mean target per (`type`, `atom_0`, `atom_1`) combination and fall back to the per‑type mean, then the global mean. This keeps the original simple‑mean logic while giving more specific predictions, which should lower the log‑MAE and move the score toward the target without altering the core modeling approach. The script is reorganized into sequential cells and still writes a valid `submission.csv`.'
- What this solution (achieved 1.23566) has done: 'I add a few more fallback mean tables that use progressively coarser groupings (reversed atom order, single‑atom type, then the original per‑type mean). The prediction function try them in order, so more specific statistics are used when available, which should reduce the MAE and move the log‑MAE closer to the target without changing the overall modelling approach.'

# 9. Code solution

## === cell 0
import os
import pandas as pd

BASE_PATH = os.path.abspath(os.path.join("..", "input", "champs-scalar-coupling"))
print("Data directory contents:", os.listdir(BASE_PATH))



## === cell 1
train_path = os.path.join(BASE_PATH, "train.csv")
test_path = os.path.join(BASE_PATH, "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)



## === cell 2
structures_path = os.path.join(BASE_PATH, "structures.csv")
structures = pd.read_csv(structures_path)[["molecule_name", "atom_index", "atom"]]

atom0 = structures.rename(columns={"atom_index": "atom_index_0", "atom": "atom_0"})
atom1 = structures.rename(columns={"atom_index": "atom_index_1", "atom": "atom_1"})

train_merged = train_df.merge(atom0, on=["molecule_name", "atom_index_0"], how="left")
train_merged = train_merged.merge(
    atom1, on=["molecule_name", "atom_index_1"], how="left"
)

test_merged = test_df.merge(atom0, on=["molecule_name", "atom_index_0"], how="left")
test_merged = test_merged.merge(atom1, on=["molecule_name", "atom_index_1"], how="left")



## === cell 3
global_mean = train_df["scalar_coupling_constant"].mean()
type_means = train_df.groupby("type")["scalar_coupling_constant"].mean().to_dict()

combo_means = (
    train_merged.groupby(["type", "atom_0", "atom_1"])["scalar_coupling_constant"]
    .mean()
    .to_dict()
)

combo_means_rev = (
    train_merged.groupby(["type", "atom_1", "atom_0"])["scalar_coupling_constant"]
    .mean()
    .to_dict()
)

combo_type_atom0 = (
    train_merged.groupby(["type", "atom_0"])["scalar_coupling_constant"]
    .mean()
    .to_dict()
)
combo_type_atom1 = (
    train_merged.groupby(["type", "atom_1"])["scalar_coupling_constant"]
    .mean()
    .to_dict()
)

print(f"Global mean: {global_mean:.6f}")
print(f"Number of coupling types: {len(type_means)}")
print(f"Full combos (type,atom0,atom1): {len(combo_means)}")
print(f"Reversed combos (type,atom1,atom0): {len(combo_means_rev)}")
print(f"Partial combos (type,atom0): {len(combo_type_atom0)}")
print(f"Partial combos (type,atom1): {len(combo_type_atom1)}")




## === cell 4
def predict_row(row):
    key_full = (row["type"], row["atom_0"], row["atom_1"])
    if key_full in combo_means:
        return combo_means[key_full]

    key_rev = (row["type"], row["atom_1"], row["atom_0"])
    if key_rev in combo_means_rev:
        return combo_means_rev[key_rev]

    key_a0 = (row["type"], row["atom_0"])
    if key_a0 in combo_type_atom0:
        return combo_type_atom0[key_a0]

    key_a1 = (row["type"], row["atom_1"])
    if key_a1 in combo_type_atom1:
        return combo_type_atom1[key_a1]

    if row["type"] in type_means:
        return type_means[row["type"]]

    return global_mean


predictions = test_merged.apply(predict_row, axis=1)

submission = pd.DataFrame(
    {"id": test_df["id"], "scalar_coupling_constant": predictions}
)



## === cell 5
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print(submission.head())



## === cell 6
try:
    import matplotlib.pyplot as plt

    submission["scalar_coupling_constant"].hist(bins=100)
    plt.title("Prediction Distribution")
    plt.xlabel("scalar_coupling_constant")
    plt.ylabel("Count")
    plt.show()
except Exception as e:
    print("Matplotlib not available or plotting failed:", e)
