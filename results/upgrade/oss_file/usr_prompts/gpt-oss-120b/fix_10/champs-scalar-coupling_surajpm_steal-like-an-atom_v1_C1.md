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

-1.31958

# 6. Current score

3.32802

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'I replace the failing cells that try to read non‑existent external submissions with a simple baseline that computes the average scalar coupling constant for each coupling type from the training data and uses it to predict the test set. This removes the file‑not‑found errors, ensures a valid `submission.csv` is written with the correct columns, and keeps the core logic minimal and deterministic.'
- What this solution (achieved 3.00563) has done: 'I add a lightweight feature – the Euclidean distance between the two atoms – and fit a separate simple linear regression (intercept + slope) for each coupling type using NumPy’s least‑squares. The model’s predictions replace the original type‑mean baseline when a distance is available; otherwise the baseline is kept as a safe fallback. This keeps the core logic unchanged while providing a more informative predictor that should lower the log‑MAE toward the target. The script is re‑ordered into cells starting at 1 and still writes a valid `submission.csv`.'
- What this solution (achieved 2.16634) has done: 'I added handling for missing atomic coordinates that produced NaN distances, which caused LinearRegression to raise an error. After merging the structures I fill any NaN values in the dist column with the median distance from the training set, ensuring both the training and test feature matrices contain only finite numbers. This fix removes the runtime exception and lets the script generate a valid submission.csv while preserving the original modelling approach.'
- What this solution (achieved 2.97444) has done: 'I replace the single global LinearRegression with a lightweight per‑type linear model that fits an intercept and a distance slope for each coupling type. This keeps the overall pipeline and features the same (distance and atom identifiers) while giving the model more flexibility to capture type‑specific trends, which should lower the log‑MAE toward the target. The rest of the code (data loading, merging, NaN handling, and CSV output) remains unchanged.'
- What this solution (achieved 1.23566) has done: 'We replace the per‑type distance‑based linear regression with a simple per‑type mean baseline (falling back to the global mean when a type is unseen). This keeps the same data‑loading and distance handling but uses a more stable predictor, which should lower the log‑MAE and move the score closer to the target.'
- What this solution (achieved 2.85367) has done: 'I keep the existing data loading and merging, add lightweight atomic‑number features, fit a simple per‑type linear model on distance and atom numbers, and blend its predictions with the original per‑type mean baseline (using a small weight) so the model stays stable but gains a modest improvement, moving the log‑MAE lower toward the target.'
- What this solution (achieved 1.23566) has done: 'I set the blending weight `alpha` to 0 so the model no longer mixes the per‑type linear‑regression predictions with the baseline means. This removes the potentially noisy regression contribution while keeping the original data‑loading and feature‑engineering unchanged, which should lower the log‑MAE and move the score closer to the target.'
- What this solution (achieved 3.32802) has done: 'I keep the overall data‑loading, feature‑engineering and per‑type linear‑regression unchanged, but change the blending weight `alpha` from 0 to a positive value (e.g., 0.6). This lets the model use the learned distance/atom‑number regression together with the type‑mean baseline, which is expected to lower the log‑MAE and bring the score closer to the target while preserving the original pipeline.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

print("Input directory contents:", os.listdir("../input"))

train_path = "../input/champs-scalar-coupling/train.csv"
test_path = "../input/champs-scalar-coupling/test.csv"
structures_path = "../input/champs-scalar-coupling/structures.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
structures = pd.read_csv(structures_path)

print("Train shape:", train.shape)
print("Test shape:", test.shape)



## === cell 1
struct0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
    }
)
struct1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
    }
)

train_merged = train.merge(
    struct0, on=["molecule_name", "atom_index_0"], how="left"
).merge(struct1, on=["molecule_name", "atom_index_1"], how="left")
test_merged = test.merge(
    struct0, on=["molecule_name", "atom_index_0"], how="left"
).merge(struct1, on=["molecule_name", "atom_index_1"], how="left")

for df in [train_merged, test_merged]:
    df["dist"] = np.sqrt(
        (df["x0"] - df["x1"]) ** 2
        + (df["y0"] - df["y1"]) ** 2
        + (df["z0"] - df["z1"]) ** 2
    )

median_dist = train_merged["dist"].median()
train_merged["dist"].fillna(median_dist, inplace=True)
test_merged["dist"].fillna(median_dist, inplace=True)

atomic_number = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Fe": 26,
    "Cu": 29,
    "Zn": 30,
    "Br": 35,
    "I": 53,
}
for col in ["atom_0", "atom_1"]:
    train_merged[col] = train_merged[col].map(atomic_number)
    test_merged[col] = test_merged[col].map(atomic_number)

median_atom = train_merged[["atom_0", "atom_1"]].median().median()
train_merged[["atom_0", "atom_1"]] = train_merged[["atom_0", "atom_1"]].fillna(
    median_atom
)
test_merged[["atom_0", "atom_1"]] = test_merged[["atom_0", "atom_1"]].fillna(
    median_atom
)

global_mean = train["scalar_coupling_constant"].mean()
type_means = train.groupby("type")["scalar_coupling_constant"].mean().to_dict()

lr_params = {}
for typ, grp in train_merged.groupby("type"):
    X = np.vstack(
        [
            np.ones(len(grp)),
            grp["dist"].values,
            grp["atom_0"].values,
            grp["atom_1"].values,
        ]
    ).T
    y = grp["scalar_coupling_constant"].values
    coeff, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    lr_params[typ] = coeff  # array of 4 coefficients


def predict_lr(row):
    coeff = lr_params.get(row["type"])
    if coeff is None:
        return np.nan
    return (
        coeff[0]
        + coeff[1] * row["dist"]
        + coeff[2] * row["atom_0"]
        + coeff[3] * row["atom_1"]
    )


test_lr_pred = test_merged.apply(predict_lr, axis=1).values

baseline_pred = np.array([type_means.get(t, global_mean) for t in test_merged["type"]])

alpha = 0.6
final_pred = np.where(
    np.isnan(test_lr_pred),
    baseline_pred,
    alpha * test_lr_pred + (1 - alpha) * baseline_pred,
)

final_pred = np.where(np.isnan(final_pred), global_mean, final_pred)

submission = pd.DataFrame({"id": test["id"], "scalar_coupling_constant": final_pred})
submission_file = "submission.csv"
submission.to_csv(submission_file, index=False)
print(f"Submission written to {submission_file}")
print(submission.head())
