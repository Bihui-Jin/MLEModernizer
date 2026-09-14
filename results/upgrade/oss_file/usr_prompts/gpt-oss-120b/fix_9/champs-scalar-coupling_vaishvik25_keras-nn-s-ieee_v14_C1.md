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

-1.4419498537699864

# 6. Current score

1.99777

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'I replace the failing stacking logic with a simple, robust pipeline that reads the training data, computes a per‑coupling‑type mean target, applies it to the test set (falling back to the global mean for any missing types), and writes a correctly formatted `submission.csv`. This eliminates the missing‑folder errors, ensures a valid CSV is produced, and modestly improves the baseline score without altering core modeling concepts.'
- What this solution (achieved 1.23438) has done: 'I add a simple distance feature derived from the atomic coordinates and fit a lightweight linear regression on the residuals of the per‑type mean. This keeps the original mean‑by‑type logic while giving a calibrated correction that should lower the log‑MAE toward the target. The new steps merge the structures data, compute Euclidean distances, train the regression, and apply the adjusted predictions to the test set, finally writing a proper `submission.csv`.'
- What this solution (achieved 1.20928) has done: 'I add richer linear features – distance squared and one‑hot encoded atom types – to the residual regression while keeping the per‑type mean baseline. This should give the model more expressive power and move the log‑MAE closer to the target (lower is better). The rest of the pipeline and file output remain unchanged.'
- What this solution (achieved 1.99777) has done: 'I add the scalar‑coupling contribution features (fc, sd, pso, dso) to both the training and test data, compute their sum, and use this exact sum as the prediction whenever it is available. This leverages information that directly composes the target, so the MAE – and thus the log‑MAE – should drop toward the negative target value while keeping the existing linear‑regression pipeline unchanged for any rows lacking contributions.'
- What this solution (achieved 1.99777) has done: 'I remove the unnecessary residual regression step from the test‑time prediction.  
Instead of adding a learned residual (which can introduce noise), the fallback use only the per‑type mean (or the global mean when a type is unseen). This keeps the core pipeline unchanged, guarantees a valid CSV, and should lower the log‑MAE toward the negative target score.'
- What this solution (achieved 1.99777) has done: 'I activate the learned residual correction that was previously computed but never applied. In the test‑time block I add a prediction step (`lr.predict`) to fill `pred_residual`, then keep the existing fallback logic so rows with available contribution sums use them directly while others get the per‑type mean plus the predicted residual. This minor change should lower the log‑MAE toward the target without altering the overall pipeline.'
- What this solution (achieved 1.99777) has done: 'I remove the residual correction that is added to the type‑mean fallback, because the linear‑regression residual predictions are noisy and increase the error. The model now use the exact contribution sum when available, otherwise fall back to the per‑type mean (or the global mean), which should lower the log‑MAE toward the target. The residual‑regression model is still trained (preserving the core pipeline) but its predictions are no longer applied.'
- What this solution (achieved 1.99777) has done: 'I add the residual correction to the fallback predictions: when a contribution sum isn’t available we now add the learned residual to the per‑type mean (or global mean) instead of using the mean alone. This small change uses the existing linear‑regression model and should lower the log‑MAE, moving the score toward the target while keeping the overall pipeline unchanged.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

print("Input folder contents:", os.listdir("../input"))




## === cell 1
train_path = "../input/champs-scalar-coupling/train.csv"
test_path = "../input/champs-scalar-coupling/test.csv"
structures_path = "../input/champs-scalar-coupling/structures.csv"
contrib_path = "../input/champs-scalar-coupling/scalar_coupling_contributions.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

global_mean = train_df["scalar_coupling_constant"].mean()
type_mean = train_df.groupby("type")["scalar_coupling_constant"].mean()




## === cell 2
structures = pd.read_csv(structures_path)

struct0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
    }
)[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]]

struct1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
    }
)[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]]

train_merged = train_df.merge(
    struct0, on=["molecule_name", "atom_index_0"], how="left"
).merge(struct1, on=["molecule_name", "atom_index_1"], how="left")

contrib = pd.read_csv(contrib_path)
train_merged = train_merged.merge(
    contrib,
    on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
    how="left",
)

train_merged["contrib_sum"] = train_merged[["fc", "sd", "pso", "dso"]].sum(axis=1)

train_merged["distance"] = np.sqrt(
    (train_merged["x0"] - train_merged["x1"]) ** 2
    + (train_merged["y0"] - train_merged["y1"]) ** 2
    + (train_merged["z0"] - train_merged["z1"]) ** 2
)
train_merged["distance_sq"] = train_merged["distance"] ** 2

median_dist = train_merged["distance"].median()
train_merged["distance"].fillna(median_dist, inplace=True)
train_merged["distance_sq"].fillna(median_dist**2, inplace=True)




## === cell 3
train_merged["type_mean"] = train_merged["type"].map(type_mean)

train_merged["residual"] = train_merged["contrib_sum"] - train_merged["type_mean"]

train_merged["atom_0"] = train_merged["atom_0"].fillna("X")
train_merged["atom_1"] = train_merged["atom_1"].fillna("X")
atom0_dummies = pd.get_dummies(train_merged["atom_0"], prefix="atom0")
atom1_dummies = pd.get_dummies(train_merged["atom_1"], prefix="atom1")

X_train = pd.concat(
    [train_merged[["distance", "distance_sq"]], atom0_dummies, atom1_dummies],
    axis=1,
)

y_train = train_merged["residual"]

lr = LinearRegression()
lr.fit(X_train, y_train)




## === cell 4
test_merged = test_df.merge(
    struct0, on=["molecule_name", "atom_index_0"], how="left"
).merge(struct1, on=["molecule_name", "atom_index_1"], how="left")

test_merged = test_merged.merge(
    contrib,
    on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
    how="left",
)

test_merged["contrib_sum"] = test_merged[["fc", "sd", "pso", "dso"]].sum(axis=1)

test_merged["distance"] = np.sqrt(
    (test_merged["x0"] - test_merged["x1"]) ** 2
    + (test_merged["y0"] - test_merged["y1"]) ** 2
    + (test_merged["z0"] - test_merged["z1"]) ** 2
)
test_merged["distance_sq"] = test_merged["distance"] ** 2

test_merged["distance"].fillna(median_dist, inplace=True)
test_merged["distance_sq"].fillna(median_dist**2, inplace=True)

test_merged["type_mean"] = test_merged["type"].map(type_mean)

test_merged["atom_0"] = test_merged["atom_0"].fillna("X")
test_merged["atom_1"] = test_merged["atom_1"].fillna("X")
atom0_dummies_test = pd.get_dummies(test_merged["atom_0"], prefix="atom0")
atom1_dummies_test = pd.get_dummies(test_merged["atom_1"], prefix="atom1")

X_test = pd.concat(
    [test_merged[["distance", "distance_sq"]], atom0_dummies_test, atom1_dummies_test],
    axis=1,
)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

test_merged["pred_residual"] = lr.predict(X_test)

test_merged["scalar_coupling_constant"] = test_merged["contrib_sum"]

missing_mask = test_merged["scalar_coupling_constant"].isna()
test_merged.loc[missing_mask, "scalar_coupling_constant"] = (
    test_merged.loc[missing_mask, "type_mean"].fillna(global_mean)
    + test_merged.loc[missing_mask, "pred_residual"]
)

test_merged["scalar_coupling_constant"].fillna(global_mean, inplace=True)




## === cell 5
submission = test_merged[["id", "scalar_coupling_constant"]].copy()
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False, float_format="%.6f")
print(f"Submission written to {submission_path}")
print(submission.head())
