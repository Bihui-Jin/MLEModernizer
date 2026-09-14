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
scipy==1.15.3
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

-1.965121149403092

# 6. Current score

1.65935

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.99065) has done: 'The script failed because it tried to read non‑existent model files from `../input/models`, causing a cascade of NameErrors. I removed that logic and replaced it with a simple, robust baseline: compute the median `scalar_coupling_constant` from the training set and use it as the prediction for every test row. This guarantees a valid `stack_median.csv` submission with the correct columns and avoids any missing‑file errors.'
- What this solution (achieved 1.18497) has done: 'I replace the single global‑median prediction with a type‑wise median: each coupling type gets its own median value from the training data, which better captures systematic differences between types and should lower the evaluation score toward the target. Unseen types fall back to the overall median, keeping the code safe.'
- What this solution (achieved 3.16836) has done: 'I add a few inexpensive molecular‑level features (potential energy and dipole magnitude) and a simple Ridge regression model that uses the coupling type together with these features, falling back to the previous type‑wise median when needed. This keeps the overall workflow unchanged while providing a modest predictive boost that should lower the log‑MAE toward the target.'
- What this solution (achieved 1.20802) has done: 'I replace the integer encoding of the coupling type with one‑hot columns and train the Ridge model on these richer features (type dummies + potential energy + dipole magnitude). Then I blend the model’s predictions with the type‑wise median (70 % model + 30 % median) to obtain a more accurate final prediction while keeping the overall workflow unchanged.'
- What this solution (achieved 1.22601) has done: 'I replace the single global Ridge model with lightweight per‑type Ridge models (using the same numeric features) and increase the model’s contribution in the final blend. This keeps the overall workflow and features unchanged while giving each coupling type its own fitted coefficients, which should reduce the MAE and move the log‑MAE score closer to the target.'
- What this solution (achieved 1.20801) has done: 'I replace the per‑type Ridge fitting with a single Ridge model that uses both the numeric features and one‑hot encoded coupling types. This keeps the overall linear‑model approach intact while giving the model more expressive power, and I blend its predictions with the type‑wise median (giving the median a modest weight so we don’t over‑fit).'
- What this solution (achieved 1.90478) has done: 'I add the scalar‑coupling contribution features (fc, sd, pso, dso and their sum) to the numeric feature set and give the ridge model a larger share in the final blend (85 % model + 15 % type‑wise median). This keeps the overall linear‑model pipeline while providing more predictive information, which should lower the log‑MAE toward the negative target score.'
- What this solution (achieved 1.86067) has done: 'Implemented a robust data‑directory fallback, ensured the target column has no NaNs before logging, and added safety handling for any infinite values that could arise during the log transformation. These fixes prevent the `ValueError` on model fitting and guarantee that `final_preds` is defined for the submission write‑out, resulting in a valid CSV file.'
- What this solution (achieved 1.65935) has done: 'I lower the contribution of the Ridge model and increase the weight of the robust type‑wise median fallback, which historically reduced the log‑MAE and moves the score closer to the negative target. The only change is the blending factor in cell 3, keeping all other logic intact.'
- What this solution (achieved 1.48508) has done: 'I lower the contribution of the Ridge model in the blending step, giving the robust type‑wise median a larger share (by reducing `model_weight` from 0.40 to 0.10). This keeps the core pipeline unchanged while moving the log‑MAE closer to the negative target (lower is better).'
- What this solution (achieved 1.65935) has done: 'I increase the ridge‑model contribution in the blending step from 0.10 back to 0.40, which historically lowers the log‑MAE. This small adjustment keeps the core pipeline unchanged while giving the better‑performing model more influence, moving the score closer to the negative target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

default_path = os.path.join("..", "input", "champs-scalar-coupling")
alternate_path = os.path.join(".", "input", "champs-scalar-coupling")
DATA_DIR = default_path if os.path.isdir(default_path) else alternate_path




## === cell 1
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

assert (
    "scalar_coupling_constant" in train_df.columns
), "Train file missing target column"
assert "id" in test_df.columns, "Test file missing id column"




## === cell 2
pot_path = os.path.join(DATA_DIR, "potential_energy.csv")
potential_df = pd.read_csv(pot_path)
train_df = train_df.merge(potential_df, on="molecule_name", how="left")
test_df = test_df.merge(potential_df, on="molecule_name", how="left")

dipole_path = os.path.join(DATA_DIR, "dipole_moments.csv")
dipole_df = pd.read_csv(dipole_path)
dipole_df["dipole_mag"] = np.sqrt(dipole_df[["X", "Y", "Z"]].pow(2).sum(axis=1))
dipole_df = dipole_df[["molecule_name", "dipole_mag"]]
train_df = train_df.merge(dipole_df, on="molecule_name", how="left")
test_df = test_df.merge(dipole_df, on="molecule_name", how="left")

contrib_path = os.path.join(DATA_DIR, "scalar_coupling_contributions.csv")
contrib_df = pd.read_csv(contrib_path)
contrib_df = contrib_df[
    ["molecule_name", "atom_index_0", "atom_index_1", "type", "fc", "sd", "pso", "dso"]
]
contrib_df["total_contrib"] = (
    contrib_df["fc"] + contrib_df["sd"] + contrib_df["pso"] + contrib_df["dso"]
)

merge_keys = ["molecule_name", "atom_index_0", "atom_index_1", "type"]
train_df = train_df.merge(contrib_df, on=merge_keys, how="left")
test_df = test_df.merge(contrib_df, on=merge_keys, how="left")

numeric_features = [
    "potential_energy",
    "dipole_mag",
    "fc",
    "sd",
    "pso",
    "dso",
    "total_contrib",
]

train_num = train_df[numeric_features].fillna(train_df[numeric_features].median())
test_num = test_df[numeric_features].fillna(train_df[numeric_features].median())

train_df["scalar_coupling_constant"] = (
    train_df["scalar_coupling_constant"]
    .fillna(train_df["scalar_coupling_constant"].median())
    .replace(0, 1e-9)
)

y_log = np.log(train_df["scalar_coupling_constant"])
y_log = y_log.replace([np.inf, -np.inf], np.nan)
y_log = y_log.fillna(y_log.median())




## === cell 3
type_dummies_train = pd.get_dummies(train_df["type"], prefix="type")
type_dummies_test = pd.get_dummies(test_df["type"], prefix="type")

X_train = pd.concat([train_num, type_dummies_train], axis=1)
X_test = pd.concat([test_num, type_dummies_test], axis=1).reindex(
    columns=X_train.columns, fill_value=0
)

alpha_value = 0.5
ridge_model = Ridge(alpha=alpha_value, random_state=42)
ridge_model.fit(X_train, y_log)

preds_log_model = ridge_model.predict(X_test)
preds_model = np.exp(preds_log_model)  # back to original scale

global_median_log = np.log(train_df["scalar_coupling_constant"].median())
type_median_log = np.log(train_df.groupby("type")["scalar_coupling_constant"].median())
fallback_log = test_df["type"].map(type_median_log).fillna(global_median_log)
fallback = np.exp(fallback_log)

model_weight = 0.40  # restored from earlier version
final_preds = model_weight * preds_model + (1 - model_weight) * fallback
final_preds = np.clip(final_preds, 0, None)




## === cell 4
submission = pd.DataFrame(
    {
        "id": test_df["id"],
        "scalar_coupling_constant": final_preds,
    }
)

output_path = "stack_median.csv"
submission.to_csv(output_path, index=False, float_format="%.6f")
print(f"Submission saved to {output_path}")
