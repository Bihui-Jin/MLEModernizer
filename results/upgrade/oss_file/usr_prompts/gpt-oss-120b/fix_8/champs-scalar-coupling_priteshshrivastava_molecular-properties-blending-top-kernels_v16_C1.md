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

-1.680697846605887

# 6. Current score

1.49316

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.2967) has done: 'I replace the missing blend‑file reads with a tiny end‑to‑end pipeline: load the real training and test sets, encode the few categorical columns, train a fast RandomForestRegressor on a sampled subset (to keep runtime low), predict the coupling constants for the test rows and write a correctly‑named *.csv* submission. This eliminates the FileNotFoundError and guarantees a valid submission file while preserving the overall workflow structure.'
- What this solution (achieved 1.49316) has done: 'I fix the training target handling by removing any rows that produce non‑finite log‑values (e.g., caused by negative coupling constants) before sampling and fitting the model. This prevents the “infinity or a value too large” error and allows the RandomForestRegressor to train correctly, so the script can subsequently generate a valid submission.csv.'

# 9. Code solution

## === cell 0
from pathlib import Path
import pandas as pd
import numpy as np

BASE_PATH = Path("/kaggle/input/champs-scalar-coupling")
if not BASE_PATH.exists():
    BASE_PATH = Path("../input/champs-scalar-coupling")

train_path = BASE_PATH / "train.csv"
test_path = BASE_PATH / "test.csv"
structures_path = BASE_PATH / "structures.csv"
potential_path = BASE_PATH / "potential_energy.csv"
dipole_path = BASE_PATH / "dipole_moments.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

structures = pd.read_csv(structures_path)

struct0 = structures.rename(columns={"atom_index": "atom_index_0", "atom": "atom_0"})
struct1 = structures.rename(columns={"atom_index": "atom_index_1", "atom": "atom_1"})

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

potential = pd.read_csv(potential_path)  # columns: molecule_name, potential_energy
dipole = pd.read_csv(dipole_path)  # columns: molecule_name, X, Y, Z

train = train.merge(potential, on="molecule_name", how="left")
test = test.merge(potential, on="molecule_name", how="left")

train = train.merge(dipole, on="molecule_name", how="left")
test = test.merge(dipole, on="molecule_name", how="left")

train = train.dropna(subset=["scalar_coupling_constant"]).reset_index(drop=True)




## === cell 1
train["type_enc"], type_uniques = pd.factorize(train["type"])
test["type_enc"] = type_uniques.get_indexer(test["type"])

train["mol_enc"], mol_uniques = pd.factorize(train["molecule_name"])
test["mol_enc"] = mol_uniques.get_indexer(test["molecule_name"])

train["atom0_enc"], atom0_uniques = pd.factorize(train["atom_0"])
test["atom0_enc"] = atom0_uniques.get_indexer(test["atom_0"])

train["atom1_enc"], atom1_uniques = pd.factorize(train["atom_1"])
test["atom1_enc"] = atom1_uniques.get_indexer(test["atom_1"])

feature_cols = [
    "atom_index_0",
    "atom_index_1",
    "type_enc",
    "mol_enc",
    "atom0_enc",
    "atom1_enc",
    "potential_energy",
    "X",
    "Y",
    "Z",
]

X = train[feature_cols].fillna(-1)

y_log = np.log1p(train["scalar_coupling_constant"])
finite_mask = np.isfinite(y_log)  # removes -inf / inf / NaN
X = X[finite_mask]
y_log = y_log[finite_mask]

rng = np.random.RandomState(42)
sample_idx = rng.choice(len(X), size=min(300_000, len(X)), replace=False)

X_train = X.iloc[sample_idx]
y_train = y_log.iloc[sample_idx]




## === cell 2
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    n_jobs=5,
    random_state=42,
)

model.fit(X_train, y_train)




## === cell 3
test_features = test[feature_cols].fillna(-1)
pred_log = model.predict(test_features)
test_preds = np.expm1(pred_log)  # invert log1p transformation

submission = pd.DataFrame({"id": test["id"], "scalar_coupling_constant": test_preds})

output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
