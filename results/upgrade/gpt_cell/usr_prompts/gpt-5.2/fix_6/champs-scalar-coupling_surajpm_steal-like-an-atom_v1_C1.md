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

1.18497

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.99777) has done: 'The crash happens because cell 1 tries to read three submission CSVs from other Kaggle notebook datasets under `../input/...`, but those folders don’t exist in this environment (only the `champs-scalar-coupling` competition data is available). To keep the notebook runnable and preserve the downstream interface, the minimal fix is to load the available `sample_submission.csv` three times and assign it to `sub1`, `sub2`, and `sub3` (so the expected `scalar_coupling_constant` column exists). This avoids changing any later logic while ensuring the `.describe()` calls and the arithmetic in cell 2 work deterministically.'
- What this solution (achieved 1.18497) has done: 'Your current code is only averaging three identical `sample_submission.csv` files, which keeps every prediction at 0 and yields a poor score. To move the score toward the (much better) target, the smallest legitimate change is to replace the placeholder predictions with a simple, type-wise constant baseline learned from `train.csv` (median per coupling `type`), then apply it to `test.csv`. This keeps the overall “single-pass, no model training loop” core approach (just generating a submission from simple statistics) while making predictions non-trivial and typically much closer to reasonable values for this competition. The submission is written with the required columns and row alignment by `id`.'
- What this solution (achieved 1.23596) has done: 'Your current baseline uses the per-type median, which is robust but often underfits this competition and leaves a large gap to the target. To move the score down (better) toward the target with minimal logic change, I switch the per-type statistic from median to mean, which typically better matches the MAE-driven evaluation here. I also add a tiny Bayesian-style shrinkage toward the global mean to stabilize rare coupling types without changing the overall “type-wise constant prediction” approach. The submission format, alignment by `id`, and file path/output (`submission.csv`) remain unchanged.'
- What this solution (achieved 1.18497) has done: 'Your current approach is a type-wise constant baseline with shrinkage; the smallest way to move the score down (better) toward the target is to make those constants closer to what minimizes MAE for each type. Since MAE is minimized by the median (not the mean), we switch the per-type center back to median but keep the same shrinkage idea for stability. To better match the competition metric (average log-MAE per type), we also apply shrinkage in a type-adaptive way (more shrinkage for rare types, less for frequent ones) without changing the “no model training loop” core logic. The submission file name, columns, and id alignment remain unchanged.'
- What this solution (achieved 1.18497) has done: 'Your current baseline is a type-wise constant with adaptive shrinkage, but it is still too coarse for this metric because each coupling `type` has a different scale and offset. The smallest change that usually reduces log-MAE here without changing the “single-pass statistics baseline” core logic is to also use molecule-level information: add the molecule’s mean target (computed from train) as an offset, then shrink that molecule offset back toward 0 for unseen/rare molecules. This keeps the same style of solution (no model training loop; only groupby statistics) while typically moving the score downward (better) toward your target. The submission remains aligned by `id` and written to `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print(os.listdir("../input"))



## === cell 1
DATA_DIR = "../input/champs-scalar-coupling"
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

train = pd.read_csv(
    train_path, usecols=["molecule_name", "type", "scalar_coupling_constant"]
)
test = pd.read_csv(test_path, usecols=["id", "molecule_name", "type"])
sample_sub = pd.read_csv(sample_path)

print(train.head())
print(test.head())
print(sample_sub.head())



## === cell 2
type_center = train.groupby("type")["scalar_coupling_constant"].median()
type_count = train.groupby("type")["scalar_coupling_constant"].size().astype(float)
global_center = float(train["scalar_coupling_constant"].median())

base_alpha = 200.0
alpha_t = base_alpha / np.sqrt(type_count.clip(lower=1.0))
shrunken_type_center = (type_center * type_count + global_center * alpha_t) / (
    type_count + alpha_t
)

train_type_pred = (
    train["type"].map(shrunken_type_center).fillna(global_center).astype(float)
)
train_resid = train["scalar_coupling_constant"].astype(float) - train_type_pred

mol_resid_mean = train_resid.groupby(train["molecule_name"]).mean()
mol_count = train.groupby("molecule_name").size().astype(float)

mol_alpha = 50.0
beta_m = mol_alpha / np.sqrt(mol_count.clip(lower=1.0))
shrunken_mol_resid = (mol_resid_mean * mol_count) / (mol_count + beta_m)

pred_type = test["type"].map(shrunken_type_center).fillna(global_center).astype(float)
pred_mol = test["molecule_name"].map(shrunken_mol_resid).fillna(0.0).astype(float)
pred = (pred_type + pred_mol).astype(float)

submission = pd.DataFrame(
    {
        "id": test["id"].astype(
            sample_sub["id"].dtype if "id" in sample_sub else test["id"].dtype
        ),
        "scalar_coupling_constant": pred.values,
    }
)

submission.to_csv("submission.csv", index=False)

print(submission["scalar_coupling_constant"].describe())
print(submission.head())
print("Wrote submission.csv with rows:", len(submission))
