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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.63413

# 6. Current score

1.44135

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.59259) has done: 'The crash happens because `test[col]` contains missing values (NaNs), and `ExtraTreesRegressor` cannot predict with NaNs. These NaNs come from the left-merges against `structures.csv` (unmatched keys) and/or from `dist_to_type_mean` when a type mean is missing in test. The minimal fix is to impute missing feature values deterministically using statistics computed from the training features only, and apply the same fill to both train and test for the selected feature columns. This keeps the model, features, and evaluation semantics the same while ensuring `reg.predict()` receives finite numeric input.'
- What this solution (achieved 1.57292) has done: 'The crash happens because the feature matrix `train[col]` still contains non-numeric columns (e.g., `atom0` / `atom1_struct` with values like `'H'`), but `ExtraTreesRegressor` requires purely numeric input. The minimal fix is to exclude those string/object columns when building `col`, leaving the rest of the pipeline unchanged (same split, model, metric, and prediction flow). This keeps all downstream variable names and interfaces the same while ensuring `reg.fit()` receives only numeric features. No changes are needed outside this cell.'
- What this solution (achieved 1.62093) has done: 'Your score is much worse than the target (lower is better), so the smallest safe way to move toward 0.63413 is to reduce avoidable leakage/noise without changing the model or feature logic. The biggest issue is that your validation split is row-wise even though the competition splits by molecule, which makes training less aligned with the test distribution and usually hurts generalization. I change only the split to be molecule-grouped (still the same ExtraTreesRegressor, same features, same loss/metric printout), keeping the rest intact. I also ensure `LabelEncoder` is fit on the union of train+test `type` characters to avoid any silent transform issues and keep the imputation deterministic from train only.'
- What this solution (achieved 1.3929) has done: 'I keep your feature engineering and ExtraTreesRegressor unchanged, but make two minimal fixes that typically improve generalization for this competition’s metric: (1) train one model per `type` (same model class/hyperparameters) because the evaluation averages MAE per type and each type has a different target scale, and (2) use a molecule-grouped split per type for the printed validation score to better match the competition split. I also ensure the `dist_to_type_mean` denominator is mapped using a stable per-type mean (with a global fallback) and keep deterministic, train-derived imputation to avoid any NaN/inf issues at predict time. The submission format, paths, and overall approach remain the same; it still produce `submission.csv`.'
- What this solution (achieved 1.38858) has done: 'Your current gap to the target is large (1.3929 vs 0.63413, lower is better), so we need a small change that improves generalization without changing the core model or feature set. The biggest avoidable issue is that you’re training each `type` model on a capped random subset of rows, which can bias the per-type fit and hurt the competition’s per-type averaged metric; switching to a deterministic molecule-level cap preserves the “cap” idea but better matches the split-by-molecule nature of the task. I keep the same ExtraTreesRegressor, same features, same imputation, and still train one model per `type`, but cap by sampling molecules (then taking all rows for those molecules) rather than sampling individual rows. This is typically a meaningful improvement while staying within your constraints and time limit, and it still writes a valid `submission.csv`.'
- What this solution (achieved 1.44135) has done: 'Your current score (1.38858, lower is better) is still far from the target (0.63413), so we need a small, safe generalization improvement without changing your model class or feature set. The most likely low-risk gain is that your per-type molecule split is *random*, while Kaggle’s split is by molecule; we can reduce variance and improve generalization by using a deterministic “hash-by-molecule” split (still per type) instead of shuffling. This keeps the same per-type ExtraTrees training and the same features, but makes validation/train separation more stable and typically improves leaderboard performance a bit. I also keep your molecule-capped training logic intact, only making the split deterministic and leak-safe.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from sklearn import *

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sub = pd.read_csv("../input/sample_submission.csv")
print(train.shape, test.shape, sub.shape)

train["atom1"] = train["type"].map(lambda x: str(x)[2])
train["atom2"] = train["type"].map(lambda x: str(x)[3])
test["atom1"] = test["type"].map(lambda x: str(x)[2])
test["atom2"] = test["type"].map(lambda x: str(x)[3])

for i in range(4):
    lbl = preprocessing.LabelEncoder()
    all_vals = pd.concat(
        [train["type"].map(lambda x: str(x)[i]), test["type"].map(lambda x: str(x)[i])],
        axis=0,
        ignore_index=True,
    ).astype(str)
    lbl.fit(all_vals)
    train["type" + str(i)] = lbl.transform(
        train["type"].map(lambda x: str(x)[i]).astype(str)
    )
    test["type" + str(i)] = lbl.transform(
        test["type"].map(lambda x: str(x)[i]).astype(str)
    )

structures0 = pd.read_csv("../input/structures.csv").rename(
    columns={
        "atom_index": "atom_index_0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
        "atom": "atom0",
    }
)
train = pd.merge(train, structures0, how="left", on=["molecule_name", "atom_index_0"])
test = pd.merge(test, structures0, how="left", on=["molecule_name", "atom_index_0"])
del structures0

structures1 = pd.read_csv("../input/structures.csv").rename(
    columns={
        "atom_index": "atom_index_1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
        "atom": "atom1_struct",
    }
)
train = pd.merge(train, structures1, how="left", on=["molecule_name", "atom_index_1"])
test = pd.merge(test, structures1, how="left", on=["molecule_name", "atom_index_1"])
del structures1

train["atom0_from_structure_mismatch"] = (train["atom0"] != train["atom1"]).astype(
    "float32"
)
test["atom0_from_structure_mismatch"] = (test["atom0"] != test["atom1"]).astype(
    "float32"
)

train["atom1_from_structure_mismatch"] = (
    train["atom1_struct"] != train["atom2"]
).astype("float32")
test["atom1_from_structure_mismatch"] = (test["atom1_struct"] != test["atom2"]).astype(
    "float32"
)

print(train.shape, test.shape, sub.shape)



## === cell 1
train_p0 = train[["x0", "y0", "z0"]].values
train_p1 = train[["x1", "y1", "z1"]].values
test_p0 = test[["x0", "y0", "z0"]].values
test_p1 = test[["x1", "y1", "z1"]].values

train["dist"] = np.linalg.norm(train_p0 - train_p1, axis=1)
test["dist"] = np.linalg.norm(test_p0 - test_p1, axis=1)

type_mean_train = train.groupby("type")["dist"].mean()
global_mean = float(train["dist"].mean())
train_type_mean = train["type"].map(type_mean_train).fillna(global_mean)
test_type_mean = test["type"].map(type_mean_train).fillna(global_mean)

train["dist_to_type_mean"] = train["dist"] / train_type_mean
test["dist_to_type_mean"] = test["dist"] / test_type_mean

train["dist_to_type_mean"] = train["dist_to_type_mean"].replace(
    [np.inf, -np.inf], np.nan
)
test["dist_to_type_mean"] = test["dist_to_type_mean"].replace([np.inf, -np.inf], np.nan)



## === cell 2
col = [
    c
    for c in train.columns
    if c
    not in [
        "id",
        "molecule_name",
        "scalar_coupling_constant",
        "type",
        "atom1",
        "atom2",
        "atom_index_0",
        "atom_index_1",
    ]
]
col = [c for c in col if pd.api.types.is_numeric_dtype(train[c])]

fill_values = train[col].median(numeric_only=True)
train[col] = train[col].replace([np.inf, -np.inf], np.nan).fillna(fill_values)
test[col] = test[col].replace([np.inf, -np.inf], np.nan).fillna(fill_values)

base_reg_params = dict(n_jobs=-1, n_estimators=20, random_state=4)

rng = np.random.RandomState(99)
pred_test = np.zeros(len(test), dtype=np.float64)

val_logs = []
types = train["type"].unique()
types.sort()


def _molecule_hash_split(mol_names, valid_frac=0.2, salt=17):
    h = pd.util.hash_pandas_object(pd.Index(mol_names), index=False).values
    u = ((h ^ np.uint64(salt)) % np.uint64(10**12)).astype(np.float64) / 1e12
    return u < valid_frac


for t in types:
    tr_idx_all = train.index[train["type"] == t]
    te_idx = test.index[test["type"] == t]
    if len(tr_idx_all) == 0:
        continue

    cap_rows = 250_000
    tr_mol_all = train.loc[tr_idx_all, "molecule_name"]
    if len(tr_idx_all) > cap_rows:
        mol_counts = tr_mol_all.value_counts()
        mols = mol_counts.index.values
        rng_tcap = np.random.RandomState(99)  # deterministic
        rng_tcap.shuffle(mols)
        chosen = []
        total = 0
        for m in mols:
            c = int(mol_counts[m])
            if total + c > cap_rows and total > 0:
                continue
            chosen.append(m)
            total += c
            if total >= cap_rows:
                break
        tr_idx = train.index[
            (train["type"] == t) & (train["molecule_name"].isin(chosen))
        ].values
    else:
        tr_idx = tr_idx_all.values

    tr_X = train.loc[tr_idx, col]
    tr_y = train.loc[tr_idx, "scalar_coupling_constant"]
    tr_mol = train.loc[tr_idx, "molecule_name"].astype(str)

    unique_mols = pd.Index(tr_mol.unique())
    is_valid_mol = _molecule_hash_split(unique_mols, valid_frac=0.2, salt=17)
    valid_mols = set(unique_mols[is_valid_mol].values)

    if len(valid_mols) == 0 and len(unique_mols) > 1:
        valid_mols = {unique_mols[0]}

    valid_mask = tr_mol.isin(valid_mols).values
    x1 = tr_X.loc[~valid_mask]
    y1 = tr_y.loc[~valid_mask]
    x2 = tr_X.loc[valid_mask]
    y2 = tr_y.loc[valid_mask]

    reg = ensemble.ExtraTreesRegressor(**base_reg_params)
    reg.fit(x1, y1)

    if len(x2) > 0:
        mae = metrics.mean_absolute_error(y2, reg.predict(x2))
        val_logs.append(np.log(mae))

    if len(te_idx) > 0:
        pred_test[te_idx] = reg.predict(test.loc[te_idx, col])

print(float(np.mean(val_logs)) if len(val_logs) else np.nan)

test["scalar_coupling_constant"] = pred_test
test[["id", "scalar_coupling_constant"]].to_csv("submission.csv", index=False)
