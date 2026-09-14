# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing, ensemble, metrics

BASE_CANDIDATES = [
    "/kaggle/input/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling",
    "../input/champs-scalar-coupling",
    "../input",
]
BASE = None
for b in BASE_CANDIDATES:
    if os.path.exists(b):
        BASE = b
        break
if BASE is None:
    raise FileNotFoundError(
        "Could not locate input data directory among: " + str(BASE_CANDIDATES)
    )

train = pd.read_csv(os.path.join(BASE, "train.csv"))
test = pd.read_csv(os.path.join(BASE, "test.csv"))
sub = pd.read_csv(os.path.join(BASE, "sample_submission.csv"))
print(train.shape, test.shape, sub.shape)

train["atom"] = train["type"].map(lambda x: str(x)[3])
test["atom"] = test["type"].map(lambda x: str(x)[3])

lbl = preprocessing.LabelEncoder()
for i in range(4):
    train["type" + str(i)] = lbl.fit_transform(train["type"].map(lambda x: str(x)[i]))
    test["type" + str(i)] = lbl.transform(test["type"].map(lambda x: str(x)[i]))

structures = pd.read_csv(os.path.join(BASE, "structures.csv"))

s1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
        "atom": "atom1",
    }
)
s0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
        "atom": "atom0",
    }
)

train = pd.merge(
    train,
    s1[["molecule_name", "atom_index_1", "x1", "y1", "z1", "atom1"]],
    how="left",
    on=["molecule_name", "atom_index_1"],
)
test = pd.merge(
    test,
    s1[["molecule_name", "atom_index_1", "x1", "y1", "z1", "atom1"]],
    how="left",
    on=["molecule_name", "atom_index_1"],
)

train = pd.merge(
    train,
    s0[["molecule_name", "atom_index_0", "x0", "y0", "z0", "atom0"]],
    how="left",
    on=["molecule_name", "atom_index_0"],
)
test = pd.merge(
    test,
    s0[["molecule_name", "atom_index_0", "x0", "y0", "z0", "atom0"]],
    how="left",
    on=["molecule_name", "atom_index_0"],
)

del structures, s0, s1

print(train.shape, test.shape, sub.shape)



## === cell 1
train.head()



## === cell 2
col = [
    c
    for c in train.columns
    if c not in ["id", "molecule_name", "scalar_coupling_constant", "type", "atom"]
]

X_train = train[col].copy()
X_test = test[col].copy()

obj_cols = [
    c for c in col if (X_train[c].dtype == "object" or X_test[c].dtype == "object")
]
for c in obj_cols:
    combined = pd.concat([X_train[c], X_test[c]], axis=0, ignore_index=True).astype(
        "string"
    )
    cats = pd.Categorical(combined)
    codes = cats.codes.astype(np.int32)  # -1 indicates missing
    X_train[c] = codes[: len(X_train)]
    X_test[c] = codes[len(X_train) :]

for c in col:
    if pd.api.types.is_numeric_dtype(X_train[c]):
        fill = X_train[c].median()
        if pd.isna(fill):
            fill = 0.0
    else:
        fill = X_train[c].mode(dropna=True)
        fill = fill.iloc[0] if len(fill) else "NA"
    X_train[c] = X_train[c].fillna(fill)
    X_test[c] = X_test[c].fillna(fill)

rng = np.random.RandomState(99)
unique_mols = train["molecule_name"].unique()
rng.shuffle(unique_mols)
n_val = int(0.2 * len(unique_mols))
val_mols = set(unique_mols[:n_val])
is_val = train["molecule_name"].isin(val_mols)

reg_params = dict(
    n_estimators=200,  # was 60
    n_jobs=-1,
    random_state=4,
    min_samples_leaf=1,
    min_samples_split=2,
)

types = train["type"].unique()
oof = np.empty(len(train), dtype=np.float64)
oof.fill(np.nan)

for t in types:
    idx_tr = (~is_val) & (train["type"] == t)
    idx_va = (is_val) & (train["type"] == t)

    if idx_tr.sum() == 0 or idx_va.sum() == 0:
        continue

    reg = ensemble.ExtraTreesRegressor(**reg_params)
    reg.fit(X_train.loc[idx_tr, col], train.loc[idx_tr, "scalar_coupling_constant"])
    oof[idx_va.values] = reg.predict(X_train.loc[idx_va, col])

val_scores = []
for t in types:
    m = is_val & (train["type"] == t)
    if m.sum() == 0:
        continue
    pred = oof[m.values]
    if np.isnan(pred).any():
        continue
    mae = metrics.mean_absolute_error(train.loc[m, "scalar_coupling_constant"], pred)
    mae = max(mae, 1e-12)
    val_scores.append(np.log(mae))
print("Validation mean log(MAE) across types:", float(np.mean(val_scores)))

test_pred = np.empty(len(test), dtype=np.float64)
test_pred.fill(np.nan)

for t in types:
    tr_mask = train["type"] == t
    te_mask = test["type"] == t
    if te_mask.sum() == 0:
        continue

    reg = ensemble.ExtraTreesRegressor(**reg_params)
    reg.fit(X_train.loc[tr_mask, col], train.loc[tr_mask, "scalar_coupling_constant"])
    test_pred[te_mask.values] = reg.predict(X_test.loc[te_mask, col])

if np.isnan(test_pred).any():
    reg_global = ensemble.ExtraTreesRegressor(**reg_params)
    reg_global.fit(X_train[col], train["scalar_coupling_constant"])
    miss = np.isnan(test_pred)
    test_pred[miss] = reg_global.predict(X_test.loc[miss, col])

test["scalar_coupling_constant"] = test_pred

submission = test[["id", "scalar_coupling_constant"]].copy()
submission.to_csv("submission.csv", float_format="%.9f", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
