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
from sklearn import preprocessing, ensemble, model_selection, metrics

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

DATA_ROOT = "/kaggle/data/champs-scalar-coupling"
if not os.path.exists(DATA_ROOT):
    DATA_ROOT = "/kaggle/input/champs-scalar-coupling"

train = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))
test = pd.read_csv(os.path.join(DATA_ROOT, "test.csv"))
sub = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))
print(train.shape, test.shape, sub.shape)

train_type_str = train["type"].astype(str)
test_type_str = test["type"].astype(str)
train["atom1"] = train_type_str.str[2]
train["atom2"] = train_type_str.str[3]
test["atom1"] = test_type_str.str[2]
test["atom2"] = test_type_str.str[3]

for i in range(4):
    lbl = preprocessing.LabelEncoder()
    all_chars = pd.concat([train_type_str.str[i], test_type_str.str[i]], axis=0)
    lbl.fit(all_chars.values)
    train["type" + str(i)] = lbl.transform(train_type_str.str[i].values)
    test["type" + str(i)] = lbl.transform(test_type_str.str[i].values)

structures = pd.read_csv(os.path.join(DATA_ROOT, "structures.csv"))
structures["atom_index"] = structures["atom_index"].astype(np.int32, copy=False)
structures["molecule_name"] = structures["molecule_name"].astype("category")
structures["atom"] = structures["atom"].astype(str)

mol_groups = structures.groupby("molecule_name", sort=False)
mol_to_xyz = {}
mol_to_atom = {}
for mol, g in mol_groups:
    idx = g["atom_index"].to_numpy()
    n = int(idx.max()) + 1
    xyz = np.empty((n, 3), dtype=np.float32)
    atoms = np.empty(n, dtype=object)
    xyz[idx, 0] = g["x"].to_numpy(dtype=np.float32, copy=False)
    xyz[idx, 1] = g["y"].to_numpy(dtype=np.float32, copy=False)
    xyz[idx, 2] = g["z"].to_numpy(dtype=np.float32, copy=False)
    atoms[idx] = g["atom"].to_numpy(dtype=object, copy=False)
    mol_to_xyz[mol] = xyz
    mol_to_atom[mol] = atoms
del structures, mol_groups


def _attach_coords(df, mol_to_xyz, mol_to_atom, idx_col, atom_col, prefix):
    n = len(df)
    x = np.empty(n, dtype=np.float32)
    y = np.empty(n, dtype=np.float32)
    z = np.empty(n, dtype=np.float32)
    x.fill(np.nan)
    y.fill(np.nan)
    z.fill(np.nan)

    mol_vals = df["molecule_name"].values
    idx_vals = df[idx_col].to_numpy(dtype=np.int32, copy=False)
    atom_vals = df[atom_col].astype(str).values

    mol_series = pd.Series(mol_vals)
    for mol, row_idx in mol_series.groupby(mol_series).groups.items():
        xyz = mol_to_xyz.get(mol)
        atoms = mol_to_atom.get(mol)
        if xyz is None:
            continue
        r = np.fromiter(row_idx, dtype=np.int64, count=len(row_idx))
        ai = idx_vals[r]
        ok = (ai >= 0) & (ai < len(atoms))
        if np.any(ok):
            r_ok = r[ok]
            ai_ok = ai[ok]
            ok2 = atoms[ai_ok] == atom_vals[r_ok]
            if np.any(ok2):
                r2 = r_ok[ok2]
                ai2 = ai_ok[ok2]
                x[r2] = xyz[ai2, 0]
                y[r2] = xyz[ai2, 1]
                z[r2] = xyz[ai2, 2]

    df[prefix + "0"] = x
    df[prefix + "1"] = y
    df[prefix + "2"] = z
    df.rename(
        columns={
            prefix + "0": prefix[0] + prefix[1:],
            prefix + "1": prefix[0] + prefix[1:],
            prefix + "2": prefix[0] + prefix[1:],
        },
        inplace=True,
    )


def attach_xyz(df, idx_col, atom_col, suffix):
    n = len(df)
    x = np.empty(n, dtype=np.float32)
    x.fill(np.nan)
    y = np.empty(n, dtype=np.float32)
    y.fill(np.nan)
    z = np.empty(n, dtype=np.float32)
    z.fill(np.nan)

    mol_vals = df["molecule_name"].values
    idx_vals = df[idx_col].to_numpy(dtype=np.int32, copy=False)
    atom_vals = df[atom_col].astype(str).values

    mol_series = pd.Series(mol_vals)
    for mol, row_idx in mol_series.groupby(mol_series).groups.items():
        xyz = mol_to_xyz.get(mol)
        atoms = mol_to_atom.get(mol)
        if xyz is None:
            continue
        r = np.fromiter(row_idx, dtype=np.int64, count=len(row_idx))
        ai = idx_vals[r]
        ok = (ai >= 0) & (ai < len(atoms))
        if not np.any(ok):
            continue
        r_ok = r[ok]
        ai_ok = ai[ok]
        ok2 = atoms[ai_ok] == atom_vals[r_ok]
        if not np.any(ok2):
            continue
        r2 = r_ok[ok2]
        ai2 = ai_ok[ok2]
        x[r2] = xyz[ai2, 0]
        y[r2] = xyz[ai2, 1]
        z[r2] = xyz[ai2, 2]

    df["x" + suffix] = x
    df["y" + suffix] = y
    df["z" + suffix] = z


attach_xyz(train, "atom_index_0", "atom1", "0")
attach_xyz(test, "atom_index_0", "atom1", "0")
attach_xyz(train, "atom_index_1", "atom2", "1")
attach_xyz(test, "atom_index_1", "atom2", "1")

print(train.shape, test.shape, sub.shape)



## === cell 1
train_p0 = train[["x0", "y0", "z0"]].to_numpy(dtype=np.float32, copy=False)
train_p1 = train[["x1", "y1", "z1"]].to_numpy(dtype=np.float32, copy=False)
test_p0 = test[["x0", "y0", "z0"]].to_numpy(dtype=np.float32, copy=False)
test_p1 = test[["x1", "y1", "z1"]].to_numpy(dtype=np.float32, copy=False)

d = train_p0 - train_p1
train["dist"] = np.sqrt((d * d).sum(axis=1))
d = test_p0 - test_p1
test["dist"] = np.sqrt((d * d).sum(axis=1))

type_mean_dist = train.groupby("type")["dist"].mean()
train["dist_to_type_mean"] = train["dist"] / train["type"].map(type_mean_dist)
test["dist_to_type_mean"] = test["dist"] / test["type"].map(type_mean_dist)

global_mean_dist = float(train["dist"].mean())
train["dist_to_type_mean"] = train["dist_to_type_mean"].fillna(
    train["dist"] / global_mean_dist
)
test["dist_to_type_mean"] = test["dist_to_type_mean"].fillna(
    test["dist"] / global_mean_dist
)



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

train_feat = train[col].copy()
test_feat = test[col].copy()

med = train_feat.median(numeric_only=True)
train_feat = train_feat.fillna(med)
test_feat = test_feat.fillna(med)

obj_cols = [c for c in train_feat.columns if train_feat[c].dtype == "object"]
for c in obj_cols:
    train_feat[c] = pd.to_numeric(train_feat[c], errors="coerce")
    test_feat[c] = pd.to_numeric(test_feat[c], errors="coerce")
train_feat = train_feat.fillna(train_feat.median(numeric_only=True))
test_feat = test_feat.fillna(train_feat.median(numeric_only=True))

N_TRAIN = 3_000_000
train_X = train_feat.tail(N_TRAIN)
train_y = train["scalar_coupling_constant"].tail(N_TRAIN)
train_type = train["type"].tail(N_TRAIN)
train_mol = train["molecule_name"].tail(N_TRAIN)

reg = ensemble.ExtraTreesRegressor(
    n_jobs=-1,
    n_estimators=120,
    random_state=4,
)

uniq_mols = train_mol.unique()
mol_train, mol_val = model_selection.train_test_split(
    uniq_mols, test_size=0.2, random_state=99
)
is_val = train_mol.isin(mol_val)

x1 = train_X.loc[~is_val]
y1 = train_y.loc[~is_val]
t1 = train_type.loc[~is_val]

x2 = train_X.loc[is_val]
y2 = train_y.loc[is_val]
t2 = train_type.loc[is_val]

x1_np = np.ascontiguousarray(x1.to_numpy(dtype=np.float32, copy=False))
x2_np = np.ascontiguousarray(x2.to_numpy(dtype=np.float32, copy=False))
trainX_np = np.ascontiguousarray(train_X.to_numpy(dtype=np.float32, copy=False))
testX_np = np.ascontiguousarray(test_feat.to_numpy(dtype=np.float32, copy=False))

reg.fit(x1_np, y1.to_numpy(copy=False))

val_pred = reg.predict(x2_np)

val_df = pd.DataFrame({"type": t2.values, "y": y2.values, "p": val_pred})
val_df["ae"] = (val_df["y"] - val_df["p"]).abs()
mae_by_type = val_df.groupby("type", sort=False)["ae"].mean()
print("Validation mean log(MAE by type):", float(np.mean(np.log(mae_by_type.values))))

reg.fit(trainX_np, train_y.to_numpy(copy=False))

test["scalar_coupling_constant"] = reg.predict(testX_np)

test[["id", "scalar_coupling_constant"]].to_csv("submission.csv", index=False)
print(
    "Wrote submission.csv with shape:", test[["id", "scalar_coupling_constant"]].shape
)
