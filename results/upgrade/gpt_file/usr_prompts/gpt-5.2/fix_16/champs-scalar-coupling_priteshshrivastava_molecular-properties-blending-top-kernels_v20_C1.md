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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

np.random.seed(42)

INPUT_DIR = "/kaggle/data/champs-scalar-coupling"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "/kaggle/input/champs-scalar-coupling"

print("Using INPUT_DIR:", INPUT_DIR)
print("Files:", sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")])[:10])

from sklearn.ensemble import RandomForestRegressor

train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")
structures_path = os.path.join(INPUT_DIR, "structures.csv")

train = pd.read_csv(
    train_path,
    usecols=[
        "id",
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "scalar_coupling_constant",
    ],
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
        "scalar_coupling_constant": np.float32,
    },
)
test = pd.read_csv(
    test_path,
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
    },
)
structures = pd.read_csv(
    structures_path,
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
    dtype={
        "molecule_name": "category",
        "atom_index": np.int16,
        "atom": "category",
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
    },
)

atomic_num = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
atom_cat = structures["atom"].cat.categories
atom_map = pd.Series([atomic_num.get(a, 0) for a in atom_cat], index=atom_cat).astype(
    np.int16
)
structures["atom_num"] = structures["atom"].map(atom_map).astype(np.int16)

s = structures[["molecule_name", "atom_index", "atom_num", "x", "y", "z"]]

mol_dtype = pd.CategoricalDtype(categories=s["molecule_name"].cat.categories)
train["molecule_name"] = train["molecule_name"].astype(mol_dtype)
test["molecule_name"] = test["molecule_name"].astype(mol_dtype)
s["molecule_name"] = s["molecule_name"].astype(mol_dtype)

SHIFT = np.int64(64)  # > max atom_index per molecule in this dataset
mol_codes_s = s["molecule_name"].cat.codes.to_numpy(np.int32, copy=False)
atom_idx_s = s["atom_index"].to_numpy(np.int32, copy=False)
n_mols = int(mol_codes_s.max()) + 1
table_size = int(n_mols * int(SHIFT))

atom_num_tbl = np.zeros(table_size, dtype=np.int16)
x_tbl = np.zeros(table_size, dtype=np.float32)
y_tbl = np.zeros(table_size, dtype=np.float32)
z_tbl = np.zeros(table_size, dtype=np.float32)

keys_s = (mol_codes_s.astype(np.int64) * SHIFT) + atom_idx_s.astype(np.int64)
atom_num_tbl[keys_s] = s["atom_num"].to_numpy(np.int16, copy=False)
x_tbl[keys_s] = s["x"].to_numpy(np.float32, copy=False)
y_tbl[keys_s] = s["y"].to_numpy(np.float32, copy=False)
z_tbl[keys_s] = s["z"].to_numpy(np.float32, copy=False)


def build_features_numpy_fast(base_df: pd.DataFrame):
    bm = (
        base_df["molecule_name"]
        .cat.codes.to_numpy(np.int32, copy=False)
        .astype(np.int64, copy=False)
    )
    ai0_i64 = (
        base_df["atom_index_0"]
        .to_numpy(np.int32, copy=False)
        .astype(np.int64, copy=False)
    )
    ai1_i64 = (
        base_df["atom_index_1"]
        .to_numpy(np.int32, copy=False)
        .astype(np.int64, copy=False)
    )

    key0 = bm * SHIFT + ai0_i64
    key1 = bm * SHIFT + ai1_i64

    atom_num_0 = atom_num_tbl[key0].astype(np.float32, copy=False)
    x0 = x_tbl[key0]
    y0 = y_tbl[key0]
    z0 = z_tbl[key0]

    atom_num_1 = atom_num_tbl[key1].astype(np.float32, copy=False)
    x1 = x_tbl[key1]
    y1 = y_tbl[key1]
    z1 = z_tbl[key1]

    dx = (x0 - x1).astype(np.float32, copy=False)
    dy = (y0 - y1).astype(np.float32, copy=False)
    dz = (z0 - z1).astype(np.float32, copy=False)
    dist2 = (dx * dx + dy * dy + dz * dz).astype(np.float32, copy=False)
    dist = np.sqrt(dist2, dtype=np.float32)
    inv_dist = (1.0 / (dist + 1e-6)).astype(np.float32, copy=False)

    atom_num_sum = (atom_num_0 + atom_num_1).astype(np.float32, copy=False)
    atom_num_diff = (atom_num_0 - atom_num_1).astype(np.float32, copy=False)

    ai0_f = ai0_i64.astype(np.float32, copy=False)
    ai1_f = ai1_i64.astype(np.float32, copy=False)

    return (
        ai0_f,
        ai1_f,
        atom_num_0,
        atom_num_1,
        dx,
        dy,
        dz,
        dist.astype(np.float32, copy=False),
        dist2,
        inv_dist,
        atom_num_sum,
        atom_num_diff,
    )


all_type_cats = pd.Index(train["type"].cat.categories).union(
    pd.Index(test["type"].cat.categories)
)
type_dtype = pd.CategoricalDtype(categories=all_type_cats)
train["type"] = train["type"].astype(type_dtype)
test["type"] = test["type"].astype(type_dtype)

type_code_train = (
    train["type"]
    .cat.codes.to_numpy(np.int16, copy=False)
    .astype(np.float32, copy=False)
)
type_code_test = (
    test["type"].cat.codes.to_numpy(np.int16, copy=False).astype(np.float32, copy=False)
)

print(
    "Loaded:",
    "train",
    train.shape,
    "test",
    test.shape,
    "structures",
    structures.shape,
    "n_mols",
    n_mols,
    "table_size",
    table_size,
)



## === cell 1
model_params = dict(
    n_estimators=60,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2,
    max_features=1.0,
)

preds = np.empty(len(test), dtype=np.float32)

(
    ai0_tr,
    ai1_tr,
    a0_tr,
    a1_tr,
    dx_tr,
    dy_tr,
    dz_tr,
    dist_tr,
    dist2_tr,
    invdist_tr,
    asum_tr,
    adiff_tr,
) = build_features_numpy_fast(train)

(
    ai0_te,
    ai1_te,
    a0_te,
    a1_te,
    dx_te,
    dy_te,
    dz_te,
    dist_te,
    dist2_te,
    invdist_te,
    asum_te,
    adiff_te,
) = build_features_numpy_fast(test)

y_train_all = train["scalar_coupling_constant"].to_numpy(dtype=np.float32, copy=False)

N_FEATURES = 13

X_train_all = np.empty((len(train), N_FEATURES), dtype=np.float32, order="C")
X_train_all[:, 0] = type_code_train
X_train_all[:, 1] = ai0_tr
X_train_all[:, 2] = ai1_tr
X_train_all[:, 3] = a0_tr
X_train_all[:, 4] = a1_tr
X_train_all[:, 5] = dx_tr
X_train_all[:, 6] = dy_tr
X_train_all[:, 7] = dz_tr
X_train_all[:, 8] = dist_tr
X_train_all[:, 9] = dist2_tr
X_train_all[:, 10] = invdist_tr
X_train_all[:, 11] = asum_tr
X_train_all[:, 12] = adiff_tr

X_test_all = np.empty((len(test), N_FEATURES), dtype=np.float32, order="C")
X_test_all[:, 0] = type_code_test
X_test_all[:, 1] = ai0_te
X_test_all[:, 2] = ai1_te
X_test_all[:, 3] = a0_te
X_test_all[:, 4] = a1_te
X_test_all[:, 5] = dx_te
X_test_all[:, 6] = dy_te
X_test_all[:, 7] = dz_te
X_test_all[:, 8] = dist_te
X_test_all[:, 9] = dist2_te
X_test_all[:, 10] = invdist_te
X_test_all[:, 11] = asum_te
X_test_all[:, 12] = adiff_te

model = RandomForestRegressor(**model_params)
model.fit(X_train_all, y_train_all)
preds[:] = model.predict(X_test_all).astype(np.float32, copy=False)

submission = pd.DataFrame(
    {"id": test["id"].to_numpy(copy=False), "scalar_coupling_constant": preds}
)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(submission), "cols:", list(submission.columns))
print(submission.head())
