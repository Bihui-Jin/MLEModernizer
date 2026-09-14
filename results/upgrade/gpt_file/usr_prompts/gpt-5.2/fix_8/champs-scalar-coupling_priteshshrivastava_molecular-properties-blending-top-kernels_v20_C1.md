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
np.random.seed(42)

INPUT_DIR = "/kaggle/data/champs-scalar-coupling"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "/kaggle/input/champs-scalar-coupling"

print("Using INPUT_DIR:", INPUT_DIR)
print("Files:", sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")])[:10])



## === cell 1
from sklearn.ensemble import RandomForestRegressor

train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")
structures_path = os.path.join(INPUT_DIR, "structures.csv")
sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")

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
atom_cat = structures["atom"]
mapped = atom_cat.cat.rename_categories(lambda a: atomic_num.get(a, 0))
structures["atom_num"] = mapped.astype(np.int16)

s = structures[["molecule_name", "atom_index", "atom_num", "x", "y", "z"]]

all_mols = (
    pd.Index(train["molecule_name"].cat.categories)
    .union(pd.Index(test["molecule_name"].cat.categories))
    .union(pd.Index(s["molecule_name"].cat.categories))
)
mol_dtype = pd.CategoricalDtype(categories=all_mols)

train["molecule_name"] = train["molecule_name"].astype(mol_dtype)
test["molecule_name"] = test["molecule_name"].astype(mol_dtype)
s["molecule_name"] = s["molecule_name"].astype(mol_dtype)


def build_pair_df_fast(base_df: pd.DataFrame, s_df: pd.DataFrame) -> pd.DataFrame:
    s0 = (
        s_df.rename(
            columns={
                "atom_index": "atom_index_0",
                "atom_num": "atom_num_0",
                "x": "x0",
                "y": "y0",
                "z": "z0",
            }
        )[["molecule_name", "atom_index_0", "atom_num_0", "x0", "y0", "z0"]]
        .set_index(["molecule_name", "atom_index_0"], drop=False)
        .sort_index()
    )
    s1 = (
        s_df.rename(
            columns={
                "atom_index": "atom_index_1",
                "atom_num": "atom_num_1",
                "x": "x1",
                "y": "y1",
                "z": "z1",
            }
        )[["molecule_name", "atom_index_1", "atom_num_1", "x1", "y1", "z1"]]
        .set_index(["molecule_name", "atom_index_1"], drop=False)
        .sort_index()
    )

    df = base_df  # do not copy; we will not mutate base_df in-place below
    df = df.merge(
        s0.reset_index(drop=True),
        on=["molecule_name", "atom_index_0"],
        how="left",
        copy=False,
        sort=False,
    )
    df = df.merge(
        s1.reset_index(drop=True),
        on=["molecule_name", "atom_index_1"],
        how="left",
        copy=False,
        sort=False,
    )
    return df


train_feat = build_pair_df_fast(train, s)
test_feat = build_pair_df_fast(test, s)


def add_pair_features_fast(df: pd.DataFrame) -> pd.DataFrame:
    x0 = df["x0"].to_numpy(np.float32, copy=False)
    y0 = df["y0"].to_numpy(np.float32, copy=False)
    z0 = df["z0"].to_numpy(np.float32, copy=False)
    x1 = df["x1"].to_numpy(np.float32, copy=False)
    y1 = df["y1"].to_numpy(np.float32, copy=False)
    z1 = df["z1"].to_numpy(np.float32, copy=False)

    dx = x0 - x1
    dy = y0 - y1
    dz = z0 - z1

    dist2 = dx * dx + dy * dy + dz * dz
    dist = np.sqrt(dist2, dtype=np.float32)
    inv_dist = (1.0 / (dist + 1e-6)).astype(np.float32, copy=False)

    a0 = df["atom_num_0"].to_numpy(np.int16, copy=False)
    a1 = df["atom_num_1"].to_numpy(np.int16, copy=False)
    atom_num_sum = (a0 + a1).astype(np.int16, copy=False)
    atom_num_diff = (a0 - a1).astype(np.int16, copy=False)

    df = df.assign(
        dx=dx,
        dy=dy,
        dz=dz,
        dist2=dist2,
        dist=dist,
        inv_dist=inv_dist,
        atom_num_sum=atom_num_sum,
        atom_num_diff=atom_num_diff,
    )
    return df


train_feat = add_pair_features_fast(train_feat)
test_feat = add_pair_features_fast(test_feat)

fill0_cols = [
    "x0",
    "y0",
    "z0",
    "x1",
    "y1",
    "z1",
    "atom_num_0",
    "atom_num_1",
    "dx",
    "dy",
    "dz",
    "dist",
    "dist2",
    "inv_dist",
    "atom_num_sum",
    "atom_num_diff",
]
train_feat[fill0_cols] = train_feat[fill0_cols].fillna(0)
test_feat[fill0_cols] = test_feat[fill0_cols].fillna(0)

feature_cols = [
    "atom_index_0",
    "atom_index_1",
    "atom_num_0",
    "atom_num_1",
    "dx",
    "dy",
    "dz",
    "dist",
    "dist2",
    "inv_dist",
    "atom_num_sum",
    "atom_num_diff",
]

X_train_all = train_feat[feature_cols].to_numpy(dtype=np.float32, copy=False)
y_train_all = train_feat["scalar_coupling_constant"].to_numpy(
    dtype=np.float32, copy=False
)
X_test_all = test_feat[feature_cols].to_numpy(dtype=np.float32, copy=False)

preds = np.zeros(len(test_feat), dtype=np.float32)

train_type_cat = train_feat["type"]
test_type_cat = test_feat["type"]
all_cats = pd.Index(train_type_cat.cat.categories).union(
    pd.Index(test_type_cat.cat.categories)
)
train_type_cat = train_type_cat.cat.set_categories(all_cats)
test_type_cat = test_type_cat.cat.set_categories(all_cats)

train_codes = train_type_cat.cat.codes.to_numpy(np.int16, copy=False)
test_codes = test_type_cat.cat.codes.to_numpy(np.int16, copy=False)
types = list(train_type_cat.cat.categories)
print("Training per type:", [str(t) for t in types])


def indices_by_code(codes: np.ndarray, n_codes: int):
    order = np.argsort(codes, kind="mergesort")  # stable/deterministic
    sorted_codes = codes[order]
    cuts = np.flatnonzero(sorted_codes[1:] != sorted_codes[:-1]) + 1
    splits = np.split(order, cuts)
    out = [np.empty(0, dtype=np.int64) for _ in range(n_codes)]
    for arr in splits:
        if arr.size:
            out[int(codes[arr[0]])] = arr
    return out


train_idx_by_code = indices_by_code(train_codes, len(types))
test_idx_by_code = indices_by_code(test_codes, len(types))



## === cell 2
from joblib import Parallel, delayed


def fit_predict_one_type(code: int):
    tr_idx = train_idx_by_code[code]
    te_idx = test_idx_by_code[code]
    if te_idx.size == 0:
        return te_idx, None

    X_tr = X_train_all[tr_idx]
    y_tr = y_train_all[tr_idx]
    X_te = X_test_all[te_idx]

    model = RandomForestRegressor(
        n_estimators=60,
        random_state=42,
        n_jobs=1,  # keep identical per-model parallelism setting
        min_samples_leaf=2,
        max_features=1.0,
    )
    model.fit(X_tr, y_tr)
    p = model.predict(X_te).astype(np.float32, copy=False)
    return te_idx, p


n_type_jobs = min(4, len(types))  # unchanged cap
results = Parallel(n_jobs=n_type_jobs, prefer="threads", batch_size=1)(
    delayed(fit_predict_one_type)(code) for code in range(len(types))
)

for te_idx, p in results:
    if p is not None:
        preds[te_idx] = p

submission = pd.DataFrame(
    {"id": test["id"].to_numpy(copy=False), "scalar_coupling_constant": preds}
)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(submission), "cols:", list(submission.columns))
print(submission.head())
