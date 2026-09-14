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

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

DATA_DIR = "/kaggle/data/champs-scalar-coupling"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
struct_path = os.path.join(DATA_DIR, "structures.csv")

mulliken_path = os.path.join(DATA_DIR, "mulliken_charges.csv")
shield_path = os.path.join(DATA_DIR, "magnetic_shielding_tensors.csv")
dipole_path = os.path.join(DATA_DIR, "dipole_moments.csv")
energy_path = os.path.join(DATA_DIR, "potential_energy.csv")

try:
    import pyarrow  # noqa: F401

    _csv_engine = "pyarrow"
except Exception:
    _csv_engine = "c"

np.random.seed(0)

train = pd.read_csv(
    train_path,
    engine=_csv_engine,
    dtype={
        "id": "int64",
        "molecule_name": "string",
        "atom_index_0": "int16",
        "atom_index_1": "int16",
        "type": "category",
        "scalar_coupling_constant": "float32",
    },
)
test = pd.read_csv(
    test_path,
    engine=_csv_engine,
    dtype={
        "id": "int64",
        "molecule_name": "string",
        "atom_index_0": "int16",
        "atom_index_1": "int16",
        "type": "category",
    },
)
structures = pd.read_csv(
    struct_path,
    engine=_csv_engine,
    dtype={
        "molecule_name": "string",
        "atom_index": "int16",
        "atom": "category",
        "x": "float32",
        "y": "float32",
        "z": "float32",
    },
)

mulliken = pd.read_csv(
    mulliken_path,
    engine=_csv_engine,
    dtype={
        "molecule_name": "string",
        "atom_index": "int16",
        "mulliken_charge": "float32",
    },
)
shield = pd.read_csv(
    shield_path,
    engine=_csv_engine,
    dtype={
        "molecule_name": "string",
        "atom_index": "int16",
        "XX": "float32",
        "YX": "float32",
        "ZX": "float32",
        "XY": "float32",
        "YY": "float32",
        "ZY": "float32",
        "XZ": "float32",
        "YZ": "float32",
        "ZZ": "float32",
    },
)
dipole = pd.read_csv(
    dipole_path,
    engine=_csv_engine,
    dtype={"molecule_name": "string", "X": "float32", "Y": "float32", "Z": "float32"},
)
energy = pd.read_csv(
    energy_path,
    engine=_csv_engine,
    dtype={"molecule_name": "string", "potential_energy": "float32"},
)

s0 = (
    structures.rename(
        columns={
            "atom_index": "atom_index_0",
            "atom": "atom_0",
            "x": "x0",
            "y": "y0",
            "z": "z0",
        }
    )[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]]
    .set_index(["molecule_name", "atom_index_0"])
    .sort_index()
)

s1 = (
    structures.rename(
        columns={
            "atom_index": "atom_index_1",
            "atom": "atom_1",
            "x": "x1",
            "y": "y1",
            "z": "z1",
        }
    )[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]]
    .set_index(["molecule_name", "atom_index_1"])
    .sort_index()
)

m0 = (
    mulliken.rename(
        columns={"atom_index": "atom_index_0", "mulliken_charge": "mulliken_0"}
    )[["molecule_name", "atom_index_0", "mulliken_0"]]
    .set_index(["molecule_name", "atom_index_0"])
    .sort_index()
)

m1 = (
    mulliken.rename(
        columns={"atom_index": "atom_index_1", "mulliken_charge": "mulliken_1"}
    )[["molecule_name", "atom_index_1", "mulliken_1"]]
    .set_index(["molecule_name", "atom_index_1"])
    .sort_index()
)

sh0 = (
    shield.rename(columns={"atom_index": "atom_index_0"})
    .add_prefix("sh0_")
    .rename(
        columns={
            "sh0_molecule_name": "molecule_name",
            "sh0_atom_index_0": "atom_index_0",
        }
    )
    .set_index(["molecule_name", "atom_index_0"])
    .sort_index()
)

sh1 = (
    shield.rename(columns={"atom_index": "atom_index_1"})
    .add_prefix("sh1_")
    .rename(
        columns={
            "sh1_molecule_name": "molecule_name",
            "sh1_atom_index_1": "atom_index_1",
        }
    )
    .set_index(["molecule_name", "atom_index_1"])
    .sort_index()
)

dipole_i = dipole.set_index("molecule_name").sort_index()
energy_i = energy.set_index("molecule_name").sort_index()


def add_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    key0 = pd.MultiIndex.from_frame(out[["molecule_name", "atom_index_0"]])
    key1 = pd.MultiIndex.from_frame(out[["molecule_name", "atom_index_1"]])

    a0 = s0.reindex(key0).reset_index(drop=True)
    a1 = s1.reindex(key1).reset_index(drop=True)
    c0 = m0.reindex(key0).reset_index(drop=True)
    c1 = m1.reindex(key1).reset_index(drop=True)
    t0 = sh0.reindex(key0).reset_index(drop=True)
    t1 = sh1.reindex(key1).reset_index(drop=True)

    d = dipole_i.reindex(out["molecule_name"]).reset_index(drop=True)
    e = energy_i.reindex(out["molecule_name"]).reset_index(drop=True)

    for col in a0.columns:
        out[col] = a0[col].to_numpy()
    for col in a1.columns:
        out[col] = a1[col].to_numpy()
    for col in c0.columns:
        out[col] = c0[col].to_numpy()
    for col in c1.columns:
        out[col] = c1[col].to_numpy()
    for col in t0.columns:
        out[col] = t0[col].to_numpy()
    for col in t1.columns:
        out[col] = t1[col].to_numpy()
    out["X"] = d["X"].to_numpy()
    out["Y"] = d["Y"].to_numpy()
    out["Z"] = d["Z"].to_numpy()
    out["potential_energy"] = e["potential_energy"].to_numpy()

    dx = out["x0"].to_numpy() - out["x1"].to_numpy()
    dy = out["y0"].to_numpy() - out["y1"].to_numpy()
    dz = out["z0"].to_numpy() - out["z1"].to_numpy()
    out["dx"] = dx
    out["dy"] = dy
    out["dz"] = dz
    out["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)

    eps = np.float32(1e-6)
    dist = out["dist"].to_numpy()
    out["dist2"] = (dist * dist).astype(np.float32)
    out["inv_dist"] = (1.0 / (dist + eps)).astype(np.float32)

    out["x0_plus_x1"] = (out["x0"].to_numpy() + out["x1"].to_numpy()).astype(np.float32)
    out["y0_plus_y1"] = (out["y0"].to_numpy() + out["y1"].to_numpy()).astype(np.float32)
    out["z0_plus_z1"] = (out["z0"].to_numpy() + out["z1"].to_numpy()).astype(np.float32)
    out["x0_minus_x1"] = dx
    out["y0_minus_y1"] = dy
    out["z0_minus_z1"] = dz

    out["mulliken_diff"] = (
        out["mulliken_0"].to_numpy() - out["mulliken_1"].to_numpy()
    ).astype(np.float32)
    return out


train_f = add_structure_features(train)
test_f = add_structure_features(test)


def make_swapped_from_features(df_f: pd.DataFrame) -> pd.DataFrame:
    swapped = df_f.copy()

    swapped[["atom_index_0", "atom_index_1"]] = swapped[
        ["atom_index_1", "atom_index_0"]
    ].to_numpy()
    swapped[["atom_0", "atom_1"]] = swapped[["atom_1", "atom_0"]].to_numpy()
    swapped[["x0", "y0", "z0", "x1", "y1", "z1"]] = swapped[
        ["x1", "y1", "z1", "x0", "y0", "z0"]
    ].to_numpy()
    swapped[["mulliken_0", "mulliken_1"]] = swapped[
        ["mulliken_1", "mulliken_0"]
    ].to_numpy()

    sh0_cols = [c for c in swapped.columns if c.startswith("sh0_")]
    sh1_cols = [c for c in swapped.columns if c.startswith("sh1_")]
    if sh0_cols and sh1_cols and (len(sh0_cols) == len(sh1_cols)):
        tmp = swapped[sh0_cols].to_numpy(copy=True)
        swapped[sh0_cols] = swapped[sh1_cols].to_numpy(copy=False)
        swapped[sh1_cols] = tmp

    dx = swapped["x0"].to_numpy() - swapped["x1"].to_numpy()
    dy = swapped["y0"].to_numpy() - swapped["y1"].to_numpy()
    dz = swapped["z0"].to_numpy() - swapped["z1"].to_numpy()
    swapped["dx"] = dx
    swapped["dy"] = dy
    swapped["dz"] = dz
    swapped["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)

    eps = np.float32(1e-6)
    dist = swapped["dist"].to_numpy()
    swapped["dist2"] = (dist * dist).astype(np.float32)
    swapped["inv_dist"] = (1.0 / (dist + eps)).astype(np.float32)

    swapped["x0_plus_x1"] = (
        swapped["x0"].to_numpy() + swapped["x1"].to_numpy()
    ).astype(np.float32)
    swapped["y0_plus_y1"] = (
        swapped["y0"].to_numpy() + swapped["y1"].to_numpy()
    ).astype(np.float32)
    swapped["z0_plus_z1"] = (
        swapped["z0"].to_numpy() + swapped["z1"].to_numpy()
    ).astype(np.float32)
    swapped["x0_minus_x1"] = dx
    swapped["y0_minus_y1"] = dy
    swapped["z0_minus_z1"] = dz

    swapped["mulliken_diff"] = (
        swapped["mulliken_0"].to_numpy() - swapped["mulliken_1"].to_numpy()
    ).astype(np.float32)
    return swapped


train_f_swapped = make_swapped_from_features(train_f)
train_f = pd.concat([train_f, train_f_swapped], axis=0, ignore_index=True)

for col in ["atom_0", "atom_1"]:
    train_f[col] = train_f[col].astype("category")
    test_f[col] = test_f[col].astype("category")

all_atoms = pd.concat(
    [train_f[["atom_0", "atom_1"]], test_f[["atom_0", "atom_1"]]], axis=0
)
atom0_cats = all_atoms["atom_0"].astype("category").cat.categories
atom1_cats = all_atoms["atom_1"].astype("category").cat.categories
train_f["atom_0"] = pd.Categorical(train_f["atom_0"], categories=atom0_cats)
test_f["atom_0"] = pd.Categorical(test_f["atom_0"], categories=atom0_cats)
train_f["atom_1"] = pd.Categorical(train_f["atom_1"], categories=atom1_cats)
test_f["atom_1"] = pd.Categorical(test_f["atom_1"], categories=atom1_cats)

train_atom0_code = train_f["atom_0"].cat.codes.astype(np.int16)
train_atom1_code = train_f["atom_1"].cat.codes.astype(np.int16)
test_atom0_code = test_f["atom_0"].cat.codes.astype(np.int16)
test_atom1_code = test_f["atom_1"].cat.codes.astype(np.int16)

shield_cols0 = [c for c in train_f.columns if c.startswith("sh0_")]
shield_cols1 = [c for c in train_f.columns if c.startswith("sh1_")]

num_features = (
    [
        "dx",
        "dy",
        "dz",
        "dist",
        "dist2",
        "inv_dist",
        "x0",
        "y0",
        "z0",
        "x1",
        "y1",
        "z1",
        "x0_plus_x1",
        "y0_plus_y1",
        "z0_plus_z1",
        "x0_minus_x1",
        "y0_minus_y1",
        "z0_minus_z1",
        "mulliken_0",
        "mulliken_1",
        "mulliken_diff",
        "X",
        "Y",
        "Z",
        "potential_energy",
    ]
    + shield_cols0
    + shield_cols1
)

X_train_num = train_f[num_features].astype("float32")
X_test_num = test_f[num_features].astype("float32")

X_train_full = np.ascontiguousarray(
    np.column_stack(
        [
            X_train_num.to_numpy(dtype=np.float32, copy=False),
            train_atom0_code.to_numpy(dtype=np.float32, copy=False),
            train_atom1_code.to_numpy(dtype=np.float32, copy=False),
        ]
    ),
    dtype=np.float32,
)
X_test_full = np.ascontiguousarray(
    np.column_stack(
        [
            X_test_num.to_numpy(dtype=np.float32, copy=False),
            test_atom0_code.to_numpy(dtype=np.float32, copy=False),
            test_atom1_code.to_numpy(dtype=np.float32, copy=False),
        ]
    ),
    dtype=np.float32,
)

X_train_mat = X_train_full
X_test_mat = X_test_full

y = train_f["scalar_coupling_constant"].astype("float32").to_numpy()
train_type = train_f["type"].astype("category")
test_type = test_f["type"].astype("category")


def make_model(alpha: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True)),
            (
                "sgd",
                SGDRegressor(
                    loss="huber",
                    epsilon=0.1,
                    penalty="l2",
                    alpha=alpha,
                    fit_intercept=True,
                    max_iter=2000,
                    tol=1e-3,
                    random_state=0,
                ),
            ),
        ]
    )


pred = np.zeros(len(test_f), dtype=np.float32)
global_mean = float(np.mean(y))

train_type_codes = train_type.cat.codes.to_numpy()
test_type_codes = test_type.cat.codes.to_numpy()
test_code_unique = np.unique(test_type_codes)

rng = np.random.RandomState(0)
alpha_grid = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4)

for code in test_code_unique:
    idx_test = np.flatnonzero(test_type_codes == code)
    idx_train_all = np.flatnonzero(train_type_codes == code)
    if len(idx_train_all) == 0:
        pred[idx_test] = global_mean
        continue

    y_t_all = y[idx_train_all]
    mu = float(np.mean(y_t_all))
    sigma = float(np.std(y_t_all) + 1e-6)

    n = len(idx_train_all)
    if n >= 5000:
        perm = rng.permutation(n)
        n_val = max(2000, int(0.05 * n))
        val_local = perm[:n_val]
        tr_local = perm[n_val:]
        idx_val = idx_train_all[val_local]
        idx_tr = idx_train_all[tr_local]

        y_tr = y[idx_tr]
        y_val = y[idx_val]
        y_tr_std = (y_tr - mu) / sigma

        best_alpha = None
        best_mae = np.inf
        X_tr = np.take(X_train_mat, idx_tr, axis=0)
        X_val = np.take(X_train_mat, idx_val, axis=0)
        for a in alpha_grid:
            model = make_model(alpha=a)
            model.fit(X_tr, y_tr_std)
            p_val_std = model.predict(X_val).astype(np.float32)
            p_val = p_val_std * sigma + mu
            mae = float(np.mean(np.abs(p_val - y_val)))
            if mae < best_mae:
                best_mae = mae
                best_alpha = a
        alpha_use = float(best_alpha)
    else:
        alpha_use = 1e-5  # stable default for small types

    y_t_std = (y_t_all - mu) / sigma
    model = make_model(alpha=alpha_use)
    model.fit(np.take(X_train_mat, idx_train_all, axis=0), y_t_std)

    p_std = model.predict(np.take(X_test_mat, idx_test, axis=0)).astype(np.float32)
    p = p_std * sigma + mu

    lo = float(np.quantile(y_t_all, 0.0001))
    hi = float(np.quantile(y_t_all, 0.9999))
    p = np.clip(p, lo, hi)

    pred[idx_test] = p.astype(np.float32)

pred = np.where(np.isfinite(pred), pred, global_mean).astype(np.float32)

submission = pd.DataFrame(
    {"id": test["id"].to_numpy(), "scalar_coupling_constant": pred}
)
submission["id"] = submission["id"].astype(np.int64)
submission = submission.sort_values("id").reset_index(drop=True)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(submission), "cols:", list(submission.columns))


## === cell 1
submission.head(20)


## === cell 2
submission.describe()
