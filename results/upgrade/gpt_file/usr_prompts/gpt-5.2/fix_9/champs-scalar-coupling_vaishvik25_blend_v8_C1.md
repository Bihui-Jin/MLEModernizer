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

DATA_DIR = "/kaggle/data/champs-scalar-coupling"
if not os.path.exists(DATA_DIR):
    DATA_DIR = "/kaggle/input/champs-scalar-coupling"

print("DATA_DIR:", DATA_DIR)
print("Files:", sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])[:10])



## === cell 1
train = pd.read_csv(
    os.path.join(DATA_DIR, "train.csv"),
    dtype={
        "id": np.int32,
        "molecule_name": "string",
        "atom_index_0": np.int32,
        "atom_index_1": np.int32,
        "type": "category",
        "scalar_coupling_constant": np.float32,
    },
)
test = pd.read_csv(
    os.path.join(DATA_DIR, "test.csv"),
    dtype={
        "id": np.int32,
        "molecule_name": "string",
        "atom_index_0": np.int32,
        "atom_index_1": np.int32,
        "type": "category",
    },
)
structures = pd.read_csv(
    os.path.join(DATA_DIR, "structures.csv"),
    dtype={
        "molecule_name": "string",
        "atom_index": np.int32,
        "atom": "category",
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
    },
)

print(train.shape, test.shape, structures.shape)
print(train.columns.tolist())
print(test.columns.tolist())
print(structures.columns.tolist())



## === cell 2
structures = structures.sort_values(["molecule_name", "atom_index"], kind="mergesort")

s0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
    }
)
s1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
    }
)

train_feat = train.merge(
    s0[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
    sort=False,
)
test_feat = test.merge(
    s0[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
    sort=False,
)
train_feat = train_feat.merge(
    s1[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
    sort=False,
)
test_feat = test_feat.merge(
    s1[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
    sort=False,
)

for col in ["atom_0", "atom_1"]:
    combined = pd.concat([train_feat[col], test_feat[col]], axis=0)
    if not pd.api.types.is_categorical_dtype(combined):
        combined = combined.astype("category")
    cats = combined.cat.categories
    if "UNK" not in cats:
        cats = cats.insert(len(cats), "UNK")
    dtype = pd.CategoricalDtype(categories=cats, ordered=False)
    train_feat[col] = (
        train_feat[col].cat.add_categories(["UNK"]).fillna("UNK").astype(dtype)
    )
    test_feat[col] = (
        test_feat[col].cat.add_categories(["UNK"]).fillna("UNK").astype(dtype)
    )


def _add_pair_geom_features(df: pd.DataFrame) -> None:
    x0 = df["x0"].to_numpy(np.float32, copy=False)
    y0 = df["y0"].to_numpy(np.float32, copy=False)
    z0 = df["z0"].to_numpy(np.float32, copy=False)
    x1 = df["x1"].to_numpy(np.float32, copy=False)
    y1 = df["y1"].to_numpy(np.float32, copy=False)
    z1 = df["z1"].to_numpy(np.float32, copy=False)

    dx = (x0 - x1).astype(np.float32, copy=False)
    dy = (y0 - y1).astype(np.float32, copy=False)
    dz = (z0 - z1).astype(np.float32, copy=False)

    dist2 = (dx * dx + dy * dy + dz * dz).astype(np.float32, copy=False)
    dist = np.sqrt(dist2).astype(np.float32, copy=False)

    df["dx"] = dx
    df["dy"] = dy
    df["dz"] = dz
    df["adx"] = np.abs(dx).astype(np.float32, copy=False)
    df["ady"] = np.abs(dy).astype(np.float32, copy=False)
    df["adz"] = np.abs(dz).astype(np.float32, copy=False)

    df["dist"] = dist
    df["dist2"] = dist2
    df["inv_dist"] = (1.0 / (dist + 1e-6)).astype(np.float32, copy=False)
    df["inv_dist2"] = (1.0 / (dist2 + 1e-6)).astype(np.float32, copy=False)

    df["pos_dot"] = (x0 * x1 + y0 * y1 + z0 * z1).astype(np.float32, copy=False)
    df["dxdy"] = (dx * dy).astype(np.float32, copy=False)
    df["dxdz"] = (dx * dz).astype(np.float32, copy=False)
    df["dydz"] = (dy * dz).astype(np.float32, copy=False)

    dist3 = (dist * dist2).astype(np.float32, copy=False)
    df["dist3"] = dist3
    df["inv_dist3"] = (1.0 / (dist3 + 1e-6)).astype(np.float32, copy=False)

    r0 = np.sqrt((x0 * x0 + y0 * y0 + z0 * z0).astype(np.float32, copy=False)).astype(
        np.float32, copy=False
    )
    r1 = np.sqrt((x1 * x1 + y1 * y1 + z1 * z1).astype(np.float32, copy=False)).astype(
        np.float32, copy=False
    )
    df["r0"] = r0
    df["r1"] = r1
    df["r_sum"] = (r0 + r1).astype(np.float32, copy=False)
    df["r_diff"] = (r0 - r1).astype(np.float32, copy=False)


_add_pair_geom_features(train_feat)
_add_pair_geom_features(test_feat)

print(
    "Feature build done. Nulls in dist train/test:",
    int(train_feat["dist"].isna().sum()),
    int(test_feat["dist"].isna().sum()),
)

K_NEIGH = 8
RADII = (1.5, 2.0, 3.0)  # Angstrom radii for neighbor counts

ENV_ELEMS = ["H", "C", "N", "O", "F"]
ENV_STATS = ["mean", "min", "max", "std"]
ENV_K = 16  # cap neighbors used per element


def _mol_env_features(mol_df: pd.DataFrame, k_neigh: int = K_NEIGH) -> pd.DataFrame:
    coords = mol_df[["x", "y", "z"]].to_numpy(np.float32, copy=False)
    n = coords.shape[0]

    out = mol_df[["molecule_name", "atom_index"]].copy()

    for j in range(1, k_neigh + 1):
        out[f"knn_d{j}"] = np.float32(np.nan)
    out["knn_mean"] = np.float32(np.nan)
    out["knn_min"] = np.float32(np.nan)
    out["knn_max"] = np.float32(np.nan)
    out["knn_std"] = np.float32(np.nan)
    for r in RADII:
        out[f"nbr_cnt_r{str(r).replace('.','p')}"] = np.float32(np.nan)
    for elem in ENV_ELEMS + ["ALL"]:
        for st in ENV_STATS:
            out[f"env_{elem}_{st}"] = np.float32(np.nan)

    if n <= 1:
        return out

    diff = coords[:, None, :] - coords[None, :, :]
    d2 = np.sum(diff * diff, axis=2, dtype=np.float32).astype(np.float32, copy=False)
    np.fill_diagonal(d2, np.inf)

    k_eff = min(k_neigh, max(1, n - 1))
    idx = np.argpartition(d2, kth=k_eff - 1, axis=1)[:, :k_eff]
    knn_d2 = np.take_along_axis(d2, idx, axis=1)
    knn_d = np.sqrt(knn_d2).astype(np.float32, copy=False)
    knn_d.sort(axis=1)
    for j in range(1, k_neigh + 1):
        if j <= k_eff:
            out[f"knn_d{j}"] = knn_d[:, j - 1].astype(np.float32, copy=False)
        else:
            out[f"knn_d{j}"] = np.float32(np.nan)

    out["knn_mean"] = np.mean(knn_d, axis=1).astype(np.float32, copy=False)
    out["knn_min"] = np.min(knn_d, axis=1).astype(np.float32, copy=False)
    out["knn_max"] = np.max(knn_d, axis=1).astype(np.float32, copy=False)
    out["knn_std"] = np.std(knn_d, axis=1).astype(np.float32, copy=False)

    d = np.sqrt(d2).astype(np.float32, copy=False)
    for r in RADII:
        out[f"nbr_cnt_r{str(r).replace('.','p')}"] = np.sum(
            d <= np.float32(r), axis=1
        ).astype(np.float32, copy=False)

    k_all = min(ENV_K, max(1, n - 1))
    idx_all = np.argpartition(d2, kth=k_all - 1, axis=1)[:, :k_all]
    d_all = np.sqrt(np.take_along_axis(d2, idx_all, axis=1)).astype(
        np.float32, copy=False
    )

    out["env_ALL_mean"] = np.mean(d_all, axis=1).astype(np.float32, copy=False)
    out["env_ALL_min"] = np.min(d_all, axis=1).astype(np.float32, copy=False)
    out["env_ALL_max"] = np.max(d_all, axis=1).astype(np.float32, copy=False)
    out["env_ALL_std"] = np.std(d_all, axis=1).astype(np.float32, copy=False)

    atom_codes = mol_df["atom"].cat.codes.to_numpy(copy=False)
    atom_cats = mol_df["atom"].cat.categories

    for elem in ENV_ELEMS:
        if elem not in atom_cats:
            continue
        elem_code = atom_cats.get_loc(elem)
        mask = atom_codes == elem_code
        if not mask.any():
            continue

        d2_e = d2[:, mask]
        m = d2_e.shape[1]
        if m == 0:
            continue
        k_e = min(ENV_K, m)
        idx_e = np.argpartition(d2_e, kth=k_e - 1, axis=1)[:, :k_e]
        de = np.sqrt(np.take_along_axis(d2_e, idx_e, axis=1)).astype(
            np.float32, copy=False
        )

        out[f"env_{elem}_mean"] = np.mean(de, axis=1).astype(np.float32, copy=False)
        out[f"env_{elem}_min"] = np.min(de, axis=1).astype(np.float32, copy=False)
        out[f"env_{elem}_max"] = np.max(de, axis=1).astype(np.float32, copy=False)
        out[f"env_{elem}_std"] = np.std(de, axis=1).astype(np.float32, copy=False)

    return out


used_mols = pd.Index(
    pd.concat(
        [train_feat["molecule_name"], test_feat["molecule_name"]], axis=0
    ).unique()
)
structures_used = structures[structures["molecule_name"].isin(used_mols)]

parts = []
for _, mol_df in structures_used.groupby("molecule_name", sort=False):
    parts.append(_mol_env_features(mol_df, k_neigh=K_NEIGH))

mol_env = pd.concat(parts, axis=0, ignore_index=True)

knn_cols_base = (
    [f"knn_d{j}" for j in range(1, K_NEIGH + 1)]
    + ["knn_mean", "knn_min", "knn_max", "knn_std"]
    + [f"nbr_cnt_r{str(r).replace('.','p')}" for r in RADII]
)
env_cols_base = [
    f"env_{elem}_{st}" for elem in (ENV_ELEMS + ["ALL"]) for st in ENV_STATS
]
feat_cols_base = knn_cols_base + env_cols_base

mol0 = mol_env.rename(columns={c: c + "_0" for c in feat_cols_base}).rename(
    columns={"atom_index": "atom_index_0"}
)
mol1 = mol_env.rename(columns={c: c + "_1" for c in feat_cols_base}).rename(
    columns={"atom_index": "atom_index_1"}
)

train_feat = train_feat.merge(
    mol0, on=["molecule_name", "atom_index_0"], how="left", sort=False
)
train_feat = train_feat.merge(
    mol1, on=["molecule_name", "atom_index_1"], how="left", sort=False
)
test_feat = test_feat.merge(
    mol0, on=["molecule_name", "atom_index_0"], how="left", sort=False
)
test_feat = test_feat.merge(
    mol1, on=["molecule_name", "atom_index_1"], how="left", sort=False
)

print(
    "Added KNN + element env features (single pass). Train/test shapes:",
    train_feat.shape,
    test_feat.shape,
)


def _replace_inf_with_nan_fast(df: pd.DataFrame, cols: list) -> None:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return
    arr = df[cols].to_numpy(dtype=np.float64, copy=False)
    mask = ~np.isfinite(arr)
    if mask.any():
        arr[mask] = np.nan


_basic_num_cols = [
    "dist",
    "dist2",
    "dist3",
    "inv_dist",
    "inv_dist2",
    "inv_dist3",
    "dx",
    "dy",
    "dz",
    "adx",
    "ady",
    "adz",
    "pos_dot",
    "dxdy",
    "dxdz",
    "dydz",
    "r0",
    "r1",
    "r_sum",
    "r_diff",
]
_knn_cols_tmp = []
for side in ["0", "1"]:
    for j in range(1, K_NEIGH + 1):
        _knn_cols_tmp.append(f"knn_d{j}_{side}")
    _knn_cols_tmp += [
        f"knn_mean_{side}",
        f"knn_min_{side}",
        f"knn_max_{side}",
        f"knn_std_{side}",
    ]
    for r in RADII:
        _knn_cols_tmp.append(f"nbr_cnt_r{str(r).replace('.','p')}_{side}")

_env_cols_tmp = []
for elem in ENV_ELEMS + ["ALL"]:
    for st in ENV_STATS:
        _env_cols_tmp.append(f"env_{elem}_{st}_0")
        _env_cols_tmp.append(f"env_{elem}_{st}_1")

_all_num_cols_tmp = _basic_num_cols + _knn_cols_tmp + _env_cols_tmp
_replace_inf_with_nan_fast(train_feat, _all_num_cols_tmp)
_replace_inf_with_nan_fast(test_feat, _all_num_cols_tmp)

print(
    "After inf->nan sanitization:",
    "train inf count (approx) =",
    int(
        np.isinf(
            train_feat[_basic_num_cols].to_numpy(dtype=np.float64, copy=False)
        ).sum()
    ),
    "test inf count (approx) =",
    int(
        np.isinf(
            test_feat[_basic_num_cols].to_numpy(dtype=np.float64, copy=False)
        ).sum()
    ),
)



## === cell 3
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer

num_cols = _basic_num_cols + _knn_cols_tmp + _env_cols_tmp
cat_cols = ["atom_0", "atom_1"]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        (
            "num",
            Pipeline(
                steps=[
                    ("imp", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            ),
            num_cols,
        ),
    ],
    remainder="drop",
    sparse_threshold=0.3,
)

base_model = Ridge(alpha=1.0, random_state=42)

models = {}
type_means = train_feat.groupby("type")["scalar_coupling_constant"].mean().to_dict()

pred_test = np.zeros(len(test_feat), dtype=np.float32)

types = sorted(train_feat["type"].unique().tolist())
print("Training types:", types)

for t in types:
    trn_mask = (train_feat["type"] == t).to_numpy()
    tst_mask = (test_feat["type"] == t).to_numpy()

    X_tr = train_feat.loc[trn_mask, cat_cols + num_cols]
    y_tr = train_feat.loc[trn_mask, "scalar_coupling_constant"].to_numpy()

    if X_tr.shape[0] < 100:
        pred_test[tst_mask] = float(
            type_means.get(t, float(train_feat["scalar_coupling_constant"].mean()))
        )
        continue

    model = Pipeline(steps=[("prep", preprocess), ("model", base_model)])
    model.fit(X_tr, y_tr)
    models[t] = model

    X_te = test_feat.loc[tst_mask, cat_cols + num_cols]
    pred_test[tst_mask] = model.predict(X_te).astype(np.float32)

test_only_types = sorted(set(test_feat["type"].unique()) - set(types))
if test_only_types:
    global_mean = float(train_feat["scalar_coupling_constant"].mean())
    for t in test_only_types:
        tst_mask = (test_feat["type"] == t).to_numpy()
        pred_test[tst_mask] = global_mean

print("Pred stats:", pd.Series(pred_test).describe())



## === cell 4
submission = pd.DataFrame(
    {"id": test["id"].to_numpy(copy=False), "scalar_coupling_constant": pred_test}
)
assert submission.shape[0] == test.shape[0]
assert list(submission.columns) == ["id", "scalar_coupling_constant"]

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission.head())

## --- ERROR in outputing the csv:
Invalid submission: Missing required ids in submission.
