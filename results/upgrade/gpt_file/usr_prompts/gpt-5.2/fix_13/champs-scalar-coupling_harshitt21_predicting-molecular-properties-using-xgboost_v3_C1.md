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
lightgbm==4.6.0
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
xgboost==2.0.3

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

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

from xgboost import XGBRegressor

os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)

_CANDIDATES = [
    "/kaggle/data/champs-scalar-coupling",
    "/kaggle/input/champs-scalar-coupling",
    "/kaggle/input/champs-scalar-coupling-challenge",  # just in case (won't hurt)
    "/kaggle/data/input/champs-scalar-coupling",
]
DATA_DIR = None
for p in _CANDIDATES:
    if os.path.exists(p) and os.path.exists(os.path.join(p, "train.csv")):
        DATA_DIR = p
        break
if DATA_DIR is None:
    DATA_DIR = "/kaggle/data/champs-scalar-coupling"

print("Using DATA_DIR:", DATA_DIR)



## === cell 1
train = pd.read_csv(
    f"{DATA_DIR}/train.csv",
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
    f"{DATA_DIR}/test.csv",
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
    f"{DATA_DIR}/structures.csv",
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

mulliken = pd.read_csv(
    f"{DATA_DIR}/mulliken_charges.csv",
    usecols=["molecule_name", "atom_index", "mulliken_charge"],
    dtype={
        "molecule_name": "category",
        "atom_index": np.int16,
        "mulliken_charge": np.float32,
    },
)
mst = pd.read_csv(
    f"{DATA_DIR}/magnetic_shielding_tensors.csv",
    usecols=["molecule_name", "atom_index", "XX", "YY", "ZZ"],
    dtype={
        "molecule_name": "category",
        "atom_index": np.int16,
        "XX": np.float32,
        "YY": np.float32,
        "ZZ": np.float32,
    },
)
dipole = pd.read_csv(
    f"{DATA_DIR}/dipole_moments.csv",
    usecols=["molecule_name", "X", "Y", "Z"],
    dtype={
        "molecule_name": "category",
        "X": np.float32,
        "Y": np.float32,
        "Z": np.float32,
    },
)
potential = pd.read_csv(
    f"{DATA_DIR}/potential_energy.csv",
    usecols=["molecule_name", "potential_energy"],
    dtype={"molecule_name": "category", "potential_energy": np.float32},
)

print("Train shape:", train.shape)
print("Test shape:", test.shape)



## === cell 2
pass



## === cell 3
pass



## === cell 4
pass



## === cell 5
pass



## === cell 6
pass



## === cell 7
pass



## === cell 8
mol_atom_counts = (
    structures.groupby(["molecule_name", "atom"], sort=False)["atom_index"]
    .count()
    .unstack(fill_value=0)
    .reset_index()
)
mol_atom_counts.columns = [
    "molecule_name" if c == "molecule_name" else f"mol_atom_cnt_{c}"
    for c in mol_atom_counts.columns
]




## === cell 9
def add_knn_and_env_features_to_structures(struct_df, k=3, radii=(1.0, 2.0, 3.0)):
    from sklearn.neighbors import KDTree

    struct_df = struct_df[["molecule_name", "atom_index", "x", "y", "z"]].copy()
    struct_df["x"] = struct_df["x"].astype(np.float32)
    struct_df["y"] = struct_df["y"].astype(np.float32)
    struct_df["z"] = struct_df["z"].astype(np.float32)
    struct_df["atom_index"] = struct_df["atom_index"].astype(np.int16)

    out_frames = []
    for mol, g in struct_df.groupby("molecule_name", sort=False):
        coords = g[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=False)
        n = coords.shape[0]

        gf = g[["molecule_name", "atom_index"]].copy()

        if n <= 1:
            for i in range(k):
                gf[f"knn_dist_{i+1}"] = np.float32(0.0)
            gf["atom_inv_dist_sum"] = np.float32(0.0)
            gf["env_dist_min"] = np.float32(0.0)
            gf["env_dist_max"] = np.float32(0.0)
            gf["env_dist_mean"] = np.float32(0.0)
            for r in radii:
                gf[f"nbr_cnt_r{str(r).replace('.','p')}"] = np.int16(0)
            out_frames.append(gf)
            continue

        tree = KDTree(coords, leaf_size=40, metric="euclidean")

        kk = min(k + 1, n)
        dists, inds = tree.query(coords, k=kk, return_distance=True)
        dists = dists.astype(np.float32, copy=False)

        d_no_self = dists[:, 1:]  # shape (n, kk-1)
        if d_no_self.shape[1] < k:
            pad = np.full((n, k - d_no_self.shape[1]), np.inf, dtype=np.float32)
            d_no_self = np.concatenate([d_no_self, pad], axis=1)
        else:
            d_no_self = d_no_self[:, :k]

        for i in range(k):
            col = d_no_self[:, i]
            gf[f"knn_dist_{i+1}"] = np.where(np.isfinite(col), col, 0.0).astype(
                np.float32
            )

        for r in radii:
            ind_r = tree.query_radius(coords, r=r, return_distance=False)
            gf[f"nbr_cnt_r{str(r).replace('.','p')}"] = np.fromiter(
                (len(ix) - 1 for ix in ind_r), count=n, dtype=np.int16
            )

        ind_all, dist_all = tree.query_radius(
            coords, r=np.inf, return_distance=True, sort_results=False
        )

        env_min = np.empty(n, dtype=np.float32)
        env_max = np.empty(n, dtype=np.float32)
        env_mean = np.empty(n, dtype=np.float32)
        inv_sum = np.empty(n, dtype=np.float32)

        for i in range(n):
            di = dist_all[i].astype(np.float32, copy=False)
            if di.size <= 1:
                env_min[i] = 0.0
                env_max[i] = 0.0
                env_mean[i] = 0.0
                inv_sum[i] = 0.0
                continue
            mask = di > 0.0
            di = di[mask]
            if di.size == 0:
                env_min[i] = 0.0
                env_max[i] = 0.0
                env_mean[i] = 0.0
                inv_sum[i] = 0.0
            else:
                env_min[i] = float(di.min())
                env_max[i] = float(di.max())
                env_mean[i] = float(di.mean())
                inv_sum[i] = float(np.sum(1.0 / (di + 1e-6), dtype=np.float32))

        gf["atom_inv_dist_sum"] = inv_sum
        gf["env_dist_min"] = env_min
        gf["env_dist_max"] = env_max
        gf["env_dist_mean"] = env_mean

        out_frames.append(gf)

    feat = pd.concat(out_frames, axis=0, ignore_index=True)
    return feat


knn_feat = add_knn_and_env_features_to_structures(
    structures, k=3, radii=(1.0, 2.0, 3.0)
)



## === cell 10
mol_centroid = (
    structures.groupby("molecule_name", sort=False)[["x", "y", "z"]]
    .mean()
    .reset_index()
    .rename(columns={"x": "mol_x_mean", "y": "mol_y_mean", "z": "mol_z_mean"})
)
mol_centroid[["mol_x_mean", "mol_y_mean", "mol_z_mean"]] = mol_centroid[
    ["mol_x_mean", "mol_y_mean", "mol_z_mean"]
].astype(np.float32)



## === cell 11
structures_idx = structures.set_index(
    ["molecule_name", "atom_index"], drop=False
).sort_index()
mulliken_idx = mulliken.set_index(["molecule_name", "atom_index"]).sort_index()
mst_idx = mst.set_index(["molecule_name", "atom_index"]).sort_index()
knn_idx = knn_feat.set_index(["molecule_name", "atom_index"]).sort_index()
mol_atom_counts_idx = mol_atom_counts.set_index("molecule_name").sort_index()
mol_centroid_idx = mol_centroid.set_index("molecule_name").sort_index()
dipole_idx = dipole.set_index("molecule_name").sort_index()
potential_idx = potential.set_index("molecule_name").sort_index()


def _make_key(df, atom_col):
    mi = pd.MultiIndex.from_arrays(
        [df["molecule_name"].astype(str).to_numpy(), df[atom_col].to_numpy()],
        names=["molecule_name", "atom_index"],
    )
    return mi


train = train.copy()
key0 = _make_key(train, "atom_index_0")
key1 = _make_key(train, "atom_index_1")

s0 = structures_idx.reindex(key0)[["atom", "x", "y", "z"]].reset_index(drop=True)
s1 = structures_idx.reindex(key1)[["atom", "x", "y", "z"]].reset_index(drop=True)
s0.columns = ["atom_x", "x_x", "y_x", "z_x"]
s1.columns = ["atom_y", "x_y", "y_y", "z_y"]
train = pd.concat([train.reset_index(drop=True), s0, s1], axis=1)

q0 = (
    mulliken_idx.reindex(key0)[["mulliken_charge"]]
    .reset_index(drop=True)
    .rename(columns={"mulliken_charge": "mulliken_charge_0"})
)
q1 = (
    mulliken_idx.reindex(key1)[["mulliken_charge"]]
    .reset_index(drop=True)
    .rename(columns={"mulliken_charge": "mulliken_charge_1"})
)
train = pd.concat([train, q0, q1], axis=1)

m0 = (
    mst_idx.reindex(key0)[["XX", "YY", "ZZ"]]
    .reset_index(drop=True)
    .rename(columns={"XX": "mst_XX_0", "YY": "mst_YY_0", "ZZ": "mst_ZZ_0"})
)
m1 = (
    mst_idx.reindex(key1)[["XX", "YY", "ZZ"]]
    .reset_index(drop=True)
    .rename(columns={"XX": "mst_XX_1", "YY": "mst_YY_1", "ZZ": "mst_ZZ_1"})
)
train = pd.concat([train, m0, m1], axis=1)

d = (
    dipole_idx.reindex(train["molecule_name"].astype(str).to_numpy())[["X", "Y", "Z"]]
    .reset_index(drop=True)
    .rename(columns={"X": "dipole_X", "Y": "dipole_Y", "Z": "dipole_Z"})
)
pe = potential_idx.reindex(train["molecule_name"].astype(str).to_numpy())[
    ["potential_energy"]
].reset_index(drop=True)
train = pd.concat([train, d, pe], axis=1)

mac = mol_atom_counts_idx.reindex(
    train["molecule_name"].astype(str).to_numpy()
).reset_index(drop=True)
mc = mol_centroid_idx.reindex(train["molecule_name"].astype(str).to_numpy())[
    ["mol_x_mean", "mol_y_mean", "mol_z_mean"]
].reset_index(drop=True)
train = pd.concat([train, mac, mc], axis=1)

k0 = knn_idx.reindex(key0).reset_index(drop=True)
k1 = knn_idx.reindex(key1).reset_index(drop=True)

k0 = k0.rename(
    columns={
        "knn_dist_1": "knn_dist_1_0",
        "knn_dist_2": "knn_dist_2_0",
        "knn_dist_3": "knn_dist_3_0",
        "atom_inv_dist_sum": "atom_inv_dist_sum_0",
        "env_dist_min": "env_dist_min_0",
        "env_dist_max": "env_dist_max_0",
        "env_dist_mean": "env_dist_mean_0",
        "nbr_cnt_r1p0": "nbr_cnt_r1p0_0",
        "nbr_cnt_r2p0": "nbr_cnt_r2p0_0",
        "nbr_cnt_r3p0": "nbr_cnt_r3p0_0",
    }
)
k1 = k1.rename(
    columns={
        "knn_dist_1": "knn_dist_1_1",
        "knn_dist_2": "knn_dist_2_1",
        "knn_dist_3": "knn_dist_3_1",
        "atom_inv_dist_sum": "atom_inv_dist_sum_1",
        "env_dist_min": "env_dist_min_1",
        "env_dist_max": "env_dist_max_1",
        "env_dist_mean": "env_dist_mean_1",
        "nbr_cnt_r1p0": "nbr_cnt_r1p0_1",
        "nbr_cnt_r2p0": "nbr_cnt_r2p0_1",
        "nbr_cnt_r3p0": "nbr_cnt_r3p0_1",
    }
)
train = pd.concat([train, k0, k1], axis=1)



## === cell 12
test = test.copy()

key0 = _make_key(test, "atom_index_0")
key1 = _make_key(test, "atom_index_1")

s0 = structures_idx.reindex(key0)[["atom", "x", "y", "z"]].reset_index(drop=True)
s1 = structures_idx.reindex(key1)[["atom", "x", "y", "z"]].reset_index(drop=True)
s0.columns = ["atom_x", "x_x", "y_x", "z_x"]
s1.columns = ["atom_y", "x_y", "y_y", "z_y"]
test = pd.concat([test.reset_index(drop=True), s0, s1], axis=1)

q0 = (
    mulliken_idx.reindex(key0)[["mulliken_charge"]]
    .reset_index(drop=True)
    .rename(columns={"mulliken_charge": "mulliken_charge_0"})
)
q1 = (
    mulliken_idx.reindex(key1)[["mulliken_charge"]]
    .reset_index(drop=True)
    .rename(columns={"mulliken_charge": "mulliken_charge_1"})
)
test = pd.concat([test, q0, q1], axis=1)

m0 = (
    mst_idx.reindex(key0)[["XX", "YY", "ZZ"]]
    .reset_index(drop=True)
    .rename(columns={"XX": "mst_XX_0", "YY": "mst_YY_0", "ZZ": "mst_ZZ_0"})
)
m1 = (
    mst_idx.reindex(key1)[["XX", "YY", "ZZ"]]
    .reset_index(drop=True)
    .rename(columns={"XX": "mst_XX_1", "YY": "mst_YY_1", "ZZ": "mst_ZZ_1"})
)
test = pd.concat([test, m0, m1], axis=1)

d = (
    dipole_idx.reindex(test["molecule_name"].astype(str).to_numpy())[["X", "Y", "Z"]]
    .reset_index(drop=True)
    .rename(columns={"X": "dipole_X", "Y": "dipole_Y", "Z": "dipole_Z"})
)
pe = potential_idx.reindex(test["molecule_name"].astype(str).to_numpy())[
    ["potential_energy"]
].reset_index(drop=True)
test = pd.concat([test, d, pe], axis=1)

mac = mol_atom_counts_idx.reindex(
    test["molecule_name"].astype(str).to_numpy()
).reset_index(drop=True)
mc = mol_centroid_idx.reindex(test["molecule_name"].astype(str).to_numpy())[
    ["mol_x_mean", "mol_y_mean", "mol_z_mean"]
].reset_index(drop=True)
test = pd.concat([test, mac, mc], axis=1)

k0 = knn_idx.reindex(key0).reset_index(drop=True)
k1 = knn_idx.reindex(key1).reset_index(drop=True)
k0 = k0.rename(
    columns={
        "knn_dist_1": "knn_dist_1_0",
        "knn_dist_2": "knn_dist_2_0",
        "knn_dist_3": "knn_dist_3_0",
        "atom_inv_dist_sum": "atom_inv_dist_sum_0",
        "env_dist_min": "env_dist_min_0",
        "env_dist_max": "env_dist_max_0",
        "env_dist_mean": "env_dist_mean_0",
        "nbr_cnt_r1p0": "nbr_cnt_r1p0_0",
        "nbr_cnt_r2p0": "nbr_cnt_r2p0_0",
        "nbr_cnt_r3p0": "nbr_cnt_r3p0_0",
    }
)
k1 = k1.rename(
    columns={
        "knn_dist_1": "knn_dist_1_1",
        "knn_dist_2": "knn_dist_2_1",
        "knn_dist_3": "knn_dist_3_1",
        "atom_inv_dist_sum": "atom_inv_dist_sum_1",
        "env_dist_min": "env_dist_min_1",
        "env_dist_max": "env_dist_max_1",
        "env_dist_mean": "env_dist_mean_1",
        "nbr_cnt_r1p0": "nbr_cnt_r1p0_1",
        "nbr_cnt_r2p0": "nbr_cnt_r2p0_1",
        "nbr_cnt_r3p0": "nbr_cnt_r3p0_1",
    }
)
test = pd.concat([test, k0, k1], axis=1)



## === cell 13
pass



## === cell 14
x_x_tr = train["x_x"].to_numpy(np.float32, copy=False)
y_x_tr = train["y_x"].to_numpy(np.float32, copy=False)
z_x_tr = train["z_x"].to_numpy(np.float32, copy=False)
x_y_tr = train["x_y"].to_numpy(np.float32, copy=False)
y_y_tr = train["y_y"].to_numpy(np.float32, copy=False)
z_y_tr = train["z_y"].to_numpy(np.float32, copy=False)

x_x_te = test["x_x"].to_numpy(np.float32, copy=False)
y_x_te = test["y_x"].to_numpy(np.float32, copy=False)
z_x_te = test["z_x"].to_numpy(np.float32, copy=False)
x_y_te = test["x_y"].to_numpy(np.float32, copy=False)
y_y_te = test["y_y"].to_numpy(np.float32, copy=False)
z_y_te = test["z_y"].to_numpy(np.float32, copy=False)

dx_tr = x_y_tr - x_x_tr
dy_tr = y_y_tr - y_x_tr
dz_tr = z_y_tr - z_x_tr

dx_te = x_y_te - x_x_te
dy_te = y_y_te - y_x_te
dz_te = z_y_te - z_x_te

train["dx"] = dx_tr
train["dy"] = dy_tr
train["dz"] = dz_tr

test["dx"] = dx_te
test["dy"] = dy_te
test["dz"] = dz_te

train["adx"] = np.abs(dx_tr)
train["ady"] = np.abs(dy_tr)
train["adz"] = np.abs(dz_tr)

test["adx"] = np.abs(dx_te)
test["ady"] = np.abs(dy_te)
test["adz"] = np.abs(dz_te)

train["dx2"] = dx_tr * dx_tr
train["dy2"] = dy_tr * dy_tr
train["dz2"] = dz_tr * dz_tr

test["dx2"] = dx_te * dx_te
test["dy2"] = dy_te * dy_te
test["dz2"] = dz_te * dz_te

train["dist2"] = (
    train["dx2"].to_numpy(np.float32, copy=False)
    + train["dy2"].to_numpy(np.float32, copy=False)
    + train["dz2"].to_numpy(np.float32, copy=False)
)
test["dist2"] = (
    test["dx2"].to_numpy(np.float32, copy=False)
    + test["dy2"].to_numpy(np.float32, copy=False)
    + test["dz2"].to_numpy(np.float32, copy=False)
)

train["dist"] = np.sqrt(train["dist2"].to_numpy(np.float32, copy=False))
test["dist"] = np.sqrt(test["dist2"].to_numpy(np.float32, copy=False))

eps = 1e-9
train["inv_dist"] = 1.0 / (train["dist"].to_numpy(np.float32, copy=False) + eps)
test["inv_dist"] = 1.0 / (test["dist"].to_numpy(np.float32, copy=False) + eps)

train["log_dist"] = np.log(train["dist"].to_numpy(np.float32, copy=False) + 1e-6)
test["log_dist"] = np.log(test["dist"].to_numpy(np.float32, copy=False) + 1e-6)

train["dx_over_dist"] = train["dx"].to_numpy(np.float32, copy=False) / (
    train["dist"].to_numpy(np.float32, copy=False) + eps
)
train["dy_over_dist"] = train["dy"].to_numpy(np.float32, copy=False) / (
    train["dist"].to_numpy(np.float32, copy=False) + eps
)
train["dz_over_dist"] = train["dz"].to_numpy(np.float32, copy=False) / (
    train["dist"].to_numpy(np.float32, copy=False) + eps
)

test["dx_over_dist"] = test["dx"].to_numpy(np.float32, copy=False) / (
    test["dist"].to_numpy(np.float32, copy=False) + eps
)
test["dy_over_dist"] = test["dy"].to_numpy(np.float32, copy=False) / (
    test["dist"].to_numpy(np.float32, copy=False) + eps
)
test["dz_over_dist"] = test["dz"].to_numpy(np.float32, copy=False) / (
    test["dist"].to_numpy(np.float32, copy=False) + eps
)

for col0, col1, out_base in [
    ("mulliken_charge_0", "mulliken_charge_1", "dq"),
    ("mst_XX_0", "mst_XX_1", "d_mst_XX"),
    ("mst_YY_0", "mst_YY_1", "d_mst_YY"),
    ("mst_ZZ_0", "mst_ZZ_1", "d_mst_ZZ"),
]:
    train[out_base] = train[col1].to_numpy(np.float32, copy=False) - train[
        col0
    ].to_numpy(np.float32, copy=False)
    test[out_base] = test[col1].to_numpy(np.float32, copy=False) - test[col0].to_numpy(
        np.float32, copy=False
    )
    train["a" + out_base] = np.abs(train[out_base].to_numpy(np.float32, copy=False))
    test["a" + out_base] = np.abs(test[out_base].to_numpy(np.float32, copy=False))

train["dipole_mag"] = np.sqrt(
    train["dipole_X"].to_numpy(np.float32, copy=False) ** 2
    + train["dipole_Y"].to_numpy(np.float32, copy=False) ** 2
    + train["dipole_Z"].to_numpy(np.float32, copy=False) ** 2
)
test["dipole_mag"] = np.sqrt(
    test["dipole_X"].to_numpy(np.float32, copy=False) ** 2
    + test["dipole_Y"].to_numpy(np.float32, copy=False) ** 2
    + test["dipole_Z"].to_numpy(np.float32, copy=False) ** 2
)

for i in [1, 2, 3]:
    train[f"d_knn_dist_{i}"] = train[f"knn_dist_{i}_1"].to_numpy(
        np.float32, copy=False
    ) - train[f"knn_dist_{i}_0"].to_numpy(np.float32, copy=False)
    test[f"d_knn_dist_{i}"] = test[f"knn_dist_{i}_1"].to_numpy(
        np.float32, copy=False
    ) - test[f"knn_dist_{i}_0"].to_numpy(np.float32, copy=False)
    train[f"ad_knn_dist_{i}"] = np.abs(
        train[f"d_knn_dist_{i}"].to_numpy(np.float32, copy=False)
    )
    test[f"ad_knn_dist_{i}"] = np.abs(
        test[f"d_knn_dist_{i}"].to_numpy(np.float32, copy=False)
    )

train["d_atom_inv_dist_sum"] = train["atom_inv_dist_sum_1"].to_numpy(
    np.float32, copy=False
) - train["atom_inv_dist_sum_0"].to_numpy(np.float32, copy=False)
test["d_atom_inv_dist_sum"] = test["atom_inv_dist_sum_1"].to_numpy(
    np.float32, copy=False
) - test["atom_inv_dist_sum_0"].to_numpy(np.float32, copy=False)
train["ad_atom_inv_dist_sum"] = np.abs(
    train["d_atom_inv_dist_sum"].to_numpy(np.float32, copy=False)
)
test["ad_atom_inv_dist_sum"] = np.abs(
    test["d_atom_inv_dist_sum"].to_numpy(np.float32, copy=False)
)

train["x0_c"] = x_x_tr - train["mol_x_mean"].to_numpy(np.float32, copy=False)
train["y0_c"] = y_x_tr - train["mol_y_mean"].to_numpy(np.float32, copy=False)
train["z0_c"] = z_x_tr - train["mol_z_mean"].to_numpy(np.float32, copy=False)
train["x1_c"] = x_y_tr - train["mol_x_mean"].to_numpy(np.float32, copy=False)
train["y1_c"] = y_y_tr - train["mol_y_mean"].to_numpy(np.float32, copy=False)
train["z1_c"] = z_y_tr - train["mol_z_mean"].to_numpy(np.float32, copy=False)

test["x0_c"] = x_x_te - test["mol_x_mean"].to_numpy(np.float32, copy=False)
test["y0_c"] = y_x_te - test["mol_y_mean"].to_numpy(np.float32, copy=False)
test["z0_c"] = z_x_te - test["mol_z_mean"].to_numpy(np.float32, copy=False)
test["x1_c"] = x_y_te - test["mol_x_mean"].to_numpy(np.float32, copy=False)
test["y1_c"] = y_y_te - test["mol_y_mean"].to_numpy(np.float32, copy=False)
test["z1_c"] = z_y_te - test["mol_z_mean"].to_numpy(np.float32, copy=False)

train["r0_c"] = np.sqrt(
    train["x0_c"].to_numpy(np.float32, copy=False) ** 2
    + train["y0_c"].to_numpy(np.float32, copy=False) ** 2
    + train["z0_c"].to_numpy(np.float32, copy=False) ** 2
)
train["r1_c"] = np.sqrt(
    train["x1_c"].to_numpy(np.float32, copy=False) ** 2
    + train["y1_c"].to_numpy(np.float32, copy=False) ** 2
    + train["z1_c"].to_numpy(np.float32, copy=False) ** 2
)
test["r0_c"] = np.sqrt(
    test["x0_c"].to_numpy(np.float32, copy=False) ** 2
    + test["y0_c"].to_numpy(np.float32, copy=False) ** 2
    + test["z0_c"].to_numpy(np.float32, copy=False) ** 2
)
test["r1_c"] = np.sqrt(
    test["x1_c"].to_numpy(np.float32, copy=False) ** 2
    + test["y1_c"].to_numpy(np.float32, copy=False) ** 2
    + test["z1_c"].to_numpy(np.float32, copy=False) ** 2
)

train["dr_c"] = train["r1_c"].to_numpy(np.float32, copy=False) - train["r0_c"].to_numpy(
    np.float32, copy=False
)
train["adr_c"] = np.abs(train["dr_c"].to_numpy(np.float32, copy=False))
test["dr_c"] = test["r1_c"].to_numpy(np.float32, copy=False) - test["r0_c"].to_numpy(
    np.float32, copy=False
)
test["adr_c"] = np.abs(test["dr_c"].to_numpy(np.float32, copy=False))

for base in [
    "env_dist_min",
    "env_dist_max",
    "env_dist_mean",
    "nbr_cnt_r1p0",
    "nbr_cnt_r2p0",
    "nbr_cnt_r3p0",
]:
    c0 = f"{base}_0"
    c1 = f"{base}_1"
    out = f"d_{base}"
    aout = f"ad_{base}"
    if (c0 in train.columns) and (c1 in train.columns):
        train[out] = train[c1].to_numpy(np.float32, copy=False) - train[c0].to_numpy(
            np.float32, copy=False
        )
        test[out] = test[c1].to_numpy(np.float32, copy=False) - test[c0].to_numpy(
            np.float32, copy=False
        )
        train[aout] = np.abs(train[out].to_numpy(np.float32, copy=False))
        test[aout] = np.abs(test[out].to_numpy(np.float32, copy=False))



## === cell 15
pass



## === cell 16
pass



## === cell 17
for c in ["type", "atom_x", "atom_y"]:
    le = LabelEncoder()
    all_vals = pd.concat([train[c].astype(str), test[c].astype(str)], axis=0)
    le.fit(all_vals)
    train[c + "_le"] = le.transform(train[c].astype(str)).astype(np.int16, copy=False)
    test[c + "_le"] = le.transform(test[c].astype(str)).astype(np.int16, copy=False)

mol_count_cols = [c for c in mol_atom_counts.columns if c != "molecule_name"]

features = [
    "atom_index_0",
    "atom_index_1",
    "x_x",
    "y_x",
    "z_x",
    "x_y",
    "y_y",
    "z_y",
    "dx",
    "dy",
    "dz",
    "adx",
    "ady",
    "adz",
    "dx2",
    "dy2",
    "dz2",
    "dist",
    "dist2",
    "inv_dist",
    "log_dist",
    "dx_over_dist",
    "dy_over_dist",
    "dz_over_dist",
    "type_le",
    "atom_x_le",
    "atom_y_le",
    "mulliken_charge_0",
    "mulliken_charge_1",
    "mst_XX_0",
    "mst_YY_0",
    "mst_ZZ_0",
    "mst_XX_1",
    "mst_YY_1",
    "mst_ZZ_1",
    "dq",
    "adq",
    "d_mst_XX",
    "ad_mst_XX",
    "d_mst_YY",
    "ad_mst_YY",
    "d_mst_ZZ",
    "ad_mst_ZZ",
    "dipole_X",
    "dipole_Y",
    "dipole_Z",
    "dipole_mag",
    "potential_energy",
    "knn_dist_1_0",
    "knn_dist_2_0",
    "knn_dist_3_0",
    "atom_inv_dist_sum_0",
    "knn_dist_1_1",
    "knn_dist_2_1",
    "knn_dist_3_1",
    "atom_inv_dist_sum_1",
    "d_knn_dist_1",
    "ad_knn_dist_1",
    "d_knn_dist_2",
    "ad_knn_dist_2",
    "d_knn_dist_3",
    "ad_knn_dist_3",
    "d_atom_inv_dist_sum",
    "ad_atom_inv_dist_sum",
    "mol_x_mean",
    "mol_y_mean",
    "mol_z_mean",
    "x0_c",
    "y0_c",
    "z0_c",
    "x1_c",
    "y1_c",
    "z1_c",
    "r0_c",
    "r1_c",
    "dr_c",
    "adr_c",
    "env_dist_min_0",
    "env_dist_max_0",
    "env_dist_mean_0",
    "nbr_cnt_r1p0_0",
    "nbr_cnt_r2p0_0",
    "nbr_cnt_r3p0_0",
    "env_dist_min_1",
    "env_dist_max_1",
    "env_dist_mean_1",
    "nbr_cnt_r1p0_1",
    "nbr_cnt_r2p0_1",
    "nbr_cnt_r3p0_1",
    "d_env_dist_min",
    "ad_env_dist_min",
    "d_env_dist_max",
    "ad_env_dist_max",
    "d_env_dist_mean",
    "ad_env_dist_mean",
    "d_nbr_cnt_r1p0",
    "ad_nbr_cnt_r1p0",
    "d_nbr_cnt_r2p0",
    "ad_nbr_cnt_r2p0",
    "d_nbr_cnt_r3p0",
    "ad_nbr_cnt_r3p0",
] + mol_count_cols

for col in features:
    if col not in train.columns:
        train[col] = 0.0
    if col not in test.columns:
        test[col] = 0.0

train[features] = train[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
test[features] = test[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)



## === cell 18
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, va_idx = next(
    gss.split(
        train[features],
        train["scalar_coupling_constant"],
        groups=train["molecule_name"],
    )
)

train_tr = train.loc[tr_idx].copy()
train_va = train.loc[va_idx].copy()




## === cell 19
def champs_metric(
    df_true_pred, y_true_col="y_true", y_pred_col="y_pred", type_col="type"
):
    maes = df_true_pred.groupby(type_col).apply(
        lambda g: metrics.mean_absolute_error(g[y_true_col], g[y_pred_col])
    )
    return float(np.mean(np.log(maes)))


type_values = sorted(train["type"].astype(str).unique())
models_by_type = {}
val_preds_all = np.full(len(train_va), np.nan, dtype=np.float32)

base_params = dict(
    n_estimators=700,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="reg:squarederror",
    n_jobs=8,
    random_state=42,
)

va_type_arr = train_va["type"].astype(str).to_numpy()
for t in type_values:
    tr_t = train_tr[train_tr["type"].astype(str) == t]
    va_t = train_va[train_va["type"].astype(str) == t]

    if len(tr_t) == 0 or len(va_t) == 0:
        continue

    feat_t = features

    y_train = tr_t["scalar_coupling_constant"].astype(np.float32).values
    X_train = tr_t[feat_t]
    X_val = va_t[feat_t]

    m = XGBRegressor(**base_params)
    m.fit(X_train, y_train)
    models_by_type[t] = m

    pred = m.predict(X_val).astype(np.float32)
    val_preds_all[va_type_arr == t] = pred

missing_mask = np.isnan(val_preds_all)
if missing_mask.any():
    xgb_fallback = XGBRegressor(**base_params)
    xgb_fallback.fit(
        train_tr[features],
        train_tr["scalar_coupling_constant"].astype(np.float32).values,
    )
    val_preds_all[missing_mask] = xgb_fallback.predict(
        train_va.loc[missing_mask, features]
    ).astype(np.float32)
else:
    xgb_fallback = None

val_df = pd.DataFrame(
    {
        "type": train_va["type"].astype(str).values,
        "y_true": train_va["scalar_coupling_constant"].values,
        "y_pred": val_preds_all,
    }
)

val_mae_pooled = metrics.mean_absolute_error(val_df["y_true"], val_df["y_pred"])
val_metric = champs_metric(val_df)

print("Validation pooled MAE:", val_mae_pooled)
print("Validation pooled log(MAE):", np.log(val_mae_pooled))
print("Validation CHAMPS metric (mean over types of log(MAE_type)):", val_metric)



## === cell 20
test_predictions = np.full(len(test), np.nan, dtype=np.float32)
test_type_arr = test["type"].astype(str).to_numpy()

for t, m in models_by_type.items():
    mask = test_type_arr == t
    if mask.any():
        test_predictions[mask] = m.predict(test.loc[mask, features]).astype(np.float32)

missing = np.isnan(test_predictions)
if missing.any():
    if xgb_fallback is None:
        xgb_fallback = XGBRegressor(**base_params)
        xgb_fallback.fit(
            train_tr[features],
            train_tr["scalar_coupling_constant"].astype(np.float32).values,
        )
    test_predictions[missing] = xgb_fallback.predict(
        test.loc[missing, features]
    ).astype(np.float32)



## === cell 21
pass



## === cell 22
submission = pd.DataFrame(
    {"id": test["id"].values, "scalar_coupling_constant": test_predictions}
)

submission = submission.sort_values("id").reset_index(drop=True)
submission["scalar_coupling_constant"] = submission["scalar_coupling_constant"].astype(
    np.float32
)
submission["scalar_coupling_constant"] = (
    submission["scalar_coupling_constant"]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)

submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Submission id unique:", submission["id"].is_unique)
print("Submission id count matches test:", len(submission) == len(test))
