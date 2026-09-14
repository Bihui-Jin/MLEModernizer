# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

-1.30873

# 6. Current score

1.99777

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.99777) has done: 'The current notebook fails because it tries to read external “blender” submission files that do not exist in this Kaggle environment, so nothing is produced. I replace that blending-only workflow with a self-contained baseline that trains per-coupling-type linear regression models using simple, standard geometric features from `structures.csv` (atom types and inter-atomic distance), then predicts for `test.csv`. This keeps the overall approach simple and stable, avoids heavy dependencies, and guarantees a correctly formatted `submission.csv` is written. The resulting score won’t be SOTA but should be valid and typically much better than a constant/empty submission, moving you toward the target by producing a meaningful model.'
- What this solution (achieved 1.21123) has done: 'I fix the NaN crash by adding a simple imputation step inside the existing sklearn preprocessing pipeline so Ridge never receives missing values. I also make the categorical columns consistent between train/test by using the same categories (avoids unseen-category edge cases) and add a small safety fill for any remaining coordinate nulls before distance computation. These are execution/stability fixes and should also improve score versus silently dropping/mean-filling whole types, without changing the core modeling approach (per-type Ridge on one-hot atoms + distance features). The script then run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 1.19831) has done: 'Your current pipeline is already stable and self-contained, but it’s underfitting because it only uses the pairwise distance and atom identities. To move the (lower-is-better) score closer to the target, I keep the exact same per-type Ridge setup and simply add a few standard, cheap geometric features derived from the same coordinates (absolute coordinate deltas, a dot product term, and axis-wise products), which usually reduces MAE without changing the learning approach. I also add `StandardScaler(with_mean=False)` on the numeric branch so Ridge is better conditioned while still working with sparse one-hot features. These are minimal changes that preserve the core logic (same data sources, same per-type training loop, same model family/loss) and should improve score toward your target.'
- What this solution (achieved 1.2015) has done: 'You’re already fairly close to the target (gap ≈ 1.19831 − (−1.30873) = 2.51; improving is needed since lower is better), so I keep the per-type Ridge pipeline intact and only add a few well-known CHAMPS baseline features that are cheap and consistently improve MAE: the inter-atomic distance powers (dist³, dist⁻³) and basic per-atom radial features (r0, r1 = distance to origin) plus their sum/diff. I also add the coupling `type` as a categorical feature inside each per-type model (it’s constant within a loop, so it won’t change anything; thus I won’t add it) and instead focus only on features that actually vary within type. These additions preserve your model family, training loop, and preprocessing semantics while usually moving the score downward (better) toward the target. The submission writing stays identical.'
- What this solution (achieved 1.21205) has done: 'Your current score (1.2015, lower-is-better) is far from the target (-1.30873), so we should legitimately improve the model while keeping the same per-type Ridge + one-hot atoms + geometry feature approach. The smallest high-impact improvement for this competition is to include “local chemical environment” information: for each endpoint atom, aggregate simple statistics (mean/min/max/std) of distances to other atoms in the same molecule by element (H/C/N/O/F). This preserves your core logic (same data sources, same per-type Ridge loop, same sklearn pipeline) but adds strong signal known to reduce MAE in CHAMPS. I also keep everything deterministic and ensure the submission format stays identical.'
- What this solution (achieved 1.99777) has done: 'Your current gap to the target is very large (lower-is-better, target is negative), so we should legitimately improve accuracy while keeping the same per-type Ridge + one-hot atoms + engineered geometry approach. The biggest low-risk gain within the same logic is to add stronger “local environment” signals without changing the model family: fast per-atom k-nearest-neighbor distance summaries (mean/min/max/std of the k closest atoms, plus counts within a radius), which are standard for CHAMPS and usually reduce MAE. I also keep your existing per-molecule environment-by-element features, but compute them in a much more scalable way (top-K neighbors) to fit in the 600s limit reliably. Finally, I keep the same preprocessing pipeline and submission writing, only extending `num_cols` with the new features.'

# 9. Code solution

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
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
structures = pd.read_csv(os.path.join(DATA_DIR, "structures.csv"))

print(train.shape, test.shape, structures.shape)
print(train.columns.tolist())
print(test.columns.tolist())
print(structures.columns.tolist())



## === cell 2
structures["atom"] = structures["atom"].astype("category")
train["type"] = train["type"].astype("category")
test["type"] = test["type"].astype("category")

for df in (train, test):
    df["atom_index_0"] = df["atom_index_0"].astype(np.int32)
    df["atom_index_1"] = df["atom_index_1"].astype(np.int32)

structures["atom_index"] = structures["atom_index"].astype(np.int32)
for c in ["x", "y", "z"]:
    structures[c] = pd.to_numeric(structures[c], errors="coerce").astype(np.float32)

s0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
    }
)
train_feat = train.merge(
    s0[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
)
test_feat = test.merge(
    s0[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
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
train_feat = train_feat.merge(
    s1[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
)
test_feat = test_feat.merge(
    s1[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
)

for col in ["atom_0", "atom_1"]:
    all_cats = pd.Index(
        pd.concat(
            [train_feat[col].astype("string"), test_feat[col].astype("string")], axis=0
        )
        .fillna("UNK")
        .unique()
    )
    if "UNK" not in all_cats:
        all_cats = all_cats.append(pd.Index(["UNK"]))
    dtype = pd.CategoricalDtype(categories=all_cats, ordered=False)
    train_feat[col] = train_feat[col].astype("string").fillna("UNK").astype(dtype)
    test_feat[col] = test_feat[col].astype("string").fillna("UNK").astype(dtype)

coord_cols = ["x0", "y0", "z0", "x1", "y1", "z1"]
for df in (train_feat, test_feat):
    for c in coord_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

for df in (train_feat, test_feat):
    dx = (df["x0"] - df["x1"]).astype("float32")
    dy = (df["y0"] - df["y1"]).astype("float32")
    dz = (df["z0"] - df["z1"]).astype("float32")

    adx = np.abs(dx).astype("float32")
    ady = np.abs(dy).astype("float32")
    adz = np.abs(dz).astype("float32")

    dist2 = (dx * dx + dy * dy + dz * dz).astype("float32")
    dist = np.sqrt(dist2).astype("float32")

    df["dx"] = dx
    df["dy"] = dy
    df["dz"] = dz
    df["adx"] = adx
    df["ady"] = ady
    df["adz"] = adz

    df["dist"] = dist
    df["dist2"] = dist2
    df["inv_dist"] = (1.0 / (dist + 1e-6)).astype("float32")
    df["inv_dist2"] = (1.0 / (dist2 + 1e-6)).astype("float32")

    df["pos_dot"] = (
        df["x0"] * df["x1"] + df["y0"] * df["y1"] + df["z0"] * df["z1"]
    ).astype("float32")
    df["dxdy"] = (dx * dy).astype("float32")
    df["dxdz"] = (dx * dz).astype("float32")
    df["dydz"] = (dy * dz).astype("float32")

    df["dist3"] = (dist * dist2).astype("float32")
    df["inv_dist3"] = (1.0 / (df["dist3"] + 1e-6)).astype("float32")

    r0 = np.sqrt(
        (df["x0"] * df["x0"] + df["y0"] * df["y0"] + df["z0"] * df["z0"]).astype(
            "float32"
        )
    ).astype("float32")
    r1 = np.sqrt(
        (df["x1"] * df["x1"] + df["y1"] * df["y1"] + df["z1"] * df["z1"]).astype(
            "float32"
        )
    ).astype("float32")
    df["r0"] = r0
    df["r1"] = r1
    df["r_sum"] = (r0 + r1).astype("float32")
    df["r_diff"] = (r0 - r1).astype("float32")

print(
    "Feature build done. Nulls in dist train/test:",
    int(train_feat["dist"].isna().sum()),
    int(test_feat["dist"].isna().sum()),
)

K_NEIGH = 8
RADII = (1.5, 2.0, 3.0)  # Angstrom radii for neighbor counts


def _knn_env_features_for_molecule(
    mol_df: pd.DataFrame, k: int = K_NEIGH
) -> pd.DataFrame:
    coords = mol_df[["x", "y", "z"]].to_numpy(np.float32, copy=False)
    n = coords.shape[0]
    out = mol_df[["molecule_name", "atom_index"]].copy()

    if n <= 1:
        for j in range(1, k + 1):
            out[f"knn_d{j}"] = np.float32(np.nan)
        out["knn_mean"] = np.float32(np.nan)
        out["knn_min"] = np.float32(np.nan)
        out["knn_max"] = np.float32(np.nan)
        out["knn_std"] = np.float32(np.nan)
        for r in RADII:
            out[f"nbr_cnt_r{str(r).replace('.','p')}"] = np.float32(np.nan)
        return out

    diff = coords[:, None, :] - coords[None, :, :]
    d2 = np.sum(diff * diff, axis=2, dtype=np.float32).astype(np.float32)
    np.fill_diagonal(d2, np.inf)

    k_eff = min(k, max(1, n - 1))
    idx = np.argpartition(d2, kth=k_eff - 1, axis=1)[:, :k_eff]
    knn_d2 = np.take_along_axis(d2, idx, axis=1)
    knn_d = np.sqrt(knn_d2).astype(np.float32)

    knn_d.sort(axis=1)
    for j in range(1, k + 1):
        col = f"knn_d{j}"
        if j <= k_eff:
            out[col] = knn_d[:, j - 1].astype(np.float32)
        else:
            out[col] = np.float32(np.nan)

    out["knn_mean"] = np.mean(knn_d, axis=1).astype(np.float32)
    out["knn_min"] = np.min(knn_d, axis=1).astype(np.float32)
    out["knn_max"] = np.max(knn_d, axis=1).astype(np.float32)
    out["knn_std"] = np.std(knn_d, axis=1).astype(np.float32)

    d = np.sqrt(d2).astype(np.float32)
    for r in RADII:
        out[f"nbr_cnt_r{str(r).replace('.','p')}"] = np.sum(
            d <= np.float32(r), axis=1
        ).astype(np.float32)

    return out


used_mols = pd.Index(
    pd.concat(
        [train_feat["molecule_name"], test_feat["molecule_name"]], axis=0
    ).unique()
)
structures_used = structures[structures["molecule_name"].isin(used_mols)].copy()

knn_parts = []
for mol_name, mol_df in structures_used.groupby("molecule_name", sort=False):
    knn_parts.append(_knn_env_features_for_molecule(mol_df, k=K_NEIGH))

knn_env = pd.concat(knn_parts, axis=0, ignore_index=True)

knn0 = knn_env.rename(
    columns={
        c: (c + "_0")
        for c in knn_env.columns
        if c not in ["molecule_name", "atom_index"]
    }
).rename(columns={"atom_index": "atom_index_0"})
knn1 = knn_env.rename(
    columns={
        c: (c + "_1")
        for c in knn_env.columns
        if c not in ["molecule_name", "atom_index"]
    }
).rename(columns={"atom_index": "atom_index_1"})

train_feat = train_feat.merge(knn0, on=["molecule_name", "atom_index_0"], how="left")
train_feat = train_feat.merge(knn1, on=["molecule_name", "atom_index_1"], how="left")
test_feat = test_feat.merge(knn0, on=["molecule_name", "atom_index_0"], how="left")
test_feat = test_feat.merge(knn1, on=["molecule_name", "atom_index_1"], how="left")

print("Added KNN env features. Train/test shapes:", train_feat.shape, test_feat.shape)

ENV_ELEMS = ["H", "C", "N", "O", "F"]
ENV_STATS = ["mean", "min", "max", "std"]
ENV_K = 16  # Change: cap neighbors used per element to keep runtime low while retaining signal.


def _env_features_for_molecule(mol_df: pd.DataFrame) -> pd.DataFrame:
    coords = mol_df[["x", "y", "z"]].to_numpy(np.float32, copy=False)
    atoms = mol_df["atom"].astype("string").to_numpy()
    n = coords.shape[0]
    out = mol_df[["molecule_name", "atom_index"]].copy()
    if n <= 1:
        for elem in ENV_ELEMS + ["ALL"]:
            for st in ENV_STATS:
                out[f"env_{elem}_{st}"] = np.float32(np.nan)
        return out

    diff = coords[:, None, :] - coords[None, :, :]
    d2 = np.sum(diff * diff, axis=2, dtype=np.float32).astype(np.float32)
    np.fill_diagonal(d2, np.inf)

    k_eff = min(ENV_K, max(1, n - 1))
    idx = np.argpartition(d2, kth=k_eff - 1, axis=1)[:, :k_eff]
    d_all = np.sqrt(np.take_along_axis(d2, idx, axis=1)).astype(np.float32)
    out["env_ALL_mean"] = np.mean(d_all, axis=1).astype(np.float32)
    out["env_ALL_min"] = np.min(d_all, axis=1).astype(np.float32)
    out["env_ALL_max"] = np.max(d_all, axis=1).astype(np.float32)
    out["env_ALL_std"] = np.std(d_all, axis=1).astype(np.float32)

    for elem in ENV_ELEMS:
        mask = atoms == elem
        if not mask.any():
            for st in ENV_STATS:
                out[f"env_{elem}_{st}"] = np.float32(np.nan)
            continue
        d2_e = d2[:, mask]
        m = d2_e.shape[1]
        if m == 0:
            for st in ENV_STATS:
                out[f"env_{elem}_{st}"] = np.float32(np.nan)
            continue
        k_e = min(ENV_K, m)
        idx_e = np.argpartition(d2_e, kth=k_e - 1, axis=1)[:, :k_e]
        de = np.sqrt(np.take_along_axis(d2_e, idx_e, axis=1)).astype(np.float32)

        out[f"env_{elem}_mean"] = np.mean(de, axis=1).astype(np.float32)
        out[f"env_{elem}_min"] = np.min(de, axis=1).astype(np.float32)
        out[f"env_{elem}_max"] = np.max(de, axis=1).astype(np.float32)
        out[f"env_{elem}_std"] = np.std(de, axis=1).astype(np.float32)

    return out


env_parts = []
for mol_name, mol_df in structures_used.groupby("molecule_name", sort=False):
    env_parts.append(_env_features_for_molecule(mol_df))

env = pd.concat(env_parts, axis=0, ignore_index=True)

env0 = env.rename(
    columns={
        c: (c + "_0") for c in env.columns if c not in ["molecule_name", "atom_index"]
    }
).rename(columns={"atom_index": "atom_index_0"})
env1 = env.rename(
    columns={
        c: (c + "_1") for c in env.columns if c not in ["molecule_name", "atom_index"]
    }
).rename(columns={"atom_index": "atom_index_1"})

train_feat = train_feat.merge(env0, on=["molecule_name", "atom_index_0"], how="left")
train_feat = train_feat.merge(env1, on=["molecule_name", "atom_index_1"], how="left")
test_feat = test_feat.merge(env0, on=["molecule_name", "atom_index_0"], how="left")
test_feat = test_feat.merge(env1, on=["molecule_name", "atom_index_1"], how="left")

print(
    "Added element env features. Train/test shapes:", train_feat.shape, test_feat.shape
)



## === cell 3
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer

num_cols = [
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

knn_cols = []
for side in ["0", "1"]:
    for j in range(1, K_NEIGH + 1):
        knn_cols.append(f"knn_d{j}_{side}")
    knn_cols += [
        f"knn_mean_{side}",
        f"knn_min_{side}",
        f"knn_max_{side}",
        f"knn_std_{side}",
    ]
    for r in RADII:
        knn_cols.append(f"nbr_cnt_r{str(r).replace('.','p')}_{side}")

env_cols = []
for elem in ["H", "C", "N", "O", "F", "ALL"]:
    for st in ["mean", "min", "max", "std"]:
        env_cols.append(f"env_{elem}_{st}_0")
        env_cols.append(f"env_{elem}_{st}_1")

num_cols = num_cols + knn_cols + env_cols
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
    trn_mask = (train_feat["type"] == t).values
    tst_mask = (test_feat["type"] == t).values

    X_tr = train_feat.loc[trn_mask, cat_cols + num_cols]
    y_tr = train_feat.loc[trn_mask, "scalar_coupling_constant"].values

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
        tst_mask = (test_feat["type"] == t).values
        pred_test[tst_mask] = global_mean

print("Pred stats:", pd.Series(pred_test).describe())



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/669216432.py in <cell line: 0>()
     93 
     94     model = Pipeline(steps=[("prep", preprocess), ("model", base_model)])
---> 95     model.fit(X_tr, y_tr)
     96     models[t] = model
     97 

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in fit(self, X, y, **fit_params)
    399         """
    400         fit_params_steps = self._check_fit_params(**fit_params)
--> 401         Xt = self._fit(X, y, **fit_params_steps)
    402         with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
    403             if self._final_estimator != "passthrough":

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in _fit(self, X, y, **fit_params_steps)
    357                 cloned_transformer = clone(transformer)
    358             # Fit or load from cache the current transformer
--> 359             X, fitted_transformer = fit_transform_one_cached(
    360                 cloned_transformer,
    361                 X,

/usr/local/lib/python3.11/dist-packages/joblib/memory.py in __call__(self, *args, **kwargs)
    324 
    325     def __call__(self, *args, **kwargs):
--> 326         return self.func(*args, **kwargs)
    327 
    328     def call_and_shelve(self, *args, **kwargs):

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in _fit_transform_one(transformer, X, y, weight, message_clsname, message, **fit_params)
    891     with _print_elapsed_time(message_clsname, message):
    892         if hasattr(transformer, "fit_transform"):
--> 893             res = transformer.fit_transform(X, y, **fit_params)
    894         else:
    895             res = transformer.fit(X, y, **fit_params).transform(X)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/compose/_column_transformer.py in fit_transform(self, X, y)
    725         self._validate_remainder(X)
    726 
--> 727         result = self._fit_transform(X, y, _fit_transform_one)
    728 
    729         if not result:

/usr/local/lib/python3.11/dist-packages/sklearn/compose/_column_transformer.py in _fit_transform(self, X, y, func, fitted, column_as_strings)
    656         )
    657         try:
--> 658             return Parallel(n_jobs=self.n_jobs)(
    659                 delayed(func)(
    660                     transformer=clone(trans) if not fitted else trans,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/parallel.py in __call__(self, iterable)
     61             for delayed_func, args, kwargs in iterable
     62         )
---> 63         return super().__call__(iterable_with_config)
     64 
     65 

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in __call__(self, iterable)
   1984             output = self._get_sequential_output(iterable)
   1985             next(output)
-> 1986             return output if self.return_generator else list(output)
   1987 
   1988         # Let's create an ID that uniquely identifies the current call. If the

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _get_sequential_output(self, iterable)
   1912                 self.n_dispatched_batches += 1
   1913                 self.n_dispatched_tasks += 1
-> 1914                 res = func(*args, **kwargs)
   1915                 self.n_completed_tasks += 1
   1916                 self.print_progress()

/usr/local/lib/python3.11/dist-packages/sklearn/utils/parallel.py in __call__(self, *args, **kwargs)
    121             config = {}
    122         with config_context(**config):
--> 123             return self.function(*args, **kwargs)

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in _fit_transform_one(transformer, X, y, weight, message_clsname, message, **fit_params)
    891     with _print_elapsed_time(message_clsname, message):
    892         if hasattr(transformer, "fit_transform"):
--> 893             res = transformer.fit_transform(X, y, **fit_params)
    894         else:
    895             res = transformer.fit(X, y, **fit_params).transform(X)

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in fit_transform(self, X, y, **fit_params)
    435         """
    436         fit_params_steps = self._check_fit_params(**fit_params)
--> 437         Xt = self._fit(X, y, **fit_params_steps)
    438 
    439         last_step = self._final_estimator

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in _fit(self, X, y, **fit_params_steps)
    357                 cloned_transformer = clone(transformer)
    358             # Fit or load from cache the current transformer
--> 359             X, fitted_transformer = fit_transform_one_cached(
    360                 cloned_transformer,
    361                 X,

/usr/local/lib/python3.11/dist-packages/joblib/memory.py in __call__(self, *args, **kwargs)
    324 
    325     def __call__(self, *args, **kwargs):
--> 326         return self.func(*args, **kwargs)
    327 
    328     def call_and_shelve(self, *args, **kwargs):

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in _fit_transform_one(transformer, X, y, weight, message_clsname, message, **fit_params)
    891     with _print_elapsed_time(message_clsname, message):
    892         if hasattr(transformer, "fit_transform"):
--> 893             res = transformer.fit_transform(X, y, **fit_params)
    894         else:
    895             res = transformer.fit(X, y, **fit_params).transform(X)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in fit_transform(self, X, y, **fit_params)
    879         else:
    880             # fit method of arity 2 (supervised transformation)
--> 881             return self.fit(X, y, **fit_params).transform(X)
    882 
    883 

/usr/local/lib/python3.11/dist-packages/sklearn/impute/_base.py in fit(self, X, y)
    388             )
    389 
--> 390         X = self._validate_input(X, in_fit=True)
    391 
    392         # default fill_value is 0 for numerical input and "missing_value"

/usr/local/lib/python3.11/dist-packages/sklearn/impute/_base.py in _validate_input(self, X, in_fit)
    342                 raise new_ve from None
    343             else:
--> 344                 raise ve
    345 
    346         if in_fit:

/usr/local/lib/python3.11/dist-packages/sklearn/impute/_base.py in _validate_input(self, X, in_fit)
    325 
    326         try:
--> 327             X = self._validate_data(
    328                 X,
    329                 reset=in_fit,

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    563             raise ValueError("Validation should be done on X, y or both.")
    564         elif not no_val_X and no_val_y:
--> 565             X = check_array(X, input_name="X", **check_params)
    566             out = X
    567         elif no_val_X and not no_val_y:

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    919 
    920         if force_all_finite:
--> 921             _assert_all_finite(
    922                 array,
    923                 input_name=input_name,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in _assert_all_finite(X, allow_nan, msg_dtype, estimator_name, input_name)
    159                 "#estimators-that-handle-nan-values"
    160             )
--> 161         raise ValueError(msg_err)
    162 
    163 

ValueError: Input X contains infinity or a value too large for dtype('float32').

## === cell 4
submission = pd.DataFrame(
    {"id": test["id"].values, "scalar_coupling_constant": pred_test}
)
assert submission.shape[0] == test.shape[0]
assert list(submission.columns) == ["id", "scalar_coupling_constant"]

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission.head())
