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

-1.30873

# 6. Current score

1.23566

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.99777) has done: 'The current notebook fails because it tries to read external “blender” submission files that do not exist in this Kaggle environment, so nothing is produced. I replace that blending-only workflow with a self-contained baseline that trains per-coupling-type linear regression models using simple, standard geometric features from `structures.csv` (atom types and inter-atomic distance), then predicts for `test.csv`. This keeps the overall approach simple and stable, avoids heavy dependencies, and guarantees a correctly formatted `submission.csv` is written. The resulting score won’t be SOTA but should be valid and typically much better than a constant/empty submission, moving you toward the target by producing a meaningful model.'
- What this solution (achieved 1.21123) has done: 'I fix the NaN crash by adding a simple imputation step inside the existing sklearn preprocessing pipeline so Ridge never receives missing values. I also make the categorical columns consistent between train/test by using the same categories (avoids unseen-category edge cases) and add a small safety fill for any remaining coordinate nulls before distance computation. These are execution/stability fixes and should also improve score versus silently dropping/mean-filling whole types, without changing the core modeling approach (per-type Ridge on one-hot atoms + distance features). The script then run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 1.19831) has done: 'Your current pipeline is already stable and self-contained, but it’s underfitting because it only uses the pairwise distance and atom identities. To move the (lower-is-better) score closer to the target, I keep the exact same per-type Ridge setup and simply add a few standard, cheap geometric features derived from the same coordinates (absolute coordinate deltas, a dot product term, and axis-wise products), which usually reduces MAE without changing the learning approach. I also add `StandardScaler(with_mean=False)` on the numeric branch so Ridge is better conditioned while still working with sparse one-hot features. These are minimal changes that preserve the core logic (same data sources, same per-type training loop, same model family/loss) and should improve score toward your target.'
- What this solution (achieved 1.2015) has done: 'You’re already fairly close to the target (gap ≈ 1.19831 − (−1.30873) = 2.51; improving is needed since lower is better), so I keep the per-type Ridge pipeline intact and only add a few well-known CHAMPS baseline features that are cheap and consistently improve MAE: the inter-atomic distance powers (dist³, dist⁻³) and basic per-atom radial features (r0, r1 = distance to origin) plus their sum/diff. I also add the coupling `type` as a categorical feature inside each per-type model (it’s constant within a loop, so it won’t change anything; thus I won’t add it) and instead focus only on features that actually vary within type. These additions preserve your model family, training loop, and preprocessing semantics while usually moving the score downward (better) toward the target. The submission writing stays identical.'
- What this solution (achieved 1.21205) has done: 'Your current score (1.2015, lower-is-better) is far from the target (-1.30873), so we should legitimately improve the model while keeping the same per-type Ridge + one-hot atoms + geometry feature approach. The smallest high-impact improvement for this competition is to include “local chemical environment” information: for each endpoint atom, aggregate simple statistics (mean/min/max/std) of distances to other atoms in the same molecule by element (H/C/N/O/F). This preserves your core logic (same data sources, same per-type Ridge loop, same sklearn pipeline) but adds strong signal known to reduce MAE in CHAMPS. I also keep everything deterministic and ensure the submission format stays identical.'
- What this solution (achieved 1.99777) has done: 'Your current gap to the target is very large (lower-is-better, target is negative), so we should legitimately improve accuracy while keeping the same per-type Ridge + one-hot atoms + engineered geometry approach. The biggest low-risk gain within the same logic is to add stronger “local environment” signals without changing the model family: fast per-atom k-nearest-neighbor distance summaries (mean/min/max/std of the k closest atoms, plus counts within a radius), which are standard for CHAMPS and usually reduce MAE. I also keep your existing per-molecule environment-by-element features, but compute them in a much more scalable way (top-K neighbors) to fit in the 600s limit reliably. Finally, I keep the same preprocessing pipeline and submission writing, only extending `num_cols` with the new features.'
- What this solution (achieved 1.99777) has done: 'The timeout is dominated by the per-molecule environment feature builder, which currently does an \(O(n^2)\) full distance matrix for every molecule and then iterates in Python over ~76k molecules; we keep the exact same features but compute them with a faster, vectorized blockwise distance approach and avoid constructing the full \(n \times n\) matrix. We also avoid repeated pandas groupby overhead by iterating over pre-sorted contiguous molecule slices, and we preallocate arrays for outputs to reduce per-molecule DataFrame operations while preserving identical semantics. The submission error is due to using `test["id"]` after building `test_feat` (which can be reordered by merges); we keep paths unchanged and fix correctness by using `test_feat["id"]` (aligned to `pred_test`) and then sorting by id to match Kaggle requirements. Everything else (model, preprocessing, training per type, and the exact set of features) is preserved.'
- What this solution (achieved 1.26227) has done: 'The main timeout comes from the per-molecule O(n²) distance-matrix computation in `_mol_env_features_fast` (including building full `d`), repeated for ~76k molecules, plus expensive pandas merges and DataFrame construction inside tight loops. I keep the exact same features and model logic, but make the environment-feature computation allocation-light and avoid constructing full `sqrt(d2)` matrices: compute radius neighbor counts directly from `d2` thresholds and compute KNN/env distances via selected indices only. I also reduce overhead by predefining output arrays once per molecule (instead of growing a DataFrame with many columns), and use `copy=False`/numpy views aggressively while preserving identical numerical results (within float32 rounding). The Ridge + preprocessing pipeline stays the same; only data preparation is sped up.'
- What this solution (achieved 1.25476) has done: 'We keep your exact per-type Ridge + one-hot atoms + engineered geometry + per-atom environment feature logic, and make only two score-relevant tweaks that tend to reduce MAE without changing the modeling approach. First, we add `atom_0`/`atom_1` self-inclusion handling in the element-specific environment features (exclude the atom itself when the queried element matches), because including distance=0 for same-element masks biases `min/mean/std` and hurts prediction quality. Second, we stabilize and slightly improve Ridge fitting by using a per-type `alpha` (still Ridge) based on sample count (light regularization for small types, default for large types), which typically lowers logMAE a bit without altering the training loop semantics. Everything else stays the same and we still write a valid `submission.csv`.'
- What this solution (achieved 1.25715) has done: 'We’re far from the (lower-is-better) target, so we should legitimately improve accuracy while keeping your exact per-type Ridge + one-hot atoms + engineered geometry + per-atom environment feature approach unchanged. The biggest low-risk gain with minimal code change is to add a tiny set of proven global molecule descriptors (dipole moment vector and potential energy) and merge them into both train/test; this often reduces logMAE across types without altering the training loop or model family. We also add one stable interaction feature (cosine of the angle between the two position vectors) derived from existing coordinates, which tends to help but doesn’t change the core logic. Everything else (feature builder, preprocessing with imputation+scaling, per-type Ridge training, and submission writing) stays the same.'
- What this solution (achieved 1.20308) has done: 'Your current score (1.25715, lower-is-better) is still far from the negative target, so we should make a small, legitimate accuracy improvement while keeping the same per-type Ridge + one-hot atoms + engineered geometry approach. The most impactful minimal change here is to add two widely-used CHAMPS auxiliary per-atom signal files (`mulliken_charges.csv` and `magnetic_shielding_tensors.csv`) and merge their per-atom values for atom_0/atom_1 into the existing feature table; this preserves the same training loop/model family and typically reduces MAE. To keep runtime under control, we only add a few stable tensor summaries (trace, Frobenius norm, diagonal mean) rather than all 9 tensor entries. Everything else (feature computation, preprocessing with imputation+scaling, per-type Ridge fitting, and submission writing) stays the same.'
- What this solution (achieved 1.23566) has done: 'To move your (lower-is-better) score down toward the target while keeping the same per-type Ridge + engineered-geometry approach, I make two minimal, high-signal feature additions that don’t change the training loop or model family. First, I add the per-pair coupling contribution components (fc/sd/pso/dso) by merging `scalar_coupling_contributions.csv` for train only and training Ridge to predict the residual `scalar_coupling_constant - (fc+sd+pso+dso)`; at inference we add back a small per-type mean contribution baseline, which is a standard, legitimate way to leverage this file without leaking test labels. Second, I add the raw MST components (9 values) per endpoint atom (in addition to your summaries), because they typically reduce MAE with a linear model and are a small incremental change. Everything else (feature computation, preprocessing, per-type Ridge fitting, and submission writing) remains the same.'

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

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(42)



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

s0_small = s0[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]]
s1_small = s1[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]]

train_feat = train.merge(
    s0_small,
    on=["molecule_name", "atom_index_0"],
    how="left",
    sort=False,
)
test_feat = test.merge(
    s0_small,
    on=["molecule_name", "atom_index_0"],
    how="left",
    sort=False,
)
train_feat = train_feat.merge(
    s1_small,
    on=["molecule_name", "atom_index_1"],
    how="left",
    sort=False,
)
test_feat = test_feat.merge(
    s1_small,
    on=["molecule_name", "atom_index_1"],
    how="left",
    sort=False,
)

for col in ["atom_0", "atom_1"]:
    tr = train_feat[col]
    te = test_feat[col]
    if not pd.api.types.is_categorical_dtype(tr):
        tr = tr.astype("category")
    if not pd.api.types.is_categorical_dtype(te):
        te = te.astype("category")
    cats = tr.cat.categories.union(te.cat.categories)
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

    denom = (r0 * r1 + np.float32(1e-6)).astype(np.float32, copy=False)
    df["cos01"] = (df["pos_dot"].to_numpy(np.float32, copy=False) / denom).astype(
        np.float32, copy=False
    )


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


def _mol_env_features_fast(
    mol_df: pd.DataFrame, k_neigh: int = K_NEIGH
) -> pd.DataFrame:
    coords = mol_df[["x", "y", "z"]].to_numpy(np.float32, copy=False)
    n = coords.shape[0]

    k_eff = min(k_neigh, n - 1) if n > 1 else 0
    k_all = min(ENV_K, n - 1) if n > 1 else 0

    out_dict = {
        "molecule_name": mol_df["molecule_name"].to_numpy(copy=False),
        "atom_index": mol_df["atom_index"].to_numpy(copy=False),
    }

    knn_d = np.full((n, k_neigh), np.nan, dtype=np.float32)
    out_dict.update({f"knn_d{j}": knn_d[:, j - 1] for j in range(1, k_neigh + 1)})
    out_dict["knn_mean"] = np.full(n, np.nan, dtype=np.float32)
    out_dict["knn_min"] = np.full(n, np.nan, dtype=np.float32)
    out_dict["knn_max"] = np.full(n, np.nan, dtype=np.float32)
    out_dict["knn_std"] = np.full(n, np.nan, dtype=np.float32)

    for r in RADII:
        out_dict[f"nbr_cnt_r{str(r).replace('.','p')}"] = np.full(
            n, np.nan, dtype=np.float32
        )

    for elem in ENV_ELEMS + ["ALL"]:
        for st in ENV_STATS:
            out_dict[f"env_{elem}_{st}"] = np.full(n, np.nan, dtype=np.float32)

    if n <= 1:
        return pd.DataFrame(out_dict)

    atom_codes = mol_df["atom"].cat.codes.to_numpy(copy=False)
    atom_cats = mol_df["atom"].cat.categories
    elem_code_map = {
        elem: atom_cats.get_loc(elem) for elem in ENV_ELEMS if elem in atom_cats
    }

    x2 = np.sum(coords * coords, axis=1, dtype=np.float32)
    G = coords @ coords.T  # float32
    d2 = (x2[:, None] + x2[None, :] - np.float32(2.0) * G).astype(
        np.float32, copy=False
    )
    np.fill_diagonal(d2, np.inf)

    idx_knn = np.argpartition(d2, kth=k_eff - 1, axis=1)[:, :k_eff]
    knn_sel = np.take_along_axis(d2, idx_knn, axis=1)
    knn_val = np.sqrt(knn_sel, dtype=np.float32)
    knn_val.sort(axis=1)
    knn_d[:, :k_eff] = knn_val  # remaining stay nan

    out_dict["knn_mean"][:] = np.mean(knn_val, axis=1, dtype=np.float32)
    out_dict["knn_min"][:] = np.min(knn_val, axis=1)
    out_dict["knn_max"][:] = np.max(knn_val, axis=1)
    out_dict["knn_std"][:] = np.std(knn_val, axis=1, dtype=np.float32)

    for r in RADII:
        thr = np.float32(r * r)
        out_dict[f"nbr_cnt_r{str(r).replace('.','p')}"][:] = np.sum(
            d2 <= thr, axis=1
        ).astype(np.float32, copy=False)

    idx_all = np.argpartition(d2, kth=k_all - 1, axis=1)[:, :k_all]
    env_all = np.sqrt(np.take_along_axis(d2, idx_all, axis=1), dtype=np.float32)
    out_dict["env_ALL_mean"][:] = np.mean(env_all, axis=1, dtype=np.float32)
    out_dict["env_ALL_min"][:] = np.min(env_all, axis=1)
    out_dict["env_ALL_max"][:] = np.max(env_all, axis=1)
    out_dict["env_ALL_std"][:] = np.std(env_all, axis=1, dtype=np.float32)

    for elem, code in elem_code_map.items():
        mask = atom_codes == code
        m = int(mask.sum())
        if m == 0:
            continue
        k_e = min(ENV_K, m - 1) if m > 1 else 0
        if k_e <= 0:
            continue

        d2_e = d2[:, mask].copy()
        idxs_elem = np.flatnonzero(mask)
        for j, i in enumerate(idxs_elem):
            d2_e[i, j] = np.inf

        idx_e = np.argpartition(d2_e, kth=k_e - 1, axis=1)[:, :k_e]
        env_e = np.sqrt(np.take_along_axis(d2_e, idx_e, axis=1), dtype=np.float32)

        out_dict[f"env_{elem}_mean"][:] = np.mean(env_e, axis=1, dtype=np.float32)
        out_dict[f"env_{elem}_min"][:] = np.min(env_e, axis=1)
        out_dict[f"env_{elem}_max"][:] = np.max(env_e, axis=1)
        out_dict[f"env_{elem}_std"][:] = np.std(env_e, axis=1, dtype=np.float32)

    return pd.DataFrame(out_dict)


used_mols = pd.Index(
    pd.concat(
        [train_feat["molecule_name"], test_feat["molecule_name"]], axis=0
    ).unique()
)

structures_used = (
    structures[structures["molecule_name"].isin(used_mols)]
    .sort_values(["molecule_name", "atom_index"], kind="mergesort")
    .reset_index(drop=True)
)

mol_names = structures_used["molecule_name"].to_numpy()
change = np.empty(len(structures_used), dtype=bool)
change[0] = True
change[1:] = mol_names[1:] != mol_names[:-1]
starts = np.flatnonzero(change)
ends = np.append(starts[1:], len(structures_used))

parts = []
for s, e in zip(starts, ends):
    parts.append(_mol_env_features_fast(structures_used.iloc[s:e], k_neigh=K_NEIGH))

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

dipole = pd.read_csv(
    os.path.join(DATA_DIR, "dipole_moments.csv"),
    dtype={
        "molecule_name": "string",
        "X": np.float32,
        "Y": np.float32,
        "Z": np.float32,
    },
)
potential = pd.read_csv(
    os.path.join(DATA_DIR, "potential_energy.csv"),
    dtype={"molecule_name": "string", "potential_energy": np.float32},
)

dipole["dipole_norm"] = np.sqrt(
    (
        dipole["X"] * dipole["X"]
        + dipole["Y"] * dipole["Y"]
        + dipole["Z"] * dipole["Z"]
    ).astype(np.float32)
).astype(np.float32)

mol_global = dipole.merge(potential, on="molecule_name", how="left", sort=False)

train_feat = train_feat.merge(mol_global, on="molecule_name", how="left", sort=False)
test_feat = test_feat.merge(mol_global, on="molecule_name", how="left", sort=False)

mulliken = pd.read_csv(
    os.path.join(DATA_DIR, "mulliken_charges.csv"),
    dtype={
        "molecule_name": "string",
        "atom_index": np.int32,
        "mulliken_charge": np.float32,
    },
)
m0 = mulliken.rename(
    columns={"atom_index": "atom_index_0", "mulliken_charge": "mulliken_charge_0"}
)
m1 = mulliken.rename(
    columns={"atom_index": "atom_index_1", "mulliken_charge": "mulliken_charge_1"}
)
train_feat = train_feat.merge(
    m0, on=["molecule_name", "atom_index_0"], how="left", sort=False
)
train_feat = train_feat.merge(
    m1, on=["molecule_name", "atom_index_1"], how="left", sort=False
)
test_feat = test_feat.merge(
    m0, on=["molecule_name", "atom_index_0"], how="left", sort=False
)
test_feat = test_feat.merge(
    m1, on=["molecule_name", "atom_index_1"], how="left", sort=False
)

mst = pd.read_csv(
    os.path.join(DATA_DIR, "magnetic_shielding_tensors.csv"),
    dtype={
        "molecule_name": "string",
        "atom_index": np.int32,
        "XX": np.float32,
        "YX": np.float32,
        "ZX": np.float32,
        "XY": np.float32,
        "YY": np.float32,
        "ZY": np.float32,
        "XZ": np.float32,
        "YZ": np.float32,
        "ZZ": np.float32,
    },
)

mst["mst_trace"] = (mst["XX"] + mst["YY"] + mst["ZZ"]).astype(np.float32)
mst["mst_diag_mean"] = (mst["mst_trace"] / np.float32(3.0)).astype(np.float32)
mst["mst_frob"] = np.sqrt(
    (
        mst["XX"] * mst["XX"]
        + mst["YX"] * mst["YX"]
        + mst["ZX"] * mst["ZX"]
        + mst["XY"] * mst["XY"]
        + mst["YY"] * mst["YY"]
        + mst["ZY"] * mst["ZY"]
        + mst["XZ"] * mst["XZ"]
        + mst["YZ"] * mst["YZ"]
        + mst["ZZ"] * mst["ZZ"]
    ).astype(np.float32)
).astype(np.float32)

mst_small = mst[
    [
        "molecule_name",
        "atom_index",
        "XX",
        "YX",
        "ZX",
        "XY",
        "YY",
        "ZY",
        "XZ",
        "YZ",
        "ZZ",
        "mst_trace",
        "mst_diag_mean",
        "mst_frob",
    ]
]

mst0 = mst_small.rename(
    columns={
        "atom_index": "atom_index_0",
        "XX": "mst_XX_0",
        "YX": "mst_YX_0",
        "ZX": "mst_ZX_0",
        "XY": "mst_XY_0",
        "YY": "mst_YY_0",
        "ZY": "mst_ZY_0",
        "XZ": "mst_XZ_0",
        "YZ": "mst_YZ_0",
        "ZZ": "mst_ZZ_0",
        "mst_trace": "mst_trace_0",
        "mst_diag_mean": "mst_diag_mean_0",
        "mst_frob": "mst_frob_0",
    }
)
mst1 = mst_small.rename(
    columns={
        "atom_index": "atom_index_1",
        "XX": "mst_XX_1",
        "YX": "mst_YX_1",
        "ZX": "mst_ZX_1",
        "XY": "mst_XY_1",
        "YY": "mst_YY_1",
        "ZY": "mst_ZY_1",
        "XZ": "mst_XZ_1",
        "YZ": "mst_YZ_1",
        "ZZ": "mst_ZZ_1",
        "mst_trace": "mst_trace_1",
        "mst_diag_mean": "mst_diag_mean_1",
        "mst_frob": "mst_frob_1",
    }
)

train_feat = train_feat.merge(
    mst0, on=["molecule_name", "atom_index_0"], how="left", sort=False
)
train_feat = train_feat.merge(
    mst1, on=["molecule_name", "atom_index_1"], how="left", sort=False
)
test_feat = test_feat.merge(
    mst0, on=["molecule_name", "atom_index_0"], how="left", sort=False
)
test_feat = test_feat.merge(
    mst1, on=["molecule_name", "atom_index_1"], how="left", sort=False
)

scc = pd.read_csv(
    os.path.join(DATA_DIR, "scalar_coupling_contributions.csv"),
    dtype={
        "molecule_name": "string",
        "atom_index_0": np.int32,
        "atom_index_1": np.int32,
        "type": "category",
        "fc": np.float32,
        "sd": np.float32,
        "pso": np.float32,
        "dso": np.float32,
    },
)
scc["scc_sum"] = (scc["fc"] + scc["sd"] + scc["pso"] + scc["dso"]).astype(np.float32)
scc_small = scc[
    [
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "fc",
        "sd",
        "pso",
        "dso",
        "scc_sum",
    ]
]

train_feat = train_feat.merge(
    scc_small,
    on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
    how="left",
    sort=False,
)

train_feat["scc_sum"] = train_feat["scc_sum"].astype(np.float32)
train_feat["scc_sum"] = train_feat["scc_sum"].fillna(np.float32(0.0)).astype(np.float32)
for c in ["fc", "sd", "pso", "dso"]:
    train_feat[c] = train_feat[c].astype(np.float32)
    train_feat[c] = train_feat[c].fillna(np.float32(0.0)).astype(np.float32)

type_scc_mean = train_feat.groupby("type")["scc_sum"].mean().to_dict()

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
    "cos01",
    "X",
    "Y",
    "Z",
    "dipole_norm",
    "potential_energy",
    "mulliken_charge_0",
    "mulliken_charge_1",
    "mst_trace_0",
    "mst_diag_mean_0",
    "mst_frob_0",
    "mst_trace_1",
    "mst_diag_mean_1",
    "mst_frob_1",
    "mst_XX_0",
    "mst_YX_0",
    "mst_ZX_0",
    "mst_XY_0",
    "mst_YY_0",
    "mst_ZY_0",
    "mst_XZ_0",
    "mst_YZ_0",
    "mst_ZZ_0",
    "mst_XX_1",
    "mst_YX_1",
    "mst_ZX_1",
    "mst_XY_1",
    "mst_YY_1",
    "mst_ZY_1",
    "mst_XZ_1",
    "mst_YZ_1",
    "mst_ZZ_1",
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


def _sanitize_numeric_features_inplace(df: pd.DataFrame, cols: list) -> None:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
    max_f32 = np.float32(np.finfo(np.float32).max / 1000.0)
    df[cols] = df[cols].clip(lower=-max_f32, upper=max_f32)


_sanitize_numeric_features_inplace(train_feat, _all_num_cols_tmp)
_sanitize_numeric_features_inplace(test_feat, _all_num_cols_tmp)


def _count_nonfinite(df: pd.DataFrame, cols: list) -> int:
    cols = [c for c in cols if c in df.columns]
    arr = df[cols].to_numpy(dtype=np.float64, copy=False)
    return int((~np.isfinite(arr)).sum())


print(
    "After numeric sanitization:",
    "nonfinite train count=",
    _count_nonfinite(train_feat, _all_num_cols_tmp),
    "nonfinite test count=",
    _count_nonfinite(test_feat, _all_num_cols_tmp),
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

models = {}

y_residual = (
    train_feat["scalar_coupling_constant"].to_numpy(np.float32, copy=False)
    - train_feat["scc_sum"].to_numpy(np.float32, copy=False)
).astype(np.float32, copy=False)

type_means_resid = (
    pd.DataFrame({"type": train_feat["type"], "y_resid": y_residual})
    .groupby("type")["y_resid"]
    .mean()
    .to_dict()
)

pred_test = np.zeros(len(test_feat), dtype=np.float32)

types = sorted(train_feat["type"].unique().tolist())
print("Training types:", types)

global_mean_resid = float(np.mean(y_residual))

for t in types:
    trn_mask = (train_feat["type"] == t).to_numpy()
    tst_mask = (test_feat["type"] == t).to_numpy()

    X_tr = train_feat.loc[trn_mask, cat_cols + num_cols]
    y_tr = y_residual[trn_mask]

    if X_tr.shape[0] < 100:
        pred_resid = float(type_means_resid.get(t, global_mean_resid))
        pred_test[tst_mask] = pred_resid + float(type_scc_mean.get(t, 0.0))
        continue

    n_t = X_tr.shape[0]
    alpha_t = 1.0
    if n_t < 5000:
        alpha_t = 2.0
    elif n_t < 20000:
        alpha_t = 1.5

    base_model = Ridge(alpha=alpha_t, random_state=42)
    model = Pipeline(steps=[("prep", preprocess), ("model", base_model)])
    model.fit(X_tr, y_tr)
    models[t] = model

    X_te = test_feat.loc[tst_mask, cat_cols + num_cols]
    pred_resid = model.predict(X_te).astype(np.float32, copy=False)

    pred_test[tst_mask] = (pred_resid + np.float32(type_scc_mean.get(t, 0.0))).astype(
        np.float32, copy=False
    )

test_only_types = sorted(set(test_feat["type"].unique()) - set(types))
if test_only_types:
    for t in test_only_types:
        tst_mask = (test_feat["type"] == t).to_numpy()
        pred_test[tst_mask] = np.float32(global_mean_resid) + np.float32(
            type_scc_mean.get(t, 0.0)
        )

print("Pred stats:", pd.Series(pred_test).describe())



## === cell 4
submission = pd.DataFrame(
    {"id": test_feat["id"].to_numpy(copy=False), "scalar_coupling_constant": pred_test}
)
assert submission.shape[0] == test.shape[0]
assert submission["id"].nunique() == test.shape[0]

submission = submission.sort_values("id", kind="mergesort").reset_index(drop=True)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission.head())
