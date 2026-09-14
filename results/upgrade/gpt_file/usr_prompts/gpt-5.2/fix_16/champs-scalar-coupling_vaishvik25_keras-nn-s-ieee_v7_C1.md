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
seaborn==0.12.2
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

-1.5192329259486128

# 6. Current score

1.18497

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.18497) has done: 'I fix the immediate runtime blockers by removing notebook-only magics, replacing missing `../input/top-mol` and `../input/another-one` dependencies with a self-contained baseline that uses only the provided competition files, and ensuring all variables used downstream are defined. Because you currently have no valid submission (“Not yielded”), the priority is to reliably write a correctly formatted `submission.csv` with `id,scalar_coupling_constant`. To keep changes minimal while improving score versus a trivial constant, I use a simple type-wise median target computed from `train.csv` and mapped onto `test.csv` by `type` (a common strong baseline for this metric). The script also include safe fallbacks and alignment checks so it runs end-to-end in the Kaggle environment.'
- What this solution (achieved 1.23596) has done: 'Your current score (1.18497, lower-is-better) is far worse than the target (-1.5192), so we should improve substantially while keeping the same “type-wise statistic mapping” core approach. The smallest meaningful upgrade is to predict per-`type` **mean** instead of median (often better for MAE-like objectives here) and to add a tiny amount of robust shrinkage toward the global mean to stabilize rare types. We also compute these statistics from the full `train.csv` target (same as before), keep the same submission format and ID alignment checks, and still write `submission.csv` end-to-end.'
- What this solution (achieved 1.18495) has done: 'Your current approach is a per-`type` constant predictor; to move the score substantially toward the target while preserving that core logic, the most direct improvement is to predict the **per-`type` median** (optimal for MAE) instead of a shrunk mean. To keep stability for smaller `type` groups (and avoid extreme medians from tiny counts), I apply a minimal “shrinkage toward global median” using the same pseudo-count idea you already use. This stays within the same semantics (type-wise statistic mapping) and only changes which robust statistic is mapped and how it’s smoothed. The submission writing, ID alignment, and paths remain unchanged.'
- What this solution (achieved 1.99065) has done: 'I fix the `np.isfinite` TypeError by ensuring the design matrices are purely numeric (float64) before calling `isfinite` and fitting Ridge; this is caused by mixed dtypes coming out of the pandas concatenation/dummies. I also make the per-type train/test feature matrices use aligned columns and consistent indexing without relying on `.values` from object-typed frames. Finally, I guard downstream cells so `submission` is always defined (even if a rare unexpected failure happens) and ensure `submission.csv` is written with the required columns and row count.'
- What this solution (achieved 3.87369) has done: 'We fix the NaN crash by ensuring the per-type design matrices are NaN-free before fitting/predicting with Ridge (the NaNs are coming from missing structure merges that weren’t fully filled for x/y/z, not just `dist`). We do this with minimal changes: fill missing coordinates and atom labels consistently in `add_pair_features`, and add a final safety `np.nan_to_num` right before model fit/predict so Ridge never sees NaNs. This preserves your core logic (per-type Ridge on `dist` + one-hot atom_pair, with shrinkage fallback/blending) while making the pipeline run end-to-end and produce a valid `submission.csv`. This should also improve score versus the current failure mode/fallback behavior by allowing Ridge to run for all eligible types.'
- What this solution (achieved 4.53266) has done: 'Your current score (3.87369, lower-is-better) is far worse than the target (-1.5192), and the most likely cause is that your feature merge is still producing essentially “broken” distances (because the coordinate fill uses `structures[c[0]]` which doesn’t exist, so many coords become 0/NaN-like), making Ridge ineffective and pushing predictions toward weak fallbacks. I make the smallest fix that preserves your core logic: correct the coordinate imputation to use the actual `structures["x"/"y"/"z"]` medians, and add a tiny extra geometric signal (`dist2`) while keeping the same per-type Ridge + atom_pair one-hot + blending/fallback. This should materially improve the model’s ability to learn within each coupling type without changing the overall approach. The submission writing, paths, and schema remain unchanged.'
- What this solution (achieved 1.99065) has done: 'Your current score is far worse than the target (lower is better), so we should improve meaningfully while keeping the same core “per-type Ridge on distance + atom_pair one-hot with shrinkage/blending” approach. The biggest issue is that `pd.get_dummies` inside each type creates huge dense matrices and can be unstable/slow; switching that part to a sparse one-hot via `OneHotEncoder(handle_unknown="ignore")` preserves the exact modeling semantics but makes training feasible and consistent across types. I also add a single additional geometric feature (`inv_dist`) that is a monotonic transform of distance and commonly helps this competition without changing the overall feature extraction approach. Finally, I keep your shrinkage baseline and blending, and ensure the submission is aligned and written exactly as required.'
- What this solution (achieved 21.60477) has done: 'I fix the Ridge fitting crash by forcing a stable Ridge solver that does not rely on SciPy’s `cg(tol=...)` signature, which is incompatible in your environment. This keeps the same per-type Ridge-on-distance-plus-onehot core logic and should restore end-to-end training/prediction for all eligible types (improving score versus falling back). I also keep the existing shrinkage/blending and submission alignment checks unchanged, only adjusting parameters needed for runtime stability. Finally, I ensure `submission.csv` is always written with the correct columns and row count.'
- What this solution (achieved 1.99065) has done: 'Your current score is far worse than the target (lower is better), and the biggest issue is that the model is training with Ridge but without any feature scaling; in this competition, unscaled geometric features (`dist`, `dist2`, `inv_dist`) can make Ridge behave poorly and inflate errors across types. I keep the exact same per-type Ridge + atom_pair one-hot + shrinkage/blending logic, but add a minimal per-type standardization of the 3 numeric distance features using the training split stats for that type and apply it to test. I also switch the Ridge solver to `sag`, which is robust for large sparse matrices and avoids solver incompatibilities, while keeping all I/O paths and the submission schema unchanged. These two small, directly relevant changes should move the score substantially downward (toward the target band) without changing the overall approach.'
- What this solution (achieved 25.64728) has done: 'I fix the Ridge runtime error by switching to a solver that does not call SciPy’s `cg(tol=...)` under the hood in this environment (the current failure path is effectively using `sparse_cg`). I keep your per-type Ridge + sparse one-hot + numeric distance features + standardization + blending/shrinkage logic intact, only changing the solver choice and adding a small safe fallback to ensure fitting never crashes for any type. This should restore end-to-end training/prediction (instead of failing before writing a submission), and typically improves score versus falling back to constant predictions. The script still write a valid `submission.csv` with the required columns and row count.'
- What this solution (achieved 1.18495) has done: 'Your current score (25.647, lower-is-better) indicates the model is effectively failing to learn and/or falling back to near-constant predictions for many rows; the smallest change that should materially move you toward the target is to make Ridge actually converge robustly on the large sparse per-type design matrices. I keep the same per-type Ridge + (dist, dist2, inv_dist) + atom_pair one-hot + shrinkage/blend core logic, but (1) switch the Ridge solver to a sparse-friendly, stable option (`sag`) and increase `max_iter`/set `tol` so it converges instead of producing bad fits, (2) add a per-type target centering (fit on `y - base`, then add `base` back) which preserves semantics but improves numerical stability, and (3) slightly reduce the blend-to-base so predictions rely less on the constant fallback when Ridge is working. These are minimal, directly score-relevant adjustments and keep the submission format and paths unchanged.'
- What this solution (achieved 1.18497) has done: 'Your current score (1.18495, lower-is-better) is far from the target (-1.5192), so we should improve substantially but with minimal, core-logic-preserving changes. The biggest safe gain within your existing “per-type Ridge on (dist features + atom_pair one-hot) with base shrinkage” approach is to fix underfitting by slightly increasing model capacity and making the base target more appropriate for MAE: use a per-type median baseline (already) but reduce excessive shrinkage and blending that pull predictions toward a constant. I also add one more strictly “pair-geometry” numeric feature (`log1p_dist`) that is a monotonic transform of distance and commonly helps this competition, without changing the modeling approach. Finally, I keep the same I/O paths and submission schema, and keep the solver stable while increasing iterations modestly for better convergence.'
- What this solution (achieved 1.18497) has done: 'Your current score (1.18497, lower-is-better) is far from the target (-1.5192), so we should improve materially while keeping the same per-type Ridge-on-distance+atom_pair core logic. The smallest high-impact fix is to add molecule-level context that is already available in `structures.csv`: per-atom neighbor counts and neighbor-distance summaries within a fixed radius; these are standard for this competition and preserve the same modeling approach (still linear Ridge per type). I also adjust the minimum rows-per-type gate downward so more coupling types can use Ridge instead of the constant baseline (this usually reduces MAE). Finally, I keep all I/O paths and submission schema unchanged and ensure all new features are numeric, NaN-safe, and standardized per type like your existing numeric features.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

DATA_DIR_CANDIDATES = [
    "/kaggle/input/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling",
    "../input/champs-scalar-coupling",
    "../input",  # sometimes files are directly here
]


def find_data_dir():
    for d in DATA_DIR_CANDIDATES:
        if os.path.isdir(d):
            if os.path.exists(os.path.join(d, "train.csv")) and os.path.exists(
                os.path.join(d, "test.csv")
            ):
                return d
    for d in DATA_DIR_CANDIDATES:
        if os.path.isdir(d):
            for root, _, files in os.walk(d):
                if "train.csv" in files and "test.csv" in files:
                    return root
    raise FileNotFoundError(
        "Could not locate champs-scalar-coupling data directory with train.csv/test.csv"
    )


DATA_DIR = find_data_dir()
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
structures_path = os.path.join(DATA_DIR, "structures.csv")

print("Using DATA_DIR:", DATA_DIR)
print(
    "Train exists:",
    os.path.exists(train_path),
    "Test exists:",
    os.path.exists(test_path),
    "Sample exists:",
    os.path.exists(sample_path),
    "Structures exists:",
    os.path.exists(structures_path),
)


## === cell 1
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse

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
)
test = pd.read_csv(
    test_path,
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
)

structures = pd.read_csv(
    structures_path, usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"]
)

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


def compute_atom_env_features(structures_df, radius=2.0, chunk_size=200_000):
    """
    Returns per-atom features:
      - nb_cnt_r: number of neighbors within radius (excluding itself)
      - nb_mean_dist_r: mean distance to those neighbors
      - nb_min_dist_r: min distance to those neighbors
    Computed approximately with an axis-aligned bounding box filter before exact distance,
    to keep runtime reasonable under Kaggle constraints.
    """
    sdf = structures_df.copy()
    for c in ["x", "y", "z"]:
        sdf[c] = pd.to_numeric(sdf[c], errors="coerce").astype("float64")
    sdf["atom_index"] = sdf["atom_index"].astype("int32")
    sdf["molecule_name"] = sdf["molecule_name"].astype("category")

    n = len(sdf)
    cnt = np.zeros(n, dtype=np.int32)
    sumd = np.zeros(n, dtype=np.float64)
    mind = np.full(n, np.inf, dtype=np.float64)

    groups = sdf.groupby("molecule_name", sort=False).indices
    r = float(radius)
    r2 = r * r

    x_all = sdf["x"].to_numpy()
    y_all = sdf["y"].to_numpy()
    z_all = sdf["z"].to_numpy()

    for _, idx in groups.items():
        m = len(idx)
        if m <= 1:
            continue

        xi = x_all[idx]
        yi = y_all[idx]
        zi = z_all[idx]

        for i in range(m):
            dx = xi[i] - xi
            dy = yi[i] - yi
            dz = zi[i] - zi

            mask = (np.abs(dx) <= r) & (np.abs(dy) <= r) & (np.abs(dz) <= r)
            mask[i] = False
            if not np.any(mask):
                continue
            d2 = dx[mask] * dx[mask] + dy[mask] * dy[mask] + dz[mask] * dz[mask]
            mask2 = d2 <= r2
            if not np.any(mask2):
                continue
            d = np.sqrt(d2[mask2])

            gi = idx[i]
            cnt[gi] += d.shape[0]
            sumd[gi] += d.sum()
            md = d.min()
            if md < mind[gi]:
                mind[gi] = md

    mean = np.divide(sumd, cnt, out=np.zeros_like(sumd), where=cnt > 0)
    mind = np.where(np.isfinite(mind), mind, 0.0)

    out = sdf[["molecule_name", "atom_index"]].copy()
    out["nb_cnt_r2"] = cnt.astype("float64")
    out["nb_mean_dist_r2"] = mean.astype("float64")
    out["nb_min_dist_r2"] = mind.astype("float64")
    return out


atom_env = compute_atom_env_features(structures, radius=2.0)

a0_env = atom_env.rename(
    columns={
        "atom_index": "atom_index_0",
        "nb_cnt_r2": "nb_cnt0_r2",
        "nb_mean_dist_r2": "nb_mean_dist0_r2",
        "nb_min_dist_r2": "nb_min_dist0_r2",
    }
)
a1_env = atom_env.rename(
    columns={
        "atom_index": "atom_index_1",
        "nb_cnt_r2": "nb_cnt1_r2",
        "nb_mean_dist_r2": "nb_mean_dist1_r2",
        "nb_min_dist_r2": "nb_min_dist1_r2",
    }
)


def add_pair_features(df):
    df = df.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
    df = df.merge(s1, on=["molecule_name", "atom_index_1"], how="left")

    df = df.merge(a0_env, on=["molecule_name", "atom_index_0"], how="left")
    df = df.merge(a1_env, on=["molecule_name", "atom_index_1"], how="left")

    coord_medians = {
        "x": float(pd.to_numeric(structures["x"], errors="coerce").median()),
        "y": float(pd.to_numeric(structures["y"], errors="coerce").median()),
        "z": float(pd.to_numeric(structures["z"], errors="coerce").median()),
    }

    for c in ["x0", "y0", "z0", "x1", "y1", "z1"]:
        axis = c[0]  # 'x'/'y'/'z'
        df[c] = (
            pd.to_numeric(df[c], errors="coerce")
            .fillna(coord_medians[axis])
            .astype("float64")
        )

    df["atom_0"] = df["atom_0"].astype("string").fillna("UNK")
    df["atom_1"] = df["atom_1"].astype("string").fillna("UNK")

    dx = (df["x0"] - df["x1"]).astype("float64")
    dy = (df["y0"] - df["y1"]).astype("float64")
    dz = (df["z0"] - df["z1"]).astype("float64")

    dist2 = (dx * dx + dy * dy + dz * dz).astype("float64")
    dist = np.sqrt(dist2).astype("float64")

    df["dist"] = dist
    df["dist2"] = dist2

    eps = 1e-8
    df["inv_dist"] = (1.0 / (df["dist"] + eps)).astype("float64")

    df["log1p_dist"] = np.log1p(df["dist"]).astype("float64")

    a0 = df["atom_0"].astype(str)
    a1 = df["atom_1"].astype(str)
    df["atom_pair"] = np.where(a0 <= a1, a0 + "_" + a1, a1 + "_" + a0).astype(object)

    env_cols = [
        "nb_cnt0_r2",
        "nb_mean_dist0_r2",
        "nb_min_dist0_r2",
        "nb_cnt1_r2",
        "nb_mean_dist1_r2",
        "nb_min_dist1_r2",
    ]
    for c in env_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
        df[c] = df[c].fillna(
            float(df[c].median() if np.isfinite(df[c].median()) else 0.0)
        )

    df["nb_cnt_sum_r2"] = (df["nb_cnt0_r2"] + df["nb_cnt1_r2"]).astype("float64")
    df["nb_cnt_absdiff_r2"] = (
        (df["nb_cnt0_r2"] - df["nb_cnt1_r2"]).abs().astype("float64")
    )
    df["nb_mean_dist_sum_r2"] = (
        df["nb_mean_dist0_r2"] + df["nb_mean_dist1_r2"]
    ).astype("float64")
    df["nb_min_dist_min_r2"] = np.minimum(
        df["nb_min_dist0_r2"], df["nb_min_dist1_r2"]
    ).astype("float64")

    for c in [
        "dist",
        "dist2",
        "inv_dist",
        "log1p_dist",
        "nb_cnt0_r2",
        "nb_mean_dist0_r2",
        "nb_min_dist0_r2",
        "nb_cnt1_r2",
        "nb_mean_dist1_r2",
        "nb_min_dist1_r2",
        "nb_cnt_sum_r2",
        "nb_cnt_absdiff_r2",
        "nb_mean_dist_sum_r2",
        "nb_min_dist_min_r2",
    ]:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
        df[c] = df[c].fillna(df[c].median())

    df["atom_pair"] = df["atom_pair"].fillna("UNK_UNK")
    return df


train_f = add_pair_features(train)
test_f = add_pair_features(test)

global_median = float(train_f["scalar_coupling_constant"].median())

print("Train features head:")
print(
    train_f[
        [
            "type",
            "atom_pair",
            "dist",
            "dist2",
            "inv_dist",
            "log1p_dist",
            "nb_cnt_sum_r2",
            "nb_cnt_absdiff_r2",
            "nb_mean_dist_sum_r2",
            "nb_min_dist_min_r2",
            "scalar_coupling_constant",
        ]
    ].head()
)


## === cell 2
type_stats = train_f.groupby("type")["scalar_coupling_constant"].agg(
    ["median", "count"]
)

alpha = 10.0
type_shrunk_median = (
    type_stats["median"] * type_stats["count"] + global_median * alpha
) / (type_stats["count"] + alpha)

pred = pd.Series(index=test_f.index, dtype="float64")

MIN_ROWS_PER_TYPE = 50
RIDGE_ALPHA = 0.3

RIDGE_SOLVER = "sag"
RIDGE_MAX_ITER = 6000
RIDGE_TOL = 1e-4
RIDGE_RANDOM_STATE = 42

NUM_COLS = [
    "dist",
    "dist2",
    "inv_dist",
    "log1p_dist",
    "nb_cnt_sum_r2",
    "nb_cnt_absdiff_r2",
    "nb_mean_dist_sum_r2",
    "nb_min_dist_min_r2",
]

for t, test_idx in test_f.groupby("type").groups.items():
    base = float(type_shrunk_median.get(t, global_median))

    tr_t = train_f[train_f["type"] == t]
    te_t = test_f.loc[test_idx]

    if len(tr_t) < MIN_ROWS_PER_TYPE:
        pred.loc[test_idx] = base
        continue

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64)
    X_tr_ap = enc.fit_transform(tr_t[["atom_pair"]])
    X_te_ap = enc.transform(te_t[["atom_pair"]])

    X_tr_num = tr_t[NUM_COLS].astype("float64").to_numpy(copy=False)
    X_te_num = te_t[NUM_COLS].astype("float64").to_numpy(copy=False)

    X_tr_num = np.nan_to_num(X_tr_num, nan=0.0, posinf=0.0, neginf=0.0)
    X_te_num = np.nan_to_num(X_te_num, nan=0.0, posinf=0.0, neginf=0.0)

    mu = X_tr_num.mean(axis=0)
    sigma = X_tr_num.std(axis=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)

    X_tr_num = (X_tr_num - mu) / sigma
    X_te_num = (X_te_num - mu) / sigma

    X_tr = sparse.hstack([sparse.csr_matrix(X_tr_num), X_tr_ap], format="csr")
    X_te = sparse.hstack([sparse.csr_matrix(X_te_num), X_te_ap], format="csr")

    y_tr = tr_t["scalar_coupling_constant"].astype("float64").to_numpy()
    y_tr = np.nan_to_num(y_tr, nan=base, posinf=base, neginf=base)

    y_center = y_tr - base

    model = Ridge(
        alpha=RIDGE_ALPHA,
        fit_intercept=True,
        solver=RIDGE_SOLVER,
        max_iter=RIDGE_MAX_ITER,
        tol=RIDGE_TOL,
        random_state=RIDGE_RANDOM_STATE,
    )

    try:
        model.fit(X_tr, y_center)
        y_hat = model.predict(X_te).astype("float64") + base
    except Exception as e:
        print(f"Warning: Ridge failed for type={t} with error={repr(e)}; using base.")
        y_hat = np.full(shape=(len(te_t),), fill_value=base, dtype="float64")

    blend = 0.0
    y_hat = (1.0 - blend) * y_hat + blend * base

    pred.loc[test_idx] = y_hat

pred = pred.fillna(global_median)

submission = pd.DataFrame(
    {
        "id": test_f["id"].astype(np.int64),
        "scalar_coupling_constant": pred.astype("float64"),
    }
)
submission.sort_values("id", inplace=True)

print(submission.head())
print("Submission shape:", submission.shape)


## === cell 3
if "submission" not in globals() or submission is None or len(submission) == 0:
    submission = pd.DataFrame(
        {"id": test["id"].astype(np.int64), "scalar_coupling_constant": global_median}
    ).sort_values("id")

if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    if "id" in sample.columns and len(sample) == len(submission):
        same_ids = sample["id"].astype(np.int64).values
        sub_ids = submission["id"].astype(np.int64).values
        if not np.array_equal(np.sort(same_ids), np.sort(sub_ids)):
            print("Warning: submission ids do not match sample_submission ids set.")
        else:
            print("ID set matches sample_submission.")
    else:
        print(
            "sample_submission.csv present but shape/columns unexpected; skipping strict alignment check."
        )
else:
    print("sample_submission.csv not found; skipping alignment check.")


## === cell 4
out_path = "submission.csv"
submission.to_csv(out_path, index=False, float_format="%.6f")
print("Wrote:", out_path, "size(bytes):", os.path.getsize(out_path))
print(pd.read_csv(out_path).head())
