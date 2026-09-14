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

-1.36251

# 6. Current score

3.00563

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.18497) has done: 'I replace the missing `../input/champstacks` dependency (which causes the first runtime error) with a self-contained baseline pipeline that uses only the provided competition data files, so it runs end-to-end in this Kaggle environment. To keep changes minimal while still improving score from “no submission” to a reasonable baseline, I generate predictions as the per-type median `scalar_coupling_constant` from the training set (a standard strong baseline for this competition’s per-type MAE metric). I also fix deprecated `np.bool` usage and ensure the submission has exactly the required columns (`id,scalar_coupling_constant`) and a `.csv` suffix. The correlation/stacking visualization cells be guarded so they don’t fail when external stack files are absent.'
- What this solution (achieved 1.23566) has done: 'Your current baseline (per-type median) is far from the target (lower-is-better), so we need a small but real modeling step that still preserves the “core logic” of simple per-type aggregation. The minimal upgrade is to predict the per-type mean (which optimizes MAE/L1 better than median in this competition’s distribution) and then apply a tiny per-type calibration using out-of-fold residual bias estimated by molecule-group KFold (to respect the molecule split rule). This keeps the approach as a lightweight statistics-based baseline (no new model architecture/training loop), but typically reduces log(MAE) substantially versus a pure median lookup. The submission writing and stacking-guard cells stay intact, and the code remains end-to-end with a valid `submission.csv`.'
- What this solution (achieved 3.00563) has done: 'Your current per-type mean + per-type bias correction is too weak for this competition, so to move the score substantially toward the target we need to add a small amount of structure information while keeping the “groupby-aggregation” core logic. The minimal upgrade is to build two physically meaningful features from `structures.csv` for each atom pair: the inter-atomic distance and the mean atomic numbers of the two atoms, then learn a simple per-type linear correction on top of the per-type mean using GroupKFold by molecule (so validation respects molecule grouping). This keeps the approach “lightweight statistics + per-type calibration,” but typically improves a lot versus using only type means. The submission format and paths stay the same, and we still write a valid `submission.csv`.'
- What this solution (achieved 3.00563) has done: 'We keep your existing “per-type mean + linear residual correction from (dist, Z_mean, Z_diff)” core logic, but fix the biggest scoring issue: your per-type linear coefficients are currently trained on full-data type means (leakage into OOF residuals), and then you add an additional bias computed from a different OOF baseline, which miscalibrates predictions. I compute truly out-of-fold residuals using GroupKFold (by molecule), fit the per-type linear correction on those OOF residuals, and then fit the per-type bias as the mean remaining OOF residual after the linear correction—so the final test prediction uses exactly one consistent calibration. This is a minimal change (same features, same linear least-squares, same folds) but should move the score significantly lower (better) toward your target. The script still run end-to-end and write `submission.csv` with the required columns.'
- What this solution (achieved 3.00563) has done: 'I keep your current “per-type baseline + per-type linear least-squares residual correction with GroupKFold by molecule” intact, but fix the main reason the score is catastrophically worse: the per-type linear model is currently trained to predict residuals computed from an OOF baseline, then applied on top of a full-data baseline, which creates a systematic mismatch. I compute OOF residuals from a baseline defined as (type-mean + type-bias) in each fold, fit the per-type linear correction to those OOF residuals, and then for test use the *same* final baseline (global type-mean + learned type-bias) plus the linear correction—one consistent decomposition. This is a minimal semantic change (same features, same folds, same linear algebra) but should move log(MAE) strongly downward toward your target. The submission writing/format stays identical and the script still runs end-to-end within constraints.'
- What this solution (achieved 3.00563) has done: 'I keep your existing “per-type baseline + per-type linear least-squares residual correction with GroupKFold by molecule” intact, but fix the core mismatch that is likely causing the very poor score: the linear residual model is trained against OOF residuals built from an OOF baseline, yet at inference you apply it on top of a *different* baseline (full-data type mean without the same bias decomposition). I change training so the per-type linear model is fit on residuals computed from a *consistent global baseline* (full-data type mean), and then estimate a single per-type bias *after* adding the linear correction (also on OOF), so the final formula used for test matches the decomposition used to learn the pieces. This is a minimal semantic change (same features, same folds, same lstsq) and should move the score strongly downward toward your target by removing systematic miscalibration. The submission writing and paths remain unchanged and the script still runs end-to-end within the time limit.'
- What this solution (achieved 3.00563) has done: 'Your current pipeline’s score suggests the learned linear correction is miscalibrated; the smallest fix that can materially reduce the gap toward the target is to make the per-type linear model *truly out-of-fold*: for each fold and type, fit coefficients on that fold’s training part and predict only that fold’s validation part, then refit one final coefficient per type on all data for test inference. This keeps the same core logic (type-mean baseline + linear least-squares on (dist, Z_mean, Z_diff) residuals + per-type bias) but removes the “averaged-across-folds coefficients applied everywhere” mismatch that can blow up errors. I also compute the per-type bias from the OOF predictions (after linear correction) and then apply that single bias at test time, keeping the decomposition consistent. The rest (features, GroupKFold by molecule, least squares, submission format/path) stays the same.'
- What this solution (achieved 3.00563) has done: 'Your current score (3.00563, lower-is-better) is far from the target (-1.36251), so we need a real but still minimal improvement while keeping your “per-type baseline + simple linear least-squares correction with GroupKFold by molecule” core logic. The biggest issue is that the linear correction is forced to explain a highly non-linear relationship between coupling and distance; adding a log-distance feature (and optionally inverse distance) keeps the same linear model but makes it much closer to the physics and typically improves this competition a lot. I also standardize the features *within each type* when fitting least-squares (still the same lstsq core), which stabilizes coefficients across types and avoids one feature dominating due to scale. Finally, I keep your consistent OOF bias-after-linear computation and the same submission writing/format.'
- What this solution (achieved 3.00563) has done: 'Your current score (3.00563, lower-is-better) is far from the target (-1.36251), so we need a meaningful but still “same core logic” improvement: keep the per-type baseline + simple linear least-squares residual correction, but add a small set of well-known CHAMPS geometric features that are still derived only from `structures.csv`. The biggest gain with minimal semantic change is to include per-atom coordination proxies (counts/means of neighbor distances) and basic molecule-level centering (distance to molecule centroid) for both atoms; these features dramatically improve a linear residual model without changing the training approach. I also fix a subtle feature bug: you compute `log_dist`/`inv_dist` before filling missing `dist`, which can leave NaNs in those derived columns; I compute/fill `dist` first, then derive `log_dist`/`inv_dist` consistently. Everything else (GroupKFold by molecule, per-type scaling, np.linalg.lstsq, submission format/path) remains the same.'
- What this solution (achieved 3.00563) has done: 'Your current score indicates the linear correction is likely overfitting/unstable, which can make log(MAE) explode; the smallest score-improving change is to add a tiny ridge (L2) regularization to the per-type least-squares fit while keeping the same per-type baseline, features, GroupKFold scheme, and “fit residual correction then add per-type bias” decomposition. I also make the feature scaling fold-consistent by computing (mu, sig) from the fold’s training split for each type (instead of a global per-type scaler), which avoids train/valid distribution mismatch within folds. These two changes keep the exact same core approach (linear residual correction on the same features) but stabilize coefficients and should move the metric substantially downward toward the target. The submission writing/format and all paths remain unchanged, and it still runs end-to-end within the time limit.'
- What this solution (achieved 3.00563) has done: 'Your current score (3.00563, lower-is-better) indicates the linear correction is effectively harming predictions; the most likely cause is unstable coefficients from a too-weak ridge penalty and a very expensive/approximate neighbor-feature computation that can also inject noise (via atom truncation). To move the score down toward the target with minimal core-logic change, I (1) replace the slow per-molecule neighbor loop with an exact, vectorized merge-based “within-cutoff neighbor stats” computed from `structures.csv` (no truncation), and (2) increase ridge regularization slightly to stabilize per-type linear fits while keeping the same features, GroupKFold-by-molecule, and linear residual-correction approach. Everything else (baseline = per-type mean, per-type OOF bias after linear correction, and submission writing) remains the same. This should materially reduce the log(MAE) without changing the modeling approach.'
- What this solution (achieved 3.00563) has done: 'I make two minimal fixes that should move your score down (better) toward the target by removing a major computational/feature issue that can explode errors. First, your current neighbor-feature builder does an all-pairs self-merge within each molecule, which is computationally infeasible at this scale and can silently thrash memory/produce unstable results; I replace it with a per-molecule BallTree radius-neighbor aggregation (same “neighbor count/mean distance” feature idea, but computed correctly and efficiently). Second, I add a tiny safety clamp for extreme `inv_dist` values (still derived from the same distance feature) to prevent a few near-zero distances/NaNs from dominating the linear fit and hurting log(MAE). Everything else (per-type mean baseline, per-type ridge linear residual correction with GroupKFold-by-molecule, bias-after-linear OOF calibration, and submission writing) remains the same.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

DATA_DIR = "/kaggle/input/champs-scalar-coupling"
ALT_DATA_DIR = (
    "/kaggle/data/champs-scalar-coupling"  # fallback for the provided file tree
)

if not os.path.exists(DATA_DIR) and os.path.exists(ALT_DATA_DIR):
    DATA_DIR = ALT_DATA_DIR

print("Using DATA_DIR:", DATA_DIR)
print("Files (head):", sorted(os.listdir(DATA_DIR))[:10])



## === cell 1
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
structures_path = os.path.join(DATA_DIR, "structures.csv")

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
sample_sub = pd.read_csv(sample_path, usecols=["id"])

print("train:", train.shape, "test:", test.shape, "sample:", sample_sub.shape)
print(train.head())



## === cell 2
try:
    from sklearn.model_selection import GroupKFold

    from sklearn.neighbors import BallTree
except Exception as e:
    raise RuntimeError(
        "scikit-learn is required (sklearn) but not available in this environment."
    ) from e

structures = pd.read_csv(
    structures_path,
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
)

atomic_number = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
structures["Z"] = structures["atom"].map(atomic_number).astype(np.int16)

cent = (
    structures.groupby("molecule_name")[["x", "y", "z"]]
    .mean()
    .rename(columns={"x": "cx", "y": "cy", "z": "cz"})
    .reset_index()
)
structures = structures.merge(cent, on="molecule_name", how="left")
structures["dc"] = np.sqrt(
    (structures["x"] - structures["cx"]) ** 2
    + (structures["y"] - structures["cy"]) ** 2
    + (structures["z"] - structures["cz"]) ** 2
).astype(np.float32)


def build_atom_coord_features_balltree(struct_df, cutoff=1.6):
    """
    Efficient per-molecule neighbor stats within a radius cutoff using BallTree.

    Output columns:
      molecule_name, atom_index, n_neigh, mean_neigh_dist
    where n_neigh excludes self, and mean_neigh_dist is mean distance to neighbors
    within cutoff (filled to cutoff if none).
    """
    out_frames = []
    for mol, g in struct_df.groupby("molecule_name", sort=False):
        coords = g[["x", "y", "z"]].values.astype(np.float64, copy=False)
        idx = g["atom_index"].values

        if coords.shape[0] <= 1:
            out_frames.append(
                pd.DataFrame(
                    {
                        "molecule_name": [mol] * len(idx),
                        "atom_index": idx,
                        "n_neigh": np.zeros(len(idx), dtype=np.int16),
                        "mean_neigh_dist": np.full(
                            len(idx), np.float32(cutoff), dtype=np.float32
                        ),
                    }
                )
            )
            continue

        tree = BallTree(coords, leaf_size=40)
        ind, dist = tree.query_radius(
            coords, r=float(cutoff), return_distance=True, sort_results=False
        )

        n = len(idx)
        n_neigh = np.empty(n, dtype=np.int16)
        mean_d = np.empty(n, dtype=np.float32)

        for i in range(n):
            d = dist[i]
            if d.size <= 1:
                n_neigh[i] = 0
                mean_d[i] = np.float32(cutoff)
            else:
                d2 = d[d > 0.0]
                if d2.size == 0:
                    n_neigh[i] = 0
                    mean_d[i] = np.float32(cutoff)
                else:
                    n_neigh[i] = np.int16(d2.size)
                    mean_d[i] = np.float32(d2.mean())

        out_frames.append(
            pd.DataFrame(
                {
                    "molecule_name": [mol] * n,
                    "atom_index": idx,
                    "n_neigh": n_neigh,
                    "mean_neigh_dist": mean_d,
                }
            )
        )

    out = pd.concat(out_frames, axis=0, ignore_index=True)
    return out


coord = build_atom_coord_features_balltree(structures, cutoff=1.6)
structures = structures.merge(coord, on=["molecule_name", "atom_index"], how="left")

s0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
        "Z": "Z0",
        "dc": "dc0",
        "n_neigh": "n_neigh0",
        "mean_neigh_dist": "mean_neigh_dist0",
    }
)[
    [
        "molecule_name",
        "atom_index_0",
        "x0",
        "y0",
        "z0",
        "Z0",
        "dc0",
        "n_neigh0",
        "mean_neigh_dist0",
    ]
]

s1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
        "Z": "Z1",
        "dc": "dc1",
        "n_neigh": "n_neigh1",
        "mean_neigh_dist": "mean_neigh_dist1",
    }
)[
    [
        "molecule_name",
        "atom_index_1",
        "x1",
        "y1",
        "z1",
        "Z1",
        "dc1",
        "n_neigh1",
        "mean_neigh_dist1",
    ]
]


def add_pair_features(df):
    df = df.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
    df = df.merge(s1, on=["molecule_name", "atom_index_1"], how="left")

    dx = (df["x0"] - df["x1"]).astype(np.float64)
    dy = (df["y0"] - df["y1"]).astype(np.float64)
    dz = (df["z0"] - df["z1"]).astype(np.float64)
    dist = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)
    dist = dist.fillna(np.float32(np.nanmedian(dist.to_numpy())))
    df["dist"] = dist

    df["Z_mean"] = (
        (df["Z0"].astype(np.float32) + df["Z1"].astype(np.float32)) * 0.5
    ).astype(np.float32)
    df["Z_diff"] = (
        np.abs(df["Z0"].astype(np.float32) - df["Z1"].astype(np.float32))
    ).astype(np.float32)

    eps = np.float32(1e-3)
    d = df["dist"].astype(np.float32)
    df["log_dist"] = np.log(d + eps).astype(np.float32)

    inv = (1.0 / (d + eps)).astype(np.float32)
    df["inv_dist"] = np.clip(inv, 0.0, np.float32(1000.0)).astype(np.float32)

    df["dc_mean"] = (
        (df["dc0"].astype(np.float32) + df["dc1"].astype(np.float32)) * 0.5
    ).astype(np.float32)
    df["dc_diff"] = np.abs(
        df["dc0"].astype(np.float32) - df["dc1"].astype(np.float32)
    ).astype(np.float32)

    df["n_neigh_mean"] = (
        (df["n_neigh0"].astype(np.float32) + df["n_neigh1"].astype(np.float32)) * 0.5
    ).astype(np.float32)
    df["n_neigh_diff"] = np.abs(
        df["n_neigh0"].astype(np.float32) - df["n_neigh1"].astype(np.float32)
    ).astype(np.float32)

    df["mean_neigh_dist_mean"] = (
        (
            df["mean_neigh_dist0"].astype(np.float32)
            + df["mean_neigh_dist1"].astype(np.float32)
        )
        * 0.5
    ).astype(np.float32)
    df["mean_neigh_dist_diff"] = np.abs(
        df["mean_neigh_dist0"].astype(np.float32)
        - df["mean_neigh_dist1"].astype(np.float32)
    ).astype(np.float32)

    for c in [
        "Z_mean",
        "Z_diff",
        "log_dist",
        "inv_dist",
        "dc_mean",
        "dc_diff",
        "n_neigh_mean",
        "n_neigh_diff",
        "mean_neigh_dist_mean",
        "mean_neigh_dist_diff",
    ]:
        if df[c].isna().any():
            df[c] = df[c].fillna(np.float32(df[c].median()))
    return df


train_f = add_pair_features(train)
test_f = add_pair_features(test)

print(
    "Feature check (train):",
    train_f[
        [
            "dist",
            "log_dist",
            "inv_dist",
            "Z_mean",
            "Z_diff",
            "dc_mean",
            "dc_diff",
            "n_neigh_mean",
            "n_neigh_diff",
            "mean_neigh_dist_mean",
            "mean_neigh_dist_diff",
        ]
    ]
    .describe()
    .loc[["mean", "std", "min", "max"]],
)



## === cell 3
gkf = GroupKFold(n_splits=5)

types_arr = train_f["type"].values
y_arr = train_f["scalar_coupling_constant"].values.astype(np.float64)
groups_arr = train_f["molecule_name"].values

feat_cols = [
    "dist",
    "log_dist",
    "inv_dist",
    "Z_mean",
    "Z_diff",
    "dc_mean",
    "dc_diff",
    "n_neigh_mean",
    "n_neigh_diff",
    "mean_neigh_dist_mean",
    "mean_neigh_dist_diff",
]
X_all = train_f[feat_cols].values.astype(np.float64)

type_mean_full = train_f.groupby("type")["scalar_coupling_constant"].mean()
global_mean_full = float(train_f["scalar_coupling_constant"].mean())

base_full = (
    train_f["type"]
    .map(type_mean_full)
    .fillna(global_mean_full)
    .values.astype(np.float64)
)
resid_full = y_arr - base_full

unique_types = np.sort(train_f["type"].unique())

RIDGE_L2 = 1e-1

oof_corr = np.zeros(len(train_f), dtype=np.float64)

for fold, (tr_idx, va_idx) in enumerate(
    gkf.split(train_f, y_arr, groups=groups_arr), 1
):
    tr_types = types_arr[tr_idx]
    tr_resid = resid_full[tr_idx]
    tr_X = X_all[tr_idx]

    va_types = types_arr[va_idx]
    va_X = X_all[va_idx]

    for t in np.unique(va_types):
        va_mask = va_types == t

        tr_mask = tr_types == t
        Xt = tr_X[tr_mask]
        yt = tr_resid[tr_mask]
        if Xt.shape[0] < 50:
            oof_corr[va_idx[va_mask]] = 0.0
            continue

        mu = Xt.mean(axis=0)
        sig = Xt.std(axis=0)
        sig[sig == 0.0] = 1.0

        Xt_s = (Xt - mu) / sig
        Xv = va_X[va_mask]
        Xv_s = (Xv - mu) / sig

        A_tr = np.concatenate(
            [np.ones((Xt_s.shape[0], 1), dtype=np.float64), Xt_s], axis=1
        )
        A_va = np.concatenate(
            [np.ones((Xv_s.shape[0], 1), dtype=np.float64), Xv_s], axis=1
        )

        AtA = A_tr.T @ A_tr
        reg = np.eye(AtA.shape[0], dtype=np.float64) * RIDGE_L2
        reg[0, 0] = 0.0
        coef = np.linalg.solve(AtA + reg, A_tr.T @ yt)

        oof_corr[va_idx[va_mask]] = A_va @ coef

    if fold == 1:
        print(
            "Fold",
            fold,
            "train rows:",
            len(tr_idx),
            "valid rows:",
            len(va_idx),
            "types:",
            len(np.unique(tr_types)),
        )

oof_remaining = y_arr - (base_full + oof_corr)

bias_by_type = (
    pd.DataFrame({"type": types_arr, "residual": oof_remaining})
    .groupby("type")["residual"]
    .mean()
)
global_bias = float(oof_remaining.mean())

print("Computed per-type bias after OOF linear correction (head):")
print(bias_by_type.sort_index().head(10))

type_scaler = {}
coef_by_type = {}
for t in unique_types:
    mask = types_arr == t
    Xt = X_all[mask]
    yt = resid_full[mask]
    if Xt.shape[0] < 50:
        type_scaler[t] = (
            np.zeros(X_all.shape[1], dtype=np.float64),
            np.ones(X_all.shape[1], dtype=np.float64),
        )
        coef_by_type[t] = np.zeros(1 + len(feat_cols), dtype=np.float64)
        continue

    mu = Xt.mean(axis=0)
    sig = Xt.std(axis=0)
    sig[sig == 0.0] = 1.0
    type_scaler[t] = (mu, sig)

    Xt_s = (Xt - mu) / sig
    A = np.concatenate([np.ones((Xt_s.shape[0], 1), dtype=np.float64), Xt_s], axis=1)

    AtA = A.T @ A
    reg = np.eye(AtA.shape[0], dtype=np.float64) * RIDGE_L2
    reg[0, 0] = 0.0
    coef = np.linalg.solve(AtA + reg, A.T @ yt)
    coef_by_type[t] = coef



## === cell 4
test_f = test_f.copy()

test_base = (
    test_f["type"]
    .map(type_mean_full)
    .fillna(global_mean_full)
    .astype(np.float64)
    .values
)
test_bias = (
    test_f["type"].map(bias_by_type).fillna(global_bias).astype(np.float64).values
)

Xt = test_f[feat_cols].values.astype(np.float64)
corr = np.zeros(len(test_f), dtype=np.float64)

test_types = test_f["type"].values
for t in np.unique(test_types):
    m = test_types == t
    coef = coef_by_type.get(t, None)
    if coef is None:
        continue
    mu_sig = type_scaler.get(t, None)
    if mu_sig is None:
        Xs = Xt[m]
    else:
        mu, sig = mu_sig
        Xs = (Xt[m] - mu) / sig
    A = np.concatenate([np.ones((m.sum(), 1), dtype=np.float64), Xs], axis=1)
    corr[m] = A @ coef

test_f["scalar_coupling_constant"] = (test_base + corr + test_bias).astype(np.float32)

sub = pd.merge(
    sample_sub, test_f[["id", "scalar_coupling_constant"]], on="id", how="left"
)
sub["scalar_coupling_constant"] = (
    sub["scalar_coupling_constant"].fillna(global_mean_full).astype(np.float32)
)

print(sub.head())
print("sub shape:", sub.shape)



## === cell 5
out_path = "submission.csv"
sub[["id", "scalar_coupling_constant"]].to_csv(
    out_path, index=False, float_format="%.6f"
)
print("Wrote:", out_path, "rows:", len(sub))



## === cell 6
diag = pd.DataFrame(
    {"type_mean": type_mean_full, "bias_after_linear_oof": bias_by_type}
).sort_index()
print("Type mean + bias-after-linear (head):")
print(diag.head(10))
print("Global mean:", global_mean_full, "Global bias:", global_bias)
print("Example learned coefficients (first 5 types):")
for t in list(diag.index[:5]):
    print(t, coef_by_type[t])



## === cell 7
sub_path = "../input/champstacks"
all_files = []
concat_sub = None
ncol = 0
print(
    "Note: champstacks path not used; generated structure-aware calibrated per-type mean submission instead."
)



## === cell 8
if os.path.exists(sub_path):
    all_files = os.listdir(sub_path)
    outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files]
    concat_sub = pd.concat(outs, axis=1)
    cols = list(map(lambda x: "champ" + str(x), range(len(concat_sub.columns))))
    concat_sub.columns = cols
    concat_sub.reset_index(inplace=True)
    ncol = concat_sub.shape[1]
    print("Loaded stacked submissions:", len(all_files), "ncol:", ncol)
else:
    print("champstacks folder not found; skipping stacking workflow.")



## === cell 9
if concat_sub is not None and ncol > 1:
    corr_m = concat_sub.iloc[:, 1:ncol].corr()
    mask = np.zeros_like(corr_m, dtype=bool)
    print("Correlation computed for stack matrix:", corr_m.shape)
else:
    print("No concat_sub available; skipping correlation.")



## === cell 10
cutoff_lo = -37
cutoff_hi = 205
print("cutoff_lo, cutoff_hi:", cutoff_lo, cutoff_hi)



## === cell 11
if concat_sub is not None and "champ_mean" in concat_sub.columns:
    concat_sub["scalar_coupling_constant"] = concat_sub["champ_mean"]
    concat_sub[["id", "scalar_coupling_constant"]].to_csv(
        "stack_mean.csv", index=False, float_format="%.6f"
    )
    print("Wrote stack_mean.csv")
else:
    print("Skipping stack_mean.csv (no stacked inputs).")



## === cell 12
if concat_sub is not None and "champ_median" in concat_sub.columns:
    concat_sub["scalar_coupling_constant"] = concat_sub["champ_median"]
    concat_sub[["id", "scalar_coupling_constant"]].to_csv(
        "stack_median.csv", index=False, float_format="%.6f"
    )
    print("Wrote stack_median.csv")
else:
    print("Skipping stack_median.csv (no stacked inputs).")



## === cell 13
assert os.path.exists(out_path), "submission.csv was not created"
assert sub["id"].isna().sum() == 0, "Missing ids in submission"
assert (
    sub["scalar_coupling_constant"].isna().sum() == 0
), "Missing predictions in submission"
print("Done. Final submission file:", out_path)
print(sub.head())
