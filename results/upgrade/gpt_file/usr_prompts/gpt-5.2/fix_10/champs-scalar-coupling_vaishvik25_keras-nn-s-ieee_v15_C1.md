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

-1.356722768415636

# 6. Current score

2.92814

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 3.12217) has done: 'I fix the NaNs coming from the structures merge by enforcing consistent dtypes on join keys and de-duplicating `structures` on `(molecule_name, atom_index)` before merging (the current NaNs indicate key mismatches/duplicates). Then I add a minimal, score-neutral safety net by imputing any remaining NaNs in the final feature matrix with column medians so `Ridge` can fit/predict reliably. I also make the later cells robust by only using variables that are guaranteed to exist after the training/prediction cell completes, and ensure a single valid `submission.csv` is always written with the exact required columns and id alignment. Core modeling logic (pair features + one-hot + per-type Ridge + simple averaging) is preserved.'
- What this solution (achieved 3.09449) has done: 'I fix the failure in feature-building by making the structures merge robust to the known “atom_index off-by-one” inconsistency that can exist between `train/test` and `structures.csv` in some Kaggle mirrors (this is what’s producing the NaNs and stopping execution). Instead of hard-failing, the code detect missing structure matches and, if needed, retry the merge with `atom_index_0/1 - 1`, then continue with the same pairwise distance features and per-type Ridge training. I also keep a small median-imputation safety net for any remaining numeric NaNs so Ridge always fits, but preserve the model/stacking logic exactly otherwise. The pipeline always write a valid `submission.csv` with the required columns and correct id alignment.'
- What this solution (achieved 3.10261) has done: 'Your current score (3.09449, lower-is-better) is far worse than the target (-1.3567), so we should improve model fidelity with minimal logic changes. The biggest issue is that your “stacking” columns (mol0/mol1/mol2) are almost identical transforms of the same prediction, so taking their median/mean can’t help—and the “pushout/minmax/rank” transformations can severely hurt this MAE-style metric. I keep your exact feature set and per-type Ridge approach, but add a tiny, metric-aligned calibration step: per-type shrinkage toward the per-type mean using a cross-validated weight computed only on train (no leakage), and then use that calibrated prediction directly for submission. This is a small, safe change that typically reduces MAE/logMAE without changing the model class or adding new features.'
- What this solution (achieved 2.87424) has done: 'We keep your exact feature set (pairwise coords/dist + one-hot type/atoms) and the per-type Ridge modeling approach, but fix the biggest scoring issue: Ridge is being fit on raw, unscaled numeric features mixed with sparse one-hot, which often hurts MAE badly for this competition. The minimal, metric-aligned improvement is to standardize only the numeric columns (using train+test to keep transforms consistent, as you already do for dummies) while leaving one-hot columns unchanged, and then refit the same per-type Ridge. We also slightly widen the per-type shrinkage grid around 1.0 so calibration can correct small systematic bias instead of mostly shrinking, without changing semantics. This should move the score substantially downward (better) toward the target while keeping runtime and core logic intact.'
- What this solution (achieved 3.0028) has done: 'Your current score (2.87424, lower-is-better) is still very far from the target (-1.3567), so we should improve accuracy with minimal, metric-aligned changes while preserving your Ridge-per-type core. The biggest remaining gap is that the model only uses coordinates/distance + one-hots; adding a single, very standard chemistry feature (per-atom Mulliken charge) via a safe merge on `(molecule_name, atom_index)` typically reduces MAE substantially without changing the training approach. We merge `mulliken_charges.csv` twice (for atom_0 and atom_1), add simple derived features (`q0`, `q1`, `qsum`, `qdiff`, `qprod`), standardize them together with your existing numeric columns (same approach as you already use), and keep the same per-type Ridge + calibration + submission writing. This should move the score downward (better) toward the target while keeping runtime reasonable and preserving evaluation semantics.'
- What this solution (achieved 3.08721) has done: 'We keep your per-type Ridge + calibration pipeline intact, but add one more very standard signal source from this competition: per-atom magnetic shielding tensors, merged twice (atom_0/atom_1) exactly like Mulliken charges. Then we create a few simple, low-risk derived features (sum/abs-diff) and include them in the same numeric standardization step you already do, which typically reduces MAE without changing training semantics. To avoid instability and stay within the 600s budget, we only use a small, robust subset of tensor features (diagonal + trace) rather than all 9 components. Everything else (splits, Ridge, calibration grid, submission writing and file paths) remains the same.'
- What this solution (achieved 2.93002) has done: 'We keep your per-type Ridge + calibration pipeline intact, but add the single most important missing signal for this competition: better geometry context beyond the paired atoms. The minimal change is to compute per-atom neighborhood distance summaries (min/mean/max distance to all other atoms in the same molecule) for atom_0 and atom_1 using `structures.csv`, then derive a few simple pair features (sums/diffs/ratios) and standardize them alongside your existing numeric columns. This preserves your feature-extraction pattern (merge → derive numeric features → one-hot → per-type Ridge) while typically reducing MAE substantially (moving the logMAE score downward toward the target). All file paths, model class, training loop structure, and submission writing remain unchanged.'
- What this solution (achieved 2.92814) has done: 'Your current score (2.93002, lower-is-better) is still far from the target (-1.3567), so we should legitimately improve predictive accuracy while keeping your core pipeline (feature merges → numeric standardization + one-hot → per-type Ridge → per-type calibration → submission) intact. The biggest remaining issue is that `atom_stats` is computed with an O(n²) distance matrix per molecule (slow) and the `nbr_max_dist` computation is incorrect/unstable, which can inject noisy/NaN neighborhood features that hurt MAE; we replace that with a mathematically equivalent but O(n log n) KDTree-based neighbor summary (min/mean/max distance) per atom, preserving the same feature semantics. We also fix `cutoff_lo/cutoff_hi` (currently reversed) only to avoid accidental downstream misuse, and keep everything else unchanged, including output files and submission format. These changes are directly aimed at making the neighborhood geometry features correct and consistent, which should move the score downward toward the target without altering the modeling approach.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.neighbors import KDTree



## === cell 1
BASE_PATH = "../input/champs-scalar-coupling"
TRAIN_PATH = os.path.join(BASE_PATH, "train.csv")
TEST_PATH = os.path.join(BASE_PATH, "test.csv")
STRUCTURES_PATH = os.path.join(BASE_PATH, "structures.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")
MULLIKEN_PATH = os.path.join(BASE_PATH, "mulliken_charges.csv")
MAG_TENSOR_PATH = os.path.join(BASE_PATH, "magnetic_shielding_tensors.csv")

for p in [
    TRAIN_PATH,
    TEST_PATH,
    STRUCTURES_PATH,
    SAMPLE_SUB_PATH,
    MULLIKEN_PATH,
    MAG_TENSOR_PATH,
]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required file not found: {p}")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
structures = pd.read_csv(STRUCTURES_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
mulliken = pd.read_csv(MULLIKEN_PATH)
mag = pd.read_csv(MAG_TENSOR_PATH)

assert set(["id", "molecule_name", "atom_index_0", "atom_index_1", "type"]).issubset(
    test.columns
)
assert set(["id", "scalar_coupling_constant"]).issubset(sample_sub.columns)

print(
    "train:",
    train.shape,
    "test:",
    test.shape,
    "structures:",
    structures.shape,
    "mulliken:",
    mulliken.shape,
    "mag:",
    mag.shape,
    "sample_sub:",
    sample_sub.shape,
)
print(
    "types (train):", train["type"].nunique(), "types (test):", test["type"].nunique()
)



## === cell 2
for df in (train, test, structures, mulliken, mag):
    df["molecule_name"] = df["molecule_name"].astype(str)

structures["atom_index"] = structures["atom_index"].astype(np.int32, copy=False)
mulliken["atom_index"] = mulliken["atom_index"].astype(np.int32, copy=False)
mag["atom_index"] = mag["atom_index"].astype(np.int32, copy=False)

train["atom_index_0"] = train["atom_index_0"].astype(np.int32, copy=False)
train["atom_index_1"] = train["atom_index_1"].astype(np.int32, copy=False)
test["atom_index_0"] = test["atom_index_0"].astype(np.int32, copy=False)
test["atom_index_1"] = test["atom_index_1"].astype(np.int32, copy=False)

structures = structures.drop_duplicates(
    subset=["molecule_name", "atom_index"], keep="first"
).reset_index(drop=True)
mulliken = mulliken.drop_duplicates(
    subset=["molecule_name", "atom_index"], keep="first"
).reset_index(drop=True)
mag = mag.drop_duplicates(
    subset=["molecule_name", "atom_index"], keep="first"
).reset_index(drop=True)

s0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "x_0",
        "y": "y_0",
        "z": "z_0",
    }
)
s1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "x": "x_1",
        "y": "y_1",
        "z": "z_1",
    }
)

q0 = mulliken.rename(columns={"atom_index": "atom_index_0", "mulliken_charge": "q0"})
q1 = mulliken.rename(columns={"atom_index": "atom_index_1", "mulliken_charge": "q1"})

mag_use_cols = ["XX", "YY", "ZZ"]
mag0 = mag[["molecule_name", "atom_index"] + mag_use_cols].copy()
mag0["trace"] = mag0["XX"] + mag0["YY"] + mag0["ZZ"]
mag0 = mag0.rename(
    columns={
        "atom_index": "atom_index_0",
        "XX": "ms_xx0",
        "YY": "ms_yy0",
        "ZZ": "ms_zz0",
        "trace": "ms_tr0",
    }
)

mag1 = mag[["molecule_name", "atom_index"] + mag_use_cols].copy()
mag1["trace"] = mag1["XX"] + mag1["YY"] + mag1["ZZ"]
mag1 = mag1.rename(
    columns={
        "atom_index": "atom_index_1",
        "XX": "ms_xx1",
        "YY": "ms_yy1",
        "ZZ": "ms_zz1",
        "trace": "ms_tr1",
    }
)


def _build_atom_neighbor_stats(struct_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["molecule_name", "atom_index", "x", "y", "z"]
    sdf = struct_df[cols].copy()
    sdf["x"] = sdf["x"].astype(np.float32, copy=False)
    sdf["y"] = sdf["y"].astype(np.float32, copy=False)
    sdf["z"] = sdf["z"].astype(np.float32, copy=False)

    out = []
    for mol, g in sdf.groupby("molecule_name", sort=False):
        xyz = g[["x", "y", "z"]].to_numpy(np.float32, copy=False)
        idx = g["atom_index"].to_numpy(np.int32, copy=False)
        n = xyz.shape[0]

        if n <= 1:
            dmin = np.full(n, np.nan, dtype=np.float32)
            dmean = np.full(n, np.nan, dtype=np.float32)
            dmax = np.full(n, np.nan, dtype=np.float32)
        else:
            tree = KDTree(xyz, leaf_size=40)
            dist, _ = tree.query(xyz, k=n, return_distance=True)
            dist = dist.astype(np.float32, copy=False)
            d_oth = dist[:, 1:]
            dmin = np.min(d_oth, axis=1).astype(np.float32, copy=False)
            dmean = np.mean(d_oth, axis=1).astype(np.float32, copy=False)
            dmax = np.max(d_oth, axis=1).astype(np.float32, copy=False)

        out.append(
            pd.DataFrame(
                {
                    "molecule_name": mol,
                    "atom_index": idx,
                    "nbr_min_dist": dmin,
                    "nbr_mean_dist": dmean,
                    "nbr_max_dist": dmax,
                }
            )
        )

    atom_stats = pd.concat(out, axis=0, ignore_index=True)
    atom_stats["molecule_name"] = atom_stats["molecule_name"].astype(str)
    atom_stats["atom_index"] = atom_stats["atom_index"].astype(np.int32, copy=False)
    atom_stats = atom_stats.drop_duplicates(
        subset=["molecule_name", "atom_index"], keep="first"
    ).reset_index(drop=True)
    return atom_stats


atom_stats = _build_atom_neighbor_stats(structures)
atom_stats0 = atom_stats.rename(
    columns={
        "atom_index": "atom_index_0",
        "nbr_min_dist": "nbr_min0",
        "nbr_mean_dist": "nbr_mean0",
        "nbr_max_dist": "nbr_max0",
    }
)
atom_stats1 = atom_stats.rename(
    columns={
        "atom_index": "atom_index_1",
        "nbr_min_dist": "nbr_min1",
        "nbr_mean_dist": "nbr_mean1",
        "nbr_max_dist": "nbr_max1",
    }
)


def _merge_structures(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.merge(
        s0[["molecule_name", "atom_index_0", "atom_0", "x_0", "y_0", "z_0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
        validate="many_to_one",
    )
    df = df.merge(
        s1[["molecule_name", "atom_index_1", "atom_1", "x_1", "y_1", "z_1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
        validate="many_to_one",
    )
    return df


def _merge_mulliken(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.merge(
        q0[["molecule_name", "atom_index_0", "q0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
        validate="many_to_one",
    )
    df = df.merge(
        q1[["molecule_name", "atom_index_1", "q1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
        validate="many_to_one",
    )
    return df


def _merge_mag(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.merge(
        mag0[["molecule_name", "atom_index_0", "ms_xx0", "ms_yy0", "ms_zz0", "ms_tr0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
        validate="many_to_one",
    )
    df = df.merge(
        mag1[["molecule_name", "atom_index_1", "ms_xx1", "ms_yy1", "ms_zz1", "ms_tr1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
        validate="many_to_one",
    )
    return df


def _merge_atom_stats(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.merge(
        atom_stats0[
            ["molecule_name", "atom_index_0", "nbr_min0", "nbr_mean0", "nbr_max0"]
        ],
        on=["molecule_name", "atom_index_0"],
        how="left",
        validate="many_to_one",
    )
    df = df.merge(
        atom_stats1[
            ["molecule_name", "atom_index_1", "nbr_min1", "nbr_mean1", "nbr_max1"]
        ],
        on=["molecule_name", "atom_index_1"],
        how="left",
        validate="many_to_one",
    )
    return df


def add_pair_features(df_in: pd.DataFrame) -> pd.DataFrame:
    df = _merge_structures(df_in)

    check_cols = ["x_0", "y_0", "z_0", "x_1", "y_1", "z_1", "atom_0", "atom_1"]
    bad = df[check_cols].isna().any(axis=1).sum()

    if bad > 0:
        df_try = df_in.copy()
        df_try["atom_index_0"] = np.maximum(df_try["atom_index_0"] - 1, 0).astype(
            np.int32, copy=False
        )
        df_try["atom_index_1"] = np.maximum(df_try["atom_index_1"] - 1, 0).astype(
            np.int32, copy=False
        )
        df2 = _merge_structures(df_try)
        bad2 = df2[check_cols].isna().any(axis=1).sum()

        if bad2 < bad:
            df = df2
            bad = bad2
            print(f"Applied atom_index - 1 correction; remaining missing merges: {bad}")
        else:
            print(
                f"No improvement from atom_index - 1 correction; remaining missing merges: {bad}"
            )

    dx = df["x_0"] - df["x_1"]
    dy = df["y_0"] - df["y_1"]
    dz = df["z_0"] - df["z_1"]
    df["dx"] = dx
    df["dy"] = dy
    df["dz"] = dz
    df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz)

    df["abs_dx"] = dx.abs()
    df["abs_dy"] = dy.abs()
    df["abs_dz"] = dz.abs()

    df["dist2"] = df["dist"] ** 2
    df["inv_dist"] = 1.0 / (df["dist"] + 1e-9)

    df = _merge_mulliken(df)
    df["qsum"] = df["q0"] + df["q1"]
    df["qdiff"] = (df["q0"] - df["q1"]).abs()
    df["qprod"] = df["q0"] * df["q1"]

    df = _merge_mag(df)
    df["ms_tr_sum"] = df["ms_tr0"] + df["ms_tr1"]
    df["ms_tr_diff"] = (df["ms_tr0"] - df["ms_tr1"]).abs()
    df["ms_xx_sum"] = df["ms_xx0"] + df["ms_xx1"]
    df["ms_xx_diff"] = (df["ms_xx0"] - df["ms_xx1"]).abs()
    df["ms_yy_sum"] = df["ms_yy0"] + df["ms_yy1"]
    df["ms_yy_diff"] = (df["ms_yy0"] - df["ms_yy1"]).abs()
    df["ms_zz_sum"] = df["ms_zz0"] + df["ms_zz1"]
    df["ms_zz_diff"] = (df["ms_zz0"] - df["ms_zz1"]).abs()

    df = _merge_atom_stats(df)
    df["nbr_min_sum"] = df["nbr_min0"] + df["nbr_min1"]
    df["nbr_min_diff"] = (df["nbr_min0"] - df["nbr_min1"]).abs()
    df["nbr_mean_sum"] = df["nbr_mean0"] + df["nbr_mean1"]
    df["nbr_mean_diff"] = (df["nbr_mean0"] - df["nbr_mean1"]).abs()
    df["nbr_max_sum"] = df["nbr_max0"] + df["nbr_max1"]
    df["nbr_max_diff"] = (df["nbr_max0"] - df["nbr_max1"]).abs()
    df["dist_over_nbrmin0"] = df["dist"] / (df["nbr_min0"] + 1e-9)
    df["dist_over_nbrmin1"] = df["dist"] / (df["nbr_min1"] + 1e-9)

    return df


train_f = add_pair_features(train)
test_f = add_pair_features(test)

check_cols2 = [
    "x_0",
    "y_0",
    "z_0",
    "x_1",
    "y_1",
    "z_1",
    "dist",
    "atom_0",
    "atom_1",
    "q0",
    "q1",
    "ms_tr0",
    "ms_tr1",
    "nbr_min0",
    "nbr_min1",
    "nbr_mean0",
    "nbr_mean1",
]
bad_train = train_f[check_cols2].isna().any(axis=1).sum()
bad_test = test_f[check_cols2].isna().any(axis=1).sum()
print(
    "Rows with any missing merged structure/charge/tensor/neighborhood fields - train:",
    bad_train,
    "test:",
    bad_test,
)

if bad_train > 0 or bad_test > 0:
    print(
        "Warning: still found some NaNs after merge(s). Will impute numeric NaNs later; "
        "categorical NaNs will be treated as 'unknown' via fill."
    )
    train_f["atom_0"] = train_f["atom_0"].fillna("X")
    train_f["atom_1"] = train_f["atom_1"].fillna("X")
    test_f["atom_0"] = test_f["atom_0"].fillna("X")
    test_f["atom_1"] = test_f["atom_1"].fillna("X")



## === cell 3
cat_cols = ["type", "atom_0", "atom_1"]

num_cols = [
    "dx",
    "dy",
    "dz",
    "abs_dx",
    "abs_dy",
    "abs_dz",
    "dist",
    "dist2",
    "inv_dist",
    "q0",
    "q1",
    "qsum",
    "qdiff",
    "qprod",
    "ms_xx0",
    "ms_yy0",
    "ms_zz0",
    "ms_tr0",
    "ms_xx1",
    "ms_yy1",
    "ms_zz1",
    "ms_tr1",
    "ms_tr_sum",
    "ms_tr_diff",
    "ms_xx_sum",
    "ms_xx_diff",
    "ms_yy_sum",
    "ms_yy_diff",
    "ms_zz_sum",
    "ms_zz_diff",
    "nbr_min0",
    "nbr_mean0",
    "nbr_max0",
    "nbr_min1",
    "nbr_mean1",
    "nbr_max1",
    "nbr_min_sum",
    "nbr_min_diff",
    "nbr_mean_sum",
    "nbr_mean_diff",
    "nbr_max_sum",
    "nbr_max_diff",
    "dist_over_nbrmin0",
    "dist_over_nbrmin1",
]

all_df = pd.concat(
    [train_f[["id"] + cat_cols + num_cols], test_f[["id"] + cat_cols + num_cols]],
    axis=0,
    ignore_index=True,
)

num_mean = all_df[num_cols].mean(axis=0)
num_std = all_df[num_cols].std(axis=0).replace(0.0, 1.0)
all_df[num_cols] = (all_df[num_cols] - num_mean) / num_std

all_df = pd.get_dummies(all_df, columns=cat_cols, dummy_na=False)

X_all = all_df.drop(columns=["id"])
if X_all.isna().any().any():
    med = X_all.median(axis=0)
    X_all = X_all.fillna(med)

X_train = X_all.iloc[: len(train_f), :].to_numpy(dtype=np.float32, copy=False)
X_test = X_all.iloc[len(train_f) :, :].to_numpy(dtype=np.float32, copy=False)

y_train = train_f["scalar_coupling_constant"].to_numpy(dtype=np.float32, copy=False)

print("X_train:", X_train.shape, "X_test:", X_test.shape, "y_train:", y_train.shape)



## === cell 4
type_train = train_f["type"].values
type_test = test_f["type"].values

unique_types = np.unique(type_train)

pred_test = np.zeros(len(test_f), dtype=np.float32)
pred_train_in = np.zeros(len(train_f), dtype=np.float32)

for t in unique_types:
    tr_idx = np.where(type_train == t)[0]
    te_idx = np.where(type_test == t)[0]
    if len(te_idx) == 0:
        continue

    model = Ridge(alpha=1.0, random_state=0)
    model.fit(X_train[tr_idx], y_train[tr_idx])

    pred_train_in[tr_idx] = model.predict(X_train[tr_idx]).astype(np.float32)
    pred_test[te_idx] = model.predict(X_test[te_idx]).astype(np.float32)

train_type_mean = train_f.groupby("type")["scalar_coupling_constant"].mean().to_dict()

mol_names = train_f["molecule_name"].astype(str).values
mol_hash = pd.util.hash_pandas_object(pd.Series(mol_names), index=False).to_numpy()
is_val = (mol_hash % 5) == 0  # ~20% validation, deterministic

w_by_type = {}
for t in unique_types:
    idx_t = np.where(type_train == t)[0]
    idx_val = idx_t[is_val[idx_t]]
    if len(idx_val) < 1000:
        w_by_type[t] = 0.98
        continue

    yv = y_train[idx_val]
    pv = pred_train_in[idx_val]
    mu = np.float32(train_type_mean.get(t, 0.0))

    ws = np.array([0.90, 0.94, 0.96, 0.98, 1.00, 1.02, 1.04], dtype=np.float32)

    maes = []
    for w in ws:
        cal = w * pv + (1.0 - w) * mu
        maes.append(np.mean(np.abs(yv - cal)))
    w_best = float(ws[int(np.argmin(maes))])
    w_by_type[t] = w_best

print("Per-type calibration weights (sample):", dict(list(w_by_type.items())[:5]))

pred_test_cal = pred_test.copy()
for t in unique_types:
    te_idx = np.where(type_test == t)[0]
    if len(te_idx) == 0:
        continue
    w = np.float32(w_by_type.get(t, 0.98))
    mu = np.float32(train_type_mean.get(t, 0.0))
    pred_test_cal[te_idx] = w * pred_test[te_idx] + (1.0 - w) * mu

mol0 = pred_test.astype(np.float32)
mol1 = np.array(
    [
        0.95 * p + 0.05 * train_type_mean.get(t, 0.0)
        for p, t in zip(pred_test, type_test)
    ],
    dtype=np.float32,
)
mol2 = pred_test_cal.astype(np.float32)

concat_sub = pd.DataFrame(
    {"id": test_f["id"].values, "mol0": mol0, "mol1": mol1, "mol2": mol2}
)
ncol = concat_sub.shape[1]
print(concat_sub.head())



## === cell 5
try:
    corr = concat_sub[["mol0", "mol1", "mol2"]].corr()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        corr,
        mask=mask,
        cmap=sns.diverging_palette(220, 10, as_cmap=True),
        center=0,
        annot=True,
    )
    plt.tight_layout()
except Exception as e:
    print("Skipped correlation plot due to:", repr(e))



## === cell 6
concat_sub["m_max"] = concat_sub.iloc[:, 1:ncol].max(axis=1)
concat_sub["m_min"] = concat_sub.iloc[:, 1:ncol].min(axis=1)
concat_sub["m_mean"] = concat_sub.iloc[:, 1:ncol].mean(axis=1)
concat_sub["m_median"] = concat_sub.iloc[:, 1:ncol].median(axis=1)

print(
    concat_sub.describe().loc[
        ["mean", "std", "min", "max"], ["mol0", "mol1", "mol2", "m_mean", "m_median"]
    ]
)



## === cell 7
cutoff_lo = 0.2
cutoff_hi = 0.8



## === cell 8
concat_sub["scalar_coupling_constant"] = concat_sub["mol2"]
concat_sub[["id", "scalar_coupling_constant"]].to_csv(
    "stack_calibrated.csv", index=False, float_format="%.6f"
)

concat_sub["scalar_coupling_constant"] = concat_sub["m_mean"]
concat_sub[["id", "scalar_coupling_constant"]].to_csv(
    "stack_mean.csv", index=False, float_format="%.6f"
)
concat_sub["scalar_coupling_constant"] = concat_sub["m_median"]
concat_sub[["id", "scalar_coupling_constant"]].to_csv(
    "stack_median.csv", index=False, float_format="%.6f"
)

final_pred = pd.read_csv("stack_calibrated.csv")

sub = sample_sub[["id"]].merge(final_pred, on="id", how="left")
if sub["scalar_coupling_constant"].isna().any():
    raise ValueError(
        "Submission has missing predictions after merge; id alignment issue."
    )

sub.to_csv("submission.csv", index=False, float_format="%.6f")
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())

try:
    plt.figure(figsize=(7, 3))
    sns.histplot(sub["scalar_coupling_constant"], bins=50, kde=False)
    plt.tight_layout()
except Exception as e:
    print("Skipped histogram plot due to:", repr(e))
