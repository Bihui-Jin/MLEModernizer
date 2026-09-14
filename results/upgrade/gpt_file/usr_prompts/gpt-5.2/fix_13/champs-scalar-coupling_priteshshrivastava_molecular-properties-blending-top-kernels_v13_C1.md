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

-1.666828640654786

# 6. Current score

1.22409

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.95983) has done: 'I remove the dependency on missing external blend files (which cause the `FileNotFoundError`) and replace it with a self-contained baseline that uses only the provided competition data. To keep the solution minimal and robust, the script build a simple, legitimate feature set from `structures.csv` (pairwise distance + atom types + coupling type) and train a regression model. This run end-to-end within the Kaggle environment and write a valid `submission.csv` with the required `id` and `scalar_coupling_constant` columns. The approach is score-improving versus an invalid submission, while staying simple and stable.'
- What this solution (achieved 1.2362) has done: 'Your current score (1.95983, lower-is-better) is far worse than the target (-1.6668), so we should legitimately improve the model while keeping the same overall approach (pairwise geometry + atom/type categoricals + tree-based regression). The biggest issue is that the competition metric averages MAE per coupling `type`, but your single global model implicitly overweights frequent types; training and predicting per-type with the same features/model usually improves the metric a lot with minimal logic change. I also switch the validation split to be by `molecule_name` (as the competition split is by molecule) to better match generalization and avoid leakage-like optimism; this doesn’t change submission semantics, but it guides safer choices. Finally, I keep the same estimator family and preprocessing, but use a slightly stronger (still fast) set of HGB hyperparameters and train separate models per type, then write a standard `submission.csv`.'
- What this solution (achieved 1.70834) has done: 'I fix the runtime error caused by calling `.toarray()` on a NumPy array by adding a small helper that converts sparse matrices to dense arrays only when needed, and then use it consistently for both training and prediction. This keeps your exact modeling approach (global preprocessing + per-type HGB models) unchanged while making the code compatible with both sparse and dense outputs from the `ColumnTransformer`. I also update the `OneHotEncoder` argument for broad sklearn compatibility by falling back to `sparse=` if `sparse_output=` is not supported, without changing semantics. Finally, I ensure the pipeline completes and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 1.22257) has done: 'Your current score (1.70834, lower-is-better) is far worse than the target (-1.6668), so we should legitimately improve with minimal semantic changes. The biggest low-risk gain is to stop training the “global” preprocessor on all of `train` (which can hurt generalization) and instead fit preprocessing and the per-type model together per type (so the one-hot categories and imputations are type-specific and not distorted by unrelated types). We keep the exact same feature set, the same model family (HistGradientBoostingRegressor with absolute_error), and the same per-type training approach, but we do a molecule-group split per type for an informational CV and then refit on full per-type data for test prediction. Rare types still fall back to a single global pipeline to keep runtime bounded and stable, but that global pipeline be fit only once and used only for rare types.'
- What this solution (achieved 1.20971) has done: 'Your current score (1.22257, lower-is-better) is still far from the target (-1.6668), so we should legitimately improve error with minimal disruption to your existing pipeline (same features, same per-type HGB model family, same loss). The biggest low-risk gain is to add a small set of well-known CHAMPS auxiliary features by joining `mulliken_charges.csv`, `magnetic_shielding_tensors.csv`, `dipole_moments.csv`, and `potential_energy.csv` onto the existing atom-pair table; this preserves the learning approach but gives the model much more signal. To keep semantics and runtime stable, we keep the same training/prediction flow (global fallback + per-type refit), and we only expand the numeric feature columns. We also clip the test predictions per type to the observed training target range for that type (a minimal post-processing that often reduces MAE under this metric without changing the core model).'
- What this solution (achieved 1.20971) has done: 'Your current score (1.20971, lower-is-better) is still far worse than the target (-1.6668), so we should improve the model while keeping the same overall approach (same features, per-type HGB, same loss). The smallest high-impact issue is that `HistGradientBoostingRegressor` does not benefit from sparse one-hot features; forcing the `ColumnTransformer` to output dense allows the model to properly use the categorical signals, which typically reduces MAE a lot in this competition. I also make the per-type training more consistent by ensuring the global fallback pipeline uses the same dense preprocessing (so rare types aren’t disadvantaged). No feature definitions, model family, or training/prediction flow is changed—only the preprocessor output format is adjusted to better match the estimator.'
- What this solution (achieved 1.20971) has done: 'Your score (1.20971, lower-is-better) is still far from the target (-1.6668), so we should legitimately reduce MAE with minimal disruption to your current approach (same features, same per-type HGB, same loss). The most direct, low-risk fix is to align with the competition’s metric by training a separate model per coupling `type` without one-hot encoding `type` itself (since it becomes constant within each per-type model and can add noise/complexity). Additionally, switching the evaluation-time clipping to a robust per-type percentile range (instead of strict min/max) usually reduces MAE by avoiding extreme outliers without changing the modeling core. These changes keep the same overall pipeline and estimator family, but better match how the leaderboard metric is computed.'
- What this solution (achieved 1.21025) has done: 'We keep your exact per-type HGB approach and feature set, but fix two small issues that typically hurt this competition’s metric: (1) add a tiny set of “orientation-invariant” geometric features (pair center coordinates and radial distances to origin) that are derivable from the same structures data and often reduce MAE without changing modeling semantics, and (2) change the prediction clipping from extreme quantiles (0.1%/99.9%) to a slightly tighter, more robust range (0.5%/99.5%) to reduce outlier-driven MAE per type. We also remove the now-useless `type` column from the learning features (since you already train per-type) to reduce noise/overfitting risk with minimal change. Everything still trains per type (with the same model family/loss) and writes a valid `submission.csv`.'
- What this solution (achieved 1.22409) has done: 'The timeout is dominated by `add_knn_features`: it recomputes a full pairwise distance matrix per molecule twice (train+test), repeatedly rebuilds per-molecule groupby objects, and fills the dataframe via many `.loc` assignments inside Python loops. I keep the exact same KNN feature definition (3 nearest-neighbor distances per atom, plus mean/min) but compute it once per molecule from `structures.csv`, cache the per-atom results, and then merge/join onto train/test in a vectorized way. I also avoid category→string conversions and repeated groupby creation by using integer-coded molecule IDs and a single `groupby(sort=False)` over structures. Model training/prediction logic stays the same; only feature engineering is made asymptotically/constant-factor faster.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

DATA_DIR = "/kaggle/data/champs-scalar-coupling"

print("DATA_DIR exists:", os.path.isdir(DATA_DIR))
print("Files:", sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])[:10])




## === cell 1
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
structures_path = os.path.join(DATA_DIR, "structures.csv")
mulliken_path = os.path.join(DATA_DIR, "mulliken_charges.csv")
shield_path = os.path.join(DATA_DIR, "magnetic_shielding_tensors.csv")
dipole_path = os.path.join(DATA_DIR, "dipole_moments.csv")
energy_path = os.path.join(DATA_DIR, "potential_energy.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

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
        "id": "int32",
        "molecule_name": "category",
        "atom_index_0": "int16",
        "atom_index_1": "int16",
        "type": "category",
        "scalar_coupling_constant": "float32",
    },
)
test = pd.read_csv(
    test_path,
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
    dtype={
        "id": "int32",
        "molecule_name": "category",
        "atom_index_0": "int16",
        "atom_index_1": "int16",
        "type": "category",
    },
)
structures = pd.read_csv(
    structures_path,
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
    dtype={
        "molecule_name": "category",
        "atom_index": "int16",
        "atom": "category",
        "x": "float32",
        "y": "float32",
        "z": "float32",
    },
)

mulliken = pd.read_csv(
    mulliken_path,
    usecols=["molecule_name", "atom_index", "mulliken_charge"],
    dtype={
        "molecule_name": "category",
        "atom_index": "int16",
        "mulliken_charge": "float32",
    },
)
shield = pd.read_csv(
    shield_path,
    usecols=[
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
    ],
    dtype={
        "molecule_name": "category",
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
    usecols=["molecule_name", "X", "Y", "Z"],
    dtype={"molecule_name": "category", "X": "float32", "Y": "float32", "Z": "float32"},
)
energy = pd.read_csv(
    energy_path,
    usecols=["molecule_name", "potential_energy"],
    dtype={"molecule_name": "category", "potential_energy": "float32"},
)

print(train.shape, test.shape, structures.shape)
print("Aux shapes:", mulliken.shape, shield.shape, dipole.shape, energy.shape)




## === cell 2
structures_idx = structures.set_index(["molecule_name", "atom_index"])[
    ["atom", "x", "y", "z"]
]

mulliken_idx = mulliken.set_index(["molecule_name", "atom_index"])[["mulliken_charge"]]
shield_idx = shield.set_index(["molecule_name", "atom_index"])[
    ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]
]
dipole_idx = dipole.set_index("molecule_name")[["X", "Y", "Z"]]
energy_idx = energy.set_index("molecule_name")[["potential_energy"]]


def _neighbor_dists_for_molecule(atom_index_arr, xyz_arr, k=3):
    n = xyz_arr.shape[0]
    out = np.full((n, k), np.nan, dtype=np.float32)
    if n <= 1:
        return out

    G = (xyz_arr @ xyz_arr.T).astype(np.float32)
    sq = np.sum(xyz_arr * xyz_arr, axis=1, dtype=np.float32)
    d2 = (sq[:, None] + sq[None, :] - 2.0 * G).astype(np.float32)
    d2[d2 < 0] = 0.0
    np.fill_diagonal(d2, np.inf)

    kk = min(k, n - 1)
    part = np.argpartition(d2, kth=kk - 1, axis=1)[:, :kk]
    row = np.arange(n)[:, None]
    vals = d2[row, part]
    order = np.argsort(vals, axis=1)
    nn_d2 = vals[row, order]
    nn = np.sqrt(nn_d2, dtype=np.float32)

    out[:, :kk] = nn.astype(np.float32)
    return out


def precompute_knn_table(structures_df: pd.DataFrame, k=3) -> pd.DataFrame:
    mol_codes = structures_df["molecule_name"].cat.codes.to_numpy(np.int32, copy=False)
    atom_index = structures_df["atom_index"].to_numpy(np.int16, copy=False)
    xyz = structures_df[["x", "y", "z"]].to_numpy(np.float32, copy=False)

    order = np.argsort(mol_codes, kind="mergesort")
    mol_codes_s = mol_codes[order]
    atom_index_s = atom_index[order]
    xyz_s = xyz[order]

    out_sorted = np.empty(
        (len(structures_df), k + 2), dtype=np.float32
    )  # nn1..nnk, mean, min
    out_sorted.fill(np.nan)

    change = np.nonzero(np.diff(mol_codes_s))[0] + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(mol_codes_s)]

    for st, en in zip(starts, ends):
        n = en - st
        if n <= 0:
            continue
        nn = _neighbor_dists_for_molecule(
            atom_index_s[st:en], xyz_s[st:en], k=k
        )  # (n,k)
        out_sorted[st:en, :k] = nn
        out_sorted[st:en, k] = np.nanmean(nn, axis=1).astype(np.float32)
        out_sorted[st:en, k + 1] = np.nanmin(nn, axis=1).astype(np.float32)

    out = np.empty_like(out_sorted)
    out[order] = out_sorted

    cols = [f"nn{i}" for i in range(1, k + 1)] + ["nn_mean", "nn_min"]
    knn_tbl = structures_df[["molecule_name", "atom_index"]].copy()
    for j, c in enumerate(cols):
        knn_tbl[c] = out[:, j]
    return knn_tbl


def add_knn_features(df: pd.DataFrame, knn_tbl: pd.DataFrame, k=3) -> pd.DataFrame:
    df = df.copy()

    left0 = df[["molecule_name", "atom_index_0"]].rename(
        columns={"atom_index_0": "atom_index"}
    )
    left1 = df[["molecule_name", "atom_index_1"]].rename(
        columns={"atom_index_1": "atom_index"}
    )

    knn0 = knn_tbl.rename(
        columns={
            c: f"a0_{c}"
            for c in knn_tbl.columns
            if c not in ("molecule_name", "atom_index")
        }
    )
    knn1 = knn_tbl.rename(
        columns={
            c: f"a1_{c}"
            for c in knn_tbl.columns
            if c not in ("molecule_name", "atom_index")
        }
    )

    df = df.merge(
        pd.concat([left0, left0.index.to_series(name="_row")], axis=1)
        .merge(knn0, on=["molecule_name", "atom_index"], how="left")
        .drop(columns=["molecule_name", "atom_index"])
        .set_index("_row"),
        left_index=True,
        right_index=True,
        how="left",
    )
    df = df.merge(
        pd.concat([left1, left1.index.to_series(name="_row")], axis=1)
        .merge(knn1, on=["molecule_name", "atom_index"], how="left")
        .drop(columns=["molecule_name", "atom_index"])
        .set_index("_row"),
        left_index=True,
        right_index=True,
        how="left",
    )

    for side in ("0", "1"):
        for i in range(1, k + 1):
            col = f"a{side}_nn{i}"
            if col not in df.columns:
                df[col] = np.nan
        for col in (f"a{side}_nn_mean", f"a{side}_nn_min"):
            if col not in df.columns:
                df[col] = np.nan

    return df


def add_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    mi0 = pd.MultiIndex.from_frame(
        df[["molecule_name", "atom_index_0"]].rename(
            columns={"atom_index_0": "atom_index"}
        )
    )
    mi1 = pd.MultiIndex.from_frame(
        df[["molecule_name", "atom_index_1"]].rename(
            columns={"atom_index_1": "atom_index"}
        )
    )

    a0 = (
        structures_idx.reindex(mi0)
        .rename(columns={"atom": "atom_0", "x": "x0", "y": "y0", "z": "z0"})
        .reset_index(drop=True)
    )
    a1 = (
        structures_idx.reindex(mi1)
        .rename(columns={"atom": "atom_1", "x": "x1", "y": "y1", "z": "z1"})
        .reset_index(drop=True)
    )

    df = pd.concat([df.reset_index(drop=True), a0, a1], axis=1)

    dx = df["x0"].to_numpy(dtype=np.float32) - df["x1"].to_numpy(dtype=np.float32)
    dy = df["y0"].to_numpy(dtype=np.float32) - df["y1"].to_numpy(dtype=np.float32)
    dz = df["z0"].to_numpy(dtype=np.float32) - df["z1"].to_numpy(dtype=np.float32)

    df["dx"] = dx
    df["dy"] = dy
    df["dz"] = dz
    df["abs_dx"] = np.abs(dx)
    df["abs_dy"] = np.abs(dy)
    df["abs_dz"] = np.abs(dz)

    dist = np.sqrt(dx * dx + dy * dy + dz * dz, dtype=np.float32)
    df["distance"] = dist
    df["inv_distance"] = 1.0 / (dist + np.float32(1e-6))
    df["distance2"] = dist * dist

    cx = (df["x0"].to_numpy(np.float32) + df["x1"].to_numpy(np.float32)) * np.float32(
        0.5
    )
    cy = (df["y0"].to_numpy(np.float32) + df["y1"].to_numpy(np.float32)) * np.float32(
        0.5
    )
    cz = (df["z0"].to_numpy(np.float32) + df["z1"].to_numpy(np.float32)) * np.float32(
        0.5
    )
    df["center_x"] = cx
    df["center_y"] = cy
    df["center_z"] = cz
    df["center_norm"] = np.sqrt(cx * cx + cy * cy + cz * cz, dtype=np.float32)
    r0 = np.sqrt(
        df["x0"].to_numpy(np.float32) ** 2
        + df["y0"].to_numpy(np.float32) ** 2
        + df["z0"].to_numpy(np.float32) ** 2,
        dtype=np.float32,
    )
    r1 = np.sqrt(
        df["x1"].to_numpy(np.float32) ** 2
        + df["y1"].to_numpy(np.float32) ** 2
        + df["z1"].to_numpy(np.float32) ** 2,
        dtype=np.float32,
    )
    df["r0"] = r0
    df["r1"] = r1
    df["r_sum"] = r0 + r1
    df["r_diff"] = r0 - r1
    df["abs_r_diff"] = np.abs(df["r_diff"].to_numpy(np.float32))

    m0 = (
        mulliken_idx.reindex(mi0)
        .rename(columns={"mulliken_charge": "mulliken_0"})
        .reset_index(drop=True)
    )
    m1 = (
        mulliken_idx.reindex(mi1)
        .rename(columns={"mulliken_charge": "mulliken_1"})
        .reset_index(drop=True)
    )
    df = pd.concat([df, m0, m1], axis=1)
    df["mulliken_sum"] = df["mulliken_0"].to_numpy(np.float32) + df[
        "mulliken_1"
    ].to_numpy(np.float32)
    df["mulliken_diff"] = df["mulliken_0"].to_numpy(np.float32) - df[
        "mulliken_1"
    ].to_numpy(np.float32)
    df["abs_mulliken_diff"] = np.abs(df["mulliken_diff"].to_numpy(np.float32))

    s0 = shield_idx.reindex(mi0).add_prefix("shield0_").reset_index(drop=True)
    s1 = shield_idx.reindex(mi1).add_prefix("shield1_").reset_index(drop=True)
    df = pd.concat([df, s0, s1], axis=1)

    df["shield0_trace"] = (
        df["shield0_XX"].to_numpy(np.float32)
        + df["shield0_YY"].to_numpy(np.float32)
        + df["shield0_ZZ"].to_numpy(np.float32)
    )
    df["shield1_trace"] = (
        df["shield1_XX"].to_numpy(np.float32)
        + df["shield1_YY"].to_numpy(np.float32)
        + df["shield1_ZZ"].to_numpy(np.float32)
    )
    df["shield_trace_sum"] = df["shield0_trace"].to_numpy(np.float32) + df[
        "shield1_trace"
    ].to_numpy(np.float32)
    df["shield_trace_diff"] = df["shield0_trace"].to_numpy(np.float32) - df[
        "shield1_trace"
    ].to_numpy(np.float32)
    df["abs_shield_trace_diff"] = np.abs(df["shield_trace_diff"].to_numpy(np.float32))

    d = (
        dipole_idx.reindex(df["molecule_name"])
        .rename(columns={"X": "dipole_x", "Y": "dipole_y", "Z": "dipole_z"})
        .reset_index(drop=True)
    )
    e = energy_idx.reindex(df["molecule_name"]).reset_index(drop=True)
    df = pd.concat([df, d, e], axis=1)
    df["dipole_norm"] = np.sqrt(
        df["dipole_x"].to_numpy(np.float32) ** 2
        + df["dipole_y"].to_numpy(np.float32) ** 2
        + df["dipole_z"].to_numpy(np.float32) ** 2,
        dtype=np.float32,
    )

    return df


knn_tbl = precompute_knn_table(structures, k=3)

train_f = add_structure_features(train)
test_f = add_structure_features(test)

train_f = add_knn_features(train_f, knn_tbl, k=3)
test_f = add_knn_features(test_f, knn_tbl, k=3)

feature_cols_num = [
    "distance",
    "inv_distance",
    "distance2",
    "dx",
    "dy",
    "dz",
    "abs_dx",
    "abs_dy",
    "abs_dz",
    "center_x",
    "center_y",
    "center_z",
    "center_norm",
    "r0",
    "r1",
    "r_sum",
    "r_diff",
    "abs_r_diff",
    "mulliken_0",
    "mulliken_1",
    "mulliken_sum",
    "mulliken_diff",
    "abs_mulliken_diff",
    "shield0_trace",
    "shield1_trace",
    "shield_trace_sum",
    "shield_trace_diff",
    "abs_shield_trace_diff",
    "dipole_x",
    "dipole_y",
    "dipole_z",
    "dipole_norm",
    "potential_energy",
    "a0_nn1",
    "a0_nn2",
    "a0_nn3",
    "a0_nn_mean",
    "a0_nn_min",
    "a1_nn1",
    "a1_nn2",
    "a1_nn3",
    "a1_nn_mean",
    "a1_nn_min",
]
feature_cols_cat = ["atom_0", "atom_1"]
target_col = "scalar_coupling_constant"

missing0 = train_f["atom_0"].isna().mean()
missing1 = train_f["atom_1"].isna().mean()
print("Missing atom_0 rate:", missing0, "Missing atom_1 rate:", missing1)
print("Missing mulliken_0 rate:", float(train_f["mulliken_0"].isna().mean()))
print("Missing shield0_XX rate:", float(train_f["shield0_XX"].isna().mean()))
print("Missing a0_nn1 rate:", float(train_f["a0_nn1"].isna().mean()))
print("Unique coupling types (train):", train_f["type"].nunique())




## === cell 3
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, feature_cols_num),
        ("cat", categorical_transformer, feature_cols_cat),
    ],
    remainder="drop",
)

base_model = HistGradientBoostingRegressor(
    loss="absolute_error",
    max_depth=10,
    learning_rate=0.06,
    max_iter=350,
    min_samples_leaf=20,
    l2_regularization=0.0,
    random_state=42,
)


def make_pipeline():
    return Pipeline(steps=[("preprocess", preprocess), ("model", base_model)])


gss = GroupShuffleSplit(n_splits=1, test_size=0.02, random_state=42)

X_all = train_f[feature_cols_num + feature_cols_cat]
y_all = train_f[target_col].astype(np.float32)

train_type = train_f["type"].cat.codes.to_numpy(np.int16, copy=False)
test_type = test_f["type"].cat.codes.to_numpy(np.int16, copy=False)
train_mol = train_f["molecule_name"].cat.codes.to_numpy(np.int32, copy=False)

types = np.unique(train_type)
types.sort()

RARE_MIN_ROWS = 5000

global_pipe = make_pipeline()
global_pipe.fit(X_all, y_all)

val_mae_by_type = {}
val_weighted_mae = 0.0
val_n = 0

for t in types:
    idx_t = np.flatnonzero(train_type == t)
    n_t = idx_t.size
    if n_t == 0:
        continue

    if n_t < RARE_MIN_ROWS or len(np.unique(train_mol[idx_t])) < 3:
        pred_t = global_pipe.predict(X_all.iloc[idx_t])
        mae_t = mean_absolute_error(y_all.iloc[idx_t], pred_t)
        val_mae_by_type[int(t)] = mae_t
        val_weighted_mae += mae_t * n_t
        val_n += n_t
        continue

    groups = train_mol[idx_t]
    tr_rel, va_rel = next(
        gss.split(np.zeros(n_t), y_all.iloc[idx_t].to_numpy(), groups=groups)
    )
    tr_idx = idx_t[tr_rel]
    va_idx = idx_t[va_rel]

    pipe_t = make_pipeline()
    pipe_t.fit(X_all.iloc[tr_idx], y_all.iloc[tr_idx])

    va_pred = pipe_t.predict(X_all.iloc[va_idx])
    mae_t = mean_absolute_error(y_all.iloc[va_idx], va_pred)

    val_mae_by_type[int(t)] = mae_t
    val_weighted_mae += mae_t * va_idx.size
    val_n += va_idx.size

print("Validation MAE by type (first 8):", dict(list(val_mae_by_type.items())[:8]))
print("Validation weighted MAE (informational):", val_weighted_mae / max(val_n, 1))




## === cell 4
X_test_all = test_f[feature_cols_num + feature_cols_cat]

test_pred = np.zeros(len(test_f), dtype=np.float32)

type_counts = pd.Series(train_type).value_counts().to_dict()

q_low = 0.005
q_high = 0.995

tmp = pd.DataFrame(
    {
        "type_code": train_type,
        target_col: train_f[target_col].to_numpy(np.float32, copy=False),
    }
)
y_lo_by_type = tmp.groupby("type_code")[target_col].quantile(q_low).to_dict()
y_hi_by_type = tmp.groupby("type_code")[target_col].quantile(q_high).to_dict()

y_global_lo = np.float32(train_f[target_col].quantile(q_low))
y_global_hi = np.float32(train_f[target_col].quantile(q_high))

for t in types:
    test_idx_t = np.flatnonzero(test_type == t)
    if test_idx_t.size == 0:
        continue

    n_train_t = type_counts.get(int(t), 0)
    if n_train_t < RARE_MIN_ROWS:
        preds = global_pipe.predict(X_test_all.iloc[test_idx_t]).astype(np.float32)
    else:
        train_idx_t = np.flatnonzero(train_type == t)
        pipe_t = make_pipeline()
        pipe_t.fit(X_all.iloc[train_idx_t], y_all.iloc[train_idx_t])
        preds = pipe_t.predict(X_test_all.iloc[test_idx_t]).astype(np.float32)

    lo = np.float32(y_lo_by_type.get(int(t), y_global_lo))
    hi = np.float32(y_hi_by_type.get(int(t), y_global_hi))
    preds = np.clip(preds, lo, hi)

    test_pred[test_idx_t] = preds

submission = test[["id"]].copy()
submission["scalar_coupling_constant"] = test_pred

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path, "shape:", submission.shape)
print(submission.head())
print(
    "Pred stats:",
    float(np.min(test_pred)),
    float(np.mean(test_pred)),
    float(np.max(test_pred)),
)

sample = pd.read_csv(sample_path)
print(
    "Sample submission shape:", sample.shape, "Our submission shape:", submission.shape
)
print("Columns match:", list(sample.columns) == list(submission.columns))
