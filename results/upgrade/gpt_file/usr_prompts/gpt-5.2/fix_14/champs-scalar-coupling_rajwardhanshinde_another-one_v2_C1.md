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

-1.5209593019916507

# 6. Current score

1.23566

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'I remove the dependency on missing external Kaggle datasets (the two `../input/.../submission.csv` files) that currently cause the `FileNotFoundError`. To keep the pipeline end-to-end and score-reasonable with minimal logic, I generate predictions directly from the provided `train.csv` by computing the mean `scalar_coupling_constant` per coupling `type`, then apply those means to `test.csv` (falling back to the global mean if needed). This produces a valid `stackers_blend.csv` with the exact required columns and id alignment. The rest of the code structure is kept minimal and consistent with the original intent of creating a submission file.'
- What this solution (achieved 1.23566) has done: 'Your current baseline uses only per-`type` means, which ignores huge variation by atom pair and molecule geometry, so the score is far from the target (lower is better). With minimal change to the core “mean-encoding” approach, we can add a higher-resolution backoff hierarchy: mean by (`type`, `atom_0`, `atom_1`) using the atom symbols from `structures.csv`, then back off to (`type`, `atom_0`), (`type`, `atom_1`), then `type`, then global mean. This stays deterministic, fast, and uses only provided files, and it should reduce MAE substantially (thus lowering log-MAE toward the target) without changing the overall modeling paradigm. The submission writing and id alignment via `sample_submission.csv` are preserved exactly.'
- What this solution (achieved 1.23566) has done: 'Your current hierarchy is still too coarse for this competition because it ignores geometry; we can keep the same “groupby mean-encoding with backoff” core logic but add one strong, minimal feature: the inter-atomic distance computed from `structures.csv` coordinates. Then we add distance-binned means at the highest resolution (`type, atom_0, atom_1, dist_bin`) with smooth backoff to your existing keys, which should materially reduce MAE (and thus lower the log-MAE score toward the target). To avoid overfitting noise from sparse bins without changing the paradigm, we also add simple count-based shrinkage toward the parent mean for the distance-binned level. The submission writing, id alignment via `sample_submission.csv`, and overall deterministic, fast pipeline remain unchanged.'
- What this solution (achieved 1.23566) has done: 'Your current pipeline is still “mean encoding with backoff,” but it likely underperforms because (1) the distance-bin level only shrinks toward the (`type`,`atom_0`,`atom_1`) mean (which is often missing for rare pairs), and (2) it treats swapped atom pairs as different even though coupling constants are symmetric w.r.t. atom order for these keys. I keep the same logic and add a symmetric key by sorting (`atom_0`,`atom_1`) and using a canonical distance-bin model on that key, then shrink that bin mean toward a stronger parent (pair-mean if available, otherwise type-mean) to increase coverage and reduce MAE. I also switch binning from floor to round-to-nearest to reduce edge effects at bin boundaries without changing the overall approach. These are minimal, deterministic changes intended to lower your score (lower is better) toward the target.'
- What this solution (achieved 1.23566) has done: 'Your current score (1.23566, lower-is-better) is far above the target (-1.52096), so we should improve accuracy substantially while keeping the same “groupby mean-encoding with backoff + distance bins + shrinkage” core logic. The smallest high-impact fix is to avoid leakage/overfitting in the encoding tables by building them out-of-fold at the molecule level (the competition splits by molecule), so each training row’s encoding is computed without using its own molecule’s target. Then we can tune only the shrinkage strength and bin width slightly (still the same method) based on a fast molecule-level holdout evaluation to push log-MAE downward toward the target. Submission generation, id alignment via `sample_submission.csv`, and feature construction from `structures.csv` remain unchanged.'
- What this solution (achieved 1.23566) has done: 'Your current approach is already the intended “mean-encoding with backoff + distance bins + shrinkage”, but the biggest remaining accuracy issue is that the distance-bin statistics are built on ordered atom pairs (`atom_0`,`atom_1`) even though scalar couplings are symmetric with respect to swapping the two atoms; this unnecessarily splits data and increases noise. I minimally change the highest-resolution distance-binned table to use the canonical (sorted) atom pair (`a_lo`,`a_hi`) for both train and test, while keeping all backoff tables and shrinkage logic intact. This should materially reduce MAE (thus lower the log-MAE score) without changing the core paradigm, and it stays fast/deterministic and produces the same submission file. I also make the metric computation a bit faster/safer by avoiding `groupby.apply` while keeping identical semantics.'
- What this solution (achieved 1.23566) has done: 'Your current score is far worse than the target (lower-is-better), so the smallest likely-to-help change is to make the encoding/prediction fully symmetric w.r.t. swapping atom_0/atom_1, since scalar couplings are defined on an unordered atom pair while your backoff still uses ordered keys in important places. Concretely, we canonicalize both the atom symbols and the atom indices (and therefore distance is unchanged), then build *all* mean/shrinkage tables on these canonical columns so train/test align and data isn’t split across (A,B) vs (B,A). This preserves the same core “groupby mean-encoding + distance bins + shrinkage + molecule-fold OOF” logic, just with consistent keys, which should reduce MAE and move the score downward toward the target. Submission writing and id alignment remain identical.'
- What this solution (achieved 1.23566) has done: 'Your current score is far worse than the target (lower-is-better), so we should improve accuracy while keeping the exact same “mean-encoding + distance-binned shrinkage + molecule-level folds” core logic. The biggest fix is that your selection step tunes `BIN_WIDTH`/`ALPHA` using only one fold, which is noisy; we instead pick them using proper molecule-level CV (the same folds you already created) and then train final tables on all data. This is a minimal change that usually reduces MAE without changing the modeling paradigm, and it remains deterministic and fast enough by keeping the candidate grid small. I also remove redundant table computations inside `build_tables` (duplicate groupby outputs) to keep runtime within limits while preserving identical predictions.'
- What this solution (achieved 1.23566) has done: 'Your current approach is already a “mean-encoding with backoff + distance bins + shrinkage + molecule-level folds” model, but it’s still leaving a lot of signal unused that’s available in your existing features. I keep the same paradigm and add one higher-resolution level that uses the canonical atom-index pair (`i_lo`,`i_hi`) and a distance bin, with the same shrinkage idea and a safe backoff to the existing (`type`,`a_lo`,`a_hi`) parent means. This is a minimal extension (just another groupby table + merge) that should reduce MAE substantially (lower is better) and move the score toward your target. I also add a tiny epsilon to the shrinkage denominator to avoid any edge-case division issues, without changing semantics.'
- What this solution (achieved 1.23566) has done: 'Your current approach is already strong but it still learns distance-bin means at the atom-index-pair level, which is extremely sparse and likely hurts generalization to unseen molecules (the Kaggle split is by molecule). To move the score down toward the target with minimal change, I keep the same mean-encoding + distance-bin + shrinkage paradigm, but I (1) add a more general “type + atom-pair + raw distance-bin” table as an extra high-resolution level, and (2) shrink each distance-bin level toward the strongest available parent in a consistent hierarchy (idxpair→pair_dist→pair→type), improving coverage without changing evaluation semantics. I also expand the tiny hyperparameter grid slightly (still fast) to better match the optimum without changing the training approach. Submission format, id alignment, and all existing features remain unchanged.'
- What this solution (achieved 1.23566) has done: 'Your current score (1.23566, lower-is-better) is much worse than the target (-1.52096), so we should improve accuracy while keeping the same “groupby mean encoding + distance bins + shrinkage + molecule-fold OOF” core logic. The biggest low-risk gain is to add a simple, chemistry-relevant feature that’s derivable from the same `structures.csv`: the atomic numbers of the two atoms, and then add one extra highest-resolution table keyed by (`type`, `Z_lo`, `Z_hi`, `dist_bin`) with the same shrinkage/backoff scheme. This preserves your training/prediction semantics (still deterministic mean tables with shrinkage and backoff), but typically reduces error substantially because element identity matters beyond the raw symbol string and helps generalization. I also make the shrinkage robust to missing counts by filling missing counts with 0 before the arithmetic (avoids NaNs propagating), which can only improve stability without changing the method.'
- What this solution (achieved 1.23566) has done: 'The timeout is dominated by repeated full-data groupby aggregations and repeated large DataFrame merges during the parameter grid search and folds. I keep the exact same logic (same tables, same shrinkage math, same CV semantics) but make it faster by (1) using categorical dtypes and smaller numeric dtypes to reduce memory and speed up groupby/merge, (2) computing `dist_bin` once per bin width and reusing it, (3) avoiding expensive DataFrame copies/merges in prediction by using index-aligned Series lookups via MultiIndex (equivalent to left merges), and (4) reusing pre-split fold indices/masks to avoid repeated boolean filtering overhead. These changes are algebraically and semantically identical to the original, with only negligible floating-point differences.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

BASE_PATH = "/kaggle/data/champs-scalar-coupling"
train_path = os.path.join(BASE_PATH, "train.csv")
test_path = os.path.join(BASE_PATH, "test.csv")
sample_path = os.path.join(BASE_PATH, "sample_submission.csv")
structures_path = os.path.join(BASE_PATH, "structures.csv")

print("BASE_PATH exists:", os.path.exists(BASE_PATH))
print("Files in BASE_PATH (first 15):", sorted(os.listdir(BASE_PATH))[:15])

RANDOM_STATE = 42
N_FOLDS = 5

train = pd.read_csv(
    train_path,
    usecols=[
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "scalar_coupling_constant",
    ],
    dtype={
        "molecule_name": "string",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "string",
        "scalar_coupling_constant": np.float32,
    },
)
test = pd.read_csv(
    test_path,
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
    dtype={
        "id": np.int32,
        "molecule_name": "string",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "string",
    },
)
sample = pd.read_csv(
    sample_path, usecols=["id", "scalar_coupling_constant"], dtype={"id": np.int32}
)

structures = pd.read_csv(
    structures_path,
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
    dtype={
        "molecule_name": "string",
        "atom_index": np.int16,
        "atom": "string",
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
    },
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

for df in (train, test, s0, s1):
    if "molecule_name" in df.columns:
        df["molecule_name"] = df["molecule_name"].astype("string")

train = train.merge(
    s0[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
)
train = train.merge(
    s1[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
)

test = test.merge(
    s0[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
)
test = test.merge(
    s1[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
)

train["atom_0"] = train["atom_0"].fillna("UNK")
train["atom_1"] = train["atom_1"].fillna("UNK")
test["atom_0"] = test["atom_0"].fillna("UNK")
test["atom_1"] = test["atom_1"].fillna("UNK")

for df in (train, test):
    dx = df["x0"].to_numpy(np.float32, copy=False) - df["x1"].to_numpy(
        np.float32, copy=False
    )
    dy = df["y0"].to_numpy(np.float32, copy=False) - df["y1"].to_numpy(
        np.float32, copy=False
    )
    dz = df["z0"].to_numpy(np.float32, copy=False) - df["z1"].to_numpy(
        np.float32, copy=False
    )
    df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)

global_mean = float(train["scalar_coupling_constant"].mean())

ATOM2Z = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
}


def add_canonical_pair_cols(df: pd.DataFrame) -> pd.DataFrame:
    a0 = df["atom_0"].astype("string")
    a1 = df["atom_1"].astype("string")
    i0 = df["atom_index_0"].astype(np.int16, copy=False)
    i1 = df["atom_index_1"].astype(np.int16, copy=False)

    swap = (a0 > a1) | ((a0 == a1) & (i0 > i1))

    df["a_lo"] = a0.where(~swap, a1)
    df["a_hi"] = a1.where(~swap, a0)
    df["i_lo"] = i0.where(~swap, i1).astype(np.int16)
    df["i_hi"] = i1.where(~swap, i0).astype(np.int16)

    z0 = df["atom_0"].map(ATOM2Z).fillna(0).astype(np.int16)
    z1 = df["atom_1"].map(ATOM2Z).fillna(0).astype(np.int16)
    df["z_lo"] = z0.where(~swap, z1).astype(np.int16)
    df["z_hi"] = z1.where(~swap, z0).astype(np.int16)
    return df


train = add_canonical_pair_cols(train)
test = add_canonical_pair_cols(test)

for df in (train, test):
    df["type"] = df["type"].astype("category")
    df["a_lo"] = df["a_lo"].astype("category")
    df["a_hi"] = df["a_hi"].astype("category")

mols = train["molecule_name"].drop_duplicates().sort_values().to_numpy()
rng = np.random.RandomState(RANDOM_STATE)
perm = rng.permutation(len(mols))
fold_id = np.zeros(len(mols), dtype=np.int8)
for i, idx in enumerate(perm):
    fold_id[idx] = i % N_FOLDS
mol2fold = pd.Series(fold_id, index=mols)

train["fold"] = train["molecule_name"].map(mol2fold).astype(np.int8)

fold_values = train["fold"].to_numpy(np.int8, copy=False)
fold_indices = [np.flatnonzero(fold_values == f) for f in range(N_FOLDS)]
train_indices = [np.flatnonzero(fold_values != f) for f in range(N_FOLDS)]

print("Train rows:", len(train), "Test rows:", len(test))
print("Unique train molecules:", train["molecule_name"].nunique(), "N_FOLDS:", N_FOLDS)




## === cell 1
def build_tables(df_tr: pd.DataFrame, dist_bin: pd.Series):
    d = df_tr  # no copy; dist_bin supplied and aligned
    y = d["scalar_coupling_constant"].astype(np.float32, copy=False)

    m_type = y.groupby(d["type"], sort=False).mean()
    m_type_pair = y.groupby([d["type"], d["a_lo"], d["a_hi"]], sort=False).mean()
    m_type_a0 = y.groupby([d["type"], d["a_lo"]], sort=False).mean()
    m_type_a1 = y.groupby([d["type"], d["a_hi"]], sort=False).mean()

    valid = dist_bin.notna().to_numpy()
    dd = d.loc[valid, ["type", "i_lo", "i_hi", "a_lo", "a_hi", "z_lo", "z_hi"]]
    db = dist_bin.loc[valid]
    yy = y.loc[valid]

    g_idx = [dd["type"], dd["i_lo"], dd["i_hi"], db]
    s_idx_mean = yy.groupby(g_idx, sort=False).mean()
    s_idx_cnt = yy.groupby(g_idx, sort=False).size().astype(np.float32)

    g_pair = [dd["type"], dd["a_lo"], dd["a_hi"], db]
    s_pair_mean = yy.groupby(g_pair, sort=False).mean()
    s_pair_cnt = yy.groupby(g_pair, sort=False).size().astype(np.float32)

    g_z = [dd["type"], dd["z_lo"], dd["z_hi"], db]
    s_z_mean = yy.groupby(g_z, sort=False).mean()
    s_z_cnt = yy.groupby(g_z, sort=False).size().astype(np.float32)

    return {
        "m_type": m_type,
        "m_type_pair": m_type_pair,
        "m_type_a0": m_type_a0,
        "m_type_a1": m_type_a1,
        "m_type_idxpair_db_mean": s_idx_mean,
        "m_type_idxpair_db_cnt": s_idx_cnt,
        "m_type_pair_db_mean": s_pair_mean,
        "m_type_pair_db_cnt": s_pair_cnt,
        "m_type_zpair_db_mean": s_z_mean,
        "m_type_zpair_db_cnt": s_z_cnt,
    }


def predict_with_tables(
    df_x: pd.DataFrame,
    dist_bin: pd.Series,
    tables: dict,
    alpha: float,
    alpha_idx: float,
    global_mean_: float,
):
    eps = np.float32(1e-6)

    t = df_x["type"]
    a_lo = df_x["a_lo"]
    a_hi = df_x["a_hi"]
    i_lo = df_x["i_lo"]
    i_hi = df_x["i_hi"]
    z_lo = df_x["z_lo"]
    z_hi = df_x["z_hi"]
    db = dist_bin

    m_type = tables["m_type"].reindex(t).to_numpy(dtype=np.float32)
    m_pair = (
        tables["m_type_pair"]
        .reindex(pd.MultiIndex.from_arrays([t, a_lo, a_hi]))
        .to_numpy(dtype=np.float32)
    )
    m_a0 = (
        tables["m_type_a0"]
        .reindex(pd.MultiIndex.from_arrays([t, a_lo]))
        .to_numpy(dtype=np.float32)
    )
    m_a1 = (
        tables["m_type_a1"]
        .reindex(pd.MultiIndex.from_arrays([t, a_hi]))
        .to_numpy(dtype=np.float32)
    )

    m_pair = np.where(np.isfinite(m_pair), m_pair, m_type)

    key_idx = pd.MultiIndex.from_arrays([t, i_lo, i_hi, db])
    key_pair = pd.MultiIndex.from_arrays([t, a_lo, a_hi, db])
    key_z = pd.MultiIndex.from_arrays([t, z_lo, z_hi, db])

    mean_idx = (
        tables["m_type_idxpair_db_mean"].reindex(key_idx).to_numpy(dtype=np.float32)
    )
    cnt_idx = (
        tables["m_type_idxpair_db_cnt"].reindex(key_idx).to_numpy(dtype=np.float32)
    )
    mean_pair = (
        tables["m_type_pair_db_mean"].reindex(key_pair).to_numpy(dtype=np.float32)
    )
    cnt_pair = tables["m_type_pair_db_cnt"].reindex(key_pair).to_numpy(dtype=np.float32)
    mean_z = tables["m_type_zpair_db_mean"].reindex(key_z).to_numpy(dtype=np.float32)
    cnt_z = tables["m_type_zpair_db_cnt"].reindex(key_z).to_numpy(dtype=np.float32)

    cnt_pair = np.nan_to_num(cnt_pair, nan=0.0).astype(np.float32, copy=False)
    cnt_z = np.nan_to_num(cnt_z, nan=0.0).astype(np.float32, copy=False)
    cnt_idx = np.nan_to_num(cnt_idx, nan=0.0).astype(np.float32, copy=False)

    shrunk_pair = (cnt_pair * mean_pair + np.float32(alpha) * m_pair) / (
        cnt_pair + np.float32(alpha) + eps
    )
    shrunk_z = (cnt_z * mean_z + np.float32(alpha) * m_pair) / (
        cnt_z + np.float32(alpha) + eps
    )

    parent_for_idx = np.where(
        np.isfinite(shrunk_z),
        shrunk_z,
        np.where(np.isfinite(shrunk_pair), shrunk_pair, m_pair),
    )
    shrunk_idx = (cnt_idx * mean_idx + np.float32(alpha_idx) * parent_for_idx) / (
        cnt_idx + np.float32(alpha_idx) + eps
    )

    pred = shrunk_idx
    pred = np.where(np.isfinite(pred), pred, shrunk_z)
    pred = np.where(np.isfinite(pred), pred, shrunk_pair)
    pred = np.where(np.isfinite(pred), pred, m_pair)
    pred = np.where(np.isfinite(pred), pred, m_a0)
    pred = np.where(np.isfinite(pred), pred, m_a1)
    pred = np.where(np.isfinite(pred), pred, m_type)
    pred = np.where(np.isfinite(pred), pred, np.float32(global_mean_))
    return pd.Series(pred.astype(np.float32, copy=False), index=df_x.index)


def competition_metric_log_mae(
    y_true: np.ndarray, y_pred: np.ndarray, types: np.ndarray
):
    df = pd.DataFrame({"y": y_true, "p": y_pred, "type": types})
    df["ae"] = (df["y"] - df["p"]).abs()
    maes = df.groupby("type", sort=False)["ae"].mean()
    return float(np.mean(np.log(maes.values + 1e-12)))


cand_bin_width = [0.05, 0.1, 0.2]
cand_alpha = [5.0, 10.0, 20.0, 40.0]
cand_alpha_idx_mult = [2.0, 4.0, 8.0]

dist_bins_train = {}
dist_bins_test = {}
for bw in cand_bin_width:
    dist_bins_train[bw] = pd.Series(
        np.rint(train["dist"].to_numpy(np.float32, copy=False) / np.float32(bw)).astype(
            np.int32
        ),
        index=train.index,
        dtype="Int32",
    )
    dist_bins_test[bw] = pd.Series(
        np.rint(test["dist"].to_numpy(np.float32, copy=False) / np.float32(bw)).astype(
            np.int32
        ),
        index=test.index,
        dtype="Int32",
    )

best = None
y_train = train["scalar_coupling_constant"].to_numpy(np.float32, copy=False)
types_train = train["type"].to_numpy()

for bw in cand_bin_width:
    db_train = dist_bins_train[bw]
    for a in cand_alpha:
        for m in cand_alpha_idx_mult:
            a_idx = a * m
            oof_pred = np.empty(len(train), dtype=np.float32)

            for f in range(N_FOLDS):
                trn_idx = train_indices[f]
                val_idx = fold_indices[f]
                trn_df = train.iloc[trn_idx]
                val_df = train.iloc[val_idx]

                tbl = build_tables(trn_df, dist_bin=db_train.iloc[trn_idx])
                oof_pred[val_idx] = predict_with_tables(
                    val_df,
                    dist_bin=db_train.iloc[val_idx],
                    tables=tbl,
                    alpha=a,
                    alpha_idx=a_idx,
                    global_mean_=global_mean,
                ).to_numpy(np.float32, copy=False)

            s = competition_metric_log_mae(y_train, oof_pred, types_train)
            if (best is None) or (s < best[0]):
                best = (s, bw, a, m)

best_score, BIN_WIDTH, ALPHA, ALPHA_IDX_MULT = best
ALPHA_IDX = ALPHA * ALPHA_IDX_MULT
print(
    "Selected (lower is better): CV OOF logMAE =",
    best_score,
    "BIN_WIDTH =",
    BIN_WIDTH,
    "ALPHA =",
    ALPHA,
    "ALPHA_IDX_MULT =",
    ALPHA_IDX_MULT,
    "ALPHA_IDX =",
    ALPHA_IDX,
)




## === cell 2
oof_pred = np.empty(len(train), dtype=np.float32)
db_train = dist_bins_train[BIN_WIDTH]
for f in range(N_FOLDS):
    trn_idx = train_indices[f]
    val_idx = fold_indices[f]
    trn_df = train.iloc[trn_idx]
    val_df = train.iloc[val_idx]
    tbl = build_tables(trn_df, dist_bin=db_train.iloc[trn_idx])
    oof_pred[val_idx] = predict_with_tables(
        val_df,
        dist_bin=db_train.iloc[val_idx],
        tables=tbl,
        alpha=ALPHA,
        alpha_idx=ALPHA_IDX,
        global_mean_=global_mean,
    ).to_numpy(np.float32, copy=False)

oof_score = competition_metric_log_mae(
    y_train,
    oof_pred,
    types_train,
)
print("OOF logMAE estimate (lower is better):", oof_score)

full_tables = build_tables(train, dist_bin=db_train)
db_test = dist_bins_test[BIN_WIDTH]
test_pred = predict_with_tables(
    test,
    dist_bin=db_test,
    tables=full_tables,
    alpha=ALPHA,
    alpha_idx=ALPHA_IDX,
    global_mean_=global_mean,
)

submission = pd.DataFrame(
    {"id": test["id"].values, "scalar_coupling_constant": test_pred.values}
)
submission = sample[["id"]].merge(submission, on="id", how="left")

missing = int(submission["scalar_coupling_constant"].isna().sum())
if missing:
    submission["scalar_coupling_constant"] = submission[
        "scalar_coupling_constant"
    ].fillna(global_mean)

out_path = "stackers_blend.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print("Rows:", len(submission), "Missing preds:", missing)
print("Global mean:", global_mean)
print(
    "Test atom_0/atom_1 UNK counts:",
    int((test["atom_0"] == "UNK").sum()),
    int((test["atom_1"] == "UNK").sum()),
)
print(
    "Missing dist (test):",
    int(test["dist"].isna().sum()),
    "Missing dist_bin (test):",
    int(db_test.isna().sum()),
)
