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

-1.3628733700694355

# 6. Current score

2.97415

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23596) has done: 'The current notebook fails because it tries to read other competitors’ datasets/submission files that are not present in your environment, so it never reaches the point of writing a valid `submission.csv`. I remove those external dependencies and replace them with an end-to-end, self-contained baseline that trains on the provided `train.csv` and predicts `test.csv`. To keep changes minimal while improving score from “no submission” toward your target, I use a simple per-`type` mean target encoding with light smoothing (a common strong baseline for this competition) and ensure the output matches `sample_submission.csv` exactly. The result always generate a valid `submission.csv` with columns `id,scalar_coupling_constant`.'
- What this solution (achieved 2.97414) has done: 'Your current solution predicts only a smoothed mean per coupling `type`, which ignores molecule geometry and atom information; that’s why the score is far from the (very strong) target. To move the score downward toward the target with minimal changes and without changing the overall “single-pass fit then predict” approach, I keep the type-based baseline but add a tiny set of strictly local, competition-relevant features: atom element types for the two atoms and their 3D distance from `structures.csv`. Then I replace the per-`type` mean with a per-`(type, atom_0, atom_1)` smoothed mean, and apply a small distance-based linear correction fit within each `type` (still simple regression, no new training loops). This remains fast, deterministic, self-contained, and almost certainly reduce MAE substantially versus the pure type mean, moving your score closer to the target.'
- What this solution (achieved 2.97415) has done: 'Your score is far worse than the target (lower is better), so we should legitimately improve accuracy with minimal core-logic disruption. The biggest issue in the current baseline is target leakage inside the “pair mean” encoding: each training row influences its own encoded feature, which makes the distance correction fit overly optimistic and harms generalization; we fix this by using out-of-fold (OOF) target encoding by molecule groups (consistent with the competition split). Then we fit the same per-`type` linear distance correction on OOF residuals (not leaky residuals) and apply it to test, keeping your overall “smoothed means + per-type linear correction” approach intact. Finally, we slightly stabilize the encodings by using an order-invariant atom-pair key (so (C,H) and (H,C) share stats), which typically improves generalization with negligible logic change.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_ROOT_CANDIDATES = [
    "/kaggle/data/champs-scalar-coupling",
    "/kaggle/input/champs-scalar-coupling-challenge",
    "/kaggle/input/champs-scalar-coupling",
    "../input/champs-scalar-coupling-challenge",
    "../input/champs-scalar-coupling",
]

DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(p) and os.path.exists(os.path.join(p, "train.csv")):
        DATA_ROOT = p
        break

if DATA_ROOT is None:
    for base in ["/kaggle/input", "../input", "/kaggle/data"]:
        if os.path.exists(base):
            for name in os.listdir(base):
                cand = os.path.join(base, name)
                if os.path.isdir(cand) and os.path.exists(
                    os.path.join(cand, "train.csv")
                ):
                    DATA_ROOT = cand
                    break
        if DATA_ROOT is not None:
            break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate competition data folder containing train.csv"
    )

print("Using DATA_ROOT:", DATA_ROOT)
print("Files:", sorted([f for f in os.listdir(DATA_ROOT) if f.endswith(".csv")])[:10])



## === cell 1
train_path = os.path.join(DATA_ROOT, "train.csv")
test_path = os.path.join(DATA_ROOT, "test.csv")
sample_path = os.path.join(DATA_ROOT, "sample_submission.csv")
structures_path = os.path.join(DATA_ROOT, "structures.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample = pd.read_csv(sample_path)

required_train_cols = {
    "id",
    "molecule_name",
    "atom_index_0",
    "atom_index_1",
    "type",
    "scalar_coupling_constant",
}
required_test_cols = {"id", "molecule_name", "atom_index_0", "atom_index_1", "type"}
if not required_train_cols.issubset(train.columns):
    raise ValueError(
        f"train.csv missing columns: {required_train_cols - set(train.columns)}"
    )
if not required_test_cols.issubset(test.columns):
    raise ValueError(
        f"test.csv missing columns: {required_test_cols - set(test.columns)}"
    )
if not {"id", "scalar_coupling_constant"}.issubset(sample.columns):
    raise ValueError(
        "sample_submission.csv must contain columns: id, scalar_coupling_constant"
    )

print(
    "train shape:",
    train.shape,
    "test shape:",
    test.shape,
    "sample shape:",
    sample.shape,
)
print("train types:", train["type"].nunique(), "test types:", test["type"].nunique())

structures = pd.read_csv(
    structures_path, usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"]
)
structures["atom_index"] = structures["atom_index"].astype(np.int32)

print("structures shape:", structures.shape)
print(structures.head())




## === cell 2
def add_atom_and_distance(df, structures_df):
    s0 = structures_df.rename(
        columns={
            "atom_index": "atom_index_0",
            "atom": "atom_0",
            "x": "x0",
            "y": "y0",
            "z": "z0",
        }
    )
    s1 = structures_df.rename(
        columns={
            "atom_index": "atom_index_1",
            "atom": "atom_1",
            "x": "x1",
            "y": "y1",
            "z": "z1",
        }
    )
    out = df.merge(
        s0[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    out = out.merge(
        s1[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
    )

    missing = out[["atom_0", "atom_1", "x0", "x1"]].isna().any(axis=1).mean()
    if missing > 0:
        print(f"Warning: fraction of rows with missing structure merge = {missing:.6f}")

    dx = out["x0"] - out["x1"]
    dy = out["y0"] - out["y1"]
    dz = out["z0"] - out["z1"]
    out["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float64)

    keep_cols = [
        "id",
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "atom_0",
        "atom_1",
        "dist",
    ]
    extra_cols = [
        c for c in df.columns if c not in keep_cols
    ]  # includes target for train
    keep_cols = list(dict.fromkeys(keep_cols + extra_cols))
    out = out[keep_cols]
    return out


train_feat = add_atom_and_distance(train, structures)
test_feat = add_atom_and_distance(test, structures)

global_dist_median = pd.concat([train_feat["dist"], test_feat["dist"]], axis=0).median()
train_feat["dist"] = train_feat["dist"].fillna(global_dist_median)
test_feat["dist"] = test_feat["dist"].fillna(global_dist_median)

train_feat["atom_0"] = train_feat["atom_0"].fillna("UNK")
train_feat["atom_1"] = train_feat["atom_1"].fillna("UNK")
test_feat["atom_0"] = test_feat["atom_0"].fillna("UNK")
test_feat["atom_1"] = test_feat["atom_1"].fillna("UNK")

train_feat["a_min"] = np.where(
    train_feat["atom_0"] <= train_feat["atom_1"],
    train_feat["atom_0"],
    train_feat["atom_1"],
)
train_feat["a_max"] = np.where(
    train_feat["atom_0"] <= train_feat["atom_1"],
    train_feat["atom_1"],
    train_feat["atom_0"],
)
test_feat["a_min"] = np.where(
    test_feat["atom_0"] <= test_feat["atom_1"], test_feat["atom_0"], test_feat["atom_1"]
)
test_feat["a_max"] = np.where(
    test_feat["atom_0"] <= test_feat["atom_1"], test_feat["atom_1"], test_feat["atom_0"]
)

print(
    train_feat[
        [
            "type",
            "atom_0",
            "atom_1",
            "a_min",
            "a_max",
            "dist",
            "scalar_coupling_constant",
        ]
    ].head()
)
print(test_feat[["type", "atom_0", "atom_1", "a_min", "a_max", "dist"]].head())



## === cell 3

global_mean = train_feat["scalar_coupling_constant"].mean()
SMOOTHING = 50.0

type_stats = (
    train_feat.groupby("type")["scalar_coupling_constant"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "type_mean", "count": "type_count"})
    .reset_index()
)
type_stats["type_mean_smooth"] = (
    type_stats["type_mean"] * type_stats["type_count"] + global_mean * SMOOTHING
) / (type_stats["type_count"] + SMOOTHING)

mols = train_feat["molecule_name"].unique()
rng = np.random.RandomState(RANDOM_STATE)
rng.shuffle(mols)
n_folds = 5
fold_id_by_mol = {m: (i % n_folds) for i, m in enumerate(mols)}
train_feat["fold"] = train_feat["molecule_name"].map(fold_id_by_mol).astype(np.int8)

oof_pair_smooth = np.empty(train_feat.shape[0], dtype=np.float64)

for f in range(n_folds):
    trn_mask = train_feat["fold"].values != f
    val_mask = ~trn_mask

    trn = train_feat.loc[
        trn_mask, ["type", "a_min", "a_max", "scalar_coupling_constant"]
    ]
    val = train_feat.loc[val_mask, ["type", "a_min", "a_max"]]

    pair_stats_f = (
        trn.groupby(["type", "a_min", "a_max"])["scalar_coupling_constant"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "pair_mean", "count": "pair_count"})
        .reset_index()
    )
    pair_stats_f = pair_stats_f.merge(
        type_stats[["type", "type_mean_smooth"]], on="type", how="left"
    )
    pair_stats_f["pair_mean_smooth"] = (
        pair_stats_f["pair_mean"] * pair_stats_f["pair_count"]
        + pair_stats_f["type_mean_smooth"] * SMOOTHING
    ) / (pair_stats_f["pair_count"] + SMOOTHING)

    val_merge = val.merge(
        pair_stats_f[["type", "a_min", "a_max", "pair_mean_smooth"]],
        on=["type", "a_min", "a_max"],
        how="left",
    ).merge(type_stats[["type", "type_mean_smooth"]], on="type", how="left")

    pred_val = (
        val_merge["pair_mean_smooth"]
        .fillna(val_merge["type_mean_smooth"])
        .fillna(global_mean)
        .astype(np.float64)
        .values
    )
    oof_pair_smooth[val_mask] = pred_val

train_base_pred_oof = oof_pair_smooth.astype(np.float64)

train_resid = (
    train_feat["scalar_coupling_constant"].astype(np.float64).values
    - train_base_pred_oof
)

train_tmp = pd.DataFrame(
    {
        "type": train_feat["type"].values,
        "dist": train_feat["dist"].values,
        "resid": train_resid,
    }
)

g = train_tmp.groupby("type", sort=False)
dist_mean = g["dist"].mean()
resid_mean = g["resid"].mean()
cov = g.apply(
    lambda x: ((x["dist"] - x["dist"].mean()) * (x["resid"] - x["resid"].mean())).mean()
)
var = g.apply(lambda x: ((x["dist"] - x["dist"].mean()) ** 2).mean())

coef = (cov / (var + 1e-12)).astype(np.float64)
intercept = (resid_mean - coef * dist_mean).astype(np.float64)

coef_df = pd.DataFrame(
    {
        "type": coef.index.values,
        "dist_coef": coef.values,
        "dist_intercept": intercept.loc[coef.index].values,
    }
)

pair_stats = (
    train_feat.groupby(["type", "a_min", "a_max"])["scalar_coupling_constant"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "pair_mean", "count": "pair_count"})
    .reset_index()
)
pair_stats = pair_stats.merge(
    type_stats[["type", "type_mean_smooth"]], on="type", how="left"
)
pair_stats["pair_mean_smooth"] = (
    pair_stats["pair_mean"] * pair_stats["pair_count"]
    + pair_stats["type_mean_smooth"] * SMOOTHING
) / (pair_stats["pair_count"] + SMOOTHING)

test_base = test_feat.merge(
    pair_stats[["type", "a_min", "a_max", "pair_mean_smooth"]],
    on=["type", "a_min", "a_max"],
    how="left",
).merge(type_stats[["type", "type_mean_smooth"]], on="type", how="left")

base_pred = (
    test_base["pair_mean_smooth"]
    .fillna(test_base["type_mean_smooth"])
    .fillna(global_mean)
    .astype(np.float64)
)

test_corr = test_feat[["type", "dist"]].merge(coef_df, on="type", how="left")
test_corr["dist_coef"] = test_corr["dist_coef"].fillna(0.0)
test_corr["dist_intercept"] = test_corr["dist_intercept"].fillna(0.0)

test_pred = (
    base_pred.values
    + test_corr["dist_intercept"].values
    + test_corr["dist_coef"].values * test_feat["dist"].values
).astype(np.float64)

print("Base prediction stats:", pd.Series(base_pred).describe())
print("Final prediction stats:", pd.Series(test_pred).describe())



## === cell 4
sub = sample[["id"]].merge(
    pd.DataFrame({"id": test["id"].values, "scalar_coupling_constant": test_pred}),
    on="id",
    how="left",
)

sub["scalar_coupling_constant"] = (
    sub["scalar_coupling_constant"].fillna(global_mean).astype(np.float64)
)

if sub.shape[0] != sample.shape[0]:
    raise ValueError("Submission row count does not match sample_submission")
if sub["id"].isna().any() or sub["scalar_coupling_constant"].isna().any():
    raise ValueError("Submission contains NaNs")

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote", out_path, "with shape", sub.shape)
print(sub.head())



## === cell 5
print("submission scalar_coupling_constant describe:")
print(sub["scalar_coupling_constant"].describe())
