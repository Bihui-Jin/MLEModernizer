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

-1.3684302901167014

# 6. Current score

1.19403

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.19403) has done: 'I remove the dependency on missing external blend files (which causes the FileNotFoundError) and instead build a self-contained baseline that always runs in this environment. To keep the core “blending” idea, I generate two simple but legitimate model predictions from the provided data (a type-wise mean and a type-wise median from train), then blend them with the same 0.4/0.6 weights. I also ensure strict submission alignment by merging on `id` and filling any missing types with the global mean, then write a valid `.csv` submission file.'
- What this solution (achieved 1.19403) has done: 'Your current score (1.19403, lower-is-better) is far worse than the target (-1.3684), so we need a legitimate accuracy improvement with minimal core-logic change. I keep the same “type-wise statistic + blending” approach, but compute those statistics at a more granular level using both `type` and the two atom element symbols (looked up from `structures.csv`), which is still the same idea (groupby mean/median + 0.4/0.6 blend) but gives much more informative priors. I also add a safe fallback ladder (type+atoms → type → global) and keep the submission alignment by `id` exactly as you already do. This remains fast (just a couple merges + groupbys) and produces the same required `blend_of_blends.csv`.'
- What this solution (achieved 1.19403) has done: 'Your current score (1.19403, lower-is-better) is far from the target (-1.3684), so we need a real accuracy gain while keeping the same “type/group statistic + blend” core logic. The smallest high-impact change is to compute the same mean/median group statistics on *both atom-orderings* by canonicalizing the pair (A,B) so that (H,C) and (C,H) share data, then use that symmetric lookup at inference with the same fallback ladder (type+atoms → type → global). This keeps the model identical in spirit (groupby mean/median + 0.4/0.6 blend) but reduces sparsity and should materially lower MAE across types. I also keep strict `id` alignment via `sample_submission.csv` and still write `blend_of_blends.csv`.'
- What this solution (achieved 1.19403) has done: 'You’re far worse than the target (lower-is-better; 1.19403 vs -1.3684), so we need a legitimate accuracy improvement while keeping the same “groupby statistic + 0.4/0.6 mean/median blend” core logic. The minimal high-impact change is to compute the statistics at a more informative but still purely-aggregate granularity: group by (`type`, `atom_lo`, `atom_hi`, binned interatomic distance), where distance comes directly from `structures.csv`. This keeps the same semantics (type-wise priors blended) but reduces MAE substantially by conditioning on geometry, which is crucial in this competition. We keep the same fallback ladder (full key → type+atoms → type → global) and preserve strict submission alignment by `id`, writing the same `blend_of_blends.csv`.'
- What this solution (achieved 1.19403) has done: 'Your current score (1.19403; lower-is-better) is far from the target (-1.3684), so we need a legitimate accuracy gain while keeping the same “groupby statistics + 0.4/0.6 mean/median blend” core logic. The smallest high-impact fix is to correct a key bug: you canonicalize atom symbols but you do **not** canonicalize the corresponding coordinates, so distances are often computed between the wrong atoms, corrupting the distance-binned lookup. I compute distances using the same canonical ordering (swap coordinates when atom_0/atom_1 are swapped) and keep everything else (binning, groupbys, fallback ladder, blending weights, and submission alignment) unchanged. This should materially reduce error while staying within the same approach and runtime, and it still writes a valid `blend_of_blends.csv`.'
- What this solution (achieved 1.19403) has done: 'Your current score (1.19403, lower-is-better) is far from the target (-1.3684), so we need a legitimate accuracy improvement while keeping the same “groupby statistics + mean/median 0.4/0.6 blend” core logic. The smallest high-impact change is to reduce noise in the distance-binned lookup by using a slightly wider distance bin, which increases per-bin sample sizes and typically improves MAE for this baseline without changing the modeling approach. I keep the same canonical atom-pair logic, the same fallback ladder (type+atoms+dist → type+atoms → type → global), and the same submission alignment by `id`. This should move the score downward (better) toward the target while staying stable and fast.'
- What this solution (achieved 1.19403) has done: 'Your current score (1.19403; lower-is-better) is far worse than the target (-1.3684), so we need a real accuracy gain while keeping the same core “groupby statistics + 0.4/0.6 mean/median blend” logic. The most direct improvement without changing the model class is to reduce sparsity and noise in the distance-conditioned lookup by using a slightly wider distance bin (more samples per bin) and to avoid overly hard clipping of distances that can collapse distinct pairs into the same edge bin. I also add a minimal safeguard to compute `dist_bin` in float32/float64 consistently and ensure both train/test use identical binning behavior. Everything else (features, grouping keys, fallback ladder, blending weights, and submission writing) stays the same.'
- What this solution (achieved 1.19403) has done: 'Your current score (1.19403, lower-is-better) is far worse than the target (-1.3684), so we need a legitimate accuracy gain while keeping the same “groupby statistics + mean/median blend” core logic. The smallest high-impact fix is to compute the atom-pair distance using the *actual* `atom_index_0/atom_index_1` coordinates (no swapping) while still using canonicalized atom symbols only for grouping; the current swap-based distance can mismatch coordinates and corrupt the distance-conditioned statistics. I keep the same binning width, the same group keys, the same fallback ladder, and the same 0.4/0.6 blend weights, so evaluation semantics remain identical. This change should materially reduce MAE by making the distance feature correct and consistent between train/test, while staying fast and producing the same valid `blend_of_blends.csv`.'
- What this solution (achieved 1.19403) has done: 'Your current score (1.19403; lower-is-better) is far worse than the target (-1.3684), so we should make a legitimate accuracy improvement while keeping the same core “groupby statistics + mean/median + 0.4/0.6 blend” approach. The biggest issue is that the prediction is currently based only on `type`, atom symbols, and a binned distance; in this competition, coupling depends strongly on local chemical environment, so adding a tiny amount of environment context improve MAE without changing the modeling paradigm. I minimally extend the grouping keys to include a simple, fast “neighbor count around each atom within a radius” computed from `structures.csv` (pure aggregation), and keep the exact same fallback ladder and blending weights. This remains fast (vectorized merge + groupby) and still writes the same valid `blend_of_blends.csv` submission.'
- What this solution (achieved 1.19403) has done: 'Your score is much worse than the target (lower is better), and the biggest issue is that the current “neighbor count” feature is computed via an O(N²) self-join per molecule table, which is both very slow and also likely to produce unstable/incorrect behavior under Kaggle time/memory constraints. I keep the exact same core logic (groupby mean/median statistics + 0.4/0.6 blending + same fallback ladder), but replace neighbor counting with a deterministic, chemistry-plausible, O(N) per-molecule approximation: use the known coupling-path length encoded in the first character of `type` (1J/2J/3J) to set `nn_lo/nn_hi` directly. This preserves the “include a tiny environment context key” idea without expensive geometry graph construction and should improve consistency and typically accuracy versus the current noisy neighbor counts. Everything else (distance binning, canonical atom-pair logic, aggregation keys, blending weights, and submission writing) remains unchanged and still produces `blend_of_blends.csv`.'
- What this solution (achieved 1.19403) has done: 'Your current score (1.19403; lower-is-better) is far from the target (-1.3684), so we need a real accuracy improvement while keeping the same core “groupby mean/median + 0.4/0.6 blend + fallback ladder” logic. The smallest high-impact fix is to correct how atom-pair canonicalization is applied: you canonicalize atom symbols for grouping but you still compute distance on the original (atom_index_0, atom_index_1) order, which splits identical unordered pairs into inconsistent distance bins and harms the distance-conditioned lookup. I compute a canonical (unordered) distance by swapping coordinates when `_swap` is true, while leaving the rest of your pipeline (binning width, group keys, blending weights, fallbacks, submission alignment, and output filename) unchanged. This should reduce sparsity/noise in the per-bin statistics and move the MAE/logMAE down toward the target.'
- What this solution (achieved 1.19449) has done: 'Your current score (1.19403; lower is better) is far from the target (-1.3684), so we should improve accuracy with a minimal change that keeps the same “groupby stats + 0.4/0.6 mean/median blend + fallback ladder” logic. The biggest safe gain here is to fix a leakage/consistency issue: group statistics should be computed on a molecule-level split (the competition’s split unit), otherwise the statistics can become poorly calibrated for unseen molecules. I add a deterministic molecule-based holdout inside the training data to compute the group means/medians (same exact aggregations, just fit on a proper split), then use those to predict test; this typically improves generalization without changing the model class. I also keep all existing features/keys (type, canonical atoms, distance bin, nn_lo/nn_hi) and the same submission writing.'
- What this solution (achieved 1.19373) has done: 'Your current score (1.19449; lower is better) is far from the target (-1.3684), so we need a real generalization gain with minimal changes while preserving the same “groupby mean/median + 0.4/0.6 blend + fallback ladder” core logic. The highest-impact low-risk fix here is to compute the statistics on a *stratified-by-type* molecule holdout instead of taking the first 10% of molecule names, which can bias the fit set and hurt per-type calibration (the metric is averaged over types). I keep all existing features/keys (type, canonical atoms, distance bin, nn_lo/nn_hi), the same binning, and the same blending weights; only the holdout selection becomes deterministic and type-balanced at the molecule level. This should reduce MAE across types and move the score downward toward the target while keeping runtime and memory similar and still writing `blend_of_blends.csv`.'
- What this solution (achieved 1.19403) has done: 'We keep your exact “groupby mean/median stats + 0.4/0.6 blend + fallback ladder” core logic, but fix a key generalization issue: the group statistics are currently fit on only ~90% of molecules (holdout), which throws away a lot of signal and inflates test error. Since Kaggle scoring is on the hidden test set (not your internal holdout), we instead fit those same statistics on all training molecules to move the score downward (better) toward the target. Everything else (features/keys, distance binning, blending weights, submission alignment, output filename) stays the same, and the script still runs end-to-end and writes `blend_of_blends.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

np.random.seed(42)

INPUT_ROOT = "/kaggle/input"

COMP_DIR = os.path.join(INPUT_ROOT, "champs-scalar-coupling")
if not os.path.isdir(COMP_DIR):
    candidates = [
        os.path.join(INPUT_ROOT, d) for d in os.listdir(INPUT_ROOT) if "champs" in d
    ]
    COMP_DIR = candidates[0] if candidates else INPUT_ROOT

print("Using COMP_DIR:", COMP_DIR)
print("Files in /kaggle/input:", os.listdir(INPUT_ROOT)[:50])

train_path = os.path.join(COMP_DIR, "train.csv")
test_path = os.path.join(COMP_DIR, "test.csv")
sample_path = os.path.join(COMP_DIR, "sample_submission.csv")
structures_path = os.path.join(COMP_DIR, "structures.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
submission = pd.read_csv(sample_path)

print(
    "train shape:",
    train.shape,
    "test shape:",
    test.shape,
    "sample shape:",
    submission.shape,
)
print("train columns:", list(train.columns))
print("test columns:", list(test.columns))
print("submission columns:", list(submission.columns))

structures = pd.read_csv(
    structures_path,
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
)
structures.rename(columns={"atom": "atom_symbol"}, inplace=True)

print("structures shape:", structures.shape)
print(structures.head())


## === cell 1
s0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom_symbol": "atom_0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
    }
)
s1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom_symbol": "atom_1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
    }
)

train_feat = train.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
train_feat = train_feat.merge(s1, on=["molecule_name", "atom_index_1"], how="left")

test_feat = test.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
test_feat = test_feat.merge(s1, on=["molecule_name", "atom_index_1"], how="left")

train_feat["atom_0"] = train_feat["atom_0"].fillna("UNK")
train_feat["atom_1"] = train_feat["atom_1"].fillna("UNK")
test_feat["atom_0"] = test_feat["atom_0"].fillna("UNK")
test_feat["atom_1"] = test_feat["atom_1"].fillna("UNK")

fit_mask = np.ones(len(train_feat), dtype=bool)
train_fit = train_feat.copy()

type_mean = train_fit.groupby("type")["scalar_coupling_constant"].mean()
type_median = train_fit.groupby("type")["scalar_coupling_constant"].median()
global_mean = float(train_fit["scalar_coupling_constant"].mean())


def _canon_pair_and_swapmask(df, a_col, b_col):
    a = df[a_col].astype(str).values
    b = df[b_col].astype(str).values
    swap = a > b
    lo = np.where(swap, b, a)
    hi = np.where(swap, a, b)
    return lo, hi, swap


train_feat = train_feat.copy()
train_feat["atom_lo"], train_feat["atom_hi"], train_feat["_swap"] = (
    _canon_pair_and_swapmask(train_feat, "atom_0", "atom_1")
)

test_feat = test_feat.copy()
test_feat["atom_lo"], test_feat["atom_hi"], test_feat["_swap"] = (
    _canon_pair_and_swapmask(test_feat, "atom_0", "atom_1")
)


def _add_distance_and_bin_canonical(
    df, swap_col="_swap", bin_width=0.20, max_dist=10.0
):
    x0 = df["x0"].astype(np.float64).values
    y0 = df["y0"].astype(np.float64).values
    z0 = df["z0"].astype(np.float64).values
    x1 = df["x1"].astype(np.float64).values
    y1 = df["y1"].astype(np.float64).values
    z1 = df["z1"].astype(np.float64).values

    swap = df[swap_col].values.astype(bool)

    x_lo = np.where(swap, x1, x0)
    y_lo = np.where(swap, y1, y0)
    z_lo = np.where(swap, z1, z0)
    x_hi = np.where(swap, x0, x1)
    y_hi = np.where(swap, y0, y1)
    z_hi = np.where(swap, z0, z1)

    dx = x_lo - x_hi
    dy = y_lo - y_hi
    dz = z_lo - z_hi
    dist = np.sqrt(dx * dx + dy * dy + dz * dz)

    dist = np.clip(dist, 0.0, max_dist)
    dist_bin = (np.floor(dist / bin_width) * bin_width).astype(np.float64)
    return dist, dist_bin


train_feat["dist"], train_feat["dist_bin"] = _add_distance_and_bin_canonical(train_feat)
test_feat["dist"], test_feat["dist_bin"] = _add_distance_and_bin_canonical(test_feat)


def _type_to_pathlen(series):
    s = series.astype(str).str.extract(r"^(\d)", expand=False)
    return s.fillna("0").astype(np.int16)


train_feat["_pathlen"] = _type_to_pathlen(train_feat["type"])
test_feat["_pathlen"] = _type_to_pathlen(test_feat["type"])

train_feat["nn0"] = (train_feat["_pathlen"] - 1).clip(lower=0).astype(np.int16)
train_feat["nn1"] = (train_feat["_pathlen"] - 1).clip(lower=0).astype(np.int16)
test_feat["nn0"] = (test_feat["_pathlen"] - 1).clip(lower=0).astype(np.int16)
test_feat["nn1"] = (test_feat["_pathlen"] - 1).clip(lower=0).astype(np.int16)

swap_nn = test_feat["_swap"].values
test_feat["nn_lo"] = np.where(
    swap_nn, test_feat["nn1"].values, test_feat["nn0"].values
).astype(np.int16)
test_feat["nn_hi"] = np.where(
    swap_nn, test_feat["nn0"].values, test_feat["nn1"].values
).astype(np.int16)

swap_nn_tr = train_feat["_swap"].values
train_feat["nn_lo"] = np.where(
    swap_nn_tr, train_feat["nn1"].values, train_feat["nn0"].values
).astype(np.int16)
train_feat["nn_hi"] = np.where(
    swap_nn_tr, train_feat["nn0"].values, train_feat["nn1"].values
).astype(np.int16)

train_fit = train_feat.loc[fit_mask].copy()

grp_cols_dist = ["type", "atom_lo", "atom_hi", "dist_bin", "nn_lo", "nn_hi"]
type_atom_dist_mean = train_fit.groupby(grp_cols_dist)[
    "scalar_coupling_constant"
].mean()
type_atom_dist_median = train_fit.groupby(grp_cols_dist)[
    "scalar_coupling_constant"
].median()

grp_cols_atoms = ["type", "atom_lo", "atom_hi"]
type_atom_mean = train_fit.groupby(grp_cols_atoms)["scalar_coupling_constant"].mean()
type_atom_median = train_fit.groupby(grp_cols_atoms)[
    "scalar_coupling_constant"
].median()

keys_dist = list(
    zip(
        test_feat["type"].values,
        test_feat["atom_lo"].values,
        test_feat["atom_hi"].values,
        test_feat["dist_bin"].values,
        test_feat["nn_lo"].values,
        test_feat["nn_hi"].values,
    )
)
keys_atoms = list(
    zip(
        test_feat["type"].values,
        test_feat["atom_lo"].values,
        test_feat["atom_hi"].values,
    )
)

pred_mean = pd.Series(keys_dist, index=test_feat.index).map(type_atom_dist_mean)
pred_median = pd.Series(keys_dist, index=test_feat.index).map(type_atom_dist_median)

pred_mean = (
    pred_mean.fillna(pd.Series(keys_atoms, index=test_feat.index).map(type_atom_mean))
    .fillna(test_feat["type"].map(type_mean))
    .fillna(global_mean)
    .astype(np.float64)
)
pred_median = (
    pred_median.fillna(
        pd.Series(keys_atoms, index=test_feat.index).map(type_atom_median)
    )
    .fillna(test_feat["type"].map(type_median))
    .fillna(global_mean)
    .astype(np.float64)
)

sub1 = pd.DataFrame(
    {"id": test_feat["id"].values, "scalar_coupling_constant": pred_mean.values}
)
sub2 = pd.DataFrame(
    {"id": test_feat["id"].values, "scalar_coupling_constant": pred_median.values}
)

assert (
    sub1["id"].is_unique and sub2["id"].is_unique
), "IDs must be unique in test predictions."
assert (
    len(sub1) == len(test_feat) == len(sub2)
), "Prediction lengths must match test size."

print(sub1.head())
print(sub2.head())


## === cell 2
blend = submission[["id"]].merge(sub1, on="id", how="left", suffixes=("", "_sub1"))
blend = blend.merge(sub2, on="id", how="left", suffixes=("_sub1", "_sub2"))

if (
    blend["scalar_coupling_constant_sub1"].isna().any()
    or blend["scalar_coupling_constant_sub2"].isna().any()
):
    blend["scalar_coupling_constant_sub1"] = blend[
        "scalar_coupling_constant_sub1"
    ].fillna(global_mean)
    blend["scalar_coupling_constant_sub2"] = blend[
        "scalar_coupling_constant_sub2"
    ].fillna(global_mean)

submission["scalar_coupling_constant"] = (
    0.4 * blend["scalar_coupling_constant_sub1"].values
    + 0.6 * blend["scalar_coupling_constant_sub2"].values
).astype(np.float64)

out_path = "blend_of_blends.csv"
submission[["id", "scalar_coupling_constant"]].to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())


## === cell 3
submission["scalar_coupling_constant"]
