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

-1.32544

# 6. Current score

1.23566

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'I remove the dependency on missing `../input/blender/*.csv` files (the root cause of the FileNotFoundError) and instead build a valid submission from the provided competition data. To keep changes minimal while ensuring an end-to-end run, I generate predictions using a simple, leakage-safe baseline: predict the per-`type` mean `scalar_coupling_constant` from `train.csv`, and fall back to the global mean if a type is unseen. This produces a correctly formatted `submission.csv` with the required columns and should yield a reasonable (not top) score rather than failing to submit. I also keep the original “blend” intent by leaving a small blending helper that would work if extra submissions existed, but it default to the baseline in this environment.'
- What this solution (achieved 1.23566) has done: 'You’re currently far from the (much better) target (gap ≈ 2.56 with lower-is-better), so we need a legitimate performance lift while keeping the same “type-based mean baseline” core logic. The smallest high-impact upgrade is to compute the mean at a finer granularity: use the mean by (`type`, `atom_0`, `atom_1`) derived by joining `train/test` with `structures.csv` to get element symbols for each atom index. We then back off smoothly to the existing per-`type` mean and finally the global mean for any missing combinations, preserving the same prediction approach (group means) but with richer keys. This remains leakage-safe (only uses train targets for aggregations) and should move the score substantially toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 1.23566) has done: 'I fix the crash in `make_dist_bin` by ensuring NaNs are handled before converting to integer (the current code casts a float array containing NaNs to `int32`, which raises `IntCastingNaNError`). I also add a small safeguard for any missing structure merges (resulting in NaN distances) so they get a dedicated bin value. These are execution/robustness fixes that keep the exact same modeling approach (group-mean backoff using type/atom-pair/distance-bin) and should be score-neutral aside from preventing failures. The script then run end-to-end and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 1.23566) has done: 'Your current baseline is still far from the target (lower-is-better), so the smallest legitimate lift is to keep the exact same “group-mean backoff” logic but make the distance binning smarter per coupling `type` to reduce sparsity and improve within-type calibration. Concretely, we replace the fixed `BIN_WIDTH/MAX_DIST` with per-`type` quantile bins computed from training distances, then map test distances into those bins; the rest of the backoff chain stays identical: (`type`,`atom_lo`,`atom_hi`,`dist_bin`) → (`type`,`atom_lo`,`atom_hi`) → (`type`) → global. This should improve the MAE per type (and thus the logged mean MAE metric) while preserving the original modeling approach and remaining leakage-safe. We also keep NaN/merge-miss handling robust and still write a valid `submission.csv`.'
- What this solution (achieved 1.23566) has done: 'You’re far from the target (lower-is-better), so we should improve accuracy while keeping the same group-mean/backoff core logic. The biggest easy gain with minimal semantic change is to make the “distance bin” more informative and less noisy by (1) using a log-distance transform for binning (still derived from the same coordinates) and (2) increasing bin resolution modestly while keeping per-`type` quantile bins to control sparsity. This preserves the exact same aggregation/prediction approach—only the binning feature becomes more stable across wide distance ranges—so it should move the score down toward the target without changing the modeling family. I also add a tiny safeguard to ensure bin edges are strictly increasing per type (prevents degenerate quantiles) but otherwise keep the pipeline identical and still write a valid `submission.csv`.'
- What this solution (achieved 1.23566) has done: 'You’re still far from the (better) target and lower is better, so we should legitimately reduce error without changing the overall “group-mean with backoff” approach. The biggest minimal gain is to make the distance-binning more consistent between train and test by fitting bins on the *combined* (train+test) distance distribution per `type` (unsupervised, no target leakage), while still computing means using only training targets. This reduces bin edge mismatch/shift and sparsity artifacts that can hurt MAE, but preserves the exact same prediction semantics: (`type`,`atom_lo`,`atom_hi`,`dist_bin`) → (`type`,`atom_lo`,`atom_hi`) → (`type`) → global. I also add a tiny safeguard so if a `type` is missing from the combined edge dict we fall back cleanly, keeping runtime and output format identical.'
- What this solution (achieved 1.23566) has done: 'Your current approach is a leakage-safe “group mean with backoff” model; to move the score down toward the (much better) target without changing that core logic, the smallest high-impact tweak is to make the distance feature less lossy while keeping the same aggregation scheme. Concretely, we (1) keep `dist_log` but add a second binning key that captures within-type distance rank more smoothly by using `pd.qcut`-style quantiles on the combined (train+test) distribution per type, and (2) add a very small-count smoothing/backoff so ultra-rare (`type`,`atom_lo`,`atom_hi`,`bin`) groups don’t inject noisy means. This preserves the exact same prediction semantics (hierarchical means) while reducing variance from sparse bins, which should lower MAE and thus improve the log-MAE metric. The script still run end-to-end and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 1.23566) has done: 'We keep your exact hierarchical “group-mean with backoff” model, but make the distance-binning feature slightly more informative (and less sparse) by using **two complementary distance bins** per type: your existing `dist_log` quantile bin plus an additional **linear-distance** quantile bin computed on the combined (train+test) distribution (unsupervised, so no leakage). Then we add one extra backoff level: (`type`,`atom_lo`,`atom_hi`,`dist_bin_log`,`dist_bin_lin`) → (`type`,`atom_lo`,`atom_hi`,`dist_bin_log`) → (`type`,`atom_lo`,`atom_hi`) → (`type`) → global, with the same small-count masking you already use. This is a minimal extension of the same aggregation semantics and is very likely to reduce MAE (thus lower the log-MAE score) without changing the overall approach or adding new data sources. The script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 1.23566) has done: 'We keep your exact hierarchical group-mean-with-backoff model, but make the distance conditioning less sparse and more type-appropriate by increasing the quantile bin resolution modestly (especially for linear distance) so the first-stage lookup matches more couplings with tighter local means. To avoid introducing noise from extra sparsity, we simultaneously raise the minimum-count threshold for the most granular 2D-bin group so that we only trust those finer bins when they’re well-supported, and otherwise fall back exactly as before. This preserves the same evaluation semantics (means computed only on train targets; bins fit unsupervised on train+test distances) while aiming to reduce MAE and move the score downward toward the target. The script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 1.23566) has done: 'We keep the exact same hierarchical group-mean + backoff prediction logic, but make the most granular group mean less noisy by applying simple empirical-Bayes smoothing (shrinkage) toward the next-backoff mean using the group counts you already compute. This typically lowers MAE (and thus log-MAE) versus hard cutoffs because it uses fine-grained information when available without overfitting tiny groups, so it should move your score down toward the (much better) target. The binning, features, and backoff chain stay the same; only the way we convert grouped (“mu”, “n”) into a usable mean changes from a hard threshold to a smooth, count-aware interpolation. The script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 1.23566) has done: 'We keep your exact hierarchical “group-mean with backoff + shrinkage” model, but adjust the shrinkage to be **adaptive per coupling type** instead of using one global `SHRINK_K_*` for all types. This is a minimal semantic change (still empirical-Bayes smoothing toward the same priors), but it typically reduces MAE because different coupling types have very different noise/scale characteristics. Concretely, we set `k(type) = base_k * median_group_count(type)` (clipped to safe bounds), and use that in the two shrinkage steps (2D bin → log-bin prior; log-bin → atom-pair prior). Everything else (features, bins, backoff chain, output format) stays identical and still produces `submission.csv`.'
- What this solution (achieved 1.23566) has done: 'Your current score (1.23566, lower-is-better) is still far from the target (-1.32544), so we should make a small, legitimate accuracy improvement while keeping the exact same “hierarchical group-mean with shrinkage + backoff” core logic. The most minimal high-impact tweak is to add one more conditioning variable that’s already available from your existing merge: the atom-index separation `abs(atom_index_0-atom_index_1)`, binned per type (unsupervised on train+test, no leakage). We then add just one extra (more granular) lookup level with the same style of smoothing, and fall back exactly to your existing 2D-bin → log-bin → atom-pair → type → global chain when missing. This typically reduces MAE because many coupling types correlate strongly with how many bonds separate the atoms, without changing the modeling family or training approach. The script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 1.23566) has done: 'Your current score (1.23566, lower-is-better) is still far from the target (-1.32544), so we need a real accuracy lift while keeping the same hierarchical group-mean + shrinkage core logic. The most minimal high-impact improvement is to add two physically meaningful, cheap geometric context features derived from the same merged coordinates: (1) the absolute difference in distance-to-center-of-mass between the two atoms, and (2) the sum of their distances-to-center-of-mass; both are then quantile-binned per `type` on combined train+test (unsupervised, no leakage). We then add just one more granular lookup level on top of your existing chain (… + `cm_bin`) with the same empirical-Bayes shrinkage style, and fall back exactly as before. This should reduce MAE by making the finest groups less noisy/ambiguous without changing the model family or training approach, and it still writes a valid `submission.csv`.'
- What this solution (achieved 1.23566) has done: 'Your current score is far worse than the target (lower-is-better), so we should add a small amount of extra signal while keeping the same “hierarchical group-mean with shrinkage + backoff” core logic. The minimal high-impact addition is to incorporate a molecule-level feature already present in the competition data: `potential_energy.csv`, merged by `molecule_name` and then quantile-binned per `type` (bins fit on combined train+test to avoid distribution shift, with no target leakage). We then add just one extra most-granular lookup level that includes this `pe_bin`, and shrink it toward your existing most-granular mean (without `pe_bin`) to reduce noise, falling back exactly as before. This keeps the same model family (smoothed grouped means) and should reduce MAE enough to move the score downward toward the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

BASE_DIR_CANDIDATES = [
    "/kaggle/input/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling",
    "/kaggle/data/input/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling/champs-scalar-coupling",
    "/kaggle/data/input/champs-scalar-coupling/champs-scalar-coupling",
]


def find_competition_dir(candidates):
    for d in candidates:
        if os.path.exists(d) and os.path.isfile(os.path.join(d, "train.csv")):
            return d
    for root in ["/kaggle/input", "/kaggle/data", "/kaggle/data/input"]:
        if os.path.exists(root):
            for dirpath, dirnames, filenames in os.walk(root):
                if (
                    "train.csv" in filenames
                    and "test.csv" in filenames
                    and "sample_submission.csv" in filenames
                ):
                    return dirpath
    raise FileNotFoundError(
        "Could not locate competition directory containing train.csv/test.csv/sample_submission.csv"
    )


COMP_DIR = find_competition_dir(BASE_DIR_CANDIDATES)
print("Using competition directory:", COMP_DIR)
print(
    "Files:",
    sorted([f for f in os.listdir(COMP_DIR) if f.endswith(".csv")])[:10],
    "...",
)

train_path = os.path.join(COMP_DIR, "train.csv")
test_path = os.path.join(COMP_DIR, "test.csv")
sample_path = os.path.join(COMP_DIR, "sample_submission.csv")
structures_path = os.path.join(COMP_DIR, "structures.csv")

potential_energy_path = os.path.join(COMP_DIR, "potential_energy.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample = pd.read_csv(sample_path)

print("train:", train.shape, "test:", test.shape, "sample:", sample.shape)

required_train = {
    "type",
    "scalar_coupling_constant",
    "molecule_name",
    "atom_index_0",
    "atom_index_1",
}
required_test = {"id", "type", "molecule_name", "atom_index_0", "atom_index_1"}
if not required_train.issubset(train.columns):
    raise ValueError(
        f"train.csv missing required columns: {required_train - set(train.columns)}"
    )
if not required_test.issubset(test.columns):
    raise ValueError(
        f"test.csv missing required columns: {required_test - set(test.columns)}"
    )

if not os.path.isfile(structures_path):
    raise FileNotFoundError(f"structures.csv not found at: {structures_path}")

structures = pd.read_csv(
    structures_path, usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"]
)
structures["atom_index"] = structures["atom_index"].astype(np.int32)

if not os.path.isfile(potential_energy_path):
    raise FileNotFoundError(
        f"potential_energy.csv not found at: {potential_energy_path}"
    )
potential_energy = pd.read_csv(
    potential_energy_path, usecols=["molecule_name", "potential_energy"]
)
potential_energy["potential_energy"] = potential_energy["potential_energy"].astype(
    np.float64
)


def add_atom_symbols_and_distance(df, structures_df):
    df = df.copy()
    df["atom_index_0"] = df["atom_index_0"].astype(np.int32)
    df["atom_index_1"] = df["atom_index_1"].astype(np.int32)

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

    df = df.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
    df = df.merge(s1, on=["molecule_name", "atom_index_1"], how="left")

    dx = df["x0"] - df["x1"]
    dy = df["y0"] - df["y1"]
    dz = df["z0"] - df["z1"]
    df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float64)

    return df


train_e = add_atom_symbols_and_distance(
    train[
        [
            "type",
            "scalar_coupling_constant",
            "molecule_name",
            "atom_index_0",
            "atom_index_1",
        ]
    ],
    structures,
)
test_e = add_atom_symbols_and_distance(
    test[["id", "type", "molecule_name", "atom_index_0", "atom_index_1"]], structures
)

train_e["atom_0"] = train_e["atom_0"].fillna("UNK")
train_e["atom_1"] = train_e["atom_1"].fillna("UNK")
test_e["atom_0"] = test_e["atom_0"].fillna("UNK")
test_e["atom_1"] = test_e["atom_1"].fillna("UNK")


def canonicalize_atom_pair(df):
    a0 = df["atom_0"].astype(str).values
    a1 = df["atom_1"].astype(str).values
    lo = np.minimum(a0, a1)
    hi = np.maximum(a0, a1)
    df["atom_lo"] = lo
    df["atom_hi"] = hi
    return df


train_e = canonicalize_atom_pair(train_e)
test_e = canonicalize_atom_pair(test_e)

type_mean = train.groupby("type")["scalar_coupling_constant"].mean()
global_mean = float(train["scalar_coupling_constant"].mean())

train_e["dist_log"] = np.log1p(train_e["dist"].clip(lower=0).astype(np.float64))
test_e["dist_log"] = np.log1p(test_e["dist"].clip(lower=0).astype(np.float64))

train_e["dist_lin"] = train_e["dist"].astype(np.float64)
test_e["dist_lin"] = test_e["dist"].astype(np.float64)

train_e["idx_sep"] = (
    (train_e["atom_index_0"] - train_e["atom_index_1"]).abs().astype(np.int32)
)
test_e["idx_sep"] = (
    (test_e["atom_index_0"] - test_e["atom_index_1"]).abs().astype(np.int32)
)

com = (
    structures.groupby("molecule_name", sort=False)[["x", "y", "z"]]
    .mean()
    .rename(columns={"x": "cx", "y": "cy", "z": "cz"})
    .reset_index()
)
train_e = train_e.merge(com, on="molecule_name", how="left")
test_e = test_e.merge(com, on="molecule_name", how="left")

for df in (train_e, test_e):
    r0 = np.sqrt(
        (df["x0"] - df["cx"]) ** 2
        + (df["y0"] - df["cy"]) ** 2
        + (df["z0"] - df["cz"]) ** 2
    ).astype(np.float64)
    r1 = np.sqrt(
        (df["x1"] - df["cx"]) ** 2
        + (df["y1"] - df["cy"]) ** 2
        + (df["z1"] - df["cz"]) ** 2
    ).astype(np.float64)
    df["r0_com"] = r0
    df["r1_com"] = r1
    df["r_com_sum"] = (r0 + r1).astype(np.float64)
    df["r_com_absdiff"] = (r0 - r1).abs().astype(np.float64)

train_e = train_e.merge(potential_energy, on="molecule_name", how="left")
test_e = test_e.merge(potential_energy, on="molecule_name", how="left")

pe_median = float(
    pd.concat(
        [train_e["potential_energy"], test_e["potential_energy"]], axis=0
    ).median()
)
train_e["potential_energy"] = (
    train_e["potential_energy"].fillna(pe_median).astype(np.float64)
)
test_e["potential_energy"] = (
    test_e["potential_energy"].fillna(pe_median).astype(np.float64)
)

N_BINS_LOG = 96
N_BINS_LIN = 64
N_BINS_SEP = 32

N_BINS_COMSUM = 48
N_BINS_COMDIFF = 40

N_BINS_PE = 24

BASE_SHRINK_K_2D = 0.80
BASE_SHRINK_K_LOG = 0.60
BASE_SHRINK_K_SEP = 0.70
BASE_SHRINK_K_CM = 0.75

BASE_SHRINK_K_PE = 0.75

K2D_MIN, K2D_MAX = 50.0, 1000.0
KLOG_MIN, KLOG_MAX = 30.0, 800.0
KSEP_MIN, KSEP_MAX = 40.0, 900.0
KCM_MIN, KCM_MAX = 40.0, 900.0
KPE_MIN, KPE_MAX = 40.0, 900.0


def _make_strictly_increasing(edges):
    edges = np.asarray(edges, dtype=np.float64)
    if edges.size == 0:
        return edges
    out = edges.copy()
    eps = 1e-12
    for i in range(1, out.size):
        if not np.isfinite(out[i]) or out[i] <= out[i - 1]:
            out[i] = out[i - 1] + eps
    return out


def build_type_quantile_edges_from_frames(frames, n_bins, dist_col):
    edges_by_type = {}
    comb = pd.concat([f[["type", dist_col]] for f in frames], axis=0, ignore_index=True)
    for t, g in comb.groupby("type", sort=False):
        d = g[dist_col].to_numpy(dtype=np.float64)
        d = d[np.isfinite(d)]
        if d.size < 1000:
            edges_by_type[t] = None
            continue
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(d, qs)
        edges = np.unique(edges)
        if edges.size <= 2:
            edges_by_type[t] = None
        else:
            edges_by_type[t] = _make_strictly_increasing(edges)
    return edges_by_type


def assign_type_quantile_bin(df, edges_by_type, dist_col):
    out = np.full(len(df), -1, dtype=np.int32)
    d = df[dist_col].to_numpy(dtype=np.float64)
    t = df["type"].astype(str).to_numpy()

    finite = np.isfinite(d)
    if not finite.any():
        return pd.Series(out, index=df.index, dtype=np.int32)

    for tt in np.unique(t):
        m = (t == tt) & finite
        if not m.any():
            continue
        edges = edges_by_type.get(tt, None)
        if edges is None:
            out[m] = 0
        else:
            b = np.searchsorted(edges, d[m], side="right") - 1
            b = np.clip(b, 0, len(edges) - 2).astype(np.int32)
            out[m] = b
    return pd.Series(out, index=df.index, dtype=np.int32)


type_edges_log = build_type_quantile_edges_from_frames(
    [train_e, test_e], N_BINS_LOG, dist_col="dist_log"
)
type_edges_lin = build_type_quantile_edges_from_frames(
    [train_e, test_e], N_BINS_LIN, dist_col="dist_lin"
)
type_edges_sep = build_type_quantile_edges_from_frames(
    [
        train_e.assign(idx_sep_f=train_e["idx_sep"].astype(np.float64)),
        test_e.assign(idx_sep_f=test_e["idx_sep"].astype(np.float64)),
    ],
    N_BINS_SEP,
    dist_col="idx_sep_f",
)
type_edges_comsum = build_type_quantile_edges_from_frames(
    [train_e, test_e], N_BINS_COMSUM, dist_col="r_com_sum"
)
type_edges_comdiff = build_type_quantile_edges_from_frames(
    [train_e, test_e], N_BINS_COMDIFF, dist_col="r_com_absdiff"
)

type_edges_pe = build_type_quantile_edges_from_frames(
    [train_e, test_e], N_BINS_PE, dist_col="potential_energy"
)

train_e["dist_bin_log"] = assign_type_quantile_bin(
    train_e, type_edges_log, dist_col="dist_log"
)
test_e["dist_bin_log"] = assign_type_quantile_bin(
    test_e, type_edges_log, dist_col="dist_log"
)

train_e["dist_bin_lin"] = assign_type_quantile_bin(
    train_e, type_edges_lin, dist_col="dist_lin"
)
test_e["dist_bin_lin"] = assign_type_quantile_bin(
    test_e, type_edges_lin, dist_col="dist_lin"
)

train_e["idx_sep_f"] = train_e["idx_sep"].astype(np.float64)
test_e["idx_sep_f"] = test_e["idx_sep"].astype(np.float64)
train_e["sep_bin"] = assign_type_quantile_bin(
    train_e, type_edges_sep, dist_col="idx_sep_f"
)
test_e["sep_bin"] = assign_type_quantile_bin(
    test_e, type_edges_sep, dist_col="idx_sep_f"
)

train_e["comsum_bin"] = assign_type_quantile_bin(
    train_e, type_edges_comsum, dist_col="r_com_sum"
)
test_e["comsum_bin"] = assign_type_quantile_bin(
    test_e, type_edges_comsum, dist_col="r_com_sum"
)
train_e["comdiff_bin"] = assign_type_quantile_bin(
    train_e, type_edges_comdiff, dist_col="r_com_absdiff"
)
test_e["comdiff_bin"] = assign_type_quantile_bin(
    test_e, type_edges_comdiff, dist_col="r_com_absdiff"
)

train_e["pe_bin"] = assign_type_quantile_bin(
    train_e, type_edges_pe, dist_col="potential_energy"
)
test_e["pe_bin"] = assign_type_quantile_bin(
    test_e, type_edges_pe, dist_col="potential_energy"
)

type_atom_mean = train_e.groupby(["type", "atom_lo", "atom_hi"], sort=False)[
    "scalar_coupling_constant"
].mean()

grp_cols_log = ["type", "atom_lo", "atom_hi", "dist_bin_log"]
type_atom_distlog_stats = (
    train_e.groupby(grp_cols_log, sort=False)["scalar_coupling_constant"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "mu", "count": "n"})
)

grp_cols_2d = ["type", "atom_lo", "atom_hi", "dist_bin_log", "dist_bin_lin"]
type_atom_dist2_stats = (
    train_e.groupby(grp_cols_2d, sort=False)["scalar_coupling_constant"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "mu", "count": "n"})
)

grp_cols_3d = ["type", "atom_lo", "atom_hi", "dist_bin_log", "dist_bin_lin", "sep_bin"]
type_atom_dist2sep_stats = (
    train_e.groupby(grp_cols_3d, sort=False)["scalar_coupling_constant"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "mu", "count": "n"})
)

grp_cols_5d = [
    "type",
    "atom_lo",
    "atom_hi",
    "dist_bin_log",
    "dist_bin_lin",
    "sep_bin",
    "comsum_bin",
    "comdiff_bin",
]
type_atom_dist2sepcom_stats = (
    train_e.groupby(grp_cols_5d, sort=False)["scalar_coupling_constant"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "mu", "count": "n"})
)

grp_cols_6d = grp_cols_5d + ["pe_bin"]
type_atom_dist2sepcompe_stats = (
    train_e.groupby(grp_cols_6d, sort=False)["scalar_coupling_constant"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "mu", "count": "n"})
)

klog_by_type = (
    type_atom_distlog_stats.reset_index()
    .groupby("type", sort=False)["n"]
    .median()
    .astype(np.float64)
)
k2d_by_type = (
    type_atom_dist2_stats.reset_index()
    .groupby("type", sort=False)["n"]
    .median()
    .astype(np.float64)
)
ksep_by_type = (
    type_atom_dist2sep_stats.reset_index()
    .groupby("type", sort=False)["n"]
    .median()
    .astype(np.float64)
)
kcm_by_type = (
    type_atom_dist2sepcom_stats.reset_index()
    .groupby("type", sort=False)["n"]
    .median()
    .astype(np.float64)
)
kpe_by_type = (
    type_atom_dist2sepcompe_stats.reset_index()
    .groupby("type", sort=False)["n"]
    .median()
    .astype(np.float64)
)

klog_by_type = (BASE_SHRINK_K_LOG * klog_by_type).clip(KLOG_MIN, KLOG_MAX)
k2d_by_type = (BASE_SHRINK_K_2D * k2d_by_type).clip(K2D_MIN, K2D_MAX)
ksep_by_type = (BASE_SHRINK_K_SEP * ksep_by_type).clip(KSEP_MIN, KSEP_MAX)
kcm_by_type = (BASE_SHRINK_K_CM * kcm_by_type).clip(KCM_MIN, KCM_MAX)
kpe_by_type = (BASE_SHRINK_K_PE * kpe_by_type).clip(KPE_MIN, KPE_MAX)

type_atom_distlog_stats = type_atom_distlog_stats.reset_index()
type_atom_distlog_stats["prior"] = type_atom_distlog_stats.set_index(
    ["type", "atom_lo", "atom_hi"]
).index.map(type_atom_mean)
type_atom_distlog_stats["prior"] = type_atom_distlog_stats["prior"].astype(np.float64)

type_atom_distlog_stats["k"] = (
    type_atom_distlog_stats["type"].map(klog_by_type).astype(np.float64)
)
type_atom_distlog_stats["mu_smooth"] = (
    type_atom_distlog_stats["n"].astype(np.float64)
    * type_atom_distlog_stats["mu"].astype(np.float64)
    + type_atom_distlog_stats["k"] * type_atom_distlog_stats["prior"]
) / (type_atom_distlog_stats["n"].astype(np.float64) + type_atom_distlog_stats["k"])
type_atom_distlog_mu_smooth = type_atom_distlog_stats.set_index(grp_cols_log)[
    "mu_smooth"
]

type_atom_dist2_stats = type_atom_dist2_stats.reset_index()
type_atom_dist2_stats["prior"] = type_atom_dist2_stats.set_index(
    ["type", "atom_lo", "atom_hi", "dist_bin_log"]
).index.map(type_atom_distlog_mu_smooth)
type_atom_dist2_stats["prior"] = type_atom_dist2_stats["prior"].astype(np.float64)

type_atom_dist2_stats["k"] = (
    type_atom_dist2_stats["type"].map(k2d_by_type).astype(np.float64)
)
type_atom_dist2_stats["mu_smooth"] = (
    type_atom_dist2_stats["n"].astype(np.float64)
    * type_atom_dist2_stats["mu"].astype(np.float64)
    + type_atom_dist2_stats["k"] * type_atom_dist2_stats["prior"]
) / (type_atom_dist2_stats["n"].astype(np.float64) + type_atom_dist2_stats["k"])
type_atom_dist2_mu_smooth = type_atom_dist2_stats.set_index(grp_cols_2d)["mu_smooth"]

type_atom_dist2sep_stats = type_atom_dist2sep_stats.reset_index()
type_atom_dist2sep_stats["prior"] = type_atom_dist2sep_stats.set_index(
    ["type", "atom_lo", "atom_hi", "dist_bin_log", "dist_bin_lin"]
).index.map(type_atom_dist2_mu_smooth)
type_atom_dist2sep_stats["prior"] = type_atom_dist2sep_stats["prior"].astype(np.float64)
type_atom_dist2sep_stats["k"] = (
    type_atom_dist2sep_stats["type"].map(ksep_by_type).astype(np.float64)
)
type_atom_dist2sep_stats["mu_smooth"] = (
    type_atom_dist2sep_stats["n"].astype(np.float64)
    * type_atom_dist2sep_stats["mu"].astype(np.float64)
    + type_atom_dist2sep_stats["k"] * type_atom_dist2sep_stats["prior"]
) / (type_atom_dist2sep_stats["n"].astype(np.float64) + type_atom_dist2sep_stats["k"])
type_atom_dist2sep_mu_smooth = type_atom_dist2sep_stats.set_index(grp_cols_3d)[
    "mu_smooth"
]

type_atom_dist2sepcom_stats = type_atom_dist2sepcom_stats.reset_index()
type_atom_dist2sepcom_stats["prior"] = type_atom_dist2sepcom_stats.set_index(
    ["type", "atom_lo", "atom_hi", "dist_bin_log", "dist_bin_lin", "sep_bin"]
).index.map(type_atom_dist2sep_mu_smooth)
type_atom_dist2sepcom_stats["prior"] = type_atom_dist2sepcom_stats["prior"].astype(
    np.float64
)
type_atom_dist2sepcom_stats["k"] = (
    type_atom_dist2sepcom_stats["type"].map(kcm_by_type).astype(np.float64)
)
type_atom_dist2sepcom_stats["mu_smooth"] = (
    type_atom_dist2sepcom_stats["n"].astype(np.float64)
    * type_atom_dist2sepcom_stats["mu"].astype(np.float64)
    + type_atom_dist2sepcom_stats["k"] * type_atom_dist2sepcom_stats["prior"]
) / (
    type_atom_dist2sepcom_stats["n"].astype(np.float64)
    + type_atom_dist2sepcom_stats["k"]
)
type_atom_dist2sepcom_mu_smooth = type_atom_dist2sepcom_stats.set_index(grp_cols_5d)[
    "mu_smooth"
]

type_atom_dist2sepcompe_stats = type_atom_dist2sepcompe_stats.reset_index()
type_atom_dist2sepcompe_stats["prior"] = type_atom_dist2sepcompe_stats.set_index(
    grp_cols_5d
).index.map(type_atom_dist2sepcom_mu_smooth)
type_atom_dist2sepcompe_stats["prior"] = type_atom_dist2sepcompe_stats["prior"].astype(
    np.float64
)
type_atom_dist2sepcompe_stats["k"] = (
    type_atom_dist2sepcompe_stats["type"].map(kpe_by_type).astype(np.float64)
)
type_atom_dist2sepcompe_stats["mu_smooth"] = (
    type_atom_dist2sepcompe_stats["n"].astype(np.float64)
    * type_atom_dist2sepcompe_stats["mu"].astype(np.float64)
    + type_atom_dist2sepcompe_stats["k"] * type_atom_dist2sepcompe_stats["prior"]
) / (
    type_atom_dist2sepcompe_stats["n"].astype(np.float64)
    + type_atom_dist2sepcompe_stats["k"]
)
type_atom_dist2sepcompe_mu_smooth = type_atom_dist2sepcompe_stats.set_index(
    grp_cols_6d
)["mu_smooth"]

key0000 = list(
    zip(
        test_e["type"].values,
        test_e["atom_lo"].values,
        test_e["atom_hi"].values,
        test_e["dist_bin_log"].values,
        test_e["dist_bin_lin"].values,
        test_e["sep_bin"].values,
        test_e["comsum_bin"].values,
        test_e["comdiff_bin"].values,
        test_e["pe_bin"].values,
    )
)
pred = pd.Series(key0000, index=test_e.index).map(type_atom_dist2sepcompe_mu_smooth)

key000 = list(
    zip(
        test_e["type"].values,
        test_e["atom_lo"].values,
        test_e["atom_hi"].values,
        test_e["dist_bin_log"].values,
        test_e["dist_bin_lin"].values,
        test_e["sep_bin"].values,
        test_e["comsum_bin"].values,
        test_e["comdiff_bin"].values,
    )
)
pred = pred.fillna(
    pd.Series(key000, index=test_e.index).map(type_atom_dist2sepcom_mu_smooth)
)

key00 = list(
    zip(
        test_e["type"].values,
        test_e["atom_lo"].values,
        test_e["atom_hi"].values,
        test_e["dist_bin_log"].values,
        test_e["dist_bin_lin"].values,
        test_e["sep_bin"].values,
    )
)
pred = pred.fillna(
    pd.Series(key00, index=test_e.index).map(type_atom_dist2sep_mu_smooth)
)

key0 = list(
    zip(
        test_e["type"].values,
        test_e["atom_lo"].values,
        test_e["atom_hi"].values,
        test_e["dist_bin_log"].values,
        test_e["dist_bin_lin"].values,
    )
)
pred = pred.fillna(pd.Series(key0, index=test_e.index).map(type_atom_dist2_mu_smooth))

key1 = list(
    zip(
        test_e["type"].values,
        test_e["atom_lo"].values,
        test_e["atom_hi"].values,
        test_e["dist_bin_log"].values,
    )
)
pred = pred.fillna(pd.Series(key1, index=test_e.index).map(type_atom_distlog_mu_smooth))

key2 = list(
    zip(test_e["type"].values, test_e["atom_lo"].values, test_e["atom_hi"].values)
)
pred = pred.fillna(pd.Series(key2, index=test_e.index).map(type_atom_mean))
pred = pred.fillna(test_e["type"].map(type_mean)).fillna(global_mean).astype(np.float64)

submission = pd.DataFrame(
    {"id": test_e["id"].astype(np.int64), "scalar_coupling_constant": pred.values}
)

if "id" in sample.columns and len(sample) == len(submission):
    submission = sample[["id"]].merge(submission, on="id", how="left")
    submission["scalar_coupling_constant"] = (
        submission["scalar_coupling_constant"].fillna(global_mean).astype(np.float64)
    )

assert submission.columns.tolist() == ["id", "scalar_coupling_constant"]
assert submission["id"].isna().sum() == 0
assert submission["scalar_coupling_constant"].isna().sum() == 0
assert len(submission) == len(test)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission.head())
print(submission["scalar_coupling_constant"].describe())



## === cell 1
try:
    ax = submission["scalar_coupling_constant"].plot(
        kind="hist", bins=100, title="Predicted scalar_coupling_constant"
    )
    fig = ax.get_figure()
    fig.tight_layout()
except Exception as e:
    print("Plot skipped:", repr(e))
