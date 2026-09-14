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

-1.4419498537699864

# 6. Current score

1.23566

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.99777) has done: 'I remove the dependency on missing Kaggle “../input/top-mol” and other external kernel directories by switching to the files that actually exist in your environment (the provided CHAMPS dataset). Since no model is present here to generate predictions, I make the pipeline run end-to-end by producing a valid baseline submission from `sample_submission.csv` (all zeros), ensuring correct columns, `.csv` suffix, and row alignment. I also fix notebook-only syntax (`%matplotlib inline`) and deprecated NumPy usage (`np.bool`) so the script runs as a plain Python script. The plotting/stacking cells are kept but guarded so they don’t crash when optional blend inputs are absent.'
- What this solution (achieved 1.23566) has done: 'Your current score is far worse than the target (lower is better), mainly because the code generates an all-zeros baseline from `sample_submission.csv`. To move the score sharply toward the target while preserving a simple/consistent “single-pass baseline” core, I replace the zero-fill with a lightweight, per-`type` training-mean regressor (and a global fallback) computed from `train.csv`, then merge onto `test.csv` by `type`. This keeps the pipeline fast (<600s), avoids any molecule leakage (uses only type), and guarantees correct id alignment and a valid `submission.csv`. All other stacking/blending cells remain but stay safely guarded when external blend files are absent.'
- What this solution (achieved 1.23566) has done: 'We need to move your score down (lower-is-better) from 1.23566 toward the target -1.44195, so we should improve the baseline without changing the overall “single-pass, no ML training loop” core approach. The biggest low-risk gain is to condition on more informative grouping keys than just `type`: using `(type, atom_0, atom_1)` where `atom_0/atom_1` come from `structures.csv` joined by molecule and atom indices, and then backing off to `type` mean and finally global mean. This still preserves your current semantics (a deterministic mean regressor) but makes predictions far closer to reality, which should reduce MAE substantially. I also add small correctness checks to ensure `id` alignment and guaranteed output of `submission.csv` with required columns.'
- What this solution (achieved 1.23566) has done: 'We fix the `IntCastingNaNError` by ensuring `dist` is always finite before binning: fill missing coordinates/distances, and use a sentinel bin (`-1`) for missing `dist` so the `.astype("int16")` conversion is safe. This keeps your existing mean-regressor/backoff logic intact while making the pipeline run end-to-end and always write a valid `submission.csv`. We also add small defensive checks to guarantee merge keys are consistent and IDs stay aligned with `sample_submission.csv`. The stacking/blending cells remain unchanged and safely skip when external files are absent.'
- What this solution (achieved 1.23566) has done: 'To move the score down (lower-is-better) toward the target while keeping your “group-mean regressor with backoff” core logic unchanged, I make the grouping key more specific by normalizing the atom-pair order (swap so `(atom_0, atom_1)` is always sorted) and using a canonical element-pair key for both train and test. This reduces fragmentation and increases hit-rate of the fine/dist-bin means without adding any new modeling method. I also compute `dist_bin` from the true distance without clipping to 5.0 (still binned) so long-range couplings aren’t collapsed into the same bin, which typically reduces MAE. Everything else (mean tables + backoff + submission writing and id alignment) stays the same, and it still writes `submission.csv`.'
- What this solution (achieved 1.23566) has done: 'Your current score (1.23566, lower-is-better) is far from the target (-1.44195), so we should materially improve predictions while keeping your core “group-mean regressor with backoff” logic intact. The biggest low-risk gain is to add a little more chemical context without introducing a new model: incorporate per-atom Mulliken charge and shielding tensor summaries for the two atoms, and then extend the grouping key to include coarse bins of these values plus your existing distance bin. To avoid over-fragmentation (which can hurt), we keep the same backoff chain but add intermediate levels (dist+charge/shield bins → dist only → fine only → type → global). All changes are deterministic, keep the same training-free mean-encoding approach, and still produce a valid `submission.csv` aligned by `id`.'
- What this solution (achieved 1.23566) has done: 'Your current score (1.23566, lower-is-better) is far from the target (-1.44195), so we should improve the baseline while keeping your core “group-mean regressor with backoff” logic unchanged. The biggest low-risk gain is to make the finest buckets less sparse by using *signed* bins for Mulliken charge and shielding (so negative values don’t collapse into the same extreme bin) and to canonicalize the atom-pair by **atom indices** (not element symbols) to correctly treat inherently ordered couplings like 2JHC/3JHC. We keep the same backoff chain and dataset usage, but improve hit-rate and reduce noise in the most specific mean tables. The script still runs end-to-end, stays deterministic, and writes a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 1.23566) has done: 'Your current score (1.23566, lower-is-better) is far worse than the target (-1.44195), so we should make the smallest change that materially improves accuracy while keeping your “group-mean regressor with backoff” core logic intact. The biggest issue is that you canonicalize (swap) atom pairs, which is unsafe for ordered coupling types like *JHC* and *JHN* where atom roles matter; removing this swap should reduce error without changing the overall approach. I also add two very cheap intermediate backoff levels: `(type, dist_bin)` and `(type, dist_bin, mc_sum_bin, st_sum_bin, sf_sum_bin)` to improve coverage when `atom_pair` is sparse, while preserving your existing mean-table prediction semantics. Everything else (data sources, mean-encoding style, submission writing and ID alignment) remains the same and still produces `submission.csv`.'
- What this solution (achieved 1.23566) has done: 'Your current score (1.23566, lower-is-better) is still very far from the target (-1.44195), so we should improve accuracy while keeping the same core “group-mean regressor with backoff” approach. The smallest high-impact fix is to prevent target leakage via mean-encoding on the same rows being predicted: we build all mean tables using an out-of-fold (molecule-group) scheme on the training set, then use those OOF predictions for better-calibrated fallbacks and to avoid over-optimistic buckets. We also replace the single global mean fallback with a molecule-agnostic, type-aware OOF baseline (and keep the same backoff chain), which typically reduces MAE materially without changing the modeling paradigm. Submission writing, id alignment, and the optional stacking/blending cells remain intact and still safely skip when inputs are missing.'
- What this solution (achieved 1.23566) has done: 'I fix the runtime error in `add_pair_key` caused by applying `np.minimum/maximum` to pandas `string` dtype that can contain `pd.NA`, by replacing it with a safe, vectorized compare-and-swap that never evaluates NA as boolean. This is score-neutral (it only affects feature creation) but unblocks the pipeline so it runs end-to-end and writes `submission.csv`. I also keep the rest of your mean-encoding/backoff logic unchanged and add a small guard to ensure `atom_pair` is always a regular string (no `<NA>` concatenation issues). Finally, the optional stacking/blending cells remain intact and safely skip when their external inputs are absent.'
- What this solution (achieved 1.23566) has done: 'We need to move the score down (lower-is-better) from 1.23566 toward the much better target -1.44195, so the smallest meaningful improvement is to make the mean-encoding tables less sparse and more physically aligned without changing the overall “group-mean regressor with backoff” approach. I keep your exact pipeline structure but (1) stop canonicalizing `atom_pair` (ordered couplings depend on role), and (2) replace the very high-cardinality `(type, atom_pair, dist_bin, mc/st/sf bins)` finest key with a slightly less sparse “physics-first” key based on `(type, atom_0, atom_1, dist_bin)` plus coarse charge/shield bins, while keeping your existing backoff chain. I also compute `dist_bin` with a slightly larger bin width (0.5) to increase key hit-rate (less missing means → lower MAE) while preserving the same binned-distance semantics. Submission writing, ID alignment checks, and all optional stacking/blending cells remain intact and still safely skip when external inputs are absent.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "/kaggle/data/champs-scalar-coupling"
if not os.path.exists(DATA_DIR):
    DATA_DIR = "/kaggle/input/champs-scalar-coupling"

print("Using DATA_DIR:", DATA_DIR)
print("Files in DATA_DIR (first 20):", sorted(os.listdir(DATA_DIR))[:20])



## === cell 1
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
train_path = os.path.join(DATA_DIR, "train.csv")
structures_path = os.path.join(DATA_DIR, "structures.csv")
mulliken_path = os.path.join(DATA_DIR, "mulliken_charges.csv")
shield_path = os.path.join(DATA_DIR, "magnetic_shielding_tensors.csv")

sample_sub = pd.read_csv(sample_path, usecols=["id", "scalar_coupling_constant"])
test = pd.read_csv(
    test_path, usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"]
)
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

structures = pd.read_csv(
    structures_path, usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"]
)
structures.rename(columns={"atom": "atom_symbol"}, inplace=True)

mulliken = pd.read_csv(
    mulliken_path, usecols=["molecule_name", "atom_index", "mulliken_charge"]
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
)

shield_vals = shield[["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]].astype(
    "float32"
)
shield["shield_trace"] = (shield["XX"] + shield["YY"] + shield["ZZ"]).astype("float32")
shield["shield_frob"] = np.sqrt(
    (shield_vals * shield_vals).sum(axis=1).astype("float32")
).astype("float32")
shield_small = shield[["molecule_name", "atom_index", "shield_trace", "shield_frob"]]
del shield, shield_vals

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

m0 = mulliken.rename(columns={"atom_index": "atom_index_0", "mulliken_charge": "mc0"})
m1 = mulliken.rename(columns={"atom_index": "atom_index_1", "mulliken_charge": "mc1"})
sh0 = shield_small.rename(
    columns={"atom_index": "atom_index_0", "shield_trace": "st0", "shield_frob": "sf0"}
)
sh1 = shield_small.rename(
    columns={"atom_index": "atom_index_1", "shield_trace": "st1", "shield_frob": "sf1"}
)

train = train.merge(
    s0[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
    validate="many_to_one",
)
train = train.merge(
    s1[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
    validate="many_to_one",
)
train = train.merge(
    m0[["molecule_name", "atom_index_0", "mc0"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
    validate="many_to_one",
)
train = train.merge(
    m1[["molecule_name", "atom_index_1", "mc1"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
    validate="many_to_one",
)
train = train.merge(
    sh0[["molecule_name", "atom_index_0", "st0", "sf0"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
    validate="many_to_one",
)
train = train.merge(
    sh1[["molecule_name", "atom_index_1", "st1", "sf1"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
    validate="many_to_one",
)

test = test.merge(
    s0[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
    validate="many_to_one",
)
test = test.merge(
    s1[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
    validate="many_to_one",
)
test = test.merge(
    m0[["molecule_name", "atom_index_0", "mc0"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
    validate="many_to_one",
)
test = test.merge(
    m1[["molecule_name", "atom_index_1", "mc1"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
    validate="many_to_one",
)
test = test.merge(
    sh0[["molecule_name", "atom_index_0", "st0", "sf0"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
    validate="many_to_one",
)
test = test.merge(
    sh1[["molecule_name", "atom_index_1", "st1", "sf1"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
    validate="many_to_one",
)

missing_train_atoms = int(train["atom_0"].isna().sum() + train["atom_1"].isna().sum())
missing_test_atoms = int(test["atom_0"].isna().sum() + test["atom_1"].isna().sum())
print(
    "Missing joined atom symbols - train:",
    missing_train_atoms,
    "| test:",
    missing_test_atoms,
)


def add_pair_key(df: pd.DataFrame) -> pd.DataFrame:
    a0 = df["atom_0"].astype("string")
    a1 = df["atom_1"].astype("string")
    df["atom_pair"] = (a0.fillna("UNK") + "_" + a1.fillna("UNK")).astype("string")
    return df


train = add_pair_key(train)
test = add_pair_key(test)

for df in (train, test):
    x0 = pd.to_numeric(df["x0"], errors="coerce").astype("float32")
    y0 = pd.to_numeric(df["y0"], errors="coerce").astype("float32")
    z0 = pd.to_numeric(df["z0"], errors="coerce").astype("float32")
    x1 = pd.to_numeric(df["x1"], errors="coerce").astype("float32")
    y1 = pd.to_numeric(df["y1"], errors="coerce").astype("float32")
    z1 = pd.to_numeric(df["z1"], errors="coerce").astype("float32")

    dx = x0 - x1
    dy = y0 - y1
    dz = z0 - z1

    dist = np.sqrt(dx * dx + dy * dy + dz * dz).astype("float32")
    dist = pd.Series(dist).where(np.isfinite(dist), np.nan).astype("float32")
    df["dist"] = dist.values

bin_width = 0.50
for df in (train, test):
    d = pd.to_numeric(df["dist"], errors="coerce")
    d = d.clip(lower=0.0)
    dist_bin = np.floor(d / bin_width)
    dist_bin = dist_bin.where(dist_bin.notna(), -1)
    df["dist_bin"] = dist_bin.astype("int16")


def add_aux_bins(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["mc0", "mc1", "st0", "st1", "sf0", "sf1"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    def signed_bin(
        s: pd.Series, bw: float, clip_abs: float, sentinel: int = -999
    ) -> pd.Series:
        x = s.astype("float32")
        x = x.clip(lower=-clip_abs, upper=clip_abs)
        b = np.floor(x / bw)
        b = b.where(b.notna(), sentinel)
        return b.astype("int16")

    mc_bw = (
        0.10  # Change (score-improving, minimal): coarser bins => less fragmentation.
    )
    for c in ["mc0", "mc1"]:
        df[c + "_bin"] = signed_bin(df[c], bw=mc_bw, clip_abs=2.0, sentinel=-999)

    st_bw = 2.0  # Change (score-improving, minimal): coarser bins => higher coverage.
    for c in ["st0", "st1"]:
        df[c + "_bin"] = signed_bin(df[c], bw=st_bw, clip_abs=50.0, sentinel=-999)

    sf_bw = 1.0  # Change (score-improving, minimal): coarser bins => higher coverage.
    for c in ["sf0", "sf1"]:
        df[c + "_bin"] = signed_bin(df[c], bw=sf_bw, clip_abs=50.0, sentinel=-999)

    df["mc_sum_bin"] = (
        (df["mc0_bin"].astype("int32") + df["mc1_bin"].astype("int32"))
        .clip(-32768, 32767)
        .astype("int16")
    )
    df["st_sum_bin"] = (
        (df["st0_bin"].astype("int32") + df["st1_bin"].astype("int32"))
        .clip(-32768, 32767)
        .astype("int16")
    )
    df["sf_sum_bin"] = (
        (df["sf0_bin"].astype("int32") + df["sf1_bin"].astype("int32"))
        .clip(-32768, 32767)
        .astype("int16")
    )
    return df


train = add_aux_bins(train)
test = add_aux_bins(test)


def make_molecule_folds(molecule_names: pd.Series, n_folds: int = 5) -> pd.Series:
    mols = molecule_names.drop_duplicates().sort_values()
    h = pd.util.hash_pandas_object(mols, index=False).astype("uint64")
    fold = (h % np.uint64(n_folds)).astype("int8")
    fold_map = pd.DataFrame({"molecule_name": mols.values, "fold": fold.values})
    return molecule_names.to_frame().merge(fold_map, on="molecule_name", how="left")[
        "fold"
    ]


train["fold"] = make_molecule_folds(train["molecule_name"], n_folds=5).astype("int8")

key_cols_dist_aux = [
    "type",
    "atom_0",
    "atom_1",
    "dist_bin",
    "mc_sum_bin",
    "st_sum_bin",
    "sf_sum_bin",
]
key_cols_dist = ["type", "atom_0", "atom_1", "dist_bin"]
key_cols_fine = ["type", "atom_0", "atom_1"]

key_cols_dist_aux_type = ["type", "dist_bin", "mc_sum_bin", "st_sum_bin", "sf_sum_bin"]
key_cols_dist_type = ["type", "dist_bin"]


def oof_mean_table(
    df: pd.DataFrame, key_cols: list, target_col: str, fold_col: str, pred_name: str
) -> pd.DataFrame:
    parts = []
    for f in sorted(df[fold_col].unique()):
        trn = df[df[fold_col] != f]
        val = df[df[fold_col] == f][key_cols].copy()
        m = (
            trn.groupby(key_cols, sort=False)[target_col]
            .mean()
            .rename(pred_name)
            .reset_index()
        )
        val = val.merge(m, on=key_cols, how="left", validate="many_to_one")
        parts.append(val)
    oof = pd.concat(parts, axis=0, ignore_index=True)
    return oof.groupby(key_cols, sort=False)[pred_name].mean().reset_index()


dist_aux_mean = oof_mean_table(
    train, key_cols_dist_aux, "scalar_coupling_constant", "fold", "pred_dist_aux"
)
dist_mean = oof_mean_table(
    train, key_cols_dist, "scalar_coupling_constant", "fold", "pred_dist"
)
fine_mean = oof_mean_table(
    train, key_cols_fine, "scalar_coupling_constant", "fold", "pred_fine"
)
dist_aux_type_mean = oof_mean_table(
    train,
    key_cols_dist_aux_type,
    "scalar_coupling_constant",
    "fold",
    "pred_dist_aux_type",
)
dist_type_mean = oof_mean_table(
    train, key_cols_dist_type, "scalar_coupling_constant", "fold", "pred_dist_type"
)
type_mean = oof_mean_table(
    train, ["type"], "scalar_coupling_constant", "fold", "pred_type"
)

global_mean = float(train["scalar_coupling_constant"].mean())

sub = test.merge(
    dist_aux_mean, on=key_cols_dist_aux, how="left", validate="many_to_one"
)
sub = sub.merge(dist_mean, on=key_cols_dist, how="left", validate="many_to_one")
sub = sub.merge(fine_mean, on=key_cols_fine, how="left", validate="many_to_one")
sub = sub.merge(
    dist_aux_type_mean, on=key_cols_dist_aux_type, how="left", validate="many_to_one"
)
sub = sub.merge(
    dist_type_mean, on=key_cols_dist_type, how="left", validate="many_to_one"
)
sub = sub.merge(type_mean, on=["type"], how="left", validate="many_to_one")

sub["scalar_coupling_constant"] = sub["pred_dist_aux"]
sub["scalar_coupling_constant"] = sub["scalar_coupling_constant"].fillna(
    sub["pred_dist"]
)
sub["scalar_coupling_constant"] = sub["scalar_coupling_constant"].fillna(
    sub["pred_fine"]
)
sub["scalar_coupling_constant"] = sub["scalar_coupling_constant"].fillna(
    sub["pred_dist_aux_type"]
)
sub["scalar_coupling_constant"] = sub["scalar_coupling_constant"].fillna(
    sub["pred_dist_type"]
)
sub["scalar_coupling_constant"] = sub["scalar_coupling_constant"].fillna(
    sub["pred_type"]
)
sub["scalar_coupling_constant"] = (
    sub["scalar_coupling_constant"].fillna(global_mean).astype("float64")
)

sub = sub[["id", "scalar_coupling_constant"]]
sub = sub.sort_values("id").reset_index(drop=True)

sample_ids = sample_sub.sort_values("id")["id"].to_numpy()
sub_ids = sub["id"].to_numpy()
if len(sample_ids) == len(sub_ids) and np.array_equal(sample_ids, sub_ids):
    print("ID alignment check: OK (matches sample_submission)")
else:
    print(
        "ID alignment check: WARNING (ids differ from sample_submission)",
        "| sample_n=",
        len(sample_ids),
        "| sub_n=",
        len(sub_ids),
    )

out_path = "submission.csv"
sub.to_csv(out_path, index=False, float_format="%.6f")

print("Wrote", out_path, "shape=", sub.shape, "| global_mean=", global_mean)
print(sub.head())



## === cell 2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

for p in ["/kaggle/input", "/kaggle/data", "../input"]:
    if os.path.exists(p):
        print(f"ls {p} ->", sorted(os.listdir(p))[:20])



## === cell 3
sub_path = "../input/top-mol"
if os.path.isdir(sub_path):
    all_files = os.listdir(sub_path)
else:
    all_files = []
print("sub_path exists:", os.path.isdir(sub_path), "| n_files:", len(all_files))



## === cell 4
if all_files:
    outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files]
    concat_sub = pd.concat(outs, axis=1)
    cols = list(map(lambda x: "mol" + str(x), range(len(concat_sub.columns))))
    concat_sub.columns = cols
    concat_sub.reset_index(inplace=True)
    ncol = concat_sub.shape[1]
    print(concat_sub.head())
else:
    concat_sub = None
    ncol = 0
    print("Skipping stacking: no files found in", sub_path)



## === cell 5
if concat_sub is not None:
    _ = concat_sub.iloc[:, 1:ncol].corr()
    print(_.iloc[:5, :5])
else:
    print("Skipping correlation: concat_sub is None")



## === cell 6
if concat_sub is not None:
    corr = concat_sub.iloc[:, 1 : min(7, ncol)].corr()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=0.3,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        ax=ax,
    )
    plt.close(f)
    print("Heatmap created (not displayed in script mode).")
else:
    print("Skipping heatmap: concat_sub is None")



## === cell 7
if concat_sub is not None:
    concat_sub["m_max"] = concat_sub.iloc[:, 1:ncol].max(axis=1)
    concat_sub["m_min"] = concat_sub.iloc[:, 1:ncol].min(axis=1)
    concat_sub["m_mean"] = concat_sub.iloc[:, 1:ncol].mean(axis=1)
    concat_sub["m_median"] = concat_sub.iloc[:, 1:ncol].median(axis=1)
    print(concat_sub[["m_max", "m_min", "m_mean", "m_median"]].head())
else:
    print("Skipping stacking stats: concat_sub is None")



## === cell 8
if concat_sub is not None:
    print(concat_sub.describe().T.head(15))
else:
    print("Skipping describe: concat_sub is None")



## === cell 9
cutoff_lo = 0.8
cutoff_hi = 0.2
print("cutoff_lo, cutoff_hi:", cutoff_lo, cutoff_hi)



## === cell 10
pass



## === cell 11
if concat_sub is not None and "m_mean" in concat_sub.columns:
    concat_sub["scalar_coupling_constant"] = concat_sub["m_mean"]
    concat_sub[["id", "scalar_coupling_constant"]].to_csv(
        "stack_mean.csv", index=False, float_format="%.6f"
    )
    print("Wrote stack_mean.csv")
else:
    print("Skipping stack_mean.csv: concat_sub not available")



## === cell 12
pass



## === cell 13
if concat_sub is not None and "m_median" in concat_sub.columns:
    concat_sub["scalar_coupling_constant"] = concat_sub["m_median"]
    concat_sub[["id", "scalar_coupling_constant"]].to_csv(
        "stack_median.csv", index=False, float_format="%.6f"
    )
    print("Wrote stack_median.csv")
else:
    print("Skipping stack_median.csv: concat_sub not available")



## === cell 14
pass



## === cell 15
if concat_sub is not None and ncol > 1 and "m_median" in concat_sub.columns:
    concat_sub["scalar_coupling_constant"] = np.where(
        np.all(concat_sub.iloc[:, 1:ncol] > cutoff_lo, axis=1),
        1,
        np.where(
            np.all(concat_sub.iloc[:, 1:ncol] < cutoff_hi, axis=1),
            0,
            concat_sub["m_median"],
        ),
    )
    concat_sub[["id", "scalar_coupling_constant"]].to_csv(
        "stack_pushout_median.csv", index=False, float_format="%.6f"
    )
    print("Wrote stack_pushout_median.csv")
else:
    print("Skipping stack_pushout_median.csv: concat_sub not available")



## === cell 16
pass



## === cell 17
if concat_sub is not None and ncol > 1:
    concat_sub["scalar_coupling_constant"] = np.where(
        np.all(concat_sub.iloc[:, 1:ncol] > cutoff_lo, axis=1),
        concat_sub["m_max"],
        np.where(
            np.all(concat_sub.iloc[:, 1:ncol] < cutoff_hi, axis=1),
            concat_sub["m_min"],
            concat_sub["m_mean"],
        ),
    )
    concat_sub[["id", "scalar_coupling_constant"]].to_csv(
        "stack_minmax_mean.csv", index=False, float_format="%.6f"
    )
    print("Wrote stack_minmax_mean.csv")
else:
    print("Skipping stack_minmax_mean.csv: concat_sub not available")



## === cell 18
pass



## === cell 19
if concat_sub is not None and ncol > 1:
    concat_sub["scalar_coupling_constant"] = np.where(
        np.all(concat_sub.iloc[:, 1:ncol] > cutoff_lo, axis=1),
        concat_sub["m_max"],
        np.where(
            np.all(concat_sub.iloc[:, 1:ncol] < cutoff_hi, axis=1),
            concat_sub["m_min"],
            concat_sub["m_median"],
        ),
    )
    concat_sub[["id", "scalar_coupling_constant"]].to_csv(
        "stack_minmax_median.csv", index=False, float_format="%.6f"
    )
    print("Wrote stack_minmax_median.csv")
else:
    print("Skipping stack_minmax_median.csv: concat_sub not available")



## === cell 20
pass



## === cell 21
if concat_sub is not None and all(
    c in concat_sub.columns for c in ["mol0", "mol1", "mol2"]
):
    concat_sub["scalar_coupling_constant"] = (
        concat_sub["mol0"].rank(method="min")
        + concat_sub["mol1"].rank(method="min")
        + concat_sub["mol2"].rank(method="min")
    )
    s = concat_sub["scalar_coupling_constant"]
    concat_sub["scalar_coupling_constant"] = (s - s.min()) / (s.max() - s.min())
    concat_sub[["id", "scalar_coupling_constant"]].to_csv(
        "stack_rank.csv", index=False, float_format="%.8f"
    )
    print("Wrote stack_rank.csv")
else:
    print("Skipping stack_rank.csv: need mol0,mol1,mol2 columns")



## === cell 22
pass



## === cell 23
paths = {
    "sub1": "../input/another-one/stackers_blend.csv",
    "sub2": "../input/lgb-public-kernels-plus-more-features/sub_lgb_model_individual.csv",
    "sub3": "../input/yet-another-one/stackers_blend.csv",
    "sub4": "../input/giba-r-data-table-simple-features-1-17-lb/submission-giba-1.csv",
}
loaded = {}
for k, p in paths.items():
    if os.path.exists(p):
        loaded[k] = pd.read_csv(p)
        print("Loaded", k, "from", p, "shape=", loaded[k].shape)
    else:
        print("Missing", k, "path:", p)

temp = None
if "sub1" in loaded:
    temp = loaded["sub1"].copy()



## === cell 24
if temp is not None and "sub3" in loaded and "sub4" in loaded:
    sub3 = loaded["sub3"]
    sub4 = loaded["sub4"]
    temp["scalar_coupling_constant"] = (
        0.6 * sub3["scalar_coupling_constant"] + 0.4 * sub4["scalar_coupling_constant"]
    )
    temp.to_csv("submission4.csv", index=False)
    print("Wrote submission4.csv")
else:
    print("Skipping submission4 blend: required files not available")



## === cell 25
if temp is not None and "scalar_coupling_constant" in temp.columns:
    ax = sns.histplot(temp["scalar_coupling_constant"], bins=50, kde=True)
    fig = ax.get_figure()
    fig.savefig("temp_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved temp_distribution.png")
else:
    print("Skipping distribution plot: temp not available")
