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

-1.680697846605887

# 6. Current score

1.23566

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'Your notebook fails because it references external “blending tutorial” and “otherkernelsadded” input folders that do not exist in this Kaggle environment, so no submission is produced. I replace that blending-only code with a minimal, self-contained baseline that uses only the provided CHAMPS dataset files and writes a valid `submission.csv`. To keep core logic simple and stable (and avoid heavy training within the 600s limit), the model predict the per-`type` mean coupling constant from the training data (a common safe baseline), and fall back to the global mean for any unseen types. This run end-to-end and generate a correctly formatted CSV for submission.'
- What this solution (achieved 1.23566) has done: 'Your current baseline (per-`type` mean) is far from the target, so we need a small but legitimate feature upgrade while keeping the overall approach simple and fast. I keep it as a lightweight grouped-statistics regressor, but make it more specific by conditioning on `(type, atom_0 element, atom_1 element)` using `structures.csv` to look up the elements for each atom index. This typically reduces MAE substantially versus `type`-only means, yet remains the same “predict group mean” core logic (no model architecture/training loop changes). I also add safe backoff levels (to `type` mean, then global mean) to guarantee a complete submission.'
- What this solution (achieved 1.23566) has done: 'Your current grouped-mean baseline is fast and stable but too coarse, so we keep the same “predict a group mean with backoffs” core logic and just make the grouping slightly richer using geometry you already have in `structures.csv`. Specifically, we add the inter-atomic distance feature (computed from x/y/z) and predict the mean within `(type, atom_0, atom_1, distance_bin)`; this usually lowers MAE meaningfully while staying within the same no-training approach. We also keep strict fallback levels (to `(type, atom_0, atom_1)`, then `type`, then global mean) to ensure a complete submission. This should move the score downward (better, since lower is better) toward your target without changing the overall approach or runtime drastically.'
- What this solution (achieved 1.23566) has done: 'Your current grouped-mean-with-backoffs baseline is still far from the target, so we keep the exact same core “predict a mean for a group” logic but make the grouping slightly more informative using only data you already load. Specifically, we (1) canonicalize the atom pair ordering within each row (so symmetric pairs share statistics), and (2) add a very cheap, chemistry-relevant context signal: each atom’s degree (number of atoms bonded within a distance cutoff) computed from `structures.csv`, then include `(deg0, deg1)` in the finest-grain group with backoffs unchanged. This remains a pure lookup/statistics approach (no training loop/model change), should lower MAE materially, and stays within time by computing degrees once per molecule with a small per-molecule distance-matrix routine.'
- What this solution (achieved 1.23566) has done: 'Your current grouped-mean lookup is still far above (worse than) the target, so we should make the finest-grain grouping a bit more informative while keeping the exact same “predict a group mean with backoffs” core logic. The smallest high-impact addition is to include more molecule context that correlates with coupling: (1) per-atom Mulliken charge and (2) per-atom magnetic shielding tensor trace (XX+YY+ZZ), then discretize them into small quantile bins computed on train and use those bins only in the *finest* group. This keeps runtime low (just a couple merges + binning) and preserves evaluation semantics (still mean-by-group with deterministic fallbacks), while typically reducing MAE meaningfully versus geometry-only bins. All existing backoffs remain unchanged to guarantee a complete submission.'
- What this solution (achieved 1.23566) has done: 'Your current approach is a pure grouped-mean lookup with fallbacks; to move the score down (better) toward the target while preserving that exact core logic, I only make the finest-grain statistics less sparse and better aligned with the evaluation (per-type MAE). Concretely: (1) use count-aware shrinkage (additive smoothing) for each grouping level to reduce noisy means when groups are small, and (2) use per-`type` backoff means (instead of global mean) as the prior so the model respects type-specific scale. I also fix a subtle issue where `fillna(np.nan)` does nothing (so missingness handling is inconsistent), by leaving NaNs as-is and binning them to `-1` as intended. These are minimal, deterministic changes that keep the same “predict group mean with backoffs” semantics but usually improve MAE significantly.'
- What this solution (achieved 1.23566) has done: 'Your current score (1.23566, lower-is-better) is still far above the target (-1.6807), so we should improve the predictions while keeping the same “grouped mean with deterministic backoffs” core logic. The smallest high-impact fix is to stop mapping test keys via a plain `Series(key).map(...)` (which relies on Python tuple hashing and can be slow/memory-heavy) and instead compute predictions by merging pre-aggregated group-mean tables back onto `test`; this keeps identical semantics but avoids accidental key dtype issues and typically yields slightly better stability. Then, to move the score down further without changing the approach, we add one more cheap, competition-relevant context signal to the finest grouping: molecule-level dipole magnitude (binned), merged from `dipole_moments.csv`; this preserves the “lookup mean by group with fallbacks” design and is fast (single merge + binning). All existing backoffs and smoothing remain, so the pipeline stays robust and always produces a valid `submission.csv`.'
- What this solution (achieved 1.23566) has done: 'Your current score is far worse than the target (lower-is-better), so we should legitimately improve accuracy while keeping the same “grouped mean with smoothed backoffs” core logic. The biggest issue is that canonicalizing atom pairs by atom symbol can mix direction-sensitive coupling types (the CHAMPS `type` encodes ordered atom roles), which hurts the per-type MAE. I make canonicalization conditional: only canonicalize for same-element pairs (safe symmetry), and otherwise preserve the original `(atom_index_0, atom_index_1)` direction. This is a minimal change that typically reduces error materially without changing the overall approach, runtime, or submission format.'
- What this solution (achieved 1.23566) has done: 'Your current score is far worse than the target (lower-is-better), so we should improve accuracy while keeping the exact same “smoothed grouped-mean lookup with backoffs” core logic. The most likely remaining high-impact issue is that the finest groups are still too sparse and can overfit noise; we can reduce MAE by adding one more deterministic *backoff level* that uses a simpler, well-aligned continuous signal: distance-only binning per `type` (and optionally per `(type, atom_0, atom_1)`), without changing the overall approach. Concretely, we add two extra smoothed tables: `(type, dist_bin)` and `(type, atom_0, atom_1, dist_bin)` with moderate prior weights, and insert them into the existing backoff chain between `pred4` and `pred3`. This preserves evaluation semantics, keeps runtime within limits (just extra groupbys/merges), and should move the score downward toward the target.'
- What this solution (achieved 1.23566) has done: 'Your current score (1.23566, lower-is-better) is still far above the target (-1.6807), so we need a legitimate accuracy lift while keeping your exact “smoothed grouped-mean lookup with backoffs” core logic intact. The smallest high-impact fix is to stop using global quantile distance bins, because coupling–distance relationships differ strongly by `type`; instead we compute `dist_bin` per-`type` using train quantiles and apply the same edges to test, then keep the rest of your tables/merges/backoffs unchanged. This typically improves MAE without changing the modeling approach (still deterministic group means + smoothing) and stays within the time limit by using vectorized `groupby().quantile()` and a fast `searchsorted` bin assignment. All I/O paths and submission writing remain the same.'
- What this solution (achieved 1.23566) has done: 'The crash comes from trying to merge on `dist_bin` even though `pred_df` doesn’t include that column (it only includes the `cols11` features). I fix this by building `pred_df` from `test` with **all** columns needed for every subsequent merge (including `dist_bin` and `dist_bin_ta`), which is score-neutral and preserves your exact prediction/backoff logic. I also add a couple of assertions to catch missing feature columns earlier, instead of failing deep inside a merge. The rest of the approach (feature creation, smoothed group tables, backoff chain, and submission writing) is unchanged.'
- What this solution (achieved 1.23566) has done: 'Your current score is far worse than the target (lower-is-better), so we should improve accuracy while keeping the exact same “smoothed grouped-mean lookup with backoffs” core logic intact. The highest-impact minimal fix is to align smoothing priors with the competition metric by using a *more specific prior* for groups: `(type, atom_0, atom_1)` prior for the finer tables, instead of always using `type` prior; this reduces bias and typically lowers MAE without changing the approach. In the same spirit, we add one additional, very cheap backoff table `(type, dist_bin_ta)` (distance behavior is strongly type-specific) inserted between existing levels; this is still just another smoothed mean table + merge. All paths, feature engineering, and prediction/fallback semantics remain deterministic and the script still writes a valid `submission.csv`.'
- What this solution (achieved 1.23566) has done: 'We’re far from the target (lower-is-better), so we need a legitimate accuracy lift while keeping your exact “smoothed grouped-mean lookup with backoffs” core logic unchanged. The smallest high-impact improvement that fits your current framework is to compute degrees using a chemically more appropriate bond cutoff **per atom-pair type** (H vs heavy atoms) instead of a single 1.8Å cutoff, because your finest tables depend on `(deg0, deg1)` and current degrees are noisy. I replace `compute_degrees()` with a vectorized, per-molecule routine that uses covalent-radius-based thresholds (still deterministic, no learning), and keep everything else (features, bins, smoothing, backoff chain, output) the same. This should reduce MAE (and thus log-MAE) without changing evaluation semantics or runtime materially.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

DATA_DIR = "/kaggle/data/champs-scalar-coupling"

print("Listing data dir:", DATA_DIR)
print(sorted(os.listdir(DATA_DIR))[:20])

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
structures_path = os.path.join(DATA_DIR, "structures.csv")

mulliken_path = os.path.join(DATA_DIR, "mulliken_charges.csv")
shield_path = os.path.join(DATA_DIR, "magnetic_shielding_tensors.csv")
dipole_path = os.path.join(DATA_DIR, "dipole_moments.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample = pd.read_csv(sample_path)
structures = pd.read_csv(structures_path)

required_train_cols = {
    "molecule_name",
    "atom_index_0",
    "atom_index_1",
    "type",
    "scalar_coupling_constant",
}
required_test_cols = {"id", "molecule_name", "atom_index_0", "atom_index_1", "type"}
assert required_train_cols.issubset(
    train.columns
), f"train.csv missing columns: {required_train_cols - set(train.columns)}"
assert required_test_cols.issubset(
    test.columns
), f"test.csv missing columns: {required_test_cols - set(test.columns)}"
assert {"id", "scalar_coupling_constant"}.issubset(
    sample.columns
), "sample_submission.csv has unexpected columns"

structures = structures[["molecule_name", "atom_index", "atom", "x", "y", "z"]].copy()
structures["atom_index"] = structures["atom_index"].astype(np.int32, copy=False)
for c in ["x", "y", "z"]:
    structures[c] = structures[c].astype(np.float32, copy=False)

mulliken = pd.read_csv(
    mulliken_path, usecols=["molecule_name", "atom_index", "mulliken_charge"]
)
mulliken["atom_index"] = mulliken["atom_index"].astype(np.int32, copy=False)
mulliken["mulliken_charge"] = mulliken["mulliken_charge"].astype(np.float32, copy=False)

shield = pd.read_csv(
    shield_path,
    usecols=["molecule_name", "atom_index", "XX", "YY", "ZZ"],
)
shield["atom_index"] = shield["atom_index"].astype(np.int32, copy=False)
for c in ["XX", "YY", "ZZ"]:
    shield[c] = shield[c].astype(np.float32, copy=False)
shield["shield_trace"] = (shield["XX"] + shield["YY"] + shield["ZZ"]).astype(
    np.float32, copy=False
)
shield = shield[["molecule_name", "atom_index", "shield_trace"]]

dipole = pd.read_csv(dipole_path, usecols=["molecule_name", "X", "Y", "Z"])
for c in ["X", "Y", "Z"]:
    dipole[c] = dipole[c].astype(np.float32, copy=False)
dipole["dipole_mag"] = np.sqrt(
    dipole["X"] * dipole["X"] + dipole["Y"] * dipole["Y"] + dipole["Z"] * dipole["Z"]
).astype(np.float32, copy=False)
dipole = dipole[["molecule_name", "dipole_mag"]]


def compute_degrees(struct_df: pd.DataFrame, fudge: float = 0.45) -> pd.DataFrame:
    rad = {
        "H": 0.31,
        "C": 0.76,
        "N": 0.71,
        "O": 0.66,
        "F": 0.57,
        "P": 1.07,
        "S": 1.05,
        "Cl": 1.02,
        "Br": 1.20,
        "I": 1.39,
    }

    out = np.empty(len(struct_df), dtype=np.int16)

    r = struct_df["atom"].map(rad).fillna(0.85).astype(np.float32).to_numpy(copy=False)

    for mol, g in struct_df.groupby("molecule_name", sort=False):
        idx = g.index.values
        xyz = g[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=False)
        rr = r[idx]
        n = xyz.shape[0]
        if n <= 1:
            out[idx] = 0
            continue

        diff = xyz[:, None, :] - xyz[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff).astype(np.float32, copy=False)

        thr = (rr[:, None] + rr[None, :] + np.float32(fudge)).astype(
            np.float32, copy=False
        )
        thr2 = thr * thr

        neigh = (d2 <= thr2) & (d2 > 0.0)
        out[idx] = neigh.sum(axis=1).astype(np.int16, copy=False)

    deg_df = struct_df[["molecule_name", "atom_index"]].copy()
    deg_df["degree"] = out
    return deg_df


deg = compute_degrees(structures, fudge=0.45)

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

deg0 = deg.rename(columns={"atom_index": "atom_index_0", "degree": "deg0"})
deg1 = deg.rename(columns={"atom_index": "atom_index_1", "degree": "deg1"})

m0 = mulliken.rename(columns={"atom_index": "atom_index_0", "mulliken_charge": "q0"})
m1 = mulliken.rename(columns={"atom_index": "atom_index_1", "mulliken_charge": "q1"})
sh0 = shield.rename(columns={"atom_index": "atom_index_0", "shield_trace": "sh0"})
sh1 = shield.rename(columns={"atom_index": "atom_index_1", "shield_trace": "sh1"})

train = train.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
train = train.merge(s1, on=["molecule_name", "atom_index_1"], how="left")
test = test.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
test = test.merge(s1, on=["molecule_name", "atom_index_1"], how="left")

train = train.merge(deg0, on=["molecule_name", "atom_index_0"], how="left")
train = train.merge(deg1, on=["molecule_name", "atom_index_1"], how="left")
test = test.merge(deg0, on=["molecule_name", "atom_index_0"], how="left")
test = test.merge(deg1, on=["molecule_name", "atom_index_1"], how="left")

train = train.merge(m0, on=["molecule_name", "atom_index_0"], how="left")
train = train.merge(m1, on=["molecule_name", "atom_index_1"], how="left")
test = test.merge(m0, on=["molecule_name", "atom_index_0"], how="left")
test = test.merge(m1, on=["molecule_name", "atom_index_1"], how="left")

train = train.merge(sh0, on=["molecule_name", "atom_index_0"], how="left")
train = train.merge(sh1, on=["molecule_name", "atom_index_1"], how="left")
test = test.merge(sh0, on=["molecule_name", "atom_index_0"], how="left")
test = test.merge(sh1, on=["molecule_name", "atom_index_1"], how="left")

train = train.merge(dipole, on="molecule_name", how="left")
test = test.merge(dipole, on="molecule_name", how="left")

train["atom_0"] = train["atom_0"].fillna("UNK")
train["atom_1"] = train["atom_1"].fillna("UNK")
test["atom_0"] = test["atom_0"].fillna("UNK")
test["atom_1"] = test["atom_1"].fillna("UNK")

train["deg0"] = train["deg0"].fillna(-1).astype(np.int16)
train["deg1"] = train["deg1"].fillna(-1).astype(np.int16)
test["deg0"] = test["deg0"].fillna(-1).astype(np.int16)
test["deg1"] = test["deg1"].fillna(-1).astype(np.int16)

for c in ["q0", "q1", "sh0", "sh1", "dipole_mag"]:
    train[c] = train[c].astype(np.float32, copy=False)
    test[c] = test[c].astype(np.float32, copy=False)


def add_distance(df: pd.DataFrame) -> pd.DataFrame:
    dx = (df["x0"] - df["x1"]).astype("float32")
    dy = (df["y0"] - df["y1"]).astype("float32")
    dz = (df["z0"] - df["z1"]).astype("float32")
    d = np.sqrt(dx * dx + dy * dy + dz * dz).astype("float32")
    df["dist"] = d
    return df


train = add_distance(train)
test = add_distance(test)


def canonicalize_pairs(df: pd.DataFrame) -> pd.DataFrame:
    a0 = df["atom_0"].astype(str)
    a1 = df["atom_1"].astype(str)

    same = a0.eq(a1)
    swap = same & (df["atom_index_0"].to_numpy() > df["atom_index_1"].to_numpy())

    if swap.any():
        df.loc[swap, ["atom_0", "atom_1"]] = df.loc[
            swap, ["atom_1", "atom_0"]
        ].to_numpy()
        df.loc[swap, ["x0", "y0", "z0", "x1", "y1", "z1"]] = df.loc[
            swap, ["x1", "y1", "z1", "x0", "y0", "z0"]
        ].to_numpy()
        df.loc[swap, ["deg0", "deg1"]] = df.loc[swap, ["deg1", "deg0"]].to_numpy()
        df.loc[swap, ["atom_index_0", "atom_index_1"]] = df.loc[
            swap, ["atom_index_1", "atom_index_0"]
        ].to_numpy()
        df.loc[swap, ["q0", "q1"]] = df.loc[swap, ["q1", "q0"]].to_numpy()
        df.loc[swap, ["sh0", "sh1"]] = df.loc[swap, ["sh1", "sh0"]].to_numpy()
    return df


train = canonicalize_pairs(train)
test = canonicalize_pairs(test)


def add_groupwise_dist_bins(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_cols,
    dist_col: str = "dist",
    n_bins: int = 30,
    outcol: str = "dist_bin",
) -> None:
    qs = np.linspace(0.0, 1.0, n_bins + 1)

    qt = (
        train_df[group_cols + [dist_col]]
        .dropna(subset=[dist_col])
        .groupby(group_cols, sort=False)[dist_col]
        .quantile(qs)
        .unstack(-1)
        .astype(np.float32)
    )

    def assign_bins(df: pd.DataFrame) -> np.ndarray:
        out = np.full(len(df), -1, dtype=np.int16)
        for key, idx in df.groupby(group_cols, sort=False).groups.items():
            if key not in qt.index:
                continue
            edges = qt.loc[key].to_numpy(copy=False)
            edges = np.unique(edges[~np.isnan(edges)])
            if edges.size < 2:
                out[idx] = 0
                continue
            v = df.loc[idx, dist_col].to_numpy(dtype=np.float32, copy=False)
            b = np.searchsorted(edges, v, side="right") - 1
            b = np.clip(b, 0, edges.size - 2).astype(np.int16, copy=False)
            b[np.isnan(v)] = -1
            out[idx] = b
        return out

    train_df[outcol] = assign_bins(train_df)
    test_df[outcol] = assign_bins(test_df)


add_groupwise_dist_bins(train, test, group_cols=["type"], n_bins=40, outcol="dist_bin")

add_groupwise_dist_bins(
    train,
    test,
    group_cols=["type", "atom_0", "atom_1"],
    n_bins=25,
    outcol="dist_bin_ta",
)


def quantile_bins(train_series: pd.Series, n_bins: int):
    v = train_series.dropna().values
    if v.size == 0:
        return np.array([0.0, 1.0], dtype=np.float32)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(v, qs))
    if edges.size < 2:
        edges = np.array(
            [float(np.min(v)) - 1e-6, float(np.max(v)) + 1e-6], dtype=np.float32
        )
    return edges.astype(np.float32, copy=False)


def apply_bins(df: pd.DataFrame, col: str, edges: np.ndarray, outcol: str) -> None:
    b = pd.cut(df[col], bins=edges, labels=False, include_lowest=True)
    df[outcol] = b.fillna(-1).astype(np.int16)


q_edges = quantile_bins(
    pd.concat([train["q0"], train["q1"]], ignore_index=True), n_bins=16
)
sh_edges = quantile_bins(
    pd.concat([train["sh0"], train["sh1"]], ignore_index=True), n_bins=16
)
dip_edges = quantile_bins(train["dipole_mag"], n_bins=16)

apply_bins(train, "q0", q_edges, "q0_bin")
apply_bins(train, "q1", q_edges, "q1_bin")
apply_bins(test, "q0", q_edges, "q0_bin")
apply_bins(test, "q1", q_edges, "q1_bin")

apply_bins(train, "sh0", sh_edges, "sh0_bin")
apply_bins(train, "sh1", sh_edges, "sh1_bin")
apply_bins(test, "sh0", sh_edges, "sh0_bin")
apply_bins(test, "sh1", sh_edges, "sh1_bin")

apply_bins(train, "dipole_mag", dip_edges, "dip_bin")
apply_bins(test, "dipole_mag", dip_edges, "dip_bin")

global_mean = float(train["scalar_coupling_constant"].mean())
type_mean = train.groupby("type")["scalar_coupling_constant"].mean()
type_prior = type_mean

W10 = 50.0
W6 = 30.0
W4 = 20.0
W3 = 10.0




## === cell 1
def smoothed_group_table(
    df: pd.DataFrame,
    group_cols,
    target_col: str,
    prior_mean: pd.Series,
    prior_weight: float,
    outcol: str,
) -> pd.DataFrame:
    gb = df.groupby(group_cols, sort=False)[target_col]
    m = gb.mean()
    c = gb.size().astype(np.int64)

    t = m.index.get_level_values(0)
    p = prior_mean.reindex(t).to_numpy(dtype=np.float64, copy=False)

    m_np = m.to_numpy(dtype=np.float64, copy=False)
    c_np = c.to_numpy(dtype=np.float64, copy=False)
    sm = (m_np * c_np + p * prior_weight) / (c_np + prior_weight)

    tab = m.index.to_frame(index=False)
    tab[outcol] = sm.astype(np.float32, copy=False)
    return tab


cols3 = ["type", "atom_0", "atom_1"]

cols4 = ["type", "atom_0", "atom_1", "dist_bin_ta"]
cols6 = ["type", "atom_0", "atom_1", "dist_bin_ta", "deg0", "deg1"]
cols11 = [
    "type",
    "atom_0",
    "atom_1",
    "dist_bin_ta",
    "deg0",
    "deg1",
    "q0_bin",
    "q1_bin",
    "sh0_bin",
    "sh1_bin",
    "dip_bin",
]

cols2_type_dist = ["type", "dist_bin"]
cols4_type_atoms_dist = ["type", "atom_0", "atom_1", "dist_bin_ta"]

ta_mean = train.groupby(cols3, sort=False)["scalar_coupling_constant"].mean()

W2 = 15.0
W4b = 18.0
W_td_ta = 18.0  # moderate smoothing for (type, dist_bin_ta) table

tab3 = smoothed_group_table(
    train, cols3, "scalar_coupling_constant", type_prior, W3, "pred3"
)

tab4 = smoothed_group_table(
    train, cols4, "scalar_coupling_constant", ta_mean, W4, "pred4"
)
tab6 = smoothed_group_table(
    train, cols6, "scalar_coupling_constant", ta_mean, W6, "pred6"
)
tab11 = smoothed_group_table(
    train, cols11, "scalar_coupling_constant", ta_mean, W10, "pred11"
)

tab2_td = smoothed_group_table(
    train, cols2_type_dist, "scalar_coupling_constant", type_prior, W2, "pred2_td"
)
tab4_ad = smoothed_group_table(
    train,
    cols4_type_atoms_dist,
    "scalar_coupling_constant",
    ta_mean,
    W4b,
    "pred4_ad",
)

cols2_type_dist_ta = ["type", "dist_bin_ta"]
tab2_tdt = smoothed_group_table(
    train,
    cols2_type_dist_ta,
    "scalar_coupling_constant",
    type_prior,
    W_td_ta,
    "pred2_tdt",
)

needed_pred_cols = sorted(set(["id"] + cols11 + cols2_type_dist + cols2_type_dist_ta))
missing = [c for c in needed_pred_cols if c not in test.columns]
assert not missing, f"Missing required prediction feature columns in test: {missing}"

pred_df = test[needed_pred_cols].copy()

pred_df = pred_df.merge(tab11, on=cols11, how="left")
pred_df = pred_df.merge(tab6, on=cols6, how="left")
pred_df = pred_df.merge(tab4, on=cols4, how="left")
pred_df = pred_df.merge(tab4_ad, on=cols4_type_atoms_dist, how="left")
pred_df = pred_df.merge(tab2_tdt, on=cols2_type_dist_ta, how="left")
pred_df = pred_df.merge(tab2_td, on=cols2_type_dist, how="left")
pred_df = pred_df.merge(tab3, on=cols3, how="left")

pred = pred_df["pred11"]
pred = pred.fillna(pred_df["pred6"])
pred = pred.fillna(pred_df["pred4"])
pred = pred.fillna(pred_df["pred4_ad"])
pred = pred.fillna(pred_df["pred2_tdt"])
pred = pred.fillna(pred_df["pred2_td"])
pred = pred.fillna(pred_df["pred3"])
pred = pred.fillna(test["type"].map(type_mean).astype("float64"))
pred = pred.fillna(global_mean)

submission = pd.DataFrame(
    {
        "id": test["id"].astype(sample["id"].dtype, copy=False),
        "scalar_coupling_constant": pred.to_numpy(dtype=np.float64, copy=False),
    }
)

submission = sample[["id"]].merge(submission, on="id", how="left")
if submission["scalar_coupling_constant"].isna().any():
    submission["scalar_coupling_constant"] = submission[
        "scalar_coupling_constant"
    ].fillna(global_mean)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print("Rows:", len(submission), "Expected:", len(sample))
assert len(submission) == len(sample), "Submission row count mismatch"
assert list(submission.columns) == [
    "id",
    "scalar_coupling_constant",
], "Submission columns mismatch"
assert (
    not submission["scalar_coupling_constant"].isna().any()
), "Submission contains NaN predictions"
