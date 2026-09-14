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

-1.664883110448491

# 6. Current score

1.23596

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.18497) has done: 'Your notebook fails because it tries to blend several external submissions that are not present in this environment (`../input/champs-blending-tutorial/...` and `../input/otherkernelsadded/...`). The minimal fix is to remove that dependency and instead generate a valid baseline submission from the provided competition data files. To keep this robust and fast, the code create a per-`type` median from `train.csv` and use it to predict `test.csv`, falling back to the global median for any unseen types. This runs end-to-end and writes a correctly formatted `.csv` submission file.'
- What this solution (achieved 1.23596) has done: 'Your current approach predicts a per-`type` constant (the median), which is robust but leaves a large gap to the target. To move the score closer to the target while keeping the same “type-based constant prediction” core logic, I switch from median to mean, because MAE is minimized by the median but the competition metric is log(MAE) averaged across types, and the mean often reduces large systematic offsets for some coupling types in this dataset. I also compute the statistic on the target-transformed space per type using a small shrinkage toward the global value to stabilize rare types; this is still the same constant-per-type baseline, just slightly better calibrated. The code still run fast, read only the needed columns, and write a valid `submission.csv`.'
- What this solution (achieved 1.23596) has done: 'Your current baseline is a constant-per-`type` predictor; to move the (lower-is-better) score much closer to the target with minimal semantic change, we should still predict a per-`type` constant but make it a better constant for this metric. The smallest strong upgrade is to use the provided `scalar_coupling_contributions.csv` and predict `scalar_coupling_constant ≈ fc+sd+pso+dso` per row, falling back to your shrunk per-`type` mean only if a row can’t be matched. This keeps the “no model training, deterministic mapping to a scalar” core approach, but uses more informative provided features that directly sum to the target. The code below reads only needed columns, performs a keyed merge, validates row counts, and writes a valid `submission.csv`.'
- What this solution (achieved 1.23596) has done: 'Your current solution already uses the strongest “no-training” signal available (the contributions sum), so the main remaining score drag is likely join mismatches (atom order) and duplicates in `scalar_coupling_contributions.csv`. I make the merge robust to swapped atom indices by creating an order-invariant key (min/max of the two atom indices) for both test and contributions, without changing the prediction logic (still `fc+sd+pso+dso`, with the same per-type shrunk-mean fallback). I also de-duplicate the contributions on the merge key to avoid 1:1 validation failures or silent mismatches, and I report the new miss rate so you can verify the improvement. These are minimal, metric-relevant fixes that should reduce MAE (thus lower the log-MAE score) toward your target.'
- What this solution (achieved 1.23596) has done: 'Your current code already uses the strongest available “no-training” signal (fc+sd+pso+dso), so to move the score down toward the target the most likely remaining issue is residual join mismatch caused by duplicate contribution rows and/or atom-index order inconsistencies. I make the merge fully order-invariant by constructing a canonical key in both `test` and `scalar_coupling_contributions` and then aggregating duplicates on that key (summing the four terms) instead of keeping an arbitrary first row. This preserves your core logic (predict contributions sum, fallback to shrunk per-type mean) while reducing wrong matches that inflate MAE. I also keep the output format/paths identical and add a couple of sanity prints to confirm the miss rate and duplicate handling.'
- What this solution (achieved 1.23596) has done: 'Your current score is far from the target (lower-is-better; 1.23596 vs -1.66488), and the main reason is that `scalar_coupling_contributions.csv` only provides target decompositions for the training set—so your merge on the test set mostly misses and you fall back to a weak per-type constant. The smallest legitimate change that materially moves the score toward the target is to keep your “per-type constant predictor” core logic, but compute that constant from richer per-row geometric features (distance between the atom pair) using `structures.csv`, then aggregate to a per-`type`×binned-distance mean and use it for test predictions. This is still a deterministic lookup (no model training/loops), but it dramatically improves calibration vs. type-only constants while staying fast and robust. I keep your shrinkage fallback as a final safety net and still write a valid `submission.csv`.'
- What this solution (achieved 1.23596) has done: 'Your current approach is a deterministic per-`type`×distance-bin lookup with shrinkage; to move the score down (better) toward the target without changing that core logic, the most leverage is to make the distance feature consistent with the physics by using an order-invariant atom pair (min/max) for the distance computation and for binning. This reduces noise from swapped `atom_index_0/1` ordering and improves the stability of the (type, bin) statistics while keeping the same “lookup table” prediction semantics. I also make the binning robust by extending the first/last bin edges slightly so test distances don’t fall outside the cut range (which otherwise increases fallback-to-type-mean rate). These are minimal, metric-relevant changes that should reduce MAE and therefore lower the log-MAE score toward your target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

DATA_DIR = "/kaggle/data/champs-scalar-coupling"

print("DATA_DIR exists:", os.path.isdir(DATA_DIR))
print("Files (head):", sorted(os.listdir(DATA_DIR))[:10])



## === cell 1
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
structures_path = os.path.join(DATA_DIR, "structures.csv")
contrib_path = os.path.join(
    DATA_DIR, "scalar_coupling_contributions.csv"
)  # kept for compatibility/prints

train = pd.read_csv(
    train_path,
    usecols=[
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
sample = pd.read_csv(sample_path, usecols=["id"])

structures = pd.read_csv(
    structures_path,
    usecols=["molecule_name", "atom_index", "x", "y", "z"],
)

for df in (train, test):
    df["atom_index_0"] = df["atom_index_0"].astype("int32", copy=False)
    df["atom_index_1"] = df["atom_index_1"].astype("int32", copy=False)

structures["atom_index"] = structures["atom_index"].astype("int32", copy=False)
for c in ["x", "y", "z"]:
    structures[c] = structures[c].astype("float32", copy=False)


def add_canonical_pair(df):
    a0 = df["atom_index_0"]
    a1 = df["atom_index_1"]
    df["a_min"] = a0.where(a0 <= a1, a1).astype("int32", copy=False)
    df["a_max"] = a1.where(a0 <= a1, a0).astype("int32", copy=False)
    return df


train = add_canonical_pair(train)
test = add_canonical_pair(test)

s_min = structures.rename(
    columns={"atom_index": "a_min", "x": "x0", "y": "y0", "z": "z0"}
)
s_max = structures.rename(
    columns={"atom_index": "a_max", "x": "x1", "y": "y1", "z": "z1"}
)

train_xy = train.merge(
    s_min, on=["molecule_name", "a_min"], how="left", sort=False, validate="m:1"
)
train_xy = train_xy.merge(
    s_max, on=["molecule_name", "a_max"], how="left", sort=False, validate="m:1"
)

test_xy = test.merge(
    s_min, on=["molecule_name", "a_min"], how="left", sort=False, validate="m:1"
)
test_xy = test_xy.merge(
    s_max, on=["molecule_name", "a_max"], how="left", sort=False, validate="m:1"
)


def add_distance(df):
    dx = (df["x0"] - df["x1"]).astype("float32")
    dy = (df["y0"] - df["y1"]).astype("float32")
    dz = (df["z0"] - df["z1"]).astype("float32")
    df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype("float32")
    return df


train_xy = add_distance(train_xy)
test_xy = add_distance(test_xy)

train_dist_missing = float(train_xy["dist"].isna().mean())
test_dist_missing = float(test_xy["dist"].isna().mean())
print(
    "Distance missing rate | train:", train_dist_missing, "| test:", test_dist_missing
)

n_bins = 120
train_dist_for_bins = train_xy.loc[train_xy["dist"].notna(), "dist"].astype("float64")

bin_edges = np.quantile(train_dist_for_bins.values, np.linspace(0.0, 1.0, n_bins + 1))
bin_edges = np.unique(bin_edges)

if len(bin_edges) < 10:
    raise RuntimeError(
        "Distance bin edges degenerated; check structures merge/dist computation."
    )

bin_edges = bin_edges.copy()
bin_edges[0] = bin_edges[0] - 1e-6
bin_edges[-1] = bin_edges[-1] + 1e-6

train_xy["dist_bin"] = pd.cut(
    train_xy["dist"].astype("float64"),
    bins=bin_edges,
    include_lowest=True,
    labels=False,
)
test_xy["dist_bin"] = pd.cut(
    test_xy["dist"].astype("float64"),
    bins=bin_edges,
    include_lowest=True,
    labels=False,
)

type_stats = train.groupby("type")["scalar_coupling_constant"].agg(["mean", "count"])
global_mean = float(train["scalar_coupling_constant"].mean())
alpha = 50.0  # keep same regularization strength for stability
type_shrunk_mean = (type_stats["mean"] * type_stats["count"] + global_mean * alpha) / (
    type_stats["count"] + alpha
)

bin_stats = (
    train_xy.groupby(["type", "dist_bin"])["scalar_coupling_constant"]
    .agg(["mean", "count"])
    .reset_index()
)

bin_alpha = 30.0  # mild shrinkage; deterministic, stabilizes rare (type,bin) combos
bin_stats["type_mean_shrunk"] = (
    bin_stats["type"].map(type_shrunk_mean).astype("float64")
)
bin_stats["mean_shrunk"] = (
    bin_stats["mean"] * bin_stats["count"] + bin_stats["type_mean_shrunk"] * bin_alpha
) / (bin_stats["count"] + bin_alpha)

lookup = bin_stats.set_index(["type", "dist_bin"])["mean_shrunk"]

test_key = pd.MultiIndex.from_frame(test_xy[["type", "dist_bin"]])
pred = lookup.reindex(test_key).to_numpy(dtype="float64")
pred = pd.Series(pred, index=test_xy.index, dtype="float64")
pred = pred.fillna(test_xy["type"].map(type_shrunk_mean).astype("float64")).fillna(
    global_mean
)

submission = pd.DataFrame(
    {
        "id": test_xy["id"].astype(sample["id"].dtype, copy=False),
        "scalar_coupling_constant": pred.values,
    }
)

assert submission.shape[0] == test.shape[0], "Submission row count mismatch."
assert list(submission.columns) == [
    "id",
    "scalar_coupling_constant",
], "Wrong submission columns."

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(submission))

miss_rate = float(pd.isna(lookup.reindex(test_key)).mean())
print("Miss rate for (type, dist_bin) lookup:", miss_rate)

if os.path.exists(contrib_path):
    contrib = pd.read_csv(
        contrib_path,
        usecols=[
            "molecule_name",
            "atom_index_0",
            "atom_index_1",
            "type",
            "fc",
            "sd",
            "pso",
            "dso",
        ],
    )
    for df in (test, contrib):
        df["atom_index_0"] = df["atom_index_0"].astype("int32", copy=False)
        df["atom_index_1"] = df["atom_index_1"].astype("int32", copy=False)
        a0 = df["atom_index_0"]
        a1 = df["atom_index_1"]
        df["a_min"] = a0.where(a0 <= a1, a1).astype("int32", copy=False)
        df["a_max"] = a1.where(a0 <= a1, a0).astype("int32", copy=False)

    dup_key = ["molecule_name", "a_min", "a_max", "type"]
    contrib = contrib.groupby(dup_key, as_index=False)[["fc", "sd", "pso", "dso"]].sum()

    merged = test.merge(
        contrib, on=["molecule_name", "a_min", "a_max", "type"], how="left", sort=False
    )
    contrib_missing_rate = float(merged["fc"].isna().mean())
    print(
        "Informational: contribution merge missing rate on test (expected ~1.0):",
        contrib_missing_rate,
    )

print(submission.head())
