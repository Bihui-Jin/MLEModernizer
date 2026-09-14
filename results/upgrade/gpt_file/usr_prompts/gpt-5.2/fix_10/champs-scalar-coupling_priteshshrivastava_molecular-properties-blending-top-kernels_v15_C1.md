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

-1.679859728490842

# 6. Current score

1.23566

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'The crash happens because the notebook tries to read out-of-environment blend files from `../input/champs-blending-tutorial/` and `../input/otherkernelsadded/`, which don’t exist in your provided dataset. To make this run end-to-end and still produce a valid submission, I replace that broken blending step with a minimal, deterministic baseline that uses only the available competition files: predict the mean `scalar_coupling_constant` per coupling `type` from `train.csv`, and fall back to the global mean for any unseen types. This keeps the “core logic” of generating a submission from CSV inputs without adding heavy modeling code, and it yield a nontrivial score (better than constant-zero) while guaranteeing a correct `id,scalar_coupling_constant` CSV. The script also writes the submission with a `.csv` suffix to the working directory.'
- What this solution (achieved 1.21163) has done: 'Your current baseline predicts a per-`type` mean, which ignores large within-type variation driven by atom pair geometry and atom identities, so the score is far from the target (lower is better). With minimal change to the approach (still a deterministic aggregation from train and mapping onto test), I add a couple of very lightweight, competition-relevant grouping keys: `(type, atom_index_0)` and `(type, atom_index_1)`, and combine them with the existing `type` mean using smoothed backoff. This keeps the same “groupby → map → fillna” core logic, but captures some structure while remaining fast and safe. The output remains a valid `id,scalar_coupling_constant` CSV with identical paths and schema.'
- What this solution (achieved 1.23566) has done: 'I fix the crash by ensuring `dist` never produces NaN/inf before converting to the integer `dist_bin` (the current error is caused by missing structure merges leaving x/y/z as NaN). I do this with a minimal, score-neutral imputation: set missing coordinates to 0.0 and also guard against non-finite distances. I also add small sanity checks after merges to make failures explicit rather than silently propagating NaNs. The rest of the aggregation + shrinkage logic and submission schema stays unchanged, and it write a valid `.csv` submission.'
- What this solution (achieved 1.23566) has done: 'Your current score (1.23566, lower is better) is far from the target (-1.6799), so we should improve legitimately while keeping the same “groupby statistics + shrinkage + weighted blend” core logic. The biggest low-risk gain within that same approach is to add one more highly-informative aggregation key that you already have all ingredients for: `(type, atom_0, atom_1, dist_bin)` with shrinkage backoff to `(type, atom_0, atom_1)` and `(type, dist_bin)` and then to `type`. We keep your existing features/merges, keep the same prediction semantics (a convex-ish blend of shrunk means), and only adjust weights to incorporate this extra term. This should reduce MAE across coupling types because it captures geometry-dependent variation within each atom pair/type without introducing any new model class.'
- What this solution (achieved 1.23566) has done: 'Your current score (1.23566, lower is better) is still very far from the target (-1.67986), so we should legitimately improve while keeping your exact “groupby stats + shrinkage + weighted blend” approach. The smallest high-impact change within that same logic is to stop using raw `atom_index_0/1` aggregations (these indices are molecule-local and don’t generalize across molecules), and instead aggregate by chemistry/geometry keys you already have: `(type, atom_0)`, `(type, atom_1)`, plus a symmetric `(type, atom_min, atom_max, dist_bin)` term. I keep your existing type/atompair/dist_bin/ap_db statistics and shrinkage, add these two atom-type terms and symmetry, and then reweight slightly toward the more generalizable signals. This preserves evaluation semantics and still writes a valid `id,scalar_coupling_constant` submission CSV.'
- What this solution (achieved 1.23566) has done: 'Your current score is far worse than the target (lower is better), so the smallest legitimate improvement while preserving your exact “groupby stats + shrinkage + weighted blend” logic is to (1) fix a key mismatch that prevents your symmetric `(type, atom_min, atom_max, dist_bin)` stats from matching most test rows, and (2) make the `(atom_0, atom_1)` term symmetric as well so it generalizes regardless of index ordering. Concretely, we build `atom_min/atom_max` using the real atom symbols (not stringified), and we compute `m_ap_s` from a symmetric `(type, atom_min, atom_max)` group instead of `(type, atom_0, atom_1)`. Everything else (distance binning, shrinkage formulae, weights, paths, and submission schema) stays the same and still runs fast end-to-end.'
- What this solution (achieved 1.23566) has done: 'Your current score (1.23566, lower is better) is still very far from the target (-1.6799), so we should improve legitimately while keeping the same “groupby stats + shrinkage + weighted blend” core logic. The biggest safe gain without changing the approach is to add a molecule-level context term: include `potential_energy` (available for all molecules) and build an additional aggregated mean keyed by `(type, dist_bin, potential_energy_bin)` with shrinkage backoff to your existing `(type, dist_bin)` and `type` means. This preserves your existing features and prediction semantics (still a convex blend of shrunk group means) but captures systematic shifts across molecules that the current model can’t see. I also keep all paths the same and ensure the script still writes a valid `id,scalar_coupling_constant` CSV.'
- What this solution (achieved 1.23566) has done: 'Your current approach is already a deterministic “groupby stats + shrinkage + weighted blend” model, but the weights are fixed globally even though each coupling `type` has very different behavior and noise. To move the score down toward the target with minimal logic change, I keep all your existing feature merges, bins, group statistics, and shrinkage formulas, and only change the *final blending step* to be **type-adaptive**: for each row, we weight components by their available group counts (higher count ⇒ more trust), with a small set of per-component base weights to preserve your original mix. This usually reduces MAE (and therefore log-MAE) because rare/unreliable group means stop dominating predictions in sparse regions. The submission schema, paths, and runtime remain essentially the same, and it still writes a valid `.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

DATA_DIR = "/kaggle/data/champs-scalar-coupling"

print("Listing DATA_DIR:", DATA_DIR)
print(sorted([p for p in os.listdir(DATA_DIR) if p.endswith(".csv")])[:20])



## === cell 1
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
structures_path = os.path.join(DATA_DIR, "structures.csv")
pe_path = os.path.join(DATA_DIR, "potential_energy.csv")

train = pd.read_csv(
    train_path,
    usecols=[
        "molecule_name",
        "type",
        "atom_index_0",
        "atom_index_1",
        "scalar_coupling_constant",
    ],
)
test = pd.read_csv(
    test_path,
    usecols=["id", "molecule_name", "type", "atom_index_0", "atom_index_1"],
)
sample_sub = pd.read_csv(sample_path, usecols=["id"])

structures = pd.read_csv(
    structures_path,
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
)

potential_energy = pd.read_csv(pe_path, usecols=["molecule_name", "potential_energy"])

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

train = train.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
train = train.merge(s1, on=["molecule_name", "atom_index_1"], how="left")
test = test.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
test = test.merge(s1, on=["molecule_name", "atom_index_1"], how="left")

train = train.merge(potential_energy, on="molecule_name", how="left")
test = test.merge(potential_energy, on="molecule_name", how="left")

coord_cols = ["x0", "y0", "z0", "x1", "y1", "z1"]
for df_name, df in (("train", train), ("test", test)):
    missing_coords = df[coord_cols].isna().any(axis=1).sum()
    if missing_coords:
        print(
            f"WARNING: {df_name} has {missing_coords} rows with missing coordinates after merge; imputing 0.0"
        )
        df[coord_cols] = df[coord_cols].fillna(0.0)

pe_train_mean = train["potential_energy"].mean()
for df_name, df in (("train", train), ("test", test)):
    n_missing_pe = df["potential_energy"].isna().sum()
    if n_missing_pe:
        print(
            f"WARNING: {df_name} has {n_missing_pe} missing potential_energy; imputing train mean"
        )
        df["potential_energy"] = df["potential_energy"].fillna(pe_train_mean)

for df in (train, test):
    dx = df["x0"].to_numpy() - df["x1"].to_numpy()
    dy = df["y0"].to_numpy() - df["y1"].to_numpy()
    dz = df["z0"].to_numpy() - df["z1"].to_numpy()
    dist = np.sqrt(dx * dx + dy * dy + dz * dz).astype("float64")

    dist[~np.isfinite(dist)] = 0.0
    df["dist"] = dist

    df["dist_bin"] = np.clip(np.rint(df["dist"] * 10.0), 0, 300).astype("int16")

pe_edges = (
    train["potential_energy"].quantile([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).to_numpy()
)
pe_edges = np.unique(pe_edges)
if pe_edges.shape[0] < 3:
    pe_edges = np.unique(
        np.array(
            [
                train["potential_energy"].min(),
                train["potential_energy"].mean(),
                train["potential_energy"].max(),
            ],
            dtype="float64",
        )
    )
    if pe_edges.shape[0] < 3:
        pe_edges = np.array(
            [
                train["potential_energy"].min() - 1.0,
                train["potential_energy"].min(),
                train["potential_energy"].min() + 1.0,
            ],
            dtype="float64",
        )

for df in (train, test):
    df["pe_bin"] = pd.cut(
        df["potential_energy"], bins=pe_edges, include_lowest=True, labels=False
    ).astype("int16")

global_mean = train["scalar_coupling_constant"].mean()

type_stats = train.groupby("type")["scalar_coupling_constant"].agg(["mean", "count"])
type_mean = type_stats["mean"]
type_cnt = type_stats["count"]

tm = test["type"].map(type_mean)
tc = test["type"].map(type_cnt)

g_a0 = train.groupby(["type", "atom_0"])["scalar_coupling_constant"].agg(
    ["mean", "count"]
)
g_a1 = train.groupby(["type", "atom_1"])["scalar_coupling_constant"].agg(
    ["mean", "count"]
)

test_key_a0 = pd.MultiIndex.from_frame(test[["type", "atom_0"]])
test_key_a1 = pd.MultiIndex.from_frame(test[["type", "atom_1"]])

m_a0 = pd.Series(g_a0["mean"].reindex(test_key_a0).to_numpy(), index=test.index)
c_a0 = pd.Series(g_a0["count"].reindex(test_key_a0).to_numpy(), index=test.index)

m_a1 = pd.Series(g_a1["mean"].reindex(test_key_a1).to_numpy(), index=test.index)
c_a1 = pd.Series(g_a1["count"].reindex(test_key_a1).to_numpy(), index=test.index)

for df in (train, test):
    df[["atom_0", "atom_1"]] = df[["atom_0", "atom_1"]].astype("object")
    df["atom_min"] = df[["atom_0", "atom_1"]].min(axis=1)
    df["atom_max"] = df[["atom_0", "atom_1"]].max(axis=1)

g_atompair_sym = train.groupby(["type", "atom_min", "atom_max"])[
    "scalar_coupling_constant"
].agg(["mean", "count"])

test_key_atompair_sym = pd.MultiIndex.from_frame(test[["type", "atom_min", "atom_max"]])
m_ap = pd.Series(
    g_atompair_sym["mean"].reindex(test_key_atompair_sym).to_numpy(), index=test.index
)
c_ap = pd.Series(
    g_atompair_sym["count"].reindex(test_key_atompair_sym).to_numpy(), index=test.index
)

g_distbin = train.groupby(["type", "dist_bin"])["scalar_coupling_constant"].agg(
    ["mean", "count"]
)
test_key_distbin = pd.MultiIndex.from_frame(test[["type", "dist_bin"]])
m_db = pd.Series(
    g_distbin["mean"].reindex(test_key_distbin).to_numpy(), index=test.index
)
c_db = pd.Series(
    g_distbin["count"].reindex(test_key_distbin).to_numpy(), index=test.index
)

g_ap_db_sym = train.groupby(["type", "atom_min", "atom_max", "dist_bin"])[
    "scalar_coupling_constant"
].agg(["mean", "count"])
test_key_ap_db_sym = pd.MultiIndex.from_frame(
    test[["type", "atom_min", "atom_max", "dist_bin"]]
)
m_ap_db = pd.Series(
    g_ap_db_sym["mean"].reindex(test_key_ap_db_sym).to_numpy(), index=test.index
)
c_ap_db = pd.Series(
    g_ap_db_sym["count"].reindex(test_key_ap_db_sym).to_numpy(), index=test.index
)

g_db_pe = train.groupby(["type", "dist_bin", "pe_bin"])["scalar_coupling_constant"].agg(
    ["mean", "count"]
)
test_key_db_pe = pd.MultiIndex.from_frame(test[["type", "dist_bin", "pe_bin"]])
m_db_pe = pd.Series(
    g_db_pe["mean"].reindex(test_key_db_pe).to_numpy(), index=test.index
)
c_db_pe = pd.Series(
    g_db_pe["count"].reindex(test_key_db_pe).to_numpy(), index=test.index
)

K = 50.0
m_a0_s = (m_a0 * c_a0 + tm * K) / (c_a0 + K)
m_a1_s = (m_a1 * c_a1 + tm * K) / (c_a1 + K)

K_ap = 200.0
m_ap_s = (m_ap * c_ap + tm * K_ap) / (c_ap + K_ap)

K_db = 200.0
m_db_s = (m_db * c_db + tm * K_db) / (c_db + K_db)

K_ap_db = 300.0
m_ap_db_s = (m_ap_db * c_ap_db + m_ap_s * K_ap_db) / (c_ap_db + K_ap_db)

K_db_pe = 200.0
m_db_pe_s = (m_db_pe * c_db_pe + m_db_s * K_db_pe) / (c_db_pe + K_db_pe)

base_w_a0 = 0.15
base_w_a1 = 0.15
base_w_ap = 0.15
base_w_db = 0.08
base_w_ap_db = 0.37
base_w_db_pe = 0.10

eps = 1e-12
r_a0 = np.power(
    (
        c_a0.fillna(0.0).to_numpy(dtype="float64")
        / (c_a0.fillna(0.0).to_numpy(dtype="float64") + K + eps)
    ),
    0.5,
)
r_a1 = np.power(
    (
        c_a1.fillna(0.0).to_numpy(dtype="float64")
        / (c_a1.fillna(0.0).to_numpy(dtype="float64") + K + eps)
    ),
    0.5,
)
r_ap = np.power(
    (
        c_ap.fillna(0.0).to_numpy(dtype="float64")
        / (c_ap.fillna(0.0).to_numpy(dtype="float64") + K_ap + eps)
    ),
    0.5,
)
r_db = np.power(
    (
        c_db.fillna(0.0).to_numpy(dtype="float64")
        / (c_db.fillna(0.0).to_numpy(dtype="float64") + K_db + eps)
    ),
    0.5,
)
r_ap_db = np.power(
    (
        c_ap_db.fillna(0.0).to_numpy(dtype="float64")
        / (c_ap_db.fillna(0.0).to_numpy(dtype="float64") + K_ap_db + eps)
    ),
    0.5,
)
r_db_pe = np.power(
    (
        c_db_pe.fillna(0.0).to_numpy(dtype="float64")
        / (c_db_pe.fillna(0.0).to_numpy(dtype="float64") + K_db_pe + eps)
    ),
    0.5,
)

w_a0 = base_w_a0 * r_a0
w_a1 = base_w_a1 * r_a1
w_ap = base_w_ap * r_ap
w_db = base_w_db * r_db
w_ap_db = base_w_ap_db * r_ap_db
w_db_pe = base_w_db_pe * r_db_pe

den = w_a0 + w_a1 + w_ap + w_db + w_ap_db + w_db_pe
den = np.where(den <= 0.0, 1.0, den)

pred = (
    (w_a0 * m_a0_s.to_numpy(dtype="float64"))
    + (w_a1 * m_a1_s.to_numpy(dtype="float64"))
    + (w_ap * m_ap_s.to_numpy(dtype="float64"))
    + (w_db * m_db_s.to_numpy(dtype="float64"))
    + (w_ap_db * m_ap_db_s.to_numpy(dtype="float64"))
    + (w_db_pe * m_db_pe_s.to_numpy(dtype="float64"))
) / den
pred = pd.Series(pred, index=test.index)

pred = pred.fillna(tm).fillna(global_mean)

submission = pd.DataFrame(
    {
        "id": test["id"].astype(sample_sub["id"].dtype, copy=False),
        "scalar_coupling_constant": pred.astype("float64"),
    }
)

assert submission.shape[0] == test.shape[0], "Submission row count mismatch vs test.csv"
assert submission["id"].is_unique, "Test ids are expected to be unique"
assert set(submission.columns) == {"id", "scalar_coupling_constant"}

out_path = "my_blend_6.csv"
submission.to_csv(out_path, index=False)
print("Wrote submission:", out_path, "rows:", len(submission))
print(submission.head())
