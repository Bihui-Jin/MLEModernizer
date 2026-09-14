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

-1.6630743227479918

# 6. Current score

1.23566

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'I remove the dependency on missing external blend files and replace it with a minimal, self-contained pipeline that only uses the provided CHAMPS dataset files in this environment. To keep core logic simple and stable, the model be a per-`type` baseline using the training mean `scalar_coupling_constant` for that type (and global mean as fallback), which guarantees end-to-end execution and a valid `submission.csv`. I also fix paths to match the given `/kaggle/data/champs-scalar-coupling/` layout and ensure the submission rows align exactly to `test.csv` `id` order. This is score-safe (not optimal) but should produce a valid score and unblock you from “Not yielded.”'
- What this solution (achieved 1.23566) has done: 'I fix the runtime error caused by trying to construct a `Series` directly from a `MultiIndex`, which is no longer supported in recent pandas. The minimal, score-neutral fix is to avoid creating a `MultiIndex`-backed Series and instead merge the precomputed `type/atom_0/atom_1` means onto `test2`. I keep the same fallback logic (type mean, then global mean) and preserve the exact submission formatting and ID alignment checks so a valid `submission.csv` is always produced end-to-end.'
- What this solution (achieved 1.23566) has done: 'The crash comes from `dist` being NaN for some rows (missing structure merges), which then makes `dist_bin.astype(int32)` fail; I make the structure merge robust and compute `dist`/`dist_bin` with explicit NaN handling using pandas’ nullable integer type. This keeps the same core logic (distance binning + hierarchical mean fallback) while ensuring groupby/merge keys can carry missing values safely. I also add a small sanity check to quantify missing coordinates (for debugging) and keep the submission ID alignment assertion so the output is always valid. The result run end-to-end and write `submission.csv`.'
- What this solution (achieved 1.23566) has done: 'Your current score is far above (worse than) the target on a lower-is-better metric, so we should improve accuracy with minimal changes while keeping the same “hierarchical mean fallback” core logic. The smallest high-impact tweak is to add molecule-level means (per type) and use them before the global type mean, because the split is by molecule and these aggregates are still fully train-derived and leakage-free. We also make the distance binning slightly more stable by using an integer bin via floor (still the same distance-binning idea) to reduce boundary noise, while preserving the same overall pipeline and submission alignment checks. The output remains a valid `submission.csv` with the required columns and id order.'
- What this solution (achieved 1.23566) has done: 'Your current score (1.23566, lower-is-better) is far worse than the target (-1.663), so we should legitimately improve accuracy with minimal changes while keeping the same “hierarchical mean fallback” approach. The smallest high-impact fix is to correct the fallback order to avoid using molecule-level means: because the split is by molecule, `molecule_name` seen in test is never in train, so `mean_mol_type`/`mean_mol` are always missing and add no value. Then we add one extra, still-aggregated fallback level that is available in both train and test: `type + dist_bin` mean (distance-binned per coupling type), inserted between the most specific and less specific means. This keeps the same core logic (distance binning + chained means) while typically improving MAE a lot versus type-only baselines, and still produces a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 1.23566) has done: 'Your current score (1.23566, lower-is-better) is far worse than the target (-1.663), so we should legitimately improve accuracy with minimal changes while keeping the same “distance + hierarchical mean fallback” core logic. The biggest missing signal in your current aggregates is that coupling strength depends strongly on bond separation (1J/2J/3J) beyond raw Euclidean distance, so we add a tiny, train-derived feature: the shortest-path “graph distance” between the two atoms using a naive bond graph built from inter-atomic distances (no new model, just another bin/key for means). Then we insert one extra aggregation level `type + graph_dist` (and optionally `type+atom_0+atom_1+graph_dist`) ahead of the coarser fallbacks, keeping all existing levels and submission alignment intact. This stays leakage-free (test molecules are disjoint) and should move the score substantially toward the target while preserving the same overall approach and runtime under the limit.'
- What this solution (achieved 1.23566) has done: 'We keep the same “distance + graph shortest-path + hierarchical mean fallback” core approach, but make two small, score-relevant fixes that typically reduce MAE without changing semantics. First, we correct an inefficiency/bug in `dist_bin` creation that can misalign indices (because `pd.Series(dist_bin)` creates a fresh RangeIndex), by creating it with `index=df.index` so merges and group keys are consistent. Second, we compute graph distances once per molecule by building the bond graph a single time and then answering all needed pairs (train+test) for that molecule together; this preserves the exact graph logic/thresholds but avoids timeouts and makes it feasible to use the `type+graph_dist` aggregates reliably. The rest of the aggregation levels, fallback order, paths, and submission alignment checks remain the same.'
- What this solution (achieved 1.23566) has done: 'Your current score (1.23566, lower-is-better) is far worse than the target (-1.663), so we should improve accuracy with minimal, core-logic-preserving changes to your hierarchical-mean pipeline. The largest accuracy bug/risk is that `graph_dist` is computed by iterating over a Python `set(mols)`, which makes the run non-deterministic; since the code uses `.at[...]` scalar writes into the same output Series, the final `graph_dist` values can vary between runs and harm score stability—so we make molecule iteration deterministic. Next, we add one extra aggregation level that is still “means + fallback” (same modeling idea) but uses a strong, cheap signal you already computed: `type + graph_dist + dist_bin`, inserted ahead of coarser fallbacks to reduce MAE without changing the overall approach. Finally, we slightly reduce oversmoothing by using `median` instead of `mean` for the most specific bucketed aggregates (still an aggregate baseline, no new model), which tends to lower MAE under heavy-tailed errors and should move the score toward the target.'
- What this solution (achieved 1.23566) has done: 'Your current score (1.23566, lower-is-better) is far worse than the target (-1.663), so we should materially improve accuracy while keeping the same core “graph distance + distance bin + hierarchical aggregate fallback” logic. The biggest likely accuracy issue here is the naive bond graph: a single global `BOND_FUDGE` can create many spurious bonds (especially for H), which makes `graph_dist` noisy and harms the downstream `type+graph_dist` aggregates. I keep the exact BFS/aggregate pipeline, but tighten the bond criterion in a minimal, chemistry-consistent way by (a) using a smaller fudge and (b) adding a special-case cap for H–H bonds (rarely bonded), which typically stabilizes graph distances. I also add a tiny sanity print of graph distance distribution (no behavior change) to confirm the feature became less degenerate, and keep submission alignment identical.'
- What this solution (achieved 1.23566) has done: 'Your current score (1.23566, lower-is-better) is far from the target (-1.663), so we need a legitimate accuracy improvement while keeping the same “feature keys + hierarchical aggregate fallback” core logic. The biggest low-risk gain is to add CHAMPS-provided per-atom and per-molecule signals (mulliken charges, magnetic shielding tensors, dipole, potential energy) and use them only to create additional aggregated lookup levels (no model change) that are available for both train/test. We keep your existing distance/graph distance computation and current fallback chain, but insert a couple of new, stronger keys (e.g., `type+atom_0+atom_1+graph_dist+dist_bin+charge_bin` and `type+dist_bin+charge_bin`) ahead of the coarser fallbacks. This should materially reduce MAE and move the score toward the target, while remaining deterministic, leakage-free, and producing the same submission format.'
- What this solution (achieved 1.23566) has done: 'We need to substantially improve accuracy (lower logMAE) toward the target, and the largest low-risk gain within your current “feature keys + hierarchical aggregate fallback” logic is to ensure the most specific aggregates don’t overfit tiny buckets. I keep your exact features and fallback chain, but add simple, deterministic smoothing: for each aggregated lookup we also compute its group count and only use that lookup when the count exceeds a small threshold; otherwise we fall back to the next level. This preserves the same semantics (still just per-bucket train statistics with fallbacks), but reduces noisy medians/means that can inflate MAE on unseen molecules. Changes are localized to the aggregation construction and the `pred.fillna(...)` chain; paths, graph logic, and submission formatting remain unchanged.'
- What this solution (achieved 1.23566) has done: 'Your current score (1.23566, lower-is-better) is still far from the target (-1.663), so we should improve accuracy (reduce MAE) with minimal, core-logic-preserving changes. The biggest low-risk gain inside your existing “hierarchical aggregate fallback” approach is to (1) make the most-specific buckets less noisy via deterministic shrinkage toward the next coarser level (still just train-derived aggregates, not a new model), and (2) ensure binning/keys handle missing values cleanly so we don’t lose matches. Concretely, we add count-aware smoothed estimates for the top few aggregation levels using `smoothed = (cnt*stat + alpha*backoff)/(cnt+alpha)` and use those values directly in the same fallback chain. This keeps all your existing features (dist/graph_dist/qbin/stbin/etc.), your BFS graph logic, and produces the same valid `submission.csv` format and ID alignment.'
- What this solution (achieved 1.23566) has done: 'Your current score is much worse than the target on a lower-is-better metric, so we should legitimately improve accuracy while keeping your exact “engineered features + hierarchical aggregate fallback” approach. The smallest high-impact fix is to remove the early, overly-coarse `type_atom_dist` bucket from the top of the fallback chain (it tends to dominate and can be noisy/misalleading versus the richer graph/charge-based keys you already compute). Next, we add one stronger-but-still-consistent aggregation level that uses information already present: `type + graph_dist + dist_bin + stbin`, inserted ahead of the coarser fallbacks to reduce MAE. Finally, we make binning robust to missing aux values by using pandas nullable integers for bins (so NaNs don’t turn into arbitrary integers), preserving the same semantics while increasing match rates for the aggregate joins.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from collections import deque, defaultdict

BASE_PATH = "/kaggle/data/champs-scalar-coupling"

assert os.path.exists(BASE_PATH), f"Dataset path not found: {BASE_PATH}"
print("Files in dataset dir (sample):", sorted(os.listdir(BASE_PATH))[:20])



## === cell 1
train_path = os.path.join(BASE_PATH, "train.csv")
test_path = os.path.join(BASE_PATH, "test.csv")
sample_path = os.path.join(BASE_PATH, "sample_submission.csv")
structures_path = os.path.join(BASE_PATH, "structures.csv")

mulliken_path = os.path.join(BASE_PATH, "mulliken_charges.csv")
shield_path = os.path.join(BASE_PATH, "magnetic_shielding_tensors.csv")
dipole_path = os.path.join(BASE_PATH, "dipole_moments.csv")
energy_path = os.path.join(BASE_PATH, "potential_energy.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

required_train_cols = {
    "type",
    "scalar_coupling_constant",
    "molecule_name",
    "atom_index_0",
    "atom_index_1",
}
required_test_cols = {"id", "type", "molecule_name", "atom_index_0", "atom_index_1"}
required_sub_cols = {"id", "scalar_coupling_constant"}

missing_train = required_train_cols - set(train.columns)
missing_test = required_test_cols - set(test.columns)
missing_sub = required_sub_cols - set(sample_sub.columns)

assert not missing_train, f"train.csv missing columns: {missing_train}"
assert not missing_test, f"test.csv missing columns: {missing_test}"
assert not missing_sub, f"sample_submission.csv missing columns: {missing_sub}"

structures = pd.read_csv(
    structures_path, usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"]
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

train2 = train.merge(
    s0, on=["molecule_name", "atom_index_0"], how="left", validate="m:1"
)
train2 = train2.merge(
    s1, on=["molecule_name", "atom_index_1"], how="left", validate="m:1"
)
test2 = test.merge(s0, on=["molecule_name", "atom_index_0"], how="left", validate="m:1")
test2 = test2.merge(
    s1, on=["molecule_name", "atom_index_1"], how="left", validate="m:1"
)

for df in (train2, test2):
    dx = df["x0"] - df["x1"]
    dy = df["y0"] - df["y1"]
    dz = df["z0"] - df["z1"]
    df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype("float64")

train_missing_dist = int(train2["dist"].isna().sum())
test_missing_dist = int(test2["dist"].isna().sum())
print(
    f"Missing dist: train={train_missing_dist} ({train_missing_dist/len(train2):.6f}), "
    f"test={test_missing_dist} ({test_missing_dist/len(test2):.6f})"
)



## === cell 2
COV_RAD = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57}

BOND_FUDGE = 0.25
HH_MAX_BOND = 0.90


def _bond_threshold(a, b):
    ra = COV_RAD.get(a, 0.70)
    rb = COV_RAD.get(b, 0.70)
    thr = ra + rb + BOND_FUDGE
    if (a == "H") and (b == "H"):
        thr = min(thr, HH_MAX_BOND)
    return thr


def compute_graph_dist_for_pairs_combined(struct_df, train_pairs_df, test_pairs_df):
    """
    Compute graph shortest-path distance on a naive bond graph.
    This preserves the exact bond rule + BFS logic, but builds each molecule graph once
    and answers all (train+test) queries for that molecule together for speed/stability.
    """
    train_out = pd.Series(
        pd.array([pd.NA] * len(train_pairs_df), dtype="Int16"),
        index=train_pairs_df.index,
    )
    test_out = pd.Series(
        pd.array([pd.NA] * len(test_pairs_df), dtype="Int16"), index=test_pairs_df.index
    )

    structs_by_mol = struct_df.groupby("molecule_name", sort=False)
    train_by_mol = train_pairs_df.groupby("molecule_name", sort=False)
    test_by_mol = test_pairs_df.groupby("molecule_name", sort=False)

    mols = sorted(set(train_by_mol.indices.keys()) | set(test_by_mol.indices.keys()))

    for mol in mols:
        if mol not in structs_by_mol.indices:
            continue

        s = structs_by_mol.get_group(mol)
        atom_idx = s["atom_index"].to_numpy()
        atoms = s["atom"].to_numpy()
        coords = s[["x", "y", "z"]].to_numpy(dtype=np.float64)

        n = len(s)
        if n <= 1:
            continue

        idx_to_pos = {int(ai): i for i, ai in enumerate(atom_idx)}

        adj = [[] for _ in range(n)]
        for i in range(n):
            ai = atoms[i]
            xi, yi, zi = coords[i]
            for j in range(i + 1, n):
                aj = atoms[j]
                thr = _bond_threshold(ai, aj)
                dx = xi - coords[j, 0]
                dy = yi - coords[j, 1]
                dz = zi - coords[j, 2]
                d = (dx * dx + dy * dy + dz * dz) ** 0.5
                if d <= thr:
                    adj[i].append(j)
                    adj[j].append(i)

        by_src = defaultdict(list)

        if mol in train_by_mol.indices:
            g = train_by_mol.get_group(mol)
            p = g[["atom_index_0", "atom_index_1"]].to_numpy(dtype=np.int32)
            for row_i, (a0, a1) in zip(g.index, p):
                by_src[int(a0)].append(("train", row_i, int(a1)))

        if mol in test_by_mol.indices:
            g = test_by_mol.get_group(mol)
            p = g[["atom_index_0", "atom_index_1"]].to_numpy(dtype=np.int32)
            for row_i, (a0, a1) in zip(g.index, p):
                by_src[int(a0)].append(("test", row_i, int(a1)))

        for a0, targets in by_src.items():
            if a0 not in idx_to_pos:
                continue
            src = idx_to_pos[a0]
            dist = np.full(n, -1, dtype=np.int16)
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                du = dist[u] + 1
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du
                        q.append(v)

            for which, row_i, a1 in targets:
                if a1 not in idx_to_pos:
                    continue
                d = dist[idx_to_pos[a1]]
                if d >= 0:
                    if which == "train":
                        train_out.at[row_i] = int(d)
                    else:
                        test_out.at[row_i] = int(d)

    return train_out, test_out


train_gd, test_gd = compute_graph_dist_for_pairs_combined(
    structures[["molecule_name", "atom_index", "atom", "x", "y", "z"]],
    train2[["molecule_name", "atom_index_0", "atom_index_1"]],
    test2[["molecule_name", "atom_index_0", "atom_index_1"]],
)
train2["graph_dist"] = train_gd
test2["graph_dist"] = test_gd

print(
    "graph_dist missing:",
    f"train={int(train2['graph_dist'].isna().sum())} ({float(train2['graph_dist'].isna().mean()):.6f}),",
    f"test={int(test2['graph_dist'].isna().sum())} ({float(test2['graph_dist'].isna().mean()):.6f})",
)

print("graph_dist value counts (train, top 12 incl NA):")
print(train2["graph_dist"].value_counts(dropna=False).head(12))



## === cell 3
mull = pd.read_csv(
    mulliken_path, usecols=["molecule_name", "atom_index", "mulliken_charge"]
)
m0 = mull.rename(columns={"atom_index": "atom_index_0", "mulliken_charge": "q0"})
m1 = mull.rename(columns={"atom_index": "atom_index_1", "mulliken_charge": "q1"})
train2 = train2.merge(
    m0, on=["molecule_name", "atom_index_0"], how="left", validate="m:1"
)
train2 = train2.merge(
    m1, on=["molecule_name", "atom_index_1"], how="left", validate="m:1"
)
test2 = test2.merge(
    m0, on=["molecule_name", "atom_index_0"], how="left", validate="m:1"
)
test2 = test2.merge(
    m1, on=["molecule_name", "atom_index_1"], how="left", validate="m:1"
)

shield = pd.read_csv(
    shield_path,
    usecols=["molecule_name", "atom_index", "XX", "YY", "ZZ"],
)
shield["shield_trace"] = (shield["XX"] + shield["YY"] + shield["ZZ"]).astype("float64")
shield0 = shield[["molecule_name", "atom_index", "shield_trace"]].rename(
    columns={"atom_index": "atom_index_0", "shield_trace": "st0"}
)
shield1 = shield[["molecule_name", "atom_index", "shield_trace"]].rename(
    columns={"atom_index": "atom_index_1", "shield_trace": "st1"}
)
train2 = train2.merge(
    shield0, on=["molecule_name", "atom_index_0"], how="left", validate="m:1"
)
train2 = train2.merge(
    shield1, on=["molecule_name", "atom_index_1"], how="left", validate="m:1"
)
test2 = test2.merge(
    shield0, on=["molecule_name", "atom_index_0"], how="left", validate="m:1"
)
test2 = test2.merge(
    shield1, on=["molecule_name", "atom_index_1"], how="left", validate="m:1"
)

dip = pd.read_csv(dipole_path, usecols=["molecule_name", "X", "Y", "Z"])
dip["dipole_mag"] = np.sqrt(
    dip["X"] * dip["X"] + dip["Y"] * dip["Y"] + dip["Z"] * dip["Z"]
).astype("float64")
dip = dip[["molecule_name", "dipole_mag"]]
energy = pd.read_csv(energy_path, usecols=["molecule_name", "potential_energy"])

train2 = train2.merge(dip, on="molecule_name", how="left", validate="m:1")
train2 = train2.merge(energy, on="molecule_name", how="left", validate="m:1")
test2 = test2.merge(dip, on="molecule_name", how="left", validate="m:1")
test2 = test2.merge(energy, on="molecule_name", how="left", validate="m:1")

Q_BIN_W = 0.02
ST_BIN_W = 2.0
DIPOLE_BIN_W = 0.2
ENERGY_BIN_W = 0.05

for df in (train2, test2):
    df["dq"] = (df["q0"] - df["q1"]).astype("float64")
    df["abs_dq"] = np.abs(df["dq"]).astype("float64")
    df["qbin"] = pd.array(np.floor(df["abs_dq"] / Q_BIN_W), dtype="Int16")

    df["dst"] = (df["st0"] - df["st1"]).astype("float64")
    df["abs_dst"] = np.abs(df["dst"]).astype("float64")
    df["stbin"] = pd.array(np.floor(df["abs_dst"] / ST_BIN_W), dtype="Int16")

    df["dipole_bin"] = pd.array(
        np.floor(df["dipole_mag"] / DIPOLE_BIN_W), dtype="Int16"
    )
    df["energy_bin"] = pd.array(
        np.floor((df["potential_energy"] - (-76.0)) / ENERGY_BIN_W), dtype="Int16"
    )

print(
    "Aux missing rates:",
    "q0",
    float(train2["q0"].isna().mean()),
    "st0",
    float(train2["st0"].isna().mean()),
    "dipole_mag",
    float(train2["dipole_mag"].isna().mean()),
    "potential_energy",
    float(train2["potential_energy"].isna().mean()),
)



## === cell 4
BIN_WIDTH = 0.05
for df in (train2, test2):
    df["dist_bin"] = pd.array(np.floor(df["dist"] / BIN_WIDTH), dtype="Int32")

key_cols = ["type", "atom_0", "atom_1"]
key_cols_dist = ["type", "atom_0", "atom_1", "dist_bin"]


def agg_with_count(df, by_cols, target_col, agg_name, agg_func):
    g = df.groupby(by_cols, dropna=False)[target_col]
    out = g.agg([(agg_name, agg_func), ("cnt", "size")]).reset_index()
    out = out.rename(columns={"cnt": f"cnt_{agg_name}"})
    return out


ALPHA_VERY_SPEC = 100.0
ALPHA_SPEC = 300.0
ALPHA_MED = 800.0

MIN_CNT_VERY_SPEC = 50
MIN_CNT_SPEC = 200
MIN_CNT_MED = 500

type_atom_dist_med = agg_with_count(
    train2, key_cols_dist, "scalar_coupling_constant", "median_type_atom_dist", "median"
)

type_atom_graph_distbin_qbin_med = agg_with_count(
    train2,
    ["type", "atom_0", "atom_1", "graph_dist", "dist_bin", "qbin"],
    "scalar_coupling_constant",
    "median_type_atom_graph_distbin_qbin",
    "median",
)

type_graph_distbin_qbin_mean = agg_with_count(
    train2,
    ["type", "graph_dist", "dist_bin", "qbin"],
    "scalar_coupling_constant",
    "mean_type_graph_distbin_qbin",
    "mean",
)

type_dist_qbin_mean = agg_with_count(
    train2,
    ["type", "dist_bin", "qbin"],
    "scalar_coupling_constant",
    "mean_type_dist_qbin",
    "mean",
)

type_dist_stbin_mean = agg_with_count(
    train2,
    ["type", "dist_bin", "stbin"],
    "scalar_coupling_constant",
    "mean_type_dist_stbin",
    "mean",
)

type_graph_distbin_stbin_mean = agg_with_count(
    train2,
    ["type", "graph_dist", "dist_bin", "stbin"],
    "scalar_coupling_constant",
    "mean_type_graph_distbin_stbin",
    "mean",
)

type_atom_graph_med = agg_with_count(
    train2,
    ["type", "atom_0", "atom_1", "graph_dist"],
    "scalar_coupling_constant",
    "median_type_atom_graph",
    "median",
)

type_atom_mean = agg_with_count(
    train2, key_cols, "scalar_coupling_constant", "mean_type_atom", "mean"
)

type_graph_distbin_mean = agg_with_count(
    train2,
    ["type", "graph_dist", "dist_bin"],
    "scalar_coupling_constant",
    "mean_type_graph_distbin",
    "mean",
)

type_graph_mean = agg_with_count(
    train2,
    ["type", "graph_dist"],
    "scalar_coupling_constant",
    "mean_type_graph",
    "mean",
)

type_dist_mean = agg_with_count(
    train2,
    ["type", "dist_bin"],
    "scalar_coupling_constant",
    "mean_type_dist",
    "mean",
)

type_mean = train2.groupby("type")["scalar_coupling_constant"].mean()
global_mean = float(train2["scalar_coupling_constant"].mean())

test2m = test2.copy()

test2m = test2m.merge(type_dist_mean, on=["type", "dist_bin"], how="left")
test2m = test2m.merge(type_graph_mean, on=["type", "graph_dist"], how="left")
test2m = test2m.merge(
    type_graph_distbin_mean, on=["type", "graph_dist", "dist_bin"], how="left"
)

test2m = test2m.merge(type_atom_mean, on=key_cols, how="left")
test2m = test2m.merge(
    type_atom_graph_med, on=["type", "atom_0", "atom_1", "graph_dist"], how="left"
)
test2m = test2m.merge(
    type_dist_stbin_mean, on=["type", "dist_bin", "stbin"], how="left"
)
test2m = test2m.merge(
    type_graph_distbin_stbin_mean,
    on=["type", "graph_dist", "dist_bin", "stbin"],
    how="left",
)
test2m = test2m.merge(type_dist_qbin_mean, on=["type", "dist_bin", "qbin"], how="left")
test2m = test2m.merge(
    type_graph_distbin_qbin_mean,
    on=["type", "graph_dist", "dist_bin", "qbin"],
    how="left",
)
test2m = test2m.merge(
    type_atom_graph_distbin_qbin_med,
    on=["type", "atom_0", "atom_1", "graph_dist", "dist_bin", "qbin"],
    how="left",
)
test2m = test2m.merge(type_atom_dist_med, on=key_cols_dist, how="left")


eps = 1e-12

back_type_dist = test2m["mean_type_dist"].astype("float64")
back_type_graph = test2m["mean_type_graph"].astype("float64")
back_type_graph_dist = test2m["mean_type_graph_distbin"].astype("float64")
back_type_atom_graph = test2m["median_type_atom_graph"].astype("float64")

cnt = test2m["cnt_mean_type_graph_distbin_stbin"].astype("float64")
stat = test2m["mean_type_graph_distbin_stbin"].astype("float64")
test2m["smooth_type_graph_distbin_stbin"] = (
    cnt * stat + ALPHA_MED * back_type_graph_dist
) / (cnt + ALPHA_MED + eps)

cnt = test2m["cnt_median_type_atom_dist"].astype("float64")
stat = test2m["median_type_atom_dist"].astype("float64")
test2m["smooth_type_atom_dist"] = (cnt * stat + ALPHA_SPEC * back_type_dist) / (
    cnt + ALPHA_SPEC + eps
)

cnt = test2m["cnt_median_type_atom_graph"].astype("float64")
stat = test2m["median_type_atom_graph"].astype("float64")
test2m["smooth_type_atom_graph"] = (cnt * stat + ALPHA_SPEC * back_type_graph) / (
    cnt + ALPHA_SPEC + eps
)

cnt = test2m["cnt_mean_type_graph_distbin_qbin"].astype("float64")
stat = test2m["mean_type_graph_distbin_qbin"].astype("float64")
test2m["smooth_type_graph_distbin_qbin"] = (
    cnt * stat + ALPHA_MED * back_type_graph_dist
) / (cnt + ALPHA_MED + eps)

cnt = test2m["cnt_mean_type_dist_qbin"].astype("float64")
stat = test2m["mean_type_dist_qbin"].astype("float64")
test2m["smooth_type_dist_qbin"] = (cnt * stat + ALPHA_MED * back_type_dist) / (
    cnt + ALPHA_MED + eps
)

cnt = test2m["cnt_mean_type_dist_stbin"].astype("float64")
stat = test2m["mean_type_dist_stbin"].astype("float64")
test2m["smooth_type_dist_stbin"] = (cnt * stat + ALPHA_MED * back_type_dist) / (
    cnt + ALPHA_MED + eps
)

cnt = test2m["cnt_median_type_atom_graph_distbin_qbin"].astype("float64")
stat = test2m["median_type_atom_graph_distbin_qbin"].astype("float64")
test2m["smooth_type_atom_graph_distbin_qbin"] = (
    cnt * stat + ALPHA_VERY_SPEC * back_type_atom_graph
) / (cnt + ALPHA_VERY_SPEC + eps)

pred = pd.Series(np.nan, index=test2m.index, dtype="float64")

c = test2m["cnt_median_type_atom_graph_distbin_qbin"]
v = test2m["smooth_type_atom_graph_distbin_qbin"]
pred = pred.fillna(v.where(c >= MIN_CNT_VERY_SPEC))

c = test2m["cnt_mean_type_graph_distbin_qbin"]
v = test2m["smooth_type_graph_distbin_qbin"]
pred = pred.fillna(v.where(c >= MIN_CNT_MED))

c = test2m["cnt_mean_type_graph_distbin_stbin"]
v = test2m["smooth_type_graph_distbin_stbin"]
pred = pred.fillna(v.where(c >= MIN_CNT_MED))

c = test2m["cnt_mean_type_dist_qbin"]
v = test2m["smooth_type_dist_qbin"]
pred = pred.fillna(v.where(c >= MIN_CNT_MED))

c = test2m["cnt_mean_type_dist_stbin"]
v = test2m["smooth_type_dist_stbin"]
pred = pred.fillna(v.where(c >= MIN_CNT_MED))

c = test2m["cnt_median_type_atom_graph"]
v = test2m["smooth_type_atom_graph"]
pred = pred.fillna(v.where(c >= MIN_CNT_SPEC))

c = test2m["cnt_mean_type_atom"]
v = test2m["mean_type_atom"]
pred = pred.fillna(v.where(c >= MIN_CNT_MED))

c = test2m["cnt_mean_type_graph_distbin"]
v = test2m["mean_type_graph_distbin"]
pred = pred.fillna(v.where(c >= MIN_CNT_MED))

c = test2m["cnt_mean_type_graph"]
v = test2m["mean_type_graph"]
pred = pred.fillna(v.where(c >= MIN_CNT_MED))

c = test2m["cnt_mean_type_dist"]
v = test2m["mean_type_dist"]
pred = pred.fillna(v.where(c >= MIN_CNT_MED))

pred = pred.fillna(test2m["type"].map(type_mean))
pred = pred.fillna(global_mean).astype("float64")

submission = pd.DataFrame(
    {
        "id": test2m["id"].astype(sample_sub["id"].dtype, copy=False),
        "scalar_coupling_constant": pred.values,
    }
)

submission = submission.sort_values("id").reset_index(drop=True)
expected_ids = sample_sub.sort_values("id")["id"].reset_index(drop=True)
assert submission["id"].equals(
    expected_ids
), "Submission ids do not match sample_submission ids/order."

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", submission.shape)
print(submission.head())
