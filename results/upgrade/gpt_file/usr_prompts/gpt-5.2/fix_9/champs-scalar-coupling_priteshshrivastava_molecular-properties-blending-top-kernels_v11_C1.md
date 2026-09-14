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
BOND_FUDGE = 0.45  # tolerant margin to allow for geometry variation


def _bond_threshold(a, b):
    ra = COV_RAD.get(a, 0.70)
    rb = COV_RAD.get(b, 0.70)
    return ra + rb + BOND_FUDGE


def compute_graph_dist_for_pairs(struct_df, pairs_df):
    """
    struct_df columns: molecule_name, atom_index, atom, x, y, z
    pairs_df columns: molecule_name, atom_index_0, atom_index_1
    returns: pd.Series aligned to pairs_df index with nullable Int16 graph_dist (1,2,3,...), or <NA> if not found
    """
    out = pd.Series(
        pd.array([pd.NA] * len(pairs_df), dtype="Int16"), index=pairs_df.index
    )

    pairs_by_mol = pairs_df.groupby("molecule_name", sort=False)
    structs_by_mol = struct_df.groupby("molecule_name", sort=False)

    mols = pairs_by_mol.indices.keys()

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

        pairs_m = pairs_by_mol.get_group(mol)
        needed = pairs_m[["atom_index_0", "atom_index_1"]].to_numpy(dtype=np.int32)
        by_src = defaultdict(list)
        for row_i, (a0, a1) in zip(pairs_m.index, needed):
            by_src[int(a0)].append((row_i, int(a1)))

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
            for row_i, a1 in targets:
                if a1 not in idx_to_pos:
                    continue
                d = dist[idx_to_pos[a1]]
                if d >= 0:
                    out.at[row_i] = int(d)

    return out


train2["graph_dist"] = compute_graph_dist_for_pairs(
    structures[["molecule_name", "atom_index", "atom", "x", "y", "z"]],
    train2[["molecule_name", "atom_index_0", "atom_index_1"]],
)
test2["graph_dist"] = compute_graph_dist_for_pairs(
    structures[["molecule_name", "atom_index", "atom", "x", "y", "z"]],
    test2[["molecule_name", "atom_index_0", "atom_index_1"]],
)

print(
    "graph_dist missing:",
    f"train={int(train2['graph_dist'].isna().sum())} ({float(train2['graph_dist'].isna().mean()):.6f}),",
    f"test={int(test2['graph_dist'].isna().sum())} ({float(test2['graph_dist'].isna().mean()):.6f})",
)



## === cell 3
BIN_WIDTH = 0.05
for df in (train2, test2):
    dist_bin = np.floor(df["dist"] / BIN_WIDTH)
    df["dist_bin"] = pd.Series(dist_bin).astype("Int32")

key_cols = ["type", "atom_0", "atom_1"]
key_cols_dist = ["type", "atom_0", "atom_1", "dist_bin"]

type_graph_mean = (
    train2.groupby(["type", "graph_dist"], dropna=False)["scalar_coupling_constant"]
    .mean()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "mean_type_graph"})
)

type_atom_graph_mean = (
    train2.groupby(["type", "atom_0", "atom_1", "graph_dist"], dropna=False)[
        "scalar_coupling_constant"
    ]
    .mean()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "mean_type_atom_graph"})
)

type_dist_mean = (
    train2.groupby(["type", "dist_bin"], dropna=False)["scalar_coupling_constant"]
    .mean()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "mean_type_dist"})
)

type_atom_dist_mean = (
    train2.groupby(key_cols_dist, dropna=False)["scalar_coupling_constant"]
    .mean()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "mean_type_atom_dist"})
)

type_atom_mean = (
    train2.groupby(key_cols, dropna=False)["scalar_coupling_constant"]
    .mean()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "mean_type_atom"})
)

type_mean = train2.groupby("type")["scalar_coupling_constant"].mean()
global_mean = float(train2["scalar_coupling_constant"].mean())

test2m = test2.merge(type_atom_dist_mean, on=key_cols_dist, how="left")
test2m = test2m.merge(type_atom_mean, on=key_cols, how="left")
test2m = test2m.merge(type_dist_mean, on=["type", "dist_bin"], how="left")
test2m = test2m.merge(
    type_atom_graph_mean, on=["type", "atom_0", "atom_1", "graph_dist"], how="left"
)
test2m = test2m.merge(type_graph_mean, on=["type", "graph_dist"], how="left")

pred = test2m["mean_type_atom_dist"]
pred = pred.fillna(test2m["mean_type_atom_graph"])
pred = pred.fillna(test2m["mean_type_atom"])
pred = pred.fillna(test2m["mean_type_graph"])
pred = pred.fillna(test2m["mean_type_dist"])
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
