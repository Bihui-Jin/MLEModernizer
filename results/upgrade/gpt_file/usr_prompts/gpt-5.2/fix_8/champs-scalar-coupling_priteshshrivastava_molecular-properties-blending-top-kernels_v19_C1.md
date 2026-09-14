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

-1.6819062841656574

# 6. Current score

1.23566

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'Your notebook fails because it references external Kaggle datasets (`champs-blending-tutorial`, `otherkernelsadded`) that are not present in this environment, so no submission is produced. I replace that broken blending step with a minimal, self-contained baseline that only uses the provided competition files and writes a valid `submission.csv`. To keep changes minimal and stable, the baseline predict the mean `scalar_coupling_constant` per `type` from `train.csv` and apply it to `test.csv`, with a safe fallback to the global mean for any unseen types. This run end-to-end within the time limit and generate a correctly formatted CSV.'
- What this solution (achieved 1.23566) has done: 'Your current baseline (mean by `type`) is leaving a lot of error on the table; the smallest legitimate improvement that preserves the “simple aggregate-statistics model” core logic is to predict using a more specific conditional mean. I switch from `E[y|type]` to `E[y|type, atom0, atom1]` by joining atom identities from `structures.csv`, and use a safe backoff chain (`type+pair` → `type+atom0`/`type+atom1` → `type` → global) to avoid NaNs and keep it robust. This stays within the same approach (groupby means applied to test), but should move the log-MAE score substantially downward (better) toward your negative target. I also keep the submission alignment check and ensure we still write a valid `submission.csv`.'
- What this solution (achieved 1.23566) has done: 'Your current solution is a “groupby-mean with backoff” baseline; the smallest score-improving change that preserves this core logic is to condition the means on a bit more structural context without introducing a new model. I add a cheap geometric feature (inter-atomic distance) from `structures.csv`, then use type+pair+distance-bin conditional means as the highest-priority lookup, with your existing backoff chain unchanged. This should reduce MAE (and thus the log-MAE metric) versus using only atom identities, moving the score downward toward your negative target. I also keep the submission alignment guard and ensure the code still runs within time and writes a valid `submission.csv`.'
- What this solution (achieved 1.23566) has done: 'Your current score (1.23566, lower is better) is still far from the negative target, so we should improve accuracy while keeping the same “groupby mean with backoff” core logic. The smallest impactful change is to make the distance conditioning less sparse by using a coarser distance bin and a per-`type,pair` quantile binning (computed from train) so test rows land in well-populated bins more often. We keep your exact backoff chain (`type+pair+bin` → `type+pair` → `type+atom0/atom1` → `type` → global) and only swap the top-level bin definition to reduce NaNs and noise. We also add a tiny amount of smoothing at the top level by falling back to `type+pair` whenever the `type+pair+bin` group has too few samples (still the same conditional-mean approach, just a safer lookup).'
- What this solution (achieved 1.23566) has done: 'I fix the runtime error caused by joining a dataframe that already contains `molecule_name` (Pandas now errors on overlapping columns without suffixes). This is score-neutral and keeps your current “groupby-mean with backoff” logic identical; it simply passes the correct three columns into `compute_bond_distance_for_df`. I also make the `bond_dist` assignment preserve the index directly (no unnecessary `.join()` and no `.to_numpy()` roundtrips), improving stability without changing predictions. The script then run end-to-end and write a valid `submission.csv` with the correct columns and row order.'
- What this solution (achieved 1.23566) has done: 'Your current score (1.23566, lower is better) is still far from the negative target, so we should improve accuracy while keeping the exact same “groupby mean with backoff” core logic. The most direct, minimal gain is to add *lightweight smoothing* (shrinking noisy small groups toward a parent mean) instead of hard cutoffs, which reduces variance and typically lowers MAE/log-MAE without changing the overall approach. Concretely, we keep your same hierarchy/keys, but replace the “use mean only if count ≥ MIN_COUNT_TOP else NaN” with a smoothed mean computed from (group_sum, group_count) blended with a parent mean using a small prior weight. This is a small code change, keeps evaluation semantics identical (still conditional means with backoff), and should move the score downward toward the target while remaining robust and within time.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

DATA_DIR = "/kaggle/data/champs-scalar-coupling"

print("Listing /kaggle/data:")
print(os.listdir("/kaggle/data")[:20])

print(f"\nListing {DATA_DIR}:")
print(os.listdir(DATA_DIR)[:20])



## === cell 1
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
structures_path = os.path.join(DATA_DIR, "structures.csv")

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
    test_path, usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"]
)
sample = pd.read_csv(sample_path, usecols=["id"])

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

train = train.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
train = train.merge(s1, on=["molecule_name", "atom_index_1"], how="left")
test = test.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
test = test.merge(s1, on=["molecule_name", "atom_index_1"], how="left")

train["atom_0"] = train["atom_0"].fillna("UNK")
train["atom_1"] = train["atom_1"].fillna("UNK")
test["atom_0"] = test["atom_0"].fillna("UNK")
test["atom_1"] = test["atom_1"].fillna("UNK")

global_mean = float(train["scalar_coupling_constant"].mean())
mean_type = train.groupby("type")["scalar_coupling_constant"].mean()

train_pair_key = np.where(
    train["atom_0"] <= train["atom_1"],
    train["atom_0"].astype(str) + "_" + train["atom_1"].astype(str),
    train["atom_1"].astype(str) + "_" + train["atom_0"].astype(str),
)
test_pair_key = np.where(
    test["atom_0"] <= test["atom_1"],
    test["atom_0"].astype(str) + "_" + test["atom_1"].astype(str),
    test["atom_1"].astype(str) + "_" + test["atom_0"].astype(str),
)
train = train.assign(pair=train_pair_key)
test = test.assign(pair=test_pair_key)

train = train.assign(
    pair_ord=train["atom_0"].astype(str) + "_" + train["atom_1"].astype(str)
)
test = test.assign(
    pair_ord=test["atom_0"].astype(str) + "_" + test["atom_1"].astype(str)
)


def add_distance(df: pd.DataFrame) -> pd.DataFrame:
    dx = (df["x0"] - df["x1"]).astype(np.float32)
    dy = (df["y0"] - df["y1"]).astype(np.float32)
    dz = (df["z0"] - df["z1"]).astype(np.float32)
    dist = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)
    return df.assign(dist=dist)


train = add_distance(train)
test = add_distance(test)

BOND_CUTOFF = 1.75  # broad, type-agnostic; robust enough for CHONF typical bonds


def compute_bond_distance_for_df(
    df: pd.DataFrame, structures_df: pd.DataFrame, cutoff: float
) -> pd.Series:
    mols = df["molecule_name"].unique()
    s = structures_df[structures_df["molecule_name"].isin(mols)].copy()
    s["atom_index"] = s["atom_index"].astype(np.int32)

    a = s.rename(columns={"atom_index": "i", "x": "xi", "y": "yi", "z": "zi"})[
        ["molecule_name", "i", "xi", "yi", "zi"]
    ]
    b = s.rename(columns={"atom_index": "j", "x": "xj", "y": "yj", "z": "zj"})[
        ["molecule_name", "j", "xj", "yj", "zj"]
    ]
    ab = a.merge(b, on="molecule_name", how="inner")
    ab = ab[ab["i"] < ab["j"]]

    dx = ab["xi"].to_numpy(np.float32) - ab["xj"].to_numpy(np.float32)
    dy = ab["yi"].to_numpy(np.float32) - ab["yj"].to_numpy(np.float32)
    dz = ab["zi"].to_numpy(np.float32) - ab["zj"].to_numpy(np.float32)
    dist = np.sqrt(dx * dx + dy * dy + dz * dz)
    mask = dist <= np.float32(cutoff)
    ab = ab.loc[mask, ["molecule_name", "i", "j"]]

    adj = {}
    for mol, grp in ab.groupby("molecule_name", sort=False):
        edges = grp[["i", "j"]].to_numpy(np.int32, copy=False)
        g = {}
        for ii, jj in edges:
            g.setdefault(ii, []).append(jj)
            g.setdefault(jj, []).append(ii)
        adj[mol] = g

    from collections import deque

    cache = {}  # (mol, src) -> dict distances
    out = np.full(
        len(df), 99, dtype=np.int16
    )  # 99 = "far/unreachable" bucket (safe fallback conditioning)
    mol_arr = df["molecule_name"].to_numpy()
    a0 = df["atom_index_0"].to_numpy(np.int32, copy=False)
    a1 = df["atom_index_1"].to_numpy(np.int32, copy=False)

    for k in range(len(df)):
        mol = mol_arr[k]
        src = int(a0[k])
        tgt = int(a1[k])
        if mol not in adj:
            continue
        key = (mol, src)
        if key not in cache:
            g = adj[mol]
            dist_map = {src: 0}
            q = deque([src])
            while q:
                u = q.popleft()
                du = dist_map[u]
                for v in g.get(u, ()):
                    if v not in dist_map:
                        dist_map[v] = du + 1
                        q.append(v)
            cache[key] = dist_map
        dmap = cache[key]
        if tgt in dmap:
            out[k] = np.int16(dmap[tgt])
    return pd.Series(out, index=df.index, name="bond_dist")


train["bond_dist"] = compute_bond_distance_for_df(
    train[["molecule_name", "atom_index_0", "atom_index_1"]], structures, BOND_CUTOFF
)
test["bond_dist"] = compute_bond_distance_for_df(
    test[["molecule_name", "atom_index_0", "atom_index_1"]], structures, BOND_CUTOFF
)

N_QBINS = 10
MIN_SAMPLES_FOR_QBINS = 80
FALLBACK_BIN_WIDTH = 0.10
FALLBACK_MAX_DIST = 5.0


def fixed_bins(dist_arr: np.ndarray, bin_width: float, max_dist: float) -> np.ndarray:
    dist_arr = dist_arr.astype(np.float32, copy=False)
    dist_clipped = np.minimum(dist_arr, np.float32(max_dist))
    b = np.floor(dist_clipped / np.float32(bin_width)).astype(np.int16)
    b = np.where(np.isnan(dist_arr), np.int16(-1), b).astype(np.int16)
    return b


train["dist_bin_fixed"] = fixed_bins(
    train["dist"].to_numpy(), FALLBACK_BIN_WIDTH, FALLBACK_MAX_DIST
)
test["dist_bin_fixed"] = fixed_bins(
    test["dist"].to_numpy(), FALLBACK_BIN_WIDTH, FALLBACK_MAX_DIST
)

grp_sizes = train.groupby(["type", "pair"]).size()
eligible = grp_sizes[grp_sizes >= MIN_SAMPLES_FOR_QBINS].index

train_elig = train.set_index(["type", "pair"]).loc[eligible].reset_index()

q_levels = np.linspace(0.0, 1.0, N_QBINS + 1)[1:-1]
q_edges = (
    train_elig.groupby(["type", "pair"])["dist"].quantile(q_levels).unstack(level=-1)
)


def apply_quantile_bins(df: pd.DataFrame, q_edges_table: pd.DataFrame) -> np.ndarray:
    out = df["dist_bin_fixed"].to_numpy().astype(np.int16)

    key = pd.MultiIndex.from_arrays([df["type"].to_numpy(), df["pair"].to_numpy()])
    mask = key.isin(q_edges_table.index)
    if not mask.any():
        return out

    sub = df.loc[mask, ["type", "pair", "dist"]].copy()
    sub_key = pd.MultiIndex.from_frame(sub[["type", "pair"]])
    edges = q_edges_table.reindex(sub_key).to_numpy(dtype=np.float32)

    d = sub["dist"].to_numpy(dtype=np.float32)
    edges_valid = np.where(np.isnan(edges), np.inf, edges)
    qb = (d[:, None] > edges_valid).sum(axis=1).astype(np.int16)

    out[mask] = qb
    return out


train["dist_bin"] = apply_quantile_bins(train, q_edges)
test["dist_bin"] = apply_quantile_bins(test, q_edges)

y = train["scalar_coupling_constant"]
sum_top_ord = train.groupby(["type", "pair_ord", "bond_dist", "dist_bin"])[
    "scalar_coupling_constant"
].sum()
cnt_top_ord = train.groupby(["type", "pair_ord", "bond_dist", "dist_bin"])[
    "scalar_coupling_constant"
].size()

sum_mid_ord = train.groupby(["type", "pair_ord", "bond_dist"])[
    "scalar_coupling_constant"
].sum()
cnt_mid_ord = train.groupby(["type", "pair_ord", "bond_dist"])[
    "scalar_coupling_constant"
].size()

sum_type_pair_bond_distbin = train.groupby(["type", "pair", "bond_dist", "dist_bin"])[
    "scalar_coupling_constant"
].sum()
cnt_type_pair_bond_distbin = train.groupby(["type", "pair", "bond_dist", "dist_bin"])[
    "scalar_coupling_constant"
].size()

sum_type_pair_distbin = train.groupby(["type", "pair", "dist_bin"])[
    "scalar_coupling_constant"
].sum()
cnt_type_pair_distbin = train.groupby(["type", "pair", "dist_bin"])[
    "scalar_coupling_constant"
].size()

sum_type_pair = train.groupby(["type", "pair"])["scalar_coupling_constant"].sum()
cnt_type_pair = train.groupby(["type", "pair"])["scalar_coupling_constant"].size()

mean_type_atom0 = train.groupby(["type", "atom_0"])["scalar_coupling_constant"].mean()
mean_type_atom1 = train.groupby(["type", "atom_1"])["scalar_coupling_constant"].mean()

PRIOR_TOP = 30.0
PRIOR_MID = 20.0
PRIOR_TPBD = 30.0
PRIOR_TPD = 20.0
PRIOR_TP = 10.0

pred = np.full(len(test), np.nan, dtype=np.float64)

idx_top = pd.MultiIndex.from_arrays(
    [
        test["type"].values,
        test["pair_ord"].values,
        test["bond_dist"].values,
        test["dist_bin"].values,
    ]
)
top_sum = sum_top_ord.reindex(idx_top).to_numpy()
top_cnt = cnt_top_ord.reindex(idx_top).to_numpy()

idx_mid_for_top = pd.MultiIndex.from_arrays(
    [test["type"].values, test["pair_ord"].values, test["bond_dist"].values]
)
mid_mean_for_top = (
    sum_mid_ord.reindex(idx_mid_for_top) / cnt_mid_ord.reindex(idx_mid_for_top)
).to_numpy()

top_mean = top_sum / top_cnt
top_smooth = (top_sum + PRIOR_TOP * mid_mean_for_top) / (top_cnt + PRIOR_TOP)
p = np.where(~np.isnan(top_mean), top_smooth, np.nan)
pred = np.where(np.isnan(pred), p, pred)

mask = np.isnan(pred)
if mask.any():
    idx_mid = pd.MultiIndex.from_arrays(
        [
            test.loc[mask, "type"].values,
            test.loc[mask, "pair_ord"].values,
            test.loc[mask, "bond_dist"].values,
        ]
    )
    mid_sum = sum_mid_ord.reindex(idx_mid).to_numpy()
    mid_cnt = cnt_mid_ord.reindex(idx_mid).to_numpy()

    idx_tp_parent = pd.MultiIndex.from_arrays(
        [test.loc[mask, "type"].values, test.loc[mask, "pair"].values]
    )
    tp_mean_parent = (
        sum_type_pair.reindex(idx_tp_parent) / cnt_type_pair.reindex(idx_tp_parent)
    ).to_numpy()

    mid_mean = mid_sum / mid_cnt
    mid_smooth = (mid_sum + PRIOR_MID * tp_mean_parent) / (mid_cnt + PRIOR_MID)
    pred[mask] = np.where(~np.isnan(mid_mean), mid_smooth, np.nan)

mask = np.isnan(pred)
if mask.any():
    idx3 = pd.MultiIndex.from_arrays(
        [
            test.loc[mask, "type"].values,
            test.loc[mask, "pair"].values,
            test.loc[mask, "bond_dist"].values,
            test.loc[mask, "dist_bin"].values,
        ]
    )
    s3 = sum_type_pair_bond_distbin.reindex(idx3).to_numpy()
    c3 = cnt_type_pair_bond_distbin.reindex(idx3).to_numpy()

    idx_tp = pd.MultiIndex.from_arrays(
        [test.loc[mask, "type"].values, test.loc[mask, "pair"].values]
    )
    tp_mean = (sum_type_pair.reindex(idx_tp) / cnt_type_pair.reindex(idx_tp)).to_numpy()

    m3 = s3 / c3
    p3 = (s3 + PRIOR_TPBD * tp_mean) / (c3 + PRIOR_TPBD)
    pred[mask] = np.where(~np.isnan(m3), p3, np.nan)

mask = np.isnan(pred)
if mask.any():
    idx4 = pd.MultiIndex.from_arrays(
        [
            test.loc[mask, "type"].values,
            test.loc[mask, "pair"].values,
            test.loc[mask, "dist_bin"].values,
        ]
    )
    s4 = sum_type_pair_distbin.reindex(idx4).to_numpy()
    c4 = cnt_type_pair_distbin.reindex(idx4).to_numpy()

    idx_tp = pd.MultiIndex.from_arrays(
        [test.loc[mask, "type"].values, test.loc[mask, "pair"].values]
    )
    tp_mean = (sum_type_pair.reindex(idx_tp) / cnt_type_pair.reindex(idx_tp)).to_numpy()

    m4 = s4 / c4
    p4 = (s4 + PRIOR_TPD * tp_mean) / (c4 + PRIOR_TPD)
    pred[mask] = np.where(~np.isnan(m4), p4, np.nan)

mask = np.isnan(pred)
if mask.any():
    idx5 = pd.MultiIndex.from_arrays(
        [test.loc[mask, "type"].values, test.loc[mask, "pair"].values]
    )
    s5 = sum_type_pair.reindex(idx5).to_numpy()
    c5 = cnt_type_pair.reindex(idx5).to_numpy()
    type_mean_parent = test.loc[mask, "type"].map(mean_type).to_numpy(dtype=np.float64)

    m5 = s5 / c5
    p5 = (s5 + PRIOR_TP * type_mean_parent) / (c5 + PRIOR_TP)
    pred[mask] = np.where(~np.isnan(m5), p5, np.nan)

mask = np.isnan(pred)
if mask.any():
    idx_ta0 = pd.MultiIndex.from_arrays(
        [test.loc[mask, "type"].values, test.loc[mask, "atom_0"].values]
    )
    idx_ta1 = pd.MultiIndex.from_arrays(
        [test.loc[mask, "type"].values, test.loc[mask, "atom_1"].values]
    )

    p0 = mean_type_atom0.reindex(idx_ta0).to_numpy()
    p1 = mean_type_atom1.reindex(idx_ta1).to_numpy()

    p_atom = np.where(
        np.isnan(p0) & np.isnan(p1),
        np.nan,
        np.where(np.isnan(p0), p1, np.where(np.isnan(p1), p0, 0.5 * (p0 + p1))),
    )
    pred[mask] = p_atom

mask = np.isnan(pred)
if mask.any():
    pred[mask] = test.loc[mask, "type"].map(mean_type).to_numpy()

mask = np.isnan(pred)
if mask.any():
    pred[mask] = global_mean

pred = pred.astype(np.float32)

submission = pd.DataFrame({"id": test["id"].values, "scalar_coupling_constant": pred})

if len(submission) == len(sample) and not submission["id"].equals(sample["id"]):
    submission = submission.set_index("id").loc[sample["id"]].reset_index()

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print(submission.head())
print(
    f"\nWrote {out_path} with shape {submission.shape} and columns {list(submission.columns)}"
)
print("id unique:", submission["id"].is_unique)
print("Any NaNs:", submission["scalar_coupling_constant"].isna().any())
print("dist_bin stats (test):", pd.Series(test["dist_bin"]).describe())
print("bond_dist stats (test):", pd.Series(test["bond_dist"]).describe())
