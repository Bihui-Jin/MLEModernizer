# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.18497) has done: 'I remove the dependency on missing external blend files (`../input/blend/...` and `../input/statstack/...`) that cause the `FileNotFoundError`, and instead generate a valid baseline prediction directly from the provided competition data. To keep the core approach “simple blending/statistical” and score-improving versus all-zeros, I compute per-`type` median `scalar_coupling_constant` from `train.csv` and use that to predict `test.csv`, with a global median fallback for any unseen types. This run end-to-end in the given environment and write a correctly formatted `.csv` submission file.'
- What this solution (achieved 1.18497) has done: 'Your current baseline uses only per-`type` medians, which ignores key signal that varies strongly by atom pair geometry and chemistry; to move the score down toward the target, we add a small set of well-known CHAMPS features while keeping the same “simple statistical model” core logic. Specifically, we merge `structures.csv` to get both atoms’ coordinates and element types, compute the inter-atomic distance and a few lightweight derived features, then predict using per-(`type`, `atom_0`, `atom_1`) medians with sensible fallbacks to per-(`type`, binned distance) medians and finally to your existing per-`type` median. This preserves evaluation semantics (still a deterministic median-based estimator) but typically improves MAE a lot versus type-only. The script still runs end-to-end and writes a valid `submission.csv` with the required columns and row alignment.'
- What this solution (achieved 1.18497) has done: 'Your current median-based estimator is already valid but it “leaks” some avoidable error because it doesn’t canonicalize atom order consistently and uses a very fine distance bin (0.05Å) that sparsifies medians and pushes many test rows to weaker fallbacks. I keep the same core logic (deterministic medians with fallbacks) but (1) compute only canonicalized (`atom_a`,`atom_b`) medians (so train/test keying matches regardless of atom order), (2) use slightly wider distance bins (0.10Å) to reduce sparsity and improve the mid-level fallback, and (3) add an additional, still-statistical fallback using per-(`type`,`atom_a`,`atom_b`,`dist_bin`) medians before backing off to per-(`type`,`atom_a`,`atom_b`) and then per-`type`. These are minimal changes that usually reduce MAE materially (lower score is better) without changing the overall approach or adding any learning model. The script still runs end-to-end and writes `submission.csv` in the correct format.'
- What this solution (achieved 1.18497) has done: 'Your current score is far from the (much better) target, so we should improve accuracy while keeping your median/fallback core intact. The biggest gap in your current approach is that atom-pair order isn’t canonicalized consistently for geometric features: you canonicalize (`atom_a`,`atom_b`) but you still bin distance from the original (`atom_index_0`,`atom_index_1`) direction, which can mismatch when medians are learned on one ordering and applied on another. I minimally fix this by also canonicalizing the geometry used for distance/atom labels so that the same physical pair always maps to the same keys, then rebuild the same median tables and fallbacks. This preserves your deterministic “statistical median with fallbacks” logic but reduces key sparsity/mismatches and should move the log-MAE downward toward the target.'
- What this solution (achieved 1.18497) has done: 'Your current median-with-fallbacks logic is sound but it’s leaving easy error on the table because (a) coupling strength depends strongly on pair distance in a smooth way and (b) your current 0.10Å binning can still be sparse for the highest-specificity key. I keep the exact same deterministic “groupby-median then fallback” approach, but add a second, coarser distance bin (0.20Å) as an intermediate fallback to reduce sparsity before backing off to broader keys. I also add one more still-statistical table keyed by (`type`, `atom_a`, `atom_b`) with a *trimmed* median (remove extreme 1% tails within each group) to reduce outlier impact without changing the model class. These are minimal changes intended to reduce MAE (lower is better) and move your score downward toward the target.'
- What this solution (achieved 1.18497) has done: 'Your current score is much worse than the target (lower is better), so we should improve accuracy while keeping your deterministic median-with-fallbacks approach intact. The biggest low-risk gain is to add a simple, still-statistical correction that captures the strong per-molecule shifts in coupling constants: incorporate `potential_energy` and (optionally) `dipole_moments` as binned features in the median tables. We compute per-molecule features once, merge them into train/test, create coarse bins, and add two new intermediate fallbacks keyed by (`type`,`atom_a`,`atom_b`,`dist_bin`,`pe_bin`) and (`type`,`dist_bin`,`pe_bin`) before your existing fallbacks; this reduces error without changing the model class. All paths remain the same and the script still writes a valid `submission.csv`.'
- What this solution (achieved 1.18497) has done: 'Your current score (1.18497, lower is better) is far from the target (-1.3684), so we should improve accuracy while keeping your deterministic “median tables + fallbacks” approach intact. The biggest low-risk gain within the same core logic is to use the provided per-atom `mulliken_charges` and `magnetic_shielding_tensors` by merging each atom’s values into train/test and then adding a couple of extra high-signal median tables keyed by (`type`, `atom_a`, `atom_b`, `dist_bin`, per-atom bins) with sensible fallbacks. This preserves the existing feature extraction style (simple merges + binning) and prediction semantics (groupby median + hierarchical fallback), but should materially reduce MAE. I also keep your existing tables/fallback chain unchanged and only insert the new tables near the top so they help when available without breaking anything.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

INPUT_ROOT = "/kaggle/data/champs-scalar-coupling"

print("Listing /kaggle/data:", os.listdir("/kaggle/data")[:20])
print("Using INPUT_ROOT:", INPUT_ROOT)
print("Files in INPUT_ROOT:", sorted(os.listdir(INPUT_ROOT))[:20])



## === cell 1
train_path = os.path.join(INPUT_ROOT, "train.csv")
test_path = os.path.join(INPUT_ROOT, "test.csv")
sample_path = os.path.join(INPUT_ROOT, "sample_submission.csv")
structures_path = os.path.join(INPUT_ROOT, "structures.csv")
potential_energy_path = os.path.join(INPUT_ROOT, "potential_energy.csv")
dipole_path = os.path.join(INPUT_ROOT, "dipole_moments.csv")
mulliken_path = os.path.join(INPUT_ROOT, "mulliken_charges.csv")
mst_path = os.path.join(INPUT_ROOT, "magnetic_shielding_tensors.csv")

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
submission = pd.read_csv(sample_path)

assert "id" in submission.columns and "scalar_coupling_constant" in submission.columns
assert len(submission) == len(
    test
), "sample_submission and test must have the same number of rows"

structures = pd.read_csv(
    structures_path,
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
)

pe = pd.read_csv(potential_energy_path, usecols=["molecule_name", "potential_energy"])
dip = pd.read_csv(dipole_path, usecols=["molecule_name", "X", "Y", "Z"])
dip["dipole_norm"] = np.sqrt(
    dip["X"].to_numpy(dtype="float64") ** 2
    + dip["Y"].to_numpy(dtype="float64") ** 2
    + dip["Z"].to_numpy(dtype="float64") ** 2
)
mol_feat = pe.merge(
    dip[["molecule_name", "dipole_norm"]], on="molecule_name", how="left"
)

mull = pd.read_csv(
    mulliken_path, usecols=["molecule_name", "atom_index", "mulliken_charge"]
)
mull0 = mull.rename(
    columns={"atom_index": "atom_index_0", "mulliken_charge": "mulliken_0"}
)
mull1 = mull.rename(
    columns={"atom_index": "atom_index_1", "mulliken_charge": "mulliken_1"}
)

mst = pd.read_csv(mst_path, usecols=["molecule_name", "atom_index", "XX", "YY", "ZZ"])
mst["mst_trace"] = (
    mst["XX"].to_numpy(dtype="float64")
    + mst["YY"].to_numpy(dtype="float64")
    + mst["ZZ"].to_numpy(dtype="float64")
)
mst = mst[["molecule_name", "atom_index", "mst_trace"]]
mst0 = mst.rename(columns={"atom_index": "atom_index_0", "mst_trace": "mst_trace_0"})
mst1 = mst.rename(columns={"atom_index": "atom_index_1", "mst_trace": "mst_trace_1"})

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


def add_struct_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
    df = df.merge(s1, on=["molecule_name", "atom_index_1"], how="left")

    df = df.merge(mull0, on=["molecule_name", "atom_index_0"], how="left")
    df = df.merge(mull1, on=["molecule_name", "atom_index_1"], how="left")
    df = df.merge(mst0, on=["molecule_name", "atom_index_0"], how="left")
    df = df.merge(mst1, on=["molecule_name", "atom_index_1"], how="left")

    atom0 = df["atom_0"].astype("string")
    atom1 = df["atom_1"].astype("string")
    swap = atom0.to_numpy() > atom1.to_numpy()
    swap = pd.Series(swap, index=df.index)

    df["atom_index_a"] = df["atom_index_0"].where(~swap, df["atom_index_1"])
    df["atom_index_b"] = df["atom_index_1"].where(~swap, df["atom_index_0"])

    df["atom_a"] = df["atom_0"].where(~swap, df["atom_1"])
    df["atom_b"] = df["atom_1"].where(~swap, df["atom_0"])

    xa = df["x0"].where(~swap, df["x1"]).astype("float64")
    ya = df["y0"].where(~swap, df["y1"]).astype("float64")
    za = df["z0"].where(~swap, df["z1"]).astype("float64")

    xb = df["x1"].where(~swap, df["x0"]).astype("float64")
    yb = df["y1"].where(~swap, df["y0"]).astype("float64")
    zb = df["z1"].where(~swap, df["z0"]).astype("float64")

    dx = xa - xb
    dy = ya - yb
    dz = za - zb
    df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz)

    df["inv_dist"] = 1.0 / (df["dist"] + 1e-12)
    df["dist2"] = df["dist"] * df["dist"]

    df["dist_bin"] = np.floor((df["dist"] / 0.10)).astype("Int64")  # 0.10 Å bins
    df["dist_bin2"] = np.floor((df["dist"] / 0.20)).astype("Int64")  # 0.20 Å bins

    df["mulliken_a"] = df["mulliken_0"].where(~swap, df["mulliken_1"]).astype("float64")
    df["mulliken_b"] = df["mulliken_1"].where(~swap, df["mulliken_0"]).astype("float64")
    df["mst_a"] = df["mst_trace_0"].where(~swap, df["mst_trace_1"]).astype("float64")
    df["mst_b"] = df["mst_trace_1"].where(~swap, df["mst_trace_0"]).astype("float64")

    df["mulliken_sum"] = df["mulliken_a"] + df["mulliken_b"]
    df["mulliken_absdiff"] = (df["mulliken_a"] - df["mulliken_b"]).abs()
    df["mst_sum"] = df["mst_a"] + df["mst_b"]
    df["mst_absdiff"] = (df["mst_a"] - df["mst_b"]).abs()
    return df


train_f = add_struct_features(train)
test_f = add_struct_features(test)

train_f = train_f.merge(mol_feat, on="molecule_name", how="left")
test_f = test_f.merge(mol_feat, on="molecule_name", how="left")

pe_edges = np.quantile(
    train_f["potential_energy"].to_numpy(dtype="float64"), np.linspace(0, 1, 51)
)
pe_edges = np.unique(pe_edges)
if pe_edges.size < 3:
    pe_edges = np.array(
        [train_f["potential_energy"].min(), train_f["potential_energy"].max()],
        dtype="float64",
    )
train_f["pe_bin"] = pd.cut(
    train_f["potential_energy"], bins=pe_edges, include_lowest=True, labels=False
).astype("Int64")
test_f["pe_bin"] = pd.cut(
    test_f["potential_energy"], bins=pe_edges, include_lowest=True, labels=False
).astype("Int64")

dip_edges = np.quantile(
    train_f["dipole_norm"]
    .fillna(train_f["dipole_norm"].median())
    .to_numpy(dtype="float64"),
    np.linspace(0, 1, 31),
)
dip_edges = np.unique(dip_edges)
if dip_edges.size < 3:
    dip_edges = np.array(
        [train_f["dipole_norm"].min(), train_f["dipole_norm"].max()], dtype="float64"
    )
train_f["dip_bin"] = pd.cut(
    train_f["dipole_norm"], bins=dip_edges, include_lowest=True, labels=False
).astype("Int64")
test_f["dip_bin"] = pd.cut(
    test_f["dipole_norm"], bins=dip_edges, include_lowest=True, labels=False
).astype("Int64")


def make_edges_from_train(s: pd.Series, q: int) -> np.ndarray:
    a = s.to_numpy(dtype="float64")
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.array([-1.0, 1.0], dtype="float64")
    edges = np.quantile(a, np.linspace(0, 1, q + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        mn = float(np.min(a))
        mx = float(np.max(a))
        if mn == mx:
            edges = np.array([mn - 1e-6, mx + 1e-6], dtype="float64")
        else:
            edges = np.array([mn, mx], dtype="float64")
    return edges


mull_edges = make_edges_from_train(train_f["mulliken_sum"], q=20)
mst_edges = make_edges_from_train(train_f["mst_sum"], q=20)

train_f["mull_bin"] = pd.cut(
    train_f["mulliken_sum"], bins=mull_edges, include_lowest=True, labels=False
).astype("Int64")
test_f["mull_bin"] = pd.cut(
    test_f["mulliken_sum"], bins=mull_edges, include_lowest=True, labels=False
).astype("Int64")
train_f["mst_bin"] = pd.cut(
    train_f["mst_sum"], bins=mst_edges, include_lowest=True, labels=False
).astype("Int64")
test_f["mst_bin"] = pd.cut(
    test_f["mst_sum"], bins=mst_edges, include_lowest=True, labels=False
).astype("Int64")


def add_type_dist_quantile_bins(
    train_df: pd.DataFrame, test_df: pd.DataFrame, n_bins: int, col_name: str
) -> None:
    train_bins = pd.Series(
        pd.array([pd.NA] * len(train_df), dtype="Int64"), index=train_df.index
    )
    test_bins = pd.Series(
        pd.array([pd.NA] * len(test_df), dtype="Int64"), index=test_df.index
    )

    for t, gtr in train_df.groupby("type", sort=False):
        dtr = gtr["dist"].astype("float64")
        if dtr.notna().sum() < n_bins * 50 or dtr.nunique(dropna=True) < n_bins:
            edges = np.unique(
                np.quantile(
                    dtr.to_numpy(), np.linspace(0, 1, min(11, max(3, n_bins // 2 + 1)))
                )
            )
            if edges.size < 3:
                edges = np.array(
                    [float(dtr.min()) - 1e-6, float(dtr.max()) + 1e-6], dtype="float64"
                )
            train_bins.loc[gtr.index] = pd.cut(
                dtr, bins=edges, include_lowest=True, labels=False
            ).astype("Int64")
        else:
            train_bins.loc[gtr.index] = pd.qcut(
                dtr, q=n_bins, labels=False, duplicates="drop"
            ).astype("Int64")

        gte = test_df[test_df["type"] == t]
        if len(gte) > 0:
            dte = gte["dist"].astype("float64")
            edges = np.unique(
                np.quantile(dtr.to_numpy(), np.linspace(0, 1, n_bins + 1))
            )
            if edges.size < 3:
                edges = np.array(
                    [float(dtr.min()) - 1e-6, float(dtr.max()) + 1e-6], dtype="float64"
                )
            test_bins.loc[gte.index] = pd.cut(
                dte, bins=edges, include_lowest=True, labels=False
            ).astype("Int64")

    train_df[col_name] = train_bins
    test_df[col_name] = test_bins


add_type_dist_quantile_bins(train_f, test_f, n_bins=30, col_name="dist_qbin30")
add_type_dist_quantile_bins(train_f, test_f, n_bins=15, col_name="dist_qbin15")

type_median = train_f.groupby("type")["scalar_coupling_constant"].median()
global_median = float(train_f["scalar_coupling_constant"].median())



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3444911276.py in <cell line: 0>()
    139 
    140 train_f = add_struct_features(train)
--> 141 test_f = add_struct_features(test)
    142 
    143 train_f = train_f.merge(mol_feat, on="molecule_name", how="left")

/tmp/ipykernel_11/3444911276.py in add_struct_features(df)
     98     atom0 = df["atom_0"].astype("string")
     99     atom1 = df["atom_1"].astype("string")
--> 100     swap = atom0.to_numpy() > atom1.to_numpy()
    101     swap = pd.Series(swap, index=df.index)
    102 

missing.pyx in pandas._libs.missing.NAType.__bool__()

TypeError: boolean value of NA is ambiguous

## === cell 2
med_type_atoms_distbin_mull = (
    train_f.groupby(["type", "atom_a", "atom_b", "dist_bin", "mull_bin"])[
        "scalar_coupling_constant"
    ]
    .median()
    .rename("m")
)
med_type_atoms_distbin_mst = (
    train_f.groupby(["type", "atom_a", "atom_b", "dist_bin", "mst_bin"])[
        "scalar_coupling_constant"
    ]
    .median()
    .rename("m")
)

med_type_atoms_distbin = (
    train_f.groupby(["type", "atom_a", "atom_b", "dist_bin"])[
        "scalar_coupling_constant"
    ]
    .median()
    .rename("m")
)

med_type_atoms_distbin2 = (
    train_f.groupby(["type", "atom_a", "atom_b", "dist_bin2"])[
        "scalar_coupling_constant"
    ]
    .median()
    .rename("m")
)

med_type_atoms_sorted = (
    train_f.groupby(["type", "atom_a", "atom_b"])["scalar_coupling_constant"]
    .median()
    .rename("m")
)


def trimmed_median_1pct(x: pd.Series) -> float:
    a = x.to_numpy(dtype="float64", copy=False)
    if a.size == 0:
        return np.nan
    lo = np.quantile(a, 0.01)
    hi = np.quantile(a, 0.99)
    a2 = a[(a >= lo) & (a <= hi)]
    if a2.size == 0:
        return float(np.median(a))
    return float(np.median(a2))


med_type_atoms_sorted_trim = (
    train_f.groupby(["type", "atom_a", "atom_b"])["scalar_coupling_constant"]
    .apply(trimmed_median_1pct)
    .rename("m")
)

med_type_distbin = (
    train_f.groupby(["type", "dist_bin"])["scalar_coupling_constant"]
    .median()
    .rename("m")
)
med_type_distbin2 = (
    train_f.groupby(["type", "dist_bin2"])["scalar_coupling_constant"]
    .median()
    .rename("m")
)

med_type_atoms_distbin_pe = (
    train_f.groupby(["type", "atom_a", "atom_b", "dist_bin", "pe_bin"])[
        "scalar_coupling_constant"
    ]
    .median()
    .rename("m")
)
med_type_distbin_pe = (
    train_f.groupby(["type", "dist_bin", "pe_bin"])["scalar_coupling_constant"]
    .median()
    .rename("m")
)

med_type_atoms_distbin_dip = (
    train_f.groupby(["type", "atom_a", "atom_b", "dist_bin", "dip_bin"])[
        "scalar_coupling_constant"
    ]
    .median()
    .rename("m")
)
med_type_distbin_dip = (
    train_f.groupby(["type", "dist_bin", "dip_bin"])["scalar_coupling_constant"]
    .median()
    .rename("m")
)

med_type_atoms_qbin30 = (
    train_f.groupby(["type", "atom_a", "atom_b", "dist_qbin30"])[
        "scalar_coupling_constant"
    ]
    .median()
    .rename("m")
)
med_type_qbin30 = (
    train_f.groupby(["type", "dist_qbin30"])["scalar_coupling_constant"]
    .median()
    .rename("m")
)
med_type_atoms_qbin15 = (
    train_f.groupby(["type", "atom_a", "atom_b", "dist_qbin15"])[
        "scalar_coupling_constant"
    ]
    .median()
    .rename("m")
)
med_type_qbin15 = (
    train_f.groupby(["type", "dist_qbin15"])["scalar_coupling_constant"]
    .median()
    .rename("m")
)

pred = pd.Series(np.nan, index=test_f.index, dtype="float64")

k_mull = pd.MultiIndex.from_frame(
    test_f[["type", "atom_a", "atom_b", "dist_bin", "mull_bin"]]
)
pred_mull = pd.Series(
    med_type_atoms_distbin_mull.reindex(k_mull).to_numpy(),
    index=test_f.index,
    dtype="float64",
)
pred = pred.fillna(pred_mull)

k_mst = pd.MultiIndex.from_frame(
    test_f[["type", "atom_a", "atom_b", "dist_bin", "mst_bin"]]
)
pred_mst = pd.Series(
    med_type_atoms_distbin_mst.reindex(k_mst).to_numpy(),
    index=test_f.index,
    dtype="float64",
)
pred = pred.fillna(pred_mst)

k_pe = pd.MultiIndex.from_frame(
    test_f[["type", "atom_a", "atom_b", "dist_bin", "pe_bin"]]
)
pred_pe = pd.Series(
    med_type_atoms_distbin_pe.reindex(k_pe).to_numpy(),
    index=test_f.index,
    dtype="float64",
)
pred = pred.fillna(pred_pe)

k_dip = pd.MultiIndex.from_frame(
    test_f[["type", "atom_a", "atom_b", "dist_bin", "dip_bin"]]
)
pred_dip = pd.Series(
    med_type_atoms_distbin_dip.reindex(k_dip).to_numpy(),
    index=test_f.index,
    dtype="float64",
)
pred = pred.fillna(pred_dip)

kq0 = pd.MultiIndex.from_frame(test_f[["type", "atom_a", "atom_b", "dist_qbin30"]])
pred_q0 = pd.Series(
    med_type_atoms_qbin30.reindex(kq0).to_numpy(), index=test_f.index, dtype="float64"
)
pred = pred.fillna(pred_q0)

kq1 = pd.MultiIndex.from_frame(test_f[["type", "atom_a", "atom_b", "dist_qbin15"]])
pred_q1 = pd.Series(
    med_type_atoms_qbin15.reindex(kq1).to_numpy(), index=test_f.index, dtype="float64"
)
pred = pred.fillna(pred_q1)

k0 = pd.MultiIndex.from_frame(test_f[["type", "atom_a", "atom_b", "dist_bin"]])
pred0 = pd.Series(
    med_type_atoms_distbin.reindex(k0).to_numpy(), index=test_f.index, dtype="float64"
)
pred = pred.fillna(pred0)

k0b = pd.MultiIndex.from_frame(test_f[["type", "atom_a", "atom_b", "dist_bin2"]])
pred0b = pd.Series(
    med_type_atoms_distbin2.reindex(k0b).to_numpy(), index=test_f.index, dtype="float64"
)
pred = pred.fillna(pred0b)

k2 = pd.MultiIndex.from_frame(test_f[["type", "atom_a", "atom_b"]])
pred2t = pd.Series(
    med_type_atoms_sorted_trim.reindex(k2).to_numpy(),
    index=test_f.index,
    dtype="float64",
)
pred = pred.fillna(pred2t)

pred2 = pd.Series(
    med_type_atoms_sorted.reindex(k2).to_numpy(), index=test_f.index, dtype="float64"
)
pred = pred.fillna(pred2)

k3_pe = pd.MultiIndex.from_frame(test_f[["type", "dist_bin", "pe_bin"]])
pred3_pe = pd.Series(
    med_type_distbin_pe.reindex(k3_pe).to_numpy(), index=test_f.index, dtype="float64"
)
pred = pred.fillna(pred3_pe)

k3_dip = pd.MultiIndex.from_frame(test_f[["type", "dist_bin", "dip_bin"]])
pred3_dip = pd.Series(
    med_type_distbin_dip.reindex(k3_dip).to_numpy(), index=test_f.index, dtype="float64"
)
pred = pred.fillna(pred3_dip)

kq3 = pd.MultiIndex.from_frame(test_f[["type", "dist_qbin30"]])
pred_q3 = pd.Series(
    med_type_qbin30.reindex(kq3).to_numpy(), index=test_f.index, dtype="float64"
)
pred = pred.fillna(pred_q3)

kq4 = pd.MultiIndex.from_frame(test_f[["type", "dist_qbin15"]])
pred_q4 = pd.Series(
    med_type_qbin15.reindex(kq4).to_numpy(), index=test_f.index, dtype="float64"
)
pred = pred.fillna(pred_q4)

k3 = pd.MultiIndex.from_frame(test_f[["type", "dist_bin"]])
pred3 = pd.Series(
    med_type_distbin.reindex(k3).to_numpy(), index=test_f.index, dtype="float64"
)
pred = pred.fillna(pred3)

k3b = pd.MultiIndex.from_frame(test_f[["type", "dist_bin2"]])
pred3b = pd.Series(
    med_type_distbin2.reindex(k3b).to_numpy(), index=test_f.index, dtype="float64"
)
pred = pred.fillna(pred3b)

pred4 = test_f["type"].map(type_median).astype("float64")
pred = pred.fillna(pred4)

pred = pred.fillna(global_median)

assert pred.isna().sum() == 0



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3568581456.py in <cell line: 0>()
      1 med_type_atoms_distbin_mull = (
----> 2     train_f.groupby(["type", "atom_a", "atom_b", "dist_bin", "mull_bin"])[
      3         "scalar_coupling_constant"
      4     ]
      5     .median()

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in groupby(self, by, axis, level, as_index, sort, group_keys, observed, dropna)
   9181             raise TypeError("You have to supply one of 'by' and 'level'")
   9182 
-> 9183         return DataFrameGroupBy(
   9184             obj=self,
   9185             keys=by,

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in __init__(self, obj, keys, axis, level, grouper, exclusions, selection, as_index, sort, group_keys, observed, dropna)
   1327 
   1328         if grouper is None:
-> 1329             grouper, exclusions, obj = get_grouper(
   1330                 obj,
   1331                 keys,

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/grouper.py in get_grouper(obj, key, axis, level, sort, observed, validate, dropna)
   1041                 in_axis, level, gpr = False, gpr, None
   1042             else:
-> 1043                 raise KeyError(gpr)
   1044         elif isinstance(gpr, Grouper) and gpr.key is not None:
   1045             # Add key to exclusions

KeyError: 'mull_bin'

## === cell 3
submission = submission.copy()
submission["scalar_coupling_constant"] = pred.to_numpy()

assert len(submission) == len(test)
assert submission["scalar_coupling_constant"].isna().sum() == 0
assert submission["id"].nunique() == len(submission)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission.head())



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3533939811.py in <cell line: 0>()
      1 submission = submission.copy()
----> 2 submission["scalar_coupling_constant"] = pred.to_numpy()
      3 
      4 assert len(submission) == len(test)
      5 assert submission["scalar_coupling_constant"].isna().sum() == 0

NameError: name 'pred' is not defined

## === cell 4
submission["scalar_coupling_constant"].describe()
