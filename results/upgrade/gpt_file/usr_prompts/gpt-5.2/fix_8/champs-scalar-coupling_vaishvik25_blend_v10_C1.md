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

-1.3300563636911908

# 6. Current score

1.62315

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.18497) has done: 'Your notebook is trying to blend several external submission files that don’t exist in this Kaggle environment, which causes the FileNotFoundError and prevents any submission from being written. I keep the blending core idea, but make it robust: it attempt to load those files if present, otherwise fall back to a simple, valid baseline built from the provided competition data. To nudge score toward your target (and avoid a terrible constant baseline), the fallback train a lightweight per-`type` median model on `train.csv` and apply it to `test.csv`, which is score-improving while staying minimal and fast. Finally, it always write a valid `submission1236.csv` with the required columns and row alignment.'
- What this solution (achieved 1.18497) has done: 'Your current score (1.18497, lower-is-better) is far worse than the target (-1.3301), so we should improve the fallback baseline while keeping the same “no real model, just statistics from train applied to test” core logic. The biggest minimal gain here is to use molecule-level grouping to create simple geometry-based features (distance between the two atoms) from `structures.csv`, then still predict per-`type` but conditioned on distance bins (a stronger version of your per-`type` median). This stays within the same training approach (groupby → median lookup) and runs fast enough by reading only needed columns and using merges. The blending path remains unchanged: if external submissions exist, it blend them; otherwise it build this improved baseline and always write `submission1236.csv`.'
- What this solution (achieved 1.18497) has done: 'Your current score (1.18497, lower-is-better) is far worse than the target (-1.3301), so we should improve predictions while keeping your same “lookup statistics from train and apply to test” core logic. The biggest low-risk gain is to make the fallback median table more informative by conditioning not only on `(type, distance_bin)` but also on the actual atom elements involved (`atom_0`, `atom_1`), which strongly shifts coupling scales and reduces MAE. This is still the same approach (groupby medians with fallbacks), just with a slightly richer key, and it stays fast by merging only the needed `atom` and coordinates from `structures.csv`. The blending path remains unchanged; we only upgrade the fallback so the notebook always writes a valid `submission1236.csv`.'
- What this solution (achieved 1.63558) has done: 'Your current score (1.18497, lower-is-better) is far from the target (-1.33), so we should materially improve the fallback baseline while preserving the same “groupby medians lookup with fallbacks” core logic. The biggest low-risk gain is to build the distance bins *per coupling type* (each type lives on different bond-length regimes) and to use a **mean** aggregation (L1-optimal median is good, but the competition scores log(MAE) averaged by type, and in practice this baseline often improves with mean when conditioning is already strong). We also avoid slow Python key loops by using merges for the lookup tables (same semantics, faster, less missing-key noise). The blending path stays intact; if external submissions exist, it blend, otherwise it produce the improved baseline and always write `submission1236.csv`.'
- What this solution (achieved 1.62315) has done: 'I fix the crash in the fallback feature builder: `np.where(...).astype("category")` fails because NumPy can’t cast to pandas `category` dtype. The minimal, score-neutral fix is to create `atom_min/atom_max` as pandas Series and then cast using pandas’ `.astype("category")`, while also handling missing merges safely. I also make sure `sub1` is always defined (blend or fallback path) so the submission cell always runs and writes `submission1236.csv` with correct columns and row alignment. No changes are made to the core “groupby statistic lookup with fallbacks” logic beyond this dtype/robustness fix.'
- What this solution (achieved 1.62315) has done: 'Your current score (1.62315, lower-is-better) is still far worse than the target (-1.3301), so we should improve the fallback baseline while keeping the same “groupby statistic lookup with fallbacks” core logic. The most direct minimal gain is to add a couple of very cheap geometry features derived from the same two atoms (inverse distance and a short polynomial of distance), then condition the same lookup-table medians on an additional “feature-bin” to reduce residual variation within each (type, atoms, distance) bucket. This preserves your existing approach (no ML training loop; just merges + groupby aggregations) and remains fast because it only adds a few columns and one extra groupby/merge. We also keep your blending path untouched and still always write `submission1236.csv` with correct `id` alignment via `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os

BASE_INPUT = "/kaggle/data/champs-scalar-coupling"
ALT_INPUT = "/kaggle/input/champs-scalar-coupling"

if os.path.exists(BASE_INPUT):
    INPUT_DIR = BASE_INPUT
elif os.path.exists(ALT_INPUT):
    INPUT_DIR = ALT_INPUT
else:
    INPUT_DIR = "../input/champs-scalar-coupling"

INPUT_DIR



## === cell 1
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

for p in ["../input", "/kaggle/input", "/kaggle/data"]:
    if os.path.exists(p):
        print(p, "->", os.listdir(p)[:20])

print("Using INPUT_DIR =", INPUT_DIR)
print("INPUT_DIR exists:", os.path.exists(INPUT_DIR))
print(
    "INPUT_DIR files sample:",
    os.listdir(INPUT_DIR)[:20] if os.path.exists(INPUT_DIR) else "N/A",
)




## === cell 2
def safe_read_submission(path):
    """Return (df, err). df is None if read fails."""
    try:
        df = pd.read_csv(path)
        if not {"id", "scalar_coupling_constant"}.issubset(df.columns):
            return (
                None,
                f"Missing required columns in {path}. Found: {list(df.columns)}",
            )
        return df[["id", "scalar_coupling_constant"]].copy(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


blend_paths = [
    "../input/blender/LGB_2019-07-11_-1.4378.csv",
    "../input/blender/submission-2.csv",
    "../input/blender/stack_minmax_median.csv",
    "../input/blender2/submission.csv",
    "../input/blender2/submission-giba-1.csv",
    "../input/blender/workingsubmission-test.csv",
]

subs = []
errors = []
for p in blend_paths:
    df, err = safe_read_submission(p)
    if df is not None:
        subs.append(df)
    else:
        errors.append((p, err))

print(f"Found {len(subs)} blendable submission files.")
if errors:
    print(
        "Missing/unreadable blend inputs (expected in original notebook, not required here):"
    )
    for p, err in errors:
        print(" -", p, "->", err)

if len(subs) >= 2:
    base = subs[0].set_index("id")
    aligned = [base]
    for s in subs[1:]:
        aligned.append(s.set_index("id").reindex(base.index))

    for i in range(len(aligned)):
        if aligned[i]["scalar_coupling_constant"].isna().any():
            aligned[i]["scalar_coupling_constant"] = aligned[i][
                "scalar_coupling_constant"
            ].fillna(aligned[0]["scalar_coupling_constant"])

    weights = [0.25, 0.25, 0.10, 0.20, 0.15, 0.05]
    weights = weights[: len(aligned)]
    weights = np.array(weights, dtype=np.float64)
    weights = weights / weights.sum()

    blended = np.zeros(len(base), dtype=np.float64)
    for w, a in zip(weights, aligned):
        blended += w * a["scalar_coupling_constant"].to_numpy(dtype=np.float64)

    sub1 = pd.DataFrame({"id": base.index.values, "scalar_coupling_constant": blended})
    print(sub1["scalar_coupling_constant"].describe())
else:
    train_path = os.path.join(INPUT_DIR, "train.csv")
    test_path = os.path.join(INPUT_DIR, "test.csv")
    sample_path = os.path.join(INPUT_DIR, "sample_submission.csv")
    structures_path = os.path.join(INPUT_DIR, "structures.csv")

    train = pd.read_csv(
        train_path,
        usecols=[
            "molecule_name",
            "atom_index_0",
            "atom_index_1",
            "type",
            "scalar_coupling_constant",
        ],
        dtype={
            "molecule_name": "category",
            "atom_index_0": np.int32,
            "atom_index_1": np.int32,
            "type": "category",
            "scalar_coupling_constant": np.float32,
        },
    )
    test = pd.read_csv(
        test_path,
        usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
        dtype={
            "id": np.int32,
            "molecule_name": "category",
            "atom_index_0": np.int32,
            "atom_index_1": np.int32,
            "type": "category",
        },
    )
    sample = pd.read_csv(sample_path, usecols=["id"], dtype={"id": np.int32})

    structures = pd.read_csv(
        structures_path,
        usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
        dtype={
            "molecule_name": "category",
            "atom_index": np.int32,
            "atom": "category",
            "x": np.float32,
            "y": np.float32,
            "z": np.float32,
        },
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

    def add_atom_and_distance(df):
        df = df.merge(
            s0[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]],
            on=["molecule_name", "atom_index_0"],
            how="left",
        )
        df = df.merge(
            s1[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]],
            on=["molecule_name", "atom_index_1"],
            how="left",
        )

        dx = (df["x0"] - df["x1"]).astype(np.float32)
        dy = (df["y0"] - df["y1"]).astype(np.float32)
        dz = (df["z0"] - df["z1"]).astype(np.float32)
        dist2 = (dx * dx + dy * dy + dz * dz).astype(np.float32)
        df["dist2"] = dist2
        df["dist"] = np.sqrt(dist2).astype(np.float32)

        eps = np.float32(1e-6)
        d = df["dist"].astype(np.float32)
        df["inv_dist"] = (np.float32(1.0) / (d + eps)).astype(np.float32)
        df["dist3"] = (d * d * d).astype(np.float32)

        a0 = df["atom_0"].astype("string")
        a1 = df["atom_1"].astype("string")
        cond = (a0 <= a1) & a0.notna() & a1.notna()
        atom_min = pd.Series(pd.NA, index=df.index, dtype="string")
        atom_max = pd.Series(pd.NA, index=df.index, dtype="string")
        atom_min.loc[cond] = a0.loc[cond]
        atom_max.loc[cond] = a1.loc[cond]
        atom_min.loc[~cond] = a1.loc[~cond]
        atom_max.loc[~cond] = a0.loc[~cond]
        df["atom_min"] = atom_min.astype("category")
        df["atom_max"] = atom_max.astype("category")
        return df

    train_f = add_atom_and_distance(train)
    test_f = add_atom_and_distance(test)

    agg_func = "median"
    n_bins = 60  # keep moderate to reduce sparsity / missing-key fallbacks

    train_f["dist_bin"] = np.int16(-1)
    test_f["dist_bin"] = np.int16(-1)
    train_f["dist2_bin"] = np.int16(-1)
    test_f["dist2_bin"] = np.int16(-1)

    train_f["inv_bin"] = np.int16(-1)
    test_f["inv_bin"] = np.int16(-1)

    for t in train_f["type"].cat.categories:
        tr_mask = (train_f["type"] == t).to_numpy()
        te_mask = (test_f["type"] == t).to_numpy()
        if not tr_mask.any():
            continue

        d_tr = train_f.loc[tr_mask, "dist"].to_numpy(dtype=np.float32)
        d_te = (
            test_f.loc[te_mask, "dist"].to_numpy(dtype=np.float32)
            if te_mask.any()
            else np.array([], dtype=np.float32)
        )
        d_all = np.concatenate([d_tr[np.isfinite(d_tr)], d_te[np.isfinite(d_te)]])
        if d_all.size < 200:
            train_f.loc[tr_mask, "dist_bin"] = np.int16(0)
            if te_mask.any():
                test_f.loc[te_mask, "dist_bin"] = np.int16(0)
        else:
            qs = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)
            edges = np.quantile(d_all, qs)
            edges = np.unique(edges)
            if edges.size < 3:
                train_f.loc[tr_mask, "dist_bin"] = np.int16(0)
                if te_mask.any():
                    test_f.loc[te_mask, "dist_bin"] = np.int16(0)
            else:
                train_f.loc[tr_mask, "dist_bin"] = np.digitize(
                    d_tr, edges[1:-1], right=False
                ).astype(np.int16)
                if te_mask.any():
                    test_f.loc[te_mask, "dist_bin"] = np.digitize(
                        d_te, edges[1:-1], right=False
                    ).astype(np.int16)

        d2_tr = train_f.loc[tr_mask, "dist2"].to_numpy(dtype=np.float32)
        d2_te = (
            test_f.loc[te_mask, "dist2"].to_numpy(dtype=np.float32)
            if te_mask.any()
            else np.array([], dtype=np.float32)
        )
        d2_all = np.concatenate([d2_tr[np.isfinite(d2_tr)], d2_te[np.isfinite(d2_te)]])
        if d2_all.size < 200:
            train_f.loc[tr_mask, "dist2_bin"] = np.int16(0)
            if te_mask.any():
                test_f.loc[te_mask, "dist2_bin"] = np.int16(0)
        else:
            qs2 = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)
            edges2 = np.quantile(d2_all, qs2)
            edges2 = np.unique(edges2)
            if edges2.size < 3:
                train_f.loc[tr_mask, "dist2_bin"] = np.int16(0)
                if te_mask.any():
                    test_f.loc[te_mask, "dist2_bin"] = np.int16(0)
            else:
                train_f.loc[tr_mask, "dist2_bin"] = np.digitize(
                    d2_tr, edges2[1:-1], right=False
                ).astype(np.int16)
                if te_mask.any():
                    test_f.loc[te_mask, "dist2_bin"] = np.digitize(
                        d2_te, edges2[1:-1], right=False
                    ).astype(np.int16)

        inv_tr = train_f.loc[tr_mask, "inv_dist"].to_numpy(dtype=np.float32)
        inv_te = (
            test_f.loc[te_mask, "inv_dist"].to_numpy(dtype=np.float32)
            if te_mask.any()
            else np.array([], dtype=np.float32)
        )
        inv_all = np.concatenate(
            [inv_tr[np.isfinite(inv_tr)], inv_te[np.isfinite(inv_te)]]
        )
        if inv_all.size < 200:
            train_f.loc[tr_mask, "inv_bin"] = np.int16(0)
            if te_mask.any():
                test_f.loc[te_mask, "inv_bin"] = np.int16(0)
        else:
            qs3 = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)
            edges3 = np.quantile(inv_all, qs3)
            edges3 = np.unique(edges3)
            if edges3.size < 3:
                train_f.loc[tr_mask, "inv_bin"] = np.int16(0)
                if te_mask.any():
                    test_f.loc[te_mask, "inv_bin"] = np.int16(0)
            else:
                train_f.loc[tr_mask, "inv_bin"] = np.digitize(
                    inv_tr, edges3[1:-1], right=False
                ).astype(np.int16)
                if te_mask.any():
                    test_f.loc[te_mask, "inv_bin"] = np.digitize(
                        inv_te, edges3[1:-1], right=False
                    ).astype(np.int16)

    global_center = float(train_f["scalar_coupling_constant"].median())
    type_center = train_f.groupby("type", observed=True)[
        "scalar_coupling_constant"
    ].agg(agg_func)

    tad2i_center = (
        train_f.groupby(
            ["type", "atom_min", "atom_max", "dist_bin", "dist2_bin", "inv_bin"],
            observed=True,
        )["scalar_coupling_constant"]
        .agg(agg_func)
        .reset_index()
        .rename(columns={"scalar_coupling_constant": "pred"})
    )
    td2i_center = (
        train_f.groupby(["type", "dist_bin", "dist2_bin", "inv_bin"], observed=True)[
            "scalar_coupling_constant"
        ]
        .agg(agg_func)
        .reset_index()
        .rename(columns={"scalar_coupling_constant": "pred_td"})
    )
    tdi_center = (
        train_f.groupby(["type", "dist_bin", "inv_bin"], observed=True)[
            "scalar_coupling_constant"
        ]
        .agg(agg_func)
        .reset_index()
        .rename(columns={"scalar_coupling_constant": "pred_t"})
    )

    td_center = (
        train_f.groupby(["type", "dist_bin"], observed=True)["scalar_coupling_constant"]
        .agg(agg_func)
        .reset_index()
        .rename(columns={"scalar_coupling_constant": "pred_t2"})
    )

    pred = (
        test_f.merge(
            tad2i_center,
            on=["type", "atom_min", "atom_max", "dist_bin", "dist2_bin", "inv_bin"],
            how="left",
        )
        .merge(td2i_center, on=["type", "dist_bin", "dist2_bin", "inv_bin"], how="left")
        .merge(tdi_center, on=["type", "dist_bin", "inv_bin"], how="left")
        .merge(td_center, on=["type", "dist_bin"], how="left")
    )

    pred["scalar_coupling_constant"] = pred["pred"]
    pred["scalar_coupling_constant"] = pred["scalar_coupling_constant"].fillna(
        pred["pred_td"]
    )
    pred["scalar_coupling_constant"] = pred["scalar_coupling_constant"].fillna(
        pred["pred_t"]
    )
    pred["scalar_coupling_constant"] = pred["scalar_coupling_constant"].fillna(
        pred["pred_t2"]
    )
    pred["scalar_coupling_constant"] = pred["scalar_coupling_constant"].fillna(
        pred["type"].map(type_center).astype(np.float64)
    )
    pred["scalar_coupling_constant"] = (
        pred["scalar_coupling_constant"].fillna(global_center).astype(np.float64)
    )

    sub1 = pred[["id", "scalar_coupling_constant"]].copy()

    sub1 = sample.merge(sub1, on="id", how="left")
    sub1["scalar_coupling_constant"] = (
        sub1["scalar_coupling_constant"].fillna(global_center).astype(np.float64)
    )

    print(
        "Fallback baseline built (per-type quantile bins on dist, dist^2, inv_dist; per-(type, unordered atoms, dist_bin, dist2_bin, inv_bin) median with fallbacks)."
    )
    print(sub1["scalar_coupling_constant"].describe())



## === cell 3
sub1.to_csv("submission1236.csv", index=False)
print("Wrote submission1236.csv with shape:", sub1.shape)
print(sub1.head())



## === cell 4
try:
    ax = sub1["scalar_coupling_constant"].plot(
        kind="hist", bins=100, title="scalar_coupling_constant histogram"
    )
    fig = ax.get_figure()
    fig.show()
except Exception as e:
    print("Plot skipped due to:", type(e).__name__, str(e))
