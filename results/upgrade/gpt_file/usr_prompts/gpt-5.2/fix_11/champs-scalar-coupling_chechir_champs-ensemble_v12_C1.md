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

-2.357440763184845

# 6. Current score

1.18497

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.18497) has done: 'The notebook fails because it references external Kaggle datasets (`../input/champ-preds`, `../input/nnet-try*`) that are not present in your provided environment, so the prediction columns never get created and downstream cells crash. I keep the same ensemble structure but make it robust: automatically fall back to a simple, valid baseline prediction when those files are missing. I also fix pathing to match your available data (`/kaggle/data/champs-scalar-coupling/...`), ensure predictions align by `id` (avoids silent row-order mismatches), and always write a valid `sub_ensemble.csv` with the required columns.'
- What this solution (achieved 1.18497) has done: 'Your current score is far worse than the target (lower-is-better), and it’s likely because all the external prediction files are missing so you’re effectively submitting a very weak “type median” baseline. To move the score toward the target without changing the core “ensemble from precomputed preds” logic, I keep the same fallback mechanism but make the fallback substantially stronger by engineering a few standard CHAMPS geometric features from `structures.csv` (atom-wise coordinates) and then predicting `scalar_coupling_constant` by (type, atom0, atom1, distance-bin) medians. This stays in the same spirit as your current approach (no new model training loops/architectures), but uses legitimate structure information available in your environment. I also keep strict `id` alignment and still write `sub_ensemble.csv` in the correct format.'
- What this solution (achieved 1.18497) has done: 'Your current score (1.18497, lower-is-better) is far from the target (-2.3574), so we should improve the fallback baseline (which is what’s effectively being used because the external prediction files are missing). I keep the same overall ensemble structure and final weighted blend, but strengthen the baseline by (1) adding molecule-level features from `dipole_moments.csv` and `potential_energy.csv`, and (2) improving geometric binning by using a log-distance bin (more aligned with coupling decay) in addition to the existing distance bin. Predictions are still pure train-set medians with a conservative backoff chain (no new model training loops/architectures), and everything stays aligned by `id` and always writes `sub_ensemble.csv`.'
- What this solution (achieved 1.18497) has done: 'I fix the merge KeyError by making the environment-aggregation merge output keep the original `atom_index_0/atom_index_1` columns (your helper currently drops/renames them away, so the subsequent merge can’t find the keys). That single bug prevents the baseline from being built, which then cascades into missing `n1/n2/...` columns and later `final_preds`/submission failures; fixing it makes the pipeline run end-to-end and always produce `sub_ensemble.csv`. I also add a small safety fallback so that if any unexpected exception happens during baseline construction, the code still fills prediction columns with a valid type-median baseline and writes a proper submission (score-neutral vs crashing). No model/ensemble weights or core approach is changed beyond correcting the broken feature-merge logic.'
- What this solution (achieved 1.18497) has done: 'Your current score is far worse than the target (lower-is-better), and the biggest limiter is that the fallback baseline is being built from a loop over molecules that computes full pairwise distance matrices, which is too slow/fragile and can silently degrade coverage/quality. I keep the same ensemble logic and the same “train-median with backoff” prediction approach, but make the fallback baseline both faster and more faithful by (1) computing per-atom environment features via a global spatial hash (grid) instead of O(n²) per molecule, and (2) adding a minimal, standard CHAMPS “neighbor count within radii” feature set (two radii) that improves type-wise calibration without changing the modeling family. I also make the binning deterministic and consistent between train/test by using edges learned on train (already done for dist/logdist) and extend that to the new neighbor-count features. This should legitimately improve the baseline (hence the blended submission) while staying within your core “no new model training loop/architecture” constraint and still always writing `sub_ensemble.csv`.'
- What this solution (achieved 1.18497) has done: 'Your current score is far worse than the target (lower is better), and the main reason is that all ensemble inputs are missing so the submission is effectively the fallback baseline. To move the score toward the target with minimal, core-logic-preserving changes, I keep the same “median/backoff baseline from train+structures” approach but add two very standard CHAMPS physics-driven features: (1) inverse-distance terms (1/r, 1/r², 1/r³) binned per type, and (2) atom-pair ordering normalization so (atom_0, atom_1) and (atom_1, atom_0) share statistics. I also fix a small binning bug where the env “r2/r3” columns referenced didn’t match the generated column names (“r2.0/r3.0”), which was silently weakening the primary grouping. These changes only affect the fallback baseline construction (used when external files are absent) and keep the rest of your ensemble + submission logic identical.'
- What this solution (achieved 1.18497) has done: 'Your current score (1.18497, lower-is-better) is far from the target (-2.3574), so we should improve the fallback baseline (which is effectively what you submit when the external prediction files are missing) with minimal, core-logic-preserving changes. I keep the same overall ensemble and “median/backoff from train statistics” approach, but add a single strong CHAMPS feature: the 3D distance between the coupled atoms (and its log/inverse variants) computed from `structures.csv`. I then switch the primary grouping to include `type + atom_0 + atom_1 + dist_bin` (and keep your existing backoff chain), which is a small change but typically a large quality jump versus type-only medians. I also ensure we use a deterministic, train-derived bin edge mapping for both train and test to avoid mis-binning drift.'
- What this solution (achieved 1.18497) has done: 'Your current score (1.18497, lower-is-better) is far from the target (-2.3574), and the biggest bottleneck is that all external ensemble inputs are missing so you effectively submit the fallback baseline. Keeping the same “no training loop / median-with-backoff” core approach, I strengthen that baseline with two minimal but high-impact additions: (1) enforce atom-pair order normalization so (atom_0, atom_1) and (atom_1, atom_0) share statistics, and (2) add a very small, standard CHAMPS context signal by counting heavy-atom neighbors (non-H) within 2.0Å and 3.0Å around each coupled atom, then include binned neighbor-counts in the primary median group. This stays purely as train-statistics aggregation (no model fitting), is deterministic, and should reduce MAE materially versus distance-only binning while keeping the rest of your blending/submission logic unchanged. I also keep the safety fallback and strict `id` alignment, and still always write `sub_ensemble.csv`.'

# 9. Code solution

## === cell 0
import os
import glob
import pandas as pd
import numpy as np

INPUT_BASE_CANDIDATES = [
    "../input/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling",
    "/kaggle/input/champs-scalar-coupling",
]


def resolve_input_base():
    for p in INPUT_BASE_CANDIDATES:
        if os.path.exists(p) and os.path.exists(os.path.join(p, "test.csv")):
            return p
    for root in ["/kaggle", "../input", "../"]:
        matches = glob.glob(os.path.join(root, "**", "test.csv"), recursive=True)
        for m in matches:
            base = os.path.dirname(m)
            if os.path.exists(os.path.join(base, "sample_submission.csv")):
                return base
    raise FileNotFoundError(
        "Could not locate champs-scalar-coupling dataset folder containing test.csv"
    )


DATA_DIR = resolve_input_base()
print("Using DATA_DIR:", DATA_DIR)




## === cell 1
def read_pred_series(
    path, target_col="scalar_coupling_constant", id_col="id", ref_ids=None
):
    """
    Read a prediction file and return a Series aligned to ref_ids (if provided).
    Supports either:
      - a column named target_col, optionally with id_col
      - or first numeric column besides id_col
    """
    df = pd.read_csv(path)
    if target_col not in df.columns:
        cand_cols = [c for c in df.columns if c != id_col]
        if len(cand_cols) == 0:
            raise ValueError(
                f"No prediction column found in {path}. Columns={df.columns.tolist()}"
            )
        pred_col = cand_cols[0]
    else:
        pred_col = target_col

    if id_col in df.columns and ref_ids is not None:
        s = df.set_index(id_col)[pred_col]
        return s.reindex(ref_ids).astype(float)
    else:
        s = df[pred_col].astype(float)
        if ref_ids is not None and len(s) != len(ref_ids):
            raise ValueError(
                f"Length mismatch for {path}: {len(s)} vs test {len(ref_ids)}"
            )
        s.index = ref_ids if ref_ids is not None else s.index
        return s


def get_one_df(files, ref_ids):
    """Median ensemble across multiple files (aligned by id when possible)."""
    outs = []
    for f in files:
        outs.append(read_pred_series(f, ref_ids=ref_ids))
    concat_sub = pd.concat(outs, axis=1)
    champ_median = concat_sub.median(axis=1).values
    return champ_median




## === cell 2
test = pd.read_csv(
    os.path.join(DATA_DIR, "test.csv"),
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
    },
)
TARGET = "scalar_coupling_constant"

test = test.sort_values("id").reset_index(drop=True)

external_sources = {
    "n1": "../input/champ-preds/gnn_median_2255.csv",
    "n2": "../input/champ-preds/chain_sep_2069.csv",
    "lgb_a": [
        "../input/champ-preds/submission_type_2085.csv",
        "../input/champ-preds/submission_type_2082.csv",
    ],
    "lgb_m": [
        "../input/champ-preds/lgb_type_full_f286_10.csv",
        "../input/champ-preds/lgb_type_full_f262_10.csv",
    ],
    "nnet": [
        "../input/nnet-try/lgb_type_cv-2.108373877033157_mae0.12143527465528912_bags-1_f120_fd5_10.csv",
        "../input/nnet-try-seed-11/lgb_type_cv-1.64944_mae0.23965_bags-1_f120_fd5_11.csv",
        "../input/nnet-try-seed-12/lgb_type_cv-1.64875_mae0.2412_bags-1_f120_fd5_12.csv",
    ],
}

ref_ids = test["id"].values

missing_any = False

if isinstance(external_sources["n1"], str) and os.path.exists(external_sources["n1"]):
    test["n1"] = read_pred_series(
        external_sources["n1"], target_col=TARGET, ref_ids=ref_ids
    ).values
else:
    missing_any = True

if isinstance(external_sources["n2"], str) and os.path.exists(external_sources["n2"]):
    test["n2"] = read_pred_series(
        external_sources["n2"], target_col=TARGET, ref_ids=ref_ids
    ).values
else:
    missing_any = True

if all(os.path.exists(p) for p in external_sources["lgb_a"]):
    test["lgb_a"] = get_one_df(external_sources["lgb_a"], ref_ids=ref_ids)
else:
    missing_any = True

if all(os.path.exists(p) for p in external_sources["lgb_m"]):
    test["lgb_m"] = get_one_df(external_sources["lgb_m"], ref_ids=ref_ids)
else:
    missing_any = True

if all(os.path.exists(p) for p in external_sources["nnet"]):
    test["nnet"] = get_one_df(external_sources["nnet"], ref_ids=ref_ids)
else:
    missing_any = True

if missing_any:
    print(
        "One or more external prediction files are missing. Building a stronger baseline from train.csv + structures.csv "
        "(pair-normalized + distance-binned medians + tiny neighbor-count context)..."
    )

    try:
        train = pd.read_csv(
            os.path.join(DATA_DIR, "train.csv"),
            usecols=["molecule_name", "atom_index_0", "atom_index_1", "type", TARGET],
            dtype={
                "molecule_name": "category",
                "atom_index_0": np.int16,
                "atom_index_1": np.int16,
                "type": "category",
                TARGET: np.float32,
            },
        )

        structs = pd.read_csv(
            os.path.join(DATA_DIR, "structures.csv"),
            usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
            dtype={
                "molecule_name": "category",
                "atom_index": np.int16,
                "atom": "category",
                "x": np.float32,
                "y": np.float32,
                "z": np.float32,
            },
        )

        s0 = structs.rename(
            columns={
                "atom_index": "atom_index_0",
                "atom": "atom_0",
                "x": "x0",
                "y": "y0",
                "z": "z0",
            }
        )
        s1 = structs.rename(
            columns={
                "atom_index": "atom_index_1",
                "atom": "atom_1",
                "x": "x1",
                "y": "y1",
                "z": "z1",
            }
        )

        tr = train.merge(
            s0[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]],
            on=["molecule_name", "atom_index_0"],
            how="left",
            copy=False,
        ).merge(
            s1[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]],
            on=["molecule_name", "atom_index_1"],
            how="left",
            copy=False,
        )

        a0 = tr["atom_0"].astype(str)
        a1 = tr["atom_1"].astype(str)
        swap = a0.values > a1.values
        if swap.any():
            tr.loc[swap, ["atom_0", "atom_1"]] = tr.loc[
                swap, ["atom_1", "atom_0"]
            ].values
            tr.loc[swap, ["atom_index_0", "atom_index_1"]] = tr.loc[
                swap, ["atom_index_1", "atom_index_0"]
            ].values
            tr.loc[swap, ["x0", "y0", "z0", "x1", "y1", "z1"]] = tr.loc[
                swap, ["x1", "y1", "z1", "x0", "y0", "z0"]
            ].values

        dx = tr["x0"].values - tr["x1"].values
        dy = tr["y0"].values - tr["y1"].values
        dz = tr["z0"].values - tr["z1"].values
        tr["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)
        tr["log_dist"] = np.log1p(tr["dist"].astype(np.float64)).astype(np.float32)

        eps = np.float32(1e-6)
        inv = (np.float32(1.0) / (tr["dist"].values + eps)).astype(np.float32)
        tr["inv_dist2"] = (inv * inv).astype(np.float32)

        def build_edges_by_type(frame, col, n_bins):
            q = np.linspace(0, 1, n_bins + 1)
            q_edges = (
                frame.groupby("type", observed=True)[col]
                .quantile(q)
                .reset_index()
                .rename(columns={"level_1": "q", col: "edge"})
            )
            edges_by_type = {}
            for t, g in q_edges.groupby("type", sort=False, observed=True):
                edges = np.unique(g["edge"].values.astype(np.float64))
                if len(edges) < 3:
                    c = frame[col].astype(np.float64)
                    edges = np.array([c.min(), c.median(), c.max()], dtype=np.float64)
                edges_by_type[str(t)] = edges
            return edges_by_type

        def assign_bin_index_per_row(types, vals, edges_by_type):
            types_s = pd.Series(types).astype(str)
            vals = np.asarray(vals, dtype=np.float64)
            out = np.full(vals.shape[0], -1, dtype=np.int16)
            for t, edges in edges_by_type.items():
                m = (types_s.values == t) & np.isfinite(vals)
                if not m.any():
                    continue
                v = vals[m]
                idx = np.searchsorted(edges, v, side="right") - 1
                idx = np.clip(idx, 0, len(edges) - 2)
                out[m] = idx.astype(np.int16)
            return out

        edges_dist = build_edges_by_type(tr, "dist", 30)
        edges_logdist = build_edges_by_type(tr, "log_dist", 30)
        edges_inv2 = build_edges_by_type(tr, "inv_dist2", 30)

        tr["dist_bin"] = assign_bin_index_per_row(
            tr["type"].values, tr["dist"].values, edges_dist
        )
        tr["logdist_bin"] = assign_bin_index_per_row(
            tr["type"].values, tr["log_dist"].values, edges_logdist
        )
        tr["inv2_bin"] = assign_bin_index_per_row(
            tr["type"].values, tr["inv_dist2"].values, edges_inv2
        )

        def add_heavy_neighbor_counts(pairs_df, structs_df, radii=(2.0, 3.0)):
            mols = pd.Index(pairs_df["molecule_name"].unique())
            st = structs_df[structs_df["molecule_name"].isin(mols)].copy()
            st["atom_is_heavy"] = (st["atom"].astype(str) != "H").astype(np.int8)

            out_cols0 = [f"nH{r}_0" for r in radii]
            out_cols1 = [f"nH{r}_1" for r in radii]
            for c in out_cols0 + out_cols1:
                pairs_df[c] = np.int16(-1)

            counts = {}
            r2 = [float(r) * float(r) for r in radii]

            for mol, g in st.groupby("molecule_name", sort=False, observed=True):
                coords = g[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=True)
                atom_idx = g["atom_index"].to_numpy(dtype=np.int16, copy=True)
                heavy = g["atom_is_heavy"].to_numpy(dtype=np.int8, copy=True)

                diff = coords[:, None, :] - coords[None, :, :]
                d2 = np.sum(diff * diff, axis=2)

                for i, ai in enumerate(atom_idx):
                    mask_other = np.ones(d2.shape[0], dtype=bool)
                    mask_other[i] = False
                    for rr, rr2 in zip(radii, r2):
                        cnt = int(
                            np.sum(
                                (d2[i, mask_other] <= rr2) & (heavy[mask_other] == 1)
                            )
                        )
                        counts[(mol, int(ai), rr)] = cnt

            mol_vals = pairs_df["molecule_name"].astype(str).values
            a0_vals = pairs_df["atom_index_0"].values
            a1_vals = pairs_df["atom_index_1"].values
            for rr in radii:
                c0 = f"nH{rr}_0"
                c1 = f"nH{rr}_1"
                v0 = np.empty(len(pairs_df), dtype=np.int16)
                v1 = np.empty(len(pairs_df), dtype=np.int16)
                for i in range(len(pairs_df)):
                    mol = mol_vals[i]
                    v0[i] = np.int16(counts.get((mol, int(a0_vals[i]), rr), -1))
                    v1[i] = np.int16(counts.get((mol, int(a1_vals[i]), rr), -1))
                pairs_df[c0] = v0
                pairs_df[c1] = v1
            return pairs_df

        tr = add_heavy_neighbor_counts(tr, structs, radii=(2.0, 3.0))

        def build_int_edges_by_type(frame, col, n_bins):
            q = np.linspace(0, 1, n_bins + 1)
            q_edges = (
                frame.groupby("type", observed=True)[col]
                .quantile(q)
                .reset_index()
                .rename(columns={"level_1": "q", col: "edge"})
            )
            edges_by_type = {}
            for t, g in q_edges.groupby("type", sort=False, observed=True):
                edges = np.unique(g["edge"].values.astype(np.float64))
                if len(edges) < 3:
                    c = frame[col].astype(np.float64)
                    edges = np.array([c.min(), c.median(), c.max()], dtype=np.float64)
                edges_by_type[str(t)] = edges
            return edges_by_type

        edges_nH2_0 = build_int_edges_by_type(tr, "nH2.0_0", 10)
        edges_nH2_1 = build_int_edges_by_type(tr, "nH2.0_1", 10)
        edges_nH3_0 = build_int_edges_by_type(tr, "nH3.0_0", 10)
        edges_nH3_1 = build_int_edges_by_type(tr, "nH3.0_1", 10)

        tr["nH2_0_bin"] = assign_bin_index_per_row(
            tr["type"].values, tr["nH2.0_0"].values, edges_nH2_0
        )
        tr["nH2_1_bin"] = assign_bin_index_per_row(
            tr["type"].values, tr["nH2.0_1"].values, edges_nH2_1
        )
        tr["nH3_0_bin"] = assign_bin_index_per_row(
            tr["type"].values, tr["nH3.0_0"].values, edges_nH3_0
        )
        tr["nH3_1_bin"] = assign_bin_index_per_row(
            tr["type"].values, tr["nH3.0_1"].values, edges_nH3_1
        )

        grp_cols_primary = [
            "type",
            "atom_0",
            "atom_1",
            "dist_bin",
            "nH2_0_bin",
            "nH2_1_bin",
        ]
        med_primary = tr.groupby(grp_cols_primary, observed=True)[TARGET].median()

        med_pair = tr.groupby(["type", "atom_0", "atom_1"], observed=True)[
            TARGET
        ].median()
        med_pair_nH3 = tr.groupby(
            ["type", "atom_0", "atom_1", "nH3_0_bin", "nH3_1_bin"], observed=True
        )[TARGET].median()
        med_t_inv2bin = tr.groupby(["type", "inv2_bin"], observed=True)[TARGET].median()
        med_t_logbin = tr.groupby(["type", "logdist_bin"], observed=True)[
            TARGET
        ].median()
        med_t_distbin = tr.groupby(["type", "dist_bin"], observed=True)[TARGET].median()
        med_type = tr.groupby(["type"], observed=True)[TARGET].median()
        global_med = float(tr[TARGET].median())

        te = test.merge(
            s0[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]],
            on=["molecule_name", "atom_index_0"],
            how="left",
            copy=False,
        ).merge(
            s1[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]],
            on=["molecule_name", "atom_index_1"],
            how="left",
            copy=False,
        )

        a0 = te["atom_0"].astype(str)
        a1 = te["atom_1"].astype(str)
        swap = a0.values > a1.values
        if swap.any():
            te.loc[swap, ["atom_0", "atom_1"]] = te.loc[
                swap, ["atom_1", "atom_0"]
            ].values
            te.loc[swap, ["atom_index_0", "atom_index_1"]] = te.loc[
                swap, ["atom_index_1", "atom_index_0"]
            ].values
            te.loc[swap, ["x0", "y0", "z0", "x1", "y1", "z1"]] = te.loc[
                swap, ["x1", "y1", "z1", "x0", "y0", "z0"]
            ].values

        dx = te["x0"].values - te["x1"].values
        dy = te["y0"].values - te["y1"].values
        dz = te["z0"].values - te["z1"].values
        te["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)
        te["log_dist"] = np.log1p(te["dist"].astype(np.float64)).astype(np.float32)
        inv = (np.float32(1.0) / (te["dist"].values + eps)).astype(np.float32)
        te["inv_dist2"] = (inv * inv).astype(np.float32)

        te["dist_bin"] = assign_bin_index_per_row(
            te["type"].values, te["dist"].values, edges_dist
        )
        te["logdist_bin"] = assign_bin_index_per_row(
            te["type"].values, te["log_dist"].values, edges_logdist
        )
        te["inv2_bin"] = assign_bin_index_per_row(
            te["type"].values, te["inv_dist2"].values, edges_inv2
        )

        te = add_heavy_neighbor_counts(te, structs, radii=(2.0, 3.0))
        te["nH2_0_bin"] = assign_bin_index_per_row(
            te["type"].values, te["nH2.0_0"].values, edges_nH2_0
        )
        te["nH2_1_bin"] = assign_bin_index_per_row(
            te["type"].values, te["nH2.0_1"].values, edges_nH2_1
        )
        te["nH3_0_bin"] = assign_bin_index_per_row(
            te["type"].values, te["nH3.0_0"].values, edges_nH3_0
        )
        te["nH3_1_bin"] = assign_bin_index_per_row(
            te["type"].values, te["nH3.0_1"].values, edges_nH3_1
        )

        te_key = te[
            grp_cols_primary + ["logdist_bin", "inv2_bin", "nH3_0_bin", "nH3_1_bin"]
        ].copy()

        te_key["pred"] = (
            te_key.set_index(grp_cols_primary).index.map(med_primary).astype("float64")
        )

        mask = te_key["pred"].isna()
        if mask.any():
            idx = pd.MultiIndex.from_frame(
                te.loc[mask, ["type", "atom_0", "atom_1", "nH3_0_bin", "nH3_1_bin"]]
            )
            te_key.loc[mask, "pred"] = idx.map(med_pair_nH3).astype("float64")

        mask = te_key["pred"].isna()
        if mask.any():
            idx = pd.MultiIndex.from_frame(te.loc[mask, ["type", "atom_0", "atom_1"]])
            te_key.loc[mask, "pred"] = idx.map(med_pair).astype("float64")

        mask = te_key["pred"].isna()
        if mask.any():
            idx = pd.MultiIndex.from_frame(te.loc[mask, ["type", "inv2_bin"]])
            te_key.loc[mask, "pred"] = idx.map(med_t_inv2bin).astype("float64")

        mask = te_key["pred"].isna()
        if mask.any():
            idx = pd.MultiIndex.from_frame(te.loc[mask, ["type", "logdist_bin"]])
            te_key.loc[mask, "pred"] = idx.map(med_t_logbin).astype("float64")

        mask = te_key["pred"].isna()
        if mask.any():
            idx = pd.MultiIndex.from_frame(te.loc[mask, ["type", "dist_bin"]])
            te_key.loc[mask, "pred"] = idx.map(med_t_distbin).astype("float64")

        mask = te_key["pred"].isna()
        if mask.any():
            te_key.loc[mask, "pred"] = (
                te.loc[mask, "type"].map(med_type).astype("float64")
            )

        te_key["pred"] = te_key["pred"].fillna(global_med).astype(np.float32)
        baseline = te_key["pred"].values

    except Exception as e:
        print("Baseline construction failed with exception:", repr(e))
        print("Falling back to simple per-type median baseline.")
        train_small = pd.read_csv(
            os.path.join(DATA_DIR, "train.csv"),
            usecols=["type", TARGET],
            dtype={"type": "category", TARGET: np.float32},
        )
        med_type = train_small.groupby("type", observed=True)[TARGET].median()
        global_med = float(train_small[TARGET].median())
        baseline = (
            test["type"].map(med_type).astype(np.float32).fillna(global_med).values
        )

    for col in ["n1", "n2", "lgb_a", "lgb_m", "nnet"]:
        if col not in test.columns:
            test[col] = baseline

test.head(10)



## === cell 3
test["final_preds"] = (
    test["n1"] * 0.65
    + test["n2"] * 0.05
    + test["lgb_a"] * 0.15
    + test["lgb_m"] * 0.10
    + test["nnet"] * 0.05
)

assert test["final_preds"].isna().sum() == 0, "NaNs in final predictions"
test.head(20)



## === cell 4
submission = pd.DataFrame(
    {
        "id": test["id"].values,
        "scalar_coupling_constant": test["final_preds"].astype(float).values,
    }
)

submission = submission.sort_values("id").reset_index(drop=True)

out_path = "sub_ensemble.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape=", submission.shape)
print(submission.head())



## === cell 5
submission.head(20)
