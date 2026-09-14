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
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
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
        "One or more external prediction files are missing. Building a stronger baseline from train.csv + structures.csv (+ molecule features)..."
    )

    dip = pd.read_csv(os.path.join(DATA_DIR, "dipole_moments.csv"))
    pot = pd.read_csv(os.path.join(DATA_DIR, "potential_energy.csv"))
    mol_feat = dip.merge(pot, on="molecule_name", how="left")
    mol_feat["dip_norm"] = np.sqrt(
        mol_feat["X"].values ** 2
        + mol_feat["Y"].values ** 2
        + mol_feat["Z"].values ** 2
    ).astype(np.float32)

    mol_feat["dip_bin"] = pd.qcut(mol_feat["dip_norm"], q=10, duplicates="drop").astype(
        str
    )
    mol_feat["pe_bin"] = pd.qcut(
        mol_feat["potential_energy"], q=10, duplicates="drop"
    ).astype(str)
    mol_feat = mol_feat[["molecule_name", "dip_bin", "pe_bin"]]

    train = pd.read_csv(
        os.path.join(DATA_DIR, "train.csv"),
        usecols=["molecule_name", "atom_index_0", "atom_index_1", "type", TARGET],
    )

    structs = pd.read_csv(
        os.path.join(DATA_DIR, "structures.csv"),
        usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
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

    tr = (
        train.merge(
            s0[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]],
            on=["molecule_name", "atom_index_0"],
            how="left",
        )
        .merge(
            s1[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]],
            on=["molecule_name", "atom_index_1"],
            how="left",
        )
        .merge(mol_feat, on="molecule_name", how="left")
    )

    dx = tr["x0"].values - tr["x1"].values
    dy = tr["y0"].values - tr["y1"].values
    dz = tr["z0"].values - tr["z1"].values
    tr["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)

    tr["log_dist"] = np.log1p(tr["dist"].astype(np.float64)).astype(np.float32)

    tr["dist_bin"] = (
        tr.groupby("type")["dist"]
        .transform(lambda s: pd.qcut(s, q=20, duplicates="drop"))
        .astype(str)
    )
    tr["logdist_bin"] = (
        tr.groupby("type")["log_dist"]
        .transform(lambda s: pd.qcut(s, q=20, duplicates="drop"))
        .astype(str)
    )

    grp_cols_primary = ["type", "atom_0", "atom_1", "logdist_bin", "dip_bin", "pe_bin"]
    med_primary = tr.groupby(grp_cols_primary)[TARGET].median()

    med_pair_mol = tr.groupby(["type", "atom_0", "atom_1", "dip_bin", "pe_bin"])[
        TARGET
    ].median()
    med_pair = tr.groupby(["type", "atom_0", "atom_1"])[TARGET].median()
    med_t_logbin = tr.groupby(["type", "logdist_bin"])[TARGET].median()
    med_t_distbin = tr.groupby(["type", "dist_bin"])[TARGET].median()
    med_type = tr.groupby(["type"])[TARGET].median()
    global_med = float(tr[TARGET].median())

    te = (
        test.merge(
            s0[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]],
            on=["molecule_name", "atom_index_0"],
            how="left",
        )
        .merge(
            s1[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]],
            on=["molecule_name", "atom_index_1"],
            how="left",
        )
        .merge(mol_feat, on="molecule_name", how="left")
    )

    dx = te["x0"].values - te["x1"].values
    dy = te["y0"].values - te["y1"].values
    dz = te["z0"].values - te["z1"].values
    te["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)
    te["log_dist"] = np.log1p(te["dist"].astype(np.float64)).astype(np.float32)

    def build_edges_by_type(frame, col, n_bins):
        q_edges = (
            frame.groupby("type")[col]
            .quantile(np.linspace(0, 1, n_bins + 1))
            .reset_index()
            .rename(columns={"level_1": "q", col: "edge"})
        )
        edges_by_type = {}
        for t, g in q_edges.groupby("type"):
            edges = np.unique(g["edge"].values.astype(np.float64))
            if len(edges) < 3:
                c = frame[col]
                edges = np.array([c.min(), c.median(), c.max()], dtype=np.float64)
            edges_by_type[t] = edges
        return edges_by_type

    edges_dist = build_edges_by_type(tr, "dist", 20)
    edges_logdist = build_edges_by_type(tr, "log_dist", 20)

    def assign_bin_per_row(types, vals, edges_by_type):
        out = np.empty(len(vals), dtype=object)
        for i, (t, v) in enumerate(zip(types, vals)):
            edges = edges_by_type.get(t)
            if edges is None or not np.isfinite(v):
                out[i] = "nan"
                continue
            idx = np.searchsorted(edges, v, side="right") - 1
            if idx < 0:
                idx = 0
            if idx >= len(edges) - 1:
                idx = len(edges) - 2
            out[i] = f"({edges[idx]:.6g},{edges[idx+1]:.6g}]"
        return out

    te["dist_bin"] = assign_bin_per_row(
        te["type"].values, te["dist"].values, edges_dist
    )
    te["logdist_bin"] = assign_bin_per_row(
        te["type"].values, te["log_dist"].values, edges_logdist
    )

    te_key = te[
        ["type", "atom_0", "atom_1", "logdist_bin", "dip_bin", "pe_bin", "dist_bin"]
    ].copy()
    te_key["pred"] = (
        te_key.set_index(grp_cols_primary).index.map(med_primary).astype("float64")
    )

    mask = te_key["pred"].isna()
    if mask.any():
        idx = pd.MultiIndex.from_frame(
            te_key.loc[mask, ["type", "atom_0", "atom_1", "dip_bin", "pe_bin"]]
        )
        te_key.loc[mask, "pred"] = idx.map(med_pair_mol).astype("float64")

    mask = te_key["pred"].isna()
    if mask.any():
        idx = pd.MultiIndex.from_frame(te_key.loc[mask, ["type", "atom_0", "atom_1"]])
        te_key.loc[mask, "pred"] = idx.map(med_pair).astype("float64")

    mask = te_key["pred"].isna()
    if mask.any():
        idx = pd.MultiIndex.from_frame(te_key.loc[mask, ["type", "logdist_bin"]])
        te_key.loc[mask, "pred"] = idx.map(med_t_logbin).astype("float64")

    mask = te_key["pred"].isna()
    if mask.any():
        idx = pd.MultiIndex.from_frame(te_key.loc[mask, ["type", "dist_bin"]])
        te_key.loc[mask, "pred"] = idx.map(med_t_distbin).astype("float64")

    mask = te_key["pred"].isna()
    if mask.any():
        te_key.loc[mask, "pred"] = (
            te_key.loc[mask, "type"].map(med_type).astype("float64")
        )

    te_key["pred"] = te_key["pred"].fillna(global_med).astype(np.float32)
    baseline = te_key["pred"].values

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
