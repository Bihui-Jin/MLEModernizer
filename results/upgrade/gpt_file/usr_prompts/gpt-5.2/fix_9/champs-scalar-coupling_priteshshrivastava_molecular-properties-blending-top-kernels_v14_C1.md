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

-1.6708838544682014

# 6. Current score

1.23566

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'Your current notebook is a blender that depends on external submission files (e.g., `../input/champs-blending-tutorial/1.csv`) which do not exist in this Kaggle environment, causing the FileNotFoundError and preventing any `.csv` submission from being written. To make it run end-to-end here while preserving the “blend” semantics, I (1) switch all inputs to the provided CHAMPS dataset paths, (2) replace the missing blended components with a single simple, deterministic baseline prediction built only from available training data (mean per coupling `type`), and (3) ensure the submission is aligned to `test.csv` `id` ordering and written with the required columns and a `.csv` suffix. This yield a valid submission file and should produce a reasonable (not necessarily SOTA) score without changing to a totally different modeling approach.'
- What this solution (achieved 1.23566) has done: 'Your current baseline predicts a single mean per coupling `type`, which is simple but leaves a lot of signal unused and explains why the log-MAE is far from the target. To move the score downward toward the target while keeping the same “groupby-aggregate then map to test” core logic, I add one extra level of granularity: a mean per `(type, atom_0, atom_1)` pair using `structures.csv` to look up the element symbols. This keeps the approach deterministic and purely target-encoding from train (no leakage from test labels), while usually being a large quality jump versus `type`-only means. I also add safe backoff (pair mean → type mean → global mean) and ensure the submission aligns exactly to `test.csv` ids and writes a valid `.csv`.'
- What this solution (achieved 1.23566) has done: 'Your current score (1.23566, lower-is-better) is far from the target (-1.6709), so we should improve substantially while keeping the same “groupby aggregate from train then map onto test with backoff” core logic. The biggest safe gain without changing the approach is to add a more informative grouping key using only available files: include the `type` plus both atoms’ element symbols and their inter-atomic distance from `structures.csv`. We do a deterministic “bin-and-mean” target encoding: compute distance per row, bin it, then take the mean `scalar_coupling_constant` per `(type, atom_0, atom_1, dist_bin)` with the same fallback chain as before. This preserves evaluation semantics, avoids leakage (only uses train targets for aggregates), and typically reduces MAE a lot for CHAMPS because coupling strength depends strongly on distance.'
- What this solution (achieved 1.23566) has done: 'The crash comes from `pd.cut(..., labels=False)` producing NaNs for some rows (typically when `dist` is NaN due to a failed merge / missing coordinates, or when bin edges collapse), and then trying to `.astype(np.int16)` which cannot represent NaN. I keep the same core “type + (atom_0, atom_1) + distance-bin mean with fallback” logic, but make the binning robust by (1) ensuring `dist` is finite (fill with per-type median, then global median), and (2) using pandas’ nullable integer (`Int16`) during bin assignment and only converting to a real int after filling missing bins with a default. This is score-neutral in intent (it just prevents NaNs from breaking execution) and finally write a valid `my_blend_1.csv` submission.'
- What this solution (achieved 1.23566) has done: 'Your current approach is a deterministic target-encoding lookup table keyed by `(type, atom_0, atom_1, dist_bin)` with a backoff chain, which is good but still too coarse to reach the target. To move the score downward (lower is better) toward the target while preserving the same “groupby mean then map to test” core logic, I only increase the distance-bin granularity and make the bin edges a bit more stable by using quantile-based edges with a slightly higher bin count. This usually reduces within-type bias because coupling strength varies sharply with distance, without changing the modeling paradigm. I also keep the existing robust NaN handling and submission alignment intact.'
- What this solution (achieved 1.23566) has done: 'Your current model is a deterministic lookup table using `(type, atom_0, atom_1, dist_bin)` means with backoff, and the biggest remaining “minimal-change” lever is making `dist_bin` better aligned with the target by adding slightly more granularity and stabilizing the quantile bin edges. I increase the number of quantile bins and also compute quantiles on the *training distribution per type* exactly as you already do, but with a safer edge construction that avoids too many collapsed/duplicate edges (which otherwise reduces effective bin resolution). This keeps the same core logic (groupby mean → map to test → fallback chain), but should move the score downward (better, since lower is better) toward your target. I also keep your robust NaN-distance handling and exact submission alignment unchanged.'
- What this solution (achieved 1.23566) has done: 'Your current score (1.23566, lower-is-better) is still far from the target (-1.6709), so we should improve substantially while keeping the same deterministic “groupby mean lookup with fallback” core logic. The smallest high-impact change is to make the distance conditioning smoother by using two distance resolutions (coarse + fine) and blending their lookup predictions, while keeping the same features (type, atom_0/atom_1, distance) and the same fallback chain. This typically reduces MAE because some coupling types benefit from fine bins while others are more stable with coarser bins, and averaging the two reduces quantile-edge noise. I also keep your robust NaN/inf distance handling and exact submission alignment unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

DATA_DIR = "/kaggle/data/champs-scalar-coupling"

print("DATA_DIR exists:", os.path.exists(DATA_DIR))
print("Files:", sorted(os.listdir(DATA_DIR))[:20])



## === cell 1
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
structures_path = os.path.join(DATA_DIR, "structures.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

structures = pd.read_csv(
    structures_path, usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"]
)

a0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
    }
)
a1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
    }
)

train = train.merge(a0, on=["molecule_name", "atom_index_0"], how="left")
train = train.merge(a1, on=["molecule_name", "atom_index_1"], how="left")
test = test.merge(a0, on=["molecule_name", "atom_index_0"], how="left")
test = test.merge(a1, on=["molecule_name", "atom_index_1"], how="left")

for df in (train, test):
    dx = (df["x0"] - df["x1"]).to_numpy(dtype=np.float64)
    dy = (df["y0"] - df["y1"]).to_numpy(dtype=np.float64)
    dz = (df["z0"] - df["z1"]).to_numpy(dtype=np.float64)
    df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz)

for df in (train, test):
    swap = (df["atom_0"] > df["atom_1"]) | (
        (df["atom_0"] == df["atom_1"]) & (df["atom_index_0"] > df["atom_index_1"])
    )
    df.loc[swap, ["atom_0", "atom_1"]] = df.loc[swap, ["atom_1", "atom_0"]].to_numpy()

global_dist_median = float(np.nanmedian(train["dist"].to_numpy(dtype=np.float64)))
type_dist_median = train.groupby("type", sort=False)["dist"].median()

for df in (train, test):
    bad = ~np.isfinite(df["dist"].to_numpy(dtype=np.float64))
    if bad.any():
        df.loc[bad, "dist"] = df.loc[bad, "type"].map(type_dist_median)
        df["dist"] = df["dist"].fillna(global_dist_median)




## === cell 2
def assign_quantile_bins_per_type(
    train_dist: pd.Series,
    test_dist: pd.Series,
    train_type: pd.Series,
    test_type: pd.Series,
    n_bins: int,
):
    """
    Same core logic as before: per-type quantile edges, pd.cut to labels=False, Int16 temporary, then fill.
    Change is only parameterization (n_bins) and packaging so we can do two resolutions and blend.
    """
    tr_bin = pd.Series(
        pd.array([pd.NA] * len(train_dist), dtype="Int16"), index=train_dist.index
    )
    te_bin = pd.Series(
        pd.array([pd.NA] * len(test_dist), dtype="Int16"), index=test_dist.index
    )

    edges_by_type = {}
    for t, d in train_dist.groupby(train_type, sort=False):
        q = np.linspace(0.0, 1.0, n_bins + 1)
        arr = d.to_numpy(dtype=np.float64)
        edges = np.quantile(arr, q)
        edges = np.unique(edges)

        if edges.size < (n_bins // 2):
            mn = float(np.nanmin(arr))
            mx = float(np.nanmax(arr))
            if not np.isfinite(mn) or not np.isfinite(mx) or mn == mx:
                mn, mx = global_dist_median - 1.0, global_dist_median + 1.0
            edges = np.linspace(mn, mx, num=min(n_bins + 1, 10), dtype=np.float64)
            edges = np.unique(edges)

        if edges.size < 3:
            mn = float(np.nanmin(arr))
            mx = float(np.nanmax(arr))
            if not np.isfinite(mn) or not np.isfinite(mx) or mn == mx:
                mn, mx = global_dist_median - 1.0, global_dist_median + 1.0
            edges = np.array([mn, (mn + mx) / 2.0, mx], dtype=np.float64)

        edges = edges.astype(np.float64, copy=False)
        edges[0] = -np.inf
        edges[-1] = np.inf
        edges_by_type[t] = edges

    tr_type_arr = train_type.to_numpy()
    te_type_arr = test_type.to_numpy()
    for t, edges in edges_by_type.items():
        tr_mask = tr_type_arr == t
        te_mask = te_type_arr == t

        tr_bins = pd.cut(
            train_dist.loc[tr_mask], bins=edges, labels=False, include_lowest=True
        )
        te_bins = pd.cut(
            test_dist.loc[te_mask], bins=edges, labels=False, include_lowest=True
        )

        tr_bin.loc[tr_mask] = tr_bins.astype("Int16")
        te_bin.loc[te_mask] = te_bins.astype("Int16")

    tr_bin = tr_bin.fillna(0).astype(np.int16)
    te_bin = te_bin.fillna(0).astype(np.int16)
    return tr_bin, te_bin


def predict_from_bins(train_df: pd.DataFrame, test_df: pd.DataFrame, dist_bin_col: str):
    """
    Same core as your current predictor: (type, atom_0, atom_1, dist_bin) -> mean, with backoff.
    """
    pair_dist_mean = (
        train_df.groupby(["type", "atom_0", "atom_1", dist_bin_col], sort=False)[
            "scalar_coupling_constant"
        ]
        .mean()
        .astype(np.float64)
    )
    pair_mean = (
        train_df.groupby(["type", "atom_0", "atom_1"], sort=False)[
            "scalar_coupling_constant"
        ]
        .mean()
        .astype(np.float64)
    )
    type_mean = (
        train_df.groupby("type", sort=False)["scalar_coupling_constant"]
        .mean()
        .astype(np.float64)
    )
    global_mean = float(train_df["scalar_coupling_constant"].mean())

    test_key_4 = pd.MultiIndex.from_frame(
        test_df[["type", "atom_0", "atom_1", dist_bin_col]]
    )
    pred = pair_dist_mean.reindex(test_key_4).to_numpy(dtype=np.float64)

    missing = np.isnan(pred)
    if missing.any():
        test_key_3 = pd.MultiIndex.from_frame(
            test_df.loc[missing, ["type", "atom_0", "atom_1"]]
        )
        pred_pair = pair_mean.reindex(test_key_3).to_numpy(dtype=np.float64)
        pred[missing] = pred_pair

    missing2 = np.isnan(pred)
    if missing2.any():
        pred_type = (
            test_df.loc[missing2, "type"].map(type_mean).to_numpy(dtype=np.float64)
        )
        pred[missing2] = pred_type

    missing3 = np.isnan(pred)
    if missing3.any():
        pred[missing3] = global_mean

    return pred




## === cell 3

n_bins_coarse = 24
n_bins_fine = 48

train["dist_bin_c"], test["dist_bin_c"] = assign_quantile_bins_per_type(
    train["dist"], test["dist"], train["type"], test["type"], n_bins=n_bins_coarse
)
train["dist_bin_f"], test["dist_bin_f"] = assign_quantile_bins_per_type(
    train["dist"], test["dist"], train["type"], test["type"], n_bins=n_bins_fine
)

pred_c = predict_from_bins(train, test, "dist_bin_c")
pred_f = predict_from_bins(train, test, "dist_bin_f")

w_fine = 0.65
pred = w_fine * pred_f + (1.0 - w_fine) * pred_c

submission = pd.DataFrame(
    {"id": test["id"].to_numpy(), "scalar_coupling_constant": pred}
)

assert (
    submission.shape[0] == sample_sub.shape[0]
), "Row count mismatch vs sample_submission"
assert list(submission.columns) == [
    "id",
    "scalar_coupling_constant",
], "Wrong submission columns"
assert submission["id"].is_unique, "Submission id must be unique"

out_path = "my_blend_1.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print(submission["scalar_coupling_constant"].describe())
print(
    "Any NaN preds:", np.isnan(submission["scalar_coupling_constant"].to_numpy()).any()
)
