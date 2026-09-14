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
seaborn==0.12.2
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

-1.5209593019916507

# 6. Current score

1.18497

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.18497) has done: 'Your notebook fails because it tries to read two external submissions from `../input/...` folders that are not present in your environment, so `sub1/sub2` never get created and all downstream cells error. I keep the same “blend two submissions into `sample_submission` and write a CSV” core logic, but make it robust by (1) loading those files only if they exist, otherwise falling back to valid baseline predictions built from the provided competition data. This guarantees the notebook runs end-to-end and always writes `stackers_blend.csv` with the required columns. The fallback uses only train statistics per coupling `type` (median) which is score-reasonable and should move you toward the target much more than a constant-0 submission.'
- What this solution (achieved 1.18497) has done: 'Your current blend is effectively blending the same fallback twice (type-median), so it can’t get close to the target. To move the score down toward -1.52 while keeping the “baseline stats → predict test → write CSV” core logic, I upgrade the fallback to a slightly richer, still-tabular aggregation: per (`type`, `atom_0`, `atom_1`) median coupling, with safe fallbacks to per-`type` median and then global median. This uses only provided competition files (`train.csv`, `test.csv`, `structures.csv`) and keeps the rest of your pipeline (alignment + blending + output file) unchanged. I keep the existing external-submission loading logic intact, but if those files are missing, the new fallback should materially improve MAE and therefore reduce the log-MAE score toward your target.'
- What this solution (achieved 1.18497) has done: 'Your current score is far worse than the target (lower is better), so we should legitimately improve predictions while keeping your “aggregation baseline → optional blending → write CSV” core logic intact. The smallest meaningful gain is to extend the fallback from just (`type`, `atom_0`, `atom_1`) medians to also use a geometry-derived feature that is highly predictive: the inter-atomic distance. We add a binned-distance median lookup per (`type`, `atom_0`, `atom_1`, dist_bin), with safe fallbacks to your existing (`type`, `atom_0`, `atom_1`) median, then per-`type` median, then global median. External submission loading and the final blend/write logic remain unchanged; only the fallback generator is upgraded.'
- What this solution (achieved 1.18497) has done: 'Your current score (1.18497, lower-is-better) is far from the target (-1.52096), so we should legitimately improve the fallback predictions while keeping your same “baseline aggregation → optional blending → write CSV” core logic. The simplest high-signal fix is to make the distance binning *type-aware* (different coupling types have very different distance regimes), so the per-bin medians stop being over-smoothed and become much more accurate. I also tighten the distance clipping to a type-specific max derived from train quantiles to avoid bin-collisions at the upper cap, while preserving your same median lookup + fallback chain and the same submission writing/blending behavior. External submission loading stays unchanged; these changes only affect the fallback generator used when those files aren’t present.'
- What this solution (achieved 1.18497) has done: 'Your current score is much worse than the target (lower is better), so we should improve the fallback predictions while keeping your same “aggregation baseline → optional blend → write CSV” core logic. The biggest issue is that `pred = pd.Series(key4).map(...)` won’t match a MultiIndex reliably; we change this to a proper merge/join on grouped medians so the distance-bin lookup actually gets used. We also ensure symmetry is handled by building both (atom_0,atom_1) and swapped keys for the lookup tables, again via merges (not map-on-tuples). Finally, we keep your blending/output exactly the same so the pipeline remains stable and still writes `stackers_blend.csv`.'
- What this solution (achieved 1.18497) has done: 'Your current score is far worse than the target (lower is better), and the core issue is that your two “submissions” are identical because both fall back to the same baseline, so blending cannot improve anything. I keep your exact pipeline (optional external submissions → otherwise fallback → align → blend → write `stackers_blend.csv`) but make the two fallbacks deliberately different, so the blend becomes meaningful and should reduce MAE. Specifically, I keep your existing distance-binned median baseline as `sub2`, and create a complementary `sub1` baseline that is still aggregation-based but uses a different signal (type + atom-pair + distance with a different binning scheme plus a simple 1/dist feature grouped median). This is a minimal change confined to fallback generation; submission format and I/O paths remain unchanged.'
- What this solution (achieved 1.18497) has done: 'We keep your exact “two fallbacks → align → blend → write CSV” pipeline, but fix a key accuracy issue: both baselines currently ignore the coupling directionality because swapped atom pairs are deduplicated, so one direction can overwrite the other even though `atom_0/atom_1` order is meaningful. The minimal score-improving change is to make the aggregation symmetric by canonicalizing `(atom_0, atom_1)` into an unordered pair (min/max) consistently in both train and test before grouping/merging, so the model uses all relevant training rows and matches test rows reliably. We also preserve your distance and inv-distance binning logic, just applying it after canonicalization, and keep the blend weights/output unchanged to maintain evaluation semantics while improving MAE (lower log-MAE). The script still runs end-to-end and writes `stackers_blend.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path


def find_existing_path(candidates):
    for p in candidates:
        if p is None:
            continue
        p = Path(p)
        if p.exists():
            return str(p)
    return None


BASE_INPUT = find_existing_path(
    [
        "/kaggle/data/champs-scalar-coupling",
        "/kaggle/input/champs-scalar-coupling",
        "../input/champs-scalar-coupling",
        "/kaggle/data/input/champs-scalar-coupling",
        "/kaggle/input",
        "../input",
    ]
)

if BASE_INPUT is None:
    raise FileNotFoundError(
        "Could not locate the input directory containing champs-scalar-coupling data."
    )

print("Resolved BASE_INPUT:", BASE_INPUT)



## === cell 1
import numpy as np
import pandas as pd
import seaborn as sns

import os

try:
    print("Contents of BASE_INPUT:", os.listdir(BASE_INPUT)[:20])
except Exception as e:
    print("Could not list BASE_INPUT due to:", repr(e))



## === cell 2
sample_path = os.path.join(BASE_INPUT, "sample_submission.csv")
train_path = os.path.join(BASE_INPUT, "train.csv")
test_path = os.path.join(BASE_INPUT, "test.csv")
structures_path = os.path.join(BASE_INPUT, "structures.csv")

sample = pd.read_csv(sample_path)
test = pd.read_csv(test_path)

sub1_path = find_existing_path(
    [
        "../input/lgb-public-kernels-plus-more-features/sub_lgb_model_individual.csv",
        "/kaggle/input/lgb-public-kernels-plus-more-features/sub_lgb_model_individual.csv",
    ]
)
sub2_path = find_existing_path(
    [
        "../input/staking-and-stealing-like-a-molecule/submission.csv",
        "/kaggle/input/staking-and-stealing-like-a-molecule/submission.csv",
    ]
)

sub1 = None
sub2 = None


def validate_submission(df, name):
    if df is None:
        return False
    ok = ("id" in df.columns) and ("scalar_coupling_constant" in df.columns)
    if not ok:
        print(f"{name} invalid columns: {df.columns.tolist()}")
        return False
    if df["id"].nunique() != len(df):
        print(f"{name} has duplicate ids; will re-aggregate by id with mean.")
        df = df.groupby("id", as_index=False)["scalar_coupling_constant"].mean()
    return True


if sub1_path is not None:
    sub1 = pd.read_csv(sub1_path)
    if not validate_submission(sub1, "sub1"):
        sub1 = None
else:
    print("sub1 file not found; will use fallback predictions for sub1.")

if sub2_path is not None:
    sub2 = pd.read_csv(sub2_path)
    if not validate_submission(sub2, "sub2"):
        sub2 = None
else:
    print("sub2 file not found; will use fallback predictions for sub2.")


def _load_minimal_train_test_structures(train_csv_path, test_df, structures_csv_path):
    train = pd.read_csv(
        train_csv_path,
        usecols=[
            "molecule_name",
            "atom_index_0",
            "atom_index_1",
            "type",
            "scalar_coupling_constant",
        ],
    )
    test_small = test_df[
        ["id", "molecule_name", "atom_index_0", "atom_index_1", "type"]
    ].copy()

    structs = pd.read_csv(
        structures_csv_path,
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

    train = train.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
    train = train.merge(s1, on=["molecule_name", "atom_index_1"], how="left")

    test_small = test_small.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
    test_small = test_small.merge(s1, on=["molecule_name", "atom_index_1"], how="left")

    for df in (train, test_small):
        df["atom_0"] = df["atom_0"].fillna("UNK")
        df["atom_1"] = df["atom_1"].fillna("UNK")

    for df in (train, test_small):
        dx = (df["x0"] - df["x1"]).astype(np.float32)
        dy = (df["y0"] - df["y1"]).astype(np.float32)
        dz = (df["z0"] - df["z1"]).astype(np.float32)
        df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)

    return train, test_small


def _canonicalize_atom_pair(df):
    a0 = df["atom_0"].astype(str)
    a1 = df["atom_1"].astype(str)
    lo = np.where(a0.values <= a1.values, a0.values, a1.values)
    hi = np.where(a0.values <= a1.values, a1.values, a0.values)
    df = df.copy()
    df["atom_lo"] = lo
    df["atom_hi"] = hi
    return df


def make_type_atompair_distbin_median_baseline(
    train_csv_path, test_df, structures_csv_path
):
    train, test_small = _load_minimal_train_test_structures(
        train_csv_path, test_df, structures_csv_path
    )

    train = _canonicalize_atom_pair(train)
    test_small = _canonicalize_atom_pair(test_small)

    dist_cap_by_type = (
        train.groupby("type")["dist"]
        .quantile(0.995)
        .clip(lower=np.float32(1.0), upper=np.float32(20.0))
    )
    global_cap = float(train["dist"].quantile(0.995))
    if not np.isfinite(global_cap):
        global_cap = 10.0

    bin_width_by_type = {
        "1JHC": np.float32(0.02),
        "1JHN": np.float32(0.02),
        "1JHH": np.float32(0.02),
        "1JCN": np.float32(0.02),
        "2JHC": np.float32(0.04),
        "2JHH": np.float32(0.04),
        "2JHN": np.float32(0.04),
        "3JHC": np.float32(0.06),
        "3JHH": np.float32(0.06),
        "3JHN": np.float32(0.06),
        "default": np.float32(0.05),
    }

    def _make_bins(df):
        caps = df["type"].map(dist_cap_by_type).astype(np.float32)
        caps = caps.fillna(np.float32(global_cap))
        d = df["dist"].fillna(np.float32(0.0)).astype(np.float32)
        d = np.minimum(np.maximum(d, np.float32(0.0)), caps)

        bw = df["type"].map(bin_width_by_type).astype(np.float32)
        bw = bw.fillna(bin_width_by_type["default"])
        dist_bin = np.floor(d / bw).astype(np.int16)
        return dist_bin

    train["dist_bin"] = _make_bins(train)
    test_small["dist_bin"] = _make_bins(test_small)

    med4 = (
        train.groupby(["type", "atom_lo", "atom_hi", "dist_bin"], as_index=False)[
            "scalar_coupling_constant"
        ]
        .median()
        .rename(columns={"scalar_coupling_constant": "pred4"})
    )

    med3 = (
        train.groupby(["type", "atom_lo", "atom_hi"], as_index=False)[
            "scalar_coupling_constant"
        ]
        .median()
        .rename(columns={"scalar_coupling_constant": "pred3"})
    )

    med_type = (
        train.groupby("type", as_index=False)["scalar_coupling_constant"]
        .median()
        .rename(columns={"scalar_coupling_constant": "pred_type"})
    )
    global_median = float(train["scalar_coupling_constant"].median())

    out = test_small[["id", "type", "atom_lo", "atom_hi", "dist_bin"]].copy()

    out = out.merge(med4, on=["type", "atom_lo", "atom_hi", "dist_bin"], how="left")
    out = out.merge(med3, on=["type", "atom_lo", "atom_hi"], how="left")
    out = out.merge(med_type, on=["type"], how="left")

    pred = out["pred4"]
    pred = pred.fillna(out["pred3"])
    pred = pred.fillna(out["pred_type"])
    pred = pred.fillna(global_median).astype(np.float32)

    return pd.DataFrame(
        {"id": out["id"].values, "scalar_coupling_constant": pred.values}
    )


def make_complementary_inverse_distance_baseline(
    train_csv_path, test_df, structures_csv_path
):
    train, test_small = _load_minimal_train_test_structures(
        train_csv_path, test_df, structures_csv_path
    )

    train = _canonicalize_atom_pair(train)
    test_small = _canonicalize_atom_pair(test_small)

    eps = np.float32(1e-3)
    train["inv_dist"] = (
        np.float32(1.0) / (train["dist"].astype(np.float32) + eps)
    ).astype(np.float32)
    test_small["inv_dist"] = (
        np.float32(1.0) / (test_small["dist"].astype(np.float32) + eps)
    ).astype(np.float32)

    edges_by_type = {}
    q = np.linspace(0.0, 1.0, 33, dtype=np.float32)  # 32 bins
    for t, grp in train.groupby("type", sort=False):
        vals = grp["inv_dist"].values
        if len(vals) < 1000:
            continue
        edges = np.quantile(vals, q)
        edges = np.unique(edges.astype(np.float32))
        if len(edges) >= 5:
            edges_by_type[t] = edges

    global_edges = np.unique(
        np.quantile(train["inv_dist"].values, q).astype(np.float32)
    )
    if len(global_edges) < 5:
        global_edges = np.array(
            [np.float32(train["inv_dist"].min()), np.float32(train["inv_dist"].max())],
            dtype=np.float32,
        )

    def _assign_bin(df):
        out_bins = np.empty(len(df), dtype=np.int16)
        for t in df["type"].unique():
            mask = (df["type"] == t).values
            edges = edges_by_type.get(t, global_edges)
            b = pd.cut(
                df.loc[mask, "inv_dist"], bins=edges, labels=False, include_lowest=True
            )
            out_bins[mask] = b.fillna(-1).astype(np.int16).values
        return out_bins

    train["inv_bin"] = _assign_bin(train)
    test_small["inv_bin"] = _assign_bin(test_small)

    med4 = (
        train.groupby(["type", "atom_lo", "atom_hi", "inv_bin"], as_index=False)[
            "scalar_coupling_constant"
        ]
        .median()
        .rename(columns={"scalar_coupling_constant": "pred4"})
    )

    med3 = (
        train.groupby(["type", "atom_lo", "atom_hi"], as_index=False)[
            "scalar_coupling_constant"
        ]
        .median()
        .rename(columns={"scalar_coupling_constant": "pred3"})
    )

    med_type = (
        train.groupby("type", as_index=False)["scalar_coupling_constant"]
        .median()
        .rename(columns={"scalar_coupling_constant": "pred_type"})
    )
    global_median = float(train["scalar_coupling_constant"].median())

    out = test_small[["id", "type", "atom_lo", "atom_hi", "inv_bin"]].copy()
    out = out.merge(med4, on=["type", "atom_lo", "atom_hi", "inv_bin"], how="left")
    out = out.merge(med3, on=["type", "atom_lo", "atom_hi"], how="left")
    out = out.merge(med_type, on=["type"], how="left")

    pred = out["pred4"]
    pred = pred.fillna(out["pred3"])
    pred = pred.fillna(out["pred_type"])
    pred = pred.fillna(global_median).astype(np.float32)

    return pd.DataFrame(
        {"id": out["id"].values, "scalar_coupling_constant": pred.values}
    )


if sub1 is None:
    sub1 = make_complementary_inverse_distance_baseline(
        train_path, test, structures_path
    )
if sub2 is None:
    sub2 = make_type_atompair_distbin_median_baseline(train_path, test, structures_path)


def align_to_sample(sub, sample_ids):
    if sub["id"].nunique() != len(sub):
        sub = sub.groupby("id", as_index=False)["scalar_coupling_constant"].mean()
    sub = sub.set_index("id").reindex(sample_ids)
    if sub["scalar_coupling_constant"].isna().any():
        sub["scalar_coupling_constant"] = sub["scalar_coupling_constant"].fillna(0.0)
    return sub["scalar_coupling_constant"].values


ids = sample["id"].values
pred1 = align_to_sample(sub1, ids)
pred2 = align_to_sample(sub2, ids)

print("Loaded/created predictions:", pred1.shape, pred2.shape)
print(
    "Prediction stats sub1:",
    float(np.min(pred1)),
    float(np.max(pred1)),
    float(np.mean(pred1)),
)
print(
    "Prediction stats sub2:",
    float(np.min(pred2)),
    float(np.max(pred2)),
    float(np.mean(pred2)),
)



## === cell 3
sub1_aligned_series = pd.Series(pred1, name="scalar_coupling_constant")
print(sub1_aligned_series.describe())



## === cell 4
sub2_aligned_series = pd.Series(pred2, name="scalar_coupling_constant")
print(sub2_aligned_series.describe())



## === cell 5
sample["scalar_coupling_constant"] = (0.6 * pred2 + 0.4 * pred1).astype(np.float32)

assert list(sample.columns) == [
    "id",
    "scalar_coupling_constant",
], f"Unexpected submission columns: {sample.columns.tolist()}"
assert len(sample) == len(ids)
assert sample["scalar_coupling_constant"].notna().all()

sample.to_csv("stackers_blend.csv", index=False)
print("Wrote submission:", "stackers_blend.csv", "rows:", len(sample))
print(sample.head())



## === cell 6
sns.histplot(sample["scalar_coupling_constant"], bins=100)
