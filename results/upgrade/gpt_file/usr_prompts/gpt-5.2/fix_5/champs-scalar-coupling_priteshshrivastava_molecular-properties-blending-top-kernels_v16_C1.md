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

-1.680697846605887

# 6. Current score

1.23566

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'Your notebook fails because it references external “blending tutorial” and “otherkernelsadded” input folders that do not exist in this Kaggle environment, so no submission is produced. I replace that blending-only code with a minimal, self-contained baseline that uses only the provided CHAMPS dataset files and writes a valid `submission.csv`. To keep core logic simple and stable (and avoid heavy training within the 600s limit), the model predict the per-`type` mean coupling constant from the training data (a common safe baseline), and fall back to the global mean for any unseen types. This run end-to-end and generate a correctly formatted CSV for submission.'
- What this solution (achieved 1.23566) has done: 'Your current baseline (per-`type` mean) is far from the target, so we need a small but legitimate feature upgrade while keeping the overall approach simple and fast. I keep it as a lightweight grouped-statistics regressor, but make it more specific by conditioning on `(type, atom_0 element, atom_1 element)` using `structures.csv` to look up the elements for each atom index. This typically reduces MAE substantially versus `type`-only means, yet remains the same “predict group mean” core logic (no model architecture/training loop changes). I also add safe backoff levels (to `type` mean, then global mean) to guarantee a complete submission.'
- What this solution (achieved 1.23566) has done: 'Your current grouped-mean baseline is fast and stable but too coarse, so we keep the same “predict a group mean with backoffs” core logic and just make the grouping slightly richer using geometry you already have in `structures.csv`. Specifically, we add the inter-atomic distance feature (computed from x/y/z) and predict the mean within `(type, atom_0, atom_1, distance_bin)`; this usually lowers MAE meaningfully while staying within the same no-training approach. We also keep strict fallback levels (to `(type, atom_0, atom_1)`, then `type`, then global mean) to ensure a complete submission. This should move the score downward (better, since lower is better) toward your target without changing the overall approach or runtime drastically.'
- What this solution (achieved 1.23566) has done: 'Your current grouped-mean-with-backoffs baseline is still far from the target, so we keep the exact same core “predict a mean for a group” logic but make the grouping slightly more informative using only data you already load. Specifically, we (1) canonicalize the atom pair ordering within each row (so symmetric pairs share statistics), and (2) add a very cheap, chemistry-relevant context signal: each atom’s degree (number of atoms bonded within a distance cutoff) computed from `structures.csv`, then include `(deg0, deg1)` in the finest-grain group with backoffs unchanged. This remains a pure lookup/statistics approach (no training loop/model change), should lower MAE materially, and stays within time by computing degrees once per molecule with a small per-molecule distance-matrix routine.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

DATA_DIR = "/kaggle/data/champs-scalar-coupling"

print("Listing data dir:", DATA_DIR)
print(sorted(os.listdir(DATA_DIR))[:20])



## === cell 1
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
structures_path = os.path.join(DATA_DIR, "structures.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample = pd.read_csv(sample_path)
structures = pd.read_csv(structures_path)

required_train_cols = {
    "molecule_name",
    "atom_index_0",
    "atom_index_1",
    "type",
    "scalar_coupling_constant",
}
required_test_cols = {"id", "molecule_name", "atom_index_0", "atom_index_1", "type"}
assert required_train_cols.issubset(
    train.columns
), f"train.csv missing columns: {required_train_cols - set(train.columns)}"
assert required_test_cols.issubset(
    test.columns
), f"test.csv missing columns: {required_test_cols - set(test.columns)}"
assert {"id", "scalar_coupling_constant"}.issubset(
    sample.columns
), "sample_submission.csv has unexpected columns"

structures = structures[["molecule_name", "atom_index", "atom", "x", "y", "z"]].copy()
structures["atom_index"] = structures["atom_index"].astype(np.int32, copy=False)
for c in ["x", "y", "z"]:
    structures[c] = structures[c].astype(np.float32, copy=False)


def compute_degrees(struct_df: pd.DataFrame, cutoff: float = 1.8) -> pd.DataFrame:
    out = np.empty(len(struct_df), dtype=np.int16)
    for mol, g in struct_df.groupby("molecule_name", sort=False):
        idx = g.index.values
        xyz = g[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=False)
        n = xyz.shape[0]
        if n <= 1:
            out[idx] = 0
            continue
        diff = xyz[:, None, :] - xyz[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff).astype(np.float32, copy=False)
        neigh = (d2 <= (cutoff * cutoff)) & (d2 > 0.0)
        out[idx] = neigh.sum(axis=1).astype(np.int16, copy=False)
    deg_df = struct_df[["molecule_name", "atom_index"]].copy()
    deg_df["degree"] = out
    return deg_df


deg = compute_degrees(structures, cutoff=1.8)

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

deg0 = deg.rename(columns={"atom_index": "atom_index_0", "degree": "deg0"})
deg1 = deg.rename(columns={"atom_index": "atom_index_1", "degree": "deg1"})

train = train.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
train = train.merge(s1, on=["molecule_name", "atom_index_1"], how="left")
test = test.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
test = test.merge(s1, on=["molecule_name", "atom_index_1"], how="left")

train = train.merge(deg0, on=["molecule_name", "atom_index_0"], how="left")
train = train.merge(deg1, on=["molecule_name", "atom_index_1"], how="left")
test = test.merge(deg0, on=["molecule_name", "atom_index_0"], how="left")
test = test.merge(deg1, on=["molecule_name", "atom_index_1"], how="left")

train["atom_0"] = train["atom_0"].fillna("UNK")
train["atom_1"] = train["atom_1"].fillna("UNK")
test["atom_0"] = test["atom_0"].fillna("UNK")
test["atom_1"] = test["atom_1"].fillna("UNK")

train["deg0"] = train["deg0"].fillna(-1).astype(np.int16)
train["deg1"] = train["deg1"].fillna(-1).astype(np.int16)
test["deg0"] = test["deg0"].fillna(-1).astype(np.int16)
test["deg1"] = test["deg1"].fillna(-1).astype(np.int16)


def add_distance(df: pd.DataFrame) -> pd.DataFrame:
    dx = (df["x0"] - df["x1"]).astype("float32")
    dy = (df["y0"] - df["y1"]).astype("float32")
    dz = (df["z0"] - df["z1"]).astype("float32")
    d = np.sqrt(dx * dx + dy * dy + dz * dz).astype("float32")
    df["dist"] = d
    return df


train = add_distance(train)
test = add_distance(test)


def canonicalize_pairs(df: pd.DataFrame) -> pd.DataFrame:
    a0 = df["atom_0"].astype(str)
    a1 = df["atom_1"].astype(str)
    swap = a0.gt(a1)  # swap if atom_0 string > atom_1 string
    if swap.any():
        df.loc[swap, ["atom_0", "atom_1"]] = df.loc[
            swap, ["atom_1", "atom_0"]
        ].to_numpy()
        df.loc[swap, ["x0", "y0", "z0", "x1", "y1", "z1"]] = df.loc[
            swap, ["x1", "y1", "z1", "x0", "y0", "z0"]
        ].to_numpy()
        df.loc[swap, ["deg0", "deg1"]] = df.loc[swap, ["deg1", "deg0"]].to_numpy()
        df.loc[swap, ["atom_index_0", "atom_index_1"]] = df.loc[
            swap, ["atom_index_1", "atom_index_0"]
        ].to_numpy()
    return df


train = canonicalize_pairs(train)
test = canonicalize_pairs(test)

valid_dist = train["dist"].dropna()
if len(valid_dist) == 0:
    train["dist_bin"] = -1
    test["dist_bin"] = -1
else:
    q = np.linspace(0.0, 1.0, 41)
    edges = np.unique(np.quantile(valid_dist.values, q))
    if edges.size < 2:
        train["dist_bin"] = 0
        test["dist_bin"] = 0
    else:
        train["dist_bin"] = pd.cut(
            train["dist"], bins=edges, labels=False, include_lowest=True
        )
        test["dist_bin"] = pd.cut(
            test["dist"], bins=edges, labels=False, include_lowest=True
        )
        train["dist_bin"] = train["dist_bin"].fillna(-1).astype(np.int16)
        test["dist_bin"] = test["dist_bin"].fillna(-1).astype(np.int16)

global_mean = float(train["scalar_coupling_constant"].mean())

type_mean = train.groupby("type")["scalar_coupling_constant"].mean()
type_atom_mean = train.groupby(["type", "atom_0", "atom_1"])[
    "scalar_coupling_constant"
].mean()

type_atom_dist_mean = train.groupby(["type", "atom_0", "atom_1", "dist_bin"])[
    "scalar_coupling_constant"
].mean()

type_atom_dist_deg_mean = train.groupby(
    ["type", "atom_0", "atom_1", "dist_bin", "deg0", "deg1"]
)["scalar_coupling_constant"].mean()

key6 = list(
    zip(
        test["type"],
        test["atom_0"],
        test["atom_1"],
        test["dist_bin"],
        test["deg0"],
        test["deg1"],
    )
)
pred = pd.Series(key6).map(type_atom_dist_deg_mean).astype("float64")

key4 = list(zip(test["type"], test["atom_0"], test["atom_1"], test["dist_bin"]))
pred = pred.fillna(pd.Series(key4).map(type_atom_dist_mean).astype("float64"))

key3 = list(zip(test["type"], test["atom_0"], test["atom_1"]))
pred = pred.fillna(pd.Series(key3).map(type_atom_mean).astype("float64"))

pred = pred.fillna(test["type"].map(type_mean).astype("float64"))
pred = pred.fillna(global_mean)

submission = pd.DataFrame(
    {
        "id": test["id"].astype(sample["id"].dtype, copy=False),
        "scalar_coupling_constant": pred.values,
    }
)

submission = sample[["id"]].merge(submission, on="id", how="left")
if submission["scalar_coupling_constant"].isna().any():
    submission["scalar_coupling_constant"] = submission[
        "scalar_coupling_constant"
    ].fillna(global_mean)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print("Rows:", len(submission), "Expected:", len(sample))
assert len(submission) == len(sample), "Submission row count mismatch"
assert list(submission.columns) == [
    "id",
    "scalar_coupling_constant",
], "Submission columns mismatch"
assert (
    not submission["scalar_coupling_constant"].isna().any()
), "Submission contains NaN predictions"
