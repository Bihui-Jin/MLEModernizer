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

-1.314363717973649

# 6. Current score

1.23566

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'I remove the dependency on missing external Kaggle datasets (the four `../input/.../submission*.csv` files) by switching to an in-notebook baseline model that uses only the provided competition files in `/kaggle/input/champs-scalar-coupling/`. To keep changes minimal while ensuring a valid `.csv` submission is always produced, I build a simple, deterministic per-`type` mean target model from `train.csv` and apply it to `test.csv`. This runs end-to-end under the listed installed packages (NumPy/Pandas only), avoids runtime errors, and yields a reasonable score better than the all-zero baseline. The output be written as `submission.csv` with the exact required columns.'
- What this solution (achieved 1.23566) has done: 'Your current baseline only uses the coupling `type`, so it can’t capture the large within-type variation driven by atom identities and geometry; that’s why the MAE stays high. To move the score down toward your target with minimal core-logic change, I keep the same “simple deterministic aggregation model” approach but add one more readily available signal: the distance between the two atoms (computed from `structures.csv`), and then predict using per-`(type, distance_bin)` mean targets with a safe fallback to per-type mean and then global mean. This keeps training/prediction fully deterministic, uses only provided competition files, and should materially reduce error without introducing new modeling frameworks. The output submission schema and path remain unchanged (`submission.csv` with `id,scalar_coupling_constant`).'
- What this solution (achieved 1.23566) has done: 'I fix the runtime error caused by using NumPy’s `astype("category")` (NumPy doesn’t understand pandas’ `category` dtype) by creating `atom_pair` as a normal string (or pandas categorical via pandas, if needed). Then I ensure downstream mapping keys are consistent (avoid mixed string/category tuples) so the groupby means and `.map()` lookups work reliably. Finally, I keep the same core “per-(type, atom_pair, dist_bin) mean with fallbacks” logic and make sure a valid `submission.csv` is always written with correct columns, row count, and no NaNs.'
- What this solution (achieved 1.23566) has done: 'I fix the bin-assignment crash by making `assign_type_quantile_bins` robust to NaNs and to `qcut` returning missing bin labels when there aren’t enough unique distances (common in some coupling types). This keeps the exact same core logic (per-`(type, atom_pair, dist_bin)` mean with fallbacks) while ensuring `dist_bin` is always a valid integer (using a dedicated “missing bin” code when needed). I also guard against any rare missing structure merges by assigning a safe fallback bin and atom-pair, so prediction never produces NaNs. Finally, I ensure `submission.csv` is always written even if earlier cells failed by keeping the pipeline end-to-end and validating the output schema/rowcount.'
- What this solution (achieved 1.23566) has done: 'I fix the crash in `assign_bins_with_precomputed_edges` caused by `pd.cut` returning a NumPy array in this pandas version, by safely converting the result to a NumPy float array regardless of whether it’s a Series/Index/ndarray. I also make the bin assignment write-back robust by assigning via `out.loc[...]` with the actual row labels for each group (instead of using `.iloc` with label values), which avoids subtle misassignment when indices aren’t 0..N-1. These changes keep the same core logic (per-(type, atom_pair, dist_bin) mean with fallbacks) while ensuring the pipeline runs end-to-end. Finally, I ensure `submission.csv` is always produced with the correct columns and row count.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

DATA_DIR_CANDIDATES = [
    "/kaggle/input/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling",
    "/kaggle/input/champs-scalar-coupling/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling/champs-scalar-coupling",
]


def find_data_dir():
    for d in DATA_DIR_CANDIDATES:
        if os.path.exists(os.path.join(d, "train.csv")) and os.path.exists(
            os.path.join(d, "test.csv")
        ):
            return d
    base = "/kaggle/input"
    if os.path.isdir(base):
        for root, _, files in os.walk(base):
            if "train.csv" in files and "test.csv" in files:
                return root
    raise FileNotFoundError(
        "Could not locate champs-scalar-coupling data directory containing train.csv and test.csv"
    )


DATA_DIR = find_data_dir()
print("Using DATA_DIR:", DATA_DIR)
print("Files (head):", sorted(os.listdir(DATA_DIR))[:20])



## === cell 1
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
structures_path = os.path.join(DATA_DIR, "structures.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

print("train shape:", train.shape)
print("test shape:", test.shape)
print("sample_submission shape:", sample_sub.shape)
print("train columns:", train.columns.tolist())
print("test columns:", test.columns.tolist())

required_train_cols = {
    "id",
    "molecule_name",
    "atom_index_0",
    "atom_index_1",
    "type",
    "scalar_coupling_constant",
}
required_test_cols = {"id", "molecule_name", "atom_index_0", "atom_index_1", "type"}
if not required_train_cols.issubset(set(train.columns)):
    raise ValueError(
        f"train.csv missing required columns. Found={train.columns.tolist()}"
    )
if not required_test_cols.issubset(set(test.columns)):
    raise ValueError(
        f"test.csv missing required columns. Found={test.columns.tolist()}"
    )

structures = pd.read_csv(
    structures_path, usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"]
)
structures["atom_index"] = structures["atom_index"].astype(np.int32)
structures["atom"] = structures["atom"].astype(str)
for c in ["x", "y", "z"]:
    structures[c] = structures[c].astype(np.float32)

print("structures shape:", structures.shape)
print(structures.head())




## === cell 2
def add_distance_and_atoms(df, structures_df):
    base = df[["id", "molecule_name", "atom_index_0", "atom_index_1", "type"]].copy()
    base["atom_index_0"] = base["atom_index_0"].astype(np.int32)
    base["atom_index_1"] = base["atom_index_1"].astype(np.int32)

    s0 = structures_df.rename(
        columns={
            "atom_index": "atom_index_0",
            "atom": "atom_0",
            "x": "x0",
            "y": "y0",
            "z": "z0",
        }
    )
    s1 = structures_df.rename(
        columns={
            "atom_index": "atom_index_1",
            "atom": "atom_1",
            "x": "x1",
            "y": "y1",
            "z": "z1",
        }
    )

    merged = base.merge(
        s0[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    merged = merged.merge(
        s1[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
    )

    dx = merged["x0"].astype(np.float64) - merged["x1"].astype(np.float64)
    dy = merged["y0"].astype(np.float64) - merged["y1"].astype(np.float64)
    dz = merged["z0"].astype(np.float64) - merged["z1"].astype(np.float64)
    merged["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float64)

    a0 = merged["atom_0"].astype("string")
    a1 = merged["atom_1"].astype("string")
    mask_valid = a0.notna() & a1.notna()
    atom_pair = pd.Series("UNK-UNK", index=merged.index, dtype="string")
    a0v = a0[mask_valid]
    a1v = a1[mask_valid]
    atom_pair.loc[mask_valid] = np.where(a0v <= a1v, a0v + "-" + a1v, a1v + "-" + a0v)
    merged["atom_pair"] = atom_pair.astype(str)

    return merged


train_feat = add_distance_and_atoms(train, structures)
test_feat = add_distance_and_atoms(test, structures)

print("train_feat shape:", train_feat.shape, "test_feat shape:", test_feat.shape)
print("NaN dist in train:", int(train_feat["dist"].isna().sum()), "of", len(train_feat))
print("NaN dist in test:", int(test_feat["dist"].isna().sum()), "of", len(test_feat))
print("NaN atom_pair in train:", int(pd.isna(train_feat["atom_pair"]).sum()))
print("NaN atom_pair in test:", int(pd.isna(test_feat["atom_pair"]).sum()))
print(train_feat[["id", "type", "atom_pair", "dist"]].head())



## === cell 3
global_mean = float(train["scalar_coupling_constant"].mean())
type_mean = train.groupby("type")["scalar_coupling_constant"].mean()

train_feat["scalar_coupling_constant"] = train["scalar_coupling_constant"].astype(
    np.float64
)

N_BINS_PER_TYPE = 60  # deterministic; moderate granularity without exploding sparsity
MISSING_BIN_CODE = np.int16(-1)


def compute_type_bin_edges_from_train(
    train_dist: pd.Series, train_type: pd.Series, n_bins: int
) -> dict:
    edges_by_type = {}
    for t, idx in train_type.groupby(train_type, sort=False).groups.items():
        d = train_dist.loc[idx]
        valid = d.notna() & np.isfinite(d.to_numpy(dtype=np.float64, na_value=np.nan))
        if valid.sum() < 2:
            continue
        dv = d.loc[valid.index[valid]].astype(np.float64).to_numpy()
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        e = np.quantile(dv, qs)
        e = np.unique(e)
        if e.size >= 2:
            edges_by_type[t] = e
    return edges_by_type


def _as_float_array(x, na_value=np.nan) -> np.ndarray:
    if isinstance(x, np.ndarray):
        arr = x
    else:
        arr = np.asarray(x)
    if arr.dtype == object:
        out = np.empty(arr.shape, dtype=np.float64)
        for i, v in enumerate(arr):
            try:
                out[i] = float(v)
            except Exception:
                out[i] = na_value
        return out
    return arr.astype(np.float64, copy=False)


def assign_bins_with_precomputed_edges(
    dist_s: pd.Series,
    type_s: pd.Series,
    edges_by_type: dict,
    n_bins: int,
    missing_bin_code: np.int16 = MISSING_BIN_CODE,
) -> pd.Series:
    out = pd.Series(missing_bin_code, index=dist_s.index, dtype=np.int16)
    dist_arr = dist_s.to_numpy(dtype=np.float64, na_value=np.nan)

    for t, idx in type_s.groupby(type_s, sort=False).groups.items():
        edges = edges_by_type.get(t, None)
        if edges is None or len(edges) < 2:
            continue

        idx_labels = pd.Index(idx)
        d = dist_arr[idx_labels.to_numpy(dtype=np.int64, copy=False)]
        valid_mask = np.isfinite(d)
        if valid_mask.sum() == 0:
            continue

        dv = d[valid_mask]

        b = pd.cut(dv, bins=edges, labels=False, include_lowest=True)
        b_np = _as_float_array(b, na_value=np.nan)

        nan_mask = ~np.isfinite(b_np)
        if nan_mask.any():
            b_fill = b_np.copy()
            b_fill[nan_mask & (dv <= edges[0])] = 0.0
            b_fill[nan_mask & (dv >= edges[-1])] = float(len(edges) - 2)
            b_np = b_fill

        b_int = np.clip(np.floor(b_np), 0, len(edges) - 2).astype(np.int16)

        out_rows = idx_labels[valid_mask]
        out.loc[out_rows] = b_int

    return out


edges_by_type = compute_type_bin_edges_from_train(
    train_feat["dist"], train_feat["type"], N_BINS_PER_TYPE
)
train_feat["dist_bin"] = assign_bins_with_precomputed_edges(
    train_feat["dist"], train_feat["type"], edges_by_type, N_BINS_PER_TYPE
)
test_feat["dist_bin"] = assign_bins_with_precomputed_edges(
    test_feat["dist"], test_feat["type"], edges_by_type, N_BINS_PER_TYPE
)

type_atompair_dist_mean = train_feat.groupby(
    ["type", "atom_pair", "dist_bin"], sort=False
)["scalar_coupling_constant"].mean()

type_atompair_mean = train_feat.groupby(["type", "atom_pair"], sort=False)[
    "scalar_coupling_constant"
].mean()

type_dist_mean = train_feat.groupby(["type", "dist_bin"], sort=False)[
    "scalar_coupling_constant"
].mean()

print("Global mean:", global_mean)
print("Per-type means (head):")
print(type_mean.head(10))
print("Per-(type, atom_pair, dist_bin) means (head):")
print(type_atompair_dist_mean.head(10))

k1 = list(
    zip(
        test_feat["type"].values,
        test_feat["atom_pair"].values,
        test_feat["dist_bin"].values,
    )
)
pred = pd.Series(k1, index=test_feat.index).map(type_atompair_dist_mean)

k2 = list(zip(test_feat["type"].values, test_feat["atom_pair"].values))
pred = pred.fillna(pd.Series(k2, index=test_feat.index).map(type_atompair_mean))

k3 = list(zip(test_feat["type"].values, test_feat["dist_bin"].values))
pred = pred.fillna(pd.Series(k3, index=test_feat.index).map(type_dist_mean))

pred = (
    pred.fillna(test_feat["type"].map(type_mean)).fillna(global_mean).astype(np.float64)
)

submission = pd.DataFrame(
    {"id": test["id"].astype(np.int64), "scalar_coupling_constant": pred}
)

submission = submission.sort_values("id").reset_index(drop=True)

if list(submission.columns) != ["id", "scalar_coupling_constant"]:
    raise ValueError("Submission has wrong columns/order")
if submission.isna().any().any():
    raise ValueError("Submission contains NaNs")
if submission.shape[0] != test.shape[0]:
    raise ValueError("Submission row count mismatch with test")

print(submission.head())
print(submission["scalar_coupling_constant"].describe())



## === cell 4
out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(submission))
print(
    "File exists:", os.path.exists(out_path), "size_bytes:", os.path.getsize(out_path)
)
