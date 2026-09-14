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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.33542

# 6. Current score

1.57639

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 2.14492) has done: 'Diagnosis: The crash happens in cell 2 during `reg.predict(test[col])` because `test[col]` contains NaN values, and `ExtraTreesRegressor` in scikit-learn does not accept NaNs. These NaNs are introduced earlier by the left merges with `structures.csv` (missing coordinates for some rows) and/or by `dist_to_type_mean` when a `type` appears in test but not in train (group mean becomes NaN). Since we must not change earlier feature engineering, we need to make the model input finite inside cell 2.

Patch summary: In cell 2, compute the feature columns as before, then fill missing values in both train and test feature matrices with a deterministic constant (0.0) before fitting/predicting. This preserves the existing model, features, and training approach while preventing scikit-learn validation from failing.

Updated cells:'
- What this solution (achieved 2.14492) has done: 'Your current score (2.14492, lower-is-better) is far from the target (0.33542), so we should improve performance but with minimal changes and identical overall approach. The biggest issue in your feature engineering is that `dist_to_type_mean` for test is computed using test-only type means, which is inconsistent with training and can badly distort this key feature. I change this single line to compute test normalization using the training per-type means (and fall back to the global mean if a type is missing), while keeping the same ExtraTrees model and the same feature set. This should materially reduce the error and move the score toward the target without altering the core pipeline.'
- What this solution (achieved 1.58914) has done: 'Your score is much worse than the target (lower-is-better), so we should improve it with the smallest change that fixes a key modeling mismatch. The biggest issue is that you’re fitting one global model across all coupling `type`s; because each type has very different target distributions, this inflates MAE and hurts the per-type log-MAE metric. I keep the exact same features and the same ExtraTreesRegressor, but train/predict one model per `type` and write predictions back in the original test row order. I also keep the existing NaN/inf handling so the pipeline remains robust and still produces a valid `submission.csv`.'
- What this solution (achieved 1.23566) has done: 'Main bottlenecks are (1) repeated `MultiIndex.reindex` lookups for coordinates (4×, each O(N log M)), (2) `groupby(...).transform("mean")` on the full 4.19M rows, and (3) training *two* ExtraTrees models per type during grid search (doubling fit time). The refactor keeps the exact same features and model logic, but replaces the coordinate attachment with a single vectorized join using precomputed integer keys, replaces the groupby-transform with a map from precomputed means, and makes the validation loop reuse fitted models (no extra re-fit). These changes are provably equivalent (same values produced) while cutting a large amount of overhead and redundant training work.'
- What this solution (achieved 1.54256) has done: 'I speed up the run by eliminating the per-type hyperparameter search (which doubles model fits and is the dominant cost) while keeping the same model family, per-type training scheme, and features; we use the already-intended “best” setting directly (100 trees). I also reduce overhead in feature assembly by building a compact float32 feature matrix without repeated pandas conversions, and I keep the fast coordinate attach but avoid extra category/string churn. Finally, I enable Intel-accelerated scikit-learn (available via scikit-learn-intelex) to accelerate tree training/prediction without changing semantics.'
- What this solution (achieved 1.57639) has done: 'Diagnosis: Cell 1 fails because `train["type"]` is a pandas `category`, and in pandas 2.x `Series.map()` with a categorical key can yield a `Categorical` result when mapping to a Series with a categorical index. That makes the denominator categorical, so `train["dist"] / ...` triggers `TypeError: Object with dtype category cannot perform the numpy op divide`. The same risk exists for the test mapping path.  

Patch summary: In cell 1, force the mapped means to be numeric (`float32`) by mapping using `type.astype(str)` against a string-indexed mean Series and explicitly converting the result to a NumPy float array before division. This keeps the feature semantics identical (distance divided by per-type mean distance, with a global fallback for unseen types).  

Updated cells: Only cell 1 is changed.  

Compatibility notes for cell k+1: The created columns `dist` and `dist_to_type_mean` remain present with numeric dtypes and the same names, so cell 2’s feature selection and `to_numpy(dtype=np.float32)` continue to work unchanged.  

Assumptions: The `type` values are consistently representable as strings like in the original code, and no downstream code relies on `dist_to_type_mean` being categorical (it should be numeric).'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from sklearn import preprocessing, ensemble

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

np.random.seed(4)

TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SUB_PATH = "../input/sample_submission.csv"
STRUCT_PATH = "../input/structures.csv"

train = pd.read_csv(
    TRAIN_PATH,
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
        "scalar_coupling_constant": np.float64,
    },
)
test = pd.read_csv(
    TEST_PATH,
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
    },
)
sub = pd.read_csv(
    SUB_PATH, dtype={"id": np.int32, "scalar_coupling_constant": np.float64}
)
print(train.shape, test.shape, sub.shape)

train_type_str = train["type"].astype(str)
test_type_str = test["type"].astype(str)

train["atom1"] = train_type_str.str[2]
train["atom2"] = train_type_str.str[3]
test["atom1"] = test_type_str.str[2]
test["atom2"] = test_type_str.str[3]

lbl = preprocessing.LabelEncoder()
for i in range(4):
    trn_ch = train_type_str.str[i]
    tst_ch = test_type_str.str[i]
    train[f"type{i}"] = lbl.fit_transform(trn_ch)
    test[f"type{i}"] = lbl.transform(tst_ch)

structures = pd.read_csv(
    STRUCT_PATH,
    dtype={
        "molecule_name": "category",
        "atom_index": np.int16,
        "atom": "category",
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
    },
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
)

all_molecules = (
    pd.Index(train["molecule_name"].astype(str).to_numpy())
    .append(pd.Index(test["molecule_name"].astype(str).to_numpy()))
    .append(pd.Index(structures["molecule_name"].astype(str).to_numpy()))
    .unique()
)
shared_mol_dtype = pd.api.types.CategoricalDtype(
    categories=all_molecules, ordered=False
)
train["molecule_name"] = train["molecule_name"].astype(str).astype(shared_mol_dtype)
test["molecule_name"] = test["molecule_name"].astype(str).astype(shared_mol_dtype)
structures["molecule_name"] = (
    structures["molecule_name"].astype(str).astype(shared_mol_dtype)
)


def _attach_coords_fast(df: pd.DataFrame, idx_col: str, suffix: str) -> pd.DataFrame:
    mol_code = df["molecule_name"].cat.codes.to_numpy(np.int32, copy=False)
    atom_idx = df[idx_col].to_numpy(np.int32, copy=False)

    cache = _attach_coords_fast.__dict__.setdefault("_cache", {})
    if not cache:
        smol = structures["molecule_name"].cat.codes.to_numpy(np.int32, copy=False)
        sidx = structures["atom_index"].to_numpy(np.int32, copy=False)

        stride = int(sidx.max()) + 1
        cache["stride"] = stride

        skey = smol.astype(np.int64) * stride + sidx.astype(np.int64)
        order = np.argsort(skey, kind="mergesort")
        cache["skey_sorted"] = skey[order]
        cache["x_sorted"] = structures["x"].to_numpy(copy=False)[order]
        cache["y_sorted"] = structures["y"].to_numpy(copy=False)[order]
        cache["z_sorted"] = structures["z"].to_numpy(copy=False)[order]

    stride = cache["stride"]
    key = mol_code.astype(np.int64) * stride + atom_idx.astype(np.int64)

    skey_sorted = cache["skey_sorted"]
    pos = np.searchsorted(skey_sorted, key)

    n = skey_sorted.size
    in_bounds = pos < n
    found = np.zeros_like(in_bounds, dtype=bool)
    if np.any(in_bounds):
        found[in_bounds] = skey_sorted[pos[in_bounds]] == key[in_bounds]

    x_out = np.full(len(df), np.nan, dtype=np.float32)
    y_out = np.full(len(df), np.nan, dtype=np.float32)
    z_out = np.full(len(df), np.nan, dtype=np.float32)

    if np.any(found):
        p = pos[found]
        x_out[found] = cache["x_sorted"][p]
        y_out[found] = cache["y_sorted"][p]
        z_out[found] = cache["z_sorted"][p]

    df[f"x{suffix}"] = x_out
    df[f"y{suffix}"] = y_out
    df[f"z{suffix}"] = z_out
    return df


train = _attach_coords_fast(train, "atom_index_0", "0")
test = _attach_coords_fast(test, "atom_index_0", "0")
train = _attach_coords_fast(train, "atom_index_1", "1")
test = _attach_coords_fast(test, "atom_index_1", "1")

del structures

print(train.shape, test.shape, sub.shape)



## === cell 1
train_p0 = train[["x0", "y0", "z0"]].to_numpy(dtype=np.float32, copy=False)
train_p1 = train[["x1", "y1", "z1"]].to_numpy(dtype=np.float32, copy=False)
test_p0 = test[["x0", "y0", "z0"]].to_numpy(dtype=np.float32, copy=False)
test_p1 = test[["x1", "y1", "z1"]].to_numpy(dtype=np.float32, copy=False)

train["dist"] = np.linalg.norm(train_p0 - train_p1, axis=1)
test["dist"] = np.linalg.norm(test_p0 - test_p1, axis=1)

type_mean_train = train.groupby("type", sort=False)["dist"].mean()
global_mean_train = float(train["dist"].mean())

type_mean_train_str = type_mean_train.copy()
type_mean_train_str.index = type_mean_train_str.index.astype(str)

train_type_mean = (
    train["type"]
    .astype(str)
    .map(type_mean_train_str)
    .to_numpy(dtype=np.float32, copy=False)
)
train["dist_to_type_mean"] = (
    train["dist"].to_numpy(dtype=np.float32, copy=False) / train_type_mean
)

test_type_mean = (
    test["type"].astype(str).map(type_mean_train_str).fillna(global_mean_train)
).to_numpy(dtype=np.float32, copy=False)
test["dist_to_type_mean"] = (
    test["dist"].to_numpy(dtype=np.float32, copy=False) / test_type_mean
)


## === cell 2
col = [
    c
    for c in train.columns
    if c
    not in [
        "id",
        "molecule_name",
        "scalar_coupling_constant",
        "type",
        "atom1",
        "atom2",
        "atom_index_0",
        "atom_index_1",
    ]
]

X_train_all = train.loc[:, col].to_numpy(dtype=np.float32, copy=False)
X_test_all = test.loc[:, col].to_numpy(dtype=np.float32, copy=False)

np.nan_to_num(X_train_all, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
np.nan_to_num(X_test_all, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

y_train_all = train["scalar_coupling_constant"].to_numpy(copy=False)

rng = np.random.RandomState(4)
mol_codes = train["molecule_name"].cat.codes.to_numpy(np.int32, copy=False)
unique_mol_codes = np.unique(mol_codes)
rng.shuffle(unique_mol_codes)
n_val = max(1, int(0.05 * len(unique_mol_codes)))
val_mol_codes = unique_mol_codes[:n_val]
is_val = np.isin(mol_codes, val_mol_codes, assume_unique=False)

test_pred = np.empty(len(test), dtype=np.float64)

BEST_N_ESTIMATORS = 100

train_type_codes, type_uniques = pd.factorize(train["type"].astype(str), sort=False)
test_type_codes = pd.Categorical(
    test["type"].astype(str), categories=type_uniques
).codes

train_idx_by_type = [
    np.flatnonzero(train_type_codes == k) for k in range(len(type_uniques))
]
test_idx_by_type = [
    np.flatnonzero(test_type_codes == k) for k in range(len(type_uniques))
]

for k, t in enumerate(type_uniques):
    idx_full = train_idx_by_type[k]
    if idx_full.size == 0:
        continue

    idx_te = test_idx_by_type[k]

    X_full = X_train_all[idx_full]
    y_full = y_train_all[idx_full]
    reg = ensemble.ExtraTreesRegressor(
        n_jobs=-1, n_estimators=BEST_N_ESTIMATORS, random_state=4
    )
    reg.fit(X_full, y_full)

    if idx_te.size:
        test_pred[idx_te] = reg.predict(X_test_all[idx_te])

test["scalar_coupling_constant"] = test_pred
test[["id", "scalar_coupling_constant"]].to_csv("submission.csv", index=False)
print(
    "Wrote submission.csv with shape:", test[["id", "scalar_coupling_constant"]].shape
)
