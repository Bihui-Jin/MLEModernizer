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

category_encoders==2.7.0
geopandas==0.14.4
lightgbm==4.6.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.7869

# 6. Current score

1.18497

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 2.42039) has done: 'The test row mismatch is caused by the merges with `structures` using `sort=True`, which reorders rows and can also silently drop rows if any merge keys don’t match, so `X_test_new` no longer aligns 1:1 with `test`. I make the merges stable and non-dropping by using `how="left"` and removing `sort=True`, and I restore alignment by carrying `id` through feature engineering and reindexing back to the original `row_id` order before training/predicting. This fixes the runtime assertions and ensures `y_predict` is defined so a valid `submission.csv` is always written. The modeling logic (LightGBM regressor + same feature set) is preserved; changes are strictly to fix row alignment/stability.'
- What this solution (achieved 1.68) has done: 'Your current score (2.42039, lower is better) is far from the target (0.7869), so we should improve performance while keeping the same core approach (LightGBM regressor on engineered features). The biggest gain with minimal logic change is to avoid the “single global model” mismatch across coupling `type`: train one LightGBM model per `type` (same model class, same fit/predict loop concept) and predict test rows for that type, which aligns with the metric being averaged per type and typically drops error substantially. I also switch to LightGBM’s native categorical handling (keep your category dtypes and do not one-hot), and I add a safe median fallback per type for any rare edge case where a type might be missing in training. The rest of your feature engineering, row alignment, and submission writing stays intact.'
- What this solution (achieved 1.68285) has done: 'You’re still far from the target (lower is better), so we should improve accuracy while keeping the same core LightGBM-per-`type` approach and the same engineered features. The largest remaining avoidable error source is that your training/validation logic ignores the molecule-wise split requirement, which can make generalization worse and destabilize performance; we switch CV (and training fit behavior) to be molecule-group aware without changing the model class or feature set. We also add a minimal, metric-aligned log-MAE monitoring via a GroupKFold CV printout per type to sanity-check that we’re moving toward the target, and we slightly regularize LightGBM (still the same model) to reduce overfit that hurts the molecule-split test. Everything still writes a valid `submission.csv` with correct row alignment and schema.'
- What this solution (achieved 1.71436) has done: 'Your current score is much worse than the target (lower is better), so we should improve generalization without changing the overall approach (same LightGBM per `type`, same engineered features). The largest avoidable issue is that `num_atoms` is currently computed from only `atom_index_0` inside train/test separately, which is noisy and inconsistent across splits; we instead compute `num_atoms` from `structures.csv` once per molecule and merge it into both train/test. Next, we ensure categorical columns are aligned between train/test by making test categories match train categories, which reduces unpredictable handling of unseen categories and improves stability. Finally, we add a minimal, metric-aligned per-type target transform: fit on `scalar_coupling_constant - median(type)` and add the median back at prediction time, which often reduces per-type MAE without changing the model class or training loop structure.'
- What this solution (achieved 1.18497) has done: 'We fix two blocking bugs so the notebook runs end-to-end and always writes a valid `submission.csv`. First, the `num_atoms` merge fails because some molecules produce NaNs during the groupby/max pipeline; we compute `num_atoms` robustly from `structures` and fill any missing values before casting to int. Second, LightGBM errors because `to_numpy()` includes categorical/string columns (e.g., `molecule_name`), so we keep the exact same per-`type` LightGBM training logic but feed LightGBM a pandas DataFrame (with categorical dtypes) instead of a float NumPy array, letting LightGBM handle categorical features properly. These changes are correctness/stability fixes and should also improve score vs the broken pipeline by enabling the intended model to train/predict.'
- What this solution (achieved 1.18497) has done: 'Your score is still worse than the target (lower is better), so we should make small, metric-aligned improvements without changing the core approach (LightGBM per `type` with the same engineered features). The biggest low-risk gain is to add a few standard, cheap geometric features (squared distance and inverse distance) that help the model capture non-linear distance effects while staying within the same feature-engineering paradigm. We also reduce a known source of instability by enforcing consistent `category` dtypes for `atom_0/atom_1` based on the global `structures["atom"]` categories (instead of creating fresh Categoricals from arrays), which improves categorical handling across train/test and per-type training. Finally, we add a tiny amount of additional regularization (`min_child_samples`) and slightly more boosting rounds to improve generalization toward the molecule-split test, while keeping the same training loop and objective.'
- What this solution (achieved 1.18497) has done: 'We’re still significantly worse than the target (current 1.18497 vs target 0.7869, lower is better), so the smallest score-improving change is to stop throwing away the most predictive categorical signal: the per-`type` model currently drops the `type` feature entirely (even though the original feature set includes it). I keep the same “one LightGBM model per type” core approach and the same target-centering by type, but include the `type` column as a constant categorical feature inside each type-specific training set (and likewise in test), which helps the tree split structure remain consistent and improves calibration with essentially no architectural change. I also add `molecule_name` back into the feature set (it was being dropped implicitly by selecting only numeric columns earlier), leveraging LightGBM categorical handling; this is a minimal change that often materially improves this competition because molecules have characteristic coupling patterns. Finally, I keep all alignment guarantees and submission writing unchanged.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing
import gc
import lightgbm as lgbm
from sklearn.model_selection import GroupKFold
import os

print(os.listdir("../input"))



## === cell 1
gc.collect()



## === cell 2
BASE_PATH = "../input"
if os.path.exists(os.path.join(BASE_PATH, "champs-scalar-coupling", "train.csv")):
    DATA_PATH = os.path.join(BASE_PATH, "champs-scalar-coupling")
else:
    DATA_PATH = BASE_PATH

train_dtypes = {
    "id": np.int32,
    "molecule_name": "category",
    "atom_index_0": np.int16,
    "atom_index_1": np.int16,
    "type": "category",
    "scalar_coupling_constant": np.float32,
}
test_dtypes = {
    "id": np.int32,
    "molecule_name": "category",
    "atom_index_0": np.int16,
    "atom_index_1": np.int16,
    "type": "category",
}
structures_dtypes = {
    "molecule_name": "category",
    "atom_index": np.int16,
    "atom": "category",
    "x": np.float32,
    "y": np.float32,
    "z": np.float32,
}
mulliken_dtypes = {
    "molecule_name": "category",
    "atom_index": np.int16,
    "mulliken_charge": np.float32,
}
shield_dtypes = {
    "molecule_name": "category",
    "atom_index": np.int16,
    "XX": np.float32,
    "YX": np.float32,
    "ZX": np.float32,
    "XY": np.float32,
    "YY": np.float32,
    "ZY": np.float32,
    "XZ": np.float32,
    "YZ": np.float32,
    "ZZ": np.float32,
}

train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"), dtype=train_dtypes)
test = pd.read_csv(os.path.join(DATA_PATH, "test.csv"), dtype=test_dtypes)

sample_sub = pd.read_csv(
    os.path.join(DATA_PATH, "sample_submission.csv"),
    usecols=["id", "scalar_coupling_constant"],
)

structures = pd.read_csv(
    os.path.join(DATA_PATH, "structures.csv"),
    dtype=structures_dtypes,
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
)
mulliken = pd.read_csv(
    os.path.join(DATA_PATH, "mulliken_charges.csv"),
    dtype=mulliken_dtypes,
    usecols=["molecule_name", "atom_index", "mulliken_charge"],
)
shield = pd.read_csv(
    os.path.join(DATA_PATH, "magnetic_shielding_tensors.csv"), dtype=shield_dtypes
)

print(f"train.shape: {train.shape}")
print(f"test.shape: {test.shape}")
print(f"structures.shape: {structures.shape}")
print(f"mulliken.shape: {mulliken.shape}")
print(f"shield.shape: {shield.shape}")



## === cell 3
X_train = train.drop(columns=["scalar_coupling_constant"]).copy()
y_train = train["scalar_coupling_constant"].copy()
X_test = test.copy()

print(f"X_train.shape: {X_train.shape}")
print(f"X_test.shape: {X_test.shape}")



## === cell 4
train_id = X_train["id"].copy()
test_id = X_test["id"].copy()



## === cell 5
X_train = X_train.copy()
X_test = X_test.copy()
X_train["row_id"] = np.arange(len(X_train), dtype=np.int32)
X_test["row_id"] = np.arange(len(X_test), dtype=np.int32)




## === cell 6
def convert_object_to_categories(X_train, X_test):
    for col in X_train.columns:
        if X_train[col].dtype == "O":
            X_train[col] = X_train[col].astype("category")
            if col in X_test.columns:
                X_test[col] = X_test[col].astype("category")
    for col in X_test.columns:
        if X_test[col].dtype == "O":
            X_test[col] = X_test[col].astype("category")
    return X_train, X_test


X_train, X_test = convert_object_to_categories(X_train, X_test)




## === cell 7
def calc_score_by_type(df_type, y_true, y_pred):
    err = np.mean(np.abs(y_true - y_pred))
    return float(np.log(err + 1e-12))


def calc_overall_score(types, y_true, y_pred):
    tmp = pd.DataFrame({"type": types.astype(str).values, "y": y_true, "p": y_pred})
    per_type = tmp.groupby("type").apply(
        lambda g: calc_score_by_type(g["type"], g["y"].values, g["p"].values)
    )
    return float(per_type.mean())


print(f"{X_train['type'].unique()}")
print(f"{X_test['type'].unique()}")




## === cell 8
def cross_val_grouped_by_molecule(X, y, groups, cat_cols, n_splits=3, lgb_params=None):
    if lgb_params is None:
        lgb_params = {}
    gkf = GroupKFold(n_splits=n_splits)

    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        Xtr = X.iloc[tr_idx]
        ytr = y.iloc[tr_idx]
        Xva = X.iloc[va_idx]
        yva = y.iloc[va_idx]

        model = lgbm.LGBMRegressor(**lgb_params)
        model.fit(Xtr, ytr, categorical_feature=cat_cols)
        pva = model.predict(Xva)

        score = calc_overall_score(Xva["type"], yva.values, pva)
        fold_scores.append(score)
        print(f"GroupKFold fold {fold} logMAE-by-type score: {score:.5f}")

    print(f"Mean CV score (GroupKFold): {np.mean(fold_scores):.5f}")
    return float(np.mean(fold_scores))




## === cell 9
def _pack_key(mol_cat_codes: np.ndarray, atom_index: np.ndarray) -> np.ndarray:
    return (mol_cat_codes.astype(np.int64) << 20) | atom_index.astype(np.int64)


all_mols = pd.Categorical(
    pd.concat(
        [train["molecule_name"], test["molecule_name"], structures["molecule_name"]],
        axis=0,
    ),
    ordered=False,
)
mol_categories = all_mols.categories

for df in (train, test, structures, mulliken, shield, X_train, X_test):
    if "molecule_name" in df.columns:
        df["molecule_name"] = df["molecule_name"].cat.set_categories(mol_categories)

mol_codes_struct = structures["molecule_name"].cat.codes.to_numpy(np.int32, copy=False)
atom_idx_struct = structures["atom_index"].to_numpy(np.int16, copy=False)
k_struct = _pack_key(mol_codes_struct, atom_idx_struct)
order_struct = np.argsort(k_struct, kind="mergesort")
k_struct_sorted = k_struct[order_struct]

x_s = structures["x"].to_numpy(np.float32, copy=False)[order_struct]
y_s = structures["y"].to_numpy(np.float32, copy=False)[order_struct]
z_s = structures["z"].to_numpy(np.float32, copy=False)[order_struct]
atom_s = structures["atom"].to_numpy(copy=False)[order_struct]


def _lookup_struct_features(mol_codes: np.ndarray, atom_idx: np.ndarray):
    keys = _pack_key(mol_codes, atom_idx)
    pos = np.searchsorted(k_struct_sorted, keys)
    found = (pos < k_struct_sorted.size) & (k_struct_sorted[pos] == keys)
    x = np.full(keys.shape[0], np.nan, dtype=np.float32)
    y = np.full(keys.shape[0], np.nan, dtype=np.float32)
    z = np.full(keys.shape[0], np.nan, dtype=np.float32)
    a = np.empty(keys.shape[0], dtype=object)
    a[:] = None
    if found.any():
        pf = pos[found]
        x[found] = x_s[pf]
        y[found] = y_s[pf]
        z[found] = z_s[pf]
        a[found] = atom_s[pf]
    return x, y, z, a


mol_codes_train = X_train["molecule_name"].cat.codes.to_numpy(np.int32, copy=False)
mol_codes_test = X_test["molecule_name"].cat.codes.to_numpy(np.int32, copy=False)

a0_train = X_train["atom_index_0"].to_numpy(np.int16, copy=False)
a0_test = X_test["atom_index_0"].to_numpy(np.int16, copy=False)
x0, y0, z0, atom0 = _lookup_struct_features(mol_codes_train, a0_train)
X_train["atom_index_0_x"] = x0
X_train["atom_index_0_y"] = y0
X_train["atom_index_0_z"] = z0
X_train["atom_0"] = pd.Categorical(atom0, categories=structures["atom"].cat.categories)

x0, y0, z0, atom0 = _lookup_struct_features(mol_codes_test, a0_test)
X_test["atom_index_0_x"] = x0
X_test["atom_index_0_y"] = y0
X_test["atom_index_0_z"] = z0
X_test["atom_0"] = pd.Categorical(atom0, categories=structures["atom"].cat.categories)

a1_train = X_train["atom_index_1"].to_numpy(np.int16, copy=False)
a1_test = X_test["atom_index_1"].to_numpy(np.int16, copy=False)
x1, y1, z1, atom1 = _lookup_struct_features(mol_codes_train, a1_train)
X_train["atom_index_1_x"] = x1
X_train["atom_index_1_y"] = y1
X_train["atom_index_1_z"] = z1
X_train["atom_1"] = pd.Categorical(atom1, categories=structures["atom"].cat.categories)

x1, y1, z1, atom1 = _lookup_struct_features(mol_codes_test, a1_test)
X_test["atom_index_1_x"] = x1
X_test["atom_index_1_y"] = y1
X_test["atom_index_1_z"] = z1
X_test["atom_1"] = pd.Categorical(atom1, categories=structures["atom"].cat.categories)



## === cell 10
pass



## === cell 11
X_train.head()



## === cell 12
dx = X_train["atom_index_0_x"].to_numpy() - X_train["atom_index_1_x"].to_numpy()
dy = X_train["atom_index_0_y"].to_numpy() - X_train["atom_index_1_y"].to_numpy()
dz = X_train["atom_index_0_z"].to_numpy() - X_train["atom_index_1_z"].to_numpy()
dist2 = dx * dx + dy * dy + dz * dz
X_train["distance"] = np.sqrt(dist2)
X_train["distance2"] = dist2.astype(np.float32)
X_train["inv_distance"] = (1.0 / (X_train["distance"].to_numpy() + 1e-6)).astype(
    np.float32
)

dx = X_test["atom_index_0_x"].to_numpy() - X_test["atom_index_1_x"].to_numpy()
dy = X_test["atom_index_0_y"].to_numpy() - X_test["atom_index_1_y"].to_numpy()
dz = X_test["atom_index_0_z"].to_numpy() - X_test["atom_index_1_z"].to_numpy()
dist2 = dx * dx + dy * dy + dz * dz
X_test["distance"] = np.sqrt(dist2)
X_test["distance2"] = dist2.astype(np.float32)
X_test["inv_distance"] = (1.0 / (X_test["distance"].to_numpy() + 1e-6)).astype(
    np.float32
)



## === cell 13
X_train["join_type"] = X_train["type"].astype(str).str.slice(0, 2)
X_test["join_type"] = X_test["type"].astype(str).str.slice(0, 2)



## === cell 14
print(X_train["atom_0"].unique())
print(X_test["atom_0"].unique())



## === cell 15
mol_num_atoms = (
    structures.groupby("molecule_name", sort=False)["atom_index"]
    .max()
    .astype("float32")
    .add(1.0)
    .fillna(0.0)
    .astype(np.int32)
    .rename("num_atoms")
    .reset_index()
)

X_train = X_train.merge(mol_num_atoms, on="molecule_name", how="left")
X_test = X_test.merge(mol_num_atoms, on="molecule_name", how="left")

X_train["num_atoms"] = X_train["num_atoms"].fillna(0).astype(np.int32)
X_test["num_atoms"] = X_test["num_atoms"].fillna(0).astype(np.int32)



## === cell 16
mol_codes_mul = mulliken["molecule_name"].cat.codes.to_numpy(np.int32, copy=False)
atom_idx_mul = mulliken["atom_index"].to_numpy(np.int16, copy=False)
k_mul = _pack_key(mol_codes_mul, atom_idx_mul)
order_mul = np.argsort(k_mul, kind="mergesort")
k_mul_sorted = k_mul[order_mul]
mul_charge_sorted = mulliken["mulliken_charge"].to_numpy(np.float32, copy=False)[
    order_mul
]


def _lookup_mulliken(mol_codes: np.ndarray, atom_idx: np.ndarray) -> np.ndarray:
    keys = _pack_key(mol_codes, atom_idx)
    pos = np.searchsorted(k_mul_sorted, keys)
    found = (pos < k_mul_sorted.size) & (k_mul_sorted[pos] == keys)
    out = np.full(keys.shape[0], np.nan, dtype=np.float32)
    if found.any():
        out[found] = mul_charge_sorted[pos[found]]
    return out


X_train["mulliken_charge_0"] = _lookup_mulliken(mol_codes_train, a0_train)
X_train["mulliken_charge_1"] = _lookup_mulliken(mol_codes_train, a1_train)
X_test["mulliken_charge_0"] = _lookup_mulliken(mol_codes_test, a0_test)
X_test["mulliken_charge_1"] = _lookup_mulliken(mol_codes_test, a1_test)



## === cell 17
shield_cols = ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]
mol_codes_sh = shield["molecule_name"].cat.codes.to_numpy(np.int32, copy=False)
atom_idx_sh = shield["atom_index"].to_numpy(np.int16, copy=False)
k_sh = _pack_key(mol_codes_sh, atom_idx_sh)
order_sh = np.argsort(k_sh, kind="mergesort")
k_sh_sorted = k_sh[order_sh]
shield_mat_sorted = shield[shield_cols].to_numpy(np.float32, copy=False)[order_sh]


def _lookup_shield(mol_codes: np.ndarray, atom_idx: np.ndarray) -> np.ndarray:
    keys = _pack_key(mol_codes, atom_idx)
    pos = np.searchsorted(k_sh_sorted, keys)
    found = (pos < k_sh_sorted.size) & (k_sh_sorted[pos] == keys)
    out = np.full((keys.shape[0], len(shield_cols)), np.nan, dtype=np.float32)
    if found.any():
        out[found, :] = shield_mat_sorted[pos[found], :]
    return out


sh0 = _lookup_shield(mol_codes_train, a0_train)
sh1 = _lookup_shield(mol_codes_train, a1_train)
for i, c in enumerate(shield_cols):
    X_train[f"{c}_0"] = sh0[:, i]
    X_train[f"{c}_1"] = sh1[:, i]

sh0 = _lookup_shield(mol_codes_test, a0_test)
sh1 = _lookup_shield(mol_codes_test, a1_test)
for i, c in enumerate(shield_cols):
    X_test[f"{c}_0"] = sh0[:, i]
    X_test[f"{c}_1"] = sh1[:, i]



## === cell 18
X_train, X_test = convert_object_to_categories(X_train, X_test)



## === cell 19
for c in X_train.columns:
    if (
        str(X_train[c].dtype) == "category"
        and c in X_test.columns
        and str(X_test[c].dtype) == "category"
    ):
        X_test[c] = X_test[c].cat.set_categories(X_train[c].cat.categories)



## === cell 20
X_train_new = X_train.set_index("row_id", drop=True)
X_test_new = X_test.set_index("row_id", drop=True)



## === cell 21
num_cols_train = X_train_new.select_dtypes(include=[np.number]).columns
num_cols_test = X_test_new.select_dtypes(include=[np.number]).columns
X_train_new[num_cols_train] = X_train_new[num_cols_train].fillna(0.0)
X_test_new[num_cols_test] = X_test_new[num_cols_test].fillna(0.0)

if "scalar_coupling_constant" in X_train_new.columns:
    X_train_new = X_train_new.drop(columns=["scalar_coupling_constant"])
if "scalar_coupling_constant" in X_test_new.columns:
    X_test_new = X_test_new.drop(columns=["scalar_coupling_constant"])

if "id" in X_train_new.columns:
    X_train_new = X_train_new.drop(columns=["id"])
if "id" in X_test_new.columns:
    X_test_new = X_test_new.drop(columns=["id"])

X_test_new = X_test_new.reindex(columns=X_train_new.columns)

assert X_train_new.shape[0] == y_train.shape[0], "Train features/target row mismatch."
assert X_test_new.shape[0] == test.shape[0], "Test row mismatch after processing."
assert X_test_new.shape[1] > 0, "No feature columns available after alignment."



## === cell 22
cat_cols = [c for c in X_train_new.columns if str(X_train_new[c].dtype) == "category"]

lgb_params = dict(
    objective="regression_l1",
    n_estimators=1600,
    learning_rate=0.05,
    num_leaves=128,
    max_depth=-1,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
)

print("Skipping CV to fit 600s runtime budget (does not affect final predictions).")



## === cell 23
y_pred = np.zeros(X_test_new.shape[0], dtype=np.float64)

type_medians = train.groupby("type")["scalar_coupling_constant"].median().to_dict()
global_median = float(train["scalar_coupling_constant"].median())

train_types = X_train_new["type"].astype(str).to_numpy()
test_types = X_test_new["type"].astype(str).to_numpy()

feature_cols = list(X_train_new.columns)  # includes 'type' + 'molecule_name'
cat_cols_in_model = cat_cols

train_order = np.argsort(train_types, kind="mergesort")
train_types_sorted = train_types[train_order]
train_unique, train_start = np.unique(train_types_sorted, return_index=True)
train_end = np.r_[train_start[1:], train_types_sorted.size]

test_order = np.argsort(test_types, kind="mergesort")
test_types_sorted = test_types[test_order]
test_unique, test_start = np.unique(test_types_sorted, return_index=True)
test_end = np.r_[test_start[1:], test_types_sorted.size]

train_type_to_slice = {
    t: (train_start[i], train_end[i]) for i, t in enumerate(train_unique)
}

base_params = dict(lgb_params)
n_estimators = int(base_params.pop("n_estimators"))
base_params.setdefault("verbosity", -1)

base_params.setdefault("seed", 42)
base_params.setdefault("feature_fraction_seed", 42)
base_params.setdefault("bagging_seed", 42)
base_params.setdefault("data_random_seed", 42)
base_params.setdefault("deterministic", True)
base_params.setdefault("force_row_wise", True)

X_train_base = X_train_new[feature_cols]
X_test_base = X_test_new[feature_cols]
y_train_arr = y_train.to_numpy(copy=False).astype(np.float64, copy=False)

ref_train = lgbm.Dataset(
    X_train_base.iloc[:1, :],
    label=y_train_arr[:1],
    categorical_feature=cat_cols_in_model,
    free_raw_data=False,
)

for i, t in enumerate(test_unique):
    s0, s1 = test_start[i], test_end[i]
    test_idx = test_order[s0:s1]

    median_t = float(type_medians.get(t, global_median))
    sl = train_type_to_slice.get(t, None)
    if sl is None:
        y_pred[test_idx] = median_t
        continue

    tr_s0, tr_s1 = sl
    train_idx = train_order[tr_s0:tr_s1]

    Xtr = X_train_base.iloc[train_idx, :]
    ytr = y_train_arr[train_idx] - median_t
    Xte = X_test_base.iloc[test_idx, :]

    dtrain = lgbm.Dataset(
        Xtr,
        label=ytr,
        categorical_feature=cat_cols_in_model,
        free_raw_data=False,
    )
    dtrain.set_reference(ref_train)

    booster = lgbm.train(
        params=base_params,
        train_set=dtrain,
        num_boost_round=n_estimators,
    )
    y_pred[test_idx] = booster.predict(Xte, num_iteration=n_estimators) + median_t

y_predict = y_pred



## === cell 24
sub = pd.DataFrame({"id": test_id.values, "scalar_coupling_constant": y_predict})
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))
print("NaNs in predictions:", np.isnan(sub["scalar_coupling_constant"]).sum())
assert (
    sub.shape[0] == sample_sub.shape[0]
), "Submission row count mismatch vs sample_submission."
assert list(sub.columns) == [
    "id",
    "scalar_coupling_constant",
], "Submission columns incorrect."
