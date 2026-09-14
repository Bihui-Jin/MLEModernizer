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

1.24022

# 6. Current score

2.96913

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 2.16541) has done: 'We fix the Ridge prediction error by ensuring train/test feature matrices have no NaNs after the merges and distance calculation (left-joins can create missing coordinates/atoms). The minimal score-neutral approach is to impute missing numeric features with the training median and missing categorical values with a constant before one-hot encoding, then align columns exactly. We also make the CV helper correct (it currently passes an integer instead of a CV splitter) while keeping the same Ridge model and training flow. Finally, we guarantee a valid `submission.csv` is written with the required columns.'
- What this solution (achieved 2.16556) has done: 'I fix the crash in cell 17 by ensuring the post–one-hot-encoding matrices are strictly numeric before checking for NaNs (the error indicates some columns remain non-numeric/object). I also make sure `train` and `test` are aligned and converted to `float32` consistently so Ridge and CV run without dtype issues, without changing the model or feature logic. Finally, I keep the same training/inference flow and guarantee a correctly formatted `submission.csv` is written. These fixes are execution/stability-oriented and should be score-neutral to slightly positive (by preventing silent dtype/object issues).'
- What this solution (achieved 2.16549) has done: 'I fix the categorical `fillna(0)` bug by only filling numeric columns and then one-hot encoding, which prevents Pandas from trying to insert `0` into categorical dtypes. I also ensure no string/categorical columns (like `molecule_name`) survive into the model matrix by explicitly dropping `molecule_name` after saving the grouping vector, so `X.to_numpy(float32)` works. Finally, I keep the same Ridge + GroupKFold CV flow but make the feature alignment/NaN handling consistent so the notebook runs end-to-end and always writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 2.95947) has done: 'You’re currently far from the target (2.16549 vs 1.24022; lower is better), so we need a modest, legitimate improvement without changing the core “Ridge on simple geometry + one-hot atoms/types” approach. The biggest score win with minimal semantic change is to train **separate Ridge models per coupling `type`** (the metric is averaged per type, and different types have very different scales), while keeping the same features and Ridge regression. To preserve the same training flow, we keep GroupKFold but run it **within each type**, pick an alpha per type from the same grid, then fit on full data of that type and predict only that type in test. This typically reduces log(MAE) substantially versus a single global Ridge, while staying within the same model family and feature set.'
- What this solution (achieved 2.96913) has done: 'We keep the exact same Ridge-per-type approach and feature set, but fix a key mismatch with the competition metric: your CV currently computes `log(MAE)` on a single type subset, while the real metric uses `log(mean_abs_error)` with the *mean taken before the log* (equivalently `log1p`? no, `log(mae)`), but averaged across types on the full set. For per-type tuning, the correct objective is simply `log(MAE)` for that type, i.e., compute MAE for that fold/type then take log; your current `champs_metric` averages logs across “types” inside the fold, which becomes a constant/no-op per type subset but can still behave oddly with the 1e-12 stabilization. We replace the per-type CV scoring with a direct per-type `log(mae)` (still legitimate, same evaluation semantics), use a slightly wider alpha grid that stays in the same Ridge family, and (most importantly) train/predict in float64 internally for numerical stability while keeping outputs float32—these are minimal changes that should improve score from 2.959 toward 1.240 without changing the modeling approach. The script still writes a valid `submission.csv` with required columns.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os

INPUT_DIR = (
    "/kaggle/input/champs-scalar-coupling"
    if os.path.exists("/kaggle/input/champs-scalar-coupling")
    else "../input"
)
print("INPUT_DIR =", INPUT_DIR)
print("Top-level /kaggle/input exists:", os.path.exists("/kaggle/input"))
if os.path.exists("/kaggle/input"):
    print("Available datasets under /kaggle/input:", os.listdir("/kaggle/input")[:20])



## === cell 1
structures = pd.read_csv(
    f"{INPUT_DIR}/structures.csv",
    dtype={
        "molecule_name": "category",
        "atom_index": "int16",
        "atom": "category",
        "x": "float32",
        "y": "float32",
        "z": "float32",
    },
)
train = pd.read_csv(
    f"{INPUT_DIR}/train.csv",
    dtype={
        "id": "int32",
        "molecule_name": "category",
        "atom_index_0": "int16",
        "atom_index_1": "int16",
        "type": "category",
        "scalar_coupling_constant": "float32",
    },
)
test = pd.read_csv(
    f"{INPUT_DIR}/test.csv",
    dtype={
        "id": "int32",
        "molecule_name": "category",
        "atom_index_0": "int16",
        "atom_index_1": "int16",
        "type": "category",
    },
)



## === cell 2
structures_idx = structures.set_index(["molecule_name", "atom_index"], drop=True)[
    ["atom", "x", "y", "z"]
].sort_index()


def map_atom_info(df, atom_idx):
    key = pd.MultiIndex.from_arrays(
        [df["molecule_name"].values, df[f"atom_index_{atom_idx}"].values],
        names=["molecule_name", "atom_index"],
    )
    add = structures_idx.reindex(key)
    add = add.rename(
        columns={
            "atom": f"atom_{atom_idx}",
            "x": f"x_{atom_idx}",
            "y": f"y_{atom_idx}",
            "z": f"z_{atom_idx}",
        }
    ).reset_index(drop=True)
    return pd.concat([df.reset_index(drop=True), add], axis=1)


train = map_atom_info(train, 0)
train = map_atom_info(train, 1)

test = map_atom_info(test, 0)
test = map_atom_info(test, 1)



## === cell 3
train.head()



## === cell 4
dx = train["x_1"].to_numpy() - train["x_0"].to_numpy()
dy = train["y_1"].to_numpy() - train["y_0"].to_numpy()
dz = train["z_1"].to_numpy() - train["z_0"].to_numpy()
train["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)

dx = test["x_1"].to_numpy() - test["x_0"].to_numpy()
dy = test["y_1"].to_numpy() - test["y_0"].to_numpy()
dz = test["z_1"].to_numpy() - test["z_0"].to_numpy()
test["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)



## === cell 5
print(train["atom_0"].value_counts())

train = train.drop(["atom_0", "atom_index_1", "atom_index_0"], axis=1)
test = test.drop(["atom_0", "atom_index_1", "atom_index_0"], axis=1)



## === cell 6
train.head()



## === cell 7
test.head()



## === cell 8
print("Skipping plots for performance (no effect on training/predictions).")



## === cell 9
print("Skipping plots for performance (no effect on training/predictions).")



## === cell 10
print("Skipping plots for performance (no effect on training/predictions).")



## === cell 11
print("Skipping plots for performance (no effect on training/predictions).")



## === cell 12
print("Skipping plots for performance (no effect on training/predictions).")



## === cell 13
print("Skipping plots for performance (no effect on training/predictions).")



## === cell 14
train_mol = train["molecule_name"].copy()
train_type = train["type"].copy()
test_id = test["id"].copy()
test_type = test["type"].copy()

train = train.drop(["id", "molecule_name"], axis=1)
test = test.drop(["id", "molecule_name"], axis=1)



## === cell 15
cat_cols = train.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in train.columns if c not in cat_cols]

for c in cat_cols:
    train[c] = (
        train[c].astype("category").cat.add_categories(["Unknown"]).fillna("Unknown")
    )
    if c in test.columns:
        test[c] = (
            test[c].astype("category").cat.add_categories(["Unknown"]).fillna("Unknown")
        )

num_fill_cols = [c for c in num_cols if c != "scalar_coupling_constant"]
medians = train[num_fill_cols].median(numeric_only=True)
train[num_fill_cols] = train[num_fill_cols].fillna(medians)
test[num_fill_cols] = test[num_fill_cols].fillna(medians)

train = pd.get_dummies(train)
test = pd.get_dummies(test)



## === cell 16
Y = train["scalar_coupling_constant"].astype(np.float32)
X = train.drop(["scalar_coupling_constant"], axis=1)

test = test.reindex(columns=X.columns, fill_value=0)

fill_vals = X.median(numeric_only=True)
X = X.fillna(fill_vals)
test = test.fillna(fill_vals)

X = X.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
test = test.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)

print("X shape:", X.shape, "test shape:", test.shape)
print(
    "NaNs in X:",
    int(np.isnan(X.to_numpy()).sum()),
    "NaNs in test:",
    int(np.isnan(test.to_numpy()).sum()),
)



## === cell 17
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge


def log_mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = np.mean(np.abs(y_true - y_pred))
    return float(np.log(mae + 1e-12))


def cv_logmae(alpha, X_np, Y_np, splits):
    fold_scores = []
    for tr_idx, va_idx in splits:
        model = Ridge(alpha=alpha)
        model.fit(X_np[tr_idx], Y_np[tr_idx])
        pred = model.predict(X_np[va_idx])
        fold_scores.append(log_mae(Y_np[va_idx], pred))
    return np.asarray(fold_scores, dtype=np.float64)




## === cell 18
gkf = GroupKFold(n_splits=5)
groups_np_all = train_mol.astype(str).to_numpy()
types_np_all = train_type.astype(str).to_numpy()
test_types_np_all = test_type.astype(str).to_numpy()

X_np_all = X.to_numpy(dtype=np.float64, copy=False)
Y_np_all = Y.to_numpy(dtype=np.float64, copy=False)
X_test_np_all = test.to_numpy(dtype=np.float64, copy=False)

alpha_list = np.concatenate(
    [
        np.array([0.01, 0.03], dtype=np.float64),
        np.linspace(start=0.05, stop=2.0, num=20, dtype=np.float64),
    ]
)

unique_types = np.unique(types_np_all)
best_alpha_by_type = {}
cv_mean_by_type = {}

plt.figure(figsize=(10, 4))

for t_i, t in enumerate(unique_types):
    tr_mask = types_np_all == t
    idx = np.where(tr_mask)[0]

    X_np = X_np_all[idx]
    Y_np = Y_np_all[idx]
    groups_np = groups_np_all[idx]

    if len(idx) < 1000:
        best_alpha_by_type[t] = 0.5
        cv_mean_by_type[t] = np.nan
        continue

    splits = [
        (tr_idx, va_idx) for tr_idx, va_idx in gkf.split(X_np, Y_np, groups=groups_np)
    ]

    L = []
    for alpha_val in alpha_list:
        fold_scores = cv_logmae(
            alpha=float(alpha_val),
            X_np=X_np,
            Y_np=Y_np,
            splits=splits,
        )
        L.append(fold_scores.mean())

    L = np.asarray(L, dtype=np.float64)
    best_alpha = float(alpha_list[int(np.argmin(L))])

    best_alpha_by_type[t] = best_alpha
    cv_mean_by_type[t] = float(np.min(L))

    plt.plot(alpha_list, L, label=t)

plt.xlabel("alpha")
plt.ylabel("CV mean log(MAE) per type")
plt.legend(ncol=4, fontsize=8)
plt.tight_layout()
plt.show()

print("Per-type best alphas (sample):", dict(list(best_alpha_by_type.items())[:5]))
print("Per-type best CV (sample):", dict(list(cv_mean_by_type.items())[:5]))
print(
    "Overall mean of per-type CV mins (nan ignored):",
    float(np.nanmean(list(cv_mean_by_type.values()))),
)



## === cell 19
pred_test = np.empty(X_test_np_all.shape[0], dtype=np.float64)

for t in unique_types:
    tr_mask = types_np_all == t
    te_mask = test_types_np_all == t
    if not np.any(te_mask):
        continue

    idx_tr = np.where(tr_mask)[0]
    idx_te = np.where(te_mask)[0]

    alpha = float(best_alpha_by_type.get(t, 0.5))
    model = Ridge(alpha=alpha)
    model.fit(X_np_all[idx_tr], Y_np_all[idx_tr])
    pred_test[idx_te] = model.predict(X_test_np_all[idx_te])

submission = pd.DataFrame(
    {
        "id": test_id.values.astype(np.int32, copy=False),
        "scalar_coupling_constant": pred_test.astype(np.float32),
    }
)
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
print("submission.csv exists:", os.path.exists("submission.csv"))
