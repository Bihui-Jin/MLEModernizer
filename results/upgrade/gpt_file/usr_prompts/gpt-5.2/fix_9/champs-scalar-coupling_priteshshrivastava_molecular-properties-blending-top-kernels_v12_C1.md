# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import time
import numpy as np
import pandas as pd

INPUT_DIR = "/kaggle/input/champs-scalar-coupling"
WORKING_DIR = "/kaggle/working"

print("Input dir exists:", os.path.exists(INPUT_DIR))
print("Files (sample):", sorted(os.listdir(INPUT_DIR))[:10])

t0 = time.time()



## === cell 1
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor

train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")
structures_path = os.path.join(INPUT_DIR, "structures.csv")
mulliken_path = os.path.join(INPUT_DIR, "mulliken_charges.csv")
mst_path = os.path.join(INPUT_DIR, "magnetic_shielding_tensors.csv")
dipole_path = os.path.join(INPUT_DIR, "dipole_moments.csv")
potential_path = os.path.join(INPUT_DIR, "potential_energy.csv")
sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")

train = pd.read_csv(
    train_path,
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int32,
        "atom_index_1": np.int32,
        "type": "category",
        "scalar_coupling_constant": np.float32,
    },
)
test = pd.read_csv(
    test_path,
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int32,
        "atom_index_1": np.int32,
        "type": "category",
    },
)
structures = pd.read_csv(
    structures_path,
    dtype={
        "molecule_name": "category",
        "atom_index": np.int32,
        "atom": "category",
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
    },
)

print(train.shape, test.shape, structures.shape)
print(train.columns)
print(test.columns)
print(structures.columns)
print("Load elapsed (s):", round(time.time() - t0, 2))



## === cell 2
mull = pd.read_csv(
    mulliken_path,
    usecols=["molecule_name", "atom_index", "mulliken_charge"],
    dtype={
        "molecule_name": "category",
        "atom_index": np.int32,
        "mulliken_charge": np.float32,
    },
)

mst = pd.read_csv(
    mst_path,
    usecols=[
        "molecule_name",
        "atom_index",
        "XX",
        "YX",
        "ZX",
        "XY",
        "YY",
        "ZY",
        "XZ",
        "YZ",
        "ZZ",
    ],
    dtype={
        "molecule_name": "category",
        "atom_index": np.int32,
        "XX": np.float32,
        "YX": np.float32,
        "ZX": np.float32,
        "XY": np.float32,
        "YY": np.float32,
        "ZY": np.float32,
        "XZ": np.float32,
        "YZ": np.float32,
        "ZZ": np.float32,
    },
)

mst["mst_trace"] = (mst["XX"] + mst["YY"] + mst["ZZ"]).astype(np.float32)
mst_vals = mst[["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]].to_numpy(
    copy=False
)
mst["mst_frob"] = np.sqrt((mst_vals * mst_vals).sum(axis=1)).astype(np.float32)

dip = pd.read_csv(
    dipole_path,
    usecols=["molecule_name", "X", "Y", "Z"],
    dtype={
        "molecule_name": "category",
        "X": np.float32,
        "Y": np.float32,
        "Z": np.float32,
    },
).rename(columns={"X": "dipole_X", "Y": "dipole_Y", "Z": "dipole_Z"})

pot = pd.read_csv(
    potential_path,
    usecols=["molecule_name", "potential_energy"],
    dtype={"molecule_name": "category", "potential_energy": np.float32},
)

struct_aux = structures.merge(
    mull, on=["molecule_name", "atom_index"], how="left", copy=False
)
struct_aux = struct_aux.merge(
    mst, on=["molecule_name", "atom_index"], how="left", copy=False
)

struct_aux = struct_aux.set_index(["molecule_name", "atom_index"], drop=False)

s0_cols = {
    "atom": "atom_0",
    "x": "x_0",
    "y": "y_0",
    "z": "z_0",
    "mulliken_charge": "mulliken_charge_0",
    "XX": "mst_XX_0",
    "YX": "mst_YX_0",
    "ZX": "mst_ZX_0",
    "XY": "mst_XY_0",
    "YY": "mst_YY_0",
    "ZY": "mst_ZY_0",
    "XZ": "mst_XZ_0",
    "YZ": "mst_YZ_0",
    "ZZ": "mst_ZZ_0",
    "mst_trace": "mst_trace_0",
    "mst_frob": "mst_frob_0",
}
s1_cols = {k: v.replace("_0", "_1") for k, v in s0_cols.items()}

need_cols = [
    "atom",
    "x",
    "y",
    "z",
    "mulliken_charge",
    "XX",
    "YX",
    "ZX",
    "XY",
    "YY",
    "ZY",
    "XZ",
    "YZ",
    "ZZ",
    "mst_trace",
    "mst_frob",
]


def _attach_atom_block(df, atom_index_col, rename_map):
    idx = pd.MultiIndex.from_arrays(
        [df["molecule_name"].to_numpy(), df[atom_index_col].to_numpy()]
    )
    block = struct_aux.reindex(idx)[need_cols]
    block = block.rename(columns=rename_map)
    block = block.reset_index(drop=True)
    return pd.concat([df.reset_index(drop=True), block], axis=1)


def add_structure_features(df):
    df2 = _attach_atom_block(df, "atom_index_0", s0_cols)
    df2 = _attach_atom_block(df2, "atom_index_1", s1_cols)

    dip_idx = dip.set_index("molecule_name")
    pot_idx = pot.set_index("molecule_name")
    mol = df2["molecule_name"]

    df2["dipole_X"] = mol.map(dip_idx["dipole_X"]).astype(np.float32)
    df2["dipole_Y"] = mol.map(dip_idx["dipole_Y"]).astype(np.float32)
    df2["dipole_Z"] = mol.map(dip_idx["dipole_Z"]).astype(np.float32)
    df2["potential_energy"] = mol.map(pot_idx["potential_energy"]).astype(np.float32)

    dx = (df2["x_0"] - df2["x_1"]).astype(np.float32)
    dy = (df2["y_0"] - df2["y_1"]).astype(np.float32)
    dz = (df2["z_0"] - df2["z_1"]).astype(np.float32)

    df2["dx"] = dx
    df2["dy"] = dy
    df2["dz"] = dz

    dist2 = (dx * dx + dy * dy + dz * dz).astype(np.float32)
    df2["dist2"] = dist2
    df2["dist"] = np.sqrt(dist2.astype(np.float64)).astype(np.float32)

    df2["abs_dx"] = dx.abs()
    df2["abs_dy"] = dy.abs()
    df2["abs_dz"] = dz.abs()

    eps = np.float32(1e-12)
    df2["inv_dist"] = (1.0 / (df2["dist"] + eps)).astype(np.float32)
    df2["inv_dist2"] = (1.0 / (df2["dist2"] + eps)).astype(np.float32)
    df2["abs_dx_over_dist"] = (df2["abs_dx"] / (df2["dist"] + eps)).astype(np.float32)
    df2["abs_dy_over_dist"] = (df2["abs_dy"] / (df2["dist"] + eps)).astype(np.float32)
    df2["abs_dz_over_dist"] = (df2["abs_dz"] / (df2["dist"] + eps)).astype(np.float32)
    df2["dx_dy"] = (df2["dx"] * df2["dy"]).astype(np.float32)
    df2["dx_dz"] = (df2["dx"] * df2["dz"]).astype(np.float32)
    df2["dy_dz"] = (df2["dy"] * df2["dz"]).astype(np.float32)

    df2["mulliken_sum"] = (df2["mulliken_charge_0"] + df2["mulliken_charge_1"]).astype(
        np.float32
    )
    df2["mulliken_diff"] = (df2["mulliken_charge_0"] - df2["mulliken_charge_1"]).astype(
        np.float32
    )
    df2["mst_trace_sum"] = (df2["mst_trace_0"] + df2["mst_trace_1"]).astype(np.float32)
    df2["mst_trace_diff"] = (df2["mst_trace_0"] - df2["mst_trace_1"]).astype(np.float32)
    df2["mst_frob_sum"] = (df2["mst_frob_0"] + df2["mst_frob_1"]).astype(np.float32)
    df2["mst_frob_diff"] = (df2["mst_frob_0"] - df2["mst_frob_1"]).astype(np.float32)

    df2["dipole_mag"] = np.sqrt(
        (df2["dipole_X"].astype(np.float64) ** 2)
        + (df2["dipole_Y"].astype(np.float64) ** 2)
        + (df2["dipole_Z"].astype(np.float64) ** 2)
    ).astype(np.float32)

    return df2


train_f = add_structure_features(train)
test_f = add_structure_features(test)


def add_molecule_distance_norm(df):
    g = df.groupby("molecule_name", sort=False)["dist"]
    df["mol_dist_mean"] = g.transform("mean").astype(np.float32)
    df["mol_dist_std"] = g.transform("std").fillna(0.0).astype(np.float32)
    eps = np.float32(1e-12)
    df["dist_z"] = (
        (df["dist"] - df["mol_dist_mean"]) / (df["mol_dist_std"] + eps)
    ).astype(np.float32)
    df["dist_over_mean"] = (df["dist"] / (df["mol_dist_mean"] + eps)).astype(np.float32)
    return df


train_f = add_molecule_distance_norm(train_f)
test_f = add_molecule_distance_norm(test_f)

missing_train = train_f[["atom_0", "atom_1", "x_0", "x_1", "dist"]].isna().mean().max()
missing_test = test_f[["atom_0", "atom_1", "x_0", "x_1", "dist"]].isna().mean().max()
print("Max missing fraction (train,test):", float(missing_train), float(missing_test))
print("Feature build elapsed (s):", round(time.time() - t0, 2))



## === cell 3
target = "scalar_coupling_constant"
y = train_f[target].to_numpy()

feature_cols_num = [
    "x_0",
    "y_0",
    "z_0",
    "x_1",
    "y_1",
    "z_1",
    "dx",
    "dy",
    "dz",
    "dist",
    "dist2",
    "inv_dist",
    "inv_dist2",
    "abs_dx",
    "abs_dy",
    "abs_dz",
    "abs_dx_over_dist",
    "abs_dy_over_dist",
    "abs_dz_over_dist",
    "dx_dy",
    "dx_dz",
    "dy_dz",
    "mol_dist_mean",
    "mol_dist_std",
    "dist_z",
    "dist_over_mean",
    "mulliken_charge_0",
    "mulliken_charge_1",
    "mulliken_sum",
    "mulliken_diff",
    "mst_trace_0",
    "mst_trace_1",
    "mst_trace_sum",
    "mst_trace_diff",
    "mst_frob_0",
    "mst_frob_1",
    "mst_frob_sum",
    "mst_frob_diff",
    "dipole_X",
    "dipole_Y",
    "dipole_Z",
    "dipole_mag",
    "potential_energy",
]
feature_cols_cat = ["type", "atom_0", "atom_1"]

X = train_f[feature_cols_num + feature_cols_cat]
X_test = test_f[feature_cols_num + feature_cols_cat]

numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, feature_cols_num),
        ("cat", categorical_transformer, feature_cols_cat),
    ],
    remainder="drop",
)

model = HistGradientBoostingRegressor(
    loss="squared_error",
    max_depth=8,
    max_iter=250,
    learning_rate=0.05,
    random_state=42,
)

pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def champs_metric(y_true, y_pred, types, eps=1e-9):
    dfm = pd.DataFrame({"y": y_true, "p": y_pred, "type": types})
    maes = dfm.groupby("type").apply(
        lambda g: np.mean(np.abs(g["y"].values - g["p"].values))
    )
    score = float(np.mean(np.log(maes.values + eps)))
    return score, maes


gkf = GroupKFold(n_splits=3)
groups = train_f["molecule_name"].to_numpy()
types_all = train_f["type"].to_numpy()

cv_maes = []
cv_scores = []

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
    X_tr = X.iloc[tr_idx]
    X_va = X.iloc[va_idx]
    y_tr = y[tr_idx]
    y_va = y[va_idx]
    t_tr = types_all[tr_idx]
    t_va = types_all[va_idx]

    y_pred_va = np.zeros_like(y_va, dtype=np.float64)

    uniq_types = pd.unique(t_tr)

    for t in uniq_types:
        tr_m = t_tr == t
        va_m = t_va == t
        if not np.any(va_m):
            continue

        X_tr_t = X_tr.loc[tr_m]
        X_va_t = X_va.loc[va_m]

        y_tr_t = y_tr[tr_m].astype(np.float64, copy=False)
        y_min_t = float(np.min(y_tr_t))
        shift = max(0.0, -y_min_t) + 1e-6
        y_tr_t_trans = np.log1p(y_tr_t + shift)

        prep = preprocess
        Xt_tr = prep.fit_transform(X_tr_t)
        Xt_va = prep.transform(X_va_t)

        model_t = HistGradientBoostingRegressor(
            loss="squared_error",
            max_depth=8,
            max_iter=250,
            learning_rate=0.05,
            random_state=42,
        )
        model_t.fit(Xt_tr, y_tr_t_trans)
        pred_trans = model_t.predict(Xt_va).astype(np.float64, copy=False)
        y_pred_va[va_m] = np.expm1(pred_trans) - shift

    mae = mean_absolute_error(y_va, y_pred_va)
    score, _ = champs_metric(y_va, y_pred_va, t_va)
    cv_maes.append(mae)
    cv_scores.append(score)
    print(
        f"Fold {fold} MAE (global): {mae:.6f} | CHAMPS metric (mean log MAE by type): {score:.6f}"
    )

print("CV MAE mean (global):", float(np.mean(cv_maes)))
print("CV CHAMPS metric mean:", float(np.mean(cv_scores)))
print("CV elapsed (s):", round(time.time() - t0, 2))



## === cell 4
test_pred = np.zeros(len(test_f), dtype=np.float64)

type_stats = (
    train_f.groupby("type")[target]
    .agg(["mean", "min"])
    .rename(columns={"mean": "y_mean", "min": "y_min"})
)

shrink_alpha = 0.12  # previously 0.08

for t in sorted(train_f["type"].unique()):
    tr_mask = train_f["type"].to_numpy() == t
    te_mask = test_f["type"].to_numpy() == t

    X_tr_t = train_f.loc[tr_mask, feature_cols_num + feature_cols_cat]
    y_tr_t = train_f.loc[tr_mask, target].to_numpy(dtype=np.float64, copy=False)
    X_te_t = test_f.loc[te_mask, feature_cols_num + feature_cols_cat]

    y_min_t = float(type_stats.loc[t, "y_min"])
    shift = max(0.0, -y_min_t) + 1e-6
    y_tr_t_trans = np.log1p(y_tr_t + shift)

    Xt_tr = preprocess.fit_transform(X_tr_t)
    Xt_te = preprocess.transform(X_te_t)

    model_t = HistGradientBoostingRegressor(
        loss="squared_error",
        max_depth=8,
        max_iter=250,
        learning_rate=0.05,
        random_state=42,
    )
    model_t.fit(Xt_tr, y_tr_t_trans)
    preds_trans = model_t.predict(Xt_te).astype(np.float64, copy=False)
    preds_t = np.expm1(preds_trans) - shift

    y_mean_t = float(type_stats.loc[t, "y_mean"])
    preds_t = (1.0 - shrink_alpha) * preds_t + shrink_alpha * y_mean_t

    test_pred[te_mask] = preds_t
    print(f"type={t}: train_n={int(tr_mask.sum())}, test_n={int(te_mask.sum())}")

test_pred = np.where(np.isfinite(test_pred), test_pred, 0.0)

sub = pd.read_csv(sample_sub_path)
pred_df = pd.DataFrame(
    {"id": test_f["id"].to_numpy(), "scalar_coupling_constant": test_pred}
)
sub = sub[["id"]].merge(pred_df, on="id", how="left")
sub["scalar_coupling_constant"] = sub["scalar_coupling_constant"].fillna(0.0)

out_path = os.path.join(WORKING_DIR, "submission.csv")
sub.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(sub.head())
print(sub.shape)
print("Any NaNs:", sub.isna().any().any())
print(
    "Submission id match sample:", sub["id"].equals(pd.read_csv(sample_sub_path)["id"])
)
print("Total elapsed (s):", round(time.time() - t0, 2))
