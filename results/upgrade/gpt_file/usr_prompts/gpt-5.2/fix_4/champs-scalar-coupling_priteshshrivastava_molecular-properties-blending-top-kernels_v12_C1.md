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

-1.6821991287997458

# 6. Current score

1.21757

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.8228) has done: 'The crash is because the notebook expects out-of-scope datasets (other people’s submission files) under `../input/...`, which do not exist in your environment, so no submission CSV is ever written. I replace that external-file blending with an in-notebook baseline model trained on the provided `train.csv/structures.csv` and used to predict `test.csv`, keeping the pipeline simple and deterministic. This fixes runtime, guarantees a correctly formatted `submission.csv`, and should achieve a reasonable score (better than a constant baseline) without relying on unavailable inputs. I also add safety checks to ensure feature columns align and no NaNs remain.'
- What this solution (achieved 1.23132) has done: 'Your current score (1.8228, lower is better) is far from the target (-1.6822), so we should make a small, legitimate improvement that better matches the competition’s metric without changing your overall approach (structure-merge features + a single sklearn regressor). The biggest gap is that CV is using global MAE, while the leaderboard uses **mean of log(MAE) computed per coupling `type`**, so the model is not being tuned/validated against the right objective. I (1) add an “official metric” cross-validation calculation (log MAE per type averaged) for visibility, and (2) train **separate models per `type`** (same pipeline/model class and same features) which usually improves this competition materially while staying within the same core logic. This keeps deterministic training, produces the same submission schema, and should move the score down toward the target.'
- What this solution (achieved 1.21757) has done: 'Your current score (1.23132, lower is better) is still far from the target (-1.6822), so we should make a small, legitimate improvement without changing the core approach (structures merge → basic distance features → sklearn tree regressor per type). The biggest low-risk gain here is to add a few standard “pairwise geometry” features (signed deltas and simple ratios) plus a molecule-level distance normalization, which typically reduces MAE across coupling types while keeping the same model class and training loop. I also add a deterministic per-type group-CV report (same metric) to verify the change is moving in the right direction, but keep training on full data for submission exactly as before. All paths stay the same and the code still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

INPUT_DIR = "/kaggle/input/champs-scalar-coupling"
WORKING_DIR = "/kaggle/working"

print("Input dir exists:", os.path.exists(INPUT_DIR))
print("Files (sample):", sorted(os.listdir(INPUT_DIR))[:10])



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
sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
structures = pd.read_csv(structures_path)

print(train.shape, test.shape, structures.shape)
print(train.columns)
print(test.columns)
print(structures.columns)



## === cell 2
for c in ["atom_index_0", "atom_index_1"]:
    train[c] = train[c].astype(np.int32)
    test[c] = test[c].astype(np.int32)
structures["atom_index"] = structures["atom_index"].astype(np.int32)

s0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "x_0",
        "y": "y_0",
        "z": "z_0",
    }
)
s1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "x": "x_1",
        "y": "y_1",
        "z": "z_1",
    }
)


def add_structure_features(df):
    df = df.merge(
        s0[["molecule_name", "atom_index_0", "atom_0", "x_0", "y_0", "z_0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    df = df.merge(
        s1[["molecule_name", "atom_index_1", "atom_1", "x_1", "y_1", "z_1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
    )

    dx = df["x_0"] - df["x_1"]
    dy = df["y_0"] - df["y_1"]
    dz = df["z_0"] - df["z_1"]

    df["dx"] = dx
    df["dy"] = dy
    df["dz"] = dz

    df["dist2"] = dx * dx + dy * dy + dz * dz
    df["dist"] = np.sqrt(df["dist2"])

    df["abs_dx"] = dx.abs()
    df["abs_dy"] = dy.abs()
    df["abs_dz"] = dz.abs()

    eps = 1e-12
    df["inv_dist"] = 1.0 / (df["dist"] + eps)
    df["inv_dist2"] = 1.0 / (df["dist2"] + eps)
    df["abs_dx_over_dist"] = df["abs_dx"] / (df["dist"] + eps)
    df["abs_dy_over_dist"] = df["abs_dy"] / (df["dist"] + eps)
    df["abs_dz_over_dist"] = df["abs_dz"] / (df["dist"] + eps)
    df["dx_dy"] = df["dx"] * df["dy"]
    df["dx_dz"] = df["dx"] * df["dz"]
    df["dy_dz"] = df["dy"] * df["dz"]

    return df


train_f = add_structure_features(train)
test_f = add_structure_features(test)


def add_molecule_distance_norm(df):
    g = df.groupby("molecule_name")["dist"]
    df["mol_dist_mean"] = g.transform("mean")
    df["mol_dist_std"] = g.transform("std").fillna(0.0)
    eps = 1e-12
    df["dist_z"] = (df["dist"] - df["mol_dist_mean"]) / (df["mol_dist_std"] + eps)
    df["dist_over_mean"] = df["dist"] / (df["mol_dist_mean"] + eps)
    return df


train_f = add_molecule_distance_norm(train_f)
test_f = add_molecule_distance_norm(test_f)

missing_train = train_f[["atom_0", "atom_1", "x_0", "x_1", "dist"]].isna().mean().max()
missing_test = test_f[["atom_0", "atom_1", "x_0", "x_1", "dist"]].isna().mean().max()
print("Max missing fraction (train,test):", float(missing_train), float(missing_test))



## === cell 3
target = "scalar_coupling_constant"
y = train_f[target].values

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
]
feature_cols_cat = ["type", "atom_0", "atom_1"]

X = train_f[feature_cols_num + feature_cols_cat]
X_test = test_f[feature_cols_num + feature_cols_cat]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
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
    loss="absolute_error",  # aligns with MAE-based metric
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
groups = train_f["molecule_name"].values
types_all = train_f["type"].values

cv_maes = []
cv_scores = []
for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    t_va = types_all[va_idx]
    pipe.fit(X_tr, y_tr)
    pred_va = pipe.predict(X_va)
    mae = mean_absolute_error(y_va, pred_va)
    score, _ = champs_metric(y_va, pred_va, t_va)
    cv_maes.append(mae)
    cv_scores.append(score)
    print(
        f"Fold {fold} MAE (global): {mae:.6f} | CHAMPS metric (mean log MAE by type): {score:.6f}"
    )

print("CV MAE mean (global):", float(np.mean(cv_maes)))
print("CV CHAMPS metric mean:", float(np.mean(cv_scores)))



## === cell 4
test_pred = np.zeros(len(test_f), dtype=np.float64)

for t in sorted(train_f["type"].unique()):
    tr_mask = train_f["type"].values == t
    te_mask = test_f["type"].values == t

    X_tr_t = train_f.loc[tr_mask, feature_cols_num + feature_cols_cat]
    y_tr_t = train_f.loc[tr_mask, target].values
    X_te_t = test_f.loc[te_mask, feature_cols_num + feature_cols_cat]

    pipe.fit(X_tr_t, y_tr_t)
    preds_t = pipe.predict(X_te_t)

    test_pred[te_mask] = preds_t
    print(f"type={t}: train_n={int(tr_mask.sum())}, test_n={int(te_mask.sum())}")

test_pred = np.where(np.isfinite(test_pred), test_pred, 0.0)

sub = pd.read_csv(sample_sub_path)
pred_df = pd.DataFrame(
    {"id": test_f["id"].values, "scalar_coupling_constant": test_pred}
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
