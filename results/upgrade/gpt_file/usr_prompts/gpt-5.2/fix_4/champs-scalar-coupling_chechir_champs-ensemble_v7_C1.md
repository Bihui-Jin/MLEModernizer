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

-2.100464760474246

# 6. Current score

3.55362

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.2116) has done: 'The crash is from reading external “../input/champ-preds/…” and “../input/1-mpnn/…” files that are not present in your environment, so no submission ever gets created. I keep the same overall idea (produce `id, scalar_coupling_constant` and write a `.csv`) but replace the missing-file ensemble with a self-contained baseline that trains per coupling `type` and predicts for test. This fixes the runtime errors, avoids leakage by splitting by `molecule_name` for a quick sanity check, and produces `sub_ensemble.csv` in the required format. The approach is intentionally simple/fast to ensure it finishes under the time limit and yields a valid submission.'
- What this solution (achieved 1.2116) has done: 'The crash comes from the strict NA check after merging `structures.csv`: in this dataset a tiny number of rows can fail to match (data quirks or dtype mismatches), and raising stops the whole pipeline. I make the merge keys type-consistent, then replace the hard failure with safe handling: keep training rows with valid structure matches and impute missing structure-derived features in test (so we still output predictions for every `id`). This preserves your core modeling approach (per-`type` Ridge on simple distance/atom features) while ensuring the notebook runs end-to-end and always writes a valid `.csv` submission. The score should also improve versus using broken/partial features because training won’t be poisoned by unmatched-merge NaNs.'
- What this solution (achieved 3.55362) has done: 'Your current model is extremely underpowered for CHAMPS, so to move the score down toward the target we should add a few high-signal structure-derived features while keeping the same per-`type` Ridge + OHE pipeline. I minimally extend `add_structure_features()` to include molecule-level center-of-mass style coordinates (mean x/y/z per molecule) and the relative positions of each atom to that center, plus a couple of simple interaction terms; this preserves the core logic and only enriches the existing feature set. I also include the coupling `type` as a feature while switching to a single global Ridge model (still Ridge, same preprocessing, same semantics) so the model can share statistics across types without changing the training approach. These changes are typically enough to significantly reduce MAE/logMAE while remaining fast and self-contained.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

DATA_DIR = "/kaggle/data/champs-scalar-coupling"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
structures_path = os.path.join(DATA_DIR, "structures.csv")

assert os.path.exists(train_path), f"Missing: {train_path}"
assert os.path.exists(test_path), f"Missing: {test_path}"
assert os.path.exists(structures_path), f"Missing: {structures_path}"




## === cell 1
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
structures = pd.read_csv(structures_path)

for df in (train, test, structures):
    df["molecule_name"] = df["molecule_name"].astype(str)

train["atom_index_0"] = train["atom_index_0"].astype(np.int32)
train["atom_index_1"] = train["atom_index_1"].astype(np.int32)
test["atom_index_0"] = test["atom_index_0"].astype(np.int32)
test["atom_index_1"] = test["atom_index_1"].astype(np.int32)
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

mol_centroid = (
    structures.groupby("molecule_name", sort=False)[["x", "y", "z"]]
    .mean()
    .rename(columns={"x": "x_c", "y": "y_c", "z": "z_c"})
    .reset_index()
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
    df = df.merge(mol_centroid, on="molecule_name", how="left")

    dx = df["x_0"] - df["x_1"]
    dy = df["y_0"] - df["y_1"]
    dz = df["z_0"] - df["z_1"]
    df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz)
    df["dist2"] = dx * dx + dy * dy + dz * dz
    df["abs_dx"] = dx.abs()
    df["abs_dy"] = dy.abs()
    df["abs_dz"] = dz.abs()

    df["x0_c"] = df["x_0"] - df["x_c"]
    df["y0_c"] = df["y_0"] - df["y_c"]
    df["z0_c"] = df["z_0"] - df["z_c"]
    df["x1_c"] = df["x_1"] - df["x_c"]
    df["y1_c"] = df["y_1"] - df["y_c"]
    df["z1_c"] = df["z_1"] - df["z_c"]

    df["r0_c"] = np.sqrt(
        df["x0_c"] * df["x0_c"] + df["y0_c"] * df["y0_c"] + df["z0_c"] * df["z0_c"]
    )
    df["r1_c"] = np.sqrt(
        df["x1_c"] * df["x1_c"] + df["y1_c"] * df["y1_c"] + df["z1_c"] * df["z1_c"]
    )

    df["r0_r1"] = df["r0_c"] * df["r1_c"]
    df["inv_dist"] = 1.0 / (df["dist"] + 1e-6)

    return df


train_feat = add_structure_features(train)
test_feat = add_structure_features(test)

req_cols = ["atom_0", "atom_1", "dist", "x_c", "y_c", "z_c"]
train_missing_mask = train_feat[req_cols].isna().any(axis=1)
test_missing_mask = test_feat[req_cols].isna().any(axis=1)

n_train_missing = int(train_missing_mask.sum())
n_test_missing = int(test_missing_mask.sum())

if n_train_missing > 0 or n_test_missing > 0:
    print(
        f"Warning: missing structure merges detected. "
        f"train_missing={n_train_missing}/{len(train_feat)}, "
        f"test_missing={n_test_missing}/{len(test_feat)}. "
        f"Dropping missing train rows; imputing missing test features."
    )

if n_train_missing > 0:
    train_feat = train_feat.loc[~train_missing_mask].reset_index(drop=True)




## === cell 2
feature_cols_num = [
    "dist",
    "dist2",
    "abs_dx",
    "abs_dy",
    "abs_dz",
    "x0_c",
    "y0_c",
    "z0_c",
    "x1_c",
    "y1_c",
    "z1_c",
    "r0_c",
    "r1_c",
    "r0_r1",
    "inv_dist",
]
feature_cols_cat = ["atom_0", "atom_1", "type"]

preprocess = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([("imputer", SimpleImputer(strategy="median"))]),
            feature_cols_num,
        ),
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            feature_cols_cat,
        ),
    ],
    remainder="drop",
)

gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
train_idx, val_idx = next(gss.split(train_feat, groups=train_feat["molecule_name"]))
tr = train_feat.iloc[train_idx]
va = train_feat.iloc[val_idx]

model = Pipeline(
    steps=[("prep", preprocess), ("reg", Ridge(alpha=1.0, random_state=42))]
)

X_tr = tr[feature_cols_num + feature_cols_cat]
y_tr = tr["scalar_coupling_constant"].values
X_va = va[feature_cols_num + feature_cols_cat]
y_va = va["scalar_coupling_constant"].values

model.fit(X_tr, y_tr)
pred_va = model.predict(X_va)

va_tmp = va[["type"]].copy()
va_tmp["y"] = y_va
va_tmp["p"] = pred_va
val_mae_by_type = (
    va_tmp.groupby("type")
    .apply(lambda g: mean_absolute_error(g["y"].values, g["p"].values))
    .sort_values()
)
print("Validation MAE by type (lower is better):")
print(val_mae_by_type.head(10))
print(f"Validated types: {val_mae_by_type.shape[0]}/{train_feat['type'].nunique()}")




## === cell 3
X_full = train_feat[feature_cols_num + feature_cols_cat]
y_full = train_feat["scalar_coupling_constant"].values
model.fit(X_full, y_full)

X_test = test_feat[feature_cols_num + feature_cols_cat]
test_pred = model.predict(X_test)

submission = pd.DataFrame(
    {
        "id": test_feat["id"].values,
        "scalar_coupling_constant": test_pred.astype(np.float64),
    }
)

submission = submission.sort_values("id").reset_index(drop=True)
submission["id"] = submission["id"].astype(np.int64)

out_path = "sub_ensemble.csv"
submission.to_csv(out_path, index=False)
print(f"Wrote submission: {out_path} with shape={submission.shape}")
print(submission.head(10))




## === cell 4
submission.head(20)
