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

-1.6838160788283738

# 6. Current score

7.87216

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.18497) has done: 'Your notebook fails because it tries to read out-of-environment blend files (e.g., `../input/champs-blending-tutorial/1.csv`) that do not exist in the provided filesystem. To make it run end-to-end and generate a valid `.csv` submission, I replace that external blending step with a minimal, self-contained baseline model that uses only the provided competition data (`train.csv`/`test.csv`) and produces predictions per coupling `type` (median target per type, with a global fallback). This preserves the overall “simple aggregation-based predictor” spirit while removing missing dependencies. The output be written as `my_blend_1.csv` with the required columns and row alignment by `id`.'
- What this solution (achieved 4.11578) has done: 'I fix the Ridge crash by ensuring the numeric distance features never contain NaN/inf (these can appear after merges if any structure rows are missing, or if a distance is zero leading to inf in `inv_dist`). Concretely, I coerce the coordinate columns to numeric, compute `dist/dist2/inv_dist` robustly, then replace any remaining NaN/inf in the numeric feature matrix with safe finite values (medians, with 0 fallback). This keeps the same model/feature logic and training loops, but makes the pipeline run end-to-end and produce a valid `my_blend_1.csv` submission.'
- What this solution (achieved 7.87216) has done: 'Your current score (4.11578, lower-is-better) is far worse than the target (-1.6838), so we should improve while keeping the same overall “Ridge per type on simple structure-distance + atom pair one-hot” core. The biggest likely issue is that the numeric feature scales are extremely ill-conditioned (dist, dist2, inv_dist), which makes Ridge behave poorly without standardization; adding a `StandardScaler` inside a `Pipeline` preserves the same linear model logic but typically yields a large MAE reduction on this competition. I also make the train/test one-hot alignment lighter and safer by using `OneHotEncoder(handle_unknown="ignore")` in a `ColumnTransformer`, which avoids giant dense matrices and keeps consistent columns without depending on concatenation order. Everything else (per-type training loop, fallbacks, submission alignment) stays the same.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

DATA_DIR = "/kaggle/data/champs-scalar-coupling"

print("Listing /kaggle/data:")
print(os.listdir("/kaggle/data")[:20])
print("\nListing DATA_DIR:")
print(os.listdir(DATA_DIR)[:20])



## === cell 1
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
structures_path = os.path.join(DATA_DIR, "structures.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample = pd.read_csv(sample_path)

structures = pd.read_csv(structures_path)

for c in ["x", "y", "z"]:
    structures[c] = pd.to_numeric(structures[c], errors="coerce").astype("float32")
structures["atom_index"] = pd.to_numeric(
    structures["atom_index"], errors="coerce"
).astype("int32")

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


def add_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(
        s0[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    df = df.merge(
        s1[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
    )

    x0 = df["x0"].astype("float32")
    y0 = df["y0"].astype("float32")
    z0 = df["z0"].astype("float32")
    x1 = df["x1"].astype("float32")
    y1 = df["y1"].astype("float32")
    z1 = df["z1"].astype("float32")

    dx = (x0 - x1).astype("float32")
    dy = (y0 - y1).astype("float32")
    dz = (z0 - z1).astype("float32")

    dist2 = (dx * dx + dy * dy + dz * dz).astype("float32")
    dist = np.sqrt(dist2.astype("float64")).astype("float32")  # stable sqrt
    inv_dist = (1.0 / (dist.astype("float64") + 1e-6)).astype("float32")

    df["dist"] = dist
    df["dist2"] = dist2
    df["inv_dist"] = inv_dist

    for c in ["dist", "dist2", "inv_dist"]:
        v = df[c].to_numpy()
        bad = ~np.isfinite(v)
        if bad.any():
            finite = v[np.isfinite(v)]
            fill = float(np.median(finite)) if finite.size else 0.0
            v[bad] = fill
            df[c] = v.astype("float32")

    return df


train_f = add_structure_features(train)
test_f = add_structure_features(test)

type_median = train.groupby("type")["scalar_coupling_constant"].median()
global_median = float(train["scalar_coupling_constant"].median())

cat_cols = ["atom_0", "atom_1"]
for c in cat_cols:
    train_f[c] = train_f[c].fillna("UNK").astype("string")
    test_f[c] = test_f[c].fillna("UNK").astype("string")

num_cols = ["dist", "dist2", "inv_dist"]
for c in num_cols:
    train_f[c] = pd.to_numeric(train_f[c], errors="coerce").astype("float32")
    test_f[c] = pd.to_numeric(test_f[c], errors="coerce").astype("float32")

preprocess = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(steps=[("scaler", StandardScaler(with_mean=True, with_std=True))]),
            num_cols,
        ),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
    ],
    remainder="drop",
    sparse_threshold=0.3,
)

pred_test = np.empty(len(test_f), dtype=np.float32)
pred_test[:] = np.nan

train_f = train_f.reset_index(drop=True)
test_f = test_f.reset_index(drop=True)
y_train_full = (
    train_f["scalar_coupling_constant"].astype("float32").reset_index(drop=True)
)

for t in test_f["type"].unique():
    test_idx = test_f.index[test_f["type"] == t].to_numpy(dtype=np.int64)
    trn_mask = (train_f["type"] == t).to_numpy()

    if trn_mask.sum() < 1000:
        pred_test[test_idx] = float(type_median.get(t, global_median))
        continue

    X_trn_df = train_f.loc[trn_mask, cat_cols + num_cols]
    y_trn = y_train_full.loc[trn_mask]
    X_tst_df = test_f.loc[test_idx, cat_cols + num_cols]

    model = Pipeline(
        steps=[
            ("prep", preprocess),
            ("ridge", Ridge(alpha=1.0, random_state=0)),
        ]
    )
    model.fit(X_trn_df, y_trn)
    pred_test[test_idx] = model.predict(X_tst_df).astype(np.float32)

pred_series = pd.Series(pred_test, index=test_f.index)
pred_series = (
    pred_series.fillna(test["type"].map(type_median))
    .fillna(global_median)
    .astype("float64")
)

submission = pd.DataFrame(
    {"id": test["id"].values, "scalar_coupling_constant": pred_series.values}
)

if not submission["id"].equals(sample["id"]):
    submission = sample[["id"]].merge(submission, on="id", how="left")
    submission["scalar_coupling_constant"] = submission[
        "scalar_coupling_constant"
    ].fillna(global_median)

out_path = "my_blend_1.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print("Rows:", len(submission), "Columns:", submission.columns.tolist())
assert out_path.endswith(".csv")
assert submission.shape[0] == sample.shape[0]
assert list(submission.columns) == ["id", "scalar_coupling_constant"]
assert np.isfinite(submission["scalar_coupling_constant"].to_numpy()).all()
