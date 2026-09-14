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

-1.3517984281028244

# 6. Current score

3.42157

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.99777) has done: 'Your notebook is trying to ensemble (“stack”) external submissions from `../input/top-mol` and several other `../input/...` folders that do not exist in this environment, causing the early `FileNotFoundError` and cascading `NameError`s. To keep the core “stacking/aggregation” intent while making it run end-to-end, I switch the input source to the available `sample_submission.csv` and generate a simple, deterministic baseline prediction (all zeros) in the exact required submission format. I also remove notebook-only magic (`%matplotlib inline`) so it runs as a plain script, and guard any plotting/correlation code so it won’t break execution. This reliably create a valid `.csv` submission file (though the score not be competitive without actual model predictions).'
- What this solution (achieved 1.99777) has done: 'Your current code submits all zeros, which is why the score is far from the (much lower/better) target; we need a small, legitimate model-based prediction while staying simple and deterministic. I keep the pipeline lightweight by training a separate regularized linear model per coupling `type` using only minimal geometric features computed from `structures.csv` (distance and coordinate deltas between the two atoms). This preserves the “no deep architecture/training loop” simplicity, runs within the time limit, and should substantially reduce MAE vs zeros, moving the score toward the target band. I also ensure strict alignment by predicting in the original test row order and writing a valid `submission.csv`.'
- What this solution (achieved 2.85744) has done: 'The crash is because some merged structure coordinates are missing, which makes the geometric features (`dx/dy/dz/dist/inv_dist`) become NaN/inf and Ridge refuses to predict with NaNs. I fix this with minimal, score-neutral preprocessing: compute features, then deterministically impute any non-finite feature values using train-set medians (and apply the same to test). I also make the `one_hot` routine robust to missing atom labels without changing the modeling approach. This keeps the per-type Ridge training core logic intact while allowing the notebook to run end-to-end and generate a valid `submission.csv`.'
- What this solution (achieved 2.82349) has done: 'Your current per-type Ridge baseline is sound but it’s leaving a lot of signal unused; the biggest safe gain (without changing the learning approach) is to add a few more physically meaningful, cheap geometric features derived from the same merged coordinates (squared distance, absolute deltas, and unit direction components). This keeps the exact same training loop (one Ridge per type) and still uses the same data sources, but gives the linear model enough feature richness to reduce MAE materially toward your target. I also standardize numeric features using train statistics (applied to test) to make the single global `alpha=1.0` behave more consistently across feature scales; this is a minimal preprocessing change that typically improves Ridge stability. Finally, I keep the existing non-finite imputation to ensure the run completes and produces a valid `submission.csv`.'
- What this solution (achieved 3.42157) has done: 'You’re far worse than the target (lower is better), so we should legitimately improve accuracy with minimal disruption to your current per-type Ridge setup. The biggest safe gain without changing the learning approach is to add a few more “cheap but strong” geometric features (dot products and higher-order distance transforms) and to include coupling `type` as an explicit one-hot feature (while still training per-type, it helps stabilize intercept/feature space in edge cases). I also add a tiny, deterministic per-type alpha selection from a very small fixed grid using a molecule-grouped validation split; this keeps the same model family/training loop (Ridge per type) but avoids a clearly suboptimal global `alpha=1.0`. All changes are deterministic, keep the same data sources, and still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge

BASE_INPUT = "/kaggle/data/champs-scalar-coupling"
TRAIN_PATH = os.path.join(BASE_INPUT, "train.csv")
TEST_PATH = os.path.join(BASE_INPUT, "test.csv")
STRUCT_PATH = os.path.join(BASE_INPUT, "structures.csv")

print("Found train:", os.path.exists(TRAIN_PATH))
print("Found test:", os.path.exists(TEST_PATH))
print("Found structures:", os.path.exists(STRUCT_PATH))



## === cell 1
train = pd.read_csv(
    TRAIN_PATH,
    usecols=[
        "id",
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "scalar_coupling_constant",
    ],
)
test = pd.read_csv(
    TEST_PATH,
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
)

print("train:", train.shape, "test:", test.shape)
print("train types:", train["type"].nunique(), "test types:", test["type"].nunique())



## === cell 2
structures = pd.read_csv(
    STRUCT_PATH,
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
)

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

train_m = train.merge(s0, on=["molecule_name", "atom_index_0"], how="left").merge(
    s1, on=["molecule_name", "atom_index_1"], how="left"
)
test_m = test.merge(s0, on=["molecule_name", "atom_index_0"], how="left").merge(
    s1, on=["molecule_name", "atom_index_1"], how="left"
)

for name, df in [("train_m", train_m), ("test_m", test_m)]:
    missing = df[["x0", "y0", "z0", "x1", "y1", "z1"]].isna().any(axis=1).mean()
    print(name, "missing_coord_frac:", float(missing))




## === cell 3
def add_geom_features(df: pd.DataFrame) -> pd.DataFrame:
    x0 = df["x0"].values
    y0 = df["y0"].values
    z0 = df["z0"].values
    x1 = df["x1"].values
    y1 = df["y1"].values
    z1 = df["z1"].values

    dx = x0 - x1
    dy = y0 - y1
    dz = z0 - z1

    dist2 = dx * dx + dy * dy + dz * dz
    dist = np.sqrt(dist2)
    inv_dist = 1.0 / (dist + 1e-6)

    out = df.copy()
    out["dx"] = dx
    out["dy"] = dy
    out["dz"] = dz
    out["dist2"] = dist2
    out["dist"] = dist
    out["inv_dist"] = inv_dist

    out["abs_dx"] = np.abs(dx)
    out["abs_dy"] = np.abs(dy)
    out["abs_dz"] = np.abs(dz)

    out["ux"] = dx * inv_dist
    out["uy"] = dy * inv_dist
    out["uz"] = dz * inv_dist

    out["inv_dist2"] = 1.0 / (dist2 + 1e-6)
    out["inv_dist3"] = out["inv_dist2"] * inv_dist  # ~ 1/r^3
    out["dist3"] = dist2 * dist
    out["dist4"] = dist2 * dist2

    out["dx_dy"] = dx * dy
    out["dx_dz"] = dx * dz
    out["dy_dz"] = dy * dz

    return out


train_m = add_geom_features(train_m)
test_m = add_geom_features(test_m)

base_num_feats = [
    "dx",
    "dy",
    "dz",
    "abs_dx",
    "abs_dy",
    "abs_dz",
    "dist2",
    "dist",
    "dist3",
    "dist4",
    "inv_dist",
    "inv_dist2",
    "inv_dist3",
    "ux",
    "uy",
    "uz",
    "dx_dy",
    "dx_dz",
    "dy_dz",
]


def impute_nonfinite_with_train_median(train_df, test_df, cols):
    train_imp = train_df.copy()
    test_imp = test_df.copy()

    medians = {}
    for c in cols:
        tr_vals = train_imp[c].astype(np.float64).values
        tr_vals = np.where(np.isfinite(tr_vals), tr_vals, np.nan)
        med = float(np.nanmedian(tr_vals))
        if not np.isfinite(med):
            med = 0.0
        medians[c] = med

        for d in (train_imp, test_imp):
            vals = d[c].astype(np.float64).values
            bad = ~np.isfinite(vals)
            if bad.any():
                vals[bad] = med
                d[c] = vals
    return train_imp, test_imp, medians


train_m, test_m, num_medians = impute_nonfinite_with_train_median(
    train_m, test_m, base_num_feats
)
print(
    "Numeric feature medians used for imputation:",
    {k: round(v, 6) for k, v in num_medians.items()},
)

cats = pd.concat([train_m[["atom_0", "atom_1"]], test_m[["atom_0", "atom_1"]]], axis=0)
atom0_levels = sorted(cats["atom_0"].dropna().unique().tolist())
atom1_levels = sorted(cats["atom_1"].dropna().unique().tolist())

type_levels = sorted(
    pd.concat([train_m["type"], test_m["type"]], axis=0).dropna().unique().tolist()
)


def one_hot(series: pd.Series, levels):
    m = np.zeros((len(series), len(levels)), dtype=np.float32)
    idx = {v: i for i, v in enumerate(levels)}
    vals = series.values
    for r in range(len(vals)):
        v = vals[r]
        if v in idx:
            m[r, idx[v]] = 1.0
    return m


X_atom0_tr = one_hot(train_m["atom_0"], atom0_levels)
X_atom1_tr = one_hot(train_m["atom_1"], atom1_levels)
X_atom0_te = one_hot(test_m["atom_0"], atom0_levels)
X_atom1_te = one_hot(test_m["atom_1"], atom1_levels)

X_type_tr = one_hot(train_m["type"], type_levels)
X_type_te = one_hot(test_m["type"], type_levels)

X_num_tr = train_m[base_num_feats].astype(np.float32).values
X_num_te = test_m[base_num_feats].astype(np.float32).values

num_mean = X_num_tr.mean(axis=0, dtype=np.float64)
num_std = X_num_tr.std(axis=0, dtype=np.float64)
num_std = np.where(num_std > 1e-12, num_std, 1.0)

X_num_tr = ((X_num_tr - num_mean) / num_std).astype(np.float32)
X_num_te = ((X_num_te - num_mean) / num_std).astype(np.float32)

X_tr_all = np.hstack([X_num_tr, X_atom0_tr, X_atom1_tr, X_type_tr])
X_te_all = np.hstack([X_num_te, X_atom0_te, X_atom1_te, X_type_te])

assert np.isfinite(X_tr_all).all(), "Non-finite values remain in training features."
assert np.isfinite(X_te_all).all(), "Non-finite values remain in test features."

y = train_m["scalar_coupling_constant"].astype(np.float32).values

print("X_tr_all shape:", X_tr_all.shape, "X_te_all shape:", X_te_all.shape)




## === cell 4
def make_molecule_val_mask(molecule_names: np.ndarray, frac: float = 0.1) -> np.ndarray:
    mols = pd.Series(molecule_names).unique()
    mols_sorted = np.sort(mols.astype(str))
    n_val = max(1, int(len(mols_sorted) * frac))
    val_mols = set(mols_sorted[:n_val])  # deterministic
    return np.array([m in val_mols for m in molecule_names], dtype=bool)


pred_test = np.zeros(len(test_m), dtype=np.float32)

types = sorted(train_m["type"].unique().tolist())
print("Training per-type models:", len(types))

alpha_grid = [0.05, 0.2, 1.0, 5.0, 20.0]

for t in types:
    tr_idx = train_m["type"].values == t
    te_idx = test_m["type"].values == t

    if not te_idx.any():
        continue

    n_tr = int(tr_idx.sum())
    if n_tr < 50:
        type_mean = (
            float(train_m.loc[tr_idx, "scalar_coupling_constant"].mean())
            if n_tr > 0
            else 0.0
        )
        pred_test[te_idx] = type_mean
        continue

    mol_names_t = train_m.loc[tr_idx, "molecule_name"].values
    val_mask_t = make_molecule_val_mask(mol_names_t, frac=0.1)
    if val_mask_t.sum() < 200 or (~val_mask_t).sum() < 200:
        best_alpha = 1.0
    else:
        X_t = X_tr_all[tr_idx]
        y_t = y[tr_idx]
        X_tr_t, y_tr_t = X_t[~val_mask_t], y_t[~val_mask_t]
        X_va_t, y_va_t = X_t[val_mask_t], y_t[val_mask_t]

        best_alpha = None
        best_mae = None
        for a in alpha_grid:
            m = Ridge(alpha=a, random_state=0)
            m.fit(X_tr_t, y_tr_t)
            p = m.predict(X_va_t)
            mae = float(np.mean(np.abs(p - y_va_t)))
            if (best_mae is None) or (mae < best_mae):
                best_mae = mae
                best_alpha = a

    model = Ridge(alpha=float(best_alpha), random_state=0)
    model.fit(X_tr_all[tr_idx], y[tr_idx])
    pred_test[te_idx] = model.predict(X_te_all[te_idx]).astype(np.float32)

assert (test_m["id"].values == test["id"].values).all()



## === cell 5
submission = pd.DataFrame(
    {"id": test["id"].values, "scalar_coupling_constant": pred_test.astype(np.float64)}
)

assert submission.shape[0] == test.shape[0]
assert list(submission.columns) == ["id", "scalar_coupling_constant"]
assert submission["id"].isna().sum() == 0

out_path = "submission.csv"
submission.to_csv(out_path, index=False, float_format="%.6f")
print("Wrote:", out_path)
print(submission.describe())
