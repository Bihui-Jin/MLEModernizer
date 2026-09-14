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

0.90666

# 6. Current score

1.21011

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.34178) has done: 'Diagnosis: Cell 2 crashes because the merged structure features (`x`, `y`, `z`) introduce missing values (NaNs) for some rows, and `ExtraTreesRegressor` in scikit-learn refuses NaNs at predict-time (and also at fit-time, though your split subset likely had fewer NaNs). The feature list `col` includes these coordinates, so `test[col]` contains NaNs and triggers the `ValueError`.  
Patch summary: In cell 2 only, add a minimal preprocessing step that imputes missing feature values using medians computed from the training features, then apply the same fill values to the validation split and test set. This preserves the model, training loop, and evaluation semantics while making the inputs finite for sklearn.  
Updated cells: Only cell 2 is modified.  
Compatibility notes for cell k+1: No interface changes—`submission.csv` is still written with the same columns and formats; variables (`reg`, `test`) remain as expected.  
Assumptions: Median imputation is acceptable as a minimal deterministic fix and all model features in `col` are numeric after the earlier encoding/merge steps.'
- What this solution (achieved 1.28849) has done: 'Your current score (1.34178, lower-is-better) is far worse than the target (0.90666), so we should improve performance with minimal, safe changes. The biggest easy gain while preserving the same model/training approach is to actually use more of the available “official” auxiliary feature files that you already had in the code but commented out (potential_energy, mulliken_charges, dipole_moments, magnetic_shielding_tensors); these are standard strong signal for this competition and should move the score significantly toward the target without changing the core estimator or training loop. I keep your ExtraTreesRegressor setup and the median-imputation step, just expanding the feature table via the existing merges (and aligning the merges to avoid accidental duplicates). The script still writes `submission.csv` with the required columns.'
- What this solution (achieved 1.89821) has done: 'Diagnosis: The crash happens when fitting `ExtraTreesRegressor` because `train_X` still contains non-numeric (string) columns (e.g., `atom_0`, `atom_1`), so scikit-learn cannot convert values like `'C'` to float. `median(numeric_only=True)` only fills numeric columns but does not remove or encode string columns, so the error persists at model fit time.  
Patch summary: In cell 2, restrict the feature column list `col` to numeric dtypes only (after excluding ID/target/text fields), so the model receives purely numeric matrices; keep the rest of the logic (median fill, split, model, training, prediction, submission) unchanged.  
Updated cells: Only cell 2 is modified.  
Compatibility notes for cell k+1: No interface changes—`train_X`, `test_X`, `reg`, and `test["scalar_coupling_constant"]` are still created as before; only the selected feature set is filtered to numeric columns.  
Assumptions: The intended baseline model uses only numeric engineered features; dropping raw string atom labels is consistent with the existing use of `LabelEncoder`-based `type0..type3` features.'
- What this solution (achieved 1.21951) has done: 'Your current score (1.89821, lower-is-better) is far worse than the target (0.90666), so we should improve model performance with minimal, safe edits while keeping the same ExtraTrees approach. The biggest issue is that you’re training a single regressor across all coupling `type`s, while the competition metric averages MAE per type; training one model per `type` (same estimator, same features) is a minimal change that usually reduces error a lot for this competition. I keep your feature engineering/merges and median-imputation, but fit/predict separately per `type` and then stitch predictions back in `id` order. I also add a couple of simple, type-agnostic geometric features (abs(dx/dy/dz) and squared distance) that don’t change the approach but typically help ExtraTrees.'
- What this solution (achieved 1.20212) has done: 'You’re still materially above the target (1.21951 vs 0.90666, lower-is-better), so we should improve accuracy but keep the same ExtraTrees-per-type core logic. The smallest high-impact change here is to add a few standard “pairwise” features for atom_index_0 *and* atom_index_1 from files you already load for atom_index_0 (mulliken charges + magnetic shielding tensors); these help because the coupling depends on both atoms, and this doesn’t change the model/training approach. I also add two tiny deterministic geometric features (inverse distance and distance^3) which commonly help tree models without altering semantics. Everything else (per-type loop, imputation-by-median, ExtraTrees settings, submission writing) stays the same.'
- What this solution (achieved 1.21011) has done: 'The main timeout drivers are repeated heavy merges on 4M+ rows, re-reading the same large CSVs multiple times, and doing per-type pandas `fillna` (which copies large frames). I keep the exact same features and per-type ExtraTrees training/prediction logic, but speed it up by (1) reading only needed columns with explicit dtypes, (2) concatenating train+test once and doing each merge once (instead of twice), (3) loading mulliken charges and magnetic shielding tensors once and self-merging via two renamed views, and (4) computing features using NumPy arrays to reduce pandas overhead and intermediate copies. The model, hyperparameters, validation diagnostic, and prediction semantics remain unchanged (only negligible float differences possible).'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from sklearn import preprocessing, ensemble, model_selection, metrics

SEED = 99
np.random.seed(SEED)

DATA_DIR = "../input"

train_dtypes = {
    "id": "int32",
    "molecule_name": "category",
    "atom_index_0": "int16",
    "atom_index_1": "int16",
    "type": "category",
    "scalar_coupling_constant": "float32",
}
test_dtypes = {
    "id": "int32",
    "molecule_name": "category",
    "atom_index_0": "int16",
    "atom_index_1": "int16",
    "type": "category",
}
usecols_train = [
    "id",
    "molecule_name",
    "atom_index_0",
    "atom_index_1",
    "type",
    "scalar_coupling_constant",
]
usecols_test = ["id", "molecule_name", "atom_index_0", "atom_index_1", "type"]

train = pd.read_csv(f"{DATA_DIR}/train.csv", usecols=usecols_train, dtype=train_dtypes)
test = pd.read_csv(f"{DATA_DIR}/test.csv", usecols=usecols_test, dtype=test_dtypes)
sub = pd.read_csv(
    f"{DATA_DIR}/sample_submission.csv", usecols=["id", "scalar_coupling_constant"]
)
print(train.shape, test.shape, sub.shape)

train["atom"] = train["type"].astype("string").str[3]
test["atom"] = test["type"].astype("string").str[3]

lbl = preprocessing.LabelEncoder()
type_str_tr = train["type"].astype("string")
type_str_te = test["type"].astype("string")
for i in range(4):
    col = f"type{i}"
    train[col] = lbl.fit_transform(type_str_tr.str[i].to_numpy())
    test[col] = lbl.transform(type_str_te.str[i].to_numpy())

train["_is_train"] = 1
test["_is_train"] = 0
df_all = pd.concat([train, test], axis=0, ignore_index=True, copy=False)
del train, test

structures = pd.read_csv(
    f"{DATA_DIR}/structures.csv",
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
    dtype={
        "molecule_name": "category",
        "atom_index": "int16",
        "atom": "category",
        "x": "float32",
        "y": "float32",
        "z": "float32",
    },
)

mol_agg = (
    structures.groupby("molecule_name", sort=False)
    .agg(
        n_atoms=("atom_index", "count"),
        x_mean=("x", "mean"),
        y_mean=("y", "mean"),
        z_mean=("z", "mean"),
        x_std=("x", "std"),
        y_std=("y", "std"),
        z_std=("z", "std"),
    )
    .reset_index()
)
df_all = pd.merge(df_all, mol_agg, how="left", on="molecule_name", copy=False)
del mol_agg

structures_1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "x": "x_1",
        "y": "y_1",
        "z": "z_1",
    }
)
df_all = pd.merge(
    df_all, structures_1, how="left", on=["molecule_name", "atom_index_1"], copy=False
)
del structures_1

structures_0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "x_0",
        "y": "y_0",
        "z": "z_0",
    }
)
df_all = pd.merge(
    df_all, structures_0, how="left", on=["molecule_name", "atom_index_0"], copy=False
)
del structures_0, structures

x0 = df_all["x_0"].to_numpy(dtype=np.float64, copy=False)
y0 = df_all["y_0"].to_numpy(dtype=np.float64, copy=False)
z0 = df_all["z_0"].to_numpy(dtype=np.float64, copy=False)
x1 = df_all["x_1"].to_numpy(dtype=np.float64, copy=False)
y1 = df_all["y_1"].to_numpy(dtype=np.float64, copy=False)
z1 = df_all["z_1"].to_numpy(dtype=np.float64, copy=False)

dx = x0 - x1
dy = y0 - y1
dz = z0 - z1

dist2 = dx * dx + dy * dy + dz * dz
dist = np.sqrt(dist2)

df_all["dist"] = dist
df_all["dist2"] = dist2
df_all["abs_dx"] = np.abs(dx)
df_all["abs_dy"] = np.abs(dy)
df_all["abs_dz"] = np.abs(dz)

eps = 1e-6
df_all["inv_dist"] = 1.0 / (dist + eps)
df_all["dist3"] = dist * dist2

x_mean = df_all["x_mean"].to_numpy(dtype=np.float64, copy=False)
y_mean = df_all["y_mean"].to_numpy(dtype=np.float64, copy=False)
z_mean = df_all["z_mean"].to_numpy(dtype=np.float64, copy=False)

x0_c = x0 - x_mean
y0_c = y0 - y_mean
z0_c = z0 - z_mean
x1_c = x1 - x_mean
y1_c = y1 - y_mean
z1_c = z1 - z_mean

df_all["x0_c"] = x0_c
df_all["y0_c"] = y0_c
df_all["z0_c"] = z0_c
df_all["x1_c"] = x1_c
df_all["y1_c"] = y1_c
df_all["z1_c"] = z1_c

r0_c = np.sqrt(x0_c * x0_c + y0_c * y0_c + z0_c * z0_c)
r1_c = np.sqrt(x1_c * x1_c + y1_c * y1_c + z1_c * z1_c)
df_all["r0_c"] = r0_c
df_all["r1_c"] = r1_c
df_all["abs_dr_c"] = np.abs(r0_c - r1_c)

pe = pd.read_csv(
    f"{DATA_DIR}/potential_energy.csv",
    usecols=["molecule_name", "potential_energy"],
    dtype={"molecule_name": "category", "potential_energy": "float32"},
)
df_all = pd.merge(df_all, pe, how="left", on=["molecule_name"], copy=False)
del pe

mc = pd.read_csv(
    f"{DATA_DIR}/mulliken_charges.csv",
    usecols=["molecule_name", "atom_index", "mulliken_charge"],
    dtype={
        "molecule_name": "category",
        "atom_index": "int16",
        "mulliken_charge": "float32",
    },
)
mc0 = mc.rename(
    columns={"atom_index": "atom_index_0", "mulliken_charge": "mulliken_charge_0"}
)
df_all = pd.merge(
    df_all, mc0, how="left", on=["molecule_name", "atom_index_0"], copy=False
)
mc1 = mc.rename(
    columns={"atom_index": "atom_index_1", "mulliken_charge": "mulliken_charge_1"}
)
df_all = pd.merge(
    df_all, mc1, how="left", on=["molecule_name", "atom_index_1"], copy=False
)
del mc, mc0, mc1

dm = pd.read_csv(
    f"{DATA_DIR}/dipole_moments.csv",
    usecols=["molecule_name", "X", "Y", "Z"],
    dtype={"molecule_name": "category", "X": "float32", "Y": "float32", "Z": "float32"},
)
df_all = pd.merge(df_all, dm, how="left", on=["molecule_name"], copy=False)
del dm

mst = pd.read_csv(
    f"{DATA_DIR}/magnetic_shielding_tensors.csv",
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
        "atom_index": "int16",
        "XX": "float32",
        "YX": "float32",
        "ZX": "float32",
        "XY": "float32",
        "YY": "float32",
        "ZY": "float32",
        "XZ": "float32",
        "YZ": "float32",
        "ZZ": "float32",
    },
)
mst0 = mst.rename(
    columns={
        "atom_index": "atom_index_0",
        "XX": "XX_0",
        "YX": "YX_0",
        "ZX": "ZX_0",
        "XY": "XY_0",
        "YY": "YY_0",
        "ZY": "ZY_0",
        "XZ": "XZ_0",
        "YZ": "YZ_0",
        "ZZ": "ZZ_0",
    }
)
df_all = pd.merge(
    df_all, mst0, how="left", on=["molecule_name", "atom_index_0"], copy=False
)

mst1 = mst.rename(
    columns={
        "atom_index": "atom_index_1",
        "XX": "XX_1",
        "YX": "YX_1",
        "ZX": "ZX_1",
        "XY": "XY_1",
        "YY": "YY_1",
        "ZY": "ZY_1",
        "XZ": "XZ_1",
        "YZ": "YZ_1",
        "ZZ": "ZZ_1",
    }
)
df_all = pd.merge(
    df_all, mst1, how="left", on=["molecule_name", "atom_index_1"], copy=False
)
del mst, mst0, mst1

train = (
    df_all[df_all["_is_train"] == 1].drop(columns=["_is_train"]).reset_index(drop=True)
)
test = (
    df_all[df_all["_is_train"] == 0]
    .drop(columns=["_is_train", "scalar_coupling_constant"])
    .reset_index(drop=True)
)
del df_all

print(train.shape, test.shape, sub.shape)



## === cell 1
train.head()



## === cell 2
base_col = [
    c
    for c in train.columns
    if c not in ["id", "molecule_name", "scalar_coupling_constant", "type", "atom"]
]
base_col = [c for c in base_col if pd.api.types.is_numeric_dtype(train[c])]

test_pred = np.empty(len(test), dtype=np.float64)
test_pred[:] = np.nan

rng_state = 99
val_scores = {}

types = sorted(train["type"].unique())
print("Num types:", len(types), "Types:", types)

for t in types:
    tr_mask = train["type"].to_numpy() == t
    te_mask = test["type"].to_numpy() == t

    tr_t = train.loc[tr_mask, base_col + ["scalar_coupling_constant"]]
    te_t = test.loc[te_mask, base_col]

    fill_values_t = tr_t[base_col].median(numeric_only=True)

    X_tr = tr_t[base_col].fillna(fill_values_t)
    y_tr = tr_t["scalar_coupling_constant"].to_numpy()
    X_te = te_t.fillna(fill_values_t)

    reg = ensemble.ExtraTreesRegressor(n_jobs=-1, random_state=4, n_estimators=20)

    n_val_cap = 250_000
    if len(tr_t) > 10_000:
        X_small = X_tr.tail(min(n_val_cap, len(tr_t)))
        y_small = y_tr[-len(X_small) :]
        x1, x2, y1, y2 = model_selection.train_test_split(
            X_small,
            y_small,
            test_size=0.2,
            random_state=rng_state,
        )
        reg.fit(x1, y1)
        mae = metrics.mean_absolute_error(y2, reg.predict(x2))
        val_scores[t] = float(np.log(mae))
        print(f"type={t:>4s} log(MAE)={val_scores[t]:.5f} (diagnostic)")

    reg.fit(X_tr, y_tr)
    test_pred[te_mask] = reg.predict(X_te)

n_missing = int(np.isnan(test_pred).sum())
print("Missing test predictions:", n_missing)
if n_missing != 0:
    global_med = float(train["scalar_coupling_constant"].median())
    test_pred[np.isnan(test_pred)] = global_med

test_out = test[["id"]].copy()
test_out["scalar_coupling_constant"] = test_pred

test_out.to_csv("submission.csv", float_format="%.9f", index=False)
print("Wrote submission.csv with shape:", test_out.shape)
print(test_out.head())
