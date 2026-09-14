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

-1.36251

# 6. Current score

1.23566

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.18497) has done: 'I replace the missing `../input/champstacks` dependency (which causes the first runtime error) with a self-contained baseline pipeline that uses only the provided competition data files, so it runs end-to-end in this Kaggle environment. To keep changes minimal while still improving score from “no submission” to a reasonable baseline, I generate predictions as the per-type median `scalar_coupling_constant` from the training set (a standard strong baseline for this competition’s per-type MAE metric). I also fix deprecated `np.bool` usage and ensure the submission has exactly the required columns (`id,scalar_coupling_constant`) and a `.csv` suffix. The correlation/stacking visualization cells be guarded so they don’t fail when external stack files are absent.'
- What this solution (achieved 1.23566) has done: 'Your current baseline (per-type median) is far from the target (lower-is-better), so we need a small but real modeling step that still preserves the “core logic” of simple per-type aggregation. The minimal upgrade is to predict the per-type mean (which optimizes MAE/L1 better than median in this competition’s distribution) and then apply a tiny per-type calibration using out-of-fold residual bias estimated by molecule-group KFold (to respect the molecule split rule). This keeps the approach as a lightweight statistics-based baseline (no new model architecture/training loop), but typically reduces log(MAE) substantially versus a pure median lookup. The submission writing and stacking-guard cells stay intact, and the code remains end-to-end with a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

DATA_DIR = "/kaggle/input/champs-scalar-coupling"
ALT_DATA_DIR = (
    "/kaggle/data/champs-scalar-coupling"  # fallback for the provided file tree
)

if not os.path.exists(DATA_DIR) and os.path.exists(ALT_DATA_DIR):
    DATA_DIR = ALT_DATA_DIR

print("Using DATA_DIR:", DATA_DIR)
print("Files (head):", sorted(os.listdir(DATA_DIR))[:10])



## === cell 1
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

train = pd.read_csv(
    train_path, usecols=["molecule_name", "type", "scalar_coupling_constant"]
)
test = pd.read_csv(test_path, usecols=["id", "type"])
sample_sub = pd.read_csv(sample_path, usecols=["id"])

print("train:", train.shape, "test:", test.shape, "sample:", sample_sub.shape)
print(train.head())



## === cell 2
try:
    from sklearn.model_selection import GroupKFold
except Exception as e:
    raise RuntimeError(
        "scikit-learn is required (sklearn) but not available in this environment."
    ) from e

type_mean = train.groupby("type")["scalar_coupling_constant"].mean()
global_mean = float(train["scalar_coupling_constant"].mean())

gkf = GroupKFold(n_splits=5)
oof_pred = np.empty(len(train), dtype=np.float64)

types_arr = train["type"].values
y_arr = train["scalar_coupling_constant"].values.astype(np.float64)
groups_arr = train["molecule_name"].values

for fold, (tr_idx, va_idx) in enumerate(gkf.split(train, y_arr, groups=groups_arr), 1):
    tr = train.iloc[tr_idx]
    fold_type_mean = tr.groupby("type")["scalar_coupling_constant"].mean()
    fold_global_mean = float(tr["scalar_coupling_constant"].mean())

    va_types = types_arr[va_idx]
    va_pred = (
        pd.Series(va_types)
        .map(fold_type_mean)
        .fillna(fold_global_mean)
        .values.astype(np.float64)
    )
    oof_pred[va_idx] = va_pred
    if fold == 1:
        print("Fold", fold, "train rows:", len(tr_idx), "valid rows:", len(va_idx))

residual = y_arr - oof_pred
bias_by_type = (
    pd.DataFrame({"type": types_arr, "residual": residual})
    .groupby("type")["residual"]
    .mean()
)
global_bias = float(residual.mean())

print("Computed per-type bias (head):")
print(bias_by_type.sort_index().head(10))
print("Global mean:", global_mean, "Global bias:", global_bias)



## === cell 3
test = test.copy()
test_base = test["type"].map(type_mean).fillna(global_mean).astype(np.float64)
test_bias = test["type"].map(bias_by_type).fillna(global_bias).astype(np.float64)

test["scalar_coupling_constant"] = (test_base + test_bias).astype(np.float32)

sub = pd.merge(
    sample_sub, test[["id", "scalar_coupling_constant"]], on="id", how="left"
)
sub["scalar_coupling_constant"] = (
    sub["scalar_coupling_constant"].fillna(global_mean).astype(np.float32)
)

sub.head(), sub.shape



## === cell 4
out_path = "submission.csv"
sub[["id", "scalar_coupling_constant"]].to_csv(
    out_path, index=False, float_format="%.6f"
)
print("Wrote:", out_path, "rows:", len(sub))



## === cell 5
diag = pd.DataFrame(
    {
        "type_mean": type_mean,
        "oof_bias": bias_by_type,
    }
).sort_index()
print("Type mean + OOF bias used (head):")
print(diag.head(10))
print("Global mean:", global_mean, "Global bias:", global_bias)



## === cell 6
sub_path = "../input/champstacks"
all_files = []
concat_sub = None
ncol = 0
print(
    "Note: champstacks path not used; generated calibrated per-type mean submission instead."
)



## === cell 7
if os.path.exists(sub_path):
    all_files = os.listdir(sub_path)
    outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files]
    concat_sub = pd.concat(outs, axis=1)
    cols = list(map(lambda x: "champ" + str(x), range(len(concat_sub.columns))))
    concat_sub.columns = cols
    concat_sub.reset_index(inplace=True)
    ncol = concat_sub.shape[1]
    print("Loaded stacked submissions:", len(all_files), "ncol:", ncol)
else:
    print("champstacks folder not found; skipping stacking workflow.")



## === cell 8
if concat_sub is not None and ncol > 1:
    corr = concat_sub.iloc[:, 1:ncol].corr()
    mask = np.zeros_like(corr, dtype=bool)
    print("Correlation computed for stack matrix:", corr.shape)
else:
    print("No concat_sub available; skipping correlation.")



## === cell 9
cutoff_lo = -37
cutoff_hi = 205
print("cutoff_lo, cutoff_hi:", cutoff_lo, cutoff_hi)



## === cell 10
if concat_sub is not None and "champ_mean" in concat_sub.columns:
    concat_sub["scalar_coupling_constant"] = concat_sub["champ_mean"]
    concat_sub[["id", "scalar_coupling_constant"]].to_csv(
        "stack_mean.csv", index=False, float_format="%.6f"
    )
    print("Wrote stack_mean.csv")
else:
    print("Skipping stack_mean.csv (no stacked inputs).")



## === cell 11
if concat_sub is not None and "champ_median" in concat_sub.columns:
    concat_sub["scalar_coupling_constant"] = concat_sub["champ_median"]
    concat_sub[["id", "scalar_coupling_constant"]].to_csv(
        "stack_median.csv", index=False, float_format="%.6f"
    )
    print("Wrote stack_median.csv")
else:
    print("Skipping stack_median.csv (no stacked inputs).")



## === cell 12
assert os.path.exists(out_path), "submission.csv was not created"
assert sub["id"].isna().sum() == 0, "Missing ids in submission"
assert (
    sub["scalar_coupling_constant"].isna().sum() == 0
), "Missing predictions in submission"
print("Done. Final submission file:", out_path)
print(sub.head())
