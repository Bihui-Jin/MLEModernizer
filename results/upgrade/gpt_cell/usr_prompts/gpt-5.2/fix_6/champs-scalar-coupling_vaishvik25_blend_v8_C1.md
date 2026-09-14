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

-1.30873

# 6. Current score

1.26183

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.99777) has done: 'Diagnosis: Cell 1 crashes because it tries to read four “blender” submission CSVs from `../input/blender/…`, but that directory/files do not exist in this environment (only the competition input files under `../input/champs-scalar-coupling/` are available). The rest of the notebook expects `sub1/sub2/sub3/sub6` to exist and to have a `scalar_coupling_constant` column so cell 2 can blend them.  
Patch summary: Modify cell 1 to load those submissions if present, otherwise fall back deterministically to the provided `sample_submission.csv` for each missing file, ensuring the expected columns exist and keeping the same interfaces for cell 2.  
Updated cells: Only cell 1 is changed.  
Compatibility notes for cell k+1: `sub1`, `sub2`, `sub3`, and `sub6` remain pandas DataFrames with a `scalar_coupling_constant` column (and `id`), so cell 2 can run unchanged.  
Assumptions: When blender files are unavailable, using `sample_submission.csv` as a fallback is acceptable to unblock execution; it preserves schema but not original ensemble performance.'
- What this solution (achieved 1.99777) has done: 'The crash happens because recent pandas versions disallow passing the plot “kind” as a positional argument to `Series.plot()`. In this cell, `sub1['scalar_coupling_constant'].plot('hist', bins=100)` triggers a `TypeError`. The minimal fix is to pass `kind='hist'` as a keyword argument, keeping the same histogram semantics and output. No other cells or variables are changed, so downstream compatibility is preserved.'
- What this solution (achieved 1.23566) has done: 'Your current score is poor because the intended blender submission files are missing, so the blend effectively becomes a near-constant/zero prediction from `sample_submission.csv`. The smallest legitimate improvement (without changing the approach of “blending submissions”) is to replace those missing submissions with a simple baseline model trained on the provided training data, using only existing tabular features and the same target. We create per-coupling-type mean targets (a standard CHAMPS baseline) and, when a blender file is absent, fill predictions by mapping the test `type` to those means. This keeps the core logic as an ensemble/blend while making the substituted inputs far more informative, moving the score substantially toward the target.'
- What this solution (achieved 1.98802) has done: 'Your current blend is effectively just four identical copies of the same type-mean baseline (because all “blender” files are missing), so the weighted average cannot improve beyond that baseline. To move the score closer to the target with minimal change and without altering the overall “blend submissions” logic, I generate four *slightly different but legitimate* baselines when files are missing: (1) type mean, (2) type median, (3) type mean with molecule-level energy feature, and (4) a shrinkage blend between type mean and global mean. This preserves the existing blending cell unchanged while making the ensemble meaningfully better calibrated per type and molecule, which should reduce MAE and move the log-MAE down toward the target. The submission file path/format remains the same.'
- What this solution (achieved 1.26183) has done: 'Your current score (1.98802, lower-is-better) is far from the target (-1.30873), so we should make a small but meaningful accuracy improvement while keeping the “blend submissions / fallback predictions” core logic intact. The biggest win with minimal disruption is to make the energy-based fallback more informative by using out-of-fold (OOF) smoothing on `(type, pe_bin)` means to reduce noise/leakage-like overfitting and improve generalization on the molecule-split test. This keeps the same overall baseline idea (type + potential_energy bin), but stabilizes it with simple CV over molecules and Bayesian-style shrinkage. All I/O paths and the blending cell stay the same, and we still write a valid `submission1236.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print(os.listdir("../input"))



## === cell 1
COMP_PATH = "../input/champs-scalar-coupling"
SAMPLE_PATH = f"{COMP_PATH}/sample_submission.csv"
TRAIN_PATH = f"{COMP_PATH}/train.csv"
TEST_PATH = f"{COMP_PATH}/test.csv"
PE_PATH = f"{COMP_PATH}/potential_energy.csv"


def _type_mean_baseline_submission():
    train = pd.read_csv(TRAIN_PATH, usecols=["type", "scalar_coupling_constant"])
    type_mean = train.groupby("type")["scalar_coupling_constant"].mean()

    test = pd.read_csv(TEST_PATH, usecols=["id", "type"])
    pred = test["type"].map(type_mean).astype(np.float32)

    global_mean = float(train["scalar_coupling_constant"].mean())
    pred = pred.fillna(global_mean)

    sub = pd.DataFrame(
        {"id": test["id"].values, "scalar_coupling_constant": pred.values}
    )
    return sub


def _type_median_baseline_submission():
    train = pd.read_csv(TRAIN_PATH, usecols=["type", "scalar_coupling_constant"])
    type_median = train.groupby("type")["scalar_coupling_constant"].median()

    test = pd.read_csv(TEST_PATH, usecols=["id", "type"])
    pred = test["type"].map(type_median).astype(np.float32)

    global_median = float(train["scalar_coupling_constant"].median())
    pred = pred.fillna(global_median)

    return pd.DataFrame(
        {"id": test["id"].values, "scalar_coupling_constant": pred.values}
    )


def _type_mean_plus_energy_baseline_submission():
    train = pd.read_csv(
        TRAIN_PATH,
        usecols=["molecule_name", "type", "scalar_coupling_constant"],
    )
    pe = pd.read_csv(PE_PATH, usecols=["molecule_name", "potential_energy"])
    train = train.merge(pe, on="molecule_name", how="left")

    test = pd.read_csv(TEST_PATH, usecols=["id", "molecule_name", "type"])
    test = test.merge(pe, on="molecule_name", how="left")

    all_pe = pd.concat([train["potential_energy"], test["potential_energy"]], axis=0)
    if all_pe.notna().any():
        try:
            bins = pd.qcut(all_pe, q=50, duplicates="drop")
        except Exception:
            bins = pd.cut(all_pe, bins=50)
    else:
        bins = pd.Series([np.nan] * len(all_pe))

    train_bins = bins.iloc[: len(train)]
    test_bins = bins.iloc[len(train) :].reset_index(drop=True)

    train = train.assign(pe_bin=train_bins.values)
    test = test.assign(pe_bin=test_bins.values)

    global_mean = float(train["scalar_coupling_constant"].mean())
    type_sum = train.groupby("type")["scalar_coupling_constant"].sum()
    type_cnt = train.groupby("type")["scalar_coupling_constant"].size()
    type_mean = (type_sum / type_cnt).astype(np.float64)

    mols = train["molecule_name"].unique()
    mols = np.sort(mols)
    n_folds = 5
    fold_id = (np.arange(len(mols)) % n_folds).astype(np.int32)
    mol2fold = pd.Series(fold_id, index=mols)
    train_fold = train["molecule_name"].map(mol2fold).astype(np.int32)

    g_all = train.groupby(["type", "pe_bin"])["scalar_coupling_constant"].agg(
        ["sum", "count"]
    )
    g_all_sum = g_all["sum"].astype(np.float64)
    g_all_cnt = g_all["count"].astype(np.int64)

    g_fold = (
        train.groupby([train_fold, "type", "pe_bin"])["scalar_coupling_constant"]
        .agg(["sum", "count"])
        .reset_index()
    )
    g_fold = g_fold.rename(columns={"sum": "f_sum", "count": "f_cnt"})
    g_fold_key = pd.MultiIndex.from_frame(
        g_fold[
            [
                train_fold.name if hasattr(train_fold, "name") else "molecule_name",
                "type",
                "pe_bin",
            ]
        ]
    )  # not used
    g_fold = g_fold.rename(columns={0: "fold"}) if 0 in g_fold.columns else g_fold
    if "fold" not in g_fold.columns:
        g_fold = g_fold.rename(columns={g_fold.columns[0]: "fold"})

    train_key = pd.DataFrame(
        {
            "fold": train_fold.values,
            "type": train["type"].values,
            "pe_bin": train["pe_bin"].values,
        }
    )
    train_key = train_key.merge(g_fold, on=["fold", "type", "pe_bin"], how="left")

    idx_tb = pd.MultiIndex.from_frame(train[["type", "pe_bin"]])
    all_sum = g_all_sum.reindex(idx_tb).to_numpy(dtype=np.float64)
    all_cnt = g_all_cnt.reindex(idx_tb).to_numpy(dtype=np.float64)

    f_sum = train_key["f_sum"].to_numpy(dtype=np.float64)
    f_cnt = train_key["f_cnt"].to_numpy(dtype=np.float64)

    oof_sum = all_sum - np.nan_to_num(f_sum, nan=0.0)
    oof_cnt = all_cnt - np.nan_to_num(f_cnt, nan=0.0)

    strength = 50.0
    tmean_arr = train["type"].map(type_mean).to_numpy(dtype=np.float64)
    oof_mean_tb = np.where(
        oof_cnt > 0,
        (oof_sum + strength * tmean_arr) / (oof_cnt + strength),
        tmean_arr,
    )

    test_idx_tb = pd.MultiIndex.from_frame(test[["type", "pe_bin"]])
    test_sum = g_all_sum.reindex(test_idx_tb).to_numpy(dtype=np.float64)
    test_cnt = g_all_cnt.reindex(test_idx_tb).to_numpy(dtype=np.float64)
    test_tmean = test["type"].map(type_mean).to_numpy(dtype=np.float64)

    test_mean_tb = np.where(
        np.isfinite(test_sum) & np.isfinite(test_cnt) & (test_cnt > 0),
        (np.nan_to_num(test_sum, nan=0.0) + strength * test_tmean)
        / (np.nan_to_num(test_cnt, nan=0.0) + strength),
        test_tmean,
    )

    base_test = test_tmean
    pred_test = (base_test + (test_mean_tb - base_test)).astype(np.float32)

    pred_test = (
        pd.Series(pred_test).fillna(np.float32(global_mean)).to_numpy(dtype=np.float32)
    )

    return pd.DataFrame(
        {"id": test["id"].values, "scalar_coupling_constant": pred_test}
    )


def _type_mean_shrinkage_baseline_submission(alpha=0.85):
    train = pd.read_csv(TRAIN_PATH, usecols=["type", "scalar_coupling_constant"])
    g = float(train["scalar_coupling_constant"].mean())
    type_mean = train.groupby("type")["scalar_coupling_constant"].mean()

    test = pd.read_csv(TEST_PATH, usecols=["id", "type"])
    tm = test["type"].map(type_mean).astype(np.float64)
    pred = (alpha * tm + (1.0 - alpha) * g).astype(np.float32)
    pred = pred.fillna(np.float32(g))

    return pd.DataFrame(
        {"id": test["id"].values, "scalar_coupling_constant": pred.values}
    )


_FALLBACKS = {}


def _read_submission_or_fallback(path, fallback_kind="mean"):
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        if fallback_kind not in _FALLBACKS:
            if fallback_kind == "mean":
                _FALLBACKS[fallback_kind] = _type_mean_baseline_submission()
            elif fallback_kind == "median":
                _FALLBACKS[fallback_kind] = _type_median_baseline_submission()
            elif fallback_kind == "energy":
                _FALLBACKS[fallback_kind] = _type_mean_plus_energy_baseline_submission()
            elif fallback_kind == "shrink":
                _FALLBACKS[fallback_kind] = _type_mean_shrinkage_baseline_submission(
                    alpha=0.85
                )
            else:
                _FALLBACKS[fallback_kind] = _type_mean_baseline_submission()
        df = _FALLBACKS[fallback_kind].copy()

    if "scalar_coupling_constant" not in df.columns:
        df["scalar_coupling_constant"] = 0.0

    if "id" in df.columns:
        df = df[["id", "scalar_coupling_constant"]]
    return df


sub1 = _read_submission_or_fallback(
    "../input/blender/LGB_2019-07-11_-1.4378.csv", fallback_kind="energy"
)
sub2 = _read_submission_or_fallback(
    "../input/blender/submission-2.csv", fallback_kind="mean"
)
sub3 = _read_submission_or_fallback(
    "../input/blender/stack_minmax_median.csv", fallback_kind="median"
)
sub6 = _read_submission_or_fallback(
    "../input/blender/workingsubmission-test.csv", fallback_kind="shrink"
)

print(sub1["scalar_coupling_constant"].describe())
print(sub2["scalar_coupling_constant"].describe())
print(sub3["scalar_coupling_constant"].describe())
print(sub6["scalar_coupling_constant"].describe())



## === cell 2
sub2 = sub2.set_index("id").loc[sub1["id"].values].reset_index()
sub3 = sub3.set_index("id").loc[sub1["id"].values].reset_index()
sub6 = sub6.set_index("id").loc[sub1["id"].values].reset_index()

sub1["scalar_coupling_constant"] = (
    0.3 * sub1["scalar_coupling_constant"]
    + 0.3 * sub2["scalar_coupling_constant"]
    + 0.20 * sub3["scalar_coupling_constant"]
    + 0.20 * sub6["scalar_coupling_constant"]
)
sub1.to_csv("submission1236.csv", index=False)



## === cell 3
sub1["scalar_coupling_constant"].plot(kind="hist", bins=100)
