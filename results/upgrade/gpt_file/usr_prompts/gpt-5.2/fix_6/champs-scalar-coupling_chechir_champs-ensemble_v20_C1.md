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

-2.4086276598616023

# 6. Current score

2.92101

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.99777) has done: 'I fix the pipeline so it runs even when the external “champ-preds/nnet/chemistry-of-best-models” input folders aren’t present in your environment by adding safe fallbacks that default to a baseline prediction. I also make the file reading robust to different CSV shapes (with/without index column, with `id` column) and ensure all prediction vectors align to `test` by `id` to avoid silent misalignment. The correlation/heatmap cell be made numeric-only so it won’t crash on the string `type` column. Finally, I always write a valid `ensemble_sub.csv` with the required columns and `.csv` suffix.'
- What this solution (achieved 1.99777) has done: 'Your current score is much better than the target (lower is better), so to move toward the target we should *reduce* performance in a controlled, minimal way without breaking the submission format. The simplest stable approach is to keep your existing ensemble logic intact but “shrink” predictions toward a constant baseline (0.0), which generally worsen MAE and thus increase the log-MAE score toward the target. I add a single calibration parameter `SHRINK_ALPHA` applied after the weighted blend, and set it to a conservative value (0.15) to degrade accuracy while remaining numerically safe and deterministic. Everything else (file loading, alignment by `id`, and CSV writing) stays the same.'
- What this solution (achieved 1.99777) has done: 'Your current score (1.99777, lower-is-better) is far better than the target (-2.4086), so to move *toward* the target we should intentionally and minimally degrade performance. The smallest stable lever in your existing pipeline is the existing post-blend shrink step; I reduce `SHRINK_ALPHA` further so predictions move closer to the constant 0 baseline, which should increase MAE and therefore increase the (log-MAE) score toward the target. I keep all file loading, alignment-by-`id`, blending weights, and submission writing identical to preserve core logic and avoid breaking the pipeline. I also make the shrink operation explicitly include an additive baseline term (still 0.0) to keep the semantics clear and deterministic.'
- What this solution (achieved 1.99777) has done: 'Your current score (1.99777) is far better than the target (-2.4086) on a lower-is-better metric, so to move closer to the target we should intentionally (but minimally and deterministically) worsen predictions. The least invasive lever that preserves your existing ensemble logic is the existing post-blend shrink-to-constant step; by setting `SHRINK_ALPHA` to exactly `0.0`, the submission becomes a constant baseline (0.0) while keeping all loading/alignment/weights unchanged. This should increase MAE substantially and push the log-MAE score downward toward the negative range, reducing the absolute gap to the target. All file paths, blending, alignment by `id`, and CSV writing remain intact to ensure the pipeline still runs end-to-end and produces a valid `ensemble_sub.csv`.'
- What this solution (achieved 2.92101) has done: 'Your current score (1.99777) is much better than the target (-2.4086) on a lower-is-better metric, so we should intentionally degrade predictions to move closer to the target. The smallest, most stable change that preserves your existing ensemble pipeline is to keep your current blending logic intact but switch from an all-zero constant to a per-coupling-type constant baseline (computed from train), which is still a “simple baseline” but generally be less catastrophic than zero and should move the score toward the target. We keep `SHRINK_ALPHA = 0.0` so the ensemble inputs are still loaded exactly the same way, but the final output becomes `type`-wise constants. This is deterministic, avoids any training/validation complexity, and should run well within the time limit while producing a valid `ensemble_sub.csv`.'

# 9. Code solution

## === cell 0
import os, glob
import numpy as np
import pandas as pd

BASE_INPUT_CANDIDATES = [
    "../input/champs-scalar-coupling",
    "/kaggle/input/champs-scalar-coupling",
    "../input",  # sometimes files are directly under ../input
    "/kaggle/input",
]


def find_file(filename):
    """Find a file by trying known base directories; return the first existing path."""
    for base in BASE_INPUT_CANDIDATES:
        p = os.path.join(base, filename)
        if os.path.exists(p):
            return p
    for root in ["../input", "/kaggle/input"]:
        if os.path.exists(root):
            hits = glob.glob(os.path.join(root, "**", filename), recursive=True)
            if hits:
                return hits[0]
    return None


TEST_PATH = find_file("test.csv")
SAMPLE_SUB_PATH = find_file("sample_submission.csv")
TRAIN_PATH = find_file("train.csv")

if TEST_PATH is None or SAMPLE_SUB_PATH is None:
    raise FileNotFoundError(
        f"Could not locate required files test.csv/sample_submission.csv under {BASE_INPUT_CANDIDATES}"
    )

test = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

TARGET = "scalar_coupling_constant"

test = test.sort_values("id").reset_index(drop=True)
sample_sub = sample_sub.sort_values("id").reset_index(drop=True)

assert "id" in test.columns, "test.csv must contain an 'id' column"
assert list(sample_sub.columns) == [
    "id",
    TARGET,
], "sample_submission.csv must have columns: id, scalar_coupling_constant"




## === cell 1
def _read_pred_csv(path, target_col=TARGET):
    """
    Read a prediction CSV in a robust way. Supports:
    - submission-style: columns include ['id', target_col]
    - index-style: first column is id index (index_col=0) and one column is target_col
    - single column of predictions in the same order as test (fallback)
    Returns: pd.Series indexed by id if possible, else a plain Series in file order.
    """
    df = pd.read_csv(path)
    if "id" in df.columns and target_col in df.columns:
        s = df.set_index("id")[target_col]
        return s
    df2 = pd.read_csv(path, index_col=0)
    if target_col in df2.columns:
        s = df2[target_col]
        return s
    if df.shape[1] == 1:
        return df.iloc[:, 0]
    if df2.shape[1] == 1:
        return df2.iloc[:, 0]
    raise ValueError(f"Unrecognized prediction file format for: {path}")


def get_preds_from_files(files, test_ids, how="median", fill_value=0.0):
    """
    Load multiple prediction files and aggregate (median or mean).
    Align by id where possible; otherwise assume file order matches test order.
    Missing files are skipped.
    """
    series_list = []
    used = []
    for f in files:
        if f is None or (not os.path.exists(f)):
            continue
        try:
            s = _read_pred_csv(f)
        except Exception:
            continue

        if isinstance(s.index, pd.Index) and np.issubdtype(s.index.dtype, np.number):
            aligned = pd.Series(index=test_ids, dtype="float64")
            idx = s.index.intersection(test_ids)
            aligned.loc[idx] = s.loc[idx].astype("float64")
            series_list.append(aligned)
        else:
            if len(s) != len(test_ids):
                continue
            series_list.append(pd.Series(s.values, index=test_ids, dtype="float64"))
        used.append(f)

    if not series_list:
        return pd.Series(fill_value, index=test_ids, dtype="float64"), used

    mat = pd.concat(series_list, axis=1)
    if how == "median":
        out = mat.median(axis=1, skipna=True)
    elif how == "mean":
        out = mat.mean(axis=1, skipna=True)
    else:
        raise ValueError("how must be 'median' or 'mean'")

    out = out.fillna(fill_value).astype("float64")
    return out, used


test_ids = test["id"].values



## === cell 2
n1_path = "../input/champ-preds/gnn_median_2302.csv"
n2_path = "../input/champ-preds/gnn0_median_2068.csv"
lgb_a_files = [
    "../input/champ-preds/submission_type_2100.csv",
    "../input/champ-preds/submission_type_2085.csv",
]
lgb_m_files = [
    "../input/champ-preds/lgb_type_full_f286_10.csv",
    "../input/champ-preds/lgb_type_full_f262_10.csv",
]
nnet_files = [
    "../input/nn-seed-10/nnet_sub.csv",
    "../input/nn-seed-11/nnet_sub.csv",
    "../input/nnet-c-seed-10/lgb_type_cv-1.7126_mae0.23572_fd5_10.csv",
    "../input/nnet-c-seed-11/lgb_type_cv-1.70994_mae0.23497_fd5_11.csv",
    "../input/nnet-c-seed-12/lgb_type_cv-1.71029_mae0.23523_fd5_12.csv",
    "../input/nnet-b-seed-10/lgb_type_cv-1.72296_mae0.24019_bags-1_f120_fd5_10.csv",
    "../input/nnet-b-seed-11/lgb_type_cv-1.70647_mae0.2376_bags-1_f120_fd5_11.csv",
    "../input/nnet-b-seed-12/lgb_type_cv-1.69833_mae0.24106_bags-1_f120_fd5_12.csv",
    "../input/nnet-try-seed-11/lgb_type_cv-1.64944_mae0.23965_bags-1_f120_fd5_11.csv",
]
lb_path = "../input/chemistry-of-best-models-1-839/stack_median.csv"


def try_load_single(path, test_ids, fill_value=0.0):
    if path is None or (not os.path.exists(path)):
        return pd.Series(fill_value, index=test_ids, dtype="float64"), False
    try:
        s = _read_pred_csv(path)
        if (
            isinstance(s.index, pd.Index)
            and np.issubdtype(s.index.dtype, np.number)
            and len(set(s.index).intersection(set(test_ids))) > 0
        ):
            out = pd.Series(index=test_ids, dtype="float64")
            idx = s.index.intersection(test_ids)
            out.loc[idx] = s.loc[idx].astype("float64")
            out = out.fillna(fill_value)
        else:
            if len(s) != len(test_ids):
                return pd.Series(fill_value, index=test_ids, dtype="float64"), False
            out = pd.Series(s.values, index=test_ids, dtype="float64")
        return out, True
    except Exception:
        return pd.Series(fill_value, index=test_ids, dtype="float64"), False


test["n1"], ok_n1 = try_load_single(n1_path, test_ids, fill_value=0.0)
test["n2"], ok_n2 = try_load_single(n2_path, test_ids, fill_value=0.0)

test["lgb_a"], used_lgb_a = get_preds_from_files(
    lgb_a_files, test_ids, how="median", fill_value=0.0
)
test["lgb_m"], used_lgb_m = get_preds_from_files(
    lgb_m_files, test_ids, how="median", fill_value=0.0
)
test["nnet"], used_nnet = get_preds_from_files(
    nnet_files, test_ids, how="median", fill_value=0.0
)

test["lb"], ok_lb = try_load_single(lb_path, test_ids, fill_value=0.0)

print(
    "Loaded sources:",
    {
        "n1": ok_n1,
        "n2": ok_n2,
        "lb": ok_lb,
        "lgb_a_n": len(used_lgb_a),
        "lgb_m_n": len(used_lgb_m),
        "nnet_n": len(used_nnet),
    },
)

test.head(10)



## === cell 3
import matplotlib.pyplot as plt
import seaborn as sns

numeric_cols = test.select_dtypes(include=[np.number]).columns.tolist()
exclude = {"id", "atom_index_0", "atom_index_1"}
pred_cols = [c for c in numeric_cols if c not in exclude]

if len(pred_cols) > 1:
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    sns.heatmap(test[pred_cols].corr(), ax=ax)
    plt.tight_layout()
    _ = test[pred_cols].corr()
else:
    _ = pd.DataFrame()

_



## === cell 4
if TRAIN_PATH is None or (not os.path.exists(TRAIN_PATH)):
    raise FileNotFoundError(
        "train.csv is required for the per-type baseline but could not be located."
    )

train = pd.read_csv(TRAIN_PATH, usecols=["type", TARGET])

type_mean = train.groupby("type")[TARGET].mean()
global_mean = float(train[TARGET].mean())

SHRINK_ALPHA = 0.0

raw_blend = (
    test["n1"] * 0.65
    + test["n2"] * 0.07
    + test["lgb_a"] * 0.14
    + test["lgb_m"] * 0.03
    + test["nnet"] * 0.04
    + test["lb"] * 0.07
).astype("float64")

baseline_by_type = test["type"].map(type_mean).fillna(global_mean).astype("float64")

test["final_preds"] = (
    baseline_by_type + SHRINK_ALPHA * (raw_blend - baseline_by_type)
).astype("float64")

test[["id", "type", "final_preds"]].head(20)



## === cell 5
submission = sample_sub.copy()
submission = submission.sort_values("id").reset_index(drop=True)

pred_series = pd.Series(test["final_preds"].values, index=test["id"].values)
submission[TARGET] = submission["id"].map(pred_series).astype("float64")
submission[TARGET] = submission[TARGET].fillna(float(global_mean))

submission.to_csv("ensemble_sub.csv", index=False)
print("Wrote submission:", "ensemble_sub.csv", "rows:", len(submission))
submission.head(20)



## === cell 6
assert os.path.exists("ensemble_sub.csv")
assert submission.shape[0] == sample_sub.shape[0]
assert list(submission.columns) == ["id", TARGET]
submission.describe(include="all")
