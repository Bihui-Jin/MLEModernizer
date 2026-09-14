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

-2.3117754596968823

# 6. Current score

2.7291

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.99777) has done: 'I fix the pipeline so it no longer crashes when the referenced “../input/*” model-prediction files are missing in your environment. The main change is to make prediction loading robust: if an external file isn’t found, we fall back to a safe baseline (zeros) while keeping the ensembling logic intact and ensuring all required columns exist. I also align all loaded predictions by `id` (instead of relying on row order/index_col quirks) to prevent silent misalignment bugs. Finally, I always write a valid `submission.csv` with the required columns and `.csv` suffix.'
- What this solution (achieved 1.99777) has done: 'Your current score is far worse than the target (lower is better), and the biggest reason is that almost all referenced `../input/...` prediction files are missing here, so the ensemble collapses to (mostly) zeros. To move the score toward the target while preserving your ensembling logic, I add a minimal fallback: train a fast per-`type` baseline model (ridge regression) on `train.csv` using only geometric + atom-type features from `structures.csv`, and use its predictions only for rows where the external predictions are missing (all-zero across ensemble sources). This keeps your original weighted blend intact when real OOF/submission predictions are available, but prevents the catastrophic all-zero submission when they aren’t. The approach is deterministic, runs within the time budget, and still writes a valid `submission.csv`.'
- What this solution (achieved 2.41977) has done: 'I fix the crash in the fallback feature builder by avoiding pandas `string`/`pd.NA` comparisons in `np.where`, which currently triggers “boolean value of NA is ambiguous” when any merged atom labels are missing. The minimal change is to ensure `atom_0`/`atom_1` are plain Python strings with a safe fill value before building `atom_pair`, making the comparison well-defined and deterministic. I also add a small safety check that merged structure coordinates exist (otherwise distance features become NaN), and fill any remaining NaNs in the final fallback design matrix with zeros so Ridge prediction can’t error. This should restore end-to-end execution and keep the ensemble logic identical, only improving score versus the all-zero external-preds case by enabling the intended Ridge fallback.'
- What this solution (achieved 3.16394) has done: 'Your current score (2.41977, lower-is-better) is far from the target (-2.3118), so we should improve (decrease) the error without changing your ensemble’s core logic. The biggest low-risk gain here is to make the Ridge fallback (used when external prediction files are missing) stronger while keeping the same “only replace when all external preds are zero” behavior. I minimally extend the fallback features by merging in Mulliken charges and magnetic shielding tensors for the two atoms and adding simple aggregate tensor statistics; this is still the same training approach (single Ridge fit) and same prediction semantics. I also ensure the fallback design matrix remains fully numeric and NaN-safe so it runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 2.42871) has done: 'Your score is much worse than the target (lower is better), so we should improve it (decrease error) with minimal risk while keeping your ensemble and Ridge fallback logic intact. The biggest issue is that your Ridge fallback is trained on the full training set and only used when external preds are missing; without any validation-calibration, its scale/offset can be poor per coupling `type`, which strongly hurts the competition’s per-type log-MAE metric. I keep the same features and same Ridge model, but fit a simple out-of-fold (OOF) calibration per `type` (a linear `a*pred + b`) on a molecule-group split, then apply that calibration to the fallback test predictions only. This is a minimal post-processing step aligned to the metric and should move the score substantially toward the target when external prediction files are absent/zero.'
- What this solution (achieved 2.83671) has done: 'Your current score (2.42871, lower-is-better) is far from the target (-2.3118), so we should improve (decrease) it with minimal risk while keeping the ensemble + Ridge fallback core intact. The biggest issue is that the fallback Ridge is trained globally and then calibrated with a potentially unstable per-type linear fit; because the metric is per-type log-MAE, a more robust per-type calibration (with shrinkage and clipping) typically reduces large per-type errors without changing the model/features. I keep the exact same features and Ridge training, but replace the raw `np.polyfit` calibration with a numerically-stable ridge regression calibration per type, shrink its slope toward 1.0 based on sample size, and clip extreme slopes to avoid blowing up predictions. This is a minimal post-processing adjustment applied only to the fallback path (only when external preds are missing), so it should move your score down toward the target while preserving your ensemble behavior.'
- What this solution (achieved 4.65501) has done: 'Your score is far worse than the target (lower is better), and the current pipeline likely still behaves like a weak baseline because the external prediction files are missing and the Ridge fallback is underpowered for a per-type log-MAE metric. I keep your ensemble logic identical, but strengthen only the fallback path by (1) using a per-`type` Ridge model (same Ridge approach, just fitted separately per coupling type) and (2) applying the same stable OOF calibration you already have, now computed per type on that per-type model. This is a minimal, metric-aligned change that typically reduces large per-type errors without changing features, loss, or introducing approximations. The fallback still only be used when all external preds are effectively zero, and the script still write a valid `submission.csv`.'
- What this solution (achieved 4.64762) has done: 'Your current score is much worse than the target (lower is better), and the most likely cause is that the fallback path is producing very poor predictions because the categorical one-hot columns are being built separately for train/test and then reindexed, which is correct, but the numeric feature columns still include NaNs from missing merges (especially tensor/charges) that then get zero-filled, collapsing signal. I keep your ensemble and Ridge fallback logic intact, but make the fallback features more robust and informative with minimal changes: (1) add simple per-atom element encoding as numeric atomic numbers, and (2) add per-tensor summary stats (mean/std/min/max) for each atom, which are cheap and stable. I also fix a subtle bug risk in `t0_trace/t1_trace` where `df.get()` can return `None` and break addition, by explicitly filling missing tensor columns with 0 before computing derived stats. These are all confined to the fallback path and should reduce fallback MAE substantially when external predictions are missing.'
- What this solution (achieved 4.64762) has done: 'Your current score is much worse than the target (lower is better), so we should reduce the error with the smallest safe change in the fallback path (since external prediction files are missing here). I keep your ensemble logic and per-type Ridge fallback intact, but fix the biggest metric mismatch: your calibration currently uses OOF predictions from a *different* model than you use at inference (it fits per-type Ridge, but OOF is built on the full one-hot “type” features, causing leakage/instability and very poor calibration). I build OOF predictions using the exact same per-type Ridge setup you use for test, then fit the same (a*pred+b) ridge calibration per type on those OOF preds, and apply it only to fallback rows as you already do. This should substantially decrease per-type MAE (and thus the averaged log-MAE), moving the score toward the target while preserving core semantics and producing the same submission format.'
- What this solution (achieved 8.85288) has done: 'Your current score (4.64762, lower-is-better) is far worse than the target (-2.3118), and in this environment almost all external prediction files are missing so the ensemble is effectively falling back to a weak model. To move the score down substantially without changing your overall pipeline/ensemble semantics, I keep the same Ridge-per-type fallback + calibration approach but make one minimal, metric-aligned fix: train the fallback Ridge on a log1p-transformed target per type (and invert with expm1 for predictions), which typically reduces per-type MAE spread and improves the competition’s averaged log-MAE. I also make the calibration operate in the same transformed space (so it’s consistent), and leave the “only replace when external preds are all-zero” rule unchanged. This should improve fallback accuracy materially while keeping architecture/training approach the same and still writing a valid `submission.csv`.'
- What this solution (achieved 1.99777) has done: 'Your current score (8.85288, lower-is-better) is far worse than the target (-2.3118), so we should improve (decrease) error with the smallest safe change in the fallback path (since external prediction files are missing here). The main regression is that the per-type log1p shift uses the minimum of the full training set, which can over-shift and distort the transformed target distribution, hurting Ridge fit and calibration; I replace that with a robust per-type shift based on a low quantile (and a small safety margin) while keeping the same log1p/expm1 core logic. I also clip the inverse-transformed fallback predictions to a reasonable per-type range derived from train quantiles, which reduces catastrophic outliers that heavily penalize the competition’s per-type log-MAE. These changes only affect rows where all external preds are missing (your existing all-zero mask), preserving ensemble semantics otherwise and still producing a valid `submission.csv`.'
- What this solution (achieved 2.89774) has done: 'I fix the crash in the Ridge fallback by ensuring the fallback training target has no NaNs before fitting (the current error comes from NaNs in `scalar_coupling_constant` after reading/merging). I do this minimally by filtering out rows with non-finite target values and applying the exact same mask consistently to `train`, `Xtr`, `types_tr`, and `ytr_raw` so shapes remain aligned. I also add a small safety guard for the log1p transform so any remaining bad values can’t create NaNs/inf in `ytr_trans`. These changes are correctness/stability fixes for the fallback path and should improve the score versus the current broken run (which can’t generate a valid submission).'
- What this solution (achieved 2.7291) has done: 'Your current score (2.89774, lower-is-better) is far from the target (-2.3118), so we should reduce error with the smallest change that strengthens the only part that matters in this environment: the fallback (since all external pred files are missing and thus currently produce zeros). I keep your ensemble weights and fallback Ridge-per-type + calibration + log1p/clip semantics intact, but make the fallback features more informative by also merging in molecule-level `dipole_moments.csv` and `potential_energy.csv` (cheap, strong signals) and by adding a couple of geometry-derived invariants (vector components and inverse distance) that don’t change the modeling approach. This should materially improve MAE per type (and thus the averaged log-MAE) while preserving the same training loop and submission format. I also add a small safeguard to ensure the new merged numeric columns are always present and NaN-safe so the pipeline remains stable and deterministic.'

# 9. Code solution

## === cell 0
import os, glob
import pandas as pd
import numpy as np

for p in [
    "../input/1-mpn",
    "../input/",
    "../input/nnet-b-seed-10",
    "../input/nnet-b-seed-11",
    "../input/nnet-b-seed-12",
    "../input/champ-preds",
]:
    print(f"Listing: {p}")
    if os.path.exists(p):
        print("\n".join(os.listdir(p)[:20]))
    else:
        print("Path not found (OK for this environment).")



## === cell 1
import pandas as pd
import numpy as np
import os

BASE_PATH = "/kaggle/data/champs-scalar-coupling"
test_path = os.path.join(BASE_PATH, "test.csv")
sample_path = os.path.join(BASE_PATH, "sample_submission.csv")

test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

TARGET = "scalar_coupling_constant"

assert "id" in test.columns, "test.csv must have an 'id' column"
assert (
    "id" in sample_sub.columns and TARGET in sample_sub.columns
), "sample_submission must have id and scalar_coupling_constant"

test.shape, sample_sub.shape




## === cell 2
def _safe_read_pred_series(path, ids, target_col=TARGET):
    """
    Robustly load predictions from a CSV and align to provided ids.
    Supports two common formats:
      - columns: ['id', 'scalar_coupling_constant']
      - index as id (first column), with a prediction column
    If file is missing/unreadable, returns zeros (score will be poor but submission valid).
    """
    ids = pd.Index(ids)
    if (path is None) or (not os.path.exists(path)):
        return pd.Series(
            np.zeros(len(ids), dtype=np.float64), index=ids, name=target_col
        )

    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.Series(
            np.zeros(len(ids), dtype=np.float64), index=ids, name=target_col
        )

    if "id" in df.columns:
        if target_col not in df.columns:
            if df.shape[1] == 2:
                pred_col = [c for c in df.columns if c != "id"][0]
                s = df.set_index("id")[pred_col]
            else:
                return pd.Series(
                    np.zeros(len(ids), dtype=np.float64), index=ids, name=target_col
                )
        else:
            s = df.set_index("id")[target_col]
        return s.reindex(ids).fillna(0.0).astype(np.float64)

    if df.shape[1] >= 2:
        first = df.columns[0]
        pred_col = target_col if target_col in df.columns else df.columns[-1]
        try:
            tmp = df.set_index(first)[pred_col]
            tmp.index = pd.to_numeric(tmp.index, errors="ignore")
            return tmp.reindex(ids).fillna(0.0).astype(np.float64)
        except Exception:
            return pd.Series(
                np.zeros(len(ids), dtype=np.float64), index=ids, name=target_col
            )

    return pd.Series(np.zeros(len(ids), dtype=np.float64), index=ids, name=target_col)


def get_median_from_files(files, ids):
    """
    Load multiple prediction files robustly and take median per id.
    Missing files contribute a zero series; this keeps the pipeline runnable.
    """
    series_list = []
    for f in files:
        series_list.append(_safe_read_pred_series(f, ids))
    concat_sub = pd.concat(series_list, axis=1)
    champ_median = concat_sub.median(axis=1).values
    return champ_median


ids = test["id"].values

test["n1"] = get_median_from_files(
    [
        "../input/champ-preds/gnn_median_2302.csv",
        "../input/champ-preds/gnn_median_2301.csv",
        "../input/champ-preds/gnn_median_adjusted_1JHC_2296.csv",
    ],
    ids=ids,
)

gnn_2312 = _safe_read_pred_series(
    "../input/champ-preds/gnn_median_65_68_2312.csv", ids
).values
test["lastgnn"] = _safe_read_pred_series(
    "../input/champ-preds/gnn_median_69_73.csv", ids
).values
test["n1"] = gnn_2312 * 0.6 + test["n1"] * 0.4

test["n2"] = _safe_read_pred_series(
    "../input/champ-preds/gnn0_median_2068.csv", ids
).values

test["lgb_a"] = get_median_from_files(
    [
        "../input/champ-preds/submission_type_2100.csv",
        "../input/champ-preds/submission_type_2085.csv",
    ],
    ids=ids,
)

test["lgb_m"] = get_median_from_files(
    [
        "../input/champ-preds/lgb_type_full_f286_10.csv",
        "../input/champ-preds/lgb_type_full_f262_10.csv",
    ],
    ids=ids,
)

test["nnet"] = get_median_from_files(
    [
        "../input/nnpvals-seed-20/nnet_sub_s11.csv",
        "../input/champ-preds/nnet_sub.csv",
        "../input/nn-seed-10/nnet_sub.csv",
        "../input/nn-seed-11/nnet_sub.csv",
        "../input/nnet-c-seed-10/lgb_type_cv-1.7126_mae0.23572_fd5_10.csv",
        "../input/nnet-c-seed-11/lgb_type_cv-1.70994_mae0.23497_fd5_11.csv",
        "../input/nnet-c-seed-12/lgb_type_cv-1.71029_mae0.23523_fd5_12.csv",
        "../input/nnet-b-seed-10/lgb_type_cv-1.72296_mae0.24019_bags-1_f120_fd5_10.csv",
        "../input/nnet-b-seed-11/lgb_type_cv-1.70647_mae0.2376_bags-1_f120_fd5_11.csv",
        "../input/nnet-b-seed-12/lgb_type_cv-1.69833_mae0.24106_bags-1_f120_fd5_12.csv",
        "../input/nnet-try-seed-11/lgb_type_cv-1.64944_mae0.23965_bags-1_f120_fd5_11.csv",
    ],
    ids=ids,
)

test["nnet_cont"] = get_median_from_files(
    [
        "../input/nncont-seed-23-p/nnetCont_sub_only_predict.csv",
        "../input/nncont-seed-22/nnetCont_sub-1.8213.csv",
    ],
    ids=ids,
)

nn_contof = _safe_read_pred_series(
    "../input/nncont-seed-26/nnetCont_sub_-2.6677.csv", ids
).values
test["nnet_cont"] = nn_contof * 0.6 + test["nnet_cont"] * 0.4

test["final_mpnn"] = _safe_read_pred_series(
    "../input/champ-preds/final_mpnn.csv", ids
).values
test["mpnn"] = _safe_read_pred_series(
    "../input/champ-preds/mpnn_5_fold_pseudo_1449.csv", ids
).values

test["lb"] = _safe_read_pred_series(
    "../input/chemistry-of-best-models-1-895/stack_median.csv", ids
).values

test.head(3)



## === cell 3
import matplotlib.pyplot as plt
import seaborn as sns

test["nnet_ens"] = test["nnet_cont"] * 0.6 + test["nnet"] * 0.4
test["lgb_ens"] = test["lgb_a"] * 0.8 + test["lgb_m"] * 0.2

cols = [
    c
    for c in test.columns
    if c not in ["id", "molecule_name", "atom_index_0", "atom_index_1", "type"]
]
corr = test[cols].corr(numeric_only=True)

plt.figure(figsize=(12, 8))
sns.heatmap(corr, cmap="viridis", square=True)
plt.tight_layout()
corr



## === cell 4
test["final_preds"] = (
    test["n1"] * 0.65
    + test["n2"] * 0.07
    + test["lgb_ens"] * 0.15
    + test["nnet_ens"] * 0.06
    + test["lb"] * 0.07
)

test[["id", "final_preds"]].head(5)



## === cell 5
from sklearn.linear_model import Ridge

train_path = os.path.join(BASE_PATH, "train.csv")
structures_path = os.path.join(BASE_PATH, "structures.csv")
mulliken_path = os.path.join(BASE_PATH, "mulliken_charges.csv")
mst_path = os.path.join(BASE_PATH, "magnetic_shielding_tensors.csv")
dipole_path = os.path.join(BASE_PATH, "dipole_moments.csv")
energy_path = os.path.join(BASE_PATH, "potential_energy.csv")

train = pd.read_csv(
    train_path,
    usecols=["molecule_name", "atom_index_0", "atom_index_1", "type", TARGET],
)
structures = pd.read_csv(
    structures_path, usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"]
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

mulliken = pd.read_csv(
    mulliken_path, usecols=["molecule_name", "atom_index", "mulliken_charge"]
)
m0 = mulliken.rename(
    columns={"atom_index": "atom_index_0", "mulliken_charge": "mulliken_0"}
)
m1 = mulliken.rename(
    columns={"atom_index": "atom_index_1", "mulliken_charge": "mulliken_1"}
)

mst_cols = [
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
]
mst = pd.read_csv(mst_path, usecols=mst_cols)

t0_rename = {c: f"t0_{c}" for c in mst_cols if c not in ["molecule_name", "atom_index"]}
t0_rename["atom_index"] = "atom_index_0"
t1_rename = {c: f"t1_{c}" for c in mst_cols if c not in ["molecule_name", "atom_index"]}
t1_rename["atom_index"] = "atom_index_1"
t0 = mst.rename(columns=t0_rename)
t1 = mst.rename(columns=t1_rename)

tensor_elems = ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]
t0_elem_cols = [f"t0_{c}" for c in tensor_elems]
t1_elem_cols = [f"t1_{c}" for c in tensor_elems]

dipole = pd.read_csv(dipole_path, usecols=["molecule_name", "X", "Y", "Z"]).rename(
    columns={"X": "dipole_x", "Y": "dipole_y", "Z": "dipole_z"}
)
energy = pd.read_csv(energy_path, usecols=["molecule_name", "potential_energy"])

_ATOMIC_NUM = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}


def _add_geo_atom_features(df):
    df = df.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
    df = df.merge(s1, on=["molecule_name", "atom_index_1"], how="left")

    df = df.merge(m0, on=["molecule_name", "atom_index_0"], how="left")
    df = df.merge(m1, on=["molecule_name", "atom_index_1"], how="left")
    df = df.merge(t0, on=["molecule_name", "atom_index_0"], how="left")
    df = df.merge(t1, on=["molecule_name", "atom_index_1"], how="left")

    df = df.merge(dipole, on="molecule_name", how="left")
    df = df.merge(energy, on="molecule_name", how="left")

    for c in t0_elem_cols + t1_elem_cols:
        if c not in df.columns:
            df[c] = 0.0
    df[t0_elem_cols] = (
        df[t0_elem_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    )
    df[t1_elem_cols] = (
        df[t1_elem_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    )

    for c in ["x0", "y0", "z0", "x1", "y1", "z1"]:
        if c not in df.columns:
            df[c] = np.nan

    dx = (df["x0"] - df["x1"]).astype(np.float32)
    dy = (df["y0"] - df["y1"]).astype(np.float32)
    dz = (df["z0"] - df["z1"]).astype(np.float32)

    df["dx"] = dx
    df["dy"] = dy
    df["dz"] = dz

    df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)
    df["dist2"] = (df["dist"] * df["dist"]).astype(np.float32)
    df["inv_dist"] = (1.0 / (df["dist"].astype(np.float64) + 1e-6)).astype(np.float32)

    a0 = df["atom_0"].astype("object").fillna("X").astype(str)
    a1 = df["atom_1"].astype("object").fillna("X").astype(str)
    df["atom_pair"] = np.where(a0 <= a1, a0 + "_" + a1, a1 + "_" + a0)

    df["z0_num"] = a0.map(_ATOMIC_NUM).fillna(0).astype(np.float32)
    df["z1_num"] = a1.map(_ATOMIC_NUM).fillna(0).astype(np.float32)

    if "mulliken_0" not in df.columns:
        df["mulliken_0"] = np.nan
    if "mulliken_1" not in df.columns:
        df["mulliken_1"] = np.nan
    df["mulliken_0"] = pd.to_numeric(df["mulliken_0"], errors="coerce").fillna(0.0)
    df["mulliken_1"] = pd.to_numeric(df["mulliken_1"], errors="coerce").fillna(0.0)
    df["mulliken_sum"] = (df["mulliken_0"] + df["mulliken_1"]).astype(np.float32)
    df["mulliken_diff"] = (df["mulliken_0"] - df["mulliken_1"]).astype(np.float32)

    for c in ["t0_XX", "t0_YY", "t0_ZZ", "t1_XX", "t1_YY", "t1_ZZ"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["t0_trace"] = (df["t0_XX"] + df["t0_YY"] + df["t0_ZZ"]).astype(np.float32)
    df["t1_trace"] = (df["t1_XX"] + df["t1_YY"] + df["t1_ZZ"]).astype(np.float32)

    df["t0_mean"] = df[t0_elem_cols].mean(axis=1).astype(np.float32)
    df["t0_std"] = df[t0_elem_cols].std(axis=1).astype(np.float32)
    df["t0_min"] = df[t0_elem_cols].min(axis=1).astype(np.float32)
    df["t0_max"] = df[t0_elem_cols].max(axis=1).astype(np.float32)

    df["t1_mean"] = df[t1_elem_cols].mean(axis=1).astype(np.float32)
    df["t1_std"] = df[t1_elem_cols].std(axis=1).astype(np.float32)
    df["t1_min"] = df[t1_elem_cols].min(axis=1).astype(np.float32)
    df["t1_max"] = df[t1_elem_cols].max(axis=1).astype(np.float32)

    df["t0_abs_mean"] = df[t0_elem_cols].abs().mean(axis=1).astype(np.float32)
    df["t1_abs_mean"] = df[t1_elem_cols].abs().mean(axis=1).astype(np.float32)

    for c in ["dipole_x", "dipole_y", "dipole_z", "potential_energy"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)
    df["dipole_mag"] = np.sqrt(
        (df["dipole_x"].astype(np.float64) ** 2)
        + (df["dipole_y"].astype(np.float64) ** 2)
        + (df["dipole_z"].astype(np.float64) ** 2)
    ).astype(np.float32)

    keep = [
        "type",
        "atom_pair",
        "z0_num",
        "z1_num",
        "dx",
        "dy",
        "dz",
        "dist",
        "dist2",
        "inv_dist",
        "mulliken_sum",
        "mulliken_diff",
        "t0_trace",
        "t1_trace",
        "t0_abs_mean",
        "t1_abs_mean",
        "t0_mean",
        "t0_std",
        "t0_min",
        "t0_max",
        "t1_mean",
        "t1_std",
        "t1_min",
        "t1_max",
        "dipole_x",
        "dipole_y",
        "dipole_z",
        "dipole_mag",
        "potential_energy",
    ]
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    return df[keep]


Xtr_cat = _add_geo_atom_features(train)
ytr_raw = train[TARGET].astype(np.float64).values

Xtr_ohe = pd.get_dummies(Xtr_cat[["type", "atom_pair"]], drop_first=False, sparse=False)
Xtr_num = (
    Xtr_cat.drop(columns=["type", "atom_pair"])
    .astype(np.float32)
    .reset_index(drop=True)
)
Xtr = np.hstack([Xtr_num.values, Xtr_ohe.values]).astype(np.float32)
Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)

Xte_cat = _add_geo_atom_features(test)
Xte_ohe = pd.get_dummies(Xte_cat[["type", "atom_pair"]], drop_first=False, sparse=False)
Xte_ohe = Xte_ohe.reindex(columns=Xtr_ohe.columns, fill_value=0.0)
Xte_num = (
    Xte_cat.drop(columns=["type", "atom_pair"])
    .astype(np.float32)
    .reset_index(drop=True)
)
Xte = np.hstack([Xte_num.values, Xte_ohe.values]).astype(np.float32)
Xte = np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0)

types_tr = train["type"].astype(str).values
types_te = test["type"].astype(str).values
uniq_types = np.unique(types_tr)

valid_y_mask = np.isfinite(ytr_raw)
if valid_y_mask.sum() < len(valid_y_mask):
    print("Dropping invalid y rows:", int((~valid_y_mask).sum()))
train = train.loc[valid_y_mask].reset_index(drop=True)
Xtr = Xtr[valid_y_mask]
types_tr = types_tr[valid_y_mask]
ytr_raw = ytr_raw[valid_y_mask]
uniq_types = np.unique(types_tr)

type_shift = {}
type_clip = {}  # per-type clipping bounds in original space to reduce outlier blow-ups
ytr_trans = np.zeros_like(ytr_raw, dtype=np.float64)
for t in uniq_types:
    m = types_tr == t
    y_t = ytr_raw[m]

    q_low = float(np.quantile(y_t, 0.001))
    s = max(0.0, -q_low + 1e-3)
    type_shift[t] = s
    yt_shifted = y_t + s
    yt_shifted = np.maximum(yt_shifted, -1.0 + 1e-6)
    ytr_trans[m] = np.log1p(yt_shifted)

    q1 = float(np.quantile(y_t, 0.001))
    q2 = float(np.quantile(y_t, 0.999))
    pad = 0.05 * (q2 - q1) if np.isfinite(q2 - q1) else 0.0
    lo = q1 - pad
    hi = q2 + pad
    if not np.isfinite(lo):
        lo = float(np.min(y_t))
    if not np.isfinite(hi):
        hi = float(np.max(y_t))
    if lo > hi:
        lo, hi = hi, lo
    type_clip[t] = (lo, hi)

fallback_preds_trans = np.zeros(len(test), dtype=np.float64)

for t in uniq_types:
    tr_m = types_tr == t
    te_m = types_te == t
    if te_m.sum() == 0:
        continue
    rm = Ridge(alpha=1.0, random_state=0)
    rm.fit(Xtr[tr_m], ytr_trans[tr_m])
    fallback_preds_trans[te_m] = rm.predict(Xte[te_m]).astype(np.float64)


def _fit_type_calibration_oof_per_type(train_df, Xtr_full, y_trans_full, seed=0):
    """
    Compute OOF + calibration in the SAME transformed space used by the fallback model.
    """
    mol = train_df["molecule_name"].astype(str).values
    h = pd.util.hash_pandas_object(pd.Series(mol), index=False).values
    fold = (h ^ (seed * 9973)) % 5  # deterministic 5-fold by molecule name

    types = train_df["type"].astype(str).values
    y = y_trans_full.astype(np.float64)

    oof = np.zeros(len(train_df), dtype=np.float64)

    for t in np.unique(types):
        m_t = types == t
        if m_t.sum() == 0:
            continue
        for k in range(5):
            tr_idx = m_t & (fold != k)
            va_idx = m_t & (fold == k)
            if va_idx.sum() == 0 or tr_idx.sum() == 0:
                continue
            rm = Ridge(alpha=1.0, random_state=0)
            rm.fit(Xtr_full[tr_idx], y_trans_full[tr_idx])
            oof[va_idx] = rm.predict(Xtr_full[va_idx]).astype(np.float64)

    calib = {}
    lam = 1e-2
    shrink_N = 5000.0
    a_min, a_max = 0.5, 1.5
    b_clip = 2.0  # clip in log-space

    for t in np.unique(types):
        m = types == t
        n = int(m.sum())
        xp = oof[m]
        yt = y[m]
        if n < 100 or np.std(xp) < 1e-12:
            calib[t] = (1.0, 0.0)
            continue

        X = np.vstack([xp, np.ones_like(xp)]).T
        XtX = X.T @ X
        Xty = X.T @ yt
        A = XtX + lam * np.eye(2, dtype=np.float64)
        try:
            ab = np.linalg.solve(A, Xty)
            a_hat = float(ab[0])
            b_hat = float(ab[1])
        except Exception:
            a_hat, b_hat = 1.0, 0.0

        if (not np.isfinite(a_hat)) or (not np.isfinite(b_hat)):
            a_hat, b_hat = 1.0, 0.0

        w = n / (n + shrink_N)
        a = 1.0 + w * (a_hat - 1.0)
        b = 0.0 + w * (b_hat - 0.0)

        a = float(np.clip(a, a_min, a_max))
        b = float(np.clip(b, -b_clip, b_clip))
        calib[t] = (a, b)

    return calib


calib = _fit_type_calibration_oof_per_type(train, Xtr, ytr_trans, seed=0)

a = np.array([calib.get(t, (1.0, 0.0))[0] for t in types_te], dtype=np.float64)
b = np.array([calib.get(t, (1.0, 0.0))[1] for t in types_te], dtype=np.float64)
fallback_preds_trans_cal = fallback_preds_trans * a + b

shift_te = np.array([type_shift.get(t, 0.0) for t in types_te], dtype=np.float64)
fallback_preds_cal = np.expm1(fallback_preds_trans_cal) - shift_te

clip_lo = np.array(
    [type_clip.get(t, (-np.inf, np.inf))[0] for t in types_te], dtype=np.float64
)
clip_hi = np.array(
    [type_clip.get(t, (-np.inf, np.inf))[1] for t in types_te], dtype=np.float64
)
fallback_preds_cal = np.clip(fallback_preds_cal, clip_lo, clip_hi)

pred_source_cols = ["n1", "n2", "lgb_ens", "nnet_ens", "lb"]
all_zero_mask = (test[pred_source_cols].abs().sum(axis=1) < 1e-12).values

print(
    "Rows with missing external preds (using fallback):",
    int(all_zero_mask.sum()),
    "/",
    len(test),
)

test.loc[all_zero_mask, "final_preds"] = fallback_preds_cal[all_zero_mask]

test[["id", "final_preds"]].head(5)



## === cell 6
submission = pd.DataFrame({"id": test["id"].values, TARGET: test["final_preds"].values})

submission = submission.sort_values("id").reset_index(drop=True)

submission.to_csv("submission.csv", index=False)
submission.to_csv("last_gnn.csv", index=False)

submission.head(10), submission.shape



## === cell 7
assert submission.columns.tolist() == ["id", TARGET]
assert submission["id"].isna().sum() == 0
assert submission[TARGET].isna().sum() == 0
print("Wrote submission.csv with", len(submission), "rows")
print(submission.describe(include="all"))
