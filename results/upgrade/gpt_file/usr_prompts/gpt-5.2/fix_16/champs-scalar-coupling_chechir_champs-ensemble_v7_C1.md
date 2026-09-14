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

1.2481

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.2116) has done: 'The crash is from reading external “../input/champ-preds/…” and “../input/1-mpnn/…” files that are not present in your environment, so no submission ever gets created. I keep the same overall idea (produce `id, scalar_coupling_constant` and write a `.csv`) but replace the missing-file ensemble with a self-contained baseline that trains per coupling `type` and predicts for test. This fixes the runtime errors, avoids leakage by splitting by `molecule_name` for a quick sanity check, and produces `sub_ensemble.csv` in the required format. The approach is intentionally simple/fast to ensure it finishes under the time limit and yields a valid submission.'
- What this solution (achieved 1.2116) has done: 'The crash comes from the strict NA check after merging `structures.csv`: in this dataset a tiny number of rows can fail to match (data quirks or dtype mismatches), and raising stops the whole pipeline. I make the merge keys type-consistent, then replace the hard failure with safe handling: keep training rows with valid structure matches and impute missing structure-derived features in test (so we still output predictions for every `id`). This preserves your core modeling approach (per-`type` Ridge on simple distance/atom features) while ensuring the notebook runs end-to-end and always writes a valid `.csv` submission. The score should also improve versus using broken/partial features because training won’t be poisoned by unmatched-merge NaNs.'
- What this solution (achieved 3.55362) has done: 'Your current model is extremely underpowered for CHAMPS, so to move the score down toward the target we should add a few high-signal structure-derived features while keeping the same per-`type` Ridge + OHE pipeline. I minimally extend `add_structure_features()` to include molecule-level center-of-mass style coordinates (mean x/y/z per molecule) and the relative positions of each atom to that center, plus a couple of simple interaction terms; this preserves the core logic and only enriches the existing feature set. I also include the coupling `type` as a feature while switching to a single global Ridge model (still Ridge, same preprocessing, same semantics) so the model can share statistics across types without changing the training approach. These changes are typically enough to significantly reduce MAE/logMAE while remaining fast and self-contained.'
- What this solution (achieved 3.54936) has done: 'Your current Ridge model is missing two very strong, competition-standard signals: the per-atom electric environment (Mulliken charges) and local magnetic environment (shielding tensors). I minimally extend your existing `add_structure_features()` to left-merge these two provided files for atom 0 and atom 1, keeping the same global Ridge + OHE pipeline and the same train/test flow. This should reduce MAE across types (and therefore logMAE) substantially while remaining fast enough and producing the same `id,scalar_coupling_constant` submission. All I/O paths stay within the provided dataset directory and we still impute any missing merged values.'
- What this solution (achieved 3.32712) has done: 'Your current score is far from the target (lower is better), so we need a meaningful-but-still-minimal improvement while keeping the same overall pipeline: merge tabular atom-level features, build numeric+categorical features, and fit a Ridge regression with the same preprocessing. The biggest issue is that a single global linear model struggles because each coupling `type` has a very different scale/distribution, and the evaluation averages log(MAE) per type. I keep Ridge + the same feature extraction, but train one Ridge model per `type` (same preprocessing) and predict test rows by their `type`; this directly aligns with the metric and typically reduces per-type errors substantially. I also add a tiny, safe “type-wise target centering” (fit on centered target per type, then add back the mean) which improves calibration without changing the loss/architecture and stays fast.'
- What this solution (achieved 3.19135) has done: 'We need to move your score down (lower-is-better) toward the target, and your current model is still missing a couple of very strong tabular signals already provided by the competition that can be merged with minimal disruption. I keep the same per-`type` Ridge + impute + OHE pipeline and the same target-centering, but add `dipole_moments.csv` (molecule-level) and `potential_energy.csv` (molecule-level) as additional numeric features, since they’re fast to merge and typically reduce per-type MAE. I also include `type` as an additional categorical feature inside each per-type model (it becomes constant within each model, so it won’t change those models’ learned weights; but it keeps the feature schema robust and won’t hurt), and I increase Ridge regularization slightly to reduce overfitting across noisy high-dimensional OHE+tensor features (a small, safe calibration move rather than an architectural change). The script still runs end-to-end and writes a valid `sub_ensemble.csv`.'
- What this solution (achieved 3.65527) has done: 'Your current score (3.19135, lower-is-better) is still far from the target (-2.10046), so we should make a meaningful improvement but keep the same core approach (feature merges → numeric/categorical preprocessing → per-type Ridge). The biggest low-risk gain available without changing the modeling paradigm is to add the provided `scalar_coupling_contributions.csv` terms (fc/sd/pso/dso): they are extremely predictive and can be merged directly on the same keys, improving per-type MAE (and thus logMAE) substantially. I also add the (common) symmetric “swapped-atom” features (e.g., dist is symmetric; per-atom features can be averaged/differenced) which helps linear models without changing the model class or training loop. Everything else (per-type Ridge, target-centering, imputation/OHE, submission writing) stays the same.'
- What this solution (achieved 1.23566) has done: 'Your score is far worse than the target (lower-is-better), so we should make a small change that legitimately improves predictions without changing the modeling paradigm (still per-type Ridge with the same preprocessing and target-centering). The largest single issue is that `scalar_coupling_contributions.csv` features are only merged for train and set to NaN for test, so the model learns to rely on highly predictive `fc/sd/pso/dso` but cannot use them at inference—this creates a big train/test feature mismatch that hurts score. The minimal, metric-aligned fix is to (1) train the model to predict the residual `scalar_coupling_constant - (fc+sd+pso+dso)` using the same Ridge pipeline, and (2) at test time, approximate the contributions by using per-`type` means from train and add them back, so train/test semantics match and the model focuses on what’s predictable from structure/tabular features. This keeps core logic (feature merges → preprocess → per-type Ridge → submission) intact, runs fast, and should move the score down toward the target.'
- What this solution (achieved 1.23566) has done: 'Your current score is still far from the (lower-is-better) target, so the safest way to improve without changing the core pipeline is to remove the train/test mismatch introduced by using exact `fc/sd/pso/dso` only in train but type-means in test. I keep the exact same feature engineering and per-`type` Ridge approach, but stop using `fc/sd/pso/dso` as model features (they remain used only to form the residual target and to add back a type-mean contribution at inference). This makes the learning problem consistent between train and test and typically improves logMAE substantially for this competition. I also ensure the test-side contribution fallback is robust by computing type means from the training merge (so it matches the training population) and filling any missing contribution means safely.'
- What this solution (achieved 3.04778) has done: 'Your current score is much worse than the target (lower-is-better), and the biggest minimal fix is to remove the remaining train/test mismatch around `contrib_sum`: right now test uses per-type mean contributions, but the residual target in train uses the exact per-row contributions, so the model is asked to learn a residual distribution at train time that it never see at test time. I keep the exact same per-type Ridge + preprocessing pipeline, but redefine the residual target to be `y - type_mean(contrib_sum)` (and then add back that same type-mean at inference), making the learning target consistent between train and test while still leveraging contributions in a legitimate, non-leaky way. This should materially reduce MAE across types and move the score down toward the target without changing the model class, training loop, or feature extraction. I also add a safe fallback for missing type means and ensure `contrib_sum` is always finite after merges/imputation.'
- What this solution (achieved 3.04778) has done: 'Your current score (3.04778, lower-is-better) is still far above the target (-2.10046), so we should make a meaningful but still minimal improvement that keeps the same per-type Ridge + preprocessing pipeline. The biggest low-risk issue is that your “type-mean contribution” approximation is computed only by `type`, but the provided contributions vary strongly by both `type` and the actual atom pair identities; we can tighten this approximation without changing the model by computing a *much more specific* fallback hierarchy for test contributions using only training data statistics (no leakage): first by `(type, atom_0, atom_1)`, then `(type, atom_1, atom_0)`, then `type`, then global. This reduces residual-target noise consistently across types (aligned with the metric) while preserving the exact same feature set, model class, and training loops. I also keep submission writing identical and ensure all required columns are present/finite.'
- What this solution (achieved 1.19671) has done: 'We need to move your score down (lower-is-better) toward the target, and the current 3.04778 indicates the model is likely being hurt by (1) noisy/incorrect “contrib_sum_type_mean” approximations and (2) lack of basic scale normalization for Ridge on very differently-scaled numeric features (coords, tensors, energy). I keep the exact same per-type Ridge + preprocessing + residualization approach, but improve the contribution fallback hierarchy using *atom indices* (much more specific than element symbols) and a safe multi-level backoff computed from train only. I also add a `StandardScaler` for numeric features inside the existing numeric pipeline (same model/loss, just better-conditioned linear regression), which typically improves MAE without changing core semantics. Everything else (splitting by molecule, per-type training loop, submission writing) remains the same and it still writes `sub_ensemble.csv`.'
- What this solution (achieved 1.25332) has done: 'Your current score (1.19671, lower-is-better) is still far above the target (-2.10046), so we should make a small, safe improvement without changing your core approach (feature merges → per-type Ridge → residualization → submission). The most likely issue holding you back is that the “contrib_sum_type_mean” backoff currently uses raw `atom_index_0/1`, which are molecule-local indices and therefore do not generalize across molecules; this makes most lookups miss and injects noise into the residual target. I change the contribution fallback hierarchy to use chemically meaningful keys that generalize: `(type, atom_0, atom_1)` (and swapped), and only then back off to type/global means. Everything else (features, scaler, per-type Ridge, training loops, and submission schema) stays the same.'
- What this solution (achieved 1.2481) has done: 'Your current score (1.25332, lower-is-better) is still far above the target (-2.10046), so we should make a small, safe improvement that keeps the exact same per-type Ridge + residualization pipeline. The biggest low-risk gain is to improve the “contrib_sum_type_mean” approximation: right now it keys only on `(type, atom_0, atom_1)` which is still too coarse; we can add a more specific backoff level using `(type, atom_0, atom_1, atom_index_0, atom_index_1)` computed from train only, then fall back to your existing element-pair means and then type/global means. This preserves your core logic (no architecture/training-loop changes) but reduces residual noise and should move the score downward. I also make one tiny robustness fix: ensure the contrib lookup doesn’t break when structure merges are missing by filling missing `atom_0/atom_1` with a sentinel before building MultiIndex.'
- What this solution (achieved 1.2481) has done: 'Your current score (1.2481, lower-is-better) is still far from the target (-2.10046), so we should make a small, safe improvement that keeps the exact same pipeline (feature merges → contrib residualization → per-type Ridge with the same preprocess → submission). The main remaining weakness is that the residual target still contains a strong, type-dependent offset/scale, and Ridge can struggle when different `type`s have very different target magnitudes; we can reduce this by adding a minimal, type-wise scale normalization of the residual during training and inverting it at prediction time (no change to model class or loss). This directly aligns with the metric (log(MAE) per type) because it tends to balance errors across types. All I/O paths and feature sets stay unchanged, and the script still writes `sub_ensemble.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

DATA_DIR = "/kaggle/data/champs-scalar-coupling"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
structures_path = os.path.join(DATA_DIR, "structures.csv")
mulliken_path = os.path.join(DATA_DIR, "mulliken_charges.csv")
shielding_path = os.path.join(DATA_DIR, "magnetic_shielding_tensors.csv")
dipole_path = os.path.join(DATA_DIR, "dipole_moments.csv")
potential_path = os.path.join(DATA_DIR, "potential_energy.csv")
scc_contrib_path = os.path.join(DATA_DIR, "scalar_coupling_contributions.csv")

assert os.path.exists(train_path), f"Missing: {train_path}"
assert os.path.exists(test_path), f"Missing: {test_path}"
assert os.path.exists(structures_path), f"Missing: {structures_path}"
assert os.path.exists(mulliken_path), f"Missing: {mulliken_path}"
assert os.path.exists(shielding_path), f"Missing: {shielding_path}"
assert os.path.exists(dipole_path), f"Missing: {dipole_path}"
assert os.path.exists(potential_path), f"Missing: {potential_path}"
assert os.path.exists(scc_contrib_path), f"Missing: {scc_contrib_path}"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
structures = pd.read_csv(structures_path)

mulliken = pd.read_csv(mulliken_path)
shielding = pd.read_csv(shielding_path)

dipole = pd.read_csv(dipole_path)
potential = pd.read_csv(potential_path)

scc_contrib = pd.read_csv(scc_contrib_path)

for df in (
    train,
    test,
    structures,
    mulliken,
    shielding,
    dipole,
    potential,
    scc_contrib,
):
    df["molecule_name"] = df["molecule_name"].astype(str)

train["atom_index_0"] = train["atom_index_0"].astype(np.int32)
train["atom_index_1"] = train["atom_index_1"].astype(np.int32)
test["atom_index_0"] = test["atom_index_0"].astype(np.int32)
test["atom_index_1"] = test["atom_index_1"].astype(np.int32)
structures["atom_index"] = structures["atom_index"].astype(np.int32)

mulliken["atom_index"] = mulliken["atom_index"].astype(np.int32)
shielding["atom_index"] = shielding["atom_index"].astype(np.int32)

scc_contrib["atom_index_0"] = scc_contrib["atom_index_0"].astype(np.int32)
scc_contrib["atom_index_1"] = scc_contrib["atom_index_1"].astype(np.int32)

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

m0 = mulliken.rename(
    columns={"atom_index": "atom_index_0", "mulliken_charge": "mulliken_0"}
)
m1 = mulliken.rename(
    columns={"atom_index": "atom_index_1", "mulliken_charge": "mulliken_1"}
)

sh_cols = ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]
sh0 = shielding.rename(
    columns={"atom_index": "atom_index_0", **{c: f"sh0_{c}" for c in sh_cols}}
)
sh1 = shielding.rename(
    columns={"atom_index": "atom_index_1", **{c: f"sh1_{c}" for c in sh_cols}}
)

dipole_use = dipole.rename(columns={"X": "dipole_X", "Y": "dipole_Y", "Z": "dipole_Z"})
potential_use = potential.copy()

contrib_use = scc_contrib[
    ["molecule_name", "atom_index_0", "atom_index_1", "type", "fc", "sd", "pso", "dso"]
].copy()
contrib_use["contrib_sum"] = (
    contrib_use["fc"].astype(np.float64)
    + contrib_use["sd"].astype(np.float64)
    + contrib_use["pso"].astype(np.float64)
    + contrib_use["dso"].astype(np.float64)
)


def add_structure_features(df, is_train: bool):
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

    df = df.merge(
        m0[["molecule_name", "atom_index_0", "mulliken_0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    df = df.merge(
        m1[["molecule_name", "atom_index_1", "mulliken_1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
    )
    df = df.merge(
        sh0[["molecule_name", "atom_index_0"] + [f"sh0_{c}" for c in sh_cols]],
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    df = df.merge(
        sh1[["molecule_name", "atom_index_1"] + [f"sh1_{c}" for c in sh_cols]],
        on=["molecule_name", "atom_index_1"],
        how="left",
    )

    df = df.merge(dipole_use, on="molecule_name", how="left")
    df = df.merge(potential_use, on="molecule_name", how="left")

    if is_train:
        df = df.merge(
            contrib_use,
            on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
            how="left",
        )
    else:
        df = df.copy()
        df["fc"] = np.nan
        df["sd"] = np.nan
        df["pso"] = np.nan
        df["dso"] = np.nan
        df["contrib_sum"] = np.nan

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

    df["mulliken_sum"] = df["mulliken_0"] + df["mulliken_1"]
    df["mulliken_diff"] = (df["mulliken_0"] - df["mulliken_1"]).abs()
    df["mulliken_mean"] = 0.5 * (df["mulliken_0"] + df["mulliken_1"])

    for c in sh_cols:
        df[f"sh_{c}_mean"] = 0.5 * (df[f"sh0_{c}"] + df[f"sh1_{c}"])
        df[f"sh_{c}_diff_abs"] = (df[f"sh0_{c}"] - df[f"sh1_{c}"]).abs()

    df["dipole_norm"] = np.sqrt(
        df["dipole_X"] * df["dipole_X"]
        + df["dipole_Y"] * df["dipole_Y"]
        + df["dipole_Z"] * df["dipole_Z"]
    )

    return df


train_feat = add_structure_features(train, is_train=True)
test_feat = add_structure_features(test, is_train=False)

train_feat["contrib_sum"] = (
    train_feat["fc"].astype(np.float64)
    + train_feat["sd"].astype(np.float64)
    + train_feat["pso"].astype(np.float64)
    + train_feat["dso"].astype(np.float64)
)

contrib_sum_type_mean = train_feat.groupby("type", sort=False)["contrib_sum"].mean()
global_contrib_mean = float(train_feat["contrib_sum"].mean())

pair_means_elem = (
    train_feat.groupby(["type", "atom_0", "atom_1"], sort=False)["contrib_sum"]
    .mean()
    .astype(np.float64)
)
pair_means_elem_swap = (
    train_feat.groupby(["type", "atom_1", "atom_0"], sort=False)["contrib_sum"]
    .mean()
    .astype(np.float64)
)
pair_means_elem_swap.index = pair_means_elem_swap.index.set_names(
    ["type", "atom_0", "atom_1"]
)

pair_means_elem_idx = (
    train_feat.groupby(
        ["type", "atom_0", "atom_1", "atom_index_0", "atom_index_1"], sort=False
    )["contrib_sum"]
    .mean()
    .astype(np.float64)
)
pair_means_elem_idx_swap = (
    train_feat.groupby(
        ["type", "atom_1", "atom_0", "atom_index_1", "atom_index_0"], sort=False
    )["contrib_sum"]
    .mean()
    .astype(np.float64)
)
pair_means_elem_idx_swap.index = pair_means_elem_idx_swap.index.set_names(
    ["type", "atom_0", "atom_1", "atom_index_0", "atom_index_1"]
)


def assign_contrib_approx(df: pd.DataFrame) -> pd.Series:
    tmp = df[["type", "atom_0", "atom_1", "atom_index_0", "atom_index_1"]].copy()
    tmp["atom_0"] = tmp["atom_0"].fillna("__MISSING__").astype(str)
    tmp["atom_1"] = tmp["atom_1"].fillna("__MISSING__").astype(str)

    idx_ei = pd.MultiIndex.from_frame(
        tmp[["type", "atom_0", "atom_1", "atom_index_0", "atom_index_1"]]
    )
    approx = pd.Series(
        pair_means_elem_idx.reindex(idx_ei).to_numpy(),
        index=df.index,
        dtype=np.float64,
    )

    miss = approx.isna()
    if miss.any():
        tmp_m = tmp.loc[
            miss, ["type", "atom_1", "atom_0", "atom_index_1", "atom_index_0"]
        ]
        tmp_m.columns = ["type", "atom_0", "atom_1", "atom_index_0", "atom_index_1"]
        idx_ei_s = pd.MultiIndex.from_frame(tmp_m)
        approx.loc[miss] = pair_means_elem_idx_swap.reindex(idx_ei_s).to_numpy()

    miss = approx.isna()
    if miss.any():
        idx_e = pd.MultiIndex.from_frame(tmp.loc[miss, ["type", "atom_0", "atom_1"]])
        approx.loc[miss] = pair_means_elem.reindex(idx_e).to_numpy()

    miss = approx.isna()
    if miss.any():
        idx_es = pd.MultiIndex.from_frame(tmp.loc[miss, ["type", "atom_1", "atom_0"]])
        idx_es = idx_es.set_names(["type", "atom_0", "atom_1"])
        approx.loc[miss] = pair_means_elem_swap.reindex(idx_es).to_numpy()

    miss = approx.isna()
    if miss.any():
        approx.loc[miss] = (
            df.loc[miss, "type"]
            .map(contrib_sum_type_mean)
            .astype(np.float64)
            .to_numpy()
        )

    return approx.fillna(global_contrib_mean).astype(np.float64)


train_feat["contrib_sum_type_mean"] = assign_contrib_approx(train_feat)
test_feat["contrib_sum_type_mean"] = assign_contrib_approx(test_feat)
test_feat["contrib_sum"] = test_feat["contrib_sum_type_mean"]

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



## === cell 1
feature_cols_num = (
    [
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
        "mulliken_0",
        "mulliken_1",
        "mulliken_sum",
        "mulliken_diff",
        "mulliken_mean",
        "dipole_X",
        "dipole_Y",
        "dipole_Z",
        "dipole_norm",
        "potential_energy",
    ]
    + [f"sh0_{c}" for c in sh_cols]
    + [f"sh1_{c}" for c in sh_cols]
    + [f"sh_{c}_mean" for c in sh_cols]
    + [f"sh_{c}_diff_abs" for c in sh_cols]
)

feature_cols_cat = ["atom_0", "atom_1", "type"]

preprocess = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ]
            ),
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

ALPHA = 5.0

type_models = {}
type_target_mean = {}
type_target_scale = {}

gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
train_idx, val_idx = next(gss.split(train_feat, groups=train_feat["molecule_name"]))
tr_all = train_feat.iloc[train_idx].reset_index(drop=True)
va_all = train_feat.iloc[val_idx].reset_index(drop=True)

va_pred = np.zeros(len(va_all), dtype=np.float64)

for t, tr_t in tr_all.groupby("type", sort=False):
    va_t = va_all.loc[va_all["type"] == t]
    if len(va_t) == 0:
        continue

    model_t = Pipeline(
        steps=[("prep", preprocess), ("reg", Ridge(alpha=ALPHA, random_state=42))]
    )

    X_tr = tr_t[feature_cols_num + feature_cols_cat]
    y_tr = tr_t["scalar_coupling_constant"].values.astype(np.float64)

    contrib_tr_approx = tr_t["contrib_sum_type_mean"].values.astype(np.float64)
    y_tr_resid = y_tr - contrib_tr_approx

    mu = float(np.mean(y_tr_resid))
    sigma = float(np.std(y_tr_resid - mu))
    if not np.isfinite(sigma) or sigma < 1e-6:
        sigma = 1.0

    y_tr_cs = (y_tr_resid - mu) / sigma

    model_t.fit(X_tr, y_tr_cs)
    type_models[t] = model_t
    type_target_mean[t] = mu
    type_target_scale[t] = sigma

    X_va = va_t[feature_cols_num + feature_cols_cat]
    contrib_va_approx = va_t["contrib_sum_type_mean"].values.astype(np.float64)
    va_pred[va_t.index.values] = (
        (model_t.predict(X_va) * sigma) + mu + contrib_va_approx
    )

va_tmp = va_all[["type"]].copy()
va_tmp["y"] = va_all["scalar_coupling_constant"].values
va_tmp["p"] = va_pred

val_mae_by_type = (
    va_tmp.groupby("type")
    .apply(lambda g: mean_absolute_error(g["y"].values, g["p"].values))
    .sort_values()
)

print("Validation MAE by type (lower is better):")
print(val_mae_by_type.head(10))
print(f"Validated types: {val_mae_by_type.shape[0]}/{train_feat['type'].nunique()}")



## === cell 2
type_models_full = {}
type_target_mean_full = {}
type_target_scale_full = {}

for t, df_t in train_feat.groupby("type", sort=False):
    model_t = Pipeline(
        steps=[("prep", preprocess), ("reg", Ridge(alpha=ALPHA, random_state=42))]
    )
    X_t = df_t[feature_cols_num + feature_cols_cat]

    y_t = df_t["scalar_coupling_constant"].values.astype(np.float64)
    contrib_t_approx = df_t["contrib_sum_type_mean"].values.astype(np.float64)
    y_t_resid = y_t - contrib_t_approx

    mu = float(np.mean(y_t_resid))
    sigma = float(np.std(y_t_resid - mu))
    if not np.isfinite(sigma) or sigma < 1e-6:
        sigma = 1.0

    y_t_cs = (y_t_resid - mu) / sigma

    model_t.fit(X_t, y_t_cs)
    type_models_full[t] = model_t
    type_target_mean_full[t] = mu
    type_target_scale_full[t] = sigma

test_pred = np.zeros(len(test_feat), dtype=np.float64)
global_mean = float(train_feat["scalar_coupling_constant"].mean())

for t, te_t in test_feat.groupby("type", sort=False):
    idx = te_t.index.values
    if t in type_models_full:
        X_te = te_t[feature_cols_num + feature_cols_cat]
        contrib_te_approx = te_t["contrib_sum_type_mean"].values.astype(np.float64)
        mu = type_target_mean_full[t]
        sigma = type_target_scale_full[t]
        test_pred[idx] = (
            (type_models_full[t].predict(X_te) * sigma) + mu + contrib_te_approx
        )
    else:
        test_pred[idx] = global_mean

submission = (
    pd.DataFrame(
        {
            "id": test_feat["id"].values.astype(np.int64),
            "scalar_coupling_constant": test_pred.astype(np.float64),
        }
    )
    .sort_values("id")
    .reset_index(drop=True)
)

out_path = "sub_ensemble.csv"
submission.to_csv(out_path, index=False)
print(f"Wrote submission: {out_path} with shape={submission.shape}")
print(submission.head(10))



## === cell 3
submission.head(20)
