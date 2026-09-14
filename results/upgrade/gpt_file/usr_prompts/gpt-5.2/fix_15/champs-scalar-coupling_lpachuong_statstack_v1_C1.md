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

-1.222736449496261

# 6. Current score

2.85297

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.99777) has done: 'Your notebook is trying to ensemble (“stack”) multiple precomputed submission files from a `../input/champstacks` dataset that does not exist in your environment, which causes the first `FileNotFoundError` and then cascades into `NameError`s. To make this run end-to-end and produce a valid `.csv` submission, I keep the stacking logic but switch the input source to the available `sample_submission.csv` as a safe fallback when no stack files are present. I also fix a pandas/numpy compatibility bug (`np.bool` removed) and ensure we always write at least one valid submission file with the exact required columns (`id,scalar_coupling_constant`). Since no current score exists and there are no base model predictions available, this prioritize correctness and producing a submission (score be poor but valid).'
- What this solution (achieved 3.00563) has done: 'Your current pipeline produces mostly-zero predictions because the `../input/champstacks` ensemble inputs are missing, which explains the poor score (1.99777) relative to the much better target (-1.2227; lower is better). To move the score substantially toward the target without changing the overall “single-pass tabular regression → write submission.csv” logic, I replace the empty-stack fallback with a lightweight per-type baseline model trained from `train.csv` only: predict each coupling `type` using the mean (and optionally a small linear correction vs. atom-pair distance computed from `structures.csv`). This keeps the core semantics (no leakage, no architecture/training loops) while giving nontrivial predictions that should drastically reduce MAE vs. zeros. I also ensure exact `id` alignment to `sample_submission.csv` and always write `submission.csv`.'
- What this solution (achieved 4.74289) has done: 'Your current fallback model is a per-type linear fit using only the atom-pair distance, which is too weak for this competition and explains the large gap to the target (lower is better). To move the score toward the target with minimal semantic changes, I keep the same “single-pass tabular regression → write submission.csv” structure but strengthen the baseline by adding a tiny set of well-known CHAMPS features (per-atom: mulliken charge + shielding tensor diagonal + element; per-molecule: potential energy + dipole vector), then fit the same kind of per-type linear regression (closed form, no training loop) with ridge stabilization. I also compute distance robustly and ensure train/test feature columns align and prediction is always finite, while preserving the existing stacking paths when `../input/champstacks` exists. This should substantially reduce MAE versus the distance-only fit and move the log-MAE score downward toward your target.'
- What this solution (achieved 3.7284) has done: 'Your current score is far worse than the target (lower is better), so we should improve the fallback model while keeping the same overall “single-pass feature join → per-type closed-form ridge regression → write submission.csv” core logic. The biggest low-risk gain is to add a few standard CHAMPS geometric features derived from the same merged structures: coordinate deltas, per-axis absolute deltas, and reciprocal powers of distance, which helps linear models capture nonlinearity without changing the training approach. We also stabilize per-type fitting by standardizing numeric features using train-set mean/std (within the same pipeline) so ridge behaves consistently across heterogeneous feature scales. Finally, we keep all existing stacking/output files and ensure `submission.csv` is always written with the correct `id` alignment.'
- What this solution (achieved 2.93249) has done: 'We keep your exact “feature join → per-type closed-form ridge regression → write submission.csv” pipeline, but fix one key issue that likely hurts the leaderboard score: the regression currently has no explicit coupling `type` signal in the features, so each per-type model still must absorb a type-dependent intercept/scale indirectly. Adding a one-hot encoding of `type` (and aligning train/test columns) is a minimal, safe change that preserves core semantics while usually improving log-MAE materially in CHAMPS. To reduce numerical instability without changing the training approach, we also slightly increase ridge strength (alpha) and ensure we don’t accidentally include non-feature columns. Output file names/format stay identical, and `submission.csv` is always produced.'
- What this solution (achieved 14.88971) has done: 'Your current score (2.93249, lower-is-better) is much worse than the target (-1.2227), so we should improve the fallback model, but keep the same core “feature join → per-type closed-form ridge regression → write submission.csv” pipeline. The highest-impact minimal fix is to add a few additional lightweight, competition-standard engineered features that are still purely derived from already-merged columns: per-atom shielding anisotropy/trace, per-pair sums/differences, and simple interaction terms with inverse distance. We also make ridge slightly more stable by using a per-type alpha scaled by sample count (same closed-form solve, no new training loop), while keeping all stacking behavior intact when `../input/champstacks` exists. The submission writing remains unchanged, and we still align predictions strictly to `sample_submission.csv` ids.'
- What this solution (achieved 5.75105) has done: 'Your current score (14.88971, lower-is-better) is far worse than the target (-1.2227), so we should improve the fallback model while keeping the same core pipeline: feature joins → per-type closed-form ridge regression → write `submission.csv`. The most likely culprit for such an extreme degradation is numerical instability/poor regularization in the per-type solve (alpha shrinking with sample size can make the system nearly unregularized for big types) and lack of a stabilizing target transform. I (1) switch the ridge regularization to increase with sample size (minimal change, same closed-form ridge) and (2) fit on `log1p(abs(y))` with sign and then invert at prediction time (keeps per-type ridge, but better matches log-MAE evaluation behavior and stabilizes heavy-tailed targets). I also ensure prediction columns are used consistently (avoid accidentally mixing derived champ_* columns into stacking logic) and keep the same submission writing paths/names.'
- What this solution (achieved 8.74652) has done: 'We need to move your score down (lower-is-better) toward the target, and the current 5.75 suggests the last change (signed log1p target transform + inversion) likely harmed performance for this metric. I keep the same core pipeline (feature joins → per-type closed-form ridge regression → write submission) but remove the target transform and fit ridge directly on the original target, which is a minimal semantic adjustment that typically improves MAE for CHAMPS. I also make the ridge regularization stable across types (constant alpha, not scaled by sample count) to avoid under/over-regularizing big/small coupling types, while keeping the same per-type solve and feature set. Finally, I ensure the predictions are finite and aligned to `sample_submission.csv`, and still write all the same output CSVs including `submission.csv`.'
- What this solution (achieved 4.14797) has done: 'Your current score (8.74652; lower is better) is far worse than the target (-1.2227), so we should improve the fallback model while keeping the exact same core pipeline: merged tabular features → per-type closed-form ridge regression → write submission CSVs. The biggest minimal fix is to avoid fitting per-type while also including `type` one-hot features (those are constant within each per-type slice and add unnecessary collinearity); instead we keep the per-type intercept but drop `type` dummies from the regression features. To better match the competition’s per-type log-MAE aggregation without changing training approach, we also tune a single global ridge `alpha` slightly upward for stability and add a tiny per-type shrinkage of predictions toward the per-type mean (a calibration step, not a new model), which typically reduces MAE/log-MAE. All stacking logic and output files stay the same, and we still always produce `submission.csv` with correct `id` alignment.'
- What this solution (achieved 3.9318) has done: 'We need to move the score down (lower-is-better) toward the target, and the current fallback ridge is likely hurting because it fits one global coefficient vector per coupling `type` but **shares** `atom_pair` dummy columns across types, which introduces unnecessary sparsity/collinearity and can overfit/underfit differently per type. I keep the exact same pipeline (same feature joins, same engineered features, same per-type closed-form ridge, same shrink-to-mean calibration, same output semantics), but make one minimal modeling fix: fit each per-type model using only the atom_pair categories that actually occur in that type (dropping all-zero dummy columns within that type). I also add a tiny numerical stabilization by using `float64` just for the linear solve (inputs/outputs remain float32/float), which reduces `LinAlgError`/conditioning issues without changing the approach. This should improve MAE/log-MAE materially compared with the current unstable per-type solve, moving the leaderboard score closer to the target while staying within the same core logic.'
- What this solution (achieved 3.17103) has done: 'Your current score (3.9318, lower-is-better) is far from the target (-1.2227), so we need a real accuracy improvement but with minimal core-logic change. I keep the same pipeline (feature joins → per-type closed-form ridge → shrink-to-mean → write submission) and only fix two high-impact modeling issues: (1) add molecule-split-aware training/validation-style regularization by increasing ridge strength and (2) replace the brittle “nonzero-only” column filter with a proper “non-constant” filter so we don’t keep near-constant columns that destabilize solves. I also make the shrink-to-mean weight depend mildly on per-type sample size (same calibration step, not a new model) to reduce overfitting on small types while keeping large types mostly unchanged. Output paths and submission format remain identical, and `submission.csv` is always written.'
- What this solution (achieved 3.3006) has done: 'Your current score (3.171, lower-is-better) is far worse than the target (-1.223), so we should improve accuracy without changing the overall pipeline (feature joins → per-type closed-form ridge → shrink-to-mean → write submission). The biggest low-risk issue is that your per-type ridge uses a single global `alpha=1.0`, which is too weak for some types and too strong for others; we switch to a tiny fixed per-type alpha map (still the same closed-form ridge solve) and keep the same shrinkage step. We also make sure missing types in test (if any) get a safe per-type mean fallback, and we remove the “pushout/minmax” heuristics from the final `submission.csv` by writing the best-performing straightforward aggregator (`champ0` directly) to avoid degrading predictions when only one model exists. All file paths and the submission schema stay unchanged, and `submission.csv` is always produced.'
- What this solution (achieved 3.06124) has done: 'Your score is much worse than the target (lower-is-better), so we should improve the fallback model accuracy while keeping the same core pipeline: merged tabular features → per-type closed-form ridge → shrink-to-mean → write submission. The most impactful minimal fix here is that CHAMPS errors are very type-dependent, and your features are missing the single strongest low-cost signal: one-hot encoding of the coupling `type` within each per-type model is redundant, but **type-specific nonlinearities** can be captured better by adding a few more simple distance-basis terms (e.g., `inv_dist4`, `inv_dist5`, and `exp(-dist)` / `exp(-dist^2)`), without changing the modeling approach. I also make alpha slightly more conservative (a small increase) and keep the same shrinkage scheme to reduce overfitting/instability per type. Submission writing and all existing stack CSV outputs remain unchanged, and `submission.csv` is always produced.'
- What this solution (achieved 2.85297) has done: 'We need to move your score down (lower-is-better) toward the target, and the current 3.061 is still far off, so we should improve the fallback model while keeping the same core pipeline (feature joins → per-type closed-form ridge → shrink-to-mean → write submission). The biggest low-risk gain without changing the training approach is to add a small set of “type-local” categorical signals that matter a lot in CHAMPS: the individual atoms (`atom_0`, `atom_1`) and an ordered atom-pair within each coupling type, instead of sharing a single global one-hot across all types. Concretely, we keep your same ridge solve but build the dummy columns *inside each type slice* so the model sees only relevant categories (less collinearity/conditioning) and can use both `atom_pair` and `atom_0/atom_1` dummies. We also slightly tune the ridge strengths to be a bit more conservative on the largest/noisiest types to reduce MAE, while keeping your shrinkage calibration step intact. Output stays identical (`submission.csv` and the other stack_*.csv files) with the same column names and id alignment.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from subprocess import check_output
import seaborn as sns

sns.set()
import matplotlib.pyplot as plt

try:
    print(check_output(["ls", "../input/"]).decode("utf8"))
except Exception as e:
    print("Could not list ../input:", repr(e))



## === cell 1
DATA_ROOT_CANDIDATES = [
    "../input/champs-scalar-coupling",
    "../input/champs-scalar-coupling/champs-scalar-coupling",
    "../input/champs-scalar-coupling/",  # tolerate trailing slash
    "../input/champs-scalar-coupling/champs-scalar-coupling/",
    "/kaggle/input/champs-scalar-coupling",
    "/kaggle/input/champs-scalar-coupling/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    p2 = p.rstrip("/")
    if os.path.exists(p2) and os.path.exists(os.path.join(p2, "sample_submission.csv")):
        DATA_ROOT = p2
        break

if DATA_ROOT is None:
    DATA_ROOT = "/kaggle/data/champs-scalar-coupling"

print("Using DATA_ROOT:", DATA_ROOT)
print(
    "sample_submission exists:",
    os.path.exists(os.path.join(DATA_ROOT, "sample_submission.csv")),
)



## === cell 2
sub_path = "../input/champstacks"

if os.path.isdir(sub_path):
    all_files = [f for f in os.listdir(sub_path) if f.lower().endswith(".csv")]
else:
    all_files = []

all_files[:10], len(all_files)



## === cell 3
sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)

if len(all_files) > 0:
    outs = []
    for f in sorted(all_files):
        df = pd.read_csv(os.path.join(sub_path, f))
        if "id" in df.columns:
            df = df.set_index("id")
        else:
            df.index.name = "id"
        pred_col = (
            "scalar_coupling_constant"
            if "scalar_coupling_constant" in df.columns
            else df.columns[0]
        )
        outs.append(df[[pred_col]].rename(columns={pred_col: f}))

    concat_sub = pd.concat(outs, axis=1)
    cols = list(map(lambda x: "champ" + str(x), range(len(concat_sub.columns))))
    concat_sub.columns = cols
    concat_sub.reset_index(inplace=True)
else:
    train_path = os.path.join(DATA_ROOT, "train.csv")
    test_path = os.path.join(DATA_ROOT, "test.csv")
    structures_path = os.path.join(DATA_ROOT, "structures.csv")
    mulliken_path = os.path.join(DATA_ROOT, "mulliken_charges.csv")
    shielding_path = os.path.join(DATA_ROOT, "magnetic_shielding_tensors.csv")
    dipole_path = os.path.join(DATA_ROOT, "dipole_moments.csv")
    energy_path = os.path.join(DATA_ROOT, "potential_energy.csv")

    train = pd.read_csv(
        train_path,
        usecols=[
            "molecule_name",
            "atom_index_0",
            "atom_index_1",
            "type",
            "scalar_coupling_constant",
        ],
    )
    test = pd.read_csv(
        test_path,
        usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
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
        columns={"atom_index": "atom_index_0", "mulliken_charge": "q0"}
    )
    m1 = mulliken.rename(
        columns={"atom_index": "atom_index_1", "mulliken_charge": "q1"}
    )

    shielding = pd.read_csv(
        shielding_path,
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
    )
    sh0 = shielding.rename(
        columns={
            "atom_index": "atom_index_0",
            "XX": "shXX0",
            "YX": "shYX0",
            "ZX": "shZX0",
            "XY": "shXY0",
            "YY": "shYY0",
            "ZY": "shZY0",
            "XZ": "shXZ0",
            "YZ": "shYZ0",
            "ZZ": "shZZ0",
        }
    )
    sh1 = shielding.rename(
        columns={
            "atom_index": "atom_index_1",
            "XX": "shXX1",
            "YX": "shYX1",
            "ZX": "shZX1",
            "XY": "shXY1",
            "YY": "shYY1",
            "ZY": "shZY1",
            "XZ": "shXZ1",
            "YZ": "shYZ1",
            "ZZ": "shZZ1",
        }
    )

    dipole = pd.read_csv(dipole_path, usecols=["molecule_name", "X", "Y", "Z"]).rename(
        columns={"X": "dipX", "Y": "dipY", "Z": "dipZ"}
    )
    energy = pd.read_csv(energy_path, usecols=["molecule_name", "potential_energy"])

    def add_features(df):
        df = (
            df.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
            .merge(s1, on=["molecule_name", "atom_index_1"], how="left")
            .merge(m0, on=["molecule_name", "atom_index_0"], how="left")
            .merge(m1, on=["molecule_name", "atom_index_1"], how="left")
            .merge(sh0, on=["molecule_name", "atom_index_0"], how="left")
            .merge(sh1, on=["molecule_name", "atom_index_1"], how="left")
            .merge(dipole, on=["molecule_name"], how="left")
            .merge(energy, on=["molecule_name"], how="left")
        )

        dx = (df["x0"] - df["x1"]).astype(np.float32)
        dy = (df["y0"] - df["y1"]).astype(np.float32)
        dz = (df["z0"] - df["z1"]).astype(np.float32)

        df["dx"] = dx
        df["dy"] = dy
        df["dz"] = dz
        df["adx"] = np.abs(dx).astype(np.float32)
        df["ady"] = np.abs(dy).astype(np.float32)
        df["adz"] = np.abs(dz).astype(np.float32)

        dist = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)
        df["dist"] = dist
        df["dist2"] = (dist * dist).astype(np.float32)

        inv = (1.0 / (dist + 1e-3)).astype(np.float32)
        df["inv_dist"] = inv
        df["inv_dist2"] = (1.0 / (dist * dist + 1e-3)).astype(np.float32)
        df["inv_dist3"] = (1.0 / (dist * dist * dist + 1e-3)).astype(np.float32)
        df["inv_dist4"] = (inv**4).astype(np.float32)
        df["inv_dist5"] = (inv**5).astype(np.float32)
        df["exp_dist"] = np.exp(-dist).astype(np.float32)
        df["exp_dist2"] = np.exp(-df["dist2"]).astype(np.float32)

        df["atom_pair"] = (
            df["atom_0"].fillna("X") + "_" + df["atom_1"].fillna("X")
        ).astype(str)

        df["atom_0"] = df["atom_0"].fillna("X").astype(str)
        df["atom_1"] = df["atom_1"].fillna("X").astype(str)

        df["dq"] = (df["q0"] - df["q1"]).astype(np.float32)
        df["qsum"] = (df["q0"] + df["q1"]).astype(np.float32)
        df["qabsdiff"] = np.abs(df["dq"]).astype(np.float32)

        sh0_diag = df[["shXX0", "shYY0", "shZZ0"]]
        sh1_diag = df[["shXX1", "shYY1", "shZZ1"]]
        df["sh0_iso"] = sh0_diag.mean(axis=1).astype(np.float32)
        df["sh1_iso"] = sh1_diag.mean(axis=1).astype(np.float32)
        df["dsh_iso"] = (df["sh0_iso"] - df["sh1_iso"]).astype(np.float32)
        df["shsum_iso"] = (df["sh0_iso"] + df["sh1_iso"]).astype(np.float32)

        df["sh0_trace"] = (df["shXX0"] + df["shYY0"] + df["shZZ0"]).astype(np.float32)
        df["sh1_trace"] = (df["shXX1"] + df["shYY1"] + df["shZZ1"]).astype(np.float32)
        df["dsh_trace"] = (df["sh0_trace"] - df["sh1_trace"]).astype(np.float32)
        df["shsum_trace"] = (df["sh0_trace"] + df["sh1_trace"]).astype(np.float32)

        df["sh0_aniso"] = (df["shZZ0"] - 0.5 * (df["shXX0"] + df["shYY0"])).astype(
            np.float32
        )
        df["sh1_aniso"] = (df["shZZ1"] - 0.5 * (df["shXX1"] + df["shYY1"])).astype(
            np.float32
        )
        df["dsh_aniso"] = (df["sh0_aniso"] - df["sh1_aniso"]).astype(np.float32)
        df["shsum_aniso"] = (df["sh0_aniso"] + df["sh1_aniso"]).astype(np.float32)

        df["sh0_offdiag_abs"] = (
            np.abs(df["shXY0"])
            + np.abs(df["shXZ0"])
            + np.abs(df["shYZ0"])
            + np.abs(df["shYX0"])
            + np.abs(df["shZX0"])
            + np.abs(df["shZY0"])
        ).astype(np.float32)
        df["sh1_offdiag_abs"] = (
            np.abs(df["shXY1"])
            + np.abs(df["shXZ1"])
            + np.abs(df["shYZ1"])
            + np.abs(df["shYX1"])
            + np.abs(df["shZX1"])
            + np.abs(df["shZY1"])
        ).astype(np.float32)
        df["dsh_offdiag_abs"] = (df["sh0_offdiag_abs"] - df["sh1_offdiag_abs"]).astype(
            np.float32
        )
        df["shsum_offdiag_abs"] = (
            df["sh0_offdiag_abs"] + df["sh1_offdiag_abs"]
        ).astype(np.float32)

        df["qsum_inv"] = (df["qsum"] * df["inv_dist"]).astype(np.float32)
        df["dq_inv"] = (df["dq"] * df["inv_dist"]).astype(np.float32)
        df["shsum_iso_inv"] = (df["shsum_iso"] * df["inv_dist"]).astype(np.float32)
        df["dsh_iso_inv"] = (df["dsh_iso"] * df["inv_dist"]).astype(np.float32)

        df["dip_norm"] = np.sqrt(
            df["dipX"].astype(np.float32) ** 2
            + df["dipY"].astype(np.float32) ** 2
            + df["dipZ"].astype(np.float32) ** 2
        ).astype(np.float32)

        return df

    train_feat = add_features(train)
    test_feat = add_features(test)

    base_num_cols = [
        "dist",
        "dist2",
        "inv_dist",
        "inv_dist2",
        "inv_dist3",
        "inv_dist4",
        "inv_dist5",
        "exp_dist",
        "exp_dist2",
        "dx",
        "dy",
        "dz",
        "adx",
        "ady",
        "adz",
        "q0",
        "q1",
        "dq",
        "qsum",
        "qabsdiff",
        "qsum_inv",
        "dq_inv",
        "sh0_iso",
        "sh1_iso",
        "dsh_iso",
        "shsum_iso",
        "sh0_trace",
        "sh1_trace",
        "dsh_trace",
        "shsum_trace",
        "sh0_aniso",
        "sh1_aniso",
        "dsh_aniso",
        "shsum_aniso",
        "sh0_offdiag_abs",
        "sh1_offdiag_abs",
        "dsh_offdiag_abs",
        "shsum_offdiag_abs",
        "shsum_iso_inv",
        "dsh_iso_inv",
        "dipX",
        "dipY",
        "dipZ",
        "dip_norm",
        "potential_energy",
    ]

    num_medians = train_feat[base_num_cols].median(numeric_only=True)
    train_feat[base_num_cols] = (
        train_feat[base_num_cols].fillna(num_medians).astype(np.float32)
    )
    test_feat[base_num_cols] = (
        test_feat[base_num_cols].fillna(num_medians).astype(np.float32)
    )

    means = train_feat[base_num_cols].mean()
    stds = train_feat[base_num_cols].std().replace(0.0, 1.0)
    train_feat[base_num_cols] = ((train_feat[base_num_cols] - means) / stds).astype(
        np.float32
    )
    test_feat[base_num_cols] = ((test_feat[base_num_cols] - means) / stds).astype(
        np.float32
    )

    X_train_num = train_feat[base_num_cols].copy()
    X_test_num = test_feat[base_num_cols].copy()

    y_train_all = train_feat["scalar_coupling_constant"].astype(np.float32)
    y_train_np = y_train_all.to_numpy(dtype=np.float32)

    preds = np.empty(len(test_feat), dtype=np.float32)
    preds[:] = np.nan

    global_mean_y = float(np.mean(y_train_np))

    train_types = train_feat["type"].values
    test_types = test_feat["type"].values

    alpha_map = {
        "1JHC": 10.0,
        "1JHN": 10.0,
        "2JHC": 6.0,
        "2JHH": 6.0,
        "2JHN": 6.0,
        "3JHC": 6.0,
        "3JHH": 6.0,
        "3JHN": 6.0,
    }
    default_alpha = 8.0

    type_mean = train_feat.groupby("type")["scalar_coupling_constant"].mean().to_dict()

    for t in pd.unique(train_feat["type"]):
        tr_idx = np.where(train_types == t)[0]
        te_idx = np.where(test_types == t)[0]
        if te_idx.size == 0:
            continue

        yt = y_train_np[tr_idx]
        mu_t = (
            float(np.mean(yt))
            if tr_idx.size > 0
            else float(type_mean.get(t, global_mean_y))
        )

        if tr_idx.size < 50:
            preds[te_idx] = mu_t
            continue

        tr_slice = train_feat.iloc[tr_idx]
        te_slice = test_feat.iloc[te_idx]

        Xt_num = X_train_num.iloc[tr_idx]
        Xte_num = X_test_num.iloc[te_idx]

        ap_tr = pd.get_dummies(tr_slice["atom_pair"], prefix="ap", dtype=np.int8)
        ap_cols = ap_tr.columns.tolist()
        ap_te = pd.get_dummies(
            te_slice["atom_pair"], prefix="ap", dtype=np.int8
        ).reindex(columns=ap_cols, fill_value=0)

        a0_tr = pd.get_dummies(tr_slice["atom_0"], prefix="a0", dtype=np.int8)
        a0_cols = a0_tr.columns.tolist()
        a0_te = pd.get_dummies(te_slice["atom_0"], prefix="a0", dtype=np.int8).reindex(
            columns=a0_cols, fill_value=0
        )

        a1_tr = pd.get_dummies(tr_slice["atom_1"], prefix="a1", dtype=np.int8)
        a1_cols = a1_tr.columns.tolist()
        a1_te = pd.get_dummies(te_slice["atom_1"], prefix="a1", dtype=np.int8).reindex(
            columns=a1_cols, fill_value=0
        )

        Xt_df = pd.concat([Xt_num, ap_tr, a0_tr, a1_tr], axis=1)
        Xte_df = pd.concat([Xte_num, ap_te, a0_te, a1_te], axis=1)

        Xt_full = Xt_df.to_numpy(dtype=np.float32)
        Xte_full = Xte_df.to_numpy(dtype=np.float32)

        ones_train = np.ones((Xt_full.shape[0], 1), dtype=np.float32)
        ones_test = np.ones((Xte_full.shape[0], 1), dtype=np.float32)
        Xt_full = np.concatenate([ones_train, Xt_full], axis=1)
        Xte_full = np.concatenate([ones_test, Xte_full], axis=1)

        col_std = Xt_full.astype(np.float64).std(axis=0)
        col_var = col_std > 1e-12
        col_var[0] = True  # keep intercept
        Xt = Xt_full[:, col_var]
        Xte = Xte_full[:, col_var]

        p = Xt.shape[1]
        I = np.eye(p, dtype=np.float64)
        I[0, 0] = 0.0  # don't regularize intercept

        alpha = float(alpha_map.get(t, default_alpha))
        Xt64 = Xt.astype(np.float64)
        yt64 = yt.astype(np.float64)
        A = Xt64.T @ Xt64 + (alpha * I)
        bvec = Xt64.T @ yt64

        shrink_w = float(np.clip(25.0 / np.sqrt(tr_idx.size), 0.02, 0.20))

        try:
            beta = np.linalg.solve(A, bvec)
            pred_t = (Xte.astype(np.float64) @ beta).astype(np.float32)
            preds[te_idx] = ((1.0 - shrink_w) * pred_t + shrink_w * mu_t).astype(
                np.float32
            )
        except np.linalg.LinAlgError:
            preds[te_idx] = mu_t

    for t in pd.unique(test_feat["type"]):
        te_idx = np.where(test_types == t)[0]
        if te_idx.size == 0:
            continue
        mask_missing = ~np.isfinite(preds[te_idx])
        if np.any(mask_missing):
            preds[te_idx[mask_missing]] = float(type_mean.get(t, global_mean_y))

    preds = np.where(np.isfinite(preds), preds, global_mean_y).astype(np.float32)

    concat_sub = pd.DataFrame(
        {"id": test_feat["id"].values, "champ0": preds.astype(np.float32)}
    )
    concat_sub = sample_sub[["id"]].merge(concat_sub, on="id", how="left")
    concat_sub["champ0"] = (
        concat_sub["champ0"].fillna(float(global_mean_y)).astype(float)
    )

concat_sub.head()
ncol = concat_sub.shape[1]



## === cell 4
if ncol > 2:
    _corr = concat_sub.iloc[:, 1:ncol].corr()
    try:
        display(_corr)
    except NameError:
        print(_corr)
else:
    print("Only one prediction column available; skipping correlation.")



## === cell 5
if ncol > 2:
    corr = concat_sub.iloc[:, 1 : min(7, ncol)].corr()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=0.3,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )
    plt.show()
else:
    print("Only one prediction column available; skipping heatmap.")



## === cell 6
concat_sub["champ_max"] = concat_sub.iloc[:, 1:ncol].max(axis=1)
concat_sub["champ_min"] = concat_sub.iloc[:, 1:ncol].min(axis=1)
concat_sub["champ_mean"] = concat_sub.iloc[:, 1:ncol].mean(axis=1)
concat_sub["champ_median"] = concat_sub.iloc[:, 1:ncol].median(axis=1)



## === cell 7
concat_sub.describe()



## === cell 8
cutoff_lo = 300
cutoff_hi = 0



## === cell 9
if not concat_sub["id"].is_unique:
    concat_sub = concat_sub.drop_duplicates(subset=["id"], keep="first")

if len(concat_sub) != len(sample_sub) or set(concat_sub["id"]) != set(sample_sub["id"]):
    concat_sub = sample_sub[["id"]].merge(concat_sub, on="id", how="left")

pred_cols = [
    c
    for c in concat_sub.columns
    if c.startswith("champ")
    or c in ["champ_max", "champ_min", "champ_mean", "champ_median"]
]
for c in pred_cols:
    concat_sub[c] = concat_sub[c].astype(float)

fallback_val = (
    float(concat_sub["champ_mean"].mean())
    if "champ_mean" in concat_sub.columns
    else 0.0
)
concat_sub[pred_cols] = concat_sub[pred_cols].fillna(fallback_val)

concat_sub.head(), concat_sub.shape



## === cell 10
concat_sub["scalar_coupling_constant"] = concat_sub["champ0"].astype(float)
concat_sub[["id", "scalar_coupling_constant"]].to_csv(
    "stack_mean.csv", index=False, float_format="%.6f"
)
print("Wrote stack_mean.csv with rows:", len(concat_sub))



## === cell 11
pass



## === cell 12
pass



## === cell 13
concat_sub["scalar_coupling_constant"] = concat_sub["champ0"].astype(float)
concat_sub[["id", "scalar_coupling_constant"]].to_csv(
    "stack_median.csv", index=False, float_format="%.6f"
)
print("Wrote stack_median.csv with rows:", len(concat_sub))



## === cell 14
pass



## === cell 15
pass



## === cell 16
champ_only_cols = [
    c
    for c in concat_sub.columns
    if c.startswith("champ")
    and c not in ["champ_max", "champ_min", "champ_mean", "champ_median"]
]
if len(champ_only_cols) == 0:
    champ_only_cols = ["champ_mean"]

concat_sub["scalar_coupling_constant"] = concat_sub["champ0"].astype(float)
concat_sub[["id", "scalar_coupling_constant"]].to_csv(
    "stack_pushout_median.csv", index=False, float_format="%.6f"
)
print("Wrote stack_pushout_median.csv with rows:", len(concat_sub))



## === cell 17
pass



## === cell 18
pass



## === cell 19
concat_sub["scalar_coupling_constant"] = concat_sub["champ0"].astype(float)
concat_sub[["id", "scalar_coupling_constant"]].to_csv(
    "stack_minmax_mean.csv", index=False, float_format="%.6f"
)
print("Wrote stack_minmax_mean.csv with rows:", len(concat_sub))



## === cell 20
pass



## === cell 21
pass



## === cell 22
concat_sub["scalar_coupling_constant"] = concat_sub["champ0"].astype(float)
concat_sub[["id", "scalar_coupling_constant"]].to_csv(
    "stack_minmax_median.csv", index=False, float_format="%.6f"
)
print("Wrote stack_minmax_median.csv with rows:", len(concat_sub))



## === cell 23
pass



## === cell 24
pass



## === cell 25
best_base_path = "../input/champstacks/submission-giba-1 (2).csv"

if os.path.exists(best_base_path):
    sub_base = pd.read_csv(best_base_path)
    if "id" in sub_base.columns:
        sub_base = sub_base.sort_values("id").reset_index(drop=True)
    print("Loaded base submission:", best_base_path, "rows:", len(sub_base))
else:
    sub_base = pd.DataFrame(
        {
            "id": concat_sub["id"].values,
            "scalar_coupling_constant": concat_sub["champ0"].values,
        }
    )
    print("Base submission not found; using champ0 as base.")



## === cell 26
sub_base = sub_base[["id", "scalar_coupling_constant"]].copy()
concat_sub = concat_sub.merge(
    sub_base.rename(columns={"scalar_coupling_constant": "champ_base"}),
    on="id",
    how="left",
)
concat_sub["champ_base"] = (
    concat_sub["champ_base"].fillna(concat_sub["champ0"]).astype(float)
)

concat_sub["scalar_coupling_constant"] = concat_sub["champ0"].astype(float)

concat_sub[["id", "scalar_coupling_constant"]].to_csv(
    "stack_minmax_bestbase.csv", index=False, float_format="%.6f"
)
print("Wrote stack_minmax_bestbase.csv with rows:", len(concat_sub))

concat_sub[["id", "scalar_coupling_constant"]].to_csv(
    "submission.csv", index=False, float_format="%.6f"
)
print("Wrote submission.csv with rows:", len(concat_sub))
