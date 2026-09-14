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
scipy==1.15.3
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

-1.965121149403092

# 6. Current score

2.58431

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.99777) has done: 'I fix the immediate runtime failure by removing the hardcoded `../input/models` dependency and instead reading the competition’s `sample_submission.csv` so the notebook can always produce a valid submission. Since your current code is a stacking/blending notebook that expects external model prediction CSVs (which do not exist in this environment), the minimal correct fallback is to generate a baseline prediction file with the required columns and `.csv` suffix. I also remove notebook-only/IPython commands (`%matplotlib inline`) and deprecation issues (`np.bool`) so it runs as a plain Python script on Kaggle. The output be a valid `stack_median.csv` submission.'
- What this solution (achieved 1.18497) has done: 'Your current score (1.99777, lower-is-better) is far worse than the target (-1.9651), and the reason is that the notebook is producing an all-zeros fallback because no external model prediction files exist. To move the score toward the target with minimal change, I keep the same “single submission CSV generation” flow but replace the zero fallback with a simple, legitimate baseline: train a per-`type` median on `train.csv` and predict those medians for `test.csv` by `type`. This preserves the overall lightweight approach (no new model architecture/training loop) while aligning predictions with the competition’s per-type error behavior, which should materially reduce MAE. I also keep the existing stacking logic intact when `../input/models` exists, and only use the baseline when it doesn’t.'
- What this solution (achieved 1.19861) has done: 'I fix the root runtime error by ensuring the LinearRegression branch never sees NaNs: the current `tr[feature_cols].isna()` check only looks at the training slice, but NaNs can exist in `X_te` due to missing structure merges, which is why `predict()` crashes. I add a minimal, score-neutral imputation step (fill NaNs in both train/test feature matrices using per-type training medians, with a safe fallback to global medians) while keeping the same per-type modeling logic and features. Then I add a small guard so downstream cells don’t crash when only a single prediction column exists (baseline case), by creating `m_median` directly in that case. Finally, the script always write a valid `stack_median.csv` with the required columns.'
- What this solution (achieved 1.19732) has done: 'Your current score is far worse than the (much lower) target, so we should make a small, legitimate accuracy improvement without changing the overall “per-type simple model” approach. The biggest gain with minimal risk here is to enrich the existing linear-regression-per-type baseline with a couple of physically meaningful geometric features derived from the same merged coordinates (no new data sources, no new training loop). Specifically, we add absolute coordinate deltas and a stabilized log-distance, and we keep the same per-type fitting/prediction flow with the same NaN-handling safeguards. This typically reduces MAE noticeably for this competition while staying lightweight and deterministic, and it still always write a valid `stack_median.csv`.'
- What this solution (achieved 1.18487) has done: 'We need to move your score down (lower-is-better) toward the much better target, but with minimal changes and without changing the overall “per-type simple regression with geometry features” core logic. The biggest safe gain here is to fit the linear regression per coupling `type` on a log1p-transformed target (still the same model/training loop), then invert with expm1; this stabilizes heavy-tailed targets and typically reduces MAE for this competition. To keep predictions physically plausible and reduce catastrophic errors (which dominate log-MAE), we also clip each type’s predictions to a robust train-derived range (e.g., 0.5–99.5 percentile) with a small padding. Finally, we keep your existing NaN guards and always write a valid `stack_median.csv`.'
- What this solution (achieved 1.39163) has done: 'Your current score (1.18487, lower-is-better) is still far above the target (-1.9651), so we should make a small, legitimate accuracy improvement while keeping the same per-type LinearRegression-on-geometry core logic. The lowest-risk gain here is to add a couple of very standard, physically meaningful features (atom-pair midpoint radius, per-atom radii, and simple dot-products) computed from the same merged coordinates; this often reduces per-type MAE without changing the modeling approach. I also add one tiny robustness tweak by using a slightly stronger, still-robust clipping range (1–99%) to reduce occasional extreme errors that dominate the log-MAE metric. Everything else (data paths, per-type loop, log1p/expm1 target transform, and submission writing) stays the same.'
- What this solution (achieved 1.46982) has done: 'Your current score is much worse than the target (lower-is-better), and the recent feature additions likely hurt because the regression is trying to fit strong type-specific offsets without explicit intercept-like features. To move the score back down with minimal change and the same per-type LinearRegression core logic, I (1) add very small, standardization-within-type feature scaling (fit on train per type, apply to test) to stabilize coefficients across heterogeneous feature magnitudes, and (2) add an explicit per-type target centering (fit on centered log1p target, then add the mean back) which often reduces bias without changing the model class. I keep your existing log1p/expm1 transform, clipping, NaN handling, and submission writing unchanged otherwise. This should plausibly recover performance toward your previous ~1.18 band (closer to the target than 1.39) without altering the overall approach.'
- What this solution (achieved 1.18489) has done: 'Your current score is worse than your best recent run (~1.18), so we should revert the specific “new features + scaling/centering” pieces that likely destabilized LinearRegression while keeping the same per-type LinearRegression-on-geometry core approach. I make the smallest change that usually improves this competition baseline: remove the recently added dot/radius features and per-type feature scaling/target-centering, and go back to a simpler, stable feature set (distance + coordinate deltas + atom encodings) with the same log1p/expm1 and per-type clipping. This keeps the training loop and model class identical (still per-type LinearRegression) and should move the score back down toward the earlier ~1.18 band. The submission writing path and format remain unchanged and still always produce `stack_median.csv`.'
- What this solution (achieved 1.57689) has done: 'Your current score is far worse than the target (lower is better), so we should make a small, low-risk improvement while keeping the same per-type LinearRegression + geometry-feature pipeline. The main issue is that the model is trying to learn large type-specific offsets without an explicit bias feature tied to atom identities; adding a couple of simple, physically meaningful atom-identity features (atomic number and electronegativity for each atom, plus differences/sums) usually improves MAE a lot in this competition without changing the training loop or model class. I also add two very small geometric features (squared distance components) that are still derived from the same coordinates and keep the existing log1p/expm1 + per-type clipping intact. Everything else (fallback behavior, per-type fit, NaN handling, and writing `stack_median.csv`) remains the same.'
- What this solution (achieved 1.54845) has done: 'Your current score (1.57689, lower-is-better) is still far from the target (-1.9651), so we should make a small, legitimate improvement while keeping the same per-type LinearRegression + geometry features pipeline. The biggest low-risk gain here is to add two classic interaction features (`dist*inv_dist` and `dist*log_dist`) and a per-type intercept-like feature by including the train-per-type mean of the log-target as an explicit feature (computed only from train, then merged to test by `type`). This keeps the exact same training loop/model class (still LinearRegression per type on log1p target) and typically reduces bias by type without changing evaluation semantics. I also make the `Z0/Z1/EN0/EN1` mapping robust to missing atoms by filling NaNs (so we don’t silently fall back as often), which should reduce errors rather than change the approach.'
- What this solution (achieved 2.58431) has done: 'Your current score (1.54845, lower-is-better) is far worse than the target, so we should improve accuracy with the smallest possible change while keeping the same per-type LinearRegression-on-geometry core. The biggest low-risk gain here is to stop using `log1p(y)` on a target that can be negative for several coupling types; this transform distorts/invalidates values and can silently degrade fit, so we instead use a sign-preserving transform `sign(y)*log1p(abs(y))` and invert it after prediction. Everything else stays the same: same features, same per-type training loop, same NaN filling, and same per-type clipping (applied in the original target space). This should reduce MAE substantially and move the score down toward the target band without changing the overall approach.'
- What this solution (achieved 2.64123) has done: 'Your current score (2.58431, lower-is-better) is far worse than the target (-1.9651), so we should make a small, safe improvement without changing the core “per-type LinearRegression on merged structure geometry + simple atom features” logic. The biggest likely issue is the current per-type clipping, which is applied in the raw target space and can badly distort types whose values are negative or span wide ranges; we instead clip in the transformed space (the same sign-preserving log space you already train in), then invert—this keeps the model/training loop identical but prevents extreme raw-space clipping artifacts. We also ensure the submission `id` order exactly matches `sample_submission.csv` (stable alignment is score-critical and cheap), without changing any feature or model behavior. Everything else (data paths, feature set, per-type loop, NaN filling, and writing `stack_median.csv`) stays the same.'
- What this solution (achieved 2.58431) has done: 'Your current score is much worse than target (lower-is-better), but the core model/feature pipeline is already the “right shape”; the biggest likely regression is the recent change to clip in transformed space, which can mis-calibrate after inversion and inflate MAE. I make the smallest corrective change: keep the sign-preserving transform training exactly as-is, but move clipping back to the original target space using robust per-type train percentiles (applied after `y_inverse`). I also add a tiny safety guard for types with near-constant targets (where percentile ranges collapse) to avoid over-clipping. Everything else (data reading, features, per-type LinearRegression loop, NaN filling, and submission alignment to `sample_submission.csv`) remains unchanged, and it still write `stack_median.csv`.'
- What this solution (achieved 2.58431) has done: 'Your current score (2.58431, lower-is-better) is far worse than the target, and the largest likely cause inside your existing per-type LinearRegression pipeline is the sign-preserving log transform being fed a raw target that spans negative to positive—this can still be hard for plain OLS to fit because different coupling types have very different scales and offsets. With minimal change and the same core training loop/model class, I add one safe, competition-standard feature: one-hot encoding of the `type` (merged as fixed columns into both train/test), which gives the linear model an explicit type-specific bias and reduces systematic error without changing the approach. I keep your current geometry + atom features, NaN filling, and per-type clipping logic intact, and I also ensure the submission remains aligned to `sample_submission.csv` by `id`. This should move the score down (improve) toward the target band while keeping runtime under the limit.'

# 9. Code solution

## === cell 0
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling",
    "../input/champs-scalar-coupling",
    "../input",  # sometimes files are directly here
    "/kaggle/input",
    "/kaggle/data",
]


def find_file(filename: str) -> str:
    for root in DATA_ROOT_CANDIDATES:
        p = os.path.join(root, filename)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Could not find {filename} under any of: {DATA_ROOT_CANDIDATES}"
    )




## === cell 1
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import gmean

for root in ["/kaggle/input", "/kaggle/data", "../input", "."]:
    if os.path.isdir(root):
        try:
            print(root, "->", sorted(os.listdir(root))[:20])
        except Exception as e:
            print(root, "-> (could not list)", e)



## === cell 2
sub_path = "../input/models"
all_files = []
if os.path.isdir(sub_path):
    all_files = [f for f in os.listdir(sub_path) if f.lower().endswith(".csv")]
all_files



## === cell 3
to_remove = [
    "submission-1.701.csv",
    "submission-1.643.csv",
    "submission-1.481.csv",
    "submission-1.302.csv",
    "submission-1.619.csv",
    "submission-1.662.csv",
    "submission-1.696.csv",
    "submission-1.780.csv",
    "submission-1.708.csv",
    "submission-1.714.csv",
]
all_files = [f for f in all_files if f not in set(to_remove)]
all_files



## === cell 4
all_files



## === cell 5
if len(all_files) > 0:
    outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files]
    concat_sub = pd.concat(outs, axis=1)
    cols = list(map(lambda x: "mol" + str(x), range(len(concat_sub.columns))))
    concat_sub.columns = cols
    concat_sub.reset_index(inplace=True)
    if "index" in concat_sub.columns and "id" not in concat_sub.columns:
        concat_sub.rename(columns={"index": "id"}, inplace=True)
else:
    from sklearn.linear_model import LinearRegression

    train_path = find_file("train.csv")
    test_path = find_file("test.csv")
    structures_path = find_file("structures.csv")
    sample_sub_path = find_file("sample_submission.csv")

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
        structures_path,
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

    train = train.merge(
        s0[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    train = train.merge(
        s1[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
    )

    test = test.merge(
        s0[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    test = test.merge(
        s1[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
    )

    def add_dist(df: pd.DataFrame) -> pd.DataFrame:
        dx = (df["x0"] - df["x1"]).astype(np.float64)
        dy = (df["y0"] - df["y1"]).astype(np.float64)
        dz = (df["z0"] - df["z1"]).astype(np.float64)

        df["dx"] = dx
        df["dy"] = dy
        df["dz"] = dz
        df["adx"] = np.abs(dx)
        df["ady"] = np.abs(dy)
        df["adz"] = np.abs(dz)

        df["dx2"] = dx * dx
        df["dy2"] = dy * dy
        df["dz2"] = dz * dz

        df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz)
        df["inv_dist"] = 1.0 / (df["dist"] + 1e-6)
        df["dist2"] = df["dist"] * df["dist"]
        df["log_dist"] = np.log(df["dist"] + 1e-6)

        df["dist_inv"] = df["dist"] * df["inv_dist"]
        df["dist_log"] = df["dist"] * df["log_dist"]
        return df

    train = add_dist(train)
    test = add_dist(test)

    atom_values = pd.concat(
        [train["atom_0"], train["atom_1"], test["atom_0"], test["atom_1"]], axis=0
    )
    atom_values = atom_values.dropna().unique().tolist()
    atom_to_int = {a: i for i, a in enumerate(sorted(atom_values))}
    for df in (train, test):
        df["atom0_i"] = df["atom_0"].map(atom_to_int).fillna(-1).astype(np.int16)
        df["atom1_i"] = df["atom_1"].map(atom_to_int).fillna(-1).astype(np.int16)

    atomic_number = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
    electronegativity = {"H": 2.20, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98}
    for df in (train, test):
        df["Z0"] = df["atom_0"].map(atomic_number).fillna(0.0).astype(np.float64)
        df["Z1"] = df["atom_1"].map(atomic_number).fillna(0.0).astype(np.float64)
        df["EN0"] = df["atom_0"].map(electronegativity).fillna(0.0).astype(np.float64)
        df["EN1"] = df["atom_1"].map(electronegativity).fillna(0.0).astype(np.float64)

        df["dZ"] = (df["Z0"] - df["Z1"]).astype(np.float64)
        df["sZ"] = (df["Z0"] + df["Z1"]).astype(np.float64)
        df["dEN"] = (df["EN0"] - df["EN1"]).astype(np.float64)
        df["sEN"] = (df["EN0"] + df["EN1"]).astype(np.float64)

    def y_transform(y: np.ndarray) -> np.ndarray:
        y = y.astype(np.float64)
        return np.sign(y) * np.log1p(np.abs(y))

    def y_inverse(yt: np.ndarray) -> np.ndarray:
        yt = yt.astype(np.float64)
        return np.sign(yt) * np.expm1(np.abs(yt))

    type_log_mean = (
        train.assign(
            ytr=y_transform(train["scalar_coupling_constant"].astype(np.float64).values)
        )
        .groupby("type")["ytr"]
        .mean()
        .rename("type_ylog_mean")
        .astype(np.float64)
    )
    global_type_log_mean = float(type_log_mean.mean())
    train = train.merge(type_log_mean.reset_index(), on="type", how="left")
    test = test.merge(type_log_mean.reset_index(), on="type", how="left")
    train["type_ylog_mean"] = train["type_ylog_mean"].fillna(global_type_log_mean)
    test["type_ylog_mean"] = test["type_ylog_mean"].fillna(global_type_log_mean)

    all_types = pd.Index(pd.concat([train["type"], test["type"]], axis=0).unique())
    for df in (train, test):
        for t in all_types:
            df[f"type_oh_{t}"] = (df["type"] == t).astype(np.int8)

    type_oh_cols = [f"type_oh_{t}" for t in all_types]

    feature_cols = [
        "dist",
        "inv_dist",
        "dist2",
        "log_dist",
        "dist_inv",
        "dist_log",
        "dx",
        "dy",
        "dz",
        "adx",
        "ady",
        "adz",
        "dx2",
        "dy2",
        "dz2",
        "atom0_i",
        "atom1_i",
        "Z0",
        "Z1",
        "dZ",
        "sZ",
        "EN0",
        "EN1",
        "dEN",
        "sEN",
        "type_ylog_mean",
    ] + type_oh_cols

    type_median = (
        train.groupby("type")["scalar_coupling_constant"].median().astype(np.float64)
    )
    global_median = float(train["scalar_coupling_constant"].median())

    global_feat_median = train[feature_cols].astype(np.float64).median()

    preds = np.empty(len(test), dtype=np.float64)
    preds[:] = np.nan

    for t, idx in test.groupby("type").indices.items():
        te_idx = np.array(list(idx), dtype=np.int64)
        tr = train[train["type"] == t]

        if len(tr) < 50:
            fallback = float(type_median.get(t, global_median))
            preds[te_idx] = fallback
            continue

        X_tr_df = tr[feature_cols].astype(np.float64)
        y_tr_raw = tr["scalar_coupling_constant"].astype(np.float64).values
        y_tr = y_transform(y_tr_raw)

        X_te_df = test.loc[te_idx, feature_cols].astype(np.float64)

        type_feat_median = X_tr_df.median()
        fill_vals = type_feat_median.fillna(global_feat_median).to_dict()

        X_tr = X_tr_df.fillna(fill_vals).values
        X_te = X_te_df.fillna(fill_vals).values

        if not (
            np.isfinite(X_tr).all()
            and np.isfinite(X_te).all()
            and np.isfinite(y_tr).all()
        ):
            fallback = float(type_median.get(t, global_median))
            preds[te_idx] = fallback
            continue

        model = LinearRegression(n_jobs=None)
        model.fit(X_tr, y_tr)

        pred_t = model.predict(X_te)
        pred_raw = y_inverse(pred_t)

        lo = np.nanpercentile(y_tr_raw, 1.0)
        hi = np.nanpercentile(y_tr_raw, 99.0)
        rng = hi - lo
        if np.isfinite(rng) and rng > 1e-12 and np.isfinite(lo) and np.isfinite(hi):
            pad = 0.05 * rng
            lo2 = lo - pad
            hi2 = hi + pad
            if lo2 < hi2:
                pred_raw = np.clip(pred_raw, lo2, hi2)

        preds[te_idx] = pred_raw

    preds = np.where(np.isfinite(preds), preds, global_median)

    sample_sub = pd.read_csv(sample_sub_path, usecols=["id"])
    pred_df = pd.DataFrame(
        {"id": test["id"].astype(int), "mol0": preds.astype(np.float64)}
    )
    concat_sub = sample_sub.merge(pred_df, on="id", how="left")
    concat_sub["mol0"] = concat_sub["mol0"].fillna(global_median).astype(np.float64)

concat_sub.head()
ncol = concat_sub.shape[1]
ncol



## === cell 6
if concat_sub.shape[1] > 2:
    corr = concat_sub.iloc[:, 1:].corr()
else:
    corr = pd.DataFrame()
corr



## === cell 7
if concat_sub.shape[1] > 3:
    corr = concat_sub.iloc[:, 1:].corr()
    mask = np.zeros_like(corr, dtype=bool)  # fix deprecated np.bool
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="prism",
        vmin=0.96,
        center=0,
        square=True,
        linewidths=1,
        annot=True,
        fmt=".4f",
        ax=ax,
    )
    plt.show()



## === cell 8
pred_cols = [c for c in concat_sub.columns if c != "id"]
concat_sub["m_max"] = concat_sub[pred_cols].max(axis=1)
concat_sub["m_min"] = concat_sub[pred_cols].min(axis=1)
concat_sub["m_median"] = concat_sub[pred_cols].median(axis=1)



## === cell 9
concat_sub.describe()



## === cell 10
concat_sub.head(10)



## === cell 11
cutoff_lo = 0.8
cutoff_hi = 0.2



## === cell 12
concat_sub["scalar_coupling_constant"] = concat_sub["m_median"]

sub = concat_sub[["id", "scalar_coupling_constant"]].copy()
sub["id"] = sub["id"].astype(int)

out_path = "stack_median.csv"
sub.to_csv(out_path, index=False, float_format="%.6f")

print("Wrote submission:", out_path)
print(sub.head())
print("Rows:", len(sub))
