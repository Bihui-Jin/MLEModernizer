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

-2.366647166218125

# 6. Current score

1.18497

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.18497) has done: 'The failure is because the notebook expects external prediction files in `../input/champ-preds` and `../input/nnet-*` that are not present in your environment, so the first `read_csv` raises `FileNotFoundError` and all later columns never get created. I keep the ensemble logic intact, but make the code robust by (1) automatically using the correct base data path you actually have (`/kaggle/data/champs-scalar-coupling`), (2) gracefully skipping missing prediction files and re-normalizing the remaining weights, and (3) falling back to a safe baseline (per-type median from train) so a valid submission CSV is always produced. Plotting be made conditional so it won’t crash when optional columns are missing. This is score-improving versus “no submission”, and otherwise score-neutral with respect to the intended ensemble when those files are available.'
- What this solution (achieved 1.18497) has done: 'Your current score (1.18497, lower-is-better) is far worse than the target (-2.3666), and the main reason is that all external prediction files are missing so you’re falling back to a per-type median baseline. To move the score substantially toward the target while preserving the same “ensemble of model predictions” semantics, I keep the exact ensemble/weighting logic but add a lightweight in-notebook surrogate to generate the missing `n1/n2/lgb_a/lgb_m/nnet` columns from the provided training data. Concretely, when external files are absent, we compute per-`type` linear coefficients based on a physically meaningful proxy feature (inverse inter-atomic distance from `structures.csv`) and predict `scalar_coupling_constant` as `a_type + b_type*(1/distance)`. This is a minimal, fast, deterministic addition that typically beats the median baseline by a lot on CHAMPS and should move the metric closer to your target, while still writing a valid `ensemble_sub.csv`.'
- What this solution (achieved 1.18497) has done: 'Your current score is far worse than the target (lower-is-better), so we should improve the surrogate predictions while keeping the same “fill missing model preds then weighted-average ensemble” core logic intact. The minimal high-impact fix is to make the surrogate less underfit by using a tiny per-type **ridge regression** on a small set of geometry-derived features computed from `structures.csv` (still deterministic, fast, and aligned with the task physics), instead of only `1/distance`. We keep the ensemble weights and prediction-column semantics unchanged; we only replace the fallback surrogate generation used when external prediction files are missing. This should move your score substantially toward the target while still writing a valid `ensemble_sub.csv`.'
- What this solution (achieved 1.18497) has done: 'We keep your ensemble logic and weights exactly the same, but improve the surrogate predictions that fill in for missing external model files (which is currently the main driver of the poor score). The minimal high-impact change is to add a couple of physics-aligned, cheap geometry features (distance powers) and fit a per-type ridge model **in log1p-space** of the target with a sign-preserving transform; this better matches the competition’s log-MAE behavior without changing the overall training approach (still per-type closed-form ridge). We also standardize features per type using train statistics (applied to test) to stabilize coefficients and reduce extreme extrapolation. Everything still runs end-to-end within time and writes `ensemble_sub.csv` in the required format.'
- What this solution (achieved 1.18497) has done: 'Your current score is much worse than the target (lower-is-better), so we should improve the surrogate predictions that replace the missing external model files, while keeping the ensemble logic and weights unchanged. The smallest high-impact change is to add a few cheap, chemistry-aligned features beyond pair distance: atom identity (for both atoms) and simple molecule-level aggregates (mean/std of coordinates and atom count), then refit the same per-type ridge in signed-log space. This preserves your training approach (closed-form ridge per type) and evaluation semantics, but should substantially reduce MAE versus the geometry-only baseline and move the score toward the target band. We also make the ridge solve slightly more numerically stable without changing the algorithm.'
- What this solution (achieved 1.18497) has done: 'Your current score is much worse than the (lower-is-better) target, so we should improve the fallback surrogate predictions (used because the external ensemble prediction files are missing) while keeping your ensemble weights/logic unchanged. The smallest high-impact fix is to add a few “official auxiliary” features (mulliken charges, shielding tensors, dipole moments, potential energy) into the same per-type ridge surrogate—this preserves the same training approach (closed-form ridge per type) and prediction semantics, but typically reduces MAE a lot on CHAMPS. We also make the ridge solve slightly more numerically robust (use `np.linalg.lstsq` fallback if `solve` fails) without changing the algorithm. Everything still runs end-to-end, stays within the provided data paths, and writes `ensemble_sub.csv` in the correct format.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path

CANDIDATE_BASES = [
    Path("../input/champs-scalar-coupling"),  # Kaggle classic
    Path("/kaggle/input/champs-scalar-coupling"),  # Kaggle classic absolute
    Path("/kaggle/data/champs-scalar-coupling"),  # this environment
    Path("/kaggle/data/input/champs-scalar-coupling"),  # possible alternative
]
BASE = next((p for p in CANDIDATE_BASES if p.exists()), None)

print("Resolved BASE:", BASE)
if BASE is None:
    raise FileNotFoundError(
        "Could not find champs-scalar-coupling data directory in any of: "
        + ", ".join(map(str, CANDIDATE_BASES))
    )

print("Listing BASE files (head):")
print(sorted([p.name for p in BASE.iterdir()])[:20])



## === cell 1
import pandas as pd
import numpy as np

TRAIN_PATH = BASE / "train.csv"
TEST_PATH = BASE / "test.csv"
SAMPLE_SUB_PATH = BASE / "sample_submission.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

print(
    "train shape:",
    train.shape,
    "test shape:",
    test.shape,
    "sample_sub shape:",
    sample_sub.shape,
)
print("train cols:", train.columns.tolist())
print("test cols:", test.columns.tolist())



## === cell 2
TARGET = "scalar_coupling_constant"


def _read_pred_series(pred_path, target_col=TARGET):
    """Read a prediction file and return a 1D numpy array aligned by test row order.
    Supports either:
      - column 'scalar_coupling_constant' in file (same row order as test)
      - OR first column is an index and target col exists
      - OR a single unnamed prediction column
    Returns None if file doesn't exist.
    """
    pred_path = Path(pred_path)
    if not pred_path.exists():
        return None

    df = pd.read_csv(pred_path)
    if target_col in df.columns:
        return df[target_col].to_numpy()

    try:
        df2 = pd.read_csv(pred_path, index_col=0)
        if target_col in df2.columns:
            return df2[target_col].to_numpy()
        if df2.shape[1] == 1:
            return df2.iloc[:, 0].to_numpy()
    except Exception:
        pass

    if df.shape[1] == 1:
        return df.iloc[:, 0].to_numpy()

    raise ValueError(
        f"Don't know how to extract predictions from {pred_path} with columns {df.columns.tolist()}"
    )


def get_median_from_files(files):
    series_list = []
    for f in files:
        s = _read_pred_series(f)
        if s is not None:
            series_list.append(pd.Series(s))
        else:
            print("Missing (skipped):", f)
    if not series_list:
        return None
    concat_sub = pd.concat(series_list, axis=1, sort=False)
    return concat_sub.median(axis=1).to_numpy()


def get_mean_from_files(files):
    series_list = []
    for f in files:
        s = _read_pred_series(f)
        if s is not None:
            series_list.append(pd.Series(s))
        else:
            print("Missing (skipped):", f)
    if not series_list:
        return None
    concat_sub = pd.concat(series_list, axis=1, sort=False)
    return concat_sub.mean(axis=1).to_numpy()


test["n1"] = np.nan
test["n2"] = np.nan
test["lgb_a"] = np.nan
test["lgb_m"] = np.nan
test["nnet"] = np.nan

n1 = _read_pred_series("../input/champ-preds/gnn_median_2279.csv")
if n1 is not None:
    test["n1"] = n1
else:
    print("Missing (skipped): ../input/champ-preds/gnn_median_2279.csv")

n2 = _read_pred_series("../input/champ-preds/gnn_train_sep_2258.csv")
if n2 is not None:
    test["n2"] = n2
else:
    print("Missing (skipped): ../input/champ-preds/gnn_train_sep_2258.csv")

lgb_a = get_mean_from_files(
    [
        "../input/champ-preds/submission_type_2085.csv",
        "../input/champ-preds/submission_type_2082.csv",
    ]
)
if lgb_a is not None:
    test["lgb_a"] = lgb_a

lgb_m = get_mean_from_files(
    [
        "../input/champ-preds/lgb_type_full_f286_10.csv",
        "../input/champ-preds/lgb_type_full_f262_10.csv",
    ]
)
if lgb_m is not None:
    test["lgb_m"] = lgb_m

nnet = get_median_from_files(
    [
        "../input/nnet-b-seed-10/lgb_type_cv-1.72296_mae0.24019_bags-1_f120_fd5_10.csv",
        "../input/nnet-b-seed-11/lgb_type_cv-1.70647_mae0.2376_bags-1_f120_fd5_11.csv",
        "../input/nnet-b-seed-12/lgb_type_cv-1.69833_mae0.24106_bags-1_f120_fd5_12.csv",
        "../input/nnet-try/lgb_type_cv-2.108373877033157_mae0.12143527465528912_bags-1_f120_fd5_10.csv",
        "../input/nnet-try-seed-11/lgb_type_cv-1.64944_mae0.23965_bags-1_f120_fd5_11.csv",
        "../input/nnet-try-seed-12/lgb_type_cv-1.64875_mae0.2412_bags-1_f120_fd5_12.csv",
    ]
)
if nnet is not None:
    test["nnet"] = nnet

print("Available prediction columns (non-null counts):")
print(test[["n1", "n2", "lgb_a", "lgb_m", "nnet"]].notnull().sum())

test.head(10)



## === cell 3

STRUCTURES_PATH = BASE / "structures.csv"
MULLIKEN_PATH = BASE / "mulliken_charges.csv"
TENSORS_PATH = BASE / "magnetic_shielding_tensors.csv"
DIPOLE_PATH = BASE / "dipole_moments.csv"
POT_PATH = BASE / "potential_energy.csv"

structures = pd.read_csv(
    STRUCTURES_PATH,
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
)

mol_agg = (
    structures.groupby("molecule_name")[["x", "y", "z"]]
    .agg(["mean", "std"])
    .reset_index()
)
mol_agg.columns = [
    "molecule_name",
    "x_mean",
    "x_std",
    "y_mean",
    "y_std",
    "z_mean",
    "z_std",
]
mol_natoms = (
    structures.groupby("molecule_name")["atom_index"]
    .max()
    .add(1)
    .rename("n_atoms")
    .reset_index()
)
mol_agg = mol_agg.merge(
    mol_natoms, on="molecule_name", how="left", validate="one_to_one"
)

dipole = pd.read_csv(DIPOLE_PATH)  # molecule_name, X,Y,Z
dipole = dipole.rename(columns={"X": "dip_x", "Y": "dip_y", "Z": "dip_z"})
pot = pd.read_csv(POT_PATH)  # molecule_name, potential_energy

mol_aux = mol_agg.merge(dipole, on="molecule_name", how="left", validate="one_to_one")
mol_aux = mol_aux.merge(pot, on="molecule_name", how="left", validate="one_to_one")

mull = pd.read_csv(MULLIKEN_PATH)  # molecule_name, atom_index, mulliken_charge
tens = pd.read_csv(TENSORS_PATH)  # molecule_name, atom_index, 9 tensor comps
atom_aux = mull.merge(
    tens, on=["molecule_name", "atom_index"], how="left", validate="one_to_one"
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

a0 = atom_aux.rename(
    columns={
        "atom_index": "atom_index_0",
        "mulliken_charge": "q0",
        "XX": "t0_XX",
        "YX": "t0_YX",
        "ZX": "t0_ZX",
        "XY": "t0_XY",
        "YY": "t0_YY",
        "ZY": "t0_ZY",
        "XZ": "t0_XZ",
        "YZ": "t0_YZ",
        "ZZ": "t0_ZZ",
    }
)
a1 = atom_aux.rename(
    columns={
        "atom_index": "atom_index_1",
        "mulliken_charge": "q1",
        "XX": "t1_XX",
        "YX": "t1_YX",
        "ZX": "t1_ZX",
        "XY": "t1_XY",
        "YY": "t1_YY",
        "ZY": "t1_ZY",
        "XZ": "t1_XZ",
        "YZ": "t1_YZ",
        "ZZ": "t1_ZZ",
    }
)

train_xy = (
    train[["molecule_name", "atom_index_0", "atom_index_1", "type", TARGET]]
    .merge(s0, on=["molecule_name", "atom_index_0"], how="left", validate="many_to_one")
    .merge(s1, on=["molecule_name", "atom_index_1"], how="left", validate="many_to_one")
    .merge(mol_aux, on="molecule_name", how="left", validate="many_to_one")
    .merge(a0, on=["molecule_name", "atom_index_0"], how="left", validate="many_to_one")
    .merge(a1, on=["molecule_name", "atom_index_1"], how="left", validate="many_to_one")
)

test_xy = (
    test[["id", "molecule_name", "atom_index_0", "atom_index_1", "type"]]
    .merge(s0, on=["molecule_name", "atom_index_0"], how="left", validate="many_to_one")
    .merge(s1, on=["molecule_name", "atom_index_1"], how="left", validate="many_to_one")
    .merge(mol_aux, on="molecule_name", how="left", validate="many_to_one")
    .merge(a0, on=["molecule_name", "atom_index_0"], how="left", validate="many_to_one")
    .merge(a1, on=["molecule_name", "atom_index_1"], how="left", validate="many_to_one")
)

atom_categories = pd.Index(sorted(structures["atom"].dropna().unique().tolist()))
atom_to_code = {a: i for i, a in enumerate(atom_categories)}
print("Atom categories:", atom_categories.tolist())


def _geom_features(df):
    dx = (df["x0"] - df["x1"]).astype("float64")
    dy = (df["y0"] - df["y1"]).astype("float64")
    dz = (df["z0"] - df["z1"]).astype("float64")
    d2 = dx * dx + dy * dy + dz * dz
    d = np.sqrt(d2)
    d = np.clip(d, 1e-6, None)

    inv_d = 1.0 / d
    inv_d2 = 1.0 / np.clip(d2, 1e-12, None)
    inv_d3 = inv_d2 * inv_d
    inv_sqrt_d = 1.0 / np.sqrt(d)

    a0c = df["atom_0"].map(atom_to_code).fillna(-1).astype("float64")
    a1c = df["atom_1"].map(atom_to_code).fillna(-1).astype("float64")
    a_sum = a0c + a1c
    a_diff = a0c - a1c
    a_prod = a0c * a1c

    x_mean = df["x_mean"].fillna(0.0).astype("float64")
    y_mean = df["y_mean"].fillna(0.0).astype("float64")
    z_mean = df["z_mean"].fillna(0.0).astype("float64")
    x_std = df["x_std"].fillna(0.0).astype("float64")
    y_std = df["y_std"].fillna(0.0).astype("float64")
    z_std = df["z_std"].fillna(0.0).astype("float64")
    n_atoms = df["n_atoms"].fillna(0.0).astype("float64")

    x0c = df["x0"].astype("float64") - x_mean
    y0c = df["y0"].astype("float64") - y_mean
    z0c = df["z0"].astype("float64") - z_mean
    x1c = df["x1"].astype("float64") - x_mean
    y1c = df["y1"].astype("float64") - y_mean
    z1c = df["z1"].astype("float64") - z_mean

    dip_x = df["dip_x"].fillna(0.0).astype("float64")
    dip_y = df["dip_y"].fillna(0.0).astype("float64")
    dip_z = df["dip_z"].fillna(0.0).astype("float64")
    pot_e = df["potential_energy"].fillna(0.0).astype("float64")

    q0 = df["q0"].fillna(0.0).astype("float64")
    q1 = df["q1"].fillna(0.0).astype("float64")
    q_sum = q0 + q1
    q_diff = q0 - q1
    q_prod = q0 * q1

    t0_cols = [
        "t0_XX",
        "t0_YX",
        "t0_ZX",
        "t0_XY",
        "t0_YY",
        "t0_ZY",
        "t0_XZ",
        "t0_YZ",
        "t0_ZZ",
    ]
    t1_cols = [
        "t1_XX",
        "t1_YX",
        "t1_ZX",
        "t1_XY",
        "t1_YY",
        "t1_ZY",
        "t1_XZ",
        "t1_YZ",
        "t1_ZZ",
    ]
    t0 = df[t0_cols].fillna(0.0).astype("float64")
    t1 = df[t1_cols].fillna(0.0).astype("float64")
    t_sum = t0.to_numpy() + t1.to_numpy()
    t_diff = t0.to_numpy() - t1.to_numpy()

    out = {
        "bias": 1.0,
        "inv_d": inv_d,
        "inv_d2": inv_d2,
        "inv_d3": inv_d3,
        "inv_sqrt_d": inv_sqrt_d,
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "a0": a0c,
        "a1": a1c,
        "a_sum": a_sum,
        "a_diff": a_diff,
        "a_prod": a_prod,
        "n_atoms": n_atoms,
        "x_std": x_std,
        "y_std": y_std,
        "z_std": z_std,
        "x0c": x0c,
        "y0c": y0c,
        "z0c": z0c,
        "x1c": x1c,
        "y1c": y1c,
        "z1c": z1c,
        "dip_x": dip_x,
        "dip_y": dip_y,
        "dip_z": dip_z,
        "potential_energy": pot_e,
        "q0": q0,
        "q1": q1,
        "q_sum": q_sum,
        "q_diff": q_diff,
        "q_prod": q_prod,
    }

    for j, name in enumerate(["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]):
        out[f"t_sum_{name}"] = t_sum[:, j]
        out[f"t_diff_{name}"] = t_diff[:, j]

    return pd.DataFrame(out, index=df.index)


X_train = _geom_features(train_xy)
X_test = _geom_features(test_xy)
y_train = train_xy[TARGET].astype("float64")

LAMBDA = 1e-3


def _signed_log1p(x):
    x = x.astype("float64")
    return np.sign(x) * np.log1p(np.abs(x))


def _signed_expm1(x):
    x = x.astype("float64")
    return np.sign(x) * np.expm1(np.abs(x))


def _solve_ridge(A, b):
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def _ridge_fit_predict_per_type(train_types, Xtr, ytr, test_types, Xte, lam=LAMBDA):
    ytr_t = _signed_log1p(ytr)

    cols = Xtr.columns.tolist()
    p = len(cols)
    I = np.eye(p, dtype="float64")
    I[0, 0] = 0.0  # do not regularize bias

    Xtr_np_all = Xtr.to_numpy(dtype="float64")
    Xte_np_all = Xte.to_numpy(dtype="float64")

    mu_g = Xtr_np_all.mean(axis=0)
    sd_g = Xtr_np_all.std(axis=0)
    sd_g = np.where(sd_g < 1e-12, 1.0, sd_g)
    sd_g[0] = 1.0  # keep bias as-is
    mu_g[0] = 0.0

    Xtr_g = (Xtr_np_all - mu_g) / sd_g
    XtX_g = Xtr_g.T @ Xtr_g
    Xty_g = Xtr_g.T @ ytr_t.to_numpy(dtype="float64")
    beta_g = _solve_ridge(XtX_g + lam * I, Xty_g)

    betas = {}
    mus = {}
    sds = {}

    for t, idx in train_types.groupby(train_types).groups.items():
        X_t = Xtr.loc[idx].to_numpy(dtype="float64")
        y_t = ytr_t.loc[idx].to_numpy(dtype="float64")
        if X_t.shape[0] < 50:
            betas[t] = beta_g
            mus[t] = mu_g
            sds[t] = sd_g
            continue

        mu = X_t.mean(axis=0)
        sd = X_t.std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        sd[0] = 1.0
        mu[0] = 0.0

        Xs = (X_t - mu) / sd
        XtX = Xs.T @ Xs
        Xty = Xs.T @ y_t
        betas[t] = _solve_ridge(XtX + lam * I, Xty)
        mus[t] = mu
        sds[t] = sd

    preds_t = np.empty(Xte.shape[0], dtype="float64")
    for t, idx in test_types.groupby(test_types).groups.items():
        beta = betas.get(t, beta_g)
        mu = mus.get(t, mu_g)
        sd = sds.get(t, sd_g)
        X_block = (Xte_np_all[idx] - mu) / sd
        preds_t[idx] = X_block @ beta

    return _signed_expm1(preds_t)


sur_preds = _ridge_fit_predict_per_type(
    train_xy["type"], X_train, y_train, test_xy["type"], X_test
)

for c in ["n1", "n2", "lgb_a", "lgb_m", "nnet"]:
    if (c not in test.columns) or (not test[c].notnull().any()):
        test[c] = sur_preds

print("After surrogate fill, non-null counts:")
print(test[["n1", "n2", "lgb_a", "lgb_m", "nnet"]].notnull().sum())



## === cell 4
import matplotlib.pyplot as plt
import seaborn as sns

pred_cols = ["n1", "n2", "lgb_a", "lgb_m", "nnet"]
available_cols = [c for c in pred_cols if c in test.columns and test[c].notnull().any()]

print("Correlation cols available:", available_cols)
if len(available_cols) >= 2:
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    sns.heatmap(test[available_cols].corr(), ax=ax)
    plt.show()
    try:
        display(test[available_cols].corr())
    except NameError:
        print(test[available_cols].corr())
else:
    print("Not enough prediction columns to compute correlation heatmap (need >=2).")



## === cell 5
weights = {"n1": 0.7, "n2": 0.05, "lgb_a": 0.15, "lgb_m": 0.04, "nnet": 0.06}

available = [
    k for k, w in weights.items() if k in test.columns and test[k].notnull().all()
]
print("Models with complete predictions:", available)

if available:
    w_sum = sum(weights[k] for k in available)
    test["final_preds"] = 0.0
    for k in available:
        test["final_preds"] += test[k] * (weights[k] / w_sum)
else:
    print("No prediction columns available; using per-type median baseline from train.")
    type_median = train.groupby("type")[TARGET].median()
    global_median = train[TARGET].median()
    test["final_preds"] = (
        test["type"].map(type_median).fillna(global_median).astype("float64")
    )

test[["id", "type", "final_preds"]].head(20)



## === cell 6
sub = sample_sub[["id"]].merge(
    test[["id", "final_preds"]], on="id", how="left", validate="one_to_one"
)

if sub["final_preds"].isna().any():
    global_median = float(train[TARGET].median())
    sub["final_preds"] = sub["final_preds"].fillna(global_median)

submission = pd.DataFrame(
    {"id": sub["id"].astype(np.int64), TARGET: sub["final_preds"].astype(np.float64)}
)

OUT_PATH = "ensemble_sub.csv"
submission.to_csv(OUT_PATH, index=False)
print("Wrote:", OUT_PATH, "shape:", submission.shape)
submission.head(20)



## === cell 7
assert (
    submission.shape[0] == sample_sub.shape[0]
), "Submission row count must match sample_submission"
assert submission.columns.tolist() == [
    "id",
    TARGET,
], "Submission columns must be ['id','scalar_coupling_constant']"
assert OUT_PATH.endswith(".csv"), "Submission filename must end with .csv"
print(submission.describe(include="all"))
