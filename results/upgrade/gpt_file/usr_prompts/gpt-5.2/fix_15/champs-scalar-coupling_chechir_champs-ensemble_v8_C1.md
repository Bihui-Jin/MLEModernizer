# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

-2.1018426058695074

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.222) has done: 'Your notebook fails because it depends on external Kaggle datasets (`../input/champ-preds` and `../input/1-mpnn`) that are not present in your provided environment; this prevents any submission from being written. I replace those missing-file reads with an in-notebook baseline that trains a fast, deterministic per-`type` linear regression using only `train.csv`/`test.csv` plus `structures.csv`-derived geometric features (distance and a few coordinate deltas), which preserves the overall “blend/ensemble-like” intent while making it runnable end-to-end. I also ensure strict ID alignment (merge back on `id`) and write a valid `submission.csv` with the exact required columns. The remaining cells be updated to display the produced submission and a preview of the test predictions, so no downstream `NameError` occurs.'
- What this solution (achieved 2.75915) has done: 'Your current Ridge-per-type baseline is leaving a lot of signal on the table because the metric is averaged per coupling `type`, and the data provides several strong, “allowed” auxiliary physics features (mulliken charges, shielding tensors, dipole moments, potential energy) that can be merged without changing the core modeling approach. I keep the exact same training loop (per-`type` Ridge in a Pipeline) and just extend the feature set with these extra merged features plus a couple of minimal, stable geometric transforms (`dist^2`, `1/dist`) that often help linear models. I also set `fit_intercept=False` since we already include scaling and type-wise training; this is a small calibration change that typically reduces MAE without altering the overall approach. The result still runs end-to-end within constraints and writes a valid `submission.csv`.'
- What this solution (achieved 1.21973) has done: 'Your current score (2.75915, lower-is-better) is far worse than the target (-2.1018), so we should improve it while keeping the same core approach (per-type Ridge). The biggest issue is that your linear model is forced through the origin (`fit_intercept=False`) even though you are not explicitly providing a per-type bias feature; for MAE-style targets this typically harms calibration badly, so enabling the intercept is a minimal, high-impact fix. I also make the `StandardScaler` use `with_mean=True` (safe here because we’re using dense pandas dataframes), which better conditions Ridge and works naturally with an intercept. Everything else (features, per-type loop, Ridge, submission schema/paths) stays the same.'
- What this solution (achieved 1.21977) has done: 'Your current score is far from the target and (since lower is better) we should improve it, but with minimal core-logic changes: keep the same per-type Ridge pipeline and the same feature set. The biggest likely issue is metric mismatch: the competition evaluates log(MAE) per type, and `scalar_coupling_constant` has very different scales per type, so training on the raw target makes Ridge focus disproportionately on large-magnitude types. I therefore apply a simple per-type target standardization during training and invert it at prediction time (same model, same loop, just a scale normalization that aligns optimization across types). I also clip non-finite/absurd predictions to a safe range based on each type’s training distribution to reduce occasional outlier penalties without changing the modeling approach.'
- What this solution (achieved 1.31417) has done: 'I fix the failing shielding-tensor merge by removing the incorrect intermediate `add_prefix(...)[...]` selection that references non-existent column names, and build `sh0`/`sh1` in one consistent step before setting the MultiIndex. I also make the join logic in `add_structure_features()` robust by joining directly on the `["molecule_name","atom_index_*"]` columns (instead of passing a MultiIndex to `on=`), which avoids pandas version edge-cases. These changes are execution-blocking bug fixes and are score-neutral (they preserve the same features and model). Finally, I keep the submission-writing unchanged and ensure `submission` exists so the later preview cells run.'
- What this solution (achieved 1.29311) has done: 'Your current score is still far from the target (lower-is-better), so we should improve it while keeping the same per-type Ridge pipeline and the same feature engineering. The smallest high-impact fix is to align training with the competition metric: minimize MAE rather than MSE by switching Ridge to `SGDRegressor(loss="epsilon_insensitive")`, while keeping the same preprocessing pipeline (median impute + standardize) and the same per-type loop and target standardization/inversion. I also remove the very aggressive per-type prediction clipping (0.1%/99.9%), replacing it with a much milder 0.01%/99.99% to avoid harming legitimate tails that matter for MAE on some coupling types. These are minimal semantic changes that typically reduce MAE-type errors without changing the overall approach or breaking submission format.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

DATA_DIR = "/kaggle/data/champs-scalar-coupling"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
struct_path = os.path.join(DATA_DIR, "structures.csv")

mulliken_path = os.path.join(DATA_DIR, "mulliken_charges.csv")
shield_path = os.path.join(DATA_DIR, "magnetic_shielding_tensors.csv")
dipole_path = os.path.join(DATA_DIR, "dipole_moments.csv")
energy_path = os.path.join(DATA_DIR, "potential_energy.csv")

try:
    import pyarrow  # noqa: F401

    _csv_engine = "pyarrow"
except Exception:
    _csv_engine = "c"

np.random.seed(0)

train = pd.read_csv(
    train_path,
    engine=_csv_engine,
    dtype={
        "id": "int64",
        "molecule_name": "string",
        "atom_index_0": "int16",
        "atom_index_1": "int16",
        "type": "category",
        "scalar_coupling_constant": "float32",
    },
)
test = pd.read_csv(
    test_path,
    engine=_csv_engine,
    dtype={
        "id": "int64",
        "molecule_name": "string",
        "atom_index_0": "int16",
        "atom_index_1": "int16",
        "type": "category",
    },
)
structures = pd.read_csv(
    struct_path,
    engine=_csv_engine,
    dtype={
        "molecule_name": "string",
        "atom_index": "int16",
        "atom": "category",
        "x": "float32",
        "y": "float32",
        "z": "float32",
    },
)

mulliken = pd.read_csv(
    mulliken_path,
    engine=_csv_engine,
    dtype={
        "molecule_name": "string",
        "atom_index": "int16",
        "mulliken_charge": "float32",
    },
)
shield = pd.read_csv(
    shield_path,
    engine=_csv_engine,
    dtype={
        "molecule_name": "string",
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
dipole = pd.read_csv(
    dipole_path,
    engine=_csv_engine,
    dtype={"molecule_name": "string", "X": "float32", "Y": "float32", "Z": "float32"},
)
energy = pd.read_csv(
    energy_path,
    engine=_csv_engine,
    dtype={"molecule_name": "string", "potential_energy": "float32"},
)

s0 = (
    structures.rename(
        columns={
            "atom_index": "atom_index_0",
            "atom": "atom_0",
            "x": "x0",
            "y": "y0",
            "z": "z0",
        }
    )[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]]
    .set_index(["molecule_name", "atom_index_0"])
    .sort_index()
)
s1 = (
    structures.rename(
        columns={
            "atom_index": "atom_index_1",
            "atom": "atom_1",
            "x": "x1",
            "y": "y1",
            "z": "z1",
        }
    )[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]]
    .set_index(["molecule_name", "atom_index_1"])
    .sort_index()
)
m0 = (
    mulliken.rename(
        columns={"atom_index": "atom_index_0", "mulliken_charge": "mulliken_0"}
    )[["molecule_name", "atom_index_0", "mulliken_0"]]
    .set_index(["molecule_name", "atom_index_0"])
    .sort_index()
)
m1 = (
    mulliken.rename(
        columns={"atom_index": "atom_index_1", "mulliken_charge": "mulliken_1"}
    )[["molecule_name", "atom_index_1", "mulliken_1"]]
    .set_index(["molecule_name", "atom_index_1"])
    .sort_index()
)

sh0 = (
    shield.rename(columns={"atom_index": "atom_index_0"})
    .add_prefix("sh0_")
    .rename(
        columns={
            "sh0_molecule_name": "molecule_name",
            "sh0_atom_index_0": "atom_index_0",
        }
    )
    .set_index(["molecule_name", "atom_index_0"])
    .sort_index()
)
sh1 = (
    shield.rename(columns={"atom_index": "atom_index_1"})
    .add_prefix("sh1_")
    .rename(
        columns={
            "sh1_molecule_name": "molecule_name",
            "sh1_atom_index_1": "atom_index_1",
        }
    )
    .set_index(["molecule_name", "atom_index_1"])
    .sort_index()
)

dipole_i = dipole.set_index("molecule_name").sort_index()
energy_i = energy.set_index("molecule_name").sort_index()


def _gather_by_multiindex(table: pd.DataFrame, key_mi: pd.MultiIndex) -> pd.DataFrame:
    if table.shape[1] == 0:
        return pd.DataFrame(index=np.arange(len(key_mi)))
    idx = table.index.get_indexer(key_mi)

    out_arr = np.empty((len(key_mi), table.shape[1]), dtype=np.float32)
    out_arr.fill(np.nan)

    mask = idx >= 0
    if mask.any():
        out_arr[mask] = table.iloc[idx[mask]].to_numpy(copy=False)

    return pd.DataFrame(out_arr, columns=table.columns)


def _gather_by_index(table: pd.DataFrame, keys: pd.Series) -> pd.DataFrame:
    if table.shape[1] == 0:
        return pd.DataFrame(index=np.arange(len(keys)))
    idx = table.index.get_indexer(keys)

    out_arr = np.empty((len(keys), table.shape[1]), dtype=np.float32)
    out_arr.fill(np.nan)

    mask = idx >= 0
    if mask.any():
        out_arr[mask] = table.iloc[idx[mask]].to_numpy(copy=False)

    return pd.DataFrame(out_arr, columns=table.columns)


def add_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    key0 = pd.MultiIndex.from_frame(
        out[["molecule_name", "atom_index_0"]], names=["molecule_name", "atom_index_0"]
    )
    key1 = pd.MultiIndex.from_frame(
        out[["molecule_name", "atom_index_1"]], names=["molecule_name", "atom_index_1"]
    )

    a0 = _gather_by_multiindex(s0, key0)
    a1 = _gather_by_multiindex(s1, key1)
    c0 = _gather_by_multiindex(m0, key0)
    c1 = _gather_by_multiindex(m1, key1)
    t0 = _gather_by_multiindex(sh0, key0)
    t1 = _gather_by_multiindex(sh1, key1)

    d = _gather_by_index(dipole_i, out["molecule_name"])
    e = _gather_by_index(energy_i, out["molecule_name"])

    out = pd.concat([out.reset_index(drop=True), a0, a1, c0, c1, t0, t1, d, e], axis=1)

    x0 = out["x0"].to_numpy(dtype=np.float32, copy=False)
    y0 = out["y0"].to_numpy(dtype=np.float32, copy=False)
    z0 = out["z0"].to_numpy(dtype=np.float32, copy=False)
    x1 = out["x1"].to_numpy(dtype=np.float32, copy=False)
    y1 = out["y1"].to_numpy(dtype=np.float32, copy=False)
    z1 = out["z1"].to_numpy(dtype=np.float32, copy=False)

    dx = x0 - x1
    dy = y0 - y1
    dz = z0 - z1
    out["dx"] = dx
    out["dy"] = dy
    out["dz"] = dz
    out["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)

    eps = np.float32(1e-6)
    dist = out["dist"].to_numpy(dtype=np.float32, copy=False)
    out["dist2"] = (dist * dist).astype(np.float32)
    out["inv_dist"] = (1.0 / (dist + eps)).astype(np.float32)

    out["x0_plus_x1"] = (x0 + x1).astype(np.float32)
    out["y0_plus_y1"] = (y0 + y1).astype(np.float32)
    out["z0_plus_z1"] = (z0 + z1).astype(np.float32)

    out["x0_minus_x1"] = dx
    out["y0_minus_y1"] = dy
    out["z0_minus_z1"] = dz

    out["mulliken_diff"] = (
        out["mulliken_0"].to_numpy(dtype=np.float32, copy=False)
        - out["mulliken_1"].to_numpy(dtype=np.float32, copy=False)
    ).astype(np.float32)
    return out


train_f = add_structure_features(train)
test_f = add_structure_features(test)


def make_swapped_from_features(df_f: pd.DataFrame) -> pd.DataFrame:
    swapped = df_f.copy()

    swapped[["atom_index_0", "atom_index_1"]] = swapped[
        ["atom_index_1", "atom_index_0"]
    ].to_numpy()
    swapped[["atom_0", "atom_1"]] = swapped[["atom_1", "atom_0"]].to_numpy()
    swapped[["x0", "y0", "z0", "x1", "y1", "z1"]] = swapped[
        ["x1", "y1", "z1", "x0", "y0", "z0"]
    ].to_numpy()
    swapped[["mulliken_0", "mulliken_1"]] = swapped[
        ["mulliken_1", "mulliken_0"]
    ].to_numpy()

    sh0_cols = [c for c in swapped.columns if c.startswith("sh0_")]
    sh1_cols = [c for c in swapped.columns if c.startswith("sh1_")]
    if sh0_cols and sh1_cols and (len(sh0_cols) == len(sh1_cols)):
        tmp = swapped[sh0_cols].to_numpy(copy=True)
        swapped[sh0_cols] = swapped[sh1_cols].to_numpy(copy=False)
        swapped[sh1_cols] = tmp

    x0 = swapped["x0"].to_numpy(dtype=np.float32, copy=False)
    y0 = swapped["y0"].to_numpy(dtype=np.float32, copy=False)
    z0 = swapped["z0"].to_numpy(dtype=np.float32, copy=False)
    x1 = swapped["x1"].to_numpy(dtype=np.float32, copy=False)
    y1 = swapped["y1"].to_numpy(dtype=np.float32, copy=False)
    z1 = swapped["z1"].to_numpy(dtype=np.float32, copy=False)

    dx = x0 - x1
    dy = y0 - y1
    dz = z0 - z1
    swapped["dx"] = dx
    swapped["dy"] = dy
    swapped["dz"] = dz
    swapped["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)

    eps = np.float32(1e-6)
    dist = swapped["dist"].to_numpy(dtype=np.float32, copy=False)
    swapped["dist2"] = (dist * dist).astype(np.float32)
    swapped["inv_dist"] = (1.0 / (dist + eps)).astype(np.float32)

    swapped["x0_plus_x1"] = (x0 + x1).astype(np.float32)
    swapped["y0_plus_y1"] = (y0 + y1).astype(np.float32)
    swapped["z0_plus_z1"] = (z0 + z1).astype(np.float32)
    swapped["x0_minus_x1"] = dx
    swapped["y0_minus_y1"] = dy
    swapped["z0_minus_z1"] = dz

    swapped["mulliken_diff"] = (
        swapped["mulliken_0"].to_numpy(dtype=np.float32, copy=False)
        - swapped["mulliken_1"].to_numpy(dtype=np.float32, copy=False)
    ).astype(np.float32)
    return swapped


train_f_swapped = make_swapped_from_features(train_f)
train_f = pd.concat([train_f, train_f_swapped], axis=0, ignore_index=True)

for col in ["atom_0", "atom_1"]:
    train_f[col] = train_f[col].astype("category")
    test_f[col] = test_f[col].astype("category")

all_atoms = pd.concat(
    [train_f[["atom_0", "atom_1"]], test_f[["atom_0", "atom_1"]]], axis=0
)
atom0_cats = all_atoms["atom_0"].astype("category").cat.categories
atom1_cats = all_atoms["atom_1"].astype("category").cat.categories
train_f["atom_0"] = pd.Categorical(train_f["atom_0"], categories=atom0_cats)
test_f["atom_0"] = pd.Categorical(test_f["atom_0"], categories=atom0_cats)
train_f["atom_1"] = pd.Categorical(train_f["atom_1"], categories=atom1_cats)
test_f["atom_1"] = pd.Categorical(test_f["atom_1"], categories=atom1_cats)

train_atom0_code = train_f["atom_0"].cat.codes.astype(np.int16)
train_atom1_code = train_f["atom_1"].cat.codes.astype(np.int16)
test_atom0_code = test_f["atom_0"].cat.codes.astype(np.int16)
test_atom1_code = test_f["atom_1"].cat.codes.astype(np.int16)

shield_cols0 = [c for c in train_f.columns if c.startswith("sh0_")]
shield_cols1 = [c for c in train_f.columns if c.startswith("sh1_")]

num_features = (
    [
        "dx",
        "dy",
        "dz",
        "dist",
        "dist2",
        "inv_dist",
        "x0",
        "y0",
        "z0",
        "x1",
        "y1",
        "z1",
        "x0_plus_x1",
        "y0_plus_y1",
        "z0_plus_z1",
        "x0_minus_x1",
        "y0_minus_y1",
        "z0_minus_z1",
        "mulliken_0",
        "mulliken_1",
        "mulliken_diff",
        "X",
        "Y",
        "Z",
        "potential_energy",
    ]
    + shield_cols0
    + shield_cols1
)

X_train_num = train_f[num_features].astype("float32")
X_test_num = test_f[num_features].astype("float32")

X_train_full = np.ascontiguousarray(
    np.column_stack(
        [
            X_train_num.to_numpy(dtype=np.float32, copy=False),
            train_atom0_code.to_numpy(dtype=np.float32, copy=False),
            train_atom1_code.to_numpy(dtype=np.float32, copy=False),
        ]
    ),
    dtype=np.float32,
)
X_test_full = np.ascontiguousarray(
    np.column_stack(
        [
            X_test_num.to_numpy(dtype=np.float32, copy=False),
            test_atom0_code.to_numpy(dtype=np.float32, copy=False),
            test_atom1_code.to_numpy(dtype=np.float32, copy=False),
        ]
    ),
    dtype=np.float32,
)

X_train_mat = X_train_full
X_test_mat = X_test_full

y = train_f["scalar_coupling_constant"].astype("float32").to_numpy()
train_type = train_f["type"].astype("category")
test_type = test_f["type"].astype("category")


def make_model(alpha: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True)),
            (
                "sgd",
                SGDRegressor(
                    loss="huber",
                    epsilon=0.1,
                    penalty="l2",
                    alpha=alpha,
                    fit_intercept=True,
                    max_iter=2000,
                    tol=1e-3,
                    random_state=0,
                    n_jobs=-1,
                ),
            ),
        ]
    )


pred = np.zeros(len(test_f), dtype=np.float32)
global_mean = float(np.mean(y))

train_type_codes = train_type.cat.codes.to_numpy()
test_type_codes = test_type.cat.codes.to_numpy()
test_code_unique = np.unique(test_type_codes)

rng = np.random.RandomState(0)
alpha_grid = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4)

for code in test_code_unique:
    idx_test = np.flatnonzero(test_type_codes == code)
    idx_train_all = np.flatnonzero(train_type_codes == code)
    if len(idx_train_all) == 0:
        pred[idx_test] = global_mean
        continue

    y_t_all = y[idx_train_all]
    mu = float(np.mean(y_t_all))
    sigma = float(np.std(y_t_all) + 1e-6)

    n = len(idx_train_all)
    if n >= 5000:
        perm = rng.permutation(n)
        n_val = max(2000, int(0.05 * n))
        val_local = perm[:n_val]
        tr_local = perm[n_val:]
        idx_val = idx_train_all[val_local]
        idx_tr = idx_train_all[tr_local]

        y_tr = y[idx_tr]
        y_val = y[idx_val]
        y_tr_std = (y_tr - mu) / sigma

        X_tr = np.take(X_train_mat, idx_tr, axis=0)
        X_val = np.take(X_train_mat, idx_val, axis=0)

        best_alpha = None
        best_mae = np.inf
        for a in alpha_grid:
            model = make_model(alpha=a)
            model.fit(X_tr, y_tr_std)
            p_val_std = model.predict(X_val).astype(np.float32)
            p_val = p_val_std * sigma + mu
            mae = float(np.mean(np.abs(p_val - y_val)))
            if mae < best_mae:
                best_mae = mae
                best_alpha = a
        alpha_use = float(best_alpha)
    else:
        alpha_use = 1e-5  # stable default for small types

    y_t_std = (y_t_all - mu) / sigma
    model = make_model(alpha=alpha_use)

    X_train_t = np.take(X_train_mat, idx_train_all, axis=0)
    model.fit(X_train_t, y_t_std)

    X_test_t = np.take(X_test_mat, idx_test, axis=0)
    p_std = model.predict(X_test_t).astype(np.float32)
    p = p_std * sigma + mu

    lo = float(np.quantile(y_t_all, 0.0001))
    hi = float(np.quantile(y_t_all, 0.9999))
    p = np.clip(p, lo, hi)

    pred[idx_test] = p.astype(np.float32)

pred = np.where(np.isfinite(pred), pred, global_mean).astype(np.float32)

submission = pd.DataFrame(
    {"id": test["id"].to_numpy(), "scalar_coupling_constant": pred}
)
submission["id"] = submission["id"].astype(np.int64)
submission = submission.sort_values("id").reset_index(drop=True)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(submission), "cols:", list(submission.columns))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_12/2423608428.py in <cell line: 0>()
    261 
    262 
--> 263 train_f = add_structure_features(train)
    264 test_f = add_structure_features(test)
    265 

/tmp/ipykernel_12/2423608428.py in add_structure_features(df)
    214     )
    215 
--> 216     a0 = _gather_by_multiindex(s0, key0)
    217     a1 = _gather_by_multiindex(s1, key1)
    218     c0 = _gather_by_multiindex(m0, key0)

/tmp/ipykernel_12/2423608428.py in _gather_by_multiindex(table, key_mi)
    184     mask = idx >= 0
    185     if mask.any():
--> 186         out_arr[mask] = table.iloc[idx[mask]].to_numpy(copy=False)
    187 
    188     return pd.DataFrame(out_arr, columns=table.columns)

ValueError: could not convert string to float: 'H'

## === cell 1
if "submission" in globals():
    print(submission.head(20))
else:
    print("submission is not defined (upstream cell failed).")



## === cell 2
if "submission" in globals():
    print(submission.describe())
else:
    print("submission is not defined (upstream cell failed).")
