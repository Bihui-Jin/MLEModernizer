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

-1.3517984281028244

# 6. Current score

4.5078

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.99777) has done: 'Your notebook is trying to ensemble (“stack”) external submissions from `../input/top-mol` and several other `../input/...` folders that do not exist in this environment, causing the early `FileNotFoundError` and cascading `NameError`s. To keep the core “stacking/aggregation” intent while making it run end-to-end, I switch the input source to the available `sample_submission.csv` and generate a simple, deterministic baseline prediction (all zeros) in the exact required submission format. I also remove notebook-only magic (`%matplotlib inline`) so it runs as a plain script, and guard any plotting/correlation code so it won’t break execution. This reliably create a valid `.csv` submission file (though the score not be competitive without actual model predictions).'
- What this solution (achieved 1.99777) has done: 'Your current code submits all zeros, which is why the score is far from the (much lower/better) target; we need a small, legitimate model-based prediction while staying simple and deterministic. I keep the pipeline lightweight by training a separate regularized linear model per coupling `type` using only minimal geometric features computed from `structures.csv` (distance and coordinate deltas between the two atoms). This preserves the “no deep architecture/training loop” simplicity, runs within the time limit, and should substantially reduce MAE vs zeros, moving the score toward the target band. I also ensure strict alignment by predicting in the original test row order and writing a valid `submission.csv`.'
- What this solution (achieved 2.85744) has done: 'The crash is because some merged structure coordinates are missing, which makes the geometric features (`dx/dy/dz/dist/inv_dist`) become NaN/inf and Ridge refuses to predict with NaNs. I fix this with minimal, score-neutral preprocessing: compute features, then deterministically impute any non-finite feature values using train-set medians (and apply the same to test). I also make the `one_hot` routine robust to missing atom labels without changing the modeling approach. This keeps the per-type Ridge training core logic intact while allowing the notebook to run end-to-end and generate a valid `submission.csv`.'
- What this solution (achieved 2.82349) has done: 'Your current per-type Ridge baseline is sound but it’s leaving a lot of signal unused; the biggest safe gain (without changing the learning approach) is to add a few more physically meaningful, cheap geometric features derived from the same merged coordinates (squared distance, absolute deltas, and unit direction components). This keeps the exact same training loop (one Ridge per type) and still uses the same data sources, but gives the linear model enough feature richness to reduce MAE materially toward your target. I also standardize numeric features using train statistics (applied to test) to make the single global `alpha=1.0` behave more consistently across feature scales; this is a minimal preprocessing change that typically improves Ridge stability. Finally, I keep the existing non-finite imputation to ensure the run completes and produces a valid `submission.csv`.'
- What this solution (achieved 3.42157) has done: 'You’re far worse than the target (lower is better), so we should legitimately improve accuracy with minimal disruption to your current per-type Ridge setup. The biggest safe gain without changing the learning approach is to add a few more “cheap but strong” geometric features (dot products and higher-order distance transforms) and to include coupling `type` as an explicit one-hot feature (while still training per-type, it helps stabilize intercept/feature space in edge cases). I also add a tiny, deterministic per-type alpha selection from a very small fixed grid using a molecule-grouped validation split; this keeps the same model family/training loop (Ridge per type) but avoids a clearly suboptimal global `alpha=1.0`. All changes are deterministic, keep the same data sources, and still write a valid `submission.csv`.'
- What this solution (achieved 3.01362) has done: 'Your current Ridge-per-type pipeline is valid but is likely underperforming because it ignores strong, “free” signals available from the provided auxiliary CSVs. I keep the exact same core approach (same per-type Ridge training loop and objective), and only add a minimal set of extra features by left-merging `mulliken_charges.csv` (per-atom) and `magnetic_shielding_tensors.csv` (per-atom) for both atoms in the pair. I reuse your existing non-finite imputation + standardization so the new numeric features are safe and stable, and keep the same deterministic per-type alpha selection. This should materially reduce MAE (thus log-MAE) and move your score closer to the target without changing the modeling family.'
- What this solution (achieved 3.01016) has done: 'Your current score is much worse than the (lower-is-better) target, so we should improve legitimate predictive signal while keeping the same per-type Ridge setup. The biggest minimal issue is that you’re fitting Ridge without an intercept after globally standardizing features; this can materially hurt calibration per type, so I enable `fit_intercept=True` consistently in both validation and final fits. Next, to add strong “free” molecule-level information without changing the model family or loop, I left-merge `dipole_moments.csv` and `potential_energy.csv` and add a few simple derived features (norm and per-axis components, plus per-type centering via Ridge intercept). All changes keep the same data sources, deterministic behavior, and still write a valid `submission.csv`.'
- What this solution (achieved 1.99984) has done: 'Your current per-type Ridge is sound but still far from the (lower-is-better) target, so the smallest legitimate accuracy gain is to add a couple of strong, cheap domain features without changing the model family or training loop. Specifically, I merge `scalar_coupling_contributions.csv` for the training rows only and add its four components (fc/sd/pso/dso) plus their sum; this is not label leakage because these are given per training sample and are unavailable for test, and we set them to 0 for test so the model can still run. I also add the well-known CHAMPS feature `1/dist` and `1/dist^2`-scaled charge interactions (`mulliken_0*mulliken_1*inv_dist`, etc.), which are simple transforms of existing features and often reduce MAE. Everything else (per-type Ridge, molecule-based validation for alpha grid, standardization/imputation, submission alignment) is kept intact.'
- What this solution (achieved 3.39374) has done: 'Your current score (1.99984, lower-is-better) is far worse than the target (-1.3518), so we should legitimately improve predictive accuracy with the smallest changes that keep your per-type Ridge setup intact. The biggest issue hurting generalization is that `fc/sd/pso/dso` are merged for train only and then set to 0 for test, creating a train/test feature distribution mismatch; I keep the same feature set but instead train on a “clean” feature matrix that excludes these contribution columns (and their sum) so train/test are consistent. I also make the molecule-based validation split slightly more representative by hashing molecules (still deterministic) rather than taking the first sorted 10%, which reduces systematic bias without changing the training approach. Everything else (merges, geometric features, standardization/imputation, per-type Ridge + small alpha grid, submission alignment) remains the same and still writes a valid `submission.csv`.'
- What this solution (achieved 16.9374) has done: 'Your current score is far worse than the (lower-is-better) target, so we should improve accuracy with minimal disruption to your existing per-type Ridge pipeline. The biggest safe gain (without changing the model family or loop) is to fix a likely evaluation mismatch: the competition metric is MAE on `scalar_coupling_constant`, but it’s common to model each type in log-space and then invert, which often reduces MAE because targets are heavy-tailed and type-dependent. I keep the same per-type Ridge training, alpha grid search, merges, and feature construction, but train/predict on a signed log1p-transformed target per type (and then invert back) to improve robustness. I also ensure we don’t use the unavailable `fc/sd/pso/dso` columns at all (they currently still get created and used indirectly via `contrib_sum`), keeping train/test feature distributions consistent.'
- What this solution (achieved 3.39374) has done: 'Your current score is dramatically worse than the (lower-is-better) target, and the most likely cause is a train/test evaluation mismatch introduced by the signed log1p target transform: it changes the optimization objective away from MAE on the original scale, which can badly hurt this competition’s log-MAE metric. I make the smallest core-logic-preserving change by removing the target transform and training/predicting Ridge directly on the original target per type, while keeping the exact same features, per-type loop, molecule-hash validation split, and alpha grid selection. This should move the score substantially down toward the target without changing architecture, data sources, or submission semantics. I also keep all existing imputation/standardization and alignment checks unchanged to preserve stability.'
- What this solution (achieved 4.5078) has done: 'Your current score is far worse than the (lower-is-better) target, so we should improve accuracy with the smallest possible change that keeps your per-type Ridge approach intact. The biggest bug-like issue is that you’re standardizing *all* numeric features globally across all coupling types; since feature scales and distributions differ by type, this can hurt per-type calibration and Ridge alpha selection. I keep the exact same features, per-type Ridge loop, and alpha grid, but compute numeric standardization (mean/std) **per type** using that type’s training rows, then apply it to both train/test rows of that type. This preserves the same core modeling logic while typically reducing MAE substantially for CHAMPS baselines.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge

BASE_INPUT = "/kaggle/data/champs-scalar-coupling"
TRAIN_PATH = os.path.join(BASE_INPUT, "train.csv")
TEST_PATH = os.path.join(BASE_INPUT, "test.csv")
STRUCT_PATH = os.path.join(BASE_INPUT, "structures.csv")
MULLIKEN_PATH = os.path.join(BASE_INPUT, "mulliken_charges.csv")
MAGTENSOR_PATH = os.path.join(BASE_INPUT, "magnetic_shielding_tensors.csv")
DIPOLE_PATH = os.path.join(BASE_INPUT, "dipole_moments.csv")
POTENTIAL_PATH = os.path.join(BASE_INPUT, "potential_energy.csv")
CONTRIB_PATH = os.path.join(BASE_INPUT, "scalar_coupling_contributions.csv")

print("Found train:", os.path.exists(TRAIN_PATH))
print("Found test:", os.path.exists(TEST_PATH))
print("Found structures:", os.path.exists(STRUCT_PATH))
print("Found mulliken:", os.path.exists(MULLIKEN_PATH))
print("Found mag_tensor:", os.path.exists(MAGTENSOR_PATH))
print("Found dipole:", os.path.exists(DIPOLE_PATH))
print("Found potential:", os.path.exists(POTENTIAL_PATH))
print("Found contributions:", os.path.exists(CONTRIB_PATH))



## === cell 1
train = pd.read_csv(
    TRAIN_PATH,
    usecols=[
        "id",
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "scalar_coupling_constant",
    ],
)
test = pd.read_csv(
    TEST_PATH,
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
)

print("train:", train.shape, "test:", test.shape)
print("train types:", train["type"].nunique(), "test types:", test["type"].nunique())



## === cell 2
structures = pd.read_csv(
    STRUCT_PATH,
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

train_m = train.merge(s0, on=["molecule_name", "atom_index_0"], how="left").merge(
    s1, on=["molecule_name", "atom_index_1"], how="left"
)
test_m = test.merge(s0, on=["molecule_name", "atom_index_0"], how="left").merge(
    s1, on=["molecule_name", "atom_index_1"], how="left"
)

for name, df in [("train_m", train_m), ("test_m", test_m)]:
    missing = df[["x0", "y0", "z0", "x1", "y1", "z1"]].isna().any(axis=1).mean()
    print(name, "missing_coord_frac:", float(missing))

mulliken = pd.read_csv(
    MULLIKEN_PATH, usecols=["molecule_name", "atom_index", "mulliken_charge"]
)
m0 = mulliken.rename(
    columns={"atom_index": "atom_index_0", "mulliken_charge": "mulliken_0"}
)
m1 = mulliken.rename(
    columns={"atom_index": "atom_index_1", "mulliken_charge": "mulliken_1"}
)

mag = pd.read_csv(
    MAGTENSOR_PATH,
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
g0 = mag.rename(
    columns={
        "atom_index": "atom_index_0",
        "XX": "gXX_0",
        "YX": "gYX_0",
        "ZX": "gZX_0",
        "XY": "gXY_0",
        "YY": "gYY_0",
        "ZY": "gZY_0",
        "XZ": "gXZ_0",
        "YZ": "gYZ_0",
        "ZZ": "gZZ_0",
    }
)
g1 = mag.rename(
    columns={
        "atom_index": "atom_index_1",
        "XX": "gXX_1",
        "YX": "gYX_1",
        "ZX": "gZX_1",
        "XY": "gXY_1",
        "YY": "gYY_1",
        "ZY": "gZY_1",
        "XZ": "gXZ_1",
        "YZ": "gYZ_1",
        "ZZ": "gZZ_1",
    }
)

train_m = train_m.merge(m0, on=["molecule_name", "atom_index_0"], how="left").merge(
    m1, on=["molecule_name", "atom_index_1"], how="left"
)
test_m = test_m.merge(m0, on=["molecule_name", "atom_index_0"], how="left").merge(
    m1, on=["molecule_name", "atom_index_1"], how="left"
)

train_m = train_m.merge(g0, on=["molecule_name", "atom_index_0"], how="left").merge(
    g1, on=["molecule_name", "atom_index_1"], how="left"
)
test_m = test_m.merge(g0, on=["molecule_name", "atom_index_0"], how="left").merge(
    g1, on=["molecule_name", "atom_index_1"], how="left"
)

dip = pd.read_csv(DIPOLE_PATH, usecols=["molecule_name", "X", "Y", "Z"]).rename(
    columns={"X": "dipX", "Y": "dipY", "Z": "dipZ"}
)
pot = pd.read_csv(POTENTIAL_PATH, usecols=["molecule_name", "potential_energy"]).rename(
    columns={"potential_energy": "potential_energy"}
)

train_m = train_m.merge(dip, on="molecule_name", how="left").merge(
    pot, on="molecule_name", how="left"
)
test_m = test_m.merge(dip, on="molecule_name", how="left").merge(
    pot, on="molecule_name", how="left"
)

for c in ["fc", "sd", "pso", "dso"]:
    if c in train_m.columns:
        train_m.drop(columns=[c], inplace=True)
    if c in test_m.columns:
        test_m.drop(columns=[c], inplace=True)

extra_cols = (
    ["mulliken_0", "mulliken_1"]
    + ["gXX_0", "gYX_0", "gZX_0", "gXY_0", "gYY_0", "gZY_0", "gXZ_0", "gYZ_0", "gZZ_0"]
    + ["gXX_1", "gYX_1", "gZX_1", "gXY_1", "gYY_1", "gZY_1", "gXZ_1", "gYZ_1", "gZZ_1"]
    + ["dipX", "dipY", "dipZ", "potential_energy"]
)
for name, df in [("train_m", train_m), ("test_m", test_m)]:
    miss = df[extra_cols].isna().mean().sort_values(ascending=False).head(10)
    print(name, "top missing extra feature fracs:\n", miss)




## === cell 3
def add_geom_features(df: pd.DataFrame) -> pd.DataFrame:
    x0 = df["x0"].values
    y0 = df["y0"].values
    z0 = df["z0"].values
    x1 = df["x1"].values
    y1 = df["y1"].values
    z1 = df["z1"].values

    dx = x0 - x1
    dy = y0 - y1
    dz = z0 - z1

    dist2 = dx * dx + dy * dy + dz * dz
    dist = np.sqrt(dist2)
    inv_dist = 1.0 / (dist + 1e-6)

    out = df.copy()
    out["dx"] = dx
    out["dy"] = dy
    out["dz"] = dz
    out["dist2"] = dist2
    out["dist"] = dist
    out["inv_dist"] = inv_dist

    out["abs_dx"] = np.abs(dx)
    out["abs_dy"] = np.abs(dy)
    out["abs_dz"] = np.abs(dz)

    out["ux"] = dx * inv_dist
    out["uy"] = dy * inv_dist
    out["uz"] = dz * inv_dist

    out["inv_dist2"] = 1.0 / (dist2 + 1e-6)
    out["inv_dist3"] = out["inv_dist2"] * inv_dist  # ~ 1/r^3
    out["dist3"] = dist2 * dist
    out["dist4"] = dist2 * dist2

    out["dx_dy"] = dx * dy
    out["dx_dz"] = dx * dz
    out["dy_dz"] = dy * dz

    out["mulliken_sum"] = out["mulliken_0"].values + out["mulliken_1"].values
    out["mulliken_diff"] = out["mulliken_0"].values - out["mulliken_1"].values

    for comp in ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]:
        a0 = out[f"g{comp}_0"].values
        a1 = out[f"g{comp}_1"].values
        out[f"g{comp}_sum"] = a0 + a1
        out[f"g{comp}_diff"] = a0 - a1

    dipx = out["dipX"].values
    dipy = out["dipY"].values
    dipz = out["dipZ"].values
    out["dip_norm"] = np.sqrt(dipx * dipx + dipy * dipy + dipz * dipz)

    q0 = out["mulliken_0"].values
    q1 = out["mulliken_1"].values
    out["qprod"] = q0 * q1
    out["qprod_inv_dist"] = out["qprod"].values * inv_dist
    out["qprod_inv_dist2"] = out["qprod"].values * out["inv_dist2"].values
    out["qsum_inv_dist"] = out["mulliken_sum"].values * inv_dist
    out["qdiff_inv_dist"] = out["mulliken_diff"].values * inv_dist

    return out


train_m = add_geom_features(train_m)
test_m = add_geom_features(test_m)

base_num_feats = [
    "dx",
    "dy",
    "dz",
    "abs_dx",
    "abs_dy",
    "abs_dz",
    "dist2",
    "dist",
    "dist3",
    "dist4",
    "inv_dist",
    "inv_dist2",
    "inv_dist3",
    "ux",
    "uy",
    "uz",
    "dx_dy",
    "dx_dz",
    "dy_dz",
]

extra_num_feats = (
    ["mulliken_0", "mulliken_1", "mulliken_sum", "mulliken_diff"]
    + ["gXX_0", "gYX_0", "gZX_0", "gXY_0", "gYY_0", "gZY_0", "gXZ_0", "gYZ_0", "gZZ_0"]
    + ["gXX_1", "gYX_1", "gZX_1", "gXY_1", "gYY_1", "gZY_1", "gXZ_1", "gYZ_1", "gZZ_1"]
    + [
        f"g{comp}_sum"
        for comp in ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]
    ]
    + [
        f"g{comp}_diff"
        for comp in ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]
    ]
    + ["dipX", "dipY", "dipZ", "dip_norm", "potential_energy"]
    + ["qprod", "qprod_inv_dist", "qprod_inv_dist2", "qsum_inv_dist", "qdiff_inv_dist"]
)

num_feats = base_num_feats + extra_num_feats


def impute_nonfinite_with_train_median(train_df, test_df, cols):
    train_imp = train_df.copy()
    test_imp = test_df.copy()

    medians = {}
    for c in cols:
        tr_vals = train_imp[c].astype(np.float64).values
        tr_vals = np.where(np.isfinite(tr_vals), tr_vals, np.nan)
        med = float(np.nanmedian(tr_vals))
        if not np.isfinite(med):
            med = 0.0
        medians[c] = med

        for d in (train_imp, test_imp):
            vals = d[c].astype(np.float64).values
            bad = ~np.isfinite(vals)
            if bad.any():
                vals[bad] = med
                d[c] = vals
    return train_imp, test_imp, medians


train_m, test_m, num_medians = impute_nonfinite_with_train_median(
    train_m, test_m, num_feats
)
print(
    "Numeric feature medians used for imputation (first 12):",
    {k: round(num_medians[k], 6) for k in list(num_medians.keys())[:12]},
)

cats = pd.concat([train_m[["atom_0", "atom_1"]], test_m[["atom_0", "atom_1"]]], axis=0)
atom0_levels = sorted(cats["atom_0"].dropna().unique().tolist())
atom1_levels = sorted(cats["atom_1"].dropna().unique().tolist())

type_levels = sorted(
    pd.concat([train_m["type"], test_m["type"]], axis=0).dropna().unique().tolist()
)


def one_hot(series: pd.Series, levels):
    m = np.zeros((len(series), len(levels)), dtype=np.float32)
    idx = {v: i for i, v in enumerate(levels)}
    vals = series.values
    for r in range(len(vals)):
        v = vals[r]
        if v in idx:
            m[r, idx[v]] = 1.0
    return m


X_atom0_tr = one_hot(train_m["atom_0"], atom0_levels)
X_atom1_tr = one_hot(train_m["atom_1"], atom1_levels)
X_atom0_te = one_hot(test_m["atom_0"], atom0_levels)
X_atom1_te = one_hot(test_m["atom_1"], atom1_levels)

X_type_tr = one_hot(train_m["type"], type_levels)
X_type_te = one_hot(test_m["type"], type_levels)

X_num_tr_raw = train_m[num_feats].astype(np.float32).values
X_num_te_raw = test_m[num_feats].astype(np.float32).values

X_tr_all_raw = np.hstack([X_num_tr_raw, X_atom0_tr, X_atom1_tr, X_type_tr]).astype(
    np.float32
)
X_te_all_raw = np.hstack([X_num_te_raw, X_atom0_te, X_atom1_te, X_type_te]).astype(
    np.float32
)

assert np.isfinite(X_tr_all_raw).all(), "Non-finite values remain in training features."
assert np.isfinite(X_te_all_raw).all(), "Non-finite values remain in test features."

y_raw = train_m["scalar_coupling_constant"].astype(np.float32).values

num_dim = len(num_feats)
print(
    "X_tr_all_raw shape:", X_tr_all_raw.shape, "X_te_all_raw shape:", X_te_all_raw.shape
)
print("Numeric dim:", num_dim, "One-hot dim:", X_tr_all_raw.shape[1] - num_dim)




## === cell 4
def make_molecule_val_mask(molecule_names: np.ndarray, frac: float = 0.1) -> np.ndarray:
    mols = pd.Series(molecule_names).astype(str).values
    h = pd.util.hash_pandas_object(pd.Series(mols), index=False).values
    threshold = np.quantile(h.astype(np.float64), 1.0 - frac)
    return h >= threshold


pred_test = np.zeros(len(test_m), dtype=np.float32)

types = sorted(train_m["type"].unique().tolist())
print("Training per-type models:", len(types))

alpha_grid = [0.05, 0.2, 1.0, 5.0, 20.0]

for t in types:
    tr_idx = train_m["type"].values == t
    te_idx = test_m["type"].values == t

    if not te_idx.any():
        continue

    n_tr = int(tr_idx.sum())
    if n_tr < 50:
        type_mean = (
            float(train_m.loc[tr_idx, "scalar_coupling_constant"].mean())
            if n_tr > 0
            else 0.0
        )
        pred_test[te_idx] = type_mean
        continue

    mol_names_t = train_m.loc[tr_idx, "molecule_name"].values
    val_mask_t = make_molecule_val_mask(mol_names_t, frac=0.1)

    X_t_tr = X_tr_all_raw[tr_idx].astype(np.float32)
    y_t = y_raw[tr_idx].astype(np.float32)

    X_t_te = X_te_all_raw[te_idx].astype(np.float32)

    mu = X_t_tr[:, :num_dim].mean(axis=0, dtype=np.float64)
    sd = X_t_tr[:, :num_dim].std(axis=0, dtype=np.float64)
    sd = np.where(sd > 1e-12, sd, 1.0)

    X_t_tr_std = X_t_tr.copy()
    X_t_te_std = X_t_te.copy()
    X_t_tr_std[:, :num_dim] = ((X_t_tr_std[:, :num_dim] - mu) / sd).astype(np.float32)
    X_t_te_std[:, :num_dim] = ((X_t_te_std[:, :num_dim] - mu) / sd).astype(np.float32)

    if val_mask_t.sum() < 200 or (~val_mask_t).sum() < 200:
        best_alpha = 1.0
    else:
        X_tr_t, y_tr_t = X_t_tr_std[~val_mask_t], y_t[~val_mask_t]
        X_va_t, y_va_t = X_t_tr_std[val_mask_t], y_t[val_mask_t]

        best_alpha = None
        best_mae = None
        for a in alpha_grid:
            m = Ridge(alpha=a, random_state=0, fit_intercept=True)
            m.fit(X_tr_t, y_tr_t)
            p = m.predict(X_va_t).astype(np.float64)
            mae = float(np.mean(np.abs(p - y_va_t.astype(np.float64))))
            if (best_mae is None) or (mae < best_mae):
                best_mae = mae
                best_alpha = a

    model = Ridge(alpha=float(best_alpha), random_state=0, fit_intercept=True)
    model.fit(X_t_tr_std, y_t)
    pred_test[te_idx] = model.predict(X_t_te_std).astype(np.float32)

assert (test_m["id"].values == test["id"].values).all()



## === cell 5
submission = pd.DataFrame(
    {"id": test["id"].values, "scalar_coupling_constant": pred_test.astype(np.float64)}
)

assert submission.shape[0] == test.shape[0]
assert list(submission.columns) == ["id", "scalar_coupling_constant"]
assert submission["id"].isna().sum() == 0

out_path = "submission.csv"
submission.to_csv(out_path, index=False, float_format="%.6f")
print("Wrote:", out_path)
print(submission.describe())
