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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

1.23424

# 6. Current score

2.84748

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 2.17067) has done: 'I fix the merge logic so atom coordinates are joined correctly (by molecule and atom index) instead of creating a cartesian product, which currently makes the dataset explode and is logically wrong. Then I make the one-hot encoding consistent between train and test by fitting on train and reindexing test to the same feature columns, which resolves the feature-name mismatch error. I also correct the input paths to match your environment (`/kaggle/data/champs-scalar-coupling/...`) and ensure the script always writes a valid `submission.csv` with the required columns. These changes preserve the core “linear regression on distance + type dummies” approach while making it run end-to-end and improving the expected score substantially versus the broken merge.'
- What this solution (achieved 27.9907) has done: 'We need to move the score down (lower is better) from 2.17067 toward 1.23424, so we should add a small amount of extra signal without changing the overall “linear regression on engineered features” approach. The most minimal gain in this competition usually comes from (1) adding atom identity dummies for atom_0/atom_1 (and their interaction as a combined category), and (2) adding simple distance transforms (inverse distance and squared distance) while keeping the same LinearRegression model and single fit/predict loop. These additions keep the core logic intact (same data sources, same merge, same model family) but typically reduce MAE per type because different coupling types/atom pairs follow different distance-response curves. I also ensure feature columns are aligned between train/test exactly as before and keep writing a valid `submission.csv`.'
- What this solution (achieved 20.30863) has done: 'We need to move the score down (lower is better) from 27.9907 toward 1.23424, so the smallest legitimate improvement is to add a bit more physics signal while keeping the same “single LinearRegression fit on engineered tabular features” core. I merge in three auxiliary files you already have (mulliken charges, magnetic shielding tensors, dipole moments, plus potential energy) and derive simple pair features (charge sum/diff/product; shielding row-wise stats per atom and pair diffs; dipole magnitude and projections onto the bond vector). These additions are lightweight, preserve the same training approach and model family, and typically improve this competition materially because they correlate with coupling constants across types. I keep the existing distance/type/atom dummies and ensure train/test columns stay perfectly aligned, still writing a valid `submission.csv`.'
- What this solution (achieved 21.86587) has done: 'Your current score is far above the target (20.31 vs 1.234, lower is better), so we need a legitimate performance improvement without changing the core “single LinearRegression on engineered tabular features” approach. The most impactful minimal fix for CHAMPS is to train separate linear models per coupling `type` (because the metric is averaged per type and relationships differ strongly by type) while keeping the same features and model class. I also make the one-hot encoding consistent by fitting dummies on the combined train+test categories (still no label leakage since it’s only categories), which avoids missing-category degradation. Finally, I keep the same submission format and ensure row alignment by predicting into `test_2` by type and writing `submission.csv`.'
- What this solution (achieved 21.86587) has done: 'I keep your per-`type` LinearRegression approach and the same core engineered features, but fix the main reason your score is still very poor: the model is being trained/predicted on raw coupling constants whose scale differs wildly by type, and LinearRegression is very sensitive to feature scaling across the large one-hot + numeric mix. I apply a **per-type target standardization** (fit mean/std on train rows of that type, train on standardized y, then unstandardize predictions), which is a minimal post-processing/calibration change that typically improves log-MAE a lot without changing model family or features. I also add a tiny robustness fix: for any type that is missing in train (shouldn’t happen, but safe), fall back to the global mean. Finally, I keep the submission format identical and still write `submission.csv`.'
- What this solution (achieved 21.03352) has done: 'Your score is far from the target (21.86587 vs 1.23424, lower is better), so we need a real modeling improvement while keeping the same overall “linear regression on engineered tabular features” approach. The biggest issue is that you’re still feeding huge, unscaled one-hot blocks plus numeric features into plain LinearRegression, which is ill-conditioned and typically performs terribly here; switching to Ridge (same linear family, same fit/predict loop) is a minimal, legitimate regularization change that usually drops log-MAE dramatically. I also add a very small, safe per-type numeric feature standardization (fit mean/std on train rows of that type, apply to that type’s train/test rows) to reduce scale mismatch without changing features or training semantics. Finally, I keep your per-type target standardization and submission writing exactly as-is so it still produces a valid `submission.csv`.'
- What this solution (achieved 21.03352) has done: 'To move your log-MAE down toward the 1.234 target without changing the overall “per-type linear model on engineered tabular features” core, I make two minimal, score-relevant adjustments. First, I replace the dense `get_dummies` design matrix with a `ColumnTransformer` + `OneHotEncoder` feeding a `Pipeline(StandardScaler(with_mean=False) -> Ridge)` per type; this keeps the same model family and features, but fixes the ill-conditioning and scaling problems caused by huge dense one-hot blocks (a common reason for very poor CHAMPS scores). Second, I keep your per-type target standardization, but drop the redundant manual numeric scaling (the pipeline handles it), improving stability and usually lowering log-MAE substantially. The script still run end-to-end on the same files and always write a valid `submission.csv`.'
- What this solution (achieved 21.03352) has done: 'Your current score (21.03352, lower is better) is far worse than the target (1.23424), so we need a real but still minimal fix that improves model fit without changing the overall “per-type linear Ridge on engineered tabular features” approach. The biggest bug here is that you reuse the same `Pipeline` (and thus the same `OneHotEncoder` category mapping) across types; after the first `.fit()`, subsequent types encode categories using the first type’s learned categories, badly corrupting features and tanking the score. I create a fresh per-type pipeline inside the loop (same exact components) so each type gets its own correct encoding/scaling. I also set `OneHotEncoder(min_frequency=...)` to reduce extreme sparsity/overfitting from ultra-rare categories within each type, which usually lowers log-MAE in CHAMPS while keeping the model family and features unchanged.'
- What this solution (achieved 21.03352) has done: 'Your current score (21.03, lower is better) is far from the target (1.234), so we need a legitimate improvement while keeping the same per-type linear Ridge pipeline and engineered features. The biggest score issue left is that the features and model are still not aligned with the competition metric: the metric averages log-MAE per coupling `type`, so we should weight each training row within a type by the inverse frequency of that type to avoid big types dominating the fit. Additionally, `Ridge` in scikit-learn does not use `random_state`, and leaving `solver` implicit can change behavior; we set a deterministic solver and enable `sample_weight` in `.fit()` (no change to core model family). These are minimal changes (same features, same per-type loop, same Ridge model) that typically reduce per-type MAE and therefore log-MAE.'
- What this solution (achieved 19.99662) has done: 'The timeout is dominated by repeated heavy merges and, especially, repeatedly fitting a new `ColumnTransformer`/`OneHotEncoder` pipeline for each coupling `type`, which re-does expensive category scanning and sparse matrix construction many times. I keep the exact same model (Ridge(sag)), per-type training loop, and features, but (1) make all CSV reads and merges faster/more memory-efficient via dtypes and selecting only needed columns, (2) replace the slow `groupby + set_index + index.map(lambda...)` weight computation with a fully vectorized merge, and (3) pre-split the data by `type` once (using integer row indices) to avoid repeated boolean masks and repeated DataFrame slicing overhead. These changes are runtime-only and preserve evaluation semantics; the model, preprocessing steps, feature definitions, and per-type normalization remain identical.'
- What this solution (achieved 2.84748) has done: 'The main timeout drivers here are repeated huge pandas merges, unnecessary dtype upcasting, and per-type pipeline fitting overhead made worse by slow sparse construction. I keep the exact feature set and Ridge-per-type training loop, but speed up joins by indexing structures/aux tables once and using fast `.join`/reindex patterns, compute MST aggregates only once, and avoid repeated pandas object creation inside the type loop. I also enable Intel-accelerated scikit-learn (available as `scikit-learn-intelex`) to speed up Ridge/SAG and preprocessing without changing the math. Finally, I reduce categorical string conversions and keep categoricals until the OneHotEncoder step to cut memory/time while preserving identical semantics.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_PATH = "/kaggle/data/champs-scalar-coupling"
train_path = os.path.join(BASE_PATH, "train.csv")
test_path = os.path.join(BASE_PATH, "test.csv")
structures_path = os.path.join(BASE_PATH, "structures.csv")

train_dtypes = {
    "id": np.int32,
    "molecule_name": "category",
    "atom_index_0": np.int16,
    "atom_index_1": np.int16,
    "type": "category",
    "scalar_coupling_constant": np.float64,
}
test_dtypes = {
    "id": np.int32,
    "molecule_name": "category",
    "atom_index_0": np.int16,
    "atom_index_1": np.int16,
    "type": "category",
}
structures_dtypes = {
    "molecule_name": "category",
    "atom_index": np.int16,
    "atom": "category",
    "x": np.float32,
    "y": np.float32,
    "z": np.float32,
}

train = pd.read_csv(train_path, dtype=train_dtypes, usecols=list(train_dtypes.keys()))
test = pd.read_csv(test_path, dtype=test_dtypes, usecols=list(test_dtypes.keys()))
structures = pd.read_csv(
    structures_path, dtype=structures_dtypes, usecols=list(structures_dtypes.keys())
)

mulliken_path = os.path.join(BASE_PATH, "mulliken_charges.csv")
mst_path = os.path.join(BASE_PATH, "magnetic_shielding_tensors.csv")
dipole_path = os.path.join(BASE_PATH, "dipole_moments.csv")
pe_path = os.path.join(BASE_PATH, "potential_energy.csv")

mulliken = (
    pd.read_csv(
        mulliken_path,
        usecols=["molecule_name", "atom_index", "mulliken_charge"],
        dtype={
            "molecule_name": "category",
            "atom_index": np.int16,
            "mulliken_charge": np.float32,
        },
    )
    if os.path.exists(mulliken_path)
    else None
)

mst = (
    pd.read_csv(
        mst_path,
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
        dtype={
            "molecule_name": "category",
            "atom_index": np.int16,
            "XX": np.float32,
            "YX": np.float32,
            "ZX": np.float32,
            "XY": np.float32,
            "YY": np.float32,
            "ZY": np.float32,
            "XZ": np.float32,
            "YZ": np.float32,
            "ZZ": np.float32,
        },
    )
    if os.path.exists(mst_path)
    else None
)

dipole = (
    pd.read_csv(
        dipole_path,
        usecols=["molecule_name", "X", "Y", "Z"],
        dtype={
            "molecule_name": "category",
            "X": np.float32,
            "Y": np.float32,
            "Z": np.float32,
        },
    )
    if os.path.exists(dipole_path)
    else None
)

potential_energy = (
    pd.read_csv(
        pe_path,
        usecols=["molecule_name", "potential_energy"],
        dtype={"molecule_name": "category", "potential_energy": np.float32},
    )
    if os.path.exists(pe_path)
    else None
)


## === cell 1

s0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "x_0",
        "y": "y_0",
        "z": "z_0",
    }
)[["molecule_name", "atom_index_0", "atom_0", "x_0", "y_0", "z_0"]].set_index(
    ["molecule_name", "atom_index_0"]
)

s1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "x": "x_1",
        "y": "y_1",
        "z": "z_1",
    }
)[["molecule_name", "atom_index_1", "atom_1", "x_1", "y_1", "z_1"]].set_index(
    ["molecule_name", "atom_index_1"]
)

train_2 = train.copy()
test_2 = test.copy()

train_2 = train_2.join(s0, on=["molecule_name", "atom_index_0"])
train_2 = train_2.join(s1, on=["molecule_name", "atom_index_1"])
test_2 = test_2.join(s0, on=["molecule_name", "atom_index_0"])
test_2 = test_2.join(s1, on=["molecule_name", "atom_index_1"])

if mulliken is not None:
    m0 = mulliken.rename(
        columns={"atom_index": "atom_index_0", "mulliken_charge": "mcharge_0"}
    )[["molecule_name", "atom_index_0", "mcharge_0"]].set_index(
        ["molecule_name", "atom_index_0"]
    )
    m1 = mulliken.rename(
        columns={"atom_index": "atom_index_1", "mulliken_charge": "mcharge_1"}
    )[["molecule_name", "atom_index_1", "mcharge_1"]].set_index(
        ["molecule_name", "atom_index_1"]
    )

    train_2 = train_2.join(m0, on=["molecule_name", "atom_index_0"])
    train_2 = train_2.join(m1, on=["molecule_name", "atom_index_1"])
    test_2 = test_2.join(m0, on=["molecule_name", "atom_index_0"])
    test_2 = test_2.join(m1, on=["molecule_name", "atom_index_1"])
else:
    train_2["mcharge_0"] = np.nan
    train_2["mcharge_1"] = np.nan
    test_2["mcharge_0"] = np.nan
    test_2["mcharge_1"] = np.nan

if mst is not None:
    tensor_cols = ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]
    vals = mst[tensor_cols].to_numpy(dtype=np.float64, copy=False)
    mst_ag = pd.DataFrame(
        {
            "molecule_name": mst["molecule_name"].to_numpy(copy=False),
            "atom_index": mst["atom_index"].to_numpy(copy=False),
            "mst_mean": vals.mean(axis=1),
            "mst_std": vals.std(axis=1),
            "mst_absmean": np.abs(vals).mean(axis=1),
            "mst_trace": mst["XX"].astype(np.float64).to_numpy(copy=False)
            + mst["YY"].astype(np.float64).to_numpy(copy=False)
            + mst["ZZ"].astype(np.float64).to_numpy(copy=False),
        }
    )
    mst0 = mst_ag.rename(columns={"atom_index": "atom_index_0"}).set_index(
        ["molecule_name", "atom_index_0"]
    )
    mst0.columns = [c + "_0" for c in mst0.columns]
    mst1 = mst_ag.rename(columns={"atom_index": "atom_index_1"}).set_index(
        ["molecule_name", "atom_index_1"]
    )
    mst1.columns = [c + "_1" for c in mst1.columns]

    train_2 = train_2.join(mst0, on=["molecule_name", "atom_index_0"])
    train_2 = train_2.join(mst1, on=["molecule_name", "atom_index_1"])
    test_2 = test_2.join(mst0, on=["molecule_name", "atom_index_0"])
    test_2 = test_2.join(mst1, on=["molecule_name", "atom_index_1"])
else:
    for col in [
        "mst_mean_0",
        "mst_std_0",
        "mst_absmean_0",
        "mst_trace_0",
        "mst_mean_1",
        "mst_std_1",
        "mst_absmean_1",
        "mst_trace_1",
    ]:
        train_2[col] = np.nan
        test_2[col] = np.nan

if dipole is not None:
    dip = dipole.set_index("molecule_name")
    train_2 = train_2.join(dip, on="molecule_name")
    test_2 = test_2.join(dip, on="molecule_name")
else:
    train_2["X"] = np.nan
    train_2["Y"] = np.nan
    train_2["Z"] = np.nan
    test_2["X"] = np.nan
    test_2["Y"] = np.nan
    test_2["Z"] = np.nan

if potential_energy is not None:
    pe = potential_energy.set_index("molecule_name")
    train_2 = train_2.join(pe, on="molecule_name")
    test_2 = test_2.join(pe, on="molecule_name")
else:
    train_2["potential_energy"] = np.nan
    test_2["potential_energy"] = np.nan

for c in ["x_0", "y_0", "z_0", "x_1", "y_1", "z_1"]:
    train_2[c] = train_2[c].fillna(0.0)
    test_2[c] = test_2[c].fillna(0.0)

num_fill_zero = [
    "mcharge_0",
    "mcharge_1",
    "mst_mean_0",
    "mst_std_0",
    "mst_absmean_0",
    "mst_trace_0",
    "mst_mean_1",
    "mst_std_1",
    "mst_absmean_1",
    "mst_trace_1",
    "X",
    "Y",
    "Z",
    "potential_energy",
]
for c in num_fill_zero:
    if c in train_2.columns:
        train_2[c] = train_2[c].fillna(0.0)
    if c in test_2.columns:
        test_2[c] = test_2[c].fillna(0.0)

x0_tr = train_2["x_0"].to_numpy(dtype=np.float64, copy=False)
y0_tr = train_2["y_0"].to_numpy(dtype=np.float64, copy=False)
z0_tr = train_2["z_0"].to_numpy(dtype=np.float64, copy=False)
x1_tr = train_2["x_1"].to_numpy(dtype=np.float64, copy=False)
y1_tr = train_2["y_1"].to_numpy(dtype=np.float64, copy=False)
z1_tr = train_2["z_1"].to_numpy(dtype=np.float64, copy=False)

dx_tr = x0_tr - x1_tr
dy_tr = y0_tr - y1_tr
dz_tr = z0_tr - z1_tr
dist_tr = np.sqrt(dx_tr * dx_tr + dy_tr * dy_tr + dz_tr * dz_tr)

x0_te = test_2["x_0"].to_numpy(dtype=np.float64, copy=False)
y0_te = test_2["y_0"].to_numpy(dtype=np.float64, copy=False)
z0_te = test_2["z_0"].to_numpy(dtype=np.float64, copy=False)
x1_te = test_2["x_1"].to_numpy(dtype=np.float64, copy=False)
y1_te = test_2["y_1"].to_numpy(dtype=np.float64, copy=False)
z1_te = test_2["z_1"].to_numpy(dtype=np.float64, copy=False)

dx_te = x0_te - x1_te
dy_te = y0_te - y1_te
dz_te = z0_te - z1_te
dist_te = np.sqrt(dx_te * dx_te + dy_te * dy_te + dz_te * dz_te)

train_2["distance"] = dist_tr
test_2["distance"] = dist_te

eps = 1e-6
train_2["distance2"] = train_2["distance"] ** 2
test_2["distance2"] = test_2["distance"] ** 2
train_2["inv_distance"] = 1.0 / (train_2["distance"] + eps)
test_2["inv_distance"] = 1.0 / (test_2["distance"] + eps)

train_2["atom_pair"] = (
    train_2["atom_0"].astype("string") + "_" + train_2["atom_1"].astype("string")
).astype("string")
test_2["atom_pair"] = (
    test_2["atom_0"].astype("string") + "_" + test_2["atom_1"].astype("string")
).astype("string")

train_2["mcharge_sum"] = train_2["mcharge_0"].astype(np.float64) + train_2[
    "mcharge_1"
].astype(np.float64)
test_2["mcharge_sum"] = test_2["mcharge_0"].astype(np.float64) + test_2[
    "mcharge_1"
].astype(np.float64)
train_2["mcharge_diff"] = train_2["mcharge_0"].astype(np.float64) - train_2[
    "mcharge_1"
].astype(np.float64)
test_2["mcharge_diff"] = test_2["mcharge_0"].astype(np.float64) - test_2[
    "mcharge_1"
].astype(np.float64)
train_2["mcharge_prod"] = train_2["mcharge_0"].astype(np.float64) * train_2[
    "mcharge_1"
].astype(np.float64)
test_2["mcharge_prod"] = test_2["mcharge_0"].astype(np.float64) * test_2[
    "mcharge_1"
].astype(np.float64)

train_2["mst_mean_diff"] = train_2["mst_mean_0"].astype(np.float64) - train_2[
    "mst_mean_1"
].astype(np.float64)
test_2["mst_mean_diff"] = test_2["mst_mean_0"].astype(np.float64) - test_2[
    "mst_mean_1"
].astype(np.float64)
train_2["mst_trace_diff"] = train_2["mst_trace_0"].astype(np.float64) - train_2[
    "mst_trace_1"
].astype(np.float64)
test_2["mst_trace_diff"] = test_2["mst_trace_0"].astype(np.float64) - test_2[
    "mst_trace_1"
].astype(np.float64)

train_2["dipole_mag"] = np.sqrt(
    train_2["X"].astype(np.float64) ** 2
    + train_2["Y"].astype(np.float64) ** 2
    + train_2["Z"].astype(np.float64) ** 2
)
test_2["dipole_mag"] = np.sqrt(
    test_2["X"].astype(np.float64) ** 2
    + test_2["Y"].astype(np.float64) ** 2
    + test_2["Z"].astype(np.float64) ** 2
)

train_2["ux"] = dx_tr / (train_2["distance"] + eps)
train_2["uy"] = dy_tr / (train_2["distance"] + eps)
train_2["uz"] = dz_tr / (train_2["distance"] + eps)
test_2["ux"] = dx_te / (test_2["distance"] + eps)
test_2["uy"] = dy_te / (test_2["distance"] + eps)
test_2["uz"] = dz_te / (test_2["distance"] + eps)

train_2["dipole_proj"] = (
    train_2["X"].astype(np.float64) * train_2["ux"]
    + train_2["Y"].astype(np.float64) * train_2["uy"]
    + train_2["Z"].astype(np.float64) * train_2["uz"]
)
test_2["dipole_proj"] = (
    test_2["X"].astype(np.float64) * test_2["ux"]
    + test_2["Y"].astype(np.float64) * test_2["uy"]
    + test_2["Z"].astype(np.float64) * test_2["uz"]
)


## === cell 2
numeric_features = [
    "distance",
    "distance2",
    "inv_distance",
    "mcharge_0",
    "mcharge_1",
    "mcharge_sum",
    "mcharge_diff",
    "mcharge_prod",
    "mst_mean_0",
    "mst_std_0",
    "mst_absmean_0",
    "mst_trace_0",
    "mst_mean_1",
    "mst_std_1",
    "mst_absmean_1",
    "mst_trace_1",
    "mst_mean_diff",
    "mst_trace_diff",
    "dipole_mag",
    "dipole_proj",
    "potential_energy",
]
cat_features = ["type", "atom_0", "atom_1", "atom_pair"]

for c in numeric_features:
    tr = train_2[c].to_numpy(dtype=np.float64, copy=False)
    te = test_2[c].to_numpy(dtype=np.float64, copy=False)
    tr = np.nan_to_num(tr, nan=0.0, posinf=0.0, neginf=0.0)
    te = np.nan_to_num(te, nan=0.0, posinf=0.0, neginf=0.0)
    train_2[c] = tr
    test_2[c] = te

for c in cat_features:
    train_2[c] = train_2[c].astype("string").fillna("NA")
    test_2[c] = test_2[c].astype("string").fillna("NA")

y_train = train_2["scalar_coupling_constant"].astype(np.float64).to_numpy(copy=False)
global_y_mean = float(np.mean(y_train))

type_pair_counts = (
    train_2.groupby(["type", "atom_pair"], sort=False)
    .size()
    .rename("tp_count")
    .reset_index()
)
train_key = train_2[["type", "atom_pair"]].copy()
inv_type_pair_weight = train_key.merge(
    type_pair_counts, on=["type", "atom_pair"], how="left", sort=False
)["tp_count"].to_numpy(dtype=np.float64, copy=False)
inv_type_pair_weight = 1.0 / inv_type_pair_weight


def make_pipe(min_freq=None):
    ohe_kwargs = dict(handle_unknown="ignore", sparse_output=True)
    if min_freq is not None and min_freq >= 2:
        ohe_kwargs["min_frequency"] = int(min_freq)

    preprocess = ColumnTransformer(
        transformers=[
            ("cats", OneHotEncoder(**ohe_kwargs), cat_features),
            ("nums", StandardScaler(with_mean=False), numeric_features),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
    model = Ridge(alpha=10.0, fit_intercept=True, solver="sag")
    return Pipeline(steps=[("prep", preprocess), ("ridge", model)])


test_pred = np.zeros(len(test_2), dtype=np.float64)

train_type_to_idx = train_2.groupby("type", sort=False).indices
test_type_to_idx = test_2.groupby("type", sort=False).indices

types_unique_train = list(train_type_to_idx.keys())

feature_cols = cat_features + numeric_features

for t in types_unique_train:
    te_idx = test_type_to_idx.get(t)
    if te_idx is None or len(te_idx) == 0:
        continue
    tr_idx = train_type_to_idx.get(t)
    te_idx = np.asarray(te_idx, dtype=np.int64)

    if tr_idx is None or len(tr_idx) == 0:
        test_pred[te_idx] = global_y_mean
        continue

    tr_idx = np.asarray(tr_idx, dtype=np.int64)
    y_t = y_train[tr_idx]
    mu = float(np.mean(y_t))
    sd = float(np.std(y_t))
    if not np.isfinite(sd) or sd < 1e-12:
        test_pred[te_idx] = mu
        continue

    Xtr_df = train_2[feature_cols].take(tr_idx)
    Xte_df = test_2[feature_cols].take(te_idx)

    w_t = inv_type_pair_weight[tr_idx]
    w_t = w_t / np.mean(w_t)  # keep weight scale stable

    n_tr = len(tr_idx)
    if n_tr < 20000:
        min_freq = None
    else:
        min_freq = 10

    pipe = make_pipe(min_freq=min_freq)
    pipe.fit(Xtr_df, (y_t - mu) / sd, ridge__sample_weight=w_t)
    yhat_std = pipe.predict(Xte_df)
    yhat = yhat_std * sd + mu

    lo, hi = np.quantile(y_t, [0.001, 0.999])
    yhat = np.clip(yhat, lo, hi)

    test_pred[te_idx] = yhat

test_2["scalar_coupling_constant"] = test_pred

submission = test_2[["id", "scalar_coupling_constant"]].copy()
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Unique types in test:", test_2["type"].nunique())
