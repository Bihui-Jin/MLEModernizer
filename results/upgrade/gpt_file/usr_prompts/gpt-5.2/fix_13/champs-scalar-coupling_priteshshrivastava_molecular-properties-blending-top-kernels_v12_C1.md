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

-1.6821991287997458

# 6. Current score

1.59661

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.8228) has done: 'The crash is because the notebook expects out-of-scope datasets (other people’s submission files) under `../input/...`, which do not exist in your environment, so no submission CSV is ever written. I replace that external-file blending with an in-notebook baseline model trained on the provided `train.csv/structures.csv` and used to predict `test.csv`, keeping the pipeline simple and deterministic. This fixes runtime, guarantees a correctly formatted `submission.csv`, and should achieve a reasonable score (better than a constant baseline) without relying on unavailable inputs. I also add safety checks to ensure feature columns align and no NaNs remain.'
- What this solution (achieved 1.23132) has done: 'Your current score (1.8228, lower is better) is far from the target (-1.6822), so we should make a small, legitimate improvement that better matches the competition’s metric without changing your overall approach (structure-merge features + a single sklearn regressor). The biggest gap is that CV is using global MAE, while the leaderboard uses **mean of log(MAE) computed per coupling `type`**, so the model is not being tuned/validated against the right objective. I (1) add an “official metric” cross-validation calculation (log MAE per type averaged) for visibility, and (2) train **separate models per `type`** (same pipeline/model class and same features) which usually improves this competition materially while staying within the same core logic. This keeps deterministic training, produces the same submission schema, and should move the score down toward the target.'
- What this solution (achieved 1.21757) has done: 'Your current score (1.23132, lower is better) is still far from the target (-1.6822), so we should make a small, legitimate improvement without changing the core approach (structures merge → basic distance features → sklearn tree regressor per type). The biggest low-risk gain here is to add a few standard “pairwise geometry” features (signed deltas and simple ratios) plus a molecule-level distance normalization, which typically reduces MAE across coupling types while keeping the same model class and training loop. I also add a deterministic per-type group-CV report (same metric) to verify the change is moving in the right direction, but keep training on full data for submission exactly as before. All paths stay the same and the code still writes a valid `submission.csv`.'
- What this solution (achieved 1.21282) has done: 'Your score is much worse than the target (lower is better), so we should make a small, legitimate accuracy gain without changing the overall approach (structures merge → handcrafted geometry features → per-type sklearn regressor). The most direct low-risk improvement here is to add a few well-known CHAMPS auxiliary features (Mulliken charges and magnetic shielding tensors) for each of the two atoms, merged by `(molecule_name, atom_index)`; this keeps the same training loop and model class but usually reduces per-type MAE materially. To keep runtime under control, we only load the needed columns and cast dtypes to reduce memory, and we add a couple of simple derived tensor summaries (trace and Frobenius norm) to help the tree model. Submission writing and schema remain identical.'
- What this solution (achieved 1.21017) has done: 'Your current score (1.21282, lower is better) is still far from the target (-1.6822), so we should make a small, legitimate improvement without changing the core approach (same features + same per-type HistGradientBoostingRegressor training loop). The biggest low-risk boost here is to add the CHAMPS-provided **molecule-level auxiliary features** (dipole moments and potential energy) via a simple merge on `molecule_name`, which often reduces error across coupling types while keeping everything else identical. I also set `categorical_transformer` to output a sparse matrix (instead of dense) to reduce memory pressure and keep training more stable on this large dataset; this doesn’t change semantics and is usually faster. Submission writing, schema, and paths remain the same.'
- What this solution (achieved 1.20675) has done: 'To move your score down (lower is better) with minimal change, I keep the exact same feature set and per-type `HistGradientBoostingRegressor` training loop, but fix a key mismatch with the competition metric: the target distribution is highly type-dependent, so a per-type **target transformation** stabilizes MAE and usually improves the log-MAE-by-type score. Concretely, we train each type-model on `log1p(y - y_min_type)` (a safe, strictly monotone transform per type), then invert it at prediction time; this preserves the same model architecture/loop and keeps evaluation semantics (predicting scalar coupling constants) while improving robustness. I also add a tiny, deterministic “shrink-to-type-mean” calibration on predictions (a convex blend with the train mean per type) to reduce outlier MAE without changing the model class. The code still runs end-to-end and writes a valid `submission.csv` in `/kaggle/working`.'
- What this solution (achieved 1.21965) has done: 'The main timeout comes from repeatedly fitting the full preprocessing (imputer + one-hot encoder) inside the per-type loops for every fold and again for test training, plus expensive `reindex`/`concat` and repeated `set_index`/`map` work during feature construction. I keep the exact model/training logic but make it faster by (1) caching molecule-level lookups (dipole/potential) once, (2) replacing the per-row `MultiIndex` reindex approach with two hash-merge joins for atom_0/atom_1 features (provably equivalent), (3) computing groupby transforms (mean/std) in a single pass, and (4) reusing boolean masks and avoiding repeated `.to_numpy()` inside loops. These changes reduce overhead and memory churn substantially while preserving identical semantics and accuracy (only negligible float differences from equivalent vectorized operations).'
- What this solution (achieved 1.59661) has done: 'The timeout is dominated by repeated expensive DataFrame operations in feature building (MultiIndex reindex + concat on 4.6M rows twice) and by training 3-fold CV *per coupling type* (roughly 3×8 model fits) on large sparse matrices with slow boolean indexing/slicing. I keep the exact same features, preprocessing, models, CV, and calibration logic, but make them faster by: (1) replacing MultiIndex reindex with vectorized merges on integer-encoded molecule IDs, (2) building distance statistics once on train+test and joining back (same semantics), (3) avoiding pandas-heavy metric computation in CV, and (4) reducing repeated sparse slicing overhead by precomputing fold indices per type using integer codes. These changes are algebraically equivalent (same joins/aggregations, same model fits and predictions) and should finish within 600s.'

# 9. Code solution

## === cell 0
import os
import time
import numpy as np
import pandas as pd

INPUT_DIR = "/kaggle/input/champs-scalar-coupling"
WORKING_DIR = "/kaggle/working"

print("Input dir exists:", os.path.exists(INPUT_DIR))
print("Files (sample):", sorted(os.listdir(INPUT_DIR))[:10])

t0 = time.time()

os.environ.setdefault("PYTHONHASHSEED", "42")
np.random.seed(42)



## === cell 1
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor

train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")
structures_path = os.path.join(INPUT_DIR, "structures.csv")
mulliken_path = os.path.join(INPUT_DIR, "mulliken_charges.csv")
mst_path = os.path.join(INPUT_DIR, "magnetic_shielding_tensors.csv")
dipole_path = os.path.join(INPUT_DIR, "dipole_moments.csv")
potential_path = os.path.join(INPUT_DIR, "potential_energy.csv")
sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")

train = pd.read_csv(
    train_path,
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int32,
        "atom_index_1": np.int32,
        "type": "category",
        "scalar_coupling_constant": np.float32,
    },
)
test = pd.read_csv(
    test_path,
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int32,
        "atom_index_1": np.int32,
        "type": "category",
    },
)
structures = pd.read_csv(
    structures_path,
    dtype={
        "molecule_name": "category",
        "atom_index": np.int32,
        "atom": "category",
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
    },
)

print(train.shape, test.shape, structures.shape)
print(train.columns)
print(test.columns)
print(structures.columns)
print("Load elapsed (s):", round(time.time() - t0, 2))



## === cell 2
mull = pd.read_csv(
    mulliken_path,
    usecols=["molecule_name", "atom_index", "mulliken_charge"],
    dtype={
        "molecule_name": "category",
        "atom_index": np.int32,
        "mulliken_charge": np.float32,
    },
)

mst = pd.read_csv(
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
        "atom_index": np.int32,
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

mst["mst_trace"] = (mst["XX"] + mst["YY"] + mst["ZZ"]).astype(np.float32)
mst_vals = mst[["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]].to_numpy(
    copy=False
)
mst["mst_frob"] = np.sqrt(np.einsum("ij,ij->i", mst_vals, mst_vals)).astype(np.float32)

dip = pd.read_csv(
    dipole_path,
    usecols=["molecule_name", "X", "Y", "Z"],
    dtype={
        "molecule_name": "category",
        "X": np.float32,
        "Y": np.float32,
        "Z": np.float32,
    },
).rename(columns={"X": "dipole_X", "Y": "dipole_Y", "Z": "dipole_Z"})

pot = pd.read_csv(
    potential_path,
    usecols=["molecule_name", "potential_energy"],
    dtype={"molecule_name": "category", "potential_energy": np.float32},
)

mol_union = pd.Categorical(
    pd.concat(
        [train["molecule_name"], test["molecule_name"], structures["molecule_name"]],
        ignore_index=True,
    )
)
mol_cat = mol_union.categories


def _add_mol_id(df, col="molecule_name"):
    return pd.Categorical(df[col], categories=mol_cat).codes.astype(
        np.int32, copy=False
    )


train["mol_id"] = _add_mol_id(train)
test["mol_id"] = _add_mol_id(test)
structures["mol_id"] = _add_mol_id(structures)
mull["mol_id"] = _add_mol_id(mull)
mst["mol_id"] = _add_mol_id(mst)
dip["mol_id"] = _add_mol_id(dip)
pot["mol_id"] = _add_mol_id(pot)

dip = dip.sort_values("mol_id", kind="mergesort")
pot = pot.sort_values("mol_id", kind="mergesort")
max_mol = int(
    max(train["mol_id"].max(), test["mol_id"].max(), structures["mol_id"].max())
)
dip_X_arr = np.full(max_mol + 1, np.nan, dtype=np.float32)
dip_Y_arr = np.full(max_mol + 1, np.nan, dtype=np.float32)
dip_Z_arr = np.full(max_mol + 1, np.nan, dtype=np.float32)
pot_E_arr = np.full(max_mol + 1, np.nan, dtype=np.float32)
dip_X_arr[dip["mol_id"].to_numpy()] = dip["dipole_X"].to_numpy(
    dtype=np.float32, copy=False
)
dip_Y_arr[dip["mol_id"].to_numpy()] = dip["dipole_Y"].to_numpy(
    dtype=np.float32, copy=False
)
dip_Z_arr[dip["mol_id"].to_numpy()] = dip["dipole_Z"].to_numpy(
    dtype=np.float32, copy=False
)
pot_E_arr[pot["mol_id"].to_numpy()] = pot["potential_energy"].to_numpy(
    dtype=np.float32, copy=False
)

struct_aux = structures.merge(
    mull[["mol_id", "atom_index", "mulliken_charge"]],
    on=["mol_id", "atom_index"],
    how="left",
    copy=False,
)
struct_aux = struct_aux.merge(
    mst[
        [
            "mol_id",
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
            "mst_trace",
            "mst_frob",
        ]
    ],
    on=["mol_id", "atom_index"],
    how="left",
    copy=False,
)

need_cols = [
    "mol_id",
    "molecule_name",
    "atom_index",
    "atom",
    "x",
    "y",
    "z",
    "mulliken_charge",
    "XX",
    "YX",
    "ZX",
    "XY",
    "YY",
    "ZY",
    "XZ",
    "YZ",
    "ZZ",
    "mst_trace",
    "mst_frob",
]
struct_aux = struct_aux[need_cols]

s0_cols = {
    "atom": "atom_0",
    "x": "x_0",
    "y": "y_0",
    "z": "z_0",
    "mulliken_charge": "mulliken_charge_0",
    "XX": "mst_XX_0",
    "YX": "mst_YX_0",
    "ZX": "mst_ZX_0",
    "XY": "mst_XY_0",
    "YY": "mst_YY_0",
    "ZY": "mst_ZY_0",
    "XZ": "mst_XZ_0",
    "YZ": "mst_YZ_0",
    "ZZ": "mst_ZZ_0",
    "mst_trace": "mst_trace_0",
    "mst_frob": "mst_frob_0",
}
s1_cols = {k: v.replace("_0", "_1") for k, v in s0_cols.items()}

a0 = struct_aux.rename(columns={"atom_index": "atom_index_0", **s0_cols})
a1 = struct_aux.rename(columns={"atom_index": "atom_index_1", **s1_cols})


def add_structure_features(df):
    df2 = df.copy()

    df2 = df2.merge(
        a0.drop(columns=["molecule_name"], errors="ignore"),
        on=["mol_id", "atom_index_0"],
        how="left",
        copy=False,
    )
    df2 = df2.merge(
        a1.drop(columns=["molecule_name"], errors="ignore"),
        on=["mol_id", "atom_index_1"],
        how="left",
        copy=False,
    )

    mol_id = df2["mol_id"].to_numpy(dtype=np.int32, copy=False)
    df2["dipole_X"] = dip_X_arr[mol_id]
    df2["dipole_Y"] = dip_Y_arr[mol_id]
    df2["dipole_Z"] = dip_Z_arr[mol_id]
    df2["potential_energy"] = pot_E_arr[mol_id]

    dx = (df2["x_0"] - df2["x_1"]).astype(np.float32)
    dy = (df2["y_0"] - df2["y_1"]).astype(np.float32)
    dz = (df2["z_0"] - df2["z_1"]).astype(np.float32)

    df2["dx"] = dx
    df2["dy"] = dy
    df2["dz"] = dz

    dist2 = (dx * dx + dy * dy + dz * dz).astype(np.float32)
    df2["dist2"] = dist2
    df2["dist"] = np.sqrt(dist2.astype(np.float64)).astype(np.float32)

    df2["abs_dx"] = dx.abs()
    df2["abs_dy"] = dy.abs()
    df2["abs_dz"] = dz.abs()

    eps = np.float32(1e-12)
    dist = df2["dist"]
    df2["inv_dist"] = (1.0 / (dist + eps)).astype(np.float32)
    df2["inv_dist2"] = (1.0 / (dist2 + eps)).astype(np.float32)
    df2["abs_dx_over_dist"] = (df2["abs_dx"] / (dist + eps)).astype(np.float32)
    df2["abs_dy_over_dist"] = (df2["abs_dy"] / (dist + eps)).astype(np.float32)
    df2["abs_dz_over_dist"] = (df2["abs_dz"] / (dist + eps)).astype(np.float32)
    df2["dx_dy"] = (dx * dy).astype(np.float32)
    df2["dx_dz"] = (dx * dz).astype(np.float32)
    df2["dy_dz"] = (dy * dz).astype(np.float32)

    df2["mulliken_sum"] = (df2["mulliken_charge_0"] + df2["mulliken_charge_1"]).astype(
        np.float32
    )
    df2["mulliken_diff"] = (df2["mulliken_charge_0"] - df2["mulliken_charge_1"]).astype(
        np.float32
    )
    df2["mst_trace_sum"] = (df2["mst_trace_0"] + df2["mst_trace_1"]).astype(np.float32)
    df2["mst_trace_diff"] = (df2["mst_trace_0"] - df2["mst_trace_1"]).astype(np.float32)
    df2["mst_frob_sum"] = (df2["mst_frob_0"] + df2["mst_frob_1"]).astype(np.float32)
    df2["mst_frob_diff"] = (df2["mst_frob_0"] - df2["mst_frob_1"]).astype(np.float32)

    df2["dipole_mag"] = np.sqrt(
        (df2["dipole_X"].astype(np.float64) ** 2)
        + (df2["dipole_Y"].astype(np.float64) ** 2)
        + (df2["dipole_Z"].astype(np.float64) ** 2)
    ).astype(np.float32)

    return df2


train_f = add_structure_features(train)
test_f = add_structure_features(test)


def add_molecule_distance_norm_both(train_df, test_df):
    combined = pd.concat(
        [train_df[["mol_id", "dist"]], test_df[["mol_id", "dist"]]],
        axis=0,
        ignore_index=True,
        copy=False,
    )
    stats = (
        combined.groupby("mol_id", sort=False)["dist"]
        .agg(mol_dist_mean="mean", mol_dist_std="std")
        .astype(np.float32)
    )

    def _join_and_compute(df):
        df2 = df.join(stats, on="mol_id")
        df2["mol_dist_std"] = (
            df2["mol_dist_std"].fillna(np.float32(0.0)).astype(np.float32)
        )
        eps = np.float32(1e-12)
        df2["dist_z"] = (
            (df2["dist"] - df2["mol_dist_mean"]) / (df2["mol_dist_std"] + eps)
        ).astype(np.float32)
        df2["dist_over_mean"] = (df2["dist"] / (df2["mol_dist_mean"] + eps)).astype(
            np.float32
        )
        return df2

    return _join_and_compute(train_df), _join_and_compute(test_df)


train_f, test_f = add_molecule_distance_norm_both(train_f, test_f)

missing_train = train_f[["atom_0", "atom_1", "x_0", "x_1", "dist"]].isna().mean().max()
missing_test = test_f[["atom_0", "atom_1", "x_0", "x_1", "dist"]].isna().mean().max()
print("Max missing fraction (train,test):", float(missing_train), float(missing_test))
print("Feature build elapsed (s):", round(time.time() - t0, 2))



## === cell 3
target = "scalar_coupling_constant"
y = train_f[target].to_numpy()

feature_cols_num = [
    "x_0",
    "y_0",
    "z_0",
    "x_1",
    "y_1",
    "z_1",
    "dx",
    "dy",
    "dz",
    "dist",
    "dist2",
    "inv_dist",
    "inv_dist2",
    "abs_dx",
    "abs_dy",
    "abs_dz",
    "abs_dx_over_dist",
    "abs_dy_over_dist",
    "abs_dz_over_dist",
    "dx_dy",
    "dx_dz",
    "dy_dz",
    "mol_dist_mean",
    "mol_dist_std",
    "dist_z",
    "dist_over_mean",
    "mulliken_charge_0",
    "mulliken_charge_1",
    "mulliken_sum",
    "mulliken_diff",
    "mst_trace_0",
    "mst_trace_1",
    "mst_trace_sum",
    "mst_trace_diff",
    "mst_frob_0",
    "mst_frob_1",
    "mst_frob_sum",
    "mst_frob_diff",
    "dipole_X",
    "dipole_Y",
    "dipole_Z",
    "dipole_mag",
    "potential_energy",
]
feature_cols_cat = ["type", "atom_0", "atom_1"]

X = train_f[feature_cols_num + feature_cols_cat]
X_test = test_f[feature_cols_num + feature_cols_cat]

numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, feature_cols_num),
        ("cat", categorical_transformer, feature_cols_cat),
    ],
    remainder="drop",
)


def champs_metric(y_true, y_pred, types, eps=1e-9):
    codes, uniques = pd.factorize(types, sort=False)
    ae = np.abs(y_true - y_pred)
    sums = np.bincount(codes, weights=ae.astype(np.float64, copy=False))
    cnts = np.bincount(codes)
    maes = sums / np.maximum(cnts, 1)
    score = float(np.mean(np.log(maes + eps)))
    maes_s = pd.Series(maes, index=uniques)
    return score, maes_s


gkf = GroupKFold(n_splits=3)
groups = train_f["molecule_name"].to_numpy()
types_all = train_f["type"].to_numpy()

Xt_all = preprocess.fit_transform(X)
Xt_test_all = preprocess.transform(X_test)

print("Preprocess fit/transform elapsed (s):", round(time.time() - t0, 2))



## === cell 4
oof_pred = np.full(len(train_f), np.nan, dtype=np.float64)

cv_maes = []
cv_scores = []

type_codes_all, type_uniques = pd.factorize(types_all, sort=False)
type_to_code = {t: i for i, t in enumerate(type_uniques)}

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
    y_tr = y[tr_idx]
    y_va = y[va_idx]
    t_tr_codes = type_codes_all[tr_idx]
    t_va_codes = type_codes_all[va_idx]
    t_va = types_all[va_idx]

    y_pred_va = np.zeros_like(y_va, dtype=np.float64)

    uniq_codes = np.unique(t_tr_codes)
    for c in uniq_codes:
        tr_pos = np.flatnonzero(t_tr_codes == c)
        va_pos = np.flatnonzero(t_va_codes == c)
        if va_pos.size == 0:
            continue

        tr_rows = tr_idx[tr_pos]
        va_rows = va_idx[va_pos]

        Xt_tr = Xt_all[tr_rows, :]
        Xt_va = Xt_all[va_rows, :]

        y_tr_t = y_tr[tr_pos].astype(np.float64, copy=False)
        y_min_t = float(np.min(y_tr_t))
        shift = max(0.0, -y_min_t) + 1e-6
        y_tr_t_trans = np.log1p(y_tr_t + shift)

        model_t = HistGradientBoostingRegressor(
            loss="squared_error",
            max_depth=8,
            max_iter=250,
            learning_rate=0.05,
            random_state=42,
        )
        model_t.fit(Xt_tr, y_tr_t_trans)
        pred_trans = model_t.predict(Xt_va).astype(np.float64, copy=False)
        y_pred_va[va_pos] = np.expm1(pred_trans) - shift

    oof_pred[va_idx] = y_pred_va

    mae = mean_absolute_error(y_va, y_pred_va)
    score, _ = champs_metric(y_va.astype(np.float64, copy=False), y_pred_va, t_va)
    cv_maes.append(mae)
    cv_scores.append(score)
    print(
        f"Fold {fold} MAE (global): {mae:.6f} | CHAMPS metric (mean log MAE by type): {score:.6f}"
    )

assert np.isfinite(oof_pred).all()
print("CV MAE mean (global):", float(np.mean(cv_maes)))
print("CV CHAMPS metric mean:", float(np.mean(cv_scores)))
print("CV+OOF build elapsed (s):", round(time.time() - t0, 2))

type_stats = (
    train_f.groupby("type")[target]
    .agg(["mean", "min"])
    .rename(columns={"mean": "y_mean", "min": "y_min"})
)

alphas = {}
for t in train_f["type"].cat.categories:
    mask = types_all == t
    if not np.any(mask):
        continue
    p = oof_pred[mask]
    y_t = train_f.loc[mask, target].to_numpy(dtype=np.float64, copy=False)
    mu = float(type_stats.loc[t, "y_mean"])
    d = mu - p
    num = float(np.dot(d, (y_t - p)))
    den = float(np.dot(d, d)) + 1e-12
    a = num / den
    a = float(np.clip(a, 0.0, 0.5))
    alphas[t] = a

alpha_series = pd.Series(alphas).astype(np.float64)
print("Per-type alpha summary:", alpha_series.describe().to_dict())

oof_pred_shrunk = oof_pred.copy()
for t, a in alphas.items():
    m = types_all == t
    mu = float(type_stats.loc[t, "y_mean"])
    oof_pred_shrunk[m] = (1.0 - a) * oof_pred_shrunk[m] + a * mu

oof_score, _ = champs_metric(
    train_f[target].to_numpy(dtype=np.float64, copy=False), oof_pred, types_all
)
oof_score_shrunk, _ = champs_metric(
    train_f[target].to_numpy(dtype=np.float64, copy=False), oof_pred_shrunk, types_all
)
print("OOF CHAMPS metric (no shrink):", float(oof_score))
print("OOF CHAMPS metric (per-type shrink):", float(oof_score_shrunk))
print("OOF-calibration elapsed (s):", round(time.time() - t0, 2))



## === cell 5
test_pred = np.zeros(len(test_f), dtype=np.float64)

train_type_arr = train_f["type"].to_numpy()
test_type_arr = test_f["type"].to_numpy()

for t in sorted(train_f["type"].unique()):
    tr_mask = train_type_arr == t
    te_mask = test_type_arr == t

    y_tr_t = train_f.loc[tr_mask, target].to_numpy(dtype=np.float64, copy=False)

    y_min_t = float(type_stats.loc[t, "y_min"])
    shift = max(0.0, -y_min_t) + 1e-6
    y_tr_t_trans = np.log1p(y_tr_t + shift)

    Xt_tr = Xt_all[tr_mask, :]
    Xt_te = Xt_test_all[te_mask, :]

    model_t = HistGradientBoostingRegressor(
        loss="squared_error",
        max_depth=8,
        max_iter=250,
        learning_rate=0.05,
        random_state=42,
    )
    model_t.fit(Xt_tr, y_tr_t_trans)
    preds_trans = model_t.predict(Xt_te).astype(np.float64, copy=False)
    preds_t = np.expm1(preds_trans) - shift

    a = float(alphas.get(t, 0.12))
    y_mean_t = float(type_stats.loc[t, "y_mean"])
    preds_t = (1.0 - a) * preds_t + a * y_mean_t

    test_pred[te_mask] = preds_t
    print(
        f"type={t}: train_n={int(tr_mask.sum())}, test_n={int(te_mask.sum())}, alpha={a:.4f}"
    )

test_pred = np.where(np.isfinite(test_pred), test_pred, 0.0)

sub = pd.read_csv(sample_sub_path)
pred_df = pd.DataFrame(
    {"id": test_f["id"].to_numpy(), "scalar_coupling_constant": test_pred}
)
sub = sub[["id"]].merge(pred_df, on="id", how="left")
sub["scalar_coupling_constant"] = sub["scalar_coupling_constant"].fillna(0.0)

out_path = os.path.join(WORKING_DIR, "submission.csv")
sub.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(sub.head())
print(sub.shape)
print("Any NaNs:", sub.isna().any().any())
print(
    "Submission id match sample:", sub["id"].equals(pd.read_csv(sample_sub_path)["id"])
)
print("Total elapsed (s):", round(time.time() - t0, 2))
