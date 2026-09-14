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

-1.3534795878235684

# 6. Current score

27.32578

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.27184) has done: 'I fix the root cause of the crash in feature generation by ensuring merge-produced NaNs are handled before casting to integer dtypes (the current code casts first, which raises `IntCastingNaNError`). Then I make the pipeline robust so downstream cells always have `test_f`/`submission` defined, allowing end-to-end execution and a valid `submission.csv` to be written. These changes are score-neutral in intent (they preserve the same features/model), but they unblock training/inference so you can actually obtain a Kaggle score and iterate toward the target.'
- What this solution (achieved 25.39513) has done: 'Your current score (6.27184, lower-is-better) is far worse than the target (-1.35348), so we should improve model accuracy while keeping the same overall approach (per-type linear regression on simple pairwise features). The biggest minimal win is to add the coupling `type` into the feature space (as an integer code) and to expand the geometric features slightly (absolute deltas + inverse-distance terms) while still using the same LinearRegression training loop per type. I also fix the atom type encoding to be stable (sorted unique atoms) so train/test mappings are deterministic, and I keep submission formatting identical. These changes preserve the same model family and training semantics but should materially reduce MAE across types, moving the score closer to the target band.'
- What this solution (achieved 25.39486) has done: 'Your current score (25.39513, lower-is-better) is much worse than the target (-1.35348), so we should legitimately improve accuracy while keeping the same core approach: per-type `LinearRegression` on simple pairwise features. The smallest high-impact fix is to stop treating categorical IDs (`atom_int_*`, `atom_pair`, `type_int`) as ordinal numbers and instead one-hot encode them, which linear regression handles far better without changing the model family. To keep runtime/memory within limits, we one-hot encode only `type_int` and `atom_pair` (largest signal) and keep the continuous geometric features as-is. We also ensure the one-hot columns are aligned between train/test for each type so predictions are stable and valid.'
- What this solution (achieved 24.31367) has done: 'Your current score is far worse than the target (lower-is-better), so we should improve accuracy while keeping the same per-type LinearRegression core. The biggest issue is that the current per-type one-hot encoding explodes dimensionality (especially `atom_pair`) and tends to overfit/behave poorly; we can keep the same model family but switch the categorical handling to a fixed-size, information-preserving numeric representation (atom_0/atom_1 + their interaction) and add a simple, standard geometric signal (dot product) without changing the overall approach. We also fix the inconsistent atom-pair directionality by using an order-invariant `atom_pair_sorted`, which improves generalization while preserving semantics. These are minimal feature-engineering adjustments that should materially reduce MAE and move the score toward the target band.'
- What this solution (achieved 24.31366) has done: 'Your current score (24.31367, lower-is-better) is far worse than the target (-1.35348), so we need a legitimate accuracy boost while keeping the same core approach (per-type `LinearRegression` on simple engineered pairwise features). The smallest high-impact change here is to stop feeding categorical identifiers (`atom_int_*`, `atom_pair_sorted`) as raw ordinal numbers and instead one-hot encode just `atom_int_0`, `atom_int_1`, and `atom_pair_sorted` within each coupling `type` (keeping continuous geometry features unchanged). This preserves the same training loop and model family, but makes the linear model treat categories properly, which should significantly reduce per-type MAE. To keep memory/time safe, the one-hot encoding is done per-type and we align train/test columns for each type before fitting/predicting.'
- What this solution (achieved 23.6799) has done: 'Your current score (24.31366, lower-is-better) is far worse than the target (-1.35348), so we should improve accuracy while keeping the same overall per-type linear-regression pipeline. The biggest minimal win is to fix the one-hot alignment bug: you currently align train to test columns (dropping train-only categories), which harms fit; we should align both to the union so train keeps all informative columns and test gets missing ones as zeros. Next, we add a small, standard set of continuous geometric features (signed deltas, midpoint radius, and distance powers) that are cheap and usually strongly predictive in this competition, without changing the model family or training loop. These changes should materially reduce MAE (and thus log-MAE) while preserving your core approach and still producing a valid `submission.csv`.'
- What this solution (achieved 21.3531) has done: 'Your current score (23.6799, lower-is-better) is far from the target (-1.3535), so we should make a small but meaningful accuracy improvement while keeping the same per-type `LinearRegression` pipeline and engineered pairwise geometry features. The biggest low-risk gain is to incorporate the atom-wise auxiliary signals you already have available (Mulliken charges and magnetic shielding tensors) by merging them for atom_index_0/1 and adding only a few derived per-pair features (diff/sum/dot/norm) to the existing feature matrix. This preserves the same modeling approach/loop and semantics, but adds strong physics-informed predictors that usually reduce MAE substantially in CHAMPS. I also keep the exact submission format and ensure all new merged columns are NaN-safe and float32 for stability and runtime.'
- What this solution (achieved 27.32578) has done: 'Your current score (21.3531, lower-is-better) is still far worse than the target (-1.3535), so we need a real accuracy gain while keeping the same per-type linear-regression approach. The most impactful minimal change here is to standardize the numeric features (continuous + auxiliary) **within each coupling type** before fitting/predicting, because `LinearRegression` is very sensitive to wildly different feature scales (e.g., distance powers vs. charges/tensors) and this hurts MAE. I implement a lightweight per-type z-score normalization (fit mean/std on that type’s train rows, apply to that type’s test rows), leaving the model, loop structure, and features intact. I also make the `get_dummies` input explicitly string-typed to ensure stable categorical column naming across pandas versions, which avoids subtle misalignment issues and improves robustness without changing semantics.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_DIR_CANDIDATES = [
    "/kaggle/input/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling",
    "/kaggle/data/input/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling/champs-scalar-coupling",
    "../input/champs-scalar-coupling",
]


def find_data_dir():
    for d in DATA_DIR_CANDIDATES:
        if os.path.exists(d) and os.path.exists(os.path.join(d, "train.csv")):
            return d
    for root in ["/kaggle/input", "/kaggle/data", "../input"]:
        if os.path.exists(root):
            for dirpath, dirnames, filenames in os.walk(root):
                if (
                    "train.csv" in filenames
                    and "test.csv" in filenames
                    and "structures.csv" in filenames
                ):
                    return dirpath
    raise FileNotFoundError(
        "Could not locate champs-scalar-coupling data directory containing train.csv/test.csv/structures.csv"
    )


DATA_DIR = find_data_dir()
print("Using DATA_DIR:", DATA_DIR)
print("Files:", sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])[:10])



## === cell 1
import numpy as np  # linear algebra
import pandas as pd  # data processing
import os

for p in ["/kaggle/input", "/kaggle/data", "../input"]:
    if os.path.exists(p):
        print(p, "->", os.listdir(p)[:10])



## === cell 2
from sklearn.linear_model import LinearRegression

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
structures_path = os.path.join(DATA_DIR, "structures.csv")
sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

mulliken_path = os.path.join(DATA_DIR, "mulliken_charges.csv")
shield_path = os.path.join(DATA_DIR, "magnetic_shielding_tensors.csv")

usecols_train = [
    "id",
    "molecule_name",
    "atom_index_0",
    "atom_index_1",
    "type",
    "scalar_coupling_constant",
]
usecols_test = ["id", "molecule_name", "atom_index_0", "atom_index_1", "type"]

print("Loading CSVs...")
train = pd.read_csv(train_path, usecols=usecols_train)
test = pd.read_csv(test_path, usecols=usecols_test)
structures = pd.read_csv(
    structures_path, usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"]
)
sample_sub = pd.read_csv(sample_sub_path)

print("Shapes:", train.shape, test.shape, structures.shape, sample_sub.shape)
print("Train types:", train["type"].nunique(), "Test types:", test["type"].nunique())

print("Loading auxiliary atom-level features...")
mulliken = pd.read_csv(
    mulliken_path, usecols=["molecule_name", "atom_index", "mulliken_charge"]
)
shield = pd.read_csv(
    shield_path,
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

mulliken["mulliken_charge"] = mulliken["mulliken_charge"].astype(np.float32)
for c in ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]:
    shield[c] = shield[c].astype(np.float32)

atom_types = pd.Index(sorted(structures["atom"].unique()))
atom2int = {a: i for i, a in enumerate(atom_types)}
structures["atom_int"] = structures["atom"].map(atom2int).astype(np.int16)

s0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "atom_int": "atom_int_0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
    }
)[["molecule_name", "atom_index_0", "atom_0", "atom_int_0", "x0", "y0", "z0"]]

s1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "atom_int": "atom_int_1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
    }
)[["molecule_name", "atom_index_1", "atom_1", "atom_int_1", "x1", "y1", "z1"]]

m0 = mulliken.rename(
    columns={"atom_index": "atom_index_0", "mulliken_charge": "mulliken_charge_0"}
)[["molecule_name", "atom_index_0", "mulliken_charge_0"]]
m1 = mulliken.rename(
    columns={"atom_index": "atom_index_1", "mulliken_charge": "mulliken_charge_1"}
)[["molecule_name", "atom_index_1", "mulliken_charge_1"]]

sh0 = shield.rename(columns={"atom_index": "atom_index_0"}).add_suffix("_0")
sh0 = sh0.rename(
    columns={"molecule_name_0": "molecule_name", "atom_index_0_0": "atom_index_0"}
)
sh0 = sh0[
    ["molecule_name", "atom_index_0", "XX_0", "YY_0", "ZZ_0", "XY_0", "XZ_0", "YZ_0"]
]

sh1 = shield.rename(columns={"atom_index": "atom_index_1"}).add_suffix("_1")
sh1 = sh1.rename(
    columns={"molecule_name_1": "molecule_name", "atom_index_1_1": "atom_index_1"}
)
sh1 = sh1[
    ["molecule_name", "atom_index_1", "XX_1", "YY_1", "ZZ_1", "XY_1", "XZ_1", "YZ_1"]
]


def add_pair_features(df):
    df = df.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
    df = df.merge(s1, on=["molecule_name", "atom_index_1"], how="left")

    df = df.merge(m0, on=["molecule_name", "atom_index_0"], how="left")
    df = df.merge(m1, on=["molecule_name", "atom_index_1"], how="left")
    df = df.merge(sh0, on=["molecule_name", "atom_index_0"], how="left")
    df = df.merge(sh1, on=["molecule_name", "atom_index_1"], how="left")

    fill0_cols = ["x0", "y0", "z0", "x1", "y1", "z1", "atom_int_0", "atom_int_1"]
    for c in fill0_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    aux_cols = [
        "mulliken_charge_0",
        "mulliken_charge_1",
        "XX_0",
        "YY_0",
        "ZZ_0",
        "XY_0",
        "XZ_0",
        "YZ_0",
        "XX_1",
        "YY_1",
        "ZZ_1",
        "XY_1",
        "XZ_1",
        "YZ_1",
    ]
    for c in aux_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(np.float32)

    dx = (df["x0"] - df["x1"]).astype(np.float32)
    dy = (df["y0"] - df["y1"]).astype(np.float32)
    dz = (df["z0"] - df["z1"]).astype(np.float32)

    dist2 = (dx * dx + dy * dy + dz * dz).astype(np.float32)
    dist = np.sqrt(dist2).astype(np.float32)
    df["dist"] = dist
    df["dist2"] = dist2

    df["abs_dx"] = np.abs(dx).astype(np.float32)
    df["abs_dy"] = np.abs(dy).astype(np.float32)
    df["abs_dz"] = np.abs(dz).astype(np.float32)

    inv_dist = (1.0 / (dist + 1e-3)).astype(np.float32)
    df["inv_dist"] = inv_dist
    df["inv_dist2"] = (1.0 / (dist2 + 1e-6)).astype(np.float32)

    df["dot"] = (
        df["x0"].astype(np.float32) * df["x1"].astype(np.float32)
        + df["y0"].astype(np.float32) * df["y1"].astype(np.float32)
        + df["z0"].astype(np.float32) * df["z1"].astype(np.float32)
    ).astype(np.float32)

    df["dx"] = dx
    df["dy"] = dy
    df["dz"] = dz
    df["dist3"] = (dist2 * dist).astype(np.float32)
    df["dist4"] = (dist2 * dist2).astype(np.float32)
    mx = ((df["x0"].astype(np.float32) + df["x1"].astype(np.float32)) * 0.5).astype(
        np.float32
    )
    my = ((df["y0"].astype(np.float32) + df["y1"].astype(np.float32)) * 0.5).astype(
        np.float32
    )
    mz = ((df["z0"].astype(np.float32) + df["z1"].astype(np.float32)) * 0.5).astype(
        np.float32
    )
    df["mid_r"] = np.sqrt(mx * mx + my * my + mz * mz).astype(np.float32)

    df["atom_int_0"] = df["atom_int_0"].astype(np.int16)
    df["atom_int_1"] = df["atom_int_1"].astype(np.int16)

    a0 = df["atom_int_0"].astype(np.int32)
    a1 = df["atom_int_1"].astype(np.int32)
    amin = np.minimum(a0, a1)
    amax = np.maximum(a0, a1)
    df["atom_pair_sorted"] = (amin * 100 + amax).astype(np.int32)

    df["atom_int_prod"] = (a0 * a1).astype(np.int32)

    df["mulliken_diff"] = (df["mulliken_charge_0"] - df["mulliken_charge_1"]).astype(
        np.float32
    )
    df["mulliken_sum"] = (df["mulliken_charge_0"] + df["mulliken_charge_1"]).astype(
        np.float32
    )

    df["shield_trace_0"] = (df["XX_0"] + df["YY_0"] + df["ZZ_0"]).astype(np.float32)
    df["shield_trace_1"] = (df["XX_1"] + df["YY_1"] + df["ZZ_1"]).astype(np.float32)
    df["shield_trace_diff"] = (df["shield_trace_0"] - df["shield_trace_1"]).astype(
        np.float32
    )
    df["shield_trace_sum"] = (df["shield_trace_0"] + df["shield_trace_1"]).astype(
        np.float32
    )

    sh0_vec = np.stack(
        [df["XY_0"].to_numpy(), df["XZ_0"].to_numpy(), df["YZ_0"].to_numpy()], axis=1
    ).astype(np.float32)
    sh1_vec = np.stack(
        [df["XY_1"].to_numpy(), df["XZ_1"].to_numpy(), df["YZ_1"].to_numpy()], axis=1
    ).astype(np.float32)
    df["shield_offdiag_norm0"] = np.sqrt((sh0_vec * sh0_vec).sum(axis=1)).astype(
        np.float32
    )
    df["shield_offdiag_norm1"] = np.sqrt((sh1_vec * sh1_vec).sum(axis=1)).astype(
        np.float32
    )
    df["shield_offdiag_dot"] = (sh0_vec * sh1_vec).sum(axis=1).astype(np.float32)

    return df


print("Adding features...")
train_f = add_pair_features(train)
test_f = add_pair_features(test)

all_types = pd.Index(sorted(pd.concat([train_f["type"], test_f["type"]]).unique()))
type2int = {t: i for i, t in enumerate(all_types)}
train_f["type_int"] = train_f["type"].map(type2int).astype(np.int16)
test_f["type_int"] = test_f["type"].map(type2int).astype(np.int16)

cont_feature_cols = [
    "dist",
    "dist2",
    "dist3",
    "dist4",
    "dx",
    "dy",
    "dz",
    "abs_dx",
    "abs_dy",
    "abs_dz",
    "inv_dist",
    "inv_dist2",
    "dot",
    "mid_r",
    "mulliken_charge_0",
    "mulliken_charge_1",
    "mulliken_diff",
    "mulliken_sum",
    "shield_trace_0",
    "shield_trace_1",
    "shield_trace_diff",
    "shield_trace_sum",
    "shield_offdiag_norm0",
    "shield_offdiag_norm1",
    "shield_offdiag_dot",
]
train_f[cont_feature_cols] = train_f[cont_feature_cols].fillna(0)
test_f[cont_feature_cols] = test_f[cont_feature_cols].fillna(0)

print("NaN counts (train, cont):", train_f[cont_feature_cols].isna().sum().to_dict())
print("NaN counts (test, cont):", test_f[cont_feature_cols].isna().sum().to_dict())



## === cell 3
pred = np.zeros(len(test_f), dtype=np.float32)

global_mean_by_type = (
    train_f.groupby("type")["scalar_coupling_constant"].mean().to_dict()
)
overall_mean = float(train_f["scalar_coupling_constant"].mean())

cat_cols_to_ohe = ["atom_int_0", "atom_int_1", "atom_pair_sorted"]
cat_numeric_cols = ["atom_int_prod", "type_int"]

for t, test_idx in test_f.groupby("type").groups.items():
    train_idx = train_f.index[train_f["type"] == t]

    if len(train_idx) < 50:
        fill_val = global_mean_by_type.get(t, overall_mean)
        pred[test_idx] = fill_val
        continue

    Xtr_cont = train_f.loc[train_idx, cont_feature_cols].astype(np.float32)
    Xte_cont = test_f.loc[test_idx, cont_feature_cols].astype(np.float32)

    Xtr_catn = train_f.loc[train_idx, cat_numeric_cols].astype(np.float32)
    Xte_catn = test_f.loc[test_idx, cat_numeric_cols].astype(np.float32)

    Xtr_num = pd.concat([Xtr_cont, Xtr_catn], axis=1)
    Xte_num = pd.concat([Xte_cont, Xte_catn], axis=1)

    mu = Xtr_num.mean(axis=0)
    sigma = Xtr_num.std(axis=0).replace(0.0, 1.0)

    Xtr_num = ((Xtr_num - mu) / sigma).astype(np.float32)
    Xte_num = ((Xte_num - mu) / sigma).astype(np.float32)

    Xtr_ohe = pd.get_dummies(
        train_f.loc[train_idx, cat_cols_to_ohe].astype(np.int32).astype(str),
        columns=cat_cols_to_ohe,
        dummy_na=False,
    )
    Xte_ohe = pd.get_dummies(
        test_f.loc[test_idx, cat_cols_to_ohe].astype(np.int32).astype(str),
        columns=cat_cols_to_ohe,
        dummy_na=False,
    )

    Xtr_ohe, Xte_ohe = Xtr_ohe.align(Xte_ohe, join="outer", axis=1, fill_value=0)

    X_train = pd.concat(
        [
            Xtr_num.reset_index(drop=True),
            Xtr_ohe.reset_index(drop=True),
        ],
        axis=1,
    ).to_numpy(dtype=np.float32)

    X_test = pd.concat(
        [
            Xte_num.reset_index(drop=True),
            Xte_ohe.reset_index(drop=True),
        ],
        axis=1,
    ).to_numpy(dtype=np.float32)

    y_train = train_f.loc[train_idx, "scalar_coupling_constant"].to_numpy(
        dtype=np.float32
    )

    model = LinearRegression(n_jobs=None)
    model.fit(X_train, y_train)
    pred[test_idx] = model.predict(X_test).astype(np.float32)

submission = test_f[["id"]].copy()
submission["scalar_coupling_constant"] = pred.astype(np.float32)

submission = submission.sort_values("id").reset_index(drop=True)

assert submission.shape[0] == test.shape[0], "Submission row count mismatch"
assert submission.columns.tolist() == [
    "id",
    "scalar_coupling_constant",
], "Submission columns mismatch"

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote", out_path, "with shape", submission.shape)
print(submission.head())



## === cell 4
desc = submission["scalar_coupling_constant"].describe()
print(desc)
