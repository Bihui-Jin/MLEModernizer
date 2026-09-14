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

0.3023773260272986

# 6. Current score

1.18838

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.18838) has done: 'The code currently depends on four external Kaggle datasets (`champs-scalar-coupling-*-...`) that are not present in your environment, causing the initial `FileNotFoundError` and preventing any submission from being written. To keep the core approach intact (a weighted blend optimized by random search), I replace those missing inputs with four simple, deterministic baseline “models” trained directly from the provided CHAMPS training data (type-wise mean, type-wise median, and two shrinkage variants). I also fix the missing cell numbering and remove the notebook-only `%%time` magic so the script runs as a plain Python kernel. Finally, the script writes a valid `submission.csv` with the required `id,scalar_coupling_constant` columns.'
- What this solution (achieved 1.18838) has done: 'Your current score (1.18838, lower-is-better) is far from the target (0.30238), so we should legitimately improve the model while keeping your “type-based blending” core logic intact. The biggest gain with minimal semantic change is to avoid training-set leakage in the weight search by using a molecule-wise validation split (the competition splits by molecule), then fit the type statistics on train-fold only and optimize blend weights on the held-out fold. This keeps the same four base predictors (type mean/median + two shrinkage means) and the same random-search blending approach, but makes the chosen weights generalize better and thus should move the score substantially toward the target. The code below implements that split, recalculates mappings on the training fold, selects weights on the validation fold, then rebuilds mappings on full training data and writes `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import sys
import numpy as np
import pandas as pd



## === cell 1
SEED = 31
TRIALS = 200
TARGET = "scalar_coupling_constant"
PREDICTION = "pred"

DATA_DIR_CANDIDATES = [
    "/kaggle/data/champs-scalar-coupling",
    "/kaggle/input/champs-scalar-coupling",
    "/kaggle/data/input/champs-scalar-coupling",
    "/kaggle/input",
    "/kaggle/data",
]


def find_data_dir():
    for d in DATA_DIR_CANDIDATES:
        if os.path.exists(os.path.join(d, "train.csv")) and os.path.exists(
            os.path.join(d, "test.csv")
        ):
            return d
    if os.path.exists("train.csv") and os.path.exists("test.csv"):
        return "."
    raise FileNotFoundError(
        "Could not find train.csv/test.csv in expected Kaggle paths. "
        f"Tried: {DATA_DIR_CANDIDATES} and current directory."
    )


DATA_DIR = find_data_dir()
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

print("Using DATA_DIR =", DATA_DIR)




## === cell 2
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


seed_everything(SEED)




## === cell 3
def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    """
    Fast metric computation for CHAMPS scalar coupling.
    """
    maes = (y_true - y_pred).abs().groupby(types).mean()
    maes = np.log(maes.map(lambda x: max(x, floor)))
    return maes.mean()




## === cell 4
usecols_train = ["id", "molecule_name", "type", TARGET]
usecols_test = ["id", "type"]

train = pd.read_csv(TRAIN_PATH, usecols=usecols_train)
test = pd.read_csv(TEST_PATH, usecols=usecols_test)

train["type"] = train["type"].astype("category")
test["type"] = test["type"].astype("category")

print("Loaded:", train.shape, test.shape)
print("Unique molecules (train):", train["molecule_name"].nunique())




## === cell 5
def molecule_holdout_split(df, valid_frac=0.10, seed=SEED):
    mols = df["molecule_name"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(mols)
    n_valid = max(1, int(len(mols) * valid_frac))
    valid_mols = set(mols[:n_valid])
    is_valid = df["molecule_name"].isin(valid_mols)
    return df[~is_valid].copy(), df[is_valid].copy()


train_tr, train_va = molecule_holdout_split(train, valid_frac=0.10, seed=SEED)
print("Train fold:", train_tr.shape, "Valid fold:", train_va.shape)
print(
    "Molecule overlap:",
    len(set(train_tr["molecule_name"]).intersection(set(train_va["molecule_name"]))),
)




## === cell 6
def build_type_mappings(train_df, k1=25.0, k2=100.0):
    type_mean = train_df.groupby("type")[TARGET].mean()
    type_median = train_df.groupby("type")[TARGET].median()
    global_mean = float(train_df[TARGET].mean())

    type_count = train_df.groupby("type")[TARGET].size().astype(float)
    shrink_mean_k1 = (type_mean * type_count + global_mean * k1) / (type_count + k1)
    shrink_mean_k2 = (type_mean * type_count + global_mean * k2) / (type_count + k2)

    return {
        "type_mean": type_mean,
        "type_median": type_median,
        "shrink_k1": shrink_mean_k1,
        "shrink_k2": shrink_mean_k2,
        "global_mean": global_mean,
    }


def build_pred_df(base, mapping, global_mean):
    df = base[["id", "type"]].copy()
    df[PREDICTION] = df["type"].map(mapping).astype(float)
    df[PREDICTION] = df[PREDICTION].fillna(global_mean)
    return df


def build_train_pred_df(base, mapping, global_mean):
    df = base[["id", "type", TARGET]].copy()
    df[PREDICTION] = df["type"].map(mapping).astype(float)
    df[PREDICTION] = df[PREDICTION].fillna(global_mean)
    return df


maps_tr = build_type_mappings(train_tr, k1=25.0, k2=100.0)
gm_tr = maps_tr["global_mean"]

va_m0 = build_train_pred_df(train_va, maps_tr["type_mean"], gm_tr)
va_m1 = build_train_pred_df(train_va, maps_tr["type_median"], gm_tr)
va_m2 = build_train_pred_df(train_va, maps_tr["shrink_k1"], gm_tr)
va_m3 = build_train_pred_df(train_va, maps_tr["shrink_k2"], gm_tr)

va_m0 = va_m0.sort_values("id").reset_index(drop=True)
va_m1 = va_m1.sort_values("id").reset_index(drop=True)
va_m2 = va_m2.sort_values("id").reset_index(drop=True)
va_m3 = va_m3.sort_values("id").reset_index(drop=True)

assert (va_m0["id"].values == va_m1["id"].values).all()
assert (va_m0["id"].values == va_m2["id"].values).all()
assert (va_m0["id"].values == va_m3["id"].values).all()

valid_sets = [va_m0, va_m1, va_m2, va_m3]

print("Prepared validation prediction sets:", [df.shape for df in valid_sets])




## === cell 7
def weights(n, min_weight=0.01, max_weight=0.99):
    if n < 1:
        raise ValueError("n must not be less than 1")
    res = []
    remainder = 1.0
    for _ in range(n - 1):
        w = random.uniform(min_weight, max_weight) * remainder
        res.append(w)
        remainder -= w
    res.append(remainder)
    return res


def trial(train_sets, prediction_column, target_column):
    ws = weights(len(train_sets))
    df = train_sets[0][["id", "type", target_column]].copy()
    df[prediction_column] = 0.0
    for i, t in enumerate(train_sets):
        df[prediction_column] += t[prediction_column].astype(float).values * ws[i]
    score = group_mean_log_mae(df[target_column], df[prediction_column], df["type"])
    return float(score), ws


best = sys.maxsize
best_weights = None
for i in range(TRIALS):
    score, ws = trial(
        train_sets=valid_sets, prediction_column=PREDICTION, target_column=TARGET
    )
    if score < best:
        best = score
        best_weights = ws

print(f"best(valid)={best:.6f}")
print(
    f"""best weights (sum={sum(best_weights):.6f})
  model0_type_mean       ={best_weights[0]:.6f}
  model1_type_median     ={best_weights[1]:.6f}
  model2_shrink_mean_k25 ={best_weights[2]:.6f}
  model3_shrink_mean_k100={best_weights[3]:.6f}
"""
)



## === cell 8
maps_full = build_type_mappings(train, k1=25.0, k2=100.0)
global_mean = maps_full["global_mean"]

test_m0 = (
    build_pred_df(test, maps_full["type_mean"], global_mean)
    .sort_values("id")
    .reset_index(drop=True)
)
test_m1 = (
    build_pred_df(test, maps_full["type_median"], global_mean)
    .sort_values("id")
    .reset_index(drop=True)
)
test_m2 = (
    build_pred_df(test, maps_full["shrink_k1"], global_mean)
    .sort_values("id")
    .reset_index(drop=True)
)
test_m3 = (
    build_pred_df(test, maps_full["shrink_k2"], global_mean)
    .sort_values("id")
    .reset_index(drop=True)
)

assert (test_m0["id"].values == test_m1["id"].values).all()
assert (test_m0["id"].values == test_m2["id"].values).all()
assert (test_m0["id"].values == test_m3["id"].values).all()

test_sets = [
    test_m0.rename(columns={PREDICTION: TARGET}),
    test_m1.rename(columns={PREDICTION: TARGET}),
    test_m2.rename(columns={PREDICTION: TARGET}),
    test_m3.rename(columns={PREDICTION: TARGET}),
]

submission = test_sets[0][["id"]].copy()
submission[TARGET] = 0.0
for i, t in enumerate(test_sets):
    submission[TARGET] += t[TARGET].astype(float).values * best_weights[i]

if os.path.exists(SAMPLE_SUB_PATH):
    sample = pd.read_csv(SAMPLE_SUB_PATH, usecols=["id"])
    submission = sample.merge(submission, on="id", how="left")
    submission[TARGET] = submission[TARGET].fillna(global_mean)

print(submission.head())
print(submission.shape)



## === cell 9
out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote", out_path, "with shape", submission.shape)
print(submission.dtypes)
print("Files in working dir:", os.listdir("."))
