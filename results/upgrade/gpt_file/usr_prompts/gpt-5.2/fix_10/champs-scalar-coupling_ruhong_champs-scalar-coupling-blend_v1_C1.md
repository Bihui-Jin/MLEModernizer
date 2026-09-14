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

3.93505

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.18838) has done: 'The code currently depends on four external Kaggle datasets (`champs-scalar-coupling-*-...`) that are not present in your environment, causing the initial `FileNotFoundError` and preventing any submission from being written. To keep the core approach intact (a weighted blend optimized by random search), I replace those missing inputs with four simple, deterministic baseline “models” trained directly from the provided CHAMPS training data (type-wise mean, type-wise median, and two shrinkage variants). I also fix the missing cell numbering and remove the notebook-only `%%time` magic so the script runs as a plain Python kernel. Finally, the script writes a valid `submission.csv` with the required `id,scalar_coupling_constant` columns.'
- What this solution (achieved 1.18838) has done: 'Your current score (1.18838, lower-is-better) is far from the target (0.30238), so we should legitimately improve the model while keeping your “type-based blending” core logic intact. The biggest gain with minimal semantic change is to avoid training-set leakage in the weight search by using a molecule-wise validation split (the competition splits by molecule), then fit the type statistics on train-fold only and optimize blend weights on the held-out fold. This keeps the same four base predictors (type mean/median + two shrinkage means) and the same random-search blending approach, but makes the chosen weights generalize better and thus should move the score substantially toward the target. The code below implements that split, recalculates mappings on the training fold, selects weights on the validation fold, then rebuilds mappings on full training data and writes `submission.csv`.'
- What this solution (achieved 1.20105) has done: 'Your current score (1.18838, lower-is-better) is still far from the target (0.30238), so the smallest meaningful improvement is to keep your type-based blending core intact but make the base predictors noticeably stronger with minimal extra features. I add two additional per-type baseline models derived from the training labels: a molecule-aware “mean of (atom0,atom1) element-pair means” using `structures.csv`, and a simple distance-based correction (fit linear regression of target vs inter-atomic distance per coupling type) using `structures.csv`. These additions preserve your overall approach (deterministic feature aggregation → a few base predictors → random-search blend on molecule-holdout validation) while typically reducing MAE substantially for CHAMPS. The script still writes a valid `submission.csv` with `id,scalar_coupling_constant`, keeps the molecule-wise split, and stays within Kaggle constraints.'
- What this solution (achieved 1.23115) has done: 'Your current score (1.20105, lower-is-better) is far from the target (0.30238), so we should improve generalization while keeping your existing “few deterministic base predictors + random-search blending on a molecule-holdout validation” core intact. The biggest issue is that the two added base predictors (atom-pair mean and type-wise distance linear) can overfit sparse atom-pairs and noisy distances; adding small, deterministic shrinkage to these mappings and clipping distance outliers per type improves robustness without changing the approach. I (1) shrink the atom-pair mean toward the type mean using a count-based prior, (2) fit the distance linear model with a tiny ridge term and train-time winsorization of distances per type, and (3) increase the random-search trials modestly (still fast) to better exploit the improved bases. These changes keep the same evaluation semantics and produce the same `submission.csv` format.'
- What this solution (achieved 2.70823) has done: 'Your current score (1.23115, lower-is-better) is still far from the target (0.30238), so we should improve predictive strength without changing your core “deterministic baselines + random-search blending with molecule-wise holdout” approach. The biggest missing signal in your current features is that couplings are extremely dependent on the chemical environment, and you already have strong per-atom signals available (Mulliken charges and magnetic shielding tensors) in the provided dataset. I add two additional base predictors that use only deterministic aggregations of those provided files: per-type linear models on (distance + mulliken charges) and on (distance + shielding tensor summaries), fit separately per coupling type with tiny ridge (same as your current distance-linear base). Then we keep the same validation split and the same random-search blending, just with a slightly larger model set and a small increase in trials to let blending exploit the new bases.'
- What this solution (achieved 3.93505) has done: 'Your current score (2.70823, lower-is-better) is far worse than the target (0.30238), so we should fix the most likely correctness bug that can severely degrade performance without changing the overall “deterministic base predictors + type-wise models + random-search blending” approach. The main issue is that your weight search uses `random.uniform(min_weight,max_weight)*remainder`, which biases weights toward early models and prevents fair exploration; replacing it with an unbiased Dirichlet draw keeps the same blending logic but finds much better weights. I also fix a subtle bug in `build_atompair_mean_mapping()` where the type-mean alignment for each `(type, atom_pair)` is incorrect, which can badly corrupt that base predictor and thus the blend. These two minimal fixes should move the score substantially toward the target while keeping the same models, data, metric, and submission semantics.'
- What this solution (achieved 3.93505) has done: 'Your current score is far worse than the target (lower is better), so the smallest likely “correctness” fix that can yield a big improvement is to make sure the validation setup matches the competition metric: GroupMeanLogMAE is computed per coupling `type`, and your weight search currently uses only one random molecule-holdout split that can be noisy and mislead the blend. I keep your exact base models and blending logic, but switch the weight search to use two different molecule-holdout folds and minimize the average validation metric across them (more stable weights, better generalization). I also ensure the `type` dtype is consistent (`category` with shared categories) and that the MultiIndex reindex in the atompair model uses the same `type` values (not string-cast) to avoid silent mismatches that can badly degrade that predictor. These are minimal changes that preserve core semantics while addressing instability/misalignment that can explain the regression to 3.93.'
- What this solution (achieved 3.93505) has done: 'Your current score is much worse than the target (lower-is-better), so we need a correctness-level fix rather than tuning. The biggest issue is that you’re loading **structures/mulliken/shielding for both train+test molecules**, which is extremely heavy and can cause silent timeouts/OOM/partial execution in Kaggle and lead to broken/degenerate predictions; we instead load/merge those auxiliary tables **only for the molecules needed per fold** (train-fold + valid-fold) and for final training (train+test separately). To keep core logic identical, all models and the blending procedure remain the same; we only change data-loading scope, enforce consistent categorical `type` handling, and ensure the atompair MultiIndex uses the same dtype as the mapping index. These minimal changes should restore “intended” feature joins and stabilize the blend, moving the score back toward your earlier ~1.x range (closer to the 0.30 target than 3.93).'
- What this solution (achieved 3.93505) has done: 'Your current score (3.935, lower-is-better) indicates a correctness failure rather than a small modeling gap; the most likely culprit here is inconsistent `type` handling: using `Categorical` in some places and raw `.values` (categorical codes) in others can break all type-wise mappings and degrade predictions massively. I make `group_mean_log_mae`, all type-mapping/prediction functions, and the atompair MultiIndex construction explicitly operate on the *string labels* for `type` (not categorical codes), while keeping the exact same models and blending logic. I also ensure both folds and final train/test share the same `type` categories and that every place we iterate/check membership uses the label form, which should restore the intended joins and move the score back toward your earlier ~1.x range (closer to the 0.30 target) without changing the approach. No training loops, model definitions, or feature sets are changed—only dtype alignment and safe conversions.'

# 9. Code solution

## === cell 0
import os
import random
import sys
import numpy as np
import pandas as pd



## === cell 1
SEED = 31
TRIALS = (
    1000  # keep trials as-is; we'll stabilize data joins instead of increasing trials
)

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
STRUCTURES_PATH = os.path.join(DATA_DIR, "structures.csv")
MULLIKEN_PATH = os.path.join(DATA_DIR, "mulliken_charges.csv")
SHIELD_PATH = os.path.join(DATA_DIR, "magnetic_shielding_tensors.csv")

print("Using DATA_DIR =", DATA_DIR)
print("TRAIN_PATH =", TRAIN_PATH)
print("TEST_PATH  =", TEST_PATH)
print(
    "STRUCTURES_PATH =", STRUCTURES_PATH, "| exists:", os.path.exists(STRUCTURES_PATH)
)
print("MULLIKEN_PATH   =", MULLIKEN_PATH, "| exists:", os.path.exists(MULLIKEN_PATH))
print("SHIELD_PATH     =", SHIELD_PATH, "| exists:", os.path.exists(SHIELD_PATH))




## === cell 2
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


seed_everything(SEED)




## === cell 3
def _type_labels(x):
    if isinstance(x, pd.Series) and isinstance(x.dtype, pd.CategoricalDtype):
        return x.astype(str)
    if isinstance(x, pd.Categorical):
        return pd.Series(x.astype(str))
    if isinstance(x, (pd.Series, pd.Index)):
        return x.astype(str)
    return pd.Series(x).astype(str)


def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    t = _type_labels(types)
    maes = (y_true - y_pred).abs().groupby(t).mean()
    maes = np.log(maes.map(lambda x: max(float(x), floor)))
    return float(maes.mean())




## === cell 4
usecols_train = ["id", "molecule_name", "atom_index_0", "atom_index_1", "type", TARGET]
usecols_test = ["id", "molecule_name", "atom_index_0", "atom_index_1", "type"]

train = pd.read_csv(TRAIN_PATH, usecols=usecols_train)
test = pd.read_csv(TEST_PATH, usecols=usecols_test)

all_types = pd.Index(
    pd.concat([train["type"], test["type"]], axis=0).astype(str).unique()
)
train["type"] = pd.Categorical(train["type"].astype(str), categories=all_types)
test["type"] = pd.Categorical(test["type"].astype(str), categories=all_types)

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
train_tr2, train_va2 = molecule_holdout_split(train, valid_frac=0.10, seed=SEED + 997)

print("Fold1 Train:", train_tr.shape, "Valid:", train_va.shape)
print("Fold2 Train:", train_tr2.shape, "Valid:", train_va2.shape)




## === cell 6
def build_type_mappings(train_df, k1=25.0, k2=100.0):
    t = _type_labels(train_df["type"])
    type_mean = train_df[TARGET].groupby(t).mean()
    type_median = train_df[TARGET].groupby(t).median()
    global_mean = float(train_df[TARGET].mean())

    type_count = train_df[TARGET].groupby(t).size().astype(float)
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
    df[PREDICTION] = _type_labels(df["type"]).map(mapping).astype(float)
    df[PREDICTION] = df[PREDICTION].fillna(global_mean)
    return df


def build_train_pred_df(base, mapping, global_mean):
    df = base[["id", "type", TARGET]].copy()
    df[PREDICTION] = _type_labels(df["type"]).map(mapping).astype(float)
    df[PREDICTION] = df[PREDICTION].fillna(global_mean)
    return df




## === cell 7
def load_structures_needed(structures_path, mol_names):
    usecols = ["molecule_name", "atom_index", "atom", "x", "y", "z"]
    st = pd.read_csv(structures_path, usecols=usecols)
    st = st[st["molecule_name"].isin(mol_names)].copy()
    st["atom_index"] = st["atom_index"].astype(np.int32)
    return st


def attach_atoms_and_distance(df, structures_df):
    s0 = structures_df.rename(
        columns={
            "atom_index": "atom_index_0",
            "atom": "atom_0",
            "x": "x0",
            "y": "y0",
            "z": "z0",
        }
    )
    out = df.merge(
        s0[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
    )

    s1 = structures_df.rename(
        columns={
            "atom_index": "atom_index_1",
            "atom": "atom_1",
            "x": "x1",
            "y": "y1",
            "z": "z1",
        }
    )
    out = out.merge(
        s1[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
    )

    dx = out["x0"].values - out["x1"].values
    dy = out["y0"].values - out["y1"].values
    dz = out["z0"].values - out["z1"].values
    out["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)

    a0 = out["atom_0"].astype(str)
    a1 = out["atom_1"].astype(str)
    out["atom_pair"] = np.where(a0 <= a1, a0 + "_" + a1, a1 + "_" + a0)

    return out


def winsorize_dist_per_type(train_ref_df, df_to_clip, lo_q=0.01, hi_q=0.99):
    qs = (
        train_ref_df.groupby(_type_labels(train_ref_df["type"]))["dist"]
        .quantile([lo_q, hi_q])
        .unstack()
    )
    qs.columns = ["lo", "hi"]
    out = df_to_clip.copy()
    lo = _type_labels(out["type"]).map(qs["lo"]).astype(float)
    hi = _type_labels(out["type"]).map(qs["hi"]).astype(float)
    lo = lo.fillna(float(train_ref_df["dist"].quantile(lo_q)))
    hi = hi.fillna(float(train_ref_df["dist"].quantile(hi_q)))
    out["dist"] = np.clip(
        out["dist"].astype(float).to_numpy(), lo.to_numpy(), hi.to_numpy()
    ).astype(np.float32)
    return out


fold1_mols = set(
    pd.concat([train_tr["molecule_name"], train_va["molecule_name"]]).unique()
)
fold2_mols = set(
    pd.concat([train_tr2["molecule_name"], train_va2["molecule_name"]]).unique()
)
final_train_mols = set(train["molecule_name"].unique())
final_test_mols = set(test["molecule_name"].unique())

structures_f1 = load_structures_needed(STRUCTURES_PATH, fold1_mols)
structures_f2 = load_structures_needed(STRUCTURES_PATH, fold2_mols)
structures_train = load_structures_needed(STRUCTURES_PATH, final_train_mols)
structures_test = load_structures_needed(STRUCTURES_PATH, final_test_mols)

print("Loaded structures fold1:", structures_f1.shape)
print("Loaded structures fold2:", structures_f2.shape)
print("Loaded structures train:", structures_train.shape)
print("Loaded structures test :", structures_test.shape)

train_tr_f = attach_atoms_and_distance(train_tr, structures_f1)
train_va_f = attach_atoms_and_distance(train_va, structures_f1)

train_tr2_f = attach_atoms_and_distance(train_tr2, structures_f2)
train_va2_f = attach_atoms_and_distance(train_va2, structures_f2)

train_f = attach_atoms_and_distance(train, structures_train)
test_f = attach_atoms_and_distance(test, structures_test)

for df_name, df_ in [
    ("train_tr_f", train_tr_f),
    ("train_va_f", train_va_f),
    ("train_tr2_f", train_tr2_f),
    ("train_va2_f", train_va2_f),
    ("train_f", train_f),
    ("test_f", test_f),
]:
    if df_["dist"].isna().any():
        df_["dist"] = df_["dist"].fillna(df_["dist"].median())
        print(df_name, "had missing dist; filled with median.")

train_tr_f = winsorize_dist_per_type(train_tr_f, train_tr_f, 0.01, 0.99)
train_va_f = winsorize_dist_per_type(train_tr_f, train_va_f, 0.01, 0.99)

train_tr2_f = winsorize_dist_per_type(train_tr2_f, train_tr2_f, 0.01, 0.99)
train_va2_f = winsorize_dist_per_type(train_tr2_f, train_va2_f, 0.01, 0.99)

train_f = winsorize_dist_per_type(train_f, train_f, 0.01, 0.99)
test_f = winsorize_dist_per_type(train_f, test_f, 0.01, 0.99)




## === cell 8
def build_atompair_mean_mapping(train_df, k_pair=50.0):
    global_mean = float(train_df[TARGET].mean())
    type_mean = train_df.groupby(_type_labels(train_df["type"]))[TARGET].mean()

    tmp = train_df.copy()
    tmp["type_lbl"] = _type_labels(tmp["type"])
    pair_mean = tmp.groupby(["type_lbl", "atom_pair"])[TARGET].mean()
    pair_cnt = tmp.groupby(["type_lbl", "atom_pair"])[TARGET].size().astype(float)

    type_mean_for_pair = (
        pair_mean.index.get_level_values(0).map(type_mean).astype(float)
    )

    pair_shrunk = (pair_mean * pair_cnt + type_mean_for_pair * k_pair) / (
        pair_cnt + k_pair
    )
    pair_shrunk.index = pair_shrunk.index.set_names(["type", "atom_pair"])
    return global_mean, type_mean, pair_shrunk


def predict_atompair_mean(df, global_mean, type_mean, pair_mean):
    out = df[["id", "type"]].copy()
    idx = pd.MultiIndex.from_arrays(
        [_type_labels(df["type"]).values, df["atom_pair"].astype(str).values],
        names=["type", "atom_pair"],
    )
    vals = pair_mean.reindex(idx).to_numpy()
    tmean = _type_labels(df["type"]).map(type_mean).astype(float).to_numpy()
    pred = np.where(np.isfinite(vals), vals, tmean)
    pred = np.where(np.isfinite(pred), pred, global_mean)
    out[PREDICTION] = pred.astype(float)
    return out


def fit_typewise_dist_linear(train_df, ridge=1e-3):
    global_mean = float(train_df[TARGET].mean())
    type_mean = train_df.groupby(_type_labels(train_df["type"]))[TARGET].mean()

    params = {}
    tmp = train_df.copy()
    tmp["type_lbl"] = _type_labels(tmp["type"])
    for t, g in tmp.groupby("type_lbl"):
        x = g["dist"].astype(float).to_numpy()
        y = g[TARGET].astype(float).to_numpy()
        if len(x) < 2 or not np.isfinite(x).all():
            params[t] = (float(type_mean.loc[t]), 0.0)
            continue
        x_mean = float(x.mean())
        y_mean = float(y.mean())
        denom = float(((x - x_mean) ** 2).sum()) + float(ridge)
        b = float(((x - x_mean) * (y - y_mean)).sum() / denom)
        a = y_mean - b * x_mean
        params[t] = (a, b)
    return global_mean, type_mean, params


def predict_typewise_dist_linear(df, global_mean, type_mean, params):
    out = df[["id", "type"]].copy()
    t_vals = _type_labels(df["type"]).values
    x = df["dist"].astype(float).to_numpy()
    a = np.empty(len(df), dtype=np.float64)
    b = np.empty(len(df), dtype=np.float64)
    for i, tv in enumerate(t_vals):
        if tv in params:
            ai, bi = params[tv]
        else:
            ai = float(type_mean.get(tv, global_mean))
            bi = 0.0
        a[i] = ai
        b[i] = bi
    pred = a + b * x
    pred = np.where(
        np.isfinite(pred),
        pred,
        _type_labels(df["type"]).map(type_mean).astype(float).to_numpy(),
    )
    pred = np.where(np.isfinite(pred), pred, global_mean)
    out[PREDICTION] = pred.astype(float)
    return out


def load_mulliken_needed(path, mol_names):
    usecols = ["molecule_name", "atom_index", "mulliken_charge"]
    mc = pd.read_csv(path, usecols=usecols)
    mc = mc[mc["molecule_name"].isin(mol_names)].copy()
    mc["atom_index"] = mc["atom_index"].astype(np.int32)
    mc["mulliken_charge"] = mc["mulliken_charge"].astype(np.float32)
    return mc


def load_shielding_needed(path, mol_names):
    usecols = [
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
    ]
    sh = pd.read_csv(path, usecols=usecols)
    sh = sh[sh["molecule_name"].isin(mol_names)].copy()
    sh["atom_index"] = sh["atom_index"].astype(np.int32)
    for c in ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]:
        sh[c] = sh[c].astype(np.float32)
    sh["shield_trace"] = (sh["XX"] + sh["YY"] + sh["ZZ"]).astype(np.float32)
    sh["shield_abs_sum"] = (
        sh[["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]]
        .abs()
        .sum(axis=1)
        .astype(np.float32)
    )
    return sh[["molecule_name", "atom_index", "shield_trace", "shield_abs_sum"]]


def attach_mulliken(df, mulliken_df):
    m0 = mulliken_df.rename(
        columns={"atom_index": "atom_index_0", "mulliken_charge": "q0"}
    )
    m1 = mulliken_df.rename(
        columns={"atom_index": "atom_index_1", "mulliken_charge": "q1"}
    )
    out = df.merge(
        m0[["molecule_name", "atom_index_0", "q0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    out = out.merge(
        m1[["molecule_name", "atom_index_1", "q1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
    )
    return out


def attach_shielding(df, shielding_df):
    s0 = shielding_df.rename(
        columns={
            "atom_index": "atom_index_0",
            "shield_trace": "st0",
            "shield_abs_sum": "sa0",
        }
    )
    s1 = shielding_df.rename(
        columns={
            "atom_index": "atom_index_1",
            "shield_trace": "st1",
            "shield_abs_sum": "sa1",
        }
    )
    out = df.merge(
        s0[["molecule_name", "atom_index_0", "st0", "sa0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    out = out.merge(
        s1[["molecule_name", "atom_index_1", "st1", "sa1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
    )
    return out


def fit_typewise_linear_multi(train_df, feat_cols, ridge=1e-3):
    global_mean = float(train_df[TARGET].mean())
    type_mean = train_df.groupby(_type_labels(train_df["type"]))[TARGET].mean()
    params = {}

    tmp = train_df.copy()
    tmp["type_lbl"] = _type_labels(tmp["type"])
    for t, g in tmp.groupby("type_lbl"):
        X = g[feat_cols].astype(float).to_numpy()
        y = g[TARGET].astype(float).to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if len(y) < (len(feat_cols) + 2):
            params[t] = (
                float(type_mean.loc[t]),
                np.zeros(len(feat_cols), dtype=np.float64),
            )
            continue
        X_mean = X.mean(axis=0, keepdims=True)
        y_mean = float(y.mean())
        Xc = X - X_mean
        yc = y - y_mean
        A = Xc.T @ Xc + ridge * np.eye(Xc.shape[1])
        b = Xc.T @ yc
        w = np.linalg.solve(A, b)
        intercept = y_mean - float((X_mean @ w.reshape(-1, 1)).ravel()[0])
        params[t] = (float(intercept), w.astype(np.float64))
    return global_mean, type_mean, params


def predict_typewise_linear_multi(df, feat_cols, global_mean, type_mean, params):
    out = df[["id", "type"]].copy()
    X = df[feat_cols].astype(float).to_numpy()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    t_vals = _type_labels(df["type"]).values
    pred = np.empty(len(df), dtype=np.float64)
    for i, tv in enumerate(t_vals):
        if tv in params:
            a, w = params[tv]
        else:
            a = float(type_mean.get(tv, global_mean))
            w = np.zeros(len(feat_cols), dtype=np.float64)
        pred[i] = a + float(X[i].dot(w))
    pred = np.where(
        np.isfinite(pred),
        pred,
        _type_labels(df["type"]).map(type_mean).astype(float).to_numpy(),
    )
    pred = np.where(np.isfinite(pred), pred, global_mean)
    out[PREDICTION] = pred.astype(float)
    return out




## === cell 9
def build_valid_sets(train_tr_df, train_va_df, train_tr_f_df, train_va_f_df):
    maps_tr = build_type_mappings(train_tr_df, k1=25.0, k2=100.0)
    gm_tr = maps_tr["global_mean"]

    va_m0 = build_train_pred_df(train_va_df, maps_tr["type_mean"], gm_tr)
    va_m1 = build_train_pred_df(train_va_df, maps_tr["type_median"], gm_tr)
    va_m2 = build_train_pred_df(train_va_df, maps_tr["shrink_k1"], gm_tr)
    va_m3 = build_train_pred_df(train_va_df, maps_tr["shrink_k2"], gm_tr)

    gmean_ap, tmean_ap, pair_mean = build_atompair_mean_mapping(
        train_tr_f_df, k_pair=50.0
    )
    va_m4 = train_va_f_df[["id", "type", TARGET]].copy()
    va_m4[PREDICTION] = predict_atompair_mean(
        train_va_f_df, gmean_ap, tmean_ap, pair_mean
    )[PREDICTION].values

    gmean_lin, tmean_lin, params_lin = fit_typewise_dist_linear(
        train_tr_f_df, ridge=1e-3
    )
    va_m5 = train_va_f_df[["id", "type", TARGET]].copy()
    va_m5[PREDICTION] = predict_typewise_dist_linear(
        train_va_f_df, gmean_lin, tmean_lin, params_lin
    )[PREDICTION].values

    valid_sets_local = [va_m0, va_m1, va_m2, va_m3, va_m4, va_m5]

    fold_mols = set(
        pd.concat([train_tr_df["molecule_name"], train_va_df["molecule_name"]]).unique()
    )

    if os.path.exists(MULLIKEN_PATH):
        mulliken = load_mulliken_needed(MULLIKEN_PATH, fold_mols)
        train_tr_q = attach_mulliken(train_tr_f_df, mulliken)
        train_va_q = attach_mulliken(train_va_f_df, mulliken)

        for d in [train_tr_q, train_va_q]:
            d["q0"] = d["q0"].astype(np.float32)
            d["q1"] = d["q1"].astype(np.float32)
            d["qprod"] = (d["q0"] * d["q1"]).astype(np.float32)
            d["qdiff"] = (d["q0"] - d["q1"]).abs().astype(np.float32)

        feat_q = ["dist", "q0", "q1", "qprod", "qdiff"]
        gmean_q, tmean_q, params_q = fit_typewise_linear_multi(
            train_tr_q, feat_q, ridge=1e-3
        )
        va_m6 = train_va_q[["id", "type", TARGET]].copy()
        va_m6[PREDICTION] = predict_typewise_linear_multi(
            train_va_q, feat_q, gmean_q, tmean_q, params_q
        )[PREDICTION].values
        valid_sets_local.append(va_m6)

    if os.path.exists(SHIELD_PATH):
        shield = load_shielding_needed(SHIELD_PATH, fold_mols)
        train_tr_s = attach_shielding(train_tr_f_df, shield)
        train_va_s = attach_shielding(train_va_f_df, shield)

        for d in [train_tr_s, train_va_s]:
            d["st0"] = d["st0"].astype(np.float32)
            d["st1"] = d["st1"].astype(np.float32)
            d["sa0"] = d["sa0"].astype(np.float32)
            d["sa1"] = d["sa1"].astype(np.float32)
            d["st_sum"] = (d["st0"] + d["st1"]).astype(np.float32)
            d["st_diff"] = (d["st0"] - d["st1"]).abs().astype(np.float32)
            d["sa_sum"] = (d["sa0"] + d["sa1"]).astype(np.float32)

        feat_s = ["dist", "st0", "st1", "st_sum", "st_diff", "sa_sum"]
        gmean_s, tmean_s, params_s = fit_typewise_linear_multi(
            train_tr_s, feat_s, ridge=1e-3
        )
        va_m7 = train_va_s[["id", "type", TARGET]].copy()
        va_m7[PREDICTION] = predict_typewise_linear_multi(
            train_va_s, feat_s, gmean_s, tmean_s, params_s
        )[PREDICTION].values
        valid_sets_local.append(va_m7)

    for i in range(len(valid_sets_local)):
        valid_sets_local[i] = (
            valid_sets_local[i].sort_values("id").reset_index(drop=True)
        )
    for i in range(1, len(valid_sets_local)):
        assert (
            valid_sets_local[0]["id"].values == valid_sets_local[i]["id"].values
        ).all()

    return valid_sets_local


valid_sets_fold1 = build_valid_sets(train_tr, train_va, train_tr_f, train_va_f)
valid_sets_fold2 = build_valid_sets(train_tr2, train_va2, train_tr2_f, train_va2_f)

print(
    "Prepared validation prediction sets fold1:", [df.shape for df in valid_sets_fold1]
)
print(
    "Prepared validation prediction sets fold2:", [df.shape for df in valid_sets_fold2]
)

assert len(valid_sets_fold1) == len(valid_sets_fold2), (
    len(valid_sets_fold1),
    len(valid_sets_fold2),
)




## === cell 10
def weights(n, alpha=1.0):
    if n < 1:
        raise ValueError("n must not be less than 1")
    return np.random.dirichlet(alpha=np.full(n, alpha)).tolist()


def eval_blend(valid_sets, ws):
    df = valid_sets[0][["id", "type", TARGET]].copy()
    df[PREDICTION] = 0.0
    for i, t in enumerate(valid_sets):
        df[PREDICTION] += t[PREDICTION].astype(float).values * ws[i]
    return float(group_mean_log_mae(df[TARGET], df[PREDICTION], df["type"]))


def trial_twofold(valid_sets1, valid_sets2):
    ws = weights(len(valid_sets1), alpha=1.0)
    s1 = eval_blend(valid_sets1, ws)
    s2 = eval_blend(valid_sets2, ws)
    return 0.5 * (s1 + s2), ws, s1, s2


best = sys.maxsize
best_weights = None
best_s1 = None
best_s2 = None

for i in range(TRIALS):
    score, ws, s1, s2 = trial_twofold(valid_sets_fold1, valid_sets_fold2)
    if score < best:
        best = score
        best_weights = ws
        best_s1 = s1
        best_s2 = s2

print(f"best(avg valid)={best:.6f} | fold1={best_s1:.6f} fold2={best_s2:.6f}")
print(f"best weights (sum={sum(best_weights):.6f})")
for i, w in enumerate(best_weights):
    print(f"  model{i} weight={w:.6f}")



## === cell 11
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

gmean_ap_f, tmean_ap_f, pair_mean_f = build_atompair_mean_mapping(train_f, k_pair=50.0)
test_m4 = (
    predict_atompair_mean(test_f, gmean_ap_f, tmean_ap_f, pair_mean_f)
    .sort_values("id")
    .reset_index(drop=True)
)

gmean_lin_f, tmean_lin_f, params_lin_f = fit_typewise_dist_linear(train_f, ridge=1e-3)
test_m5 = (
    predict_typewise_dist_linear(test_f, gmean_lin_f, tmean_lin_f, params_lin_f)
    .sort_values("id")
    .reset_index(drop=True)
)

test_models = [test_m0, test_m1, test_m2, test_m3, test_m4, test_m5]

if os.path.exists(MULLIKEN_PATH):
    mulliken_train = load_mulliken_needed(MULLIKEN_PATH, final_train_mols)
    mulliken_test = load_mulliken_needed(MULLIKEN_PATH, final_test_mols)
    train_q = attach_mulliken(train_f, mulliken_train)
    test_q = attach_mulliken(test_f, mulliken_test)
    for d in [train_q, test_q]:
        d["q0"] = d["q0"].astype(np.float32)
        d["q1"] = d["q1"].astype(np.float32)
        d["qprod"] = (d["q0"] * d["q1"]).astype(np.float32)
        d["qdiff"] = (d["q0"] - d["q1"]).abs().astype(np.float32)
    feat_q = ["dist", "q0", "q1", "qprod", "qdiff"]
    gmean_q_f, tmean_q_f, params_q_f = fit_typewise_linear_multi(
        train_q, feat_q, ridge=1e-3
    )
    test_m6 = (
        predict_typewise_linear_multi(test_q, feat_q, gmean_q_f, tmean_q_f, params_q_f)
        .sort_values("id")
        .reset_index(drop=True)
    )
    test_models.append(test_m6)

if os.path.exists(SHIELD_PATH):
    shield_train = load_shielding_needed(SHIELD_PATH, final_train_mols)
    shield_test = load_shielding_needed(SHIELD_PATH, final_test_mols)
    train_s = attach_shielding(train_f, shield_train)
    test_s = attach_shielding(test_f, shield_test)
    for d in [train_s, test_s]:
        d["st0"] = d["st0"].astype(np.float32)
        d["st1"] = d["st1"].astype(np.float32)
        d["sa0"] = d["sa0"].astype(np.float32)
        d["sa1"] = d["sa1"].astype(np.float32)
        d["st_sum"] = (d["st0"] + d["st1"]).astype(np.float32)
        d["st_diff"] = (d["st0"] - d["st1"]).abs().astype(np.float32)
        d["sa_sum"] = (d["sa0"] + d["sa1"]).astype(np.float32)
    feat_s = ["dist", "st0", "st1", "st_sum", "st_diff", "sa_sum"]
    gmean_s_f, tmean_s_f, params_s_f = fit_typewise_linear_multi(
        train_s, feat_s, ridge=1e-3
    )
    test_m7 = (
        predict_typewise_linear_multi(test_s, feat_s, gmean_s_f, tmean_s_f, params_s_f)
        .sort_values("id")
        .reset_index(drop=True)
    )
    test_models.append(test_m7)

for i in range(1, len(test_models)):
    assert (test_models[0]["id"].values == test_models[i]["id"].values).all()

if len(best_weights) != len(test_models):
    min_len = min(len(best_weights), len(test_models))
    test_models = test_models[:min_len]
    best_weights = best_weights[:min_len]
    s = sum(best_weights)
    best_weights = [w / s for w in best_weights]

submission = test_models[0][["id"]].copy()
submission[TARGET] = 0.0
for i, t in enumerate(test_models):
    submission[TARGET] += t[PREDICTION].astype(float).values * best_weights[i]

if os.path.exists(SAMPLE_SUB_PATH):
    sample = pd.read_csv(SAMPLE_SUB_PATH, usecols=["id"])
    submission = sample.merge(submission, on="id", how="left")
    submission[TARGET] = submission[TARGET].fillna(global_mean)

print(submission.head())
print(submission.shape)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote", out_path, "with shape", submission.shape)
print(submission.dtypes)
print("Files in working dir:", os.listdir("."))
