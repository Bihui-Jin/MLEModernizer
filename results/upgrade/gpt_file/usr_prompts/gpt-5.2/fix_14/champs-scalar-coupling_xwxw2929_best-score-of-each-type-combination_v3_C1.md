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

-1.6777209112242684

# 6. Current score

3.00563

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 3.00563) has done: 'The notebook fails because it tries to read two external submissions from `../input/keras-*` datasets that are not present in your environment, so `sub1/sub2` are undefined and the concatenation never builds a full set of test ids. To make it run end-to-end and always output a valid submission, I replace those missing inputs with a simple, deterministic baseline model trained from the provided competition files only (no extra datasets). The model preserves the original “combine predictions into one submission” intent by predicting per coupling `type` and then filling every test `id`, guaranteeing no missing ids. This should also produce a non-trivial score (better than a constant/empty submission) while keeping changes minimal and focused on correctness.'
- What this solution (achieved 3.00563) has done: 'Your current model uses only distance, which is too weak for this competition and explains the large gap to the target (lower is better). I keep the same “fit per coupling `type` and predict for test” core approach, but upgrade the per-type model from 1D linear regression on `dist` to a small closed-form linear regression on a few safe, cheap geometric features (distance powers and coordinate deltas), which better matches known signal while staying deterministic and fast. I also add light ridge regularization for numerical stability and clip extreme predictions per type to the training range to reduce MAE blow-ups that disproportionately hurt the log-MAE metric. The pipeline remains end-to-end, uses only provided competition files, and still outputs a valid `submission.csv` with all test ids.'
- What this solution (achieved 3.00563) has done: 'I fix the crash in the atom list creation by ensuring we only sort non-null string atom symbols (the current concat/sort mixes NaN floats with strings). This let `make_design_matrix()` be defined successfully, which in turn fixes the downstream `NameError` failures in training, submission checks, and CSV writing. I also add a small safety fill for any missing `atom_0/atom_1` values right after feature creation so the one-hot encoding never encounters NaNs. These changes are execution/robustness fixes and should be score-neutral aside from enabling the intended model to run and generate a valid submission.'
- What this solution (achieved 3.00563) has done: 'Your score is far worse than the target (lower-is-better), so we should make the smallest legitimate quality improvements without changing the overall “per-type deterministic linear model on engineered geometry + atom one-hot” approach. The biggest likely issue is the forced single sign per coupling `type` (median-sign) combined with modeling only `log1p(|y|)`, which systematically flips many predictions for types whose targets mix signs; instead, we keep the same closed-form ridge regression but learn the sign with a second ridge model on `sign(y)` and combine it with the magnitude model. To better match the competition’s per-type MAE and stabilize predictions, we also fit the model directly on `y` (not `log1p(|y|)`) in a second head and blend it lightly with the sign*magnitude prediction; this preserves core logic (linear ridge per type) while reducing large errors. Finally, we keep your clipping and add a tiny fallback for unseen/rare types to avoid NaNs and extreme outputs.'
- What this solution (achieved 3.00563) has done: 'Your current gap to the target is very large (lower-is-better, and 3.00563 >> -1.6777), so we need a legitimate quality boost while keeping the same core “per-type closed-form ridge regression on engineered geometry + atom one-hot” approach. The most direct minimal improvement is to (1) add a few strong, cheap physics-inspired features (atom-level Mulliken charge and magnetic shielding tensors for the two atoms, plus molecule-level dipole and potential energy) into the same design matrix, and (2) standardize features per type before solving ridge to make the linear system better conditioned. These changes keep the exact same training/prediction structure (still per-type ridge via normal equations, still the same blending of signed-magnitude and direct-y heads), but typically reduce large errors a lot on this competition. Submission writing/format stays identical.'
- What this solution (achieved 3.00563) has done: 'Your current gap to the (much better) target is very large for a lower-is-better metric, so we need a modest but meaningful quality lift while keeping the same per-type closed-form ridge regression core. The most leverage with minimal conceptual change is to add a few cheap, competition-proven geometric features: per-atom absolute coordinates, center-of-pair coordinates, and dot products with the molecule dipole vector; these stay within your linear model and use data you already load. I also make the standardization ignore the intercept column so the bias term remains interpretable and stable, which typically reduces systematic offsets per type. Finally, I tighten numerical robustness around one-hot mapping (avoid potential NaN indices) without changing semantics.'
- What this solution (achieved 3.00563) has done: 'Your current score (3.00563, lower-is-better) is far from the target (-1.6777), so we need a real quality lift while keeping the same core approach: per-`type` standardized closed-form ridge regression on engineered features. The biggest safe gain with minimal conceptual change is to add the known-strong `scalar_coupling_contributions.csv` terms (`fc/sd/pso/dso`) as additional linear features during training only (then use a second per-type ridge model to predict those four terms from the same existing features at test time, and sum them to get the final prediction). This preserves your per-type ridge setup and feature extraction, but aligns tightly with the physics decomposition of the target and typically reduces MAE substantially. I also blend this “predicted contributions sum” with your existing prediction (small weight) to improve stability without changing the overall semantics too aggressively.'
- What this solution (achieved 3.00563) has done: 'Your current score is much worse than the target (lower-is-better), so the smallest safe way to move toward the target is to keep your exact per-type closed-form ridge setup, but align it better with the competition metric by (1) fitting and predicting **per type on a log1p(|y|) scale with sign**, and (2) applying a lightweight **per-type mean calibration** using the training set so each type’s predictions match that type’s central tendency. I also make the contribution-sum head consistent by predicting **contributions on their native scale** but blending them **after** calibrating the main prediction, so it helps without dominating. Finally, I add a tiny numerical stabilization for the expm1 step to avoid rare overflow-induced huge MAE spikes that can severely hurt log-MAE.'
- What this solution (achieved 3.00563) has done: 'Your current score (3.00563; lower-is-better) is far from the target (-1.6777), so we need a legitimate accuracy lift while keeping the same per-type standardized closed-form ridge setup. The biggest low-risk gain without changing the model family is to stop forcing a hard +/- sign via classification and instead learn a continuous signed prediction on the log-|y| scale, then invert it; this reduces large sign-flip MAE spikes that heavily hurt the log-MAE metric. I also switch the per-type calibration from a mean-shift to a robust median-shift (less sensitive to outliers) while keeping the same “one scalar shift per type” idea. Finally, I slightly rebalance the blend weights to rely more on the direct-y head (usually stabilizes extremes) while keeping the contribution-sum head in place.'
- What this solution (achieved 3.00563) has done: 'Your current score (3.00563, lower-is-better) is far from the target (-1.6777), so we should make a small, legitimate accuracy improvement without changing the core “per-type standardized closed-form ridge regression on engineered features” approach. The biggest low-risk gain is to add a tiny within-train molecule split to compute a per-type blend weight between your main prediction and the contribution-sum head, because the contributions sum is often very predictive but not equally reliable across types; this keeps the same models and features, only adjusts blending per type using training-only information. I also fix a likely join issue by merging contributions in a symmetric way (since train/test atom_index order may not match the contributions file ordering), which can currently zero-out many contribution rows and hurt the contribution head. These two changes are minimal, deterministic, and should reduce MAE spikes (and thus log-MAE) while preserving the overall pipeline and submission format.'
- What this solution (achieved 3.00563) has done: 'You’re far worse than the target (lower-is-better), so the smallest likely-to-help change is to fix the blending logic so the per-type “contribution-sum head” is actually used as intended and not distorted by the current weight-search bug. I replace the incorrect alpha/weight computation with a deterministic closed-form least-squares fit (on a molecule-heldout fold) for the blend weights between your existing three heads, then renormalize to keep weights stable and non-negative. This keeps the exact same core approach (same features, same per-type ridge models, same calibration/clipping), but should reduce large per-type MAE and therefore improve the log-MAE metric. I also add a tiny guard for types with too-few molecules to avoid unstable weight fitting.'
- What this solution (achieved 3.00563) has done: 'Your current score is much worse than the target (lower-is-better), so the smallest likely-to-help change is to fix a bug in the per-type blending logic that can distort weights and effectively ignore useful heads. I remove the unintended “no-op” weight computation and instead compute stable non-negative blend weights from the molecule-heldout fold, then apply a mild shrinkage back toward your base weights to avoid overfitting while still improving accuracy. I also make the blend-weight fitting use the *same clipped* signed-log predictions used at inference (to reduce mismatch), and keep all models/features/training loops identical otherwise. This should move the score downward (better) toward your target while preserving the core per-type ridge + engineered features approach and producing the same valid submission format.'

# 9. Code solution

## === cell 0
import warnings

warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd

BASE_CANDIDATES = [
    "../input/champs-scalar-coupling",
    "/kaggle/input/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling",
    "/kaggle/data/input/champs-scalar-coupling",
]
BASE_PATH = None
for p in BASE_CANDIDATES:
    if os.path.exists(p):
        BASE_PATH = p
        break

if BASE_PATH is None:
    for root, dirs, files in os.walk("/kaggle"):
        if "train.csv" in files and "test.csv" in files:
            BASE_PATH = root
            break

print("BASE_PATH:", BASE_PATH)
print("Listing BASE_PATH (first 20):", sorted(os.listdir(BASE_PATH))[:20])



## === cell 1
train = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
test = pd.read_csv(os.path.join(BASE_PATH, "test.csv"))
structures = pd.read_csv(os.path.join(BASE_PATH, "structures.csv"))

print(train.shape, test.shape, structures.shape)
print(train.head())
print(test.head())



## === cell 2
mulliken = pd.read_csv(os.path.join(BASE_PATH, "mulliken_charges.csv"))
shield = pd.read_csv(os.path.join(BASE_PATH, "magnetic_shielding_tensors.csv"))
dipole = pd.read_csv(os.path.join(BASE_PATH, "dipole_moments.csv"))
energy = pd.read_csv(os.path.join(BASE_PATH, "potential_energy.csv"))

print("Loaded:", mulliken.shape, shield.shape, dipole.shape, energy.shape)




## === cell 3
def add_pair_distance(df_pairs, structures_df):
    s = structures_df[["molecule_name", "atom_index", "atom", "x", "y", "z"]].copy()

    s0 = s.rename(
        columns={
            "atom_index": "atom_index_0",
            "atom": "atom_0",
            "x": "x0",
            "y": "y0",
            "z": "z0",
        }
    )
    s1 = s.rename(
        columns={
            "atom_index": "atom_index_1",
            "atom": "atom_1",
            "x": "x1",
            "y": "y1",
            "z": "z1",
        }
    )

    out = df_pairs.merge(s0, how="left", on=["molecule_name", "atom_index_0"])
    out = out.merge(s1, how="left", on=["molecule_name", "atom_index_1"])

    dx = out["x0"] - out["x1"]
    dy = out["y0"] - out["y1"]
    dz = out["z0"] - out["z1"]
    out["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz)

    if out["dist"].isna().any():
        out["dist"] = out["dist"].fillna(out["dist"].median())

    return out


def add_side_info(df_pairs, mulliken_df, shield_df, dipole_df, energy_df):
    m0 = mulliken_df.rename(
        columns={"atom_index": "atom_index_0", "mulliken_charge": "q0"}
    )
    m1 = mulliken_df.rename(
        columns={"atom_index": "atom_index_1", "mulliken_charge": "q1"}
    )

    sh_cols = ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]
    s0 = shield_df.rename(
        columns={c: f"sh0_{c}" for c in sh_cols} | {"atom_index": "atom_index_0"}
    )
    s1 = shield_df.rename(
        columns={c: f"sh1_{c}" for c in sh_cols} | {"atom_index": "atom_index_1"}
    )

    out = df_pairs.merge(
        m0[["molecule_name", "atom_index_0", "q0"]],
        how="left",
        on=["molecule_name", "atom_index_0"],
    )
    out = out.merge(
        m1[["molecule_name", "atom_index_1", "q1"]],
        how="left",
        on=["molecule_name", "atom_index_1"],
    )

    out = out.merge(
        s0[["molecule_name", "atom_index_0"] + [f"sh0_{c}" for c in sh_cols]],
        how="left",
        on=["molecule_name", "atom_index_0"],
    )
    out = out.merge(
        s1[["molecule_name", "atom_index_1"] + [f"sh1_{c}" for c in sh_cols]],
        how="left",
        on=["molecule_name", "atom_index_1"],
    )

    out = out.merge(
        dipole_df.rename(columns={"X": "dipX", "Y": "dipY", "Z": "dipZ"}),
        how="left",
        on="molecule_name",
    )
    out = out.merge(energy_df, how="left", on="molecule_name")

    for c in (
        ["q0", "q1", "dipX", "dipY", "dipZ", "potential_energy"]
        + [f"sh0_{k}" for k in sh_cols]
        + [f"sh1_{k}" for k in sh_cols]
    ):
        if c in out.columns:
            if out[c].isna().any():
                out[c] = out[c].fillna(out[c].median())

    return out


train_feat = add_pair_distance(train, structures)
test_feat = add_pair_distance(test, structures)

train_feat = add_side_info(train_feat, mulliken, shield, dipole, energy)
test_feat = add_side_info(test_feat, mulliken, shield, dipole, energy)

for df_ in (train_feat, test_feat):
    df_["atom_0"] = df_["atom_0"].astype("string").fillna("UNK")
    df_["atom_1"] = df_["atom_1"].astype("string").fillna("UNK")

print(
    train_feat[
        [
            "id",
            "type",
            "atom_0",
            "atom_1",
            "dist",
            "q0",
            "q1",
            "potential_energy",
            "scalar_coupling_constant",
        ]
    ].head()
)
print(
    test_feat[
        ["id", "type", "atom_0", "atom_1", "dist", "q0", "q1", "potential_energy"]
    ].head()
)



## === cell 4
ALL_ATOMS = sorted(
    pd.concat(
        [
            train_feat["atom_0"],
            train_feat["atom_1"],
            test_feat["atom_0"],
            test_feat["atom_1"],
        ],
        axis=0,
        ignore_index=True,
    )
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)
ATOM_TO_IDX = {a: i for i, a in enumerate(ALL_ATOMS)}
N_AT = len(ALL_ATOMS)

SH_COLS = ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]


def make_design_matrix(df):
    dx = (df["x0"] - df["x1"]).to_numpy(np.float64)
    dy = (df["y0"] - df["y1"]).to_numpy(np.float64)
    dz = (df["z0"] - df["z1"]).to_numpy(np.float64)
    d = df["dist"].to_numpy(np.float64)

    x0 = df["x0"].to_numpy(np.float64)
    y0 = df["y0"].to_numpy(np.float64)
    z0 = df["z0"].to_numpy(np.float64)
    x1 = df["x1"].to_numpy(np.float64)
    y1 = df["y1"].to_numpy(np.float64)
    z1 = df["z1"].to_numpy(np.float64)
    xc = 0.5 * (x0 + x1)
    yc = 0.5 * (y0 + y1)
    zc = 0.5 * (z0 + z1)

    d2 = d * d
    d3 = d2 * d
    invd = 1.0 / np.maximum(d, 1e-8)
    invd2 = invd * invd
    invd3 = invd2 * invd

    q0 = df["q0"].to_numpy(np.float64)
    q1 = df["q1"].to_numpy(np.float64)
    dipX = df["dipX"].to_numpy(np.float64)
    dipY = df["dipY"].to_numpy(np.float64)
    dipZ = df["dipZ"].to_numpy(np.float64)
    pe = df["potential_energy"].to_numpy(np.float64)

    dip_dot_d = dipX * dx + dipY * dy + dipZ * dz
    dip_dot_c = dipX * xc + dipY * yc + dipZ * zc

    sh0 = np.column_stack([df[f"sh0_{c}"].to_numpy(np.float64) for c in SH_COLS])
    sh1 = np.column_stack([df[f"sh1_{c}"].to_numpy(np.float64) for c in SH_COLS])

    base = np.column_stack(
        [
            np.ones_like(d),
            d,
            d2,
            d3,
            invd,
            invd2,
            invd3,
            np.abs(dx),
            np.abs(dy),
            np.abs(dz),
            x0,
            y0,
            z0,
            x1,
            y1,
            z1,
            xc,
            yc,
            zc,
            dip_dot_d,
            dip_dot_c,
            q0,
            q1,
            q0 - q1,
            q0 * q1,
            dipX,
            dipY,
            dipZ,
            pe,
        ]
    )

    base = np.concatenate([base, sh0, sh1, sh0 - sh1], axis=1)

    a0 = (
        df["atom_0"]
        .astype(str)
        .map(ATOM_TO_IDX)
        .fillna(ATOM_TO_IDX.get("UNK", 0))
        .to_numpy(np.int32)
    )
    a1 = (
        df["atom_1"]
        .astype(str)
        .map(ATOM_TO_IDX)
        .fillna(ATOM_TO_IDX.get("UNK", 0))
        .to_numpy(np.int32)
    )
    oh0 = np.zeros((len(df), N_AT), dtype=np.float64)
    oh1 = np.zeros((len(df), N_AT), dtype=np.float64)
    oh0[np.arange(len(df)), a0] = 1.0
    oh1[np.arange(len(df)), a1] = 1.0

    X = np.concatenate([base, oh0, oh1], axis=1)
    return X


def ridge_solve_from_stats(XtX, Xty, ridge):
    XtX = XtX.copy()
    XtX.flat[:: XtX.shape[0] + 1] += ridge
    return np.linalg.solve(XtX, Xty)


def standardize_fit(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    mu[0] = 0.0
    sigma[0] = 1.0
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    Xs = (X - mu) / sigma
    return Xs, mu, sigma


def standardize_apply(X, mu, sigma):
    return (X - mu) / sigma




## === cell 5
contrib = pd.read_csv(os.path.join(BASE_PATH, "scalar_coupling_contributions.csv"))
contrib = contrib[
    ["molecule_name", "atom_index_0", "atom_index_1", "type", "fc", "sd", "pso", "dso"]
].copy()

contrib["a_min"] = np.minimum(
    contrib["atom_index_0"].values, contrib["atom_index_1"].values
)
contrib["a_max"] = np.maximum(
    contrib["atom_index_0"].values, contrib["atom_index_1"].values
)
contrib = contrib.drop(columns=["atom_index_0", "atom_index_1"])

train_feat["a_min"] = np.minimum(
    train_feat["atom_index_0"].values, train_feat["atom_index_1"].values
)
train_feat["a_max"] = np.maximum(
    train_feat["atom_index_0"].values, train_feat["atom_index_1"].values
)

train_feat = train_feat.merge(
    contrib,
    how="left",
    on=["molecule_name", "a_min", "a_max", "type"],
)

for c in ["fc", "sd", "pso", "dso"]:
    if train_feat[c].isna().any():
        train_feat[c] = train_feat[c].fillna(0.0)

print(
    train_feat[["fc", "sd", "pso", "dso"]].describe().T[["mean", "std", "min", "max"]]
)



## === cell 6
types = sorted(train_feat["type"].unique())

params = {}

ridge_signedlog = 1e-2
ridge_y = 1e-2
ridge_contrib = 1e-2

w_signedlog_base = 0.60
w_direct_y_base = 0.25
w_contrib_sum_base = 0.15

rng = np.random.RandomState(42)

for t in types:
    tr = train_feat[train_feat["type"] == t]
    y = tr["scalar_coupling_constant"].to_numpy(np.float64)

    X = make_design_matrix(tr)
    Xs, mu, sigma = standardize_fit(X)
    XtX = Xs.T @ Xs

    y_signedlog = np.sign(y) * np.log1p(np.abs(y))
    beta_signedlog = ridge_solve_from_stats(XtX, Xs.T @ y_signedlog, ridge_signedlog)

    beta_y = ridge_solve_from_stats(XtX, Xs.T @ y, ridge_y)

    Yc = tr[["fc", "sd", "pso", "dso"]].to_numpy(np.float64)  # (n,4)
    XtYc = Xs.T @ Yc
    Bc = np.empty((XtX.shape[0], 4), dtype=np.float64)
    XtX_reg = XtX.copy()
    XtX_reg.flat[:: XtX_reg.shape[0] + 1] += ridge_contrib
    for j in range(4):
        Bc[:, j] = np.linalg.solve(XtX_reg, XtYc[:, j])

    pt_signedlog_tr = Xs @ beta_signedlog
    pt_signedlog_tr = np.clip(pt_signedlog_tr, -50.0, 50.0)
    p_signedlog_tr = np.sign(pt_signedlog_tr) * np.expm1(np.abs(pt_signedlog_tr))

    p_direct_tr = Xs @ beta_y
    p_contrib_sum_tr = (Xs @ Bc).sum(axis=1)

    mol = tr["molecule_name"].to_numpy()
    uniq = pd.unique(mol)
    rng.shuffle(uniq)
    cut = int(0.8 * len(uniq))
    m_tr = set(uniq[:cut])
    in_mask = np.array([m in m_tr for m in mol], dtype=bool)

    w_signedlog = w_signedlog_base
    w_direct_y = w_direct_y_base
    w_contrib = w_contrib_sum_base

    if in_mask.sum() > 500 and (~in_mask).sum() > 500:
        y_va = y[~in_mask]
        A = np.column_stack(
            [
                p_signedlog_tr[~in_mask],
                p_direct_tr[~in_mask],
                p_contrib_sum_tr[~in_mask],
            ]
        ).astype(np.float64)

        lam_w = 1e-6
        AtA = A.T @ A
        AtA.flat[:: AtA.shape[0] + 1] += lam_w
        Aty = A.T @ y_va
        w = np.linalg.solve(AtA, Aty)

        w = np.maximum(w, 0.0)
        s = float(w.sum())
        if s > 1e-12:
            w = w / s
            w_signedlog, w_direct_y, w_contrib = float(w[0]), float(w[1]), float(w[2])
        else:
            w_signedlog, w_direct_y, w_contrib = (
                w_signedlog_base,
                w_direct_y_base,
                w_contrib_sum_base,
            )

        w_contrib = float(np.clip(w_contrib, 0.0, 0.40))

        shrink = 0.25
        w_signedlog = (1.0 - shrink) * w_signedlog + shrink * w_signedlog_base
        w_direct_y = (1.0 - shrink) * w_direct_y + shrink * w_direct_y_base
        w_contrib = (1.0 - shrink) * w_contrib + shrink * w_contrib_sum_base

        w_contrib = float(np.clip(w_contrib, 0.0, 0.40))
        rem = max(1e-12, 1.0 - w_contrib)
        wd_sum = max(1e-12, w_signedlog + w_direct_y)
        w_signedlog = rem * (w_signedlog / wd_sum)
        w_direct_y = rem * (w_direct_y / wd_sum)

    p_tr = (
        (w_signedlog * p_signedlog_tr)
        + (w_direct_y * p_direct_tr)
        + (w_contrib * p_contrib_sum_tr)
    )

    calib_shift = float(np.median(y) - np.median(p_tr))

    y_lo = float(np.quantile(y, 0.002))
    y_hi = float(np.quantile(y, 0.998))

    params[t] = (
        beta_signedlog,
        beta_y,
        Bc,
        mu,
        sigma,
        y_lo,
        y_hi,
        calib_shift,
        w_signedlog,
        w_direct_y,
        w_contrib,
    )

pred = np.empty(len(test_feat), dtype=np.float64)
pred[:] = np.nan

global_mean = float(train_feat["scalar_coupling_constant"].mean())

for t in types:
    (
        beta_signedlog,
        beta_y,
        Bc,
        mu,
        sigma,
        y_lo,
        y_hi,
        calib_shift,
        w_signedlog,
        w_direct_y,
        w_contrib,
    ) = params[t]
    mask = test_feat["type"].values == t
    if not np.any(mask):
        continue

    Xte = make_design_matrix(test_feat.loc[mask])
    Xte = standardize_apply(Xte, mu, sigma)

    pt_signedlog = Xte @ beta_signedlog
    pt_signedlog = np.clip(pt_signedlog, -50.0, 50.0)
    p_signedlog = np.sign(pt_signedlog) * np.expm1(np.abs(pt_signedlog))

    p_direct = Xte @ beta_y
    p_contrib_sum = (Xte @ Bc).sum(axis=1)

    p = (
        (w_signedlog * p_signedlog)
        + (w_direct_y * p_direct)
        + (w_contrib * p_contrib_sum)
    )

    p = p + calib_shift
    p = np.clip(p, y_lo, y_hi)
    pred[mask] = p

if np.isnan(pred).any():
    pred[np.isnan(pred)] = global_mean

sub = pd.DataFrame({"id": test_feat["id"].values, "scalar_coupling_constant": pred})
sub = sub.sort_values("id").reset_index(drop=True)

print(sub.head())
print("Submission shape:", sub.shape, "Expected rows:", test.shape[0])



## === cell 7
missing = set(test["id"].values) - set(sub["id"].values)
extra = set(sub["id"].values) - set(test["id"].values)
dup = sub["id"].duplicated().sum()

print("Missing ids:", len(missing))
print("Extra ids:", len(extra))
print("Duplicate ids:", int(dup))

assert len(missing) == 0, "Invalid submission: missing required ids."
assert len(extra) == 0, "Invalid submission: extra ids."
assert dup == 0, "Invalid submission: duplicate ids."
assert list(sub.columns) == [
    "id",
    "scalar_coupling_constant",
], "Invalid submission columns."



## === cell 8
out_path = "/kaggle/working/submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(pd.read_csv(out_path).head())
