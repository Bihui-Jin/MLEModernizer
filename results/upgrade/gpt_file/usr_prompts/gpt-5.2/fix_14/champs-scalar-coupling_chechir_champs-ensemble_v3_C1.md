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

-2.0809984084130706

# 6. Current score

1.83922

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.99777) has done: 'I remove the dependency on missing `../input/champ-preds/*.csv` files (which causes the `FileNotFoundError`) and instead build a simple, robust baseline model directly from the provided CHAMPS dataset files in `/kaggle/data/champs-scalar-coupling/`. To keep the core approach minimal and stable, the solution use lightweight feature engineering (atom types + pairwise distance + coupling type) and train one `HistGradientBoostingRegressor` per coupling `type`, then predict test rows of that type. Finally, it write a valid `sub_ensemble.csv` with exactly `id,scalar_coupling_constant` columns.'
- What this solution (achieved 1.74698) has done: 'The NaNs come from a path mismatch: `train.csv`/`test.csv` reference molecule names that are not present in the `structures.csv` you loaded (different dataset copy/folder), so the merges fail and produce missing atom coordinates. I fix this by loading *all* files (train/test/structures/sample) from the same base directory and adding a small validation that auto-switches to the alternate mirrored folder if needed. This is a correctness/stability fix (not a modeling change) and should restore the previous working behavior and score trajectory. I also make the categorical mapping robust to any remaining missing values by using an explicit “__UNK__” category, preventing `IntCastingNaNError` and ensuring a valid `sub_ensemble.csv` is always written.'
- What this solution (achieved 1.68538) has done: 'Your current score (1.74698, lower-is-better) is far worse than the target (-2.08099), so we should improve performance but keep the same overall approach (per-type HistGradientBoostingRegressor on simple geometric/categorical features). The biggest minimal win within your existing feature paradigm is to add a couple of cheap, physically meaningful pairwise features (squared distance and inverse distance) and to include simple per-molecule aggregate context (atom count + mean/std of coordinates) derived only from `structures.csv`. These additions don’t change the training loop or model family, but usually reduce MAE substantially for this competition’s baseline models. I also ensure the submission alignment is strictly by `id` (no unnecessary merge) to avoid any accidental ordering issues.'
- What this solution (achieved 1.6489) has done: 'Your current score (1.68538, lower-is-better) is still far from the target (-2.08099), so we should improve it while keeping the same per-type `HistGradientBoostingRegressor` approach and the same overall feature style. The smallest high-impact improvement here is to add a few more physically meaningful but cheap geometric features (absolute coordinate deltas, direction cosines, and centered coordinates relative to molecule centroid) without changing the model family or training loop. These features usually reduce MAE significantly for CHAMPS because coupling strength depends on both distance and relative orientation within a molecule. I also keep the submission alignment-by-`id` logic intact, only ensuring feature columns are filled deterministically.'
- What this solution (achieved 1.63506) has done: 'Your current score (1.6489, lower-is-better) is still far from the target (-2.08099), so we should improve it while keeping the same per-type `HistGradientBoostingRegressor` training loop and the same “simple geometry + categorical atoms/types” feature style. The smallest high-impact, competition-specific improvement that stays within your core logic is to add the provided per-atom physics features (Mulliken charge + magnetic shielding tensor diagonals) by merging them for atom_0 and atom_1, then training the same per-type models on the extended feature set. This typically reduces MAE materially on CHAMPS without changing the model family, loss, or training approach. I also add a couple of cheap derived tensor summaries (trace and anisotropy magnitude) and keep the submission writing exactly `id,scalar_coupling_constant` to ensure validity.'
- What this solution (achieved 1.6417) has done: 'Your current score (1.63506, lower-is-better) is still far above the target (-2.08099), so we should improve predictive accuracy while keeping the same per-type `HistGradientBoostingRegressor` approach. The most impactful minimal change that stays within your existing feature paradigm is to add the provided molecule-level physics features (`dipole_moments` and `potential_energy`) and merge them into both train/test, which typically reduces MAE without altering the training loop or model family. To better match the competition metric (average log(MAE) per type), we also add a tiny, type-specific post-fit bias correction using out-of-fold residual medians computed via a molecule-group split; this is a lightweight calibration step and does not change the model or loss. Finally, we keep submission alignment strictly by `id` and keep runtime bounded by using small per-type validation splits (not cross-validation).'
- What this solution (achieved 1.6324) has done: 'Your current score (1.6417, lower-is-better) is still far from the target (-2.081), so we should improve accuracy while keeping the same per-type HistGradientBoostingRegressor training loop and the same overall feature paradigm. The smallest high-impact fix is to stop leaking the numeric `type` feature into the per-type models (it’s constant within each model and can add noise), and instead add a cheap but very predictive geometric feature: the bond-angle proxy via dot product between centered atom vectors and its cosine. I also make the type-specific bias correction consistent by computing the bias from out-of-fold predictions (fit on ~90% molecules, predict ~10% held-out) rather than in-sample residuals, which should reduce systematic error without changing the model family or loss. Submission writing remains identical (`id,scalar_coupling_constant`) and aligned strictly by `id`.'
- What this solution (achieved 1.60317) has done: 'Your current score (1.6324, lower-is-better) is still far from the target (-2.081), so we should improve accuracy while keeping the same per-type `HistGradientBoostingRegressor` approach and feature paradigm. The smallest high-impact fix is to align training with the competition metric by training on `log1p(abs(y))` per type (monotonic transform) and then inverting with `sign * (expm1(pred))`; this often reduces relative error across coupling types without changing the model family or loop. To keep semantics stable and avoid leakage, we keep the exact same per-type training, just transform `y` and apply the same OOF median-bias correction in the transformed space. Finally, we ensure deterministic, correct `id` alignment by writing predictions directly in `test` order (still matching `sample_submission` ids) and keeping the submission schema unchanged.'
- What this solution (achieved 1.99777) has done: 'Your current score (1.60317, lower-is-better) is still far from the target (-2.08099), so we should improve accuracy while keeping your per-type `HistGradientBoostingRegressor` training loop and the same overall feature style. The smallest high-impact, competition-specific improvement that doesn’t change model family/loop is to predict the 4 contribution terms (`fc, sd, pso, dso`) per type using the same features, then sum them to get `scalar_coupling_constant` (this aligns with how the target is generated and typically reduces MAE). We keep your existing direct-target model as a fallback and blend it lightly with the contribution-sum prediction for stability (still the same regressor, same per-type training). Submission writing and `id` alignment remain unchanged and deterministic.'
- What this solution (achieved 1.83998) has done: 'I fix the runtime error by ensuring the `type` column used for merging `train_fe` with `scalar_coupling_contributions.csv` stays as the original string labels (not the encoded int codes), since right now you encode `type` in-place and then try to merge int16 with object. To keep core logic unchanged, I preserve your encoded numeric `type` for modeling, but add a separate `type_str` column (and similarly keep string atoms only if needed) and use that only for the contribution merge keys. I also make the merge robust by validating and coercing dtypes for `atom_index_0/1` and by dropping duplicate rows in contributions on the key to avoid rare duplicate-key failures. These changes are correctness/stability-focused and should restore end-to-end execution and produce a valid `sub_ensemble.csv`.'
- What this solution (achieved 1.83922) has done: 'The timeout is dominated by repeated model fitting: for each coupling type you fit one “direct” model plus four contribution models, and each fit is done twice (once for final train and once inside the bias/OoF function), causing ~10 fits per type. The fastest equivalent fix is to reuse the already-trained model to compute the validation residual bias instead of training a second identical model, preserving the same split and exact semantics. Additionally, converting `molecule_name` to `category` once and reusing precomputed hashes/codes avoids repeated expensive hashing/Series creation, and switching join-heavy feature creation to `merge` on pre-indexed small frames reduces overhead without changing features. These changes keep the algorithm, features, models, and training loops intact while cutting fit-count roughly in half and reducing pandas overhead.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "42")
np.random.seed(42)

DATA_DIR_CANDIDATES = [
    "/kaggle/data/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling/champs-scalar-coupling",
    "/kaggle/input/champs-scalar-coupling",
    "/kaggle/input/champs-scalar-coupling/champs-scalar-coupling",
]


def resolve_data_dir():
    required = ["train.csv", "test.csv", "structures.csv", "sample_submission.csv"]
    for d in DATA_DIR_CANDIDATES:
        if all(os.path.exists(os.path.join(d, f)) for f in required):
            return d
    return "/kaggle/data/champs-scalar-coupling"


DATA_DIR = resolve_data_dir()

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
structures_path = os.path.join(DATA_DIR, "structures.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

mulliken_path = os.path.join(DATA_DIR, "mulliken_charges.csv")
shielding_path = os.path.join(DATA_DIR, "magnetic_shielding_tensors.csv")

dipole_path = os.path.join(DATA_DIR, "dipole_moments.csv")
energy_path = os.path.join(DATA_DIR, "potential_energy.csv")

contrib_path = os.path.join(DATA_DIR, "scalar_coupling_contributions.csv")

for p in [
    train_path,
    test_path,
    structures_path,
    sample_path,
    mulliken_path,
    shielding_path,
    dipole_path,
    energy_path,
    contrib_path,
]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing expected file: {p}")

train = pd.read_csv(
    train_path,
    usecols=[
        "id",
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "scalar_coupling_constant",
    ],
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
        "scalar_coupling_constant": np.float32,
    },
)
test = pd.read_csv(
    test_path,
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
    },
)
structures = pd.read_csv(
    structures_path,
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
    dtype={
        "molecule_name": "category",
        "atom_index": np.int16,
        "atom": "category",
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
    },
)

mulliken = pd.read_csv(
    mulliken_path,
    usecols=["molecule_name", "atom_index", "mulliken_charge"],
    dtype={
        "molecule_name": "category",
        "atom_index": np.int16,
        "mulliken_charge": np.float32,
    },
)
shielding = pd.read_csv(
    shielding_path,
    usecols=["molecule_name", "atom_index", "XX", "YY", "ZZ"],
    dtype={
        "molecule_name": "category",
        "atom_index": np.int16,
        "XX": np.float32,
        "YY": np.float32,
        "ZZ": np.float32,
    },
)

dipole = pd.read_csv(
    dipole_path,
    usecols=["molecule_name", "X", "Y", "Z"],
    dtype={
        "molecule_name": "category",
        "X": np.float32,
        "Y": np.float32,
        "Z": np.float32,
    },
)
energy = pd.read_csv(
    energy_path,
    usecols=["molecule_name", "potential_energy"],
    dtype={"molecule_name": "category", "potential_energy": np.float32},
)

contrib = pd.read_csv(
    contrib_path,
    usecols=[
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "fc",
        "sd",
        "pso",
        "dso",
    ],
    dtype={
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
        "fc": np.float32,
        "sd": np.float32,
        "pso": np.float32,
        "dso": np.float32,
    },
)

print("Using DATA_DIR:", DATA_DIR)
print("train/test/structures:", train.shape, test.shape, structures.shape)
print("mulliken/shielding:", mulliken.shape, shielding.shape)
print("dipole/energy:", dipole.shape, energy.shape)
print("contrib:", contrib.shape)
train.head()



## === cell 1
structures = structures[["molecule_name", "atom_index", "atom", "x", "y", "z"]].copy()

mol_feats = (
    structures.groupby("molecule_name", sort=False, observed=True)
    .agg(
        atom_count=("atom_index", "count"),
        x_mean=("x", "mean"),
        y_mean=("y", "mean"),
        z_mean=("z", "mean"),
        x_std=("x", "std"),
        y_std=("y", "std"),
        z_std=("z", "std"),
        x_min=("x", "min"),
        y_min=("y", "min"),
        z_min=("z", "min"),
        x_max=("x", "max"),
        y_max=("y", "max"),
        z_max=("z", "max"),
    )
    .reset_index()
)
mol_feats["x_range"] = (mol_feats["x_max"] - mol_feats["x_min"]).astype(np.float32)
mol_feats["y_range"] = (mol_feats["y_max"] - mol_feats["y_min"]).astype(np.float32)
mol_feats["z_range"] = (mol_feats["z_max"] - mol_feats["z_min"]).astype(np.float32)
mol_feats = mol_feats.drop(
    columns=["x_min", "y_min", "z_min", "x_max", "y_max", "z_max"]
)
for c in mol_feats.columns:
    if c != "molecule_name":
        mol_feats[c] = mol_feats[c].astype(np.float32)

shield_cols = ["XX", "YY", "ZZ"]
shielding = shielding[["molecule_name", "atom_index"] + shield_cols].copy()
shielding["shield_trace"] = (
    shielding["XX"] + shielding["YY"] + shielding["ZZ"]
).astype(np.float32)
shielding["shield_aniso"] = np.sqrt(
    (
        (shielding["XX"] - shielding["YY"]) ** 2
        + (shielding["YY"] - shielding["ZZ"]) ** 2
        + (shielding["ZZ"] - shielding["XX"]) ** 2
    ).astype(np.float32)
).astype(np.float32)

atom_phys = shielding.merge(
    mulliken, on=["molecule_name", "atom_index"], how="left", copy=False
)
for c in ["shield_trace", "shield_aniso", "mulliken_charge"] + shield_cols:
    atom_phys[c] = atom_phys[c].astype(np.float32)

dipole = dipole.rename(columns={"X": "dipole_x", "Y": "dipole_y", "Z": "dipole_z"})
dipole["dipole_norm"] = np.sqrt(
    (
        dipole["dipole_x"] ** 2 + dipole["dipole_y"] ** 2 + dipole["dipole_z"] ** 2
    ).astype(np.float32)
).astype(np.float32)

mol_phys = dipole.merge(energy, on="molecule_name", how="left", copy=False)
mol_phys["potential_energy"] = mol_phys["potential_energy"].astype(np.float32)

_s0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
    }
)[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]]
_s1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
    }
)[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]]

_ap0 = atom_phys.rename(
    columns={
        "atom_index": "atom_index_0",
        "mulliken_charge": "q0",
        "XX": "shield_xx0",
        "YY": "shield_yy0",
        "ZZ": "shield_zz0",
        "shield_trace": "shield_trace0",
        "shield_aniso": "shield_aniso0",
    }
)[
    [
        "molecule_name",
        "atom_index_0",
        "q0",
        "shield_xx0",
        "shield_yy0",
        "shield_zz0",
        "shield_trace0",
        "shield_aniso0",
    ]
]

_ap1 = atom_phys.rename(
    columns={
        "atom_index": "atom_index_1",
        "mulliken_charge": "q1",
        "XX": "shield_xx1",
        "YY": "shield_yy1",
        "ZZ": "shield_zz1",
        "shield_trace": "shield_trace1",
        "shield_aniso": "shield_aniso1",
    }
)[
    [
        "molecule_name",
        "atom_index_1",
        "q1",
        "shield_xx1",
        "shield_yy1",
        "shield_zz1",
        "shield_trace1",
        "shield_aniso1",
    ]
]

_s0i = _s0.set_index(["molecule_name", "atom_index_0"], drop=True)
_s1i = _s1.set_index(["molecule_name", "atom_index_1"], drop=True)
_ap0i = _ap0.set_index(["molecule_name", "atom_index_0"], drop=True)
_ap1i = _ap1.set_index(["molecule_name", "atom_index_1"], drop=True)
_mol_feats_i = mol_feats.set_index("molecule_name", drop=True)
_mol_phys_i = mol_phys.set_index("molecule_name", drop=True)


def add_pair_features(df, s0i, s1i, mol_feats_i, atom0_i, atom1_i, mol_phys_i):
    df = df.copy()

    df = df.join(s0i, on=["molecule_name", "atom_index_0"])
    df = df.join(s1i, on=["molecule_name", "atom_index_1"])

    dx = (df["x0"] - df["x1"]).astype(np.float32)
    dy = (df["y0"] - df["y1"]).astype(np.float32)
    dz = (df["z0"] - df["z1"]).astype(np.float32)
    dist2 = (dx * dx + dy * dy + dz * dz).astype(np.float32)
    dist = np.sqrt(dist2).astype(np.float32)

    df["dist"] = dist
    df["dist2"] = dist2
    invd = (1.0 / (dist + np.float32(1e-3))).astype(np.float32)
    df["inv_dist"] = invd

    df["abs_dx"] = np.abs(dx).astype(np.float32)
    df["abs_dy"] = np.abs(dy).astype(np.float32)
    df["abs_dz"] = np.abs(dz).astype(np.float32)

    df["cos_dx"] = (dx * invd).astype(np.float32)
    df["cos_dy"] = (dy * invd).astype(np.float32)
    df["cos_dz"] = (dz * invd).astype(np.float32)

    df = df.join(mol_feats_i, on="molecule_name")
    df = df.join(mol_phys_i, on="molecule_name")

    df["x0_c"] = (df["x0"].astype(np.float32) - df["x_mean"]).astype(np.float32)
    df["y0_c"] = (df["y0"].astype(np.float32) - df["y_mean"]).astype(np.float32)
    df["z0_c"] = (df["z0"].astype(np.float32) - df["z_mean"]).astype(np.float32)
    df["x1_c"] = (df["x1"].astype(np.float32) - df["x_mean"]).astype(np.float32)
    df["y1_c"] = (df["y1"].astype(np.float32) - df["y_mean"]).astype(np.float32)
    df["z1_c"] = (df["z1"].astype(np.float32) - df["z_mean"]).astype(np.float32)

    df["r0_c"] = np.sqrt(
        (
            df["x0_c"] * df["x0_c"] + df["y0_c"] * df["y0_c"] + df["z0_c"] * df["z0_c"]
        ).astype(np.float32)
    ).astype(np.float32)
    df["r1_c"] = np.sqrt(
        (
            df["x1_c"] * df["x1_c"] + df["y1_c"] * df["y1_c"] + df["z1_c"] * df["z1_c"]
        ).astype(np.float32)
    ).astype(np.float32)

    dot_c = (
        df["x0_c"] * df["x1_c"] + df["y0_c"] * df["y1_c"] + df["z0_c"] * df["z1_c"]
    ).astype(np.float32)
    df["dot_c"] = dot_c
    df["cos_c"] = (dot_c / (df["r0_c"] * df["r1_c"] + np.float32(1e-3))).astype(
        np.float32
    )

    df = df.join(atom0_i, on=["molecule_name", "atom_index_0"])
    df = df.join(atom1_i, on=["molecule_name", "atom_index_1"])

    df["dq"] = (df["q0"] - df["q1"]).astype(np.float32)
    df["q_sum"] = (df["q0"] + df["q1"]).astype(np.float32)
    df["d_shield_trace"] = (df["shield_trace0"] - df["shield_trace1"]).astype(
        np.float32
    )
    df["shield_trace_sum"] = (df["shield_trace0"] + df["shield_trace1"]).astype(
        np.float32
    )
    df["shield_aniso_sum"] = (df["shield_aniso0"] + df["shield_aniso1"]).astype(
        np.float32
    )

    df = df.drop(columns=["x0", "y0", "z0", "x1", "y1", "z1"])
    return df


train_fe = add_pair_features(train, _s0i, _s1i, _mol_feats_i, _ap0i, _ap1i, _mol_phys_i)
test_fe = add_pair_features(test, _s0i, _s1i, _mol_feats_i, _ap0i, _ap1i, _mol_phys_i)

na_cols = [
    "atom_0",
    "atom_1",
    "dist",
    "dist2",
    "inv_dist",
    "atom_count",
    "abs_dx",
    "abs_dy",
    "abs_dz",
    "cos_dx",
    "cos_dy",
    "cos_dz",
    "r0_c",
    "r1_c",
    "dot_c",
    "cos_c",
    "q0",
    "q1",
    "shield_trace0",
    "shield_trace1",
    "dipole_x",
    "dipole_y",
    "dipole_z",
    "potential_energy",
]
train_na = train_fe[na_cols].isna().any(axis=1).mean()
test_na = test_fe[na_cols].isna().any(axis=1).mean()
print(f"NaN rate after merge - train: {train_na:.6f}, test: {test_na:.6f}")

train_fe.head()



## === cell 2
cat_cols = ["type", "atom_0", "atom_1"]

for c in ["atom_0", "atom_1"]:
    train_fe[c] = train_fe[c].cat.add_categories(["__UNK__"]).fillna("__UNK__")
    test_fe[c] = test_fe[c].cat.add_categories(["__UNK__"]).fillna("__UNK__")

num_fill_cols = [
    "dist",
    "dist2",
    "inv_dist",
    "abs_dx",
    "abs_dy",
    "abs_dz",
    "cos_dx",
    "cos_dy",
    "cos_dz",
    "x0_c",
    "y0_c",
    "z0_c",
    "x1_c",
    "y1_c",
    "z1_c",
    "r0_c",
    "r1_c",
    "dot_c",
    "cos_c",
    "atom_count",
    "x_mean",
    "y_mean",
    "z_mean",
    "x_std",
    "y_std",
    "z_std",
    "x_range",
    "y_range",
    "z_range",
    "q0",
    "q1",
    "dq",
    "q_sum",
    "shield_xx0",
    "shield_yy0",
    "shield_zz0",
    "shield_trace0",
    "shield_aniso0",
    "shield_xx1",
    "shield_yy1",
    "shield_zz1",
    "shield_trace1",
    "shield_aniso1",
    "d_shield_trace",
    "shield_trace_sum",
    "shield_aniso_sum",
    "dipole_x",
    "dipole_y",
    "dipole_z",
    "dipole_norm",
    "potential_energy",
]

meds = train_fe[num_fill_cols].median(numeric_only=True)
fill_values = {
    c: float(meds.get(c)) if pd.notna(meds.get(c, np.nan)) else 0.0
    for c in num_fill_cols
}
train_fe[num_fill_cols] = train_fe[num_fill_cols].fillna(fill_values).astype(np.float32)
test_fe[num_fill_cols] = test_fe[num_fill_cols].fillna(fill_values).astype(np.float32)

full_cat = pd.concat([train_fe[cat_cols], test_fe[cat_cols]], axis=0, ignore_index=True)
for c in cat_cols:
    full_cat[c] = full_cat[c].astype("category")
cats = {c: full_cat[c].cat.categories for c in cat_cols}
for c in cat_cols:
    train_fe[c] = (
        train_fe[c]
        .astype(pd.CategoricalDtype(categories=cats[c]))
        .cat.codes.astype(np.int16)
    )
    test_fe[c] = (
        test_fe[c]
        .astype(pd.CategoricalDtype(categories=cats[c]))
        .cat.codes.astype(np.int16)
    )

train_fe["type_code"] = train_fe["type"].astype(np.int16)
test_fe["type_code"] = test_fe["type"].astype(np.int16)

feature_cols = [
    "type",
    "atom_0",
    "atom_1",
    "dist",
    "dist2",
    "inv_dist",
    "abs_dx",
    "abs_dy",
    "abs_dz",
    "cos_dx",
    "cos_dy",
    "cos_dz",
    "x0_c",
    "y0_c",
    "z0_c",
    "x1_c",
    "y1_c",
    "z1_c",
    "r0_c",
    "r1_c",
    "dot_c",
    "cos_c",
    "atom_count",
    "x_mean",
    "y_mean",
    "z_mean",
    "x_std",
    "y_std",
    "z_std",
    "x_range",
    "y_range",
    "z_range",
    "q0",
    "q1",
    "dq",
    "q_sum",
    "shield_xx0",
    "shield_yy0",
    "shield_zz0",
    "shield_trace0",
    "shield_aniso0",
    "shield_xx1",
    "shield_yy1",
    "shield_zz1",
    "shield_trace1",
    "shield_aniso1",
    "d_shield_trace",
    "shield_trace_sum",
    "shield_aniso_sum",
    "dipole_x",
    "dipole_y",
    "dipole_z",
    "dipole_norm",
    "potential_energy",
]
X_train_all = train_fe[feature_cols]
y_train_all = train_fe["scalar_coupling_constant"].astype(np.float32)
X_test_all = test_fe[feature_cols]

X_train_all.head()



## === cell 3
from sklearn.ensemble import HistGradientBoostingRegressor

RANDOM_STATE = 42

test_pred = np.zeros(len(test_fe), dtype=np.float32)

train_types = train_fe["type_code"].unique()
test_types = test_fe["type_code"].unique()

missing_types = set(test_types) - set(train_types)
if missing_types:
    print(
        "Warning: types in test but not in train (will predict 0 for them):",
        missing_types,
    )

per_type_features = [
    "atom_0",
    "atom_1",
    "dist",
    "dist2",
    "inv_dist",
    "abs_dx",
    "abs_dy",
    "abs_dz",
    "cos_dx",
    "cos_dy",
    "cos_dz",
    "x0_c",
    "y0_c",
    "z0_c",
    "x1_c",
    "y1_c",
    "z1_c",
    "r0_c",
    "r1_c",
    "dot_c",
    "cos_c",
    "atom_count",
    "x_mean",
    "y_mean",
    "z_mean",
    "x_std",
    "y_std",
    "z_std",
    "x_range",
    "y_range",
    "z_range",
    "q0",
    "q1",
    "dq",
    "q_sum",
    "shield_xx0",
    "shield_yy0",
    "shield_zz0",
    "shield_trace0",
    "shield_aniso0",
    "shield_xx1",
    "shield_yy1",
    "shield_zz1",
    "shield_trace1",
    "shield_aniso1",
    "d_shield_trace",
    "shield_trace_sum",
    "shield_aniso_sum",
    "dipole_x",
    "dipole_y",
    "dipole_z",
    "dipole_norm",
    "potential_energy",
]


def _y_to_log_abs(y):
    y = y.astype(np.float32, copy=False)
    return np.log1p(np.abs(y)).astype(np.float32)


def _log_abs_to_y(log_abs_pred, sign_hint):
    mag = np.expm1(log_abs_pred.astype(np.float32, copy=False)).astype(np.float32)
    return (mag * np.float32(sign_hint)).astype(np.float32)


def bias_from_trained_model_logspace(model, Xtr_np, ytr_np, val_mask):
    val_rate = float(val_mask.mean())
    if val_rate < 0.01 or val_rate > 0.5:
        return 0.0
    X_val = Xtr_np[val_mask]
    y_val_t = _y_to_log_abs(ytr_np[val_mask])
    val_pred_t = model.predict(X_val).astype(np.float32)
    resid_t = y_val_t - val_pred_t
    return float(np.median(resid_t))


base_params = dict(
    loss="absolute_error",
    max_depth=8,
    max_iter=300,
    learning_rate=0.05,
    min_samples_leaf=40,
    l2_regularization=0.0,
    random_state=RANDOM_STATE,
)

contrib_key = ["molecule_name", "atom_index_0", "atom_index_1", "type"]
contrib_small = contrib[contrib_key + ["fc", "sd", "pso", "dso"]].copy()
contrib_small = contrib_small.drop_duplicates(subset=contrib_key, keep="first")

train_merge_base = train_fe.copy()
train_with_contrib = train_merge_base.merge(
    contrib_small, on=contrib_key, how="left", validate="one_to_one", copy=False
)
miss_rate = train_with_contrib[["fc", "sd", "pso", "dso"]].isna().any(axis=1).mean()
print(f"Contribution merge missing rate in train: {miss_rate:.6f}")
for cc in ["fc", "sd", "pso", "dso"]:
    train_with_contrib[cc] = train_with_contrib[cc].fillna(0.0).astype(np.float32)

X_train_np = X_train_all[per_type_features].to_numpy(dtype=np.float32, copy=False)
X_test_np = X_test_all[per_type_features].to_numpy(dtype=np.float32, copy=False)
y_train_np = y_train_all.to_numpy(dtype=np.float32, copy=False)

test_type_code = test_fe["type_code"].to_numpy(dtype=np.int16, copy=False)
train_type_code = train_fe["type_code"].to_numpy(dtype=np.int16, copy=False)

train_mol_hash = pd.util.hash_pandas_object(
    train_fe["molecule_name"], index=False
).to_numpy()
train_val_mask_all = (train_mol_hash % 10) == 0

test_pred_direct = np.zeros(len(test_fe), dtype=np.float32)

for t in np.sort(test_types):
    test_mask = test_type_code == t
    train_mask = train_type_code == t
    if not np.any(train_mask):
        continue

    Xtr = X_train_np[train_mask]
    ytr = y_train_np[train_mask]
    Xte = X_test_np[test_mask]

    ytr_t = _y_to_log_abs(ytr)

    model = HistGradientBoostingRegressor(**base_params)
    model.fit(Xtr, ytr_t)

    val_mask_t = train_val_mask_all[train_mask]
    bias_t = bias_from_trained_model_logspace(model, Xtr, ytr, val_mask_t)

    sign_hint = 1.0 if float(np.median(ytr)) >= 0.0 else -1.0
    pred_t = (model.predict(Xte).astype(np.float32) + np.float32(bias_t)).astype(
        np.float32
    )
    test_pred_direct[test_mask] = _log_abs_to_y(pred_t, sign_hint)

train_with_contrib_type = train_with_contrib["type_code"].to_numpy(
    dtype=np.int16, copy=False
)
train_with_contrib_mol_hash = pd.util.hash_pandas_object(
    train_with_contrib["molecule_name"], index=False
).to_numpy()
train_with_contrib_val_mask_all = (train_with_contrib_mol_hash % 10) == 0
X_train_contrib_np = train_with_contrib[per_type_features].to_numpy(
    dtype=np.float32, copy=False
)

test_pred_contribsum = np.zeros(len(test_fe), dtype=np.float32)
components = ["fc", "sd", "pso", "dso"]

for t in np.sort(test_types):
    test_mask = test_type_code == t
    train_mask = train_with_contrib_type == t
    if not np.any(train_mask):
        continue

    Xtr = X_train_contrib_np[train_mask]
    Xte = X_test_np[test_mask]

    comp_sum = np.zeros(int(test_mask.sum()), dtype=np.float32)
    val_mask_t = train_with_contrib_val_mask_all[train_mask]

    for cc in components:
        ycc = train_with_contrib.loc[train_mask, cc].to_numpy(
            dtype=np.float32, copy=False
        )
        ycc_t = _y_to_log_abs(ycc)

        m = HistGradientBoostingRegressor(**base_params)
        m.fit(Xtr, ycc_t)

        bias_cc = bias_from_trained_model_logspace(m, Xtr, ycc, val_mask_t)

        sign_hint_cc = 1.0 if float(np.median(ycc)) >= 0.0 else -1.0
        pred_cc_t = (m.predict(Xte).astype(np.float32) + np.float32(bias_cc)).astype(
            np.float32
        )
        comp_sum += _log_abs_to_y(pred_cc_t, sign_hint_cc)

    test_pred_contribsum[test_mask] = comp_sum

alpha = np.float32(0.80)
test_pred = (
    alpha * test_pred_contribsum + (np.float32(1.0) - alpha) * test_pred_direct
).astype(np.float32)

print(
    "Pred stats:",
    float(np.min(test_pred)),
    float(np.mean(test_pred)),
    float(np.max(test_pred)),
)



## === cell 4
sample = pd.read_csv(sample_path, usecols=["id"], dtype={"id": np.int32})[["id"]].copy()

submission = pd.DataFrame(
    {"id": test["id"].values, "scalar_coupling_constant": test_pred.astype(np.float32)}
)

submission = sample.merge(submission, on="id", how="left", validate="one_to_one")
submission["scalar_coupling_constant"] = (
    submission["scalar_coupling_constant"].fillna(0.0).astype(np.float32)
)

out_path = "sub_ensemble.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path, "rows:", len(submission))
submission.head()
