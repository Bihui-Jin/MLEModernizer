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

1.63506

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.99777) has done: 'I remove the dependency on missing `../input/champ-preds/*.csv` files (which causes the `FileNotFoundError`) and instead build a simple, robust baseline model directly from the provided CHAMPS dataset files in `/kaggle/data/champs-scalar-coupling/`. To keep the core approach minimal and stable, the solution use lightweight feature engineering (atom types + pairwise distance + coupling type) and train one `HistGradientBoostingRegressor` per coupling `type`, then predict test rows of that type. Finally, it write a valid `sub_ensemble.csv` with exactly `id,scalar_coupling_constant` columns.'
- What this solution (achieved 1.74698) has done: 'The NaNs come from a path mismatch: `train.csv`/`test.csv` reference molecule names that are not present in the `structures.csv` you loaded (different dataset copy/folder), so the merges fail and produce missing atom coordinates. I fix this by loading *all* files (train/test/structures/sample) from the same base directory and adding a small validation that auto-switches to the alternate mirrored folder if needed. This is a correctness/stability fix (not a modeling change) and should restore the previous working behavior and score trajectory. I also make the categorical mapping robust to any remaining missing values by using an explicit “__UNK__” category, preventing `IntCastingNaNError` and ensuring a valid `sub_ensemble.csv` is always written.'
- What this solution (achieved 1.68538) has done: 'Your current score (1.74698, lower-is-better) is far worse than the target (-2.08099), so we should improve performance but keep the same overall approach (per-type HistGradientBoostingRegressor on simple geometric/categorical features). The biggest minimal win within your existing feature paradigm is to add a couple of cheap, physically meaningful pairwise features (squared distance and inverse distance) and to include simple per-molecule aggregate context (atom count + mean/std of coordinates) derived only from `structures.csv`. These additions don’t change the training loop or model family, but usually reduce MAE substantially for this competition’s baseline models. I also ensure the submission alignment is strictly by `id` (no unnecessary merge) to avoid any accidental ordering issues.'
- What this solution (achieved 1.6489) has done: 'Your current score (1.68538, lower-is-better) is still far from the target (-2.08099), so we should improve it while keeping the same per-type `HistGradientBoostingRegressor` approach and the same overall feature style. The smallest high-impact improvement here is to add a few more physically meaningful but cheap geometric features (absolute coordinate deltas, direction cosines, and centered coordinates relative to molecule centroid) without changing the model family or training loop. These features usually reduce MAE significantly for CHAMPS because coupling strength depends on both distance and relative orientation within a molecule. I also keep the submission alignment-by-`id` logic intact, only ensuring feature columns are filled deterministically.'
- What this solution (achieved 1.63506) has done: 'Your current score (1.6489, lower-is-better) is still far from the target (-2.08099), so we should improve it while keeping the same per-type `HistGradientBoostingRegressor` training loop and the same “simple geometry + categorical atoms/types” feature style. The smallest high-impact, competition-specific improvement that stays within your core logic is to add the provided per-atom physics features (Mulliken charge + magnetic shielding tensor diagonals) by merging them for atom_0 and atom_1, then training the same per-type models on the extended feature set. This typically reduces MAE materially on CHAMPS without changing the model family, loss, or training approach. I also add a couple of cheap derived tensor summaries (trace and anisotropy magnitude) and keep the submission writing exactly `id,scalar_coupling_constant` to ensure validity.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

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

for p in [
    train_path,
    test_path,
    structures_path,
    sample_path,
    mulliken_path,
    shielding_path,
]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing expected file: {p}")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
structures = pd.read_csv(structures_path)

mulliken = pd.read_csv(mulliken_path)
shielding = pd.read_csv(shielding_path)

print("Using DATA_DIR:", DATA_DIR)
print("train/test/structures:", train.shape, test.shape, structures.shape)
print("mulliken/shielding:", mulliken.shape, shielding.shape)
train.head()



## === cell 1
structures = structures[["molecule_name", "atom_index", "atom", "x", "y", "z"]].copy()
structures["atom_index"] = structures["atom_index"].astype(np.int16)

mol_feats = (
    structures.groupby("molecule_name")
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

mulliken = mulliken[["molecule_name", "atom_index", "mulliken_charge"]].copy()
mulliken["atom_index"] = mulliken["atom_index"].astype(np.int16)
mulliken["mulliken_charge"] = mulliken["mulliken_charge"].astype(np.float32)

shield_cols = ["XX", "YY", "ZZ"]  # keep minimal + informative
shielding = shielding[["molecule_name", "atom_index"] + shield_cols].copy()
shielding["atom_index"] = shielding["atom_index"].astype(np.int16)
for c in shield_cols:
    shielding[c] = shielding[c].astype(np.float32)

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

atom_phys = shielding.merge(mulliken, on=["molecule_name", "atom_index"], how="left")
for c in ["shield_trace", "shield_aniso", "mulliken_charge"] + shield_cols:
    atom_phys[c] = atom_phys[c].astype(np.float32)


def add_pair_features(df, structures_df, mol_feats_df, atom_phys_df):
    df = df.copy()
    df["atom_index_0"] = df["atom_index_0"].astype(np.int16)
    df["atom_index_1"] = df["atom_index_1"].astype(np.int16)

    s0 = structures_df.rename(
        columns={
            "atom_index": "atom_index_0",
            "atom": "atom_0",
            "x": "x0",
            "y": "y0",
            "z": "z0",
        }
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

    df = df.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
    df = df.merge(s1, on=["molecule_name", "atom_index_1"], how="left")

    dx = (df["x0"] - df["x1"]).astype(np.float32)
    dy = (df["y0"] - df["y1"]).astype(np.float32)
    dz = (df["z0"] - df["z1"]).astype(np.float32)
    dist2 = (dx * dx + dy * dy + dz * dz).astype(np.float32)
    dist = np.sqrt(dist2).astype(np.float32)

    df["dist"] = dist
    df["dist2"] = dist2
    df["inv_dist"] = (1.0 / (dist + np.float32(1e-3))).astype(np.float32)

    df["abs_dx"] = np.abs(dx).astype(np.float32)
    df["abs_dy"] = np.abs(dy).astype(np.float32)
    df["abs_dz"] = np.abs(dz).astype(np.float32)

    invd = (1.0 / (dist + np.float32(1e-3))).astype(np.float32)
    df["cos_dx"] = (dx * invd).astype(np.float32)
    df["cos_dy"] = (dy * invd).astype(np.float32)
    df["cos_dz"] = (dz * invd).astype(np.float32)

    df = df.merge(mol_feats_df, on="molecule_name", how="left")

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

    ap0 = atom_phys_df.rename(
        columns={
            "atom_index": "atom_index_0",
            "mulliken_charge": "q0",
            "XX": "shield_xx0",
            "YY": "shield_yy0",
            "ZZ": "shield_zz0",
            "shield_trace": "shield_trace0",
            "shield_aniso": "shield_aniso0",
        }
    )
    ap1 = atom_phys_df.rename(
        columns={
            "atom_index": "atom_index_1",
            "mulliken_charge": "q1",
            "XX": "shield_xx1",
            "YY": "shield_yy1",
            "ZZ": "shield_zz1",
            "shield_trace": "shield_trace1",
            "shield_aniso": "shield_aniso1",
        }
    )
    df = df.merge(
        ap0[
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
        ],
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    df = df.merge(
        ap1[
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
        ],
        on=["molecule_name", "atom_index_1"],
        how="left",
    )

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


train_fe = add_pair_features(train, structures, mol_feats, atom_phys)
test_fe = add_pair_features(test, structures, mol_feats, atom_phys)

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
    "q0",
    "q1",
    "shield_trace0",
    "shield_trace1",
]
train_na = train_fe[na_cols].isna().any(axis=1).mean()
test_na = test_fe[na_cols].isna().any(axis=1).mean()
print(f"NaN rate after merge - train: {train_na:.6f}, test: {test_na:.6f}")

train_fe.head()



## === cell 2
cat_cols = ["type", "atom_0", "atom_1"]

for c in ["atom_0", "atom_1"]:
    train_fe[c] = train_fe[c].fillna("__UNK__")
    test_fe[c] = test_fe[c].fillna("__UNK__")

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
]

fill_values = {}
for c in num_fill_cols:
    if train_fe[c].notna().any():
        fill_values[c] = float(train_fe[c].median())
    else:
        fill_values[c] = 0.0

for c in num_fill_cols:
    train_fe[c] = train_fe[c].fillna(fill_values[c]).astype(np.float32)
    test_fe[c] = test_fe[c].fillna(fill_values[c]).astype(np.float32)

full_cat = pd.concat([train_fe[cat_cols], test_fe[cat_cols]], axis=0, ignore_index=True)
for c in cat_cols:
    full_cat[c] = full_cat[c].astype("category")

cat_maps = {
    c: {k: i for i, k in enumerate(full_cat[c].cat.categories)} for c in cat_cols
}


def apply_cat_maps(df, maps):
    df = df.copy()
    for c, mp in maps.items():
        codes = df[c].map(mp)
        if codes.isna().any():
            unk_code = mp.get("__UNK__", 0)
            codes = codes.fillna(unk_code)
        df[c] = codes.astype(np.int16)
    return df


train_fe = apply_cat_maps(train_fe, cat_maps)
test_fe = apply_cat_maps(test_fe, cat_maps)

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
]
X_train_all = train_fe[feature_cols]
y_train_all = train_fe["scalar_coupling_constant"].astype(np.float32)
X_test_all = test_fe[feature_cols]

X_train_all.head()



## === cell 3
from sklearn.ensemble import HistGradientBoostingRegressor

RANDOM_STATE = 42

test_pred = np.zeros(len(test_fe), dtype=np.float32)

train_types = train_fe["type"].unique()
test_types = test_fe["type"].unique()

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
]

for t in sorted(test_types):
    test_mask = test_fe["type"].values == t
    train_mask = train_fe["type"].values == t

    if not np.any(train_mask):
        continue

    Xtr = X_train_all.loc[train_mask, per_type_features]
    ytr = y_train_all.loc[train_mask]
    Xte = X_test_all.loc[test_mask, per_type_features]

    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_depth=8,
        max_iter=300,
        learning_rate=0.05,
        min_samples_leaf=40,
        l2_regularization=0.0,
        random_state=RANDOM_STATE,
    )
    model.fit(Xtr, ytr)
    test_pred[test_mask] = model.predict(Xte).astype(np.float32)

print(
    "Pred stats:",
    float(np.min(test_pred)),
    float(np.mean(test_pred)),
    float(np.max(test_pred)),
)



## === cell 4
submission = pd.read_csv(sample_path)[["id"]].copy()

pred_by_id = pd.DataFrame(
    {"id": test["id"].values, "scalar_coupling_constant": test_pred}
)
submission = submission.merge(pred_by_id, on="id", how="left", validate="one_to_one")

submission["scalar_coupling_constant"] = (
    submission["scalar_coupling_constant"].fillna(0.0).astype(np.float32)
)

out_path = "sub_ensemble.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path, "rows:", len(submission))
submission.head()
