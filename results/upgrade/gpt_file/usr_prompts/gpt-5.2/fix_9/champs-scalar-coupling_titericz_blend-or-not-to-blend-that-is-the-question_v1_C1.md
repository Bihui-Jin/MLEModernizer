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

-1.2781586533261562

# 6. Current score

1.23566

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23578) has done: 'The current notebook fails because it tries to read two external submissions from Kaggle Datasets that are not available in your environment, so `sub1`/`sub2` are never created and downstream cells crash. To make it run end-to-end and always produce a valid `.csv` submission, I replace those missing inputs with a simple, deterministic baseline model trained only from the provided `train.csv`/`test.csv` (group mean by coupling `type`). This preserves the “blending” idea by creating two lightweight baseline predictors and then weighted-averaging them, but without relying on unavailable files. The result writes `weighted-avg-blend-lgb-keras-1.csv` with the required `id,scalar_coupling_constant` columns.'
- What this solution (achieved 1.23566) has done: 'I fix the crash by preventing NaNs from propagating into `dist_bin` and other integer-binned features, which currently happens when atom coordinate merges fail (missing structure rows) and `dist` becomes non-finite. The minimal change is to make `add_features()` robust (drop duplicate structure rows, validate merge cardinality, and fill missing coordinates with molecule-level means) and to compute bins using pandas’ nullable integer type before filling. This preserves the existing “two predictors + weighted blend” core logic and ensures the notebook always reaches the CSV write step. Finally, I add a small submission sanity check (column names, length, and id sorting) so Kaggle accepts the output.'
- What this solution (achieved 1.23566) has done: 'Your current score (1.23566, lower-is-better) is far from the target (-1.278...), so we need a legitimate improvement while keeping the same “group-mean baselines + blended prediction” core logic. The smallest high-impact change is to prevent noisy/rare `dist_bin/pair/mol_n_atoms_bin` group means from dominating by increasing the shrinkage (k) and only trusting group statistics when there is sufficient support. We also use a more stable `dist` binning (clipping extreme bins) to reduce tail sparsity without changing the feature set. These changes keep the same feature engineering and prediction semantics (type mean + smoothed group mean + weighted blend), but should move the score downward (better) toward the target.'
- What this solution (achieved 1.23566) has done: 'Your current score (1.23566, lower-is-better) is extremely far from the target (-1.278...), so we should make a legitimate accuracy improvement while keeping the same overall “type mean + smoothed group mean + weighted blend” approach. The smallest high-impact upgrade is to compute the smoothed group statistics in an out-of-fold (OOF) manner by splitting on `molecule_name`, so the group means used for each training row don’t leak its own target; this typically produces better-calibrated smoothing that generalizes better to the test-by-molecule split. We then fit the final group statistics on all training data (as before) for test prediction, and we keep the same feature set and the same blend weights. This change is directly tied to the competition’s molecule-based split and should move the score downward toward your target without changing the core modeling semantics.'
- What this solution (achieved 1.23566) has done: 'Your current score (1.23566, lower-is-better) is far from the target (-1.278...), so we should make a legitimate accuracy improvement while preserving your existing “type mean + smoothed group mean + weighted blend” logic. The biggest easy win without changing the model family is to add a few more atom/molecule context features that are already available (mulliken charges, magnetic shielding tensors, dipole moments, potential energy) and include them in the same smoothed group-mean framework. This keeps the exact same prediction semantics (shrunk group means back to type mean) but makes the groups much more informative, which should reduce logMAE. I also keep the submission creation unchanged and add only lightweight numeric aggregates so runtime stays within limits.'
- What this solution (achieved 1.23566) has done: 'We move the score downward (better) by making the smoothed group-mean predictor generalize closer to the competition’s molecule-based split, without changing your overall “type mean + smoothed group mean + weighted blend” approach. The minimal high-impact fix is to make the smoothing parameters selection (k and min count) match the leaderboard metric more faithfully by computing OOF logMAE on a stratified-by-type molecule split and using the *global* training mean (not per-fold mean) as the fallback, reducing fold-to-fold bias. We also add one more candidate value for k and min_count (small search expansion) to avoid getting stuck in a poor local choice while keeping runtime within limits. Finally, we keep the same submission generation, but ensure predictions are always finite.'
- What this solution (achieved 1.23566) has done: 'Your current score (1.23566, lower-is-better) is still very far from the target (-1.278...), so we should make a legitimate accuracy improvement while keeping the same core “type mean + smoothed group mean + weighted blend” approach. The biggest minimal-change gain here is to compute group statistics in a way that respects the competition’s *molecule-split* and reduces sparsity: (1) make the atom-pair feature order-invariant (canonicalize pair and swap indices/atom features when needed), and (2) compute the distance-based bin as a *type-specific quantile bin* (still a binning of dist, but far less sparse than a fixed 0.1Å grid). These changes preserve your modeling semantics (same smoothing, same blending), but should noticeably reduce noise from rare bins and improve generalization. I also keep your OOF-based parameter selection, but adapt it to the new bin column name and add small safety checks to keep everything finite and submission-valid.'

# 9. Code solution

## === cell 0
import os

DATA_ROOT = "/kaggle/data/champs-scalar-coupling"
ALT_DATA_ROOT = "/kaggle/input/champs-scalar-coupling"

if not os.path.exists(DATA_ROOT) and os.path.exists(ALT_DATA_ROOT):
    DATA_ROOT = ALT_DATA_ROOT

print("DATA_ROOT =", DATA_ROOT)
print("Files in DATA_ROOT (sample):", sorted(os.listdir(DATA_ROOT))[:10])



## === cell 1
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print(os.listdir("/kaggle/data"))



## === cell 2
train_path = os.path.join(DATA_ROOT, "train.csv")
test_path = os.path.join(DATA_ROOT, "test.csv")
structures_path = os.path.join(DATA_ROOT, "structures.csv")
mulliken_path = os.path.join(DATA_ROOT, "mulliken_charges.csv")
shielding_path = os.path.join(DATA_ROOT, "magnetic_shielding_tensors.csv")
dipole_path = os.path.join(DATA_ROOT, "dipole_moments.csv")
energy_path = os.path.join(DATA_ROOT, "potential_energy.csv")

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
)
test = pd.read_csv(
    test_path, usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"]
)

structures = pd.read_csv(
    structures_path, usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"]
)

structures = structures.drop_duplicates(
    subset=["molecule_name", "atom_index"], keep="first"
)

mulliken = pd.read_csv(
    mulliken_path, usecols=["molecule_name", "atom_index", "mulliken_charge"]
)
mulliken = mulliken.drop_duplicates(
    subset=["molecule_name", "atom_index"], keep="first"
)

shield = pd.read_csv(
    shielding_path,
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
shield = shield.drop_duplicates(subset=["molecule_name", "atom_index"], keep="first")

shield["shield_trace"] = (shield["XX"] + shield["YY"] + shield["ZZ"]).astype(np.float64)
shield["shield_frob"] = np.sqrt(
    (
        shield[["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]].astype(
            np.float64
        )
        ** 2
    ).sum(axis=1)
).astype(np.float64)
shield = shield[["molecule_name", "atom_index", "shield_trace", "shield_frob"]]

dipole = pd.read_csv(dipole_path, usecols=["molecule_name", "X", "Y", "Z"])
dipole["dipole_norm"] = np.sqrt(
    (dipole[["X", "Y", "Z"]].astype(np.float64) ** 2).sum(axis=1)
).astype(np.float64)
dipole = dipole.rename(columns={"X": "dipole_x", "Y": "dipole_y", "Z": "dipole_z"})[
    ["molecule_name", "dipole_x", "dipole_y", "dipole_z", "dipole_norm"]
]

energy = pd.read_csv(
    energy_path, usecols=["molecule_name", "potential_energy"]
).drop_duplicates(subset=["molecule_name"], keep="first")

mol_agg = (
    structures.groupby("molecule_name", sort=False)
    .agg(
        mol_x_mean=("x", "mean"),
        mol_y_mean=("y", "mean"),
        mol_z_mean=("z", "mean"),
        mol_n_atoms=("atom_index", "count"),
    )
    .reset_index()
)

coords = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
    }
)
coords2 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
    }
)

mull0 = mulliken.rename(columns={"atom_index": "atom_index_0", "mulliken_charge": "q0"})
mull1 = mulliken.rename(columns={"atom_index": "atom_index_1", "mulliken_charge": "q1"})
sh0 = shield.rename(
    columns={
        "atom_index": "atom_index_0",
        "shield_trace": "sh_trace0",
        "shield_frob": "sh_frob0",
    }
)
sh1 = shield.rename(
    columns={
        "atom_index": "atom_index_1",
        "shield_trace": "sh_trace1",
        "shield_frob": "sh_frob1",
    }
)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(
        coords[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
        validate="m:1",
    )
    df = df.merge(
        coords2[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
        validate="m:1",
    )
    df = df.merge(mol_agg, on="molecule_name", how="left", validate="m:1")

    df = df.merge(
        mull0[["molecule_name", "atom_index_0", "q0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
        validate="m:1",
    )
    df = df.merge(
        mull1[["molecule_name", "atom_index_1", "q1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
        validate="m:1",
    )
    df = df.merge(
        sh0[["molecule_name", "atom_index_0", "sh_trace0", "sh_frob0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
        validate="m:1",
    )
    df = df.merge(
        sh1[["molecule_name", "atom_index_1", "sh_trace1", "sh_frob1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
        validate="m:1",
    )
    df = df.merge(dipole, on="molecule_name", how="left", validate="m:1")
    df = df.merge(energy, on="molecule_name", how="left", validate="m:1")

    for c in ["x0", "y0", "z0", "x1", "y1", "z1"]:
        df[c] = df[c].astype(np.float64)
    df["mol_x_mean"] = df["mol_x_mean"].astype(np.float64)
    df["mol_y_mean"] = df["mol_y_mean"].astype(np.float64)
    df["mol_z_mean"] = df["mol_z_mean"].astype(np.float64)

    df["x0"] = df["x0"].fillna(df["mol_x_mean"]).fillna(0.0)
    df["y0"] = df["y0"].fillna(df["mol_y_mean"]).fillna(0.0)
    df["z0"] = df["z0"].fillna(df["mol_z_mean"]).fillna(0.0)
    df["x1"] = df["x1"].fillna(df["mol_x_mean"]).fillna(0.0)
    df["y1"] = df["y1"].fillna(df["mol_y_mean"]).fillna(0.0)
    df["z1"] = df["z1"].fillna(df["mol_z_mean"]).fillna(0.0)

    dx = (df["x0"] - df["x1"]).astype(np.float64)
    dy = (df["y0"] - df["y1"]).astype(np.float64)
    dz = (df["z0"] - df["z1"]).astype(np.float64)
    df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float64)

    cdx0 = (df["x0"] - df["mol_x_mean"]).astype(np.float64)
    cdy0 = (df["y0"] - df["mol_y_mean"]).astype(np.float64)
    cdz0 = (df["z0"] - df["mol_z_mean"]).astype(np.float64)
    cdx1 = (df["x1"] - df["mol_x_mean"]).astype(np.float64)
    cdy1 = (df["y1"] - df["mol_y_mean"]).astype(np.float64)
    cdz1 = (df["z1"] - df["mol_z_mean"]).astype(np.float64)
    df["r0"] = np.sqrt(cdx0 * cdx0 + cdy0 * cdy0 + cdz0 * cdz0).astype(np.float64)
    df["r1"] = np.sqrt(cdx1 * cdx1 + cdy1 * cdy1 + cdz1 * cdz1).astype(np.float64)

    swap = (df["atom_0"].astype(str) > df["atom_1"].astype(str)).values
    if swap.any():
        for a, b in [
            ("atom_index_0", "atom_index_1"),
            ("atom_0", "atom_1"),
            ("x0", "x1"),
            ("y0", "y1"),
            ("z0", "z1"),
            ("r0", "r1"),
            ("q0", "q1"),
            ("sh_trace0", "sh_trace1"),
            ("sh_frob0", "sh_frob1"),
        ]:
            tmp = df.loc[swap, a].copy()
            df.loc[swap, a] = df.loc[swap, b].values
            df.loc[swap, b] = tmp.values

    df["pair"] = df["atom_0"].astype(str) + "-" + df["atom_1"].astype(str)

    df["q0"] = df["q0"].astype(np.float64)
    df["q1"] = df["q1"].astype(np.float64)
    df["dq"] = (df["q0"] - df["q1"]).astype(np.float64)
    df["q_abs_sum"] = (df["q0"].abs() + df["q1"].abs()).astype(np.float64)

    for c in ["sh_trace0", "sh_frob0", "sh_trace1", "sh_frob1"]:
        df[c] = df[c].astype(np.float64)
    df["d_sh_trace"] = (df["sh_trace0"] - df["sh_trace1"]).astype(np.float64)
    df["d_sh_frob"] = (df["sh_frob0"] - df["sh_frob1"]).astype(np.float64)

    df["dipole_x"] = df["dipole_x"].astype(np.float64)
    df["dipole_y"] = df["dipole_y"].astype(np.float64)
    df["dipole_z"] = df["dipole_z"].astype(np.float64)
    df["dipole_norm"] = df["dipole_norm"].astype(np.float64)
    df["potential_energy"] = df["potential_energy"].astype(np.float64)

    return df


train_f = add_features(train.copy())
test_f = add_features(test.copy())


def _bin_series(s: pd.Series, width: float, max_abs_bin: int) -> pd.Series:
    x = s.astype(np.float64)
    b = np.floor(x / width)
    b = (
        pd.Series(b)
        .clip(lower=-max_abs_bin, upper=max_abs_bin)
        .astype("Int64")
        .fillna(0)
        .astype(np.int32)
    )
    return b


N_DIST_BINS = 50


def _add_type_quantile_dist_bin(
    train_df: pd.DataFrame, test_df: pd.DataFrame, n_bins: int = 50
):
    train_bins = np.empty(len(train_df), dtype=np.int16)
    test_bins = np.empty(len(test_df), dtype=np.int16)

    for t in train_df["type"].unique():
        tr_mask = (train_df["type"] == t).values
        te_mask = (test_df["type"] == t).values

        tr_dist = train_df.loc[tr_mask, "dist"].astype(np.float64).values
        te_dist = test_df.loc[te_mask, "dist"].astype(np.float64).values

        tr_dist = np.nan_to_num(
            tr_dist, nan=np.nanmedian(tr_dist) if len(tr_dist) else 0.0
        )
        te_dist = np.nan_to_num(
            te_dist, nan=np.nanmedian(tr_dist) if len(tr_dist) else 0.0
        )

        if len(tr_dist) < max(1000, n_bins * 10):
            width = 0.1
            train_bins[tr_mask] = np.clip(np.floor(tr_dist / width), 0, 300).astype(
                np.int16
            )
            test_bins[te_mask] = np.clip(np.floor(te_dist / width), 0, 300).astype(
                np.int16
            )
            continue

        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(tr_dist, qs)
        edges = np.unique(edges)
        if len(edges) <= 2:
            train_bins[tr_mask] = 0
            test_bins[te_mask] = 0
            continue

        tr_b = np.digitize(tr_dist, edges[1:-1], right=True)
        te_b = np.digitize(te_dist, edges[1:-1], right=True)

        train_bins[tr_mask] = tr_b.astype(np.int16)
        test_bins[te_mask] = te_b.astype(np.int16)

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["dist_bin_q"] = train_bins.astype(np.int16)
    test_df["dist_bin_q"] = test_bins.astype(np.int16)
    return train_df, test_df


train_f, test_f = _add_type_quantile_dist_bin(train_f, test_f, n_bins=N_DIST_BINS)

train_f["mol_n_atoms_bin"] = (
    (train_f["mol_n_atoms"] // 5).astype("Int64").fillna(-1).astype(np.int32)
)
test_f["mol_n_atoms_bin"] = (
    (test_f["mol_n_atoms"] // 5).astype("Int64").fillna(-1).astype(np.int32)
)

train_f["dq_bin"] = _bin_series(
    train_f["dq"].fillna(0.0), width=0.02, max_abs_bin=250
)  # ~[-5,5]
test_f["dq_bin"] = _bin_series(test_f["dq"].fillna(0.0), width=0.02, max_abs_bin=250)

train_f["d_sh_trace_bin"] = _bin_series(
    train_f["d_sh_trace"].fillna(0.0), width=1.0, max_abs_bin=300
)
test_f["d_sh_trace_bin"] = _bin_series(
    test_f["d_sh_trace"].fillna(0.0), width=1.0, max_abs_bin=300
)

train_f["dipole_norm_bin"] = _bin_series(
    train_f["dipole_norm"].fillna(0.0), width=0.2, max_abs_bin=200
)
test_f["dipole_norm_bin"] = _bin_series(
    test_f["dipole_norm"].fillna(0.0), width=0.2, max_abs_bin=200
)

train_f["energy_bin"] = _bin_series(
    train_f["potential_energy"].fillna(0.0), width=50.0, max_abs_bin=300
)
test_f["energy_bin"] = _bin_series(
    test_f["potential_energy"].fillna(0.0), width=50.0, max_abs_bin=300
)

global_mean = float(train_f["scalar_coupling_constant"].mean())

grp_cols = [
    "type",
    "dist_bin_q",
    "pair",
    "mol_n_atoms_bin",
    "dq_bin",
    "d_sh_trace_bin",
    "dipole_norm_bin",
    "energy_bin",
]




## === cell 3
def _make_stratified_molecule_folds(
    train_df: pd.DataFrame, n_folds: int, seed: int = 2020
):
    mol_type_counts = (
        train_df.groupby(["molecule_name", "type"]).size().unstack(fill_value=0)
    )
    mols = mol_type_counts.index.values
    X = mol_type_counts.values.astype(np.int64)

    rng = np.random.RandomState(seed)
    order = np.argsort(-X.sum(axis=1))  # place largest molecules first
    order = order[rng.permutation(len(order))]  # deterministic shuffle among ties

    fold_sums = np.zeros((n_folds, X.shape[1]), dtype=np.int64)
    folds = [[] for _ in range(n_folds)]

    for idx in order:
        x = X[idx]
        best_fold = None
        best_score = None
        for f in range(n_folds):
            new = fold_sums.copy()
            new[f] += x
            target = new.sum(axis=0) / n_folds
            score = ((new - target) ** 2).sum()
            if best_score is None or score < best_score:
                best_score = score
                best_fold = f
        fold_sums[best_fold] += x
        folds[best_fold].append(mols[idx])

    return [np.array(f, dtype=object) for f in folds]


def _make_oof_pred(
    train_df: pd.DataFrame, k_grp: float, min_grp_count: float, n_folds: int = 5
) -> np.ndarray:
    folds = _make_stratified_molecule_folds(train_df, n_folds=n_folds, seed=2020)

    oof = np.empty(len(train_df), dtype=np.float64)

    for fold_mols in folds:
        is_val = train_df["molecule_name"].isin(fold_mols).values
        tr = train_df.loc[~is_val]
        va = train_df.loc[is_val]

        type_stats_tr = (
            tr.groupby("type")["scalar_coupling_constant"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "type_mean", "count": "type_count"})
        )

        grp_stats_tr = (
            tr.groupby(grp_cols)["scalar_coupling_constant"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "grp_mean", "count": "grp_count"})
            .reset_index()
        )

        va_stats = va.merge(type_stats_tr, how="left", left_on="type", right_index=True)
        va_stats = va_stats.merge(grp_stats_tr, how="left", on=grp_cols)

        grp_count = va_stats["grp_count"].fillna(0.0).astype(np.float64)
        grp_mean = (
            va_stats["grp_mean"]
            .fillna(va_stats["type_mean"])
            .fillna(global_mean)
            .astype(np.float64)
        )
        type_mean = va_stats["type_mean"].fillna(global_mean).astype(np.float64)

        w_grp = grp_count / (grp_count + k_grp)
        w_grp = np.where(grp_count >= min_grp_count, w_grp, 0.0).astype(np.float64)

        pred = (w_grp * grp_mean + (1.0 - w_grp) * type_mean).astype(np.float64)
        oof[is_val] = pred.values

    return oof


def _mae_by_type_log(y_true: np.ndarray, y_pred: np.ndarray, types: pd.Series) -> float:
    df = pd.DataFrame({"y": y_true, "p": y_pred, "type": types.values})
    mae_by_type = df.groupby("type", sort=False).apply(
        lambda g: np.mean(np.abs(g["y"] - g["p"]))
    )
    return float(np.mean(np.log(mae_by_type.clip(lower=1e-9).values)))


y = train_f["scalar_coupling_constant"].astype(np.float64).values
types = train_f["type"]

candidates_k = [50.0, 100.0, 200.0, 400.0, 800.0]
candidates_min = [1.0, 3.0, 5.0, 10.0, 20.0]

best = None
best_params = None

for k_grp in candidates_k:
    for min_cnt in candidates_min:
        oof_pred = _make_oof_pred(
            train_f, k_grp=k_grp, min_grp_count=min_cnt, n_folds=5
        )
        score = _mae_by_type_log(y, oof_pred, types)
        if (best is None) or (score < best):
            best = score
            best_params = (k_grp, min_cnt)

k_grp, min_grp_count = best_params
print(
    "Selected smoothing params from stratified molecule-OOF: k_grp =",
    k_grp,
    "min_grp_count =",
    min_grp_count,
    "OOF logMAE =",
    best,
)

type_stats = (
    train_f.groupby("type")["scalar_coupling_constant"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "type_mean", "count": "type_count"})
)

grp_stats = (
    train_f.groupby(grp_cols)["scalar_coupling_constant"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "grp_mean", "count": "grp_count"})
    .reset_index()
)

test_stats = test_f.merge(type_stats, how="left", left_on="type", right_index=True)
test_stats = test_stats.merge(grp_stats, how="left", on=grp_cols)

pred1 = test_stats["type_mean"].fillna(global_mean).astype(np.float64)

grp_count = test_stats["grp_count"].fillna(0.0).astype(np.float64)
grp_mean = (
    test_stats["grp_mean"]
    .fillna(test_stats["type_mean"])
    .fillna(global_mean)
    .astype(np.float64)
)
type_mean = test_stats["type_mean"].fillna(global_mean).astype(np.float64)

w_grp = grp_count / (grp_count + float(k_grp))
w_grp = np.where(grp_count >= float(min_grp_count), w_grp, 0.0).astype(np.float64)

pred2 = (w_grp * grp_mean + (1.0 - w_grp) * type_mean).astype(np.float64)

sub1 = pd.DataFrame(
    {"id": test_stats["id"].values, "scalar_coupling_constant": pred1.values}
)
sub2 = pd.DataFrame(
    {"id": test_stats["id"].values, "scalar_coupling_constant": pred2.values}
)

print("pred1 describe:\n", sub1["scalar_coupling_constant"].describe())
print("pred2 describe:\n", sub2["scalar_coupling_constant"].describe())



## === cell 4
(sub1["scalar_coupling_constant"] - sub2["scalar_coupling_constant"]).abs().mean()



## === cell 5
sub1["scalar_coupling_constant"] = (
    0.35 * sub1["scalar_coupling_constant"] + 0.65 * sub2["scalar_coupling_constant"]
)

sub1 = sub1[["id", "scalar_coupling_constant"]].sort_values("id").reset_index(drop=True)

sub1["scalar_coupling_constant"] = (
    sub1["scalar_coupling_constant"]
    .astype(np.float64)
    .replace([np.inf, -np.inf], np.nan)
    .fillna(global_mean)
)

sample_path = os.path.join(DATA_ROOT, "sample_submission.csv")
sample = pd.read_csv(sample_path, usecols=["id"])
if len(sample) == len(sub1):
    if not np.array_equal(sample["id"].values, sub1["id"].values):
        sub1 = sample.merge(sub1, on="id", how="left", validate="1:1")
        sub1["scalar_coupling_constant"] = (
            sub1["scalar_coupling_constant"]
            .astype(np.float64)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(global_mean)
        )

out_path = "weighted-avg-blend-lgb-keras-1.csv"
sub1.to_csv(out_path, index=False)

print("Wrote:", out_path)
print("submission shape:", sub1.shape)
print("id monotonic:", sub1["id"].is_monotonic_increasing)
print(sub1["scalar_coupling_constant"].describe())
