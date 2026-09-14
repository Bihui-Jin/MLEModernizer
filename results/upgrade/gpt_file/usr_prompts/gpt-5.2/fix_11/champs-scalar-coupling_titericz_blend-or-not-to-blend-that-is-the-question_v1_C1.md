# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

Not yielded

# 7. Whether higher score is better

Lower is better

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

for df in (train, test):
    df["molecule_name"] = df["molecule_name"].astype("category")
    df["type"] = df["type"].astype("category")

structures = pd.read_csv(
    structures_path, usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"]
)
structures = structures.drop_duplicates(
    subset=["molecule_name", "atom_index"], keep="first"
)
structures["molecule_name"] = structures["molecule_name"].astype("category")
structures["atom"] = structures["atom"].astype("category")

mulliken = pd.read_csv(
    mulliken_path, usecols=["molecule_name", "atom_index", "mulliken_charge"]
)
mulliken = mulliken.drop_duplicates(
    subset=["molecule_name", "atom_index"], keep="first"
)
mulliken["molecule_name"] = mulliken["molecule_name"].astype("category")

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
shield["molecule_name"] = shield["molecule_name"].astype("category")

sh_arr = shield[["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]].to_numpy(
    dtype=np.float64, copy=False
)
shield_trace = (
    shield["XX"].to_numpy(dtype=np.float64, copy=False)
    + shield["YY"].to_numpy(dtype=np.float64, copy=False)
    + shield["ZZ"].to_numpy(dtype=np.float64, copy=False)
)
shield_frob = np.sqrt((sh_arr * sh_arr).sum(axis=1))
shield = shield[["molecule_name", "atom_index"]].copy()
shield["shield_trace"] = shield_trace.astype(np.float64, copy=False)
shield["shield_frob"] = shield_frob.astype(np.float64, copy=False)

dipole = pd.read_csv(dipole_path, usecols=["molecule_name", "X", "Y", "Z"])
dipole["molecule_name"] = dipole["molecule_name"].astype("category")
d_arr = dipole[["X", "Y", "Z"]].to_numpy(dtype=np.float64, copy=False)
dipole_norm = np.sqrt((d_arr * d_arr).sum(axis=1))
dipole = dipole.rename(columns={"X": "dipole_x", "Y": "dipole_y", "Z": "dipole_z"})
dipole["dipole_norm"] = dipole_norm.astype(np.float64, copy=False)
dipole = dipole[["molecule_name", "dipole_x", "dipole_y", "dipole_z", "dipole_norm"]]

energy = pd.read_csv(
    energy_path, usecols=["molecule_name", "potential_energy"]
).drop_duplicates(subset=["molecule_name"], keep="first")
energy["molecule_name"] = energy["molecule_name"].astype("category")

mol_agg = (
    structures.groupby("molecule_name", sort=False, observed=True)
    .agg(
        mol_x_mean=("x", "mean"),
        mol_y_mean=("y", "mean"),
        mol_z_mean=("z", "mean"),
        mol_n_atoms=("atom_index", "count"),
    )
    .reset_index()
)

_ATOMIC_NUM = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
}

structures_idx = structures.set_index(["molecule_name", "atom_index"], drop=False)
mulliken_idx = mulliken.set_index(["molecule_name", "atom_index"], drop=False)
shield_idx = shield.set_index(["molecule_name", "atom_index"], drop=False)
dipole_idx = dipole.set_index("molecule_name", drop=False)
energy_idx = energy.set_index("molecule_name", drop=False)
mol_agg_idx = mol_agg.set_index("molecule_name", drop=False)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    k0 = pd.MultiIndex.from_arrays(
        [df["molecule_name"], df["atom_index_0"]], names=["molecule_name", "atom_index"]
    )
    k1 = pd.MultiIndex.from_arrays(
        [df["molecule_name"], df["atom_index_1"]], names=["molecule_name", "atom_index"]
    )

    a0 = structures_idx.loc[:, ["atom", "x", "y", "z"]].reindex(k0)
    a0.columns = ["atom_0", "x0", "y0", "z0"]
    df = df.join(a0.reset_index(drop=True))

    a1 = structures_idx.loc[:, ["atom", "x", "y", "z"]].reindex(k1)
    a1.columns = ["atom_1", "x1", "y1", "z1"]
    df = df.join(a1.reset_index(drop=True))

    df = df.join(
        mol_agg_idx.loc[:, ["mol_x_mean", "mol_y_mean", "mol_z_mean", "mol_n_atoms"]]
        .reindex(df["molecule_name"])
        .reset_index(drop=True)
    )
    df = df.join(
        dipole_idx.loc[:, ["dipole_x", "dipole_y", "dipole_z", "dipole_norm"]]
        .reindex(df["molecule_name"])
        .reset_index(drop=True)
    )
    df = df.join(
        energy_idx.loc[:, ["potential_energy"]]
        .reindex(df["molecule_name"])
        .reset_index(drop=True)
    )

    q0 = (
        mulliken_idx.loc[:, ["mulliken_charge"]]
        .reindex(k0)
        .rename(columns={"mulliken_charge": "q0"})
    )
    q1 = (
        mulliken_idx.loc[:, ["mulliken_charge"]]
        .reindex(k1)
        .rename(columns={"mulliken_charge": "q1"})
    )
    df = df.join(q0.reset_index(drop=True)).join(q1.reset_index(drop=True))

    s0 = (
        shield_idx.loc[:, ["shield_trace", "shield_frob"]]
        .reindex(k0)
        .rename(columns={"shield_trace": "sh_trace0", "shield_frob": "sh_frob0"})
    )
    s1 = (
        shield_idx.loc[:, ["shield_trace", "shield_frob"]]
        .reindex(k1)
        .rename(columns={"shield_trace": "sh_trace1", "shield_frob": "sh_frob1"})
    )
    df = df.join(s0.reset_index(drop=True)).join(s1.reset_index(drop=True))

    for c in [
        "x0",
        "y0",
        "z0",
        "x1",
        "y1",
        "z1",
        "mol_x_mean",
        "mol_y_mean",
        "mol_z_mean",
    ]:
        df[c] = df[c].astype(np.float64)

    df["x0"] = df["x0"].fillna(df["mol_x_mean"]).fillna(0.0)
    df["y0"] = df["y0"].fillna(df["mol_y_mean"]).fillna(0.0)
    df["z0"] = df["z0"].fillna(df["mol_z_mean"]).fillna(0.0)
    df["x1"] = df["x1"].fillna(df["mol_x_mean"]).fillna(0.0)
    df["y1"] = df["y1"].fillna(df["mol_y_mean"]).fillna(0.0)
    df["z1"] = df["z1"].fillna(df["mol_z_mean"]).fillna(0.0)

    x0 = df["x0"].to_numpy(dtype=np.float64, copy=False)
    y0 = df["y0"].to_numpy(dtype=np.float64, copy=False)
    z0 = df["z0"].to_numpy(dtype=np.float64, copy=False)
    x1 = df["x1"].to_numpy(dtype=np.float64, copy=False)
    y1 = df["y1"].to_numpy(dtype=np.float64, copy=False)
    z1 = df["z1"].to_numpy(dtype=np.float64, copy=False)
    mx = df["mol_x_mean"].to_numpy(dtype=np.float64, copy=False)
    my = df["mol_y_mean"].to_numpy(dtype=np.float64, copy=False)
    mz = df["mol_z_mean"].to_numpy(dtype=np.float64, copy=False)

    dx = x0 - x1
    dy = y0 - y1
    dz = z0 - z1
    df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float64, copy=False)

    cdx0 = x0 - mx
    cdy0 = y0 - my
    cdz0 = z0 - mz
    cdx1 = x1 - mx
    cdy1 = y1 - my
    cdz1 = z1 - mz
    df["r0"] = np.sqrt(cdx0 * cdx0 + cdy0 * cdy0 + cdz0 * cdz0).astype(
        np.float64, copy=False
    )
    df["r1"] = np.sqrt(cdx1 * cdx1 + cdy1 * cdy1 + cdz1 * cdz1).astype(
        np.float64, copy=False
    )

    swap = (df["atom_0"].astype(str) > df["atom_1"].astype(str)).to_numpy()
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
            df.loc[swap, a] = df.loc[swap, b].to_numpy()
            df.loc[swap, b] = tmp.to_numpy()

    df["pair"] = df["atom_0"].astype(str) + "-" + df["atom_1"].astype(str)

    z0s = df["atom_0"].map(_ATOMIC_NUM).fillna(0).astype(np.int16)
    z1s = df["atom_1"].map(_ATOMIC_NUM).fillna(0).astype(np.int16)
    df["pair_z"] = (z0s.astype(np.int32) * 100 + z1s.astype(np.int32)).astype(np.int32)

    df["q0"] = df["q0"].astype(np.float64)
    df["q1"] = df["q1"].astype(np.float64)
    df["dq"] = (df["q0"] - df["q1"]).astype(np.float64)
    df["q_abs_sum"] = (df["q0"].abs() + df["q1"].abs()).astype(np.float64)

    for c in ["sh_trace0", "sh_frob0", "sh_trace1", "sh_frob1"]:
        df[c] = df[c].astype(np.float64)
    df["d_sh_trace"] = (df["sh_trace0"] - df["sh_trace1"]).astype(np.float64)
    df["d_sh_frob"] = (df["sh_frob0"] - df["sh_frob1"]).astype(np.float64)

    for c in ["dipole_x", "dipole_y", "dipole_z", "dipole_norm", "potential_energy"]:
        df[c] = df[c].astype(np.float64)

    return df


train_f = add_features(train)
test_f = add_features(test)


def _bin_series(s: pd.Series, width: float, max_abs_bin: int) -> pd.Series:
    x = s.to_numpy(dtype=np.float64, copy=False)
    b = np.floor(x / width)
    b = np.clip(b, -max_abs_bin, max_abs_bin)
    b = np.nan_to_num(b, nan=0.0).astype(np.int32)
    return pd.Series(b, index=s.index)


def _add_type_quantile_dist_bin(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_bins: int = 50,
    col_name: str = "dist_bin_q",
):
    train_bins = np.empty(len(train_df), dtype=np.int16)
    test_bins = np.empty(len(test_df), dtype=np.int16)

    for t in train_df["type"].cat.categories:
        tr_mask = (train_df["type"] == t).to_numpy()
        te_mask = (test_df["type"] == t).to_numpy()

        tr_dist = train_df.loc[tr_mask, "dist"].to_numpy(dtype=np.float64, copy=False)
        te_dist = test_df.loc[te_mask, "dist"].to_numpy(dtype=np.float64, copy=False)

        tr_med = np.nanmedian(tr_dist) if len(tr_dist) else 0.0
        tr_dist = np.nan_to_num(tr_dist, nan=tr_med)
        te_dist = np.nan_to_num(te_dist, nan=tr_med)

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

        train_bins[tr_mask] = np.digitize(tr_dist, edges[1:-1], right=True).astype(
            np.int16
        )
        test_bins[te_mask] = np.digitize(te_dist, edges[1:-1], right=True).astype(
            np.int16
        )

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[col_name] = train_bins.astype(np.int16, copy=False)
    test_df[col_name] = test_bins.astype(np.int16, copy=False)
    return train_df, test_df


train_f, test_f = _add_type_quantile_dist_bin(
    train_f, test_f, n_bins=50, col_name="dist_bin_q"
)
train_f, test_f = _add_type_quantile_dist_bin(
    train_f, test_f, n_bins=20, col_name="dist_bin_q20"
)
train_f, test_f = _add_type_quantile_dist_bin(
    train_f, test_f, n_bins=10, col_name="dist_bin_q10"
)

train_f["mol_n_atoms_bin"] = (train_f["mol_n_atoms"] // 5).fillna(-1).astype(np.int32)
test_f["mol_n_atoms_bin"] = (test_f["mol_n_atoms"] // 5).fillna(-1).astype(np.int32)

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
    "dist_bin_q20",
    "dist_bin_q10",
    "pair",
    "pair_z",
    "mol_n_atoms_bin",
    "dq_bin",
    "d_sh_trace_bin",
    "dipole_norm_bin",
    "energy_bin",
]




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/163266606.py in <cell line: 0>()
    281 
    282 
--> 283 train_f = add_features(train)
    284 test_f = add_features(test)
    285 

/tmp/ipykernel_11/163266606.py in add_features(df)
    261     df["pair"] = df["atom_0"].astype(str) + "-" + df["atom_1"].astype(str)
    262 
--> 263     z0s = df["atom_0"].map(_ATOMIC_NUM).fillna(0).astype(np.int16)
    264     z1s = df["atom_1"].map(_ATOMIC_NUM).fillna(0).astype(np.int16)
    265     df["pair_z"] = (z0s.astype(np.int32) * 100 + z1s.astype(np.int32)).astype(np.int32)

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in fillna(self, value, method, axis, inplace, limit, downcast)
   7347                     )
   7348 
-> 7349                 new_data = self._mgr.fillna(
   7350                     value=value, limit=limit, inplace=inplace, downcast=downcast
   7351                 )

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/base.py in fillna(self, value, limit, inplace, downcast)
    184             limit = libalgos.validate_limit(None, limit=limit)
    185 
--> 186         return self.apply_with_block(
    187             "fillna",
    188             value=value,

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in apply(self, f, align_keys, **kwargs)
    361                 applied = b.apply(f, **kwargs)
    362             else:
--> 363                 applied = getattr(b, f)(**kwargs)
    364             result_blocks = extend_blocks(applied, result_blocks)
    365 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in fillna(self, value, limit, inplace, downcast, using_cow, already_warned)
   2332                 # 3rd party EA that has not implemented copy keyword yet
   2333                 refs = None
-> 2334                 new_values = self.values.fillna(value=value, method=None, limit=limit)
   2335                 # issue the warning *after* retrying, in case the TypeError
   2336                 #  was caused by an invalid fill_value

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/_mixins.py in fillna(self, value, method, limit, copy)
    374             # We validate the fill_value even if there is nothing to fill
    375             if value is not None:
--> 376                 self._validate_setitem_value(value)
    377 
    378             if not copy:

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/categorical.py in _validate_setitem_value(self, value)
   1587             return self._validate_listlike(value)
   1588         else:
-> 1589             return self._validate_scalar(value)
   1590 
   1591     def _validate_scalar(self, fill_value):

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/categorical.py in _validate_scalar(self, fill_value)
   1612             fill_value = self._unbox_scalar(fill_value)
   1613         else:
-> 1614             raise TypeError(
   1615                 "Cannot setitem on a Categorical with a new "
   1616                 f"category ({fill_value}), set the categories first"

TypeError: Cannot setitem on a Categorical with a new category (0), set the categories first

## === cell 3
def _make_stratified_molecule_folds(
    train_df: pd.DataFrame, n_folds: int, seed: int = 2020
):
    mol_type_counts = (
        train_df.groupby(["molecule_name", "type"], sort=False, observed=True)
        .size()
        .unstack(fill_value=0)
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


def _prepare_oof_cache(train_df: pd.DataFrame, n_folds: int = 5, seed: int = 2020):
    folds = _make_stratified_molecule_folds(train_df, n_folds=n_folds, seed=seed)
    mol_arr = train_df["molecule_name"].to_numpy()
    caches = []
    for fold_mols in folds:
        fold_set = set(fold_mols.tolist())
        is_val = np.fromiter(
            (m in fold_set for m in mol_arr), dtype=bool, count=len(mol_arr)
        )

        tr = train_df.loc[
            ~is_val, ["type", "scalar_coupling_constant"] + grp_cols[1:]
        ].copy()
        va = train_df.loc[is_val, ["type"] + grp_cols[1:]].copy()

        type_stats_tr = (
            tr.groupby("type", sort=False, observed=True)["scalar_coupling_constant"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "type_mean", "count": "type_count"})
        )

        grp_stats_tr = (
            tr.groupby(grp_cols, sort=False, observed=True)["scalar_coupling_constant"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "grp_mean", "count": "grp_count"})
            .reset_index()
        )

        va_stats = va.merge(type_stats_tr, how="left", left_on="type", right_index=True)
        va_stats = va_stats.merge(grp_stats_tr, how="left", on=grp_cols)

        caches.append((is_val, va_stats[["type_mean", "grp_mean", "grp_count"]].copy()))
    return caches


def _make_oof_pred_from_cache(
    oof_cache, k_grp: float, min_grp_count: float
) -> np.ndarray:
    oof = np.empty(
        sum(len(x[0]) for x in [oof_cache[:1]]), dtype=np.float64
    )  # placeholder; resized below
    n = len(oof_cache[0][0])
    oof = np.empty(n, dtype=np.float64)

    for is_val, va_stats_small in oof_cache:
        grp_count = (
            va_stats_small["grp_count"]
            .fillna(0.0)
            .to_numpy(dtype=np.float64, copy=False)
        )
        type_mean = (
            va_stats_small["type_mean"]
            .fillna(global_mean)
            .to_numpy(dtype=np.float64, copy=False)
        )
        grp_mean = va_stats_small["grp_mean"].to_numpy(dtype=np.float64, copy=False)
        grp_mean = np.where(np.isnan(grp_mean), type_mean, grp_mean)
        grp_mean = np.where(np.isnan(grp_mean), global_mean, grp_mean)

        w_grp = grp_count / (grp_count + float(k_grp))
        w_grp = np.where(grp_count >= float(min_grp_count), w_grp, 0.0).astype(
            np.float64, copy=False
        )

        pred = (w_grp * grp_mean + (1.0 - w_grp) * type_mean).astype(
            np.float64, copy=False
        )
        oof[is_val] = pred

    return oof


def _mae_by_type_log(y_true: np.ndarray, y_pred: np.ndarray, types: pd.Series) -> float:
    df = pd.DataFrame({"y": y_true, "p": y_pred, "type": types.values})
    mae_by_type = df.groupby("type", sort=False, observed=True).apply(
        lambda g: np.mean(np.abs(g["y"] - g["p"]))
    )
    return float(np.mean(np.log(mae_by_type.clip(lower=1e-9).values)))


y = train_f["scalar_coupling_constant"].astype(np.float64).to_numpy()
types = train_f["type"]

candidates_k = [25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0]
candidates_min = [1.0, 3.0, 5.0, 10.0, 20.0, 50.0]

oof_cache = _prepare_oof_cache(train_f, n_folds=5, seed=2020)

best = None
best_params = None

for k_grp in candidates_k:
    for min_cnt in candidates_min:
        oof_pred = _make_oof_pred_from_cache(
            oof_cache, k_grp=k_grp, min_grp_count=min_cnt
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
    train_f.groupby("type", sort=False, observed=True)["scalar_coupling_constant"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "type_mean", "count": "type_count"})
)

grp_stats = (
    train_f.groupby(grp_cols, sort=False, observed=True)["scalar_coupling_constant"]
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



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/227582666.py in <cell line: 0>()
    122 
    123 
--> 124 y = train_f["scalar_coupling_constant"].astype(np.float64).to_numpy()
    125 types = train_f["type"]
    126 

NameError: name 'train_f' is not defined

## === cell 4
(sub1["scalar_coupling_constant"] - sub2["scalar_coupling_constant"]).abs().mean()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4050153359.py in <cell line: 0>()
----> 1 (sub1["scalar_coupling_constant"] - sub2["scalar_coupling_constant"]).abs().mean()
      2 

NameError: name 'sub1' is not defined

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

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/656335458.py in <cell line: 0>()
      1 sub1["scalar_coupling_constant"] = (
----> 2     0.35 * sub1["scalar_coupling_constant"] + 0.65 * sub2["scalar_coupling_constant"]
      3 )
      4 
      5 sub1 = sub1[["id", "scalar_coupling_constant"]].sort_values("id").reset_index(drop=True)

NameError: name 'sub1' is not defined
