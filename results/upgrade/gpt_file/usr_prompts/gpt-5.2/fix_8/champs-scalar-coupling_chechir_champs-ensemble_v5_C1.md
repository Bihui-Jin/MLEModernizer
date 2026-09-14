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

-2.058625093818288

# 6. Current score

1.81795

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'I remove the dependency on missing external input folders (`../input/champ-preds` and `../input/1-mpnn`) that cause the `FileNotFoundError`, and instead build a simple, valid baseline submission directly from the provided competition data. To keep the change minimal and score-stable, the script use the per-`type` mean target computed from `train.csv` and apply it to `test.csv`, which is a standard baseline for this competition. I also make the path handling robust to the provided directory layout and ensure the output file is a valid `.csv` with the required columns (`id,scalar_coupling_constant`). This guarantees an end-to-end run and a submission artifact without changing any hidden evaluation semantics.'
- What this solution (achieved 1.18497) has done: 'You’re currently far better than the target (lower is better, and 1.23566 is much lower than -2.0586 is not possible because the metric is log(MAE) and should be negative for good models; this indicates your score is being *worse* than target because 1.23566 > -2.0586). To move toward the target, we should legitimately reduce MAE by making a minimal upgrade over the per-type mean baseline without changing the overall “simple tabular baseline” core approach. The smallest meaningful gain here is to use per-`type` **median** (more robust than mean) and add a single strong physics feature: the inter-atomic distance computed from `structures.csv`, then fit a separate **per-type** linear regression on distance only (very lightweight, fast, and stays within the same simple supervised tabular paradigm). We keep safe fallbacks (median) for any missing merges so the script always produces a valid submission CSV.'
- What this solution (achieved 1.18497) has done: 'Your current score (1.18497, lower-is-better) is still far worse than the target (-2.0586), so we should make a small legitimate improvement without changing the overall “simple per-type tabular regression” approach. The biggest low-risk gain in this competition is to fit the same per-type linear model but on a slightly richer, still physics-minimal feature set: distance plus its inverse and squared terms, and include basic per-atom coordinates so the model can partially capture orientation-related effects (still linear, still per-type, still closed-form least squares). To keep runtime and memory safe, we avoid any huge new joins beyond the existing structures merge and we use a tiny ridge term to stabilize the normal equation solve (prevents numeric blow-ups and should improve generalization). Submission writing and schema stay identical.'
- What this solution (achieved 1.18497) has done: 'Your current score (1.18497, lower-is-better) is still far worse than the target (-2.0586), so we should make a small, safe improvement that keeps the same core “per-type lightweight linear ridge regression on geometry-derived features” approach. The biggest low-risk gain without changing the modeling paradigm is to add atom identity information (the `atom` column from `structures.csv`) for each endpoint as simple one-hot features, because coupling types depend strongly on which elements are involved. This is a minimal feature augmentation: we keep the same per-type closed-form ridge solve, same data sources/paths, and the same prediction + fallback logic. We also keep the output schema identical and still write a valid submission CSV.'
- What this solution (achieved 1.18497) has done: 'Your current score (1.18497, lower-is-better) is still far worse than the target (-2.0586), so we should improve legitimately with the smallest extension of your existing per-type linear ridge regression approach. The most impactful minimal change is to add a few more competition-provided, molecule/atom-level features (mulliken charges, magnetic shielding tensor diagonals, and dipole/potential energy at the molecule level) via simple left-joins, then feed those values into the same closed-form per-type ridge solve. This keeps the same core model/training loop and evaluation semantics while giving the linear model more signal than geometry alone. We keep the same robust fallbacks (per-type median/global median) so submission generation remains guaranteed.'
- What this solution (achieved 1.23126) has done: 'We keep your per-type closed-form ridge regression and feature set unchanged, but fix a key training issue: `dropna()` is currently throwing away most rows due to missing optional feature files/values, so the model trains on a tiny (and biased) subset and generalizes poorly. Instead, we impute missing numeric features using per-type medians computed on the training data (with a global-median fallback) and carry those same fill values into the test set, preserving evaluation semantics and staying purely tabular. We also ensure atom one-hots remain stable by filling missing atom labels to `"UNK"` (mapped to all-zeros) rather than dropping rows. This should legitimately reduce MAE (thus logMAE) and move the score downward toward your negative target without changing the core approach.'
- What this solution (achieved 1.81795) has done: 'We keep your exact per-type closed-form ridge regression and feature set, but fix two small issues that can materially hurt logMAE: (1) training uses both (0,1) and (1,0) pairs while test uses a fixed order, so we canonicalize atom order for both train and test to make the learned mapping consistent; and (2) predictions occasionally produce extreme outliers, so we apply a conservative per-type winsorization (clip to the 0.5%–99.5% train quantiles for that type), which typically reduces MAE without changing the model. Both changes are lightweight, stay within the same modeling approach, and should move the score downward (better) toward your negative target. The script still runs end-to-end and writes a valid `sub_ensemble.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

CANDIDATE_ROOTS = [
    "/kaggle/data/champs-scalar-coupling",
    "/kaggle/input/champs-scalar-coupling",
    "/kaggle/data",
    "/kaggle/input",
]
DATA_ROOT = None
for p in CANDIDATE_ROOTS:
    if os.path.exists(os.path.join(p, "train.csv")) and os.path.exists(
        os.path.join(p, "test.csv")
    ):
        DATA_ROOT = p
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not find train.csv/test.csv in expected locations. "
        f"Tried: {CANDIDATE_ROOTS}"
    )

print("Using DATA_ROOT =", DATA_ROOT)
print(
    "Files:",
    [
        f
        for f in ["train.csv", "test.csv", "sample_submission.csv", "structures.csv"]
        if os.path.exists(os.path.join(DATA_ROOT, f))
    ],
)



## === cell 1
train = pd.read_csv(
    os.path.join(DATA_ROOT, "train.csv"),
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
    os.path.join(DATA_ROOT, "test.csv"),
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
)

structures = pd.read_csv(
    os.path.join(DATA_ROOT, "structures.csv"),
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
)

s0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
    }
)
s1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
    }
)

train = train.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
train = train.merge(s1, on=["molecule_name", "atom_index_1"], how="left")
test = test.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
test = test.merge(s1, on=["molecule_name", "atom_index_1"], how="left")

mulliken_path = os.path.join(DATA_ROOT, "mulliken_charges.csv")
if os.path.exists(mulliken_path):
    mull = pd.read_csv(
        mulliken_path, usecols=["molecule_name", "atom_index", "mulliken_charge"]
    )
    m0 = mull.rename(columns={"atom_index": "atom_index_0", "mulliken_charge": "q0"})
    m1 = mull.rename(columns={"atom_index": "atom_index_1", "mulliken_charge": "q1"})
    train = train.merge(m0, on=["molecule_name", "atom_index_0"], how="left")
    train = train.merge(m1, on=["molecule_name", "atom_index_1"], how="left")
    test = test.merge(m0, on=["molecule_name", "atom_index_0"], how="left")
    test = test.merge(m1, on=["molecule_name", "atom_index_1"], how="left")
else:
    train["q0"] = np.nan
    train["q1"] = np.nan
    test["q0"] = np.nan
    test["q1"] = np.nan

mst_path = os.path.join(DATA_ROOT, "magnetic_shielding_tensors.csv")
if os.path.exists(mst_path):
    mst = pd.read_csv(
        mst_path, usecols=["molecule_name", "atom_index", "XX", "YY", "ZZ"]
    )
    t0 = mst.rename(
        columns={"atom_index": "atom_index_0", "XX": "XX0", "YY": "YY0", "ZZ": "ZZ0"}
    )
    t1 = mst.rename(
        columns={"atom_index": "atom_index_1", "XX": "XX1", "YY": "YY1", "ZZ": "ZZ1"}
    )
    train = train.merge(t0, on=["molecule_name", "atom_index_0"], how="left")
    train = train.merge(t1, on=["molecule_name", "atom_index_1"], how="left")
    test = test.merge(t0, on=["molecule_name", "atom_index_0"], how="left")
    test = test.merge(t1, on=["molecule_name", "atom_index_1"], how="left")
else:
    for c in ["XX0", "YY0", "ZZ0", "XX1", "YY1", "ZZ1"]:
        train[c] = np.nan
        test[c] = np.nan

dip_path = os.path.join(DATA_ROOT, "dipole_moments.csv")
if os.path.exists(dip_path):
    dip = pd.read_csv(dip_path, usecols=["molecule_name", "X", "Y", "Z"]).rename(
        columns={"X": "dipX", "Y": "dipY", "Z": "dipZ"}
    )
    train = train.merge(dip, on="molecule_name", how="left")
    test = test.merge(dip, on="molecule_name", how="left")
else:
    for c in ["dipX", "dipY", "dipZ"]:
        train[c] = np.nan
        test[c] = np.nan

pe_path = os.path.join(DATA_ROOT, "potential_energy.csv")
if os.path.exists(pe_path):
    pe = pd.read_csv(pe_path, usecols=["molecule_name", "potential_energy"]).rename(
        columns={"potential_energy": "potE"}
    )
    train = train.merge(pe, on="molecule_name", how="left")
    test = test.merge(pe, on="molecule_name", how="left")
else:
    train["potE"] = np.nan
    test["potE"] = np.nan


def canonicalize_pair_order(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["atom0"] = out["atom0"].fillna("UNK")
    out["atom1"] = out["atom1"].fillna("UNK")

    swap = (out["atom_index_0"] > out["atom_index_1"]) | (
        (out["atom_index_0"] == out["atom_index_1"]) & (out["atom0"] > out["atom1"])
    )

    cols0 = ["atom_index_0", "atom0", "x0", "y0", "z0", "q0", "XX0", "YY0", "ZZ0"]
    cols1 = ["atom_index_1", "atom1", "x1", "y1", "z1", "q1", "XX1", "YY1", "ZZ1"]
    for c0, c1 in zip(cols0, cols1):
        tmp = out.loc[swap, c0].copy()
        out.loc[swap, c0] = out.loc[swap, c1].values
        out.loc[swap, c1] = tmp.values

    return out


train = canonicalize_pair_order(train)
test = canonicalize_pair_order(test)

for df in (train, test):
    dx = df["x0"] - df["x1"]
    dy = df["y0"] - df["y1"]
    dz = df["z0"] - df["z1"]
    df["dist"] = (dx * dx + dy * dy + dz * dz) ** 0.5

type_median = train.groupby("type", sort=False)["scalar_coupling_constant"].median()
global_median = float(train["scalar_coupling_constant"].median())

ATOM_VOCAB = ["H", "C", "N", "O", "F"]
ATOM_TO_IDX = {a: i for i, a in enumerate(ATOM_VOCAB)}

clip_lo = train.groupby("type", sort=False)["scalar_coupling_constant"].quantile(0.005)
clip_hi = train.groupby("type", sort=False)["scalar_coupling_constant"].quantile(0.995)




## === cell 2
def make_features(df: pd.DataFrame) -> np.ndarray:
    d = df["dist"].to_numpy(dtype=np.float64)
    dx = (df["x0"] - df["x1"]).to_numpy(dtype=np.float64)
    dy = (df["y0"] - df["y1"]).to_numpy(dtype=np.float64)
    dz = (df["z0"] - df["z1"]).to_numpy(dtype=np.float64)

    inv_d = 1.0 / np.clip(d, 1e-6, None)
    d2 = d * d

    a0 = df["atom0"].astype("object").to_numpy()
    a1 = df["atom1"].astype("object").to_numpy()
    oh0 = np.zeros((len(df), len(ATOM_VOCAB)), dtype=np.float64)
    oh1 = np.zeros((len(df), len(ATOM_VOCAB)), dtype=np.float64)
    for i, a in enumerate(a0):
        j = ATOM_TO_IDX.get(a)
        if j is not None:
            oh0[i, j] = 1.0
    for i, a in enumerate(a1):
        j = ATOM_TO_IDX.get(a)
        if j is not None:
            oh1[i, j] = 1.0

    q0 = df["q0"].to_numpy(dtype=np.float64)
    q1 = df["q1"].to_numpy(dtype=np.float64)

    XX0 = df["XX0"].to_numpy(dtype=np.float64)
    YY0 = df["YY0"].to_numpy(dtype=np.float64)
    ZZ0 = df["ZZ0"].to_numpy(dtype=np.float64)
    XX1 = df["XX1"].to_numpy(dtype=np.float64)
    YY1 = df["YY1"].to_numpy(dtype=np.float64)
    ZZ1 = df["ZZ1"].to_numpy(dtype=np.float64)

    dipX = df["dipX"].to_numpy(dtype=np.float64)
    dipY = df["dipY"].to_numpy(dtype=np.float64)
    dipZ = df["dipZ"].to_numpy(dtype=np.float64)
    potE = df["potE"].to_numpy(dtype=np.float64)

    dip_norm = np.sqrt(dipX * dipX + dipY * dipY + dipZ * dipZ)

    X = np.column_stack(
        [
            np.ones_like(d),
            d,
            d2,
            inv_d,
            inv_d * inv_d,
            dx,
            dy,
            dz,
            q0,
            q1,
            XX0,
            YY0,
            ZZ0,
            XX1,
            YY1,
            ZZ1,
            dipX,
            dipY,
            dipZ,
            dip_norm,
            potE,
            oh0,
            oh1,
        ]
    )
    return X


NUMERIC_COLS = [
    "x0",
    "y0",
    "z0",
    "x1",
    "y1",
    "z1",
    "dist",
    "q0",
    "q1",
    "XX0",
    "YY0",
    "ZZ0",
    "XX1",
    "YY1",
    "ZZ1",
    "dipX",
    "dipY",
    "dipZ",
    "potE",
]

for df in (train, test):
    df["atom0"] = df["atom0"].fillna("UNK")
    df["atom1"] = df["atom1"].fillna("UNK")

global_num_median = train[NUMERIC_COLS].median(numeric_only=True)
type_num_median = train.groupby("type", sort=False)[NUMERIC_COLS].median(
    numeric_only=True
)


def fill_numeric_by_type(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in NUMERIC_COLS:
        fill_map = type_num_median[c]
        out[c] = out[c].fillna(out["type"].map(fill_map))
        out[c] = out[c].fillna(global_num_median[c])
    return out


train_filled = fill_numeric_by_type(train)
test_filled = fill_numeric_by_type(test)

models = {}
ridge_alpha = 1e-6  # tiny stabilization, not a model/approach change

train_cols_needed = (
    ["type", "atom0", "atom1"] + NUMERIC_COLS + ["scalar_coupling_constant"]
)
train_model_df = train_filled[train_cols_needed]

for t, g in train_model_df.groupby("type", sort=False):
    y = g["scalar_coupling_constant"].to_numpy(dtype=np.float64)
    if len(g) < 10:
        continue

    X = make_features(g)

    XtX = X.T @ X
    Xty = X.T @ y

    reg = np.eye(XtX.shape[0], dtype=np.float64) * ridge_alpha
    reg[0, 0] = 0.0

    try:
        beta = np.linalg.solve(XtX + reg, Xty)
        models[t] = beta.astype(np.float64)
    except np.linalg.LinAlgError:
        models[t] = np.array([y.mean()] + [0.0] * (X.shape[1] - 1), dtype=np.float64)

pred = pd.Series(index=test.index, dtype="float64")

test_cols_needed = ["type", "atom0", "atom1"] + NUMERIC_COLS
test_feat_df = test_filled[test_cols_needed]

for t, beta in models.items():
    mask = test_filled["type"] == t
    if not mask.any():
        continue
    g = test_feat_df.loc[mask]
    ok = g.notna().all(axis=1)
    if ok.any():
        X = make_features(g.loc[ok])
        pred.loc[g.loc[ok].index] = (X @ beta).astype(np.float64)

pred = (
    pred.fillna(test["type"].map(type_median)).fillna(global_median).astype("float64")
)

lo = test["type"].map(clip_lo)
hi = test["type"].map(clip_hi)
pred = pred.clip(lower=lo, upper=hi)

submission = pd.DataFrame(
    {"id": test["id"].astype("int64"), "scalar_coupling_constant": pred}
)

out_path = "sub_ensemble.csv"
submission.to_csv(out_path, index=False)
print("Wrote submission:", out_path, "shape=", submission.shape)
print(submission.head())



## === cell 3
assert list(submission.columns) == ["id", "scalar_coupling_constant"]
assert submission["scalar_coupling_constant"].isna().sum() == 0
assert submission["id"].isna().sum() == 0
assert submission["id"].is_unique

submission.head(20)



## === cell 4
sample_path = os.path.join(DATA_ROOT, "sample_submission.csv")
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    print("sample_submission.csv shape=", sample.shape)
    print("submission shape=", submission.shape)
    if len(sample) == len(submission):
        print("Same number of rows as sample submission.")
    else:
        print("Warning: Row count differs from sample submission.")
else:
    print("sample_submission.csv not found; skipping comparison.")
