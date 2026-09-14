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

-1.3547912745601447

# 6. Current score

1.28381

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'The current notebook fails because it tries to read four external submission files from `../input/...` folders that don’t exist in your environment, so `sub1..sub4` are never created and everything downstream crashes. To make it run end-to-end and generate a valid `submission.csv`, I replace that broken ensemble-with-missing-files with a minimal, self-contained baseline that uses only the provided competition data. Specifically, it compute the mean `scalar_coupling_constant` per `type` from `train.csv` and use that as predictions for `test.csv` (a standard “type-mean” baseline), then write `id,scalar_coupling_constant` to `submission.csv`. This preserves the overall “simple blending/aggregation” spirit while ensuring correct I/O, correct submission format, and a nontrivial score (better than all-zeros) without introducing new modeling complexity.'
- What this solution (achieved 1.21344) has done: 'You’re hitting NaNs in the design matrix because some atom-coordinate merges fail (missing structure rows or unexpected indices), which propagates into `dist/dist2` and makes `Ridge` crash; fixing that unblocks everything downstream. I add a minimal, score-neutral imputation: fill missing `dist/dist2` with the per-`type` median distance computed from train (fallback to global median), and also guard against any remaining NaNs in dummy columns. I also make `cell 3`/`cell 4` robust by ensuring `sub` is always defined once predictions are created and by importing matplotlib for plotting. Core approach (per-type Ridge on distance + atom one-hot) is preserved.'
- What this solution (achieved 1.28381) has done: 'Your current Ridge-per-type baseline is far from the target (lower is better), so we should improve it modestly without changing the overall approach. The biggest safe gain here is to add a few physically meaningful, cheap-to-compute geometry features (absolute deltas and inverse-distance terms) while keeping the same per-type Ridge training loop and loss semantics. I also standardize features within each coupling type (using train statistics only) to make the single global `alpha` behave more consistently across types, which typically reduces MAE without changing the model family. All changes keep the same I/O paths and still produce a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

BASE_INPUT = "/kaggle/data/champs-scalar-coupling"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "/kaggle/input/champs-scalar-coupling"

assert os.path.exists(BASE_INPUT), f"Could not find dataset directory at {BASE_INPUT}"

TRAIN_PATH = os.path.join(BASE_INPUT, "train.csv")
TEST_PATH = os.path.join(BASE_INPUT, "test.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")
STRUCTURES_PATH = os.path.join(BASE_INPUT, "structures.csv")



## === cell 1
import numpy as np  # linear algebra
import pandas as pd  # data processing
import matplotlib.pyplot as plt

print("BASE_INPUT =", BASE_INPUT)
print("Files in BASE_INPUT (first 20):", sorted(os.listdir(BASE_INPUT))[:20])



## === cell 2
from sklearn.linear_model import Ridge

train = pd.read_csv(
    TRAIN_PATH,
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
    TEST_PATH,
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH, usecols=["id", "scalar_coupling_constant"])

structures = pd.read_csv(
    STRUCTURES_PATH,
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
)

s0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "x_0",
        "y": "y_0",
        "z": "z_0",
    }
)
s1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "x": "x_1",
        "y": "y_1",
        "z": "z_1",
    }
)


def add_pair_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
    df = df.merge(s1, on=["molecule_name", "atom_index_1"], how="left")

    dx = (df["x_0"].values - df["x_1"].values).astype(np.float32)
    dy = (df["y_0"].values - df["y_1"].values).astype(np.float32)
    dz = (df["z_0"].values - df["z_1"].values).astype(np.float32)

    df["dx"] = dx
    df["dy"] = dy
    df["dz"] = dz
    df["adx"] = np.abs(dx).astype(np.float32)
    df["ady"] = np.abs(dy).astype(np.float32)
    df["adz"] = np.abs(dz).astype(np.float32)

    dist = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)
    df["dist"] = dist
    df["dist2"] = (dist * dist).astype(np.float32)

    eps = np.float32(1e-3)
    inv = (1.0 / np.maximum(dist, eps)).astype(np.float32)
    df["inv_dist"] = inv
    df["inv_dist2"] = (inv * inv).astype(np.float32)

    df.drop(columns=["x_0", "y_0", "z_0", "x_1", "y_1", "z_1"], inplace=True)
    return df


train_f = add_pair_features(train)
test_f = add_pair_features(test)

global_dist_median = float(np.nanmedian(train_f["dist"].values))
type_dist_median = train_f.groupby("type")["dist"].median()


def impute_dist_features(df: pd.DataFrame) -> pd.DataFrame:
    dist = df["dist"]
    if dist.isna().any():
        fill_vals = df["type"].map(type_dist_median).astype(np.float32)
        fill_vals = fill_vals.fillna(global_dist_median).astype(np.float32)
        df.loc[dist.isna(), "dist"] = fill_vals.loc[dist.isna()].values
        df.loc[df["dist2"].isna(), "dist2"] = (
            df.loc[df["dist2"].isna(), "dist"].astype(np.float32) ** 2
        ).values

    if df["dist2"].isna().any():
        df.loc[df["dist2"].isna(), "dist2"] = (
            df.loc[df["dist2"].isna(), "dist"].astype(np.float32) ** 2
        ).values

    eps = np.float32(1e-3)
    if df["inv_dist"].isna().any() or df["inv_dist2"].isna().any():
        inv = (1.0 / np.maximum(df["dist"].astype(np.float32).values, eps)).astype(
            np.float32
        )
        df["inv_dist"] = inv
        df["inv_dist2"] = (inv * inv).astype(np.float32)

    for c in ["dx", "dy", "dz", "adx", "ady", "adz"]:
        if c in df.columns and df[c].isna().any():
            df[c] = df[c].fillna(np.float32(0.0))

    return df


train_f = impute_dist_features(train_f)
test_f = impute_dist_features(test_f)

type_mean = train_f.groupby("type")["scalar_coupling_constant"].mean()
global_mean = float(train_f["scalar_coupling_constant"].mean())

cats = pd.concat(
    [
        train_f[["atom_0", "atom_1", "type"]],
        test_f[["atom_0", "atom_1", "type"]],
    ],
    axis=0,
    ignore_index=True,
)

atom0_levels = pd.Index(cats["atom_0"].fillna("UNK").unique())
atom1_levels = pd.Index(cats["atom_1"].fillna("UNK").unique())


def make_design(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame(index=df.index)

    num_cols = ["dist", "dist2", "inv_dist", "inv_dist2", "adx", "ady", "adz"]
    for c in num_cols:
        d[c] = df[c].astype(np.float32)

    a0 = df["atom_0"].fillna("UNK")
    a1 = df["atom_1"].fillna("UNK")

    a0_oh = pd.get_dummies(a0, prefix="a0").reindex(
        columns=[f"a0_{x}" for x in atom0_levels], fill_value=0
    )
    a1_oh = pd.get_dummies(a1, prefix="a1").reindex(
        columns=[f"a1_{x}" for x in atom1_levels], fill_value=0
    )

    d = pd.concat([d, a0_oh, a1_oh], axis=1)

    if d.isna().any().any():
        d = d.fillna(0.0)

    return d


X_train_all = make_design(train_f)
X_test_all = make_design(test_f)

pred_test = np.empty(len(test_f), dtype=np.float64)

alpha = 1.0

numeric_cols = ["dist", "dist2", "inv_dist", "inv_dist2", "adx", "ady", "adz"]
all_cols = X_train_all.columns

for t, idx in test_f.groupby("type").indices.items():
    tr_mask = train_f["type"].values == t
    tr_idx = train_f.index[tr_mask]
    if len(tr_idx) < 50:
        pred_test[idx] = float(type_mean.get(t, global_mean))
        continue

    Xtr = X_train_all.loc[tr_idx].copy()
    Xte = X_test_all.loc[idx].copy()

    mu = Xtr[numeric_cols].mean(axis=0)
    sig = Xtr[numeric_cols].std(axis=0).replace(0.0, 1.0)

    Xtr.loc[:, numeric_cols] = (Xtr[numeric_cols] - mu) / sig
    Xte.loc[:, numeric_cols] = (Xte[numeric_cols] - mu) / sig

    model = Ridge(alpha=alpha, random_state=0)
    model.fit(Xtr, train_f.loc[tr_idx, "scalar_coupling_constant"].values)
    pred_test[idx] = model.predict(Xte)

sub = pd.DataFrame({"id": test_f["id"].values, "scalar_coupling_constant": pred_test})

if sample_sub.shape[0] == sub.shape[0] and sample_sub["id"].is_unique:
    sub = sample_sub[["id"]].merge(sub, on="id", how="left")
    sub["scalar_coupling_constant"] = sub["scalar_coupling_constant"].fillna(
        global_mean
    )

print(sub.head())
print(sub["scalar_coupling_constant"].describe())
print("Any NaN predictions?", sub["scalar_coupling_constant"].isna().any())



## === cell 3
SUB_PATH = "submission.csv"
sub.to_csv(SUB_PATH, index=False)
print("Wrote:", SUB_PATH, "rows:", len(sub), "cols:", list(sub.columns))



## === cell 4
ax = sub["scalar_coupling_constant"].plot(
    kind="hist", bins=100, title="Predicted scalar_coupling_constant distribution"
)
ax.set_xlabel("scalar_coupling_constant")
plt.show()
