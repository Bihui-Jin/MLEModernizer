# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

BASE_PATH = "/kaggle/data/champs-scalar-coupling"
train_path = os.path.join(BASE_PATH, "train.csv")
test_path = os.path.join(BASE_PATH, "test.csv")
sample_path = os.path.join(BASE_PATH, "sample_submission.csv")
structures_path = os.path.join(BASE_PATH, "structures.csv")

print("BASE_PATH exists:", os.path.exists(BASE_PATH))
print("Files in BASE_PATH (first 15):", sorted(os.listdir(BASE_PATH))[:15])

RANDOM_STATE = 42
N_FOLDS = 5

train = pd.read_csv(
    train_path,
    usecols=[
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "scalar_coupling_constant",
    ],
)
test = pd.read_csv(
    test_path,
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
)
sample = pd.read_csv(sample_path, usecols=["id", "scalar_coupling_constant"])

structures = pd.read_csv(
    structures_path, usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"]
)

s0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
    }
)
s1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
    }
)

train = train.merge(
    s0[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
)
train = train.merge(
    s1[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
)

test = test.merge(
    s0[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
)
test = test.merge(
    s1[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
)

train["atom_0"] = train["atom_0"].fillna("UNK")
train["atom_1"] = train["atom_1"].fillna("UNK")
test["atom_0"] = test["atom_0"].fillna("UNK")
test["atom_1"] = test["atom_1"].fillna("UNK")

for df in (train, test):
    dx = df["x0"] - df["x1"]
    dy = df["y0"] - df["y1"]
    dz = df["z0"] - df["z1"]
    df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)

global_mean = float(train["scalar_coupling_constant"].mean())

ATOM2Z = {
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


def add_canonical_pair_cols(df: pd.DataFrame) -> pd.DataFrame:
    a0 = df["atom_0"].astype(str)
    a1 = df["atom_1"].astype(str)
    i0 = df["atom_index_0"].astype(np.int32)
    i1 = df["atom_index_1"].astype(np.int32)

    swap = (a0 > a1) | ((a0 == a1) & (i0 > i1))

    df["a_lo"] = a0.where(~swap, a1)
    df["a_hi"] = a1.where(~swap, a0)
    df["i_lo"] = i0.where(~swap, i1)
    df["i_hi"] = i1.where(~swap, i0)

    z0 = df["atom_0"].map(ATOM2Z).fillna(0).astype(np.int16)
    z1 = df["atom_1"].map(ATOM2Z).fillna(0).astype(np.int16)
    df["z_lo"] = z0.where(~swap, z1)
    df["z_hi"] = z1.where(~swap, z0)
    return df


train = add_canonical_pair_cols(train)
test = add_canonical_pair_cols(test)

mols = train["molecule_name"].drop_duplicates().sort_values().values
rng = np.random.RandomState(RANDOM_STATE)
perm = rng.permutation(len(mols))
fold_id = np.zeros(len(mols), dtype=np.int8)
for i, idx in enumerate(perm):
    fold_id[idx] = i % N_FOLDS
mol2fold = pd.Series(fold_id, index=mols)

train["fold"] = train["molecule_name"].map(mol2fold).astype(np.int8)

print("Train rows:", len(train), "Test rows:", len(test))
print("Unique train molecules:", train["molecule_name"].nunique(), "N_FOLDS:", N_FOLDS)




## === cell 1
def build_tables(df_tr: pd.DataFrame, bin_width: float):
    d = df_tr.copy()
    d["dist_bin"] = np.rint(d["dist"] / bin_width).astype("Int32")

    m_type_pair = (
        d.groupby(["type", "a_lo", "a_hi"], as_index=False)["scalar_coupling_constant"]
        .mean()
        .rename(columns={"scalar_coupling_constant": "m_type_pair"})
    )
    m_type_a0 = (
        d.groupby(["type", "a_lo"], as_index=False)["scalar_coupling_constant"]
        .mean()
        .rename(columns={"scalar_coupling_constant": "m_type_a0"})
    )
    m_type_a1 = (
        d.groupby(["type", "a_hi"], as_index=False)["scalar_coupling_constant"]
        .mean()
        .rename(columns={"scalar_coupling_constant": "m_type_a1"})
    )
    m_type = (
        d.groupby(["type"], as_index=False)["scalar_coupling_constant"]
        .mean()
        .rename(columns={"scalar_coupling_constant": "m_type"})
    )

    m_type_idxpair_db = (
        d.dropna(subset=["dist_bin"])
        .groupby(["type", "i_lo", "i_hi", "dist_bin"])["scalar_coupling_constant"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "m_type_idxpair_db", "count": "c_type_idxpair_db"})
    )

    m_type_pair_db = (
        d.dropna(subset=["dist_bin"])
        .groupby(["type", "a_lo", "a_hi", "dist_bin"])["scalar_coupling_constant"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "m_type_pair_db", "count": "c_type_pair_db"})
    )

    m_type_zpair_db = (
        d.dropna(subset=["dist_bin"])
        .groupby(["type", "z_lo", "z_hi", "dist_bin"])["scalar_coupling_constant"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "m_type_zpair_db", "count": "c_type_zpair_db"})
    )

    return {
        "m_type_pair": m_type_pair,
        "m_type_a0": m_type_a0,
        "m_type_a1": m_type_a1,
        "m_type": m_type,
        "m_type_idxpair_db": m_type_idxpair_db,
        "m_type_pair_db": m_type_pair_db,
        "m_type_zpair_db": m_type_zpair_db,
        "bin_width": bin_width,
    }


def predict_with_tables(
    df_x: pd.DataFrame,
    tables: dict,
    alpha: float,
    alpha_idx: float,
    global_mean_: float,
):
    bin_width = tables["bin_width"]
    x = df_x.copy()
    x["dist_bin"] = np.rint(x["dist"] / bin_width).astype("Int32")

    pred_df = x[
        ["type", "a_lo", "a_hi", "i_lo", "i_hi", "z_lo", "z_hi", "dist_bin"]
    ].copy()
    pred_df = pred_df.merge(
        tables["m_type_pair"], on=["type", "a_lo", "a_hi"], how="left"
    )
    pred_df = pred_df.merge(tables["m_type_a0"], on=["type", "a_lo"], how="left")
    pred_df = pred_df.merge(tables["m_type_a1"], on=["type", "a_hi"], how="left")
    pred_df = pred_df.merge(tables["m_type"], on=["type"], how="left")

    pred_df = pred_df.merge(
        tables["m_type_idxpair_db"], on=["type", "i_lo", "i_hi", "dist_bin"], how="left"
    )
    pred_df = pred_df.merge(
        tables["m_type_pair_db"], on=["type", "a_lo", "a_hi", "dist_bin"], how="left"
    )
    pred_df = pred_df.merge(
        tables["m_type_zpair_db"], on=["type", "z_lo", "z_hi", "dist_bin"], how="left"
    )

    eps = np.float32(1e-6)

    m_type = pred_df["m_type"]
    m_pair = pred_df["m_type_pair"].fillna(m_type)

    num_pair = pred_df["c_type_pair_db"].fillna(0).astype(np.float32)
    mean_pair = pred_df["m_type_pair_db"]
    shrunk_pair = (num_pair * mean_pair + alpha * m_pair) / (num_pair + alpha + eps)

    num_z = pred_df["c_type_zpair_db"].fillna(0).astype(np.float32)
    mean_z = pred_df["m_type_zpair_db"]
    shrunk_z = (num_z * mean_z + alpha * m_pair) / (num_z + alpha + eps)

    parent_for_idx = shrunk_z.fillna(shrunk_pair).fillna(m_pair)

    num_idx = pred_df["c_type_idxpair_db"].fillna(0).astype(np.float32)
    mean_idx = pred_df["m_type_idxpair_db"]
    shrunk_idx = (num_idx * mean_idx + alpha_idx * parent_for_idx) / (
        num_idx + alpha_idx + eps
    )

    pred = (
        shrunk_idx.fillna(shrunk_z)
        .fillna(shrunk_pair)
        .fillna(pred_df["m_type_pair"])
        .fillna(pred_df["m_type_a0"])
        .fillna(pred_df["m_type_a1"])
        .fillna(pred_df["m_type"])
        .fillna(global_mean_)
        .astype(np.float32)
    )
    return pred


def competition_metric_log_mae(
    y_true: np.ndarray, y_pred: np.ndarray, types: np.ndarray
):
    df = pd.DataFrame({"y": y_true, "p": y_pred, "type": types})
    df["ae"] = (df["y"] - df["p"]).abs()
    maes = df.groupby("type", sort=False)["ae"].mean()
    return float(np.mean(np.log(maes.values + 1e-12)))


cand_bin_width = [0.05, 0.1, 0.2]
cand_alpha = [5.0, 10.0, 20.0, 40.0]

cand_alpha_idx_mult = [2.0, 4.0, 8.0]

best = None
for bw in cand_bin_width:
    for a in cand_alpha:
        for m in cand_alpha_idx_mult:
            a_idx = a * m
            oof_pred = np.empty(len(train), dtype=np.float32)
            for f in range(N_FOLDS):
                trn_df = train[train["fold"] != f]
                val_df = train[train["fold"] == f]
                tbl = build_tables(trn_df, bin_width=bw)
                oof_pred[train["fold"].values == f] = predict_with_tables(
                    val_df,
                    tables=tbl,
                    alpha=a,
                    alpha_idx=a_idx,
                    global_mean_=global_mean,
                ).values
            s = competition_metric_log_mae(
                train["scalar_coupling_constant"].values.astype(np.float32),
                oof_pred,
                train["type"].values,
            )
            if (best is None) or (s < best[0]):
                best = (s, bw, a, m)

best_score, BIN_WIDTH, ALPHA, ALPHA_IDX_MULT = best
ALPHA_IDX = ALPHA * ALPHA_IDX_MULT
print(
    "Selected (lower is better): CV OOF logMAE =",
    best_score,
    "BIN_WIDTH =",
    BIN_WIDTH,
    "ALPHA =",
    ALPHA,
    "ALPHA_IDX_MULT =",
    ALPHA_IDX_MULT,
    "ALPHA_IDX =",
    ALPHA_IDX,
)



## === cell 2
oof_pred = np.empty(len(train), dtype=np.float32)
for f in range(N_FOLDS):
    trn_df = train[train["fold"] != f]
    val_df = train[train["fold"] == f]
    tbl = build_tables(trn_df, bin_width=BIN_WIDTH)
    oof_pred[train["fold"].values == f] = predict_with_tables(
        val_df,
        tables=tbl,
        alpha=ALPHA,
        alpha_idx=ALPHA_IDX,
        global_mean_=global_mean,
    ).values

oof_score = competition_metric_log_mae(
    train["scalar_coupling_constant"].values.astype(np.float32),
    oof_pred,
    train["type"].values,
)
print("OOF logMAE estimate (lower is better):", oof_score)

full_tables = build_tables(train, bin_width=BIN_WIDTH)
test_pred = predict_with_tables(
    test,
    tables=full_tables,
    alpha=ALPHA,
    alpha_idx=ALPHA_IDX,
    global_mean_=global_mean,
)

submission = pd.DataFrame(
    {"id": test["id"].values, "scalar_coupling_constant": test_pred.values}
)
submission = sample[["id"]].merge(submission, on="id", how="left")

missing = int(submission["scalar_coupling_constant"].isna().sum())
if missing:
    submission["scalar_coupling_constant"] = submission[
        "scalar_coupling_constant"
    ].fillna(global_mean)

out_path = "stackers_blend.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print("Rows:", len(submission), "Missing preds:", missing)
print("Global mean:", global_mean)
print(
    "Test atom_0/atom_1 UNK counts:",
    int((test["atom_0"] == "UNK").sum()),
    int((test["atom_1"] == "UNK").sum()),
)
print(
    "Missing dist (test):",
    int(test["dist"].isna().sum()),
    "Missing dist_bin (test):",
    int(np.rint(test["dist"] / BIN_WIDTH).astype("Int32").isna().sum()),
)
