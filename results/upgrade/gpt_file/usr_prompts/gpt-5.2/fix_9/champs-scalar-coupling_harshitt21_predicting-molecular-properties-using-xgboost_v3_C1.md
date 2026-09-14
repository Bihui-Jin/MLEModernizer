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
lightgbm==4.6.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
xgboost==2.0.3

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

0.9998

# 6. Current score

1.99777

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.69855) has done: 'I first make the script reliably run end-to-end and actually yield a Kaggle-submittable file: your current paths don’t match the provided data layout and the output filename isn’t a plain `.csv`. Then, with minimal core-logic changes, I add the `type` and atom element features via label encoding (these are central to the metric being averaged by type, and typically give a large MAE reduction with no architecture/training-loop changes). Finally, I make the train/validation split molecule-grouped so validation reflects the competition’s molecule split and avoids overly optimistic validation that can hide issues.'
- What this solution (achieved 1.88247) has done: 'You’re currently optimizing plain MAE on a pooled validation set, while Kaggle scores the mean of per-`type` log(MAE). To move your score down toward the 0.9998 target without changing the modeling approach, I keep XGBoost and your features, but switch training to a custom objective that minimizes MAE in log-space per coupling `type` by training one model per `type` (same regressor, same features subset). This aligns the optimization with the metric and typically yields a large drop from ~1.7 without needing new feature sources. I also compute a local validation score using the exact competition metric so you can verify the direction before submitting, and I keep the submission schema/paths unchanged.'
- What this solution (achieved 1.89419) has done: 'Your current score (1.88247, lower-is-better) is still far from the target (0.9998), so we need a meaningful but still “core-logic-preserving” improvement. The biggest low-risk gain here is to add a few standard, cheap geometry-derived features (vector deltas and inverse distance) while keeping the same per-type XGBoost training loop and objective. This typically reduces MAE across types without changing the modeling approach, and it stays well within the 600s budget since it’s just extra columns. I also make the “missing predictions” logic robust (avoid treating genuine 0 predictions as missing) to prevent accidental fallback overwriting.'
- What this solution (achieved 1.88607) has done: 'To move your score down toward the 0.9998 target (lower is better) without changing the core per-type XGBoost loop, I add a few very standard, cheap geometry features that are strongly predictive in CHAMPS: absolute deltas, per-axis squared deltas, and normalized direction cosines (dx/dist, etc.). I also add a safe log-distance feature and clip tiny distances to avoid numerical issues; these are minimal feature-only changes and keep the same training approach and objective. Finally, I keep your molecule-grouped split and submission writing intact, just extending the feature list so the model can fit interactions it currently has to learn indirectly.'
- What this solution (achieved 1.96865) has done: 'We keep your per-`type` XGBoost training loop and geometry feature logic intact, but add a few “official” CHAMPS side-table features that are cheap to join and are strongly predictive (Mulliken charges, magnetic shielding tensor diagonals, dipole moments, potential energy). These additions don’t change the modeling approach or loss, but typically reduce MAE substantially across coupling types, moving the score down toward your 0.9998 target. We also add simple charge/tensor pairwise deltas/abs deltas to mirror your existing dx/dy/dz pattern (still just feature engineering), and ensure all joins are left-joins with safe fillna so the pipeline stays robust and produces a valid `submission.csv`.'
- What this solution (achieved 1.70405) has done: 'Your current score (1.96865, lower-is-better) is still far from the target (0.9998), so we need a modest-but-meaningful improvement without changing the per-type XGBoost approach. The biggest issue is that the model is currently missing key “pair environment” signals; with minimal feature-engineering only, we can add standard CHAMPS aggregation features: per-molecule counts by element, and for each endpoint atom, distance-to-k-nearest atoms (and a simple sum of inverse distances). These are cheap to compute by merging structures into the pair table and doing small groupby operations, and they typically reduce MAE substantially across types while preserving the same model/training loop. I also keep all joins as left joins, fill NaNs safely, and keep the submission writing unchanged so you still get a valid `submission.csv`.'
- What this solution (achieved 1.57532) has done: 'You’re still far above the target (lower-is-better), so we need a meaningful but still core-logic-preserving improvement. The simplest high-impact fix is to align training with the metric by training per-`type` on a log1p-transformed target (and inverting with expm1 at prediction time); this keeps the same per-type XGBoost loop/architecture but typically reduces the mean log(MAE_type) substantially. To keep it stable and avoid harming types with negative targets, we shift by a per-type constant before log1p and undo that shift after expm1. I also include `type_le` inside each per-type model (it becomes a constant column, so it won’t change behavior but keeps feature parity), and keep the same submission writing.'
- What this solution (achieved 1.99777) has done: 'Your current gap to the target is large (1.57532 vs 0.9998, lower-is-better), so we need a meaningful but still core-logic-preserving improvement. The biggest low-risk win now is to stop forcing the target to be positive and log1p-transformed (which can distort negative couplings) and instead train each per-`type` model on the raw target while still evaluating with the exact CHAMPS metric; this keeps your per-type XGBoost approach intact and usually reduces per-type MAE substantially. To further reduce MAE without changing the approach, we also add a small set of standard “distance-to-mean” (molecule-centered) geometry features that are cheap and highly predictive. Finally, we keep the same paths and ensure the submission is aligned and fully filled.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

from xgboost import XGBRegressor

DATA_DIR = "/kaggle/data/champs-scalar-coupling"



## === cell 1
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
structures = pd.read_csv(f"{DATA_DIR}/structures.csv")

mulliken = pd.read_csv(f"{DATA_DIR}/mulliken_charges.csv")
mst = pd.read_csv(f"{DATA_DIR}/magnetic_shielding_tensors.csv")
dipole = pd.read_csv(f"{DATA_DIR}/dipole_moments.csv")
potential = pd.read_csv(f"{DATA_DIR}/potential_energy.csv")



## === cell 2
print("Train shape: {}".format(train.shape))
print("Test shape: {}".format(test.shape))



## === cell 3
train.info()



## === cell 4
test.info()



## === cell 5
train.describe()



## === cell 6
train[["molecule_name", "scalar_coupling_constant"]].groupby("molecule_name").mean()[
    :100
]



## === cell 7
train[["type", "scalar_coupling_constant"]].groupby("type").count()



## === cell 8
pass



## === cell 9
mol_atom_counts = structures.pivot_table(
    index="molecule_name",
    columns="atom",
    values="atom_index",
    aggfunc="count",
    fill_value=0,
).reset_index()
mol_atom_counts.columns = [
    "molecule_name" if c == "molecule_name" else f"mol_atom_cnt_{c}"
    for c in mol_atom_counts.columns
]




## === cell 10
def add_knn_features_to_structures(struct_df, k=3):
    struct_df = struct_df.copy()
    struct_df["x"] = struct_df["x"].astype(np.float32)
    struct_df["y"] = struct_df["y"].astype(np.float32)
    struct_df["z"] = struct_df["z"].astype(np.float32)
    struct_df["atom_index"] = struct_df["atom_index"].astype(np.int32)

    out_frames = []
    for mol, g in struct_df.groupby("molecule_name", sort=False):
        coords = g[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=True)
        n = coords.shape[0]
        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.sqrt((diff * diff).sum(axis=2) + 1e-12).astype(np.float32)
        np.fill_diagonal(dist, np.inf)

        kth = min(k, n - 1) if n > 1 else 0
        if kth > 0:
            knn = np.partition(dist, kth=kth, axis=1)[:, :kth]
            knn.sort(axis=1)
            if kth < k:
                pad = np.full((n, k - kth), np.inf, dtype=np.float32)
                knn = np.concatenate([knn, pad], axis=1)
        else:
            knn = np.full((n, k), np.inf, dtype=np.float32)

        inv_sum = np.sum(1.0 / (dist + 1e-6), axis=1).astype(np.float32)

        gf = g[["molecule_name", "atom_index"]].copy()
        for i in range(k):
            gf[f"knn_dist_{i+1}"] = knn[:, i]
        gf["atom_inv_dist_sum"] = inv_sum
        out_frames.append(gf)

    knn_feat = pd.concat(out_frames, axis=0, ignore_index=True)
    for i in range(k):
        knn_feat[f"knn_dist_{i+1}"] = knn_feat[f"knn_dist_{i+1}"].replace([np.inf], 0.0)
    return knn_feat


knn_feat = add_knn_features_to_structures(structures, k=3)



## === cell 11
mol_centroid = (
    structures.groupby("molecule_name")[["x", "y", "z"]]
    .mean()
    .reset_index()
    .rename(columns={"x": "mol_x_mean", "y": "mol_y_mean", "z": "mol_z_mean"})
)
mol_centroid[["mol_x_mean", "mol_y_mean", "mol_z_mean"]] = mol_centroid[
    ["mol_x_mean", "mol_y_mean", "mol_z_mean"]
].astype(np.float32)



## === cell 12
train = pd.merge(
    train,
    structures,
    how="left",
    left_on=["molecule_name", "atom_index_0"],
    right_on=["molecule_name", "atom_index"],
    suffixes=("", "_x"),
)
train = pd.merge(
    train,
    structures,
    how="left",
    left_on=["molecule_name", "atom_index_1"],
    right_on=["molecule_name", "atom_index"],
    suffixes=("_x", "_y"),
)

train = pd.merge(
    train,
    mulliken,
    how="left",
    left_on=["molecule_name", "atom_index_0"],
    right_on=["molecule_name", "atom_index"],
    suffixes=("", "_q0_drop"),
)
train.rename(columns={"mulliken_charge": "mulliken_charge_0"}, inplace=True)
train.drop(
    columns=[c for c in ["atom_index_q0_drop"] if c in train.columns], inplace=True
)

train = pd.merge(
    train,
    mulliken,
    how="left",
    left_on=["molecule_name", "atom_index_1"],
    right_on=["molecule_name", "atom_index"],
    suffixes=("", "_q1_drop"),
)
train.rename(columns={"mulliken_charge": "mulliken_charge_1"}, inplace=True)
train.drop(
    columns=[c for c in ["atom_index_q1_drop"] if c in train.columns], inplace=True
)

mst_diag = mst[["molecule_name", "atom_index", "XX", "YY", "ZZ"]].copy()

train = pd.merge(
    train,
    mst_diag,
    how="left",
    left_on=["molecule_name", "atom_index_0"],
    right_on=["molecule_name", "atom_index"],
    suffixes=("", "_mst0_drop"),
)
train.rename(
    columns={"XX": "mst_XX_0", "YY": "mst_YY_0", "ZZ": "mst_ZZ_0"}, inplace=True
)
train.drop(
    columns=[c for c in ["atom_index_mst0_drop"] if c in train.columns], inplace=True
)

train = pd.merge(
    train,
    mst_diag,
    how="left",
    left_on=["molecule_name", "atom_index_1"],
    right_on=["molecule_name", "atom_index"],
    suffixes=("", "_mst1_drop"),
)
train.rename(
    columns={"XX": "mst_XX_1", "YY": "mst_YY_1", "ZZ": "mst_ZZ_1"}, inplace=True
)
train.drop(
    columns=[c for c in ["atom_index_mst1_drop"] if c in train.columns], inplace=True
)

train = pd.merge(train, dipole, how="left", on="molecule_name")
train.rename(columns={"X": "dipole_X", "Y": "dipole_Y", "Z": "dipole_Z"}, inplace=True)

train = pd.merge(train, potential, how="left", on="molecule_name")

train = pd.merge(train, mol_atom_counts, how="left", on="molecule_name")

train = pd.merge(
    train,
    knn_feat,
    how="left",
    left_on=["molecule_name", "atom_index_0"],
    right_on=["molecule_name", "atom_index"],
    suffixes=("", "_knn0_drop"),
)
train.rename(
    columns={
        "knn_dist_1": "knn_dist_1_0",
        "knn_dist_2": "knn_dist_2_0",
        "knn_dist_3": "knn_dist_3_0",
        "atom_inv_dist_sum": "atom_inv_dist_sum_0",
    },
    inplace=True,
)
train.drop(
    columns=[c for c in ["atom_index_knn0_drop"] if c in train.columns], inplace=True
)

train = pd.merge(
    train,
    knn_feat,
    how="left",
    left_on=["molecule_name", "atom_index_1"],
    right_on=["molecule_name", "atom_index"],
    suffixes=("", "_knn1_drop"),
)
train.rename(
    columns={
        "knn_dist_1": "knn_dist_1_1",
        "knn_dist_2": "knn_dist_2_1",
        "knn_dist_3": "knn_dist_3_1",
        "atom_inv_dist_sum": "atom_inv_dist_sum_1",
    },
    inplace=True,
)
train.drop(
    columns=[c for c in ["atom_index_knn1_drop"] if c in train.columns], inplace=True
)

train = pd.merge(train, mol_centroid, how="left", on="molecule_name")



## === cell 13
test = pd.merge(
    test,
    structures,
    how="left",
    left_on=["molecule_name", "atom_index_0"],
    right_on=["molecule_name", "atom_index"],
    suffixes=("", "_x"),
)
test = pd.merge(
    test,
    structures,
    how="left",
    left_on=["molecule_name", "atom_index_1"],
    right_on=["molecule_name", "atom_index"],
    suffixes=("_x", "_y"),
)

test = pd.merge(
    test,
    mulliken,
    how="left",
    left_on=["molecule_name", "atom_index_0"],
    right_on=["molecule_name", "atom_index"],
    suffixes=("", "_q0_drop"),
)
test.rename(columns={"mulliken_charge": "mulliken_charge_0"}, inplace=True)
test.drop(
    columns=[c for c in ["atom_index_q0_drop"] if c in test.columns], inplace=True
)

test = pd.merge(
    test,
    mulliken,
    how="left",
    left_on=["molecule_name", "atom_index_1"],
    right_on=["molecule_name", "atom_index"],
    suffixes=("", "_q1_drop"),
)
test.rename(columns={"mulliken_charge": "mulliken_charge_1"}, inplace=True)
test.drop(
    columns=[c for c in ["atom_index_q1_drop"] if c in test.columns], inplace=True
)

test = pd.merge(
    test,
    mst_diag,
    how="left",
    left_on=["molecule_name", "atom_index_0"],
    right_on=["molecule_name", "atom_index"],
    suffixes=("", "_mst0_drop"),
)
test.rename(
    columns={"XX": "mst_XX_0", "YY": "mst_YY_0", "ZZ": "mst_ZZ_0"}, inplace=True
)
test.drop(
    columns=[c for c in ["atom_index_mst0_drop"] if c in test.columns], inplace=True
)

test = pd.merge(
    test,
    mst_diag,
    how="left",
    left_on=["molecule_name", "atom_index_1"],
    right_on=["molecule_name", "atom_index"],
    suffixes=("", "_mst1_drop"),
)
test.rename(
    columns={"XX": "mst_XX_1", "YY": "mst_YY_1", "ZZ": "mst_ZZ_1"}, inplace=True
)
test.drop(
    columns=[c for c in ["atom_index_mst1_drop"] if c in test.columns], inplace=True
)

test = pd.merge(test, dipole, how="left", on="molecule_name")
test.rename(columns={"X": "dipole_X", "Y": "dipole_Y", "Z": "dipole_Z"}, inplace=True)

test = pd.merge(test, potential, how="left", on="molecule_name")

test = pd.merge(test, mol_atom_counts, how="left", on="molecule_name")

test = pd.merge(
    test,
    knn_feat,
    how="left",
    left_on=["molecule_name", "atom_index_0"],
    right_on=["molecule_name", "atom_index"],
    suffixes=("", "_knn0_drop"),
)
test.rename(
    columns={
        "knn_dist_1": "knn_dist_1_0",
        "knn_dist_2": "knn_dist_2_0",
        "knn_dist_3": "knn_dist_3_0",
        "atom_inv_dist_sum": "atom_inv_dist_sum_0",
    },
    inplace=True,
)
test.drop(
    columns=[c for c in ["atom_index_knn0_drop"] if c in test.columns], inplace=True
)

test = pd.merge(
    test,
    knn_feat,
    how="left",
    left_on=["molecule_name", "atom_index_1"],
    right_on=["molecule_name", "atom_index"],
    suffixes=("", "_knn1_drop"),
)
test.rename(
    columns={
        "knn_dist_1": "knn_dist_1_1",
        "knn_dist_2": "knn_dist_2_1",
        "knn_dist_3": "knn_dist_3_1",
        "atom_inv_dist_sum": "atom_inv_dist_sum_1",
    },
    inplace=True,
)
test.drop(
    columns=[c for c in ["atom_index_knn1_drop"] if c in test.columns], inplace=True
)

test = pd.merge(test, mol_centroid, how="left", on="molecule_name")



## === cell 14
train.head()



## === cell 15
dx_tr = train["x_y"] - train["x_x"]
dy_tr = train["y_y"] - train["y_x"]
dz_tr = train["z_y"] - train["z_x"]

dx_te = test["x_y"] - test["x_x"]
dy_te = test["y_y"] - test["y_x"]
dz_te = test["z_y"] - test["z_x"]

train["dx"] = dx_tr
train["dy"] = dy_tr
train["dz"] = dz_tr

test["dx"] = dx_te
test["dy"] = dy_te
test["dz"] = dz_te

train["adx"] = np.abs(dx_tr)
train["ady"] = np.abs(dy_tr)
train["adz"] = np.abs(dz_tr)

test["adx"] = np.abs(dx_te)
test["ady"] = np.abs(dy_te)
test["adz"] = np.abs(dz_te)

train["dx2"] = dx_tr * dx_tr
train["dy2"] = dy_tr * dy_tr
train["dz2"] = dz_tr * dz_tr

test["dx2"] = dx_te * dx_te
test["dy2"] = dy_te * dy_te
test["dz2"] = dz_te * dz_te

train["dist2"] = train["dx2"] + train["dy2"] + train["dz2"]
test["dist2"] = test["dx2"] + test["dy2"] + test["dz2"]

train["dist"] = np.sqrt(train["dist2"])
test["dist"] = np.sqrt(test["dist2"])

eps = 1e-9
train["inv_dist"] = 1.0 / (train["dist"] + eps)
test["inv_dist"] = 1.0 / (test["dist"] + eps)

train["log_dist"] = np.log(train["dist"] + 1e-6)
test["log_dist"] = np.log(test["dist"] + 1e-6)

train["dx_over_dist"] = train["dx"] / (train["dist"] + eps)
train["dy_over_dist"] = train["dy"] / (train["dist"] + eps)
train["dz_over_dist"] = train["dz"] / (train["dist"] + eps)

test["dx_over_dist"] = test["dx"] / (test["dist"] + eps)
test["dy_over_dist"] = test["dy"] / (test["dist"] + eps)
test["dz_over_dist"] = test["dz"] / (test["dist"] + eps)

for col0, col1, out_base in [
    ("mulliken_charge_0", "mulliken_charge_1", "dq"),
    ("mst_XX_0", "mst_XX_1", "d_mst_XX"),
    ("mst_YY_0", "mst_YY_1", "d_mst_YY"),
    ("mst_ZZ_0", "mst_ZZ_1", "d_mst_ZZ"),
]:
    train[out_base] = train[col1] - train[col0]
    test[out_base] = test[col1] - test[col0]
    train["a" + out_base] = np.abs(train[out_base])
    test["a" + out_base] = np.abs(test[out_base])

train["dipole_mag"] = np.sqrt(
    train["dipole_X"] ** 2 + train["dipole_Y"] ** 2 + train["dipole_Z"] ** 2
)
test["dipole_mag"] = np.sqrt(
    test["dipole_X"] ** 2 + test["dipole_Y"] ** 2 + test["dipole_Z"] ** 2
)

for i in [1, 2, 3]:
    train[f"d_knn_dist_{i}"] = train[f"knn_dist_{i}_1"] - train[f"knn_dist_{i}_0"]
    test[f"d_knn_dist_{i}"] = test[f"knn_dist_{i}_1"] - test[f"knn_dist_{i}_0"]
    train[f"ad_knn_dist_{i}"] = np.abs(train[f"d_knn_dist_{i}"])
    test[f"ad_knn_dist_{i}"] = np.abs(test[f"ad_knn_dist_{i}"])

train["d_atom_inv_dist_sum"] = (
    train["atom_inv_dist_sum_1"] - train["atom_inv_dist_sum_0"]
)
test["d_atom_inv_dist_sum"] = test["atom_inv_dist_sum_1"] - test["atom_inv_dist_sum_0"]
train["ad_atom_inv_dist_sum"] = np.abs(train["d_atom_inv_dist_sum"])
test["ad_atom_inv_dist_sum"] = np.abs(test["d_atom_inv_dist_sum"])

train["x0_c"] = train["x_x"] - train["mol_x_mean"]
train["y0_c"] = train["y_x"] - train["mol_y_mean"]
train["z0_c"] = train["z_x"] - train["mol_z_mean"]
train["x1_c"] = train["x_y"] - train["mol_x_mean"]
train["y1_c"] = train["y_y"] - train["mol_y_mean"]
train["z1_c"] = train["z_y"] - train["mol_z_mean"]

test["x0_c"] = test["x_x"] - test["mol_x_mean"]
test["y0_c"] = test["y_x"] - test["mol_y_mean"]
test["z0_c"] = test["z_x"] - test["mol_z_mean"]
test["x1_c"] = test["x_y"] - test["mol_x_mean"]
test["y1_c"] = test["y_y"] - test["mol_y_mean"]
test["z1_c"] = test["z_y"] - test["mol_z_mean"]

train["r0_c"] = np.sqrt(train["x0_c"] ** 2 + train["y0_c"] ** 2 + train["z0_c"] ** 2)
train["r1_c"] = np.sqrt(train["x1_c"] ** 2 + train["y1_c"] ** 2 + train["z1_c"] ** 2)
test["r0_c"] = np.sqrt(test["x0_c"] ** 2 + test["y0_c"] ** 2 + test["z0_c"] ** 2)
test["r1_c"] = np.sqrt(test["x1_c"] ** 2 + test["y1_c"] ** 2 + test["z1_c"] ** 2)

train["dr_c"] = train["r1_c"] - train["r0_c"]
train["adr_c"] = np.abs(train["dr_c"])
test["dr_c"] = test["r1_c"] - test["r0_c"]
test["adr_c"] = np.abs(test["dr_c"])



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'ad_knn_dist_1'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2601986247.py in <cell line: 0>()
     74     test[f"d_knn_dist_{i}"] = test[f"knn_dist_{i}_1"] - test[f"knn_dist_{i}_0"]
     75     train[f"ad_knn_dist_{i}"] = np.abs(train[f"d_knn_dist_{i}"])
---> 76     test[f"ad_knn_dist_{i}"] = np.abs(test[f"ad_knn_dist_{i}"])
     77 
     78 train["d_atom_inv_dist_sum"] = (

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'ad_knn_dist_1'

## === cell 16
train.head()



## === cell 17
train.columns



## === cell 18
for c in ["type", "atom_x", "atom_y"]:
    le = LabelEncoder()
    all_vals = pd.concat([train[c].astype(str), test[c].astype(str)], axis=0)
    le.fit(all_vals)
    train[c + "_le"] = le.transform(train[c].astype(str))
    test[c + "_le"] = le.transform(test[c].astype(str))

mol_count_cols = [c for c in mol_atom_counts.columns if c != "molecule_name"]

features = [
    "atom_index_0",
    "atom_index_1",
    "x_x",
    "y_x",
    "z_x",
    "x_y",
    "y_y",
    "z_y",
    "dx",
    "dy",
    "dz",
    "adx",
    "ady",
    "adz",
    "dx2",
    "dy2",
    "dz2",
    "dist",
    "dist2",
    "inv_dist",
    "log_dist",
    "dx_over_dist",
    "dy_over_dist",
    "dz_over_dist",
    "type_le",
    "atom_x_le",
    "atom_y_le",
    "mulliken_charge_0",
    "mulliken_charge_1",
    "mst_XX_0",
    "mst_YY_0",
    "mst_ZZ_0",
    "mst_XX_1",
    "mst_YY_1",
    "mst_ZZ_1",
    "dq",
    "adq",
    "d_mst_XX",
    "ad_mst_XX",
    "d_mst_YY",
    "ad_mst_YY",
    "d_mst_ZZ",
    "ad_mst_ZZ",
    "dipole_X",
    "dipole_Y",
    "dipole_Z",
    "dipole_mag",
    "potential_energy",
    "knn_dist_1_0",
    "knn_dist_2_0",
    "knn_dist_3_0",
    "atom_inv_dist_sum_0",
    "knn_dist_1_1",
    "knn_dist_2_1",
    "knn_dist_3_1",
    "atom_inv_dist_sum_1",
    "d_knn_dist_1",
    "ad_knn_dist_1",
    "d_knn_dist_2",
    "ad_knn_dist_2",
    "d_knn_dist_3",
    "ad_knn_dist_3",
    "d_atom_inv_dist_sum",
    "ad_atom_inv_dist_sum",
    "mol_x_mean",
    "mol_y_mean",
    "mol_z_mean",
    "x0_c",
    "y0_c",
    "z0_c",
    "x1_c",
    "y1_c",
    "z1_c",
    "r0_c",
    "r1_c",
    "dr_c",
    "adr_c",
] + mol_count_cols

train[features] = train[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
test[features] = test[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/876678021.py in <cell line: 0>()
     89 ] + mol_count_cols
     90 
---> 91 train[features] = train[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
     92 test[features] = test[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
     93 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['d_knn_dist_2', 'ad_knn_dist_2', 'd_knn_dist_3', 'ad_knn_dist_3', 'd_atom_inv_dist_sum', 'ad_atom_inv_dist_sum', 'x0_c', 'y0_c', 'z0_c', 'x1_c', 'y1_c', 'z1_c', 'r0_c', 'r1_c', 'dr_c', 'adr_c'] not in index"

## === cell 19
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, va_idx = next(
    gss.split(
        train[features],
        train["scalar_coupling_constant"],
        groups=train["molecule_name"],
    )
)

train_tr = train.loc[tr_idx].copy()
train_va = train.loc[va_idx].copy()




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2083893279.py in <cell line: 0>()
      2 tr_idx, va_idx = next(
      3     gss.split(
----> 4         train[features],
      5         train["scalar_coupling_constant"],
      6         groups=train["molecule_name"],

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['d_knn_dist_2', 'ad_knn_dist_2', 'd_knn_dist_3', 'ad_knn_dist_3', 'd_atom_inv_dist_sum', 'ad_atom_inv_dist_sum', 'x0_c', 'y0_c', 'z0_c', 'x1_c', 'y1_c', 'z1_c', 'r0_c', 'r1_c', 'dr_c', 'adr_c'] not in index"

## === cell 20
def champs_metric(
    df_true_pred, y_true_col="y_true", y_pred_col="y_pred", type_col="type"
):
    maes = df_true_pred.groupby(type_col).apply(
        lambda g: metrics.mean_absolute_error(g[y_true_col], g[y_pred_col])
    )
    return float(np.mean(np.log(maes)))


type_values = sorted(train["type"].unique())
models_by_type = {}
val_preds_all = np.full(len(train_va), np.nan, dtype=np.float32)

base_params = dict(
    n_estimators=700,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="reg:squarederror",
    n_jobs=8,
    random_state=42,
)

for t in type_values:
    tr_t = train_tr[train_tr["type"] == t]
    va_t = train_va[train_va["type"] == t]

    if len(tr_t) == 0 or len(va_t) == 0:
        continue

    feat_t = features  # keep identical feature set within each per-type model

    y_train = tr_t["scalar_coupling_constant"].astype(np.float32).values
    X_train = tr_t[feat_t]
    X_val = va_t[feat_t]

    m = XGBRegressor(**base_params)
    m.fit(X_train, y_train)
    models_by_type[t] = m

    pred = m.predict(X_val).astype(np.float32)
    val_preds_all[train_va["type"].values == t] = pred

missing_mask = np.isnan(val_preds_all)
if missing_mask.any():
    xgb_fallback = XGBRegressor(**base_params)
    xgb_fallback.fit(
        train_tr[features],
        train_tr["scalar_coupling_constant"].astype(np.float32).values,
    )
    val_preds_all[missing_mask] = xgb_fallback.predict(
        train_va.loc[missing_mask, features]
    ).astype(np.float32)
else:
    xgb_fallback = None

val_df = pd.DataFrame(
    {
        "type": train_va["type"].values,
        "y_true": train_va["scalar_coupling_constant"].values,
        "y_pred": val_preds_all,
    }
)

val_mae_pooled = metrics.mean_absolute_error(val_df["y_true"], val_df["y_pred"])
val_metric = champs_metric(val_df)

print("Validation pooled MAE:", val_mae_pooled)
print("Validation pooled log(MAE):", np.log(val_mae_pooled))
print("Validation CHAMPS metric (mean over types of log(MAE_type)):", val_metric)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1802361981.py in <cell line: 0>()
     10 type_values = sorted(train["type"].unique())
     11 models_by_type = {}
---> 12 val_preds_all = np.full(len(train_va), np.nan, dtype=np.float32)
     13 
     14 base_params = dict(

NameError: name 'train_va' is not defined

## === cell 21
test_predictions = np.full(len(test), np.nan, dtype=np.float32)

for t, m in models_by_type.items():
    mask = test["type"].values == t
    if mask.any():
        test_predictions[mask] = m.predict(test.loc[mask, features]).astype(np.float32)

missing = np.isnan(test_predictions)
if missing.any():
    if xgb_fallback is None:
        xgb_fallback = XGBRegressor(**base_params)
        xgb_fallback.fit(
            train_tr[features],
            train_tr["scalar_coupling_constant"].astype(np.float32).values,
        )
    test_predictions[missing] = xgb_fallback.predict(
        test.loc[missing, features]
    ).astype(np.float32)



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1184267361.py in <cell line: 0>()
      8 missing = np.isnan(test_predictions)
      9 if missing.any():
---> 10     if xgb_fallback is None:
     11         xgb_fallback = XGBRegressor(**base_params)
     12         xgb_fallback.fit(

NameError: name 'xgb_fallback' is not defined

## === cell 22
pass



## === cell 23
submission = pd.DataFrame(
    {"id": test["id"].values, "scalar_coupling_constant": test_predictions}
)

submission = submission.sort_values("id").reset_index(drop=True)
submission["scalar_coupling_constant"] = submission["scalar_coupling_constant"].astype(
    np.float32
)
submission["scalar_coupling_constant"] = (
    submission["scalar_coupling_constant"]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)

submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Submission id unique:", submission["id"].is_unique)
print("Submission id count matches test:", len(submission) == len(test))
