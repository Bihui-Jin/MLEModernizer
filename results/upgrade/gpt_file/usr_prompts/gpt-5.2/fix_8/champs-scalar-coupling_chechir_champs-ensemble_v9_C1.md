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

-0.8555725082919478

# 6. Current score

1.21494

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 3.00563) has done: 'I fix the missing input dependency by removing the hard-coded `../input/champ-preds` / `../input/nnet-try*` reads and instead generate a valid baseline prediction from the provided competition data files. To keep the logic as close as possible to your “ensemble of multiple predictors,” I build several simple, legitimate per-`type` and per-atom-pair aggregations from `train.csv` and blend them with the same weighted-sum pattern, producing a stable prediction for every test row. I also make the code robust to missing categories by falling back to global means, and ensure the output file is written as a valid `.csv` with the required columns (`id,scalar_coupling_constant`). This run end-to-end in the given environment and yield a reasonable score (not optimal, but far better than a constant), moving toward the target from “no submission”.'
- What this solution (achieved 1.21224) has done: 'I fix the pandas 2.2 `fillna` error by ensuring we always pass a scalar or a properly indexed `Series` (not a NumPy array) when filling missing predictions. This is a correctness/runtime fix that preserves your existing “multiple aggregations + weighted blend” core logic and should restore the ability to write a valid `.csv` submission end-to-end. I also make the mapping steps explicitly return `Series` aligned to `test.index` to avoid silent misalignment and keep predictions stable. No model/feature approach changes are introduced beyond these alignment/fill fixes.'
- What this solution (achieved 1.21934) has done: 'Your current blend is dominated by very coarse fallbacks (especially `n1`/type mean at 0.6 weight), which tends to over-smooth and hurts the per-type MAE. To move the score down toward the target without changing the core approach (still “multiple aggregations + weighted blend”), I (1) add a minimal, legitimate distance feature from `structures.csv` and learn a simple per-`type` distance→target mean table from train, then map it onto test as another prediction column. Then I (2) slightly rebalance the blend weights to rely less on the global type mean and more on the new distance-based signal and the more specific group means; no new models, no new training loops. This should improve accuracy while keeping runtime reasonable and producing the same valid submission format.'
- What this solution (achieved 1.21494) has done: 'I keep your “multiple aggregations + weighted blend” approach intact and make one minimal, high-impact correction: compute the distance-bin mean map on out-of-fold (molecule-wise) folds to reduce leakage/overfit in `dist_pred`, which should lower the per-type MAE on the hidden test. Then I slightly rebalance the blend to rely a bit more on the stronger pair-specific means (`n2`, `lgb_m`) and a bit less on the coarse `n1`, which tends to over-smooth. All I/O paths and the required submission schema stay the same, and the script still runs end-to-end and writes a valid `.csv`.'
- What this solution (achieved 1.21494) has done: 'Your current score (1.21494, lower-is-better) is still far from the target (-0.8556), so we should make a small, legitimate accuracy improvement without changing the overall “multiple aggregations + weighted blend” approach. The biggest low-risk gain is to strengthen the distance-based signal by (1) making the distance binning a bit finer (0.02 instead of 0.05) and (2) adding one more similarly-constructed geometric predictor (inverse-distance binned mean), then blending it in with a small weight while slightly reducing reliance on the coarse `n1`. These are still the same core logic (groupby-mean maps + weighted sum), just one extra geometry-derived aggregation and a minor reweighting. All paths remain unchanged and the script still writes a valid `sub_ensemble.csv`.'
- What this solution (achieved 1.21494) has done: 'Your current score (1.21494, lower-is-better) is still far from the target (-0.8556), so we should make a small but meaningful accuracy improvement while keeping the same “groupby mean maps + weighted blend” core logic. The most direct gain without changing the modeling approach is to add a more specific geometry-aware mapping: per-`type` + `(atom_0, atom_1)` + `distance_bin` mean, built in a molecule-wise OOF way (same anti-leakage idea you already use), then blend it in with a modest weight. To avoid hurting rare groups, we also add a simple shrinkage toward the existing `dist_pred` using the count per bin (still just aggregations, no new model). Finally, we slightly rebalance weights away from the coarser `n1` toward the new more specific geometry signal, keeping everything deterministic and still producing a valid `sub_ensemble.csv`.'
- What this solution (achieved 1.21494) has done: 'We keep your existing “groupby-mean maps + weighted blend” core logic, but make one targeted improvement that usually lowers MAE on this competition: add a per-`type` + `(atom_0, atom_1)` + `dist_bin` **median** predictor (OOF by molecule, just like your mean-based versions) because the target has heavy tails and the median is often more robust than the mean. We then blend this new robust geometric signal in with a small weight while slightly reducing the weight on the coarse `n1` type-mean, which is typically the least specific component. All paths, output schema, and the rest of your pipeline remain unchanged, and it still writes `sub_ensemble.csv` end-to-end.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

DATA_DIR = "/kaggle/data/champs-scalar-coupling"
if not os.path.exists(DATA_DIR):
    DATA_DIR = "../input/champs-scalar-coupling"

print("Using DATA_DIR:", DATA_DIR)
print("Files sample:", sorted(os.listdir(DATA_DIR))[:10])



## === cell 1
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

TARGET = "scalar_coupling_constant"
print(train.shape, test.shape, sample_sub.shape)
train.head()



## === cell 2
global_mean = float(train[TARGET].mean())
type_mean = train.groupby("type", sort=False)[TARGET].mean()  # Series indexed by type

test["n1"] = test["type"].map(type_mean).astype(float)

structures = pd.read_csv(
    os.path.join(DATA_DIR, "structures.csv"),
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
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

train_aug = train.merge(s0, on=["molecule_name", "atom_index_0"], how="left").merge(
    s1, on=["molecule_name", "atom_index_1"], how="left"
)
test_aug = test.merge(s0, on=["molecule_name", "atom_index_0"], how="left").merge(
    s1, on=["molecule_name", "atom_index_1"], how="left"
)

grp_cols_n2 = ["type", "atom_0", "atom_1"]
n2_map = train_aug.groupby(grp_cols_n2, sort=False)[TARGET].mean()
test["n2"] = pd.Series(
    n2_map.reindex(pd.MultiIndex.from_frame(test_aug[grp_cols_n2])).to_numpy(),
    index=test.index,
    dtype="float64",
)

m0 = train.groupby(["type", "atom_index_0"], sort=False)[TARGET].mean()
m1 = train.groupby(["type", "atom_index_1"], sort=False)[TARGET].mean()
v0 = pd.Series(
    m0.reindex(pd.MultiIndex.from_frame(test[["type", "atom_index_0"]])).to_numpy(),
    index=test.index,
    dtype="float64",
)
v1 = pd.Series(
    m1.reindex(pd.MultiIndex.from_frame(test[["type", "atom_index_1"]])).to_numpy(),
    index=test.index,
    dtype="float64",
)
test["lgb_a"] = 0.5 * v0 + 0.5 * v1

m01 = train.groupby(["type", "atom_index_0", "atom_index_1"], sort=False)[TARGET].mean()
test["lgb_m"] = pd.Series(
    m01.reindex(
        pd.MultiIndex.from_frame(test[["type", "atom_index_0", "atom_index_1"]])
    ).to_numpy(),
    index=test.index,
    dtype="float64",
)

m_atoms = train_aug.groupby(["atom_0", "atom_1"], sort=False)[TARGET].mean()
test["nnet"] = pd.Series(
    m_atoms.reindex(
        pd.MultiIndex.from_frame(test_aug[["atom_0", "atom_1"]])
    ).to_numpy(),
    index=test.index,
    dtype="float64",
)


def add_distance_based_pred_oof(
    train_aug_df: pd.DataFrame,
    test_aug_df: pd.DataFrame,
    bin_width: float = 0.02,
    n_folds: int = 5,
) -> pd.Series:
    dx_tr = train_aug_df["x0"].to_numpy() - train_aug_df["x1"].to_numpy()
    dy_tr = train_aug_df["y0"].to_numpy() - train_aug_df["y1"].to_numpy()
    dz_tr = train_aug_df["z0"].to_numpy() - train_aug_df["z1"].to_numpy()
    dist_tr = np.sqrt(dx_tr * dx_tr + dy_tr * dy_tr + dz_tr * dz_tr)
    tr_bin = np.floor(dist_tr / bin_width).astype(np.int32)

    dx_te = test_aug_df["x0"].to_numpy() - test_aug_df["x1"].to_numpy()
    dy_te = test_aug_df["y0"].to_numpy() - test_aug_df["y1"].to_numpy()
    dz_te = test_aug_df["z0"].to_numpy() - test_aug_df["z1"].to_numpy()
    dist_te = np.sqrt(dx_te * dx_te + dy_te * dy_te + dz_te * dz_te)
    te_bin = np.floor(dist_te / bin_width).astype(np.int32)

    mol = train_aug_df["molecule_name"].astype(str)
    fold_id = (
        pd.util.hash_pandas_object(mol, index=False).astype(np.uint64)
        % np.uint64(n_folds)
    ).astype(np.int16)
    fold_id = pd.Series(fold_id.to_numpy(), index=train_aug_df.index)

    oof_pred = np.empty(train_aug_df.shape[0], dtype=np.float64)
    y = train_aug_df[TARGET].to_numpy(dtype=np.float64)
    types = train_aug_df["type"].to_numpy()

    for f in range(n_folds):
        tr_mask = (fold_id != f).to_numpy()
        val_mask = ~tr_mask

        tmp = pd.DataFrame(
            {"type": types[tr_mask], "dist_bin": tr_bin[tr_mask], TARGET: y[tr_mask]}
        )
        dist_map_f = tmp.groupby(["type", "dist_bin"], sort=False)[TARGET].mean()

        val_key = pd.MultiIndex.from_arrays(
            [types[val_mask], tr_bin[val_mask]], names=["type", "dist_bin"]
        )
        oof_pred[val_mask] = dist_map_f.reindex(val_key).to_numpy(dtype=np.float64)

    tmp_oof = pd.DataFrame(
        {"type": train_aug_df["type"].values, "dist_bin": tr_bin, "oof_pred": oof_pred}
    )
    dist_map_oof = tmp_oof.groupby(["type", "dist_bin"], sort=False)["oof_pred"].mean()

    te_key = pd.MultiIndex.from_arrays(
        [test_aug_df["type"].values, te_bin], names=["type", "dist_bin"]
    )
    out = pd.Series(
        dist_map_oof.reindex(te_key).to_numpy(dtype=np.float64),
        index=test_aug_df.index,
        dtype="float64",
    )
    return out


def add_inv_distance_based_pred_oof(
    train_aug_df: pd.DataFrame,
    test_aug_df: pd.DataFrame,
    inv_bin_width: float = 0.1,
    n_folds: int = 5,
    eps: float = 1e-6,
) -> pd.Series:
    dx_tr = train_aug_df["x0"].to_numpy() - train_aug_df["x1"].to_numpy()
    dy_tr = train_aug_df["y0"].to_numpy() - train_aug_df["y1"].to_numpy()
    dz_tr = train_aug_df["z0"].to_numpy() - train_aug_df["z1"].to_numpy()
    dist_tr = np.sqrt(dx_tr * dx_tr + dy_tr * dy_tr + dz_tr * dz_tr)
    inv_tr = 1.0 / (dist_tr + eps)
    tr_bin = np.floor(inv_tr / inv_bin_width).astype(np.int32)

    dx_te = test_aug_df["x0"].to_numpy() - test_aug_df["x1"].to_numpy()
    dy_te = test_aug_df["y0"].to_numpy() - test_aug_df["y1"].to_numpy()
    dz_te = test_aug_df["z0"].to_numpy() - test_aug_df["z1"].to_numpy()
    dist_te = np.sqrt(dx_te * dx_te + dy_te * dy_te + dz_te * dz_te)
    inv_te = 1.0 / (dist_te + eps)
    te_bin = np.floor(inv_te / inv_bin_width).astype(np.int32)

    mol = train_aug_df["molecule_name"].astype(str)
    fold_id = (
        pd.util.hash_pandas_object(mol, index=False).astype(np.uint64)
        % np.uint64(n_folds)
    ).astype(np.int16)
    fold_id = pd.Series(fold_id.to_numpy(), index=train_aug_df.index)

    oof_pred = np.empty(train_aug_df.shape[0], dtype=np.float64)
    y = train_aug_df[TARGET].to_numpy(dtype=np.float64)
    types = train_aug_df["type"].to_numpy()

    for f in range(n_folds):
        tr_mask = (fold_id != f).to_numpy()
        val_mask = ~tr_mask

        tmp = pd.DataFrame(
            {"type": types[tr_mask], "inv_bin": tr_bin[tr_mask], TARGET: y[tr_mask]}
        )
        inv_map_f = tmp.groupby(["type", "inv_bin"], sort=False)[TARGET].mean()

        val_key = pd.MultiIndex.from_arrays(
            [types[val_mask], tr_bin[val_mask]], names=["type", "inv_bin"]
        )
        oof_pred[val_mask] = inv_map_f.reindex(val_key).to_numpy(dtype=np.float64)

    tmp_oof = pd.DataFrame(
        {"type": train_aug_df["type"].values, "inv_bin": tr_bin, "oof_pred": oof_pred}
    )
    inv_map_oof = tmp_oof.groupby(["type", "inv_bin"], sort=False)["oof_pred"].mean()

    te_key = pd.MultiIndex.from_arrays(
        [test_aug_df["type"].values, te_bin], names=["type", "inv_bin"]
    )
    out = pd.Series(
        inv_map_oof.reindex(te_key).to_numpy(dtype=np.float64),
        index=test_aug_df.index,
        dtype="float64",
    )
    return out


def add_pair_distance_pred_oof_shrunk(
    train_aug_df: pd.DataFrame,
    test_aug_df: pd.DataFrame,
    base_test_dist_pred: pd.Series,
    bin_width: float = 0.02,
    n_folds: int = 5,
    k_smooth: float = 30.0,
) -> pd.Series:
    dx_tr = train_aug_df["x0"].to_numpy() - train_aug_df["x1"].to_numpy()
    dy_tr = train_aug_df["y0"].to_numpy() - train_aug_df["y1"].to_numpy()
    dz_tr = train_aug_df["z0"].to_numpy() - train_aug_df["z1"].to_numpy()
    dist_tr = np.sqrt(dx_tr * dx_tr + dy_tr * dy_tr + dz_tr * dz_tr)
    tr_bin = np.floor(dist_tr / bin_width).astype(np.int32)

    dx_te = test_aug_df["x0"].to_numpy() - test_aug_df["x1"].to_numpy()
    dy_te = test_aug_df["y0"].to_numpy() - test_aug_df["y1"].to_numpy()
    dz_te = test_aug_df["z0"].to_numpy() - test_aug_df["z1"].to_numpy()
    dist_te = np.sqrt(dx_te * dx_te + dy_te * dy_te + dz_te * dz_te)
    te_bin = np.floor(dist_te / bin_width).astype(np.int32)

    mol = train_aug_df["molecule_name"].astype(str)
    fold_id = (
        pd.util.hash_pandas_object(mol, index=False).astype(np.uint64)
        % np.uint64(n_folds)
    ).astype(np.int16)
    fold_id = pd.Series(fold_id.to_numpy(), index=train_aug_df.index)

    oof_pred = np.empty(train_aug_df.shape[0], dtype=np.float64)
    y = train_aug_df[TARGET].to_numpy(dtype=np.float64)
    types = train_aug_df["type"].to_numpy()
    a0 = train_aug_df["atom_0"].to_numpy()
    a1 = train_aug_df["atom_1"].to_numpy()

    for f in range(n_folds):
        tr_mask = (fold_id != f).to_numpy()
        val_mask = ~tr_mask

        tmp = pd.DataFrame(
            {
                "type": types[tr_mask],
                "atom_0": a0[tr_mask],
                "atom_1": a1[tr_mask],
                "dist_bin": tr_bin[tr_mask],
                TARGET: y[tr_mask],
            }
        )
        m_f = tmp.groupby(["type", "atom_0", "atom_1", "dist_bin"], sort=False)[
            TARGET
        ].mean()

        val_key = pd.MultiIndex.from_arrays(
            [types[val_mask], a0[val_mask], a1[val_mask], tr_bin[val_mask]],
            names=["type", "atom_0", "atom_1", "dist_bin"],
        )
        oof_pred[val_mask] = m_f.reindex(val_key).to_numpy(dtype=np.float64)

    tmp_oof = pd.DataFrame(
        {
            "type": train_aug_df["type"].values,
            "atom_0": train_aug_df["atom_0"].values,
            "atom_1": train_aug_df["atom_1"].values,
            "dist_bin": tr_bin,
            "oof_pred": oof_pred,
        }
    )
    mean_map = tmp_oof.groupby(["type", "atom_0", "atom_1", "dist_bin"], sort=False)[
        "oof_pred"
    ].mean()
    cnt_map = tmp_oof.groupby(["type", "atom_0", "atom_1", "dist_bin"], sort=False)[
        "oof_pred"
    ].size()

    te_key = pd.MultiIndex.from_arrays(
        [
            test_aug_df["type"].values,
            test_aug_df["atom_0"].values,
            test_aug_df["atom_1"].values,
            te_bin,
        ],
        names=["type", "atom_0", "atom_1", "dist_bin"],
    )

    te_mean = mean_map.reindex(te_key).to_numpy(dtype=np.float64)
    te_cnt = cnt_map.reindex(te_key).to_numpy(dtype=np.float64)

    te_mean_s = pd.Series(te_mean, index=test_aug_df.index, dtype="float64")
    te_cnt_s = pd.Series(te_cnt, index=test_aug_df.index, dtype="float64")

    base = base_test_dist_pred.reindex(test_aug_df.index).astype(float)
    alpha = te_cnt_s / (te_cnt_s + float(k_smooth))
    out = alpha * te_mean_s + (1.0 - alpha) * base
    return out


def add_pair_distance_pred_oof_shrunk_median(
    train_aug_df: pd.DataFrame,
    test_aug_df: pd.DataFrame,
    base_test_dist_pred: pd.Series,
    bin_width: float = 0.02,
    n_folds: int = 5,
    k_smooth: float = 30.0,
) -> pd.Series:
    dx_tr = train_aug_df["x0"].to_numpy() - train_aug_df["x1"].to_numpy()
    dy_tr = train_aug_df["y0"].to_numpy() - train_aug_df["y1"].to_numpy()
    dz_tr = train_aug_df["z0"].to_numpy() - train_aug_df["z1"].to_numpy()
    dist_tr = np.sqrt(dx_tr * dx_tr + dy_tr * dy_tr + dz_tr * dz_tr)
    tr_bin = np.floor(dist_tr / bin_width).astype(np.int32)

    dx_te = test_aug_df["x0"].to_numpy() - test_aug_df["x1"].to_numpy()
    dy_te = test_aug_df["y0"].to_numpy() - test_aug_df["y1"].to_numpy()
    dz_te = test_aug_df["z0"].to_numpy() - test_aug_df["z1"].to_numpy()
    dist_te = np.sqrt(dx_te * dx_te + dy_te * dy_te + dz_te * dz_te)
    te_bin = np.floor(dist_te / bin_width).astype(np.int32)

    mol = train_aug_df["molecule_name"].astype(str)
    fold_id = (
        pd.util.hash_pandas_object(mol, index=False).astype(np.uint64)
        % np.uint64(n_folds)
    ).astype(np.int16)
    fold_id = pd.Series(fold_id.to_numpy(), index=train_aug_df.index)

    oof_pred = np.empty(train_aug_df.shape[0], dtype=np.float64)
    y = train_aug_df[TARGET].to_numpy(dtype=np.float64)
    types = train_aug_df["type"].to_numpy()
    a0 = train_aug_df["atom_0"].to_numpy()
    a1 = train_aug_df["atom_1"].to_numpy()

    for f in range(n_folds):
        tr_mask = (fold_id != f).to_numpy()
        val_mask = ~tr_mask

        tmp = pd.DataFrame(
            {
                "type": types[tr_mask],
                "atom_0": a0[tr_mask],
                "atom_1": a1[tr_mask],
                "dist_bin": tr_bin[tr_mask],
                TARGET: y[tr_mask],
            }
        )
        m_f = tmp.groupby(["type", "atom_0", "atom_1", "dist_bin"], sort=False)[
            TARGET
        ].median()

        val_key = pd.MultiIndex.from_arrays(
            [types[val_mask], a0[val_mask], a1[val_mask], tr_bin[val_mask]],
            names=["type", "atom_0", "atom_1", "dist_bin"],
        )
        oof_pred[val_mask] = m_f.reindex(val_key).to_numpy(dtype=np.float64)

    tmp_oof = pd.DataFrame(
        {
            "type": train_aug_df["type"].values,
            "atom_0": train_aug_df["atom_0"].values,
            "atom_1": train_aug_df["atom_1"].values,
            "dist_bin": tr_bin,
            "oof_pred": oof_pred,
        }
    )
    med_map = tmp_oof.groupby(["type", "atom_0", "atom_1", "dist_bin"], sort=False)[
        "oof_pred"
    ].median()
    cnt_map = tmp_oof.groupby(["type", "atom_0", "atom_1", "dist_bin"], sort=False)[
        "oof_pred"
    ].size()

    te_key = pd.MultiIndex.from_arrays(
        [
            test_aug_df["type"].values,
            test_aug_df["atom_0"].values,
            test_aug_df["atom_1"].values,
            te_bin,
        ],
        names=["type", "atom_0", "atom_1", "dist_bin"],
    )

    te_med = med_map.reindex(te_key).to_numpy(dtype=np.float64)
    te_cnt = cnt_map.reindex(te_key).to_numpy(dtype=np.float64)

    te_med_s = pd.Series(te_med, index=test_aug_df.index, dtype="float64")
    te_cnt_s = pd.Series(te_cnt, index=test_aug_df.index, dtype="float64")

    base = base_test_dist_pred.reindex(test_aug_df.index).astype(float)
    alpha = te_cnt_s / (te_cnt_s + float(k_smooth))
    out = alpha * te_med_s + (1.0 - alpha) * base
    return out


test["dist_pred"] = add_distance_based_pred_oof(
    train_aug, test_aug, bin_width=0.02, n_folds=5
)
test["inv_dist_pred"] = add_inv_distance_based_pred_oof(
    train_aug, test_aug, inv_bin_width=0.1, n_folds=5
)

test["pair_dist_pred"] = add_pair_distance_pred_oof_shrunk(
    train_aug_df=train_aug,
    test_aug_df=test_aug,
    base_test_dist_pred=test["dist_pred"],
    bin_width=0.02,
    n_folds=5,
    k_smooth=30.0,
)

test["pair_dist_med_pred"] = add_pair_distance_pred_oof_shrunk_median(
    train_aug_df=train_aug,
    test_aug_df=test_aug,
    base_test_dist_pred=test["dist_pred"],
    bin_width=0.02,
    n_folds=5,
    k_smooth=30.0,
)

type_fallback = test["type"].map(type_mean).astype(float)  # Series aligned to test
for c in [
    "n1",
    "n2",
    "lgb_a",
    "lgb_m",
    "nnet",
    "dist_pred",
    "inv_dist_pred",
    "pair_dist_pred",
    "pair_dist_med_pred",
]:
    test[c] = test[c].astype(float)
    test[c] = test[c].fillna(type_fallback)
    test[c] = test[c].fillna(global_mean)

test[
    [
        "n1",
        "n2",
        "lgb_a",
        "lgb_m",
        "nnet",
        "dist_pred",
        "inv_dist_pred",
        "pair_dist_pred",
        "pair_dist_med_pred",
    ]
].head(10)



## === cell 3
test["final_preds"] = (
    test["n1"] * 0.14
    + test["n2"] * 0.12
    + test["lgb_a"] * 0.16
    + test["lgb_m"] * 0.17
    + test["nnet"] * 0.05
    + test["dist_pred"] * 0.18
    + test["inv_dist_pred"] * 0.08
    + test["pair_dist_pred"] * 0.08
    + test["pair_dist_med_pred"] * 0.02
)

test["final_preds"] = test["final_preds"].fillna(global_mean).astype(float)
test[["id", "final_preds"]].head()



## === cell 4
submission = sample_sub[["id"]].merge(test[["id", "final_preds"]], on="id", how="left")
submission["scalar_coupling_constant"] = (
    submission["final_preds"].fillna(global_mean).astype(float)
)
submission = submission[["id", "scalar_coupling_constant"]]

out_path = "sub_ensemble.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path, "shape:", submission.shape)
submission.head()



## === cell 5
assert out_path.endswith(".csv")
assert submission.columns.tolist() == ["id", "scalar_coupling_constant"]
assert submission["id"].isna().sum() == 0
assert submission["scalar_coupling_constant"].isna().sum() == 0
print("Submission looks valid.")
submission.tail(10)
