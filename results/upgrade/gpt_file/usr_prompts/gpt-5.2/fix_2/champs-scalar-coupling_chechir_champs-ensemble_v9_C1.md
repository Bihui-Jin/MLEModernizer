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

3.00563

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 3.00563) has done: 'I fix the missing input dependency by removing the hard-coded `../input/champ-preds` / `../input/nnet-try*` reads and instead generate a valid baseline prediction from the provided competition data files. To keep the logic as close as possible to your “ensemble of multiple predictors,” I build several simple, legitimate per-`type` and per-atom-pair aggregations from `train.csv` and blend them with the same weighted-sum pattern, producing a stable prediction for every test row. I also make the code robust to missing categories by falling back to global means, and ensure the output file is written as a valid `.csv` with the required columns (`id,scalar_coupling_constant`). This run end-to-end in the given environment and yield a reasonable score (not optimal, but far better than a constant), moving toward the target from “no submission”.'

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


def safe_map(series_key_df, group_cols, target_col):
    """
    Returns a Series aligned to series_key_df.index by mapping group means from train.
    """
    grp = train.groupby(group_cols, sort=False)[target_col].mean()
    key = pd.MultiIndex.from_frame(series_key_df[group_cols])
    return pd.Series(grp.reindex(key).to_numpy(), index=series_key_df.index)


global_mean = train[TARGET].mean()

test["n1"] = (
    train.groupby("type", sort=False)[TARGET].mean().reindex(test["type"]).to_numpy()
)

structures = pd.read_csv(
    os.path.join(DATA_DIR, "structures.csv"),
    usecols=["molecule_name", "atom_index", "atom"],
)
s0 = structures.rename(columns={"atom_index": "atom_index_0", "atom": "atom_0"})
s1 = structures.rename(columns={"atom_index": "atom_index_1", "atom": "atom_1"})

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
)

m0 = train.groupby(["type", "atom_index_0"], sort=False)[TARGET].mean()
m1 = train.groupby(["type", "atom_index_1"], sort=False)[TARGET].mean()
v0 = pd.Series(
    m0.reindex(pd.MultiIndex.from_frame(test[["type", "atom_index_0"]])).to_numpy(),
    index=test.index,
)
v1 = pd.Series(
    m1.reindex(pd.MultiIndex.from_frame(test[["type", "atom_index_1"]])).to_numpy(),
    index=test.index,
)
test["lgb_a"] = 0.5 * v0 + 0.5 * v1

m01 = train.groupby(["type", "atom_index_0", "atom_index_1"], sort=False)[TARGET].mean()
test["lgb_m"] = pd.Series(
    m01.reindex(
        pd.MultiIndex.from_frame(test[["type", "atom_index_0", "atom_index_1"]])
    ).to_numpy(),
    index=test.index,
)

m_atoms = train_aug.groupby(["atom_0", "atom_1"], sort=False)[TARGET].mean()
test["nnet"] = pd.Series(
    m_atoms.reindex(
        pd.MultiIndex.from_frame(test_aug[["atom_0", "atom_1"]])
    ).to_numpy(),
    index=test.index,
)

type_mean = train.groupby("type", sort=False)[TARGET].mean()
for c in ["n1", "n2", "lgb_a", "lgb_m", "nnet"]:
    test[c] = test[c].fillna(type_mean.reindex(test["type"]).to_numpy())
    test[c] = test[c].fillna(global_mean)

test[["n1", "n2", "lgb_a", "lgb_m", "nnet"]].head(10)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4209473981.py in <cell line: 0>()
     79 type_mean = train.groupby("type", sort=False)[TARGET].mean()
     80 for c in ["n1", "n2", "lgb_a", "lgb_m", "nnet"]:
---> 81     test[c] = test[c].fillna(type_mean.reindex(test["type"]).to_numpy())
     82     test[c] = test[c].fillna(global_mean)
     83 

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in fillna(self, value, method, axis, inplace, limit, downcast)
   7341                     pass
   7342                 else:
-> 7343                     raise TypeError(
   7344                         '"value" parameter must be a scalar, dict '
   7345                         "or Series, but you passed a "

TypeError: "value" parameter must be a scalar, dict or Series, but you passed a "ndarray"

## === cell 3
test["final_preds"] = (
    test["n1"] * 0.6
    + test["n2"] * 0.05
    + test["lgb_a"] * 0.15
    + test["lgb_m"] * 0.1
    + test["nnet"] * 0.05
)

test["final_preds"] = test["final_preds"].fillna(global_mean)

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
