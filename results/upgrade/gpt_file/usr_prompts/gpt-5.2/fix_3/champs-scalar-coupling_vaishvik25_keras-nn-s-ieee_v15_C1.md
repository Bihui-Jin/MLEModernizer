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
seaborn==0.12.2
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

-1.356722768415636

# 6. Current score

3.12217

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 3.12217) has done: 'I fix the NaNs coming from the structures merge by enforcing consistent dtypes on join keys and de-duplicating `structures` on `(molecule_name, atom_index)` before merging (the current NaNs indicate key mismatches/duplicates). Then I add a minimal, score-neutral safety net by imputing any remaining NaNs in the final feature matrix with column medians so `Ridge` can fit/predict reliably. I also make the later cells robust by only using variables that are guaranteed to exist after the training/prediction cell completes, and ensure a single valid `submission.csv` is always written with the exact required columns and id alignment. Core modeling logic (pair features + one-hot + per-type Ridge + simple averaging) is preserved.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge



## === cell 1
BASE_PATH = "../input/champs-scalar-coupling"
TRAIN_PATH = os.path.join(BASE_PATH, "train.csv")
TEST_PATH = os.path.join(BASE_PATH, "test.csv")
STRUCTURES_PATH = os.path.join(BASE_PATH, "structures.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")

for p in [TRAIN_PATH, TEST_PATH, STRUCTURES_PATH, SAMPLE_SUB_PATH]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required file not found: {p}")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
structures = pd.read_csv(STRUCTURES_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

assert set(["id", "molecule_name", "atom_index_0", "atom_index_1", "type"]).issubset(
    test.columns
)
assert set(["id", "scalar_coupling_constant"]).issubset(sample_sub.columns)

print(
    "train:",
    train.shape,
    "test:",
    test.shape,
    "structures:",
    structures.shape,
    "sample_sub:",
    sample_sub.shape,
)
print(
    "types (train):", train["type"].nunique(), "types (test):", test["type"].nunique()
)



## === cell 2
for df in (train, test, structures):
    df["molecule_name"] = df["molecule_name"].astype(str)

structures["atom_index"] = structures["atom_index"].astype(np.int32)
train["atom_index_0"] = train["atom_index_0"].astype(np.int32)
train["atom_index_1"] = train["atom_index_1"].astype(np.int32)
test["atom_index_0"] = test["atom_index_0"].astype(np.int32)
test["atom_index_1"] = test["atom_index_1"].astype(np.int32)

structures = structures.drop_duplicates(
    subset=["molecule_name", "atom_index"], keep="first"
).reset_index(drop=True)

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


def add_pair_features(df):
    df = df.merge(
        s0[["molecule_name", "atom_index_0", "atom_0", "x_0", "y_0", "z_0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
        validate="many_to_one",
    )
    df = df.merge(
        s1[["molecule_name", "atom_index_1", "atom_1", "x_1", "y_1", "z_1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
        validate="many_to_one",
    )

    dx = df["x_0"] - df["x_1"]
    dy = df["y_0"] - df["y_1"]
    dz = df["z_0"] - df["z_1"]
    df["dx"] = dx
    df["dy"] = dy
    df["dz"] = dz
    df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz)

    df["abs_dx"] = dx.abs()
    df["abs_dy"] = dy.abs()
    df["abs_dz"] = dz.abs()

    df["dist2"] = df["dist"] ** 2
    df["inv_dist"] = 1.0 / (df["dist"] + 1e-9)

    return df


train_f = add_pair_features(train)
test_f = add_pair_features(test)

check_cols = ["x_0", "y_0", "z_0", "x_1", "y_1", "z_1", "dist", "atom_0", "atom_1"]
bad_train = train_f[check_cols].isna().any(axis=1).sum()
bad_test = test_f[check_cols].isna().any(axis=1).sum()
print(
    "Rows with any missing merged structure fields - train:",
    bad_train,
    "test:",
    bad_test,
)

if bad_train > 0 or bad_test > 0:
    raise ValueError(
        "Still found NaNs after merge; check molecule_name/atom_index key integrity."
    )



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3504436485.py in <cell line: 0>()
     83     # In the official dataset these should be 0 after dtype/dup fixes; raise to catch genuine data/path issues.
     84     # (We still add later imputation for numeric matrices as an extra safety net.)
---> 85     raise ValueError(
     86         "Still found NaNs after merge; check molecule_name/atom_index key integrity."
     87     )

ValueError: Still found NaNs after merge; check molecule_name/atom_index key integrity.

## === cell 3
cat_cols = ["type", "atom_0", "atom_1"]
num_cols = ["dx", "dy", "dz", "abs_dx", "abs_dy", "abs_dz", "dist", "dist2", "inv_dist"]

all_df = pd.concat(
    [train_f[["id"] + cat_cols + num_cols], test_f[["id"] + cat_cols + num_cols]],
    axis=0,
    ignore_index=True,
)

all_df = pd.get_dummies(all_df, columns=cat_cols, dummy_na=False)

X_all = all_df.drop(columns=["id"])
if X_all.isna().any().any():
    med = X_all.median(axis=0)
    X_all = X_all.fillna(med)

X_train = X_all.iloc[: len(train_f), :].to_numpy(dtype=np.float32, copy=False)
X_test = X_all.iloc[len(train_f) :, :].to_numpy(dtype=np.float32, copy=False)

y_train = train_f["scalar_coupling_constant"].to_numpy(dtype=np.float32, copy=False)

print("X_train:", X_train.shape, "X_test:", X_test.shape, "y_train:", y_train.shape)



## === cell 4
type_train = train_f["type"].values
type_test = test_f["type"].values

unique_types = np.unique(type_train)
pred_test = np.zeros(len(test_f), dtype=np.float32)

for t in unique_types:
    tr_idx = np.where(type_train == t)[0]
    te_idx = np.where(type_test == t)[0]
    if len(te_idx) == 0:
        continue

    model = Ridge(alpha=1.0, random_state=0)
    model.fit(X_train[tr_idx], y_train[tr_idx])
    pred_test[te_idx] = model.predict(X_test[te_idx]).astype(np.float32)

mol0 = pred_test

train_type_mean = train_f.groupby("type")["scalar_coupling_constant"].mean().to_dict()
mol1 = np.array(
    [
        0.95 * p + 0.05 * train_type_mean.get(t, 0.0)
        for p, t in zip(pred_test, type_test)
    ],
    dtype=np.float32,
)

mol2 = (0.98 * pred_test).astype(np.float32)

concat_sub = pd.DataFrame(
    {
        "id": test_f["id"].values,
        "mol0": mol0,
        "mol1": mol1,
        "mol2": mol2,
    }
)
ncol = concat_sub.shape[1]
print(concat_sub.head())



## === cell 5
try:
    corr = concat_sub[["mol0", "mol1", "mol2"]].corr()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        corr,
        mask=mask,
        cmap=sns.diverging_palette(220, 10, as_cmap=True),
        center=0,
        annot=True,
    )
    plt.tight_layout()
except Exception as e:
    print("Skipped correlation plot due to:", repr(e))



## === cell 6
concat_sub["m_max"] = concat_sub.iloc[:, 1:ncol].max(axis=1)
concat_sub["m_min"] = concat_sub.iloc[:, 1:ncol].min(axis=1)
concat_sub["m_mean"] = concat_sub.iloc[:, 1:ncol].mean(axis=1)
concat_sub["m_median"] = concat_sub.iloc[:, 1:ncol].median(axis=1)

print(
    concat_sub.describe().loc[
        ["mean", "std", "min", "max"], ["mol0", "mol1", "mol2", "m_mean", "m_median"]
    ]
)



## === cell 7
cutoff_lo = 0.8
cutoff_hi = 0.2



## === cell 8
concat_sub["scalar_coupling_constant"] = concat_sub["m_mean"]
concat_sub[["id", "scalar_coupling_constant"]].to_csv(
    "stack_mean.csv", index=False, float_format="%.6f"
)

concat_sub["scalar_coupling_constant"] = concat_sub["m_median"]
concat_sub[["id", "scalar_coupling_constant"]].to_csv(
    "stack_median.csv", index=False, float_format="%.6f"
)

concat_sub["scalar_coupling_constant"] = np.where(
    np.all(concat_sub.iloc[:, 1:ncol] > cutoff_lo, axis=1),
    1,
    np.where(
        np.all(concat_sub.iloc[:, 1:ncol] < cutoff_hi, axis=1),
        0,
        concat_sub["m_median"],
    ),
)
concat_sub[["id", "scalar_coupling_constant"]].to_csv(
    "stack_pushout_median.csv", index=False, float_format="%.6f"
)

concat_sub["scalar_coupling_constant"] = np.where(
    np.all(concat_sub.iloc[:, 1:ncol] > cutoff_lo, axis=1),
    concat_sub["m_max"],
    np.where(
        np.all(concat_sub.iloc[:, 1:ncol] < cutoff_hi, axis=1),
        concat_sub["m_min"],
        concat_sub["m_mean"],
    ),
)
concat_sub[["id", "scalar_coupling_constant"]].to_csv(
    "stack_minmax_mean.csv", index=False, float_format="%.6f"
)

concat_sub["scalar_coupling_constant"] = np.where(
    np.all(concat_sub.iloc[:, 1:ncol] > cutoff_lo, axis=1),
    concat_sub["m_max"],
    np.where(
        np.all(concat_sub.iloc[:, 1:ncol] < cutoff_hi, axis=1),
        concat_sub["m_min"],
        concat_sub["m_median"],
    ),
)
concat_sub[["id", "scalar_coupling_constant"]].to_csv(
    "stack_minmax_median.csv", index=False, float_format="%.6f"
)

rank_sum = None
for c in ["mol0", "mol1", "mol2"]:
    r = concat_sub[c].rank(method="min")
    rank_sum = r if rank_sum is None else (rank_sum + r)

rank_scaled = (rank_sum - rank_sum.min()) / (rank_sum.max() - rank_sum.min())
concat_sub["scalar_coupling_constant"] = rank_scaled.astype(np.float32)
concat_sub[["id", "scalar_coupling_constant"]].to_csv(
    "stack_rank.csv", index=False, float_format="%.8f"
)



## === cell 9
final_pred = pd.read_csv("stack_median.csv")

sub = sample_sub[["id"]].merge(final_pred, on="id", how="left")
if sub["scalar_coupling_constant"].isna().any():
    raise ValueError(
        "Submission has missing predictions after merge; id alignment issue."
    )

sub.to_csv("submission.csv", index=False, float_format="%.6f")
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())



## === cell 10
try:
    plt.figure(figsize=(7, 3))
    sns.histplot(sub["scalar_coupling_constant"], bins=50, kde=False)
    plt.tight_layout()
except Exception as e:
    print("Skipped histogram plot due to:", repr(e))
