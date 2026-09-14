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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from subprocess import check_output
import seaborn as sns

sns.set()
import matplotlib.pyplot as plt

print(check_output(["ls", "../input/"]).decode("utf8"))



## === cell 1
from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

DATA_DIR_CANDIDATES = [
    "../input/champs-scalar-coupling",
    "../input/kaggle/data/champs-scalar-coupling",
    "../input/kaggle/input/champs-scalar-coupling",
]

data_dir = None
for p in DATA_DIR_CANDIDATES:
    if os.path.isdir(p) and os.path.isfile(os.path.join(p, "train.csv")):
        data_dir = p
        break
if data_dir is None:
    raise FileNotFoundError(
        "Could not locate champs-scalar-coupling data directory with train.csv"
    )

train_path = os.path.join(data_dir, "train.csv")
test_path = os.path.join(data_dir, "test.csv")
sample_path = os.path.join(data_dir, "sample_submission.csv")
structures_path = os.path.join(data_dir, "structures.csv")

mulliken_path = os.path.join(data_dir, "mulliken_charges.csv")
shielding_path = os.path.join(data_dir, "magnetic_shielding_tensors.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_path)

structures = pd.read_csv(
    structures_path,
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

mulliken = pd.read_csv(
    mulliken_path, usecols=["molecule_name", "atom_index", "mulliken_charge"]
)
m0 = mulliken.rename(
    columns={"atom_index": "atom_index_0", "mulliken_charge": "mulliken_0"}
)
m1 = mulliken.rename(
    columns={"atom_index": "atom_index_1", "mulliken_charge": "mulliken_1"}
)

shield = pd.read_csv(
    shielding_path,
    usecols=["molecule_name", "atom_index", "XX", "YY", "ZZ", "XY", "XZ", "YZ"],
)

shield["trace"] = shield["XX"] + shield["YY"] + shield["ZZ"]

sh0 = shield.rename(
    columns={
        "atom_index": "atom_index_0",
        "XX": "shield_xx_0",
        "YY": "shield_yy_0",
        "ZZ": "shield_zz_0",
        "XY": "shield_xy_0",
        "XZ": "shield_xz_0",
        "YZ": "shield_yz_0",
        "trace": "shield_trace_0",
    }
)
sh1 = shield.rename(
    columns={
        "atom_index": "atom_index_1",
        "XX": "shield_xx_1",
        "YY": "shield_yy_1",
        "ZZ": "shield_zz_1",
        "XY": "shield_xy_1",
        "XZ": "shield_xz_1",
        "YZ": "shield_yz_1",
        "trace": "shield_trace_1",
    }
)


def add_distance_feature(df):
    df2 = df.merge(s0, on=["molecule_name", "atom_index_0"], how="left").merge(
        s1, on=["molecule_name", "atom_index_1"], how="left"
    )

    dx = df2["x0"].values - df2["x1"].values
    dy = df2["y0"].values - df2["y1"].values
    dz = df2["z0"].values - df2["z1"].values
    dist = np.sqrt(dx * dx + dy * dy + dz * dz)
    df2["dx"] = dx
    df2["dy"] = dy
    df2["dz"] = dz
    df2["dist"] = dist
    df2["inv_dist"] = 1.0 / (dist + 1e-6)

    df2 = (
        df2.merge(m0, on=["molecule_name", "atom_index_0"], how="left")
        .merge(m1, on=["molecule_name", "atom_index_1"], how="left")
        .merge(sh0, on=["molecule_name", "atom_index_0"], how="left")
        .merge(sh1, on=["molecule_name", "atom_index_1"], how="left")
    )

    df2["mulliken_sum"] = df2["mulliken_0"] + df2["mulliken_1"]
    df2["mulliken_diff"] = df2["mulliken_0"] - df2["mulliken_1"]

    df2["shield_xx_sum"] = df2["shield_xx_0"] + df2["shield_xx_1"]
    df2["shield_xx_diff"] = df2["shield_xx_0"] - df2["shield_xx_1"]
    df2["shield_yy_sum"] = df2["shield_yy_0"] + df2["shield_yy_1"]
    df2["shield_yy_diff"] = df2["shield_yy_0"] - df2["shield_yy_1"]
    df2["shield_zz_sum"] = df2["shield_zz_0"] + df2["shield_zz_1"]
    df2["shield_zz_diff"] = df2["shield_zz_0"] - df2["shield_zz_1"]

    df2["shield_xy_sum"] = df2["shield_xy_0"] + df2["shield_xy_1"]
    df2["shield_xy_diff"] = df2["shield_xy_0"] - df2["shield_xy_1"]
    df2["shield_xz_sum"] = df2["shield_xz_0"] + df2["shield_xz_1"]
    df2["shield_xz_diff"] = df2["shield_xz_0"] - df2["shield_xz_1"]
    df2["shield_yz_sum"] = df2["shield_yz_0"] + df2["shield_yz_1"]
    df2["shield_yz_diff"] = df2["shield_yz_0"] - df2["shield_yz_1"]

    df2["shield_trace_sum"] = df2["shield_trace_0"] + df2["shield_trace_1"]
    df2["shield_trace_diff"] = df2["shield_trace_0"] - df2["shield_trace_1"]

    return df2


train_fe = add_distance_feature(train)
test_fe = add_distance_feature(test)

feature_cols = [
    "type",
    "atom_0",
    "atom_1",
    "atom_index_0",
    "atom_index_1",
    "x0",
    "y0",
    "z0",
    "x1",
    "y1",
    "z1",
    "dx",
    "dy",
    "dz",
    "dist",
    "inv_dist",
    "mulliken_0",
    "mulliken_1",
    "mulliken_sum",
    "mulliken_diff",
    "shield_xx_0",
    "shield_xx_1",
    "shield_xx_sum",
    "shield_xx_diff",
    "shield_yy_0",
    "shield_yy_1",
    "shield_yy_sum",
    "shield_yy_diff",
    "shield_zz_0",
    "shield_zz_1",
    "shield_zz_sum",
    "shield_zz_diff",
    "shield_xy_0",
    "shield_xy_1",
    "shield_xy_sum",
    "shield_xy_diff",
    "shield_xz_0",
    "shield_xz_1",
    "shield_xz_sum",
    "shield_xz_diff",
    "shield_yz_0",
    "shield_yz_1",
    "shield_yz_sum",
    "shield_yz_diff",
    "shield_trace_0",
    "shield_trace_1",
    "shield_trace_sum",
    "shield_trace_diff",
]
target_col = "scalar_coupling_constant"
group_col = "molecule_name"

X = train_fe[feature_cols].copy()
y = train_fe[target_col].astype(float).values
groups = train_fe[group_col].values

X_test = test_fe[feature_cols].copy()

cat_features = ["type", "atom_0", "atom_1"]
num_features = [
    "atom_index_0",
    "atom_index_1",
    "x0",
    "y0",
    "z0",
    "x1",
    "y1",
    "z1",
    "dx",
    "dy",
    "dz",
    "dist",
    "inv_dist",
    "mulliken_0",
    "mulliken_1",
    "mulliken_sum",
    "mulliken_diff",
    "shield_xx_0",
    "shield_xx_1",
    "shield_xx_sum",
    "shield_xx_diff",
    "shield_yy_0",
    "shield_yy_1",
    "shield_yy_sum",
    "shield_yy_diff",
    "shield_zz_0",
    "shield_zz_1",
    "shield_zz_sum",
    "shield_zz_diff",
    "shield_xy_0",
    "shield_xy_1",
    "shield_xy_sum",
    "shield_xy_diff",
    "shield_xz_0",
    "shield_xz_1",
    "shield_xz_sum",
    "shield_xz_diff",
    "shield_yz_0",
    "shield_yz_1",
    "shield_yz_sum",
    "shield_yz_diff",
    "shield_trace_0",
    "shield_trace_1",
    "shield_trace_sum",
    "shield_trace_diff",
]

preprocess = ColumnTransformer(
    transformers=[
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            cat_features,
        ),
        (
            "num",
            Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
            num_features,
        ),
    ],
    remainder="drop",
    sparse_threshold=0.3,
)

gkf = GroupKFold(n_splits=3)

test_pred_folds = [np.zeros(len(test), dtype=np.float64) for _ in range(3)]
test_pred_full = np.zeros(len(test), dtype=np.float64)

ALPHA_GRID = [0.1, 0.3, 1.0, 3.0, 10.0]

for t in sorted(train_fe["type"].unique()):
    tr_mask = train_fe["type"].values == t
    te_mask = test_fe["type"].values == t

    X_t = X.loc[tr_mask]
    y_t = y[tr_mask]
    groups_t = train_fe.loc[tr_mask, group_col].values

    X_test_t = X_test.loc[te_mask]

    if len(X_test_t) == 0:
        continue

    best_alpha = 1.0
    best_mae = np.inf

    for a in ALPHA_GRID:
        fold_maes = []
        for tr_idx, va_idx in gkf.split(X_t, y_t, groups=groups_t):
            m = Pipeline(steps=[("prep", preprocess), ("model", Ridge(alpha=a))])
            m.fit(X_t.iloc[tr_idx], y_t[tr_idx])
            va_pred = m.predict(X_t.iloc[va_idx])
            fold_maes.append(mean_absolute_error(y_t[va_idx], va_pred))
        mae_a = float(np.mean(fold_maes))
        if mae_a < best_mae:
            best_mae = mae_a
            best_alpha = a

    for fold_id, (tr_idx, va_idx) in enumerate(gkf.split(X_t, y_t, groups=groups_t)):
        model = Pipeline(
            steps=[("prep", preprocess), ("model", Ridge(alpha=best_alpha))]
        )
        model.fit(X_t.iloc[tr_idx], y_t[tr_idx])
        test_pred_folds[fold_id][te_mask] = model.predict(X_test_t)

    full_model = Pipeline(
        steps=[("prep", preprocess), ("model", Ridge(alpha=best_alpha))]
    )
    full_model.fit(X_t, y_t)
    test_pred_full[te_mask] = full_model.predict(X_test_t)

concat_sub = pd.DataFrame(
    {
        "id": test["id"].values,
        "champ0": test_pred_folds[0],
        "champ1": test_pred_folds[1],
        "champ2": test_pred_folds[2],
        "champ3": test_pred_full,
    }
)

cols = ["id"] + list(map(lambda x: "champ" + str(x), range(concat_sub.shape[1] - 1)))
concat_sub.columns = cols
ncol = concat_sub.shape[1]

sub_base = pd.DataFrame(
    {"id": test["id"].values, "scalar_coupling_constant": test_pred_full}
)

concat_sub.head()



## === cell 2
concat_sub.iloc[:, 1:ncol].corr()



## === cell 3
corr = concat_sub.iloc[:, 1 : min(7, ncol)].corr()

mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(
    corr,
    mask=mask,
    cmap=cmap,
    vmax=0.3,
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.5},
)



## === cell 4
concat_sub["champ_max"] = concat_sub.iloc[:, 1:ncol].max(axis=1)
concat_sub["champ_min"] = concat_sub.iloc[:, 1:ncol].min(axis=1)
concat_sub["champ_mean"] = concat_sub.iloc[:, 1:ncol].mean(axis=1)
concat_sub["champ_median"] = concat_sub.iloc[:, 1:ncol].median(axis=1)



## === cell 5
concat_sub.describe()



## === cell 6
cutoff_lo = -37
cutoff_hi = 205



## === cell 7
concat_sub["scalar_coupling_constant"] = concat_sub["champ_mean"]
concat_sub[["id", "scalar_coupling_constant"]].to_csv(
    "stack_mean.csv", index=False, float_format="%.6f"
)



## === cell 8
concat_sub["scalar_coupling_constant"] = concat_sub["champ_median"]
concat_sub[["id", "scalar_coupling_constant"]].to_csv(
    "stack_median.csv", index=False, float_format="%.6f"
)



## === cell 9
concat_sub["scalar_coupling_constant"] = np.where(
    np.all(concat_sub.iloc[:, 1:ncol] > cutoff_lo, axis=1),
    1,
    np.where(
        np.all(concat_sub.iloc[:, 1:ncol] < cutoff_hi, axis=1),
        0,
        concat_sub["champ_median"],
    ),
)
concat_sub[["id", "scalar_coupling_constant"]].to_csv(
    "stack_pushout_median.csv", index=False, float_format="%.6f"
)



## === cell 10
concat_sub["scalar_coupling_constant"] = np.where(
    np.all(concat_sub.iloc[:, 1:ncol] > cutoff_lo, axis=1),
    concat_sub["champ_max"],
    np.where(
        np.all(concat_sub.iloc[:, 1:ncol] < cutoff_hi, axis=1),
        concat_sub["champ_min"],
        concat_sub["champ_mean"],
    ),
)
concat_sub[["id", "scalar_coupling_constant"]].to_csv(
    "stack_minmax_mean.csv", index=False, float_format="%.6f"
)



## === cell 11
concat_sub["scalar_coupling_constant"] = np.where(
    np.all(concat_sub.iloc[:, 1:ncol] > cutoff_lo, axis=1),
    concat_sub["champ_max"],
    np.where(
        np.all(concat_sub.iloc[:, 1:ncol] < cutoff_hi, axis=1),
        concat_sub["champ_min"],
        concat_sub["champ_median"],
    ),
)
concat_sub[["id", "scalar_coupling_constant"]].to_csv(
    "stack_minmax_median.csv", index=False, float_format="%.6f"
)



## === cell 12
if "scalar_coupling_constant" not in sub_base.columns:
    raise ValueError(
        f"Expected 'scalar_coupling_constant' column in sub_base. Got columns: {list(sub_base.columns)}"
    )

sub_base = sub_base.set_index("id").loc[concat_sub["id"].values].reset_index()



## === cell 13
concat_sub["champ_base"] = sub_base["scalar_coupling_constant"]
concat_sub["scalar_coupling_constant"] = np.where(
    np.all(concat_sub.iloc[:, 1:ncol] > cutoff_lo, axis=1),
    concat_sub["champ_max"],
    np.where(
        np.all(concat_sub.iloc[:, 1:ncol] < cutoff_hi, axis=1),
        concat_sub["champ_min"],
        concat_sub["champ_base"],
    ),
)
concat_sub[["id", "scalar_coupling_constant"]].to_csv(
    "stack_minmax_bestbase.csv", index=False, float_format="%.6f"
)
