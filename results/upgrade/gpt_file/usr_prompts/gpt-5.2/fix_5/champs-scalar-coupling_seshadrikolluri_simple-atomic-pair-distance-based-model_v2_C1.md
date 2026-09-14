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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

1.23424

# 6. Current score

21.86587

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 2.17067) has done: 'I fix the merge logic so atom coordinates are joined correctly (by molecule and atom index) instead of creating a cartesian product, which currently makes the dataset explode and is logically wrong. Then I make the one-hot encoding consistent between train and test by fitting on train and reindexing test to the same feature columns, which resolves the feature-name mismatch error. I also correct the input paths to match your environment (`/kaggle/data/champs-scalar-coupling/...`) and ensure the script always writes a valid `submission.csv` with the required columns. These changes preserve the core “linear regression on distance + type dummies” approach while making it run end-to-end and improving the expected score substantially versus the broken merge.'
- What this solution (achieved 27.9907) has done: 'We need to move the score down (lower is better) from 2.17067 toward 1.23424, so we should add a small amount of extra signal without changing the overall “linear regression on engineered features” approach. The most minimal gain in this competition usually comes from (1) adding atom identity dummies for atom_0/atom_1 (and their interaction as a combined category), and (2) adding simple distance transforms (inverse distance and squared distance) while keeping the same LinearRegression model and single fit/predict loop. These additions keep the core logic intact (same data sources, same merge, same model family) but typically reduce MAE per type because different coupling types/atom pairs follow different distance-response curves. I also ensure feature columns are aligned between train/test exactly as before and keep writing a valid `submission.csv`.'
- What this solution (achieved 20.30863) has done: 'We need to move the score down (lower is better) from 27.9907 toward 1.23424, so the smallest legitimate improvement is to add a bit more physics signal while keeping the same “single LinearRegression fit on engineered tabular features” core. I merge in three auxiliary files you already have (mulliken charges, magnetic shielding tensors, dipole moments, plus potential energy) and derive simple pair features (charge sum/diff/product; shielding row-wise stats per atom and pair diffs; dipole magnitude and projections onto the bond vector). These additions are lightweight, preserve the same training approach and model family, and typically improve this competition materially because they correlate with coupling constants across types. I keep the existing distance/type/atom dummies and ensure train/test columns stay perfectly aligned, still writing a valid `submission.csv`.'
- What this solution (achieved 21.86587) has done: 'Your current score is far above the target (20.31 vs 1.234, lower is better), so we need a legitimate performance improvement without changing the core “single LinearRegression on engineered tabular features” approach. The most impactful minimal fix for CHAMPS is to train separate linear models per coupling `type` (because the metric is averaged per type and relationships differ strongly by type) while keeping the same features and model class. I also make the one-hot encoding consistent by fitting dummies on the combined train+test categories (still no label leakage since it’s only categories), which avoids missing-category degradation. Finally, I keep the same submission format and ensure row alignment by predicting into `test_2` by type and writing `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

BASE_PATH = "/kaggle/data/champs-scalar-coupling"
train_path = os.path.join(BASE_PATH, "train.csv")
test_path = os.path.join(BASE_PATH, "test.csv")
structures_path = os.path.join(BASE_PATH, "structures.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
structures = pd.read_csv(structures_path)

mulliken_path = os.path.join(BASE_PATH, "mulliken_charges.csv")
mst_path = os.path.join(BASE_PATH, "magnetic_shielding_tensors.csv")
dipole_path = os.path.join(BASE_PATH, "dipole_moments.csv")
pe_path = os.path.join(BASE_PATH, "potential_energy.csv")

mulliken = pd.read_csv(mulliken_path) if os.path.exists(mulliken_path) else None
mst = pd.read_csv(mst_path) if os.path.exists(mst_path) else None
dipole = pd.read_csv(dipole_path) if os.path.exists(dipole_path) else None
potential_energy = pd.read_csv(pe_path) if os.path.exists(pe_path) else None



## === cell 1
structures_0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "x_0",
        "y": "y_0",
        "z": "z_0",
    }
)
structures_1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "x": "x_1",
        "y": "y_1",
        "z": "z_1",
    }
)

train_2 = train.merge(structures_0, on=["molecule_name", "atom_index_0"], how="left")
train_2 = train_2.merge(structures_1, on=["molecule_name", "atom_index_1"], how="left")

test_2 = test.merge(structures_0, on=["molecule_name", "atom_index_0"], how="left")
test_2 = test_2.merge(structures_1, on=["molecule_name", "atom_index_1"], how="left")

if mulliken is not None:
    mull_0 = mulliken.rename(
        columns={"atom_index": "atom_index_0", "mulliken_charge": "mcharge_0"}
    )
    mull_1 = mulliken.rename(
        columns={"atom_index": "atom_index_1", "mulliken_charge": "mcharge_1"}
    )
    train_2 = train_2.merge(
        mull_0[["molecule_name", "atom_index_0", "mcharge_0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    train_2 = train_2.merge(
        mull_1[["molecule_name", "atom_index_1", "mcharge_1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
    )
    test_2 = test_2.merge(
        mull_0[["molecule_name", "atom_index_0", "mcharge_0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    test_2 = test_2.merge(
        mull_1[["molecule_name", "atom_index_1", "mcharge_1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
    )
else:
    train_2["mcharge_0"] = np.nan
    train_2["mcharge_1"] = np.nan
    test_2["mcharge_0"] = np.nan
    test_2["mcharge_1"] = np.nan

if mst is not None:
    tensor_cols = ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]
    mst_0 = mst.rename(columns={"atom_index": "atom_index_0"})
    mst_1 = mst.rename(columns={"atom_index": "atom_index_1"})
    mst_0 = mst_0[["molecule_name", "atom_index_0"] + tensor_cols].copy()
    mst_1 = mst_1[["molecule_name", "atom_index_1"] + tensor_cols].copy()

    for df, suf in [(mst_0, "_0"), (mst_1, "_1")]:
        vals = df[tensor_cols].astype(np.float64)
        df["mst_mean" + suf] = vals.mean(axis=1)
        df["mst_std" + suf] = vals.std(axis=1)
        df["mst_absmean" + suf] = vals.abs().mean(axis=1)
        df["mst_trace" + suf] = (df["XX"] + df["YY"] + df["ZZ"]).astype(np.float64)
        df.drop(columns=tensor_cols, inplace=True)

    train_2 = train_2.merge(mst_0, on=["molecule_name", "atom_index_0"], how="left")
    train_2 = train_2.merge(mst_1, on=["molecule_name", "atom_index_1"], how="left")
    test_2 = test_2.merge(mst_0, on=["molecule_name", "atom_index_0"], how="left")
    test_2 = test_2.merge(mst_1, on=["molecule_name", "atom_index_1"], how="left")
else:
    for col in [
        "mst_mean_0",
        "mst_std_0",
        "mst_absmean_0",
        "mst_trace_0",
        "mst_mean_1",
        "mst_std_1",
        "mst_absmean_1",
        "mst_trace_1",
    ]:
        train_2[col] = np.nan
        test_2[col] = np.nan

if dipole is not None:
    train_2 = train_2.merge(dipole, on="molecule_name", how="left")
    test_2 = test_2.merge(dipole, on="molecule_name", how="left")
else:
    train_2["X"] = np.nan
    train_2["Y"] = np.nan
    train_2["Z"] = np.nan
    test_2["X"] = np.nan
    test_2["Y"] = np.nan
    test_2["Z"] = np.nan

if potential_energy is not None:
    train_2 = train_2.merge(potential_energy, on="molecule_name", how="left")
    test_2 = test_2.merge(potential_energy, on="molecule_name", how="left")
else:
    train_2["potential_energy"] = np.nan
    test_2["potential_energy"] = np.nan

for c in ["x_0", "y_0", "z_0", "x_1", "y_1", "z_1"]:
    train_2[c] = train_2[c].fillna(0.0)
    test_2[c] = test_2[c].fillna(0.0)

num_fill_zero = [
    "mcharge_0",
    "mcharge_1",
    "mst_mean_0",
    "mst_std_0",
    "mst_absmean_0",
    "mst_trace_0",
    "mst_mean_1",
    "mst_std_1",
    "mst_absmean_1",
    "mst_trace_1",
    "X",
    "Y",
    "Z",
    "potential_energy",
]
for c in num_fill_zero:
    if c in train_2.columns:
        train_2[c] = train_2[c].fillna(0.0).astype(np.float64)
    if c in test_2.columns:
        test_2[c] = test_2[c].fillna(0.0).astype(np.float64)

dx_tr = (train_2.x_0 - train_2.x_1).astype(np.float64)
dy_tr = (train_2.y_0 - train_2.y_1).astype(np.float64)
dz_tr = (train_2.z_0 - train_2.z_1).astype(np.float64)
dist_tr = np.sqrt(dx_tr * dx_tr + dy_tr * dy_tr + dz_tr * dz_tr)

dx_te = (test_2.x_0 - test_2.x_1).astype(np.float64)
dy_te = (test_2.y_0 - test_2.y_1).astype(np.float64)
dz_te = (test_2.z_0 - test_2.z_1).astype(np.float64)
dist_te = np.sqrt(dx_te * dx_te + dy_te * dy_te + dz_te * dz_te)

train_2["distance"] = dist_tr
test_2["distance"] = dist_te

eps = 1e-6
train_2["distance2"] = train_2["distance"] ** 2
test_2["distance2"] = test_2["distance"] ** 2
train_2["inv_distance"] = 1.0 / (train_2["distance"] + eps)
test_2["inv_distance"] = 1.0 / (test_2["distance"] + eps)

train_2["atom_pair"] = (
    train_2["atom_0"].astype(str) + "_" + train_2["atom_1"].astype(str)
)
test_2["atom_pair"] = test_2["atom_0"].astype(str) + "_" + test_2["atom_1"].astype(str)

train_2["mcharge_sum"] = train_2["mcharge_0"] + train_2["mcharge_1"]
test_2["mcharge_sum"] = test_2["mcharge_0"] + test_2["mcharge_1"]
train_2["mcharge_diff"] = train_2["mcharge_0"] - train_2["mcharge_1"]
test_2["mcharge_diff"] = test_2["mcharge_0"] - test_2["mcharge_1"]
train_2["mcharge_prod"] = train_2["mcharge_0"] * train_2["mcharge_1"]
test_2["mcharge_prod"] = test_2["mcharge_0"] * test_2["mcharge_1"]

train_2["mst_mean_diff"] = train_2["mst_mean_0"] - train_2["mst_mean_1"]
test_2["mst_mean_diff"] = test_2["mst_mean_0"] - test_2["mst_mean_1"]
train_2["mst_trace_diff"] = train_2["mst_trace_0"] - train_2["mst_trace_1"]
test_2["mst_trace_diff"] = test_2["mst_trace_0"] - test_2["mst_trace_1"]

train_2["dipole_mag"] = np.sqrt(
    train_2["X"] ** 2 + train_2["Y"] ** 2 + train_2["Z"] ** 2
)
test_2["dipole_mag"] = np.sqrt(test_2["X"] ** 2 + test_2["Y"] ** 2 + test_2["Z"] ** 2)

train_2["ux"] = dx_tr / (train_2["distance"] + eps)
train_2["uy"] = dy_tr / (train_2["distance"] + eps)
train_2["uz"] = dz_tr / (train_2["distance"] + eps)
test_2["ux"] = dx_te / (test_2["distance"] + eps)
test_2["uy"] = dy_te / (test_2["distance"] + eps)
test_2["uz"] = dz_te / (test_2["distance"] + eps)

train_2["dipole_proj"] = (
    train_2["X"] * train_2["ux"]
    + train_2["Y"] * train_2["uy"]
    + train_2["Z"] * train_2["uz"]
)
test_2["dipole_proj"] = (
    test_2["X"] * test_2["ux"] + test_2["Y"] * test_2["uy"] + test_2["Z"] * test_2["uz"]
)



## === cell 2
numeric_features = [
    "distance",
    "distance2",
    "inv_distance",
    "mcharge_0",
    "mcharge_1",
    "mcharge_sum",
    "mcharge_diff",
    "mcharge_prod",
    "mst_mean_0",
    "mst_std_0",
    "mst_absmean_0",
    "mst_trace_0",
    "mst_mean_1",
    "mst_std_1",
    "mst_absmean_1",
    "mst_trace_1",
    "mst_mean_diff",
    "mst_trace_diff",
    "dipole_mag",
    "dipole_proj",
    "potential_energy",
]

cats_type = pd.concat([train_2["type"], test_2["type"]], axis=0)
cats_atom0 = pd.concat([train_2["atom_0"], test_2["atom_0"]], axis=0)
cats_atom1 = pd.concat([train_2["atom_1"], test_2["atom_1"]], axis=0)
cats_pair = pd.concat([train_2["atom_pair"], test_2["atom_pair"]], axis=0)

type_dum_all = pd.get_dummies(cats_type, prefix="type")
atom0_dum_all = pd.get_dummies(cats_atom0, prefix="atom0")
atom1_dum_all = pd.get_dummies(cats_atom1, prefix="atom1")
pair_dum_all = pd.get_dummies(cats_pair, prefix="pair")

ntr = len(train_2)
type_tr, type_te = type_dum_all.iloc[:ntr].reset_index(drop=True), type_dum_all.iloc[
    ntr:
].reset_index(drop=True)
a0_tr, a0_te = atom0_dum_all.iloc[:ntr].reset_index(drop=True), atom0_dum_all.iloc[
    ntr:
].reset_index(drop=True)
a1_tr, a1_te = atom1_dum_all.iloc[:ntr].reset_index(drop=True), atom1_dum_all.iloc[
    ntr:
].reset_index(drop=True)
pair_tr, pair_te = pair_dum_all.iloc[:ntr].reset_index(drop=True), pair_dum_all.iloc[
    ntr:
].reset_index(drop=True)

X_train = pd.concat(
    [
        type_tr,
        a0_tr,
        a1_tr,
        pair_tr,
        train_2[numeric_features].reset_index(drop=True).astype(np.float64),
    ],
    axis=1,
)

X_test = pd.concat(
    [
        type_te,
        a0_te,
        a1_te,
        pair_te,
        test_2[numeric_features].reset_index(drop=True).astype(np.float64),
    ],
    axis=1,
)

y_train = train_2["scalar_coupling_constant"].astype(np.float64)

test_pred = np.zeros(len(test_2), dtype=np.float64)
for t in train_2["type"].unique():
    tr_idx = train_2["type"].values == t
    te_idx = test_2["type"].values == t
    if not np.any(te_idx):
        continue

    reg = LinearRegression()
    reg.fit(X_train.loc[tr_idx], y_train.loc[tr_idx])
    test_pred[te_idx] = reg.predict(X_test.loc[te_idx])

test_2["scalar_coupling_constant"] = test_pred

submission = test_2[["id", "scalar_coupling_constant"]].copy()
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Unique types predicted:", submission.shape[0], "rows")
