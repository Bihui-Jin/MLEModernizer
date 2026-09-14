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

# 5. Code solution

## === cell 0
import numpy as np
import pandas as pd
from sklearn import *

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sub = pd.read_csv("../input/sample_submission.csv")
print(train.shape, test.shape, sub.shape)

train["atom"] = train["type"].map(lambda x: str(x)[3])
test["atom"] = test["type"].map(lambda x: str(x)[3])

lbl = preprocessing.LabelEncoder()
for i in range(4):
    train["type" + str(i)] = lbl.fit_transform(train["type"].map(lambda x: str(x)[i]))
    test["type" + str(i)] = lbl.transform(test["type"].map(lambda x: str(x)[i]))

structures = pd.read_csv("../input/structures.csv")

mol_agg = (
    structures.groupby("molecule_name", sort=False)
    .agg(
        n_atoms=("atom_index", "count"),
        x_mean=("x", "mean"),
        y_mean=("y", "mean"),
        z_mean=("z", "mean"),
        x_std=("x", "std"),
        y_std=("y", "std"),
        z_std=("z", "std"),
    )
    .reset_index()
)
train = pd.merge(train, mol_agg, how="left", on="molecule_name")
test = pd.merge(test, mol_agg, how="left", on="molecule_name")
del mol_agg

structures_1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "x": "x_1",
        "y": "y_1",
        "z": "z_1",
    }
)
train = pd.merge(train, structures_1, how="left", on=["molecule_name", "atom_index_1"])
test = pd.merge(test, structures_1, how="left", on=["molecule_name", "atom_index_1"])
del structures_1

structures_0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "x_0",
        "y": "y_0",
        "z": "z_0",
    }
)
train = pd.merge(train, structures_0, how="left", on=["molecule_name", "atom_index_0"])
test = pd.merge(test, structures_0, how="left", on=["molecule_name", "atom_index_0"])
del structures_0, structures

for df in (train, test):
    dx = df["x_0"] - df["x_1"]
    dy = df["y_0"] - df["y_1"]
    dz = df["z_0"] - df["z_1"]
    df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz)
    df["dist2"] = dx * dx + dy * dy + dz * dz
    df["abs_dx"] = dx.abs()
    df["abs_dy"] = dy.abs()
    df["abs_dz"] = dz.abs()

    eps = 1e-6
    df["inv_dist"] = 1.0 / (df["dist"] + eps)
    df["dist3"] = df["dist"] * df["dist2"]

    df["x0_c"] = df["x_0"] - df["x_mean"]
    df["y0_c"] = df["y_0"] - df["y_mean"]
    df["z0_c"] = df["z_0"] - df["z_mean"]
    df["x1_c"] = df["x_1"] - df["x_mean"]
    df["y1_c"] = df["y_1"] - df["y_mean"]
    df["z1_c"] = df["z_1"] - df["z_mean"]
    df["r0_c"] = np.sqrt(
        df["x0_c"] * df["x0_c"] + df["y0_c"] * df["y0_c"] + df["z0_c"] * df["z0_c"]
    )
    df["r1_c"] = np.sqrt(
        df["x1_c"] * df["x1_c"] + df["y1_c"] * df["y1_c"] + df["z1_c"] * df["z1_c"]
    )
    df["abs_dr_c"] = (df["r0_c"] - df["r1_c"]).abs()

pe = pd.read_csv("../input/potential_energy.csv")
train = pd.merge(train, pe, how="left", on=["molecule_name"])
test = pd.merge(test, pe, how="left", on=["molecule_name"])
del pe

mc0 = pd.read_csv("../input/mulliken_charges.csv").rename(
    columns={"atom_index": "atom_index_0", "mulliken_charge": "mulliken_charge_0"}
)
train = pd.merge(train, mc0, how="left", on=["molecule_name", "atom_index_0"])
test = pd.merge(test, mc0, how="left", on=["molecule_name", "atom_index_0"])
del mc0

mc1 = pd.read_csv("../input/mulliken_charges.csv").rename(
    columns={"atom_index": "atom_index_1", "mulliken_charge": "mulliken_charge_1"}
)
train = pd.merge(train, mc1, how="left", on=["molecule_name", "atom_index_1"])
test = pd.merge(test, mc1, how="left", on=["molecule_name", "atom_index_1"])
del mc1

dm = pd.read_csv("../input/dipole_moments.csv")
train = pd.merge(train, dm, how="left", on=["molecule_name"])
test = pd.merge(test, dm, how="left", on=["molecule_name"])
del dm

mst0 = pd.read_csv("../input/magnetic_shielding_tensors.csv").rename(
    columns={
        "atom_index": "atom_index_0",
        "XX": "XX_0",
        "YX": "YX_0",
        "ZX": "ZX_0",
        "XY": "XY_0",
        "YY": "YY_0",
        "ZY": "ZY_0",
        "XZ": "XZ_0",
        "YZ": "YZ_0",
        "ZZ": "ZZ_0",
    }
)
train = pd.merge(train, mst0, how="left", on=["molecule_name", "atom_index_0"])
test = pd.merge(test, mst0, how="left", on=["molecule_name", "atom_index_0"])
del mst0

mst1 = pd.read_csv("../input/magnetic_shielding_tensors.csv").rename(
    columns={
        "atom_index": "atom_index_1",
        "XX": "XX_1",
        "YX": "YX_1",
        "ZX": "ZX_1",
        "XY": "XY_1",
        "YY": "YY_1",
        "ZY": "ZY_1",
        "XZ": "XZ_1",
        "YZ": "YZ_1",
        "ZZ": "ZZ_1",
    }
)
train = pd.merge(train, mst1, how="left", on=["molecule_name", "atom_index_1"])
test = pd.merge(test, mst1, how="left", on=["molecule_name", "atom_index_1"])
del mst1

print(train.shape, test.shape, sub.shape)



## === cell 1
train.head()



## === cell 2
base_col = [
    c
    for c in train.columns
    if c not in ["id", "molecule_name", "scalar_coupling_constant", "type", "atom"]
]

base_col = [c for c in base_col if pd.api.types.is_numeric_dtype(train[c])]

test_pred = np.empty(len(test), dtype=np.float64)
test_pred[:] = np.nan

rng_state = 99
val_scores = {}

types = sorted(train["type"].unique())
print("Num types:", len(types), "Types:", types)

for t in types:
    tr_mask = train["type"].values == t
    te_mask = test["type"].values == t

    tr_t = train.loc[tr_mask]
    te_t = test.loc[te_mask]

    fill_values_t = tr_t[base_col].median(numeric_only=True)
    X_tr = tr_t[base_col].fillna(fill_values_t)
    y_tr = tr_t["scalar_coupling_constant"].values
    X_te = te_t[base_col].fillna(fill_values_t)

    reg = ensemble.ExtraTreesRegressor(n_jobs=-1, random_state=4, n_estimators=20)

    n_val_cap = 250_000
    if len(tr_t) > 10_000:
        X_small = X_tr.tail(min(n_val_cap, len(tr_t)))
        y_small = y_tr[-len(X_small) :]
        x1, x2, y1, y2 = model_selection.train_test_split(
            X_small,
            y_small,
            test_size=0.2,
            random_state=rng_state,
        )
        reg.fit(x1, y1)
        mae = metrics.mean_absolute_error(y2, reg.predict(x2))
        val_scores[t] = float(np.log(mae))
        print(f"type={t:>4s} log(MAE)={val_scores[t]:.5f} (diagnostic)")

    reg.fit(X_tr, y_tr)
    test_pred[te_mask] = reg.predict(X_te)

n_missing = int(np.isnan(test_pred).sum())
print("Missing test predictions:", n_missing)
if n_missing != 0:
    global_med = float(train["scalar_coupling_constant"].median())
    test_pred[np.isnan(test_pred)] = global_med

test_out = test[["id"]].copy()
test_out["scalar_coupling_constant"] = test_pred

test_out.to_csv("submission.csv", float_format="%.9f", index=False)
print("Wrote submission.csv with shape:", test_out.shape)
print(test_out.head())
