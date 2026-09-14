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

INPUT_DIR = "../input"

train = pd.read_csv(f"{INPUT_DIR}/train.csv")
test = pd.read_csv(f"{INPUT_DIR}/test.csv")
sub = pd.read_csv(f"{INPUT_DIR}/sample_submission.csv")
print(train.shape, test.shape, sub.shape)

train["atom1"] = train["type"].map(lambda x: str(x)[2])
train["atom2"] = train["type"].map(lambda x: str(x)[3])
test["atom1"] = test["type"].map(lambda x: str(x)[2])
test["atom2"] = test["type"].map(lambda x: str(x)[3])

for i in range(4):
    lbl_i = preprocessing.LabelEncoder()
    all_vals = pd.concat(
        [
            train["type"].map(lambda x: str(x)[i]),
            test["type"].map(lambda x: str(x)[i]),
        ],
        axis=0,
        ignore_index=True,
    )
    lbl_i.fit(all_vals)
    train["type" + str(i)] = lbl_i.transform(train["type"].map(lambda x: str(x)[i]))
    test["type" + str(i)] = lbl_i.transform(test["type"].map(lambda x: str(x)[i]))

structures_all = pd.read_csv(
    f"{INPUT_DIR}/structures.csv",
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
)

s0 = structures_all.rename(
    columns={
        "atom_index": "atom_index_0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
        "atom": "atom1",
    }
)
train = pd.merge(train, s0, how="left", on=["molecule_name", "atom_index_0", "atom1"])
test = pd.merge(test, s0, how="left", on=["molecule_name", "atom_index_0", "atom1"])

s1 = structures_all.rename(
    columns={
        "atom_index": "atom_index_1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
        "atom": "atom2",
    }
)
train = pd.merge(train, s1, how="left", on=["molecule_name", "atom_index_1", "atom2"])
test = pd.merge(test, s1, how="left", on=["molecule_name", "atom_index_1", "atom2"])
print(train.shape, test.shape, sub.shape)

atom_map0 = structures_all[["molecule_name", "atom_index", "atom"]].rename(
    columns={"atom_index": "atom_index_0", "atom": "atom0"}
)
train = pd.merge(train, atom_map0, how="left", on=["molecule_name", "atom_index_0"])
test = pd.merge(test, atom_map0, how="left", on=["molecule_name", "atom_index_0"])

atom_map1 = structures_all[["molecule_name", "atom_index", "atom"]].rename(
    columns={"atom_index": "atom_index_1", "atom": "atom1s"}
)
train = pd.merge(train, atom_map1, how="left", on=["molecule_name", "atom_index_1"])
test = pd.merge(test, atom_map1, how="left", on=["molecule_name", "atom_index_1"])

atom_lbl = preprocessing.LabelEncoder()
all_atoms = pd.concat(
    [train["atom0"], train["atom1s"], test["atom0"], test["atom1s"]],
    axis=0,
    ignore_index=True,
).astype(str)
atom_lbl.fit(all_atoms)
train["atom0_enc"] = atom_lbl.transform(train["atom0"].astype(str))
train["atom1s_enc"] = atom_lbl.transform(train["atom1s"].astype(str))
test["atom0_enc"] = atom_lbl.transform(test["atom0"].astype(str))
test["atom1s_enc"] = atom_lbl.transform(test["atom1s"].astype(str))

mulliken = pd.read_csv(
    f"{INPUT_DIR}/mulliken_charges.csv",
    usecols=["molecule_name", "atom_index", "mulliken_charge"],
)
m0 = mulliken.rename(
    columns={"atom_index": "atom_index_0", "mulliken_charge": "mulliken_0"}
)
m1 = mulliken.rename(
    columns={"atom_index": "atom_index_1", "mulliken_charge": "mulliken_1"}
)
train = pd.merge(train, m0, how="left", on=["molecule_name", "atom_index_0"])
test = pd.merge(test, m0, how="left", on=["molecule_name", "atom_index_0"])
train = pd.merge(train, m1, how="left", on=["molecule_name", "atom_index_1"])
test = pd.merge(test, m1, how="left", on=["molecule_name", "atom_index_1"])
del mulliken, m0, m1

mst = pd.read_csv(
    f"{INPUT_DIR}/magnetic_shielding_tensors.csv",
    usecols=["molecule_name", "atom_index", "XX", "YY", "ZZ"],
)
mst0 = mst.rename(
    columns={
        "atom_index": "atom_index_0",
        "XX": "mst_xx_0",
        "YY": "mst_yy_0",
        "ZZ": "mst_zz_0",
    }
)
mst1 = mst.rename(
    columns={
        "atom_index": "atom_index_1",
        "XX": "mst_xx_1",
        "YY": "mst_yy_1",
        "ZZ": "mst_zz_1",
    }
)
train = pd.merge(train, mst0, how="left", on=["molecule_name", "atom_index_0"])
test = pd.merge(test, mst0, how="left", on=["molecule_name", "atom_index_0"])
train = pd.merge(train, mst1, how="left", on=["molecule_name", "atom_index_1"])
test = pd.merge(test, mst1, how="left", on=["molecule_name", "atom_index_1"])
del mst, mst0, mst1

dipole = pd.read_csv(
    f"{INPUT_DIR}/dipole_moments.csv", usecols=["molecule_name", "X", "Y", "Z"]
).rename(columns={"X": "dipole_x", "Y": "dipole_y", "Z": "dipole_z"})
pe = pd.read_csv(
    f"{INPUT_DIR}/potential_energy.csv", usecols=["molecule_name", "potential_energy"]
)
train = pd.merge(train, dipole, how="left", on="molecule_name")
test = pd.merge(test, dipole, how="left", on="molecule_name")
train = pd.merge(train, pe, how="left", on="molecule_name")
test = pd.merge(test, pe, how="left", on="molecule_name")
del dipole, pe


def _build_graph_dist_features(structures_df, pairs_df, max_molecules=9000):
    pairs_keys = pairs_df[["molecule_name", "atom_index_0", "atom_index_1"]].copy()
    pairs_keys["a"] = np.minimum(
        pairs_keys["atom_index_0"], pairs_keys["atom_index_1"]
    ).astype(np.int32)
    pairs_keys["b"] = np.maximum(
        pairs_keys["atom_index_0"], pairs_keys["atom_index_1"]
    ).astype(np.int32)
    pairs_keys = pairs_keys[["molecule_name", "a", "b"]].drop_duplicates()

    mols = pairs_keys["molecule_name"].drop_duplicates().sort_values()
    if len(mols) > max_molecules:
        mols = mols.iloc[:max_molecules]
    mols_set = set(mols.values.tolist())

    s = structures_df[structures_df["molecule_name"].isin(mols_set)].copy()
    s["atom_index"] = s["atom_index"].astype(np.int32)
    s[["x", "y", "z"]] = s[["x", "y", "z"]].astype(np.float32)

    rad = {
        "H": 0.31,
        "C": 0.76,
        "N": 0.71,
        "O": 0.66,
        "F": 0.57,
        "P": 1.07,
        "S": 1.05,
        "Cl": 1.02,
        "Br": 1.20,
        "I": 1.39,
    }
    s["r"] = s["atom"].map(rad).fillna(0.77).astype(np.float32)

    out = []
    for mol, g in s.groupby("molecule_name", sort=False):
        ai = g["atom_index"].to_numpy()
        xyz = g[["x", "y", "z"]].to_numpy(dtype=np.float32)
        r = g["r"].to_numpy(dtype=np.float32)

        n = len(ai)
        if n <= 1:
            continue

        diff = xyz[:, None, :] - xyz[None, :, :]
        d2 = (diff * diff).sum(axis=2)
        d = np.sqrt(d2, dtype=np.float32)

        thresh = (r[:, None] + r[None, :]) + np.float32(0.45)
        adj = (d > 0) & (d < thresh)

        neigh = [np.where(adj[i])[0].tolist() for i in range(n)]
        idx_pos = {int(a): int(i) for i, a in enumerate(ai.tolist())}

        pk = pairs_keys[pairs_keys["molecule_name"] == mol]
        if pk.empty:
            continue

        for a, b in pk[["a", "b"]].itertuples(index=False, name=None):
            a = int(a)
            b = int(b)
            if a not in idx_pos or b not in idx_pos:
                out.append((mol, a, b, np.nan))
                continue
            ia = idx_pos[a]
            ib = idx_pos[b]
            if ia == ib:
                out.append((mol, a, b, 0.0))
                continue

            visited = np.zeros(n, dtype=np.uint8)
            visited[ia] = 1
            frontier = [ia]
            dist = 0
            found = False
            while frontier and dist <= 8:  # cap at 8 bonds (enough for this dataset)
                dist += 1
                nxt = []
                for u in frontier:
                    for v in neigh[u]:
                        if not visited[v]:
                            if v == ib:
                                found = True
                                nxt = []
                                break
                            visited[v] = 1
                            nxt.append(v)
                    if found:
                        break
                if found:
                    out.append((mol, a, b, float(dist)))
                    break
                frontier = nxt
            if not found:
                out.append((mol, a, b, np.nan))

    if not out:
        return pd.DataFrame(columns=["molecule_name", "a", "b", "graph_dist"])

    df_out = pd.DataFrame(out, columns=["molecule_name", "a", "b", "graph_dist"])
    return df_out.drop_duplicates(subset=["molecule_name", "a", "b"], keep="last")


pairs_all = pd.concat(
    [
        train[["molecule_name", "atom_index_0", "atom_index_1"]],
        test[["molecule_name", "atom_index_0", "atom_index_1"]],
    ],
    axis=0,
    ignore_index=True,
)
graph_df = _build_graph_dist_features(structures_all, pairs_all, max_molecules=9000)

for df in (train, test):
    df["a"] = np.minimum(df["atom_index_0"], df["atom_index_1"]).astype(np.int32)
    df["b"] = np.maximum(df["atom_index_0"], df["atom_index_1"]).astype(np.int32)

train = train.merge(
    graph_df,
    how="left",
    left_on=["molecule_name", "a", "b"],
    right_on=["molecule_name", "a", "b"],
)
test = test.merge(
    graph_df,
    how="left",
    left_on=["molecule_name", "a", "b"],
    right_on=["molecule_name", "a", "b"],
)
train.drop(columns=["a", "b"], inplace=True)
test.drop(columns=["a", "b"], inplace=True)
del graph_df, pairs_all

gd_mean_by_type = train.groupby("type")["graph_dist"].mean()
gd_global_mean = (
    float(train["graph_dist"].mean()) if train["graph_dist"].notna().any() else 2.0
)
train_gd_mean = train["type"].map(gd_mean_by_type).fillna(gd_global_mean)
test_gd_mean = test["type"].map(gd_mean_by_type).fillna(gd_global_mean)
train["graph_dist_to_type_mean"] = train["graph_dist"] / (train_gd_mean + 1e-12)
test["graph_dist_to_type_mean"] = test["graph_dist"] / (test_gd_mean + 1e-12)



## === cell 1
train["dx"] = train["x0"] - train["x1"]
train["dy"] = train["y0"] - train["y1"]
train["dz"] = train["z0"] - train["z1"]
test["dx"] = test["x0"] - test["x1"]
test["dy"] = test["y0"] - test["y1"]
test["dz"] = test["z0"] - test["z1"]

train["dist2"] = (
    train["dx"] * train["dx"] + train["dy"] * train["dy"] + train["dz"] * train["dz"]
)
test["dist2"] = (
    test["dx"] * test["dx"] + test["dy"] * test["dy"] + test["dz"] * test["dz"]
)

train["dist"] = np.sqrt(train["dist2"])
test["dist"] = np.sqrt(test["dist2"])

train_type_mean = train.groupby("type")["dist"].mean()
global_mean = train["dist"].mean()
eps = 1e-12

train_mean_for_row = train["type"].map(train_type_mean).fillna(global_mean)
test_mean_for_row = test["type"].map(train_type_mean).fillna(global_mean)

train["dist_to_type_mean"] = train["dist"] / (train_mean_for_row + eps)
test["dist_to_type_mean"] = test["dist"] / (test_mean_for_row + eps)

train["inv_dist"] = 1.0 / (train["dist"] + eps)
test["inv_dist"] = 1.0 / (test["dist"] + eps)

train["J_order"] = (
    train["type"]
    .map(lambda s: int(str(s)[0]) if str(s)[0].isdigit() else -1)
    .astype(np.int16)
)
test["J_order"] = (
    test["type"]
    .map(lambda s: int(str(s)[0]) if str(s)[0].isdigit() else -1)
    .astype(np.int16)
)

r0_norm_tr = np.sqrt(train["x0"] ** 2 + train["y0"] ** 2 + train["z0"] ** 2) + eps
r1_norm_tr = np.sqrt(train["x1"] ** 2 + train["y1"] ** 2 + train["z1"] ** 2) + eps
r0_norm_te = np.sqrt(test["x0"] ** 2 + test["y0"] ** 2 + test["z0"] ** 2) + eps
r1_norm_te = np.sqrt(test["x1"] ** 2 + test["y1"] ** 2 + test["z1"] ** 2) + eps

train["cos_r0_r1"] = (
    train["x0"] * train["x1"] + train["y0"] * train["y1"] + train["z0"] * train["z1"]
) / (r0_norm_tr * r1_norm_tr)
test["cos_r0_r1"] = (
    test["x0"] * test["x1"] + test["y0"] * test["y1"] + test["z0"] * test["z1"]
) / (r0_norm_te * r1_norm_te)

train["dipole_bond_dot"] = (
    train["dipole_x"] * train["dx"]
    + train["dipole_y"] * train["dy"]
    + train["dipole_z"] * train["dz"]
)
test["dipole_bond_dot"] = (
    test["dipole_x"] * test["dx"]
    + test["dipole_y"] * test["dy"]
    + test["dipole_z"] * test["dz"]
)

train.replace([np.inf, -np.inf], np.nan, inplace=True)
test.replace([np.inf, -np.inf], np.nan, inplace=True)



## === cell 2
col = [
    c
    for c in train.columns
    if c
    not in [
        "id",
        "molecule_name",
        "scalar_coupling_constant",
        "type",
        "atom1",
        "atom2",
        "atom0",
        "atom1s",
    ]
]


def make_regressor():
    return ensemble.ExtraTreesRegressor(n_jobs=-1, n_estimators=60, random_state=4)


MAX_ROWS = 2_000_000

y_all = train["scalar_coupling_constant"].values
valid_mask = np.isfinite(y_all)
train_valid = train.loc[valid_mask].copy()

if len(train_valid) > MAX_ROWS:
    type_counts = train_valid["type"].value_counts()
    min_per_type = 2000
    alloc = (type_counts / type_counts.sum() * MAX_ROWS).astype(int)
    alloc = np.maximum(alloc, min_per_type)

    if alloc.sum() != MAX_ROWS:
        if alloc.sum() > MAX_ROWS:
            scale = MAX_ROWS / alloc.sum()
            alloc = np.maximum((alloc * scale).astype(int), 1)

        diff = int(MAX_ROWS - alloc.sum())
        if diff != 0:
            order = alloc.sort_values(ascending=False).index.tolist()
            i = 0
            step = 1 if diff > 0 else -1
            diff_abs = abs(diff)
            while diff_abs > 0 and i < 10_000_000:
                t = order[i % len(order)]
                new_val = int(alloc.loc[t] + step)
                if new_val >= 1:
                    alloc.loc[t] = new_val
                    diff_abs -= 1
                i += 1

    parts = []
    for t, n in alloc.items():
        df_t = train_valid[train_valid["type"] == t]
        if len(df_t) <= n:
            parts.append(df_t)
        else:
            parts.append(df_t.sample(n=int(n), random_state=99))
    train_sub = pd.concat(parts, axis=0, ignore_index=False)
else:
    train_sub = train_valid

X_global = train_sub[col]
global_num_cols = X_global.columns[
    X_global.dtypes.apply(lambda dt: np.issubdtype(dt, np.number))
]
global_fill_values = X_global[global_num_cols].median()

test_X_full = test[col].copy()
pred_test_full = np.zeros(len(test), dtype=np.float64)
cv_mae_by_type = {}

types = sorted(train_sub["type"].unique().tolist())
print("Training per-type models for", len(types), "types:", types)

for t in types:
    df_t = train_sub[train_sub["type"] == t].copy()
    if df_t.empty:
        continue

    mols = df_t["molecule_name"].unique()
    if len(mols) < 5:
        is_valid = np.zeros(len(df_t), dtype=bool)
    else:
        mols_train, mols_valid = model_selection.train_test_split(
            mols, test_size=0.2, random_state=99
        )
        is_valid = df_t["molecule_name"].isin(mols_valid).values

    X_train = df_t.loc[~is_valid, col].copy()
    y_train = df_t.loc[~is_valid, "scalar_coupling_constant"].copy()

    if len(X_train) == 0:
        X_train = df_t[col].copy()
        y_train = df_t["scalar_coupling_constant"].copy()
        X_valid = None
        y_valid = None
    else:
        X_valid = df_t.loc[is_valid, col].copy()
        y_valid = df_t.loc[is_valid, "scalar_coupling_constant"].copy()

    num_cols_t = X_train.columns[
        X_train.dtypes.apply(lambda dt: np.issubdtype(dt, np.number))
    ]
    fill_values_t = X_train[num_cols_t].median()
    fill_values_t = fill_values_t.fillna(global_fill_values.reindex(num_cols_t))
    fill_values_t = fill_values_t.fillna(0.0)

    X_train[num_cols_t] = X_train[num_cols_t].fillna(fill_values_t)
    X_train[num_cols_t] = X_train[num_cols_t].fillna(0.0)

    if X_valid is not None and len(X_valid) > 0:
        X_valid[num_cols_t] = X_valid[num_cols_t].fillna(fill_values_t)
        X_valid[num_cols_t] = X_valid[num_cols_t].fillna(0.0)

    y_train_vals = y_train.values.astype(np.float64)
    y_train_t = np.sign(y_train_vals) * np.log1p(np.abs(y_train_vals))

    reg = make_regressor()
    reg.fit(X_train, y_train_t)

    bias = 0.0
    if X_valid is not None and len(X_valid) > 0:
        pred_valid_t = reg.predict(X_valid).astype(np.float64)
        pred_valid = np.sign(pred_valid_t) * np.expm1(np.abs(pred_valid_t))
        resid = y_valid.values.astype(np.float64) - pred_valid.astype(np.float64)
        bias = (
            float(np.median(resid[np.isfinite(resid)]))
            if np.any(np.isfinite(resid))
            else 0.0
        )

        pred_valid_bc = pred_valid + bias
        mae_t = float(np.mean(np.abs(y_valid.values - pred_valid_bc)))
        cv_mae_by_type[t] = mae_t

    test_mask = (test["type"] == t).values
    if np.any(test_mask):
        X_test_t = test_X_full.loc[test_mask, :].copy()
        X_test_t[num_cols_t] = X_test_t[num_cols_t].fillna(fill_values_t)
        X_test_t[num_cols_t] = X_test_t[num_cols_t].fillna(0.0)

        pred_t_t = reg.predict(X_test_t).astype(np.float64)
        pred_t = np.sign(pred_t_t) * np.expm1(np.abs(pred_t_t))
        pred_t = pred_t + bias

        pred_t = np.where(np.isfinite(pred_t), pred_t, np.nan)
        pred_t = np.nan_to_num(pred_t, nan=np.nanmedian(pred_t))
        pred_test_full[test_mask] = pred_t

if len(cv_mae_by_type) > 0:
    cv_metric = float(np.mean(np.log(np.array(list(cv_mae_by_type.values())) + 1e-12)))
    print("Approx CV metric (mean log MAE over types with valid split):", cv_metric)

test_pred_df = pd.DataFrame(
    {"id": test["id"].values, "scalar_coupling_constant": pred_test_full}
)
submission = sub[["id"]].merge(test_pred_df, on="id", how="left")

if submission["scalar_coupling_constant"].isna().any():
    fallback = float(np.nanmedian(submission["scalar_coupling_constant"].values))
    submission["scalar_coupling_constant"] = submission[
        "scalar_coupling_constant"
    ].fillna(fallback)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
