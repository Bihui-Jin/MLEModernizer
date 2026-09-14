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
import os

DATA_ROOT = "/kaggle/data/champs-scalar-coupling"
ALT_DATA_ROOT = "/kaggle/input/champs-scalar-coupling"

if not os.path.exists(DATA_ROOT) and os.path.exists(ALT_DATA_ROOT):
    DATA_ROOT = ALT_DATA_ROOT

print("DATA_ROOT =", DATA_ROOT)
print("Files in DATA_ROOT (sample):", sorted(os.listdir(DATA_ROOT))[:10])



## === cell 1
import numpy as np  # linear algebra
import pandas as pd  # data processing
import os
import gc

print("Files in /kaggle/data (sample):", sorted(os.listdir("/kaggle/data"))[:20])

np.random.seed(2020)



## === cell 2
train_path = os.path.join(DATA_ROOT, "train.csv")
test_path = os.path.join(DATA_ROOT, "test.csv")
structures_path = os.path.join(DATA_ROOT, "structures.csv")
mulliken_path = os.path.join(DATA_ROOT, "mulliken_charges.csv")
shielding_path = os.path.join(DATA_ROOT, "magnetic_shielding_tensors.csv")
dipole_path = os.path.join(DATA_ROOT, "dipole_moments.csv")
energy_path = os.path.join(DATA_ROOT, "potential_energy.csv")

train = pd.read_csv(
    train_path,
    usecols=[
        "id",
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "scalar_coupling_constant",
    ],
    dtype={
        "id": np.int32,
        "molecule_name": "string",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "string",
        "scalar_coupling_constant": np.float64,
    },
)
test = pd.read_csv(
    test_path,
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
    dtype={
        "id": np.int32,
        "molecule_name": "string",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "string",
    },
)

all_mol_cats = pd.Index(
    pd.concat(
        [train["molecule_name"], test["molecule_name"]], ignore_index=True
    ).unique()
)
all_mol_cats = all_mol_cats.sort_values()

for df in (train, test):
    df["molecule_name"] = pd.Categorical(df["molecule_name"], categories=all_mol_cats)
    df["type"] = df["type"].astype("category")

structures = pd.read_csv(
    structures_path,
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
    dtype={
        "molecule_name": "string",
        "atom_index": np.int16,
        "atom": "string",
        "x": np.float64,
        "y": np.float64,
        "z": np.float64,
    },
)
structures = structures.drop_duplicates(
    subset=["molecule_name", "atom_index"], keep="first"
)
structures["molecule_name"] = pd.Categorical(
    structures["molecule_name"], categories=all_mol_cats
)
structures["atom"] = structures["atom"].astype("category")

mulliken = pd.read_csv(
    mulliken_path,
    usecols=["molecule_name", "atom_index", "mulliken_charge"],
    dtype={
        "molecule_name": "string",
        "atom_index": np.int16,
        "mulliken_charge": np.float64,
    },
)
mulliken = mulliken.drop_duplicates(
    subset=["molecule_name", "atom_index"], keep="first"
)
mulliken["molecule_name"] = pd.Categorical(
    mulliken["molecule_name"], categories=all_mol_cats
)

shield = pd.read_csv(
    shielding_path,
    usecols=[
        "molecule_name",
        "atom_index",
        "XX",
        "YX",
        "ZX",
        "XY",
        "YY",
        "ZY",
        "XZ",
        "YZ",
        "ZZ",
    ],
    dtype={
        "molecule_name": "string",
        "atom_index": np.int16,
        "XX": np.float64,
        "YX": np.float64,
        "ZX": np.float64,
        "XY": np.float64,
        "YY": np.float64,
        "ZY": np.float64,
        "XZ": np.float64,
        "YZ": np.float64,
        "ZZ": np.float64,
    },
)
shield = shield.drop_duplicates(subset=["molecule_name", "atom_index"], keep="first")
shield["molecule_name"] = pd.Categorical(
    shield["molecule_name"], categories=all_mol_cats
)

sh_arr = shield[["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]].to_numpy(
    dtype=np.float64, copy=False
)
shield_trace = (
    shield["XX"].to_numpy(dtype=np.float64, copy=False)
    + shield["YY"].to_numpy(dtype=np.float64, copy=False)
    + shield["ZZ"].to_numpy(dtype=np.float64, copy=False)
)
shield_frob = np.sqrt((sh_arr * sh_arr).sum(axis=1))
shield = shield[["molecule_name", "atom_index"]].copy()
shield["shield_trace"] = shield_trace.astype(np.float64, copy=False)
shield["shield_frob"] = shield_frob.astype(np.float64, copy=False)

dipole = pd.read_csv(
    dipole_path,
    usecols=["molecule_name", "X", "Y", "Z"],
    dtype={
        "molecule_name": "string",
        "X": np.float64,
        "Y": np.float64,
        "Z": np.float64,
    },
)
dipole["molecule_name"] = pd.Categorical(
    dipole["molecule_name"], categories=all_mol_cats
)
d_arr = dipole[["X", "Y", "Z"]].to_numpy(dtype=np.float64, copy=False)
dipole_norm = np.sqrt((d_arr * d_arr).sum(axis=1))
dipole = dipole.rename(columns={"X": "dipole_x", "Y": "dipole_y", "Z": "dipole_z"})
dipole["dipole_norm"] = dipole_norm.astype(np.float64, copy=False)
dipole = dipole[["molecule_name", "dipole_x", "dipole_y", "dipole_z", "dipole_norm"]]

energy = pd.read_csv(
    energy_path,
    usecols=["molecule_name", "potential_energy"],
    dtype={"molecule_name": "string", "potential_energy": np.float64},
).drop_duplicates(subset=["molecule_name"], keep="first")
energy["molecule_name"] = pd.Categorical(
    energy["molecule_name"], categories=all_mol_cats
)

mol_agg = (
    structures.groupby("molecule_name", sort=False, observed=True)
    .agg(
        mol_x_mean=("x", "mean"),
        mol_y_mean=("y", "mean"),
        mol_z_mean=("z", "mean"),
        mol_n_atoms=("atom_index", "count"),
    )
    .reset_index()
)

_ATOMIC_NUM = {
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


def _build_atom_lookup(df_atoms: pd.DataFrame, cols, mol_categories: pd.Index):
    mol_codes = pd.Categorical(
        df_atoms["molecule_name"], categories=mol_categories
    ).codes.astype(np.int32, copy=False)
    atom_idx = df_atoms["atom_index"].to_numpy(np.int32, copy=False)
    key = (mol_codes.astype(np.int64) << 16) + atom_idx.astype(np.int64)
    order = np.argsort(key, kind="mergesort")
    key_sorted = key[order]
    vals = df_atoms.loc[df_atoms.index[order], cols].to_numpy(copy=False)
    return key_sorted, vals


_struct_cols = ["atom", "x", "y", "z"]
_struct_key, _struct_vals = _build_atom_lookup(structures, _struct_cols, all_mol_cats)
_mull_key, _mull_vals = _build_atom_lookup(mulliken, ["mulliken_charge"], all_mol_cats)
_sh_key, _sh_vals = _build_atom_lookup(
    shield, ["shield_trace", "shield_frob"], all_mol_cats
)

_mol_categories = all_mol_cats  # shared categories used everywhere

_dip_mol_codes = pd.Categorical(
    dipole["molecule_name"], categories=_mol_categories
).codes.astype(np.int32, copy=False)
_en_mol_codes = pd.Categorical(
    energy["molecule_name"], categories=_mol_categories
).codes.astype(np.int32, copy=False)
_agg_mol_codes = pd.Categorical(
    mol_agg["molecule_name"], categories=_mol_categories
).codes.astype(np.int32, copy=False)


def _dense_mol_array(codes, vals, n_mols, fill_val=np.nan):
    out = np.full((n_mols, vals.shape[1]), fill_val, dtype=np.float64)
    m = codes >= 0
    out[codes[m]] = vals[m]
    return out


n_mols = len(_mol_categories)
dip_vals = dipole[["dipole_x", "dipole_y", "dipole_z", "dipole_norm"]].to_numpy(
    np.float64, copy=False
)
ene_vals = energy[["potential_energy"]].to_numpy(np.float64, copy=False)
agg_vals = mol_agg[["mol_x_mean", "mol_y_mean", "mol_z_mean", "mol_n_atoms"]].to_numpy(
    np.float64, copy=False
)

_dip_dense = _dense_mol_array(_dip_mol_codes, dip_vals, n_mols, fill_val=np.nan)
_ene_dense = _dense_mol_array(_en_mol_codes, ene_vals, n_mols, fill_val=np.nan)
_agg_dense = _dense_mol_array(_agg_mol_codes, agg_vals, n_mols, fill_val=np.nan)


def _lookup_sorted(
    key_sorted: np.ndarray,
    vals_sorted: np.ndarray,
    query_key: np.ndarray,
    out_shape_last: int,
):
    pos = np.searchsorted(key_sorted, query_key)
    hit = pos < len(key_sorted)
    hit = hit & (key_sorted[pos.clip(0, len(key_sorted) - 1)] == query_key)
    out = np.full((len(query_key), out_shape_last), np.nan, dtype=vals_sorted.dtype)
    if hit.any():
        out[hit] = vals_sorted[pos[hit]]
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    mol_codes = pd.Categorical(
        df["molecule_name"], categories=_mol_categories
    ).codes.astype(np.int32, copy=False)
    a0 = df["atom_index_0"].to_numpy(np.int32, copy=False)
    a1 = df["atom_index_1"].to_numpy(np.int32, copy=False)
    k0 = (mol_codes.astype(np.int64) << 16) + a0.astype(np.int64)
    k1 = (mol_codes.astype(np.int64) << 16) + a1.astype(np.int64)

    v0 = _lookup_sorted(_struct_key, _struct_vals, k0, 4)
    v1 = _lookup_sorted(_struct_key, _struct_vals, k1, 4)
    df["atom_0"] = pd.Categorical(
        v0[:, 0], categories=structures["atom"].cat.categories
    )
    df["x0"] = v0[:, 1].astype(np.float64, copy=False)
    df["y0"] = v0[:, 2].astype(np.float64, copy=False)
    df["z0"] = v0[:, 3].astype(np.float64, copy=False)
    df["atom_1"] = pd.Categorical(
        v1[:, 0], categories=structures["atom"].cat.categories
    )
    df["x1"] = v1[:, 1].astype(np.float64, copy=False)
    df["y1"] = v1[:, 2].astype(np.float64, copy=False)
    df["z1"] = v1[:, 3].astype(np.float64, copy=False)

    agg = _agg_dense[mol_codes]
    df["mol_x_mean"] = agg[:, 0]
    df["mol_y_mean"] = agg[:, 1]
    df["mol_z_mean"] = agg[:, 2]
    df["mol_n_atoms"] = agg[:, 3].astype(np.float64, copy=False)

    dip = _dip_dense[mol_codes]
    df["dipole_x"] = dip[:, 0]
    df["dipole_y"] = dip[:, 1]
    df["dipole_z"] = dip[:, 2]
    df["dipole_norm"] = dip[:, 3]

    ene = _ene_dense[mol_codes]
    df["potential_energy"] = ene[:, 0]

    q0 = _lookup_sorted(_mull_key, _mull_vals, k0, 1)[:, 0]
    q1 = _lookup_sorted(_mull_key, _mull_vals, k1, 1)[:, 0]
    df["q0"] = q0.astype(np.float64, copy=False)
    df["q1"] = q1.astype(np.float64, copy=False)

    s0 = _lookup_sorted(_sh_key, _sh_vals, k0, 2)
    s1 = _lookup_sorted(_sh_key, _sh_vals, k1, 2)
    df["sh_trace0"] = s0[:, 0].astype(np.float64, copy=False)
    df["sh_frob0"] = s0[:, 1].astype(np.float64, copy=False)
    df["sh_trace1"] = s1[:, 0].astype(np.float64, copy=False)
    df["sh_frob1"] = s1[:, 1].astype(np.float64, copy=False)

    df["x0"] = pd.Series(df["x0"]).fillna(df["mol_x_mean"]).fillna(0.0).to_numpy()
    df["y0"] = pd.Series(df["y0"]).fillna(df["mol_y_mean"]).fillna(0.0).to_numpy()
    df["z0"] = pd.Series(df["z0"]).fillna(df["mol_z_mean"]).fillna(0.0).to_numpy()
    df["x1"] = pd.Series(df["x1"]).fillna(df["mol_x_mean"]).fillna(0.0).to_numpy()
    df["y1"] = pd.Series(df["y1"]).fillna(df["mol_y_mean"]).fillna(0.0).to_numpy()
    df["z1"] = pd.Series(df["z1"]).fillna(df["mol_z_mean"]).fillna(0.0).to_numpy()

    x0 = df["x0"].to_numpy(dtype=np.float64, copy=False)
    y0 = df["y0"].to_numpy(dtype=np.float64, copy=False)
    z0 = df["z0"].to_numpy(dtype=np.float64, copy=False)
    x1 = df["x1"].to_numpy(dtype=np.float64, copy=False)
    y1 = df["y1"].to_numpy(dtype=np.float64, copy=False)
    z1 = df["z1"].to_numpy(dtype=np.float64, copy=False)
    mx = df["mol_x_mean"].to_numpy(dtype=np.float64, copy=False)
    my = df["mol_y_mean"].to_numpy(dtype=np.float64, copy=False)
    mz = df["mol_z_mean"].to_numpy(dtype=np.float64, copy=False)

    dx = x0 - x1
    dy = y0 - y1
    dz = z0 - z1
    df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float64, copy=False)

    cdx0 = x0 - mx
    cdy0 = y0 - my
    cdz0 = z0 - mz
    cdx1 = x1 - mx
    cdy1 = y1 - my
    cdz1 = z1 - mz
    df["r0"] = np.sqrt(cdx0 * cdx0 + cdy0 * cdy0 + cdz0 * cdz0).astype(
        np.float64, copy=False
    )
    df["r1"] = np.sqrt(cdx1 * cdx1 + cdy1 * cdy1 + cdz1 * cdz1).astype(
        np.float64, copy=False
    )

    atom0_str = df["atom_0"].astype("string")
    atom1_str = df["atom_1"].astype("string")

    swap = (atom0_str.fillna("") > atom1_str.fillna("")).to_numpy()
    if swap.any():
        for a, b in [
            ("atom_index_0", "atom_index_1"),
            ("atom_0", "atom_1"),
            ("x0", "x1"),
            ("y0", "y1"),
            ("z0", "z1"),
            ("r0", "r1"),
            ("q0", "q1"),
            ("sh_trace0", "sh_trace1"),
            ("sh_frob0", "sh_frob1"),
        ]:
            tmp = df.loc[swap, a].copy()
            df.loc[swap, a] = df.loc[swap, b].to_numpy()
            df.loc[swap, b] = tmp.to_numpy()
        atom0_str = df["atom_0"].astype("string")
        atom1_str = df["atom_1"].astype("string")

    df["pair"] = atom0_str.fillna("X") + "-" + atom1_str.fillna("X")

    z0s = atom0_str.map(_ATOMIC_NUM).astype("float64").fillna(0.0).astype(np.int16)
    z1s = atom1_str.map(_ATOMIC_NUM).astype("float64").fillna(0.0).astype(np.int16)
    df["pair_z"] = (z0s.astype(np.int32) * 100 + z1s.astype(np.int32)).astype(np.int32)

    df["dq"] = (df["q0"] - df["q1"]).astype(np.float64)
    df["q_abs_sum"] = (
        (pd.Series(df["q0"]).abs() + pd.Series(df["q1"]).abs())
        .astype(np.float64)
        .to_numpy()
    )

    df["d_sh_trace"] = (df["sh_trace0"] - df["sh_trace1"]).astype(np.float64)
    df["d_sh_frob"] = (df["sh_frob0"] - df["sh_frob1"]).astype(np.float64)

    return df


train_f = add_features(train)
test_f = add_features(test)


def _bin_series(s: pd.Series, width: float, max_abs_bin: int) -> pd.Series:
    x = s.to_numpy(dtype=np.float64, copy=False)
    b = np.floor(x / width)
    b = np.clip(b, -max_abs_bin, max_abs_bin)
    b = np.nan_to_num(b, nan=0.0).astype(np.int32)
    return pd.Series(b, index=s.index)


def _add_type_quantile_dist_bin(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_bins: int = 50,
    col_name: str = "dist_bin_q",
):
    train_bins = np.empty(len(train_df), dtype=np.int16)
    test_bins = np.empty(len(test_df), dtype=np.int16)

    for t in train_df["type"].cat.categories:
        tr_mask = (train_df["type"] == t).to_numpy()
        te_mask = (test_df["type"] == t).to_numpy()

        tr_dist = train_df.loc[tr_mask, "dist"].to_numpy(dtype=np.float64, copy=False)
        te_dist = test_df.loc[te_mask, "dist"].to_numpy(dtype=np.float64, copy=False)

        tr_med = np.nanmedian(tr_dist) if len(tr_dist) else 0.0
        tr_dist = np.nan_to_num(tr_dist, nan=tr_med)
        te_dist = np.nan_to_num(te_dist, nan=tr_med)

        if len(tr_dist) < max(1000, n_bins * 10):
            width = 0.1
            train_bins[tr_mask] = np.clip(np.floor(tr_dist / width), 0, 300).astype(
                np.int16
            )
            test_bins[te_mask] = np.clip(np.floor(te_dist / width), 0, 300).astype(
                np.int16
            )
            continue

        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(tr_dist, qs)
        edges = np.unique(edges)
        if len(edges) <= 2:
            train_bins[tr_mask] = 0
            test_bins[te_mask] = 0
            continue

        train_bins[tr_mask] = np.digitize(tr_dist, edges[1:-1], right=True).astype(
            np.int16
        )
        test_bins[te_mask] = np.digitize(te_dist, edges[1:-1], right=True).astype(
            np.int16
        )

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[col_name] = train_bins.astype(np.int16, copy=False)
    test_df[col_name] = test_bins.astype(np.int16, copy=False)
    return train_df, test_df


train_f, test_f = _add_type_quantile_dist_bin(
    train_f, test_f, n_bins=50, col_name="dist_bin_q"
)
train_f, test_f = _add_type_quantile_dist_bin(
    train_f, test_f, n_bins=20, col_name="dist_bin_q20"
)
train_f, test_f = _add_type_quantile_dist_bin(
    train_f, test_f, n_bins=10, col_name="dist_bin_q10"
)

train_f["mol_n_atoms_bin"] = (train_f["mol_n_atoms"] // 5).fillna(-1).astype(np.int32)
test_f["mol_n_atoms_bin"] = (test_f["mol_n_atoms"] // 5).fillna(-1).astype(np.int32)

train_f["dq_bin"] = _bin_series(train_f["dq"].fillna(0.0), width=0.02, max_abs_bin=250)
test_f["dq_bin"] = _bin_series(test_f["dq"].fillna(0.0), width=0.02, max_abs_bin=250)

train_f["d_sh_trace_bin"] = _bin_series(
    train_f["d_sh_trace"].fillna(0.0), width=1.0, max_abs_bin=300
)
test_f["d_sh_trace_bin"] = _bin_series(
    test_f["d_sh_trace"].fillna(0.0), width=1.0, max_abs_bin=300
)

train_f["dipole_norm_bin"] = _bin_series(
    train_f["dipole_norm"].fillna(0.0), width=0.2, max_abs_bin=200
)
test_f["dipole_norm_bin"] = _bin_series(
    test_f["dipole_norm"].fillna(0.0), width=0.2, max_abs_bin=200
)

train_f["energy_bin"] = _bin_series(
    train_f["potential_energy"].fillna(0.0), width=50.0, max_abs_bin=300
)
test_f["energy_bin"] = _bin_series(
    test_f["potential_energy"].fillna(0.0), width=50.0, max_abs_bin=300
)

global_mean = float(train_f["scalar_coupling_constant"].mean())

grp_cols = [
    "type",
    "dist_bin_q",
    "dist_bin_q20",
    "dist_bin_q10",
    "pair",
    "pair_z",
    "mol_n_atoms_bin",
    "dq_bin",
    "d_sh_trace_bin",
    "dipole_norm_bin",
    "energy_bin",
]

print("train_f shape:", train_f.shape, "test_f shape:", test_f.shape)

del train, test, structures, mulliken, shield, dipole, energy, mol_agg
gc.collect()




## === cell 3
def _make_stratified_molecule_folds(
    train_df: pd.DataFrame, n_folds: int, seed: int = 2020
):
    mol_type_counts = (
        train_df.groupby(["molecule_name", "type"], sort=False, observed=True)
        .size()
        .unstack(fill_value=0)
    )
    mols = mol_type_counts.index.values
    X = mol_type_counts.values.astype(np.int64)

    rng = np.random.RandomState(seed)
    order = np.argsort(-X.sum(axis=1))
    order = order[rng.permutation(len(order))]

    fold_sums = np.zeros((n_folds, X.shape[1]), dtype=np.int64)
    folds = [[] for _ in range(n_folds)]

    for idx in order:
        x = X[idx]
        best_fold = None
        best_score = None
        for f in range(n_folds):
            new = fold_sums.copy()
            new[f] += x
            target = new.sum(axis=0) / n_folds
            score = ((new - target) ** 2).sum()
            if best_score is None or score < best_score:
                best_score = score
                best_fold = f
        fold_sums[best_fold] += x
        folds[best_fold].append(mols[idx])

    return [np.array(f, dtype=object) for f in folds]


def _prepare_oof_arrays(train_df: pd.DataFrame, n_folds: int = 5, seed: int = 2020):
    folds = _make_stratified_molecule_folds(train_df, n_folds=n_folds, seed=seed)
    mol_arr = train_df["molecule_name"].to_numpy()

    caches = []
    for fold_mols in folds:
        is_val = np.isin(mol_arr, fold_mols)

        tr = train_df.loc[~is_val, ["type", "scalar_coupling_constant"] + grp_cols[1:]]
        va = train_df.loc[is_val, ["type"] + grp_cols[1:]]

        type_stats_tr = (
            tr.groupby("type", sort=False, observed=True)["scalar_coupling_constant"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "type_mean", "count": "type_count"})
        )

        grp_stats_tr = (
            tr.groupby(grp_cols, sort=False, observed=True)["scalar_coupling_constant"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "grp_mean", "count": "grp_count"})
            .reset_index()
        )

        va_stats = va.merge(type_stats_tr, how="left", left_on="type", right_index=True)
        va_stats = va_stats.merge(grp_stats_tr, how="left", on=grp_cols)

        type_mean = va_stats["type_mean"].to_numpy(np.float64, copy=False)
        grp_mean = va_stats["grp_mean"].to_numpy(np.float64, copy=False)
        grp_count = va_stats["grp_count"].to_numpy(np.float64, copy=False)
        caches.append((is_val, type_mean, grp_mean, grp_count))
        del tr, va, type_stats_tr, grp_stats_tr, va_stats
        gc.collect()

    return caches


def _make_oof_pred_from_arrays(
    oof_arrays, k_grp: float, min_grp_count: float
) -> np.ndarray:
    n = len(oof_arrays[0][0])
    oof = np.empty(n, dtype=np.float64)

    for is_val, type_mean, grp_mean, grp_count in oof_arrays:
        tm = np.where(np.isnan(type_mean), global_mean, type_mean)
        gm = np.where(np.isnan(grp_mean), tm, grp_mean)
        gm = np.where(np.isnan(gm), global_mean, gm)
        gcnt = np.where(np.isnan(grp_count), 0.0, grp_count)

        w_grp = gcnt / (gcnt + float(k_grp))
        w_grp = np.where(gcnt >= float(min_grp_count), w_grp, 0.0)

        pred = w_grp * gm + (1.0 - w_grp) * tm
        oof[is_val] = pred

    return oof


def _mae_by_type_log(y_true: np.ndarray, y_pred: np.ndarray, types: pd.Series) -> float:
    t = types.to_numpy()
    abs_err = np.abs(y_true - y_pred)
    df = pd.DataFrame({"type": t, "e": abs_err})
    mae_by_type = df.groupby("type", sort=False, observed=True)["e"].mean().to_numpy()
    return float(np.mean(np.log(np.clip(mae_by_type, 1e-9, None))))


y = train_f["scalar_coupling_constant"].astype(np.float64).to_numpy(copy=False)
types = train_f["type"]

candidates_k = [25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0]
candidates_min = [1.0, 3.0, 5.0, 10.0, 20.0, 50.0]

oof_arrays = _prepare_oof_arrays(train_f, n_folds=5, seed=2020)

best = None
best_params = None

for k_grp in candidates_k:
    for min_cnt in candidates_min:
        oof_pred = _make_oof_pred_from_arrays(
            oof_arrays, k_grp=k_grp, min_grp_count=min_cnt
        )
        score = _mae_by_type_log(y, oof_pred, types)
        if (best is None) or (score < best):
            best = score
            best_params = (k_grp, min_cnt)

k_grp, min_grp_count = best_params
print(
    "Selected smoothing params from stratified molecule-OOF: k_grp =",
    k_grp,
    "min_grp_count =",
    min_grp_count,
    "OOF logMAE =",
    best,
)

type_stats = (
    train_f.groupby("type", sort=False, observed=True)["scalar_coupling_constant"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "type_mean", "count": "type_count"})
)

grp_stats = (
    train_f.groupby(grp_cols, sort=False, observed=True)["scalar_coupling_constant"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "grp_mean", "count": "grp_count"})
    .reset_index()
)

test_stats = test_f.merge(type_stats, how="left", left_on="type", right_index=True)
test_stats = test_stats.merge(grp_stats, how="left", on=grp_cols)

pred1 = test_stats["type_mean"].fillna(global_mean).astype(np.float64)

grp_count = test_stats["grp_count"].fillna(0.0).astype(np.float64)
grp_mean = (
    test_stats["grp_mean"]
    .fillna(test_stats["type_mean"])
    .fillna(global_mean)
    .astype(np.float64)
)
type_mean = test_stats["type_mean"].fillna(global_mean).astype(np.float64)

w_grp = grp_count / (grp_count + float(k_grp))
w_grp = np.where(grp_count >= float(min_grp_count), w_grp, 0.0).astype(np.float64)

pred2 = (w_grp * grp_mean + (1.0 - w_grp) * type_mean).astype(np.float64)

sub1 = pd.DataFrame(
    {"id": test_stats["id"].values, "scalar_coupling_constant": pred1.values}
)
sub2 = pd.DataFrame(
    {"id": test_stats["id"].values, "scalar_coupling_constant": pred2.values}
)

print("pred1 describe:\n", sub1["scalar_coupling_constant"].describe())
print("pred2 describe:\n", sub2["scalar_coupling_constant"].describe())



## === cell 4
(sub1["scalar_coupling_constant"] - sub2["scalar_coupling_constant"]).abs().mean()



## === cell 5
sub1["scalar_coupling_constant"] = (
    0.35 * sub1["scalar_coupling_constant"] + 0.65 * sub2["scalar_coupling_constant"]
)

sub1 = sub1[["id", "scalar_coupling_constant"]].sort_values("id").reset_index(drop=True)

sub1["scalar_coupling_constant"] = (
    sub1["scalar_coupling_constant"]
    .astype(np.float64)
    .replace([np.inf, -np.inf], np.nan)
    .fillna(global_mean)
)

sample_path = os.path.join(DATA_ROOT, "sample_submission.csv")
sample = pd.read_csv(sample_path, usecols=["id"], dtype={"id": np.int32})
if len(sample) == len(sub1):
    if not np.array_equal(sample["id"].values, sub1["id"].values):
        sub1 = sample.merge(sub1, on="id", how="left", validate="1:1")
        sub1["scalar_coupling_constant"] = (
            sub1["scalar_coupling_constant"]
            .astype(np.float64)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(global_mean)
        )

out_path = "weighted-avg-blend-lgb-keras-1.csv"
sub1.to_csv(out_path, index=False)

print("Wrote:", out_path)
print("submission shape:", sub1.shape)
print("id monotonic:", sub1["id"].is_monotonic_increasing)
print(sub1["scalar_coupling_constant"].describe())
print("Columns:", list(sub1.columns))
assert out_path.endswith(".csv")
assert list(sub1.columns) == ["id", "scalar_coupling_constant"]
assert len(sub1) == len(sample)
assert sub1["scalar_coupling_constant"].notna().all()
assert np.isfinite(sub1["scalar_coupling_constant"].to_numpy()).all()
