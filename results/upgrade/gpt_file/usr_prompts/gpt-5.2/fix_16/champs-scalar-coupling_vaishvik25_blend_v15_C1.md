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
import numpy as np
import pandas as pd

RANDOM_STATE = 42

DATA_DIR_CANDIDATES = [
    "/kaggle/data/champs-scalar-coupling",
    "/kaggle/input/champs-scalar-coupling",
    "/kaggle/data",
    "/kaggle/input",
]


def find_file(filename: str) -> str:
    for d in DATA_DIR_CANDIDATES:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Could not find {filename} in any of: {DATA_DIR_CANDIDATES}"
    )


train_path = find_file("train.csv")
test_path = find_file("test.csv")
structures_path = find_file("structures.csv")
sample_sub_path = find_file("sample_submission.csv")

mulliken_path = find_file("mulliken_charges.csv")
magshield_path = find_file("magnetic_shielding_tensors.csv")
dipole_path = find_file("dipole_moments.csv")
potential_path = find_file("potential_energy.csv")

scc_contrib_path = find_file("scalar_coupling_contributions.csv")

print("Using paths:")
print("train:", train_path)
print("test:", test_path)
print("structures:", structures_path)
print("sample_submission:", sample_sub_path)
print("mulliken:", mulliken_path)
print("magnetic_shielding_tensors:", magshield_path)
print("dipole_moments:", dipole_path)
print("potential_energy:", potential_path)
print("scalar_coupling_contributions:", scc_contrib_path)



## === cell 1
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

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

mulliken = pd.read_csv(
    mulliken_path,
    usecols=["molecule_name", "atom_index", "mulliken_charge"],
    dtype={
        "molecule_name": "string",
        "atom_index": np.int16,
        "mulliken_charge": np.float64,
    },
)
mag_cols = ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]
magshield = pd.read_csv(
    magshield_path,
    usecols=["molecule_name", "atom_index"] + mag_cols,
    dtype={
        "molecule_name": "string",
        "atom_index": np.int16,
        **{c: np.float64 for c in mag_cols},
    },
)
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
potential = pd.read_csv(
    potential_path,
    usecols=["molecule_name", "potential_energy"],
    dtype={"molecule_name": "string", "potential_energy": np.float64},
)

scc = pd.read_csv(
    scc_contrib_path,
    usecols=[
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "fc",
        "sd",
        "pso",
        "dso",
    ],
    dtype={
        "molecule_name": "string",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "string",
        "fc": np.float64,
        "sd": np.float64,
        "pso": np.float64,
        "dso": np.float64,
    },
)

print(train.shape, test.shape, structures.shape)
print(
    "aux shapes:",
    mulliken.shape,
    magshield.shape,
    dipole.shape,
    potential.shape,
    scc.shape,
)



## === cell 2
mol_centroid = (
    structures.groupby("molecule_name", sort=False)[["x", "y", "z"]]
    .mean()
    .rename(columns={"x": "cx", "y": "cy", "z": "cz"})
)

ATOM_Z = {
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


def _neighbor_block(mask: np.ndarray, dist: np.ndarray, atoms: np.ndarray, suffix: str):
    degree = mask.sum(axis=1).astype(np.int32)
    dist_masked = np.where(mask, dist, np.nan)
    neigh_mean = np.nanmean(dist_masked, axis=1)
    neigh_min = np.nanmin(dist_masked, axis=1)

    is_H = atoms == "H"
    is_C = atoms == "C"
    is_N = atoms == "N"
    is_O = atoms == "O"
    is_other = ~(is_H | is_C | is_N | is_O)

    neigh_H = (mask & is_H[None, :]).sum(axis=1).astype(np.int16)
    neigh_C = (mask & is_C[None, :]).sum(axis=1).astype(np.int16)
    neigh_N = (mask & is_N[None, :]).sum(axis=1).astype(np.int16)
    neigh_O = (mask & is_O[None, :]).sum(axis=1).astype(np.int16)
    neigh_other = (mask & is_other[None, :]).sum(axis=1).astype(np.int16)

    return {
        f"degree{suffix}": degree,
        f"neigh_dist_mean{suffix}": neigh_mean,
        f"neigh_dist_min{suffix}": neigh_min,
        f"neigh_H{suffix}": neigh_H,
        f"neigh_C{suffix}": neigh_C,
        f"neigh_N{suffix}": neigh_N,
        f"neigh_O{suffix}": neigh_O,
        f"neigh_other{suffix}": neigh_other,
    }


def build_atom_neighbor_features_both(
    structures_df: pd.DataFrame, cutoff1: float = 2.0, cutoff2: float = 4.0
) -> pd.DataFrame:
    coords = structures_df[
        ["molecule_name", "atom_index", "atom", "x", "y", "z"]
    ].copy()
    coords["atom"] = coords["atom"].astype(str)

    suf1 = f"_{cutoff1:.1f}".replace(".", "p")
    suf2 = f"_{cutoff2:.1f}".replace(".", "p")

    rows = []
    eye_cache = {}

    for mol, g in coords.groupby("molecule_name", sort=False):
        g = g.sort_values("atom_index")
        idx = g["atom_index"].to_numpy()
        xyz = g[["x", "y", "z"]].to_numpy(dtype=np.float64, copy=False)
        atoms = g["atom"].to_numpy()

        n = len(g)
        if n == 0:
            continue

        eye = eye_cache.get(n)
        if eye is None:
            eye = np.eye(n, dtype=bool)
            eye_cache[n] = eye

        diff = xyz[:, None, :] - xyz[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        dist = np.sqrt(d2, dtype=np.float64)

        m1 = (dist <= cutoff1) & (~eye)
        m2 = (dist <= cutoff2) & (~eye)

        d = {"molecule_name": mol, "atom_index": idx}
        d.update(_neighbor_block(m1, dist, atoms, suf1))
        d.update(_neighbor_block(m2, dist, atoms, suf2))

        rows.append(pd.DataFrame(d))

    if not rows:
        cols = ["molecule_name", "atom_index"]
        for suf in (suf1, suf2):
            cols += [
                f"degree{suf}",
                f"neigh_dist_mean{suf}",
                f"neigh_dist_min{suf}",
                f"neigh_H{suf}",
                f"neigh_C{suf}",
                f"neigh_N{suf}",
                f"neigh_O{suf}",
                f"neigh_other{suf}",
            ]
        return pd.DataFrame(columns=cols)

    return pd.concat(rows, axis=0, ignore_index=True)


atom_env = build_atom_neighbor_features_both(structures, cutoff1=2.0, cutoff2=4.0)
print("Atom env features (both cutoffs):", atom_env.shape)



## === cell 3
mulliken_feat = mulliken.copy()

magshield_feat = magshield.copy()
magshield_feat["ms_trace"] = (
    magshield_feat["XX"] + magshield_feat["YY"] + magshield_feat["ZZ"]
)
magshield_feat["ms_frob"] = np.sqrt(
    (magshield_feat[mag_cols].to_numpy(dtype=np.float64, copy=False) ** 2).sum(axis=1)
)

dipole_feat = dipole.copy()
dipole_feat["dipole_norm"] = np.sqrt(
    (dipole_feat[["X", "Y", "Z"]].to_numpy(dtype=np.float64, copy=False) ** 2).sum(
        axis=1
    )
)

potential_feat = potential.copy()

print(
    "Prepared aux feats:",
    mulliken_feat.shape,
    magshield_feat.shape,
    dipole_feat.shape,
    potential_feat.shape,
)



## === cell 4
s0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
    }
)[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]]

s1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
    }
)[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]]

env0 = atom_env.rename(columns={"atom_index": "atom_index_0"}).copy()
env1 = atom_env.rename(columns={"atom_index": "atom_index_1"}).copy()
env1 = env1.rename(
    columns={
        c: (c + "_1")
        for c in env1.columns
        if c not in ["molecule_name", "atom_index_1"]
    }
)

m0 = mulliken_feat.rename(
    columns={"atom_index": "atom_index_0", "mulliken_charge": "mulliken_0"}
)[["molecule_name", "atom_index_0", "mulliken_0"]]
m1 = mulliken_feat.rename(
    columns={"atom_index": "atom_index_1", "mulliken_charge": "mulliken_1"}
)[["molecule_name", "atom_index_1", "mulliken_1"]]

ms0 = magshield_feat.rename(columns={"atom_index": "atom_index_0"})[
    ["molecule_name", "atom_index_0"] + mag_cols + ["ms_trace", "ms_frob"]
]
ms1 = magshield_feat.rename(columns={"atom_index": "atom_index_1"})[
    ["molecule_name", "atom_index_1"] + mag_cols + ["ms_trace", "ms_frob"]
].copy()
ms1 = ms1.rename(
    columns={
        c: (c + "_1") for c in ms1.columns if c not in ["molecule_name", "atom_index_1"]
    }
)

dipole_m = dipole_feat.rename(
    columns={"X": "dipole_X", "Y": "dipole_Y", "Z": "dipole_Z"}
)[["molecule_name", "dipole_X", "dipole_Y", "dipole_Z", "dipole_norm"]]
potential_m = potential_feat[["molecule_name", "potential_energy"]].copy()

assert not s0.duplicated(["molecule_name", "atom_index_0"]).any()
assert not s1.duplicated(["molecule_name", "atom_index_1"]).any()
assert not env0.duplicated(["molecule_name", "atom_index_0"]).any()
assert not env1.duplicated(["molecule_name", "atom_index_1"]).any()
assert not m0.duplicated(["molecule_name", "atom_index_0"]).any()
assert not m1.duplicated(["molecule_name", "atom_index_1"]).any()
assert not ms0.duplicated(["molecule_name", "atom_index_0"]).any()
assert not ms1.duplicated(["molecule_name", "atom_index_1"]).any()


def add_pair_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out = out.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
    out = out.merge(s1, on=["molecule_name", "atom_index_1"], how="left")

    out = out.merge(mol_centroid.reset_index(), on="molecule_name", how="left")

    out = out.merge(env0, on=["molecule_name", "atom_index_0"], how="left")
    out = out.merge(env1, on=["molecule_name", "atom_index_1"], how="left")

    out = out.merge(m0, on=["molecule_name", "atom_index_0"], how="left")
    out = out.merge(m1, on=["molecule_name", "atom_index_1"], how="left")

    out = out.merge(ms0, on=["molecule_name", "atom_index_0"], how="left")
    out = out.merge(ms1, on=["molecule_name", "atom_index_1"], how="left")

    out = out.merge(dipole_m, on="molecule_name", how="left")
    out = out.merge(potential_m, on="molecule_name", how="left")

    dx = out["x0"] - out["x1"]
    dy = out["y0"] - out["y1"]
    dz = out["z0"] - out["z1"]

    out["dx"] = dx
    out["dy"] = dy
    out["dz"] = dz

    out["abs_dx"] = dx.abs()
    out["abs_dy"] = dy.abs()
    out["abs_dz"] = dz.abs()

    out["dist2"] = dx * dx + dy * dy + dz * dz
    out["dist"] = np.sqrt(out["dist2"])

    out["x0c"] = out["x0"] - out["cx"]
    out["y0c"] = out["y0"] - out["cy"]
    out["z0c"] = out["z0"] - out["cz"]
    out["x1c"] = out["x1"] - out["cx"]
    out["y1c"] = out["y1"] - out["cy"]
    out["z1c"] = out["z1"] - out["cz"]

    out["dxc"] = out["x0c"] - out["x1c"]
    out["dyc"] = out["y0c"] - out["y1c"]
    out["dzc"] = out["z0c"] - out["z1c"]

    eps = 1e-12
    out["r0c2"] = (
        out["x0c"] * out["x0c"] + out["y0c"] * out["y0c"] + out["z0c"] * out["z0c"]
    )
    out["r1c2"] = (
        out["x1c"] * out["x1c"] + out["y1c"] * out["y1c"] + out["z1c"] * out["z1c"]
    )
    out["r0c"] = np.sqrt(out["r0c2"] + eps)
    out["r1c"] = np.sqrt(out["r1c2"] + eps)
    out["r_sum"] = out["r0c"] + out["r1c"]
    out["r_absdiff"] = (out["r0c"] - out["r1c"]).abs()
    out["cos_c0c1"] = (
        out["x0c"] * out["x1c"] + out["y0c"] * out["y1c"] + out["z0c"] * out["z1c"]
    ) / (out["r0c"] * out["r1c"] + eps)

    out["inv_dist"] = 1.0 / (out["dist"] + eps)
    out["inv_dist2"] = 1.0 / (out["dist2"] + eps)
    out["inv_dist3"] = out["inv_dist"] * out["inv_dist2"]

    out["mx"] = 0.5 * (out["x0"] + out["x1"])
    out["my"] = 0.5 * (out["y0"] + out["y1"])
    out["mz"] = 0.5 * (out["z0"] + out["z1"])
    out["mxc"] = out["mx"] - out["cx"]
    out["myc"] = out["my"] - out["cy"]
    out["mzc"] = out["mz"] - out["cz"]
    out["mr2"] = (
        out["mxc"] * out["mxc"] + out["myc"] * out["myc"] + out["mzc"] * out["mzc"]
    )
    out["mr"] = np.sqrt(out["mr2"] + eps)

    out["ux"] = out["dx"] / (out["dist"] + eps)
    out["uy"] = out["dy"] / (out["dist"] + eps)
    out["uz"] = out["dz"] / (out["dist"] + eps)
    out["abs_ux"] = out["ux"].abs()
    out["abs_uy"] = out["uy"].abs()
    out["abs_uz"] = out["uz"].abs()

    az0 = out["atom_0"].map(ATOM_Z).astype("float64")
    az1 = out["atom_1"].map(ATOM_Z).astype("float64")
    out["az0"] = az0
    out["az1"] = az1
    out["az_sum"] = az0 + az1
    out["az_diff"] = (az0 - az1).abs()
    out["az_prod"] = az0 * az1
    out["az_prod_inv_dist"] = out["az_prod"] * out["inv_dist"]
    out["az_sum_inv_dist2"] = out["az_sum"] * out["inv_dist2"]
    out["az_prod_inv_dist3"] = out["az_prod"] * out["inv_dist3"]

    out["mulliken_sum"] = out["mulliken_0"] + out["mulliken_1"]
    out["mulliken_absdiff"] = (out["mulliken_0"] - out["mulliken_1"]).abs()
    out["mulliken_prod"] = out["mulliken_0"] * out["mulliken_1"]

    out["mulliken_sum_inv_dist"] = out["mulliken_sum"] * out["inv_dist"]
    out["mulliken_absdiff_inv_dist"] = out["mulliken_absdiff"] * out["inv_dist"]
    out["dipole_norm_inv_dist"] = out["dipole_norm"] * out["inv_dist"]
    out["potential_energy_inv_dist"] = out["potential_energy"] * out["inv_dist"]

    return out


train_f = add_pair_features(train)
test_f = add_pair_features(test)

train_f = train_f.merge(
    scc,
    on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
    how="left",
    validate="one_to_one",
)

miss = train_f[["fc", "sd", "pso", "dso"]].isna().mean()
print("Contribution missing rate in train:", miss.to_dict())

needed_cols = [
    "type",
    "atom_0",
    "atom_1",
    "dx",
    "dy",
    "dz",
    "dist",
    "inv_dist",
    "mr",
    "ux",
    "az0",
    "az1",
    "mulliken_0",
    "mulliken_1",
    "potential_energy",
    "dipole_X",
]
print("Train missing counts (selected):")
print(train_f[needed_cols].isna().sum())
print("Test missing counts (selected):")
print(test_f[needed_cols].isna().sum())



## === cell 5
env_base_2 = [
    "degree_2p0",
    "neigh_dist_mean_2p0",
    "neigh_dist_min_2p0",
    "neigh_H_2p0",
    "neigh_C_2p0",
    "neigh_N_2p0",
    "neigh_O_2p0",
    "neigh_other_2p0",
]
env_base_4 = [
    "degree_4p0",
    "neigh_dist_mean_4p0",
    "neigh_dist_min_4p0",
    "neigh_H_4p0",
    "neigh_C_4p0",
    "neigh_N_4p0",
    "neigh_O_4p0",
    "neigh_other_4p0",
]
env_base = env_base_2 + env_base_4

env_0 = env_base
env_1 = [c + "_1" for c in env_base]

for base in env_base:
    a = train_f[base]
    b = train_f[base + "_1"]
    train_f[base + "_sum"] = a + b
    train_f[base + "_absdiff"] = (a - b).abs()

    a = test_f[base]
    b = test_f[base + "_1"]
    test_f[base + "_sum"] = a + b
    test_f[base + "_absdiff"] = (a - b).abs()

env_sym = [b + "_sum" for b in env_base] + [b + "_absdiff" for b in env_base]

ms_base = mag_cols + ["ms_trace", "ms_frob"]
ms_0 = ms_base
ms_1 = [c + "_1" for c in ms_base]

for base in ms_base:
    train_f[base + "_sum"] = train_f[base] + train_f[base + "_1"]
    train_f[base + "_absdiff"] = (train_f[base] - train_f[base + "_1"]).abs()
    train_f[base + "_prod"] = train_f[base] * train_f[base + "_1"]

    test_f[base + "_sum"] = test_f[base] + test_f[base + "_1"]
    test_f[base + "_absdiff"] = (test_f[base] - test_f[base + "_1"]).abs()
    test_f[base + "_prod"] = test_f[base] * test_f[base + "_1"]

ms_sym = (
    [b + "_sum" for b in ms_base]
    + [b + "_absdiff" for b in ms_base]
    + [b + "_prod" for b in ms_base]
)

feature_cols_num = (
    [
        "dx",
        "dy",
        "dz",
        "abs_dx",
        "abs_dy",
        "abs_dz",
        "dist",
        "dist2",
        "dxc",
        "dyc",
        "dzc",
        "inv_dist",
        "inv_dist2",
        "inv_dist3",
        "mr",
        "mr2",
        "ux",
        "uy",
        "uz",
        "abs_ux",
        "abs_uy",
        "abs_uz",
        "az0",
        "az1",
        "az_sum",
        "az_diff",
        "az_prod",
        "az_prod_inv_dist",
        "az_sum_inv_dist2",
        "az_prod_inv_dist3",
        "mulliken_0",
        "mulliken_1",
        "mulliken_sum",
        "mulliken_absdiff",
        "mulliken_prod",
        "mulliken_sum_inv_dist",
        "mulliken_absdiff_inv_dist",
        "dipole_X",
        "dipole_Y",
        "dipole_Z",
        "dipole_norm",
        "dipole_norm_inv_dist",
        "potential_energy",
        "potential_energy_inv_dist",
        "r0c",
        "r1c",
        "r_sum",
        "r_absdiff",
        "cos_c0c1",
    ]
    + ms_0
    + ms_1
    + ms_sym
    + env_0
    + env_1
    + env_sym
)

feature_cols_cat = ["atom_0", "atom_1"]

missing_num = [c for c in feature_cols_num if c not in train_f.columns]
missing_cat = [c for c in feature_cols_cat if c not in train_f.columns]
if missing_num or missing_cat:
    raise KeyError(
        f"Missing feature cols. num missing={missing_num[:10]} cat missing={missing_cat}"
    )

numeric_tf = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ]
)

categorical_tf = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_tf, feature_cols_num),
        ("cat", categorical_tf, feature_cols_cat),
    ],
    remainder="drop",
)

model = Ridge(alpha=3.0)
pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])




## === cell 6
def _to_object_with_nan(df: pd.DataFrame, cols):
    out = df.copy()
    for c in cols:
        s = out[c]
        out[c] = s.astype("object").where(~pd.isna(s), np.nan)
    return out


X_tr_all = train_f.loc[:, feature_cols_num + feature_cols_cat]
X_te_all = test_f.loc[:, feature_cols_num + feature_cols_cat]
X_tr_all = _to_object_with_nan(X_tr_all, feature_cols_cat)
X_te_all = _to_object_with_nan(X_te_all, feature_cols_cat)

targets = ["fc", "sd", "pso", "dso"]
preds_comp = {k: np.zeros(len(test_f), dtype=np.float64) for k in targets}

types = sorted(train_f["type"].astype(str).unique().tolist())
print("Training per-type models (contrib targets):", types)

valid_targets_mask = ~train_f[targets].isna().any(axis=1).to_numpy()

train_type_arr = train_f["type"].astype(str).to_numpy()
test_type_arr = test_f["type"].astype(str).to_numpy()

for t in types:
    tr_mask = train_type_arr == t
    te_mask = test_type_arr == t

    if te_mask.sum() == 0:
        continue

    tr_mask_valid = tr_mask & valid_targets_mask

    X_tr = X_tr_all.loc[tr_mask_valid, :]
    X_te = X_te_all.loc[te_mask, :]

    if X_tr.shape[0] == 0:
        continue

    for k in targets:
        y_tr = train_f.loc[tr_mask_valid, k].to_numpy(dtype=np.float64, copy=False)
        pipe.fit(X_tr, y_tr)
        preds_comp[k][te_mask] = pipe.predict(X_te).astype(np.float64, copy=False)

preds = preds_comp["fc"] + preds_comp["sd"] + preds_comp["pso"] + preds_comp["dso"]
print("Pred summary:", pd.Series(preds).describe())



## === cell 7
sub = pd.read_csv(sample_sub_path, usecols=["id"], dtype={"id": np.int32})
pred_df = pd.DataFrame(
    {
        "id": test_f["id"].to_numpy(dtype=np.int32, copy=False),
        "scalar_coupling_constant": preds,
    }
)
sub = sub.merge(pred_df, on="id", how="left", validate="one_to_one")
sub["scalar_coupling_constant"] = (
    sub["scalar_coupling_constant"].fillna(0.0).astype(np.float64)
)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(sub.head())
print(sub.shape)
print("Submission dtypes:", sub.dtypes.to_dict())
