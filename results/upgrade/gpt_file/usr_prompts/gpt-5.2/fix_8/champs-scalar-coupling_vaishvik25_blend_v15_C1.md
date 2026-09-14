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

-1.3268705351930474

# 6. Current score

1.24534

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.21147) has done: 'I remove the dependency on missing external “blender” submission files (the current FileNotFoundError root cause) and replace it with a self-contained baseline that trains on the provided competition data and writes a valid `submission.csv`. To keep the core idea minimal and stable, the solution build a small set of standard structural features (atom types + pairwise distance) from `structures.csv`, encode categorical variables, and fit a robust regression model per coupling `type`. This run end-to-end in the given environment using only the listed packages and the provided `/kaggle/data/champs-scalar-coupling/` paths, and it always produce a correctly formatted submission file with the required columns.'
- What this solution (achieved 1.20011) has done: 'Your current model is a per-type Ridge on raw coordinates and simple pairwise distances; the biggest score win with minimal semantic change is to make the geometry features translation/rotation invariant so the model doesn’t waste capacity on arbitrary coordinate frames. I keep the same pipeline (per-type Ridge + OneHot for atom types) but replace the absolute x/y/z coordinates with invariant pair features: dx, dy, dz, abs(dx/dy/dz), dist, and dist² (still derived from the same merged structures). I also standardize numeric features (a common, low-risk improvement for Ridge) without changing the estimator family or training loop. The script still run end-to-end and write a valid `submission.csv` in the required format.'
- What this solution (achieved 1.2009) has done: 'Your current score (1.20011; lower-is-better) is far from the target (-1.32687), so we need a real accuracy lift while keeping your core approach (per-type Ridge on simple structural pair features). The smallest high-impact change is to add a few proven invariant/low-risk features that keep the same geometry basis: reciprocal distance terms (strong in coupling physics) and a simple per-molecule “centered coordinates” variant to reduce arbitrary translation effects without changing the model family. I also add `type` as a categorical feature inside each per-type model (it’s constant within a loop so it won’t change anything) only if we later remove the loop; for now we keep the loop unchanged and instead add a couple more numeric features. Finally, I slightly increase Ridge regularization (alpha) to stabilize generalization with the expanded feature set (a small calibration-like change, not a new model).'
- What this solution (achieved 1.19864) has done: 'Your current score is much worse than the (very strong) target, so we should improve accuracy with minimal, low-risk changes that keep your per-type Ridge + OHE pipeline intact. The biggest safe gain here is to add a few standard CHAMPS pair features that are still derived only from `structures.csv`: the midpoint radius to centroid (captures where in the molecule the interaction occurs) and normalized direction (unit vector) terms, plus simple interaction terms between distance and element identity via atomic numbers (physics-informed but still lightweight). I also fix a subtle but important issue: `Ridge(random_state=...)` isn’t a valid parameter for sklearn’s Ridge in many versions; removing it avoids silent incompatibilities while keeping the same estimator. Everything else (per-type loop, Ridge, preprocessing, submission writing/format) remains the same.'
- What this solution (achieved 1.26172) has done: 'Your current score is much worse than the target (lower is better), so we should improve accuracy while keeping your per-type Ridge + OHE pipeline intact. The highest-impact minimal change is to add standard local-environment features (counts and distance stats of neighboring atoms around each endpoint) derived only from `structures.csv`, which typically reduces MAE across coupling types without changing the model family. To keep this within Kaggle time/memory, the neighbor features are computed in a fully vectorized way by first building all within-molecule atom pairs once, then aggregating per atom, then joining to train/test pairs. Everything else (per-type loop, Ridge, preprocessing, submission writing/format) remains the same.'
- What this solution (achieved 1.25392) has done: 'Your score is still far worse than the (lower-is-better) target, so we should push accuracy upward with the smallest changes that keep your per-type Ridge + OHE pipeline intact. The most impactful low-risk fix here is that your neighbor-feature computation currently builds an in-memory all-pairs self-join per molecule, which is extremely heavy and can silently force the kernel into slowdowns/instability; replacing it with a per-molecule distance-matrix approach preserves the same feature semantics (degree + neighbor element counts + distance stats within a cutoff) but is far more reliable. While doing that, we also add a second, slightly larger cutoff (4.0) for the same neighbor stats (still the same feature family) to give the linear model more signal without changing model family/training loop. Finally, we include those new 4.0-cutoff features symmetrically (sum/absdiff) exactly like your 2.0 features so the rest of the code path is unchanged.'
- What this solution (achieved 1.24534) has done: 'We keep your per-type Ridge + OHE pipeline and the same core feature families, but fix a silent feature bug that is currently overwriting the original `z0/z1` coordinate columns with atomic numbers (this contaminates geometry features and hurts MAE). Then we slightly enrich the existing “physics” numeric set by adding `1/dist^3` and `z_prod/dist^3`, which are standard for coupling strength and are a minimal extension of your current `inv_dist`/`inv_dist2` terms. Finally, we make sure these changes propagate cleanly through the numeric feature list without changing the training loop or submission semantics.'

# 9. Code solution

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

print("Using paths:")
print("train:", train_path)
print("test:", test_path)
print("structures:", structures_path)
print("sample_submission:", sample_sub_path)



## === cell 1
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
structures = pd.read_csv(structures_path)

print(train.shape, test.shape, structures.shape)
print(train.columns.tolist())
print(test.columns.tolist())
print(structures.columns.tolist())



## === cell 2
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

mol_centroid = (
    structures.groupby("molecule_name", sort=False)[["x", "y", "z"]]
    .mean()
    .rename(columns={"x": "cx", "y": "cy", "z": "cz"})
    .reset_index()
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


def build_atom_neighbor_features(
    structures_df: pd.DataFrame, cutoff: float = 2.0
) -> pd.DataFrame:
    """
    Same semantics as before (degree + neighbor element counts + dist mean/min within cutoff),
    but implemented per-molecule via distance matrices to avoid the enormous all-pairs merge.
    This improves stability and typically improves score because the features are computed
    reliably without memory-pressure artifacts.
    """
    coords = structures_df[
        ["molecule_name", "atom_index", "atom", "x", "y", "z"]
    ].copy()
    coords["atom"] = coords["atom"].astype(str)

    rows = []
    suf = f"_{cutoff:.1f}".replace(".", "p")

    for mol, g in coords.groupby("molecule_name", sort=False):
        g = g.sort_values("atom_index")
        idx = g["atom_index"].to_numpy()
        xyz = g[["x", "y", "z"]].to_numpy(dtype=np.float64, copy=False)
        atoms = g["atom"].to_numpy()

        n = len(g)
        if n == 0:
            continue

        diff = xyz[:, None, :] - xyz[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        dist = np.sqrt(d2, dtype=np.float64)

        mask = (dist <= cutoff) & (~np.eye(n, dtype=bool))
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

        out = pd.DataFrame(
            {
                "molecule_name": mol,
                "atom_index": idx,
                f"degree{suf}": degree,
                f"neigh_dist_mean{suf}": neigh_mean,
                f"neigh_dist_min{suf}": neigh_min,
                f"neigh_H{suf}": neigh_H,
                f"neigh_C{suf}": neigh_C,
                f"neigh_N{suf}": neigh_N,
                f"neigh_O{suf}": neigh_O,
                f"neigh_other{suf}": neigh_other,
            }
        )
        rows.append(out)

    if not rows:
        return pd.DataFrame(
            columns=[
                "molecule_name",
                "atom_index",
                f"degree{suf}",
                f"neigh_dist_mean{suf}",
                f"neigh_dist_min{suf}",
                f"neigh_H{suf}",
                f"neigh_C{suf}",
                f"neigh_N{suf}",
                f"neigh_O{suf}",
                f"neigh_other{suf}",
            ]
        )

    return pd.concat(rows, axis=0, ignore_index=True)


atom_env_2 = build_atom_neighbor_features(structures, cutoff=2.0)
atom_env_4 = build_atom_neighbor_features(structures, cutoff=4.0)
atom_env = atom_env_2.merge(
    atom_env_4, on=["molecule_name", "atom_index"], how="outer", validate="one_to_one"
)

print("Atom env features (merged 2.0 + 4.0):", atom_env.shape)
print(atom_env.head())




## === cell 3
def add_pair_features(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.merge(
            s0[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]],
            on=["molecule_name", "atom_index_0"],
            how="left",
            validate="many_to_one",
        )
        .merge(
            s1[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]],
            on=["molecule_name", "atom_index_1"],
            how="left",
            validate="many_to_one",
        )
        .merge(
            mol_centroid,
            on="molecule_name",
            how="left",
            validate="many_to_one",
        )
        .merge(
            atom_env.rename(columns={"atom_index": "atom_index_0"}),
            on=["molecule_name", "atom_index_0"],
            how="left",
            validate="many_to_one",
        )
        .merge(
            atom_env.rename(columns={"atom_index": "atom_index_1"})
            .add_suffix("_1")
            .rename(
                columns={
                    "molecule_name_1": "molecule_name",
                    "atom_index_1_1": "atom_index_1",
                }
            ),
            on=["molecule_name", "atom_index_1"],
            how="left",
            validate="many_to_one",
        )
    )

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

    return out


train_f = add_pair_features(train)
test_f = add_pair_features(test)

needed_cols = [
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
]
print("Train missing counts (selected):")
print(train_f[needed_cols].isna().sum())
print("Test missing counts (selected):")
print(test_f[needed_cols].isna().sum())



## === cell 4
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
        "inv_dist3",  # Change: include new minimal inverse-distance feature
        "mr",
        "mr2",
        "ux",
        "uy",
        "uz",
        "abs_ux",
        "abs_uy",
        "abs_uz",
        "az0",  # Change: atomic numbers now in az*
        "az1",
        "az_sum",
        "az_diff",
        "az_prod",
        "az_prod_inv_dist",
        "az_sum_inv_dist2",
        "az_prod_inv_dist3",  # Change: pair interaction at 1/r^3
    ]
    + env_0
    + env_1
    + env_sym
)
feature_cols_cat = ["atom_0", "atom_1"]

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

pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", model),
    ]
)

preds = np.zeros(len(test_f), dtype=np.float64)

types = sorted(train_f["type"].unique().tolist())
print("Num types:", len(types))

for t in types:
    tr_idx = train_f["type"].values == t
    te_idx = test_f["type"].values == t

    X_tr = train_f.loc[tr_idx, feature_cols_num + feature_cols_cat]
    y_tr = train_f.loc[tr_idx, "scalar_coupling_constant"].values
    X_te = test_f.loc[te_idx, feature_cols_num + feature_cols_cat]

    pipe.fit(X_tr, y_tr)
    preds[te_idx] = pipe.predict(X_te)

print("Pred summary:", pd.Series(preds).describe())



## === cell 5
sub = pd.read_csv(sample_sub_path)
pred_df = pd.DataFrame({"id": test_f["id"].values, "scalar_coupling_constant": preds})
sub = sub[["id"]].merge(pred_df, on="id", how="left", validate="one_to_one")

sub["scalar_coupling_constant"] = sub["scalar_coupling_constant"].fillna(0.0)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(sub.head())
print(sub.shape)
