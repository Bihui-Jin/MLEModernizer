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

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.21147) has done: 'I remove the dependency on missing external “blender” submission files (the current FileNotFoundError root cause) and replace it with a self-contained baseline that trains on the provided competition data and writes a valid `submission.csv`. To keep the core idea minimal and stable, the solution build a small set of standard structural features (atom types + pairwise distance) from `structures.csv`, encode categorical variables, and fit a robust regression model per coupling `type`. This run end-to-end in the given environment using only the listed packages and the provided `/kaggle/data/champs-scalar-coupling/` paths, and it always produce a correctly formatted submission file with the required columns.'
- What this solution (achieved 1.20011) has done: 'Your current model is a per-type Ridge on raw coordinates and simple pairwise distances; the biggest score win with minimal semantic change is to make the geometry features translation/rotation invariant so the model doesn’t waste capacity on arbitrary coordinate frames. I keep the same pipeline (per-type Ridge + OneHot for atom types) but replace the absolute x/y/z coordinates with invariant pair features: dx, dy, dz, abs(dx/dy/dz), dist, and dist² (still derived from the same merged structures). I also standardize numeric features (a common, low-risk improvement for Ridge) without changing the estimator family or training loop. The script still run end-to-end and write a valid `submission.csv` in the required format.'
- What this solution (achieved 1.2009) has done: 'Your current score (1.20011; lower-is-better) is far from the target (-1.32687), so we need a real accuracy lift while keeping your core approach (per-type Ridge on simple structural pair features). The smallest high-impact change is to add a few proven invariant/low-risk features that keep the same geometry basis: reciprocal distance terms (strong in coupling physics) and a simple per-molecule “centered coordinates” variant to reduce arbitrary translation effects without changing the model family. I also add `type` as a categorical feature inside each per-type model (it’s constant within a loop so it won’t change anything) only if we later remove the loop; for now we keep the loop unchanged and instead add a couple more numeric features. Finally, I slightly increase Ridge regularization (alpha) to stabilize generalization with the expanded feature set (a small calibration-like change, not a new model).'
- What this solution (achieved 1.19864) has done: 'Your current score is much worse than the (very strong) target, so we should improve accuracy with minimal, low-risk changes that keep your per-type Ridge + OHE pipeline intact. The biggest safe gain here is to add a few standard CHAMPS pair features that are still derived only from `structures.csv`: the midpoint radius to centroid (captures where in the molecule the interaction occurs) and normalized direction (unit vector) terms, plus simple interaction terms between distance and element identity via atomic numbers (physics-informed but still lightweight). I also fix a subtle but important issue: `Ridge(random_state=...)` isn’t a valid parameter for sklearn’s Ridge in many versions; removing it avoids silent incompatibilities while keeping the same estimator. Everything else (per-type loop, Ridge, preprocessing, submission writing/format) remains the same.'
- What this solution (achieved 1.26172) has done: 'Your current score is much worse than the target (lower is better), so we should improve accuracy while keeping your per-type Ridge + OHE pipeline intact. The highest-impact minimal change is to add standard local-environment features (counts and distance stats of neighboring atoms around each endpoint) derived only from `structures.csv`, which typically reduces MAE across coupling types without changing the model family. To keep this within Kaggle time/memory, the neighbor features are computed in a fully vectorized way by first building all within-molecule atom pairs once, then aggregating per atom, then joining to train/test pairs. Everything else (per-type loop, Ridge, preprocessing, submission writing/format) remains the same.'
- What this solution (achieved 1.25392) has done: 'Your score is still far worse than the (lower-is-better) target, so we should push accuracy upward with the smallest changes that keep your per-type Ridge + OHE pipeline intact. The most impactful low-risk fix here is that your neighbor-feature computation currently builds an in-memory all-pairs self-join per molecule, which is extremely heavy and can silently force the kernel into slowdowns/instability; replacing it with a per-molecule distance-matrix approach preserves the same feature semantics (degree + neighbor element counts + distance stats within a cutoff) but is far more reliable. While doing that, we also add a second, slightly larger cutoff (4.0) for the same neighbor stats (still the same feature family) to give the linear model more signal without changing model family/training loop. Finally, we include those new 4.0-cutoff features symmetrically (sum/absdiff) exactly like your 2.0 features so the rest of the code path is unchanged.'
- What this solution (achieved 1.24534) has done: 'We keep your per-type Ridge + OHE pipeline and the same core feature families, but fix a silent feature bug that is currently overwriting the original `z0/z1` coordinate columns with atomic numbers (this contaminates geometry features and hurts MAE). Then we slightly enrich the existing “physics” numeric set by adding `1/dist^3` and `z_prod/dist^3`, which are standard for coupling strength and are a minimal extension of your current `inv_dist`/`inv_dist2` terms. Finally, we make sure these changes propagate cleanly through the numeric feature list without changing the training loop or submission semantics.'
- What this solution (achieved 1.22395) has done: 'Your current score is still much worse than the target (lower is better), so we should increase accuracy while keeping your exact per-type Ridge + OHE pipeline. The smallest high-impact improvement available within your existing feature family is to add the CHAMPS-proven auxiliary tables you already have (mulliken charges, magnetic shielding tensors, and dipole moments / potential energy) as additional per-atom and per-molecule numeric features; this doesn’t change the model/training loop, it only enriches inputs. We merge these tables onto atom_0/atom_1 and molecule_name, then add symmetric sum/absdiff versions (like you already do for neighbor features) to help the linear model. All paths, estimator, preprocessing, and submission writing remain unchanged.'
- What this solution (achieved 3.91039) has done: 'Your current score is much worse than the (lower-is-better) target, so we should improve accuracy while keeping your per-type Ridge + OHE pipeline unchanged. The most leverage with minimal semantic change is to (1) add the coupling `type` as a categorical feature (and therefore remove the per-type training loop in favor of a single global Ridge), which lets the model share signal across types while still allowing type-specific offsets via one-hot. Additionally, we add a few very cheap, physically meaningful interaction features that a linear model can use (e.g., `inv_dist` times Mulliken charges / dipole norm) without changing data sources or the estimator family. Finally, we keep the exact same preprocessing (impute+scale numeric, OHE categorical) and submission writing, ensuring end-to-end execution within time.'
- What this solution (achieved 1.22972) has done: 'We need to move your score down (lower-is-better) toward the strong target, and your last change (global single Ridge across all types) clearly hurt relative to your earlier per-type Ridge approach. The minimal, high-confidence fix is to restore the per-type training loop (same Ridge + same preprocessing + same features), which matches the competition’s per-type error structure and avoids harmful cross-type parameter sharing. I keep your current enriched feature set and simply fit/predict one model per `type`, then stitch predictions back in `id` order to produce a valid `submission.csv`. This is a small code change that is very likely to improve the score back toward your prior ~1.2 range without changing the feature extraction or model family.'
- What this solution (achieved 1.32568) has done: 'Your current score (1.22972; lower-is-better) is far from the target, so we should make small, high-leverage feature tweaks without changing the per-type Ridge pipeline. The most impactful minimal improvement is to add a few additional symmetric reductions (sum/absdiff/prod) for the existing per-atom magnetic shielding tensor features, because Ridge benefits from invariants and interactions but we keep the same model family and loop. We also add a couple of very cheap geometry invariants (cosine between centered position vectors and a “radial difference” term) derived from the same coordinates, which tends to help across coupling types without changing semantics. Finally, we ensure numerical stability for divisions (eps) and keep submission alignment exactly by `id`.'

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
structures_idx = structures.set_index(["molecule_name", "atom_index"], drop=False)
atom_env_idx = atom_env.set_index(["molecule_name", "atom_index"], drop=False)
mulliken_idx = mulliken_feat.set_index(["molecule_name", "atom_index"], drop=False)
magshield_idx = magshield_feat.set_index(["molecule_name", "atom_index"], drop=False)
mol_centroid_idx = mol_centroid  # already indexed by molecule_name
dipole_idx = dipole_feat.set_index("molecule_name", drop=False)
potential_idx = potential_feat.set_index("molecule_name", drop=False)

s0 = structures_idx.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
    }
)[["atom_index_0", "atom_0", "x0", "y0", "z0"]]
s1 = structures_idx.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
    }
)[["atom_index_1", "atom_1", "x1", "y1", "z1"]]

env0 = atom_env_idx.rename(columns={"atom_index": "atom_index_0"}).drop(
    columns=["molecule_name"], errors="ignore"
)
env0 = env0.rename(columns={c: c for c in env0.columns})
env0 = env0[
    [c for c in env0.columns if c != "atom_index"]
]  # keep feature columns + atom_index_0
env0 = env0.rename(columns={"atom_index": "atom_index_0"}, errors="ignore")

env1 = atom_env_idx.rename(columns={"atom_index": "atom_index_1"}).drop(
    columns=["molecule_name"], errors="ignore"
)
env1 = env1.add_suffix("_1")
env1 = env1.rename(columns={"atom_index_1_1": "atom_index_1"})

m0 = mulliken_idx.rename(
    columns={"atom_index": "atom_index_0", "mulliken_charge": "mulliken_0"}
)[["atom_index_0", "mulliken_0"]]
m1 = mulliken_idx.rename(
    columns={"atom_index": "atom_index_1", "mulliken_charge": "mulliken_1"}
)[["atom_index_1", "mulliken_1"]]

ms0 = magshield_idx.rename(columns={"atom_index": "atom_index_0"})[
    ["atom_index_0"] + mag_cols + ["ms_trace", "ms_frob"]
]
ms1 = magshield_idx.rename(columns={"atom_index": "atom_index_1"})[
    ["atom_index_1"] + mag_cols + ["ms_trace", "ms_frob"]
].add_suffix("_1")
ms1 = ms1.rename(columns={"atom_index_1_1": "atom_index_1"})


def add_pair_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    k0 = pd.MultiIndex.from_arrays(
        [out["molecule_name"].to_numpy(), out["atom_index_0"].to_numpy()]
    )
    k1 = pd.MultiIndex.from_arrays(
        [out["molecule_name"].to_numpy(), out["atom_index_1"].to_numpy()]
    )

    out = out.join(s0, on=k0, how="left")
    out = out.join(s1, on=k1, how="left")

    out = out.join(mol_centroid_idx, on="molecule_name", how="left")
    out = out.join(
        env0.drop(columns=["atom_index"], errors="ignore"), on=k0, how="left"
    )
    out = out.join(
        env1.drop(columns=["atom_index_1"], errors="ignore"), on=k1, how="left"
    )

    out = out.join(m0, on=k0, how="left")
    out = out.join(m1, on=k1, how="left")

    out = out.join(ms0, on=k0, how="left")
    out = out.join(
        ms1.drop(columns=["atom_index_1"], errors="ignore"), on=k1, how="left"
    )

    out = out.join(
        dipole_idx.rename(columns={"X": "dipole_X", "Y": "dipole_Y", "Z": "dipole_Z"})[
            ["dipole_X", "dipole_Y", "dipole_Z", "dipole_norm"]
        ],
        on="molecule_name",
        how="left",
    )
    out = out.join(potential_idx[["potential_energy"]], on="molecule_name", how="left")

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



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2957153867.py in <cell line: 0>()
    185 
    186 
--> 187 train_f = add_pair_features(train)
    188 test_f = add_pair_features(test)
    189 

/tmp/ipykernel_11/2957153867.py in add_pair_features(df)
     71     )
     72 
---> 73     out = out.join(s0, on=k0, how="left")
     74     out = out.join(s1, on=k1, how="left")
     75 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in join(self, other, on, how, lsuffix, rsuffix, sort, validate)
  10755                     validate=validate,
  10756                 )
> 10757             return merge(
  10758                 self,
  10759                 other,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in merge(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
    168         )
    169     else:
--> 170         op = _MergeOperation(
    171             left_df,
    172             right_df,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in __init__(self, left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, indicator, validate)
    784             raise MergeError(msg)
    785 
--> 786         self.left_on, self.right_on = self._validate_left_right_on(left_on, right_on)
    787 
    788         (

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in _validate_left_right_on(self, left_on, right_on)
   1605             if self.right_index:
   1606                 if len(left_on) != self.right.index.nlevels:
-> 1607                     raise ValueError(
   1608                         "len(left_on) must equal the number "
   1609                         'of levels in the index of "right"'

ValueError: len(left_on) must equal the number of levels in the index of "right"

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



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/304562004.py in <cell line: 0>()
     25 
     26 for base in env_base:
---> 27     a = train_f[base]
     28     b = train_f[base + "_1"]
     29     train_f[base + "_sum"] = a + b

NameError: name 'train_f' is not defined

## === cell 6
X_tr_all = train_f.loc[:, feature_cols_num + feature_cols_cat]
X_te_all = test_f.loc[:, feature_cols_num + feature_cols_cat]

targets = ["fc", "sd", "pso", "dso"]
preds_comp = {k: np.zeros(len(test_f), dtype=np.float64) for k in targets}

types = sorted(train_f["type"].unique().tolist())
print("Training per-type models (contrib targets):", types)

valid_targets_mask = ~train_f[targets].isna().any(axis=1).to_numpy()

train_type_arr = train_f["type"].to_numpy()
test_type_arr = test_f["type"].to_numpy()

for t in types:
    tr_mask = train_type_arr == t
    te_mask = test_type_arr == t

    if te_mask.sum() == 0:
        continue

    tr_mask_valid = tr_mask & valid_targets_mask

    X_tr = X_tr_all.loc[tr_mask_valid, :]
    X_te = X_te_all.loc[te_mask, :]

    for k in targets:
        y_tr = train_f.loc[tr_mask_valid, k].to_numpy(dtype=np.float64, copy=False)
        pipe.fit(X_tr, y_tr)
        preds_comp[k][te_mask] = pipe.predict(X_te).astype(np.float64, copy=False)

preds = preds_comp["fc"] + preds_comp["sd"] + preds_comp["pso"] + preds_comp["dso"]
print("Pred summary:", pd.Series(preds).describe())



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3131670495.py in <cell line: 0>()
----> 1 X_tr_all = train_f.loc[:, feature_cols_num + feature_cols_cat]
      2 X_te_all = test_f.loc[:, feature_cols_num + feature_cols_cat]
      3 
      4 targets = ["fc", "sd", "pso", "dso"]
      5 preds_comp = {k: np.zeros(len(test_f), dtype=np.float64) for k in targets}

NameError: name 'train_f' is not defined

## === cell 7
sub = pd.read_csv(sample_sub_path, usecols=["id"], dtype={"id": np.int32})
pred_df = pd.DataFrame(
    {"id": test_f["id"].to_numpy(), "scalar_coupling_constant": preds}
)
sub = sub.merge(pred_df, on="id", how="left", validate="one_to_one")
sub["scalar_coupling_constant"] = sub["scalar_coupling_constant"].fillna(0.0)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(sub.head())
print(sub.shape)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2746594266.py in <cell line: 0>()
      1 sub = pd.read_csv(sample_sub_path, usecols=["id"], dtype={"id": np.int32})
      2 pred_df = pd.DataFrame(
----> 3     {"id": test_f["id"].to_numpy(), "scalar_coupling_constant": preds}
      4 )
      5 sub = sub.merge(pred_df, on="id", how="left", validate="one_to_one")

NameError: name 'test_f' is not defined
