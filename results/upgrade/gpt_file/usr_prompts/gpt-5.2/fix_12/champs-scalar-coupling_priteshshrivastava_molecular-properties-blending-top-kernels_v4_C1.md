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

-1.5735603655829546

# 6. Current score

8.86072

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'Your current notebook fails because it references external Kaggle datasets (`../input/champs-blending-tutorial/...` and `../input/otherkernelsadded/...`) that are not present in this environment. To make it run end-to-end and still produce a valid submission, I replace that missing-blend input with an in-notebook baseline model that uses only the provided competition files. The baseline be intentionally simple and stable: predict the mean `scalar_coupling_constant` per `type` from `train.csv`, and fall back to the global mean for any unseen types; this preserves correct evaluation semantics and yield a non-trivial score better than an all-zero submission. Finally, it write a correctly formatted `submission.csv` with the required columns and `.csv` suffix.'
- What this solution (achieved 3.98324) has done: 'Your current score (1.23566, lower-is-better) is far from the target (-1.57356), so we need a real model improvement rather than tiny tweaks. To keep changes minimal while preserving the same overall “tabular feature extraction + regression” approach, I add a small set of well-known CHAMPS geometry features derived from `structures.csv` (atom types for the two atoms, 3D distance, and coordinate deltas) and train separate linear models per coupling `type` using scikit-learn. This keeps the solution simple, fast (<600s), and uses only provided competition files while typically improving MAE substantially versus type-mean. The submission format and ID alignment checks are kept to ensure a valid `.csv` output.'
- What this solution (achieved 6.30222) has done: 'Your current ridge-per-type model is underfitting badly because it doesn’t include the (very informative) coupling `type` as a feature inside each per-type model (so the model can’t learn type-specific offsets/geometry interactions beyond the split), and it also misses two cheap, high-signal physics features: inverse distance terms. To move the score substantially toward the (much better) target while keeping the same core approach (structures merge → simple engineered geometry features → per-type regression), I only add `inv_dist` and `inv_dist2` numeric features and switch the per-type regressor from `Ridge` to `RidgeCV` over a small alpha grid (same model family, same training loop structure) to reduce the gap without changing the overall logic. I also ensure merges stay consistent and keep submission formatting/id alignment unchanged. This should improve MAE/logMAE materially while remaining fast and stable under the 600s limit.'
- What this solution (achieved 7.6976) has done: 'Your current score is far worse (higher) than the target, so we should improve predictive accuracy without changing the overall approach (structures merge → simple geometry features → per-type Ridge-family regression). The biggest low-risk gain is to include the coupling `type` information inside each per-type model via a couple of type-specific transformations of the same distance feature (e.g., `dist`, `inv_dist`, `inv_dist2` scaled by a per-type factor) while keeping the same model family and training loop. Additionally, standardizing numeric features (fit on train, apply to test) usually helps Ridge regressions with mixed scales and is a minimal change that preserves semantics. Finally, we keep submission formatting and ID alignment unchanged.'
- What this solution (achieved 9.96457) has done: 'Your current score is much worse than the target (lower-is-better), so we should improve accuracy while keeping the same overall pipeline: structures merge → simple geometric features → per-type Ridge-family regression. The biggest low-risk issue is that you standardize numeric features globally across all coupling types, but you then train separate per-type models; scaling should be fit per type (on that type’s train rows) and applied to that type’s test rows to match each model’s data distribution. I also add a tiny set of strictly geometry-derived interaction features (`abs_dx/dy/dz`, `dist2`, `inv_dist3`) that don’t change the approach but usually reduce MAE. Finally, I keep the same RidgeCV-per-type loop and write the same `submission.csv` format.'
- What this solution (achieved 9.96242) has done: 'Your current score is far worse than the target (lower-is-better), so the smallest safe move is to add a couple of high-signal, low-cost features while keeping the same overall pipeline (structures merge → simple geometry features → per-type RidgeCV). The biggest missing signal in your feature set is the actual endpoint coordinates (and their per-type scaling); adding those preserves the exact modeling family and training loop but typically reduces MAE a lot for this competition. I also fix a likely data-quality issue by ensuring atom indices are read as integers (avoids subtle merge misses that can silently degrade features). Finally, I keep the same submission writing and ID alignment semantics.'
- What this solution (achieved 9.96242) has done: 'Your current score is far worse than the target (lower-is-better), so we should improve accuracy while keeping the same core pipeline (structures merge → simple geometry features → per-type RidgeCV). The biggest low-risk bug is that your one-hot encoding is fitted globally but includes only `atom_0/atom_1` and not the coupling `type`; since you already train per-type models, adding `type`-specific categorical context via `type`-conditioned atom-pair tokens (still just OHE + RidgeCV) usually reduces per-type MAE without changing the modeling family. I also add a minimal set of very cheap, high-signal cross features: `atom_0atom_1` pair and a direction-symmetric version (sorted pair) to help the linear model capture element-pair effects. Everything else (per-type loop, RidgeCV, scaling per type, submission writing) is kept the same.'
- What this solution (achieved 9.96242) has done: 'Your current score is far worse than the target (lower-is-better), so we need a modest accuracy increase while keeping the same pipeline (structures merge → simple geometry features → global OHE cats → per-type StandardScaler → per-type RidgeCV). The biggest low-risk fix is to correct the per-type test indexing bug: you currently index `X_test_num_values` (row order = 0..len(test)-1) using `test_f.index`, which can be non-contiguous after merges and can silently misalign features/predictions, severely hurting MAE. I make the test featureframes carry an explicit `row_id` (original test row position) and use that for all per-type slicing, preserving identical model/training logic but ensuring predictions correspond to the right rows. I also add sanity assertions to catch any future misalignment and still write a valid `submission.csv` with the required columns.'
- What this solution (achieved 9.95328) has done: 'Your current score is far worse than the target (lower is better), so we need a real accuracy improvement while keeping the same “structures merge → engineered geometry + OHE → per-type RidgeCV” core logic. The biggest low-risk issue is that the OHE is built globally but you then slice per-type; this makes the OHE blocks (especially `type_atom_pair*`) mostly constant within a type and wastes capacity, while also making the linear system unnecessarily large and potentially ill-conditioned. I keep the same per-type RidgeCV loop, but build the one-hot features per type (fit categories on that type’s train+test rows only) and drop the redundant `type_atom_pair*` columns (since you already train per type), which typically reduces noise/conditioning problems and improves MAE. I also add a minimal, high-signal geometry feature `cos` (directional cosine between the bond vector and axes via normalized dx/dy/dz) without changing the modeling family, to move the score toward the target.'
- What this solution (achieved 9.49012) has done: 'Your score is far worse than the target (lower is better), so we need a real accuracy gain while keeping the same “structures merge → engineered geometry + OHE → per-type RidgeCV” pipeline. The most impactful minimal fix is to add a tiny set of common CHAMPS baseline features that your current code is missing: endpoint Mulliken charges and magnetic shielding tensor diagonals for the two atoms, plus molecule-level potential energy and dipole vector magnitude. These are simple merges from provided CSVs, don’t change the model family or training loop, and typically reduce per-type MAE materially. I also keep your per-type scaling/OHE logic intact and just extend `num_cols` with these new signals, ensuring the script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 8.86072) has done: 'Your current score is much worse (higher) than the target (lower is better), so we need a meaningful accuracy gain while keeping your same core pipeline (structures merges → engineered numeric + OHE categorical → per-type StandardScaler → per-type RidgeCV). The biggest missing high-signal feature family in CHAMPS is “local neighbor geometry”: for each endpoint atom, aggregate distances to its nearest atoms (and their element types). I add a very small, fast-to-compute set of neighbor features (nearest-1/2/3 distances around atom_0 and atom_1, plus counts of H/C/N/O/F neighbors within a small cutoff) computed from `structures.csv` only; this preserves your model family and training loop while typically reducing MAE a lot. I keep all existing features/merges unchanged and simply extend `add_structure_features()` + `num_cols`, ensuring the script still runs end-to-end within the time limit and writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

BASE_CANDIDATES = [
    "/kaggle/input/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling",
    "../input/champs-scalar-coupling",
    "../kaggle/data/champs-scalar-coupling",
]

DATA_DIR = None
for p in BASE_CANDIDATES:
    if os.path.exists(p) and os.path.isfile(os.path.join(p, "train.csv")):
        DATA_DIR = p
        break

if DATA_DIR is None:
    for root in ["/kaggle/input", "/kaggle/data", "../input", "../kaggle/data", "/"]:
        if os.path.exists(root):
            for dirpath, dirnames, filenames in os.walk(root):
                if (
                    "train.csv" in filenames
                    and "test.csv" in filenames
                    and "sample_submission.csv" in filenames
                ):
                    DATA_DIR = dirpath
                    break
            if DATA_DIR is not None:
                break

if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not locate champs-scalar-coupling data directory containing train.csv/test.csv."
    )

print("Using DATA_DIR:", DATA_DIR)
print("Files:", sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])[:10])




## === cell 1
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sub_path = os.path.join(DATA_DIR, "sample_submission.csv")
structures_path = os.path.join(DATA_DIR, "structures.csv")

mulliken_path = os.path.join(DATA_DIR, "mulliken_charges.csv")
shield_path = os.path.join(DATA_DIR, "magnetic_shielding_tensors.csv")
dipole_path = os.path.join(DATA_DIR, "dipole_moments.csv")
energy_path = os.path.join(DATA_DIR, "potential_energy.csv")

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
    dtype={"atom_index_0": np.int32, "atom_index_1": np.int32},
)
test = pd.read_csv(
    test_path,
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
    dtype={"atom_index_0": np.int32, "atom_index_1": np.int32},
)
sample_sub = pd.read_csv(sub_path, usecols=["id", "scalar_coupling_constant"])

structures = pd.read_csv(
    structures_path,
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
    dtype={"atom_index": np.int32},
)

mulliken = pd.read_csv(
    mulliken_path,
    usecols=["molecule_name", "atom_index", "mulliken_charge"],
    dtype={"atom_index": np.int32, "mulliken_charge": np.float64},
)
shield = pd.read_csv(
    shield_path,
    usecols=["molecule_name", "atom_index", "XX", "YY", "ZZ"],
    dtype={
        "atom_index": np.int32,
        "XX": np.float64,
        "YY": np.float64,
        "ZZ": np.float64,
    },
)
dipole = pd.read_csv(
    dipole_path,
    usecols=["molecule_name", "X", "Y", "Z"],
    dtype={"X": np.float64, "Y": np.float64, "Z": np.float64},
)
energy = pd.read_csv(
    energy_path,
    usecols=["molecule_name", "potential_energy"],
    dtype={"potential_energy": np.float64},
)

atoms0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
    }
)
atoms1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
    }
)

mull0 = mulliken.rename(
    columns={"atom_index": "atom_index_0", "mulliken_charge": "mcharge_0"}
)
mull1 = mulliken.rename(
    columns={"atom_index": "atom_index_1", "mulliken_charge": "mcharge_1"}
)

sh0 = shield.rename(
    columns={
        "atom_index": "atom_index_0",
        "XX": "shield_xx_0",
        "YY": "shield_yy_0",
        "ZZ": "shield_zz_0",
    }
)
sh1 = shield.rename(
    columns={
        "atom_index": "atom_index_1",
        "XX": "shield_xx_1",
        "YY": "shield_yy_1",
        "ZZ": "shield_zz_1",
    }
)

dipole = dipole.copy()
dipole["dipole_mag"] = np.sqrt(
    dipole["X"] * dipole["X"] + dipole["Y"] * dipole["Y"] + dipole["Z"] * dipole["Z"]
).astype(np.float64)


def _build_neighbor_features(
    structures_df: pd.DataFrame, k: int = 3, cutoff: float = 2.0
) -> pd.DataFrame:
    s = structures_df.copy()
    s["x"] = s["x"].astype(np.float64)
    s["y"] = s["y"].astype(np.float64)
    s["z"] = s["z"].astype(np.float64)
    s["atom"] = s["atom"].astype(str)

    parts = []
    for mol, g in s.groupby("molecule_name", sort=False):
        coords = g[["x", "y", "z"]].to_numpy(dtype=np.float64, copy=False)
        n = coords.shape[0]
        if n <= 1:
            out = g[["molecule_name", "atom_index"]].copy()
            for i in range(1, k + 1):
                out[f"n{i}_dist"] = 0.0
            out["n_within_cutoff"] = 0.0
            for el in ["H", "C", "N", "O", "F"]:
                out[f"n_{el}_within_cutoff"] = 0.0
            parts.append(out)
            continue

        diff = coords[:, None, :] - coords[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d2, np.inf)
        dist = np.sqrt(d2)

        nn = np.partition(dist, kth=min(k, n - 1) - 1, axis=1)[:, : min(k, n - 1)]
        nn_sorted = np.sort(nn, axis=1)
        if nn_sorted.shape[1] < k:
            pad = np.zeros((n, k - nn_sorted.shape[1]), dtype=np.float64)
            nn_sorted = np.hstack([nn_sorted, pad])

        within = dist <= float(cutoff)
        atoms = g["atom"].to_numpy()
        out = g[["molecule_name", "atom_index"]].copy()
        for i in range(1, k + 1):
            out[f"n{i}_dist"] = nn_sorted[:, i - 1].astype(np.float64)
        out["n_within_cutoff"] = within.sum(axis=1).astype(np.float64)
        for el in ["H", "C", "N", "O", "F"]:
            mask_el = (atoms == el)[None, :]
            out[f"n_{el}_within_cutoff"] = (
                (within & mask_el).sum(axis=1).astype(np.float64)
            )

        parts.append(out)

    neigh = pd.concat(parts, axis=0, ignore_index=True)
    neigh["atom_index"] = neigh["atom_index"].astype(np.int32)
    return neigh


neighbor = _build_neighbor_features(structures, k=3, cutoff=2.0)
neighbor0 = neighbor.rename(columns={"atom_index": "atom_index_0"})
neighbor1 = neighbor.rename(columns={"atom_index": "atom_index_1"})
neighbor0 = neighbor0.rename(
    columns={
        c: f"{c}_0"
        for c in neighbor0.columns
        if c not in ["molecule_name", "atom_index_0"]
    }
)
neighbor1 = neighbor1.rename(
    columns={
        c: f"{c}_1"
        for c in neighbor1.columns
        if c not in ["molecule_name", "atom_index_1"]
    }
)


def add_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(atoms0, on=["molecule_name", "atom_index_0"], how="left")
    df = df.merge(atoms1, on=["molecule_name", "atom_index_1"], how="left")

    df = df.merge(mull0, on=["molecule_name", "atom_index_0"], how="left")
    df = df.merge(mull1, on=["molecule_name", "atom_index_1"], how="left")
    df = df.merge(sh0, on=["molecule_name", "atom_index_0"], how="left")
    df = df.merge(sh1, on=["molecule_name", "atom_index_1"], how="left")
    df = df.merge(
        dipole[["molecule_name", "X", "Y", "Z", "dipole_mag"]],
        on="molecule_name",
        how="left",
    )
    df = df.merge(energy, on="molecule_name", how="left")

    df = df.merge(neighbor0, on=["molecule_name", "atom_index_0"], how="left")
    df = df.merge(neighbor1, on=["molecule_name", "atom_index_1"], how="left")

    dx = (df["x0"] - df["x1"]).astype(np.float64)
    dy = (df["y0"] - df["y1"]).astype(np.float64)
    dz = (df["z0"] - df["z1"]).astype(np.float64)
    df["dx"] = dx
    df["dy"] = dy
    df["dz"] = dz

    df["abs_dx"] = np.abs(dx)
    df["abs_dy"] = np.abs(dy)
    df["abs_dz"] = np.abs(dz)

    dist2 = dx * dx + dy * dy + dz * dz
    df["dist2"] = dist2
    df["dist"] = np.sqrt(dist2)

    eps = 1e-12
    inv_dist = 1.0 / (df["dist"].astype(np.float64) + eps)
    df["inv_dist"] = inv_dist
    df["inv_dist2"] = inv_dist * inv_dist
    df["inv_dist3"] = df["inv_dist2"] * inv_dist

    denom = df["dist"].astype(np.float64) + eps
    df["dir_x"] = dx / denom
    df["dir_y"] = dy / denom
    df["dir_z"] = dz / denom

    for c in ["x0", "y0", "z0", "x1", "y1", "z1"]:
        df[c] = df[c].astype(np.float64)

    df["atom_0"] = df["atom_0"].fillna("X")
    df["atom_1"] = df["atom_1"].fillna("X")

    num_fill0 = [
        "x0",
        "y0",
        "z0",
        "x1",
        "y1",
        "z1",
        "dx",
        "dy",
        "dz",
        "abs_dx",
        "abs_dy",
        "abs_dz",
        "dist2",
        "dist",
        "inv_dist",
        "inv_dist2",
        "inv_dist3",
        "dir_x",
        "dir_y",
        "dir_z",
        "mcharge_0",
        "mcharge_1",
        "shield_xx_0",
        "shield_yy_0",
        "shield_zz_0",
        "shield_xx_1",
        "shield_yy_1",
        "shield_zz_1",
        "X",
        "Y",
        "Z",
        "dipole_mag",
        "potential_energy",
        "n1_dist_0",
        "n2_dist_0",
        "n3_dist_0",
        "n_within_cutoff_0",
        "n_H_within_cutoff_0",
        "n_C_within_cutoff_0",
        "n_N_within_cutoff_0",
        "n_O_within_cutoff_0",
        "n_F_within_cutoff_0",
        "n1_dist_1",
        "n2_dist_1",
        "n3_dist_1",
        "n_within_cutoff_1",
        "n_H_within_cutoff_1",
        "n_C_within_cutoff_1",
        "n_N_within_cutoff_1",
        "n_O_within_cutoff_1",
        "n_F_within_cutoff_1",
    ]
    for c in num_fill0:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(np.float64, copy=False)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

    a0 = df["atom_0"].astype(str)
    a1 = df["atom_1"].astype(str)
    df["atom_pair"] = (a0 + "_" + a1).astype(str)
    df["atom_pair_sorted"] = np.where(a0 <= a1, a0 + "_" + a1, a1 + "_" + a0).astype(
        str
    )

    return df


train = train.copy()
test = test.copy()
test["row_id"] = np.arange(len(test), dtype=np.int64)

train_f = add_structure_features(train)
test_f = add_structure_features(test)

assert "row_id" in test_f.columns, "row_id was lost during feature engineering merges."
assert test_f["row_id"].notna().all(), "row_id has NaNs after merges."
assert test_f[
    "row_id"
].is_unique, "row_id is not unique; test rows duplicated during merges."
assert len(test_f) == len(
    test
), "test_f row count changed; unexpected duplication/loss."

type_stats = (
    train_f.groupby("type")["scalar_coupling_constant"]
    .agg(["mean", "std"])
    .rename(columns={"mean": "type_y_mean", "std": "type_y_std"})
)
global_std = float(train_f["scalar_coupling_constant"].std())

train_f = train_f.join(type_stats, on="type")
test_f = test_f.join(type_stats, on="type")
test_f["type_y_mean"] = test_f["type_y_mean"].fillna(
    float(train_f["scalar_coupling_constant"].mean())
)
test_f["type_y_std"] = test_f["type_y_std"].fillna(global_std)

train_f["type_y_std"] = train_f["type_y_std"].fillna(global_std)
train_f["type_y_std"] = (
    train_f["type_y_std"].replace([np.inf, -np.inf], np.nan).fillna(global_std)
)
test_f["type_y_std"] = (
    test_f["type_y_std"].replace([np.inf, -np.inf], np.nan).fillna(global_std)
)

for base in [
    "dist",
    "inv_dist",
    "inv_dist2",
    "inv_dist3",
    "x0",
    "y0",
    "z0",
    "x1",
    "y1",
    "z1",
]:
    train_f[f"{base}_x_tstd"] = train_f[base].astype(np.float64) * train_f[
        "type_y_std"
    ].astype(np.float64)
    test_f[f"{base}_x_tstd"] = test_f[base].astype(np.float64) * test_f[
        "type_y_std"
    ].astype(np.float64)

cat_cols = ["atom_0", "atom_1", "atom_pair", "atom_pair_sorted"]

num_cols = [
    "x0",
    "y0",
    "z0",
    "x1",
    "y1",
    "z1",
    "dist2",
    "dist",
    "inv_dist",
    "inv_dist2",
    "inv_dist3",
    "dx",
    "dy",
    "dz",
    "abs_dx",
    "abs_dy",
    "abs_dz",
    "dir_x",
    "dir_y",
    "dir_z",
    "dist_x_tstd",
    "inv_dist_x_tstd",
    "inv_dist2_x_tstd",
    "inv_dist3_x_tstd",
    "x0_x_tstd",
    "y0_x_tstd",
    "z0_x_tstd",
    "x1_x_tstd",
    "y1_x_tstd",
    "z1_x_tstd",
    "mcharge_0",
    "mcharge_1",
    "shield_xx_0",
    "shield_yy_0",
    "shield_zz_0",
    "shield_xx_1",
    "shield_yy_1",
    "shield_zz_1",
    "X",
    "Y",
    "Z",
    "dipole_mag",
    "potential_energy",
    "n1_dist_0",
    "n2_dist_0",
    "n3_dist_0",
    "n_within_cutoff_0",
    "n_H_within_cutoff_0",
    "n_C_within_cutoff_0",
    "n_N_within_cutoff_0",
    "n_O_within_cutoff_0",
    "n_F_within_cutoff_0",
    "n1_dist_1",
    "n2_dist_1",
    "n3_dist_1",
    "n_within_cutoff_1",
    "n_H_within_cutoff_1",
    "n_C_within_cutoff_1",
    "n_N_within_cutoff_1",
    "n_O_within_cutoff_1",
    "n_F_within_cutoff_1",
]

X_train_num = train_f[num_cols].reset_index(drop=True).astype(np.float64)
X_test_num = test_f[num_cols].reset_index(drop=True).astype(np.float64)
y = train_f["scalar_coupling_constant"].astype(np.float64).values

type_mean = train.groupby("type")["scalar_coupling_constant"].mean()
global_mean = float(train["scalar_coupling_constant"].mean())
pred_test = np.full(len(test), global_mean, dtype=np.float64)

alphas = np.array([0.01, 0.1, 1.0, 10.0, 100.0], dtype=np.float64)

X_train_num_values = X_train_num.values
X_test_num_values = X_test_num.values

assert (
    X_test_num_values.shape[0] == len(test_f) == len(test)
), "Test feature row mismatch."




## === cell 2
for t, idx_tr in train_f.groupby("type").groups.items():
    idx_tr = np.asarray(list(idx_tr), dtype=np.int64)

    idx_te_rows = np.flatnonzero(test_f["type"].values == t).astype(np.int64)
    if idx_te_rows.size == 0:
        continue
    idx_te_out = test_f.loc[test_f["type"].values == t, "row_id"].to_numpy(
        dtype=np.int64
    )

    if idx_tr.size < 50:
        pred_test[idx_te_out] = (
            float(type_mean.loc[t]) if t in type_mean.index else global_mean
        )
        continue

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_num_s = scaler.fit_transform(X_train_num_values[idx_tr])
    Xte_num_s = scaler.transform(X_test_num_values[idx_te_rows])

    cats_tr = train_f.iloc[idx_tr][cat_cols].reset_index(drop=True)
    cats_te = test_f.iloc[idx_te_rows][cat_cols].reset_index(drop=True)
    cats_comb = pd.concat([cats_tr, cats_te], axis=0, ignore_index=True)
    ohe_comb = pd.get_dummies(cats_comb, columns=cat_cols, dtype=np.float64)

    Xtr_ohe = ohe_comb.iloc[: len(cats_tr)].to_numpy(dtype=np.float64, copy=False)
    Xte_ohe = ohe_comb.iloc[len(cats_tr) :].to_numpy(dtype=np.float64, copy=False)

    Xtr = np.hstack([Xtr_num_s, Xtr_ohe])
    Xte = np.hstack([Xte_num_s, Xte_ohe])

    model = RidgeCV(alphas=alphas, fit_intercept=True, cv=3)
    model.fit(Xtr, y[idx_tr])
    pred_test[idx_te_out] = model.predict(Xte)

submission = pd.DataFrame(
    {
        "id": test["id"].astype(sample_sub["id"].dtype, copy=False),
        "scalar_coupling_constant": pred_test,
    }
)

submission = submission[["id", "scalar_coupling_constant"]]
assert len(submission) == len(
    sample_sub
), "Submission row count mismatch vs sample_submission."
assert submission["id"].is_unique, "Submission ids are not unique."
assert (
    submission["id"].values == test["id"].values
).all(), "ID order mismatch vs test.csv."

out_file = "submission.csv"
submission.to_csv(out_file, index=False)
print("Wrote:", out_file)
print(submission.head())
print(
    "Pred stats:",
    float(np.min(pred_test)),
    float(np.mean(pred_test)),
    float(np.max(pred_test)),
)
