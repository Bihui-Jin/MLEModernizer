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

1.99777

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.99777) has done: 'Your code likely didn’t yield a score because it reads from `../input/*.csv`, but in your environment the files are under `/kaggle/data/champs-scalar-coupling/`, so it may be failing before writing a submission. I make a minimal path fix (with a safe fallback to the old Kaggle path) so it runs end-to-end and always writes a valid `submission.csv`. To move the score toward your (lower-is-better) target without changing the core approach, I keep the same LinearRegression on type dummies + distance, and add two very small, metric-relevant features (inverse distance and squared distance) plus `fit_intercept=False` to slightly improve calibration while staying in the same model family and training approach. Finally, I ensure dummy columns align between train/test and the submission matches the sample row count/order by using `test[['id']]` as the base.'
- What this solution (achieved 1.99777) has done: 'Your current score (1.99777, lower-is-better) is still far from the target (1.23424), so we should make a small but meaningful improvement while keeping the same overall approach (single global linear model on type dummies + geometric distance-derived features). The biggest low-risk gain without changing the modeling family is to reduce systematic bias by incorporating atom identity for each endpoint (atom_0 and atom_1) and a couple of additional smooth distance transforms (log distance and 1/d²), which are still simple linear features consistent with your existing feature engineering style. I also set `n_jobs=-1` for faster fitting within the same algorithm, and I keep the same submission alignment logic to guarantee a valid `submission.csv`. These changes should move the MAE-per-type down (and thus log-MAE down) without altering the core training loop or introducing any extra training complexity.'
- What this solution (achieved 1.99777) has done: 'Your current score (1.99777, lower-is-better) is still far from the target (1.23424), so we should make a small, low-risk improvement without changing the core approach (a single linear model on simple engineered geometry + categorical dummies). The biggest gain-per-change here is to add a tiny set of physically-plausible linear features derived from the same already-used inputs: per-axis deltas (dx, dy, dz) and absolute deltas, which can help the linear model capture anisotropy/orientation effects that pure distance misses. I’m also switching `fit_intercept` back to `True` to reduce global bias across coupling types while keeping the same model family and training semantics. All submission alignment logic is preserved so it still writes a valid `submission.csv` with the correct `id` order and row count.'
- What this solution (achieved 1.99777) has done: 'We keep the same single global `LinearRegression` on one-hot type/atom endpoints plus simple geometry, but add two tiny, physically-plausible numeric features that often reduce per-type MAE without changing the modeling approach: the dot product (`dx*dy + ...`) and cosine-like alignment (`dot/(d^2)`), both derived from the same coordinates you already use. We also add a very light per-type de-meaning of the target (fit the linear model on residuals after subtracting the training mean for each coupling `type`, then add the corresponding type mean back at prediction time), which is still linear calibration and typically reduces systematic bias across types for this metric. All file paths, merges, training loop, and submission alignment stay the same, and it still write a valid `submission.csv`. These are minimal changes aimed at moving your score down from 1.99777 toward 1.23424 (lower is better) without changing the core solution style.'
- What this solution (achieved 1.99777) has done: 'Your score is still much worse than the target (lower-is-better, 1.99777 → 1.23424), so the smallest meaningful improvement while preserving the same core “single global LinearRegression on simple engineered features” is to add a couple of additional linear numeric features that are known to reduce per-type bias without changing the modeling approach. Specifically, we add (1) per-axis squared deltas (dx², dy², dz²) to let the linear model weight directional components separately, and (2) a simple cross-term interaction magnitude (abs(dx*dy), abs(dy*dz), abs(dz*dx)) to capture non-spherical geometry effects that distance alone cannot. These are computed from already-used coordinates, don’t change the training loop or model family, and should reduce MAE across coupling types. Submission writing/alignment stays identical to ensure a valid `submission.csv`.'
- What this solution (achieved 1.99777) has done: 'We keep your exact pipeline (structures merges → engineered geometry features → one global LinearRegression on dummies+numerics → per-type mean add-back → submission merge) and only make a tiny, metric-relevant calibration improvement. Specifically, we add a per-type scaling factor for the centered residuals (fit on the training data using an in-sample closed-form slope), because the competition metric averages log-MAE per type and this reduces systematic under/over-amplitude per coupling type without changing the model family or training approach. We also fix a small redundancy/bug in your current feature set: `dot` is identical to `distance2`, and `cos_like` becomes ~1, so we redefine them into actually informative quantities derived from the same dx/dy/dz (still minimal feature engineering). These changes are intended to move the score down from 1.99777 toward 1.23424 (lower is better) without altering core semantics or adding complexity.'
- What this solution (achieved 1.99777) has done: 'We keep your exact pipeline (structures merge → engineered geometry features → one global LinearRegression on centered target → per-type scale + add-back → submission alignment) and only make small, metric-relevant feature corrections/extensions. The main fix is that your current `dot`/`cos_like` features aren’t true dot/cosine quantities for the atom-pair vector, so they can mislead the linear model; we redefine them into informative, still-simple pairwise geometry features (sum of squared deltas, L1 distance, component products). Then we add a tiny per-type interaction between type and distance (type-weighted distance) without changing the model family, which often reduces per-type MAE for this competition metric. Everything else (paths, training loop, centering/scaling, and submission writing) stays the same to preserve semantics and stability while nudging the score down toward 1.23424.'
- What this solution (achieved 1.99777) has done: 'I keep your exact pipeline (structures merges → engineered pairwise geometry features + one-hot type/atoms → single global LinearRegression on type-centered target → per-type scaling → submission alignment) but make two minimal, metric-relevant adjustments to reduce per-type MAE. First, I replace the redundant `dot=distance2` feature and the somewhat arbitrary `cos_like` with two simple, informative, symmetric geometry features derived from the same dx/dy/dz: L1 distance (`abs_dx+abs_dy+abs_dz`) and component product sum (`dxdy+dydz+dzdx`). Second, I add a tiny, closed-form per-type intercept correction (median residual per type on train) applied after scaling; this often reduces systematic per-type bias without changing the model family or training loop. These changes should move the score down from 1.99777 toward your lower-is-better target 1.23424 while staying well within your “same core logic” constraints.'
- What this solution (achieved 1.99777) has done: 'Your current score (1.99777, lower-is-better) is still far from the target (1.23424), so we should make a small but meaningful improvement while keeping your same single global `LinearRegression` pipeline and the same training/prediction semantics. The biggest low-risk gain here is to add a couple of per-atom physical descriptors you already have available in the provided dataset (Mulliken charge and magnetic shielding tensor diagonals) for each endpoint atom, merged by `(molecule_name, atom_index_*)`, which usually reduces per-type MAE without changing the model family. To keep changes minimal and stable, we only add these few numeric columns and keep all existing features, centering/scaling-by-type, and submission alignment logic unchanged. This should move the score down (better) toward the target while staying within the same core approach and runtime constraints.'
- What this solution (achieved 1.99777) has done: 'We keep your exact global LinearRegression pipeline and all existing features/centering/scaling, and only add one small, very relevant block that improves this competition’s metric: per-type robustization of the target via a signed `log1p` transform during fitting, then inverse-transform predictions back to the original scale. This is still the same model family and training loop, but it reduces the effect of heavy-tailed coupling distributions (a known issue in CHAMPS), often lowering per-type MAE and thus the averaged log-MAE. To keep the change minimal and stable, we compute the transform scale per type from training data only (a median absolute deviation proxy) and reuse your existing per-type mean add-back and per-type slope/offset calibration. Submission writing and alignment remain identical and still produce `submission.csv`.'
- What this solution (achieved 1.99777) has done: 'To move your (lower-is-better) score down toward 1.23424 without changing the core “single global LinearRegression on engineered features” approach, I’m making two small, metric-relevant upgrades: (1) add per-atom environment counts (how many H/C/N/O/F neighbors within a small radius) for each endpoint atom, computed only from `structures.csv`, and (2) switch to `Ridge` (still linear regression, same training loop/semantics) to stabilize coefficients given the many correlated features and one-hot dummies. These are minimal feature additions that usually reduce per-type MAE in CHAMPS while staying within your existing pipeline (same merges, same feature style, same per-type centering/log1p scaling, same submission alignment). All file paths and submission writing remain unchanged, and it still writes a valid `submission.csv`.'
- What this solution (achieved 1.99777) has done: 'Your current score (1.99777, lower-is-better) is still far from the target (1.23424), so we should make a small, safe improvement that keeps the exact same core pipeline (same merges, same feature families, same Ridge fit, same per-type centering/log1p scaling/calibration, same submission alignment). The biggest “minimal change / meaningful gain” remaining is to add a single additional provided molecule-level descriptor: `potential_energy`, merged by `molecule_name`, which is strongly correlated with coupling magnitudes and usually reduces per-type MAE without changing model semantics. I also fix one small feature bug (`test_2["dzdx"]` was written in an odd way) to ensure train/test feature definitions are consistent. Everything else is left untouched to preserve stability and runtime, and it still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

BASE_CANDIDATES = [
    "/kaggle/data/champs-scalar-coupling",
    "/kaggle/input/champs-scalar-coupling",
    "../input",  # fallback for classic Kaggle notebooks
]


def _first_existing_base(cands):
    for b in cands:
        if os.path.exists(b):
            return b
    return cands[-1]


BASE = _first_existing_base(BASE_CANDIDATES)

train = pd.read_csv(os.path.join(BASE, "train.csv"))
test = pd.read_csv(os.path.join(BASE, "test.csv"))
structures = pd.read_csv(os.path.join(BASE, "structures.csv"))

mulliken = pd.read_csv(os.path.join(BASE, "mulliken_charges.csv"))
shield = pd.read_csv(os.path.join(BASE, "magnetic_shielding_tensors.csv"))

potential = pd.read_csv(os.path.join(BASE, "potential_energy.csv"))

shield = shield[["molecule_name", "atom_index", "XX", "YY", "ZZ"]]



## === cell 1
structures_0 = structures.copy()
structures_1 = structures.copy()
structures_0.columns = structures.columns + (
    [""] + ["_0"] * (len(structures.columns) - 1)
)
structures_1.columns = structures.columns + (
    [""] + ["_1"] * (len(structures.columns) - 1)
)

train_2 = pd.merge(
    pd.merge(
        train,
        structures_0,
        left_on=["molecule_name", "atom_index_0"],
        right_on=["molecule_name", "atom_index_0"],
        how="inner",
    ),
    structures_1,
    left_on=["molecule_name", "atom_index_1"],
    right_on=["molecule_name", "atom_index_1"],
    how="inner",
)

test_2 = pd.merge(
    pd.merge(
        test,
        structures_0,
        left_on=["molecule_name", "atom_index_0"],
        right_on=["molecule_name", "atom_index_0"],
        how="inner",
    ),
    structures_1,
    left_on=["molecule_name", "atom_index_1"],
    right_on=["molecule_name", "atom_index_1"],
    how="inner",
)

mulliken_0 = mulliken.rename(
    columns={"atom_index": "atom_index_0", "mulliken_charge": "mulliken_charge_0"}
)
mulliken_1 = mulliken.rename(
    columns={"atom_index": "atom_index_1", "mulliken_charge": "mulliken_charge_1"}
)
shield_0 = shield.rename(
    columns={
        "atom_index": "atom_index_0",
        "XX": "shield_XX_0",
        "YY": "shield_YY_0",
        "ZZ": "shield_ZZ_0",
    }
)
shield_1 = shield.rename(
    columns={
        "atom_index": "atom_index_1",
        "XX": "shield_XX_1",
        "YY": "shield_YY_1",
        "ZZ": "shield_ZZ_1",
    }
)

train_2 = train_2.merge(mulliken_0, on=["molecule_name", "atom_index_0"], how="left")
train_2 = train_2.merge(mulliken_1, on=["molecule_name", "atom_index_1"], how="left")
train_2 = train_2.merge(shield_0, on=["molecule_name", "atom_index_0"], how="left")
train_2 = train_2.merge(shield_1, on=["molecule_name", "atom_index_1"], how="left")

test_2 = test_2.merge(mulliken_0, on=["molecule_name", "atom_index_0"], how="left")
test_2 = test_2.merge(mulliken_1, on=["molecule_name", "atom_index_1"], how="left")
test_2 = test_2.merge(shield_0, on=["molecule_name", "atom_index_0"], how="left")
test_2 = test_2.merge(shield_1, on=["molecule_name", "atom_index_1"], how="left")

train_2 = train_2.merge(potential, on="molecule_name", how="left")
test_2 = test_2.merge(potential, on="molecule_name", how="left")

desc_cols = [
    "mulliken_charge_0",
    "mulliken_charge_1",
    "shield_XX_0",
    "shield_YY_0",
    "shield_ZZ_0",
    "shield_XX_1",
    "shield_YY_1",
    "shield_ZZ_1",
    "potential_energy",
]
desc_medians = train_2[desc_cols].median(numeric_only=True)
train_2[desc_cols] = train_2[desc_cols].fillna(desc_medians)
test_2[desc_cols] = test_2[desc_cols].fillna(desc_medians)




## === cell 2
def add_neighbor_counts(
    pair_df, structures_df, radius=1.9, atoms=("H", "C", "N", "O", "F")
):
    mols = pd.Index(pair_df["molecule_name"].unique())
    s = structures_df[structures_df["molecule_name"].isin(mols)][
        ["molecule_name", "atom_index", "atom", "x", "y", "z"]
    ].copy()
    s[["x", "y", "z"]] = s[["x", "y", "z"]].astype("float32")

    by_mol = {
        m: df.reset_index(drop=True) for m, df in s.groupby("molecule_name", sort=False)
    }

    n = len(pair_df)
    out = {}
    for pos in (0, 1):
        for a in atoms:
            out[f"nbr_{a}_{pos}"] = np.zeros(n, dtype=np.float32)
        out[f"nbr_total_{pos}"] = np.zeros(n, dtype=np.float32)

    r2 = float(radius * radius)

    pair_mol_groups = pair_df.groupby("molecule_name", sort=False).indices
    for mol, idxs in pair_mol_groups.items():
        mol_atoms = by_mol.get(mol)
        if mol_atoms is None or mol_atoms.empty:
            continue

        coords = mol_atoms[["x", "y", "z"]].to_numpy(dtype=np.float32)
        atom_types = mol_atoms["atom"].to_numpy()

        ai = mol_atoms["atom_index"].to_numpy()
        map_idx = {int(a): i for i, a in enumerate(ai)}

        for pos, col in ((0, "atom_index_0"), (1, "atom_index_1")):
            centers = pair_df.loc[idxs, col].to_numpy()
            for j, center_atom_index in enumerate(centers):
                ci = map_idx.get(int(center_atom_index), None)
                if ci is None:
                    continue
                d = coords - coords[ci]
                dist2 = d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2]
                neigh_mask = (dist2 <= r2) & (dist2 > 1e-12)  # exclude self
                out[f"nbr_total_{pos}"][idxs[j]] = float(np.sum(neigh_mask))
                if np.any(neigh_mask):
                    neigh_types = atom_types[neigh_mask]
                    for a in atoms:
                        out[f"nbr_{a}_{pos}"][idxs[j]] = float(np.sum(neigh_types == a))

    for k, v in out.items():
        pair_df[k] = v
    return pair_df


train_2 = add_neighbor_counts(
    train_2, structures, radius=1.9, atoms=("H", "C", "N", "O", "F")
)
test_2 = add_neighbor_counts(
    test_2, structures, radius=1.9, atoms=("H", "C", "N", "O", "F")
)



## === cell 3
dx_tr = (train_2.x_0 - train_2.x_1).astype("float64")
dy_tr = (train_2.y_0 - train_2.y_1).astype("float64")
dz_tr = (train_2.z_0 - train_2.z_1).astype("float64")
train_2["distance"] = np.sqrt(dx_tr * dx_tr + dy_tr * dy_tr + dz_tr * dz_tr)

dx_te = (test_2.x_0 - test_2.x_1).astype("float64")
dy_te = (test_2.y_0 - test_2.y_1).astype("float64")
dz_te = (test_2.z_0 - test_2.z_1).astype("float64")
test_2["distance"] = np.sqrt(dx_te * dx_te + dy_te * dy_te + dz_te * dz_te)

eps = 1e-6
train_2["inv_distance"] = 1.0 / (train_2["distance"] + eps)
test_2["inv_distance"] = 1.0 / (test_2["distance"] + eps)
train_2["distance2"] = train_2["distance"] * train_2["distance"]
test_2["distance2"] = test_2["distance"] * test_2["distance"]

train_2["inv_distance2"] = 1.0 / (train_2["distance2"] + eps)
test_2["inv_distance2"] = 1.0 / (test_2["distance2"] + eps)

train_2["log_distance"] = np.log(train_2["distance"] + eps)
test_2["log_distance"] = np.log(test_2["distance"] + eps)

train_2["dx"] = dx_tr
train_2["dy"] = dy_tr
train_2["dz"] = dz_tr
test_2["dx"] = dx_te
test_2["dy"] = dy_te
test_2["dz"] = dz_te

train_2["abs_dx"] = np.abs(dx_tr)
train_2["abs_dy"] = np.abs(dy_tr)
train_2["abs_dz"] = np.abs(dz_tr)
test_2["abs_dx"] = np.abs(dx_te)
test_2["abs_dy"] = np.abs(dy_te)
test_2["abs_dz"] = np.abs(dz_te)

train_2["dx2"] = dx_tr * dx_tr
train_2["dy2"] = dy_tr * dy_tr
train_2["dz2"] = dz_tr * dz_tr
test_2["dx2"] = dx_te * dx_te
test_2["dy2"] = dy_te * dy_te
test_2["dz2"] = dz_te * dz_te

train_2["abs_dxdy"] = np.abs(dx_tr * dy_tr)
train_2["abs_dydz"] = np.abs(dy_tr * dz_tr)
train_2["abs_dzdx"] = np.abs(dz_tr * dx_tr)
test_2["abs_dxdy"] = np.abs(dx_te * dy_te)
test_2["abs_dydz"] = np.abs(dy_te * dz_te)
test_2["abs_dzdx"] = np.abs(dz_te * dx_te)

train_2["l1_distance"] = train_2["abs_dx"] + train_2["abs_dy"] + train_2["abs_dz"]
test_2["l1_distance"] = test_2["abs_dx"] + test_2["abs_dy"] + test_2["abs_dz"]

train_2["dxdy"] = dx_tr * dy_tr
train_2["dydz"] = dy_tr * dz_tr
train_2["dzdx"] = dz_tr * dx_tr
test_2["dxdy"] = dx_te * dy_te
test_2["dydz"] = dy_te * dz_te

test_2["dzdx"] = dz_te * dx_te

train_2["prod_sum"] = train_2["dxdy"] + train_2["dydz"] + train_2["dzdx"]
test_2["prod_sum"] = test_2["dxdy"] + test_2["dydz"] + test_2["dzdx"]

train_2["delta_mulliken"] = (
    train_2["mulliken_charge_0"] - train_2["mulliken_charge_1"]
).astype("float64")
test_2["delta_mulliken"] = (
    test_2["mulliken_charge_0"] - test_2["mulliken_charge_1"]
).astype("float64")
train_2["abs_delta_mulliken"] = np.abs(train_2["delta_mulliken"])
test_2["abs_delta_mulliken"] = np.abs(test_2["delta_mulliken"])



## === cell 4
train_type_dum = pd.get_dummies(train_2.type, prefix="type")
test_type_dum = pd.get_dummies(test_2.type, prefix="type").reindex(
    columns=train_type_dum.columns, fill_value=0
)

train_atom0_dum = pd.get_dummies(train_2.atom_0, prefix="atom0")
test_atom0_dum = pd.get_dummies(test_2.atom_0, prefix="atom0").reindex(
    columns=train_atom0_dum.columns, fill_value=0
)

train_atom1_dum = pd.get_dummies(train_2.atom_1, prefix="atom1")
test_atom1_dum = pd.get_dummies(test_2.atom_1, prefix="atom1").reindex(
    columns=train_atom1_dum.columns, fill_value=0
)

neighbor_cols = []
for pos in (0, 1):
    neighbor_cols.append(f"nbr_total_{pos}")
    for a in ("H", "C", "N", "O", "F"):
        neighbor_cols.append(f"nbr_{a}_{pos}")

num_cols = [
    "distance",
    "inv_distance",
    "distance2",
    "inv_distance2",
    "log_distance",
    "dx",
    "dy",
    "dz",
    "abs_dx",
    "abs_dy",
    "abs_dz",
    "l1_distance",
    "dx2",
    "dy2",
    "dz2",
    "abs_dxdy",
    "abs_dydz",
    "abs_dzdx",
    "dxdy",
    "dydz",
    "dzdx",
    "prod_sum",
    "mulliken_charge_0",
    "mulliken_charge_1",
    "delta_mulliken",
    "abs_delta_mulliken",
    "shield_XX_0",
    "shield_YY_0",
    "shield_ZZ_0",
    "shield_XX_1",
    "shield_YY_1",
    "shield_ZZ_1",
    "potential_energy",
] + neighbor_cols

X_train = pd.concat(
    [train_type_dum, train_atom0_dum, train_atom1_dum, train_2[num_cols]], axis=1
)
X_test = pd.concat(
    [test_type_dum, test_atom0_dum, test_atom1_dum, test_2[num_cols]], axis=1
)

X_train_td = train_type_dum.mul(train_2["distance"].values, axis=0)
X_test_td = test_type_dum.mul(test_2["distance"].values, axis=0)
X_train_td.columns = [c + "_x_dist" for c in X_train_td.columns]
X_test_td.columns = [c + "_x_dist" for c in X_test_td.columns]
X_train = pd.concat([X_train, X_train_td], axis=1)
X_test = pd.concat([X_test, X_test_td], axis=1)

type_mean = train_2.groupby("type")["scalar_coupling_constant"].mean()
global_mean = float(train_2["scalar_coupling_constant"].mean())
y_center = train_2["scalar_coupling_constant"] - train_2["type"].map(type_mean).astype(
    "float64"
)

type_mad = train_2.groupby("type")["scalar_coupling_constant"].apply(
    lambda s: float(np.median(np.abs(s.values - np.median(s.values))))
)
global_mad = float(
    np.median(
        np.abs(
            train_2["scalar_coupling_constant"].values
            - np.median(train_2["scalar_coupling_constant"].values)
        )
    )
)
type_scale = type_mad.clip(lower=1e-3)
global_scale = max(global_mad, 1e-3)

scale_train = (
    train_2["type"].map(type_scale).fillna(global_scale).astype("float64").values
)
y_fit = np.sign(y_center.values) * np.log1p(np.abs(y_center.values) / scale_train)

reg = Ridge(alpha=1.0, fit_intercept=True, random_state=0).fit(X=X_train, y=y_fit)

train_fit_pred = reg.predict(X_train).astype("float64")

train_type = train_2["type"].values
scale_by_type = {}
for t in type_mean.index:
    mask = train_type == t
    if not np.any(mask):
        continue
    yp = train_fit_pred[mask]
    yt = y_fit[mask].astype("float64")
    denom = float(np.dot(yp, yp)) + 1e-12
    a = float(np.dot(yp, yt)) / denom  # least-squares slope through origin
    a = float(np.clip(a, 0.5, 1.5))
    scale_by_type[t] = a

offset_by_type = {}
for t in type_mean.index:
    mask = train_type == t
    if not np.any(mask):
        continue
    a = float(scale_by_type.get(t, 1.0))
    resid = y_fit[mask].astype("float64") - a * train_fit_pred[mask]
    offset_by_type[t] = float(np.median(resid))



## === cell 5
if len(test_2) == 0:
    preds = np.array([], dtype="float64")
else:
    base = test_2["type"].map(type_mean).fillna(global_mean).astype("float64").values

    fit_pred = reg.predict(X=X_test).astype("float64")
    s = test_2["type"].map(scale_by_type).fillna(1.0).astype("float64").values
    o = test_2["type"].map(offset_by_type).fillna(0.0).astype("float64").values
    fit_pred_cal = fit_pred * s + o

    scale_test = (
        test_2["type"].map(type_scale).fillna(global_scale).astype("float64").values
    )
    centered_pred = np.sign(fit_pred_cal) * (
        np.expm1(np.abs(fit_pred_cal)) * scale_test
    )

    preds = centered_pred + base

pred_df = pd.DataFrame({"id": test_2["id"].values, "scalar_coupling_constant": preds})

submission = test[["id"]].merge(pred_df, on="id", how="left")
submission["scalar_coupling_constant"] = submission["scalar_coupling_constant"].fillna(
    0.0
)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
