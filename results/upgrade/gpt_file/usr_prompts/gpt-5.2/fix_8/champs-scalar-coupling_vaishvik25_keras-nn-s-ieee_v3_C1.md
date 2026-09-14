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

# 5. Target score

-1.5059779559326905

# 6. Current score

1.99777

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.99777) has done: 'I fix the immediate runtime blocker by replacing the missing `../input/top-mol` ensemble folder with a robust fallback that uses the competition’s provided `sample_submission.csv` to generate a valid prediction file. I also remove the notebook-only `%matplotlib inline` magic (which breaks in a `.py` run context) and update deprecated `np.bool` to `bool` for NumPy 1.26 compatibility. The stacking logic is preserved when the ensemble folder exists; otherwise the script still runs end-to-end and writes a correctly formatted `.csv` submission. This at least yield a valid submission (score likely poor vs target, but no longer “Not yielded”).'
- What this solution (achieved 1.99777) has done: 'Your current script can generate a valid submission but the fallback “distance baseline” is still likely far from the target, mainly because it ignores the coupling `type` scale/offset differences in a metric that is averaged per-type. I keep your core approach (simple per-type linear regression on geometry-derived features) but add a minimal, metric-aligned improvement: fit/predict per `type` with type-specific feature scaling and a small ridge stabilization (closed-form), which typically reduces MAE variance and improves log-MAE across types without changing the overall modeling idea. I also ensure the output submission follows the exact `sample_submission.csv` id order to avoid any accidental row misalignment. These are small, safe changes that should move the score down (better) toward the target while still running within the time limit.'
- What this solution (achieved 1.99777) has done: 'Your current score (1.99777, lower is better) is far from the target (-1.5059), so we need a real modeling improvement while keeping your core “per-type ridge regression on distance/pair features” logic intact. The biggest likely issue is that the baseline ignores the strong per-type dependence on local chemical environment; a minimal, classic fix in this competition is adding a few cheap “neighbor environment” distance features from `structures.csv` (e.g., counts of nearby atoms and nearest-neighbor distances around each endpoint atom). I keep your same closed-form per-type ridge with type-specific scaling, and only expand the design matrix with these additional geometry features (computed efficiently per molecule group). I also keep the `sample_submission.csv` id alignment exactly as you already do to avoid any row-order mistakes.'
- What this solution (achieved 1.99777) has done: 'Your current score (1.99777, lower-is-better) is still very far from the target (-1.5059), so we need a real but still “same-core-logic” improvement in the fallback model (the per-type closed-form ridge on geometry-derived features). I keep your ridge-per-type approach and your existing distance + local-environment features, but add two minimal, competition-proven feature sources already available in your data: Mulliken charges and magnetic shielding tensors for both endpoint atoms. I also add a tiny, type-wise robustification by fitting on `sign(y)*log1p(|y|)` and inverting at prediction time, which usually stabilizes log-MAE across coupling types without changing the training loop/architecture (still closed-form ridge). Submission writing and `sample_submission.csv` id alignment remain unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from subprocess import check_output



## === cell 1
for base in ["../input", "/kaggle/input", "/kaggle/data"]:
    if os.path.isdir(base):
        try:
            print(f"Listing: {base}")
            print(check_output(["ls", base]).decode("utf8"))
        except Exception as e:
            print(f"Could not list {base}: {e}")



## === cell 2
sub_path = "../input/top-mol"

if not os.path.isdir(sub_path):
    candidate_paths = [
        "/kaggle/input/top-mol",
        "/kaggle/data/top-mol",
        "/kaggle/input/champs-scalar-coupling/top-mol",
        "/kaggle/data/champs-scalar-coupling/top-mol",
    ]
    for p in candidate_paths:
        if os.path.isdir(p):
            sub_path = p
            break

print("Using sub_path:", sub_path)

all_files = os.listdir(sub_path) if os.path.isdir(sub_path) else []
print("Found files:", len(all_files))




## === cell 3
def find_competition_file(filename: str) -> str:
    candidates = [
        os.path.join("/kaggle/input/champs-scalar-coupling", filename),
        os.path.join("/kaggle/data/champs-scalar-coupling", filename),
        os.path.join("/kaggle/input", filename),
        os.path.join("/kaggle/data", filename),
        os.path.join("../input/champs-scalar-coupling", filename),
        os.path.join("../input", filename),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"Could not find {filename} in known Kaggle locations.")


def distance_baseline_submission(out_csv: str = "submission.csv") -> None:
    train_path = find_competition_file("train.csv")
    test_path = find_competition_file("test.csv")
    struct_path = find_competition_file("structures.csv")
    sample_path = find_competition_file("sample_submission.csv")

    mulliken_path = find_competition_file("mulliken_charges.csv")
    shielding_path = find_competition_file("magnetic_shielding_tensors.csv")

    sample_sub = pd.read_csv(sample_path, usecols=["id"])

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
    )
    test = pd.read_csv(
        test_path,
        usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
    )
    structures = pd.read_csv(
        struct_path,
        usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
    )

    mulliken = pd.read_csv(
        mulliken_path,
        usecols=["molecule_name", "atom_index", "mulliken_charge"],
    )

    shielding = pd.read_csv(
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
    )

    s0 = structures.rename(
        columns={
            "atom_index": "atom_index_0",
            "atom": "atom0",
            "x": "x0",
            "y": "y0",
            "z": "z0",
        }
    )
    s1 = structures.rename(
        columns={
            "atom_index": "atom_index_1",
            "atom": "atom1",
            "x": "x1",
            "y": "y1",
            "z": "z1",
        }
    )

    def compute_atom_env_features(structures_df: pd.DataFrame) -> pd.DataFrame:
        radii = (1.5, 2.0, 3.0)
        k_list = (1, 2, 3)  # nearest neighbor distances

        feats = []
        for mol, g in structures_df.groupby("molecule_name", sort=False):
            xyz = g[["x", "y", "z"]].to_numpy(dtype=np.float32)
            n = xyz.shape[0]
            if n == 0:
                continue

            diff = xyz[:, None, :] - xyz[None, :, :]
            dist = np.sqrt((diff * diff).sum(axis=2)).astype(np.float32)
            np.fill_diagonal(dist, np.float32(np.inf))

            cnts = {}
            for r in radii:
                cnts[r] = (dist < np.float32(r)).sum(axis=1).astype(np.int16)

            kmax = max(k_list)
            if n == 1:
                knn = np.full((1, kmax), np.float32(10.0), dtype=np.float32)
            else:
                part = np.partition(dist, kth=min(kmax - 1, n - 2), axis=1)[:, :kmax]
                part.sort(axis=1)
                knn = part.astype(np.float32)

            out = pd.DataFrame(
                {
                    "molecule_name": g["molecule_name"].values,
                    "atom_index": g["atom_index"].values.astype(np.int32),
                    "nn1": knn[:, 0],
                    "nn2": knn[:, 1] if kmax >= 2 else np.float32(10.0),
                    "nn3": knn[:, 2] if kmax >= 3 else np.float32(10.0),
                    "cnt_r1_5": cnts[1.5],
                    "cnt_r2_0": cnts[2.0],
                    "cnt_r3_0": cnts[3.0],
                }
            )
            feats.append(out)

        atom_env = pd.concat(feats, axis=0, ignore_index=True)
        return atom_env

    atom_env = compute_atom_env_features(structures)

    a0_env = atom_env.rename(
        columns={
            "atom_index": "atom_index_0",
            "nn1": "a0_nn1",
            "nn2": "a0_nn2",
            "nn3": "a0_nn3",
            "cnt_r1_5": "a0_cnt_r1_5",
            "cnt_r2_0": "a0_cnt_r2_0",
            "cnt_r3_0": "a0_cnt_r3_0",
        }
    )
    a1_env = atom_env.rename(
        columns={
            "atom_index": "atom_index_1",
            "nn1": "a1_nn1",
            "nn2": "a1_nn2",
            "nn3": "a1_nn3",
            "cnt_r1_5": "a1_cnt_r1_5",
            "cnt_r2_0": "a1_cnt_r2_0",
            "cnt_r3_0": "a1_cnt_r3_0",
        }
    )

    m0 = mulliken.rename(
        columns={
            "atom_index": "atom_index_0",
            "mulliken_charge": "a0_mulliken_charge",
        }
    )
    m1 = mulliken.rename(
        columns={
            "atom_index": "atom_index_1",
            "mulliken_charge": "a1_mulliken_charge",
        }
    )

    sh_cols = ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]
    sh0 = shielding.rename(
        columns={"atom_index": "atom_index_0", **{c: f"a0_sh_{c}" for c in sh_cols}}
    )
    sh1 = shielding.rename(
        columns={"atom_index": "atom_index_1", **{c: f"a1_sh_{c}" for c in sh_cols}}
    )

    def add_geom(df: pd.DataFrame) -> pd.DataFrame:
        df = df.merge(s0, how="left", on=["molecule_name", "atom_index_0"])
        df = df.merge(s1, how="left", on=["molecule_name", "atom_index_1"])

        df = df.merge(a0_env, how="left", on=["molecule_name", "atom_index_0"])
        df = df.merge(a1_env, how="left", on=["molecule_name", "atom_index_1"])

        df = df.merge(m0, how="left", on=["molecule_name", "atom_index_0"])
        df = df.merge(m1, how="left", on=["molecule_name", "atom_index_1"])
        df = df.merge(sh0, how="left", on=["molecule_name", "atom_index_0"])
        df = df.merge(sh1, how="left", on=["molecule_name", "atom_index_1"])

        dx = (df["x0"] - df["x1"]).astype(np.float32)
        dy = (df["y0"] - df["y1"]).astype(np.float32)
        dz = (df["z0"] - df["z1"]).astype(np.float32)
        dist = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)

        df["dist"] = dist
        eps = np.float32(1e-6)
        inv = (1.0 / (dist + eps)).astype(np.float32)
        df["inv_dist"] = inv
        df["inv_dist2"] = (inv * inv).astype(np.float32)
        df["pair"] = df["atom0"].astype(str) + "_" + df["atom1"].astype(str)

        env_cols = [
            "a0_nn1",
            "a0_nn2",
            "a0_nn3",
            "a0_cnt_r1_5",
            "a0_cnt_r2_0",
            "a0_cnt_r3_0",
            "a1_nn1",
            "a1_nn2",
            "a1_nn3",
            "a1_cnt_r1_5",
            "a1_cnt_r2_0",
            "a1_cnt_r3_0",
        ]
        for c in env_cols:
            if c not in df.columns:
                df[c] = 0.0
        df[env_cols] = df[env_cols].fillna(0.0)

        new_cols = (
            ["a0_mulliken_charge", "a1_mulliken_charge"]
            + [f"a0_sh_{c}" for c in sh_cols]
            + [f"a1_sh_{c}" for c in sh_cols]
        )
        for c in new_cols:
            if c not in df.columns:
                df[c] = 0.0
        df[new_cols] = df[new_cols].fillna(0.0)

        df["nn1_sum"] = (df["a0_nn1"] + df["a1_nn1"]).astype(np.float32)
        df["nn1_diff"] = (df["a0_nn1"] - df["a1_nn1"]).astype(np.float32)
        df["cnt2_sum"] = (df["a0_cnt_r2_0"] + df["a1_cnt_r2_0"]).astype(np.float32)

        df["q_sum"] = (df["a0_mulliken_charge"] + df["a1_mulliken_charge"]).astype(
            np.float32
        )
        df["q_diff"] = (df["a0_mulliken_charge"] - df["a1_mulliken_charge"]).astype(
            np.float32
        )

        df["a0_sh_trace"] = (df["a0_sh_XX"] + df["a0_sh_YY"] + df["a0_sh_ZZ"]).astype(
            np.float32
        )
        df["a1_sh_trace"] = (df["a1_sh_XX"] + df["a1_sh_YY"] + df["a1_sh_ZZ"]).astype(
            np.float32
        )
        df["sh_trace_sum"] = (df["a0_sh_trace"] + df["a1_sh_trace"]).astype(np.float32)
        df["sh_trace_diff"] = (df["a0_sh_trace"] - df["a1_sh_trace"]).astype(np.float32)
        return df

    train_f = add_geom(train)
    test_f = add_geom(test)

    pair_counts = train_f["pair"].value_counts()
    top_pairs = set(pair_counts.head(32).index.tolist())

    train_f["pair"] = train_f["pair"].where(
        train_f["pair"].isin(top_pairs), "__OTHER__"
    )
    test_f["pair"] = test_f["pair"].where(test_f["pair"].isin(top_pairs), "__OTHER__")

    fixed_levels = sorted(list(top_pairs) + ["__OTHER__"])
    level_to_col = {lv: i for i, lv in enumerate(fixed_levels)}
    other_col = level_to_col["__OTHER__"]

    env_feature_cols = (
        [
            "a0_nn1",
            "a0_nn2",
            "a0_nn3",
            "a0_cnt_r1_5",
            "a0_cnt_r2_0",
            "a0_cnt_r3_0",
            "a1_nn1",
            "a1_nn2",
            "a1_nn3",
            "a1_cnt_r1_5",
            "a1_cnt_r2_0",
            "a1_cnt_r3_0",
            "nn1_sum",
            "nn1_diff",
            "cnt2_sum",
            "a0_mulliken_charge",
            "a1_mulliken_charge",
            "q_sum",
            "q_diff",
            "a0_sh_trace",
            "a1_sh_trace",
            "sh_trace_sum",
            "sh_trace_diff",
        ]
        + [f"a0_sh_{c}" for c in sh_cols]
        + [f"a1_sh_{c}" for c in sh_cols]
    )

    def make_design(df: pd.DataFrame) -> np.ndarray:
        base = np.column_stack(
            [
                np.ones(len(df), dtype=np.float64),
                df["dist"].to_numpy(dtype=np.float64),
                df["inv_dist"].to_numpy(dtype=np.float64),
                df["inv_dist2"].to_numpy(dtype=np.float64),
            ]
        )

        env = df[env_feature_cols].to_numpy(dtype=np.float64, copy=False)

        pair_idx = (
            df["pair"].map(level_to_col).fillna(other_col).to_numpy(dtype=np.int64)
        )
        onehot = np.zeros((len(df), len(fixed_levels)), dtype=np.float64)
        onehot[np.arange(len(df)), pair_idx] = 1.0
        inter = onehot * df["inv_dist"].to_numpy(dtype=np.float64)[:, None]

        X = np.concatenate([base, env, onehot, inter], axis=1)
        return X

    def fit_beta_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
        X = X.astype(np.float64, copy=False)
        y = y.astype(np.float64, copy=False)
        XtX = X.T @ X
        Xty = X.T @ y
        I = np.eye(XtX.shape[0], dtype=np.float64)
        I[0, 0] = 0.0  # do not regularize intercept
        beta = np.linalg.solve(XtX + alpha * I, Xty)
        return beta

    def y_transform(y: np.ndarray) -> np.ndarray:
        return np.sign(y) * np.log1p(np.abs(y))

    def y_inverse(z: np.ndarray) -> np.ndarray:
        return np.sign(z) * (np.expm1(np.abs(z)))

    base_dim = 4
    env_dim = len(env_feature_cols)
    cont_idx = np.array(list(range(1, base_dim + env_dim)), dtype=np.int64)

    Xg = make_design(train_f)
    yg = train_f["scalar_coupling_constant"].to_numpy(dtype=np.float64)
    zg = y_transform(yg)

    mu_g = Xg[:, cont_idx].mean(axis=0)
    sd_g = Xg[:, cont_idx].std(axis=0)
    sd_g = np.where(sd_g == 0.0, 1.0, sd_g)

    Xg_s = Xg.copy()
    Xg_s[:, cont_idx] = (Xg_s[:, cont_idx] - mu_g) / sd_g

    beta_global = fit_beta_ridge(Xg_s, zg, alpha=1e-3)

    beta_by_type = {}
    scaler_by_type = {}

    for t, idx in train_f.groupby("type", sort=False).groups.items():
        g = train_f.loc[idx]
        Xt = make_design(g)
        yt = g["scalar_coupling_constant"].to_numpy(dtype=np.float64)
        zt = y_transform(yt)

        if len(g) < 500:
            beta_by_type[t] = beta_global
            scaler_by_type[t] = (mu_g, sd_g)
            continue

        mu_t = Xt[:, cont_idx].mean(axis=0)
        sd_t = Xt[:, cont_idx].std(axis=0)
        sd_t = np.where(sd_t == 0.0, 1.0, sd_t)

        Xt_s = Xt.copy()
        Xt_s[:, cont_idx] = (Xt_s[:, cont_idx] - mu_t) / sd_t

        beta_by_type[t] = fit_beta_ridge(Xt_s, zt, alpha=1e-3)
        scaler_by_type[t] = (mu_t, sd_t)

    preds = np.empty(len(test_f), dtype=np.float64)
    for t, idx in test_f.groupby("type", sort=False).groups.items():
        beta = beta_by_type.get(t, beta_global)
        mu_t, sd_t = scaler_by_type.get(t, (mu_g, sd_g))

        Xt = make_design(test_f.loc[idx])
        Xt[:, cont_idx] = (Xt[:, cont_idx] - mu_t) / sd_t
        zpred = Xt @ beta
        preds[idx] = y_inverse(zpred)

    pred_df = pd.DataFrame(
        {"id": test_f["id"].values, "scalar_coupling_constant": preds}
    )

    pred_df = sample_sub.merge(pred_df, on="id", how="left")
    pred_df["scalar_coupling_constant"] = pred_df["scalar_coupling_constant"].fillna(
        0.0
    )

    pred_df.to_csv(out_csv, index=False, float_format="%.6f")
    print(
        f"Wrote {out_csv} using per-type ridge regression with distance + local environment + (mulliken/shielding) + target log1p transform."
    )


if len(all_files) == 0:
    distance_baseline_submission("submission.csv")
    print("No ensemble files found. Used improved distance baseline instead of zeros.")
else:
    outs = []
    for f in sorted(all_files):
        fp = os.path.join(sub_path, f)
        if os.path.isfile(fp) and f.lower().endswith(".csv"):
            df = pd.read_csv(fp, index_col=0)
            outs.append(df)
    if len(outs) == 0:
        distance_baseline_submission("submission.csv")
        print(
            "Ensemble folder had no readable CSVs. Used improved distance baseline instead of zeros."
        )
    else:
        concat_sub = pd.concat(outs, axis=1)
        cols = list(map(lambda x: "mol" + str(x), range(len(concat_sub.columns))))
        concat_sub.columns = cols
        concat_sub.reset_index(inplace=True)
        concat_sub.rename(columns={concat_sub.columns[0]: "id"}, inplace=True)
        ncol = concat_sub.shape[1]
        print("Stacked shape:", concat_sub.shape)
        concat_sub.head()



## === cell 4
if "concat_sub" in globals():
    _ = concat_sub.iloc[:, 1:ncol].corr()
    print("Computed correlation matrix for stacked predictions:", _.shape)



## === cell 5
if "concat_sub" in globals():
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
        ax=ax,
    )
    plt.close(f)  # avoid display requirement in non-notebook runs



## === cell 6
if "concat_sub" in globals():
    concat_sub["m_max"] = concat_sub.iloc[:, 1:ncol].max(axis=1)
    concat_sub["m_min"] = concat_sub.iloc[:, 1:ncol].min(axis=1)
    concat_sub["m_mean"] = concat_sub.iloc[:, 1:ncol].mean(axis=1)
    concat_sub["m_median"] = concat_sub.iloc[:, 1:ncol].median(axis=1)
    print(concat_sub[["m_max", "m_min", "m_mean", "m_median"]].describe())



## === cell 7
cutoff_lo = 0.8
cutoff_hi = 0.2



## === cell 8
if "concat_sub" in globals():
    concat_sub["scalar_coupling_constant"] = concat_sub["m_mean"]
    concat_sub[["id", "scalar_coupling_constant"]].to_csv(
        "stack_mean.csv", index=False, float_format="%.6f"
    )

    concat_sub["scalar_coupling_constant"] = concat_sub["m_median"]
    concat_sub[["id", "scalar_coupling_constant"]].to_csv(
        "stack_median.csv", index=False, float_format="%.6f"
    )

    concat_sub["scalar_coupling_constant"] = np.where(
        np.all(concat_sub.iloc[:, 1:ncol] > cutoff_lo, axis=1),
        1,
        np.where(
            np.all(concat_sub.iloc[:, 1:ncol] < cutoff_hi, axis=1),
            0,
            concat_sub["m_median"],
        ),
    )
    concat_sub[["id", "scalar_coupling_constant"]].to_csv(
        "stack_pushout_median.csv", index=False, float_format="%.6f"
    )

    concat_sub["scalar_coupling_constant"] = np.where(
        np.all(concat_sub.iloc[:, 1:ncol] > cutoff_lo, axis=1),
        concat_sub["m_max"],
        np.where(
            np.all(concat_sub.iloc[:, 1:ncol] < cutoff_hi, axis=1),
            concat_sub["m_min"],
            concat_sub["m_mean"],
        ),
    )
    concat_sub[["id", "scalar_coupling_constant"]].to_csv(
        "stack_minmax_mean.csv", index=False, float_format="%.6f"
    )

    concat_sub["scalar_coupling_constant"] = np.where(
        np.all(concat_sub.iloc[:, 1:ncol] > cutoff_lo, axis=1),
        concat_sub["m_max"],
        np.where(
            np.all(concat_sub.iloc[:, 1:ncol] < cutoff_hi, axis=1),
            concat_sub["m_min"],
            concat_sub["m_median"],
        ),
    )
    concat_sub[["id", "scalar_coupling_constant"]].to_csv(
        "stack_minmax_median.csv", index=False, float_format="%.6f"
    )

    if {"mol0", "mol1", "mol2"}.issubset(concat_sub.columns):
        concat_sub["scalar_coupling_constant"] = (
            concat_sub["mol0"].rank(method="min")
            + concat_sub["mol1"].rank(method="min")
            + concat_sub["mol2"].rank(method="min")
        )
        rng = (
            concat_sub["scalar_coupling_constant"].max()
            - concat_sub["scalar_coupling_constant"].min()
        )
        if rng != 0:
            concat_sub["scalar_coupling_constant"] = (
                concat_sub["scalar_coupling_constant"]
                - concat_sub["scalar_coupling_constant"].min()
            ) / rng
        concat_sub[["id", "scalar_coupling_constant"]].to_csv(
            "stack_rank.csv", index=False, float_format="%.8f"
        )

    if {"mol1", "mol2"}.issubset(concat_sub.columns):
        concat_sub["scalar_coupling_constant"] = (
            0.60 * concat_sub["mol1"] + 0.40 * concat_sub["mol2"]
        )
        concat_sub[["id", "scalar_coupling_constant"]].to_csv(
            "s3.csv", index=False, float_format="%.8f"
        )

    concat_sub["scalar_coupling_constant"] = concat_sub["m_median"]
    concat_sub[["id", "scalar_coupling_constant"]].to_csv(
        "submission.csv", index=False, float_format="%.6f"
    )
    print("Wrote submission.csv (median stack) and additional stack_*.csv files.")
else:
    pass
