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

-1.3628733700694355

# 6. Current score

2.86391

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23596) has done: 'The current notebook fails because it tries to read other competitors’ datasets/submission files that are not present in your environment, so it never reaches the point of writing a valid `submission.csv`. I remove those external dependencies and replace them with an end-to-end, self-contained baseline that trains on the provided `train.csv` and predicts `test.csv`. To keep changes minimal while improving score from “no submission” toward your target, I use a simple per-`type` mean target encoding with light smoothing (a common strong baseline for this competition) and ensure the output matches `sample_submission.csv` exactly. The result always generate a valid `submission.csv` with columns `id,scalar_coupling_constant`.'
- What this solution (achieved 2.97414) has done: 'Your current solution predicts only a smoothed mean per coupling `type`, which ignores molecule geometry and atom information; that’s why the score is far from the (very strong) target. To move the score downward toward the target with minimal changes and without changing the overall “single-pass fit then predict” approach, I keep the type-based baseline but add a tiny set of strictly local, competition-relevant features: atom element types for the two atoms and their 3D distance from `structures.csv`. Then I replace the per-`type` mean with a per-`(type, atom_0, atom_1)` smoothed mean, and apply a small distance-based linear correction fit within each `type` (still simple regression, no new training loops). This remains fast, deterministic, self-contained, and almost certainly reduce MAE substantially versus the pure type mean, moving your score closer to the target.'
- What this solution (achieved 2.97415) has done: 'Your score is far worse than the target (lower is better), so we should legitimately improve accuracy with minimal core-logic disruption. The biggest issue in the current baseline is target leakage inside the “pair mean” encoding: each training row influences its own encoded feature, which makes the distance correction fit overly optimistic and harms generalization; we fix this by using out-of-fold (OOF) target encoding by molecule groups (consistent with the competition split). Then we fit the same per-`type` linear distance correction on OOF residuals (not leaky residuals) and apply it to test, keeping your overall “smoothed means + per-type linear correction” approach intact. Finally, we slightly stabilize the encodings by using an order-invariant atom-pair key (so (C,H) and (H,C) share stats), which typically improves generalization with negligible logic change.'
- What this solution (achieved 3.07858) has done: 'Your current score is far worse than the (lower-is-better) target, so we should improve generalization while keeping the same “OOF smoothed mean encoding + per-type linear distance correction” core logic. The largest remaining issue is that the distance correction is fit with plain least-squares per type, which is sensitive to heavy-tailed residuals and hurts the log-MAE metric; we can make it more robust without changing the approach by using a per-type *median/ MAD*-based slope estimate (still a simple linear correction on distance). We also align the correction fit with the competition split more closely by computing the distance-correction parameters in an OOF manner (fit on each fold’s training molecules and predict that fold), then refit on full train for test. These are minimal, deterministic changes that typically reduce MAE across types without introducing new models or training loops, and they still write a valid `submission.csv`.'
- What this solution (achieved 2.92166) has done: 'We need to move the score down (lower is better) toward a very strong target, and your current baseline is still extremely feature-light; the biggest safe gain without changing the overall “smoothed target encoding + per-type linear distance correction” logic is to add a few more strictly local, precomputed scalar atom features from the provided auxiliary CSVs. I minimally extend `add_atom_and_distance()` to also merge Mulliken charges and magnetic shielding tensors for the two atoms, and then upgrade the per-pair encoding key from just `(type, a_min, a_max)` to `(type, a_min, a_max, binned_dist)` to better capture different regimes per coupling type while keeping the same smoothed-mean core. Finally, I keep your robust per-type linear distance correction exactly as-is, but fit it on the OOF residuals from the improved base predictions (same training approach, just better inputs), which should reduce MAE and move your score materially closer to the target. All changes are deterministic, use only provided files/paths, and still write a valid `submission.csv`.'
- What this solution (achieved 2.82186) has done: 'Your current pipeline is still dominated by a coarse smoothed mean on `(type, a_min, a_max, dist_bin)` plus a distance-only correction; to move the score down toward the target without changing the overall approach, we make the “base mean” grouping slightly richer using the atom-level auxiliary features you already merged. Concretely, we add *binned* deltas of Mulliken charge and shielding trace (simple discretization, no new model), and compute the same OOF smoothed-mean encoding on `(type, a_min, a_max, dist_bin, dq_bin, dsh_bin)` to reduce underfitting while keeping leakage control identical. We keep your robust per-type linear distance correction exactly as-is, but fit it on residuals from the improved OOF base predictions (same training semantics). This is a minimal change that should legitimately reduce MAE and therefore reduce the log-MAE score toward your (much lower) target.'
- What this solution (achieved 3.02322) has done: 'Your current score (2.82186, lower-is-better) is still far from the target (-1.3629), so we should legitimately improve generalization with minimal disruption to your existing “OOF smoothed-mean encoding + per-type robust linear distance correction” pipeline. The biggest safe gain without changing the training approach is to make the distance correction slightly less underfit by fitting it on two distance transforms (`dist` and `1/(dist+eps)`) per type using the same robust median/MAD mechanics, which better matches how couplings decay with distance while preserving the same simple correction idea. Additionally, we add a single, minimal geometry-derived feature (`inv_dist_bin`) into the existing target-encoding key (still OOF, still smoothed means) to reduce aliasing where different distance regimes land in the same coarse bin. These are small, deterministic changes that keep your core logic intact and should reduce MAE across types, moving the score downward toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 2.60641) has done: 'Your current score is much worse than the target (lower is better), so we should improve generalization with very small changes to the existing “OOF smoothed-mean encoding + per-type robust distance correction” logic. The biggest low-risk issue is that the robust per-type correction is fit on raw residuals, but the competition metric is MAE-on-each-type then log/average, so a robust fit should prioritize absolute error; we keep the same two-feature linear correction but fit its coefficients by a tiny, deterministic coordinate-descent minimizing L1 (median) loss per type. Additionally, we slightly strengthen the base encoding smoothing hierarchy by smoothing pair stats toward the *type* mean with a type-dependent strength (rare types get stronger smoothing), which reduces variance without changing the modeling approach. All I/O paths and submission format remain unchanged, and the code still runs end-to-end within the time limit.'
- What this solution (achieved 2.60428) has done: 'I fix the runtime error caused by duplicate `type` columns during the fold merge by ensuring `val` contains the join keys only once (dropping the extra `type` in the fold slice) and by selecting only needed columns for the merge to avoid non-unique labels. This keeps your core approach unchanged (OOF smoothed target encoding on `PAIR_KEYS` + per-type robust 2-feature distance correction) while making the notebook run end-to-end. I also add small safety assertions to catch future merge/key issues early and ensure `test_pred` is always defined before writing `submission.csv`. No score-tuning changes are introduced beyond making the intended logic execute correctly.'
- What this solution (achieved 2.76776) has done: 'Your current score (2.60428, lower-is-better) is far from the target (-1.3629), so we should make a small, legitimate accuracy improvement without changing the overall “OOF smoothed target encoding + per-type robust distance correction” approach. The biggest low-risk missing signal is that the baseline ignores molecule-level context; we can minimally add `potential_energy` and `dipole_moments` (already provided) as extra *binned* keys in the same target-encoding grouping, keeping the same OOF-by-molecule leakage control. This preserves your core logic (same smoothed means + same robust per-type 2-feature correction), but should reduce MAE by capturing systematic molecule-level shifts per coupling type. All I/O paths remain unchanged and the script still writes a valid `submission.csv`.'
- What this solution (achieved 2.76776) has done: 'Your current score (2.76776, lower-is-better) is far from the target (-1.3629), so we should make a small, legitimate accuracy gain without changing the core “OOF smoothed target encoding + per-type robust 2-feature distance correction” approach. The lowest-risk improvement is to make the target-encoding less sparse by adding a *backoff hierarchy* (pair-with-bins → pair-without-molecule-bins → type), still computed OOF by molecule to avoid leakage. This keeps the same semantics (smoothed means as the base predictor) but reduces the number of unseen keys at test time, which typically lowers MAE. The distance correction stays exactly the same, but it is now fit on residuals from the improved (less-sparse) OOF base.'
- What this solution (achieved 2.76776) has done: 'We need to move the score down (lower is better) toward the target, so the smallest safe improvement is to reduce underfitting and key sparsity without changing your core “OOF smoothed target encoding + per-type robust distance correction” approach. I add a single additional backoff level (`KEY_COARSE`) between `KEY_MID` and `type` (still OOF-by-molecule) so fewer test rows fall all the way back to `type_mean`, which typically lowers MAE. I also make the atom-pair key order-invariant by atom *index* (swap 0/1 so `atom_index_0 < atom_index_1`) before merging structures/aux features; this reduces fragmentation of statistics when the same physical pair appears in different order and is consistent with scalar coupling symmetry. Finally, I keep the same robust L1 two-feature correction but fit it on the fold’s *in-fold* base predictions (rather than using partially-unfilled `oof_base` on the training side), removing a subtle training-time mismatch that can hurt generalization.'
- What this solution (achieved 2.86391) has done: 'Your current score is far worse than the target (lower is better), so we should make a small, legitimate generalization improvement while keeping your exact core pipeline (OOF smoothed target-encoding + per-type robust 2-feature distance correction). The biggest low-risk issue is that the molecule fold assignment is currently random, which can mix very different molecule sizes/compositions unevenly across folds and make OOF encodings noisier; I replace it with a deterministic hash-based fold split by `molecule_name` to stabilize OOF stats and better match the competition’s molecule-level split semantics. Additionally, I make the molecule-level bins (`pe_bin`, `dn_bin`) *type-aware* by using per-type quantile bins (still just discretization keys), which reduces key sparsity and captures systematic shifts per type more smoothly than fixed-width bins. Everything else (features, smoothing hierarchy, robust correction fit, I/O paths, submission schema) stays the same.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_ROOT_CANDIDATES = [
    "/kaggle/data/champs-scalar-coupling",
    "/kaggle/input/champs-scalar-coupling-challenge",
    "/kaggle/input/champs-scalar-coupling",
    "../input/champs-scalar-coupling-challenge",
    "../input/champs-scalar-coupling",
]

DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(p) and os.path.exists(os.path.join(p, "train.csv")):
        DATA_ROOT = p
        break

if DATA_ROOT is None:
    for base in ["/kaggle/input", "../input", "/kaggle/data"]:
        if os.path.exists(base):
            for name in os.listdir(base):
                cand = os.path.join(base, name)
                if os.path.isdir(cand) and os.path.exists(
                    os.path.join(cand, "train.csv")
                ):
                    DATA_ROOT = cand
                    break
        if DATA_ROOT is not None:
            break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate competition data folder containing train.csv"
    )

print("Using DATA_ROOT:", DATA_ROOT)
print("Files:", sorted([f for f in os.listdir(DATA_ROOT) if f.endswith(".csv")])[:10])



## === cell 1
train_path = os.path.join(DATA_ROOT, "train.csv")
test_path = os.path.join(DATA_ROOT, "test.csv")
sample_path = os.path.join(DATA_ROOT, "sample_submission.csv")
structures_path = os.path.join(DATA_ROOT, "structures.csv")
mulliken_path = os.path.join(DATA_ROOT, "mulliken_charges.csv")
shielding_path = os.path.join(DATA_ROOT, "magnetic_shielding_tensors.csv")
potential_path = os.path.join(DATA_ROOT, "potential_energy.csv")
dipole_path = os.path.join(DATA_ROOT, "dipole_moments.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample = pd.read_csv(sample_path)

required_train_cols = {
    "id",
    "molecule_name",
    "atom_index_0",
    "atom_index_1",
    "type",
    "scalar_coupling_constant",
}
required_test_cols = {"id", "molecule_name", "atom_index_0", "atom_index_1", "type"}
if not required_train_cols.issubset(train.columns):
    raise ValueError(
        f"train.csv missing columns: {required_train_cols - set(train.columns)}"
    )
if not required_test_cols.issubset(test.columns):
    raise ValueError(
        f"test.csv missing columns: {required_test_cols - set(test.columns)}"
    )
if not {"id", "scalar_coupling_constant"}.issubset(sample.columns):
    raise ValueError(
        "sample_submission.csv must contain columns: id, scalar_coupling_constant"
    )

print(
    "train shape:",
    train.shape,
    "test shape:",
    test.shape,
    "sample shape:",
    sample.shape,
)
print("train types:", train["type"].nunique(), "test types:", test["type"].nunique())

structures = pd.read_csv(
    structures_path, usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"]
)
structures["atom_index"] = structures["atom_index"].astype(np.int32)

mulliken = pd.read_csv(
    mulliken_path, usecols=["molecule_name", "atom_index", "mulliken_charge"]
)
mulliken["atom_index"] = mulliken["atom_index"].astype(np.int32)

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
shielding["atom_index"] = shielding["atom_index"].astype(np.int32)
shielding["shield_trace"] = (
    shielding["XX"] + shielding["YY"] + shielding["ZZ"]
).astype(np.float64)
shielding = shielding[["molecule_name", "atom_index", "shield_trace"]]

potential = pd.read_csv(potential_path, usecols=["molecule_name", "potential_energy"])
dipole = pd.read_csv(dipole_path, usecols=["molecule_name", "X", "Y", "Z"])
dipole["dipole_norm"] = np.sqrt(
    dipole["X"].to_numpy(np.float64) ** 2
    + dipole["Y"].to_numpy(np.float64) ** 2
    + dipole["Z"].to_numpy(np.float64) ** 2
).astype(np.float64)
dipole = dipole[["molecule_name", "dipole_norm"]]

print("structures shape:", structures.shape)
print("mulliken shape:", mulliken.shape)
print("shielding shape:", shielding.shape)
print("potential shape:", potential.shape)
print("dipole shape:", dipole.shape)
print(structures.head())




## === cell 2
def add_atom_and_distance(
    df, structures_df, mulliken_df, shielding_df, potential_df, dipole_df
):
    out = df.copy()
    swap_mask = out["atom_index_0"].to_numpy(np.int32) > out["atom_index_1"].to_numpy(
        np.int32
    )
    if swap_mask.any():
        a0 = out.loc[swap_mask, "atom_index_0"].values
        a1 = out.loc[swap_mask, "atom_index_1"].values
        out.loc[swap_mask, "atom_index_0"] = a1
        out.loc[swap_mask, "atom_index_1"] = a0

    s0 = structures_df.rename(
        columns={
            "atom_index": "atom_index_0",
            "atom": "atom_0",
            "x": "x0",
            "y": "y0",
            "z": "z0",
        }
    )
    s1 = structures_df.rename(
        columns={
            "atom_index": "atom_index_1",
            "atom": "atom_1",
            "x": "x1",
            "y": "y1",
            "z": "z1",
        }
    )

    m0 = mulliken_df.rename(
        columns={"atom_index": "atom_index_0", "mulliken_charge": "mulliken_0"}
    )
    m1 = mulliken_df.rename(
        columns={"atom_index": "atom_index_1", "mulliken_charge": "mulliken_1"}
    )

    sh0 = shielding_df.rename(
        columns={"atom_index": "atom_index_0", "shield_trace": "shield_trace_0"}
    )
    sh1 = shielding_df.rename(
        columns={"atom_index": "atom_index_1", "shield_trace": "shield_trace_1"}
    )

    out = out.merge(
        s0[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    out = out.merge(
        s1[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
    )

    out = out.merge(
        m0[["molecule_name", "atom_index_0", "mulliken_0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    out = out.merge(
        m1[["molecule_name", "atom_index_1", "mulliken_1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
    )

    out = out.merge(
        sh0[["molecule_name", "atom_index_0", "shield_trace_0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    out = out.merge(
        sh1[["molecule_name", "atom_index_1", "shield_trace_1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
    )

    out = out.merge(potential_df, on="molecule_name", how="left")
    out = out.merge(dipole_df, on="molecule_name", how="left")

    missing = out[["atom_0", "atom_1", "x0", "x1"]].isna().any(axis=1).mean()
    if missing > 0:
        print(f"Warning: fraction of rows with missing structure merge = {missing:.6f}")

    dx = out["x0"] - out["x1"]
    dy = out["y0"] - out["y1"]
    dz = out["z0"] - out["z1"]
    out["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float64)

    keep_cols = [
        "id",
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "atom_0",
        "atom_1",
        "dist",
        "mulliken_0",
        "mulliken_1",
        "shield_trace_0",
        "shield_trace_1",
        "potential_energy",
        "dipole_norm",
    ]
    extra_cols = [
        c for c in df.columns if c not in keep_cols
    ]  # includes target for train
    keep_cols = list(dict.fromkeys(keep_cols + extra_cols))
    out = out[keep_cols]
    return out


train_feat = add_atom_and_distance(
    train, structures, mulliken, shielding, potential, dipole
)
test_feat = add_atom_and_distance(
    test, structures, mulliken, shielding, potential, dipole
)

global_dist_median = pd.concat([train_feat["dist"], test_feat["dist"]], axis=0).median()
train_feat["dist"] = train_feat["dist"].fillna(global_dist_median)
test_feat["dist"] = test_feat["dist"].fillna(global_dist_median)

for col in ["mulliken_0", "mulliken_1", "shield_trace_0", "shield_trace_1"]:
    fill_val = train_feat[col].median()
    train_feat[col] = train_feat[col].fillna(fill_val).astype(np.float64)
    test_feat[col] = test_feat[col].fillna(fill_val).astype(np.float64)

pe_fill = train_feat["potential_energy"].median()
dn_fill = train_feat["dipole_norm"].median()
train_feat["potential_energy"] = (
    train_feat["potential_energy"].fillna(pe_fill).astype(np.float64)
)
test_feat["potential_energy"] = (
    test_feat["potential_energy"].fillna(pe_fill).astype(np.float64)
)
train_feat["dipole_norm"] = train_feat["dipole_norm"].fillna(dn_fill).astype(np.float64)
test_feat["dipole_norm"] = test_feat["dipole_norm"].fillna(dn_fill).astype(np.float64)

train_feat["atom_0"] = train_feat["atom_0"].fillna("UNK")
train_feat["atom_1"] = train_feat["atom_1"].fillna("UNK")
test_feat["atom_0"] = test_feat["atom_0"].fillna("UNK")
test_feat["atom_1"] = test_feat["atom_1"].fillna("UNK")

train_feat["a_min"] = np.where(
    train_feat["atom_0"] <= train_feat["atom_1"],
    train_feat["atom_0"],
    train_feat["atom_1"],
)
train_feat["a_max"] = np.where(
    train_feat["atom_0"] <= train_feat["atom_1"],
    train_feat["atom_1"],
    train_feat["atom_0"],
)
test_feat["a_min"] = np.where(
    test_feat["atom_0"] <= test_feat["atom_1"], test_feat["atom_0"], test_feat["atom_1"]
)
test_feat["a_max"] = np.where(
    test_feat["atom_0"] <= test_feat["atom_1"], test_feat["atom_1"], test_feat["atom_0"]
)

BIN_WIDTH = 0.10
MAX_BIN = 60
train_feat["dist_bin"] = np.minimum(
    (train_feat["dist"].values / BIN_WIDTH).astype(np.int32), MAX_BIN
)
test_feat["dist_bin"] = np.minimum(
    (test_feat["dist"].values / BIN_WIDTH).astype(np.int32), MAX_BIN
)

INV_BIN_WIDTH = 0.25
INV_MAX_BIN = 80
eps_inv = 1e-3
train_inv = 1.0 / (train_feat["dist"].to_numpy(np.float64) + eps_inv)
test_inv = 1.0 / (test_feat["dist"].to_numpy(np.float64) + eps_inv)
train_feat["inv_dist_bin"] = np.minimum(
    (train_inv / INV_BIN_WIDTH).astype(np.int32), INV_MAX_BIN
)
test_feat["inv_dist_bin"] = np.minimum(
    (test_inv / INV_BIN_WIDTH).astype(np.int32), INV_MAX_BIN
)

DQ_BIN_WIDTH = 0.02
DQ_MAX_BIN = 50
DSH_BIN_WIDTH = 5.0
DSH_MAX_BIN = 50

train_feat["dq"] = (train_feat["mulliken_0"] - train_feat["mulliken_1"]).astype(
    np.float64
)
test_feat["dq"] = (test_feat["mulliken_0"] - test_feat["mulliken_1"]).astype(np.float64)
train_feat["dsh"] = (
    train_feat["shield_trace_0"] - train_feat["shield_trace_1"]
).astype(np.float64)
test_feat["dsh"] = (test_feat["shield_trace_0"] - test_feat["shield_trace_1"]).astype(
    np.float64
)

train_feat["dq_bin"] = np.clip(
    np.rint(train_feat["dq"].to_numpy(np.float64) / DQ_BIN_WIDTH).astype(np.int32),
    -DQ_MAX_BIN,
    DQ_MAX_BIN,
)
test_feat["dq_bin"] = np.clip(
    np.rint(test_feat["dq"].to_numpy(np.float64) / DQ_BIN_WIDTH).astype(np.int32),
    -DQ_MAX_BIN,
    DQ_MAX_BIN,
)
train_feat["dsh_bin"] = np.clip(
    np.rint(train_feat["dsh"].to_numpy(np.float64) / DSH_BIN_WIDTH).astype(np.int32),
    -DSH_MAX_BIN,
    DSH_MAX_BIN,
)
test_feat["dsh_bin"] = np.clip(
    np.rint(test_feat["dsh"].to_numpy(np.float64) / DSH_BIN_WIDTH).astype(np.int32),
    -DSH_MAX_BIN,
    DSH_MAX_BIN,
)


def add_type_quantile_bins(train_df, test_df, col, n_bins=20, out_col=None):
    if out_col is None:
        out_col = f"{col}_qbin"
    train_bins = np.empty(train_df.shape[0], dtype=np.int16)
    test_bins = np.empty(test_df.shape[0], dtype=np.int16)

    try:
        global_edges = np.unique(
            np.quantile(
                train_df[col].to_numpy(np.float64), np.linspace(0.0, 1.0, n_bins + 1)
            )
        )
    except Exception:
        global_edges = np.array(
            [train_df[col].min(), train_df[col].max()], dtype=np.float64
        )

    for t in train_df["type"].unique():
        trn_mask = train_df["type"].values == t
        tst_mask = test_df["type"].values == t
        x = train_df.loc[trn_mask, col].to_numpy(np.float64)

        edges = global_edges
        if x.size >= 200:
            try:
                edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)))
            except Exception:
                edges = global_edges

        if edges.size <= 2:
            train_bins[trn_mask] = 0
            test_bins[tst_mask] = 0
            continue

        train_bins[trn_mask] = np.clip(
            np.digitize(
                train_df.loc[trn_mask, col].to_numpy(np.float64),
                edges[1:-1],
                right=True,
            ),
            0,
            edges.size - 2,
        )
        test_bins[tst_mask] = np.clip(
            np.digitize(
                test_df.loc[tst_mask, col].to_numpy(np.float64), edges[1:-1], right=True
            ),
            0,
            edges.size - 2,
        )

    train_df[out_col] = train_bins.astype(np.int16)
    test_df[out_col] = test_bins.astype(np.int16)
    return train_df, test_df


train_feat, test_feat = add_type_quantile_bins(
    train_feat, test_feat, "potential_energy", n_bins=20, out_col="pe_bin"
)
train_feat, test_feat = add_type_quantile_bins(
    train_feat, test_feat, "dipole_norm", n_bins=20, out_col="dn_bin"
)

print(
    train_feat[
        [
            "type",
            "atom_0",
            "atom_1",
            "a_min",
            "a_max",
            "dist",
            "dist_bin",
            "inv_dist_bin",
            "dq",
            "dq_bin",
            "dsh",
            "dsh_bin",
            "potential_energy",
            "pe_bin",
            "dipole_norm",
            "dn_bin",
            "scalar_coupling_constant",
        ]
    ].head()
)



## === cell 3
global_mean = train_feat["scalar_coupling_constant"].mean()
BASE_SMOOTHING = 50.0

type_stats = (
    train_feat.groupby("type")["scalar_coupling_constant"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "type_mean", "count": "type_count"})
    .reset_index()
)
type_stats["type_mean_smooth"] = (
    type_stats["type_mean"] * type_stats["type_count"] + global_mean * BASE_SMOOTHING
) / (type_stats["type_count"] + BASE_SMOOTHING)

type_stats["type_smoothing"] = np.clip(
    BASE_SMOOTHING * (20000.0 / (type_stats["type_count"].astype(np.float64) + 1.0)),
    20.0,
    400.0,
)


def stable_mol_fold(mol_names, n_folds=5, seed=42):
    h = pd.util.hash_pandas_object(pd.Index(mol_names), index=False).to_numpy(np.uint64)
    h ^= np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15)
    return (h % np.uint64(n_folds)).astype(np.int8)


train_feat["fold"] = stable_mol_fold(
    train_feat["molecule_name"].values, n_folds=5, seed=RANDOM_STATE
)

KEY_FINE = [
    "type",
    "a_min",
    "a_max",
    "dist_bin",
    "inv_dist_bin",
    "dq_bin",
    "dsh_bin",
    "pe_bin",
    "dn_bin",
]
KEY_MID = [
    "type",
    "a_min",
    "a_max",
    "dist_bin",
    "inv_dist_bin",
    "dq_bin",
    "dsh_bin",
]
KEY_COARSE = [
    "type",
    "a_min",
    "a_max",
    "dist_bin",
]


def robust_l1_two_feature_fit_per_type(df_type, max_iter=25, eps=1e-12):
    x1 = df_type["dist"].to_numpy(np.float64)
    x2 = df_type["inv_dist"].to_numpy(np.float64)
    y = df_type["resid"].to_numpy(np.float64)

    if y.size < 20:
        return 0.0, 0.0, float(np.median(y)) if y.size else 0.0

    x1m = np.median(x1)
    x2m = np.median(x2)
    x1c = x1 - x1m
    x2c = x2 - x2m

    b1 = 0.0
    b2 = 0.0
    a = np.median(y)

    for _ in range(max_iter):
        r = y - (a + b1 * x1c + b2 * x2c)
        a_new = a + np.median(r)
        r = y - (a_new + b1 * x1c + b2 * x2c)

        mask1 = np.abs(x1c) > 1e-9
        if mask1.sum() >= 20:
            ratios = (r[mask1] / (x1c[mask1] + eps)).astype(np.float64)
            weights = np.abs(x1c[mask1]).astype(np.float64)
            order = np.argsort(ratios)
            ratios_s = ratios[order]
            weights_s = weights[order]
            csum = np.cumsum(weights_s)
            cutoff = 0.5 * csum[-1]
            b1_new = float(ratios_s[np.searchsorted(csum, cutoff)])
        else:
            b1_new = b1

        r = y - (a_new + b1_new * x1c + b2 * x2c)

        mask2 = np.abs(x2c) > 1e-9
        if mask2.sum() >= 20:
            ratios = (r[mask2] / (x2c[mask2] + eps)).astype(np.float64)
            weights = np.abs(x2c[mask2]).astype(np.float64)
            order = np.argsort(ratios)
            ratios_s = ratios[order]
            weights_s = weights[order]
            csum = np.cumsum(weights_s)
            cutoff = 0.5 * csum[-1]
            b2_new = float(ratios_s[np.searchsorted(csum, cutoff)])
        else:
            b2_new = b2

        if (abs(a_new - a) + abs(b1_new - b1) + abs(b2_new - b2)) < 1e-6:
            a, b1, b2 = a_new, b1_new, b2_new
            break

        a, b1, b2 = a_new, b1_new, b2_new

    intercept = a - b1 * x1m - b2 * x2m
    return float(b1), float(b2), float(intercept)


eps_inv = 1e-3
inv_dist_train = 1.0 / (train_feat["dist"].to_numpy(np.float64) + eps_inv)

oof_base = np.empty(train_feat.shape[0], dtype=np.float64)
oof_b1 = np.zeros(train_feat.shape[0], dtype=np.float64)
oof_b2 = np.zeros(train_feat.shape[0], dtype=np.float64)
oof_intercept = np.zeros(train_feat.shape[0], dtype=np.float64)

KEY_FINE = list(dict.fromkeys(KEY_FINE))
KEY_MID = list(dict.fromkeys(KEY_MID))
KEY_COARSE = list(dict.fromkeys(KEY_COARSE))

n_folds = 5
for f in range(n_folds):
    trn_mask = train_feat["fold"].values != f
    val_mask = ~trn_mask

    trn_y = train_feat.loc[trn_mask, "scalar_coupling_constant"].to_numpy(np.float64)

    trn_fine = train_feat.loc[trn_mask, KEY_FINE]
    trn_mid = train_feat.loc[trn_mask, KEY_MID]
    trn_coarse = train_feat.loc[trn_mask, KEY_COARSE]

    val_fine = train_feat.loc[val_mask, KEY_FINE]
    val_mid = train_feat.loc[val_mask, KEY_MID]
    val_coarse = train_feat.loc[val_mask, KEY_COARSE]

    fine_stats_f = (
        pd.concat([trn_fine, pd.Series(trn_y, name="y")], axis=1)
        .groupby(KEY_FINE)["y"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "fine_mean", "count": "fine_count"})
        .reset_index()
    )
    fine_stats_f = fine_stats_f.merge(
        type_stats[["type", "type_mean_smooth", "type_smoothing"]],
        on="type",
        how="left",
    )
    fine_stats_f["fine_mean_smooth"] = (
        fine_stats_f["fine_mean"] * fine_stats_f["fine_count"]
        + fine_stats_f["type_mean_smooth"] * fine_stats_f["type_smoothing"]
    ) / (fine_stats_f["fine_count"] + fine_stats_f["type_smoothing"])

    mid_stats_f = (
        pd.concat([trn_mid, pd.Series(trn_y, name="y")], axis=1)
        .groupby(KEY_MID)["y"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "mid_mean", "count": "mid_count"})
        .reset_index()
    )
    mid_stats_f = mid_stats_f.merge(
        type_stats[["type", "type_mean_smooth", "type_smoothing"]],
        on="type",
        how="left",
    )
    mid_stats_f["mid_mean_smooth"] = (
        mid_stats_f["mid_mean"] * mid_stats_f["mid_count"]
        + mid_stats_f["type_mean_smooth"] * mid_stats_f["type_smoothing"]
    ) / (mid_stats_f["mid_count"] + mid_stats_f["type_smoothing"])

    coarse_stats_f = (
        pd.concat([trn_coarse, pd.Series(trn_y, name="y")], axis=1)
        .groupby(KEY_COARSE)["y"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "coarse_mean", "count": "coarse_count"})
        .reset_index()
    )
    coarse_stats_f = coarse_stats_f.merge(
        type_stats[["type", "type_mean_smooth", "type_smoothing"]],
        on="type",
        how="left",
    )
    coarse_stats_f["coarse_mean_smooth"] = (
        coarse_stats_f["coarse_mean"] * coarse_stats_f["coarse_count"]
        + coarse_stats_f["type_mean_smooth"] * coarse_stats_f["type_smoothing"]
    ) / (coarse_stats_f["coarse_count"] + coarse_stats_f["type_smoothing"])

    if (
        val_fine.columns.duplicated().any()
        or val_mid.columns.duplicated().any()
        or val_coarse.columns.duplicated().any()
    ):
        raise RuntimeError(f"Duplicate columns detected in fold {f} val slices.")

    val_merge = (
        val_fine.merge(
            fine_stats_f[KEY_FINE + ["fine_mean_smooth"]], on=KEY_FINE, how="left"
        )
        .merge(mid_stats_f[KEY_MID + ["mid_mean_smooth"]], on=KEY_MID, how="left")
        .merge(
            coarse_stats_f[KEY_COARSE + ["coarse_mean_smooth"]],
            on=KEY_COARSE,
            how="left",
        )
        .merge(type_stats[["type", "type_mean_smooth"]], on="type", how="left")
    )

    pred_val_base = (
        val_merge["fine_mean_smooth"]
        .fillna(val_merge["mid_mean_smooth"])
        .fillna(val_merge["coarse_mean_smooth"])
        .fillna(val_merge["type_mean_smooth"])
        .fillna(global_mean)
        .astype(np.float64)
        .to_numpy(np.float64)
    )
    oof_base[val_mask] = pred_val_base

    trn_merge = (
        trn_fine.merge(
            fine_stats_f[KEY_FINE + ["fine_mean_smooth"]], on=KEY_FINE, how="left"
        )
        .merge(mid_stats_f[KEY_MID + ["mid_mean_smooth"]], on=KEY_MID, how="left")
        .merge(
            coarse_stats_f[KEY_COARSE + ["coarse_mean_smooth"]],
            on=KEY_COARSE,
            how="left",
        )
        .merge(type_stats[["type", "type_mean_smooth"]], on="type", how="left")
    )
    pred_trn_base_fold = (
        trn_merge["fine_mean_smooth"]
        .fillna(trn_merge["mid_mean_smooth"])
        .fillna(trn_merge["coarse_mean_smooth"])
        .fillna(trn_merge["type_mean_smooth"])
        .fillna(global_mean)
        .astype(np.float64)
        .to_numpy(np.float64)
    )

    tmp_trn_corr = pd.DataFrame(
        {
            "type": train_feat.loc[trn_mask, "type"].values,
            "dist": train_feat.loc[trn_mask, "dist"].values,
            "inv_dist": inv_dist_train[trn_mask],
            "resid": train_feat.loc[trn_mask, "scalar_coupling_constant"].to_numpy(
                np.float64
            )
            - pred_trn_base_fold,
        }
    )
    tmp_val_types = pd.DataFrame({"type": train_feat.loc[val_mask, "type"].values})

    params = []
    for t, g in tmp_trn_corr.groupby("type", sort=False):
        b1, b2, intercept = robust_l1_two_feature_fit_per_type(g)
        params.append((t, b1, b2, intercept))
    params_df = pd.DataFrame(params, columns=["type", "b1_dist", "b2_inv", "intercept"])

    val_params = tmp_val_types.merge(params_df, on="type", how="left")
    oof_b1[val_mask] = val_params["b1_dist"].fillna(0.0).to_numpy(np.float64)
    oof_b2[val_mask] = val_params["b2_inv"].fillna(0.0).to_numpy(np.float64)
    oof_intercept[val_mask] = val_params["intercept"].fillna(0.0).to_numpy(np.float64)

train_pred_oof_full = (
    oof_base
    + oof_intercept
    + oof_b1 * train_feat["dist"].to_numpy(np.float64)
    + oof_b2 * inv_dist_train
)

print("OOF prediction describe:")
print(pd.Series(train_pred_oof_full).describe())

fine_stats_full = (
    train_feat.groupby(KEY_FINE)["scalar_coupling_constant"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "fine_mean", "count": "fine_count"})
    .reset_index()
)
fine_stats_full = fine_stats_full.merge(
    type_stats[["type", "type_mean_smooth", "type_smoothing"]], on="type", how="left"
)
fine_stats_full["fine_mean_smooth"] = (
    fine_stats_full["fine_mean"] * fine_stats_full["fine_count"]
    + fine_stats_full["type_mean_smooth"] * fine_stats_full["type_smoothing"]
) / (fine_stats_full["fine_count"] + fine_stats_full["type_smoothing"])

mid_stats_full = (
    train_feat.groupby(KEY_MID)["scalar_coupling_constant"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "mid_mean", "count": "mid_count"})
    .reset_index()
)
mid_stats_full = mid_stats_full.merge(
    type_stats[["type", "type_mean_smooth", "type_smoothing"]], on="type", how="left"
)
mid_stats_full["mid_mean_smooth"] = (
    mid_stats_full["mid_mean"] * mid_stats_full["mid_count"]
    + mid_stats_full["type_mean_smooth"] * mid_stats_full["type_smoothing"]
) / (mid_stats_full["mid_count"] + mid_stats_full["type_smoothing"])

coarse_stats_full = (
    train_feat.groupby(KEY_COARSE)["scalar_coupling_constant"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "coarse_mean", "count": "coarse_count"})
    .reset_index()
)
coarse_stats_full = coarse_stats_full.merge(
    type_stats[["type", "type_mean_smooth", "type_smoothing"]], on="type", how="left"
)
coarse_stats_full["coarse_mean_smooth"] = (
    coarse_stats_full["coarse_mean"] * coarse_stats_full["coarse_count"]
    + coarse_stats_full["type_mean_smooth"] * coarse_stats_full["type_smoothing"]
) / (coarse_stats_full["coarse_count"] + coarse_stats_full["type_smoothing"])

train_base_full = (
    train_feat[KEY_FINE]
    .merge(fine_stats_full[KEY_FINE + ["fine_mean_smooth"]], on=KEY_FINE, how="left")
    .merge(mid_stats_full[KEY_MID + ["mid_mean_smooth"]], on=KEY_MID, how="left")
    .merge(
        coarse_stats_full[KEY_COARSE + ["coarse_mean_smooth"]],
        on=KEY_COARSE,
        how="left",
    )
    .merge(type_stats[["type", "type_mean_smooth"]], on="type", how="left")
)

base_pred_train_full = (
    train_base_full["fine_mean_smooth"]
    .fillna(train_base_full["mid_mean_smooth"])
    .fillna(train_base_full["coarse_mean_smooth"])
    .fillna(train_base_full["type_mean_smooth"])
    .fillna(global_mean)
    .astype(np.float64)
    .to_numpy(np.float64)
)

train_resid_full = (
    train_feat["scalar_coupling_constant"].to_numpy(np.float64) - base_pred_train_full
)

train_tmp_full = pd.DataFrame(
    {
        "type": train_feat["type"].values,
        "dist": train_feat["dist"].values,
        "inv_dist": inv_dist_train,
        "resid": train_resid_full,
    }
)
params_full = []
for t, g in train_tmp_full.groupby("type", sort=False):
    b1, b2, intercept = robust_l1_two_feature_fit_per_type(g)
    params_full.append((t, b1, b2, intercept))
coef_df = pd.DataFrame(params_full, columns=["type", "b1_dist", "b2_inv", "intercept"])

test_base = (
    test_feat[KEY_FINE]
    .merge(fine_stats_full[KEY_FINE + ["fine_mean_smooth"]], on=KEY_FINE, how="left")
    .merge(mid_stats_full[KEY_MID + ["mid_mean_smooth"]], on=KEY_MID, how="left")
    .merge(
        coarse_stats_full[KEY_COARSE + ["coarse_mean_smooth"]],
        on=KEY_COARSE,
        how="left",
    )
    .merge(type_stats[["type", "type_mean_smooth"]], on="type", how="left")
)

base_pred = (
    test_base["fine_mean_smooth"]
    .fillna(test_base["mid_mean_smooth"])
    .fillna(test_base["coarse_mean_smooth"])
    .fillna(test_base["type_mean_smooth"])
    .fillna(global_mean)
    .astype(np.float64)
)

test_corr = test_feat[["type", "dist"]].merge(coef_df, on="type", how="left")
test_corr["b1_dist"] = test_corr["b1_dist"].fillna(0.0)
test_corr["b2_inv"] = test_corr["b2_inv"].fillna(0.0)
test_corr["intercept"] = test_corr["intercept"].fillna(0.0)

inv_dist_test = 1.0 / (test_feat["dist"].to_numpy(np.float64) + eps_inv)

test_pred = (
    base_pred.to_numpy(np.float64)
    + test_corr["intercept"].to_numpy(np.float64)
    + test_corr["b1_dist"].to_numpy(np.float64) * test_feat["dist"].to_numpy(np.float64)
    + test_corr["b2_inv"].to_numpy(np.float64) * inv_dist_test
).astype(np.float64)

print("Base prediction stats:", pd.Series(base_pred).describe())
print("Final prediction stats:", pd.Series(test_pred).describe())



## === cell 4
if "test_pred" not in globals():
    raise RuntimeError("test_pred was not computed; check earlier cells for errors.")
if len(test_pred) != len(test):
    raise ValueError(
        f"test_pred length {len(test_pred)} does not match test length {len(test)}"
    )

sub = sample[["id"]].merge(
    pd.DataFrame({"id": test["id"].values, "scalar_coupling_constant": test_pred}),
    on="id",
    how="left",
)

sub["scalar_coupling_constant"] = (
    sub["scalar_coupling_constant"]
    .fillna(train_feat["scalar_coupling_constant"].mean())
    .astype(np.float64)
)

if sub.shape[0] != sample.shape[0]:
    raise ValueError("Submission row count does not match sample_submission")
if sub["id"].isna().any() or sub["scalar_coupling_constant"].isna().any():
    raise ValueError("Submission contains NaNs")

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote", out_path, "with shape", sub.shape)
print(sub.head())



## === cell 5
print("submission scalar_coupling_constant describe:")
print(sub["scalar_coupling_constant"].describe())
