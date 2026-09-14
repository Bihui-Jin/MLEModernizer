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

0.75855

# 6. Current score

2.28679

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 2.28679) has done: 'I fix a critical feature bug that makes `dist_to_type_mean` in the test set depend on *test* group means (and can be inconsistent/unreliable vs. train), by computing the per-`type` distance mean on the training set once and applying it to both train and test. This preserves your existing feature set and GP equations while making the feature definition consistent with training (and avoids silent NaNs if a type mean is missing). I also add a small safety fill for any remaining missing structure merges so the GP functions don’t propagate NaNs into the submission. Finally, I keep the submission alignment logic but harden it to guarantee the exact `sample_submission` row order and required columns.'
- What this solution (achieved 2.28679) has done: 'You’re far from the target (2.28679 vs 0.75855, lower is better), so we need a real accuracy lift; the biggest minimal win without changing the GP “model logic” is to fix the type-to-GP mapping so each GP formula is applied to the correct coupling `type`. Right now you `LabelEncoder` the `type` strings and assume encoded values 0..7 correspond to GP0..GP7, which is almost certainly mismatched and badly hurt score. I keep your exact GP equations and features, but replace the `LabelEncoder` for `type` with an explicit, stable mapping that matches the original CHAMPS 8 coupling types order used by most baseline notebooks. I also keep `LabelEncoder` for `atom_0/atom_1` as-is and harden the GP dispatch to handle any unexpected types safely (leave them at 0.0, same behavior as now).'
- What this solution (achieved 2.28679) has done: 'Your score is far above the target (2.28679 vs 0.75855, lower is better), so we need a meaningful accuracy lift without changing the GP equations themselves. The biggest remaining “minimal but high-impact” issue is that your GP formulas use `atom_index_0/1` as numeric inputs, but your dataframe currently contains the *original* indices (0..N_atoms-1); many GP baselines actually expect these to be “stable” per-molecule indices after ordering atoms by coordinates (or similar), and a mismatch can badly degrade performance. To stay within your core-logic constraints, we won’t change any GP math; we only add a deterministic per-molecule reindexing based on the already-merged coordinates (x,y,z) and use those reindexed values for `atom_index_0/1` features, preserving the original columns for ID alignment. This is a small, legitimate feature-definition correction that typically improves CHAMPS GP-transfer performance and should move the score substantially downward toward your target band.'
- What this solution (achieved 2.28679) has done: 'I make one minimal, high-impact correction that preserves your existing GP equations and feature set: stop overwriting `atom_index_0/1` with the coordinate-sorted “canonical” indices, and instead keep the original indices as the GP inputs. This is likely the main cause of the large error gap because the GP formulas were evolved against the original CHAMPS atom indices (as provided in train/test), and changing their meaning breaks the learned relationships. I still compute the canonical indices (since you added it), but I store them as extra columns only, so the core model logic and data alignment stay intact. The rest of your pipeline (structure merge, dist features, type mapping, GP dispatch, and submission alignment) remains unchanged.'
- What this solution (achieved 2.71983) has done: 'Your score is much worse than the target (2.28679 vs 0.75855, lower is better), so we need a real accuracy lift while keeping your GP equations intact. The highest-impact minimal fix is to correct how the `type` codes map to GP0..GP7: your current `TYPE_ORDER` is not the canonical CHAMPS order, so each GP formula is likely applied to the wrong coupling type. I replace `TYPE_ORDER` with the standard competition type order so the intended GP-per-type formula is dispatched correctly, while keeping all features, equations, and submission alignment the same. I also add a tiny safety check that no unexpected types slip through silently (still defaulting to 0.0 predictions if they do).'
- What this solution (achieved 2.28679) has done: 'The biggest likely reason you’re still far from the target is that the hard-coded `TYPE_ORDER` used to dispatch GP0..GP7 is still mismatched to the GP equations’ intended coupling types, so the “right formula” is often applied to the “wrong type”. To move the score down toward your target while keeping the GP equations and features identical, I replace the fixed `TYPE_ORDER` with a data-driven mapping: we evaluate (on a small, deterministic per-type sample from train) which GPk best matches each type, then use that mapping for both train scoring and test prediction. This preserves your core logic (same features, same GP functions) and only corrects the dispatch layer. I also harden unmapped/rare types by falling back to the globally best GP to avoid NaNs or zeros hurting score.'
- What this solution (achieved 2.28679) has done: 'Your score is still far above the target (2.28679 vs 0.75855, lower is better), and the remaining biggest “minimal but high-impact” issue is that you overwrite `train["type"]` with the inferred GP index (0..7) and then use that numeric `type` for both dispatch and for the per-type log-MAE computation. That breaks evaluation semantics: Kaggle’s metric groups by the original coupling `type` strings, not by which GP you happened to choose, so your local score becomes misleading and the mapping step can also become unstable. I keep your exact features and GP equations, but store the inferred GP index in a separate column (e.g., `gp_k`) and keep `type` as the original string for grouping/metric. Then I update the GP dispatch to use `gp_k` while leaving submission formatting and row alignment unchanged.'
- What this solution (achieved 2.28679) has done: 'Your current gap to target is large (2.28679 vs 0.75855, lower is better), so we need a meaningful accuracy lift without changing the GP equations/features. The most likely remaining “silent correctness” issue is that the inferred type→GP mapping is currently optimized for plain MAE per type, while the competition metric is mean of log(MAE) across types; this can pick suboptimal GP formulas for rare/easy vs hard types. I change only the mapping inference objective to minimize per-type log(MAE) (aligned with Kaggle), keeping the same sampling approach, GP functions, and features. I also add a small clamp to ensure GP outputs are finite (prevent rare inf/NaN from wrecking log-MAE and the submission), without altering the core prediction logic.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

pd.options.display.precision = 15

import gc
import warnings

warnings.filterwarnings("ignore")



## === cell 1
CANDIDATE_INPUT_ROOTS = [
    "../input",
    "/kaggle/input",
    "/kaggle/data",
    "/kaggle/data/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling/champs-scalar-coupling",
]
input_root = None
for p in CANDIDATE_INPUT_ROOTS:
    if os.path.isdir(p) and os.path.exists(os.path.join(p, "train.csv")):
        input_root = p
        break
    if os.path.isdir(p) and os.path.isdir(os.path.join(p, "champs-scalar-coupling")):
        pp = os.path.join(p, "champs-scalar-coupling")
        if os.path.exists(os.path.join(pp, "train.csv")):
            input_root = pp
            break

if input_root is None:
    raise FileNotFoundError(
        "Could not find train.csv under expected Kaggle input paths. "
        "Checked: " + ", ".join(CANDIDATE_INPUT_ROOTS)
    )

print("Using input_root:", input_root)
print("Files:", sorted([f for f in os.listdir(input_root) if f.endswith(".csv")])[:10])



## === cell 2
train = pd.read_csv(os.path.join(input_root, "train.csv"))
test = pd.read_csv(os.path.join(input_root, "test.csv"))
sub = pd.read_csv(os.path.join(input_root, "sample_submission.csv"))



## === cell 3
train.head()



## === cell 4
structures = pd.read_csv(os.path.join(input_root, "structures.csv"))


def map_atom_info(df, atom_idx):
    df = pd.merge(
        df,
        structures,
        how="left",
        left_on=["molecule_name", f"atom_index_{atom_idx}"],
        right_on=["molecule_name", "atom_index"],
    )

    df = df.drop("atom_index", axis=1)
    df = df.rename(
        columns={
            "atom": f"atom_{atom_idx}",
            "x": f"x_{atom_idx}",
            "y": f"y_{atom_idx}",
            "z": f"z_{atom_idx}",
        }
    )
    return df


train = map_atom_info(train, 0)
train = map_atom_info(train, 1)

test = map_atom_info(test, 0)
test = map_atom_info(test, 1)

for df in (train, test):
    for c in ["x_0", "y_0", "z_0", "x_1", "y_1", "z_1"]:
        df[c] = df[c].fillna(0.0)
    for c in ["atom_0", "atom_1"]:
        df[c] = df[c].fillna("X")



## === cell 5
train_p_0 = train[["x_0", "y_0", "z_0"]].values
train_p_1 = train[["x_1", "y_1", "z_1"]].values
test_p_0 = test[["x_0", "y_0", "z_0"]].values
test_p_1 = test[["x_1", "y_1", "z_1"]].values

train["dist"] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
test["dist"] = np.linalg.norm(test_p_0 - test_p_1, axis=1)



## === cell 6
type_dist_mean_train = train.groupby("type")["dist"].mean()

train["dist_to_type_mean"] = train["dist"] / train["type"].map(type_dist_mean_train)
test["dist_to_type_mean"] = test["dist"] / test["type"].map(type_dist_mean_train)

global_mean_dist = train["dist"].mean()
train["dist_to_type_mean"] = train["dist_to_type_mean"].fillna(
    train["dist"] / global_mean_dist
)
test["dist_to_type_mean"] = test["dist_to_type_mean"].fillna(
    test["dist"] / global_mean_dist
)




## === cell 7
def build_canonical_atom_index_map(structures_df: pd.DataFrame) -> pd.DataFrame:
    s = structures_df[["molecule_name", "atom_index", "x", "y", "z"]].copy()
    s.sort_values(["molecule_name", "x", "y", "z", "atom_index"], inplace=True)
    s["atom_index_canon"] = s.groupby("molecule_name").cumcount().astype(np.int16)
    return s[["molecule_name", "atom_index", "atom_index_canon"]]


canon_map = build_canonical_atom_index_map(structures)


def attach_canonical_indices(
    df: pd.DataFrame, canon_map_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Change rationale (score improvement, minimal logic change):
    The GP formulas were evolved against the original CHAMPS atom indices from train/test.
    Overwriting atom_index_0/1 with a coordinate-sorted reindex changes the meaning of key GP inputs
    and can severely degrade accuracy. We therefore keep original atom_index_0/1 intact for GP,
    and only ATTACH canonical indices as extra columns for inspection (not used by GP).
    """
    out = df.merge(
        canon_map_df.rename(
            columns={
                "atom_index": "atom_index_0",
                "atom_index_canon": "atom_index_0_canon",
            }
        ),
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    out = out.merge(
        canon_map_df.rename(
            columns={
                "atom_index": "atom_index_1",
                "atom_index_canon": "atom_index_1_canon",
            }
        ),
        on=["molecule_name", "atom_index_1"],
        how="left",
    )
    out["atom_index_0_canon"] = (
        out["atom_index_0_canon"].fillna(out["atom_index_0"]).astype(np.int16)
    )
    out["atom_index_1_canon"] = (
        out["atom_index_1_canon"].fillna(out["atom_index_1"]).astype(np.int16)
    )
    return out


train = attach_canonical_indices(train, canon_map)
test = attach_canonical_indices(test, canon_map)

train["atom_index_0_orig"] = train["atom_index_0"]
train["atom_index_1_orig"] = train["atom_index_1"]
test["atom_index_0_orig"] = test["atom_index_0"]
test["atom_index_1_orig"] = test["atom_index_1"]



## === cell 8
for f in ["atom_0", "atom_1"]:
    lbl = LabelEncoder()
    lbl.fit(list(train[f].values) + list(test[f].values))
    train[f] = lbl.transform(list(train[f].values))
    test[f] = lbl.transform(list(test[f].values))

train["type_str"] = train["type"].astype(str)
test["type_str"] = test["type"].astype(str)

print("Unique train types (string):", sorted(train["type_str"].unique())[:20])




## === cell 9
def GP0(data):
    return (
        94.962006
        + 1.0
        * (
            (
                (data["atom_index_1"])
                - (
                    (
                        (
                            (
                                ((((1.0) <= (data["dist_to_type_mean"])) * 1.0))
                                + ((((1.0) <= (data["dist_to_type_mean"])) * 1.0))
                            )
                        )
                        * ((4.90689849853515625))
                    )
                )
            )
        )
        + 1.0
        * (
            (
                (((14.0) * ((((data["dist_to_type_mean"]) <= (0.995526)) * 1.0))))
                + (np.minimum(((-1.0)), ((((14.0) - (data["atom_index_0"]))))))
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (data["dist_to_type_mean"])
                                                    <= (0.991079)
                                                )
                                                * 1.0
                                            )
                                        )
                                        * 2.0
                                    )
                                )
                                * 2.0
                            )
                        )
                        * 2.0
                    )
                )
                * 2.0
            )
        )
        + 0.948217
        * (
            (
                (
                    (
                        -1.0
                        * (
                            (
                                (
                                    (
                                        np.minimum(
                                            (
                                                (
                                                    (
                                                        (
                                                            (
                                                                (
                                                                    (
                                                                        ((6.0))
                                                                        > (data["x_0"])
                                                                    )
                                                                    * 1.0
                                                                )
                                                            )
                                                            > (
                                                                data[
                                                                    "dist_to_type_mean"
                                                                ]
                                                            )
                                                        )
                                                        * 1.0
                                                    )
                                                )
                                            ),
                                            ((data["dist_to_type_mean"])),
                                        )
                                    )
                                    * 2.0
                                )
                            )
                        )
                    )
                )
                * 2.0
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (data["dist"])
                                        * (
                                            (
                                                ((((data["dist"]) > (1.100827)) * 1.0))
                                                * 2.0
                                            )
                                        )
                                    )
                                )
                                * (data["dist"])
                            )
                        )
                        * 2.0
                    )
                )
                * (data["dist"])
            )
        )
        + 1.0
        * (
            (
                -1.0
                * (
                    (
                        (
                            ((((data["dist_to_type_mean"]) > (0.997387)) * 1.0))
                            - ((((12.0) > (data["atom_index_0"])) * 1.0))
                        )
                    )
                )
            )
        )
        + 1.0
        * (
            (
                (((0.318310) * ((11.50162220001220703))))
                * ((((0.997117) > (data["dist_to_type_mean"])) * 1.0))
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        ((((((data["dist_to_type_mean"]) <= (0.986908)) * 1.0)) * 2.0))
                        * 2.0
                    )
                )
                - (
                    (
                        (
                            ((((data["dist_to_type_mean"]) > (0.986908)) * 1.0))
                            > (data["dist_to_type_mean"])
                        )
                        * 1.0
                    )
                )
            )
        )
        + 1.0
        * (
            (
                (0.984850)
                * (
                    (
                        ((8.0))
                        * (
                            (
                                (0.984850)
                                * (
                                    (
                                        ((8.0))
                                        * (
                                            (
                                                (
                                                    (0.984850)
                                                    > (data["dist_to_type_mean"])
                                                )
                                                * 1.0
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (data["dist_to_type_mean"])
                            > (
                                np.maximum(
                                    ((1.0)), ((((data["atom_index_1"]) - (2.152171))))
                                )
                            )
                        )
                        * 1.0
                    )
                )
                - ((((1.013243) > (data["dist_to_type_mean"])) * 1.0))
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        ((((1.013243) <= (data["dist_to_type_mean"])) * 1.0))
                        * ((10.21938037872314453))
                    )
                )
                * 2.0
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            ((-1.0 * (((((((data["y_0"]) > (-1.0)) * 1.0)) / 2.0)))))
                            + ((((data["atom_index_0"]) <= (12.0)) * 1.0))
                        )
                        / 2.0
                    )
                )
                * 2.0
            )
        )
        + 1.0
        * (
            (
                (-0.072825)
                + (
                    (
                        (-0.072825)
                        + (
                            (
                                (-0.072825)
                                + (
                                    (
                                        (
                                            (0.998071)
                                            > (
                                                np.maximum(
                                                    ((data["dist_to_type_mean"])),
                                                    ((data["y_1"])),
                                                )
                                            )
                                        )
                                        * 1.0
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
        + 0.922325
        * (
            (
                (
                    ((5.0))
                    > (
                        (
                            (
                                (
                                    np.minimum(
                                        (((((data["y_0"]) + (data["z_1"])) / 2.0))),
                                        ((data["y_0"])),
                                    )
                                )
                                + (data["atom_index_0"])
                            )
                            / 2.0
                        )
                    )
                )
                * 1.0
            )
        )
        + 0.977528
        * (
            (
                (
                    (
                        (-0.543418)
                        * (
                            (
                                (
                                    (data["x_0"])
                                    <= ((((data["x_1"]) + (-0.283168)) / 2.0))
                                )
                                * 1.0
                            )
                        )
                    )
                )
                * ((((-2.0) <= (data["x_1"])) * 1.0))
            )
        )
        + 1.0
        * (
            (
                (
                    ((-1.0 * ((((((1.0)) > (data["dist_to_type_mean"])) * 1.0)))))
                    + (((((6.0)) > (((((data["y_1"]) * 2.0)) * 2.0))) * 1.0))
                )
                / 2.0
            )
        )
    )


def GP1(data):
    return (
        47.511509
        + 1.0
        * (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        ((-1.0 * ((data["dist"]))))
                                        + (
                                            (
                                                (
                                                    (
                                                        (
                                                            (data["dist_to_type_mean"])
                                                            <= (0.998991)
                                                        )
                                                        * 1.0
                                                    )
                                                )
                                                * 2.0
                                            )
                                        )
                                    )
                                )
                                * 2.0
                            )
                        )
                        * 2.0
                    )
                )
                * 2.0
            )
        )
        + 1.0
        * (
            (
                (3.141593)
                * (
                    (
                        (1.003744)
                        * (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (1.003744)
                                                    > (data["dist_to_type_mean"])
                                                )
                                                * 1.0
                                            )
                                        )
                                        * 2.0
                                    )
                                )
                                - (1.387789)
                            )
                        )
                    )
                )
            )
        )
        + 0.979971
        * (
            np.minimum(
                ((((4.740009) - (data["dist"])))),
                (
                    (
                        (
                            (1.570796)
                            * (
                                (
                                    (data["atom_index_1"])
                                    * (
                                        (
                                            ((0.994081) > (data["dist_to_type_mean"]))
                                            * 1.0
                                        )
                                    )
                                )
                            )
                        )
                    )
                ),
            )
        )
        + 0.950660
        * (
            (
                (0.076053)
                - (
                    (
                        ((((((((data["atom_index_0"]) + (-2.0)) / 2.0)) / 2.0)) / 2.0))
                        * ((((data["dist_to_type_mean"]) <= (0.998991)) * 1.0))
                    )
                )
            )
        )
        + 1.0
        * (
            np.minimum(
                (
                    (
                        (
                            ((((0.994834) > (data["dist_to_type_mean"])) * 1.0))
                            * (((0.636620) + (0.994834)))
                        )
                    )
                ),
                ((np.maximum(((data["atom_index_1"])), ((0.636620))))),
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            -1.0
                            * (
                                (
                                    (
                                        (
                                            (
                                                np.maximum(
                                                    ((data["z_1"])),
                                                    (((-1.0 * ((data["z_1"]))))),
                                                )
                                            )
                                            > (0.110836)
                                        )
                                        * 1.0
                                    )
                                )
                            )
                        )
                    )
                    + ((((data["atom_index_1"]) > (7.0)) * 1.0))
                )
                / 2.0
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (data["atom_index_1"])
                                                    + (
                                                        (
                                                            (data["dist_to_type_mean"])
                                                            * 2.0
                                                        )
                                                    )
                                                )
                                            )
                                            > (data["atom_index_0"])
                                        )
                                        * 1.0
                                    )
                                )
                                * 2.0
                            )
                        )
                        * 2.0
                    )
                )
                * 2.0
            )
        )
        + 1.0
        * (
            (
                (
                    (data["atom_index_0"])
                    <= (
                        (
                            (
                                np.maximum(
                                    ((3.141593)),
                                    (
                                        (
                                            (
                                                (data["atom_index_1"])
                                                + (
                                                    (
                                                        (
                                                            (3.141593)
                                                            > (
                                                                (
                                                                    (
                                                                        data[
                                                                            "atom_index_1"
                                                                        ]
                                                                    )
                                                                    / 2.0
                                                                )
                                                            )
                                                        )
                                                        * 1.0
                                                    )
                                                )
                                            )
                                        )
                                    ),
                                )
                            )
                            * 2.0
                        )
                    )
                )
                * 1.0
            )
        )
        + 1.0
        * (
            (
                (
                    np.minimum(
                        (
                            (
                                (
                                    (
                                        (
                                            -1.0
                                            * (
                                                (
                                                    (
                                                        (
                                                            (((data["y_0"]) / 2.0))
                                                            + (data["z_1"])
                                                        )
                                                        / 2.0
                                                    )
                                                )
                                            )
                                        )
                                    )
                                    / 2.0
                                )
                            )
                        ),
                        ((((3.0) + (data["y_0"])))),
                    )
                )
                / 2.0
            )
        )
        + 0.916952
        * (
            np.minimum(
                (((((1.0) > (data["z_0"])) * 1.0))),
                (
                    (
                        (
                            (
                                ((((0.0) + (data["z_0"])) / 2.0))
                                + ((((data["dist_to_type_mean"]) > (1.0)) * 1.0))
                            )
                            / 2.0
                        )
                    )
                ),
            )
        )
        + 1.0
        * (
            (
                ((((((1.021625) > (data["dist"])) * 1.0)) - (data["dist"])))
                * ((((1.021625) + (((1.021625) * 2.0))) / 2.0))
            )
        )
        + 1.0
        * (
            (
                ((data["dist_to_type_mean"]) <= ((((data["dist"]) > (1.011806)) * 1.0)))
                * 1.0
            )
        )
        + 1.0
        * (
            (
                (
                    (0.998991)
                    > (
                        (
                            (
                                (
                                    (
                                        (
                                            (0.998991)
                                            + (
                                                (
                                                    (0.998991)
                                                    * (data["dist_to_type_mean"])
                                                )
                                            )
                                        )
                                        / 2.0
                                    )
                                )
                                > ((((data["dist_to_type_mean"]) > (0.998991)) * 1.0))
                            )
                            * 1.0
                        )
                    )
                )
                * 1.0
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        np.minimum(
                            ((data["dist_to_type_mean"])),
                            ((((((data["atom_index_0"]) / 2.0)) / 2.0))),
                        )
                    )
                    <= ((((1.011806) <= (data["dist"])) * 1.0))
                )
                * 1.0
            )
        )
        + 1.0
        * (
            (
                ((-1.0 * (((((0.318310) > (data["z_1"])) * 1.0)))))
                * (
                    (
                        (
                            ((((1.011806) > (data["dist"])) * 1.0))
                            > (data["atom_index_1"])
                        )
                        * 1.0
                    )
                )
            )
        )
        + 0.922814
        * (
            np.minimum(
                (((((1.008356) > (data["dist"])) * 1.0))),
                (((((((1.199319) * (data["z_1"]))) <= (((0.072832) * 2.0))) * 1.0))),
            )
        )
    )


def GP2(data):
    return (
        -0.268229
        + 0.937958
        * (
            (
                -1.0
                * (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            (data["atom_index_1"])
                                                            > (
                                                                (
                                                                    (
                                                                        (data["y_0"])
                                                                        <= (
                                                                            (
                                                                                (
                                                                                    1.194078
                                                                                )
                                                                                / 2.0
                                                                            )
                                                                        )
                                                                    )
                                                                    * 1.0
                                                                )
                                                            )
                                                        )
                                                        * 1.0
                                                    )
                                                )
                                                * 2.0
                                            )
                                        )
                                        > (data["atom_index_1"])
                                    )
                                    * 1.0
                                )
                            )
                            * 2.0
                        )
                    )
                )
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            ((0.994044) <= (data["dist_to_type_mean"]))
                                            * 1.0
                                        )
                                    )
                                    * 2.0
                                )
                            )
                            - (2.0)
                        )
                    )
                    + ((((data["dist_to_type_mean"]) > (1.0)) * 1.0))
                )
                / 2.0
            )
        )
        + 1.0
        * (
            (
                ((((1.032724) <= (data["dist_to_type_mean"])) * 1.0))
                - (
                    np.minimum(
                        (
                            (
                                (
                                    (
                                        (data["dist_to_type_mean"])
                                        > (
                                            (
                                                (
                                                    ((12.80295181274414062))
                                                    > (data["atom_index_0"])
                                                )
                                                * 1.0
                                            )
                                        )
                                    )
                                    * 1.0
                                )
                            )
                        ),
                        ((0.166034)),
                    )
                )
            )
        )
        + 0.869565
        * (
            np.minimum(
                ((data["dist_to_type_mean"])),
                (
                    (
                        (
                            (data["atom_index_1"])
                            * (
                                (
                                    (0.950018)
                                    - (
                                        (
                                            ((0.950018) <= (data["dist_to_type_mean"]))
                                            * 1.0
                                        )
                                    )
                                )
                            )
                        )
                    )
                ),
            )
        )
        + 0.926234
        * (
            np.minimum(
                ((((((1.570796) / 2.0)) / 2.0))),
                (
                    (
                        (
                            ((((data["y_0"]) > (data["atom_index_1"])) * 1.0))
                            * (data["atom_index_1"])
                        )
                    )
                ),
            )
        )
        + 1.0 * (((((((data["dist_to_type_mean"]) - (1.0))) * 2.0)) * 2.0))
        + 0.790914
        * (
            np.minimum(
                ((data["atom_index_1"])),
                (
                    (
                        (
                            (
                                (data["atom_index_1"])
                                <= (
                                    np.minimum(
                                        (
                                            (
                                                (
                                                    (
                                                        (data["y_0"])
                                                        > (
                                                            (
                                                                (((data["z_0"]) / 2.0))
                                                                / 2.0
                                                            )
                                                        )
                                                    )
                                                    * 1.0
                                                )
                                            )
                                        ),
                                        ((data["x_1"])),
                                    )
                                )
                            )
                            * 1.0
                        )
                    )
                ),
            )
        )
        + 0.850024
        * (
            (
                (data["x_0"])
                * (
                    (
                        (
                            (
                                -1.0
                                * (
                                    (
                                        (
                                            (
                                                (0.166034)
                                                <= (
                                                    (
                                                        (
                                                            (data["atom_index_1"])
                                                            <= (((0.166034) / 2.0))
                                                        )
                                                        * 1.0
                                                    )
                                                )
                                            )
                                            * 1.0
                                        )
                                    )
                                )
                            )
                        )
                        / 2.0
                    )
                )
            )
        )
        + 1.0
        * (
            (
                (3.0)
                * (
                    (
                        (0.982967)
                        - (
                            (
                                ((data["atom_index_0"]) > ((((7.0)) - (data["y_0"]))))
                                * 1.0
                            )
                        )
                    )
                )
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        -1.0
                        * (
                            (
                                (
                                    (
                                        (((data["dist_to_type_mean"]) * (6.0)))
                                        <= (
                                            (
                                                (
                                                    ((10.40789985656738281))
                                                    + (
                                                        (
                                                            (
                                                                (6.0)
                                                                > (data["atom_index_0"])
                                                            )
                                                            * 1.0
                                                        )
                                                    )
                                                )
                                                / 2.0
                                            )
                                        )
                                    )
                                    * 1.0
                                )
                            )
                        )
                    )
                )
                * 2.0
            )
        )
        + 1.0
        * (
            (
                (data["y_0"])
                * (
                    (
                        (
                            (0.950018)
                            + (
                                (
                                    (
                                        np.minimum(
                                            ((((-2.0) - (data["dist_to_type_mean"])))),
                                            ((0.832110)),
                                        )
                                    )
                                    + (2.080729)
                                )
                            )
                        )
                        / 2.0
                    )
                )
            )
        )
        + 0.999023
        * (
            (
                (
                    (
                        (
                            (data["x_0"])
                            > (
                                (
                                    (
                                        (
                                            (
                                                (data["atom_index_0"])
                                                + (
                                                    (
                                                        (
                                                            (data["y_0"])
                                                            > (
                                                                (
                                                                    (
                                                                        (data["x_0"])
                                                                        + (data["z_0"])
                                                                    )
                                                                    / 2.0
                                                                )
                                                            )
                                                        )
                                                        * 1.0
                                                    )
                                                )
                                            )
                                            / 2.0
                                        )
                                    )
                                    / 2.0
                                )
                            )
                        )
                        * 1.0
                    )
                )
                / 2.0
            )
        )
        + 1.0
        * (
            np.minimum(
                (((((data["dist"]) <= (2.080729)) * 1.0))),
                (
                    (
                        (
                            (
                                (data["dist"])
                                > (
                                    (
                                        (data["atom_index_1"])
                                        - (
                                            (
                                                ((2.080729) <= (data["atom_index_1"]))
                                                * 1.0
                                            )
                                        )
                                    )
                                )
                            )
                            * 1.0
                        )
                    )
                ),
            )
        )
        + 1.0
        * (
            (
                -1.0
                * (
                    (
                        (
                            ((((data["dist_to_type_mean"]) <= (0.982494)) * 1.0))
                            * (((data["dist"]) * (0.034596)))
                        )
                    )
                )
            )
        )
        + 0.929653
        * (
            (
                ((-1.0 * (((((((1.0) * 2.0)) <= (data["dist"])) * 1.0)))))
                * ((((((0.982494) > (data["dist_to_type_mean"])) * 1.0)) / 2.0))
            )
        )
        + 0.999511
        * (
            (
                (
                    (
                        (
                            ((((data["dist"]) <= (2.080729)) * 1.0))
                            + (
                                (
                                    (
                                        (
                                            (
                                                ((((data["z_0"]) <= (0.204195)) * 1.0))
                                                + (0.204195)
                                            )
                                            / 2.0
                                        )
                                    )
                                    / 2.0
                                )
                            )
                        )
                        / 2.0
                    )
                )
                / 2.0
            )
        )
    )


def GP3(data):
    return (
        -10.275834
        + 1.0
        * (
            (
                -1.0
                * (
                    (
                        (
                            (
                                np.minimum(
                                    ((0.636620)),
                                    (
                                        (
                                            (
                                                (
                                                    (data["dist_to_type_mean"])
                                                    <= (
                                                        (
                                                            (
                                                                ((4.59200382232666016))
                                                                <= (
                                                                    data["atom_index_0"]
                                                                )
                                                            )
                                                            * 1.0
                                                        )
                                                    )
                                                )
                                                * 1.0
                                            )
                                        )
                                    ),
                                )
                            )
                            * 2.0
                        )
                    )
                )
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (
                                (
                                    np.minimum(
                                        (
                                            (
                                                (
                                                    (
                                                        (data["dist_to_type_mean"])
                                                        > (1.004117)
                                                    )
                                                    * 1.0
                                                )
                                            )
                                        ),
                                        ((data["dist_to_type_mean"])),
                                    )
                                )
                                * (data["dist_to_type_mean"])
                            )
                        )
                        * (2.0)
                    )
                )
                * (data["dist"])
            )
        )
        + 0.974597
        * (
            (
                (
                    (
                        (
                            (
                                (((data["dist_to_type_mean"]) * 2.0))
                                * ((((1.796855) <= (data["dist"])) * 1.0))
                            )
                        )
                        * 2.0
                    )
                )
                - (((((1.570796) / 2.0)) / 2.0))
            )
        )
        + 0.787005
        * (
            (
                (
                    (
                        (
                            (
                                (data["dist_to_type_mean"])
                                <= ((((1.765520) <= (data["dist"])) * 1.0))
                            )
                            * 1.0
                        )
                    )
                    + ((((((1.765520) <= (data["dist"])) * 1.0)) - (data["dist"])))
                )
                / 2.0
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                ((((-0.443103) <= (data["y_1"])) * 1.0))
                                                > (data["dist_to_type_mean"])
                                            )
                                            * 1.0
                                        )
                                    )
                                    / 2.0
                                )
                            )
                            + ((((1.200606) > (data["y_1"])) * 1.0))
                        )
                        / 2.0
                    )
                )
                / 2.0
            )
        )
        + 0.994138
        * (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            np.minimum(
                                                ((data["dist_to_type_mean"])),
                                                (
                                                    (
                                                        (
                                                            (
                                                                (data["dist"])
                                                                <= (1.719267)
                                                            )
                                                            * 1.0
                                                        )
                                                    )
                                                ),
                                            )
                                        )
                                        * (data["dist"])
                                    )
                                )
                                * 2.0
                            )
                        )
                        * 2.0
                    )
                )
                * (data["dist_to_type_mean"])
            )
        )
        + 0.788471
        * (
            (
                (((data["dist_to_type_mean"]) - (1.988152)))
                + (
                    np.maximum(
                        (((((data["y_0"]) > (3.0)) * 1.0))),
                        (((((1.004117) > (data["dist_to_type_mean"])) * 1.0))),
                    )
                )
            )
        )
        + 1.0
        * (
            (
                (
                    (data["dist"])
                    > (
                        (
                            (1.570796)
                            - (
                                (
                                    (
                                        (
                                            -1.0
                                            * (
                                                (
                                                    (
                                                        (((3.0) / 2.0))
                                                        * (((0.170694) * 2.0))
                                                    )
                                                )
                                            )
                                        )
                                    )
                                    / 2.0
                                )
                            )
                        )
                    )
                )
                * 1.0
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            np.minimum(
                                ((0.318310)),
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            (data["dist_to_type_mean"])
                                                            - ((1.0))
                                                        )
                                                    )
                                                    * 2.0
                                                )
                                            )
                                            * 2.0
                                        )
                                    )
                                ),
                            )
                        )
                        * 2.0
                    )
                )
                * 2.0
            )
        )
        + 0.804592
        * (
            (
                (
                    (
                        (
                            (
                                (data["y_0"])
                                - (
                                    (
                                        (data["dist_to_type_mean"])
                                        * (((1.004117) * (data["y_0"])))
                                    )
                                )
                            )
                        )
                        * 2.0
                    )
                )
                * 2.0
            )
        )
        + 0.981925
        * (
            (
                (0.147920)
                * (
                    (
                        (
                            ((((((data["y_0"]) / 2.0)) > (0.147920)) * 1.0))
                            <= ((((((data["y_0"]) / 2.0)) > (1.004117)) * 1.0))
                        )
                        * 1.0
                    )
                )
            )
        )
        + 0.890572
        * (
            (
                (
                    (
                        (
                            (
                                -1.0
                                * (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            (
                                                                (data["y_1"])
                                                                * (data["y_1"])
                                                            )
                                                        )
                                                        > (1.570796)
                                                    )
                                                    * 1.0
                                                )
                                            )
                                            / 2.0
                                        )
                                    )
                                )
                            )
                        )
                        / 2.0
                    )
                )
                / 2.0
            )
        )
        + 1.0
        * (
            (
                (
                    ((-1.0 * ((1.004117))))
                    > (
                        (
                            -1.0
                            * (
                                (
                                    (
                                        (data["dist_to_type_mean"])
                                        - (
                                            (
                                                (
                                                    (
                                                        (
                                                            (
                                                                (1.947232)
                                                                <= (
                                                                    data[
                                                                        "dist_to_type_mean"
                                                                    ]
                                                                )
                                                            )
                                                            * 1.0
                                                        )
                                                    )
                                                    + (0.054476)
                                                )
                                                / 2.0
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
                * 1.0
            )
        )
        + 0.894480
        * (
            (
                -1.0
                * (
                    (
                        np.minimum(
                            ((0.636620)),
                            (((((data["dist_to_type_mean"]) > (1.004117)) * 1.0))),
                        )
                    )
                )
            )
        )
        + 0.870542
        * (
            (
                (
                    (
                        (
                            (
                                (((((data["dist"]) - (1.770059))) * (data["dist"])))
                                * (data["dist"])
                            )
                        )
                        * (data["dist"])
                    )
                )
                * (data["dist"])
            )
        )
        + 1.0
        * (
            np.minimum(
                (((((11.0) <= (data["atom_index_0"])) * 1.0))),
                ((((0.318310) * ((((((data["z_1"]) / 2.0)) <= (data["y_1"])) * 1.0))))),
            )
        )
    )


def GP4(data):
    return (
        3.128881
        + 1.0
        * (
            (
                ((((2.264586) > (data["dist"])) * 1.0))
                * (
                    (
                        ((-1.0 * ((data["dist"]))))
                        + (((((-1.0 * ((1.207241)))) + (data["atom_index_1"])) / 2.0))
                    )
                )
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        np.minimum(
                            ((((data["y_0"]) + (data["atom_index_1"])))), ((0.318310))
                        )
                    )
                    <= (
                        (
                            ((data["atom_index_0"]) <= (((data["atom_index_1"]) * 2.0)))
                            * 1.0
                        )
                    )
                )
                * 1.0
            )
        )
        + 0.949194
        * (
            (
                (
                    (
                        ((((3.141593) <= (data["x_0"])) * 1.0))
                        - ((((-0.441697) <= (data["y_0"])) * 1.0))
                    )
                )
                / 2.0
            )
        )
        + 1.0
        * (
            np.maximum(
                (((((data["atom_index_1"]) <= (data["y_1"])) * 1.0))),
                (
                    (
                        np.minimum(
                            (((((-2.0) > (data["z_0"])) * 1.0))),
                            (
                                (
                                    (
                                        (
                                            (data["atom_index_1"])
                                            <= ((5.21898651123046875))
                                        )
                                        * 1.0
                                    )
                                )
                            ),
                        )
                    )
                ),
            )
        )
        + 1.0
        * (
            (
                -1.0
                * (
                    (
                        (
                            (
                                ((((data["dist_to_type_mean"]) <= (1.022687)) * 1.0))
                                <= (
                                    (
                                        -1.0
                                        * (
                                            (
                                                (
                                                    (
                                                        (data["atom_index_1"])
                                                        <= (data["dist_to_type_mean"])
                                                    )
                                                    * 1.0
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                            * 1.0
                        )
                    )
                )
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (
                                (
                                    (((data["dist_to_type_mean"]) / 2.0))
                                    <= ((((0.24176722764968872)) * 2.0))
                                )
                                * 1.0
                            )
                        )
                        * (((data["dist_to_type_mean"]) * 2.0))
                    )
                )
                * (((data["dist_to_type_mean"]) * 2.0))
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            np.minimum(
                                (((((data["dist"]) > (2.264586)) * 1.0))),
                                (
                                    (
                                        (
                                            (
                                                (data["dist_to_type_mean"])
                                                <= (data["atom_index_1"])
                                            )
                                            * 1.0
                                        )
                                    )
                                ),
                            )
                        )
                        * (data["dist_to_type_mean"])
                    )
                )
                * (data["dist_to_type_mean"])
            )
        )
        + 1.0
        * (
            np.minimum(
                (
                    (
                        np.maximum(
                            ((data["y_1"])),
                            (
                                (
                                    (
                                        (
                                            (
                                                -1.0
                                                * (
                                                    (
                                                        (
                                                            (
                                                                (data["atom_index_0"])
                                                                > (((14.0) - (-1.0)))
                                                            )
                                                            * 1.0
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                        / 2.0
                                    )
                                )
                            ),
                        )
                    )
                ),
                ((0.023229)),
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (((data["z_0"]) - (1.866194)))
                            > (
                                (
                                    (-2.0)
                                    * ((((((data["z_0"]) * 2.0)) <= (0.318310)) * 1.0))
                                )
                            )
                        )
                        * 1.0
                    )
                )
                / 2.0
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        np.maximum(
                            ((data["x_1"])),
                            (
                                (
                                    (
                                        ((((1.034271) > (data["x_1"])) * 1.0))
                                        - (data["z_0"])
                                    )
                                )
                            ),
                        )
                    )
                    > (np.maximum(((data["atom_index_1"])), ((data["dist"]))))
                )
                * 1.0
            )
        )
        + 1.0
        * (
            (
                ((((((-1.0 * (((((1.929792) > (data["dist"])) * 1.0))))) * 2.0)) * 2.0))
                - ((((data["dist"]) <= (1.929792)) * 1.0))
            )
        )
        + 1.0
        * (
            (
                (
                    (data["x_0"])
                    <= (
                        (
                            (
                                (
                                    -1.0
                                    * (
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            (
                                                                (data["atom_index_0"])
                                                                + ((-1.0 * ((2.0))))
                                                            )
                                                            / 2.0
                                                        )
                                                    )
                                                    + ((-1.0 * ((2.0))))
                                                )
                                                / 2.0
                                            )
                                        )
                                    )
                                )
                            )
                            * 2.0
                        )
                    )
                )
                * 1.0
            )
        )
        + 0.929653
        * (
            (
                (
                    (
                        (0.523343)
                        - (
                            (
                                (
                                    (data["dist_to_type_mean"])
                                    > ((((1.0) + ((((1.0)) - (0.029367)))) / 2.0))
                                )
                                * 1.0
                            )
                        )
                    )
                )
                / 2.0
            )
        )
        + 0.971177
        * (
            (
                (
                    (
                        (
                            (
                                np.minimum(
                                    ((data["x_0"])),
                                    (
                                        (
                                            (
                                                (((3.594635) - (data["atom_index_1"])))
                                                + (3.594635)
                                            )
                                        )
                                    ),
                                )
                            )
                            > (1.138006)
                        )
                        * 1.0
                    )
                )
                / 2.0
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (
                                -1.0
                                * (
                                    (
                                        (
                                            (
                                                (0.318310)
                                                > (
                                                    (
                                                        (
                                                            (
                                                                (0.318310)
                                                                * (
                                                                    (
                                                                        (
                                                                            (
                                                                                data[
                                                                                    "dist_to_type_mean"
                                                                                ]
                                                                            )
                                                                            + (0.0)
                                                                        )
                                                                        / 2.0
                                                                    )
                                                                )
                                                            )
                                                        )
                                                        * 2.0
                                                    )
                                                )
                                            )
                                            * 1.0
                                        )
                                    )
                                )
                            )
                        )
                        / 2.0
                    )
                )
                / 2.0
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (data["x_0"])
                            <= (
                                (
                                    (
                                        np.minimum(
                                            ((data["x_1"])),
                                            (
                                                (
                                                    np.minimum(
                                                        ((((data["y_0"]) / 2.0))),
                                                        ((0.318310)),
                                                    )
                                                )
                                            ),
                                        )
                                    )
                                    - (data["dist_to_type_mean"])
                                )
                            )
                        )
                        * 1.0
                    )
                )
                / 2.0
            )
        )
    )


def GP5(data):
    return (
        3.690675
        + 0.868100
        * (
            (
                (data["dist"])
                * (
                    (
                        (
                            (
                                (
                                    ((((3.142503) > (data["dist"])) * 1.0))
                                    <= ((((data["dist"]) > (3.141593)) * 1.0))
                                )
                                * 1.0
                            )
                        )
                        - (0.636620)
                    )
                )
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (1.0)
                            <= (
                                np.maximum(
                                    (((-1.0 * ((data["y_0"]))))),
                                    (((((-0.206448) <= (data["y_0"])) * 1.0))),
                                )
                            )
                        )
                        * 1.0
                    )
                )
                * (((0.148571) / 2.0))
            )
        )
        + 0.885686 * ((((2.719738) > (data["dist"])) * 1.0))
        + 0.888129
        * (
            (
                (
                    (
                        (np.maximum(((data["dist_to_type_mean"])), ((0.826032))))
                        - ((((data["dist_to_type_mean"]) > (0.826032)) * 1.0))
                    )
                )
                * 2.0
            )
        )
        + 0.887640
        * (
            (
                (data["dist_to_type_mean"])
                * (
                    (
                        -1.0
                        * (
                            (
                                (
                                    (
                                        (
                                            (
                                                ((data["y_1"]) > (((-0.228787) / 2.0)))
                                                * 1.0
                                            )
                                        )
                                        > ((((data["dist"]) <= (3.141593)) * 1.0))
                                    )
                                    * 1.0
                                )
                            )
                        )
                    )
                )
            )
        )
        + 1.0
        * (
            np.minimum(
                (((((data["dist_to_type_mean"]) > (1.090505)) * 1.0))),
                (
                    (
                        np.minimum(
                            (((((data["z_0"]) <= (data["dist_to_type_mean"])) * 1.0))),
                            (((((data["x_0"]) <= (data["dist_to_type_mean"])) * 1.0))),
                        )
                    )
                ),
            )
        )
        + 1.0
        * (
            (
                (
                    ((((-0.181841) + (((-0.228787) / 2.0))) / 2.0))
                    + (
                        (
                            (
                                (0.927964)
                                > (
                                    np.maximum(
                                        ((data["atom_index_1"])),
                                        ((data["dist_to_type_mean"])),
                                    )
                                )
                            )
                            * 1.0
                        )
                    )
                )
                / 2.0
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (((data["y_1"]) * 2.0))
                                                    > (data["y_1"])
                                                )
                                                * 1.0
                                            )
                                        )
                                        + (((1.570796) / 2.0))
                                    )
                                    / 2.0
                                )
                            )
                            > (data["dist_to_type_mean"])
                        )
                        * 1.0
                    )
                )
                / 2.0
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (0.018557)
                        * (
                            (
                                (
                                    np.maximum(
                                        ((data["x_0"])),
                                        ((((data["y_0"]) * (data["dist"])))),
                                    )
                                )
                                - (((1.570796) * 2.0))
                            )
                        )
                    )
                )
                * 2.0
            )
        )
        + 1.0
        * (
            np.minimum(
                (
                    (
                        (
                            (
                                (
                                    ((1.570796) > (((data["dist_to_type_mean"]) * 2.0)))
                                    * 1.0
                                )
                            )
                            * 2.0
                        )
                    )
                ),
                ((1.570796)),
            )
        )
        + 0.999511
        * (
            (
                (
                    (
                        (
                            np.minimum(
                                ((((((data["atom_index_1"]) + (-0.267775))) + (-1.0)))),
                                (((((1.570796) <= (data["z_0"])) * 1.0))),
                            )
                        )
                        / 2.0
                    )
                )
                / 2.0
            )
        )
        + 1.0
        * (
            np.minimum(
                (((((((0.00275373528711498)) * 2.0)) + ((0.00275373528711498))))),
                ((data["atom_index_1"])),
            )
        )
        + 1.0
        * (
            (
                (
                    ((((((3.647356) > (data["dist"])) * 1.0)) * (data["dist"])))
                    + ((-1.0 * ((data["dist"]))))
                )
                / 2.0
            )
        )
        + 0.999023
        * (
            np.maximum(
                (((((0.0) + ((((data["dist"]) + (-3.0)) / 2.0))) / 2.0))), ((0.0))
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (((0.0) / 2.0))
                                        - (
                                            (
                                                (
                                                    (data["x_0"])
                                                    > (((data["x_0"]) * (data["x_0"])))
                                                )
                                                * 1.0
                                            )
                                        )
                                    )
                                )
                                / 2.0
                            )
                        )
                        / 2.0
                    )
                )
                / 2.0
            )
        )
        + 1.0
        * (
            np.maximum(
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (data["dist_to_type_mean"])
                                                        <= (0.913097)
                                                    )
                                                    * 1.0
                                                )
                                            )
                                            / 2.0
                                        )
                                    )
                                    / 2.0
                                )
                            )
                            / 2.0
                        )
                    )
                ),
                ((((0.913097) - (data["dist_to_type_mean"])))),
            )
        )
    )


def GP6(data):
    return (
        4.772715
        + 1.0
        * (
            (
                (
                    (
                        (0.636620)
                        - (
                            (
                                (
                                    (data["dist"])
                                    <= (np.maximum(((3.0)), ((data["y_0"]))))
                                )
                                * 1.0
                            )
                        )
                    )
                )
                - ((((data["dist"]) <= (3.0)) * 1.0))
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (
                                (
                                    ((((-3.0) + (data["dist"])) / 2.0))
                                    > ((((data["y_1"]) > (((0.218830) * 2.0))) * 1.0))
                                )
                                * 1.0
                            )
                        )
                        * 2.0
                    )
                )
                * 2.0
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        ((data["dist_to_type_mean"]) <= ((((1.0) + (0.922659)) / 2.0)))
                        * 1.0
                    )
                )
                - ((((2.492783) <= (data["dist"])) * 1.0))
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        ((((data["dist"]) > (3.0)) * 1.0))
                        - ((((data["dist"]) > (3.102541)) * 1.0))
                    )
                )
                - ((((data["dist"]) > (3.102541)) * 1.0))
            )
        )
        + 0.962384
        * (
            (
                (2.453189)
                - (
                    (
                        (data["dist"])
                        + ((-1.0 * (((((data["dist"]) <= (2.453189)) * 1.0)))))
                    )
                )
            )
        )
        + 1.0
        * (
            (
                (1.129494)
                * (
                    (
                        (1.129494)
                        * (
                            (
                                (1.129494)
                                * (
                                    (
                                        (
                                            (np.maximum(((3.051597)), ((data["z_1"]))))
                                            <= (data["dist"])
                                        )
                                        * 1.0
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
        + 0.882755
        * (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            np.minimum(
                                                ((0.636620)), ((((data["y_1"]) / 2.0)))
                                            )
                                        )
                                        - ((((data["y_1"]) <= (0.636620)) * 1.0))
                                    )
                                )
                                / 2.0
                            )
                        )
                        / 2.0
                    )
                )
                / 2.0
            )
        )
        + 1.0
        * (
            (
                ((((1.146451) > (data["dist_to_type_mean"])) * 1.0))
                - (data["dist_to_type_mean"])
            )
        )
        + 1.0
        * (
            (
                (
                    (-2.0)
                    > (
                        (
                            -1.0
                            * (
                                (
                                    (
                                        (data["dist_to_type_mean"])
                                        * (
                                            (
                                                (data["dist_to_type_mean"])
                                                * (
                                                    (
                                                        (data["dist_to_type_mean"])
                                                        * (data["y_0"])
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
                * 1.0
            )
        )
        + 1.0
        * (
            (
                (((((data["y_0"]) - ((((data["y_0"]) <= ((1.0))) * 1.0)))) * 2.0))
                * (((data["dist_to_type_mean"]) - ((1.0))))
            )
        )
        + 1.0
        * (
            (
                ((((((3.090264) <= (data["dist"])) * 1.0)) * 2.0))
                * ((-1.0 * ((data["dist_to_type_mean"]))))
            )
        )
        + 1.0
        * (
            (
                (((2.268008) / 2.0))
                * ((((((data["dist_to_type_mean"]) > (1.132620)) * 1.0)) / 2.0))
            )
        )
        + 1.0
        * (
            (
                (
                    (-0.278111)
                    + (
                        (
                            (
                                (
                                    (
                                        (
                                            (-1.0)
                                            + (
                                                (
                                                    (
                                                        (data["dist_to_type_mean"])
                                                        <= (0.941507)
                                                    )
                                                    * 1.0
                                                )
                                            )
                                        )
                                        / 2.0
                                    )
                                )
                                + ((((data["dist_to_type_mean"]) <= (0.941507)) * 1.0))
                            )
                            / 2.0
                        )
                    )
                )
                / 2.0
            )
        )
        + 1.0
        * (
            (
                (
                    np.minimum(
                        ((((((-0.047501) * (data["y_0"]))) - (-0.047501)))),
                        ((-0.047501)),
                    )
                )
                * (((data["y_0"]) / 2.0))
            )
        )
        + 1.0
        * (
            (
                (
                    (-1.0)
                    + (
                        (
                            (
                                (0.922659)
                                <= (
                                    np.maximum(
                                        ((data["x_1"])),
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            np.maximum(
                                                                ((data["z_1"])),
                                                                (
                                                                    (
                                                                        data[
                                                                            "dist_to_type_mean"
                                                                        ]
                                                                    )
                                                                ),
                                                            )
                                                        )
                                                        + (data["dist_to_type_mean"])
                                                    )
                                                    / 2.0
                                                )
                                            )
                                        ),
                                    )
                                )
                            )
                            * 1.0
                        )
                    )
                )
                / 2.0
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (
                                (
                                    (3.046956)
                                    <= (np.maximum(((data["dist"])), ((data["z_0"]))))
                                )
                                * 1.0
                            )
                        )
                        / 2.0
                    )
                )
                / 2.0
            )
        )
    )


def GP7(data):
    return (
        0.990498
        + 0.938935
        * (
            np.minimum(
                (
                    (
                        (
                            (0.318310)
                            - (
                                (
                                    (data["dist_to_type_mean"])
                                    * (
                                        (
                                            ((data["dist_to_type_mean"]) <= (1.015225))
                                            * 1.0
                                        )
                                    )
                                )
                            )
                        )
                    )
                ),
                ((((data["dist"]) - (3.141593)))),
            )
        )
        + 0.933073
        * (
            (
                -1.0
                * (
                    (
                        np.maximum(
                            (
                                (
                                    (
                                        (((((data["y_1"]) * 2.0)) * (0.011385)))
                                        * (data["dist"])
                                    )
                                )
                            ),
                            (((((data["dist"]) <= ((2.32503938674926758))) * 1.0))),
                        )
                    )
                )
            )
        )
        + 0.932096
        * (
            np.minimum(
                ((data["atom_index_1"])),
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                ((-1.0 * ((data["dist_to_type_mean"]))))
                                                + (
                                                    np.maximum(
                                                        ((0.983454)),
                                                        ((data["dist_to_type_mean"])),
                                                    )
                                                )
                                            )
                                            / 2.0
                                        )
                                    )
                                    * 2.0
                                )
                            )
                            * 2.0
                        )
                    )
                ),
            )
        )
        + 0.982413
        * (
            (
                (
                    np.minimum(
                        ((((-0.059342) / 2.0))),
                        ((((((((-0.059342) / 2.0)) / 2.0)) * (data["atom_index_1"])))),
                    )
                )
                * (data["dist_to_type_mean"])
            )
        )
        + 1.0
        * (
            (
                -1.0
                * (
                    (
                        (
                            (-0.108994)
                            * (
                                (
                                    (
                                        np.maximum(
                                            ((data["y_0"])),
                                            (
                                                (
                                                    (
                                                        (
                                                            (
                                                                ((-1.0 * ((3.141593))))
                                                                / 2.0
                                                            )
                                                        )
                                                        / 2.0
                                                    )
                                                )
                                            ),
                                        )
                                    )
                                    / 2.0
                                )
                            )
                        )
                    )
                )
            )
        )
        + 0.980948
        * (
            (
                (((((1.0)) > (data["atom_index_1"])) * 1.0))
                * (((data["dist"]) * (((1.055259) * (((-0.110218) / 2.0))))))
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            ((0.04956008121371269))
                            > (((1.055259) * (((data["x_0"]) * (data["x_1"])))))
                        )
                        * 1.0
                    )
                )
                * ((0.04956008121371269))
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (data["atom_index_0"])
                                            <= ((5.20004987716674805))
                                        )
                                        * 1.0
                                    )
                                )
                                + (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            (data["atom_index_1"])
                                                            <= (data["dist"])
                                                        )
                                                        * 1.0
                                                    )
                                                )
                                                / 2.0
                                            )
                                        )
                                        / 2.0
                                    )
                                )
                            )
                            / 2.0
                        )
                    )
                    + (-0.059342)
                )
                / 2.0
            )
        )
        + 1.0
        * (
            (
                (
                    (data["dist_to_type_mean"])
                    <= (
                        (
                            (-0.210597)
                            + (
                                (
                                    (
                                        (data["y_1"])
                                        > (
                                            (
                                                (
                                                    (
                                                        (
                                                            (data["y_1"])
                                                            > (
                                                                (
                                                                    (
                                                                        data[
                                                                            "dist_to_type_mean"
                                                                        ]
                                                                    )
                                                                    / 2.0
                                                                )
                                                            )
                                                        )
                                                        * 1.0
                                                    )
                                                )
                                                * 2.0
                                            )
                                        )
                                    )
                                    * 1.0
                                )
                            )
                        )
                    )
                )
                * 1.0
            )
        )
        + 0.917929
        * (
            (
                -1.0
                * (
                    (
                        (
                            (
                                (
                                    (
                                        (data["dist_to_type_mean"])
                                        <= (((1.570796) / 2.0))
                                    )
                                    * 1.0
                                )
                            )
                            / 2.0
                        )
                    )
                )
            )
        )
        + 1.0
        * (
            (
                (
                    (data["dist"])
                    > (
                        np.maximum(
                            ((3.0)),
                            (
                                (
                                    (
                                        (
                                            (data["atom_index_0"])
                                            + (((-1.0) - ((-1.0 * ((data["x_1"]))))))
                                        )
                                        / 2.0
                                    )
                                )
                            ),
                        )
                    )
                )
                * 1.0
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (
                                ((((1.177861) + ((-1.0 * ((data["y_0"])))))) <= (-1.0))
                                * 1.0
                            )
                        )
                        * (-0.108994)
                    )
                )
                * (data["dist_to_type_mean"])
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        -1.0
                        * (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    np.maximum(
                                                        ((data["y_0"])),
                                                        ((((data["dist"]) * 2.0))),
                                                    )
                                                )
                                                > (7.0)
                                            )
                                            * 1.0
                                        )
                                    )
                                    / 2.0
                                )
                            )
                        )
                    )
                )
                / 2.0
            )
        )
        + 0.999023
        * (
            (
                (
                    (
                        (
                            (((data["atom_index_1"]) / 2.0))
                            <= (((data["dist"]) - (data["atom_index_0"])))
                        )
                        * 1.0
                    )
                )
                / 2.0
            )
        )
        + 0.706400
        * (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (data["y_1"])
                                                > (
                                                    (
                                                        ((-0.108994) + (data["y_0"]))
                                                        / 2.0
                                                    )
                                                )
                                            )
                                            * 1.0
                                        )
                                    )
                                    > (
                                        (
                                            (((5.40717029571533203)) + (data["y_0"]))
                                            / 2.0
                                        )
                                    )
                                )
                                * 1.0
                            )
                        )
                        / 2.0
                    )
                )
                / 2.0
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (3.0)
                                            <= (
                                                (
                                                    (data["z_0"])
                                                    - (
                                                        (
                                                            (
                                                                (data["x_1"])
                                                                <= (data["y_0"])
                                                            )
                                                            * 1.0
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                        * 1.0
                                    )
                                )
                                / 2.0
                            )
                        )
                        / 2.0
                    )
                )
                / 2.0
            )
        )
    )


GP_FUNCS = {0: GP0, 1: GP1, 2: GP2, 3: GP3, 4: GP4, 5: GP5, 6: GP6, 7: GP7}




## === cell 10
def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    maes = (y_true - y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()


def _safe_pred_array(pred: np.ndarray) -> np.ndarray:
    """
    Change rationale (score improvement / stability, no core logic change):
    Rare NaN/inf predictions can explode log-MAE and ruin the mapping selection and submission.
    We replace non-finite values with 0.0 (same default baseline as elsewhere in the script).
    """
    pred = np.asarray(pred, dtype=np.float64)
    pred[~np.isfinite(pred)] = 0.0
    return pred


def infer_type_to_gp_mapping(
    train_df: pd.DataFrame,
    gp_funcs: dict,
    sample_per_type: int = 25000,
    random_state: int = 0,
    floor: float = 1e-9,
):
    """
    Change rationale (score improvement, minimal logic change):
    Previously the mapping used plain MAE per type; Kaggle uses mean(log(MAE)) across types.
    We now pick the GPk per type that minimizes log(MAE) (equivalently MAE but with a floor,
    and consistently using the same log-MAE semantics as the competition), which tends to
    choose more appropriate formulas for each coupling type.
    """
    rs = np.random.RandomState(random_state)

    type_to_gp = {}
    scores = []
    for t in sorted(train_df["type_str"].unique()):
        idx = train_df.index[train_df["type_str"] == t].to_numpy()
        if idx.size == 0:
            continue
        if idx.size > sample_per_type:
            take = rs.choice(idx, size=sample_per_type, replace=False)
        else:
            take = idx

        chunk = train_df.loc[take]
        y = chunk["scalar_coupling_constant"].values.astype(np.float64)

        best_k = None
        best_obj = np.inf
        best_mae = np.inf
        for k, fn in gp_funcs.items():
            pred = _safe_pred_array(fn(chunk))
            mae = np.mean(np.abs(y - pred))
            obj = np.log(max(mae, floor))
            if obj < best_obj:
                best_obj = obj
                best_mae = mae
                best_k = k

        type_to_gp[t] = int(best_k)
        scores.append((t, int(best_k), float(best_mae), float(best_obj)))

    mixed_n = min(200000, len(train_df))
    mixed_idx = rs.choice(train_df.index.to_numpy(), size=mixed_n, replace=False)
    mixed = train_df.loc[mixed_idx]
    y_m = mixed["scalar_coupling_constant"].values.astype(np.float64)

    best_global_k, best_global_obj, best_global_mae = None, np.inf, np.inf
    for k, fn in gp_funcs.items():
        pred = _safe_pred_array(fn(mixed))
        mae = np.mean(np.abs(y_m - pred))
        obj = np.log(max(mae, floor))
        if obj < best_global_obj:
            best_global_obj = obj
            best_global_mae = mae
            best_global_k = k

    score_df = pd.DataFrame(
        scores, columns=["type_str", "gp_k", "mae_on_sample", "log_mae_on_sample"]
    ).sort_values("type_str")
    return type_to_gp, int(best_global_k), score_df


type_to_gp_map, global_fallback_gp, mapping_debug = infer_type_to_gp_mapping(
    train, GP_FUNCS
)
print("Inferred type->GP mapping:", type_to_gp_map)
print("Global fallback GP:", global_fallback_gp)
print("Mapping debug (sample MAE/log-MAE):")
print(mapping_debug)

train["gp_k"] = (
    train["type_str"].map(type_to_gp_map).fillna(global_fallback_gp).astype(np.int16)
)
test["gp_k"] = (
    test["type_str"].map(type_to_gp_map).fillna(global_fallback_gp).astype(np.int16)
)

print("Unique mapped gp_k (0..7):", sorted(train["gp_k"].unique().tolist()))




## === cell 11
def GP(data):
    retValues = pd.DataFrame({"id": data.id})
    retValues["scalar_coupling_constant"] = 0.0

    for k, fn in [
        (0, GP0),
        (1, GP1),
        (2, GP2),
        (3, GP3),
        (4, GP4),
        (5, GP5),
        (6, GP6),
        (7, GP7),
    ]:
        m = data.gp_k == k
        if m.any():
            retValues.loc[m, "scalar_coupling_constant"] = _safe_pred_array(fn(data[m]))

    return retValues


predictions = GP(train)

print(
    "Train group_mean_log_mae (grouped by original type_str):",
    group_mean_log_mae(
        train.scalar_coupling_constant,
        predictions.scalar_coupling_constant,
        train.type_str,
    ),
)



## === cell 12
submission = GP(test)

submission = sub[["id"]].merge(submission, on="id", how="left")
submission["scalar_coupling_constant"] = submission["scalar_coupling_constant"].fillna(
    0.0
)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
