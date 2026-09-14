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

0.5536

# 6. Current score

1.5738

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.49339) has done: 'The error is caused by NaNs created during the `dist_to_type_mean` feature calculation for coupling `type`s that appear only in the test set (the group mean becomes NaN), and ExtraTrees cannot predict with NaNs. I keep the same feature set and model, but compute the type-mean distance from the training set and map it into both train/test, filling any missing means safely. I also fix the input paths to match your provided Kaggle directory (`/kaggle/input/champs-scalar-coupling/...`) so the notebook runs end-to-end. These changes are score-neutral to slightly positive (less broken feature), and produce a valid `submission.csv`.'
- What this solution (achieved 1.63794) has done: 'We keep your exact feature set and ExtraTrees model, but make two small changes that reliably improve score without changing the overall approach: (1) use a molecule-based split for the local validation so it matches the competition’s split-by-molecule and avoids optimistic/unstable feedback, and (2) train on a larger (but still time-safe) recent subset of the data, because your current `.tail(2_000_000)` is likely underfitting and biased. We also switch the printed validation metric to the competition’s official “mean over types of log(MAE)” computed on the validation split, so you can see progress that correlates with Kaggle LB. Everything still runs end-to-end and writes `submission.csv` with the required columns.'
- What this solution (achieved 1.39256) has done: 'I keep your same ExtraTrees-per-type setup and feature set, but make two minimal fixes that typically improve log-MAE: (1) use a consistent label encoding for the `type0..type3` character features across train+test (your current per-position encoder is fit on train only, which can mis-encode when unseen chars appear), and (2) replace the biased `.tail(PER_TYPE_TRAIN)` selection with a per-type random sample (still capped to the same size) so training data is representative and less order-dependent. I also add a deterministic seed and keep the molecule-based validation split intact so the printed metric tracks Kaggle’s split-by-molecule behavior. The pipeline still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 1.33877) has done: 'Your current approach is underperforming mainly because it ignores strong, competition-standard signal sources that are already provided as CSVs (Mulliken charges, shielding tensors, dipole moments, potential energy). I keep your per-type ExtraTrees setup and all existing features intact, and minimally add these auxiliary features via safe left-joins on the same keys you already use (molecule_name + atom_index). This should reduce MAE substantially (hence lower log-MAE) without changing the modeling approach. I also keep your molecule-based validation and submission-writing exactly as-is, just extending the feature matrix with these extra columns and filling any new NaNs with train medians.'
- What this solution (achieved 1.48433) has done: 'I keep your per-type ExtraTrees pipeline and existing feature set intact, and make two small, targeted changes that typically move log-MAE down for this competition: (1) add a couple of geometry-only pair features (coordinate deltas and inverse-distance) derived from the coordinates you already merged, and (2) use the distance feature in a more linear-friendly form (also include `log1p(dist)`) so trees can split more effectively. These are minimal additions that don’t change the training loop or model family, but add missing signal strongly correlated with coupling strength. I keep the same data paths and still write a valid `submission.csv` with the required columns.'
- What this solution (achieved 1.43264) has done: 'We’re far from the target (1.48433 vs 0.5536, lower is better), so we need a meaningful-but-still-minimal improvement without changing your per-type ExtraTrees approach. The biggest low-risk gain is to add a few interaction features from the auxiliary atom-level tensors/charges you already merged (differences, absolute differences, and simple products between atom0/atom1), which trees can exploit to model coupling strength better. We keep the same training loop, same model family, same per-type sampling and molecule-based validation, and only expand the feature matrix with these derived columns. We also ensure all newly created numeric features are median-imputed consistently so the pipeline remains stable and produces a valid `submission.csv`.'
- What this solution (achieved 1.40406) has done: 'Your current gap to the target is large (1.43264 vs 0.5536, lower is better), so we need a real-but-still-minimal signal improvement while keeping the same per-type ExtraTrees approach. The biggest low-risk gain without changing the model family is to add a couple of standard CHAMPS geometry features derived from the same coordinates you already use: per-atom radial distance from molecule centroid and a simple “bond axis alignment” (cosine) feature, which help trees explain coupling strength variation. I also add a tiny set of additional symmetric interaction features for the already-merged atom-level tensors/charges (sum and squared-diff) to complement your existing diff/absdiff/prod, and keep the same median-imputation scheme so inference stays stable. Everything still trains per type, uses the same split logic, and writes a valid `submission.csv`.'
- What this solution (achieved 1.40287) has done: 'You’re far worse than the target (1.40406 vs 0.5536, lower is better), so we need a real score improvement while keeping your exact per-type ExtraTrees approach and feature families intact. The highest-impact minimal fix here is to make the merged atom-level features symmetric with respect to (atom_index_0, atom_index_1): for each coupling, we build “min/max” versions of each atom feature (e.g., mull_min/mull_max, mstXX_min/mstXX_max) so the model doesn’t have to learn two separate patterns depending on arbitrary atom ordering. This does not change the model, loss, training loop, or data sources; it only adds a small set of robust features that typically reduces MAE on CHAMPS. We keep your existing features, validation scheme, per-type sampling cap, and submission format unchanged.'
- What this solution (achieved 1.42415) has done: 'We’re far above the target (1.40287 vs 0.5536, lower is better), so we should make a small but meaningful improvement without changing your per-type ExtraTrees approach. The most “signal-dense” missing feature you can add with minimal disruption is bond-length aggregation by molecule and type (mean/std/count), computed on train and mapped into both train/test; this captures molecule-specific scaling that strongly affects couplings. I also add a tiny per-molecule atom-count feature from `structures.csv` (again just a map/join), which helps distinguish molecule regimes with virtually no risk. Everything else (features, per-type training loop, model family, submission writing) stays the same.'
- What this solution (achieved 1.43275) has done: 'Your current score (1.42415, lower is better) is far from the target (0.5536), so we need a meaningful improvement while keeping the same per-type ExtraTrees approach and overall feature families. The most effective minimal change for this competition is to train the model on the 4 known physical contributions (`fc/sd/pso/dso`) and then predict their sum; this is still ExtraTrees regression per type, but uses a much stronger target signal already provided by the dataset. To preserve your current pipeline, I only add one safe merge (train-only) with `scalar_coupling_contributions.csv` and keep the same feature set, sampling cap, validation-by-molecule, and submission writing. This should move log-MAE substantially downward toward your target without changing the model family or training loop structure.'
- What this solution (achieved 1.5738) has done: 'I keep your per-type ExtraTrees setup and all existing feature sources, but fix the main issue holding the score back: you’re training on `fc+sd+pso+dso` (the sum) while still using your original feature set, yet you’re not actually feeding the model the most predictive signal available at inference time (the *per-atom* contributions don’t exist for test). Instead, we use the contributions only to create a cleaner training target proxy: fit to the original `scalar_coupling_constant` (the competition target) while still optionally using the contributions as **train-only auxiliary features** aggregated in a leakage-safe way (molecule/type means) that can be mapped into test. This keeps the same model family and loop, adds only a small join+aggregation, and should move log-MAE down meaningfully toward your target. We also compute the validation metric exactly like Kaggle (mean over types of log(MAE)) on the molecule-split validation to ensure changes correlate with LB.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from sklearn import preprocessing, ensemble, metrics

DATA_DIR = "/kaggle/input/champs-scalar-coupling"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
print(train.shape, test.shape, sub.shape)

train["atom1"] = train["type"].map(lambda x: str(x)[2])
train["atom2"] = train["type"].map(lambda x: str(x)[3])
test["atom1"] = test["type"].map(lambda x: str(x)[2])
test["atom2"] = test["type"].map(lambda x: str(x)[3])

for i in range(4):
    lbl = preprocessing.LabelEncoder()
    all_chars = pd.concat(
        [
            train["type"].map(lambda x: str(x)[i]),
            test["type"].map(lambda x: str(x)[i]),
        ],
        axis=0,
        ignore_index=True,
    )
    lbl.fit(all_chars)
    train["type" + str(i)] = lbl.transform(train["type"].map(lambda x: str(x)[i]))
    test["type" + str(i)] = lbl.transform(test["type"].map(lambda x: str(x)[i]))

structures0 = pd.read_csv(f"{DATA_DIR}/structures.csv").rename(
    columns={
        "atom_index": "atom_index_0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
        "atom": "atom1",
    }
)
train = pd.merge(
    train, structures0, how="left", on=["molecule_name", "atom_index_0", "atom1"]
)
test = pd.merge(
    test, structures0, how="left", on=["molecule_name", "atom_index_0", "atom1"]
)
del structures0

structures1 = pd.read_csv(f"{DATA_DIR}/structures.csv").rename(
    columns={
        "atom_index": "atom_index_1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
        "atom": "atom2",
    }
)
train = pd.merge(
    train, structures1, how="left", on=["molecule_name", "atom_index_1", "atom2"]
)
test = pd.merge(
    test, structures1, how="left", on=["molecule_name", "atom_index_1", "atom2"]
)
del structures1

mull = pd.read_csv(f"{DATA_DIR}/mulliken_charges.csv").rename(
    columns={"mulliken_charge": "mulliken_charge"}
)
mull0 = mull.rename(columns={"atom_index": "atom_index_0", "mulliken_charge": "mull0"})
mull1 = mull.rename(columns={"atom_index": "atom_index_1", "mulliken_charge": "mull1"})
train = train.merge(
    mull0[["molecule_name", "atom_index_0", "mull0"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
)
test = test.merge(
    mull0[["molecule_name", "atom_index_0", "mull0"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
)
train = train.merge(
    mull1[["molecule_name", "atom_index_1", "mull1"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
)
test = test.merge(
    mull1[["molecule_name", "atom_index_1", "mull1"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
)
del mull, mull0, mull1

mst = pd.read_csv(f"{DATA_DIR}/magnetic_shielding_tensors.csv")
mst0 = mst.rename(columns={"atom_index": "atom_index_0"})
mst1 = mst.rename(columns={"atom_index": "atom_index_1"})
mst_cols = ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]
mst0 = mst0[["molecule_name", "atom_index_0"] + mst_cols].rename(
    columns={c: f"mst0_{c}" for c in mst_cols}
)
mst1 = mst1[["molecule_name", "atom_index_1"] + mst_cols].rename(
    columns={c: f"mst1_{c}" for c in mst_cols}
)
train = train.merge(mst0, on=["molecule_name", "atom_index_0"], how="left")
test = test.merge(mst0, on=["molecule_name", "atom_index_0"], how="left")
train = train.merge(mst1, on=["molecule_name", "atom_index_1"], how="left")
test = test.merge(mst1, on=["molecule_name", "atom_index_1"], how="left")
del mst, mst0, mst1

dip = pd.read_csv(f"{DATA_DIR}/dipole_moments.csv").rename(
    columns={"X": "dip_X", "Y": "dip_Y", "Z": "dip_Z"}
)
pe = pd.read_csv(f"{DATA_DIR}/potential_energy.csv")
train = train.merge(dip, on="molecule_name", how="left").merge(
    pe, on="molecule_name", how="left"
)
test = test.merge(dip, on="molecule_name", how="left").merge(
    pe, on="molecule_name", how="left"
)
del dip, pe

scc = pd.read_csv(
    f"{DATA_DIR}/scalar_coupling_contributions.csv",
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
)

scc["contrib_sum"] = scc[["fc", "sd", "pso", "dso"]].sum(axis=1)
moltype_contrib_agg = scc.groupby(["molecule_name", "type"])[
    ["fc", "sd", "pso", "dso", "contrib_sum"]
].agg(["mean", "std"])
moltype_contrib_agg.columns = [
    f"moltype_{a}_{b}" for a, b in moltype_contrib_agg.columns.to_flat_index()
]
moltype_contrib_agg = moltype_contrib_agg.reset_index()

train = train.merge(moltype_contrib_agg, on=["molecule_name", "type"], how="left")
test = test.merge(moltype_contrib_agg, on=["molecule_name", "type"], how="left")
del scc, moltype_contrib_agg

print(train.shape, test.shape, sub.shape)



## === cell 1
train_p0 = train[["x0", "y0", "z0"]].values
train_p1 = train[["x1", "y1", "z1"]].values
test_p0 = test[["x0", "y0", "z0"]].values
test_p1 = test[["x1", "y1", "z1"]].values

train_dx = train_p0[:, 0] - train_p1[:, 0]
train_dy = train_p0[:, 1] - train_p1[:, 1]
train_dz = train_p0[:, 2] - train_p1[:, 2]
test_dx = test_p0[:, 0] - test_p1[:, 0]
test_dy = test_p0[:, 1] - test_p1[:, 1]
test_dz = test_p0[:, 2] - test_p1[:, 2]

train["dx"] = train_dx
train["dy"] = train_dy
train["dz"] = train_dz
test["dx"] = test_dx
test["dy"] = test_dy
test["dz"] = test_dz

train["dist"] = np.sqrt(train_dx * train_dx + train_dy * train_dy + train_dz * train_dz)
test["dist"] = np.sqrt(test_dx * test_dx + test_dy * test_dy + test_dz * test_dz)

eps = 1e-12
train["inv_dist"] = 1.0 / (train["dist"].values + eps)
test["inv_dist"] = 1.0 / (test["dist"].values + eps)

train["log1p_dist"] = np.log1p(train["dist"].values)
test["log1p_dist"] = np.log1p(test["dist"].values)

type_mean_dist = train.groupby("type")["dist"].mean()
global_mean_dist = float(train["dist"].mean())

train_type_mean = train["type"].map(type_mean_dist).fillna(global_mean_dist)
test_type_mean = test["type"].map(type_mean_dist).fillna(global_mean_dist)

train["dist_to_type_mean"] = train["dist"] / train_type_mean
test["dist_to_type_mean"] = test["dist"] / test_type_mean

for df in (train, test):
    cx = df.groupby("molecule_name")["x0"].transform("mean")
    cy = df.groupby("molecule_name")["y0"].transform("mean")
    cz = df.groupby("molecule_name")["z0"].transform("mean")
    df["r0"] = np.sqrt(
        (df["x0"] - cx) ** 2 + (df["y0"] - cy) ** 2 + (df["z0"] - cz) ** 2
    )
    df["r1"] = np.sqrt(
        (df["x1"] - cx) ** 2 + (df["y1"] - cy) ** 2 + (df["z1"] - cz) ** 2
    )
    df["r0_plus_r1"] = df["r0"] + df["r1"]
    df["r0_minus_r1"] = df["r0"] - df["r1"]

train_r0_vec = (
    train_p0
    - train.groupby("molecule_name")[["x0", "y0", "z0"]].transform("mean").values
)
train_r1_vec = (
    train_p1
    - train.groupby("molecule_name")[["x0", "y0", "z0"]].transform("mean").values
)
test_r0_vec = (
    test_p0 - test.groupby("molecule_name")[["x0", "y0", "z0"]].transform("mean").values
)
test_r1_vec = (
    test_p1 - test.groupby("molecule_name")[["x0", "y0", "z0"]].transform("mean").values
)

train_bond = np.vstack([train_dx, train_dy, train_dz]).T
test_bond = np.vstack([test_dx, test_dy, test_dz]).T


def safe_cos(a, b, eps=1e-12):
    an = np.sqrt((a * a).sum(axis=1)) + eps
    bn = np.sqrt((b * b).sum(axis=1)) + eps
    return (a * b).sum(axis=1) / (an * bn)


train["cos_bond_r0"] = safe_cos(train_bond, train_r0_vec, eps=eps)
train["cos_bond_r1"] = safe_cos(-train_bond, train_r1_vec, eps=eps)
test["cos_bond_r0"] = safe_cos(test_bond, test_r0_vec, eps=eps)
test["cos_bond_r1"] = safe_cos(-test_bond, test_r1_vec, eps=eps)

train["mull_diff"] = train["mull0"] - train["mull1"]
test["mull_diff"] = test["mull0"] - test["mull1"]
train["mull_absdiff"] = np.abs(train["mull_diff"].values)
test["mull_absdiff"] = np.abs(test["mull_diff"].values)
train["mull_prod"] = train["mull0"] * train["mull1"]
test["mull_prod"] = test["mull0"] * test["mull1"]

train["mull_sum"] = train["mull0"] + train["mull1"]
test["mull_sum"] = test["mull0"] + test["mull1"]
train["mull_diff2"] = train["mull_diff"].values ** 2
test["mull_diff2"] = test["mull_diff"].values ** 2

for df in (train, test):
    df["mull_min"] = np.minimum(df["mull0"].values, df["mull1"].values)
    df["mull_max"] = np.maximum(df["mull0"].values, df["mull1"].values)

for c in ["dip_X", "dip_Y", "dip_Z"]:
    train[f"{c}_dist"] = train[c] * train["dist"]
    test[f"{c}_dist"] = test[c] * test["dist"]
    train[f"{c}_inv_dist"] = train[c] * train["inv_dist"]
    test[f"{c}_inv_dist"] = test[c] * test["inv_dist"]

for c in ["XX", "YY", "ZZ"]:
    a0 = f"mst0_{c}"
    a1 = f"mst1_{c}"
    train[f"{c}_diff"] = train[a0] - train[a1]
    test[f"{c}_diff"] = test[a0] - test[a1]
    train[f"{c}_absdiff"] = np.abs(train[f"{c}_diff"].values)
    test[f"{c}_absdiff"] = np.abs(test[f"{c}_diff"].values)
    train[f"{c}_prod"] = train[a0] * train[a1]
    test[f"{c}_prod"] = test[a0] * test[a1]
    train[f"{c}_sum"] = train[a0] + train[a1]
    test[f"{c}_sum"] = test[a0] + test[a1]
    train[f"{c}_diff2"] = train[f"{c}_diff"].values ** 2
    test[f"{c}_diff2"] = test[f"{c}_diff"].values ** 2

    train[f"{c}_min"] = np.minimum(train[a0].values, train[a1].values)
    train[f"{c}_max"] = np.maximum(train[a0].values, train[a1].values)
    test[f"{c}_min"] = np.minimum(test[a0].values, test[a1].values)
    test[f"{c}_max"] = np.maximum(test[a0].values, test[a1].values)

dist_agg = (
    train.groupby(["molecule_name", "type"])["dist"]
    .agg(["mean", "std", "count"])
    .rename(
        columns={
            "mean": "moltype_dist_mean",
            "std": "moltype_dist_std",
            "count": "moltype_dist_count",
        }
    )
)
train = train.join(dist_agg, on=["molecule_name", "type"])
test = test.join(dist_agg, on=["molecule_name", "type"])

structures = pd.read_csv(
    f"{DATA_DIR}/structures.csv", usecols=["molecule_name", "atom_index"]
)
mol_atom_cnt = structures.groupby("molecule_name")["atom_index"].max() + 1
mol_atom_cnt = mol_atom_cnt.rename("mol_n_atoms")
train = train.join(mol_atom_cnt, on="molecule_name")
test = test.join(mol_atom_cnt, on="molecule_name")
del structures, mol_atom_cnt, dist_agg

num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
medians = train[num_cols].median(numeric_only=True)
train[num_cols] = train[num_cols].fillna(medians)
test_num_cols = [c for c in num_cols if c in test.columns]
test[test_num_cols] = test[test_num_cols].fillna(medians.reindex(test_num_cols))



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
        "atom_index_0",
        "atom_index_1",
    ]
]

base_reg_params = dict(n_jobs=-1, n_estimators=20, random_state=4)

PER_TYPE_TRAIN = 300_000  # keep same cap to preserve runtime/approach
SEED = 99

rng = np.random.RandomState(SEED)
test_pred = np.zeros(len(test), dtype=np.float64)

val_scores = []
types = sorted(train["type"].unique().tolist())
print("Training per-type models for", len(types), "types:", types)

for t in types:
    tr_full = train[train["type"] == t]

    if len(tr_full) > PER_TYPE_TRAIN:
        tr_t = tr_full.sample(n=PER_TYPE_TRAIN, random_state=SEED).reset_index(
            drop=True
        )
    else:
        tr_t = tr_full.reset_index(drop=True)

    te_mask = test["type"].values == t

    mols = pd.unique(tr_t["molecule_name"].values)
    rng.shuffle(mols)
    n_val_mols = max(1, int(0.2 * len(mols)))
    val_mols = set(mols[:n_val_mols])

    is_val = tr_t["molecule_name"].isin(val_mols).values
    x_tr = tr_t.loc[~is_val, col]
    y_tr = tr_t.loc[~is_val, "scalar_coupling_constant"].astype(np.float64)

    x_va = tr_t.loc[is_val, col]
    y_va = tr_t.loc[is_val, "scalar_coupling_constant"].astype(np.float64)

    reg = ensemble.ExtraTreesRegressor(**base_reg_params)
    reg.fit(x_tr, y_tr)

    va_pred = reg.predict(x_va)
    mae = metrics.mean_absolute_error(y_va, va_pred)
    val_scores.append((t, float(np.log(mae))))
    print(
        f"type={t:>4s}  n_tr={x_tr.shape[0]:>7d}  n_va={x_va.shape[0]:>7d}  log(MAE)={np.log(mae):.5f}"
    )

    if np.any(te_mask):
        test_pred[te_mask] = reg.predict(test.loc[te_mask, col])

mean_log_mae_over_types = float(np.mean([s for _, s in val_scores]))
print("Validation mean log(MAE) over types:", mean_log_mae_over_types)

test["scalar_coupling_constant"] = test_pred

submission = test[["id", "scalar_coupling_constant"]].copy()
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
assert (
    submission.shape[0] == sub.shape[0]
), "Submission row count must match sample_submission."
assert submission.columns.tolist() == [
    "id",
    "scalar_coupling_constant",
], "Submission columns must match required format."
