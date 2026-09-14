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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

2.20425

# 6. Current score

4.50164

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 2.50959) has done: 'Diagnosis: The crash happens in cell 23 when calling `lr_model.predict(X_test_aligned)` because `X_test_aligned` contains NaN values. These NaNs are introduced earlier (e.g., from merges in `map_atom_data` or from feature engineering) and persist through `get_dummies`/`reindex`, and `LinearRegression` cannot handle NaNs at prediction time.  
Patch summary: In cell 23, after aligning test columns to train, explicitly replace any remaining NaN/inf values with finite defaults (0) before prediction. This is the smallest localized change that unblocks execution without changing the model or feature set.  
Updated cells: Only cell 23 is modified.  
Compatibility notes for cell k+1: No interface/variable names are changed; `y_pred`, `SCC`, and the output CSV are produced as before.  
Assumptions: Filling missing engineered/merged feature values with 0 is acceptable for this baseline linear model and preserves the intended evaluation semantics (same features, just made numerically valid).'
- What this solution (achieved 2.50959) has done: 'You’re currently above the target (2.50959 vs 2.20425, lower is better), so we should make a minimal, legitimate improvement without changing the model family or training loop. The biggest actionable issue is a bug in the distance feature: in cell 12, `test_m_1` mistakenly uses `x_0,y_0,z_0` instead of `x_1,y_1,z_1`, making `dist_vector` wrong for all test rows and hurting MAE. Fixing that single line preserves the exact same feature set/architecture but corrects the test feature to match train semantics, which should move the score down toward the target. I keep your existing NaN/inf handling and submission writing intact to ensure end-to-end execution and a valid CSV.'
- What this solution (achieved 4.07367) has done: 'We’re currently worse than the target (2.50959 vs 2.20425; lower is better), so the smallest legitimate improvement is to better match the metric: the competition scores MAE per `type`, so we should fit separate linear regressions per coupling `type` rather than one global model (this keeps the same model family and training approach, just applied per group). This typically reduces error because different coupling types have very different ranges/relationships, and it’s a minimal change that directly targets the metric. I keep your existing feature set, dummy encoding, alignment, and NaN/inf handling, and only adjust training/prediction to be per-type and written back into the sample submission by `id`. The script still run end-to-end and write a valid `Linear_Regression_model.csv`.'
- What this solution (achieved 4.07428) has done: 'Your current score (4.07367, lower is better) is far from the target (2.20425), so we need a meaningful but still minimal change that directly matches the metric. The biggest issue is that your per-type training loop is currently using the one-hot `type_*` columns as both the selector and as a feature, which provides no benefit within each type and can cause instability; we drop all `type_*` columns from the per-type feature matrix while still training separate models per type. In addition, we do the same for `id` (it should not be a predictive feature) while keeping it for output alignment, which often improves MAE for linear models without changing the model family or training approach. Everything else (feature set, one-hot encoding of atoms, distance feature, per-type LinearRegression, NaN/inf handling, and submission writing) stays the same.'
- What this solution (achieved 4.1031) has done: 'Your current score (4.07428, lower is better) is still far from the target (2.20425), so we need a small but meaningful improvement that keeps the same linear-regression-per-type core logic. The biggest remaining issue is that we’re not using any of the provided per-atom/per-molecule auxiliary tables (mulliken charges, magnetic shielding tensors, dipole moments, potential energy), which can be merged in with simple joins and usually cuts MAE substantially without changing the model family or training loop. I add minimal feature merges for atom_0/atom_1 charges and shielding tensors, plus molecule-level dipole/potential energy, then keep your existing one-hot encoding, per-type LinearRegression training, and NaN/inf cleaning unchanged. This should legitimately improve accuracy and move the score downward toward the target while still producing the same submission format.'
- What this solution (achieved 4.1031) has done: 'We’re currently worse than the target (4.1031 vs 2.20425; lower is better), so we make a minimal, legitimate improvement that better matches the competition metric while preserving your linear-regression-per-type core logic. The biggest missing piece is that the target distribution differs hugely by coupling type, so training on the raw target can overweight large-magnitude types; we instead train each per-type LinearRegression on a standardized target (z-score within that type) and then invert the transform on predictions. This keeps the same model family, same per-type loop, same features/merges/one-hot encoding, and same submission semantics—only a per-type target scaling is added. We keep your existing NaN/inf cleaning and column alignment to ensure end-to-end execution and a valid CSV.'
- What this solution (achieved 4.1031) has done: 'Your current score (4.1031, lower is better) is still far from the target (2.20425), so the smallest meaningful move is to reduce systematic error without changing the model family or training loop. The biggest remaining issue is that the per-type selector currently relies on one-hot `type_*` columns created from the features, which can desync from the true `type` labels after dummy alignment; instead we should use the original `type` column from `train_df/test_df` as the group key and train/predict per-type directly. While doing that, we keep the exact same feature matrix (dropping `id` and any `type_*` columns as before) and the same per-type target standardization/inversion, but ensure each test row is predicted by the correct type-specific model. This is a minimal change localized to the training/prediction cell and should move the score downward toward the target while preserving end-to-end submission generation.'
- What this solution (achieved 4.10278) has done: 'Your current score (4.1031, lower is better) is still well above the target (2.20425), so we should make a small, legitimate accuracy improvement without changing the core “per-type LinearRegression on engineered/merged features” approach. The biggest issue is that we are still including absolute coordinates (`x_0,y_0,z_0,x_1,y_1,z_1`) which are not physically meaningful (the target is invariant to translation/rotation) and can inject noise; replacing those with a rotation/translation-invariant relative vector (`dx,dy,dz`) keeps the same information content but improves generalization for a linear model. This is a minimal feature tweak that preserves the same model family, per-type loop, and target standardization logic. Everything else (merges, one-hot encoding, NaN/inf handling, and submission writing) stays intact.'
- What this solution (achieved 6.77051) has done: 'We’re far above the target (4.10278 vs 2.20425; lower is better), so we need a small but meaningful accuracy gain while keeping your exact “per-type LinearRegression on engineered/merged features” core logic. The most impactful minimal change is to apply a log1p transform to the distance feature (`dist_vector`) and also include interaction features between distance and the relative vector (`dx,dy,dz`), which helps a linear model capture non-linear distance decay without changing model family or training loop. We keep all your existing merges, one-hot encoding, per-type target standardization/inversion, and NaN/inf cleaning, and we still write the same submission CSV format. These added features are lightweight and should move MAE down toward the target.'
- What this solution (achieved 4.10278) has done: 'Your score regressed badly (6.77 vs target 2.204, lower is better), so we should undo the last “interaction/nonlinear feature” change that likely destabilized the linear model and increased error. I keep your core approach intact (same per-type LinearRegression loop, same auxiliary merges, same NaN/inf cleaning, same submission writing), but remove only the added `dist_log1p` and `*_over_dist` features from both feature engineering and the Attributes list. This is a minimal, directly score-relevant rollback that restores the previously-stable distance + relative vector features and should move the score back down toward the target band. No paths, file names, or training semantics change, and it still writes `Linear_Regression_model.csv`.'
- What this solution (achieved 4.41838) has done: 'Your current score (4.10278, lower is better) is still far above the target (2.20425), so we should make a small, legitimate improvement without changing the core “per-type LinearRegression on engineered/merged features” approach. The biggest accuracy leak in your current setup is that you are fitting on raw numeric atom indices (`atom_index_0/1`), which act like arbitrary IDs and often hurt generalization; we instead use simple, stable graph features derived from `structures.csv`: the per-atom degree (# of bonds) for each endpoint and the shortest-path (bond) distance between the two atoms. This preserves your model family, per-type loop, and loss/metric semantics, while adding physically meaningful features and removing the noisy index features. Everything else (aux merges, one-hot encoding, NaN/inf cleaning, per-type target standardization/inversion, and CSV writing) stays intact.'
- What this solution (achieved 4.41838) has done: 'Your current score (4.41838, lower-is-better) is far above the target (2.20425), so we need a meaningful but still minimal accuracy improvement while keeping your “per-type LinearRegression on engineered/merged features” core logic unchanged. The biggest issue is that the new bond-graph features are computed via a very expensive full O(n²) distance matrix per molecule and a potentially mismatched tolerance; this can both time out/strain resources and also introduce noisy/incorrect “bonds” that hurt the shortest-path feature. I switch the bond construction to a k-nearest-neighbor (kNN) candidate edge approach (still using the same covalent radii + tolerance rule) to get a cleaner adjacency and make the feature computation consistent and fast within the 600s budget. Everything else (features used, per-type standardization, NaN/inf cleaning, submission format/path) stays the same.'
- What this solution (achieved 4.50164) has done: 'Your current score (4.41838, lower-is-better) is far above the target (2.20425), so we need a meaningful but still minimal change that keeps your same per-type LinearRegression training loop and feature set intact. The biggest issue is that the bond graph features are likely noisy because the kNN candidate set can miss true bonds (especially in denser regions), which makes `degree_*` and `bond_path_len` unreliable and hurts MAE. I keep the exact same covalent-radii+tolerance bond rule and the same downstream features, but increase the kNN candidate size and make the path search depth consistent with the “far/unknown” bucket so fewer pairs are incorrectly truncated. This is localized to the bond-graph feature cell and should improve accuracy without changing model family or evaluation semantics, while still producing the same submission CSV.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import warnings
import os

warnings.filterwarnings("ignore")

print(os.listdir("../input"))



## === cell 1
pot_energy = pd.read_csv("../input/potential_energy.csv")
mulliken_charges = pd.read_csv("../input/mulliken_charges.csv")
train_df = pd.read_csv("../input/train.csv")
scalar_coupling_cont = pd.read_csv("../input/scalar_coupling_contributions.csv")
test_df = pd.read_csv("../input/test.csv")
magnetic_shield_tensor = pd.read_csv("../input/magnetic_shielding_tensors.csv")
dipole_moment = pd.read_csv("../input/dipole_moments.csv")
structures = pd.read_csv("../input/structures.csv")



## === cell 2
print("Shape of potential energy dataset:", pot_energy.shape)
print("Shape of mulliken_charges dataset:", mulliken_charges.shape)
print("Shape of train dataset:", train_df.shape)
print("Shape of scalar coupling contributions dataset:", scalar_coupling_cont.shape)
print("Shape of test dataset:", test_df.shape)
print("Shape of magnetic shielding tensors dataset:", magnetic_shield_tensor.shape)
print("Shape of dipole moments dataset:", dipole_moment.shape)
print("Shape of structures dataset:", structures.shape)



## === cell 3
print("Data Types:\n", pot_energy.dtypes)
print("Descriptive statistics:\n", np.round(pot_energy.describe(), 3))
pot_energy.head(6)



## === cell 4
print("Data Types:\n", mulliken_charges.dtypes)
print("Descriptive statistics:\n", np.round(mulliken_charges.describe(), 3))
mulliken_charges.head(6)



## === cell 5
print("Data Types:\n", train_df.dtypes)
print("Descriptive statistics:\n", np.round(train_df.describe(), 3))
train_df.head(6)



## === cell 6
print("Data Types:\n", scalar_coupling_cont.dtypes)
print("Descriptive statistics:\n", np.round(scalar_coupling_cont.describe(), 3))
scalar_coupling_cont.head(6)



## === cell 7
print("Data Types:\n", test_df.dtypes)
print("Descriptive statistics:\n", np.round(test_df.describe(), 3))
test_df.head(6)



## === cell 8
print("Data Types:\n", magnetic_shield_tensor.dtypes)
print("Descriptive statistics:\n", np.round(magnetic_shield_tensor.describe(), 3))
magnetic_shield_tensor.head(6)



## === cell 9
print("Data Types:\n", dipole_moment.dtypes)
print("Descriptive statistics:\n", np.round(dipole_moment.describe(), 3))
dipole_moment.head(6)



## === cell 10
print("Data Types:\n", structures.dtypes)
print("Descriptive statistics:\n", np.round(structures.describe(), 3))
structures.head(6)




## === cell 11
def map_atom_data(df, atom_idx):
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


train_df = map_atom_data(train_df, 0)
train_df = map_atom_data(train_df, 1)
test_df = map_atom_data(test_df, 0)
test_df = map_atom_data(test_df, 1)



## === cell 12
train_m_0 = train_df[["x_0", "y_0", "z_0"]].values
train_m_1 = train_df[["x_1", "y_1", "z_1"]].values
test_m_0 = test_df[["x_0", "y_0", "z_0"]].values
test_m_1 = test_df[["x_1", "y_1", "z_1"]].values  # keep correct test atom_1 coords

train_df["dist_vector"] = np.linalg.norm(train_m_0 - train_m_1, axis=1)
test_df["dist_vector"] = np.linalg.norm(test_m_0 - test_m_1, axis=1)

train_df["dx"] = train_df["x_0"] - train_df["x_1"]
train_df["dy"] = train_df["y_0"] - train_df["y_1"]
train_df["dz"] = train_df["z_0"] - train_df["z_1"]

test_df["dx"] = test_df["x_0"] - test_df["x_1"]
test_df["dy"] = test_df["y_0"] - test_df["y_1"]
test_df["dz"] = test_df["z_0"] - test_df["z_1"]



## === cell 13
display(train_df.head(6))



## === cell 14
display(test_df.head(10))




## === cell 15
def add_aux_features(df):
    out = df

    out = out.merge(pot_energy, on="molecule_name", how="left")
    out = out.merge(
        dipole_moment, on="molecule_name", how="left", suffixes=("", "_dip")
    )

    mc0 = mulliken_charges.rename(
        columns={"atom_index": "atom_index_0", "mulliken_charge": "mulliken_charge_0"}
    )
    mc1 = mulliken_charges.rename(
        columns={"atom_index": "atom_index_1", "mulliken_charge": "mulliken_charge_1"}
    )
    out = out.merge(
        mc0[["molecule_name", "atom_index_0", "mulliken_charge_0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    out = out.merge(
        mc1[["molecule_name", "atom_index_1", "mulliken_charge_1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
    )

    mst0 = magnetic_shield_tensor.rename(
        columns={
            "atom_index": "atom_index_0",
            "XX": "mst_XX_0",
            "YX": "mst_YX_0",
            "ZX": "mst_ZX_0",
            "XY": "mst_XY_0",
            "YY": "mst_YY_0",
            "ZY": "mst_ZY_0",
            "XZ": "mst_XZ_0",
            "YZ": "mst_YZ_0",
            "ZZ": "mst_ZZ_0",
        }
    )
    mst1 = magnetic_shield_tensor.rename(
        columns={
            "atom_index": "atom_index_1",
            "XX": "mst_XX_1",
            "YX": "mst_YX_1",
            "ZX": "mst_ZX_1",
            "XY": "mst_XY_1",
            "YY": "mst_YY_1",
            "ZY": "mst_ZY_1",
            "XZ": "mst_XZ_1",
            "YZ": "mst_YZ_1",
            "ZZ": "mst_ZZ_1",
        }
    )
    out = out.merge(
        mst0[
            [
                "molecule_name",
                "atom_index_0",
                "mst_XX_0",
                "mst_YX_0",
                "mst_ZX_0",
                "mst_XY_0",
                "mst_YY_0",
                "mst_ZY_0",
                "mst_XZ_0",
                "mst_YZ_0",
                "mst_ZZ_0",
            ]
        ],
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    out = out.merge(
        mst1[
            [
                "molecule_name",
                "atom_index_1",
                "mst_XX_1",
                "mst_YX_1",
                "mst_ZX_1",
                "mst_XY_1",
                "mst_YY_1",
                "mst_ZY_1",
                "mst_XZ_1",
                "mst_YZ_1",
                "mst_ZZ_1",
            ]
        ],
        on=["molecule_name", "atom_index_1"],
        how="left",
    )

    return out


train_df = add_aux_features(train_df)
test_df = add_aux_features(test_df)




## === cell 16
def build_bond_graph_features(structures_df, train_df, test_df):
    """
    Change made to move score down (closer to target) without changing model family/loop:
    - The previous kNN candidate size (k=12) can miss true covalent bonds, making degree/path noisy.
      We increase k moderately (still local and fast) so that the SAME covalent radii + tolerance
      bond rule is applied on a more complete candidate set.
    - Also make max_depth consistent with the "far/unknown bucket" (return max_depth+1).
      This reduces truncation errors in bond_path_len for larger molecules.
    """
    s = structures_df[["molecule_name", "atom_index", "atom", "x", "y", "z"]].copy()

    covalent_radii = {
        "H": 0.31,
        "C": 0.76,
        "N": 0.71,
        "O": 0.66,
        "F": 0.57,
        "P": 1.07,
        "S": 1.05,
        "Cl": 1.02,
        "Br": 1.20,
        "I": 1.39,
    }

    s["r"] = s["atom"].map(covalent_radii).astype(np.float32)
    s[["x", "y", "z"]] = s[["x", "y", "z"]].astype(np.float32)

    degree_rows = []
    adj_by_mol = {}

    tol = 0.45
    k = 24  # increased from 12 to reduce missed true bonds while staying efficient
    max_depth = 8  # slightly deeper search to reduce "truncated far" errors

    for mol, g in s.groupby("molecule_name", sort=False):
        ai = g["atom_index"].to_numpy(np.int32)
        coords = g[["x", "y", "z"]].to_numpy(np.float32)
        radii = g["r"].to_numpy(np.float32)

        n = len(g)
        if n == 0:
            continue

        adj = {int(a): set() for a in ai.tolist()}
        deg = np.zeros(n, dtype=np.int16)

        for i in range(n):
            d = coords - coords[i]
            dist2 = np.sum(d * d, axis=1)
            dist2[i] = np.inf

            kk = min(k, n - 1)
            if kk <= 0:
                continue
            cand = np.argpartition(dist2, kk)[:kk]

            cutoff = radii[i] + radii[cand] + tol
            bonded_mask = dist2[cand] <= (cutoff * cutoff)

            neigh_idx = cand[bonded_mask]
            if neigh_idx.size:
                a_i = int(ai[i])
                for j in neigh_idx.tolist():
                    a_j = int(ai[j])
                    if a_j != a_i:
                        adj[a_i].add(a_j)
                        adj[a_j].add(a_i)

        for idx, a in enumerate(ai.tolist()):
            deg[idx] = len(adj[int(a)])

        degree_rows.append(
            pd.DataFrame({"molecule_name": mol, "atom_index": ai, "degree": deg})
        )
        adj_by_mol[mol] = {k_: list(v_) for k_, v_ in adj.items()}

    degree_df = pd.concat(degree_rows, ignore_index=True)

    def add_degree(df, atom_col, out_col):
        tmp = degree_df.rename(columns={"atom_index": atom_col, "degree": out_col})
        return df.merge(
            tmp[["molecule_name", atom_col, out_col]],
            on=["molecule_name", atom_col],
            how="left",
        )

    def shortest_path_len(adj, start, goal, max_depth=max_depth):
        if start == goal:
            return 0
        visited = {start}
        frontier = [start]
        depth = 0
        while frontier and depth < max_depth:
            depth += 1
            nxt = []
            for u in frontier:
                for v in adj.get(u, []):
                    if v == goal:
                        return depth
                    if v not in visited:
                        visited.add(v)
                        nxt.append(v)
            frontier = nxt
        return max_depth + 1  # "far/unknown" bucket

    def add_path_feature(df):
        sp = np.empty(len(df), dtype=np.int16)
        mols = df["molecule_name"].astype(str).values
        a0 = df["atom_index_0"].values
        a1 = df["atom_index_1"].values
        for i in range(len(df)):
            adj = adj_by_mol.get(mols[i])
            if adj is None:
                sp[i] = max_depth + 1
            else:
                sp[i] = shortest_path_len(
                    adj, int(a0[i]), int(a1[i]), max_depth=max_depth
                )
        df["bond_path_len"] = sp.astype(np.float32)
        return df

    train_df2 = add_degree(train_df, "atom_index_0", "degree_0")
    train_df2 = add_degree(train_df2, "atom_index_1", "degree_1")
    train_df2 = add_path_feature(train_df2)

    test_df2 = add_degree(test_df, "atom_index_0", "degree_0")
    test_df2 = add_degree(test_df2, "atom_index_1", "degree_1")
    test_df2 = add_path_feature(test_df2)

    return train_df2, test_df2


train_df, test_df = build_bond_graph_features(structures, train_df, test_df)



## === cell 17
train_df["type"] = train_df.type.astype("category")
train_df["atom_0"] = train_df.atom_0.astype("category")
train_df["atom_1"] = train_df.atom_1.astype("category")

test_df["type"] = test_df.type.astype("category")
test_df["atom_0"] = test_df.atom_0.astype("category")
test_df["atom_1"] = test_df.atom_1.astype("category")



## === cell 18
Attributes = [
    "id",
    "molecule_name",
    "type",
    "atom_0",
    "atom_1",
    "dist_vector",
    "dx",
    "dy",
    "dz",
    "degree_0",
    "degree_1",
    "bond_path_len",
    "potential_energy",
    "X",
    "Y",
    "Z",
    "mulliken_charge_0",
    "mulliken_charge_1",
    "mst_XX_0",
    "mst_YX_0",
    "mst_ZX_0",
    "mst_XY_0",
    "mst_YY_0",
    "mst_ZY_0",
    "mst_XZ_0",
    "mst_YZ_0",
    "mst_ZZ_0",
    "mst_XX_1",
    "mst_YX_1",
    "mst_ZX_1",
    "mst_XY_1",
    "mst_YY_1",
    "mst_ZY_1",
    "mst_XZ_1",
    "mst_YZ_1",
    "mst_ZZ_1",
]
cat_attributes = ["type", "atom_0", "atom_1"]
target_label = ["scalar_coupling_constant"]

X_train = train_df[Attributes]
X_test = test_df[Attributes]
y_target = train_df[target_label]



## === cell 19
print(X_train.shape, X_test.shape)



## === cell 20
display(y_target.shape)



## === cell 21
X_train = pd.get_dummies(X_train, columns=cat_attributes)
print("shape of tranformed train dataframe:", X_train.shape)
X_test = pd.get_dummies(X_test, columns=cat_attributes)
print("shape of transformed test dataframe:", X_test.shape)



## === cell 22
X_train = X_train.drop("molecule_name", axis=1)
X_train.head(6)



## === cell 23
X_test = X_test.drop("molecule_name", axis=1)
X_test.head(5)



## === cell 24
from sklearn import linear_model

X_test_aligned = X_test.reindex(columns=X_train.columns, fill_value=0)
X_test_aligned = X_test_aligned.replace([np.inf, -np.inf], np.nan).fillna(0)
X_train_clean = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)

type_cols = [c for c in X_train_clean.columns if c.startswith("type_")]
feature_cols = [c for c in X_train_clean.columns if c not in (["id"] + type_cols)]

train_types = train_df["type"].astype(str).values
test_types = test_df["type"].astype(str).values

unique_types = np.unique(train_types)

y_pred_all = np.zeros(len(X_test_aligned), dtype=np.float64)

for t in unique_types:
    train_mask = train_types == t
    test_mask = test_types == t

    if train_mask.sum() == 0:
        continue

    y_t = y_target.loc[train_mask].values.ravel().astype(np.float64)
    mu = y_t.mean()
    sigma = y_t.std()
    if not np.isfinite(sigma) or sigma == 0.0:
        sigma = 1.0
    y_t_std = (y_t - mu) / sigma

    lr_model_t = linear_model.LinearRegression()
    lr_model_t.fit(X_train_clean.loc[train_mask, feature_cols], y_t_std)

    if test_mask.sum() > 0:
        pred_std = lr_model_t.predict(
            X_test_aligned.loc[test_mask, feature_cols]
        ).astype(np.float64)
        y_pred_all[test_mask] = pred_std * sigma + mu

SCC = pd.read_csv("../input/sample_submission.csv")
SCC["scalar_coupling_constant"] = y_pred_all
SCC.to_csv("Linear_Regression_model.csv", index=False)
print("Wrote submission:", "Linear_Regression_model.csv", "rows:", len(SCC))
