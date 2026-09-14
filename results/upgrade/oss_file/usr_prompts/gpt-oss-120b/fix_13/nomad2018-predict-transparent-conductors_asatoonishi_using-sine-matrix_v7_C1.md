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
Predict the formation energy and bandgap energy of a material.

## Metric
Column-wise root mean squared logarithmic error.

## Submission Format
For each id in the test set, you must predict a value for both formation_energy_ev_natom and bandgap_energy_ev. The file should contain a header and have the following format:
```
id,formation_energy_ev_natom,bandgap_energy_ev
1,0.1779,1.8892
2,0.1779,1.8892
3,0.1779,1.8892
...
```

## Dataset
The following information has been included:

- Spacegroup (a label identifying the symmetry of the material)
- Total number of Al, Ga, In and O atoms in the unit cell ($\N_{total}$)
- Relative compositions of Al, Ga, and In (x, y, z)
- Lattice vectors and angles: lv1, lv2, lv3 (which are lengths given in units of angstroms ($10^{-10}$ meters) and $\alpha, \beta, \gamma$ (which are angles in degrees between 0° and 360°)

Note: For each line of the CSV file, the corresponding spatial positions of all of the atoms in the unit cell (expressed in Cartesian coordinates) are provided as a separate file.

train.csv - contains a set of materials for which the bandgap and formation energies are provided

test.csv - contains the set of materials for which you must predict the bandgap and formation energies

/{train|test}/{id}/geometry.xyz - files with spatial information about the material. The file name corresponds to the id in the respective csv files.

# 2. Python version

3.6

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
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (89 lines)
            sample_submission.csv (241 lines)
            sample_submission.csv.zip (765 Bytes)
            test.csv (241 lines)
            test.csv.zip (6.0 kB)
            test.zip (505.0 kB)
            train.csv (2161 lines)
            train.csv.zip (56.7 kB)
            train.zip (4.5 MB)
            nomad2018-predict-transparent-conductors/
                description.md (89 lines)
                sample_submission.csv (241 lines)
                ... and 7 other files
                nomad2018-predict-transparent-conductors/
                test/
                    1/
                        geometry.xyz (3.0 kB)
                    10/
                        geometry.xyz (3.0 kB)
                    ... and 239 other folders
                train/
                    1/
                        geometry.xyz (5.5 kB)
                    10/
                        geometry.xyz (2.3 kB)
                    ... and 2159 other folders
            test/
                1/
                    geometry.xyz (3.0 kB)
                10/
                    geometry.xyz (3.0 kB)
                ... and 239 other folders
            train/
                1/
                    geometry.xyz (5.5 kB)
                10/
                    geometry.xyz (2.3 kB)
                ... and 2159 other folders
        input/
            description.md (89 lines)
            sample_submission.csv (241 lines)
            sample_submission.csv.zip (765 Bytes)
            test.csv (241 lines)
            test.csv.zip (6.0 kB)
            test.zip (505.0 kB)
            train.csv (2161 lines)
            train.csv.zip (56.7 kB)
            train.zip (4.5 MB)
            nomad2018-predict-transparent-conductors/
                description.md (89 lines)
                sample_submission.csv (241 lines)
                ... and 7 other files
                nomad2018-predict-transparent-conductors/
                test/
                    1/
                        geometry.xyz (3.0 kB)
                    10/
                        geometry.xyz (3.0 kB)
                    ... and 239 other folders
                train/
                    1/
                        geometry.xyz (5.5 kB)
                    10/
                        geometry.xyz (2.3 kB)
                    ... and 2159 other folders
            test/
                1/
                    geometry.xyz (3.0 kB)
                10/
                    geometry.xyz (3.0 kB)
                ... and 239 other folders
            train/
                1/
                    geometry.xyz (5.5 kB)
                10/
                    geometry.xyz (2.3 kB)
                ... and 2159 other folders
        working/
            nomad2018-predict-transparent-conductors/
                description.md (89 lines)
                sample_submission.csv (241 lines)
                ... and 7 other files
                nomad2018-predict-transparent-conductors/
                test/
                    1/
                        geometry.xyz (3.0 kB)
                    10/
                        geometry.xyz (3.0 kB)
                    ... and 239 other folders
                train/
                    1/
                        geometry.xyz (5.5 kB)
                    10/
                        geometry.xyz (2.3 kB)
                    ... and 2159 other folders
```

-> data/nomad2018-predict-transparent-conductors/sample_submission.csv has 240 rows and 3 columns.
The columns are: id, formation_energy_ev_natom, bandgap_energy_ev

-> data/nomad2018-predict-transparent-conductors/test.csv has 240 rows and 12 columns.
The columns are: id, spacegroup, number_of_total_atoms, percent_atom_al, percent_atom_ga, percent_atom_in, lattice_vector_1_ang, lattice_vector_2_ang, lattice_vector_3_ang, lattice_angle_alpha_degree, lattice_angle_beta_degree, lattice_angle_gamma_degree

-> data/nomad2018-predict-transparent-conductors/train.csv has 2160 rows and 14 columns.
The columns are: id, spacegroup, number_of_total_atoms, percent_atom_al, percent_atom_ga, percent_atom_in, lattice_vector_1_ang, lattice_vector_2_ang, lattice_vector_3_ang, lattice_angle_alpha_degree, lattice_angle_beta_degree, lattice_angle_gamma_degree, formation_energy_ev_natom, bandgap_energy_ev

-> data/sample_submission.csv has 240 rows and 3 columns.
The columns are: id, formation_energy_ev_natom, bandgap_energy_ev

-> data/test.csv has 240 rows and 12 columns.
The columns are: id, spacegroup, number_of_total_atoms, percent_atom_al, percent_atom_ga, percent_atom_in, lattice_vector_1_ang, lattice_vector_2_ang, lattice_vector_3_ang, lattice_angle_alpha_degree, lattice_angle_beta_degree, lattice_angle_gamma_degree

-> data/train.csv has 2160 rows and 14 columns.
The columns are: id, spacegroup, number_of_total_atoms, percent_atom_al, percent_atom_ga, percent_atom_in, lattice_vector_1_ang, lattice_vector_2_ang, lattice_vector_3_ang, lattice_angle_alpha_degree, lattice_angle_beta_degree, lattice_angle_gamma_degree, formation_energy_ev_natom, bandgap_energy_ev

-> (stopped after 10 files for performance)

# 5. Target score

0.06818

# 6. Current score

0.27273

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.27281) has done: 'I fixed the deprecated `np.float` usage, guaranteed each material’s eigen‑spectrum has a fixed length (80) so the DataFrame and PCA work, corrected the submission ID handling, removed the Jupyter‑magic line, and added a small safety check when reading geometry files. These changes unblock the pipeline, produce a valid `submission.csv` with the correct columns, and keep the original modeling approach unchanged.'
- What this solution (achieved 0.27354) has done: 'The changes fix the dtype issue in the sine‑matrix construction, correct the geometry file path, and train the models on log‑transformed targets (with exponentiation back to the original scale) to better match the RMSLE metric. These fixes unblock the pipeline, ensure a proper `submission.csv` is written, and move the score toward the target while keeping the original modelling approach unchanged.'
- What this solution (achieved 0.27364) has done: 'The main slowdown comes from over‑subscribing CPU cores: each XGBoost model was already using 4 threads while `GridSearchCV` also parallelised across parameter combinations, causing contention and long runtimes. By setting the regressor’s `n_jobs` to 1 we let `GridSearchCV` fully control parallelism without oversubscribing, which dramatically reduces fitting time while preserving the exact training logic, hyper‑parameter grid and evaluation metric.'
- What this solution (achieved 0.27295) has done: 'The changes focus on eliminating the extremely costly exhaustive GridSearchCV (which required thousands of XGBoost fits) and removing optional plotting that does not affect the final predictions. By fixing a well‑chosen hyperparameter set and fitting the models directly we keep the same XGBoost architecture and loss while reducing runtime dramatically. Plotting cells are stripped to a no‑op to avoid unnecessary computation, and comments explain each optimization, preserving all deterministic settings and the original feature‑engineering pipeline.'
- What this solution (achieved 0.27273) has done: 'Implemented fixes to unblock the pipeline and modestly improve the model:

- Replaced the erroneous `group_natoms` one‑hot encoding with a safe creation of dummy variables for `spacegroup`.
- Adjusted column dropping to retain the informative `number_of_total_atoms` feature while still removing unused identifiers.
- Added explanatory comments for the new steps.

These changes resolve the KeyError, preserve the core modelling approach, and provide a useful categorical feature that should help lower the RMSLE toward the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import linalg as LA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
from joblib import Parallel, delayed  # parallelize feature extraction
import gc  # for explicit garbage collection

BASE_INPUT = "/kaggle/input"

df_train = pd.read_csv(os.path.join(BASE_INPUT, "train.csv"))
df_train["dataset"] = "train"
df_test = pd.read_csv(os.path.join(BASE_INPUT, "test.csv"))
df_test["dataset"] = "test"
test_len = len(df_test)

df = pd.concat([df_train, df_test], axis=0, ignore_index=True, sort=False)
df_len = len(df)




## === cell 1
def get_xyz(filename):
    """Parse geometry.xyz and return atom coordinates with element symbols and lattice vectors."""
    xyz = []
    lattice = []
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Geometry file not found: {filename}")
    with open(filename) as f:
        for line in f:
            row = line.split()
            if len(row) == 0:
                continue
            if row[0] == "atom":
                xyz.append((np.array(row[1:4], dtype=float), row[4]))
            elif row[0] == "lattice_vector":
                lattice.append(np.array(row[1:4], dtype=float))
    return xyz, lattice




## === cell 2
def get_sine_matrix(xyz, lattice):
    """Vectorized version of the original double‑loop implementation."""
    n_atom = len(xyz)
    if n_atom == 0:
        return np.ones((1, 1))
    coords = np.stack([atom[0] for atom in xyz])  # shape (n,3)
    A = np.transpose(lattice)  # (3,3)
    B = LA.inv(A)  # (3,3)

    diff = coords[:, None, :] - coords[None, :, :]  # (n,n,3)
    r_ij = np.tensordot(diff, B, axes=([2], [1]))  # (n,n,3)

    sin_sq = np.sin(np.pi * r_ij) ** 2  # (n,n,3)
    transformed = np.tensordot(sin_sq, A, axes=([2], [0]))  # (n,n,3)
    distance_matrix = LA.norm(transformed, axis=2)  # (n,n)
    np.fill_diagonal(distance_matrix, 1.0)

    charge_map = {"O": 8, "Al": 13, "Ga": 31, "In": 49}
    elem_array = [atom[1] for atom in xyz]
    labels = np.array([charge_map.get(e, 0) for e in elem_array], dtype=float).reshape(
        -1, 1
    )

    charge_matrix = np.dot(labels, labels.T).astype(float)
    np.fill_diagonal(charge_matrix, 0.0)
    charge_matrix += np.diag(0.5 * (labels.flatten() ** 2.4))

    sine_matrix = charge_matrix / distance_matrix
    return sine_matrix




## === cell 3
def get_eigenspectrum(matrix, target_len=80):
    """Return the sorted eigenvalues, padded or truncated to target_len."""
    spectrum = LA.eigvalsh(matrix)
    spectrum = np.sort(spectrum)[::-1]  # descending
    if len(spectrum) < target_len:
        spectrum = np.pad(spectrum, (0, target_len - len(spectrum)), constant_values=0)
    else:
        spectrum = spectrum[:target_len]
    return spectrum




## === cell 4
def compute_spectrum(info):
    """Helper for parallel execution: read file, compute sine matrix and eigenspectrum."""
    dataset_label, row_id = info
    filename = os.path.join(
        BASE_INPUT,
        "nomad2018-predict-transparent-conductors",
        dataset_label,
        str(row_id),
        "geometry.xyz",
    )
    xyz, lattice = get_xyz(filename)
    sine_matrix = get_sine_matrix(xyz, lattice)
    spectrum = get_eigenspectrum(sine_matrix, target_len=80)
    return spectrum


row_info = [(df.loc[i, "dataset"], df.loc[i, "id"]) for i in range(df_len)]

max_jobs = max(1, os.cpu_count())
spectrum_list = Parallel(n_jobs=max_jobs, backend="loky")(
    delayed(compute_spectrum)(info) for info in row_info
)
gc.collect()




## === cell 5
pass




## === cell 6
spectrum_df = pd.DataFrame(spectrum_list, dtype=float).fillna(0)




## === cell 7
ss = StandardScaler()
spectrum_std = ss.fit_transform(spectrum_df.values)
pca = PCA(n_components=30, random_state=42)
spectrum_pca = pca.fit_transform(spectrum_std)
pca_columns = [f"pca_{i}" for i in range(spectrum_pca.shape[1])]
spectrum_pca_df = pd.DataFrame(spectrum_pca, columns=pca_columns)




## === cell 8
pass




## === cell 9
spacegroup_dummies = pd.get_dummies(df["spacegroup"], prefix="sg")
df = pd.concat([df, spacegroup_dummies], axis=1)
df.drop(["spacegroup"], axis=1, inplace=True)




## === cell 10
df.drop(["dataset"], axis=1, inplace=True)
df.drop(["id"], axis=1, inplace=True)




## === cell 11
df_new = pd.concat([spectrum_pca_df, df.reset_index(drop=True)], axis=1)




## === cell 12
rows_to_drop = [
    395,
    126,
    1215,
    1886,
    2075,
    353,
    308,
    2154,
    531,
    1379,
    2319,
    2337,
    2370,
    2333,
]
df_new.drop(rows_to_drop, axis=0, inplace=True, errors="ignore")
df_new.reset_index(drop=True, inplace=True)




## === cell 13
df_len = len(df_new)
train_len = df_len - test_len

X = df_new.drop(["formation_energy_ev_natom", "bandgap_energy_ev"], axis=1).values
X_train = X[:train_len]
X_test = X[train_len:]

y_formation = df_new["formation_energy_ev_natom"].values[:train_len]
y_bandgap = df_new["bandgap_energy_ev"].values[:train_len]

y_formation_log = np.log1p(y_formation)
y_bandgap_log = np.log1p(y_bandgap)

fixed_params = {
    "max_depth": 6,  # modestly deeper trees
    "n_estimators": 800,  # more boosting rounds
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "objective": "reg:squarederror",
    "tree_method": "hist",
}

xgb_formation = xgb.XGBRegressor(**fixed_params)
xgb_formation.fit(X_train, y_formation_log)




## === cell 14
xgb_bandgap = xgb.XGBRegressor(**fixed_params)
xgb_bandgap.fit(X_train, y_bandgap_log)




## === cell 15
pass




## === cell 16
formation_pred_log = xgb_formation.predict(X_test)
bandgap_pred_log = xgb_bandgap.predict(X_test)

formation_pred = np.expm1(formation_pred_log)
bandgap_pred = np.expm1(bandgap_pred_log)




## === cell 17
submission = df_test[["id"]].copy()
submission["formation_energy_ev_natom"] = formation_pred
submission["bandgap_energy_ev"] = bandgap_pred




## === cell 18
submission.to_csv("submission.csv", index=False)
