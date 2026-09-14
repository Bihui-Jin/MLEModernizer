# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import xgboost as xgb
from sklearn.model_selection import GridSearchCV




## === cell 1
possible_paths = [
    Path("data/nomad2018-predict-transparent-conductors"),
    Path("/kaggle/input/nomad2018-predict-transparent-conductors"),
    Path("/kaggle/working/data/nomad2018-predict-transparent-conductors"),
]
BASE_DIR = next((p for p in possible_paths if p.is_dir()), None)
if BASE_DIR is None:
    raise FileNotFoundError("Could not locate the competition data directory.")

df_train = pd.read_csv(BASE_DIR / "train.csv")
df_train["dataset"] = "train"
df_test = pd.read_csv(BASE_DIR / "test.csv")
df_test["dataset"] = "test"

test_len = len(df_test)
df = pd.concat([df_train, df_test], axis=0, ignore_index=True, sort=False)
df_len = len(df)




## === cell 2
def get_xyz(filename: Path):
    """Parse geometry.xyz and return atom coordinates & lattice vectors."""
    xyz = []
    lattice = []
    with open(filename, "r") as f:
        for line in f.readlines():
            row = line.split()
            if not row:
                continue
            if row[0] == "atom":
                xyz.append((np.array(row[1:4], dtype=float), row[4]))
            elif row[0] == "lattice_vector":
                lattice.append(np.array(row[1:4], dtype=float))
    return xyz, lattice




## === cell 3
def get_sine_matrix(xyz, lattice):
    """Create a sine‑based interaction matrix from atomic positions."""
    n_atom = len(xyz)
    if n_atom == 0:
        return np.zeros((1, 1), dtype=float)

    distance_matrix = np.ones((n_atom, n_atom), dtype=float)
    A = np.transpose(lattice)  # (a1,a2,a3) as columns
    B = np.linalg.inv(A)  # reciprocal lattice

    for i in range(n_atom):
        for j in range(i):
            r_ij = np.dot(B, xyz[i][0] - xyz[j][0])
            sin_sq_r = (np.sin(np.pi * r_ij)) ** 2
            distance = np.linalg.norm(np.dot(A, sin_sq_r))
            distance_matrix[i, j] = distance_matrix[j, i] = distance

    charge_map = {"O": 8, "Al": 13, "Ga": 31, "In": 49}
    labels = np.array(
        [charge_map.get(atom[1], 0) for atom in xyz], dtype=float
    ).reshape(-1, 1)

    charge_matrix = np.dot(labels, labels.T)  # outer product
    np.fill_diagonal(charge_matrix, 0.0)
    charge_matrix += np.diag(0.5 * (labels.squeeze() ** 2.4))

    sine_matrix = charge_matrix / distance_matrix
    return sine_matrix




## === cell 4
def get_eigenspectrum(matrix):
    spectrum = np.linalg.eigvalsh(matrix)
    spectrum = np.sort(spectrum)[::-1]
    return spectrum




## === cell 5
spectrum_list = []
max_len = 0

for idx in range(df_len):
    dataset_label = df.loc[idx, "dataset"]
    row_id = df.loc[idx, "id"]
    geom_path = BASE_DIR / dataset_label / str(row_id) / "geometry.xyz"
    if not geom_path.is_file():
        spectrum = np.zeros(1, dtype=float)
    else:
        xyz, lattice = get_xyz(geom_path)
        sine_matrix = get_sine_matrix(xyz, lattice)
        spectrum = get_eigenspectrum(sine_matrix)

    spectrum_list.append(spectrum)
    if spectrum.shape[0] > max_len:
        max_len = spectrum.shape[0]

spectrum_array = np.vstack(
    [np.pad(s, (0, max_len - s.shape[0]), constant_values=0) for s in spectrum_list]
)




## === cell 6
fig, axs = plt.subplots(1, 5, figsize=(15, 6))
for i in range(min(5, spectrum_array.shape[0])):
    axs[i].plot(spectrum_array[i])
    axs[i].hlines(0, 0, max_len, colors="r")
plt.close(fig)  # close to avoid display issues




## === cell 7
spectrum_df = pd.DataFrame(spectrum_array, dtype=float).fillna(0)
df_new = pd.concat([spectrum_df, df.reset_index(drop=True)], axis=1)
df_new.head()




## === cell 8
df_new.drop(["dataset", "id"], axis=1, inplace=True)




## === cell 9
df_len = len(df_new)
train_len = df_len - test_len

X = df_new.drop(["formation_energy_ev_natom", "bandgap_energy_ev"], axis=1).values
X_train = X[:train_len]
X_test = X[train_len:]

y_formation = df_new["formation_energy_ev_natom"].values[:train_len]
y_bandgap = df_new["bandgap_energy_ev"].values[:train_len]




## === cell 10
xgb_formation = xgb.XGBRegressor(
    objective="reg:squarederror", n_jobs=4, random_state=42
)
param_grid = {"max_depth": [2, 3, 4], "n_estimators": [100, 200, 300]}
cv_formation = GridSearchCV(xgb_formation, param_grid, cv=4, verbose=0, n_jobs=4)
cv_formation.fit(X_train, y_formation)




## === cell 11
cv_formation.best_params_




## === cell 12
xgb_bandgap = xgb.XGBRegressor(objective="reg:squarederror", n_jobs=4, random_state=42)
cv_bandgap = GridSearchCV(xgb_bandgap, param_grid, cv=4, verbose=0, n_jobs=4)
cv_bandgap.fit(X_train, y_bandgap)




## === cell 13
cv_bandgap.best_params_




## === cell 14
def plot_features(estimator, feature_names):
    importances = estimator.feature_importances_
    plt.figure(figsize=(10, 12))
    plt.barh(range(len(importances)), importances, align="center")
    plt.yticks(np.arange(len(importances)), feature_names)
    plt.gca().invert_yaxis()
    plt.show()


features = df_new.drop(
    ["formation_energy_ev_natom", "bandgap_energy_ev"], axis=1
).columns
plot_features(cv_formation.best_estimator_, features)
plot_features(cv_bandgap.best_estimator_, features)




## === cell 15
formation_pred = cv_formation.predict(X_test)
bandgap_pred = cv_bandgap.predict(X_test)




## === cell 16
submission = pd.DataFrame(
    {
        "id": df_test["id"].values,
        "formation_energy_ev_natom": formation_pred,
        "bandgap_energy_ev": bandgap_pred,
    }
)
submission.head()




## === cell 17
submission.to_csv("submission.csv", index=False)
