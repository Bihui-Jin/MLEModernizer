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
from numpy import linalg as LA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

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
        for line in f.readlines():
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
    n_atom = len(xyz)
    if n_atom == 0:
        return np.ones((1, 1))
    distance_matrix = np.ones((n_atom, n_atom))
    A = np.transpose(lattice)  # (a1, a2, a3)
    B = LA.inv(A)
    for i in range(n_atom):
        for j in range(i):
            r_ij = np.dot(B, xyz[i][0] - xyz[j][0])
            sin_sq_r = (np.sin(np.pi * r_ij)) ** 2
            distance = LA.norm(np.dot(A, sin_sq_r))
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    labels = np.array([atom[1] for atom in xyz]).reshape(-1, 1)
    for at, charge in zip(["O", "Al", "Ga", "In"], [8, 13, 31, 49]):
        labels = np.where(labels == at, charge, labels)
    labels = labels.astype(float)
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
spectrum_list = []
base_path = os.path.join(
    BASE_INPUT, "nomad2018-predict-transparent-conductors"
)  # contains train/ and test/ subfolders
for index in range(df_len):
    dataset_label = df.loc[index, "dataset"]
    row_id = df.loc[index, "id"]
    filename = os.path.join(base_path, dataset_label, str(row_id), "geometry.xyz")
    xyz, lattice = get_xyz(filename)
    sine_matrix = get_sine_matrix(xyz, lattice)
    spectrum = get_eigenspectrum(sine_matrix, target_len=80)
    spectrum_list.append(spectrum)


## === cell 5
fig, axs = plt.subplots(1, 5, figsize=(15, 6))
for i in range(5):
    ax = axs[i]
    plot_data = spectrum_list[i]
    ax.plot(range(len(plot_data)), plot_data)
    ax.hlines(0, 0, len(plot_data), colors="r")
plt.show()


## === cell 6
spectrum_df = pd.DataFrame(spectrum_list, dtype=float).fillna(0)


## === cell 7
ss = StandardScaler()
spectrum_std = ss.fit_transform(spectrum_df.values)
pca = PCA(n_components=30, random_state=42)
spectrum_pca = pca.fit_transform(spectrum_std)
spectrum_pca_df = pd.DataFrame(spectrum_pca)


## === cell 8
df["number_of_total_atoms"] = df["number_of_total_atoms"].astype(int)
df["group_natoms"] = (
    df["spacegroup"].astype(str) + "_" + df["number_of_total_atoms"].astype(str)
)

sns.lmplot(
    x="lattice_vector_1_ang",
    y="bandgap_energy_ev",
    hue="group_natoms",
    data=df,
    fit_reg=False,
)
plt.show()


## === cell 9
df = df.join(pd.get_dummies(df["group_natoms"]))
df.drop(["group_natoms"], axis=1, inplace=True)


## === cell 10
df.drop(["dataset"], axis=1, inplace=True)
df.drop(["id", "spacegroup", "number_of_total_atoms"], axis=1, inplace=True)


## === cell 11
df_new = pd.concat([spectrum_df, df.reset_index(drop=True)], axis=1)


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

xgb_formation = xgb.XGBRegressor(
    random_state=42, n_jobs=4, objective="reg:squarederror"
)
params = {
    "max_depth": [3, 4, 5, 6],
    "n_estimators": [200, 300, 400, 500],
    "learning_rate": [0.05, 0.1],
}
cv_formation = GridSearchCV(
    xgb_formation,
    param_grid=params,
    cv=4,
    verbose=1,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
)
cv_formation.fit(X_train, y_formation_log)


## === cell 14
xgb_bandgap = xgb.XGBRegressor(random_state=42, n_jobs=4, objective="reg:squarederror")
cv_bandgap = GridSearchCV(
    xgb_bandgap,
    param_grid=params,
    cv=4,
    verbose=1,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
)
cv_bandgap.fit(X_train, y_bandgap_log)




## === cell 15
def plot_features(estimator, feature_names):
    impor = estimator.feature_importances_
    plt.figure(figsize=(10, 15))
    plt.barh(range(len(impor)), impor, align="center")
    plt.yticks(range(len(impor)), feature_names)
    plt.show()


feature_names = df_new.drop(
    ["formation_energy_ev_natom", "bandgap_energy_ev"], axis=1
).columns
plot_features(cv_formation.best_estimator_, feature_names)
plot_features(cv_bandgap.best_estimator_, feature_names)


## === cell 16
formation_pred_log = cv_formation.predict(X_test)
bandgap_pred_log = cv_bandgap.predict(X_test)

formation_pred = np.expm1(formation_pred_log)
bandgap_pred = np.expm1(bandgap_pred_log)


## === cell 17
submission = df_test[["id"]].copy()
submission["formation_energy_ev_natom"] = formation_pred
submission["bandgap_energy_ev"] = bandgap_pred


## === cell 18
submission.to_csv("submission.csv", index=False)
