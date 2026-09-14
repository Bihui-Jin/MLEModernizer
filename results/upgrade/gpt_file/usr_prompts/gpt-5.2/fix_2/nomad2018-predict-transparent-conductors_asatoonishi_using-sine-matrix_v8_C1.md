# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.06668

# 6. Current score

0.26176

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 0.26176) has done: 'I fix the runtime blockers caused by deprecated `np.float`, Jupyter-only magic (`%matplotlib inline`), and incorrect file paths so geometry files can be read and features can be built end-to-end. I also make the spectrum feature matrix a consistent fixed length (pad/truncate) so PCA and downstream concatenation always work, then ensure train/test rows remain aligned. Finally, I generate the submission using the provided `sample_submission.csv` as a template to guarantee correct column names and id alignment, and clip predictions to be non-negative to avoid invalid values for RMSLE.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import numpy.linalg as LA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

np.random.seed(0)

BASE_INPUT = "../input/nomad2018-predict-transparent-conductors"



## === cell 1
train_csv = os.path.join(BASE_INPUT, "train.csv")
test_csv = os.path.join(BASE_INPUT, "test.csv")
sample_sub_csv = os.path.join(BASE_INPUT, "sample_submission.csv")

df_train = pd.read_csv(train_csv)
df_train["dataset"] = "train"
df_test = pd.read_csv(test_csv)
df_test["dataset"] = "test"

test_len = len(df_test)
df = pd.concat([df_train, df_test], axis=0, ignore_index=True, sort=False)
df_len = len(df)

print("train:", df_train.shape, "test:", df_test.shape, "all:", df.shape)



## === cell 2
df.head()



## === cell 3
df.tail()




## === cell 4
def get_xyz(filename):
    xyz = []
    lattice = []
    with open(filename, "r") as f:
        for line in f:
            row = line.split()
            if not row:
                continue
            if row[0] == "atom":
                xyz.append((np.array(row[1:4], dtype=float), row[4]))
            elif row[0] == "lattice_vector":
                lattice.append(np.array(row[1:4], dtype=float))
    return xyz, lattice




## === cell 5
def get_sine_matrix(xyz, lattice):
    n_atom = len(xyz)
    distance_matrix = np.ones((n_atom, n_atom), dtype=float)

    A = np.transpose(np.array(lattice, dtype=float))  # columns are lattice vectors
    B = LA.inv(A)

    for i in range(n_atom):
        for j in range(i):
            r_ij = np.dot(B, xyz[i][0] - xyz[j][0])
            sin_sq_r = (np.sin(np.pi * r_ij)) ** 2
            distance = LA.norm(np.dot(A, sin_sq_r))
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    labels = np.transpose(np.array(xyz, dtype=object))[1].reshape(-1, 1)
    for at, charge in zip(["O", "Al", "Ga", "In"], [8, 13, 31, 49]):
        labels = np.where(labels == at, float(charge), labels)

    labels = labels.astype(float)
    charge_matrix = np.dot(labels, labels.T).astype(float)
    charge_matrix -= np.diag(np.diag(charge_matrix))
    charge_matrix += np.diag((0.5 * (labels.reshape(-1) ** 2.4))).astype(float)

    sine_matrix = charge_matrix / distance_matrix
    return sine_matrix




## === cell 6
def get_eigenspectrum(matrix):
    spectrum = LA.eigvalsh(matrix)
    spectrum = np.sort(spectrum)[::-1]
    return spectrum




## === cell 7
N_SPEC = 80  # keep same dimensionality as later PCA(n_components=80)

spectrum_list = []
missing_files = 0

for index in range(df_len):
    dataset_label = df.dataset.values[index]
    row_id = int(df.id.values[index])
    filename = os.path.join(BASE_INPUT, dataset_label, str(row_id), "geometry.xyz")

    if not os.path.exists(filename):
        missing_files += 1
        spectrum = np.zeros(N_SPEC, dtype=float)
    else:
        xyz, lattice = get_xyz(filename)
        sine_matrix = get_sine_matrix(xyz, lattice)
        spectrum_raw = get_eigenspectrum(sine_matrix).astype(float)

        spectrum = np.zeros(N_SPEC, dtype=float)
        m = min(N_SPEC, len(spectrum_raw))
        spectrum[:m] = spectrum_raw[:m]

    spectrum_list.append(spectrum)

print("Built spectra:", len(spectrum_list), "missing geometry files:", missing_files)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/3249836180.py in <cell line: 0>()
     16     else:
     17         xyz, lattice = get_xyz(filename)
---> 18         sine_matrix = get_sine_matrix(xyz, lattice)
     19         spectrum_raw = get_eigenspectrum(sine_matrix).astype(float)
     20 

/tmp/ipykernel_11/1086461669.py in get_sine_matrix(xyz, lattice)
     15 
     16     # element symbol labels -> charges
---> 17     labels = np.transpose(np.array(xyz, dtype=object))[1].reshape(-1, 1)
     18     for at, charge in zip(["O", "Al", "Ga", "In"], [8, 13, 31, 49]):
     19         labels = np.where(labels == at, float(charge), labels)

IndexError: index 1 is out of bounds for axis 0 with size 0

## === cell 8
fig, axs = plt.subplots(1, 5, figsize=(15, 6))
for i in range(5):
    ax = axs[i]
    plot_data = spectrum_list[i]
    ax.plot(range(len(plot_data)), plot_data)
    ax.hlines(0, 0, N_SPEC, colors="r")
plt.tight_layout()
plt.show()



## === cell 9
spectrum_df = pd.DataFrame(spectrum_list).astype(float)
spectrum_df = spectrum_df.fillna(0.0)

ss = StandardScaler()
spectrum_std_df = pd.DataFrame(ss.fit_transform(spectrum_df.values))

pca = PCA(n_components=N_SPEC)
spectrum_pca_df = pd.DataFrame(pca.fit_transform(spectrum_std_df.values))

spectrum_pca_df.head()



## === cell 10
plt.scatter(x=spectrum_pca_df.loc[:100, 0], y=spectrum_pca_df.loc[:100, 1])
plt.show()



## === cell 11
df.number_of_total_atoms = df.number_of_total_atoms.astype(int)
df["group_natoms"] = (
    df.spacegroup.astype(str) + "_" + df.number_of_total_atoms.astype(str)
)



## === cell 12
try:
    sns.lmplot(
        x="lattice_vector_1_ang",
        y="bandgap_energy_ev",
        hue="group_natoms",
        data=df[df["dataset"] == "train"],
        fit_reg=False,
        height=5,
        aspect=1.2,
        legend=False,
    )
    plt.show()
except Exception as e:
    print("Skipping plot:", repr(e))



## === cell 13
df = df.join(pd.get_dummies(df.group_natoms))
df.drop(["group_natoms"], axis=1, inplace=True)



## === cell 14
df.columns



## === cell 15
df.drop(["dataset"], axis=1, inplace=True)
df.drop(["id", "spacegroup", "number_of_total_atoms"], axis=1, inplace=True)



## === cell 16
df.columns



## === cell 17
df_new = pd.concat([spectrum_pca_df, df.reset_index(drop=True)], axis=1)
df_new.head()



## === cell 18
drop_idx = [
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
drop_idx = [i for i in drop_idx if i in df_new.index]
df_new.drop(drop_idx, axis=0, inplace=True)
df_new.reset_index(drop=True, inplace=True)

print("After dropping rows:", df_new.shape)



## === cell 19
df_new.shape



## === cell 20
df_len = len(df_new)
train_len = df_len - test_len

X_all = df_new.drop(["formation_energy_ev_natom", "bandgap_energy_ev"], axis=1)
X_train = X_all.iloc[:train_len].values
X_test = X_all.iloc[train_len:].values

y_formation = df_new["formation_energy_ev_natom"].iloc[:train_len].values
y_bandgap = df_new["bandgap_energy_ev"].iloc[:train_len].values

print(
    "X_train:",
    X_train.shape,
    "X_test:",
    X_test.shape,
    "y:",
    y_formation.shape,
    y_bandgap.shape,
)



## === cell 21
xgb_formation = xgb.XGBRegressor(
    random_state=0,
)
parameters = {
    "max_depth": [2, 3, 4],
    "n_estimators": [100, 200, 300],
}

cv_formation = GridSearchCV(
    xgb_formation, param_grid=parameters, cv=4, verbose=1, n_jobs=-1
)
cv_formation.fit(X_train, y_formation)



## === cell 22
cv_formation.best_params_



## === cell 23
xgb_bandgap = xgb.XGBRegressor(
    random_state=0,
)
parameters = {
    "max_depth": [2, 3, 4],
    "n_estimators": [100, 200, 300],
}

cv_bandgap = GridSearchCV(
    xgb_bandgap, param_grid=parameters, cv=4, verbose=1, n_jobs=-1
)
cv_bandgap.fit(X_train, y_bandgap)



## === cell 24
cv_bandgap.best_params_




## === cell 25
def plot_features(estimator, features):
    importances = estimator.feature_importances_
    plt.figure(figsize=(10, 15))
    plt.barh(range(len(importances)), importances, align="center")
    plt.yticks(np.arange(len(features)), features)
    plt.show()




## === cell 26
features = df_new.drop(
    ["formation_energy_ev_natom", "bandgap_energy_ev"], axis=1
).columns
plot_features(cv_formation.best_estimator_, features)
plot_features(cv_bandgap.best_estimator_, features)



## === cell 27
formation_pred = cv_formation.predict(X_test)
bandgap_pred = cv_bandgap.predict(X_test)

formation_pred = np.clip(formation_pred, 0.0, None)
bandgap_pred = np.clip(bandgap_pred, 0.0, None)

print("pred shapes:", formation_pred.shape, bandgap_pred.shape)



## === cell 28
submission = pd.read_csv(sample_sub_csv)
submission["formation_energy_ev_natom"] = formation_pred
submission["bandgap_energy_ev"] = bandgap_pred
submission.shape



## === cell 29
submission.head()



## === cell 30
submission.to_csv("submission.csv", index=False)
print(
    "Wrote submission.csv with columns:",
    list(submission.columns),
    "and shape:",
    submission.shape,
)
