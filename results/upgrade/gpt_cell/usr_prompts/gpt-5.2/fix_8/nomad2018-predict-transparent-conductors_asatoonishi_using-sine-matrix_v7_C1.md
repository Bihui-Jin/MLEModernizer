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

0.27077

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.26859) has done: 'Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.
The crash in cell 9 is caused by using `np.float`, which was removed in NumPy 1.24+; with NumPy 1.26 this raises `AttributeError`. The minimal fix is to replace `np.float` with the builtin `float` (or `np.float64`), which preserves the intended dtype conversion. This keeps `spectrum_df` as a numeric DataFrame so cell 10’s `StandardScaler().fit_transform(spectrum_df.values)` continues to work unchanged. No other logic or variables are modified. Assumption: `spectrum_list` is already populated in earlier cells and contains numeric spectra (with NaNs filled after conversion).'
- What this solution (achieved 0.27077) has done: 'Your current score is far above the target (0.2686 vs 0.0682; lower is better), so we need a meaningful but still minimal change that better matches the competition’s RMSLE metric. I keep your sine-matrix eigenspectrum + PCA + XGBoost core pipeline intact, but train/predict in log1p-space for both targets and then invert with expm1, plus clip predictions to be non-negative (RMSLE-safe). I also set XGBoost’s objective to squared error explicitly and fix random_state/n_jobs for stability/speed without changing the overall approach. These changes typically reduce RMSLE substantially because they align optimization and prediction constraints with the log-based evaluation.'
- What this solution (achieved 0.27077) has done: 'Your score is much worse than the target (0.27077 vs 0.06818; lower is better), so we need a modest but meaningful improvement without changing your overall pipeline. The largest, safe gain here is to correct data leakage in preprocessing: StandardScaler and PCA are currently fit on train+test together, which distorts the learned representation and hurts generalization on the leaderboard. I fit StandardScaler+PCA on the training rows only, then transform the test rows with the fitted transformers, keeping your sine-matrix spectrum + PCA + XGBoost + log1p target approach identical otherwise. I also make the submission `id` come from `test.csv` (instead of assuming 1..N) to guarantee perfect row alignment.'

# 9. Code solution

## === cell 0
import numpy as np
import numpy.linalg as LA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass



## === cell 1
df_train = pd.read_csv("../input/train.csv")
df_train["dataset"] = "train"
df_test = pd.read_csv("../input/test.csv")
df_test["dataset"] = "test"
test_len = len(df_test)
df = pd.concat([df_train, df_test], axis=0, ignore_index=True, sort=False)
df_len = len(df)



## === cell 2
df.head()



## === cell 3
df.tail()




## === cell 4
def get_xyz(filename):
    row = []
    xyz = []
    lattice = []

    with open(filename) as f:
        for line in f.readlines():
            row = line.split()
            if row[0] == "atom":
                xyz.append((np.array(row[1:4], dtype=float), row[4]))
            elif row[0] == "lattice_vector":
                lattice.append(np.array(row[1:4], dtype=float))

    return xyz, lattice




## === cell 5
def get_sine_matrix(xyz, lattice):
    n_atom = len(xyz)  # number of atoms in the cell

    distance_matrix = np.ones((n_atom, n_atom))
    A = np.transpose(lattice)  # A = (a_1, a_2, a_3)
    B = LA.inv(A)  # inverse matrix of A

    for i in range(n_atom):
        for j in range(i):
            r_ij = np.dot(B, xyz[i][0] - xyz[j][0])
            sin_sq_r = (np.sin(np.pi * r_ij)) ** 2
            distance = LA.norm(np.dot(A, sin_sq_r))
            distance_matrix[i, j], distance_matrix[j, i] = distance, distance

    labels = np.array([atom[1] for atom in xyz], dtype=object).reshape(-1, 1)
    for at, charge in zip(["O", "Al", "Ga", "In"], [8, 13, 31, 49]):
        labels = np.where(labels == at, charge, labels)

    charge_matrix = np.dot(labels, np.transpose(labels)).astype(float)
    charge_matrix -= np.diag(np.diag(charge_matrix))  # let diagonal components zero
    charge_matrix += np.diag(0.5 * labels**2.4).astype(float)  # from the definition

    sine_matrix = charge_matrix / distance_matrix
    return sine_matrix




## === cell 6
def get_eigenspectrum(matrix):
    spectrum = LA.eigvalsh(matrix)
    spectrum = np.sort(spectrum)[::-1]
    return spectrum




## === cell 7
if "spectrum_list" not in globals():
    spectrum_list = []
    for _, r in df.iterrows():
        geom_path = "../input/{}/{}/geometry.xyz".format(r["dataset"], int(r["id"]))
        xyz, lattice = get_xyz(geom_path)
        sm = get_sine_matrix(xyz, lattice)
        spectrum_list.append(get_eigenspectrum(sm))

fig, axs = plt.subplots(1, 5, figsize=(15, 6))
for i in range(5):
    ax = axs[i]
    plot_data = spectrum_list[i]
    ax.plot(range(len(plot_data)), plot_data)
    ax.hlines(0, 0, 80, colors="r")
plt.show()



## === cell 8
spectrum_df = pd.DataFrame(spectrum_list).astype(float)
spectrum_df = spectrum_df.fillna(0)



## === cell 9
train_len_full = df_len - test_len

ss = StandardScaler()
spectrum_std_train = ss.fit_transform(spectrum_df.iloc[:train_len_full].values)
spectrum_std_test = ss.transform(spectrum_df.iloc[train_len_full:].values)

pca = PCA(n_components=80)
spectrum_pca_train = pca.fit_transform(spectrum_std_train)
spectrum_pca_test = pca.transform(spectrum_std_test)

spectrum_pca_df = pd.DataFrame(
    np.vstack([spectrum_pca_train, spectrum_pca_test]),
    index=spectrum_df.index,
)

spectrum_pca_df.head()



## === cell 10
plt.scatter(x=spectrum_pca_df.loc[:100, 0], y=spectrum_pca_df.loc[:100, 1])
plt.show()



## === cell 11
df.number_of_total_atoms = df.number_of_total_atoms.astype("int")
df["group_natoms"] = (
    df.spacegroup.astype("str") + "_" + df.number_of_total_atoms.astype("str")
)



## === cell 12
sns.lmplot(
    x="lattice_vector_1_ang",
    y="bandgap_energy_ev",
    hue="group_natoms",
    data=df,
    fit_reg=False,
)
plt.show()



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
df_new = pd.concat([spectrum_df, df], axis=1)
df_new.head()



## === cell 18
df_new.drop(
    [395, 126, 1215, 1886, 2075, 353, 308, 2154, 531, 1379, 2319, 2337, 2370, 2333],
    axis=0,
    inplace=True,
)



## === cell 19
df_new.shape



## === cell 20
df_len = len(df_new)
train_len = df_len - test_len
X_train = df_new.drop(["formation_energy_ev_natom", "bandgap_energy_ev"], axis=1)[
    :train_len
].values
X_test = df_new.drop(["formation_energy_ev_natom", "bandgap_energy_ev"], axis=1)[
    train_len:
].values
y_formation = df_new["formation_energy_ev_natom"][:train_len].values
y_bandgap = df_new["bandgap_energy_ev"][:train_len].values

y_formation_log = np.log1p(np.maximum(y_formation, 0))
y_bandgap_log = np.log1p(np.maximum(y_bandgap, 0))



## === cell 21
xgb_formation = xgb.XGBRegressor(
    objective="reg:squarederror", random_state=42, n_jobs=-1
)
parameters = {
    "max_depth": [2, 3, 4],
    "n_estimators": [100, 200, 300],
}

cv_formation = GridSearchCV(
    xgb_formation, param_grid=parameters, cv=4, verbose=1, n_jobs=-1
)
cv_formation.fit(X_train, y_formation_log)



## === cell 22
cv_formation.best_params_



## === cell 23
xgb_bandgap = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)
parameters = {
    "max_depth": [2, 3, 4],
    "n_estimators": [100, 200, 300],
}

cv_bandgap = GridSearchCV(
    xgb_bandgap, param_grid=parameters, cv=4, verbose=1, n_jobs=-1
)
cv_bandgap.fit(X_train, y_bandgap_log)



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
formation_pred_log = cv_formation.predict(X_test)
bandgap_pred_log = cv_bandgap.predict(X_test)

formation_pred = np.expm1(formation_pred_log)
bandgap_pred = np.expm1(bandgap_pred_log)

formation_pred = np.maximum(formation_pred, 0)
bandgap_pred = np.maximum(bandgap_pred, 0)



## === cell 28
submission = pd.DataFrame({"id": df_test["id"].values})
submission["formation_energy_ev_natom"] = formation_pred
submission["bandgap_energy_ev"] = bandgap_pred
submission.shape



## === cell 29
submission.head()



## === cell 30
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
