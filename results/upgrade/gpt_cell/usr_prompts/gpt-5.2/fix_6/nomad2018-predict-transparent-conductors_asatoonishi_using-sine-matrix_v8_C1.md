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

0.06668

# 6. Current score

0.27513

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.27547) has done: 'Diagnosis: The crash happens inside `get_sine_matrix()` when doing `np.transpose(xyz)[1]`. Here `xyz` is a Python list of tuples `(np.array([x,y,z]), symbol)`, which is ragged (mixes arrays and strings) and cannot be converted into a homogeneous NumPy array for transpose under newer NumPy, causing “setting an array element with a sequence”.  
Patch summary: In cell 7, without changing earlier function definitions, I monkey-patch `get_sine_matrix` locally to extract labels via a plain Python list comprehension (`[t[1] for t in xyz]`) and keep the rest of the sine-matrix computation identical. This keeps the feature extraction logic the same while avoiding ragged-array transpose.  
Updated cells: Only cell 7 is modified.  
Compatibility notes for cell k+1: `spectrum_list` remains a list of 1D NumPy arrays (eigenspectra) exactly as before, so plotting in cell 8 works unchanged.  
Assumptions: All structures contain only elements in `['O','Al','Ga','In']` as implied by the original mapping; paths under `../input/{train|test}/{id}/geometry.xyz` exist in this environment as in the original notebook.'
- What this solution (achieved 0.27402) has done: 'Your current approach trains XGBoost on raw targets even though the leaderboard metric is RMSLE, so the biggest safe gain (without changing the model type/loop) is to train on `log1p(target)` and then `expm1` at prediction time, plus clip predictions to be non-negative (required for RMSLE stability). I keep the same feature pipeline (sine-matrix → eigenspectrum → standardize → PCA + tabular dummies) and the same GridSearchCV training approach, only adding the target transform wrapper and ensuring the submission `id` matches `test.csv` ordering (instead of assuming 1..240). These changes should move the score substantially downward from 0.275 toward your 0.06668 target while preserving evaluation semantics for RMSLE. I also make the `np.float` compatibility patch apply before any parsing so it’s consistent across cells.'
- What this solution (achieved 0.27383) has done: 'I make two minimal, score-relevant changes to better match the RMSLE metric without changing your feature pipeline or model family. First, I set XGBoost’s objective to squared error and explicitly use an RMSLE-compatible scoring function inside GridSearchCV on the log1p-transformed targets, so hyperparameters are selected to reduce the actual competition metric rather than default R². Second, I add a small epsilon before log1p to avoid any edge-case `log(0)` behavior and keep the existing non-negativity clipping after `expm1`, improving numerical stability. Everything else (sine-matrix → eigenspectrum → scaling → PCA → dummies; separate models; GridSearchCV loop; submission format) stays the same.'
- What this solution (achieved 0.27513) has done: 'Your score is far worse than the target (0.27383 vs 0.06668; lower is better), so we should make small, metric-aligned improvements without changing the overall pipeline. The biggest minimal win is to make the CV split deterministic and less noisy for this dataset by using a shuffled KFold with a fixed seed (instead of plain `cv=4`), so GridSearch picks hyperparameters that generalize better and more consistently. I also add `eval_metric='rmse'` to XGBoost (training on log1p targets still matches RMSLE intent) and set a couple of safe, standard regularization defaults (`subsample`, `colsample_bytree`) to reduce overfitting without changing the model family or training approach. All feature extraction, PCA, and the log1p/expm1 target handling remain the same, and the script still writes a valid `submission.csv`.'

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
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_log_error, make_scorer

if not hasattr(np, "float"):
    np.float = float


def rmsle_from_log_targets(y_true_log, y_pred_log):
    y_true = np.expm1(np.asarray(y_true_log))
    y_pred = np.expm1(np.asarray(y_pred_log))
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


rmsle_scorer = make_scorer(rmsle_from_log_targets, greater_is_better=False)



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
                xyz.append((np.array(row[1:4], dtype=np.float), row[4]))
            elif row[0] == "lattice_vector":
                lattice.append(np.array(row[1:4], dtype=np.float))

    return xyz, lattice




## === cell 5
def get_sine_matrix(xyz, lattice):

    n_atom = len(xyz)  # number of atoms in the cell

    distance_matrix = np.ones((n_atom, n_atom))
    A = np.transpose(lattice)  # A = (a_1, a_2, a_3), defined as above
    B = LA.inv(A)  # inverse matrix of A

    for i in range(n_atom):
        for j in range(i):
            r_ij = np.dot(B, xyz[i][0] - xyz[j][0])
            sin_sq_r = (np.sin(np.pi * r_ij)) ** 2
            distance = LA.norm(np.dot(A, sin_sq_r))
            distance_matrix[i, j], distance_matrix[j, i] = distance, distance

    labels = np.transpose(xyz)[1]  # element symbol labels
    labels = labels.reshape(-1, 1)
    for at, charge in zip(
        ["O", "Al", "Ga", "In"], [8, 13, 31, 49]
    ):  # convert symbols into electric charges
        labels = np.where(labels == at, charge, labels)
    charge_matrix = np.dot(labels, np.transpose(labels)).astype(np.float)
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
def get_sine_matrix(xyz, lattice):
    n_atom = len(xyz)  # number of atoms in the cell

    distance_matrix = np.ones((n_atom, n_atom))
    A = np.transpose(lattice)  # A = (a_1, a_2, a_3)
    B = LA.inv(A)

    for i in range(n_atom):
        for j in range(i):
            r_ij = np.dot(B, xyz[i][0] - xyz[j][0])
            sin_sq_r = (np.sin(np.pi * r_ij)) ** 2
            distance = LA.norm(np.dot(A, sin_sq_r))
            distance_matrix[i, j], distance_matrix[j, i] = distance, distance

    labels = np.array([t[1] for t in xyz], dtype=object).reshape(-1, 1)
    for at, charge in zip(["O", "Al", "Ga", "In"], [8, 13, 31, 49]):
        labels = np.where(labels == at, charge, labels)

    charge_matrix = np.dot(labels, np.transpose(labels)).astype(np.float)
    charge_matrix -= np.diag(np.diag(charge_matrix))
    charge_matrix += np.diag(0.5 * labels**2.4).astype(float)

    sine_matrix = charge_matrix / distance_matrix
    return sine_matrix


spectrum_list = []

for index in range(df_len):
    dataset_label = df.dataset.values[index]
    row_id = df.id.values[index]
    filename = "../input/{}/{}/geometry.xyz".format(dataset_label, row_id)

    xyz, lattice = get_xyz(filename)
    sine_matrix = get_sine_matrix(xyz, lattice)
    spectrum = get_eigenspectrum(sine_matrix)

    spectrum_list.append(spectrum)



## === cell 8
fig, axs = plt.subplots(1, 5, figsize=(15, 6))
for i in range(5):
    ax = axs[i]
    plot_data = spectrum_list[i]
    ax.plot(range(len(plot_data)), plot_data)
    ax.hlines(0, 0, 80, colors="r")
plt.show()



## === cell 9
spectrum_df = pd.DataFrame(spectrum_list).astype(np.float)
spectrum_df = spectrum_df.fillna(0)



## === cell 10
ss = StandardScaler()
spectrum_std_df = pd.DataFrame(ss.fit_transform(spectrum_df.values))
pca = PCA(n_components=80)
spectrum_pca_df = pd.DataFrame(pca.fit_transform(spectrum_std_df.values))

spectrum_pca_df.head()



## === cell 11
plt.scatter(x=spectrum_pca_df.loc[:100, 0], y=spectrum_pca_df.loc[:100, 1])
plt.show()



## === cell 12
df.number_of_total_atoms = df.number_of_total_atoms.astype("int")
df["group_natoms"] = (
    df.spacegroup.astype("str") + "_" + df.number_of_total_atoms.astype("str")
)



## === cell 13
sns.lmplot(
    x="lattice_vector_1_ang",
    y="bandgap_energy_ev",
    hue="group_natoms",
    data=df,
    fit_reg=False,
)
plt.show()



## === cell 14
df = df.join(pd.get_dummies(df.group_natoms))
df.drop(["group_natoms"], axis=1, inplace=True)



## === cell 15
df.columns



## === cell 16
df.drop(["dataset"], axis=1, inplace=True)
df.drop(["id", "spacegroup", "number_of_total_atoms"], axis=1, inplace=True)



## === cell 17
df.columns



## === cell 18
df_new = pd.concat([spectrum_pca_df, df], axis=1)
df_new.head()



## === cell 19
df_new.drop(
    [395, 126, 1215, 1886, 2075, 353, 308, 2154, 531, 1379, 2319, 2337, 2370, 2333],
    axis=0,
    inplace=True,
)



## === cell 20
df_new.shape



## === cell 21
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



## === cell 22
eps = 1e-9
y_formation_log = np.log1p(np.maximum(y_formation, 0) + eps)
y_bandgap_log = np.log1p(np.maximum(y_bandgap, 0) + eps)

cv_splitter = KFold(n_splits=4, shuffle=True, random_state=0)

xgb_formation = xgb.XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=0,
    n_jobs=-1,
    subsample=0.9,
    colsample_bytree=0.9,
)
parameters = {
    "max_depth": [2, 3, 4],
    "n_estimators": [100, 200, 300],
}

cv_formation = GridSearchCV(
    xgb_formation,
    param_grid=parameters,
    cv=cv_splitter,
    scoring=rmsle_scorer,
    verbose=1,
)
cv_formation.fit(X_train, y_formation_log)



## === cell 23
cv_formation.best_params_



## === cell 24
xgb_bandgap = xgb.XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=0,
    n_jobs=-1,
    subsample=0.9,
    colsample_bytree=0.9,
)
parameters = {
    "max_depth": [2, 3, 4],
    "n_estimators": [100, 200, 300],
}

cv_bandgap = GridSearchCV(
    xgb_bandgap,
    param_grid=parameters,
    cv=cv_splitter,
    scoring=rmsle_scorer,
    verbose=1,
)
cv_bandgap.fit(X_train, y_bandgap_log)



## === cell 25
cv_bandgap.best_params_




## === cell 26
def plot_features(estimator, features):
    importances = estimator.feature_importances_
    plt.figure(figsize=(10, 15))
    plt.barh(range(len(importances)), importances, align="center")
    plt.yticks(np.arange(len(features)), features)
    plt.show()




## === cell 27
features = df_new.drop(
    ["formation_energy_ev_natom", "bandgap_energy_ev"], axis=1
).columns

plot_features(cv_formation.best_estimator_, features)
plot_features(cv_bandgap.best_estimator_, features)



## === cell 28
formation_pred = np.expm1(cv_formation.predict(X_test))
bandgap_pred = np.expm1(cv_bandgap.predict(X_test))
formation_pred = np.maximum(formation_pred, 0)
bandgap_pred = np.maximum(bandgap_pred, 0)



## === cell 29
submission = pd.DataFrame({"id": df_test["id"].values})
submission["formation_energy_ev_natom"] = formation_pred
submission["bandgap_energy_ev"] = bandgap_pred
submission.shape



## === cell 30
submission.head()



## === cell 31
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
