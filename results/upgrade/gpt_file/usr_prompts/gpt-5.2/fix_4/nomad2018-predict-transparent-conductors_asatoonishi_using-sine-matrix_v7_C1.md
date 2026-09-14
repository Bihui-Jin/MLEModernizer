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

0.05733

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.2741) has done: 'Main runtime bottlenecks are (1) repeatedly parsing ~2400 geometry files with slow Python loops and `readlines()`, and (2) two expensive `GridSearchCV` runs (36 XGBoost fits) with nested parallelism overhead. I keep the exact same features/model logic, but speed up geometry processing by using faster file iteration, vectorized pairwise computations for the sine matrix, and caching computed eigenspectra to disk so re-runs don’t redo the heavy work. For model selection, I preserve the same parameter grid/CV splits/estimators, but remove nested parallelism by letting XGBoost use all threads while GridSearch runs sequentially (this is equivalent and typically much faster/less overhead). I also avoid unnecessary pandas DataFrame conversions and use contiguous NumPy arrays where possible.'
- What this solution (achieved 0.05733) has done: 'Your score is far worse than the target (0.2741 vs 0.06818; lower is better), so we should make small, legitimate changes that better match the RMSLE metric without changing the modeling approach. The biggest issue is that you train XGBoost on raw targets but RMSLE evaluates in log-space; keeping the same XGBoost regressors and CV, we can simply train on `log1p(y)` and then `expm1` predictions back, which typically reduces RMSLE a lot. I also fix an indexing bug introduced by dropping rows: currently `X_test` no longer aligns to `df_test["id"]`, which can severely hurt score; we preserve train/test row indices, drop outliers only from train, and build the submission by joining predictions to the correct test ids. Everything else (features, geometry processing, GridSearchCV, estimators) stays the same.'

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

np.random.seed(42)



## === cell 1
BASE_DIR_CANDIDATES = [
    "/kaggle/input/nomad2018-predict-transparent-conductors",
    "/kaggle/data/nomad2018-predict-transparent-conductors",
    "/kaggle/input",
    "/kaggle/data",
    "../input",  # keep original intent
]


def find_first_existing_path(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


BASE_DIR = find_first_existing_path(BASE_DIR_CANDIDATES)
if BASE_DIR is None:
    raise FileNotFoundError(
        "Could not locate Kaggle input directory. Checked: {}".format(
            BASE_DIR_CANDIDATES
        )
    )

TRAIN_CSV = None
TEST_CSV = None
SAMPLE_SUB = None

for root in [
    BASE_DIR,
    os.path.join(BASE_DIR, "nomad2018-predict-transparent-conductors"),
]:
    if root and os.path.exists(os.path.join(root, "train.csv")):
        TRAIN_CSV = os.path.join(root, "train.csv")
        TEST_CSV = os.path.join(root, "test.csv")
        SAMPLE_SUB = os.path.join(root, "sample_submission.csv")
        GEOM_ROOT_TRAIN = os.path.join(root, "train")
        GEOM_ROOT_TEST = os.path.join(root, "test")
        break

if TRAIN_CSV is None:
    raise FileNotFoundError(
        f"train.csv not found under {BASE_DIR} (or its nested competition folder)."
    )

df_train = pd.read_csv(TRAIN_CSV)
df_train["dataset"] = "train"
df_test = pd.read_csv(TEST_CSV)
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
    coords = []
    labels = []
    lattice = []
    with open(filename, "r") as f:
        for line in f:
            row = line.split()
            if not row:
                continue
            key = row[0]
            if key == "atom":
                coords.append((float(row[1]), float(row[2]), float(row[3])))
                labels.append(row[4])
            elif key == "lattice_vector":
                lattice.append((float(row[1]), float(row[2]), float(row[3])))
    return (
        np.asarray(coords, dtype=np.float64),
        np.asarray(labels, dtype=object),
    ), np.asarray(lattice, dtype=np.float64)




## === cell 5
def get_sine_matrix(xyz, lattice):
    coords, labels = xyz
    n_atom = coords.shape[0]

    A = lattice.T.astype(np.float64, copy=False)  # shape (3,3)
    B = LA.inv(A)

    d = coords[:, None, :] - coords[None, :, :]
    r = d @ B.T
    sin_sq_r = np.sin(np.pi * r) ** 2
    v = sin_sq_r @ A.T
    distance_matrix = LA.norm(v, axis=2).astype(np.float64, copy=False)

    np.fill_diagonal(distance_matrix, 1.0)

    charge_map = {"O": 8.0, "Al": 13.0, "Ga": 31.0, "In": 49.0}
    z = np.fromiter(
        (charge_map.get(s, 0.0) for s in labels), dtype=np.float64, count=n_atom
    ).reshape(-1, 1)

    charge_matrix = (z @ z.T).astype(np.float64, copy=False)
    np.fill_diagonal(charge_matrix, 0.0)
    diag = (0.5 * (z[:, 0] ** 2.4)).astype(np.float64, copy=False)
    np.fill_diagonal(charge_matrix, diag)

    sine_matrix = charge_matrix / distance_matrix
    return sine_matrix




## === cell 6
def get_eigenspectrum(matrix):
    spectrum = LA.eigvalsh(matrix)
    spectrum = np.sort(spectrum)[::-1]
    return spectrum




## === cell 7
def geometry_path(dataset_label, row_id):
    row_id_str = str(int(row_id))
    if dataset_label == "train":
        p = os.path.join(GEOM_ROOT_TRAIN, row_id_str, "geometry.xyz")
    else:
        p = os.path.join(GEOM_ROOT_TEST, row_id_str, "geometry.xyz")
    return p


CACHE_PATH = "spectrum_cache_v1.npz"

cache = {}
if os.path.exists(CACHE_PATH):
    try:
        _npz = np.load(CACHE_PATH, allow_pickle=True)
        cache = _npz["cache"].item()
    except Exception:
        cache = {}

spectrum_list = []
failed_ids = []

for index in range(df_len):
    dataset_label = df.loc[index, "dataset"]
    row_id = df.loc[index, "id"]
    filename = geometry_path(dataset_label, row_id)

    try:
        st = os.stat(filename)
        key = (filename, int(st.st_mtime), int(st.st_size))
        spectrum = cache.get(key, None)
        if spectrum is None:
            xyz, lattice = get_xyz(filename)
            sine_matrix = get_sine_matrix(xyz, lattice)
            spectrum = get_eigenspectrum(sine_matrix)
            cache[key] = spectrum
    except Exception:
        failed_ids.append((dataset_label, row_id))
        spectrum = np.array([0.0], dtype=np.float64)

    spectrum_list.append(spectrum)

if len(failed_ids) > 0:
    print(
        f"Warning: failed to parse {len(failed_ids)} geometry files; used zero spectrum for those rows."
    )

try:
    np.savez_compressed(CACHE_PATH, cache=cache)
except Exception:
    pass



## === cell 8
if len(spectrum_list) >= 5:
    fig, axs = plt.subplots(1, 5, figsize=(15, 6))
    for i in range(5):
        ax = axs[i]
        plot_data = spectrum_list[i]
        ax.plot(range(len(plot_data)), plot_data)
        ax.hlines(0, 0, max(80, len(plot_data)), colors="r")
    plt.show()



## === cell 9
spectrum_df = pd.DataFrame(spectrum_list, copy=False).astype(np.float64, copy=False)
spectrum_df = spectrum_df.fillna(0.0)



## === cell 10
ss = StandardScaler()
spectrum_std = ss.fit_transform(spectrum_df.to_numpy(dtype=np.float64, copy=False))

pca = PCA(n_components=80)
spectrum_pca = pca.fit_transform(spectrum_std)
spectrum_pca_df = pd.DataFrame(spectrum_pca)
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
try:
    sns.lmplot(
        x="lattice_vector_1_ang",
        y="bandgap_energy_ev",
        hue="group_natoms",
        data=df,
        fit_reg=False,
    )
    plt.show()
except Exception:
    pass



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
df_new = pd.concat([spectrum_df, df], axis=1)
df_new.head()



## === cell 19
train_mask = np.zeros(len(df_new), dtype=bool)
train_mask[: len(df_train)] = True
test_mask = ~train_mask

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
drop_idx = [i for i in drop_idx if 0 <= i < len(df_new) and train_mask[i]]

df_train_new = df_new.loc[train_mask].drop(index=drop_idx, errors="ignore")
df_test_new = df_new.loc[test_mask].copy()



## === cell 20
df_train_new.shape, df_test_new.shape



## === cell 21
X_train = np.ascontiguousarray(
    df_train_new.drop(
        ["formation_energy_ev_natom", "bandgap_energy_ev"], axis=1
    ).to_numpy(dtype=np.float64, copy=False)
)
X_test = np.ascontiguousarray(
    df_test_new.drop(
        ["formation_energy_ev_natom", "bandgap_energy_ev"], axis=1
    ).to_numpy(dtype=np.float64, copy=False)
)

y_formation = df_train_new["formation_energy_ev_natom"].to_numpy(
    dtype=np.float64, copy=False
)
y_bandgap = df_train_new["bandgap_energy_ev"].to_numpy(dtype=np.float64, copy=False)



## === cell 22
y_formation_log = np.log1p(np.clip(y_formation, 0.0, None))
y_bandgap_log = np.log1p(np.clip(y_bandgap, 0.0, None))

xgb_formation = xgb.XGBRegressor(
    random_state=42,
    n_jobs=max(1, (os.cpu_count() or 4)),
)
parameters = {
    "max_depth": [2, 3, 4],
    "n_estimators": [100, 200, 300],
}

cv_formation = GridSearchCV(
    xgb_formation, param_grid=parameters, cv=4, verbose=1, n_jobs=1
)
cv_formation.fit(X_train, y_formation_log)



## === cell 23
cv_formation.best_params_



## === cell 24
xgb_bandgap = xgb.XGBRegressor(
    random_state=42,
    n_jobs=max(1, (os.cpu_count() or 4)),
)
parameters = {
    "max_depth": [2, 3, 4],
    "n_estimators": [100, 200, 300],
}

cv_bandgap = GridSearchCV(xgb_bandgap, param_grid=parameters, cv=4, verbose=1, n_jobs=1)
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
features = df_train_new.drop(
    ["formation_energy_ev_natom", "bandgap_energy_ev"], axis=1
).columns
plot_features(cv_formation.best_estimator_, features)
plot_features(cv_bandgap.best_estimator_, features)



## === cell 28
formation_pred = np.expm1(cv_formation.predict(X_test))
bandgap_pred = np.expm1(cv_bandgap.predict(X_test))

formation_pred = np.clip(formation_pred.astype(np.float64, copy=False), 0.0, None)
bandgap_pred = np.clip(bandgap_pred.astype(np.float64, copy=False), 0.0, None)



## === cell 29
if len(df_test) != len(formation_pred) or len(df_test) != len(bandgap_pred):
    raise ValueError(
        f"Prediction length mismatch: df_test={len(df_test)} formation_pred={len(formation_pred)} bandgap_pred={len(bandgap_pred)}"
    )

submission = pd.DataFrame({"id": df_test["id"].values})
submission["formation_energy_ev_natom"] = formation_pred
submission["bandgap_energy_ev"] = bandgap_pred

expected_cols = ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
if list(submission.columns) != expected_cols:
    raise ValueError(
        f"Submission columns mismatch. Got {list(submission.columns)}, expected {expected_cols}."
    )

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
