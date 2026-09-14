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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0
Theano==1.0.5
Theano-PyMC==1.1.2

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

0.10598

# 6. Current score

0.58245

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 0.58245) has done: 'I remove the failing `theano` import (it breaks with NumPy 1.26) and ensure Keras uses the installed `tf_keras` backend so the same Sequential/Dense model can train. I fix the scaler usage so the MinMaxScaler is fit once on `X_train` and then applied to both `X_test` and final `test.csv` (the current code incorrectly refits on test, causing distribution mismatch and worse score). I also disable the heavy plotting cells (pairplot/heatmap) to avoid timeouts while keeping the training/model logic intact. Finally, I make the submission file with the exact required columns and ensure predictions are clipped to be non-negative (required for RMSLE/MSLE stability) and written to a `.csv` file.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import sklearn as sk
from sklearn.model_selection import train_test_split

TRAIN_PATH_CANDIDATES = [
    "../input/train.csv",
    "/kaggle/input/train.csv",
    "/kaggle/input/nomad2018-predict-transparent-conductors/train.csv",
]
for _p in TRAIN_PATH_CANDIDATES:
    if os.path.exists(_p):
        TRAIN_PATH = _p
        break
else:
    raise FileNotFoundError("Could not find train.csv in expected locations.")

data = pd.read_csv(TRAIN_PATH)

print(data.shape)
print(data[:1])
print(data.describe())



## === cell 1
Double = data.loc[
    data.duplicated(
        keep=False,
        subset=[
            "spacegroup",
            "number_of_total_atoms",
            "percent_atom_al",
            "percent_atom_ga",
            "percent_atom_in",
            "lattice_vector_1_ang",
            "lattice_vector_2_ang",
            "lattice_vector_3_ang",
            "lattice_angle_alpha_degree",
            "lattice_angle_beta_degree",
            "lattice_angle_gamma_degree",
        ],
    ),
    :,
]
print(Double.shape)
print(
    Double.sort_values(
        [
            "spacegroup",
            "number_of_total_atoms",
            "percent_atom_al",
            "percent_atom_ga",
            "percent_atom_in",
            "lattice_vector_1_ang",
            "lattice_vector_2_ang",
            "lattice_vector_3_ang",
            "lattice_angle_alpha_degree",
            "lattice_angle_beta_degree",
            "lattice_angle_gamma_degree",
        ]
    )
)
Double = data.loc[
    data.duplicated(
        keep="last",
        subset=[
            "spacegroup",
            "number_of_total_atoms",
            "percent_atom_al",
            "percent_atom_ga",
            "percent_atom_in",
            "lattice_vector_1_ang",
            "lattice_vector_2_ang",
            "lattice_vector_3_ang",
            "lattice_angle_alpha_degree",
            "lattice_angle_beta_degree",
            "lattice_angle_gamma_degree",
        ],
    ),
    :,
]
print(Double.shape)
data.drop_duplicates(
    inplace=True,
    keep="last",
    subset=[
        "spacegroup",
        "number_of_total_atoms",
        "percent_atom_al",
        "percent_atom_ga",
        "percent_atom_in",
        "lattice_vector_1_ang",
        "lattice_vector_2_ang",
        "lattice_vector_3_ang",
        "lattice_angle_alpha_degree",
        "lattice_angle_beta_degree",
        "lattice_angle_gamma_degree",
    ],
)
print(data.shape)



## === cell 2
dataX = data.copy().iloc[:, 1:12]  # without id and last two targets
dataY = data.copy().iloc[:, 12:]  # targets
print(dataX.shape, dataY.shape)



## === cell 3
PLOT = False
if PLOT:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="whitegrid", context="notebook")
    sns.pairplot(data.iloc[:, :], height=2.0)



## === cell 4
if PLOT:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.distplot(data.iloc[:, 12:13])
    plt.show()
    sns.distplot(data.iloc[:, 13:])
    plt.show()



## === cell 5
if PLOT:
    import seaborn as sns

    corr = np.corrcoef(data.values.T)
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.set()
    hm = sns.heatmap(
        corr,
        mask=mask,
        cbar=True,
        annot=True,
        square=True,
        fmt=".1f",
        annot_kws={"size": 7},
    )



## === cell 6
print(dataX[:1])



## === cell 7
X_train, X_test, y_train, y_test = train_test_split(
    dataX, dataY, test_size=0.25, shuffle=False
)

print("Входов: " + str(X_train.shape[1]))
print("Выходов: " + str(y_train.shape[1]))
print(y_test.describe())



## === cell 8
from sklearn.preprocessing import MinMaxScaler

sk_tr = MinMaxScaler(feature_range=(0.001, 0.999))
sk_tr.fit(X_train)
X_train_std = sk_tr.transform(X_train)
X_test_std = sk_tr.transform(X_test)

y_train_std = y_train.copy()
y_test_std = y_test.copy()
y_train_std = y_train_std.drop(["bandgap_energy_ev"], axis=1)
y_test_std = y_test_std.drop(["bandgap_energy_ev"], axis=1)

y_train2_std = y_train.copy()
y_test2_std = y_test.copy()
y_train2_std = y_train2_std.drop(["formation_energy_ev_natom"], axis=1)
y_test2_std = y_test2_std.drop(["formation_energy_ev_natom"], axis=1)

print("Преобразованный вЫход:")
print(y_test_std[:1])
print(pd.DataFrame(y_test_std).describe())



## === cell 9
import tf_keras as keras
from tf_keras.models import Sequential
from tf_keras.layers import Dense, Dropout, Activation

print("Входов: " + str(X_train.shape[1]))
print("Выходов: " + str(y_train_std.shape[1]))

model = Sequential()
k_init = "random_uniform"
actForm = "tanh"
НейроновВнутри = 50
model.add(
    Dense(
        units=НейроновВнутри,
        input_shape=(X_train.shape[1],),
        kernel_initializer=k_init,
        activation=actForm,
    )
)
model.add(Dense(units=НейроновВнутри, kernel_initializer=k_init, activation=actForm))
model.add(
    Dense(units=y_train_std.shape[1], kernel_initializer=k_init, activation="relu")
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 10
import sklearn.metrics as metrics

model.compile(
    loss="mean_squared_logarithmic_error",
    optimizer="rmsprop",
    metrics=["mean_squared_logarithmic_error"],
)

history1 = model.fit(
    X_train_std,
    y_train_std,
    epochs=2000,
    batch_size=30,
    verbose=0,
    validation_split=0.1,
)

y_test_predict = model.predict(X_test_std, verbose=0)

y_test_predict = np.clip(y_test_predict, 0.0, None)

print(
    "mean_squared_log_error: "
    + str(metrics.mean_squared_log_error(y_test_std, y_test_predict))
)
print("r2_score 1 : " + str(metrics.r2_score(y_test_std, y_test_predict)))

score = model.evaluate(X_test_std, y_test_std, batch_size=10, verbose=0)

print("1 Выход истинный:")
print(y_test_std[:10])
print("1 Выход расчитанный:")
print(y_test_predict[:10])
err_per = (y_test_predict[:10]) * 100 / y_test_std[:10]
print("Ошибка в %:")
print(err_per)
print("Точность работы на тестовых данных: %.5f%%" % ((1 - score[0]) * 100))
print("Test score:", score)



## === cell 11
if PLOT:
    import matplotlib.pyplot as plt

    print(history1.history.keys())
    plt.plot(history1.history["mean_squared_logarithmic_error"])
    plt.plot(history1.history["val_mean_squared_logarithmic_error"])
    plt.title("model MSLE")
    plt.ylabel("MSLE")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")
    plt.show()
    plt.plot(history1.history["loss"])
    plt.plot(history1.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")
    plt.show()



## === cell 12
TEST_PATH_CANDIDATES = [
    "../input/test.csv",
    "/kaggle/input/test.csv",
    "/kaggle/input/nomad2018-predict-transparent-conductors/test.csv",
]
for _p in TEST_PATH_CANDIDATES:
    if os.path.exists(_p):
        TEST_PATH = _p
        break
else:
    raise FileNotFoundError("Could not find test.csv in expected locations.")

X_Fin_test = pd.read_csv(TEST_PATH)
ids = X_Fin_test["id"].copy()
X_Fin_test = X_Fin_test.drop(["id"], axis=1)

X_Fin_test_std = sk_tr.transform(X_Fin_test)

print(X_test[:1])
print(X_Fin_test[:1])

y_Fin_test_predict = model.predict(X_Fin_test_std, verbose=0)
y_Fin_test_predict = np.clip(y_Fin_test_predict, 0.0, None)



## === cell 13
model2 = Sequential()
k_init = "random_uniform"
actForm = "tanh"
НейроновВнутри = 50
model2.add(
    Dense(
        units=НейроновВнутри,
        input_shape=(X_train.shape[1],),
        kernel_initializer=k_init,
        activation=actForm,
    )
)
model2.add(Dense(units=НейроновВнутри, kernel_initializer=k_init, activation=actForm))
model2.add(
    Dense(units=y_train2_std.shape[1], kernel_initializer=k_init, activation="relu")
)

model2.compile(
    loss="mean_squared_logarithmic_error",
    optimizer="rmsprop",
    metrics=["mean_squared_logarithmic_error"],
)

history2 = model2.fit(
    X_train_std,
    y_train2_std,
    epochs=800,
    batch_size=30,
    verbose=0,
    validation_split=0.1,
)

y_test2_predict = model2.predict(X_test_std, verbose=0)
y_test2_predict = np.clip(y_test2_predict, 0.0, None)

print(
    "mean_squared_log_error: "
    + str(metrics.mean_squared_log_error(y_test2_std, y_test2_predict))
)
print("r2_score 2 : " + str(metrics.r2_score(y_test2_std, y_test2_predict)))
score2 = model2.evaluate(X_test_std, y_test2_std, batch_size=10, verbose=0)

print("2 Выход истинный:")
print(y_test2_std[:10])
print("2 Выход расчитанный:")
print(y_test2_predict[:10])
err_per2 = (y_test2_predict[:10]) * 100 / y_test2_std[:10]
print("Ошибка в %:")
print(err_per2)
print("Точность работы на тестовых данных: %.5f%%" % ((1 - score2[0]) * 100))
print("Test score 2:", score2)

y_Fin_test_predict2 = model2.predict(X_Fin_test_std, verbose=0)
y_Fin_test_predict2 = np.clip(y_Fin_test_predict2, 0.0, None)



## === cell 14
if PLOT:
    import matplotlib.pyplot as plt

    print(history2.history.keys())
    plt.plot(history2.history["mean_squared_logarithmic_error"])
    plt.plot(history2.history["val_mean_squared_logarithmic_error"])
    plt.title("model MSLE")
    plt.ylabel("MSLE")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")
    plt.show()
    plt.plot(history2.history["loss"])
    plt.plot(history2.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")
    plt.show()



## === cell 15
outDF = pd.DataFrame()
outDF["id"] = ids.values
outDF["formation_energy_ev_natom"] = np.asarray(y_Fin_test_predict).reshape(-1)
outDF["bandgap_energy_ev"] = np.asarray(y_Fin_test_predict2).reshape(-1)

print(outDF[:10])

out_path = "prediction.csv"
outDF.to_csv(out_path, index=False)
print("Wrote submission to:", out_path, "shape:", outDF.shape)
