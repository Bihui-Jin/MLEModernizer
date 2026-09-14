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

No external packages required in the script and installed.

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

0.07317

# 6. Current score

0.21205

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 0.21205) has done: 'I replace the broken TensorFlow v1 code with a simple TensorFlow Keras model that uses the same architecture (8 dense layers of 16 tanh units with L2 regularization) but works with the current TensorFlow version. I also fix the undefined variables, remove the failing batch‑normalization call, and streamline the training‑ and prediction‑steps so that a valid `subm.csv` is written. No other logic is changed.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



## === cell 1
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")



## === cell 2
train.describe()



## === cell 3
train.head()



## === cell 4
np.all(
    np.abs(
        train.loc[:, "percent_atom_al"]
        + train.loc[:, "percent_atom_ga"]
        + train.loc[:, "percent_atom_in"]
        - 1
    )
    <= 0.001
)



## === cell 5
train = train.drop(["percent_atom_in"], axis=1)
test = test.drop(["percent_atom_in"], axis=1)



## === cell 6
from sklearn.model_selection import train_test_split



## === cell 7
t1 = "formation_energy_ev_natom"
t2 = "bandgap_energy_ev"

X_train, X_validation = train_test_split(train, test_size=0.2, random_state=42)

y1_train = np.log1p(X_train[t1].values[:, np.newaxis])
y2_train = np.log1p(X_train[t2].values[:, np.newaxis])
X_train = X_train.drop(["id", t1, t2], axis=1)

y1_validation = np.log1p(X_validation[t1].values[:, np.newaxis])
y2_validation = np.log1p(X_validation[t2].values[:, np.newaxis])
X_validation = X_validation.drop(["id", t1, t2], axis=1)

print(X_train.shape, y1_train.shape, y2_train.shape)
print(X_validation.shape, y1_validation.shape, y2_validation.shape)



## === cell 8
import matplotlib.pyplot as plt



## === cell 9
plt.subplot(1, 2, 1)
plt.scatter(range(len(y1_train)), y1_train)

plt.subplot(1, 2, 2)
plt.scatter(range(len(y2_train)), y2_train)

plt.show()



## === cell 10
import tensorflow as tf



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 11
tf.random.set_seed(1)
np.random.seed(1)




## === cell 12
def build_model(input_dim):
    """Creates a Keras model with 8 dense layers of 16 tanh units and L2 regularisation."""
    l2 = tf.keras.regularizers.l2(0.001)
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    for i in range(8):
        x = tf.keras.layers.Dense(
            16, activation="tanh", kernel_regularizer=l2, name=f"dense_{i+1}"
        )(x)
    outputs = tf.keras.layers.Dense(1, name="output")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.004), loss="mse")
    return model


input_dim = X_train.shape[1]
model_fe = build_model(input_dim)  # formation energy model
model_bg = build_model(input_dim)  # bandgap model



## === cell 13
epochs = 500
batch_size = 32

history_fe = model_fe.fit(
    X_train.values,
    y1_train,
    validation_data=(X_validation.values, y1_validation),
    epochs=epochs,
    batch_size=batch_size,
    verbose=0,
)

history_bg = model_bg.fit(
    X_train.values,
    y2_train,
    validation_data=(X_validation.values, y2_validation),
    epochs=epochs,
    batch_size=batch_size,
    verbose=0,
)



## === cell 14
val_rmse_fe = np.sqrt(history_fe.history["val_loss"][-1])
val_rmse_bg = np.sqrt(history_bg.history["val_loss"][-1])
print(f"Final validation RMSE - formation energy: {val_rmse_fe:.4f}")
print(f"Final validation RMSE - bandgap: {val_rmse_bg:.4f}")



## === cell 15
sample = pd.read_csv("../input/sample_submission.csv")

X_test = test.drop(["id"], axis=1)
pred_y1_test = model_fe.predict(X_test.values, batch_size=batch_size)
pred_y2_test = model_bg.predict(X_test.values, batch_size=batch_size)

pred_y1_test = np.expm1(pred_y1_test)
pred_y2_test = np.expm1(pred_y2_test)

pred_y1_test[pred_y1_test < 0] = 0
pred_y2_test[pred_y2_test < 0] = 0

subm = pd.DataFrame(
    {
        "id": sample["id"],
        "formation_energy_ev_natom": pred_y1_test.ravel(),
        "bandgap_energy_ev": pred_y2_test.ravel(),
    }
)

subm.to_csv("subm.csv", index=False)
print("Submission file 'subm.csv' written:", subm.shape)
