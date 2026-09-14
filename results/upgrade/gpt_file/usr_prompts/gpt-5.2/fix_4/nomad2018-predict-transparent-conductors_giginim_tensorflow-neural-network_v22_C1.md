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

0.14502

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 0.14502) has done: 'I fix the TensorFlow import/runtime failure by avoiding TF1 graph construction in this environment and switching to the closest equivalent Keras Dense network that preserves the same architecture (8 layers, 16 units, tanh, L2 regularization) and the same log1p/expm1 target transformation used for RMSLE. I also remove the broken BatchNormalization placeholder wiring (it’s what triggers the unknown shape error) and keep the rest of the data prep and training loop semantics intact. Finally, I ensure the submission is written as a valid `.csv` with the required column names and aligned `id`s. These changes are necessary to get an end-to-end run and should yield a reasonable score rather than failing to produce a submission.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

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

t1 = "formation_energy_ev_natom"
t2 = "bandgap_energy_ev"

X_train, X_validation = train_test_split(train, test_size=0.2, random_state=1)

y1_train = np.log1p(X_train[t1].to_numpy(dtype=np.float32)[:, np.newaxis])
y2_train = np.log1p(X_train[t2].to_numpy(dtype=np.float32)[:, np.newaxis])
X_train = X_train.drop(["id", t1, t2], axis=1)

y1_validation = np.log1p(X_validation[t1].to_numpy(dtype=np.float32)[:, np.newaxis])
y2_validation = np.log1p(X_validation[t2].to_numpy(dtype=np.float32)[:, np.newaxis])
X_validation = X_validation.drop(["id", t1, t2], axis=1)

X_train = X_train.to_numpy(dtype=np.float32)
X_validation = X_validation.to_numpy(dtype=np.float32)

print(X_train.shape, y1_train.shape, y2_train.shape)
print(X_validation.shape, y1_validation.shape, y2_validation.shape)



## === cell 7
import matplotlib.pyplot as plt



## === cell 8
plt.subplot(1, 2, 1)
plt.scatter(range(len(y1_train)), y1_train)

plt.subplot(1, 2, 2)
plt.scatter(range(len(y2_train)), y2_train)

plt.show()



## === cell 9
import tensorflow as tf

np.random.seed(1)
tf.random.set_seed(1)

n_units = 16
n_layers = 8
activation = "tanh"
kernel_regularizer_l2 = tf.keras.regularizers.l2(0.001)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 10
inputs = tf.keras.Input(shape=(X_train.shape[1],), dtype=tf.float32, name="tf_x")

l1 = inputs
for i in range(n_layers):
    l1 = tf.keras.layers.Dense(
        n_units,
        activation=activation,
        kernel_regularizer=kernel_regularizer_l2,
        name="l%s_y1" % (i + 1),
    )(l1)
output_y1 = tf.keras.layers.Dense(1, name="output_y1")(l1)

l2 = inputs
for i in range(n_layers):
    l2 = tf.keras.layers.Dense(
        n_units,
        activation=activation,
        kernel_regularizer=kernel_regularizer_l2,
        name="l%s_y2" % (i + 1),
    )(l2)
output_y2 = tf.keras.layers.Dense(1, name="output_y2")(l2)

model = tf.keras.Model(inputs=inputs, outputs=[output_y1, output_y2])




## === cell 11
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


optimizer = tf.keras.optimizers.Adam(learning_rate=0.004)

model.compile(
    optimizer=optimizer,
    loss={"output_y1": rmse, "output_y2": rmse},
    loss_weights={"output_y1": 0.5, "output_y2": 0.5},
)

model.summary()



## === cell 12
history = model.fit(
    X_train,
    {"output_y1": y1_train, "output_y2": y2_train},
    validation_data=(
        X_validation,
        {"output_y1": y1_validation, "output_y2": y2_validation},
    ),
    epochs=500,
    batch_size=len(X_train),
    verbose=0,
)

loss_data = []
for step in range(len(history.history["loss"])):
    lt = history.history["loss"][step]
    lt_y1 = history.history["output_y1_loss"][step]
    lt_y2 = history.history["output_y2_loss"][step]
    lv = history.history["val_loss"][step]
    lv_y1 = history.history["val_output_y1_loss"][step]
    lv_y2 = history.history["val_output_y2_loss"][step]
    loss_data.append([step, lt, lt_y1, lt_y2, lv, lv_y1, lv_y2])

loss_data = np.array(loss_data, dtype=np.float32)
print(loss_data[-1][1:])



## === cell 13
plt.title("loss")
plt.plot(loss_data[:, 0], loss_data[:, 1], "r-")
plt.plot(loss_data[:, 0], loss_data[:, 4], "b-")
plt.show()



## === cell 14
plt.subplot(1, 2, 1)
plt.title("loss feen")
plt.plot(loss_data[:, 0], loss_data[:, 2], "r-")
plt.plot(loss_data[:, 0], loss_data[:, 5], "b-")

plt.subplot(1, 2, 2)
plt.title("loss bee")
plt.plot(loss_data[:, 0], loss_data[:, 3], "r-")
plt.plot(loss_data[:, 0], loss_data[:, 6], "b-")

plt.show()



## === cell 15
pred_y1, pred_y2 = model.predict(X_validation, batch_size=len(X_validation), verbose=0)
loss_val = float(
    0.5 * np.sqrt(np.mean((y1_validation - pred_y1) ** 2))
    + 0.5 * np.sqrt(np.mean((y2_validation - pred_y2) ** 2))
)
print("loss:", loss_val)

m_y1 = max(float(y1_validation.max()), float(pred_y1.max()))

ax1_y1 = plt.subplot(2, 2, 1)
ax1_y1.set_ylim([0, m_y1])
plt.scatter(range(len(y1_validation)), y1_validation)

ax2_y1 = plt.subplot(2, 2, 2)
ax2_y1.set_ylim([0, m_y1])
plt.scatter(range(len(pred_y1)), pred_y1, c="red")

m_y2 = max(float(y2_validation.max()), float(pred_y2.max()))

ax1_y2 = plt.subplot(2, 2, 3)
ax1_y2.set_ylim([0, m_y2])
plt.scatter(range(len(y2_validation)), y2_validation)

ax2_y2 = plt.subplot(2, 2, 4)
ax2_y2.set_ylim([0, m_y2])
plt.scatter(range(len(pred_y2)), pred_y2, c="red")

plt.show()



## === cell 16
sample = pd.read_csv("../input/sample_submission.csv")
sample.head()



## === cell 17
X_test = test.drop(["id"], axis=1).to_numpy(dtype=np.float32)

pred_y1, pred_y2 = model.predict(X_test, batch_size=len(X_test), verbose=0)

pred_y1 = np.expm1(pred_y1)
pred_y2 = np.expm1(pred_y2)

pred_y1[pred_y1 < 0] = 0
pred_y2[pred_y2 < 0] = 0

subm = pd.DataFrame()
subm["id"] = sample["id"].to_numpy()
subm["formation_energy_ev_natom"] = pred_y1.reshape(-1)
subm["bandgap_energy_ev"] = pred_y2.reshape(-1)

subm.to_csv("subm.csv", index=False)
print("Wrote submission:", os.path.abspath("subm.csv"), "shape=", subm.shape)
print(subm.head())
