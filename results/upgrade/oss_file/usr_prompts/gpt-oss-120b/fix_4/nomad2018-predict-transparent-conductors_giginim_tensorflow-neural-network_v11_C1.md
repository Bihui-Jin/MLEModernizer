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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.08021

# 6. Current score

0.06498

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.06694) has done: 'The fix adds a protobuf compatibility setting before importing TensorFlow, ensures loss calculation works with mixed dtypes by casting the true values to float32, and converts TensorFlow prediction tensors to NumPy arrays when creating the submission file. These changes resolve the import error, the RMSLE dtype error, and the attribute error on `squeeze`, allowing the notebook to run end‑to‑end and produce a valid CSV submission.'
- What this solution (achieved 0.06498) has done: 'The fix adds the missing protobuf version environment variable so TensorFlow can be imported without the `MessageFactory` error. This change is applied before any TensorFlow import, preserving the original model and training logic while still producing a valid `subm.csv` submission.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))




## === cell 1
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")




## === cell 2
from sklearn.model_selection import train_test_split

t1 = "formation_energy_ev_natom"
t2 = "bandgap_energy_ev"

X_train, X_validation = train_test_split(train, test_size=0.3, random_state=42)

y1_train = X_train[t1].values[:, np.newaxis]
y2_train = X_train[t2].values[:, np.newaxis]
X_train = X_train.drop(["id", t1, t2], axis=1)

y1_validation = X_validation[t1].values[:, np.newaxis]
y2_validation = X_validation[t2].values[:, np.newaxis]
X_validation = X_validation.drop(["id", t1, t2], axis=1)

print(X_train.shape, y1_train.shape, y2_train.shape)
print(X_validation.shape, y1_validation.shape, y2_validation.shape)




## === cell 3
import matplotlib.pyplot as plt




## === cell 4
plt.subplot(1, 2, 1)
plt.scatter(range(len(y1_train)), y1_train)
plt.subplot(1, 2, 2)
plt.scatter(range(len(y2_train)), y2_train)
plt.show()




## === cell 5
import tensorflow as tf




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 6
tf.random.set_seed(1)
np.random.seed(1)




## === cell 7
n_units = 16
activation = tf.tanh

inputs = tf.keras.Input(shape=(X_train.shape[1],), name="tf_x")
x_norm = tf.keras.layers.BatchNormalization()(inputs, training=True)

l1_y1 = tf.keras.layers.Dense(n_units, activation=activation, name="l1_y1")(x_norm)
l1_y1n = tf.keras.layers.BatchNormalization()(l1_y1, training=True)
l2_y1 = tf.keras.layers.Dense(n_units, activation=activation, name="l2_y1")(l1_y1n)
l2_y1n = tf.keras.layers.BatchNormalization()(l2_y1, training=True)
l3_y1 = tf.keras.layers.Dense(n_units, activation=activation, name="l3_y1")(l2_y1n)
l3_y1n = tf.keras.layers.BatchNormalization()(l3_y1, training=True)
l4_y1 = tf.keras.layers.Dense(n_units, activation=activation, name="l4_y1")(l3_y1n)
l4_y1n = tf.keras.layers.BatchNormalization()(l4_y1, training=True)
l5_y1 = tf.keras.layers.Dense(n_units, activation=activation, name="l5_y1")(l4_y1n)
l5_y1n = tf.keras.layers.BatchNormalization()(l5_y1, training=True)
output_y1 = tf.keras.layers.Dense(1, activation=tf.exp, name="output_y1")(l5_y1n)

l1_y2 = tf.keras.layers.Dense(n_units, activation=activation, name="l1_y2")(x_norm)
l1_y2n = tf.keras.layers.BatchNormalization()(l1_y2, training=True)
l2_y2 = tf.keras.layers.Dense(n_units, activation=activation, name="l2_y2")(l1_y2n)
l2_y2n = tf.keras.layers.BatchNormalization()(l2_y2, training=True)
l3_y2 = tf.keras.layers.Dense(n_units, activation=activation, name="l3_y2")(l2_y2n)
l3_y2n = tf.keras.layers.BatchNormalization()(l3_y2, training=True)
l4_y2 = tf.keras.layers.Dense(n_units, activation=activation, name="l4_y2")(l3_y2n)
l4_y2n = tf.keras.layers.BatchNormalization()(l4_y2, training=True)
l5_y2 = tf.keras.layers.Dense(n_units, activation=activation, name="l5_y2")(l4_y2n)
l5_y2n = tf.keras.layers.BatchNormalization()(l5_y2, training=True)
output_y2 = tf.keras.layers.Dense(1, activation=tf.exp, name="output_y2")(l5_y2n)

model = tf.keras.Model(inputs=inputs, outputs=[output_y1, output_y2])




## === cell 8
def rmsle(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    return tf.sqrt(
        tf.reduce_mean(tf.square(tf.math.log1p(y_true) - tf.math.log1p(y_pred)))
    )


optimizer = tf.keras.optimizers.SGD(learning_rate=0.08)




## === cell 9
epochs = 2000
for step in range(epochs):
    with tf.GradientTape() as tape:
        pred_y1, pred_y2 = model(X_train, training=True)
        loss_y1 = rmsle(y1_train, pred_y1)
        loss_y2 = rmsle(y2_train, pred_y2)
        loss = (loss_y1 + loss_y2) / 2.0
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if step % 100 == 0:
        print(f"Step {step}: loss = {loss.numpy():.6f}")




## === cell 10
val_pred_y1, val_pred_y2 = model(X_validation, training=False)
val_loss_y1 = rmsle(y1_validation, val_pred_y1).numpy()
val_loss_y2 = rmsle(y2_validation, val_pred_y2).numpy()
print(f"Validation RMSLE formation: {val_loss_y1:.6f}")
print(f"Validation RMSLE bandgap:   {val_loss_y2:.6f}")

m_y1 = max(y1_validation.max(), val_pred_y1.numpy().max())
plt.subplot(2, 2, 1)
plt.ylim([0, m_y1])
plt.scatter(range(len(y1_validation)), y1_validation, label="true")
plt.title("Formation (true)")

plt.subplot(2, 2, 2)
plt.ylim([0, m_y1])
plt.scatter(range(len(val_pred_y1)), val_pred_y1, color="red", label="pred")
plt.title("Formation (pred)")

m_y2 = max(y2_validation.max(), val_pred_y2.numpy().max())
plt.subplot(2, 2, 3)
plt.ylim([0, m_y2])
plt.scatter(range(len(y2_validation)), y2_validation, label="true")
plt.title("Bandgap (true)")

plt.subplot(2, 2, 4)
plt.ylim([0, m_y2])
plt.scatter(range(len(val_pred_y2)), val_pred_y2, color="red", label="pred")
plt.title("Bandgap (pred)")

plt.tight_layout()
plt.show()




## === cell 11
sample = pd.read_csv("../input/sample_submission.csv")
sample.head()




## === cell 12
X_test = test.drop(["id"], axis=1)
test_pred_y1, test_pred_y2 = model(X_test, training=False)

subm = pd.DataFrame(
    {
        "id": sample["id"],
        "formation_energy_ev_natom": test_pred_y1.numpy().squeeze(),
        "bandgap_energy_ev": test_pred_y2.numpy().squeeze(),
    }
)
subm.to_csv("subm.csv", index=False)
print("Submission file 'subm.csv' written.")
