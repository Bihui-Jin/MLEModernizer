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

0.07851

# 6. Current score

0.11297

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.21153) has done: 'I fix the TensorFlow import/runtime failure by avoiding the broken TF1 graph + `tf.compat.v1.layers.*` path (it’s incompatible with Keras 3 and also triggers a protobuf-related crash). I keep the same core model idea (batch norm on inputs, then an 8-layer tanh MLP with 16 units, two separate heads, Adam optimizer, RMSE on log1p targets) but implement it with `tf.keras` so it runs in the current Kaggle environment. I also ensure batch-norm updates are applied correctly (via the `training` flag) and keep determinism with fixed random seeds. Finally, I write a valid `subm.csv` with the exact required columns and row alignment to `sample_submission.csv`.'
- What this solution (achieved 0.11297) has done: 'I fix the TensorFlow import crash by forcing a compatible protobuf runtime before importing TensorFlow (the current error comes from an incompatible protobuf C++/python API mismatch). Then I keep your exact model/training logic intact, but also add a robust fallback that uses `tensorflow-cpu` if available and otherwise forces the pure-Python protobuf implementation early enough to take effect. Finally, I ensure the submission is always written with the required `.csv` suffix and exact column order aligned to `sample_submission.csv`.'

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
train = train.drop(["spacegroup", "percent_atom_in"], axis=1)
test = test.drop(["spacegroup", "percent_atom_in"], axis=1)



## === cell 6
from sklearn.model_selection import train_test_split

t1 = "formation_energy_ev_natom"
t2 = "bandgap_energy_ev"

X_train, X_validation = train_test_split(train, test_size=0.3, random_state=1)

y1_train = np.log1p(X_train[t1].to_numpy()[:, np.newaxis]).astype(np.float32)
y2_train = np.log1p(X_train[t2].to_numpy()[:, np.newaxis]).astype(np.float32)
X_train = X_train.drop(["id", t1, t2], axis=1)

y1_validation = np.log1p(X_validation[t1].to_numpy()[:, np.newaxis]).astype(np.float32)
y2_validation = np.log1p(X_validation[t2].to_numpy()[:, np.newaxis]).astype(np.float32)
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
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

import tensorflow as tf

print("TensorFlow version:", tf.__version__)

tf.random.set_seed(1)
np.random.seed(1)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 10
n_units = 16
n_layers = 8
activation = tf.nn.tanh
learning_rate = 0.004
n_steps = 500

input_dim = X_train.shape[1]




## === cell 11
class TwoHeadMLP(tf.keras.Model):
    def __init__(self, input_dim, n_units, n_layers):
        super().__init__()
        self.bn_in = tf.keras.layers.BatchNormalization(name="tf_xn")

        self.y1_layers = []
        for i in range(n_layers):
            self.y1_layers.append(
                tf.keras.layers.Dense(n_units, activation=activation, name=f"l{i+1}_y1")
            )
            self.y1_layers.append(
                tf.keras.layers.BatchNormalization(name=f"l{i+1}_y1n")
            )
        self.out_y1 = tf.keras.layers.Dense(1, name="output_y1")

        self.y2_layers = []
        for i in range(n_layers):
            self.y2_layers.append(
                tf.keras.layers.Dense(n_units, activation=activation, name=f"l{i+1}_y2")
            )
            self.y2_layers.append(
                tf.keras.layers.BatchNormalization(name=f"l{i+1}_y2n")
            )
        self.out_y2 = tf.keras.layers.Dense(1, name="output_y2")

        self.build((None, input_dim))

    def call(self, x, training=False):
        x = self.bn_in(x, training=training)

        h1 = x
        for layer in self.y1_layers:
            h1 = (
                layer(h1, training=training)
                if isinstance(layer, tf.keras.layers.BatchNormalization)
                else layer(h1)
            )
        y1 = self.out_y1(h1)

        h2 = x
        for layer in self.y2_layers:
            h2 = (
                layer(h2, training=training)
                if isinstance(layer, tf.keras.layers.BatchNormalization)
                else layer(h2)
            )
        y2 = self.out_y2(h2)

        return y1, y2


model = TwoHeadMLP(input_dim=input_dim, n_units=n_units, n_layers=n_layers)




## === cell 12
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

Xtr = tf.convert_to_tensor(X_train, dtype=tf.float32)
Xva = tf.convert_to_tensor(X_validation, dtype=tf.float32)
y1tr = tf.convert_to_tensor(y1_train, dtype=tf.float32)
y2tr = tf.convert_to_tensor(y2_train, dtype=tf.float32)
y1va = tf.convert_to_tensor(y1_validation, dtype=tf.float32)
y2va = tf.convert_to_tensor(y2_validation, dtype=tf.float32)



## === cell 13
loss_data = []

for step in range(n_steps):
    with tf.GradientTape() as tape:
        pred1_tr, pred2_tr = model(Xtr, training=True)
        loss_y1 = rmse(y1tr, pred1_tr)
        loss_y2 = rmse(y2tr, pred2_tr)
        loss = (loss_y1 + loss_y2) / 2.0

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    pred1_va, pred2_va = model(Xva, training=False)
    loss_y1_va = rmse(y1va, pred1_va)
    loss_y2_va = rmse(y2va, pred2_va)
    loss_va = (loss_y1_va + loss_y2_va) / 2.0

    loss_data.append([step, float(loss.numpy()), float(loss_va.numpy())])

loss_data = np.array(loss_data, dtype=np.float64)
print("final train/val loss:", loss_data[-1][1:])

plt.plot(loss_data[:, 0], loss_data[:, 1], "r-")
plt.plot(loss_data[:, 0], loss_data[:, 2], "b-")
plt.show()



## === cell 14
pred_y1_va, pred_y2_va = model(Xva, training=False)
pred_y1_va = pred_y1_va.numpy()
pred_y2_va = pred_y2_va.numpy()

val_loss = loss_data[-1, 2]
print("loss:", val_loss)

m_y1 = max(y1_validation.max(), pred_y1_va.max())

ax1_y1 = plt.subplot(2, 2, 1)
ax1_y1.set_ylim([0, m_y1])
plt.scatter(range(len(y1_validation)), y1_validation)

ax2_y1 = plt.subplot(2, 2, 2)
ax2_y1.set_ylim([0, m_y1])
plt.scatter(range(len(pred_y1_va)), pred_y1_va, c="red")

m_y2 = max(y2_validation.max(), pred_y2_va.max())

ax1_y2 = plt.subplot(2, 2, 3)
ax1_y2.set_ylim([0, m_y2])
plt.scatter(range(len(y2_validation)), y2_validation)

ax2_y2 = plt.subplot(2, 2, 4)
ax2_y2.set_ylim([0, m_y2])
plt.scatter(range(len(pred_y2_va)), pred_y2_va, c="red")

plt.show()



## === cell 15
sample = pd.read_csv("../input/sample_submission.csv")
sample.head()



## === cell 16
X_test = test.drop(["id"], axis=1).to_numpy(dtype=np.float32)
Xte = tf.convert_to_tensor(X_test, dtype=tf.float32)

pred_y1, pred_y2 = model(Xte, training=False)
pred_y1 = np.expm1(pred_y1.numpy())
pred_y2 = np.expm1(pred_y2.numpy())

pred_y1[pred_y1 < 0] = 0
pred_y2[pred_y2 < 0] = 0

subm = pd.DataFrame(
    {
        "id": sample["id"].to_numpy(),
        "formation_energy_ev_natom": pred_y1.reshape(-1),
        "bandgap_energy_ev": pred_y2.reshape(-1),
    }
)

subm = subm[["id", "formation_energy_ev_natom", "bandgap_energy_ev"]]

subm.to_csv("subm.csv", index=False)
print("Wrote submission:", os.path.abspath("subm.csv"), "shape:", subm.shape)
print(subm.head())
