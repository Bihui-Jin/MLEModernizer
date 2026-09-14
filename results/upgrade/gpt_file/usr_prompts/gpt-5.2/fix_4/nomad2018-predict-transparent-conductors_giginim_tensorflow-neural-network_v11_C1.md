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

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



## === cell 1
BASE_DIR = "../input/nomad2018-predict-transparent-conductors"
if os.path.exists(os.path.join(BASE_DIR, "train.csv")):
    TRAIN_PATH = os.path.join(BASE_DIR, "train.csv")
    TEST_PATH = os.path.join(BASE_DIR, "test.csv")
    SAMPLE_PATH = os.path.join(BASE_DIR, "sample_submission.csv")
else:
    TRAIN_PATH = "../input/train.csv"
    TEST_PATH = "../input/test.csv"
    SAMPLE_PATH = "../input/sample_submission.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print("train:", train.shape, "test:", test.shape)
print("train cols:", list(train.columns))



## === cell 2
from sklearn.model_selection import train_test_split

t1 = "formation_energy_ev_natom"
t2 = "bandgap_energy_ev"

X_train_df, X_validation_df = train_test_split(train, test_size=0.3, random_state=1)

y1_train = X_train_df[t1].to_numpy(dtype=np.float32)[:, np.newaxis]
y2_train = X_train_df[t2].to_numpy(dtype=np.float32)[:, np.newaxis]
X_train = X_train_df.drop(["id", t1, t2], axis=1).to_numpy(dtype=np.float32)

y1_validation = X_validation_df[t1].to_numpy(dtype=np.float32)[:, np.newaxis]
y2_validation = X_validation_df[t2].to_numpy(dtype=np.float32)[:, np.newaxis]
X_validation = X_validation_df.drop(["id", t1, t2], axis=1).to_numpy(dtype=np.float32)

print(X_train.shape, y1_train.shape, y2_train.shape)
print(X_validation.shape, y1_validation.shape, y2_validation.shape)



## === cell 3
import matplotlib.pyplot as plt

plt.subplot(1, 2, 1)
plt.scatter(range(len(y1_train)), y1_train, s=2)

plt.subplot(1, 2, 2)
plt.scatter(range(len(y2_train)), y2_train, s=2)

plt.tight_layout()
plt.show()



## === cell 4
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.random.set_seed(1)
np.random.seed(1)

n_units = 16
activation = tf.tanh

Dense = tf.keras.layers.Dense
BatchNorm = tf.keras.layers.BatchNormalization



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
tf_is_training = tf.compat.v1.placeholder_with_default(
    False, shape=(), name="tf_is_training"
)

tf_x = tf.compat.v1.placeholder(tf.float32, (None, X_train.shape[1]), name="tf_x")
tf_y1 = tf.compat.v1.placeholder(tf.float32, (None, 1), name="tf_y1")
tf_y2 = tf.compat.v1.placeholder(tf.float32, (None, 1), name="tf_y2")



## === cell 6
tf_xn = BatchNorm(name="tf_xn")(tf_x, training=tf_is_training)

l1_y1 = Dense(n_units, activation=activation, name="l1_y1")(tf_xn)
l1_y1n = BatchNorm(name="l1_y1n")(l1_y1, training=tf_is_training)
l2_y1 = Dense(n_units, activation=activation, name="l2_y1")(l1_y1n)
l2_y1n = BatchNorm(name="l2_y1n")(l2_y1, training=tf_is_training)
l3_y1 = Dense(n_units, activation=activation, name="l3_y1")(l2_y1n)
l3_y1n = BatchNorm(name="l3_y1n")(l3_y1, training=tf_is_training)
l4_y1 = Dense(n_units, activation=activation, name="l4_y1")(l3_y1n)
l4_y1n = BatchNorm(name="l4_y1n")(l4_y1, training=tf_is_training)
l5_y1 = Dense(n_units, activation=activation, name="l5_y1")(l4_y1n)
l5_y1n = BatchNorm(name="l5_y1n")(l5_y1, training=tf_is_training)
output_y1 = Dense(1, activation=tf.exp, name="output_y1")(l5_y1n)

l1_y2 = Dense(n_units, activation=activation, name="l1_y2")(tf_xn)
l1_y2n = BatchNorm(name="l1_y2n")(l1_y2, training=tf_is_training)
l2_y2 = Dense(n_units, activation=activation, name="l2_y2")(l1_y2n)
l2_y2n = BatchNorm(name="l2_y2n")(l2_y2, training=tf_is_training)
l3_y2 = Dense(n_units, activation=activation, name="l3_y2")(l2_y2n)
l3_y2n = BatchNorm(name="l3_y2n")(l3_y2, training=tf_is_training)
l4_y2 = Dense(n_units, activation=activation, name="l4_y2")(l3_y2n)
l4_y2n = BatchNorm(name="l4_y2n")(l4_y2, training=tf_is_training)
l5_y2 = Dense(n_units, activation=activation, name="l5_y2")(l4_y2n)
l5_y2n = BatchNorm(name="l5_y2n")(l5_y2, training=tf_is_training)
output_y2 = Dense(1, activation=tf.exp, name="output_y2")(l5_y2n)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
OperatorNotAllowedInGraphError            Traceback (most recent call last)
/tmp/ipykernel_11/1106083398.py in <cell line: 0>()
----> 1 tf_xn = BatchNorm(name="tf_xn")(tf_x, training=tf_is_training)
      2 
      3 l1_y1 = Dense(n_units, activation=activation, name="l1_y1")(tf_xn)
      4 l1_y1n = BatchNorm(name="l1_y1n")(l1_y1, training=tf_is_training)
      5 l2_y1 = Dense(n_units, activation=activation, name="l2_y1")(l1_y1n)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor.py in _disallow(self, task)
    301 
    302   def _disallow(self, task):
--> 303     raise errors.OperatorNotAllowedInGraphError(
    304         f"{task} is not allowed."
    305         " You can attempt the following resolutions to the problem:"

OperatorNotAllowedInGraphError: Exception encountered when calling BatchNormalization.call().

Using a symbolic `tf.Tensor` as a Python `bool` is not allowed. You can attempt the following resolutions to the problem: If you are running in Graph mode, use Eager execution mode or decorate this function with @tf.function. If you are using AutoGraph, you can try decorating this function with @tf.function. If that does not work, then you may be using an unsupported feature or your source code may not be visible to AutoGraph. See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/limitations.md#access-to-source-code for more information.

Arguments received by BatchNormalization.call():
  • inputs=tf.Tensor(shape=(None, 11), dtype=float32)
  • training=tf.Tensor(shape=(), dtype=bool)
  • mask=None

## === cell 7
eps = tf.constant(1e-7, dtype=tf.float32)

safe_y1 = tf.maximum(tf_y1, 0.0)
safe_y2 = tf.maximum(tf_y2, 0.0)
safe_o1 = tf.maximum(output_y1, 0.0)
safe_o2 = tf.maximum(output_y2, 0.0)

loss_y1 = tf.sqrt(
    tf.reduce_mean(
        tf.square(tf.math.log1p(safe_y1 + eps) - tf.math.log1p(safe_o1 + eps))
    )
)
loss_y2 = tf.sqrt(
    tf.reduce_mean(
        tf.square(tf.math.log1p(safe_y2 + eps) - tf.math.log1p(safe_o2 + eps))
    )
)
loss = (loss_y1 + loss_y2) / 2.0

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.08)

update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2930198389.py in <cell line: 0>()
      4 safe_y1 = tf.maximum(tf_y1, 0.0)
      5 safe_y2 = tf.maximum(tf_y2, 0.0)
----> 6 safe_o1 = tf.maximum(output_y1, 0.0)
      7 safe_o2 = tf.maximum(output_y2, 0.0)
      8 

NameError: name 'output_y1' is not defined

## === cell 8
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())



## === cell 9
for step in range(2000):
    _, l = sess.run(
        [train_op, loss],
        feed_dict={
            tf_x: X_train,
            tf_y1: y1_train,
            tf_y2: y2_train,
            tf_is_training: True,
        },
    )
    if step % 100 == 0:
        print(l)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3461667880.py in <cell line: 0>()
      1 for step in range(2000):
      2     _, l = sess.run(
----> 3         [train_op, loss],
      4         feed_dict={
      5             tf_x: X_train,

NameError: name 'train_op' is not defined

## === cell 10
val_loss, pred_y1, pred_y2 = sess.run(
    [loss, output_y1, output_y2],
    feed_dict={
        tf_x: X_validation,
        tf_y1: y1_validation,
        tf_y2: y2_validation,
        tf_is_training: False,
    },
)

print("loss:", float(val_loss))

m_y1 = float(max(y1_validation.max(), pred_y1.max()))
ax1_y1 = plt.subplot(2, 2, 1)
ax1_y1.set_ylim([0, m_y1])
plt.scatter(range(len(y1_validation)), y1_validation, s=2)

ax2_y1 = plt.subplot(2, 2, 2)
ax2_y1.set_ylim([0, m_y1])
plt.scatter(range(len(pred_y1)), pred_y1, c="red", s=2)

m_y2 = float(max(y2_validation.max(), pred_y2.max()))
ax1_y2 = plt.subplot(2, 2, 3)
ax1_y2.set_ylim([0, m_y2])
plt.scatter(range(len(y2_validation)), y2_validation, s=2)

ax2_y2 = plt.subplot(2, 2, 4)
ax2_y2.set_ylim([0, m_y2])
plt.scatter(range(len(pred_y2)), pred_y2, c="red", s=2)

plt.tight_layout()
plt.show()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3772490078.py in <cell line: 0>()
      1 val_loss, pred_y1, pred_y2 = sess.run(
----> 2     [loss, output_y1, output_y2],
      3     feed_dict={
      4         tf_x: X_validation,
      5         tf_y1: y1_validation,

NameError: name 'loss' is not defined

## === cell 11
sample = pd.read_csv(SAMPLE_PATH)
print(sample.head())
print("sample columns:", list(sample.columns), "shape:", sample.shape)



## === cell 12
X_test = test.drop(["id"], axis=1).to_numpy(dtype=np.float32)

pred_y1_test, pred_y2_test = sess.run(
    [output_y1, output_y2],
    feed_dict={tf_x: X_test, tf_is_training: False},
)

pred_y1_test = np.maximum(pred_y1_test.reshape(-1).astype(float), 0.0)
pred_y2_test = np.maximum(pred_y2_test.reshape(-1).astype(float), 0.0)

subm = pd.DataFrame(
    {
        "id": test["id"].astype(int).to_numpy(),
        "formation_energy_ev_natom": pred_y1_test,
        "bandgap_energy_ev": pred_y2_test,
    }
).sort_values("id")

subm.to_csv("submission.csv", index=False)
subm.to_csv("subm.csv", index=False)

print(subm.head())
print("Wrote submission.csv with shape:", subm.shape)
print("Saved to:", os.path.abspath("submission.csv"))

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/988604431.py in <cell line: 0>()
      2 
      3 pred_y1_test, pred_y2_test = sess.run(
----> 4     [output_y1, output_y2],
      5     feed_dict={tf_x: X_test, tf_is_training: False},
      6 )

NameError: name 'output_y1' is not defined
