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

0.07285

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

if os.path.exists("/kaggle/input"):
    INPUT_DIR = "/kaggle/input"
elif os.path.exists("../input"):
    INPUT_DIR = "../input"
else:
    INPUT_DIR = "/kaggle/data"

print("Using INPUT_DIR:", INPUT_DIR)
try:
    print(check_output(["ls", INPUT_DIR]).decode("utf8"))
except Exception as e:
    print("Could not list input dir:", e)



## === cell 1
train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")

if (not os.path.exists(train_path)) and os.path.exists(
    os.path.join(INPUT_DIR, "nomad2018-predict-transparent-conductors", "train.csv")
):
    INPUT_DIR = os.path.join(INPUT_DIR, "nomad2018-predict-transparent-conductors")
    train_path = os.path.join(INPUT_DIR, "train.csv")
    test_path = os.path.join(INPUT_DIR, "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print(train.shape, test.shape)
train.head()



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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



## === cell 6
t1 = "formation_energy_ev_natom"
t2 = "bandgap_energy_ev"

feature_columns = [
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

all_columns = [t1, t2, *feature_columns]



## === cell 7
all_df = pd.concat(
    [train[feature_columns], test[feature_columns]], axis=0, ignore_index=True
)

scaler = MinMaxScaler()
scaler.fit(all_df[feature_columns].astype(np.float32))

train.loc[:, feature_columns] = scaler.transform(
    train[feature_columns].astype(np.float32)
)
test.loc[:, feature_columns] = scaler.transform(
    test[feature_columns].astype(np.float32)
)



## === cell 8
X_train, X_validation = train_test_split(train, test_size=0.2, random_state=1)

y_train = np.log1p(X_train[[t1, t2]].astype(np.float32))
X_train = X_train.drop(["id", t1, t2], axis=1)

y_validation = np.log1p(X_validation[[t1, t2]].astype(np.float32))
X_validation = X_validation.drop(["id", t1, t2], axis=1)

print(X_train.shape, y_train.shape)
print(X_validation.shape, y_validation.shape)



## === cell 9
import matplotlib.pyplot as plt



## === cell 10
plt.subplot(1, 2, 1)
plt.scatter(range(y_train.shape[0]), y_train[t1])

plt.subplot(1, 2, 2)
plt.scatter(range(y_train.shape[0]), y_train[t2])

plt.show()



## === cell 11
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
print("TensorFlow version:", tf.__version__)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 12
tf.compat.v1.set_random_seed(1)
np.random.seed(1)

n_units = 8
n_layers = 16

activation = tf.tanh

kernel_regularizer_l2 = tf.keras.regularizers.l2(0.001)



## === cell 13
tf.compat.v1.reset_default_graph()

tf_is_training = tf.compat.v1.placeholder(tf.bool, None)

tf_x = tf.compat.v1.placeholder(tf.float32, (None, X_train.shape[1]), name="tf_x")
tf_y = tf.compat.v1.placeholder(tf.float32, (None, 2), name="tf_y")



## === cell 14
i = 1

l = tf.keras.layers.Dense(
    256,
    activation=activation,
    kernel_regularizer=kernel_regularizer_l2,
    name="layer%s" % i,
)(tf_x)
l = tf.keras.layers.Dropout(rate=0.5, name="dropout")(l, training=tf_is_training)

i += 1
l = tf.keras.layers.Dense(
    128,
    activation=activation,
    kernel_regularizer=kernel_regularizer_l2,
    name="layer%s" % i,
)(l)

i += 1
l = tf.keras.layers.Dense(
    64,
    activation=activation,
    kernel_regularizer=kernel_regularizer_l2,
    name="layer%s" % i,
)(l)

i += 1
l = tf.keras.layers.Dense(
    32,
    activation=activation,
    kernel_regularizer=kernel_regularizer_l2,
    name="layer%s" % i,
)(l)

i += 1
l = tf.keras.layers.Dense(
    16,
    activation=activation,
    kernel_regularizer=kernel_regularizer_l2,
    name="layer%s" % i,
)(l)

i += 1
l = tf.keras.layers.Dense(
    8,
    activation=activation,
    kernel_regularizer=kernel_regularizer_l2,
    name="layer%s" % i,
)(l)

output = tf.keras.layers.Dense(2, name="output")(l)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
OperatorNotAllowedInGraphError            Traceback (most recent call last)
/tmp/ipykernel_11/780181662.py in <cell line: 0>()
      9     name="layer%s" % i,
     10 )(tf_x)
---> 11 l = tf.keras.layers.Dropout(rate=0.5, name="dropout")(l, training=tf_is_training)
     12 
     13 i += 1

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

OperatorNotAllowedInGraphError: Exception encountered when calling Dropout.call().

Using a symbolic `tf.Tensor` as a Python `bool` is not allowed. You can attempt the following resolutions to the problem: If you are running in Graph mode, use Eager execution mode or decorate this function with @tf.function. If you are using AutoGraph, you can try decorating this function with @tf.function. If that does not work, then you may be using an unsupported feature or your source code may not be visible to AutoGraph. See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/limitations.md#access-to-source-code for more information.

Arguments received by Dropout.call():
  • inputs=tf.Tensor(shape=(None, 256), dtype=float32)
  • training=tf.Tensor(shape=<unknown>, dtype=bool)

## === cell 15
mse_y1 = tf.reduce_mean(tf.square(tf_y[:, 0] - output[:, 0]))
mse_y2 = tf.reduce_mean(tf.square(tf_y[:, 1] - output[:, 1]))
loss_y1 = tf.sqrt(mse_y1)
loss_y2 = tf.sqrt(mse_y2)
loss_total = (loss_y1 + loss_y2) / 2.0

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.004)
train_op = optimizer.minimize(loss_total)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/117833666.py in <cell line: 0>()
----> 1 mse_y1 = tf.reduce_mean(tf.square(tf_y[:, 0] - output[:, 0]))
      2 mse_y2 = tf.reduce_mean(tf.square(tf_y[:, 1] - output[:, 1]))
      3 loss_y1 = tf.sqrt(mse_y1)
      4 loss_y2 = tf.sqrt(mse_y2)
      5 loss_total = (loss_y1 + loss_y2) / 2.0

NameError: name 'output' is not defined

## === cell 16
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())



## === cell 17
loss_data = []

X_tr_np = X_train.values.astype(np.float32)
y_tr_np = y_train.values.astype(np.float32)
X_va_np = X_validation.values.astype(np.float32)
y_va_np = y_validation.values.astype(np.float32)

for step in range(500):
    _, lt, lt_y1, lt_y2 = sess.run(
        [train_op, loss_total, loss_y1, loss_y2],
        {tf_x: X_tr_np, tf_y: y_tr_np, tf_is_training: True},
    )

    lv, lv_y1, lv_y2 = sess.run(
        [loss_total, loss_y1, loss_y2],
        {tf_x: X_va_np, tf_y: y_va_np, tf_is_training: False},
    )

    loss_data.append([step, lt, lt_y1, lt_y2, lv, lv_y1, lv_y2])

loss_data = np.array(loss_data, dtype=np.float32)
print(loss_data[-1][1:])



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/579478042.py in <cell line: 0>()
      8 for step in range(500):
      9     _, lt, lt_y1, lt_y2 = sess.run(
---> 10         [train_op, loss_total, loss_y1, loss_y2],
     11         {tf_x: X_tr_np, tf_y: y_tr_np, tf_is_training: True},
     12     )

NameError: name 'train_op' is not defined

## === cell 18
plt.title("loss")
plt.plot(loss_data[:, 0], loss_data[:, 1], "r-")
plt.plot(loss_data[:, 0], loss_data[:, 4], "b-")
plt.show()



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/614300576.py in <cell line: 0>()
      1 plt.title("loss")
----> 2 plt.plot(loss_data[:, 0], loss_data[:, 1], "r-")
      3 plt.plot(loss_data[:, 0], loss_data[:, 4], "b-")
      4 plt.show()
      5 

TypeError: list indices must be integers or slices, not tuple

## === cell 19
plt.subplot(1, 2, 1)
plt.title("loss feen")
plt.plot(loss_data[:, 0], loss_data[:, 2], "r-")
plt.plot(loss_data[:, 0], loss_data[:, 5], "b-")

plt.subplot(1, 2, 2)
plt.title("loss bee")
plt.plot(loss_data[:, 0], loss_data[:, 3], "r-")
plt.plot(loss_data[:, 0], loss_data[:, 6], "b-")

plt.show()



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3870699113.py in <cell line: 0>()
      1 plt.subplot(1, 2, 1)
      2 plt.title("loss feen")
----> 3 plt.plot(loss_data[:, 0], loss_data[:, 2], "r-")
      4 plt.plot(loss_data[:, 0], loss_data[:, 5], "b-")
      5 

TypeError: list indices must be integers or slices, not tuple

## === cell 20
loss, pred_y = sess.run(
    [loss_total, output], {tf_x: X_va_np, tf_y: y_va_np, tf_is_training: False}
)

print("loss:", float(loss))

n_row = len(y_validation)

m_y1 = max(float(y_validation[t1].max()), float(pred_y[:, 0].max()))

ax1_y1 = plt.subplot(2, 2, 1)
ax1_y1.set_ylim([0, m_y1])
plt.scatter(range(n_row), y_validation[t1].values)

ax2_y1 = plt.subplot(2, 2, 2)
ax2_y1.set_ylim([0, m_y1])
plt.scatter(range(n_row), pred_y[:, 0], c="red")

m_y2 = max(float(y_validation[t2].max()), float(pred_y[:, 1].max()))

ax1_y2 = plt.subplot(2, 2, 3)
ax1_y2.set_ylim([0, m_y2])
plt.scatter(range(n_row), y_validation[t2].values)

ax2_y2 = plt.subplot(2, 2, 4)
ax2_y2.set_ylim([0, m_y2])
plt.scatter(range(n_row), pred_y[:, 1], c="red")

plt.show()



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2373015132.py in <cell line: 0>()
      1 loss, pred_y = sess.run(
----> 2     [loss_total, output], {tf_x: X_va_np, tf_y: y_va_np, tf_is_training: False}
      3 )
      4 
      5 print("loss:", float(loss))

NameError: name 'loss_total' is not defined

## === cell 21
sample_path = os.path.join(INPUT_DIR, "sample_submission.csv")
sample = pd.read_csv(sample_path)
sample.head()



## === cell 22
y_train_full = np.log1p(train[[t1, t2]].astype(np.float32))
X_train_full = train.drop(["id", t1, t2], axis=1).values.astype(np.float32)

sess.close()
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

loss_data = []
for step in range(2000):
    _, l, l_y1, l_y2 = sess.run(
        [train_op, loss_total, loss_y1, loss_y2],
        {
            tf_x: X_train_full,
            tf_y: y_train_full.values.astype(np.float32),
            tf_is_training: True,
        },
    )
    loss_data.append([step, l, l_y1, l_y2])

print(loss_data[-1][1:])

loss_data = np.array(loss_data, dtype=np.float32)



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/214346025.py in <cell line: 0>()
      9 for step in range(2000):
     10     _, l, l_y1, l_y2 = sess.run(
---> 11         [train_op, loss_total, loss_y1, loss_y2],
     12         {
     13             tf_x: X_train_full,

NameError: name 'train_op' is not defined

## === cell 23
plt.title("loss")
plt.plot(loss_data[:, 0], loss_data[:, 1])
plt.show()

plt.subplot(1, 2, 1)
plt.title("loss feen")
plt.plot(loss_data[:, 0], loss_data[:, 2])
plt.subplot(1, 2, 2)
plt.title("loss bee")
plt.plot(loss_data[:, 0], loss_data[:, 3])

plt.show()



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4124563771.py in <cell line: 0>()
      1 plt.title("loss")
----> 2 plt.plot(loss_data[:, 0], loss_data[:, 1])
      3 plt.show()
      4 
      5 plt.subplot(1, 2, 1)

TypeError: list indices must be integers or slices, not tuple

## === cell 24
X_test = test.drop(["id"], axis=1).values.astype(np.float32)
pred_y = sess.run(output, {tf_x: X_test, tf_is_training: False})

pred_y = np.expm1(pred_y)

pred_y[pred_y[:, 0] < 0, 0] = 0
pred_y[pred_y[:, 1] < 0, 1] = 0

subm = pd.DataFrame()
subm["id"] = sample["id"].values
subm[t1] = pred_y[:, 0]
subm[t2] = pred_y[:, 1]

subm.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", subm.shape)
subm.head()

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/461291321.py in <cell line: 0>()
      1 X_test = test.drop(["id"], axis=1).values.astype(np.float32)
----> 2 pred_y = sess.run(output, {tf_x: X_test, tf_is_training: False})
      3 
      4 pred_y = np.expm1(pred_y)
      5 

NameError: name 'output' is not defined
