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

0.07851

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

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
        train["percent_atom_al"]
        + train["percent_atom_ga"]
        + train["percent_atom_in"]
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

y1_train = np.log1p(X_train[t1].values[:, np.newaxis])
y2_train = np.log1p(X_train[t2].values[:, np.newaxis])
X_train = X_train.drop(["id", t1, t2], axis=1)

y1_validation = np.log1p(X_validation[t1].values[:, np.newaxis])
y2_validation = np.log1p(X_validation[t2].values[:, np.newaxis])
X_validation = X_validation.drop(["id", t1, t2], axis=1)

X_train_np = X_train.values.astype(np.float32)
X_validation_np = X_validation.values.astype(np.float32)

print(X_train.shape, y1_train.shape, y2_train.shape)
print(X_validation.shape, y1_validation.shape, y2_validation.shape)



## === cell 7
import matplotlib.pyplot as plt



## === cell 8
plt.subplot(1, 2, 1)
plt.scatter(range(len(y1_train)), y1_train, s=10)
plt.title("log1p formation energy")

plt.subplot(1, 2, 2)
plt.scatter(range(len(y2_train)), y2_train, s=10)
plt.title("log1p bandgap")
plt.show()



## === cell 9
import tensorflow as tf



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 10
tf.compat.v1.set_random_seed(1)
np.random.seed(1)
tf.compat.v1.reset_default_graph()

n_units = 16
n_layers = 8
activation = tf.tanh



## === cell 11
tf_is_training = tf.compat.v1.placeholder(tf.bool, shape=(), name="tf_is_training")
tf_x = tf.compat.v1.placeholder(tf.float32, shape=(None, X_train.shape[1]), name="tf_x")
tf_y1 = tf.compat.v1.placeholder(tf.float32, shape=(None, 1), name="tf_y1")
tf_y2 = tf.compat.v1.placeholder(tf.float32, shape=(None, 1), name="tf_y2")



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/388956687.py in <cell line: 0>()
----> 1 tf_is_training = tf.compat.v1.placeholder(tf.bool, shape=(), name="tf_is_training")
      2 tf_x = tf.compat.v1.placeholder(tf.float32, shape=(None, X_train.shape[1]), name="tf_x")
      3 tf_y1 = tf.compat.v1.placeholder(tf.float32, shape=(None, 1), name="tf_y1")
      4 tf_y2 = tf.compat.v1.placeholder(tf.float32, shape=(None, 1), name="tf_y2")
      5 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/array_ops.py in placeholder(dtype, shape, name)
   3022   """
   3023   if context.executing_eagerly():
-> 3024     raise RuntimeError("tf.placeholder() is not compatible with "
   3025                        "eager execution.")
   3026 

RuntimeError: tf.placeholder() is not compatible with eager execution.

## === cell 12
tf_xn = tf.compat.v1.layers.batch_normalization(
    tf_x, training=tf_is_training, name="tf_xn"
)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/598343753.py in <cell line: 0>()
----> 1 tf_xn = tf.compat.v1.layers.batch_normalization(
      2     tf_x, training=tf_is_training, name="tf_xn"
      3 )
      4 
      5 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/lazy_loader.py in __getattr__(self, item)
    205           "__internal__.legacy."
    206       ):
--> 207         raise AttributeError(
    208             f"`{item}` is not available with Keras 3."
    209         )

AttributeError: `batch_normalization` is not available with Keras 3.

## === cell 13
def add_norm_layer(inputs, nunits, activation, training=False, name=None):
    dense = tf.compat.v1.layers.dense(inputs, nunits, activation=activation, name=name)
    bn = tf.compat.v1.layers.batch_normalization(
        dense, training=training, name=name + "_bn"
    )
    return bn




## === cell 14
l = tf_xn
for i in range(n_layers):
    l = add_norm_layer(
        l, n_units, activation, training=tf_is_training, name="l{}_y1".format(i + 1)
    )
output_y1 = tf.compat.v1.layers.dense(l, 1, name="output_y1")

l = tf_xn
for i in range(n_layers):
    l = add_norm_layer(
        l, n_units, activation, training=tf_is_training, name="l{}_y2".format(i + 1)
    )
output_y2 = tf.compat.v1.layers.dense(l, 1, name="output_y2")



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1124553877.py in <cell line: 0>()
      1 # Formation energy branch
----> 2 l = tf_xn
      3 for i in range(n_layers):
      4     l = add_norm_layer(
      5         l, n_units, activation, training=tf_is_training, name="l{}_y1".format(i + 1)

NameError: name 'tf_xn' is not defined

## === cell 15
loss_y1 = tf.sqrt(tf.compat.v1.losses.mean_squared_error(tf_y1, output_y1))
loss_y2 = tf.sqrt(tf.compat.v1.losses.mean_squared_error(tf_y2, output_y2))
loss = (loss_y1 + loss_y2) / 2.0

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.004)
train_op = optimizer.minimize(loss)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2866093141.py in <cell line: 0>()
----> 1 loss_y1 = tf.sqrt(tf.compat.v1.losses.mean_squared_error(tf_y1, output_y1))
      2 loss_y2 = tf.sqrt(tf.compat.v1.losses.mean_squared_error(tf_y2, output_y2))
      3 loss = (loss_y1 + loss_y2) / 2.0
      4 
      5 optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.004)

NameError: name 'tf_y1' is not defined

## === cell 16
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/2478098849.py in <cell line: 0>()
      1 sess = tf.compat.v1.Session()
----> 2 sess.run(tf.compat.v1.global_variables_initializer())
      3 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/client/session.py in run(self, fetches, feed_dict, options, run_metadata)
    975 
    976     try:
--> 977       result = self._run(None, fetches, feed_dict, options_ptr,
    978                          run_metadata_ptr)
    979       if run_metadata:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/client/session.py in _run(self, handle, fetches, feed_dict, options, run_metadata)
   1145       raise RuntimeError('Attempted to use a closed Session.')
   1146     if self.graph.version == 0:
-> 1147       raise RuntimeError('The Session graph is empty. Add operations to the '
   1148                          'graph before calling run().')
   1149 

RuntimeError: The Session graph is empty. Add operations to the graph before calling run().

## === cell 17
loss_data = []

for step in range(500):
    _, lt = sess.run(
        [train_op, loss],
        feed_dict={
            tf_x: X_train_np,
            tf_y1: y1_train,
            tf_y2: y2_train,
            tf_is_training: True,
        },
    )

    lv = sess.run(
        loss,
        feed_dict={
            tf_x: X_validation_np,
            tf_y1: y1_validation,
            tf_y2: y2_validation,
            tf_is_training: False,
        },
    )

    loss_data.append([step, lt, lv])

loss_data = np.array(loss_data)

print("Final train loss, validation loss:", loss_data[-1, 1:])

plt.plot(loss_data[:, 0], loss_data[:, 1], "r-", label="train")
plt.plot(loss_data[:, 0], loss_data[:, 2], "b-", label="valid")
plt.legend()
plt.show()



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2010438557.py in <cell line: 0>()
      3 for step in range(500):
      4     _, lt = sess.run(
----> 5         [train_op, loss],
      6         feed_dict={
      7             tf_x: X_train_np,

NameError: name 'train_op' is not defined

## === cell 18
val_loss, pred_y1, pred_y2 = sess.run(
    [loss, output_y1, output_y2],
    feed_dict={
        tf_x: X_validation_np,
        tf_y1: y1_validation,
        tf_y2: y2_validation,
        tf_is_training: False,
    },
)

print("Validation loss after training:", val_loss)

m_y1 = max(y1_validation.max(), pred_y1.max())
plt.subplot(2, 2, 1)
plt.ylim(0, m_y1)
plt.scatter(range(len(y1_validation)), y1_validation, s=10)
plt.title("True log1p formation")

plt.subplot(2, 2, 2)
plt.ylim(0, m_y1)
plt.scatter(range(len(pred_y1)), pred_y1, c="red", s=10)
plt.title("Pred log1p formation")

m_y2 = max(y2_validation.max(), pred_y2.max())
plt.subplot(2, 2, 3)
plt.ylim(0, m_y2)
plt.scatter(range(len(y2_validation)), y2_validation, s=10)
plt.title("True log1p bandgap")

plt.subplot(2, 2, 4)
plt.ylim(0, m_y2)
plt.scatter(range(len(pred_y2)), pred_y2, c="red", s=10)
plt.title("Pred log1p bandgap")
plt.show()



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2440283310.py in <cell line: 0>()
      1 val_loss, pred_y1, pred_y2 = sess.run(
----> 2     [loss, output_y1, output_y2],
      3     feed_dict={
      4         tf_x: X_validation_np,
      5         tf_y1: y1_validation,

NameError: name 'loss' is not defined

## === cell 19
sample = pd.read_csv("../input/sample_submission.csv")
sample.head()



## === cell 20
X_test = test.drop(["id"], axis=1).values.astype(np.float32)

pred_y1_test, pred_y2_test = sess.run(
    [output_y1, output_y2], feed_dict={tf_x: X_test, tf_is_training: False}
)

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
print("Submission file written to subm.csv")

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3455139305.py in <cell line: 0>()
      2 
      3 pred_y1_test, pred_y2_test = sess.run(
----> 4     [output_y1, output_y2], feed_dict={tf_x: X_test, tf_is_training: False}
      5 )
      6 

NameError: name 'output_y1' is not defined
