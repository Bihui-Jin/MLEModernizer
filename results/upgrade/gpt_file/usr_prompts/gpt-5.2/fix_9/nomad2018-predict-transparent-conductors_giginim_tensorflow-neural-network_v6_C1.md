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

0.2144

# 6. Current score

0.4982

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.65708) has done: 'The crash is caused by an incompatibility between TensorFlow 2.18 and the protobuf C-extension in this runtime; the fix is to force protobuf to use the pure-Python implementation before importing TensorFlow. Once TensorFlow imports cleanly, the downstream NameErrors (missing `train_test_split`, `plt`, `tf`, `sess`) resolve because those variables are defined in the previously failing cell. I also make the input path robust to Kaggle’s nested dataset folder (so reading CSVs works regardless of whether they’re at `/kaggle/input/train.csv` or inside the competition subfolder). Finally, I keep the core model/training exactly the same and ensure the script always writes a valid `subm.csv` with the required columns.'
- What this solution (achieved 0.59943) has done: 'I fix the TensorFlow import crash by pinning protobuf to the pure-Python runtime and forcing the C++ implementation off before TensorFlow is imported (this resolves the `MessageFactory.GetPrototype` error in TF2.18+protobuf6). Then, to move the score down toward the target (lower is better) without changing the model architecture or training loop, I apply the correct inverse transform at inference: since training optimizes MSE in log1p-space, predictions should be `expm1(raw_output)` (with a safety clip) rather than using the raw linear output directly. Finally, I make sure the submission uses the exact required columns and aligns ids correctly and always writes `subm.csv`.'
- What this solution (achieved 0.49394) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing protobuf to use the pure-Python implementation *before* any TensorFlow import and by clearing conflicting protobuf env vars, which is the direct root cause of the current runtime failure. I keep the model, loss, and training loop unchanged, and only adjust the environment setup so the graph code runs in TF 2.18 reliably. I also keep the already-correct log1p↔expm1 inference handling (which legitimately improves RMSLE) and ensure the submission is written as `subm.csv` with the exact required columns and test id alignment.'
- What this solution (achieved 0.4982) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation *and* ensuring TensorFlow is imported only after that environment setup (this is the direct cause of the `MessageFactory.GetPrototype` error). I keep the model, loss, and training loop identical, and keep the already-correct log1p/expm1 handling at inference (this is score-relevant for RMSLE and should continue to move the score down toward your target). I also make the input path resolution slightly more robust so it always finds the competition folder, and ensure we always write a valid `subm.csv` with the exact required columns and id alignment.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from subprocess import check_output

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_CPLUSPLUS", None)

input_dir = "../input" if os.path.exists("../input") else "/kaggle/input"
print(check_output(["ls", input_dir]).decode("utf8"))



## === cell 1
comp_name = "nomad2018-predict-transparent-conductors"

candidates = [
    os.path.join(input_dir, comp_name),
    input_dir,
]
comp_dir = next(
    (p for p in candidates if os.path.exists(os.path.join(p, "train.csv"))), None
)

if comp_dir is None:
    try:
        for name in os.listdir(input_dir):
            p = os.path.join(input_dir, name)
            if (
                os.path.isdir(p)
                and os.path.exists(os.path.join(p, "train.csv"))
                and os.path.exists(os.path.join(p, "test.csv"))
            ):
                comp_dir = p
                break
    except Exception:
        comp_dir = None

if comp_dir is None:
    raise FileNotFoundError(
        f"Could not find train.csv under candidates={candidates}. Contents of input_dir:\n"
        + check_output(["ls", "-la", input_dir]).decode("utf8")
    )

print("Using comp_dir:", comp_dir)



## === cell 2
train_path = os.path.join(comp_dir, "train.csv")
train = pd.read_csv(train_path)
train.head()



## === cell 3
train.describe()



## === cell 4
test_path = os.path.join(comp_dir, "test.csv")
test = pd.read_csv(test_path)
test.head()



## === cell 5
test.describe()



## === cell 6
train.loc[192]



## === cell 7
geom_candidates = [
    os.path.join(comp_dir, "train", "193", "geometry.xyz"),
    os.path.join(comp_dir, comp_name, "train", "193", "geometry.xyz"),
    os.path.join(input_dir, "train", "193", "geometry.xyz"),
    os.path.join(input_dir, comp_name, "train", "193", "geometry.xyz"),
]
geom_path = next((p for p in geom_candidates if os.path.exists(p)), None)
if geom_path is None:
    raise FileNotFoundError(
        f"Could not find geometry.xyz for id=193. Tried: {geom_candidates}"
    )

with open(geom_path, "r") as f:
    print(f.read())



## === cell 8
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

tf.compat.v1.disable_eager_execution()

tf.random.set_seed(1)
np.random.seed(1)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 9
t1 = "formation_energy_ev_natom"
t2 = "bandgap_energy_ev"

X_train_df, X_validation_df = train_test_split(train, test_size=0.3, random_state=1)

y1_train = X_train_df[t1].to_numpy(dtype=np.float32)[:, np.newaxis]
y2_train = X_train_df[t2].to_numpy(dtype=np.float32)[:, np.newaxis]
X_train_df = X_train_df.drop(["id", t1, t2], axis=1)

y1_validation = X_validation_df[t1].to_numpy(dtype=np.float32)[:, np.newaxis]
y2_validation = X_validation_df[t2].to_numpy(dtype=np.float32)[:, np.newaxis]
X_validation_df = X_validation_df.drop(["id", t1, t2], axis=1)

X_train = X_train_df.to_numpy(dtype=np.float32)
X_validation = X_validation_df.to_numpy(dtype=np.float32)

print(X_train.shape, y1_train.shape, y2_train.shape)
print(X_validation.shape, y1_validation.shape, y2_validation.shape)



## === cell 10
plt.subplot(1, 2, 1)
plt.scatter(range(len(y1_train)), y1_train)

plt.subplot(1, 2, 2)
plt.scatter(range(len(y2_train)), y2_train)

plt.show()



## === cell 11
tf_x = tf.compat.v1.placeholder(tf.float32, (None, X_train.shape[1]))
tf_y1 = tf.compat.v1.placeholder(tf.float32, (None, 1))
tf_y2 = tf.compat.v1.placeholder(tf.float32, (None, 1))



## === cell 12
Dense = tf.compat.v1.keras.layers.Dense

l1_y1 = Dense(14, activation=tf.nn.sigmoid, name="y1_dense1")(tf_x)
l2_y1 = Dense(14, activation=tf.nn.sigmoid, name="y1_dense2")(l1_y1)
output_y1 = Dense(1, activation=None, name="y1_out")(l2_y1)

l1_y2 = Dense(14, activation=tf.nn.sigmoid, name="y2_dense1")(tf_x)
l2_y2 = Dense(14, activation=tf.nn.sigmoid, name="y2_dense2")(l1_y2)
output_y2 = Dense(1, activation=None, name="y2_out")(l2_y2)



## === cell 13
output_y1_safe = tf.maximum(output_y1, 0.0)
output_y2_safe = tf.maximum(output_y2, 0.0)

loss_y1 = tf.compat.v1.losses.mean_squared_error(
    labels=tf.math.log1p(tf_y1), predictions=tf.math.log1p(output_y1_safe)
)
loss_y2 = tf.compat.v1.losses.mean_squared_error(
    labels=tf.math.log1p(tf_y2), predictions=tf.math.log1p(output_y2_safe)
)
loss = (loss_y1 + loss_y2) / 2.0

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(loss)



## === cell 14
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())



## === cell 15
for step in range(1000):
    _, l = sess.run([train_op, loss], {tf_x: X_train, tf_y1: y1_train, tf_y2: y2_train})
    if step % 20 == 0:
        print(l)



## === cell 16
val_loss, pred_y1_raw, pred_y2_raw = sess.run(
    [loss, output_y1_safe, output_y2_safe],
    {tf_x: X_validation, tf_y1: y1_validation, tf_y2: y2_validation},
)

print(val_loss)

pred_y1 = np.expm1(np.clip(pred_y1_raw, 0.0, None))
pred_y2 = np.expm1(np.clip(pred_y2_raw, 0.0, None))

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



## === cell 17
sample_path = os.path.join(comp_dir, "sample_submission.csv")
sample = pd.read_csv(sample_path)
sample.head()



## === cell 18
X_test = test.drop(["id"], axis=1).to_numpy(dtype=np.float32)

pred_y1_raw, pred_y2_raw = sess.run([output_y1_safe, output_y2_safe], {tf_x: X_test})

pred_y1 = np.expm1(np.clip(pred_y1_raw.reshape(-1), 0.0, None))
pred_y2 = np.expm1(np.clip(pred_y2_raw.reshape(-1), 0.0, None))

print(sample["id"].shape)
print(pred_y1.shape)
print(pred_y2.shape)

subm = pd.DataFrame(
    {
        "id": sample["id"].to_numpy(),
        "formation_energy_ev_natom": pred_y1,
        "bandgap_energy_ev": pred_y2,
    }
)

subm.to_csv("subm.csv", index=False)
print("Wrote submission to subm.csv")
print(subm.head())
