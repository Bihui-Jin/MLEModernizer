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

0.60633

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 0.60633) has done: 'I fixed the TensorFlow 1‑style code that caused import and runtime errors by switching to TensorFlow 2’s Keras API, keeping the same network architecture (16‑unit tanh layers with batch‑normalisation, eight layers per target) and the original log‑transformed targets. The script now builds two separate Keras models, trains them, evaluates validation loss, and writes a correct `subm.csv` submission file with the required columns.'

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
assert np.all(
    np.abs(
        train["percent_atom_al"]
        + train["percent_atom_ga"]
        + train["percent_atom_in"]
        - 1
    )
    <= 0.001
), "Composition check failed"




## === cell 5
train = train.drop(["spacegroup", "percent_atom_in"], axis=1)
test = test.drop(["spacegroup", "percent_atom_in"], axis=1)




## === cell 6
from sklearn.model_selection import train_test_split

t1 = "formation_energy_ev_natom"
t2 = "bandgap_energy_ev"

X_train_df, X_val_df = train_test_split(train, test_size=0.3, random_state=1)

y1_train = np.log1p(X_train_df[t1].values[:, np.newaxis])
y2_train = np.log1p(X_train_df[t2].values[:, np.newaxis])
X_train = X_train_df.drop(["id", t1, t2], axis=1)

y1_val = np.log1p(X_val_df[t1].values[:, np.newaxis])
y2_val = np.log1p(X_val_df[t2].values[:, np.newaxis])
X_val = X_val_df.drop(["id", t1, t2], axis=1)

X_train_np = X_train.values.astype(np.float32)
X_val_np = X_val.values.astype(np.float32)

print(
    "Shapes -> X_train:",
    X_train_np.shape,
    "y1_train:",
    y1_train.shape,
    "y2_train:",
    y2_train.shape,
)
print(
    "Shapes -> X_val:", X_val_np.shape, "y1_val:", y1_val.shape, "y2_val:", y2_val.shape
)




## === cell 7
import matplotlib.pyplot as plt

plt.subplot(1, 2, 1)
plt.scatter(range(len(y1_train)), y1_train, s=10)
plt.title("log1p formation energy")

plt.subplot(1, 2, 2)
plt.scatter(range(len(y2_train)), y2_train, s=10)
plt.title("log1p bandgap")
plt.show()




## === cell 8
import tensorflow as tf

tf.random.set_seed(1)
np.random.seed(1)

n_units = 16
n_layers = 8
activation = "tanh"




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 9
def build_model(input_dim):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    x = tf.keras.layers.BatchNormalization()(x)
    for i in range(n_layers):
        x = tf.keras.layers.Dense(n_units, activation=activation, name=f"dense_{i+1}")(
            x
        )
        x = tf.keras.layers.BatchNormalization(name=f"bn_{i+1}")(x)
    outputs = tf.keras.layers.Dense(1, name="output")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


model_y1 = build_model(X_train_np.shape[1])
model_y2 = build_model(X_train_np.shape[1])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.004)
model_y1.compile(optimizer=optimizer, loss="mse")
model_y2.compile(optimizer=optimizer, loss="mse")




## === cell 10
history_y1 = model_y1.fit(
    X_train_np,
    y1_train,
    validation_data=(X_val_np, y1_val),
    epochs=50,
    batch_size=32,
    verbose=0,
)

history_y2 = model_y2.fit(
    X_train_np,
    y2_train,
    validation_data=(X_val_np, y2_val),
    epochs=50,
    batch_size=32,
    verbose=0,
)

print("Training completed.")




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NotImplementedError                       Traceback (most recent call last)
/tmp/ipykernel_11/2618641116.py in <cell line: 0>()
      9 )
     10 
---> 11 history_y2 = model_y2.fit(
     12     X_train_np,
     13     y2_train,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/core.py in convert_to_numpy(x)
    153     elif isinstance(x, tf.RaggedTensor):
    154         x = x.to_tensor()
--> 155     return np.array(x)
    156 
    157 

NotImplementedError: numpy() is only available when eager execution is enabled.

## === cell 11
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_y1.history["loss"], label="train")
plt.plot(history_y1.history["val_loss"], label="val")
plt.title("Formation energy (MSE)")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_y2.history["loss"], label="train")
plt.plot(history_y2.history["val_loss"], label="val")
plt.title("Bandgap energy (MSE)")
plt.legend()
plt.show()




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2524709251.py in <cell line: 0>()
      8 
      9 plt.subplot(1, 2, 2)
---> 10 plt.plot(history_y2.history["loss"], label="train")
     11 plt.plot(history_y2.history["val_loss"], label="val")
     12 plt.title("Bandgap energy (MSE)")

NameError: name 'history_y2' is not defined

## === cell 12
val_pred_y1 = model_y1.predict(X_val_np)
val_pred_y2 = model_y2.predict(X_val_np)
rmsle_y1 = np.sqrt(np.mean((val_pred_y1 - y1_val) ** 2))
rmsle_y2 = np.sqrt(np.mean((val_pred_y2 - y2_val) ** 2))
print(f"Validation RMSLE -> formation: {rmsle_y1:.5f}, bandgap: {rmsle_y2:.5f}")




## === cell 13
X_test = test.drop(["id"], axis=1).values.astype(np.float32)

pred_y1_test_log = model_y1.predict(X_test)
pred_y2_test_log = model_y2.predict(X_test)

pred_y1_test = np.expm1(pred_y1_test_log)
pred_y2_test = np.expm1(pred_y2_test_log)

pred_y1_test[pred_y1_test < 0] = 0
pred_y2_test[pred_y2_test < 0] = 0

sample = pd.read_csv("../input/sample_submission.csv")
subm = pd.DataFrame(
    {
        "id": sample["id"],
        "formation_energy_ev_natom": pred_y1_test.ravel(),
        "bandgap_energy_ev": pred_y2_test.ravel(),
    }
)

subm.to_csv("subm.csv", index=False)
print("Submission file written to subm.csv")
