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
Predict a patient’s severity of decline in lung function based on a CT scan of their lungs. Lung function is assessed based on output from a spirometer, which measures the forced vital capacity (`FVC`), i.e. the volume of air exhaled.

## Metric
A modified version of the Laplace Log Likelihood. 

For each true FVC measurement, you will predict both an FVC and a confidence measure (standard deviation 𝜎𝜎). The metric is computed as:

$$
\begin{gathered}
\sigma_{\text {clipped }}=\max (\sigma, 70), \\
\Delta=\min \left(\left|F V C_{\text {true }}-F V C_{\text {predicted }}\right|, 1000\right), \\
\text { metric }=-\frac{\sqrt{2} \Delta}{\sigma_{\text {clipped }}}-\ln \left(\sqrt{2} \sigma_{\text {clipped }}\right) .
\end{gathered}
$$

The error is thresholded at 1000 ml to avoid large errors adversely penalizing results, while the confidence values are clipped at 70 ml to reflect the approximate measurement uncertainty in FVC. The final score is calculated by averaging the metric across all test set `Patient_Week`s (three per patient). 

Metric values will be negative and higher is better.

## Submission Format
For each `Patient_Week`, you must predict the `FVC` and a confidence. You are asked to predict every patient's `FVC` measurement for every possible week. Those weeks which are not in the final three visits are ignored in scoring.

The file should contain a header and have the following format:

```
Patient_Week,FVC,Confidence
ID00002637202176704235138_1,2000,100
ID00002637202176704235138_2,2000,100
ID00002637202176704235138_3,2000,100
etc.

```

## Dataset
In the dataset, you are provided with a baseline chest CT scan and associated clinical information for a set of patients. A patient has an image acquired at time `Week = 0` and has numerous follow up visits over the course of approximately 1-2 years, at which time their `FVC` is measured.

- In the training set, you are provided with an anonymized, baseline CT scan and the entire history of FVC measurements.
- In the test set, you are provided with a baseline CT scan and only the initial FVC measurement. **You are asked to predict the final three `FVC` measurements for each patient, as well as a confidence value in your prediction.**

- **train.csv** - the training set, contains full history of clinical information
- **test.csv** - the test set, contains only the baseline measurement
- **train/** - contains the training patients' baseline CT scan in DICOM format
- **test/** - contains the test patients' baseline CT scan in DICOM format
- **sample_submission.csv** - demonstrates the submission format

**train.csv and test.csv**

- `Patient`a unique Id for each patient (also the name of the patient's DICOM folder)
- `Weeks`the relative number of weeks pre/post the baseline CT (may be negative)
- `FVC` - the recorded lung capacity in ml
- `Percent`a computed field which approximates the patient's FVC as a percent of the typical FVC for a person of similar characteristics
- `Age`
- `Sex`
- `SmokingStatus`

**sample submission.csv**

- `Patient_Week` - a unique Id formed by concatenating the `Patient` and `Weeks` columns (i.e. ABC_22 is a prediction for patient ABC at week 22)
- `FVC` - the predicted FVC in ml
- `Confidence` - a confidence value of your prediction (also has units of ml)

# 2. Python version

3.8

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        input/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        working/
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
```

-> data/osic-pulmonary-fibrosis-progression/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/osic-pulmonary-fibrosis-progression/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/osic-pulmonary-fibrosis-progression/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> (stopped after 10 files for performance)

# 5. Target score

-6.944662466663177

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
import tensorflow.keras.backend as K
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

SEED = 24
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF decide
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
EPOCHS = 5
NUM_IMAGES = 40  # was 140; 40 is the minimum depth that remains valid through all conv/pool layers

BATCH_SIZE = 8  # was 4

FOLDS = 5
IMAGE_DIM = (NUM_IMAGES, 60, 60)

COMP_DIR = "../input/osic-pulmonary-fibrosis-progression/"
TRAIN_PATH = "../input/osic-pulmonary-fibrosis-progression/train"
TEST_PATH = "../input/osic-pulmonary-fibrosis-progression/test"
SUB_PATH = "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"




## === cell 2
comp_dir = "../input/osic-pulmonary-fibrosis-progression"

train_data = pd.read_csv(os.path.join(comp_dir, "train.csv"))
sub = pd.read_csv(os.path.join(comp_dir, "sample_submission.csv"))
test_data = pd.read_csv(os.path.join(comp_dir, "test.csv"))

train_data = train_data.drop_duplicates(keep=False, subset=["Patient", "Weeks"])

train_data.head(), test_data.head(), sub.head()




## === cell 3
train_data_u = train_data.copy()
train_data_u = train_data_u.drop_duplicates(subset=["Patient"])
train_data_u = train_data_u.rename(
    columns={"Weeks": "Base_Week", "FVC": "Base_FVC", "Percent": "Base_Percent"}
)
train_data_u["Typical_FVC"] = (
    train_data_u.Base_FVC.values / train_data_u.Base_Percent.values
) * 100
train_data = train_data.merge(
    train_data_u.drop(["Age", "Sex", "SmokingStatus"], axis=1), on="Patient", how="left"
)

train_data.head()




## === cell 4
sub["Patient"] = sub["Patient_Week"].str.split("_").str[0]
sub["Weeks"] = sub["Patient_Week"].str.split("_").str[-1].astype(int)

sub = sub.drop(["Confidence"], axis=1)
sub = sub[["Patient", "Weeks", "Patient_Week"]]
sub.head()




## === cell 5
test_data = test_data.rename(
    columns={"Weeks": "Base_Week", "FVC": "Base_FVC", "Percent": "Base_Percent"}
)
test_data["Typical_FVC"] = (
    test_data.Base_FVC.values / test_data.Base_Percent.values
) * 100

sub = sub.merge(test_data, how="left", on="Patient")
sub.head()




## === cell 6
train_data["Type"] = "train"
sub["Type"] = "test"

data = pd.concat([train_data, sub], axis=0, ignore_index=True)

data.columns




## === cell 7
prediction_col = ["FVC"]
Continuos_cols = [
    "Weeks",
    "Base_Week",
    "Base_FVC",
    "Typical_FVC",
    "Age",
    "Percent",
    "Base_Percent",
]

Categorical_cols = ["Sex", "SmokingStatus"]




## === cell 8
for c in Continuos_cols:
    if c not in data.columns:
        data[c] = np.nan
data[Continuos_cols] = data[Continuos_cols].fillna(
    data[Continuos_cols].median(numeric_only=True)
)

scaler = MinMaxScaler()
data[Continuos_cols] = scaler.fit_transform(data[Continuos_cols])

data["Sex"] = data["Sex"].fillna("Male")
data["SmokingStatus"] = data["SmokingStatus"].fillna("Never smoked")

data.head()




## === cell 9
data["Sex"] = data["Sex"].map({"Male": 0, "Female": 1}).fillna(0).astype(np.float32)

smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}
data["SmokingStatus"] = (
    data["SmokingStatus"].map(smoke_map).fillna(1).astype(np.float32)
)

data[["Sex", "SmokingStatus"]].head()




## === cell 10
x_cols = ["Weeks", "Base_Week", "Base_FVC", "Sex", "Age"]

train_mask = data["Type"].values == "train"
test_mask = ~train_mask

x_train = data.loc[train_mask, x_cols].to_numpy(dtype=np.float32, copy=True)
y_train = data.loc[train_mask, prediction_col].to_numpy(dtype=np.float32, copy=True)
x_test = data.loc[test_mask, x_cols].to_numpy(dtype=np.float32, copy=True)

patient_ids_train = data.loc[train_mask, "Patient"].to_numpy(copy=True)
patient_ids_test = data.loc[test_mask, "Patient"].to_numpy(copy=True)

x_train.shape, y_train.shape, x_test.shape, len(patient_ids_test)




## === cell 11
_ZERO_VOL_CACHE = np.zeros((*IMAGE_DIM, 1), dtype=np.float32)
_ZERO_VOL_TF = tf.constant(_ZERO_VOL_CACHE, dtype=tf.float32)


def load_patient_volume_stub(patient_id: str, dim=IMAGE_DIM, train=True):
    if dim == IMAGE_DIM:
        return _ZERO_VOL_CACHE
    return np.zeros((dim[0], dim[1], dim[2], 1), dtype=np.float32)




## === cell 12
def make_dataset(tab_data, target=None, batch_size=8, training=False, seed=SEED):
    tab_tensor = tf.convert_to_tensor(tab_data, dtype=tf.float32)
    if target is None:
        ds = tf.data.Dataset.from_tensor_slices(tab_tensor)

        if training:
            ds = ds.shuffle(
                buffer_size=int(tab_tensor.shape[0]),
                seed=seed,
                reshuffle_each_iteration=True,
            )

        def _map_x(x_tab):
            return (_ZERO_VOL_TF, x_tab)

        ds = ds.map(_map_x, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    else:
        y_tensor = tf.convert_to_tensor(target, dtype=tf.float32)
        ds = tf.data.Dataset.from_tensor_slices((tab_tensor, y_tensor))

        if training:
            ds = ds.shuffle(
                buffer_size=int(tab_tensor.shape[0]),
                seed=seed,
                reshuffle_each_iteration=True,
            )

        def _map_xy(x_tab, y):
            return (_ZERO_VOL_TF, x_tab), y

        ds = ds.map(_map_xy, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds




## === cell 13
C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")


def score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]
    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt(tf.cast(2.0, dtype=tf.float32))
    metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
    return K.mean(metric)


def qloss(y_true, y_pred):
    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q * e, (q - 1) * e)
    return K.mean(v)


def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)

    return loss




## === cell 14
def build_model():
    inp2 = L.Input(shape=(*IMAGE_DIM, 1), name="conv_input")

    x = L.Conv3D(32, 3, activation="relu", data_format="channels_last")(inp2)
    x = L.Conv3D(64, 3, activation="relu", data_format="channels_last")(x)
    x = L.BatchNormalization()(x)
    x1 = L.MaxPooling3D()(x)

    x = L.Conv3D(64, 3, activation="relu", padding="same", data_format="channels_last")(
        x1
    )
    x = L.Conv3D(64, 3, activation="relu", padding="same", data_format="channels_last")(
        x
    )
    x = L.add([x, x1])

    x = L.Conv3D(128, 5, activation="relu", data_format="channels_last")(x)
    x = L.Conv3D(128, 5, activation="relu", data_format="channels_last")(x)
    x = L.BatchNormalization()(x)
    x2 = L.MaxPooling3D()(x)

    x = L.Conv3D(
        128, 5, activation="relu", padding="same", data_format="channels_last"
    )(x2)
    x = L.Conv3D(
        128, 5, activation="relu", padding="same", data_format="channels_last"
    )(x)
    x = L.add([x, x2])

    x = L.BatchNormalization()(x)
    x = L.MaxPooling3D()(x)
    x = L.Dropout(0.5)(x)

    op1 = L.Flatten()(x)

    inp = L.Input((5,), name="input_d")
    d = L.Dense(128, activation="relu", name="dense")(inp)
    d1 = L.Dense(128, activation="relu", name="dense_1")(d)
    op2 = L.Dense(64, activation="relu", name="dense_2")(d1)

    x = L.Concatenate()([op1, op2])

    o1 = L.Dense(3, activation="linear", name="dense_f1")(x)
    o2 = L.Dense(3, activation="relu", name="dense_f2")(x)

    pred1 = L.Lambda(lambda z: (z[0] + tf.cumsum(z[1], axis=1)), name="output")(
        [o1, o2]
    )

    model = M.Model(inputs=[inp2, inp], outputs=pred1)

    model.compile(
        loss=mloss(0.8),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=[score],
        run_eagerly=False,
        steps_per_execution=16,
    )
    return model


model = build_model()
model.summary()




## === cell 15
def get_lr_callback():
    lr_start = 0.001
    lr_min = 0.0001
    lr_ramp_ep = 5
    lr_sus_ep = 0

    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = lr_start - (lr_start - lr_min) / lr_ramp_ep * epoch
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_min
        else:
            lr = lr_min
        return lr

    return tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)




## === cell 16
def build_fast_wrapper_from_model(full_model: tf.keras.Model):
    flatten_layers = [l for l in full_model.layers if isinstance(l, L.Flatten)]
    if len(flatten_layers) != 1:
        raise RuntimeError(
            f"Expected exactly 1 Flatten layer, found {len(flatten_layers)}"
        )
    op1_tensor = flatten_layers[0].output
    img_inp = full_model.get_layer("conv_input").input

    img_feat_model = tf.keras.Model(inputs=img_inp, outputs=op1_tensor)

    const_feat = img_feat_model(
        tf.expand_dims(_ZERO_VOL_TF, axis=0), training=False
    )  # (1, F)
    const_feat = tf.stop_gradient(const_feat)

    tab_inp = full_model.get_layer("input_d").input

    dense_f1 = full_model.get_layer("dense_f1")
    dense_f2 = full_model.get_layer("dense_f2")
    out_layer = full_model.get_layer("output")

    d = full_model.get_layer("dense")(tab_inp)
    d1 = full_model.get_layer("dense_1")(d)
    op2 = full_model.get_layer("dense_2")(d1)

    def tile_const(z):
        bs = tf.shape(z)[0]
        return tf.tile(const_feat, [bs, 1])

    op1_batched = L.Lambda(tile_const, name="const_img_feat")(tab_inp)
    x = L.Concatenate(name="concat_fast")([op1_batched, op2])

    o1 = dense_f1(x)
    o2 = dense_f2(x)
    pred = out_layer([o1, o2])

    fast_model = tf.keras.Model(inputs=tab_inp, outputs=pred, name="fast_model")

    fast_model.compile(
        loss=full_model.loss,
        optimizer=full_model.optimizer.__class__.from_config(
            full_model.optimizer.get_config()
        ),
        metrics=[score],
        run_eagerly=False,
        steps_per_execution=full_model.steps_per_execution,
    )
    return fast_model


fast_model = build_fast_wrapper_from_model(model)
fast_model.summary()




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/88661767.py in <cell line: 0>()
     69 
     70 
---> 71 fast_model = build_fast_wrapper_from_model(model)
     72 fast_model.summary()
     73 

/tmp/ipykernel_11/88661767.py in build_fast_wrapper_from_model(full_model)
     22     img_inp = full_model.get_layer("conv_input").input
     23 
---> 24     img_feat_model = tf.keras.Model(inputs=img_inp, outputs=op1_tensor)
     25 
     26     # Cache constant feature from all-zero image (training=False to avoid Dropout randomness)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/tracking.py in wrapper(*args, **kwargs)
     24     def wrapper(*args, **kwargs):
     25         with DotNotTrackScope():
---> 26             return fn(*args, **kwargs)
     27 
     28     return wrapper

/usr/local/lib/python3.11/dist-packages/keras/src/models/functional.py in __init__(self, inputs, outputs, name, **kwargs)
    133             inputs, outputs = clone_graph_nodes(inputs, outputs)
    134 
--> 135         Function.__init__(self, inputs, outputs, name=name)
    136 
    137         if trainable is not None:

/usr/local/lib/python3.11/dist-packages/keras/src/ops/function.py in __init__(self, inputs, outputs, name)
     60         self._outputs = tree.flatten(outputs)
     61         if not self._inputs:
---> 62             raise ValueError(
     63                 "`inputs` argument cannot be empty. Received:\n"
     64                 f"inputs={inputs}\n"

ValueError: `inputs` argument cannot be empty. Received:
inputs=[]
outputs=<KerasTensor shape=(None, 6400), dtype=float32, sparse=False, name=keras_tensor_17>

## === cell 17
KF = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

fold0 = next(iter(KF.split(patient_ids_train)))
t_idx, v_idx = fold0

tr_ds = tf.data.Dataset.from_tensor_slices(
    (
        tf.convert_to_tensor(x_train[t_idx], tf.float32),
        tf.convert_to_tensor(y_train[t_idx], tf.float32),
    )
)
tr_ds = tr_ds.shuffle(buffer_size=len(t_idx), seed=SEED, reshuffle_each_iteration=True)
tr_ds = tr_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

va_ds = tf.data.Dataset.from_tensor_slices(
    (
        tf.convert_to_tensor(x_train[v_idx], tf.float32),
        tf.convert_to_tensor(y_train[v_idx], tf.float32),
    )
)
va_ds = va_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

steps_per_epoch = int(np.ceil(len(t_idx) / BATCH_SIZE))
validation_steps = int(np.ceil(len(v_idx) / BATCH_SIZE))

history = fast_model.fit(
    tr_ds,
    epochs=EPOCHS,
    validation_data=va_ds,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
    callbacks=[get_lr_callback()],
)




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3541768358.py in <cell line: 0>()
     25 validation_steps = int(np.ceil(len(v_idx) / BATCH_SIZE))
     26 
---> 27 history = fast_model.fit(
     28     tr_ds,
     29     epochs=EPOCHS,

NameError: name 'fast_model' is not defined

## === cell 18
te_ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(x_test, tf.float32))
te_ds = te_ds.batch(8).prefetch(tf.data.AUTOTUNE)

pred = fast_model.predict(
    te_ds,
    verbose=1,
)
pred.shape




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4133073210.py in <cell line: 0>()
      3 te_ds = te_ds.batch(8).prefetch(tf.data.AUTOTUNE)
      4 
----> 5 pred = fast_model.predict(
      6     te_ds,
      7     verbose=1,

NameError: name 'fast_model' is not defined

## === cell 19
conf = pred[:, 2] - pred[:, 0]
conf = np.maximum(conf, 70.0)

pred_fvc = pred[:, 1]

pred_df = pd.DataFrame({"FVC": pred_fvc, "Confidence": conf})
pred_df.head()




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1831016863.py in <cell line: 0>()
----> 1 conf = pred[:, 2] - pred[:, 0]
      2 conf = np.maximum(conf, 70.0)
      3 
      4 pred_fvc = pred[:, 1]
      5 

NameError: name 'pred' is not defined

## === cell 20
sub_out = sub.copy()

if len(pred_df) != len(sub_out):
    raise RuntimeError(
        f"Prediction length {len(pred_df)} != submission rows {len(sub_out)}"
    )

sub_out["FVC"] = pred_df["FVC"].values
sub_out["Confidence"] = pred_df["Confidence"].values

subm = sub_out[["Patient_Week", "FVC", "Confidence"]].copy()

subm["FVC"] = subm["FVC"].astype(np.float32).fillna(0)
subm["Confidence"] = subm["Confidence"].astype(np.float32).fillna(70)

subm.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", subm.shape)
print(subm.head(10).to_string(index=False))

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4053779954.py in <cell line: 0>()
      1 sub_out = sub.copy()
      2 
----> 3 if len(pred_df) != len(sub_out):
      4     raise RuntimeError(
      5         f"Prediction length {len(pred_df)} != submission rows {len(sub_out)}"

NameError: name 'pred_df' is not defined
