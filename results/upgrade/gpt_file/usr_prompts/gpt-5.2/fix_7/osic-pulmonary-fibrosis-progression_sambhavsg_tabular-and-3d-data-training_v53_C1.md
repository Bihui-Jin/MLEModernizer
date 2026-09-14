# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

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
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass




## === cell 1
EPOCHS = 5
NUM_IMAGES = 40  # was 140; 40 is the minimum depth that remains valid through all conv/pool layers
BATCH_SIZE = 4
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

x_train = data.loc[data["Type"] == "train", x_cols].values.astype(np.float32)
y_train = data.loc[data["Type"] == "train", prediction_col].values.astype(np.float32)
x_test = data.loc[data["Type"] == "test", x_cols].values.astype(np.float32)

patient_ids_train = data.loc[data["Type"] == "train", "Patient"].values
patient_ids_test = data.loc[data["Type"] == "test", "Patient"].values

x_train.shape, y_train.shape, x_test.shape, len(patient_ids_test)




## === cell 11
_ZERO_VOL_CACHE = np.zeros((*IMAGE_DIM, 1), dtype=np.float32)


def load_patient_volume_stub(patient_id: str, dim=IMAGE_DIM, train=True):
    if dim == IMAGE_DIM:
        return _ZERO_VOL_CACHE
    return np.zeros((dim[0], dim[1], dim[2], 1), dtype=np.float32)




## === cell 12
class Data_Generator(tf.keras.utils.Sequence):
    def __init__(
        self,
        batch_size,
        patient_ids,
        tab_data,
        dim,
        target=None,
        train=True,
        augment=False,
    ):
        self.batch_size = batch_size
        self.image_ids = np.asarray(patient_ids)
        self.augment = augment
        self.dim = dim
        self.target = target
        self.indices = np.arange(len(self.image_ids))
        self.train = train
        self.tab_data = np.asarray(tab_data, dtype=np.float32)
        if self.target is not None:
            self.target = np.asarray(self.target, dtype=np.float32)
        self.on_epoch_end()

    def getimage(self, image_id):
        return load_patient_volume_stub(image_id, dim=self.dim, train=self.train)

    def on_epoch_end(self):
        if self.train:
            np.random.shuffle(self.indices)

    def getdata(self, image_id_list):
        if self.dim == IMAGE_DIM:
            return np.broadcast_to(_ZERO_VOL_CACHE, (len(image_id_list), *self.dim, 1))
        X = np.empty((len(image_id_list), *self.dim, 1), dtype=np.float32)
        for i, im_id in enumerate(image_id_list):
            X[i] = self.getimage(im_id)
        return X

    def __getitem__(self, index):
        idx = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        image_id_list = self.image_ids[idx]
        tab_X = self.tab_data[idx]

        X = self.getdata(image_id_list)

        if self.target is not None:
            y = self.target[idx]
            return (X, tab_X), y
        return (X, tab_X)

    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))




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
KF = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

fold0 = next(iter(KF.split(patient_ids_train)))
t_idx, v_idx = fold0

tr_gen = Data_Generator(
    batch_size=BATCH_SIZE,
    patient_ids=patient_ids_train[t_idx],
    tab_data=x_train[t_idx],
    dim=IMAGE_DIM,
    target=y_train[t_idx],
    train=True,
    augment=False,
)
va_gen = Data_Generator(
    batch_size=BATCH_SIZE,
    patient_ids=patient_ids_train[v_idx],
    tab_data=x_train[v_idx],
    dim=IMAGE_DIM,
    target=y_train[v_idx],
    train=False,  # kept as-is; generator now returns y whenever target is not None
    augment=False,
)

history = model.fit(
    tr_gen,
    epochs=EPOCHS,
    validation_data=va_gen,
    verbose=1,
    callbacks=[get_lr_callback()],
)




## === cell 17
te_gen = Data_Generator(
    batch_size=2,
    patient_ids=patient_ids_test,
    tab_data=x_test,
    dim=IMAGE_DIM,
    target=None,
    train=False,
    augment=False,
)

pred = model.predict(te_gen, verbose=1)
pred.shape




## === cell 18
conf = pred[:, 2] - pred[:, 0]
conf = np.maximum(conf, 70.0)

pred_fvc = pred[:, 1]

pred_df = pd.DataFrame({"FVC": pred_fvc, "Confidence": conf})
pred_df.head()




## === cell 19
sub_out = sub.copy()
sub_out["FVC"] = pred_df["FVC"].values
sub_out["Confidence"] = pred_df["Confidence"].values

subm = sub_out[["Patient_Week", "FVC", "Confidence"]].copy()

subm["FVC"] = subm["FVC"].astype(np.float32).fillna(0)
subm["Confidence"] = subm["Confidence"].astype(np.float32).fillna(70)

subm.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", subm.shape)
print(subm.head(10).to_string(index=False))
