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

-6.913230262117238

# 6. Current score

-23.33078

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -23.33078) has done: 'I fix the environment/runtime failures that prevent any submission from being produced: (1) remove the protobuf/pydicom import crash by guarding `pydicom` usage, and (2) avoid JPEG-lossless DICOM decompression dependency errors by falling back to a safe “no-CT” mode that uses zero volumes when pixel decoding isn’t available. I also fix the missing `Adam` import so model training runs. Finally, I ensure inference and submission generation always run end-to-end and write a valid `submission.csv` with the required columns, keeping the core model/training loop intact.'
- What this solution (achieved -23.33078) has done: 'The main timeout driver is repeated, heavy Python-side work inside the Keras `Sequence` (`df.loc` per batch, per-row feature extraction loops, and repeated stacking of volumes), compounded across multiple epochs and models. I keep the same model, loss, epochs, batch size, and ensemble size, but precompute tabular feature/label arrays once, replace pandas indexing with fast NumPy indexing, and avoid per-batch Python loops where possible. I also enable TensorFlow data pipeline prefetching for `Sequence` and reduce overhead from checkpoint I/O by saving weights only at the end while keeping the same “best weights” semantics via an in-memory best-weight callback (equivalent selection criterion, no disk writes). These changes preserve evaluation semantics and accuracy while significantly reducing per-step overhead.'
- What this solution (achieved -23.33078) has done: 'Main bottlenecks are (1) the heavy 3D CNN training repeated `N_MODELS` times and (2) slow Python-side data feeding in `Sequence.__getitem__` (per-batch list comprehensions, repeated normalization/division, and single-threaded loading). To keep core logic identical while cutting runtime, the script below caches per-row image tensors once (so batches become fast array indexing), enables Keras multi-worker prefetching for the `Sequence`, and uses vectorized stacking rather than Python loops. It also avoids expensive `model.get_weights()` copies every epoch by switching the callback to checkpoint the best weights to disk (same semantics: restore best-on-val-loss), and it removes unnecessary repeated pandas slicing inside the generator by always using the prebuilt numpy arrays. No changes are made to model architecture, loss, training epochs, or prediction logic.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import gc
import random
import numpy as np
import pandas as pd

import cv2

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.utils import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

dicom = None
DICOM_AVAILABLE = False
print("INFO: Running with DICOM_AVAILABLE=False (no-CT mode) for stability.")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
MIN_MAX = {
    "Weeks": (-5.0, 133.0),
    "FVC": (827.0, 6399.0),
    "Percent": (28.877577, 153.145378),
    "Age": (49.0, 88.0),
    "typical_fvc": (827.0, 6399.0),
}




## === cell 2
IMG_SIZE = 128
NUM_OF_SCANS = 10
BATCH_SIZE = 8  # keep memory safe; does not change core logic
EPOCHS = 8  # minimal training to produce usable weights within time
N_MODELS = 3  # lightweight ensemble to stabilize predictions

DATA_DIR = "../input/osic-pulmonary-fibrosis-progression"

TRAIN_DF = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
TEST_DF = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
SAMPLE_SUBMISSION = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

training_features = [
    "Weeks",
    "min_week",
    "min_week_FVC",
    "typical_fvc",
    "Age",
    "Male",
    "Female",
    "Never smoked",
    "Currently smokes",
    "Ex-smoker",
]
num_of_features = len(training_features)

TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test")




## === cell 3
def create_typical_fvc(df):
    df = df.copy()
    df["typical_fvc"] = df["FVC"] / df["Percent"] * 100.0
    return df


def normalize(df):
    df = df.copy()
    for feature, min_max in MIN_MAX.items():
        if feature in df.columns:
            df[feature] = (df[feature] - min_max[0]) / (min_max[1] - min_max[0])
    return df


def add_onehot(df):
    df = df.copy()
    df["Never smoked"] = (df["SmokingStatus"] == "Never smoked").astype("uint8")
    df["Currently smokes"] = (df["SmokingStatus"] == "Currently smokes").astype("uint8")
    df["Ex-smoker"] = (df["SmokingStatus"] == "Ex-smoker").astype("uint8")
    df["Male"] = (df["Sex"] == "Male").astype("uint8")
    df["Female"] = (df["Sex"] == "Female").astype("uint8")
    return df


TRAIN_DF = create_typical_fvc(TRAIN_DF)
TEST_DF = create_typical_fvc(TEST_DF)

TRAIN_DF = add_onehot(TRAIN_DF)
TEST_DF = add_onehot(TEST_DF)

for df in (TRAIN_DF, TEST_DF):
    g = df.groupby("Patient", sort=False)
    df["min_week"] = g["Weeks"].transform("min").astype(np.float32)
    idx = g["Weeks"].idxmin()
    min_fvc_map = df.loc[idx, ["Patient", "FVC"]].set_index("Patient")["FVC"]
    df["min_week_FVC"] = df["Patient"].map(min_fvc_map).astype(np.float32)

TRAIN_DF = normalize(TRAIN_DF)
TEST_DF = normalize(TEST_DF)

TRAIN_DF["FVC_norm"] = (TRAIN_DF["FVC"] - MIN_MAX["FVC"][0]) / (
    MIN_MAX["FVC"][1] - MIN_MAX["FVC"][0]
)
TRAIN_DF["y0"] = TRAIN_DF["FVC_norm"] - 0.05
TRAIN_DF["y1"] = TRAIN_DF["FVC_norm"]
TRAIN_DF["y2"] = TRAIN_DF["FVC_norm"] + 0.05

patients = TRAIN_DF["Patient"].unique()
train_pat, val_pat = train_test_split(patients, test_size=0.2, random_state=SEED)

train_idx = TRAIN_DF.index[TRAIN_DF["Patient"].isin(train_pat)].to_numpy()
val_idx = TRAIN_DF.index[TRAIN_DF["Patient"].isin(val_pat)].to_numpy()




## === cell 4
def get_pixels_hu(scan):
    image = scan.pixel_array
    image = image.astype(np.int16)

    slope = getattr(scan, "RescaleSlope", 1)
    intercept = getattr(scan, "RescaleIntercept", 0)
    window_center = -200
    window_width = 2000
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
    image += np.int16(intercept)

    image_min = window_center - window_width // 2
    image_max = window_center + window_width // 2
    image = np.clip(image, image_min, image_max).astype(np.float64)
    image = (image - image_min) / (image_max - image_min) * 255.0
    return image.astype(np.uint8)


def load_patient_volume(patient_id, folder, img_size=IMG_SIZE, num_scans=NUM_OF_SCANS):
    if not DICOM_AVAILABLE:
        return np.zeros((num_scans, img_size, img_size), dtype="float32")

    image_folder = os.path.join(folder, patient_id)
    try:
        image_files = np.asarray(
            sorted(os.listdir(image_folder), key=lambda x: int(os.path.splitext(x)[0]))
        )
        if len(image_files) == 0:
            return np.zeros((num_scans, img_size, img_size), dtype="float32")

        if len(image_files) >= num_scans:
            mid = len(image_files) // 2
            half = num_scans // 2
            image_files = image_files[mid - half : mid + half]

        scans = []
        for f in image_files:
            fp = os.path.join(image_folder, f)
            scans.append(dicom.dcmread(fp))

        imgs = []
        for s in scans:
            try:
                px = get_pixels_hu(s)
            except Exception:
                return np.zeros((num_scans, img_size, img_size), dtype="float32")
            imgs.append(cv2.resize(px, (img_size, img_size)))

        if len(imgs) < num_scans:
            if len(imgs) == 0:
                imgs = [np.zeros((img_size, img_size), dtype=np.uint8)]
            while len(imgs) < num_scans:
                imgs = imgs + [imgs[-1]]
            imgs = imgs[:num_scans]
        return np.asarray(imgs, dtype="float32")
    except Exception:
        return np.zeros((num_scans, img_size, img_size), dtype="float32")


all_patients = sorted(
    set(TRAIN_DF["Patient"].unique()).union(set(TEST_DF["Patient"].unique()))
)
volumes = {}
for pid in all_patients:
    folder = (
        TRAIN_IMG_DIR
        if os.path.isdir(os.path.join(TRAIN_IMG_DIR, pid))
        else TEST_IMG_DIR
    )
    volumes[pid] = load_patient_volume(pid, folder)

gc.collect()
print(
    "Prepared volumes for patients:", len(volumes), "DICOM_AVAILABLE:", DICOM_AVAILABLE
)




## === cell 5
TRAIN_PATIENTS_ARR = TRAIN_DF["Patient"].to_numpy()
TRAIN_TAB_ARR = TRAIN_DF[training_features].astype("float32").to_numpy(copy=True)
TRAIN_Y_ARR = TRAIN_DF[["y0", "y1", "y2"]].astype("float32").to_numpy(copy=True)

_train_imgs = np.stack([volumes[pid] for pid in TRAIN_PATIENTS_ARR], axis=0).astype(
    np.float32
)
_train_imgs = (_train_imgs / 255.0)[..., None]  # (N, Z, H, W, 1)
TRAIN_IMG_ARR = _train_imgs
del _train_imgs
gc.collect()


class Dataset(Sequence):
    def __init__(
        self,
        df,
        indices,
        batch_size=BATCH_SIZE,
        mode=0,
        patients_arr=None,
        tab_arr=None,
        y_arr=None,
        img_arr=None,
    ):
        self.df = df
        self.indices = np.asarray(indices)
        self.batch_size = batch_size
        self.mode = mode  # 0 - Training/Val, 1 - Test

        self.patients_arr = patients_arr
        self.tab_arr = tab_arr
        self.y_arr = y_arr
        self.img_arr = img_arr

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def on_epoch_end(self):
        if self.mode == 0:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        batch_idx = self.indices[
            index * self.batch_size : (index + 1) * self.batch_size
        ]

        patient_ids = self.patients_arr[batch_idx]
        tabulars = self.tab_arr[batch_idx]

        if self.img_arr is not None:
            images = self.img_arr[batch_idx]
        else:
            images = np.stack([volumes[pid] for pid in patient_ids], axis=0).astype(
                np.float32
            )
            images = (images / 255.0)[..., None]

        if self.mode == 1:
            return (images, tabulars)

        y = self.y_arr[batch_idx]
        return (images, tabulars), y




## === cell 6
def swish(x):
    return x * K.sigmoid(x)


def conv_block(x, num_of_filters, num_of_layers=3):
    for i in range(num_of_layers):
        x = Conv3D(
            num_of_filters,
            kernel_size=(3, 3, 3),
            padding="same",
            kernel_initializer="he_uniform",
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    x = MaxPooling3D()(x)
    return x


def build_model():
    input_img = Input(shape=(NUM_OF_SCANS, IMG_SIZE, IMG_SIZE, 1))
    x = conv_block(input_img, 16, 4)
    x = conv_block(x, 32, 4)
    x = conv_block(x, 64, 4)
    input_latent = GlobalAveragePooling3D()(x)

    input_tabular = Input(shape=(num_of_features,))
    x = Concatenate()([input_latent, input_tabular])
    x = Dense(100)(x)
    x = Dense(100)(x)
    out = Dense(3)(x)

    model = tf.keras.Model(
        inputs=[input_img, input_tabular], outputs=out, name="tabular_model"
    )
    return model


def train_one_model(seed_offset=0):
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED + seed_offset)
    np.random.seed(SEED + seed_offset)

    model = build_model()
    model.compile(optimizer=Adam(1e-3), loss="mse")

    train_gen = Dataset(
        TRAIN_DF,
        train_idx,
        batch_size=BATCH_SIZE,
        mode=0,
        patients_arr=TRAIN_PATIENTS_ARR,
        tab_arr=TRAIN_TAB_ARR,
        y_arr=TRAIN_Y_ARR,
        img_arr=TRAIN_IMG_ARR,
    )
    val_gen = Dataset(
        TRAIN_DF,
        val_idx,
        batch_size=BATCH_SIZE,
        mode=0,
        patients_arr=TRAIN_PATIENTS_ARR,
        tab_arr=TRAIN_TAB_ARR,
        y_arr=TRAIN_Y_ARR,
        img_arr=TRAIN_IMG_ARR,
    )

    ckpt_path = f"best_weights_{seed_offset}.weights.h5"
    ckpt_cb = ModelCheckpoint(
        ckpt_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        verbose=0,
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[ckpt_cb],
        workers=max(1, (os.cpu_count() or 2) // 2),
        use_multiprocessing=True,
        max_queue_size=16,
    )
    if os.path.exists(ckpt_path):
        model.load_weights(ckpt_path)
    return model




## === cell 7
models = [train_one_model(i) for i in range(N_MODELS)]




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3761371558.py in <cell line: 0>()
----> 1 models = [train_one_model(i) for i in range(N_MODELS)]
      2 
      3 

/tmp/ipykernel_11/3761371558.py in <listcomp>(.0)
----> 1 models = [train_one_model(i) for i in range(N_MODELS)]
      2 
      3 

/tmp/ipykernel_11/2753041721.py in train_one_model(seed_offset)
     77     )
     78 
---> 79     model.fit(
     80         train_gen,
     81         validation_data=val_gen,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    117             return fn(*args, **kwargs)
    118         except Exception as e:
--> 119             filtered_tb = _process_traceback_frames(e.__traceback__)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`

TypeError: TensorFlowTrainer.fit() got an unexpected keyword argument 'workers'

## === cell 8
sub = SAMPLE_SUBMISSION.copy()
sub[["Patient", "Weeks_str"]] = sub["Patient_Week"].str.split("_", expand=True)
sub["Weeks"] = sub["Weeks_str"].astype("int32")
sub = sub.drop(columns=["Weeks_str"])

static_cols = [c for c in TEST_DF.columns if c in training_features and c != "Weeks"]
test_static = TEST_DF[["Patient"] + static_cols].drop_duplicates("Patient")
test_merge = sub.merge(test_static, on="Patient", how="left")

test_merge["Weeks"] = (test_merge["Weeks"].astype("float32") - MIN_MAX["Weeks"][0]) / (
    MIN_MAX["Weeks"][1] - MIN_MAX["Weeks"][0]
)

for col in training_features:
    if col not in test_merge.columns:
        test_merge[col] = 0.0

test_merge = test_merge.reset_index(drop=True)
TEST_PATIENTS_ARR = test_merge["Patient"].to_numpy()
TEST_TAB_ARR = test_merge[training_features].astype("float32").to_numpy(copy=False)

test_gen = Dataset(
    test_merge,
    np.arange(len(test_merge), dtype=np.int64),
    batch_size=BATCH_SIZE,
    mode=1,
    patients_arr=TEST_PATIENTS_ARR,
    tab_arr=TEST_TAB_ARR,
    y_arr=None,
    img_arr=None,
)

predictions = np.zeros((len(test_merge), 3), dtype="float32")
for model in models:
    predictions += model.predict(
        test_gen,
        verbose=1,
        workers=max(1, (os.cpu_count() or 2) // 2),
        use_multiprocessing=True,
        max_queue_size=16,
    )
predictions = predictions / len(models)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/870849641.py in <cell line: 0>()
     32 
     33 predictions = np.zeros((len(test_merge), 3), dtype="float32")
---> 34 for model in models:
     35     # Speed: predict with prefetching too.
     36     predictions += model.predict(

NameError: name 'models' is not defined

## === cell 9
def denormalize_fvc(y):
    return y * (MIN_MAX["FVC"][1] - MIN_MAX["FVC"][0]) + MIN_MAX["FVC"][0]


pred_denorm = denormalize_fvc(predictions)

FVC = pred_denorm[:, 1]
Confidence = pred_denorm[:, 2] - pred_denorm[:, 0]
Confidence = np.maximum(Confidence, 70.0)

out = SAMPLE_SUBMISSION.copy()
out["FVC"] = np.round(FVC).astype(int)
out["Confidence"] = np.round(Confidence).astype(int)
out.to_csv("submission.csv", index=False)

print(out.head())
print("Wrote submission.csv with shape:", out.shape)
print("Columns:", list(out.columns))
