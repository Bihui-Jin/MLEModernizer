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

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
pydicom==3.0.1
scikit-image==0.25.2
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
tqdm==4.67.1

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

-24.7981

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from functools import partial
from functools import reduce

import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
for gpu_instance in physical_devices:
    try:
        tf.config.experimental.set_memory_growth(gpu_instance, True)
    except Exception:
        pass

from skimage.transform import resize
import pydicom

SEED = 1337
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
MIN_WEEK = -12
MAX_WEEK = 133
MAX_FVC = 4000
INPUT_ROOT = "/kaggle/input/osic-pulmonary-fibrosis-progression"

TRAIN_SCANS_ROOT = os.path.join(INPUT_ROOT, "train")
TEST_SCANS_ROOT = os.path.join(INPUT_ROOT, "test")
SCAN_DEPTH = 20
HEIGHT = 32
WIDTH = 32
LEARNING_RATE = 0.001

TRAIN_INPUT_FILE = os.path.join(INPUT_ROOT, "train.csv")
TEST_INPUT_FILE = os.path.join(INPUT_ROOT, "test.csv")
BATCH_SIZE = 192
SEQUENCE_LENGTH = 1

STEPS_PER_EPOCH = 1500 // BATCH_SIZE
VALIDATION_STEPS = 300 // BATCH_SIZE

EPOCHS = 50

TEST_OUTPUT = "submission.csv"




## === cell 2
_SCAN_CACHE = {}


def _read_dcm_pixels(path):
    dcm = pydicom.dcmread(
        path,
        force=True,
        specific_tags=[
            "RescaleSlope",
            "RescaleIntercept",
            "PixelData",
            "Rows",
            "Columns",
        ],
    )
    slope = float(getattr(dcm, "RescaleSlope", 1.0))
    intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
    arr = dcm.pixel_array  # triggers pixel decode
    return arr, slope, intercept


def get_patient_scan(patient_dir, n_depth=5, rows=64, columns=64):
    cache_key = (patient_dir, int(n_depth), int(rows), int(columns))
    cached = _SCAN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    patient_files = [e for e in os.listdir(patient_dir) if e.lower().endswith(".dcm")]
    patient_files.sort(key=lambda fname: int(fname.split(".")[0]))
    n_slices = len(patient_files)

    slice_group = n_depth / n_slices
    slice_indexes = [int(idx / slice_group) for idx in range(n_depth)]
    slice_indexes = [min(max(i, 0), n_slices - 1) for i in slice_indexes]

    shape = (n_depth, rows, columns)
    img = np.empty(shape, dtype="float32")

    for out_idx, slice_idx in enumerate(slice_indexes):
        f = patient_files[slice_idx]
        arr, slope, intercept = _read_dcm_pixels(os.path.join(patient_dir, f))
        resized_img = resize(
            arr.astype("float32"),
            (rows, columns),
            anti_aliasing=True,
            preserve_range=True,
        ).astype("float32", copy=False)
        img[out_idx, ...] = resized_img * slope + intercept

    img = np.transpose(img, (1, 2, 0))  # (rows, cols, depth)
    _SCAN_CACHE[cache_key] = img
    return img


def get_dicom_data(patients_root, n_depth=5, rows=64, columns=64):
    def gen(patients_root):
        for patient_dir in os.listdir(patients_root):
            patient_dir = os.path.join(patients_root, patient_dir)
            img = get_patient_scan(
                patient_dir, n_depth=n_depth, rows=rows, columns=columns
            )
            yield img

    return tf.data.Dataset.from_generator(
        partial(gen, patients_root),
        output_signature=tf.TensorSpec(
            shape=(rows, columns, n_depth), dtype=tf.float32
        ),
    )




## === cell 3
def laplace_log_likelihood(y_true, y_pred):
    uncertainty_clipped = tf.maximum(y_pred[:, 1:2] * 1000.0, 70.0)
    prediction = y_pred[:, :1]
    delta = tf.minimum(tf.abs(y_true - prediction), 1000.0)
    metric = -np.sqrt(2.0) * delta / uncertainty_clipped - tf.math.log(
        np.sqrt(2.0) * uncertainty_clipped
    )
    return tf.reduce_mean(metric)


def laplace_log_likelihood_loss(y_true, y_pred):
    uncertainty_clipped = tf.maximum(y_pred[:, 1:2] * 1000.0, 70.0)
    prediction = y_pred[:, :1]
    delta = tf.minimum(tf.abs(y_true - prediction), 1000.0)
    metric = -np.sqrt(2.0) * delta / uncertainty_clipped - tf.math.log(
        np.sqrt(2.0) * uncertainty_clipped
    )
    return -tf.reduce_mean(metric)




## === cell 4
def get3dcnn_model(width, height, depth):
    inputs = tf.keras.Input(shape=(width, height, depth, 1))
    x = tf.keras.layers.Conv3D(32, kernel_size=(5, 5, 5), activation="relu")(inputs)
    x = tf.keras.layers.MaxPool3D()(x)

    x = tf.keras.layers.Conv3D(64, kernel_size=(5, 5, 5), activation="relu")(x)
    x = tf.keras.layers.MaxPool3D()(x)

    return inputs, x


def get_combined_model(sequence_length, learning_rate, width, height, depth):
    rnn_inputs = tf.keras.Input(shape=(sequence_length, 2))
    x = tf.keras.layers.Masking(mask_value=-1, input_shape=(sequence_length, 1))(
        rnn_inputs
    )
    rnn_out = 4
    rnn_out = tf.keras.layers.GRU(rnn_out)(x)

    cnn3d_inputs, cnn3d_out = get3dcnn_model(width, height, depth)
    cnn3d_out_shape = reduce(lambda x, y: x * y, cnn3d_out.shape[1:])
    cnn3d_out = tf.keras.layers.Reshape((cnn3d_out_shape,))(cnn3d_out)

    combined_out = tf.keras.layers.concatenate([rnn_out, cnn3d_out])

    prediction_output = tf.keras.layers.Dense(1)(combined_out)
    uncertainty_output = tf.keras.layers.Dense(1, activation="sigmoid")(combined_out)

    outputs = tf.keras.layers.concatenate([prediction_output, uncertainty_output])

    model = tf.keras.Model(inputs=[rnn_inputs, cnn3d_inputs], outputs=outputs)

    metrics = [laplace_log_likelihood]
    model.compile(
        loss=laplace_log_likelihood_loss,
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        metrics=metrics,
    )
    return model




## === cell 5
def get_combined_data(
    input_file,
    batch_size,
    sequence_length,
    scans_root,
    n_depth,
    rows,
    columns,
    max_fvc,
    split=0.8,
):
    train_data = pd.read_csv(input_file)
    n_features = 0
    train_data["Weeks"] = train_data["Weeks"]
    n_features += 1
    train_data["FVC"] /= max_fvc
    n_features += 1

    unique_patients = train_data["Patient"].unique().tolist()

    for patient in tqdm(unique_patients, desc="Caching train scans"):
        patient_dir = os.path.join(scans_root, patient)
        _ = get_patient_scan(patient_dir, n_depth=n_depth, rows=rows, columns=columns)

    grouped = train_data.groupby("Patient", sort=False)

    patient_payloads = []
    for patient in unique_patients:
        pdf = grouped.get_group(patient)
        fvc = pdf["FVC"].to_numpy(dtype=np.float32, copy=False)
        weeks = pdf["Weeks"].to_numpy(dtype=np.float32, copy=False)

        if len(fvc) < 2:
            continue

        fvc_prev = fvc[:-1]
        weeks_prev = weeks[:-1]
        weeks_next = weeks[1:]
        week_diff = (weeks_next - weeks_prev) / float(MAX_WEEK - MIN_WEEK)  # float32

        patient_dir = os.path.join(scans_root, patient)
        img = np.expand_dims(
            get_patient_scan(patient_dir, n_depth=n_depth, rows=rows, columns=columns),
            axis=-1,
        ).astype(
            np.float32, copy=False
        )  # (rows, cols, depth, 1)

        patient_payloads.append(
            (fvc_prev, week_diff.astype(np.float32, copy=False), img)
        )

    n_samples = int(sum(len(p[0]) for p in patient_payloads))
    if n_samples == 0:
        raise RuntimeError("No training samples constructed.")

    def gen():
        for fvc_prev, week_diff, img in tqdm(patient_payloads, desc="Building samples"):
            for i in range(len(fvc_prev)):
                sequence = np.empty((sequence_length, n_features), dtype=np.float32)
                sequence[0, 0] = fvc_prev[i]
                sequence[0, 1] = week_diff[i]
                y = np.array(fvc_prev[i], dtype=np.float32)
                yield ((sequence, img), y)

    output_signature = (
        (
            tf.TensorSpec(shape=(sequence_length, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(rows, columns, n_depth, 1), dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    )

    shuffle_buffer = min(n_samples, 4096)

    dataset = (
        tf.data.Dataset.from_generator(gen, output_signature=output_signature)
        .repeat(None)
        .shuffle(shuffle_buffer, seed=SEED, reshuffle_each_iteration=True)
    )

    train_size = int(split * n_samples)
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)

    return (
        train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE),
        val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE),
    )




## === cell 6
train_dataset, val_dataset = get_combined_data(
    TRAIN_INPUT_FILE,
    BATCH_SIZE,
    SEQUENCE_LENGTH,
    TRAIN_SCANS_ROOT,
    SCAN_DEPTH,
    HEIGHT,
    WIDTH,
    max_fvc=MAX_FVC,
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1901371451.py in <cell line: 0>()
----> 1 train_dataset, val_dataset = get_combined_data(
      2     TRAIN_INPUT_FILE,
      3     BATCH_SIZE,
      4     SEQUENCE_LENGTH,
      5     TRAIN_SCANS_ROOT,

/tmp/ipykernel_11/1156266370.py in get_combined_data(input_file, batch_size, sequence_length, scans_root, n_depth, rows, columns, max_fvc, split)
     22     for patient in tqdm(unique_patients, desc="Caching train scans"):
     23         patient_dir = os.path.join(scans_root, patient)
---> 24         _ = get_patient_scan(patient_dir, n_depth=n_depth, rows=rows, columns=columns)
     25 
     26     # Speedup: precompute per-patient arrays once to avoid heavy per-sample Pandas operations

/tmp/ipykernel_11/190791394.py in get_patient_scan(patient_dir, n_depth, rows, columns)
     42     for out_idx, slice_idx in enumerate(slice_indexes):
     43         f = patient_files[slice_idx]
---> 44         arr, slope, intercept = _read_dcm_pixels(os.path.join(patient_dir, f))
     45         # Speedup: preserve_range avoids internal scaling; float32 is kept.
     46         resized_img = resize(

/tmp/ipykernel_11/190791394.py in _read_dcm_pixels(path)
     19     slope = float(getattr(dcm, "RescaleSlope", 1.0))
     20     intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
---> 21     arr = dcm.pixel_array  # triggers pixel decode
     22     return arr, slope, intercept
     23 

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in __getattr__(self, name)
    916             return {}
    917         # Try the base class attribute getter (fix for issue 332)
--> 918         return object.__getattribute__(self, name)
    919 
    920     @property

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in pixel_array(self)
   2191             that iterates through the image frames.
   2192         """
-> 2193         self.convert_pixel_data()
   2194         return cast("numpy.ndarray", self._pixel_array)
   2195 

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in convert_pixel_data(self, handler_name)
   1724             # Use 'pydicom.pixels' backend
   1725             opts["decoding_plugin"] = name
-> 1726             self._pixel_array = pixel_array(self, **opts)
   1727             self._pixel_id = get_image_pixel_ids(self)
   1728         else:

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/utils.py in pixel_array(src, ds_out, specific_tags, index, raw, decoding_plugin, **kwargs)
   1428 
   1429         opts = as_pixel_options(ds, **kwargs)
-> 1430         return decoder.as_array(
   1431             ds,
   1432             index=index,

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py in as_array(self, src, index, validate, raw, decoding_plugin, **kwargs)
    988 
    989         if validate:
--> 990             runner.validate()
    991 
    992         if self.is_native:

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py in validate(self)
    751     def validate(self) -> None:
    752         """Validate the decoding options and source buffer (if any)."""
--> 753         self._validate_options()
    754         if self.is_dataset or self.is_buffer:
    755             self._validate_buffer()

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py in _validate_options(self)
    820     def _validate_options(self) -> None:
    821         """Validate the supplied options to ensure they meet minimum requirements."""
--> 822         super()._validate_options()
    823 
    824         # The Extended Offset Table is optional

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/common.py in _validate_options(self)
    573         prefix = "Missing required element: (0028"
    574         if self._opts.get("bits_allocated") is None:
--> 575             raise AttributeError(f"{prefix},0100) 'Bits Allocated'")
    576 
    577         if not 1 <= self.bits_allocated <= 64 or (

AttributeError: Missing required element: (0028,0100) 'Bits Allocated'

## === cell 7
model = get_combined_model(SEQUENCE_LENGTH, LEARNING_RATE, WIDTH, HEIGHT, SCAN_DEPTH)




## === cell 8
model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2183802692.py in <cell line: 0>()
      1 model.fit(
----> 2     train_dataset,
      3     epochs=EPOCHS,
      4     validation_data=val_dataset,
      5     steps_per_epoch=STEPS_PER_EPOCH,

NameError: name 'train_dataset' is not defined

## === cell 9
def get_input_data(FVC, img, sequence, week_diff):
    converted_data = {"FVC": FVC, "week_diff": [week_diff]}
    converted_df = pd.DataFrame.from_dict(converted_data)
    sequence[0] = [converted_df["FVC"].loc[0], converted_df["week_diff"].loc[0]]
    data = (np.expand_dims(sequence, axis=0), np.expand_dims(img, axis=0))
    return data


test_data = pd.read_csv(TEST_INPUT_FILE)
n_features = 0
test_data["Weeks"] = test_data["Weeks"]
n_features += 1
test_data["FVC"] /= MAX_FVC
n_features += 1
grouped = test_data.groupby(test_data.Patient)

for patient in tqdm(test_data["Patient"].unique(), desc="Caching test scans"):
    patient_dir = os.path.join(TEST_SCANS_ROOT, patient)
    _ = get_patient_scan(patient_dir, n_depth=SCAN_DEPTH, rows=HEIGHT, columns=WIDTH)

prediction_data = {"Patient_Week": [], "FVC": [], "Confidence": []}
all_weeks = np.arange(MIN_WEEK, MAX_WEEK + 1, dtype=np.int32)

for patient in tqdm(test_data["Patient"].unique(), desc="Predicting"):
    patient_df = grouped.get_group(patient)
    FVC0 = float(patient_df["FVC"].iloc[0])
    measurement_week = int(patient_df["Weeks"].iloc[0])

    prediction_weeks = all_weeks[all_weeks != measurement_week]
    week_diffs = (
        prediction_weeks.astype(np.float32) - np.float32(measurement_week)
    ) / float(MAX_WEEK - MIN_WEEK)

    patient_dir = os.path.join(TEST_SCANS_ROOT, patient)
    img = np.expand_dims(
        get_patient_scan(patient_dir, n_depth=SCAN_DEPTH, rows=HEIGHT, columns=WIDTH),
        axis=-1,
    ).astype(
        np.float32, copy=False
    )  # (rows, cols, depth, 1)

    prediction_data["Patient_Week"].append(patient + "_" + str(measurement_week))
    prediction_data["FVC"].append(int(FVC0 * MAX_FVC))
    prediction_data["Confidence"].append(100)

    seq_batch = np.empty((len(prediction_weeks), 1, n_features), dtype=np.float32)
    current_fvc = np.float32(FVC0)
    for i, wd in enumerate(week_diffs):
        seq_batch[i, 0, 0] = current_fvc
        seq_batch[i, 0, 1] = np.float32(wd)
        pass

    chunk = 64
    current_fvc = np.float32(FVC0)
    i = 0
    while i < len(prediction_weeks):
        j = min(i + chunk, len(prediction_weeks))
        for k in range(i, j):
            seq_batch[k, 0, 0] = current_fvc
            seq_batch[k, 0, 1] = week_diffs[k]
        img_b = np.repeat(img[None, ...], j - i, axis=0)

        preds = model.predict_on_batch([seq_batch[i:j], img_b]).numpy()
        for k in range(i, j):
            pred_fvc = float(preds[k - i, 0])
            pred_unc = float(preds[k - i, 1])
            current_fvc = np.float32(pred_fvc)

            week = int(prediction_weeks[k])
            prediction_data["Patient_Week"].append(patient + "_" + str(week))
            prediction_data["FVC"].append(int(pred_fvc * MAX_FVC))
            prediction_data["Confidence"].append(pred_unc * 1000.0)

        i = j

indexes = list(range(len(prediction_data["Patient_Week"])))


def get_key(patient_week):
    patient, week = patient_week.split("_")
    return (patient, int(week))


sorted_data = sorted(
    zip(
        indexes,
        prediction_data["Patient_Week"],
        prediction_data["FVC"],
        prediction_data["Confidence"],
    ),
    key=lambda e: get_key(e[1]),
)
prediction_data["Patient_Week"] = [e[1] for e in sorted_data]
prediction_data["FVC"] = [e[2] for e in sorted_data]
prediction_data["Confidence"] = [e[3] for e in sorted_data]

df = pd.DataFrame.from_dict(prediction_data)
df.to_csv(TEST_OUTPUT, index=False)
print("Wrote:", TEST_OUTPUT, "rows:", len(df), "cols:", list(df.columns))
print(df.head())

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/409243489.py in <cell line: 0>()
     20 for patient in tqdm(test_data["Patient"].unique(), desc="Caching test scans"):
     21     patient_dir = os.path.join(TEST_SCANS_ROOT, patient)
---> 22     _ = get_patient_scan(patient_dir, n_depth=SCAN_DEPTH, rows=HEIGHT, columns=WIDTH)
     23 
     24 prediction_data = {"Patient_Week": [], "FVC": [], "Confidence": []}

/tmp/ipykernel_11/190791394.py in get_patient_scan(patient_dir, n_depth, rows, columns)
     42     for out_idx, slice_idx in enumerate(slice_indexes):
     43         f = patient_files[slice_idx]
---> 44         arr, slope, intercept = _read_dcm_pixels(os.path.join(patient_dir, f))
     45         # Speedup: preserve_range avoids internal scaling; float32 is kept.
     46         resized_img = resize(

/tmp/ipykernel_11/190791394.py in _read_dcm_pixels(path)
     19     slope = float(getattr(dcm, "RescaleSlope", 1.0))
     20     intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
---> 21     arr = dcm.pixel_array  # triggers pixel decode
     22     return arr, slope, intercept
     23 

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in __getattr__(self, name)
    916             return {}
    917         # Try the base class attribute getter (fix for issue 332)
--> 918         return object.__getattribute__(self, name)
    919 
    920     @property

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in pixel_array(self)
   2191             that iterates through the image frames.
   2192         """
-> 2193         self.convert_pixel_data()
   2194         return cast("numpy.ndarray", self._pixel_array)
   2195 

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in convert_pixel_data(self, handler_name)
   1724             # Use 'pydicom.pixels' backend
   1725             opts["decoding_plugin"] = name
-> 1726             self._pixel_array = pixel_array(self, **opts)
   1727             self._pixel_id = get_image_pixel_ids(self)
   1728         else:

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/utils.py in pixel_array(src, ds_out, specific_tags, index, raw, decoding_plugin, **kwargs)
   1428 
   1429         opts = as_pixel_options(ds, **kwargs)
-> 1430         return decoder.as_array(
   1431             ds,
   1432             index=index,

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py in as_array(self, src, index, validate, raw, decoding_plugin, **kwargs)
    988 
    989         if validate:
--> 990             runner.validate()
    991 
    992         if self.is_native:

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py in validate(self)
    751     def validate(self) -> None:
    752         """Validate the decoding options and source buffer (if any)."""
--> 753         self._validate_options()
    754         if self.is_dataset or self.is_buffer:
    755             self._validate_buffer()

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py in _validate_options(self)
    820     def _validate_options(self) -> None:
    821         """Validate the supplied options to ensure they meet minimum requirements."""
--> 822         super()._validate_options()
    823 
    824         # The Extended Offset Table is optional

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/common.py in _validate_options(self)
    573         prefix = "Missing required element: (0028"
    574         if self._opts.get("bits_allocated") is None:
--> 575             raise AttributeError(f"{prefix},0100) 'Bits Allocated'")
    576 
    577         if not 1 <= self.bits_allocated <= 64 or (

AttributeError: Missing required element: (0028,0100) 'Bits Allocated'
