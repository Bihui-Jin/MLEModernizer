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
Detect breast cancer in mammograms.

## Metric
[Probabilistic F1 score](https://aclanthology.org/2020.eval4nlp-1.9.pdf) (pF1). This extension of the traditional F score accepts probabilities instead of binary classifications. 

With pX as the probabilistic version of X:

$$
pF_1 = 2 \frac{pPrecision \cdot pRecall}{pPrecision + pRecall}
$$

where:

$$
pPrecision = \frac{pTP}{pTP + pFP}
$$

$$
pRecall = \frac{pTP}{TP + FN}
$$

## Submission Format
For each `prediction_id`, you should predict the likelihood of cancer in the corresponding `cancer` column. The submission file should have the following format:

```
prediction_id,cancer
0-L,0
0-R,0.5
0-R,0.5
1-L,1
...
# Dataset

**[train/test]_images/[patient_id]/[image_id].dcm** The mammograms, in dicom format. You can expect roughly 8,000 patients in the hidden test set. There are usually but not always 4 images per patient. Note that many of the images use the jpeg 2000 format which may you may need special libraries to load.

**sample_submission.csv** A valid sample submission.

**[train/test].csv** Metadata for each patient and image. Only the first few rows of the test set are available for download.

- `site_id` - ID code for the source hospital.
- `patient_id` - ID code for the patient.
- `image_id` - ID code for the image.
- `laterality` - Whether the image is of the left or right breast.
- `view` - The orientation of the image. The default for a screening exam is to capture two views per breast.
- `age` - The patient's age in years.
- `implant` - Whether or not the patient had breast implants. Site 1 only provides breast implant information at the patient level, not at the breast level.
- `density` - A rating for how dense the breast tissue is, with A being the least dense and D being the most dense. Extremely dense tissue can make diagnosis more difficult. Only provided for train.
- `machine_id` - An ID code for the imaging device.
- `cancer` - Whether or not the breast was positive for malignant cancer. The target value. Only provided for train.
- `biopsy` - Whether or not a follow-up biopsy was performed on the breast. Only provided for train.
- `invasive` - If the breast is positive for cancer, whether or not the cancer proved to be invasive. Only provided for train.
- `BIRADS` - 0 if the breast required follow-up, 1 if the breast was rated as negative for cancer, and 2 if the breast was rated as normal. Only provided for train.
- `prediction_id` - The ID for the matching submission row. Multiple images will share the same prediction ID. Test only.
- `difficult_negative_case` - True if the case was unusually difficult. Only provided for train.

# 2. Python version

3.11

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (190 lines)
            sample_submission.csv (2385 lines)
            sample_submission.csv.zip (6.5 kB)
            test.csv (5475 lines)
            test.csv.zip (60.6 kB)
            test.zip (160 Bytes)
            test_images.zip (29.1 GB)
            train.csv (49233 lines)
            train.csv.zip (513.7 kB)
            train.zip (162 Bytes)
            train_images.zip (260.7 GB)
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
            test_images/
                10116/
                    1470873094.dcm (8.5 MB)
                    472095321.dcm (5.6 MB)
                    ... and 2 other files
                10130/
                    1013166704.dcm (9.7 MB)
                    1165309236.dcm (8.7 MB)
                    ... and 5 other files
                ... and 1191 other folders
            train_images/
                10006/
                    1459541791.dcm (4.4 MB)
                    1864590858.dcm (4.0 MB)
                    ... and 2 other files
                10011/
                    1031443799.dcm (2.1 MB)
                    220375232.dcm (1.7 MB)
                    ... and 2 other files
                ... and 10720 other folders
        input/
            description.md (190 lines)
            sample_submission.csv (2385 lines)
            sample_submission.csv.zip (6.5 kB)
            test.csv (5475 lines)
            test.csv.zip (60.6 kB)
            test.zip (160 Bytes)
            test_images.zip (29.1 GB)
            train.csv (49233 lines)
            train.csv.zip (513.7 kB)
            train.zip (162 Bytes)
            train_images.zip (260.7 GB)
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
            test_images/
                10116/
                    1470873094.dcm (8.5 MB)
                    472095321.dcm (5.6 MB)
                    ... and 2 other files
                10130/
                    1013166704.dcm (9.7 MB)
                    1165309236.dcm (8.7 MB)
                    ... and 5 other files
                ... and 1191 other folders
            train_images/
                10006/
                    1459541791.dcm (4.4 MB)
                    1864590858.dcm (4.0 MB)
                    ... and 2 other files
                10011/
                    1031443799.dcm (2.1 MB)
                    220375232.dcm (1.7 MB)
                    ... and 2 other files
                ... and 10720 other folders
        working/
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
```

-> data/rsna-breast-cancer-detection/sample_submission.csv has 2384 rows and 2 columns.
The columns are: prediction_id, cancer

-> data/rsna-breast-cancer-detection/test.csv has 5474 rows and 9 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, implant, machine_id, prediction_id

-> data/rsna-breast-cancer-detection/train.csv has 49232 rows and 14 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, cancer, biopsy, invasive, BIRADS, implant, density, machine_id, difficult_negative_case

-> data/sample_submission.csv has 2384 rows and 2 columns.
The columns are: prediction_id, cancer

-> data/test.csv has 5474 rows and 9 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, implant, machine_id, prediction_id

-> data/train.csv has 49232 rows and 14 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, cancer, biopsy, invasive, BIRADS, implant, density, machine_id, difficult_negative_case

-> (stopped after 10 files for performance)

# 5. Target score

0.2375478927203065

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import sys, platform, os

print("Python  :", sys.version)
print("Platform:", platform.platform())



## === cell 1
import tensorflow as tf

print("Tensorflow version :", tf.__version__)

import cv2

print("cv2 version", cv2.__version__)

import pylibjpeg

print("pylibjpeg version", pylibjpeg.__version__)

import libjpeg

print("libjpeg version", libjpeg.__version__)

import dicomsdl

print("dicomsdl version", dicomsdl.DICOMSDL_VERSION)

import pydicom

print("pydicom version", pydicom.__version__)

from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import gc
from joblib import Parallel, delayed
from multiprocessing import cpu_count

from keras_cv_attention_models.convnext import ConvNeXtV2Tiny



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
IMAGE_FORMAT = "JPG"
IMAGE_QUALITY = 95
TARGET_HEIGHT, TARGET_WIDTH, N_CHANNELS = (1024, 768, 1)
INPUT_SHAPE = (TARGET_HEIGHT, TARGET_WIDTH, N_CHANNELS)

THRESHOLD_BEST = 0.849352  # kept for reference – not used for final probabilities



## === cell 3
DATA_DIR = "/kaggle/input/rsna-breast-cancer-detection"
MODEL_DATA_DIR = "/kaggle/input/rsna-breast-cancer-class-convnextv2-model-weights/rsna_cancer_convnext_v2_tiny_model_weights.h5"

print("DATA_DIR      :", DATA_DIR)
print("MODEL_DATA_DIR:", MODEL_DATA_DIR)



## === cell 4
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
sample_submission_df = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
print("test_df shape:", test_df.shape)
print("sample_submission shape:", sample_submission_df.shape)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2664278422.py in <cell line: 0>()
      1 # Load CSVs
----> 2 test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
      3 sample_submission_df = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
      4 print("test_df shape:", test_df.shape)
      5 print("sample_submission shape:", sample_submission_df.shape)

NameError: name 'pd' is not defined

## === cell 5
test_file_paths = tf.io.gfile.glob(f"{DATA_DIR}/test_images/*/*.dcm")
print(f"Test size images : {len(test_file_paths)}")




## === cell 6
def image_findNonZero(img_data_uint8):
    coords = cv2.findNonZero(img_data_uint8)
    x1, y1, w, h = cv2.boundingRect(coords)
    return img_data_uint8[y1 : y1 + h, x1 : x1 + w]


def load_and_preprocess_image(file_path, save_image=True, debug=False):
    img_dicom = dicomsdl.open(file_path)
    img_data_uint16 = img_dicom.pixelData(storedvalue=True)
    img_data_uint8 = cv2.normalize(
        img_data_uint16, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    if img_dicom.PhotometricInterpretation == "MONOCHROME1":
        img_data_uint8 = np.amax(img_data_uint8) - img_data_uint8

    img_cropped = image_findNonZero(img_data_uint8)
    img_resized = cv2.resize(
        img_cropped, dsize=INPUT_SHAPE[:2][::-1], interpolation=cv2.INTER_LANCZOS4
    )
    img_resized = np.expand_dims(img_resized, 2)

    if save_image:
        patient_id, image_id = re.split(r"[\\./]", file_path)[-3:-1]
        os.makedirs(patient_id, exist_ok=True)
        if IMAGE_FORMAT == "PNG":
            cv2.imwrite(f"{patient_id}/{image_id}.png", img_resized)
        else:
            cv2.imwrite(
                f"{patient_id}/{image_id}.jpg",
                img_resized,
                [cv2.IMWRITE_JPEG_QUALITY, IMAGE_QUALITY],
            )

    if debug:
        plt.imshow(img_resized.squeeze(), cmap="bone")
        plt.title(f"{patient_id}/{image_id}")
        plt.show()




## === cell 7
def build_classifier_model(input_shape=INPUT_SHAPE, model_weights=MODEL_DATA_DIR):
    input_image = tf.keras.layers.Input(shape=input_shape, name="image", dtype=tf.uint8)
    input_patient_id = tf.keras.layers.Input(shape=(1,), name="patient_id")
    input_image_id = tf.keras.layers.Input(shape=(1,), name="image_id")

    x = tf.keras.layers.Lambda(lambda z: tf.image.grayscale_to_rgb(z))(input_image)
    x = tf.cast(x, tf.float32)
    x = tf.keras.applications.imagenet_utils.preprocess_input(x, mode="tf")

    x = ConvNeXtV2Tiny(
        input_shape=(input_shape[0], input_shape[1], 3), pretrained=None, num_classes=0
    )(x)

    x = tf.keras.layers.SpatialDropout2D(0.3)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(
        512,
        activation=tf.keras.layers.LeakyReLU(alpha=0.3),
        kernel_regularizer=tf.keras.regularizers.l1_l2(l2=0.005),
    )(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(
        256, activation="gelu", kernel_regularizer=tf.keras.regularizers.l1(0.005)
    )(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(
        128, activation="gelu", kernel_regularizer=tf.keras.regularizers.l1(0.005)
    )(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.models.Model(
        inputs=[input_image, input_patient_id, input_image_id], outputs=output
    )

    model.load_weights(model_weights)
    model.trainable = False
    model.compile()
    return model




## === cell 8
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Running on TPU")
except Exception:
    gpus = tf.config.list_logical_devices("GPU")
    if gpus:
        strategy = tf.distribute.MirroredStrategy()
        print("Running on GPU")
    else:
        strategy = tf.distribute.get_strategy()
        print("Running on CPU")
print("Number of replicas:", strategy.num_replicas_in_sync)



## === cell 9
tf.keras.backend.clear_session()
with strategy.scope():
    model = build_classifier_model()
    model.summary()




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3856374048.py in <cell line: 0>()
      1 tf.keras.backend.clear_session()
      2 with strategy.scope():
----> 3     model = build_classifier_model()
      4     model.summary()
      5 

/tmp/ipykernel_55/252861287.py in build_classifier_model(input_shape, model_weights)
      7     # Convert 1‑channel uint8 to 3‑channel float32 as expected by ConvNeXtV2
      8     x = tf.keras.layers.Lambda(lambda z: tf.image.grayscale_to_rgb(z))(input_image)
----> 9     x = tf.cast(x, tf.float32)
     10     x = tf.keras.applications.imagenet_utils.preprocess_input(x, mode="tf")
     11 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/keras_tensor.py in __tf_tensor__(self, dtype, name)
    136 
    137     def __tf_tensor__(self, dtype=None, name=None):
--> 138         raise ValueError(
    139             "A KerasTensor cannot be used as input to a TensorFlow function. "
    140             "A KerasTensor is a symbolic placeholder for a shape and dtype, "

ValueError: A KerasTensor cannot be used as input to a TensorFlow function. A KerasTensor is a symbolic placeholder for a shape and dtype, used when constructing Keras Functional models or Keras Functions. You can only use it as input to a Keras layer or a Keras operation (from the namespaces `keras.layers` and `keras.operations`). You are likely doing something like:

```
x = Input(...)
...
tf_fn(x)  # Invalid.
```

What you should do instead is wrap `tf_fn` in a layer:

```
class MyLayer(Layer):
    def call(self, x):
        return tf_fn(x)

x = MyLayer()(x)
```


## === cell 10
def preprocess_and_save_image(args):
    (patient_id, laterality), group = args
    for _, row in group.iterrows():
        image_path = f"{DATA_DIR}/test_images/{patient_id}/{row['image_id']}.dcm"
        load_and_preprocess_image(image_path)


def convert_images_dicom2jpg():
    jobs = [
        delayed(preprocess_and_save_image)(args)
        for args in tqdm(test_df.groupby(["patient_id", "laterality"]))
    ]
    Parallel(
        n_jobs=cpu_count(), verbose=0, backend="multiprocessing", prefer="threads"
    )(jobs)


with strategy.scope():
    convert_images_dicom2jpg()
    if np.random.rand() > 0.90:
        gc.collect()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/392777078.py in <cell line: 0>()
     18 
     19 with strategy.scope():
---> 20     convert_images_dicom2jpg()
     21     if np.random.rand() > 0.90:
     22         gc.collect()

/tmp/ipykernel_55/392777078.py in convert_images_dicom2jpg()
     10     jobs = [
     11         delayed(preprocess_and_save_image)(args)
---> 12         for args in tqdm(test_df.groupby(["patient_id", "laterality"]))
     13     ]
     14     Parallel(

NameError: name 'tqdm' is not defined

## === cell 11
submission_rows = []
for idx, ((patient_id, laterality), group) in enumerate(
    tqdm(test_df.groupby(["patient_id", "laterality"]))
):
    probs = []
    for _, row in group.iterrows():
        image_id = row["image_id"]
        img_path = f"{patient_id}/{image_id}.{IMAGE_FORMAT.lower()}"
        img = cv2.imread(img_path, -1)  # read as is (grayscale)
        img = np.expand_dims(img, [0, 3])  # (1, H, W, 1)
        batch = {
            "image": img,
            "patient_id": np.array([[patient_id]]),
            "image_id": np.array([[image_id]]),
        }
        prob = model.predict_on_batch(batch).squeeze()
        probs.append(prob)

        os.remove(img_path)

    mean_prob = float(np.mean(probs))
    for _, row in group.iterrows():
        submission_rows.append(
            {"prediction_id": row["prediction_id"], "cancer": mean_prob}
        )

    if np.random.rand() > 0.99:
        gc.collect()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1664740462.py in <cell line: 0>()
      2 submission_rows = []
      3 for idx, ((patient_id, laterality), group) in enumerate(
----> 4     tqdm(test_df.groupby(["patient_id", "laterality"]))
      5 ):
      6     probs = []

NameError: name 'tqdm' is not defined

## === cell 12
submission_df = pd.DataFrame(submission_rows)
print("Submission shape:", submission_df.shape)
submission_df.head()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1969713463.py in <cell line: 0>()
----> 1 submission_df = pd.DataFrame(submission_rows)
      2 print("Submission shape:", submission_df.shape)
      3 submission_df.head()
      4 

NameError: name 'pd' is not defined

## === cell 13
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Saved submission to {submission_path}")

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1900536602.py in <cell line: 0>()
      1 submission_path = "submission.csv"
----> 2 submission_df.to_csv(submission_path, index=False)
      3 print(f"Saved submission to {submission_path}")

NameError: name 'submission_df' is not defined
