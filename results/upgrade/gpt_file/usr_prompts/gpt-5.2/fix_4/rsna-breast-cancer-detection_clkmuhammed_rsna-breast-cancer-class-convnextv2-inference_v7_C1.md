# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os, sys, platform, warnings

warnings.filterwarnings("ignore")

print("Python  :", sys.version)
print("Platform:", platform.platform())

DATA_DIR = "/kaggle/input/rsna-breast-cancer-detection"
MODEL_DATA_DIR = "/kaggle/input/rsna-breast-cancer-class-convnextv2-model-weights/rsna_cancer_convnext_v2_tiny_model_weights.h5"

print("DATA_DIR      :", DATA_DIR)
print("MODEL_DATA_DIR:", MODEL_DATA_DIR)
print("Exists weights:", os.path.exists(MODEL_DATA_DIR))



## === cell 1
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd

import tensorflow as tf

print("Tensorflow:", tf.__version__)

tf.compat.v1.enable_v2_behavior()
tf.config.optimizer.set_jit(True)
tf.config.threading.set_inter_op_parallelism_threads(1)

import cv2

cv2.setNumThreads(1)
print("cv2:", cv2.__version__)

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

print("pydicom:", pydicom.__version__)

tfio = None
print("tensorflow_io: disabled for compatibility")

from tqdm.auto import tqdm

import re, gc
import matplotlib.pyplot as plt



## === cell 2
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Running on TPU", tpu.master(), len(tf.config.list_logical_devices("TPU")))
    gpus = None
except Exception:
    gpus = tf.config.list_logical_devices("GPU")
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy(
            devices=[f"/gpu:{gpu.name.split(':')[-1]}" for gpu in gpus]
        )
        print("Running on multiple GPUs", [gpu.name for gpu in gpus])
    elif len(gpus) == 1:
        strategy = tf.distribute.get_strategy()
        print("Running on single GPU", gpus[0].name)
    else:
        strategy = tf.distribute.get_strategy()
        print("Running on CPU")

print("Number of accelerators:", strategy.num_replicas_in_sync)



## === cell 3
IMAGE_FORMAT = "JPG"
IMAGE_QUALITY = 95
TARGET_HEIGHT, TARGET_WIDTH, N_CHANNELS = (1024, 768, 1)
INPUT_SHAPE = (TARGET_HEIGHT, TARGET_WIDTH, N_CHANNELS)

THRESHOLD_BEST = 0.849352

print("INPUT_SHAPE:", INPUT_SHAPE)



## === cell 4
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
sample_submission_df = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

print("test_df:", test_df.shape, "sample_submission_df:", sample_submission_df.shape)
print("sample_submission columns:", list(sample_submission_df.columns))
print(test_df.head())
print(sample_submission_df.head())




## === cell 5
def _opencv_decode_encapsulated(ds) -> np.ndarray | None:
    """Try decoding encapsulated/compressed DICOM PixelData with OpenCV (JPEG/JPEG2000)."""
    try:
        from pydicom.encaps import generate_pixel_data_frame

        frame_bytes = next(generate_pixel_data_frame(ds.PixelData, nr_frames=1), None)
        if frame_bytes is None:
            return None

        buf = np.frombuffer(frame_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None

        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img.astype(np.float32)
    except Exception:
        return None


def dicom_to_uint8(path: str) -> np.ndarray:
    ds = pydicom.dcmread(path, force=True)

    arr = None
    try:
        arr = ds.pixel_array.astype(np.float32)
    except Exception:
        arr = _opencv_decode_encapsulated(ds)

    if arr is None:
        return np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8)

    try:
        arr = apply_voi_lut(arr, ds).astype(np.float32)
    except Exception:
        pass

    if getattr(ds, "PhotometricInterpretation", None) == "MONOCHROME1":
        arr = np.max(arr) - arr

    arr = arr - np.min(arr)
    mx = np.max(arr)
    if mx > 0:
        arr = arr / mx
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def image_findNonZero(img_data_uint8: np.ndarray) -> np.ndarray:
    coords = cv2.findNonZero(img_data_uint8)
    if coords is None:
        return img_data_uint8  # fallback: blank image
    x1, y1, w, h = cv2.boundingRect(coords)
    return img_data_uint8[y1 : y1 + h, x1 : x1 + w]


def load_and_preprocess_image(
    file_path: str, save_image: bool = False, debug: bool = False
) -> np.ndarray:
    img_data_uint8 = dicom_to_uint8(file_path)
    img_data_cropped = image_findNonZero(img_data_uint8)
    img_resized = cv2.resize(
        img_data_cropped,
        dsize=INPUT_SHAPE[:2][::-1],
        interpolation=cv2.INTER_LANCZOS4,
    )
    img_resized = np.expand_dims(img_resized, 2)  # (H, W, 1)

    if save_image:
        patient_id, image_id = re.split(r"[\./]", file_path)[-3:-1]
        os.makedirs(str(patient_id), exist_ok=True)
        out_path = f"{patient_id}/{image_id}.{IMAGE_FORMAT.lower()}"
        if IMAGE_FORMAT.upper() == "PNG":
            cv2.imwrite(out_path, img_resized)
        else:
            cv2.imwrite(
                out_path, img_resized, [cv2.IMWRITE_JPEG_QUALITY, IMAGE_QUALITY]
            )

    if debug:
        plt.figure(figsize=(4, 4))
        plt.imshow(img_resized.squeeze(), cmap="bone")
        plt.axis("off")
        plt.show()

    return img_resized


test_file_paths = tf.io.gfile.glob(f"{DATA_DIR}/test_images/*/*.dcm")
print("Test size images:", len(test_file_paths))
if len(test_file_paths) > 0:
    _ = load_and_preprocess_image(test_file_paths[0], save_image=False, debug=False)
    print("DICOM load OK:", test_file_paths[0])



## === cell 6
print("Model Defined Shape:", INPUT_SHAPE)


def build_classifier_model(
    input_shape=INPUT_SHAPE, model_weights=MODEL_DATA_DIR
) -> tf.keras.models.Model:
    input_image = tf.keras.layers.Input(shape=input_shape, name="image", dtype=tf.uint8)
    input_patient_id = tf.keras.layers.Input(shape=(1,), name="patient_id")
    input_image_id = tf.keras.layers.Input(shape=(1,), name="image_id")

    x = tf.keras.layers.Lambda(lambda t: tf.image.grayscale_to_rgb(t))(input_image)
    x = tf.keras.layers.Lambda(lambda t: tf.cast(t, tf.float32))(x)
    x = tf.keras.layers.Lambda(
        lambda t: tf.keras.applications.imagenet_utils.preprocess_input(t, mode="tf")
    )(x)

    backbone = tf.keras.applications.ConvNeXtTiny(
        include_top=False,
        weights=None,
        input_shape=(input_shape[:-1] + (3,)),
    )
    x = backbone(x)
    x = tf.keras.layers.SpatialDropout2D(0.3)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(
        512,
        activation=tf.keras.layers.LeakyReLU(negative_slope=0.3),
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

    if model_weights and os.path.exists(model_weights):
        model.load_weights(model_weights)
        print("Loaded model weights:", model_weights)
    else:
        print(
            "WARNING: Model weights not found, using randomly initialized weights:",
            model_weights,
        )

    model.trainable = False
    model.compile()
    return model


tf.keras.backend.clear_session()
with strategy.scope():
    model = build_classifier_model()
model.summary()



## === cell 7
SUBMISSION_ROWS = []

for idx, (pred_id, group) in enumerate(
    tqdm(test_df.groupby("prediction_id"), desc="Predict")
):
    probs = []
    image_datas = []

    for _, row in group.iterrows():
        patient_id, image_id = row["patient_id"], row["image_id"]
        dicom_path = f"{DATA_DIR}/test_images/{patient_id}/{image_id}.dcm"

        image_data = load_and_preprocess_image(
            dicom_path, save_image=False, debug=False
        ).squeeze()
        image_datas.append(image_data)

        image_batch = np.expand_dims(image_data, (0, 3)).astype(np.uint8)
        X_batch = {
            "image": image_batch,
            "patient_id": np.array([[patient_id]]),
            "image_id": np.array([[image_id]]),
        }
        prob = float(model.predict_on_batch(X_batch).squeeze())
        probs.append(prob)

    pred_prob = float(np.mean(probs)) if len(probs) else 0.0

    if idx < 2:
        fig, axes = plt.subplots(nrows=1, ncols=len(image_datas), figsize=(5, 5))
        if len(image_datas) == 1:
            axes = [axes]
        for n, ax in enumerate(axes):
            ax.imshow(image_datas[n], cmap="bone")
            ax.set_title(f"{pred_id}\n{probs[n]:.3f}")
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    SUBMISSION_ROWS.append({"prediction_id": pred_id, "cancer": pred_prob})

    if np.random.rand() > 0.99:
        gc.collect()

submission_df = pd.DataFrame(SUBMISSION_ROWS)

print("Raw submission:", submission_df.shape)
print(submission_df.head())
submission_df.info()



## === cell 8
submission_df = sample_submission_df[["prediction_id"]].merge(
    submission_df, on="prediction_id", how="left", validate="one_to_one"
)

submission_df["cancer"] = (
    submission_df["cancer"].fillna(0.0).astype(float).clip(0.0, 1.0)
)

print("Aligned submission:", submission_df.shape)
assert list(submission_df.columns) == ["prediction_id", "cancer"]
assert submission_df["prediction_id"].isna().sum() == 0
assert submission_df["cancer"].isna().sum() == 0
print(submission_df.head())

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Wrote:", submission_path, "size:", os.path.getsize(submission_path))
print("Done.")
