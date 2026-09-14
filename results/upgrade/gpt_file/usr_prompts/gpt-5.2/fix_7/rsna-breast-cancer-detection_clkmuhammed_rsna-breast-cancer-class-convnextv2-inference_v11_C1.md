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

3.12

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
import os, sys, platform, gc, re, random, time
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:

    def display(x, *args, **kwargs):
        print(x)


print("Platform:", platform.system())
print("Python  :", platform.python_version())
print("Executable:", sys.executable)



## === cell 1
DATA_DIR = "/kaggle/input/rsna-breast-cancer-detection"
WORK_DIR = "/kaggle/working"
TMP_IMG_DIR = os.path.join(WORK_DIR, "tmp_rsna_images")
os.makedirs(TMP_IMG_DIR, exist_ok=True)

print("DATA_DIR:", DATA_DIR)
print("TMP_IMG_DIR:", TMP_IMG_DIR)
print("Listing input dir exists:", os.path.exists(DATA_DIR))



## === cell 2
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

print("Tensorflow version:", tf.__version__)
try:
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
except Exception:
    pass

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
try:
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("Available devices:")
for i, device in enumerate(tf.config.list_logical_devices()):
    print(f"{i}) {device}")

strategy = tf.distribute.get_strategy()
print("Using strategy:", type(strategy).__name__)
print("Number of accelerators:", strategy.num_replicas_in_sync)



## === cell 3
import cv2

cv2.setNumThreads(1)
print(
    "cv2:",
    cv2.__version__,
    "| cuda devices:",
    cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2, "cuda") else "N/A",
)

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

print("pydicom:", pydicom.__version__)

try:
    from tqdm.auto import tqdm
except Exception:

    def tqdm(x, *args, **kwargs):
        return x


try:
    import joblib
except Exception:
    joblib = None

gc.collect()



## === cell 4
IMAGE_FORMAT = "JPG"
IMAGE_QUALITY = 100
TARGET_HEIGHT, TARGET_WIDTH, N_CHANNELS = (624, 512, 1)
INPUT_SHAPE = (TARGET_HEIGHT, TARGET_WIDTH, N_CHANNELS)

THRESHOLD_BEST = (
    0.857292  # kept for reference; we will submit probabilities (pF1 metric)
)



## === cell 5
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
sample_submission_df = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

print("test_df:", test_df.shape)
print("sample_submission_df:", sample_submission_df.shape)
display(test_df.head())
display(sample_submission_df.head())

assert "prediction_id" in test_df.columns
assert set(sample_submission_df.columns) == {"prediction_id", "cancer"}




## === cell 6
def dicom_to_uint16_voi(path: str) -> np.ndarray | None:
    ds = pydicom.dcmread(path, force=True)

    try:
        arr = ds.pixel_array
    except Exception:
        return None

    arr = arr.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + intercept

    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = np.max(arr) - arr

    arr = arr - np.min(arr)
    mx = np.max(arr)
    if mx > 0:
        arr = arr / mx
    arr = (arr * 65535.0).clip(0, 65535).astype(np.uint16)
    return arr




## === cell 7
def analyze_components(
    img_data_voi, filtering=False, threshold=cv2.THRESH_BINARY, debug=False
):
    img_data_8u = cv2.normalize(
        img_data_voi, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    if filtering:
        blur = cv2.GaussianBlur(src=img_data_8u, ksize=(5, 5), sigmaX=0)
    else:
        blur = img_data_8u

    if threshold in range(255):
        _, black_background_mask = cv2.threshold(
            src=blur,
            thresh=25,
            maxval=255,
            type=threshold,
        )
    else:
        black_background_mask = (blur > 25).astype(np.uint8)

    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(
        image=black_background_mask, connectivity=8, ltype=cv2.CV_32S
    )

    if retval > 1:
        largest_component_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        x, y, width, height, area = stats[largest_component_index]
        img_data_roi = img_data_8u[y : y + height, x : x + width]
    else:
        img_data_roi = img_data_8u

    if debug:
        import matplotlib.pyplot as plt

        plt.subplot(121)
        plt.imshow(black_background_mask, cmap="gray")
        plt.title(f"mask threshold={threshold} filtering={filtering}")

        plt.subplot(122)
        plt.imshow(img_data_roi, cmap="gray")
        plt.title(f"roi shape={img_data_roi.shape}")
        plt.tight_layout()
        plt.show()

    return img_data_roi




## === cell 8
_IMAGE_CACHE: dict[tuple[str, str], np.ndarray] = {}
_DCM_PATH_CACHE: dict[tuple[str, str], str] = {}


def load_and_preprocess_image(file_path, save_image=True, debug=False):
    img_data_16u = dicom_to_uint16_voi(file_path)

    if img_data_16u is None:
        img_data_resized = np.zeros(INPUT_SHAPE, dtype=np.uint8)
    else:
        img_data_roi = analyze_components(img_data_16u, filtering=False)
        img_data_resized = cv2.resize(
            src=img_data_roi,
            dsize=INPUT_SHAPE[:2][::-1],  # (w, h)
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.uint8)
        img_data_resized = np.expand_dims(img_data_resized, 2)  # (H, W, 1)

    if save_image:
        parts = re.split(r"[\\/\.]", file_path)
        patient_id, image_id = parts[-3], parts[-2]
        out_dir = os.path.join(TMP_IMG_DIR, str(patient_id))
        os.makedirs(out_dir, exist_ok=True)

        if IMAGE_FORMAT.upper() == "PNG":
            out_path = os.path.join(out_dir, f"{image_id}.png")
            cv2.imwrite(out_path, img_data_resized)
        else:
            out_path = os.path.join(out_dir, f"{image_id}.jpg")
            cv2.imwrite(
                out_path, img_data_resized, [cv2.IMWRITE_JPEG_QUALITY, IMAGE_QUALITY]
            )

        if debug:
            print("Wrote:", out_path)

    return img_data_resized


def get_preprocessed_image(patient_id: str, image_id: str) -> np.ndarray:
    key = (str(patient_id), str(image_id))
    cached = _IMAGE_CACHE.get(key)
    if cached is not None:
        return cached

    dcm_path = _DCM_PATH_CACHE.get(key)
    if dcm_path is None:
        dcm_path = f"{DATA_DIR}/test_images/{key[0]}/{key[1]}.dcm"
        _DCM_PATH_CACHE[key] = dcm_path

    img_data = load_and_preprocess_image(dcm_path, save_image=False, debug=False)

    if img_data.ndim == 2:
        img_data = img_data[:, :, None]

    _IMAGE_CACHE[key] = img_data
    return img_data




## === cell 9
tf.keras.backend.clear_session()
gc.collect()

with strategy.scope():
    inp = tf.keras.Input(shape=INPUT_SHAPE, name="image", dtype=tf.uint8)
    x = tf.keras.layers.Lambda(
        lambda t: tf.cast(t, tf.float32) / 255.0, name="cast_scale"
    )(inp)
    x = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs={"image": inp}, outputs=x)
    model.trainable = False
    model.compile()

model.summary()




## === cell 10
def preprocess_and_save_image(group_df):
    for row in group_df.itertuples(index=False):
        image_path = f"{DATA_DIR}/test_images/{row.patient_id}/{row.image_id}.dcm"
        _ = load_and_preprocess_image(image_path, save_image=True, debug=False)


def convert_images_dcm2jpg():
    return


convert_images_dcm2jpg()

some_jpgs = glob(os.path.join(TMP_IMG_DIR, "*", "*.jpg"))
print("Converted JPGs (pre-existing / on-demand may add later):", len(some_jpgs))



## === cell 11
BATCH_SIZE = 32  # safe memory-wise for (624,512,1) uint8

test_df2 = test_df[["prediction_id", "patient_id", "image_id"]].copy()
test_df2["patient_id"] = test_df2["patient_id"].astype(str)
test_df2["image_id"] = test_df2["image_id"].astype(str)

preds = np.empty((len(test_df2),), dtype=np.float32)
keys = list(zip(test_df2["patient_id"].tolist(), test_df2["image_id"].tolist()))

for start in tqdm(
    range(0, len(keys), BATCH_SIZE), total=(len(keys) + BATCH_SIZE - 1) // BATCH_SIZE
):
    end = min(start + BATCH_SIZE, len(keys))
    batch_keys = keys[start:end]

    imgs = [get_preprocessed_image(pid, iid) for pid, iid in batch_keys]
    img_batch = np.stack(imgs, axis=0)  # (N, H, W, 1) uint8

    batch_pred = model.predict_on_batch({"image": img_batch})
    batch_pred = np.asarray(batch_pred).reshape(-1).astype(np.float32, copy=False)

    preds[start:end] = batch_pred

    if (start // BATCH_SIZE) % 200 == 0:
        gc.collect()

test_df2["pred"] = preds

group_pred = test_df2.groupby("prediction_id", sort=False)["pred"].mean().reset_index()
group_pred = group_pred.rename(columns={"pred": "cancer"})

submission_df = sample_submission_df[["prediction_id"]].merge(
    group_pred, on="prediction_id", how="left"
)
submission_df["cancer"] = (
    submission_df["cancer"].fillna(0.0).astype(np.float32).clip(0.0, 1.0)
)

display(submission_df.head())
print("submission_df:", submission_df.shape, submission_df.isna().sum().to_dict())



## === cell 12
out_path = os.path.join(WORK_DIR, "submission.csv")
submission_df.to_csv(out_path, index=False)
print("Wrote:", out_path)
print("Columns:", submission_df.columns.tolist())
print("Preview:")
print(open(out_path, "r").read().splitlines()[:5])
