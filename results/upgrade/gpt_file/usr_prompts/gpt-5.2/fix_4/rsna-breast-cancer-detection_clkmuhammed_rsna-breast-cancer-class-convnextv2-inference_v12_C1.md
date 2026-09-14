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
import os, sys, platform, gc, re, math, random, time
from glob import glob
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd



## === cell 1
print("Platform:", platform.system())
print("Python  :", platform.python_version())
print("Executable:", sys.executable)



## === cell 2
import cv2

cv2.setNumThreads(1)
print("cv2:", cv2.__version__)

import tensorflow as tf

print("Tensorflow:", tf.__version__)

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

print("pydicom:", pydicom.__version__)



## === cell 3
tf.config.threading.set_inter_op_parallelism_threads(1)

try:
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    strategy = tf.distribute.TPUStrategy(cluster_resolver)
    print(
        "Running on TPU",
        cluster_resolver.master(),
        len(tf.config.list_logical_devices("TPU")),
    )
except Exception:
    gpus = tf.config.list_logical_devices("GPU")
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
        print("Running on multiple GPUs", [g.name for g in gpus])
    elif len(gpus) == 1:
        strategy = tf.distribute.get_strategy()
        print("Running on single GPU", gpus[0].name)
    else:
        strategy = tf.distribute.get_strategy()
        print("Running on CPU")
print("Number of accelerators:", strategy.num_replicas_in_sync)



## === cell 4
IMAGE_FORMAT = "JPG"
IMAGE_QUALITY = 100
TARGET_HEIGHT, TARGET_WIDTH, N_CHANNELS = (624, 512, 1)
INPUT_SHAPE = (TARGET_HEIGHT, TARGET_WIDTH, N_CHANNELS)

THRESHOLD_BEST = 0.857292  # kept but unused (we output probabilities for pF1)

DATA_DIR = "/kaggle/input/rsna-breast-cancer-detection"
print("DATA_DIR:", DATA_DIR)



## === cell 5
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
sample_submission_df = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

print(
    "test_df:",
    test_df.shape,
    "train_df:",
    train_df.shape,
    "sample_submission_df:",
    sample_submission_df.shape,
)
print("sample_submission columns:", sample_submission_df.columns.tolist())




## === cell 6
def read_dicom_to_uint16(path):
    ds = pydicom.dcmread(path, force=True)
    try:
        arr = ds.pixel_array
    except Exception as e:
        return None, str(e)

    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass

    arr = np.asarray(arr)

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr

    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    if slope != 1.0 or intercept != 0.0:
        arr = arr.astype(np.float32) * slope + intercept

    arr = arr.astype(np.float32)
    mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return None, "invalid_pixel_range"
    arr = (arr - mn) / (mx - mn)
    arr = (arr * 65535.0).clip(0, 65535).astype(np.uint16)
    return arr, None




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

    return img_data_roi




## === cell 8
def load_and_preprocess_image(file_path, debug=False):
    img_data_16u, err = read_dicom_to_uint16(file_path)
    if img_data_16u is None:
        return None, err

    img_data_roi = analyze_components(img_data_16u)

    img_data_resized = cv2.resize(
        src=img_data_roi,
        dsize=INPUT_SHAPE[:2][::-1],  # cv2 expects (width, height)
        interpolation=cv2.INTER_LANCZOS4,
    )

    img = img_data_resized.astype(np.float32)
    img_min, img_max = float(img.min()), float(img.max())
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img, dtype=np.float32)

    img = np.expand_dims(img, axis=-1)
    return img, None




## === cell 9
MODEL_DIR = "/kaggle/input/rsna-mammography-breast-cancer-tensorflow-model/rsna_cancer_convnext_v2_tiny_model_legacy.h5"
print("MODEL_DIR:", MODEL_DIR)


def build_fallback_model(input_shape=INPUT_SHAPE):
    inp = tf.keras.Input(shape=input_shape, name="image")
    x = tf.keras.layers.Conv2D(8, 3, strides=2, padding="same", activation="relu")(inp)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="cancer")(x)
    m = tf.keras.Model(inp, out)
    return m


tf.keras.backend.clear_session()
gc.collect()

with strategy.scope():
    if os.path.exists(MODEL_DIR):
        model = tf.keras.models.load_model(MODEL_DIR, compile=False)
        model.trainable = False
        print("External model loaded.")
    else:
        model = build_fallback_model(INPUT_SHAPE)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy"
        )
        print("External model not found; using fallback CNN.")




## === cell 10
def make_image_batch(df_batch):
    xs = []
    ys = []
    for r in df_batch.itertuples(index=False):
        dcm_path = f"{DATA_DIR}/train_images/{r.patient_id}/{r.image_id}.dcm"
        img, err = load_and_preprocess_image(dcm_path)
        if img is None:
            continue
        xs.append(img)
        ys.append(float(r.cancer))
    if not xs:
        return None, None
    x = np.stack(xs, axis=0).astype(np.float32)
    y = np.asarray(ys, dtype=np.float32)
    return x, y


if not os.path.exists(MODEL_DIR):
    rng = np.random.default_rng(123)
    idx = np.arange(len(train_df))
    rng.shuffle(idx)

    n_train = min(2500, len(train_df))
    sub = train_df.iloc[idx[:n_train]].reset_index(drop=True)

    batch_rows = 32
    steps = 0
    for start in range(0, len(sub), batch_rows):
        chunk = sub.iloc[start : start + batch_rows]
        x, y = make_image_batch(chunk)
        if x is None:
            continue
        model.train_on_batch(x, y)
        steps += 1
        if steps >= 120:  # bounded runtime
            break
    print("Fallback CNN trained on batches:", steps)



## === cell 11
_IMG_CACHE = {}  # key -> np.ndarray [H,W,1] or None for failed decode
_FAIL_CACHE = {}  # key -> bool


def _img_key(is_test, patient_id, image_id):
    return ("test" if is_test else "train", int(patient_id), int(image_id))


def get_preprocessed_image(is_test, patient_id, image_id):
    k = _img_key(is_test, patient_id, image_id)
    if k in _IMG_CACHE:
        return _IMG_CACHE[k], _FAIL_CACHE[k]
    base = "test_images" if is_test else "train_images"
    dcm_path = f"{DATA_DIR}/{base}/{patient_id}/{image_id}.dcm"
    img, err = load_and_preprocess_image(dcm_path)
    failed = img is None
    _IMG_CACHE[k] = img
    _FAIL_CACHE[k] = failed
    return img, failed


def predict_probs_for_rows(df_pid_img, is_test, batch_size=32):
    n = len(df_pid_img)
    probs = np.empty(n, dtype=np.float32)
    fail_count = 0

    batch_imgs = []
    batch_indices = []

    def _flush():
        nonlocal batch_imgs, batch_indices, probs
        if not batch_imgs:
            return
        x = np.stack(batch_imgs, axis=0).astype(np.float32)  # [B,H,W,1]
        try:
            p = model.predict_on_batch({"image": x}).reshape(-1)
        except Exception:
            p = model.predict_on_batch(x).reshape(-1)
        p = np.asarray(p, dtype=np.float32)
        p = np.where(np.isfinite(p), p, 0.001).astype(np.float32)
        p = np.clip(p, 0.0, 1.0, out=p)
        probs[np.asarray(batch_indices, dtype=np.int64)] = p
        batch_imgs = []
        batch_indices = []

    for i, r in enumerate(df_pid_img.itertuples(index=False)):
        img, failed = get_preprocessed_image(is_test, r.patient_id, r.image_id)
        if failed:
            probs[i] = 0.001
            fail_count += 1
            continue
        batch_imgs.append(img)
        batch_indices.append(i)
        if len(batch_imgs) >= batch_size:
            _flush()
    _flush()
    return probs, fail_count




## === cell 12
pass



## === cell 13
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

train_feat = train_df[
    [
        "site_id",
        "laterality",
        "view",
        "age",
        "implant",
        "machine_id",
        "patient_id",
        "image_id",
        "cancer",
    ]
].copy()
rng = np.random.default_rng(123)
idx = np.arange(len(train_feat))
rng.shuffle(idx)
n_cal = min(12000, len(train_feat))
train_feat = train_feat.iloc[idx[:n_cal]].reset_index(drop=True)

train_probs, train_fail = predict_probs_for_rows(
    train_feat[["patient_id", "image_id"]], is_test=False, batch_size=32
)
train_feat["img_p"] = train_probs
print(
    "Calibration train rows:", len(train_feat), "image decode fails:", int(train_fail)
)

X_train = train_feat.drop(columns=["cancer"])
y_train = train_feat["cancer"].astype(int)

cat_cols = ["site_id", "laterality", "view", "implant", "machine_id"]
num_cols = ["age", "img_p"]

pre = ColumnTransformer(
    transformers=[
        (
            "cat",
            Pipeline(
                steps=[
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            cat_cols,
        ),
        ("num", Pipeline(steps=[("imp", SimpleImputer(strategy="median"))]), num_cols),
    ],
    remainder="drop",
    sparse_threshold=0.3,
)

cal_model = Pipeline(
    steps=[
        ("pre", pre),
        ("lr", LogisticRegression(max_iter=200, solver="lbfgs", n_jobs=None)),
    ]
)

cal_model.fit(X_train, y_train)
print("Calibration model fitted.")



## === cell 14
test_feat = test_df[
    [
        "prediction_id",
        "site_id",
        "laterality",
        "view",
        "age",
        "implant",
        "machine_id",
        "patient_id",
        "image_id",
    ]
].copy()

test_probs, test_fail = predict_probs_for_rows(
    test_feat[["patient_id", "image_id"]], is_test=True, batch_size=32
)
test_feat["img_p"] = test_probs
print("Test per-image decode fails (cal):", int(test_fail))

X_test = test_feat.drop(columns=["prediction_id", "patient_id", "image_id"])
test_feat["cal_p"] = cal_model.predict_proba(X_test)[:, 1].astype(np.float32)

preds_cal_by_prediction_id = (
    test_feat.groupby("prediction_id")["cal_p"].mean().to_dict()
)
print("Calibrated unique prediction_id:", len(preds_cal_by_prediction_id))



## === cell 15
submission_df = pd.DataFrame(
    {
        "prediction_id": list(preds_cal_by_prediction_id.keys()),
        "cancer": [
            float(preds_cal_by_prediction_id[k])
            for k in preds_cal_by_prediction_id.keys()
        ],
    }
)

submission_df = sample_submission_df[["prediction_id"]].merge(
    submission_df, on="prediction_id", how="left"
)
submission_df["cancer"] = (
    submission_df["cancer"].fillna(0.001).astype(float).clip(0.0, 1.0)
)

print(submission_df.head())
print(submission_df.shape, submission_df.isna().sum().to_dict())



## === cell 16
out_path = "submission.csv"
submission_df.to_csv(out_path, index=False)

check = pd.read_csv(out_path)
assert list(check.columns) == ["prediction_id", "cancer"]
assert len(check) == len(sample_submission_df)
print("Wrote", out_path, "with shape", check.shape)
print("cancer stats:", check["cancer"].describe().to_dict())
