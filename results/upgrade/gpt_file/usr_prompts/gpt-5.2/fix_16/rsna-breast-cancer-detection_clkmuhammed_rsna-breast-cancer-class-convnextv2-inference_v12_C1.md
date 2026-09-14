# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

# 5. Target score

0.139917695473251

# 6. Current score

0.05674

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00192) has done: 'I make the notebook run end-to-end on the provided Kaggle environment by removing the failing pip-installed dependencies (they are cp310 wheels while your runtime is Python 3.12) and by switching DICOM loading to pydicom (already available). I also fix the cascading `NameError`s by ensuring all needed imports exist in the cells that use them, and make preprocessing return an array so inference doesn’t rely on temporary files/folders. Finally, I fix the submission format by using `test.csv`’s `prediction_id` (not `patient_id_laterality`) and aggregating predictions per `prediction_id` to produce exactly the required rows/columns in `submission.csv`. These changes preserve the core inference approach (preprocess -> model predict -> average per group) while making it executable and generating a valid submission file.'
- What this solution (achieved 0.03276) has done: 'Main bottleneck is repeated slow DICOM decoding/preprocessing (VOI LUT + connected components + Lanczos resize) for ~12k calibration train images plus ~5.5k test images, done in Python loops and re-reading the same patient/image pairs across steps. I make decoding provably equivalent but faster by (1) using pydicom’s `stop_before_pixels` + `specific_tags` to avoid parsing unused metadata, (2) switching to faster OpenCV resize (AREA for downscale, LINEAR otherwise) while keeping the exact target shape and normalization, and (3) adding an on-disk cache of the final preprocessed float32 tensors keyed by `patient_id/image_id` so calibration and test runs never decode the same image twice across reruns/cells. I also reduce Python overhead in `predict_probs_for_rows` by preallocating arrays, using larger thread chunks, and avoiding per-item string building inside threads; the model inference path and calibration logic remain unchanged. All changes are deterministic (same seeds) and preserve evaluation semantics (still output probabilities per prediction_id averaged across images).'
- What this solution (achieved 0.03276) has done: 'I fix the runtime crash in the TensorFlow import caused by an incompatible protobuf version by forcing the pure-Python protobuf backend *before* importing TensorFlow and explicitly importing `google.protobuf` first (a known workaround for the `MessageFactory.GetPrototype` error on newer protobuf). I also make the DICOM caching write path correct (NumPy was saving to `*.tmp.npy` but replacing `*.tmp`, so caching silently failed and re-decoding slowed/injured calibration quality), which should improve both runtime stability and score by enabling the intended cached preprocessing. Finally, I keep the modeling and calibration logic unchanged, only adding a small safety fallback so `predict_probs_for_rows` always returns an array even if thread loading yields unexpected shapes.'
- What this solution (achieved 0.03276) has done: 'I fix the TensorFlow import crash caused by the protobuf API mismatch by forcing the pure-Python protobuf implementation earlier and pinning a compatible protobuf runtime behavior (without changing any modeling logic). I also make the external model path robust by falling back if the model file isn’t present (so the notebook always completes), and keep the rest of the preprocessing, prediction, calibration, and submission semantics identical. These changes are primarily correctness/stability fixes to unblock execution; score should at least return to the current pipeline’s level and can improve slightly by ensuring the intended CNN model loads when available. Finally, I ensure the script still writes a valid `submission.csv` with exactly the sample submission’s rows/columns.'
- What this solution (achieved 0.05598) has done: 'I fix the TensorFlow import crash by removing the incompatible TF dependency entirely (the environment’s TF/protobuf combination is broken) and keep the rest of the pipeline intact by using a deterministic fallback probability model. Then I fix the path-building bug that causes `UFuncTypeError` by switching from NumPy string addition to a safe Python list comprehension. Finally, I ensure calibration and test inference run end-to-end and that `preds_cal_by_prediction_id` is always defined so a valid `submission.csv` with the exact sample rows/columns is written.'
- What this solution (achieved 0.05375) has done: 'You’re currently well below the target (0.05598 vs 0.1399, higher-is-better), so we should improve score with minimal, low-risk changes that don’t alter the overall pipeline structure. The biggest issue is that your per-image “model” is a simple mean/std heuristic; without changing the approach (still deterministic image -> prob -> metadata calibration -> mean by prediction_id), we can make that heuristic more informative by adding a couple of cheap intensity/texture summaries (percentiles + gradient energy) that better correlate with suspicious findings. Then, because pF1 is sensitive to probability calibration, we lightly tune the LogisticRegression regularization (`C`) and increase `max_iter` to ensure convergence (same model family, same training approach). Finally, we aggregate per `prediction_id` using a max-like operator (1 - product(1-p)) which is still a legitimate probability aggregation for “any image positive” and typically improves breast-level detection versus a plain mean, while keeping the same grouping/aggregation step.'
- What this solution (achieved 0.05397) has done: 'Your current score (0.05375) is well below the target (0.1399, higher-is-better), so the smallest safe move is to improve probability ranking/calibration without changing the overall pipeline (DICOM->preprocess->per-image heuristic prob->LogReg calibration->aggregate by prediction_id->submission). I keep your preprocessing and training approach intact, but make the per-image heuristic slightly more informative by adding a central “bright mass” feature (top-intensity mean) and a simple spatial asymmetry feature (left-right imbalance), both computed from the already-normalized image tensor. Then I slightly adjust LogisticRegression regularization (C) to reduce underfitting on the calibration set while keeping the same model family and objective. Finally, I keep your “noisy-OR” aggregation but clip per-image probabilities a bit tighter to avoid extreme products dominating and hurting pF1 calibration.'
- What this solution (achieved 0.05674) has done: 'Your current score (0.05397) is far below the target (0.1399), so we should improve pF1 with minimal, low-risk changes while keeping your pipeline intact (DICOM→preprocess→per-image heuristic prob→LogReg calibration→aggregate by prediction_id→submission). The biggest gain with minimal semantic change is to make the calibrator train on the same “breast-level” unit the metric cares about by aggregating per-image probabilities to `prediction_id` for training (still LogisticRegression, same preprocessing, same inference), instead of training on per-image rows that mismatch the evaluation granularity. Then we align test-time and train-time aggregation to be identical (noisy-OR over per-image calibrated probs), which typically improves ranking/calibration for “any image positive per breast”. Finally, we keep everything deterministic and within time by reusing the same cached image probabilities and by only adding a lightweight grouping step.'

# 9. Code solution

## === cell 0
import os, sys, platform, gc, re, math, random, time
from glob import glob
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd

print("Platform:", platform.system())
print("Python  :", platform.python_version())
print("Executable:", sys.executable)



## === cell 1
import cv2

cv2.setNumThreads(1)
print("cv2:", cv2.__version__)

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

print("pydicom:", pydicom.__version__)

np.random.seed(123)
random.seed(123)



## === cell 2
IMAGE_FORMAT = "JPG"
IMAGE_QUALITY = 100
TARGET_HEIGHT, TARGET_WIDTH, N_CHANNELS = (624, 512, 1)
INPUT_SHAPE = (TARGET_HEIGHT, TARGET_WIDTH, N_CHANNELS)

THRESHOLD_BEST = 0.857292  # kept but unused (we output probabilities for pF1)

DATA_DIR = "/kaggle/input/rsna-breast-cancer-detection"
print("DATA_DIR:", DATA_DIR)



## === cell 3
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



## === cell 4
_DICOM_TAGS_NEEDED = [
    "PhotometricInterpretation",
    "RescaleSlope",
    "RescaleIntercept",
    "WindowCenter",
    "WindowWidth",
    "VOILUTFunction",
    "PixelRepresentation",
    "BitsStored",
    "BitsAllocated",
    "SamplesPerPixel",
    "PlanarConfiguration",
    "PixelData",
]


def read_dicom_to_uint16(path):
    photometric = ""
    try:
        ds_meta = pydicom.dcmread(
            path,
            force=True,
            stop_before_pixels=True,
            specific_tags=["PhotometricInterpretation"],
        )
        photometric = getattr(ds_meta, "PhotometricInterpretation", "") or ""
    except Exception:
        photometric = ""

    try:
        ds = pydicom.dcmread(path, force=True, specific_tags=_DICOM_TAGS_NEEDED)
        arr = ds.pixel_array
    except Exception as e2:
        return None, str(e2)

    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass

    arr = np.asarray(arr)

    if (
        photometric == "MONOCHROME1"
        or getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1"
    ):
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




## === cell 5
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
        fg = black_background_mask > 0
    else:
        fg = blur > 25

    if not fg.any():
        return img_data_8u

    ys, xs = np.where(fg)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    box_area = (y1 - y0) * (x1 - x0)
    full_area = img_data_8u.shape[0] * img_data_8u.shape[1]
    if box_area / max(1, full_area) > 0.92:
        return img_data_8u

    black_background_mask = fg.astype(np.uint8) * 255
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




## === cell 6
CACHE_DIR = "/kaggle/working/dcm_preprocessed_cache_v1"
os.makedirs(CACHE_DIR, exist_ok=True)
_CACHE_INDEX_PATH = os.path.join(CACHE_DIR, "_index_v1.npz")

_cache_known_good = set()
_cache_known_bad = set()
try:
    if os.path.exists(_CACHE_INDEX_PATH):
        d = np.load(_CACHE_INDEX_PATH, allow_pickle=False)
        _cache_known_good = set(d["good"].astype(str).tolist())
        _cache_known_bad = set(d["bad"].astype(str).tolist())
except Exception:
    _cache_known_good, _cache_known_bad = set(), set()


def _save_cache_index():
    try:
        np.savez_compressed(
            _CACHE_INDEX_PATH,
            good=np.asarray(sorted(_cache_known_good), dtype=np.str_),
            bad=np.asarray(sorted(_cache_known_bad), dtype=np.str_),
        )
    except Exception:
        pass


def _cache_path_for_dcm(file_path: str) -> str:
    p = Path(file_path)
    if len(p.parts) >= 2:
        key = f"{p.parts[-2]}_{p.stem}"
    else:
        key = re.sub(r"[^A-Za-z0-9_.-]+", "_", p.stem)
    return os.path.join(CACHE_DIR, key + ".npy")


_MEM_CACHE = {}
_MEM_CACHE_ORDER = []
_MEM_CACHE_MAX = 512  # bounded memory; deterministic eviction order (FIFO)


def _mem_cache_get(key):
    v = _MEM_CACHE.get(key, None)
    if v is None:
        return None
    return v


def _mem_cache_put(key, value):
    if key in _MEM_CACHE:
        return
    _MEM_CACHE[key] = value
    _MEM_CACHE_ORDER.append(key)
    if len(_MEM_CACHE_ORDER) > _MEM_CACHE_MAX:
        old = _MEM_CACHE_ORDER.pop(0)
        _MEM_CACHE.pop(old, None)


def load_and_preprocess_image(file_path, debug=False):
    cpath = _cache_path_for_dcm(file_path)

    mem = _mem_cache_get(cpath)
    if mem is not None:
        return mem, None

    if cpath in _cache_known_bad:
        return None, "cached_failure"
    if cpath in _cache_known_good:
        try:
            img = np.load(cpath, mmap_mode="r")
            if (
                isinstance(img, np.ndarray)
                and img.shape == INPUT_SHAPE
                and img.dtype == np.float32
            ):
                arr = np.asarray(img)
                _mem_cache_put(cpath, arr)
                return arr, None
        except Exception:
            pass

    img_data_16u, err = read_dicom_to_uint16(file_path)
    if img_data_16u is None:
        _cache_known_bad.add(cpath)
        return None, err

    img_data_roi = analyze_components(img_data_16u)

    target_h, target_w = INPUT_SHAPE[0], INPUT_SHAPE[1]
    h, w = img_data_roi.shape[:2]
    interp = cv2.INTER_AREA if (h >= target_h and w >= target_w) else cv2.INTER_LINEAR

    img_data_resized = cv2.resize(
        src=img_data_roi,
        dsize=(target_w, target_h),  # cv2 expects (width, height)
        interpolation=interp,
    )

    img = img_data_resized.astype(np.float32, copy=False)
    img_min, img_max = float(img.min()), float(img.max())
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img, dtype=np.float32)

    img = np.expand_dims(img, axis=-1).astype(np.float32, copy=False)

    try:
        tmp_path = cpath + ".tmp.npy"
        np.save(tmp_path, img)
        os.replace(tmp_path, cpath)
        _cache_known_good.add(cpath)
    except Exception:
        _cache_known_bad.add(cpath)
        try:
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

    _mem_cache_put(cpath, img)
    return img, None




## === cell 7
def image_model_predict_batch(x_batch: np.ndarray) -> np.ndarray:
    x_img = np.asarray(x_batch, dtype=np.float32)  # (B,H,W,1)
    bsz = x_img.shape[0]

    x = x_img.reshape((bsz, -1))  # (B, H*W*C)

    m = x.mean(axis=1)
    s = x.std(axis=1)

    p90 = np.quantile(x, 0.90, axis=1)
    p99 = np.quantile(x, 0.99, axis=1)
    tail = p99 - p90

    top_mean = np.empty((bsz,), dtype=np.float32)
    k = max(1, x.shape[1] // 100)  # 1%
    for i in range(bsz):
        xi = x[i]
        thr = np.partition(xi, xi.size - k)[xi.size - k]
        top_mean[i] = float(xi[xi >= thr].mean())

    hw = x_img[..., 0]  # (B,H,W)
    mid = hw.shape[2] // 2
    left_m = hw[:, :, :mid].mean(axis=(1, 2))
    right_m = hw[:, :, mid:].mean(axis=(1, 2))
    asym = np.abs(left_m - right_m)

    z = (
        2.2 * (m - 0.50)
        + 1.2 * (s - 0.25)
        + 2.0 * (p90 - 0.65)
        + 1.5 * (tail - 0.10)
        + 1.8 * (top_mean - 0.75)
        + 1.0 * (asym - 0.03)
    )

    p = 1.0 / (1.0 + np.exp(-z))
    p = p.astype(np.float32)

    np.clip(p, 0.002, 0.998, out=p)
    return p


print("Using non-TF fallback image model (deterministic, enriched summary stats).")




## === cell 8
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
    x = np.stack(xs, axis=0).astype(np.float32, copy=False)
    y = np.asarray(ys, dtype=np.float32)
    return x, y


print("No fallback CNN training step required.")



## === cell 9
import queue
from concurrent.futures import ThreadPoolExecutor


def _build_dcm_paths_from_arrays(patient_ids, image_ids, is_test: bool):
    base = "test_images" if is_test else "train_images"
    prefix = f"{DATA_DIR}/{base}"
    return [f"{prefix}/{pid}/{iid}.dcm" for pid, iid in zip(patient_ids, image_ids)]


def _predict_probs_for_paths(paths, batch_size=32, max_workers=None, queue_size=256):
    n = len(paths)
    probs = np.full(n, 0.002, dtype=np.float32)
    fail = np.ones(n, dtype=bool)

    if max_workers is None:
        max_workers = min(8, max(2, (os.cpu_count() or 2) // 2))

    xbuf = np.empty((batch_size, *INPUT_SHAPE), dtype=np.float32)
    ibuf = np.empty((batch_size,), dtype=np.int64)
    b = 0

    def _flush(bsz):
        nonlocal b
        if bsz <= 0:
            return
        x = xbuf[:bsz]
        p = image_model_predict_batch(x).reshape(-1)
        p = np.asarray(p, dtype=np.float32)
        p = np.where(np.isfinite(p), p, 0.002).astype(np.float32, copy=False)
        np.clip(p, 0.0, 1.0, out=p)
        probs[ibuf[:bsz]] = p
        b = 0

    q = queue.Queue(maxsize=queue_size)
    stop_token = object()

    def _worker(it):
        for i, pth in it:
            img, err = load_and_preprocess_image(pth)
            q.put((i, img), block=True)
        q.put(stop_token, block=True)

    indices = np.arange(n, dtype=np.int64)
    chunks = np.array_split(indices, max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for ch in chunks:
            it = ((int(i), paths[int(i)]) for i in ch.tolist())
            ex.submit(_worker, it)

        stops_seen = 0
        while stops_seen < max_workers:
            item = q.get(block=True)
            if item is stop_token:
                stops_seen += 1
                continue
            i, img = item
            if img is None or not (
                isinstance(img, np.ndarray) and img.shape == INPUT_SHAPE
            ):
                continue
            fail[i] = False
            xbuf[b] = img
            ibuf[b] = i
            b += 1
            if b >= batch_size:
                _flush(b)

        _flush(b)

    return probs, int(fail.sum())


def predict_probs_for_rows(df_pid_img, is_test, batch_size=32):
    pid = df_pid_img["patient_id"].to_numpy(dtype=str, copy=False)
    iid = df_pid_img["image_id"].to_numpy(dtype=str, copy=False)
    paths = _build_dcm_paths_from_arrays(pid, iid, is_test=is_test)
    return _predict_probs_for_paths(paths, batch_size=batch_size)




## === cell 10
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

train_feat["prediction_id"] = (
    train_feat["patient_id"].astype(str) + "-" + train_feat["laterality"].astype(str)
)

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
    "Calibration source image rows:",
    len(train_feat),
    "image decode fails:",
    int(train_fail),
)

gtr = train_feat.groupby("prediction_id", sort=False)
train_agg = gtr.agg(
    site_id=("site_id", "first"),
    laterality=("laterality", "first"),
    view=("view", "first"),
    age=("age", "median"),
    implant=("implant", "first"),
    machine_id=("machine_id", "first"),
)
train_agg["img_p_or"] = 1.0 - gtr["img_p"].apply(
    lambda s: float(np.prod(1.0 - np.clip(s.to_numpy(np.float32), 0.002, 0.998)))
).to_numpy(np.float32, copy=False)

y_train = gtr["cancer"].max().astype(int).reindex(train_agg.index)

X_train = train_agg.reset_index(drop=True)
y_train = y_train.to_numpy(dtype=int, copy=False)

cat_cols = ["site_id", "laterality", "view", "implant", "machine_id"]
num_cols = ["age", "img_p_or"]

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
        (
            "lr",
            LogisticRegression(
                max_iter=800,
                solver="lbfgs",
                n_jobs=None,
                class_weight="balanced",
                C=1.5,
            ),
        ),
    ]
)

cal_model.fit(X_train, y_train)
print("Calibration model fitted on breast-level rows:", len(X_train))

_save_cache_index()



## === cell 11
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
print("Test per-image decode fails:", int(test_fail))

g = test_feat.groupby("prediction_id", sort=False)

test_agg = g.agg(
    site_id=("site_id", "first"),
    laterality=("laterality", "first"),
    view=("view", "first"),
    age=("age", "median"),
    implant=("implant", "first"),
    machine_id=("machine_id", "first"),
)
test_agg["img_p_or"] = 1.0 - g["img_p"].apply(
    lambda s: float(np.prod(1.0 - np.clip(s.to_numpy(np.float32), 0.002, 0.998)))
).to_numpy(np.float32, copy=False)

test_agg = test_agg.reset_index()  # keep prediction_id column
X_test = test_agg.drop(columns=["prediction_id"])
test_agg["cal_p"] = cal_model.predict_proba(X_test)[:, 1].astype(np.float32)

preds_cal_by_prediction_id = dict(
    zip(
        test_agg["prediction_id"].astype(str).tolist(),
        test_agg["cal_p"].astype(float).tolist(),
    )
)

print("Calibrated unique prediction_id:", len(preds_cal_by_prediction_id))

_save_cache_index()



## === cell 12
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
    submission_df["cancer"].fillna(0.002).astype(float).clip(0.0, 1.0)
)

print(submission_df.head())
print(submission_df.shape, submission_df.isna().sum().to_dict())

out_path = "submission.csv"
submission_df.to_csv(out_path, index=False)

check = pd.read_csv(out_path)
assert list(check.columns) == ["prediction_id", "cancer"]
assert len(check) == len(sample_submission_df)
print("Wrote", out_path, "with shape", check.shape)
print("cancer stats:", check["cancer"].describe().to_dict())
