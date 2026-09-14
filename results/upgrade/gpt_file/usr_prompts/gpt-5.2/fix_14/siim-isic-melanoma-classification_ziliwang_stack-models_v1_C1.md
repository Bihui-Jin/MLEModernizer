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
Predict whether a lesion is malignant (0 denotes **benign**, and 1 indicates **malignant**).

## Metric
Area under the ROC curve.

## Submission Format
For each `image_name` in the test set, you must predict the probability (`target`) that the sample is **malignant**. The file should contain a header and have the following format:

```
image_name,target
ISIC_0052060,0.7
ISIC_0052349,0.9
ISIC_0058510,0.8
ISIC_0073313,0.5
ISIC_0073502,0.5
etc.
```

## Dataset 
The images are provided in DICOM format.

Images are also provided in JPEG and TFRecord format (in the `jpeg` and `tfrecords` directories, respectively). Images in TFRecord format have been resized to a uniform 1024x1024.

Metadata is also provided outside of the DICOM format, in CSV files. See the `Columns` section for a description.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `image_name` - unique identifier, points to filename of related DICOM image
- `patient_id` - unique patient identifier
- `sex` - the sex of the patient (when unknown, will be blank)
- `age_approx` - approximate patient age at time of imaging
- `anatom_site_general_challenge` - location of imaged site
- `diagnosis` - detailed diagnosis information (train only)
- `benign_malignant` - indicator of malignancy of imaged lesion
- `target` - binarized version of the target variable

# 2. Python version

3.8

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
        input/
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
        working/
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
```

-> data/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/siim-isic-melanoma-classification/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> data/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> input/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> (stopped after 10 files for performance)

# 5. Target score

0.924664431362698

# 6. Current score

0.81342

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.52941) has done: 'I remove the internet `wget` dependency (Kaggle notebooks run without outbound internet), which is why your downloaded CSVs are empty and `pd.read_csv` fails. To keep the core idea (blend several submissions) but make it runnable end-to-end, I replace the missing external submissions with three deterministic, locally-computed “base submissions” from `test.csv` metadata (patient/sex/age/site), then blend them the same way and write a valid `submission.csv`. I also fix the incorrect `/6` scaling so predictions remain valid probabilities (average of 3 models uses `/3`). Finally, I validate columns/order against `sample_submission.csv` to ensure Kaggle accepts the file.'
- What this solution (achieved 0.69768) has done: 'Your current 0.529 AUC is near-random because the “model” only uses test-set metadata priors, which can’t meaningfully rank malignant vs benign; to move toward the 0.924 target, we need real signal from the images while keeping the approach simple and within Kaggle/no-extra-packages constraints. I keep the existing submission-writing and blending structure, but replace the three metadata-based base submissions with three deterministic image-based predictors computed from the provided JPEGs (intensity/color/texture statistics), which usually yields a large AUC jump for this competition without changing the overall “blend several submissions then average” core idea. I also add robust JPEG path resolution (both `/kaggle/input/.../jpeg/test` and `/kaggle/data/.../jpeg/test`) and ensure predictions stay valid probabilities and align exactly to `sample_submission.csv`. This is still fast (single pass over ~4k test images) and writes a valid `submission.csv`.'
- What this solution (achieved 0.7074) has done: 'Your current approach is a deterministic “blend of 3 base submissions” derived from simple JPEG statistics; to move AUC upward toward 0.924 with minimal logic change, I keep the same feature extraction + z-scoring + 3-logit blending structure but make two small, high-impact tweaks. First, I compute z-scores robustly using median/MAD (less sensitive to outlier images) which typically improves ranking stability (AUC is ranking-based). Second, I calibrate the intercept automatically from the test prediction distribution to match a realistic melanoma prevalence (~1.76% from this competition), which improves probability scaling without changing ranking much but can help the blend behave more consistently. Everything still runs end-to-end, reads the same files/paths, and writes a valid `submission.csv`.'
- What this solution (achieved 0.76148) has done: 'The timeout is dominated by single-threaded JPEG decoding + per-image Python overhead in `image_features` for ~12k images (8k train + 4.1k test). I keep the exact same features and ridge/blending logic, but speed up extraction by (1) eliminating repeated PIL object conversions, (2) vectorizing grayscale/gradient computations more efficiently, (3) parallelizing feature extraction across CPU cores with deterministic ordering, and (4) avoiding unnecessary dtype conversions/copies. The model fitting and calibration remain identical; only the way features are computed (same math) is optimized. File paths and outputs remain unchanged.'
- What this solution (achieved 0.7772) has done: 'Your current AUC (0.76148) is far below the target (0.92466), so we should safely increase ranking signal without changing the overall “extract simple JPEG features → robust z-score → ridge-fit logits → sigmoid + blend → write submission.csv” core pipeline. The smallest high-impact change is to add a few additional cheap, deterministic image statistics (channel stds, saturation stats, and a simple center-vs-border contrast/“vignetting” measure) and include them in the same ridge models; this typically improves separability for this competition while keeping the same modeling approach. I keep the existing 3-model blend and intercept calibration, but fit scaling/weights on a larger deterministic train subset (still bounded for runtime) to reduce variance and move AUC upward. All paths remain unchanged and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.80325) has done: 'Your current score (0.7772) is well below the target (0.92466), so we should add a bit more ranking signal while keeping the exact same pipeline: extract cheap JPEG stats → robust z-score → three ridge logits → sigmoid + prevalence intercept calibration → 3-way blend → write `submission.csv`. The smallest likely win is to (1) make the training subset deterministic-but-more-representative by including all positives and a capped number of negatives (instead of the first N rows), and (2) add two very cheap, deterministic features (blue/red ratio and a simple “dark pixel fraction”) that often correlate with lesion appearance and improve AUC without changing the modeling approach. Everything else (ridge fitting, calibration, blending, file paths, output schema) stays the same, and the script still runs end-to-end and writes a valid submission.'
- What this solution (achieved 0.80123) has done: 'Your current AUC (0.80325) is still far below the target (0.92466), so we should add a small amount of extra ranking signal while keeping the same pipeline (cheap JPEG features → robust z-score → ridge logits → sigmoid + intercept calibration → 3-way blend). I make a minimal feature-only upgrade by adding a few very cheap, deterministic lesion-structure proxies (R/G and R/B ratios, green-red difference, bright pixel fraction, and a center-vs-border saturation delta), then include them in the same three ridge design matrices without changing the fitting method. This typically improves separability a bit for this competition while preserving your exact training approach and output semantics. All paths and the submission-writing/blending logic remain unchanged, and it still run end-to-end and produce `submission.csv`.'
- What this solution (achieved 0.80128) has done: 'We need to move your AUC up toward 0.9247 from 0.8012 (higher-is-better), but keep the same overall pipeline: extract cheap JPEG features → robust z-score → ridge-fit three logits → sigmoid + intercept calibration → 3-way blend → write submission.csv. The smallest likely gain without changing the training approach is to (1) make the ridge targets less “squashed” so the fitted logits preserve more ranking signal (AUC cares about ranking), and (2) very slightly enlarge the training fit subset (still bounded) to reduce variance while keeping runtime under control. I also add a tiny amount of numerically-stable standardization for the design matrices (feature-wise mean/scale computed on the fit subset) before ridge solving; it doesn’t change the model family and often improves conditioning and thus ranking quality. All paths, file outputs, and the submission schema remain identical.'
- What this solution (achieved 0.79074) has done: 'To move AUC upward toward the 0.9247 target without changing your core pipeline, I keep the exact same feature set, robust z-scoring, three ridge models, intercept calibration, and 3-way averaging blend. The minimal change is to improve the *ranking signal* by using out-of-fold (OOF) predictions on the same training subset when learning the ridge weights (a standard stacking safeguard): it reduces overfitting from “train-on-train” logits and typically improves generalization (AUC) without changing the model family. Concretely, I add a deterministic 5-fold patient-group split (to avoid leakage across the same patient) and fit each ridge on 4 folds while scoring the held-out fold; then I refit on all fit data for test predictions. Everything still runs end-to-end within the same constraints and writes a valid `submission.csv`.'
- What this solution (achieved 0.78891) has done: 'Your current AUC (0.79074) is well below the target (0.92466), so we should increase ranking signal with the smallest change that preserves the pipeline. I keep the exact same JPEG feature extraction, robust z-scoring, ridge fitting, OOF procedure, calibration, and 3-way blending, but fix a key mismatch: the intercept calibration is currently fit on **refit test logits** (`raw_*`) instead of the corresponding **OOF train logits**, which makes the calibration inconsistent and can hurt generalization. I change calibration to (1) learn the intercept on OOF train logits to match train prevalence, then (2) apply that same intercept to test logits—this keeps evaluation semantics identical while typically improving AUC. Everything still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.81342) has done: 'The timeout is dominated by per-image feature extraction (PIL open/resize + many NumPy ops) done for ~28k train + ~4k test images, plus avoidable overhead from per-image function calls and repeated intermediate arrays. I keep the exact same features and ridge/OOF logic, but speed it up by (1) extracting all 19 features in a single preallocated float32 array without creating many temporaries, (2) using PIL’s faster `Image.reduce()` (exact 2x downsampling steps) before the final resize to cut resize cost without changing the final size or algorithm, (3) using `ThreadPoolExecutor` with tuned chunksize and avoiding Python tuple packing/unpacking in the hot loop, and (4) setting thread env vars to prevent CPU oversubscription in NumPy linear algebra. All model math, feature definitions, folds, calibration, and blending remain identical.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

DATA_DIR_CANDIDATES = [
    "/kaggle/input/siim-isic-melanoma-classification",
    "/kaggle/data/siim-isic-melanoma-classification",
    "/kaggle/input",
    "/kaggle/data",
]


def find_file(filename):
    for base in DATA_DIR_CANDIDATES:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find {filename} in any of: {DATA_DIR_CANDIDATES}"
    )


def find_dir(relpath):
    for base in DATA_DIR_CANDIDATES:
        path = os.path.join(base, relpath)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        f"Could not find directory {relpath} in any of: {DATA_DIR_CANDIDATES}"
    )


test_csv_path = find_file("test.csv")
train_csv_path = find_file("train.csv")
sample_sub_path = find_file("sample_submission.csv")

test = pd.read_csv(test_csv_path)
train = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_sub_path)

assert "image_name" in test.columns
assert "image_name" in train.columns and "target" in train.columns
assert "image_name" in sample_sub.columns and "target" in sample_sub.columns

test = test.merge(sample_sub[["image_name"]], on="image_name", how="right")

JPEG_TEST_DIR = None
for candidate in ["jpeg/test", "siim-isic-melanoma-classification/jpeg/test"]:
    try:
        JPEG_TEST_DIR = find_dir(candidate)
        break
    except FileNotFoundError:
        pass
if JPEG_TEST_DIR is None:
    raise FileNotFoundError(
        "Could not locate jpeg/test directory under known data roots."
    )

JPEG_TRAIN_DIR = None
for candidate in ["jpeg/train", "siim-isic-melanoma-classification/jpeg/train"]:
    try:
        JPEG_TRAIN_DIR = find_dir(candidate)
        break
    except FileNotFoundError:
        pass
if JPEG_TRAIN_DIR is None:
    raise FileNotFoundError(
        "Could not locate jpeg/train directory under known data roots."
    )

print("Using JPEG_TEST_DIR:", JPEG_TEST_DIR)
print("Using JPEG_TRAIN_DIR:", JPEG_TRAIN_DIR)
print(
    "test rows:",
    test.shape[0],
    "train rows:",
    train.shape[0],
    "sample_sub rows:",
    sample_sub.shape[0],
)




## === cell 1
from PIL import Image
from concurrent.futures import ThreadPoolExecutor


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def image_features_to_out(image_name, jpg_dir, out_vec, size=128):
    path = os.path.join(jpg_dir, f"{image_name}.jpg")
    with Image.open(path) as im:
        im = im.convert("RGB")

        if size is not None:
            w, h = im.size
            if w >= 2 * size and h >= 2 * size:
                rf = min(w // size, h // size)
                pow2 = 1
                while (pow2 << 1) <= rf:
                    pow2 <<= 1
                if pow2 >= 2:
                    im = im.reduce(pow2)
            im = im.resize((size, size), resample=Image.BICUBIC)

        arr_u8 = np.asarray(im, dtype=np.uint8)

    arr = arr_u8.astype(np.float32) * (1.0 / 255.0)  # (H,W,3)
    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    mean_gray = gray.mean(dtype=np.float32)
    std_gray = gray.std(dtype=np.float32)

    mean_r = r.mean(dtype=np.float32)
    mean_g = g.mean(dtype=np.float32)
    mean_b = b.mean(dtype=np.float32)
    mean_rgb = arr.mean(dtype=np.float32) + 1e-6
    redness = mean_r / mean_rgb

    gx = np.abs(np.diff(gray, axis=1)).mean(dtype=np.float32)
    gy = np.abs(np.diff(gray, axis=0)).mean(dtype=np.float32)
    grad = gx + gy

    std_r = r.std(dtype=np.float32)
    std_g = g.std(dtype=np.float32)
    std_b = b.std(dtype=np.float32)

    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = mx - mn
    mean_sat = sat.mean(dtype=np.float32)
    std_sat = sat.std(dtype=np.float32)

    h, w = gray.shape
    h0, h1 = h // 4, (3 * h) // 4
    w0, w1 = w // 4, (3 * w) // 4
    center = gray[h0:h1, w0:w1]
    center_mean = center.mean(dtype=np.float32)
    total_sum = gray.sum(dtype=np.float32)
    center_sum = center.sum(dtype=np.float32)
    total_n = np.float32(h * w)
    center_n = np.float32((h1 - h0) * (w1 - w0))
    border_mean = (total_sum - center_sum) / (total_n - center_n + np.float32(1e-6))
    center_delta = center_mean - border_mean

    br_ratio = (mean_b + np.float32(1e-6)) / (mean_r + np.float32(1e-6))
    dark_frac = (gray < np.float32(0.20)).mean(dtype=np.float32)

    rg_ratio = (mean_r + np.float32(1e-6)) / (mean_g + np.float32(1e-6))
    rb_ratio = (mean_r + np.float32(1e-6)) / (mean_b + np.float32(1e-6))
    gr_diff = mean_g - mean_r
    bright_frac = (gray > np.float32(0.85)).mean(dtype=np.float32)

    center_sat = sat[h0:h1, w0:w1]
    center_sat_mean = center_sat.mean(dtype=np.float32)
    sat_total_sum = sat.sum(dtype=np.float32)
    sat_center_sum = center_sat.sum(dtype=np.float32)
    border_sat = (sat_total_sum - sat_center_sum) / (
        total_n - center_n + np.float32(1e-6)
    )
    sat_center_delta = center_sat_mean - border_sat

    g0 = gray
    lap = (
        -4.0 * g0
        + np.roll(g0, 1, axis=0)
        + np.roll(g0, -1, axis=0)
        + np.roll(g0, 1, axis=1)
        + np.roll(g0, -1, axis=1)
    )
    lap_energy = (lap * lap).mean(dtype=np.float32)

    k = (
        g0
        + np.roll(g0, 1, axis=0)
        + np.roll(g0, -1, axis=0)
        + np.roll(g0, 1, axis=1)
        + np.roll(g0, -1, axis=1)
        + np.roll(np.roll(g0, 1, axis=0), 1, axis=1)
        + np.roll(np.roll(g0, 1, axis=0), -1, axis=1)
        + np.roll(np.roll(g0, -1, axis=0), 1, axis=1)
        + np.roll(np.roll(g0, -1, axis=0), -1, axis=1)
    ) * (1.0 / 9.0)
    local_contrast = np.abs(g0 - k).mean(dtype=np.float32)

    out_vec[0] = mean_gray
    out_vec[1] = std_gray
    out_vec[2] = redness
    out_vec[3] = grad
    out_vec[4] = std_r
    out_vec[5] = std_g
    out_vec[6] = std_b
    out_vec[7] = mean_sat
    out_vec[8] = std_sat
    out_vec[9] = center_delta
    out_vec[10] = br_ratio
    out_vec[11] = dark_frac
    out_vec[12] = rg_ratio
    out_vec[13] = rb_ratio
    out_vec[14] = gr_diff
    out_vec[15] = bright_frac
    out_vec[16] = sat_center_delta
    out_vec[17] = lap_energy
    out_vec[18] = local_contrast


def robust_location_scale(x):
    x = x.astype(np.float32, copy=False)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)) + 1e-6)
    scale = float(1.4826 * mad + 1e-6)
    return med, scale


def apply_robust_zscore(x, med, scale):
    x = x.astype(np.float32, copy=False)
    return (x - med) / scale


def calibrate_intercept(base_logit, target_prevalence):
    lo, hi = -10.0, 10.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        m = float(sigmoid(base_logit + mid).mean())
        if m > target_prevalence:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


train_names_all = train["image_name"].values
train_y_all = train["target"].astype(np.float32).values

MAX_TRAIN_FIT = 24000

pos_idx = np.flatnonzero(train_y_all >= 0.5).astype(np.int32)
neg_idx = np.flatnonzero(train_y_all < 0.5).astype(np.int32)

names_bytes = train_names_all.astype("S")
name_hash = (
    np.frombuffer(b"".join(names_bytes.tolist()), dtype=np.uint8).sum().astype(np.int64)
)
rng = np.random.RandomState(int(name_hash % (2**31 - 1)))

neg_take = max(0, min(len(neg_idx), MAX_TRAIN_FIT - len(pos_idx)))
if neg_take > 0:
    neg_sel = rng.choice(neg_idx, size=neg_take, replace=False).astype(np.int32)
    fit_idx = np.concatenate([pos_idx, neg_sel], axis=0)
else:
    fit_idx = pos_idx
fit_idx = np.sort(fit_idx)

train_fit_df = train.iloc[fit_idx].reset_index(drop=True)
train_names = train_fit_df["image_name"].tolist()
train_y = train_fit_df["target"].astype(np.float32).values

print(
    "Fitting feature scaling/weights on train subset:",
    len(train_names),
    "of",
    len(train_names_all),
    "| positives included:",
    int(len(pos_idx)),
)

test_names = test["image_name"].values.tolist()


def _extract_batch(names, jpg_dir, size, max_workers):
    n = len(names)
    out = np.empty((n, 19), dtype=np.float32)

    def _one(i):
        tmp = np.empty((19,), dtype=np.float32)
        image_features_to_out(names[i], jpg_dir, tmp, size=size)
        out[i, :] = tmp

    chunksize = 128
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for k, _ in enumerate(ex.map(_one, range(n), chunksize=chunksize), start=1):
            if k % 2000 == 0 or k == n:
                print(f"Processed {k}/{n} images from {os.path.basename(jpg_dir)}")
    return out


CPU_CNT = os.cpu_count() or 2
MAX_WORKERS = min(8, max(2, CPU_CNT))

tr_feats = _extract_batch(
    train_names, JPEG_TRAIN_DIR, size=128, max_workers=MAX_WORKERS
)
te_feats = _extract_batch(test_names, JPEG_TEST_DIR, size=128, max_workers=MAX_WORKERS)

(
    tr_mg,
    tr_sg,
    tr_rd,
    tr_gr,
    tr_sr,
    tr_sgch,
    tr_sb,
    tr_msat,
    tr_ssat,
    tr_cdelta,
    tr_brr,
    tr_dark,
    tr_rgr,
    tr_rbr,
    tr_grdiff,
    tr_bright,
    tr_scdelta,
    tr_lap,
    tr_lcon,
) = (tr_feats[:, i] for i in range(19))

(
    te_mg,
    te_sg,
    te_rd,
    te_gr,
    te_sr,
    te_sgch,
    te_sb,
    te_msat,
    te_ssat,
    te_cdelta,
    te_brr,
    te_dark,
    te_rgr,
    te_rbr,
    te_grdiff,
    te_bright,
    te_scdelta,
    te_lap,
    te_lcon,
) = (te_feats[:, i] for i in range(19))

scalers = {}
for name, vec in [
    ("mg", tr_mg),
    ("sg", tr_sg),
    ("rd", tr_rd),
    ("gr", tr_gr),
    ("sr", tr_sr),
    ("sgch", tr_sgch),
    ("sb", tr_sb),
    ("msat", tr_msat),
    ("ssat", tr_ssat),
    ("cdelta", tr_cdelta),
    ("brr", tr_brr),
    ("dark", tr_dark),
    ("rgr", tr_rgr),
    ("rbr", tr_rbr),
    ("grdiff", tr_grdiff),
    ("bright", tr_bright),
    ("scdelta", tr_scdelta),
    ("lap", tr_lap),
    ("lcon", tr_lcon),
]:
    med, scale = robust_location_scale(vec)
    scalers[name] = (med, scale)

tr_z = {}
te_z = {}
for name, trv, tev in [
    ("mg", tr_mg, te_mg),
    ("sg", tr_sg, te_sg),
    ("rd", tr_rd, te_rd),
    ("gr", tr_gr, te_gr),
    ("sr", tr_sr, te_sr),
    ("sgch", tr_sgch, te_sgch),
    ("sb", tr_sb, te_sb),
    ("msat", tr_msat, te_msat),
    ("ssat", tr_ssat, te_ssat),
    ("cdelta", tr_cdelta, te_cdelta),
    ("brr", tr_brr, te_brr),
    ("dark", tr_dark, te_dark),
    ("rgr", tr_rgr, te_rgr),
    ("rbr", tr_rbr, te_rbr),
    ("grdiff", tr_grdiff, te_grdiff),
    ("bright", tr_bright, te_bright),
    ("scdelta", tr_scdelta, te_scdelta),
    ("lap", tr_lap, te_lap),
    ("lcon", tr_lcon, te_lcon),
]:
    med, scale = scalers[name]
    tr_z[name] = apply_robust_zscore(trv, med, scale)
    te_z[name] = apply_robust_zscore(tev, med, scale)


def _standardize_design_mtx(X_tr, X_te):
    mu = X_tr.mean(axis=0, dtype=np.float32)
    sd = X_tr.std(axis=0, dtype=np.float32)
    sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)
    X_tr2 = ((X_tr - mu) / sd).astype(np.float32, copy=False)
    X_te2 = ((X_te - mu) / sd).astype(np.float32, copy=False)
    return X_tr2, X_te2


def fit_ridge_weights(X, y, l2=5.0):
    y = y.astype(np.float32, copy=False)
    y_soft = np.clip(0.005 + 0.99 * y, 1e-5, 1 - 1e-5).astype(np.float32, copy=False)
    t = np.log(y_soft / (1.0 - y_soft)).astype(np.float32, copy=False)

    XtX = (X.T @ X).astype(np.float32, copy=False)
    Xtt = (X.T @ t).astype(np.float32, copy=False)
    A = XtX + (l2 * np.eye(X.shape[1], dtype=np.float32))
    w = np.linalg.solve(A, Xtt).astype(np.float32, copy=False)
    return w


def make_group_folds(groups, n_splits=5, seed=13):
    groups = np.asarray(groups)
    uniq = pd.unique(groups)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(uniq))
    uniq = uniq[perm]
    fold_id = np.arange(len(uniq)) % n_splits
    grp2fold = {g: int(f) for g, f in zip(uniq, fold_id)}
    return np.array([grp2fold[g] for g in groups], dtype=np.int32)


ones_tr = np.ones_like(tr_z["sg"], dtype=np.float32)
ones_te = np.ones_like(te_z["sg"], dtype=np.float32)

X_a_tr = np.stack(
    [
        tr_z["sg"],
        tr_z["gr"],
        tr_z["mg"],
        tr_z["rd"],
        tr_z["msat"],
        tr_z["cdelta"],
        tr_z["brr"],
        tr_z["rgr"],
        tr_z["scdelta"],
        tr_z["lap"],
        tr_z["lcon"],
        ones_tr,
    ],
    axis=1,
).astype(np.float32, copy=False)

X_b_tr = np.stack(
    [
        tr_z["gr"],
        tr_z["rd"],
        tr_z["mg"],
        tr_z["sg"],
        tr_z["ssat"],
        tr_z["sr"],
        tr_z["dark"],
        tr_z["bright"],
        tr_z["grdiff"],
        tr_z["lap"],
        tr_z["lcon"],
        ones_tr,
    ],
    axis=1,
).astype(np.float32, copy=False)

X_c_tr = np.stack(
    [
        tr_z["sg"],
        tr_z["rd"],
        tr_z["mg"],
        tr_z["gr"],
        tr_z["sb"],
        tr_z["sgch"],
        tr_z["dark"],
        tr_z["rbr"],
        tr_z["scdelta"],
        tr_z["lap"],
        tr_z["lcon"],
        ones_tr,
    ],
    axis=1,
).astype(np.float32, copy=False)

X_a_te = np.stack(
    [
        te_z["sg"],
        te_z["gr"],
        te_z["mg"],
        te_z["rd"],
        te_z["msat"],
        te_z["cdelta"],
        te_z["brr"],
        te_z["rgr"],
        te_z["scdelta"],
        te_z["lap"],
        te_z["lcon"],
        ones_te,
    ],
    axis=1,
).astype(np.float32, copy=False)

X_b_te = np.stack(
    [
        te_z["gr"],
        te_z["rd"],
        te_z["mg"],
        te_z["sg"],
        te_z["ssat"],
        te_z["sr"],
        te_z["dark"],
        te_z["bright"],
        te_z["grdiff"],
        te_z["lap"],
        te_z["lcon"],
        ones_te,
    ],
    axis=1,
).astype(np.float32, copy=False)

X_c_te = np.stack(
    [
        te_z["sg"],
        te_z["rd"],
        te_z["mg"],
        te_z["gr"],
        te_z["sb"],
        te_z["sgch"],
        te_z["dark"],
        te_z["rbr"],
        te_z["scdelta"],
        te_z["lap"],
        te_z["lcon"],
        ones_te,
    ],
    axis=1,
).astype(np.float32, copy=False)

X_a_tr, X_a_te = _standardize_design_mtx(X_a_tr, X_a_te)
X_b_tr, X_b_te = _standardize_design_mtx(X_b_tr, X_b_te)
X_c_tr, X_c_te = _standardize_design_mtx(X_c_tr, X_c_te)

groups = train_fit_df["patient_id"].fillna("NA").astype(str).values
folds = make_group_folds(groups, n_splits=5, seed=13)


def oof_and_refit_predict(X_tr, y, X_te, folds, l2):
    oof = np.empty((X_tr.shape[0],), dtype=np.float32)
    for f in range(int(folds.max()) + 1):
        tr_idx = np.flatnonzero(folds != f)
        va_idx = np.flatnonzero(folds == f)
        w = fit_ridge_weights(X_tr[tr_idx], y[tr_idx], l2=l2)
        oof[va_idx] = (X_tr[va_idx] @ w).astype(np.float32, copy=False)
    w_full = fit_ridge_weights(X_tr, y, l2=l2)
    te_raw = (X_te @ w_full).astype(np.float32, copy=False)
    return oof, te_raw


oof_a, raw_a = oof_and_refit_predict(X_a_tr, train_y, X_a_te, folds, l2=8.0)
oof_b, raw_b = oof_and_refit_predict(X_b_tr, train_y, X_b_te, folds, l2=8.0)
oof_c, raw_c = oof_and_refit_predict(X_c_tr, train_y, X_c_te, folds, l2=8.0)

train_prev = float(train["target"].mean())

delta_a = calibrate_intercept(oof_a, target_prevalence=train_prev)
delta_b = calibrate_intercept(oof_b, target_prevalence=train_prev)
delta_c = calibrate_intercept(oof_c, target_prevalence=train_prev)

pred_a = sigmoid(raw_a + delta_a).astype(np.float32, copy=False)
pred_b = sigmoid(raw_b + delta_b).astype(np.float32, copy=False)
pred_c = sigmoid(raw_c + delta_c).astype(np.float32, copy=False)

a = pd.DataFrame({"image_name": test_names, "target": pred_a})
b = pd.DataFrame({"image_name": test_names, "target": pred_b})
c = pd.DataFrame({"image_name": test_names, "target": pred_c})

a.to_csv("b0-b3-b4.csv", index=False)
b.to_csv("b5.csv", index=False)
c.to_csv("b7.csv", index=False)

print(a.head())
print("Train prevalence:", train_prev)
print("Mean preds:", float(pred_a.mean()), float(pred_b.mean()), float(pred_c.mean()))




## === cell 2
a = pd.read_csv("b0-b3-b4.csv")
b = pd.read_csv("b5.csv")
c = pd.read_csv("b7.csv")

m = (
    sample_sub[["image_name"]]
    .merge(a, on="image_name", how="left", suffixes=("", "_a"))
    .merge(b, on="image_name", how="left", suffixes=("", "_b"))
    .merge(c, on="image_name", how="left", suffixes=("", "_c"))
)

for col in ["target", "target_b", "target_c"]:
    if col not in m.columns:
        raise RuntimeError(f"Missing blended column: {col}")
    m[col] = m[col].fillna(0.02)

pred = (m["target"].values + m["target_b"].values + m["target_c"].values) / 3.0
pred = np.clip(pred, 0.0, 1.0)

submission = pd.DataFrame(
    {"image_name": m["image_name"].values, "target": pred.astype(np.float32)}
)
submission = submission[sample_sub.columns]  # enforce correct column order

assert submission.shape[0] == sample_sub.shape[0]
assert list(submission.columns) == ["image_name", "target"]
assert submission["target"].between(0, 1).all()

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Submission mean target:", float(submission["target"].mean()))
