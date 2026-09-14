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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.8612962593545803

# 6. Current score

0.77129

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the failing `pip install tensorflow-addons` and the `tensorflow_addons` import that triggers the protobuf `MessageFactory.GetPrototype` error, because it is not used anywhere in your inference-only pipeline. I also make the script self-contained by re-importing all required libraries in the first cell so `tf`, `pd`, etc. are always defined, and I fix the broken cell numbering/order while preserving your model architecture and preprocessing logic. Since your code references a non-existent external weights file, I instead use EfficientNetB1 with ImageNet weights (same architecture) so the notebook can produce meaningful predictions and a valid `submission.csv` end-to-end. Finally, I ensure inference uses the same normalization as your `preprocessing` function (divide by 255) and read `test.csv` (not `sample_submission.csv`) for the id list, keeping the required submission columns unchanged.'
- What this solution (achieved 0.71135) has done: 'The protobuf `MessageFactory.GetPrototype` crash happens during TensorFlow import due to an incompatible protobuf runtime in this environment, so the only viable way to make this run end-to-end is to remove TensorFlow usage entirely. To keep the pipeline’s semantics (image preprocessing → model → 0–4 prediction → submission) while remaining minimal and stable, I replace the TF model inference with a lightweight OpenCV feature extractor plus a scikit-learn multiclass classifier trained on `train.csv`, which is available locally. This also resolves the current score of 0.0 (from a hard crash) by producing real predictions aligned to the evaluation labels (0–4). The script keeps your Ben Graham-style preprocessing and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.72139) has done: 'Your current pipeline is already producing a valid submission and is far below the target (0.71135 vs 0.8613), so we should make small, legitimate improvements that better align the predictions with quadratic weighted kappa without changing the overall approach (Ben Graham preprocessing → simple features → scikit-learn classifier). The most direct, minimal lever here is handling the strong class imbalance: logistic regression is likely biased toward the majority class, which hurts kappa. I add `class_weight="balanced"` to the same multinomial logistic regression (same model family, same training loop) to reduce that bias. I also switch the solver explicitly to `lbfgs` (standard for multinomial) to make optimization more stable without changing semantics, keeping everything else identical and still writing `submission.csv`.'
- What this solution (achieved 0.7389) has done: 'The timeout is dominated by single-threaded image preprocessing/feature extraction over ~3.3k train + 367 test images and by redundant per-image conversions/reshapes inside feature extraction. I keep the exact same preprocessing, features, model, and threshold optimization, but make feature building faster by (1) parallelizing image read+preprocess+feature extraction with a deterministic thread pool (OpenCV releases the GIL), (2) reducing per-image overhead by computing means/stds without extra reshapes and by using OpenCV histograms, and (3) avoiding repeated constant allocations (e.g., class indices). These changes are mathematically equivalent (same operations, just reorganized/cached) and should cut wall time substantially while preserving predictions up to negligible float noise.'
- What this solution (achieved 0.74143) has done: 'We’re substantially below the target (0.7389 vs 0.8613), so the smallest legitimate lever to move QWK upward without changing your core pipeline is to calibrate the final discrete mapping. Your current thresholds are tuned on a single 80/20 split and can be noisy; instead, we learn thresholds out-of-fold (OOF) across a few stratified folds using the same model and the same “expected class” continuous score, then average those thresholds for a stabler mapping. This keeps the same feature extraction, the same LogisticRegression pipeline, and the same threshold-search method, but reduces variance and usually improves QWK on this competition. Finally, we refit once on full data and apply the averaged thresholds to test, still producing a valid `submission.csv`.'
- What this solution (achieved 0.76169) has done: 'Your current score (0.74143) is well below the target (0.86130), so we should make a small, legitimate change that’s likely to improve QWK without changing your core pipeline (Ben preprocessing → handcrafted features → multinomial LogisticRegression → thresholding). The weakest link is that LogisticRegression is optimizing log-loss, while QWK is evaluated after discretization; a minimal improvement is to tune the LR regularization strength `C` via cross-validated QWK on the same OOF thresholded predictions you already compute, then refit once on full data using that chosen `C`. This preserves the exact model family/training approach and uses the same folds you already run, but typically yields a noticeable bump by better calibrating probability separation. I keep runtime bounded by evaluating a small fixed grid of `C` values and reusing the already-built features.'
- What this solution (achieved 0.76743) has done: 'We’re still well below the target (0.7617 vs 0.8613), so the smallest legitimate change likely to move QWK upward is improving how we learn the final 0–4 discretization without changing your model/feature pipeline. Right now thresholds are fit fold-by-fold only on each fold’s validation data; I instead generate out-of-fold (OOF) expected scores for every training sample (same LogisticRegression, same folds), then fit one global set of thresholds on those OOF scores (less variance, better alignment to QWK). I keep your existing C-grid selection, but use the OOF expected vector produced during that loop to choose thresholds and compute CV QWK consistently. Everything else (Ben preprocessing, handcrafted features, multinomial LR, no extra packages) stays the same and still writes `submission.csv`.'
- What this solution (achieved 0.77067) has done: 'We’re far below the target (0.767 vs 0.861, higher-is-better), so we should make a small, legitimate improvement that better aligns your final discrete predictions with QWK without changing the core pipeline (Ben preprocessing → handcrafted features → multinomial LogisticRegression → thresholding). The most direct lever is to replace the coarse grid-search threshold fitting with a smooth, bounded optimizer (coordinate descent + golden-section search) on the same OOF expected scores; this keeps the same “expected class → 4 thresholds → 0–4” semantics but typically finds better thresholds than a 200-point grid. To avoid overfitting and keep the change minimal, we keep the exact same CV scheme and C-grid selection, but fit thresholds with a slightly stronger search and more iterations only at the OOF/global level. Everything still runs end-to-end, uses the same features/model family, and writes a valid `submission.csv`.'
- What this solution (achieved 0.77129) has done: 'We’re still below the target (0.77067 vs 0.86130, higher-is-better), so the minimal change most likely to lift QWK without changing your core pipeline is to adjust only the final discretization step. I keep your exact preprocessing, feature vector, LogisticRegression model family, CV scheme, and “expected class → 4 thresholds → 0–4” mapping, but fit thresholds with a small random-restart coordinate descent (still bounded/monotone thresholds) to avoid local optima from a single quantile initialization. This only changes how thresholds are optimized (same objective: maximize QWK on OOF expected scores), which is the most direct lever for this metric while keeping everything else intact. The script still run end-to-end within the same paths and write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import gc
import cv2
import numpy as np
import pandas as pd

np.random.seed(42)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"



## === cell 1
"""
    Config
"""
IMG_SIZE = 224
BATCH_SIZE = 16  # kept for compatibility; not used in this non-TF pipeline


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def load_ben_color(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image).astype("uint8")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image


"""
    Preprocessing for ImageDataGenerator since ImageDataGenerator reads images in rgb mode, while opencv in bgr
"""


def preprocessing(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = crop_image_from_gray(image).astype("uint8")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image.astype("float32") / 255.0




## === cell 2
def extract_features_from_ben_rgb(img_rgb_uint8):
    img_u8 = img_rgb_uint8

    img_f = img_u8.astype(np.float32) * (1.0 / 255.0)
    mean_rgb = img_f.mean(axis=(0, 1))
    std_rgb = img_f.std(axis=(0, 1))

    hsv_u8 = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
    hsv = hsv_u8.astype(np.float32) * (1.0 / 255.0)
    mean_hsv = hsv.mean(axis=(0, 1))
    std_hsv = hsv.std(axis=(0, 1))

    gray_u8 = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    gray = gray_u8.astype(np.float32) * (1.0 / 255.0)

    edges_u8 = cv2.Canny(gray_u8, 50, 150)
    edges = edges_u8.astype(np.float32) * (1.0 / 255.0)

    n_pix = float(gray.size)
    bin_w_gray = 1.0 / 16.0
    bin_w_edge = 1.0 / 8.0

    gray_hist = (
        cv2.calcHist([gray], [0], None, [16], [0.0, 1.0])
        .reshape(-1)
        .astype(np.float32, copy=False)
    )
    edge_hist = (
        cv2.calcHist([edges], [0], None, [8], [0.0, 1.0])
        .reshape(-1)
        .astype(np.float32, copy=False)
    )

    gray_hist = gray_hist / (n_pix * bin_w_gray)
    edge_hist = edge_hist / (n_pix * bin_w_edge)

    thumb_u8 = cv2.resize(img_u8, (16, 16), interpolation=cv2.INTER_AREA)
    thumb_flat = (thumb_u8.astype(np.float32) * (1.0 / 255.0)).reshape(-1)

    feat = np.concatenate(
        [
            mean_rgb.astype(np.float32, copy=False),
            std_rgb.astype(np.float32, copy=False),
            mean_hsv.astype(np.float32, copy=False),
            std_hsv.astype(np.float32, copy=False),
            gray_hist,
            edge_hist,
            thumb_flat.astype(np.float32, copy=False),
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    return feat




## === cell 3
DATA_ROOT_CANDIDATES = [
    "/kaggle/input/aptos2019-blindness-detection",
    "../input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(p):
        DATA_ROOT = p
        break
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not find aptos2019-blindness-detection dataset directory in expected locations."
    )

train_csv_path = os.path.join(DATA_ROOT, "train.csv")
test_csv_path = os.path.join(DATA_ROOT, "test.csv")
sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")
train_img_dir = os.path.join(DATA_ROOT, "train_images")
test_img_dir = os.path.join(DATA_ROOT, "test_images")

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
sub_df = pd.read_csv(sample_sub_path)

sub_df = sub_df.drop(columns=["diagnosis"], errors="ignore")
sub_df = sub_df.merge(test_df[["id_code"]], on="id_code", how="right")

print("DATA_ROOT:", DATA_ROOT)
print(
    "Train rows:",
    len(train_df),
    "Test rows:",
    len(test_df),
    "Submission rows:",
    len(sub_df),
)



## === cell 4
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from concurrent.futures import ThreadPoolExecutor

_QWK_W_CACHE = {}


def _qwk_weights(n_classes):
    W = _QWK_W_CACHE.get(n_classes)
    if W is None:
        i = np.arange(n_classes, dtype=np.float64)
        W = ((i[:, None] - i[None, :]) ** 2) / ((n_classes - 1) ** 2)
        _QWK_W_CACHE[n_classes] = W
    return W


def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_true = np.clip(y_true, 0, n_classes - 1)
    y_pred = np.clip(y_pred, 0, n_classes - 1)

    idx = y_true * n_classes + y_pred
    O = (
        np.bincount(idx, minlength=n_classes * n_classes)
        .reshape(n_classes, n_classes)
        .astype(np.float64, copy=False)
    )

    act_hist = np.bincount(y_true, minlength=n_classes).astype(np.float64, copy=False)
    pred_hist = np.bincount(y_pred, minlength=n_classes).astype(np.float64, copy=False)

    E = np.outer(act_hist, pred_hist)
    E = E / E.sum() * O.sum()

    W = _qwk_weights(n_classes)

    num = (W * O).sum()
    den = (W * E).sum()
    return 1.0 - num / den if den > 0 else 0.0


def apply_thresholds(x, thresholds):
    t0, t1, t2, t3 = thresholds
    return np.where(
        x < t0,
        0,
        np.where(x < t1, 1, np.where(x < t2, 2, np.where(x < t3, 3, 4))),
    ).astype(int)


def fit_thresholds_qwk(x_cont, y_true, n_classes=5, n_iters=10):
    x_cont = np.asarray(x_cont, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)

    qs = np.quantile(x_cont, [0.2, 0.4, 0.6, 0.8]).tolist()
    thr = np.array(sorted(qs), dtype=np.float64)

    xmin, xmax = float(x_cont.min()), float(x_cont.max())

    W = _qwk_weights(n_classes)
    y_true_clip = np.clip(y_true, 0, n_classes - 1)
    act_hist = np.bincount(y_true_clip, minlength=n_classes).astype(np.float64)
    N = float(y_true.size)

    def qwk_fast(y_pred_int):
        y_pred_int = np.asarray(y_pred_int, dtype=np.int64)
        y_pred_int = np.clip(y_pred_int, 0, n_classes - 1)

        idx = y_true_clip * n_classes + y_pred_int
        O = (
            np.bincount(idx, minlength=n_classes * n_classes)
            .reshape(n_classes, n_classes)
            .astype(np.float64, copy=False)
        )

        pred_hist = np.bincount(y_pred_int, minlength=n_classes).astype(
            np.float64, copy=False
        )
        E = np.outer(act_hist, pred_hist)
        E = E / E.sum() * N

        num = (W * O).sum()
        den = (W * E).sum()
        return 1.0 - num / den if den > 0 else 0.0

    def score_for_thr(thr_vec):
        return qwk_fast(apply_thresholds(x_cont, thr_vec))

    best_thr = thr.copy()
    best_score = score_for_thr(best_thr)

    phi = (1.0 + 5.0**0.5) / 2.0  # golden ratio

    def golden_max_1d(k, low, high, base_thr, it=28):
        if not (low < high):
            return base_thr[k], score_for_thr(base_thr)

        a, b = float(low), float(high)
        c = b - (b - a) / phi
        d = a + (b - a) / phi

        thr_c = base_thr.copy()
        thr_d = base_thr.copy()
        thr_c[k] = c
        thr_d[k] = d

        fc = score_for_thr(thr_c)
        fd = score_for_thr(thr_d)

        for _ in range(it):
            if fc > fd:
                b = d
                d = c
                fd = fc
                c = b - (b - a) / phi
                thr_c = base_thr.copy()
                thr_c[k] = c
                fc = score_for_thr(thr_c)
            else:
                a = c
                c = d
                fc = fd
                d = a + (b - a) / phi
                thr_d = base_thr.copy()
                thr_d[k] = d
                fd = score_for_thr(thr_d)

        if fc >= fd:
            return c, fc
        else:
            return d, fd

    for _ in range(n_iters):
        improved_any = False
        for k in range(4):
            low = xmin if k == 0 else best_thr[k - 1] + 1e-6
            high = xmax if k == 3 else best_thr[k + 1] - 1e-6
            if low >= high:
                continue

            best_val, best_sc = golden_max_1d(k, low, high, best_thr, it=26)
            if best_sc > best_score + 1e-12:
                best_thr[k] = best_val
                best_score = best_sc
                improved_any = True

        best_thr = np.sort(best_thr)

        if not improved_any:
            break

    return best_thr, float(best_score)


def fit_thresholds_qwk_multistart(
    x_cont, y_true, n_classes=5, n_iters=10, n_starts=7, jitter_frac=0.03, seed=42
):
    x_cont = np.asarray(x_cont, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)

    rng = np.random.default_rng(seed)
    xmin, xmax = float(x_cont.min()), float(x_cont.max())
    span = max(1e-9, xmax - xmin)

    best_thr, best_score = fit_thresholds_qwk(
        x_cont, y_true, n_classes=n_classes, n_iters=n_iters
    )

    base_q = np.quantile(x_cont, [0.2, 0.4, 0.6, 0.8]).astype(np.float64)
    for s in range(n_starts - 1):
        jitter = rng.normal(loc=0.0, scale=jitter_frac * span, size=4)
        thr0 = np.sort(np.clip(base_q + jitter, xmin, xmax))

        def _fit_from_init(thr_init):
            thr = np.array(sorted(thr_init.tolist()), dtype=np.float64)

            W = _qwk_weights(n_classes)
            y_true_clip = np.clip(y_true, 0, n_classes - 1)
            act_hist = np.bincount(y_true_clip, minlength=n_classes).astype(np.float64)
            N = float(y_true.size)

            def qwk_fast(y_pred_int):
                y_pred_int = np.asarray(y_pred_int, dtype=np.int64)
                y_pred_int = np.clip(y_pred_int, 0, n_classes - 1)

                idx = y_true_clip * n_classes + y_pred_int
                O = (
                    np.bincount(idx, minlength=n_classes * n_classes)
                    .reshape(n_classes, n_classes)
                    .astype(np.float64, copy=False)
                )

                pred_hist = np.bincount(y_pred_int, minlength=n_classes).astype(
                    np.float64, copy=False
                )
                E = np.outer(act_hist, pred_hist)
                E = E / E.sum() * N

                num = (W * O).sum()
                den = (W * E).sum()
                return 1.0 - num / den if den > 0 else 0.0

            def score_for_thr(thr_vec):
                return qwk_fast(apply_thresholds(x_cont, thr_vec))

            best_thr_local = thr.copy()
            best_score_local = score_for_thr(best_thr_local)

            phi = (1.0 + 5.0**0.5) / 2.0

            def golden_max_1d(k, low, high, base_thr, it=28):
                if not (low < high):
                    return base_thr[k], score_for_thr(base_thr)

                a, b = float(low), float(high)
                c = b - (b - a) / phi
                d = a + (b - a) / phi

                thr_c = base_thr.copy()
                thr_d = base_thr.copy()
                thr_c[k] = c
                thr_d[k] = d

                fc = score_for_thr(thr_c)
                fd = score_for_thr(thr_d)

                for _ in range(it):
                    if fc > fd:
                        b = d
                        d = c
                        fd = fc
                        c = b - (b - a) / phi
                        thr_c = base_thr.copy()
                        thr_c[k] = c
                        fc = score_for_thr(thr_c)
                    else:
                        a = c
                        c = d
                        fc = fd
                        d = a + (b - a) / phi
                        thr_d = base_thr.copy()
                        thr_d[k] = d
                        fd = score_for_thr(thr_d)

                if fc >= fd:
                    return c, fc
                else:
                    return d, fd

            for _ in range(n_iters):
                improved_any = False
                for k in range(4):
                    low = xmin if k == 0 else best_thr_local[k - 1] + 1e-6
                    high = xmax if k == 3 else best_thr_local[k + 1] - 1e-6
                    if low >= high:
                        continue

                    best_val, best_sc = golden_max_1d(
                        k, low, high, best_thr_local, it=26
                    )
                    if best_sc > best_score_local + 1e-12:
                        best_thr_local[k] = best_val
                        best_score_local = best_sc
                        improved_any = True

                best_thr_local = np.sort(best_thr_local)
                if not improved_any:
                    break

            return best_thr_local, float(best_score_local)

        thr_s, sc_s = _fit_from_init(thr0)
        if sc_s > best_score + 1e-12:
            best_thr, best_score = thr_s, sc_s

    return np.sort(best_thr), float(best_score)


MAX_TRAIN_SAMPLES = None  # use all

train_ids_all = train_df["id_code"].values
y_all = train_df["diagnosis"].astype(int).values

if MAX_TRAIN_SAMPLES is not None and len(train_ids_all) > MAX_TRAIN_SAMPLES:
    train_ids_all = train_ids_all[:MAX_TRAIN_SAMPLES]
    y_all = y_all[:MAX_TRAIN_SAMPLES]


def build_features(ids, labels=None, img_dir=None, report_every=500, tag="train"):
    n = len(ids)
    feat_dim = 804
    X = np.empty((n, feat_dim), dtype=np.float32)
    y_out = np.empty((n,), dtype=np.int64) if labels is not None else None

    def _one(i_img_id):
        i, img_id = i_img_id
        img_path = os.path.join(img_dir, f"{img_id}.png")
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return i, None
        img_rgb = load_ben_color(img_bgr)
        feat = extract_features_from_ben_rgb(img_rgb)
        lab = int(labels[i]) if labels is not None else None
        return i, (feat, lab)

    max_workers = min(8, (os.cpu_count() or 4))
    bad = 0
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, out in ex.map(_one, enumerate(ids), chunksize=32):
            if out is None:
                bad += 1
            else:
                feat, lab = out
                X[i] = feat
                if y_out is not None:
                    y_out[i] = lab
            done += 1
            if report_every is not None and done % report_every == 0:
                print(f"Processed {tag} images: {done}/{n}")

    if bad > 0:
        print(f"WARNING: {bad} {tag} images could not be read and were skipped.")
        keep_mask = np.isfinite(X).all(axis=1)
        X = X[keep_mask]
        if y_out is not None:
            y_out = y_out[keep_mask]

    if labels is not None:
        return X, y_out.astype(int, copy=False)
    return X


print("Building ALL train features once (reused for CV/val/full-train)...")
all_ids = train_df["id_code"].values
all_y = train_df["diagnosis"].astype(int).values
X_all, y_all_built = build_features(
    all_ids, labels=all_y, img_dir=train_img_dir, report_every=500, tag="all-train"
)

if X_all.shape[0] != len(all_ids):
    raise RuntimeError(
        "Some training images could not be read; for strict split consistency, this script expects all train images readable."
    )

CLASS_IDX = np.arange(5, dtype=np.float64)


def make_clf(C=2.0):
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "lr",
                LogisticRegression(
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=3000,
                    multi_class="multinomial",
                    n_jobs=-1,
                    C=float(C),
                    random_state=42,
                ),
            ),
        ]
    )


N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

C_GRID = [0.5, 1.0, 2.0, 4.0]

best_C = None
best_cv_qwk = -1e9
best_thr_for_bestC = None

print(f"Tuning C over {C_GRID} using {N_SPLITS}-fold OOF thresholds...")

for C in C_GRID:
    oof_exp = np.empty((X_all.shape[0],), dtype=np.float64)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y_all_built), start=1):
        clf_fold = make_clf(C=C)
        clf_fold.fit(X_all[tr_idx], y_all_built[tr_idx])

        va_proba = clf_fold.predict_proba(X_all[va_idx])
        oof_exp[va_idx] = (va_proba * CLASS_IDX[None, :]).sum(axis=1)

    thr_oof, _ = fit_thresholds_qwk_multistart(
        oof_exp,
        y_all_built,
        n_classes=5,
        n_iters=10,
        n_starts=7,
        jitter_frac=0.03,
        seed=42,
    )
    oof_pred = apply_thresholds(oof_exp, thr_oof)
    cv_qwk_oof = quadratic_weighted_kappa(y_all_built, oof_pred, n_classes=5)

    print(f"  C={C}: OOF-QWK={float(cv_qwk_oof):.6f}, oof_thr={thr_oof.tolist()}")

    if cv_qwk_oof > best_cv_qwk:
        best_cv_qwk = float(cv_qwk_oof)
        best_C = C
        best_thr_for_bestC = thr_oof

thr = np.array(best_thr_for_bestC, dtype=np.float64)
thr = np.sort(thr)

print("Selected C:", best_C, "with OOF QWK:", float(best_cv_qwk))
print("Selected OOF thresholds:", thr.tolist())

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(sss.split(all_ids, y_all_built))
X_train, y_train = X_all[train_idx], y_all_built[train_idx]
X_val, y_val = X_all[val_idx], y_all_built[val_idx]

clf_check = make_clf(C=best_C)
clf_check.fit(X_train, y_train)
val_proba = clf_check.predict_proba(X_val)
val_exp = (val_proba * CLASS_IDX[None, :]).sum(axis=1)
val_pred_thr = apply_thresholds(val_exp, thr)
val_qwk_thr = quadratic_weighted_kappa(y_val, val_pred_thr, n_classes=5)
print(
    "Single split (for reference) Val QWK using tuned-C + OOF thresholds:",
    float(val_qwk_thr),
)

gc.collect()



## === cell 5
print("Refitting model on full training data and predicting test...")
clf = make_clf(C=best_C)
clf.fit(X_all, y_all_built)
gc.collect()

id_code = sub_df["id_code"].values

feat_dim = 804
X_test = np.empty((len(id_code), feat_dim), dtype=np.float32)


def _one_test(i_img_id):
    i, img_id = i_img_id
    img_path = os.path.join(test_img_dir, f"{img_id}.png")
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return i, None
    img_rgb = load_ben_color(img_bgr)
    return i, extract_features_from_ben_rgb(img_rgb)


max_workers = min(8, (os.cpu_count() or 4))
bad = 0
done = 0
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for i, feat in ex.map(_one_test, enumerate(id_code), chunksize=32):
        if feat is None:
            bad += 1
        else:
            X_test[i] = feat
        done += 1
        if done % 100 == 0:
            print(f"Processed test images: {done}/{len(id_code)}")

if bad > 0:
    raise FileNotFoundError(
        f"Could not read {bad} test images (see paths under {test_img_dir})."
    )

proba = clf.predict_proba(X_test)
exp_test = (proba * CLASS_IDX[None, :]).sum(axis=1)
test_prediction = apply_thresholds(exp_test, thr).astype(np.int64)



## === cell 6
sub_df["diagnosis"] = test_prediction.astype("int64")
sub_path = "submission.csv"
sub_df[["id_code", "diagnosis"]].to_csv(sub_path, index=False)

unique, counts = np.unique(test_prediction, return_counts=True)
print(dict(zip(unique.tolist(), counts.tolist())))
print(f"Saved {sub_path} with shape {sub_df.shape}")
print("Done!")
