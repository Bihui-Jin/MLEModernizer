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

0.6376212321223429

# 6. Current score

0.34588

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.19606) has done: 'I remove the failing `tensorflow-addons` installation/import (it triggers a known protobuf `GetPrototype` error) and keep the rest of the pipeline intact. I also fix the missing `keras/pd` definitions by consolidating imports in the first cell so later cells can run. Since the referenced external pretrained model path (`../input/eff-b2-model/eff_b2_model`) is not available in your provided dataset paths, I switch to loading a model from the same competition input directory if present; otherwise, I fall back to a standard Keras application model (DenseNet121) to ensure the notebook runs end-to-end and produces `submission.csv`. Finally, I fix inference preprocessing to match the expected float input range (0–1) to avoid obvious prediction issues and ensure the submission format is correct.'
- What this solution (achieved 0.03816) has done: 'I fix the immediate runtime crash by forcing TensorFlow to use the pure-Python protobuf implementation before importing `tensorflow`, which avoids the `MessageFactory.GetPrototype` error in this environment. Then I make model loading robust by compiling the fallback model (if no external pretrained model is found) and loading weights if present, so predictions are not random; this should move the QWK score upward toward your target while keeping the same overall pipeline (single-image loop inference, same preprocessing, same DenseNet121 fallback architecture). I also add a safe path fallback for the dataset directory because Kaggle can mount it under `/kaggle/input/...` rather than `../input/...`, without changing the I/O semantics. Finally, I ensure the submission is aligned to `test.csv` ordering (not the sample submission ordering) to avoid silent id/prediction mismatches.'
- What this solution (achieved 0.09953) has done: 'To fix the runtime crash, I force TensorFlow to use the pure-Python protobuf *and* proactively remove the incompatible `google.protobuf` C-extension modules from `sys.modules` before importing TensorFlow (this is the usual root cause of the `MessageFactory.GetPrototype` error). To move the score up toward your target without changing the modeling approach, I add a minimal, standard “Ben Graham” preprocessing step at inference (the same one you already defined for training-style pipelines) so the DenseNet/ImageNet features see a more appropriate input distribution than raw resized RGB. I also make image path resolution a bit more robust (nested `test_images/test_images` appears in your directory listing) to prevent missing-file failures while keeping I/O semantics identical. The rest of the logic (single-image loop inference, same fallback DenseNet121 architecture/head, same argmax-to-class submission) is preserved.'
- What this solution (achieved -0.02911) has done: 'The crash happens before any modeling because TensorFlow is importing an incompatible protobuf build; setting the environment variable inside Python is too late in this Kaggle image. I fix this by switching to a TF import strategy that forces the pure-Python protobuf implementation from the very start of the process and avoids the `GetPrototype` call path, while keeping all modeling/inference logic identical. I also add a safe fallback to run inference without TensorFlow if TF still can’t be imported (using a deterministic, simple image-statistics classifier) so you always get a valid `submission.csv`. This is primarily a stability fix; if TF loads correctly you keep the same model path/Ben-Graham preprocessing/argmax submission behavior as before.'
- What this solution (achieved 0.3038) has done: 'Your runtime crash is caused by a protobuf/TensorFlow incompatibility (`MessageFactory.GetPrototype`) that prevents TensorFlow from importing, so the code always falls back to the heuristic predictor (which explains the very low QWK). The minimal fix is to avoid TensorFlow entirely and switch to a pure non-TF approach that is still legitimate: train a simple image-statistics classifier on the provided training images/labels and apply it to the test images. This keeps the overall pipeline structure (read images → preprocess → model → predict classes → write `submission.csv`) but removes the broken dependency and should move the score substantially upward versus the current heuristic-only brightness thresholds. The submission formatting and test ordering are preserved exactly via `test.csv`.'
- What this solution (achieved 0.40686) has done: 'Most of the runtime is spent in Python-level loops doing per-image disk I/O + preprocessing, plus the very expensive cutpoint optimization which recomputes QWK (and re-bins) thousands of times. To finish within 600s without changing the model/training logic, I will: (1) parallelize feature extraction for train/test using a deterministic thread pool (OpenCV releases the GIL heavily), (2) avoid repeated existence checks by precomputing the actual image path for each id once, and (3) make cutpoint optimization provably equivalent but much faster by precomputing the expected-class `x` once, sorting it once, and evaluating QWK for all candidate cutpoints in bulk using cached cumulative counts (same objective, same candidate grid, same coordinate-descent steps). These changes preserve the exact semantics (same features, same softmax regression training, same coordinate descent over the same grid), only removing redundant work.'
- What this solution (achieved 0.43055) has done: 'Your current pipeline is already producing a valid submission; the main gap to the target is likely from underfitting/miscalibration rather than runtime issues. With minimal changes and identical core logic (same handcrafted features, same softmax regression, same QWK-based cutpoint tuning), I (1) retrain the softmax regression on *all* standardized training data after selecting cutpoints on the validation split (so the final weights use 100% of labels), and (2) compute cutpoints using out-of-fold (OOF) expected-class predictions from a small stratified K-fold loop to reduce overfitting to a single split while keeping the same cutpoint optimizer and objective. This typically improves QWK noticeably for this style of model without changing architecture/loss/feature extraction. Submission formatting and test ordering remain unchanged, and runtime remains within the 600s budget.'
- What this solution (achieved 0.43055) has done: 'We’re currently below the target (0.43055 vs 0.6376), so we want a modest, safe boost without changing the overall model/training approach. The most likely “free” gain for QWK here is to tune the ordinal cutpoints on a prediction signal that’s less biased than the raw expected class; a standard minimal tweak is to fit a 1D linear calibration `x' = a*x + b` on OOF predictions (same model, same loss), then optimize cutpoints on `x'`. I also re-fit the feature standardization inside each fold (to avoid mild leakage from using global `mu/sigma` for OOF), then retrain the final model on all data exactly as before and apply the learned calibration + cutpoints at test time. All changes keep the same handcrafted features, same softmax regression training loop, same QWK-based cutpoint optimizer, and still write `submission.csv` in the required format.'
- What this solution (achieved 0.20166) has done: 'Your current score (0.43055) is below the target (0.63762), so we want a safe boost without changing the core model/feature logic. The most likely low-risk gain for QWK here is to address class imbalance in the same softmax-regression training loop by adding simple per-sample class weights (this keeps the exact same architecture and gradient-descent approach, only reweighting the loss). To avoid shifting behavior too much, I keep the original unweighted training for the per-fold OOF signal (used for calibration/cutpoints), and apply class-weighting only when fitting the final model on all data for test inference. This typically improves minority-class recall and moves QWK upward while keeping runtime and semantics stable and still producing a valid `submission.csv`.'
- What this solution (achieved 0.34588) has done: 'We’re well below the target (0.20166 vs 0.63762), so we should make a small, legitimate change that tends to boost QWK without altering the core pipeline (same handcrafted features, same softmax regression training loop, same cutpoint optimization). The biggest issue is that we currently tune calibration/cutpoints on OOF predictions from an *unweighted* model, but then we train the final model with *class-weighting*, which shifts the expected-class distribution and makes the learned calibration/cutpoints mismatched at test time. I make the OOF predictions use the same class-weighting that’s used for the final model, and fit calibration/cutpoints on that matched OOF signal (no architecture/loss change—just consistent weighting). This typically improves ordinal mapping and QWK while keeping runtime and semantics stable and still producing a valid `submission.csv`.'
- What this solution (achieved 0.34588) has done: 'Your current score is well below the target (0.34588 vs 0.63762, higher-is-better), so we should aim for a modest, safe improvement without changing the core model/feature/training logic. The biggest low-risk gain for QWK in this pipeline is usually better ordinal calibration: instead of fitting the linear calibration `x' = a*x+b` to raw labels (least squares), fit `a,b` by directly maximizing QWK on OOF predictions (same signal, same cutpoint optimizer; we only change how `a,b` are chosen). Then re-optimize cutpoints on the calibrated OOF signal exactly as before, and apply the same transform at test time. This preserves your handcrafted features, softmax regression, and cutpoint-based discretization, and should move score upward toward the target.'

# 9. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd
import cv2

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

cv2.setNumThreads(0)

np.random.seed(42)



## === cell 1
"""
    Config
"""
IMG_SIZE = 224


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        if not mask.any():
            return img
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        if not mask.any():
            return img
        ys = mask.any(1)
        xs = mask.any(0)
        if not ys.any() or not xs.any():
            return img
        return img[np.ix_(ys, xs)]
    return img


def load_ben_color(image, sigmaX=10):
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image.astype("float32")


def extract_features_from_rgb_uint8(img_rgb_uint8: np.ndarray) -> np.ndarray:
    img = load_ben_color(img_rgb_uint8)  # float32 in ~[0,255]
    x = (img * (1.0 / 255.0)).astype(np.float32, copy=False)

    gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).astype(np.float32, copy=False)

    r = x[..., 0]
    g = x[..., 1]
    b = x[..., 2]

    feats = np.empty(10, dtype=np.float32)
    feats[0] = float(gray.mean())
    feats[1] = float(gray.std())
    feats[2] = float(r.mean())
    feats[3] = float(g.mean())
    feats[4] = float(b.mean())
    feats[5] = float(r.std())
    feats[6] = float(g.std())
    feats[7] = float(b.std())
    feats[8] = float((np.max(x, axis=2) - np.min(x, axis=2)).mean())

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    feats[9] = float(lap.var())
    return feats


def softmax(z: np.ndarray, axis: int = 1) -> np.ndarray:
    zmax = np.max(z, axis=axis, keepdims=True)
    z = z - zmax
    e = np.exp(z)
    s = np.sum(e, axis=axis, keepdims=True) + 1e-12
    return e / s


def resolve_image_paths(ids, primary_dir, nested_dir):
    paths = []
    for img_id in ids:
        p = os.path.join(primary_dir, f"{img_id}.png")
        if not os.path.exists(p):
            alt = os.path.join(nested_dir, f"{img_id}.png")
            if os.path.exists(alt):
                p = alt
        paths.append(p)
    return np.array(paths, dtype=object)


def build_features_threaded(image_paths, labels=None, max_workers=8):
    from concurrent.futures import ThreadPoolExecutor

    n = len(image_paths)
    X = np.empty((n, 10), dtype=np.float32)
    y_out = None if labels is None else np.asarray(labels, dtype=np.int64).copy()
    ok = np.ones(n, dtype=bool)

    def _one(i):
        p = image_paths[i]
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            return i, None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        feats = extract_features_from_rgb_uint8(img)
        return i, feats

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, feats in ex.map(_one, range(n), chunksize=32):
            if feats is None:
                ok[i] = False
            else:
                X[i] = feats

    if labels is None:
        return X, ok
    else:
        return X, y_out, ok




## === cell 2
DATA_ROOT_CANDIDATES = [
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
    "../input/aptos2019-blindness-detection",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(p):
        DATA_ROOT = p
        break
if DATA_ROOT is None:
    DATA_ROOT = "../input/aptos2019-blindness-detection"

TRAIN_DIR = os.path.join(DATA_ROOT, "train_images")
TRAIN_DIR_NESTED = os.path.join(TRAIN_DIR, "train_images")
TEST_DIR = os.path.join(DATA_ROOT, "test_images")
TEST_DIR_NESTED = os.path.join(TEST_DIR, "test_images")

train_csv_path = os.path.join(DATA_ROOT, "train.csv")
test_csv_path = os.path.join(DATA_ROOT, "test.csv")
sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")

train_df = pd.read_csv(train_csv_path)
test_df = (
    pd.read_csv(test_csv_path)
    if os.path.exists(test_csv_path)
    else pd.read_csv(sample_sub_path)[["id_code"]]
)

print("DATA_ROOT:", DATA_ROOT)
print("Train:", train_df.shape, "Test:", test_df.shape)




## === cell 3
def train_softmax_regression(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int = 5,
    lr: float = 0.5,
    epochs: int = 800,
    reg: float = 1e-3,
    sample_weight: np.ndarray | None = None,
):
    """
    Optional per-sample weights to reduce class-imbalance bias.
    Core logic preserved: same softmax regression + full-batch gradient descent loop.
    """
    n, d = X.shape
    W = np.zeros((d, n_classes), dtype=np.float32)
    b = np.zeros((n_classes,), dtype=np.float32)

    Y = np.zeros((n, n_classes), dtype=np.float32)
    Y[np.arange(n), y] = 1.0

    if sample_weight is None:
        sw = None
        inv_norm = 1.0 / n
    else:
        sw = np.asarray(sample_weight, dtype=np.float32).reshape(-1)
        sw_sum = float(sw.sum())
        if sw_sum <= 0:
            sw = None
            inv_norm = 1.0 / n
        else:
            inv_norm = 1.0 / sw_sum

    logits = np.empty((n, n_classes), dtype=np.float32)
    P = np.empty((n, n_classes), dtype=np.float32)
    G = np.empty((n, n_classes), dtype=np.float32)

    for _ in range(epochs):
        np.matmul(X, W, out=logits)
        logits += b[None, :]

        zmax = np.max(logits, axis=1, keepdims=True)
        np.subtract(logits, zmax, out=P)
        np.exp(P, out=P)
        denom = np.sum(P, axis=1, keepdims=True) + 1e-12
        P /= denom

        np.subtract(P, Y, out=G)

        if sw is None:
            G *= inv_norm
        else:
            G *= sw[:, None] * inv_norm

        dW = X.T @ G
        dW += reg * W
        db = G.sum(axis=0)

        W -= lr * dW
        b -= lr * db

    return W, b


def standardize_fit(X: np.ndarray):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-6] = 1.0
    return mu.astype(np.float32), sigma.astype(np.float32)


def standardize_apply(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    return ((X - mu) / sigma).astype(np.float32, copy=False)


def quadratic_weighted_kappa(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 5
) -> float:
    y_true = y_true.astype(np.int64, copy=False)
    y_pred = y_pred.astype(np.int64, copy=False)

    mask = (y_true >= 0) & (y_true < n_classes) & (y_pred >= 0) & (y_pred < n_classes)
    yt = y_true[mask]
    yp = y_pred[mask]

    idx = yt * n_classes + yp
    O = (
        np.bincount(idx, minlength=n_classes * n_classes)
        .reshape(n_classes, n_classes)
        .astype(np.float64)
    )

    act_hist = np.bincount(yt, minlength=n_classes).astype(np.float64)
    pred_hist = np.bincount(yp, minlength=n_classes).astype(np.float64)
    E = np.outer(act_hist, pred_hist)
    sE = E.sum()
    sO = O.sum()
    if sE > 0:
        E *= sO / sE

    ij = np.arange(n_classes, dtype=np.float64)
    Wm = (ij[:, None] - ij[None, :]) ** 2 / ((n_classes - 1) ** 2)

    num = (Wm * O).sum()
    den = (Wm * E).sum()
    if den == 0:
        return 0.0
    return float(1.0 - num / den)


def probs_to_expected_class(P: np.ndarray) -> np.ndarray:
    classes = np.arange(P.shape[1], dtype=np.float32)
    return (P.astype(np.float32, copy=False) * classes[None, :]).sum(axis=1)


def apply_cutpoints(x: np.ndarray, cutpoints: np.ndarray) -> np.ndarray:
    return np.digitize(x, bins=cutpoints).astype(np.int64)


def optimize_cutpoints(x: np.ndarray, y: np.ndarray, n_classes: int = 5) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    y = y.astype(np.int64, copy=False)

    qs = np.quantile(x, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
    cp = qs.copy()

    grid = np.quantile(x, np.linspace(0.02, 0.98, 97)).astype(np.float32)

    order = np.argsort(x, kind="mergesort")  # stable/deterministic
    xs = x[order]
    ys = y[order]

    n = len(xs)
    prefix = np.zeros((n + 1, n_classes), dtype=np.int32)
    for c in range(n_classes):
        prefix[1:, c] = np.cumsum(ys == c, dtype=np.int32)

    act_hist = np.bincount(y, minlength=n_classes).astype(np.float64)
    sO = float(n)

    ij = np.arange(n_classes, dtype=np.float64)
    Wm = (ij[:, None] - ij[None, :]) ** 2 / ((n_classes - 1) ** 2)

    def _qwk_from_bins(bin_edges):
        cuts = np.searchsorted(xs, bin_edges, side="right")
        cuts = np.concatenate(([0], cuts, [n]))
        O = np.zeros((n_classes, n_classes), dtype=np.float64)
        pred_hist = np.empty(n_classes, dtype=np.float64)
        for j in range(n_classes):
            a, b = int(cuts[j]), int(cuts[j + 1])
            seg = (prefix[b] - prefix[a]).astype(np.float64)
            O[:, j] = seg
            pred_hist[j] = seg.sum()

        E = np.outer(act_hist, pred_hist)
        sE = E.sum()
        if sE > 0:
            E *= sO / sE

        num = (Wm * O).sum()
        den = (Wm * E).sum()
        if den == 0:
            return 0.0
        return float(1.0 - num / den)

    best = _qwk_from_bins(cp)

    for _pass in range(6):
        improved = False
        for j in range(4):
            low = cp[j - 1] + 1e-4 if j > 0 else -1e9
            high = cp[j + 1] - 1e-4 if j < 3 else 1e9
            candidates = grid[(grid > low) & (grid < high)]
            if candidates.size == 0:
                continue

            local_best = best
            local_cp_j = cp[j]

            for v in candidates:
                tmp0, tmp1, tmp2, tmp3 = (
                    float(cp[0]),
                    float(cp[1]),
                    float(cp[2]),
                    float(cp[3]),
                )
                if j == 0:
                    tmp0 = float(v)
                elif j == 1:
                    tmp1 = float(v)
                elif j == 2:
                    tmp2 = float(v)
                else:
                    tmp3 = float(v)

                score = _qwk_from_bins(
                    np.array([tmp0, tmp1, tmp2, tmp3], dtype=np.float32)
                )
                if score > local_best:
                    local_best = score
                    local_cp_j = float(v)

            if local_best > best + 1e-12:
                cp[j] = local_cp_j
                best = local_best
                improved = True

        if not improved:
            break

    cp = np.sort(cp)
    cp[1] = max(cp[1], cp[0] + 1e-4)
    cp[2] = max(cp[2], cp[1] + 1e-4)
    cp[3] = max(cp[3], cp[2] + 1e-4)
    return cp.astype(np.float32)


def make_stratified_folds(y: np.ndarray, n_splits: int = 5, seed: int = 42):
    y = np.asarray(y, dtype=np.int64)
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(n_splits)]
    for c in range(int(y.max()) + 1):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        for i, ix in enumerate(idx):
            folds[i % n_splits].append(int(ix))
    fold_indices = []
    all_idx = np.arange(len(y), dtype=np.int64)
    for k in range(n_splits):
        va = np.array(sorted(folds[k]), dtype=np.int64)
        tr_mask = np.ones(len(y), dtype=bool)
        tr_mask[va] = False
        tr = all_idx[tr_mask]
        fold_indices.append((tr, va))
    return fold_indices


def fit_linear_calibration(x: np.ndarray, y: np.ndarray):
    """
    Fit x' = a*x + b by least squares to map expected-class signal closer to label scale.
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    xm = float(x.mean())
    ym = float(y.mean())
    xv = float(((x - xm) ** 2).mean())
    if xv < 1e-12:
        a = 1.0
    else:
        a = float(((x - xm) * (y - ym)).mean() / xv)
    b = ym - a * xm
    return np.float32(a), np.float32(b)


def make_balanced_sample_weight(y: np.ndarray, n_classes: int = 5) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts[counts < 1.0] = 1.0
    w_per_class = counts.sum() / (n_classes * counts)  # avg weight ~ 1
    return w_per_class[y].astype(np.float32, copy=False)


def optimize_linear_calibration_for_qwk(
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int = 5,
    a_init: float | None = None,
    b_init: float | None = None,
):
    """
    Change is directly score-driven (QWK): choose (a,b) to maximize QWK after cutpointing,
    using the SAME cutpoint optimizer as the rest of the pipeline.
    This keeps core logic (expected-class signal -> linear calibration -> cutpoints) intact,
    but makes the calibration metric-aligned instead of MSE-aligned.
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    if a_init is None or b_init is None:
        a0, b0 = fit_linear_calibration(x, y)
        a = float(a0)
        b = float(b0)
    else:
        a = float(a_init)
        b = float(b_init)

    def _score(a_, b_):
        x_cal = (a_ * x + b_).astype(np.float32, copy=False)
        cps = optimize_cutpoints(x_cal, y, n_classes=n_classes)
        pred = apply_cutpoints(x_cal, cps)
        return quadratic_weighted_kappa(y, pred, n_classes=n_classes), cps

    best_score, best_cps = _score(a, b)

    for _ in range(8):
        improved = False

        a_candidates = a * np.array(
            [0.85, 0.92, 0.97, 1.03, 1.08, 1.15], dtype=np.float32
        )
        for ac in a_candidates:
            sc, cps = _score(float(ac), b)
            if sc > best_score + 1e-12:
                best_score, best_cps = sc, cps
                a = float(ac)
                improved = True

        x_std = float(x.std()) + 1e-6
        b_steps = (
            np.array([-0.40, -0.25, -0.12, 0.12, 0.25, 0.40], dtype=np.float32) * x_std
        )
        for bs in b_steps:
            bc = b + float(bs)
            sc, cps = _score(a, bc)
            if sc > best_score + 1e-12:
                best_score, best_cps = sc, cps
                b = float(bc)
                improved = True

        if not improved:
            break

    return np.float32(a), np.float32(b), best_cps.astype(np.float32), float(best_score)




## === cell 4
train_ids = train_df["id_code"].astype(str).values
y_all = train_df["diagnosis"].astype(np.int64).values

train_paths = resolve_image_paths(train_ids, TRAIN_DIR, TRAIN_DIR_NESTED)

X_all, y_all2, ok_mask = build_features_threaded(
    train_paths, labels=y_all, max_workers=8
)
skipped = int((~ok_mask).sum())
X_all = X_all[ok_mask]
y_all2 = y_all2[ok_mask]

print("Built train features:", X_all.shape, "Skipped:", skipped)

n_classes = 5
folds = make_stratified_folds(y_all2, n_splits=5, seed=42)

oof_x = np.empty((len(y_all2),), dtype=np.float32)

for k, (tr_idx, va_idx) in enumerate(folds):
    X_tr_raw = X_all[tr_idx]
    y_tr = y_all2[tr_idx]
    X_va_raw = X_all[va_idx]

    mu_k, sigma_k = standardize_fit(X_tr_raw)
    X_tr = standardize_apply(X_tr_raw, mu_k, sigma_k)
    X_va = standardize_apply(X_va_raw, mu_k, sigma_k)

    sw_tr = make_balanced_sample_weight(y_tr, n_classes=5)

    W_k, b_k = train_softmax_regression(
        X_tr, y_tr, n_classes=5, lr=0.5, epochs=800, reg=1e-3, sample_weight=sw_tr
    )
    va_logits = X_va @ W_k + b_k[None, :]
    va_P = softmax(va_logits, axis=1)
    oof_x[va_idx] = probs_to_expected_class(va_P)

    va_pred_argmax = np.argmax(va_logits, axis=1)
    qwk_fold = quadratic_weighted_kappa(y_all2[va_idx], va_pred_argmax, n_classes=5)
    print(f"Fold {k+1}/5 QWK argmax (sanity): {qwk_fold:.4f}")

a_ls, b_ls = fit_linear_calibration(oof_x, y_all2)
a_cal, b_cal, cutpoints, qwk_oof_cut = optimize_linear_calibration_for_qwk(
    oof_x, y_all2, n_classes=5, a_init=float(a_ls), b_init=float(b_ls)
)
oof_x_cal = (a_cal * oof_x + b_cal).astype(np.float32, copy=False)
oof_pred_cut = apply_cutpoints(oof_x_cal, cutpoints)

print("LS calibration (a, b):", float(a_ls), float(b_ls))
print("QWK-optimized calibration (a, b):", float(a_cal), float(b_cal))
print("OOF cutpoints:", cutpoints.tolist())
print("OOF QWK cutpoints (sanity):", qwk_oof_cut)

mu, sigma = standardize_fit(X_all)
X_all_std = standardize_apply(X_all, mu, sigma)

sw_full = make_balanced_sample_weight(y_all2, n_classes=5)

W, b = train_softmax_regression(
    X_all_std, y_all2, n_classes=5, lr=0.5, epochs=800, reg=1e-3, sample_weight=sw_full
)

train_logits_full = X_all_std @ W + b[None, :]
train_pred_argmax_full = np.argmax(train_logits_full, axis=1)
acc_full = float((train_pred_argmax_full == y_all2).mean())
print("Train acc on full data (sanity):", acc_full)



## === cell 5
id_code = test_df["id_code"].astype(str).values
test_paths = resolve_image_paths(id_code, TEST_DIR, TEST_DIR_NESTED)

X_test, ok_mask_test = build_features_threaded(test_paths, labels=None, max_workers=8)
if not ok_mask_test.all():
    bad = np.where(~ok_mask_test)[0][:5]
    raise FileNotFoundError(
        f"Could not read some test images, examples indices: {bad.tolist()} paths: {[test_paths[i] for i in bad]}"
    )

inv_sigma = (1.0 / sigma).astype(np.float32)
X_test_std = (X_test - mu[None, :]) * inv_sigma[None, :]

test_logits = X_test_std @ W + b[None, :]
test_P = softmax(test_logits, axis=1)
test_x = probs_to_expected_class(test_P)

test_x_cal = (a_cal * test_x + b_cal).astype(np.float32, copy=False)

test_prediction = apply_cutpoints(test_x_cal, cutpoints).astype(np.int64)



## === cell 6
sub = test_df[["id_code"]].copy()
sub["diagnosis"] = test_prediction.astype(np.int64)
sub.to_csv("submission.csv", index=False)

unique, counts = np.unique(test_prediction, return_counts=True)
print(dict(zip(unique.tolist(), counts.tolist())))
print("Wrote submission.csv with shape:", sub.shape)
print("Done!")
