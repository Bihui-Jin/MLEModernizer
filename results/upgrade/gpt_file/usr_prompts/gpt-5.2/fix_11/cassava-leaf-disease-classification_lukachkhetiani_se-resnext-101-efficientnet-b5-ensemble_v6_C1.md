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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8656693865216077

# 6. Current score

0.66704

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63341) has done: 'I remove the broken EfficientNet + missing checkpoint dependency that prevents `model` from being created, since `efficientnet_pytorch` isn’t installed and `b5.pth` doesn’t exist in the provided paths. To keep the pipeline valid end-to-end, I replace that section with a lightweight, deterministic image classifier built from simple color/statistics features computed from each image; this preserves the overall “read images → preprocess → predict → write submission.csv” flow and guarantees a valid CSV. I also ensure predictions are produced for every `sample_submission.csv` image_id (matching required order/row count) and add safe fallbacks for unreadable images. This should yield a non-trivial accuracy (better than random) while prioritizing correctness and Kaggle submission validity.'
- What this solution (achieved 0.63565) has done: 'You’re currently far below the target (0.63341 vs 0.86567), so we should make the smallest changes that legitimately improve accuracy without changing the overall “handcrafted features → simple linear classifier → argmax → submission.csv” core. The biggest win (with minimal disruption) is to remove the intentional downsampling and instead train on all available training images, since this competition is data-hungry and your model is very lightweight; this should move accuracy up substantially toward the target. Second, we add a very small amount of numeric stabilization (float32 throughout, clipped std) and a tiny learning-rate decay schedule (still the same training loop/optimizer math) to improve convergence without changing the model class. All paths and submission formatting remain identical.'
- What this solution (achieved 0.63453) has done: 'Your current pipeline is a simple handcrafted-feature + linear softmax classifier; to move accuracy upward toward the 0.8657 target without changing that core approach, I strengthen the feature vector slightly while keeping the same extraction flow and training loop. Specifically, I add a compact color-histogram in HSV plus a small gradient-orientation histogram (HOG-like) computed from Sobel edges, which are cheap and often improve disease/texture separability in cassava images. I also switch the training objective from one-hot MSE-on-softmax to the standard softmax cross-entropy gradient (still the same linear model and minibatch SGD), which typically converges to better classification accuracy for the same architecture. All paths and submission formatting stay identical, and the script still runs end-to-end and writes `submission.csv`.'
- What this solution (achieved 0.65882) has done: 'Your current score is far below the target (0.63453 vs 0.86567; higher-is-better), so we should improve accuracy with minimal, low-risk changes while keeping the same “handcrafted features → linear softmax classifier → argmax → submission.csv” core. The biggest likely gain without changing the model/training loop is to make the handcrafted features more disease/texture-sensitive by adding a compact Local Binary Pattern (LBP) histogram and a few simple blur/contrast/sharpness statistics. These are fast, deterministic, and fit naturally into your existing `extract_features()` flow, often giving a noticeable lift on leaf-disease imagery. All paths, training approach, loss/gradient, and submission formatting remain the same.'
- What this solution (achieved 0.53999) has done: 'Your current score (0.65882) is well below the target (0.86567), so we should make a small, legitimate accuracy improvement while preserving the same handcrafted-features → linear softmax classifier → SGD training loop. The biggest low-risk gain here is to rebalance the training signal: this dataset is class-imbalanced, so adding per-class weights inside the same cross-entropy gradient (no architecture/training-loop change) usually improves accuracy materially. I also switch the weight decay term to the standard “exclude bias from decay” variant (still identical model, just a small regularization fix), and keep everything else (features, optimizer structure, paths, submission format) unchanged. This should move the score upward toward your target without introducing new dependencies or changing core semantics.'
- What this solution (achieved 0.66368) has done: 'Your current score (0.53999) is far below the target (0.86567; higher-is-better), so we should make a small, legitimate improvement while keeping the same “handcrafted features → linear softmax classifier → SGD loop → argmax submission” core. The biggest low-risk issue in your current training is that class weights are applied by multiplying the whole gradient per-sample, which can distort optimization and hurt accuracy; we switch to the standard weighted softmax-cross-entropy gradient (multiply only the true-class subtraction term). We also compute class weights with an “effective number of samples” formulation (still just class reweighting) to avoid extreme weights and improve generalization. Everything else (features, model form, optimizer loop structure, paths, submission formatting) remains the same.'
- What this solution (achieved 0.6633) has done: 'You’re currently far below the target (0.66368 vs 0.86567; higher-is-better), so we make small, low-risk changes that improve generalization while keeping the same handcrafted-features → linear softmax classifier → SGD training loop core intact. The main fix is to correct the weighted cross-entropy gradient so that class weights scale the full per-sample gradient (not just the true-class subtraction), which is the standard formulation and typically yields a meaningful accuracy lift on imbalanced data. To avoid the weights being “neutralized” by the subsequent batch averaging, we also normalize by the sum of sample weights in the batch (a numerically small, semantics-preserving adjustment for weighted loss). Everything else (feature extraction, model form, optimizer style, paths, and submission formatting) remains unchanged and still writes a valid `submission.csv`.'
- What this solution (achieved 0.66704) has done: 'The timeout is dominated by per-image CPU feature extraction done ~37k times for training (original + flip) and ~2.7k times for test, plus extra Python overhead from list appends and repeated allocations. I keep the exact same features and training loop, but make them run faster by (1) preallocating feature arrays to avoid list growth and stack costs, (2) vectorizing/avoiding unnecessary dtype conversions and `.tolist()` round-trips inside `extract_features`, (3) using OpenCV’s optimized histogram functions where equivalent, and (4) parallelizing feature extraction with a deterministic thread pool (OpenCV releases the GIL in heavy ops) while preserving the same samples and order/labels. These changes preserve core logic and should cut wall time substantially without changing model, training semantics, or I/O paths.'

# 9. Code solution

## === cell 0
import os, sys, glob, json, random
import numpy as np
import pandas as pd



## === cell 1
import cv2
import tqdm
from concurrent.futures import ThreadPoolExecutor

random.seed(42)
np.random.seed(42)

try:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(max(1, (os.cpu_count() or 2)))
except Exception:
    pass



## === cell 2
BASE1 = "../input/cassava-leaf-disease-classification"
BASE2 = (
    "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification"
)

test_img_dir = os.path.join(BASE1, "test_images")
if not os.path.isdir(test_img_dir):
    test_img_dir = os.path.join(BASE2, "test_images")

assert os.path.isdir(test_img_dir), f"test_images directory not found at {test_img_dir}"

train_csv_path = os.path.join(BASE1, "train.csv")
if not os.path.exists(train_csv_path):
    train_csv_path = os.path.join(BASE2, "train.csv")
assert os.path.exists(train_csv_path), f"train.csv not found at {train_csv_path}"

train_img_dir = os.path.join(BASE1, "train_images")
if not os.path.isdir(train_img_dir):
    train_img_dir = os.path.join(BASE2, "train_images")
assert os.path.isdir(
    train_img_dir
), f"train_images directory not found at {train_img_dir}"

sample_sub_path = os.path.join(BASE1, "sample_submission.csv")
if not os.path.exists(sample_sub_path):
    sample_sub_path = os.path.join(BASE2, "sample_submission.csv")
assert os.path.exists(
    sample_sub_path
), f"sample_submission.csv not found at {sample_sub_path}"



## === cell 3
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _gray_world_wb_bgr(img_bgr_u8: np.ndarray) -> np.ndarray:
    x = img_bgr_u8.astype(np.float32, copy=False)
    mean_b = float(x[..., 0].mean()) + 1e-6
    mean_g = float(x[..., 1].mean()) + 1e-6
    mean_r = float(x[..., 2].mean()) + 1e-6
    mean_gray = (mean_b + mean_g + mean_r) / 3.0
    sb = mean_gray / mean_b
    sg = mean_gray / mean_g
    sr = mean_gray / mean_r
    x = x.copy()  # ensure we don't mutate caller buffer
    x[..., 0] *= sb
    x[..., 1] *= sg
    x[..., 2] *= sr
    np.clip(x, 0.0, 255.0, out=x)
    return x.astype(np.uint8, copy=False)


def _lbp_u8(gray_u8: np.ndarray) -> np.ndarray:
    """
    Minimal, fast LBP (8 neighbors, radius=1).
    Returns uint8 codes in [0,255].
    """
    g = gray_u8
    c = g[1:-1, 1:-1].astype(np.uint8, copy=False)
    code = np.zeros_like(c, dtype=np.uint8)

    code |= ((g[0:-2, 0:-2] >= c) << 7).astype(np.uint8, copy=False)
    code |= ((g[0:-2, 1:-1] >= c) << 6).astype(np.uint8, copy=False)
    code |= ((g[0:-2, 2:] >= c) << 5).astype(np.uint8, copy=False)
    code |= ((g[1:-1, 2:] >= c) << 4).astype(np.uint8, copy=False)
    code |= ((g[2:, 2:] >= c) << 3).astype(np.uint8, copy=False)
    code |= ((g[2:, 1:-1] >= c) << 2).astype(np.uint8, copy=False)
    code |= ((g[2:, 0:-2] >= c) << 1).astype(np.uint8, copy=False)
    code |= ((g[1:-1, 0:-2] >= c) << 0).astype(np.uint8, copy=False)
    return code


FEAT_DIM = 12 + 2 + 28 + 8 + 256 + 3


def extract_features(image_bgr: np.ndarray) -> np.ndarray:
    """Simple, fast features from image content; deterministic."""
    image_bgr = _gray_world_wb_bgr(image_bgr)

    img = cv2.resize(image_bgr, (256, 256), interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    feats = np.empty((FEAT_DIM,), dtype=np.float32)
    k = 0

    for arr in (img2, hsv):
        arr2 = arr.reshape(-1, 3).astype(np.float32, copy=False)
        mean = arr2.mean(axis=0)
        std = arr2.std(axis=0)
        feats[k : k + 3] = mean
        k += 3
        feats[k : k + 3] = std
        k += 3

    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 80, 160)
    feats[k] = float(edges.mean()) / 255.0
    k += 1

    bgr = img2.astype(np.float32, copy=False)
    g = bgr[..., 1]
    r = bgr[..., 2]
    bch = bgr[..., 0]
    denom = r + g + bch + 1e-6
    feats[k] = (g / denom).mean()
    k += 1

    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    h_hist = (
        cv2.calcHist([h], [0], None, [12], [0.0, 180.0])
        .reshape(-1)
        .astype(np.float32, copy=False)
    )
    s_hist = (
        cv2.calcHist([s], [0], None, [8], [0.0, 256.0])
        .reshape(-1)
        .astype(np.float32, copy=False)
    )
    v_hist = (
        cv2.calcHist([v], [0], None, [8], [0.0, 256.0])
        .reshape(-1)
        .astype(np.float32, copy=False)
    )

    h_hist /= h_hist.sum() + 1e-6
    s_hist /= s_hist.sum() + 1e-6
    v_hist /= v_hist.sum() + 1e-6

    feats[k : k + 12] = h_hist
    k += 12
    feats[k : k + 8] = s_hist
    k += 8
    feats[k : k + 8] = v_hist
    k += 8

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)  # Speed: OpenCV magnitude
    ang = cv2.phase(gx, gy, angleInDegrees=True) + 180.0  # [-180,180] -> [0,360)
    hog_hist, _ = np.histogram(ang, bins=8, range=(0.0, 360.0), weights=mag)
    hog_hist = hog_hist.astype(np.float32, copy=False)
    hog_hist = hog_hist / (hog_hist.sum() + 1e-6)
    feats[k : k + 8] = hog_hist
    k += 8

    gray_u8 = gray.astype(np.uint8, copy=False)

    lbp = _lbp_u8(gray_u8)
    lbp_hist = np.bincount(lbp.reshape(-1), minlength=256).astype(
        np.float32, copy=False
    )
    lbp_hist /= lbp_hist.sum() + 1e-6
    feats[k : k + 256] = lbp_hist
    k += 256

    lap = cv2.Laplacian(gray_u8, cv2.CV_32F, ksize=3)
    feats[k] = float(lap.var()) / (255.0 * 255.0 + 1e-6)
    k += 1

    blur = cv2.GaussianBlur(gray_u8, (5, 5), 0)
    feats[k] = float(blur.std()) / (255.0 + 1e-6)
    k += 1

    edges2 = cv2.Canny(gray_u8, 40, 120)
    feats[k] = float(edges2.mean()) / 255.0
    k += 1

    return feats


def softmax_np(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x).astype(np.float32, copy=False)
    return e / (
        np.sum(e, axis=axis, keepdims=True).astype(np.float32, copy=False) + 1e-12
    )




## === cell 4
train_df = pd.read_csv(train_csv_path)
assert {"image_id", "label"}.issubset(train_df.columns)

fit_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

n0 = len(fit_df)
X = np.empty((2 * n0, FEAT_DIM), dtype=np.float32)
y = np.empty((2 * n0,), dtype=np.int64)


def _process_one(path: str):
    img = cv2.imread(path)
    if img is None:
        return None
    f1 = extract_features(img)
    img_flip = cv2.flip(img, 1)
    f2 = extract_features(img_flip)
    return (f1, f2)


paths = [os.path.join(train_img_dir, iid) for iid in fit_df["image_id"].values]
labels = fit_df["label"].values.astype(np.int64, copy=False)

max_workers = min(8, (os.cpu_count() or 2))
missing = 0
out_i = 0

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for i, res in enumerate(
        tqdm.tqdm(ex.map(_process_one, paths, chunksize=32), total=n0)
    ):
        if res is None:
            missing += 1
            continue
        f1, f2 = res
        lab = int(labels[i])
        X[out_i] = f1
        y[out_i] = lab
        out_i += 1
        X[out_i] = f2
        y[out_i] = lab
        out_i += 1

X = X[:out_i]
y = y[:out_i]

if X.shape[0] < 1000:
    raise RuntimeError(f"Too few training images loaded ({len(y)}). Missing: {missing}")

num_classes = 5
D = X.shape[1]

mu = X.mean(axis=0, keepdims=True).astype(np.float32, copy=False)
sigma = X.std(axis=0, keepdims=True).astype(np.float32, copy=False)
sigma = np.maximum(sigma, 1e-3).astype(np.float32, copy=False)
Xn = ((X - mu) / sigma).astype(np.float32, copy=False)

W = np.zeros((D, num_classes), dtype=np.float32)
b = np.zeros((1, num_classes), dtype=np.float32)

lr0 = 0.08
wd = 1e-3
epochs = 60
batch_size = 512

counts = np.bincount(y, minlength=num_classes).astype(np.float32, copy=False)

beta = np.float32(0.999)
effective_num = 1.0 - np.power(beta, counts)
class_w = (1.0 - beta) / np.maximum(effective_num, 1e-12)
class_w = class_w.astype(np.float32, copy=False)
class_w = (class_w / class_w.mean()).astype(
    np.float32, copy=False
)  # keep average weight ~1 for LR stability

rng = np.random.RandomState(42)
idx = np.arange(Xn.shape[0])
for ep in range(epochs):
    lr = lr0 * (0.97**ep)
    rng.shuffle(idx)
    Xs = Xn[idx]
    ys = y[idx]
    for start in range(0, Xs.shape[0], batch_size):
        xb = Xs[start : start + batch_size]
        yb = ys[start : start + batch_size]

        logits = xb @ W + b  # (B,C)
        pb = softmax_np(logits, axis=1)  # (B,C)

        grad_logits = pb.copy()
        w_y = class_w[yb].astype(np.float32, copy=False)  # (B,)
        grad_logits[np.arange(xb.shape[0]), yb] -= 1.0
        grad_logits *= w_y[:, None]

        denom = np.float32(w_y.sum() + 1e-12)
        grad_logits = grad_logits / denom

        grad_W = xb.T @ grad_logits + wd * W
        grad_b = grad_logits.sum(axis=0, keepdims=True)

        W -= np.float32(lr) * grad_W.astype(np.float32, copy=False)
        b -= np.float32(lr) * grad_b.astype(np.float32, copy=False)

logits_all = Xn @ W + b
pred_all = logits_all.argmax(axis=1)
acc_fit = (pred_all == y).mean()
print(
    f"Fitting-set accuracy (sanity check): {acc_fit:.4f} | used N={len(y)} images (missing={missing}) | D={D}"
)
print(
    "Class counts:",
    counts.astype(int).tolist(),
    "| class_w:",
    np.round(class_w, 3).tolist(),
)



## === cell 5
sample_sub = pd.read_csv(sample_sub_path)
assert {"image_id", "label"}.issubset(sample_sub.columns)

test_ids = sample_sub["image_id"].astype(str).values
preds = np.zeros((len(test_ids),), dtype=np.int64)

fallback_label = int(np.bincount(y, minlength=num_classes).argmax())

test_paths = [os.path.join(test_img_dir, iid) for iid in test_ids]


def _process_test_one(path: str):
    img = cv2.imread(path)
    if img is None:
        return None
    feat = extract_features(img).astype(np.float32, copy=False)
    return feat


with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for i, feat in enumerate(
        tqdm.tqdm(
            ex.map(_process_test_one, test_paths, chunksize=32), total=len(test_paths)
        )
    ):
        if feat is None:
            preds[i] = fallback_label
            continue
        feat = feat[None, :]
        feat = ((feat - mu) / sigma).astype(np.float32, copy=False)
        logits = feat @ W + b
        preds[i] = int(np.argmax(logits, axis=1)[0])

sub = pd.DataFrame({"image_id": test_ids, "label": preds})
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with", len(sub), "rows")
