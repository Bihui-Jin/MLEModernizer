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

0.53999

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63341) has done: 'I remove the broken EfficientNet + missing checkpoint dependency that prevents `model` from being created, since `efficientnet_pytorch` isn’t installed and `b5.pth` doesn’t exist in the provided paths. To keep the pipeline valid end-to-end, I replace that section with a lightweight, deterministic image classifier built from simple color/statistics features computed from each image; this preserves the overall “read images → preprocess → predict → write submission.csv” flow and guarantees a valid CSV. I also ensure predictions are produced for every `sample_submission.csv` image_id (matching required order/row count) and add safe fallbacks for unreadable images. This should yield a non-trivial accuracy (better than random) while prioritizing correctness and Kaggle submission validity.'
- What this solution (achieved 0.63565) has done: 'You’re currently far below the target (0.63341 vs 0.86567), so we should make the smallest changes that legitimately improve accuracy without changing the overall “handcrafted features → simple linear classifier → argmax → submission.csv” core. The biggest win (with minimal disruption) is to remove the intentional downsampling and instead train on all available training images, since this competition is data-hungry and your model is very lightweight; this should move accuracy up substantially toward the target. Second, we add a very small amount of numeric stabilization (float32 throughout, clipped std) and a tiny learning-rate decay schedule (still the same training loop/optimizer math) to improve convergence without changing the model class. All paths and submission formatting remain identical.'
- What this solution (achieved 0.63453) has done: 'Your current pipeline is a simple handcrafted-feature + linear softmax classifier; to move accuracy upward toward the 0.8657 target without changing that core approach, I strengthen the feature vector slightly while keeping the same extraction flow and training loop. Specifically, I add a compact color-histogram in HSV plus a small gradient-orientation histogram (HOG-like) computed from Sobel edges, which are cheap and often improve disease/texture separability in cassava images. I also switch the training objective from one-hot MSE-on-softmax to the standard softmax cross-entropy gradient (still the same linear model and minibatch SGD), which typically converges to better classification accuracy for the same architecture. All paths and submission formatting stay identical, and the script still runs end-to-end and writes `submission.csv`.'
- What this solution (achieved 0.65882) has done: 'Your current score is far below the target (0.63453 vs 0.86567; higher-is-better), so we should improve accuracy with minimal, low-risk changes while keeping the same “handcrafted features → linear softmax classifier → argmax → submission.csv” core. The biggest likely gain without changing the model/training loop is to make the handcrafted features more disease/texture-sensitive by adding a compact Local Binary Pattern (LBP) histogram and a few simple blur/contrast/sharpness statistics. These are fast, deterministic, and fit naturally into your existing `extract_features()` flow, often giving a noticeable lift on leaf-disease imagery. All paths, training approach, loss/gradient, and submission formatting remain the same.'
- What this solution (achieved 0.53999) has done: 'Your current score (0.65882) is well below the target (0.86567), so we should make a small, legitimate accuracy improvement while preserving the same handcrafted-features → linear softmax classifier → SGD training loop. The biggest low-risk gain here is to rebalance the training signal: this dataset is class-imbalanced, so adding per-class weights inside the same cross-entropy gradient (no architecture/training-loop change) usually improves accuracy materially. I also switch the weight decay term to the standard “exclude bias from decay” variant (still identical model, just a small regularization fix), and keep everything else (features, optimizer structure, paths, submission format) unchanged. This should move the score upward toward your target without introducing new dependencies or changing core semantics.'

# 9. Code solution

## === cell 0
import os, sys, glob, json, random
import numpy as np
import pandas as pd



## === cell 1
import cv2
import tqdm

random.seed(42)
np.random.seed(42)



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


def _lbp_u8(gray_u8: np.ndarray) -> np.ndarray:
    """
    Minimal, fast LBP (8 neighbors, radius=1).
    Returns uint8 codes in [0,255].
    """
    g = gray_u8
    c = g[1:-1, 1:-1].astype(np.uint8)
    code = np.zeros_like(c, dtype=np.uint8)

    code |= ((g[0:-2, 0:-2] >= c) << 7).astype(np.uint8)
    code |= ((g[0:-2, 1:-1] >= c) << 6).astype(np.uint8)
    code |= ((g[0:-2, 2:] >= c) << 5).astype(np.uint8)
    code |= ((g[1:-1, 2:] >= c) << 4).astype(np.uint8)
    code |= ((g[2:, 2:] >= c) << 3).astype(np.uint8)
    code |= ((g[2:, 1:-1] >= c) << 2).astype(np.uint8)
    code |= ((g[2:, 0:-2] >= c) << 1).astype(np.uint8)
    code |= ((g[1:-1, 0:-2] >= c) << 0).astype(np.uint8)
    return code


def extract_features(image_bgr: np.ndarray) -> np.ndarray:
    """Simple, fast features from image content; deterministic."""
    img = cv2.resize(image_bgr, (256, 256), interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    feats = []
    for arr in (img2, hsv):
        arr2 = arr.reshape(-1, 3).astype(np.float32)
        mean = arr2.mean(axis=0)
        std = arr2.std(axis=0)
        feats.extend(mean.tolist())
        feats.extend(std.tolist())

    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 80, 160)
    feats.append(float(edges.mean()) / 255.0)

    bgr = img2.astype(np.float32)
    g = bgr[..., 1]
    r = bgr[..., 2]
    bch = bgr[..., 0]
    denom = r + g + bch + 1e-6
    green_ratio = (g / denom).mean()
    feats.append(float(green_ratio))

    h = hsv[..., 0].astype(np.float32)
    s = hsv[..., 1].astype(np.float32)
    v = hsv[..., 2].astype(np.float32)
    h_hist, _ = np.histogram(h, bins=12, range=(0.0, 180.0))
    s_hist, _ = np.histogram(s, bins=8, range=(0.0, 256.0))
    v_hist, _ = np.histogram(v, bins=8, range=(0.0, 256.0))
    h_hist = (h_hist.astype(np.float32) / (h_hist.sum() + 1e-6)).tolist()
    s_hist = (s_hist.astype(np.float32) / (s_hist.sum() + 1e-6)).tolist()
    v_hist = (v_hist.astype(np.float32) / (v_hist.sum() + 1e-6)).tolist()
    feats.extend(h_hist)
    feats.extend(s_hist)
    feats.extend(v_hist)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.arctan2(gy, gx) + np.pi) * (180.0 / np.pi)  # [0, 360)
    hog_hist, _ = np.histogram(ang, bins=8, range=(0.0, 360.0), weights=mag)
    hog_hist = hog_hist.astype(np.float32)
    hog_hist = hog_hist / (hog_hist.sum() + 1e-6)
    feats.extend(hog_hist.tolist())

    gray_u8 = gray.astype(np.uint8, copy=False)

    lbp = _lbp_u8(gray_u8)
    lbp_hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)
    feats.extend(lbp_hist.tolist())

    lap = cv2.Laplacian(gray_u8, cv2.CV_32F, ksize=3)
    feats.append(float(lap.var()) / (255.0 * 255.0 + 1e-6))

    blur = cv2.GaussianBlur(gray_u8, (5, 5), 0)
    feats.append(float(blur.std()) / (255.0 + 1e-6))

    edges2 = cv2.Canny(gray_u8, 40, 120)
    feats.append(float(edges2.mean()) / 255.0)

    return np.asarray(feats, dtype=np.float32)


def softmax_np(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x).astype(np.float32)
    return e / (np.sum(e, axis=axis, keepdims=True).astype(np.float32) + 1e-12)




## === cell 4
train_df = pd.read_csv(train_csv_path)
assert {"image_id", "label"}.issubset(train_df.columns)

fit_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

X_list = []
y_list = []

missing = 0
for image_id, label in tqdm.tqdm(
    zip(fit_df["image_id"].values, fit_df["label"].values), total=len(fit_df)
):
    path = os.path.join(train_img_dir, image_id)
    img = cv2.imread(path)
    if img is None:
        missing += 1
        continue
    X_list.append(extract_features(img))
    y_list.append(int(label))

if len(X_list) < 1000:
    raise RuntimeError(
        f"Too few training images loaded ({len(X_list)}). Missing: {missing}"
    )

X = np.stack(X_list, axis=0).astype(np.float32)  # (N, D)
y = np.asarray(y_list, dtype=np.int64)
num_classes = 5
D = X.shape[1]

mu = X.mean(axis=0, keepdims=True).astype(np.float32)
sigma = X.std(axis=0, keepdims=True).astype(np.float32)
sigma = np.maximum(sigma, 1e-3).astype(np.float32)
Xn = ((X - mu) / sigma).astype(np.float32)

W = np.zeros((D, num_classes), dtype=np.float32)
b = np.zeros((1, num_classes), dtype=np.float32)

lr0 = 0.08
wd = 1e-3
epochs = 60
batch_size = 512

counts = np.bincount(y, minlength=num_classes).astype(np.float32)
class_w = (counts.sum() / (num_classes * np.maximum(counts, 1.0))).astype(
    np.float32
)  # inverse-freq
class_w = (class_w / class_w.mean()).astype(
    np.float32
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

        grad_logits = pb
        grad_logits[np.arange(xb.shape[0]), yb] -= 1.0

        sw = class_w[yb].astype(np.float32)  # (B,)
        grad_logits = grad_logits * sw[:, None]
        grad_logits = grad_logits / (np.float32(xb.shape[0]) + 1e-12)

        grad_W = xb.T @ grad_logits + wd * W
        grad_b = grad_logits.sum(axis=0, keepdims=True)

        W -= np.float32(lr) * grad_W.astype(np.float32)
        b -= np.float32(lr) * grad_b.astype(np.float32)

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

for i, image_id in enumerate(tqdm.tqdm(test_ids)):
    path = os.path.join(test_img_dir, image_id)
    img = cv2.imread(path)
    if img is None:
        preds[i] = fallback_label
        continue
    feat = extract_features(img)[None, :].astype(np.float32)  # (1, D)
    feat = ((feat - mu) / sigma).astype(np.float32)
    logits = feat @ W + b
    preds[i] = int(np.argmax(logits, axis=1)[0])

sub = pd.DataFrame({"image_id": test_ids, "label": preds})
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with", len(sub), "rows")
