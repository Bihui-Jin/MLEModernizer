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
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scipy==1.15.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.9669491813515626

# 6. Current score

0.76991

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Your notebook fails because it tries to read four external “../input/...” ensemble submissions that are not present in this environment, so all downstream variables are undefined. I keep the same “pick the lowest-entropy model per row” ensemble core logic, but make it robust by automatically using any available submission files if they exist, otherwise falling back to a valid baseline built from `sample_submission.csv` (uniform probabilities). I also enforce correct column order (`image_id, healthy, multiple_diseases, rust, scab`) and align predictions by `image_id` to avoid silent misalignment bugs. This run end-to-end and always write a valid `submission.csv`.'
- What this solution (achieved 0.5776) has done: 'Your current 0.5 score is coming from the uniform-probability fallback (or from selecting an uninformative baseline) because no real model predictions exist in this environment. To move toward the 0.9669 target without changing the ensemble selection logic, I generate legitimate predictions from the provided train images using a minimal scikit-learn baseline: pixel-intensity features + One-vs-Rest Logistic Regression, then feed those predictions into your existing “lowest-entropy per row” selector. This keeps the core “choose lowest-entropy model per row” semantics intact, but ensures at least one non-uniform candidate model is available, which should substantially improve ROC AUC over 0.5. I also keep strict `image_id` alignment and correct submission columns/ordering, and still write `submission.csv` end-to-end.'
- What this solution (achieved 0.65757) has done: 'Your current score is far below the target, so we should improve the predictive signal while keeping your ensemble’s core “pick lowest-entropy model per row” logic unchanged. The biggest issue is the very weak image representation (raw 64×64 RGB pixels) and a too-restrictive probability renormalization that can distort one-vs-rest probabilities for a ROC-AUC metric. I keep the same scikit-learn OneVsRest+LogisticRegression approach, but switch to a stronger yet still lightweight handcrafted feature (color + gradient HOG-like summary) and remove per-row probability sum-to-1 normalization (keeping only clipping), which typically improves multi-label ROC AUC. I also fix a masking/indexing bug risk by using positional indexing consistently when writing back selected predictions.'
- What this solution (achieved 0.65245) has done: 'Your score gap to the target is large (0.6576 vs 0.9669), so we need more predictive signal while keeping your ensemble’s “pick lowest-entropy model per row” core logic unchanged. The biggest issue is that the entropy selector currently prefers overconfident models, but the entropy is computed on unnormalized one-vs-rest probabilities, which makes entropy comparisons inconsistent across models; we compute entropy on per-row normalized probabilities *only for selection*, while leaving the actual output probabilities untouched for ROC-AUC. We also strengthen the same scikit-learn OVR Logistic Regression baseline without changing the model family by standardizing features (crucial for LBFGS/LogReg) and enabling class balancing to help rare classes. Finally, we keep strict `image_id` alignment and still write a valid `submission.csv`.'
- What this solution (achieved 0.67811) has done: 'Your current gap to the target is large, so we need more predictive signal while keeping your “train a scikit-learn OVR LogisticRegression on handcrafted image features + lowest-entropy per-row selection” core logic intact. The biggest low-risk gain is to fix a feature mismatch: you compute HOG-like histograms but don’t normalize gradient magnitude, so brightness/contrast dominate and reduce generalization; I L2-normalize gradients per image before building histograms. I also add a tiny amount of additional, still-handcrafted signal (simple RGB moments) to complement HSV and gradients without changing the model family or training loop. Finally, I make the entropy selection robust to pathological overconfidence by adding a tiny temperature smoothing only for the entropy computation (output probabilities remain unchanged), improving selector stability without changing evaluation semantics.'
- What this solution (achieved 0.75084) has done: 'We need a sizable score lift (0.678 → 0.967), so we keep your core approach (handcrafted features + OVR LogisticRegression + lowest-entropy per-row selector) but strengthen it with minimal, safe changes that don’t alter the model family or training loop. The main upgrades are: (1) add a simple low-frequency spatial signal by computing the same RGB/HSV moments on a 4×4 grid and concatenating them (still handcrafted, still fast), (2) make the gradient histogram a bit more robust by using 8 spatial cells (4×2) instead of 4 (2×2) while keeping the same histogram logic, and (3) improve generalization via a tiny fixed Ridge-style regularization change by slightly lowering C (stronger regularization) and increasing max_iter for convergence stability. All submission alignment/column order logic stays identical, and we still write a valid `submission.csv`.'
- What this solution (achieved 0.77166) has done: 'Your current gap to the target is large (0.75084 vs 0.96695), so we should increase predictive signal while keeping the same core pipeline (handcrafted features → OVR LogisticRegression → lowest-entropy per-row selector). The smallest high-impact change is to fix a mismatch between the metric (mean column-wise ROC AUC) and the way probabilities are produced: `OneVsRestClassifier.predict_proba` returns probabilities from independently-trained binary classifiers, which can be poorly calibrated; using the underlying per-class decision scores passed through a sigmoid (`expit`) typically improves ranking (AUC) without changing the model family or training loop. I also slightly strengthen generalization with a minimal, safe data-augmentation-by-feature step (horizontal flip) that only doubles the training features (still the same feature extractor and classifier) and tends to help image tasks. Everything else (alignment, column order, entropy selection, and writing `submission.csv`) stays the same.'
- What this solution (achieved 0.77026) has done: 'Main runtime is spent in per-image feature extraction (Python loops over pixels/patches and repeated disk reads) and in training two LogisticRegression OVR models. To stay within 600s without changing the algorithm, I cache each image’s multiscale features (including flipped) so every image is decoded/resized only once, replace the slow per-bin HOG loop with a vectorized `np.bincount` equivalent, and avoid `iterrows()` overhead by iterating over NumPy arrays. I also parallelize feature extraction across CPU cores using `multiprocessing.Pool` (deterministic because extraction is pure) and keep the model/training settings identical. All file paths, targets, architecture (OVR LogisticRegression with StandardScaler), and ensemble entropy selection remain unchanged.'
- What this solution (achieved 0.76991) has done: 'We’re far below the target (0.770 vs 0.967), so we should increase predictive signal without changing your core pipeline (handcrafted features → OVR LogisticRegression → entropy-based per-row selector). The lowest-risk score lift here is to add two more candidate LogisticRegression models with slightly different regularization strengths (C values) so the entropy selector can pick the best per-row; this preserves the exact ensemble-selection semantics and training approach. I also make the multiprocessing start method robust (use `fork` when available, otherwise `spawn`) to avoid silent feature-extraction failures on some runners, which can otherwise degrade predictions. Everything else (feature extractor, use of decision_function+sigmoid, entropy computed on normalized probs but output unchanged, alignment, submission columns) stays the same.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.special import expit

from PIL import Image
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(0)



## === cell 1
BASE_DIR = "/kaggle/data"
COMP_DIR = os.path.join(BASE_DIR, "plant-pathology-2020-fgvc7")

sample_path_candidates = [
    os.path.join(COMP_DIR, "sample_submission.csv"),
    os.path.join(BASE_DIR, "sample_submission.csv"),
]
test_path_candidates = [
    os.path.join(COMP_DIR, "test.csv"),
    os.path.join(BASE_DIR, "test.csv"),
]
train_path_candidates = [
    os.path.join(COMP_DIR, "train.csv"),
    os.path.join(BASE_DIR, "train.csv"),
]

images_dir_candidates = [
    os.path.join(COMP_DIR, "images"),
    os.path.join(BASE_DIR, "images"),
]


def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


SAMPLE_PATH = first_existing(sample_path_candidates)
TEST_PATH = first_existing(test_path_candidates)
TRAIN_PATH = first_existing(train_path_candidates)
IMAGES_DIR = first_existing(images_dir_candidates)

if SAMPLE_PATH is None or TEST_PATH is None or TRAIN_PATH is None or IMAGES_DIR is None:
    raise FileNotFoundError(
        f"Could not find required files. SAMPLE_PATH={SAMPLE_PATH}, TEST_PATH={TEST_PATH}, "
        f"TRAIN_PATH={TRAIN_PATH}, IMAGES_DIR={IMAGES_DIR}"
    )

sub_template = pd.read_csv(SAMPLE_PATH)
test_df = pd.read_csv(TEST_PATH)
train_df = pd.read_csv(TRAIN_PATH)

target_cols = [c for c in sub_template.columns if c != "image_id"]
required_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]

if list(sub_template.columns) != required_cols:
    missing = set(required_cols) - set(sub_template.columns)
    if missing:
        raise ValueError(f"Sample submission missing required columns: {missing}")
    sub_template = sub_template[required_cols]
    target_cols = [c for c in sub_template.columns if c != "image_id"]

sub = sub_template.merge(test_df[["image_id"]], on="image_id", how="right")
sub = sub[required_cols]



## === cell 2
preferred_paths = [
    "../input/average-efficientnet/submission.csv",
    "../input/classification-densenet201-efficientnetb7/submission.csv",
    "../input/tf-zoo-models-on-tpu/submission.csv",
    "../input/fork-of-plant-2020-tpu-915e9c/submission.csv",
]

available_paths = [p for p in preferred_paths if os.path.exists(p)]

search_roots = [
    "/kaggle/input",
    "/kaggle/data",
    "/kaggle/working",
]
for root in search_roots:
    if os.path.exists(root):
        found = glob.glob(os.path.join(root, "**", "submission.csv"), recursive=True)
        available_paths.extend(found)

seen = set()
dedup_paths = []
for p in available_paths:
    if p not in seen:
        seen.add(p)
        dedup_paths.append(p)
available_paths = dedup_paths


def load_and_align_submission(path, template_image_ids, required_cols, target_cols):
    df = pd.read_csv(path)
    if "image_id" not in df.columns:
        return None
    if not set(target_cols).issubset(df.columns):
        return None
    df = df[["image_id"] + target_cols].copy()
    df = pd.DataFrame({"image_id": template_image_ids}).merge(
        df, on="image_id", how="left"
    )
    if df[target_cols].isna().any().any():
        return None
    df = df[required_cols]
    df[target_cols] = df[target_cols].astype(float).clip(1e-7, 1 - 1e-7)
    return df


template_image_ids = sub["image_id"].tolist()
subs = []
for p in available_paths:
    aligned = load_and_align_submission(
        p, template_image_ids, required_cols, target_cols
    )
    if aligned is not None:
        subs.append(aligned)

baseline = sub.copy()
baseline[target_cols] = 1.0 / len(target_cols)



## === cell 3
import multiprocessing as mp


def _extract_features_from_rgb(rgb, bins=9):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    diff = mx - mn

    h = np.zeros_like(mx, dtype=np.float32)
    mask = diff > 1e-8
    idx = (mx == r) & mask
    h[idx] = ((g[idx] - b[idx]) / diff[idx]) % 6.0
    idx = (mx == g) & mask
    h[idx] = ((b[idx] - r[idx]) / diff[idx]) + 2.0
    idx = (mx == b) & mask
    h[idx] = ((r[idx] - g[idx]) / diff[idx]) + 4.0
    h = (h / 6.0).astype(np.float32)

    s = np.zeros_like(mx, dtype=np.float32)
    mpos = mx > 1e-8
    s[mpos] = (diff[mpos] / mx[mpos]).astype(np.float32)
    v = mx.astype(np.float32)

    def moments(ch):
        return np.array(
            [
                ch.mean(),
                ch.std(),
                np.percentile(ch, 10),
                np.percentile(ch, 50),
                np.percentile(ch, 90),
            ],
            dtype=np.float32,
        )

    def grid_moments(ch, gh=4, gw=4):
        H, W = ch.shape
        feats = np.empty((gh * gw, 5), dtype=np.float32)
        t = 0
        for i in range(gh):
            i0 = (i * H) // gh
            i1 = ((i + 1) * H) // gh
            for j in range(gw):
                j0 = (j * W) // gw
                j1 = ((j + 1) * W) // gw
                patch = ch[i0:i1, j0:j1]
                feats[t] = moments(patch)
                t += 1
        return feats.ravel()

    rgb_feat_global = np.concatenate([moments(r), moments(g), moments(b)], axis=0)
    hsv_feat_global = np.concatenate([moments(h), moments(s), moments(v)], axis=0)

    rgb_feat_grid = np.concatenate(
        [grid_moments(r), grid_moments(g), grid_moments(b)], axis=0
    )
    hsv_feat_grid = np.concatenate(
        [grid_moments(h), grid_moments(s), grid_moments(v)], axis=0
    )

    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.float32)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]

    mag = np.sqrt(gx * gx + gy * gy) + 1e-8
    ang = (np.arctan2(gy, gx) + np.pi) / (2.0 * np.pi)  # 0..1

    mag = mag / (np.sqrt((mag * mag).mean()) + 1e-8)

    H, W = gray.shape
    hs = (
        slice(0, H // 4),
        slice(H // 4, H // 2),
        slice(H // 2, (3 * H) // 4),
        slice((3 * H) // 4, H),
    )
    ws = (slice(0, W // 2), slice(W // 2, W))

    hog_parts = []
    for si in hs:
        for sj in ws:
            a = ang[si, sj].ravel()
            m = mag[si, sj].ravel()
            bi = np.minimum((a * bins).astype(np.int32), bins - 1)
            hist = np.bincount(bi, weights=m, minlength=bins).astype(np.float32)
            hist = hist / (hist.sum() + 1e-8)
            hog_parts.append(hist)
    hog_feat = np.concatenate(hog_parts, axis=0)

    feat = np.concatenate(
        [rgb_feat_global, hsv_feat_global, rgb_feat_grid, hsv_feat_grid, hog_feat],
        axis=0,
    ).astype(np.float32)
    return feat


def _extract_multiscale_pair(
    image_id, images_dir, sizes=((96, 96), (160, 160)), bins=9
):
    img_path = os.path.join(images_dir, f"{image_id}.jpg")
    if not os.path.exists(img_path):
        return None

    feats_nf = []
    feats_f = []
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        for sz in sizes:
            imr = im.resize(sz)
            rgb = np.asarray(imr, dtype=np.float32) / 255.0
            feats_nf.append(_extract_features_from_rgb(rgb, bins=bins))

            imrf = imr.transpose(Image.FLIP_LEFT_RIGHT)
            rgbf = np.asarray(imrf, dtype=np.float32) / 255.0
            feats_f.append(_extract_features_from_rgb(rgbf, bins=bins))

    f_nf = np.concatenate(feats_nf, axis=0).astype(np.float32)
    f_f = np.concatenate(feats_f, axis=0).astype(np.float32)
    return (image_id, f_nf, f_f)


_MP_IMAGES_DIR = None


def _mp_init(images_dir):
    global _MP_IMAGES_DIR
    _MP_IMAGES_DIR = images_dir


def _mp_worker(image_id):
    return _extract_multiscale_pair(image_id, _MP_IMAGES_DIR)


def build_sklearn_submissions(
    train_df, test_df, images_dir, target_cols, required_cols
):
    train_ids = train_df["image_id"].to_numpy()
    test_ids = test_df["image_id"].to_numpy()

    ctx = (
        mp.get_context("fork")
        if "fork" in mp.get_all_start_methods()
        else mp.get_context("spawn")
    )

    nproc = max(1, min(8, (os.cpu_count() or 2)))
    with ctx.Pool(
        processes=nproc, initializer=_mp_init, initargs=(images_dir,)
    ) as pool:
        train_res = list(pool.imap(_mp_worker, train_ids, chunksize=32))
        test_res = list(pool.imap(_mp_worker, test_ids, chunksize=32))

    X_list = []
    y_list = []
    y_arr = train_df[target_cols].to_numpy(dtype=int)
    for i, r in enumerate(train_res):
        if r is None:
            continue
        _, f_nf, f_f = r
        X_list.append(f_nf)
        y_list.append(y_arr[i])
        X_list.append(f_f)
        y_list.append(y_arr[i])

    if len(X_list) == 0:
        return []

    X_train = np.vstack(X_list)
    y_train = np.vstack(y_list)

    X_test_list = []
    ok_ids = []
    for r in test_res:
        if r is None:
            return []
        image_id, f_nf, _ = r
        X_test_list.append(f_nf)
        ok_ids.append(image_id)
    X_test = np.vstack(X_test_list)

    submissions = []
    for C in (0.8, 1.2, 2.0, 3.0):
        base_lr = LogisticRegression(
            solver="lbfgs",
            max_iter=1600,
            C=C,
            class_weight="balanced",
            n_jobs=-1,
        )
        clf = OneVsRestClassifier(
            make_pipeline(StandardScaler(with_mean=True, with_std=True), base_lr)
        )
        clf.fit(X_train, y_train)

        scores = clf.decision_function(X_test)
        proba = expit(scores).astype(float)
        proba = np.clip(proba, 1e-7, 1 - 1e-7)

        df = pd.DataFrame({"image_id": ok_ids})
        for j, c in enumerate(target_cols):
            df[c] = proba[:, j]
        df = df[required_cols]
        submissions.append(df)

    return submissions


sk_model_subs = build_sklearn_submissions(
    train_df, test_df, IMAGES_DIR, target_cols, required_cols
)
for d in sk_model_subs:
    subs.append(d)

if len(subs) == 0:
    subs = [baseline]
else:
    subs.append(baseline)

sub1 = subs[0]
sub2 = subs[1] if len(subs) > 1 else subs[0]
sub3 = subs[2] if len(subs) > 2 else subs[0]
sub4 = subs[3] if len(subs) > 3 else subs[0]



## === cell 4
pred_stack = np.stack(
    [s[target_cols].to_numpy(dtype=float) for s in subs], axis=0
)  # (M, N, C)
pred_stack = np.clip(pred_stack, 1e-12, 1.0)  # avoid log(0) in entropy

row_sums = pred_stack.sum(axis=2, keepdims=True)  # (M, N, 1)
row_sums = np.where(row_sums <= 0, 1.0, row_sums)
pred_stack_for_entropy = pred_stack / row_sums

eps = 1e-4
pred_stack_for_entropy = (1.0 - eps) * pred_stack_for_entropy + eps * (
    1.0 / len(target_cols)
)

entropies = entropy(pred_stack_for_entropy, base=2, axis=2)  # (M, N)
selected = np.argmin(entropies, axis=0)



## === cell 5
final_preds = np.zeros((len(sub), len(target_cols)), dtype=float)
for m_idx, s in enumerate(subs):
    mask = selected == m_idx
    if np.any(mask):
        idx = np.where(mask)[0]
        final_preds[idx] = s.iloc[idx][target_cols].to_numpy(dtype=float)

sub[target_cols] = final_preds
sub[target_cols] = sub[target_cols].astype(float).clip(1e-7, 1 - 1e-7)
sub = sub[required_cols]
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print("Number of candidate models in entropy stack:", len(subs))
print(sub.head())
