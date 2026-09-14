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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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

0.9699976501828376

# 6. Current score

0.66517

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.49317) has done: 'I fix the runtime error by removing the hard dependency on a non-existent `/kaggle/input/submissions/submissions/` dataset and instead generate a valid prediction file directly from the provided competition data. To preserve the “ensemble submissions” core logic, the script still average/weight multiple “submissions”, but when none are available it create two simple baseline submissions (image-based brightness heuristics) and ensemble them. This guarantees an end-to-end run that writes `submission.csv` with the exact required columns and row alignment to `test.csv`. The approach is fully self-contained in the given environment and avoids changing the competition’s evaluation semantics (still outputs probabilities per class).'
- What this solution (achieved 0.64415) has done: 'The timeout is dominated by repeatedly opening and decoding ~1.8k JPEGs in pure Python during feature extraction; the logistic regression itself is cheap. I keep the exact same features and modeling logic, but speed up I/O and per-image compute by (1) using `PIL.ImageStat` for mean/std and grayscale stats without materializing full float arrays, (2) computing percentiles and ExG/ExR on a single grayscale/float array only once per image, and (3) parallelizing feature extraction with a deterministic thread pool (JPEG decode releases the GIL) while preserving row order. I also cache resolved image paths to avoid repeated filesystem checks. These changes are provably equivalent in semantics (same features, same training/prediction), with only negligible floating-point differences.'
- What this solution (achieved 0.63738) has done: 'Your current score (0.64415) is far below the target (0.96999), so we should improve AUC with minimal, metric-aligned changes while keeping the same overall pipeline (handcrafted image features → one-vs-rest LogisticRegression → average two modes). The biggest low-risk gain is to handle class imbalance correctly: `multiple_diseases` is rare, and plain logistic regression under-predicts it, hurting mean per-class AUC; adding `class_weight="balanced"` keeps the same model/loop but improves ranking for minority classes. I also increase `max_iter` slightly to ensure convergence with the same solver and features (no early stopping), and keep everything else (features, ensemble, submission writing) identical. These changes should move the score upward toward the target without changing the core approach.'
- What this solution (achieved 0.63738) has done: 'We keep the exact same pipeline (handcrafted image stats → per-class StandardScaler + LogisticRegression → average two modes) but make two minimal, metric-aligned fixes that typically improve mean column-wise ROC AUC substantially. First, we correct the label imbalance handling: `class_weight="balanced"` inside a per-class one-vs-rest loop can overcompensate; instead we use explicit `sample_weight` per class computed from that class’s positives/negatives, which preserves the same model and training loop but gives better ranking calibration. Second, we switch the logistic loss to `penalty="l2"` explicitly and increase `max_iter` modestly for stable convergence (no early stopping), keeping everything else identical and still writing a valid `submission.csv`.'
- What this solution (achieved 0.6246) has done: 'Your current score (0.63738) is far below the target (0.96999), so we need a meaningful AUC lift while keeping the same core pipeline (handcrafted features → per-class StandardScaler+LogisticRegression → 2-mode average). The most score-relevant minimal change is to add a few additional, cheap color/texture statistics per image (still simple global stats; no architecture/training loop change) and keep the exact same one-vs-rest LR training/prediction flow. This tends to improve separability for rust/scab vs healthy and helps the rare `multiple_diseases` class without changing the evaluation semantics. I also keep determinism and submission alignment intact.'
- What this solution (achieved 0.6246) has done: 'We keep your exact pipeline (handcrafted global image features → per-class StandardScaler+LogisticRegression → 2-mode average) but fix a score-killing misalignment: you currently extract train features using `train_df["image_id"]` order, yet fit labels using the raw `train_df[targets].values` order, which can silently mismatch if `train_df` isn’t in the same order as `train_ids` used for feature extraction (or if any later reindexing happens). I make the train/test feature matrices explicitly aligned to the exact `image_id` order used for both features and labels by setting indices and reindexing once, preserving semantics while improving AUC. I also make the submission-writing branch (when external submissions exist) align rows to `test.csv` by `image_id` (not by file row order), which prevents accidental row-order mistakes that can tank the public score. These are minimal, correctness-focused changes that typically raise ROC AUC substantially without changing the model or training loop.'
- What this solution (achieved 0.64743) has done: 'Your current score (0.6246) is far below the target (0.96999), so we need a real AUC lift while keeping your same core pipeline (global handcrafted image stats → per-class StandardScaler+LogisticRegression → 2-mode average). The most score-relevant minimal change is to add very lightweight “shape/size + saturation + edge energy” global features that are still simple per-image statistics, improving separability for rust/scab patterns without changing the model family or training loop. To preserve existing semantics, I keep the same LogisticRegression setup, per-class sample_weight balancing, and the same 2-mode averaging; only the feature vector grows. I also keep strict alignment by `image_id` and ensure we still write a valid `submission.csv` with correct ordering and columns.'
- What this solution (achieved 0.64523) has done: 'I keep your exact pipeline (global handcrafted image stats → per-class StandardScaler+LogisticRegression → 2-mode average → submission.csv), but make two metric-aligned tweaks that typically improve mean ROC AUC without changing the core approach. First, I set `fit_intercept=False` because StandardScaler already centers features and the intercept can add unnecessary degrees of freedom that hurt ranking stability on small/imbalanced classes. Second, I modestly increase `C` for both modes (less regularization) to better separate classes with these richer features, while keeping the same training loop, solver, and sample-weight balancing. Everything else (feature extraction, alignment by image_id, and submission formatting) remains unchanged.'
- What this solution (achieved 0.64782) has done: 'Your current score (0.64523) is far below the target (0.96999), so we need a real AUC lift while keeping your exact pipeline (handcrafted global image features → per-class StandardScaler + LogisticRegression → 2-mode averaging). The smallest high-impact fix is to make the LogisticRegression consistent with its own scaling: with `StandardScaler(with_mean=True)`, setting `fit_intercept=False` can hurt class ranking; switching back to `fit_intercept=True` usually improves ROC AUC without changing the model family, loss, or training loop. To preserve your two-mode ensemble semantics, I keep everything else identical and only adjust intercept usage, plus a tiny C retune to avoid overfitting when the intercept is enabled. This should move the score upward toward the target while remaining stable and within Kaggle constraints.'
- What this solution (achieved 0.64886) has done: 'I keep your exact pipeline (global handcrafted image features → per-class StandardScaler+LogisticRegression → 2-mode averaging) and make two small, score-relevant adjustments that typically improve mean per-class ROC AUC without changing the core logic. First, I increase the LogisticRegression strength slightly (higher C) to reduce underfitting on these simple features while keeping the same solver/loss/training loop. Second, I use a slightly asymmetric ensemble weight (favoring the higher-C mode) since it’s usually the stronger ranker, while still preserving the same 2-model averaging semantics. Everything else (feature extraction, sample-weight balancing, alignment by image_id, and submission formatting) remains unchanged and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.66517) has done: 'Your current score (0.64886) is far below the target (0.96999), so we need a meaningful AUC lift while keeping the same core pipeline (global handcrafted image features → per-class StandardScaler+LogisticRegression → 2-mode averaging). The most impactful minimal change is to use simple multi-label constraints during training and inference: at most one of {healthy, rust, scab} for most samples, and “multiple_diseases” should correlate with having both rust+scab; we can encode this by (a) adding two extra features (rust*scab interaction and max(rust,scab)) derived only from existing features, and (b) a lightweight post-processing that maps independent per-class probabilities into a consistent 4-class distribution (same submission columns) without touching the model family or loss. This tends to improve mean column-wise ROC AUC because it reduces contradictory rankings (e.g., high healthy and high scab) and strengthens the rare multiple_diseases signal. All I/O paths stay the same, training loop remains one-vs-rest LogisticRegression, and the script still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

np.random.seed(0)



## === cell 1
BASE_CANDIDATES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/data",
]

_FOUND_CACHE = {}


def _find_file(filename):
    if filename in _FOUND_CACHE:
        return _FOUND_CACHE[filename]

    for base in BASE_CANDIDATES:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            _FOUND_CACHE[filename] = path
            return path

    for base in BASE_CANDIDATES:
        cand = os.path.join(base, "plant-pathology-2020-fgvc7", filename)
        if os.path.exists(cand):
            _FOUND_CACHE[filename] = cand
            return cand

    raise FileNotFoundError(f"Could not locate {filename} in known Kaggle input paths.")


TRAIN_CSV = _find_file("train.csv")
TEST_CSV = _find_file("test.csv")
SAMPLE_SUB = _find_file("sample_submission.csv")


_IMAGES_DIR_CACHE = None


def _find_images_dir():
    global _IMAGES_DIR_CACHE
    if _IMAGES_DIR_CACHE is not None:
        return _IMAGES_DIR_CACHE

    candidates = []
    for base in BASE_CANDIDATES:
        candidates.extend(
            [
                os.path.join(base, "images"),
                os.path.join(base, "plant-pathology-2020-fgvc7", "images"),
            ]
        )
    for c in candidates:
        if os.path.isdir(c):
            _IMAGES_DIR_CACHE = c
            return c

    raise FileNotFoundError("Could not locate images directory.")


IMAGES_DIR = _find_images_dir()

print("Using:")
print("TRAIN_CSV:", TRAIN_CSV)
print("TEST_CSV :", TEST_CSV)
print("SAMPLE   :", SAMPLE_SUB)
print("IMAGES   :", IMAGES_DIR)



## === cell 2
SUBMISSIONS_PATH = "/kaggle/input/submissions/submissions/"
submissions_all = []
if os.path.isdir(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.endswith(".csv") or filename.endswith(".CSV"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print("Found submission files:", submissions_all)




## === cell 3
def ensemble(submissions_all, sub_idx, weights=None):
    if weights is None:
        weights = [1.0] * len(sub_idx)
    if len(sub_idx) != len(weights):
        raise ValueError("sub_idx and weights must have the same length.")

    cols = ["healthy", "multiple_diseases", "rust", "scab"]
    acc = None
    wsum = 0.0
    for i, (idx, w) in enumerate(zip(sub_idx, weights)):
        path = submissions_all[idx]
        w = float(w)
        print(f"I'm taking submission {path} with weight {w}")
        arr = pd.read_csv(path, usecols=cols).to_numpy(dtype=np.float64, copy=False)
        if acc is None:
            acc = np.zeros_like(arr, dtype=np.float64)
        acc += arr * w
        wsum += w
    if wsum != 0.0:
        acc /= wsum
    return acc


def make_submission_file(submission_avg, template_csv_path, out_path="submission.csv"):
    submission_df = pd.read_csv(template_csv_path)
    if "image_id" in submission_df.columns:
        submission_df["image_id"] = submission_df["image_id"].astype(str)
    submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = submission_avg
    submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = submission_df[
        ["healthy", "multiple_diseases", "rust", "scab"]
    ].clip(0.0, 1.0)
    submission_df.to_csv(out_path, index=False)
    print(
        f"Wrote {out_path} with shape {submission_df.shape} and columns {list(submission_df.columns)}"
    )


def _load_and_align_submission(path, test_ids):
    df = pd.read_csv(path)
    if "image_id" not in df.columns:
        raise ValueError(f"{path} is missing image_id column; cannot safely align.")
    df["image_id"] = df["image_id"].astype(str)
    cols = ["healthy", "multiple_diseases", "rust", "scab"]
    df = df.set_index("image_id")[cols].reindex(test_ids)
    if df.isna().any().any():
        raise ValueError(f"{path} is missing some test image_ids after alignment.")
    return df.to_numpy(dtype=np.float64, copy=False)


def _consistent_probs_from_independent(probs4: np.ndarray) -> np.ndarray:
    p = np.asarray(probs4, dtype=np.float64)
    p = np.clip(p, 1e-9, 1.0 - 1e-9)

    healthy = p[:, 0]
    md_ind = p[:, 1]
    rust = p[:, 2]
    scab = p[:, 3]

    both = rust * scab
    none = (1.0 - rust) * (1.0 - scab)

    md_score = 0.75 * both + 0.25 * md_ind

    rust_only_score = rust * (1.0 - scab)
    scab_only_score = scab * (1.0 - rust)

    healthy_score = healthy * none

    scores = np.stack(
        [healthy_score, md_score, rust_only_score, scab_only_score], axis=1
    )
    scores = np.clip(scores, 1e-12, None)
    scores /= scores.sum(axis=1, keepdims=True)
    return np.clip(scores, 1e-6, 1.0 - 1e-6)




## === cell 4
test_df = pd.read_csv(TEST_CSV)
train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

test_ids = test_df["image_id"].astype(str).tolist()

train_df["image_id"] = train_df["image_id"].astype(str)
train_df = train_df.set_index("image_id")
train_ids = train_df.index.tolist()

sample_df["image_id"] = sample_df["image_id"].astype(str)
sample_df = sample_df.set_index("image_id").loc[test_ids].reset_index()

from PIL import Image, ImageStat


def _build_image_map(images_dir):
    mp = {}
    for fn in os.listdir(images_dir):
        mp[fn.lower()] = fn
    return mp


_IMAGE_MAP = _build_image_map(IMAGES_DIR)

_IMG_PATH_CACHE = {}


def _safe_img_path(image_id):
    image_id = str(image_id)
    p = _IMG_PATH_CACHE.get(image_id)
    if p is not None:
        return p

    p1 = os.path.join(IMAGES_DIR, f"{image_id}.jpg")
    if os.path.exists(p1):
        _IMG_PATH_CACHE[image_id] = p1
        return p1
    p2 = os.path.join(IMAGES_DIR, image_id)
    if os.path.exists(p2):
        _IMG_PATH_CACHE[image_id] = p2
        return p2

    key1 = f"{image_id}.jpg".lower()
    key2 = image_id.lower()
    fn = _IMAGE_MAP.get(key1) or _IMAGE_MAP.get(key2)
    if fn is not None:
        p = os.path.join(IMAGES_DIR, fn)
        _IMG_PATH_CACHE[image_id] = p
        return p

    raise FileNotFoundError(f"Image not found for id={image_id} in {IMAGES_DIR}")


N_FEATS = 27


def _extract_one(image_id):
    path = _safe_img_path(image_id)
    with Image.open(path) as img:
        img = img.convert("RGB")

        w, h = img.size
        wf = float(w)
        hf = float(h)
        area = wf * hf
        aspect = wf / (hf + 1e-6)

        st = ImageStat.Stat(img)
        m = np.asarray(st.mean, dtype=np.float32) / 255.0
        s = np.asarray(st.stddev, dtype=np.float32) / 255.0

        gimg = img.convert("L")
        gst = ImageStat.Stat(gimg)
        gm = float(gst.mean[0]) / 255.0
        gs = float(gst.stddev[0]) / 255.0

        gray2d = np.asarray(gimg, dtype=np.float32) / 255.0
        gray = gray2d.reshape(-1)
        p10, p50, p90 = np.percentile(gray, [10, 50, 90])
        gmin = float(gray.min())
        gmax = float(gray.max())

        arr = np.asarray(img, dtype=np.float32) / 255.0
        r = arr[..., 0]
        g = arr[..., 1]
        b = arr[..., 2]
        exg = float((2.0 * g - r - b).mean())
        exr = float((2.0 * r - g - b).mean())

        denom = (r + g + b) + 1e-6
        r_ratio = float((r / denom).mean())
        g_ratio = float((g / denom).mean())
        b_ratio = float((b / denom).mean())
        y_proxy = float(((r + g) * 0.5 - b).mean())

        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        sat = (maxc - minc) / (maxc + 1e-6)
        sat_m = float(sat.mean())
        sat_s = float(sat.std())

        gx = np.abs(gray2d[:, 1:] - gray2d[:, :-1]).mean() if w > 1 else 0.0
        gy = np.abs(gray2d[1:, :] - gray2d[:-1, :]).mean() if h > 1 else 0.0
        edge_energy = float(gx + gy)

        green_dom = float(m[1] - 0.5 * (m[0] + m[2]))
        rg_contrast = float(m[0] - m[1])

        row = np.empty((N_FEATS,), dtype=np.float32)
        row[0] = m[0]
        row[1] = m[1]
        row[2] = m[2]
        row[3] = s[0]
        row[4] = s[1]
        row[5] = s[2]
        row[6] = gm
        row[7] = gs
        row[8] = float(p10)
        row[9] = float(p50)
        row[10] = float(p90)
        row[11] = exg + 0.25 * exr
        row[12] = gmin
        row[13] = gmax
        row[14] = r_ratio
        row[15] = g_ratio
        row[16] = b_ratio
        row[17] = y_proxy
        row[18] = wf / 4000.0
        row[19] = hf / 4000.0
        row[20] = aspect
        row[21] = area / (4000.0 * 4000.0)
        row[22] = sat_m
        row[23] = sat_s
        row[24] = edge_energy
        row[25] = green_dom
        row[26] = rg_contrast
        return row


def extract_features_for_ids(ids):
    from concurrent.futures import ThreadPoolExecutor

    n = len(ids)
    X = np.empty((n, N_FEATS), dtype=np.float32)

    max_workers = min(8, (os.cpu_count() or 2))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, row in enumerate(ex.map(_extract_one, ids, chunksize=16)):
            X[i] = row
    return X


def _binary_balanced_sample_weight(y_bin: np.ndarray) -> np.ndarray:
    y_bin = np.asarray(y_bin).astype(np.int32).ravel()
    n = y_bin.shape[0]
    pos = int(y_bin.sum())
    neg = n - pos
    pos = max(pos, 1)
    neg = max(neg, 1)
    w_pos = n / (2.0 * pos)
    w_neg = n / (2.0 * neg)
    return np.where(y_bin == 1, w_pos, w_neg).astype(np.float64)


def build_baseline_submission(mode=1, X_train=None, y_train=None, X_test=None):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    targets = ["healthy", "multiple_diseases", "rust", "scab"]

    if X_train is None or y_train is None or X_test is None:
        X_train = extract_features_for_ids(train_ids)
        y_train = train_df.loc[train_ids, targets].values.astype(np.int32)
        X_test = extract_features_for_ids(test_ids)

    C = 3.0 if mode == 1 else 1.2

    probs = np.zeros((len(test_ids), len(targets)), dtype=np.float64)
    for j in range(len(targets)):
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                (
                    "clf",
                    LogisticRegression(
                        C=C,
                        solver="lbfgs",
                        penalty="l2",
                        fit_intercept=True,
                        max_iter=1200,
                        n_jobs=None,
                        random_state=42,
                        class_weight=None,
                    ),
                ),
            ]
        )
        sw = _binary_balanced_sample_weight(y_train[:, j])
        model.fit(X_train, y_train[:, j], clf__sample_weight=sw)
        probs[:, j] = model.predict_proba(X_test)[:, 1].astype(np.float64)

    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    return probs


if len(submissions_all) >= 2:
    s0 = _load_and_align_submission(submissions_all[0], test_ids)
    s1 = _load_and_align_submission(submissions_all[1], test_ids)
    submission_avg = (0.2 * s0 + 0.8 * s1) / 1.0
    submission_avg = _consistent_probs_from_independent(submission_avg)
    make_submission_file(submission_avg, SAMPLE_SUB, out_path="submission.csv")
elif len(submissions_all) == 1:
    one = _load_and_align_submission(submissions_all[0], test_ids)
    submission_avg = one.astype(np.float64, copy=False)
    submission_avg = _consistent_probs_from_independent(submission_avg)
    make_submission_file(submission_avg, SAMPLE_SUB, out_path="submission.csv")
else:
    targets = ["healthy", "multiple_diseases", "rust", "scab"]

    X_train = extract_features_for_ids(train_ids)
    y_train = train_df.loc[train_ids, targets].values.astype(np.int32)
    X_test = extract_features_for_ids(test_ids)

    sub1 = build_baseline_submission(
        mode=1, X_train=X_train, y_train=y_train, X_test=X_test
    )
    sub2 = build_baseline_submission(
        mode=2, X_train=X_train, y_train=y_train, X_test=X_test
    )

    w1, w2 = 0.65, 0.35
    submission_avg = (sub1 * w1 + sub2 * w2) / (w1 + w2)

    submission_avg = _consistent_probs_from_independent(submission_avg)

    make_submission_file(submission_avg, SAMPLE_SUB, out_path="submission.csv")

out = pd.read_csv("submission.csv")
assert out.shape[0] == len(test_df), "Submission row count must match test.csv"
assert list(out.columns) == [
    "image_id",
    "healthy",
    "multiple_diseases",
    "rust",
    "scab",
], "Wrong submission columns/order"

out["image_id"] = out["image_id"].astype(str)
out = out.set_index("image_id").loc[test_ids].reset_index()
out.to_csv("submission.csv", index=False)

print(out.head())
print("submission.csv ready.")
