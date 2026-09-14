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

0.9700013841179632

# 6. Current score

0.70881

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The error happens because `/kaggle/input/submissions/submissions/` doesn’t exist in this environment, so `submissions_all` is empty and indexing `[0,1]` crashes. I fix this by (1) switching to the provided competition dataset path and (2) adding a safe fallback that creates a valid `submission.csv` directly from `sample_submission.csv` when no external submissions are found. This preserves your “ensemble CSVs by weighted averaging” core logic when submission files exist, and otherwise guarantees an end-to-end run producing a correctly formatted submission file. The fallback uses uniform probabilities (0.25) per class, which is score-neutral relative to not yielding any submission and meets the competition format.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 score comes from the uniform-probability fallback (0.25 each class), which is far below the 0.97 target. Since we must preserve your “ensemble existing submission CSVs if present” core logic and we have no model-training code here, the smallest legitimate way to move the score upward is to (1) reliably find any existing high-quality submission CSVs in the provided dataset folders, (2) validate/align them to `sample_submission.csv` by `image_id` to avoid silent row-order mismatch, and (3) if none exist, produce a slightly better, still-simple fallback based on the train label priors (class prevalence) instead of uniform 0.25. These changes keep the same overall approach (CSV ensembling / fallback) while making the output more competitive and correctly aligned with the evaluation format.'
- What this solution (achieved 0.59387) has done: 'Your current 0.5 score is consistent with a non-informative fallback (uniform or near-uniform predictions), so we need a slightly more informative fallback that still preserves your core “use existing submission CSVs if available, else fallback” logic. The smallest legitimate improvement is to generate image-conditioned probabilities by using simple, deterministic image features (mean RGB) from the provided `images/` folder and fitting a lightweight multi-output logistic regression on the training set. This keeps the overall approach (no deep model, no new training loops/architectures) while producing predictions that should move ROC AUC substantially upward toward your 0.97 target. If any valid submission-like CSVs are found, your original weighted-averaging ensemble path remains unchanged.'
- What this solution (achieved 0.66714) has done: 'Your current fallback uses only mean RGB features, which is likely too weak for a 4-label leaf disease task, so we keep the same “lightweight image-feature + MultiOutput logistic regression” core approach but make the features slightly richer while staying deterministic and fast. Specifically, we (1) add per-image color variability (RGB std) and simple channel ratios (r/(g+b), etc.) to improve separability, and (2) standardize features before logistic regression to make LBFGS behave better without changing the modeling family. We also ensure the image directory is resolved robustly to the existing `.../images/` folder, preventing silent missing-image fallbacks that hurt score. These are minimal changes aimed to raise AUC toward the 0.97 target without changing the overall pipeline or producing an invalid submission.'
- What this solution (achieved 0.67613) has done: 'The timeout is most likely caused by recursively walking large directory trees and repeatedly reading CSV headers to detect “submission-like” files; that can be extremely slow on Kaggle inputs. I (1) restrict the scan to the smallest necessary roots and prune `os.walk` into known-irrelevant directories, (2) replace full `pd.read_csv(..., nrows=5)` header reads with an ultra-fast “read only the header line” check and only parse with pandas if the header matches, and (3) speed up the fallback image feature extraction with parallel, deterministic thread-based loading (I/O bound) while keeping the exact same features and model/training logic. All paths, ensemble logic, feature definitions, and model settings remain unchanged.'
- What this solution (achieved 0.71045) has done: 'The timeout is most likely coming from the fallback path that reads/decodes ~1800 JPEGs and computes expensive per-image quantiles and other stats; it does this in Python for each image and can easily exceed 600s. I keep the exact same features/model/fit logic, but make the image feature extraction much faster by (1) downsampling images deterministically before computing the same statistics (equivalent semantics, just fewer pixels), (2) replacing `np.quantile` with a mathematically equivalent but faster “sort + index pick” implementation for the three quantiles, and (3) avoiding repeated array reshapes/copies and using more cache-friendly computations. I also make the thread pool mapping lighter (pre-bind locals, avoid per-call closures) while preserving determinism and outputs apart from negligible float differences. No training loop, model, targets, paths, or ensemble semantics are changed.'
- What this solution (achieved 0.70881) has done: 'Your current 0.710 score suggests the fallback image-feature logistic regression is learning something but is underfitting because the features miss key disease cues; we can move toward 0.97 by adding a tiny amount of extra image signal while keeping the exact same training approach (feature extraction → StandardScaler → LogisticRegression inside MultiOutputClassifier). I keep all existing features unchanged and append a small set of deterministic, cheap-to-compute extras (normalized green-red and green-blue indices plus simple gray-level quantiles) that often correlate with leaf discoloration and lesions, which should improve separability and ROC AUC without changing the model family or loops. I also make sure the default feature vector length stays consistent when images are missing, so we don’t silently degrade predictions. No changes are made to ensembling behavior, submission alignment, or file paths; the script still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

BASE_PATH = "/kaggle/input/plant-pathology-2020-fgvc7"
ALT_BASE_PATH = "/kaggle/input"

SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_PATH = os.path.join(BASE_PATH, "train.csv")
TEST_PATH = os.path.join(BASE_PATH, "test.csv")

IMAGES_DIR_CANDIDATES = [
    os.path.join(BASE_PATH, "images"),
    os.path.join(ALT_BASE_PATH, "plant-pathology-2020-fgvc7", "images"),
    os.path.join(ALT_BASE_PATH, "images"),
]
IMAGES_DIR = None
for p in IMAGES_DIR_CANDIDATES:
    if os.path.isdir(p):
        IMAGES_DIR = p
        break

SUBMISSIONS_PATH = "/kaggle/input/submissions/submissions/"
SEARCH_ROOTS = [SUBMISSIONS_PATH, BASE_PATH, ALT_BASE_PATH]

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

print("IMAGES_DIR:", IMAGES_DIR)
print("SAMPLE_SUB_PATH exists:", os.path.exists(SAMPLE_SUB_PATH))
print("TRAIN_PATH exists:", os.path.exists(TRAIN_PATH))
print("TEST_PATH exists:", os.path.exists(TEST_PATH))




## === cell 1
def _fast_csv_header_cols(path: str):
    try:
        with open(path, "rb") as f:
            line = f.readline(1024 * 1024)
        if not line:
            return None
        s = line.decode("utf-8-sig", errors="ignore").strip()
        if not s:
            return None
        return [c.strip().strip('"') for c in s.split(",")]
    except Exception:
        return None


def _looks_like_submission_csv_fast(path: str) -> bool:
    base = os.path.basename(path).lower()
    if base in ("train.csv", "test.csv", "sample_submission.csv"):
        return False
    cols = _fast_csv_header_cols(path)
    if not cols:
        return False
    cols_set = set(cols)
    if "image_id" not in cols_set:
        return False
    if not any(c in cols_set for c in TARGET_COLS):
        return False
    return True


def _iter_csv_paths_pruned(root: str):
    prune_dirnames = {
        "images",
        "__MACOSX",
        ".git",
        ".ipynb_checkpoints",
        "plant-pathology-2020-fgvc7",
        "working",
    }
    if not root or not os.path.exists(root):
        return
    for dirname, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [
            d for d in dirnames if d not in prune_dirnames and not d.startswith(".")
        ]
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                yield os.path.join(dirname, filename)


submissions_all = []
seen = set()

roots_to_scan = []
if SUBMISSIONS_PATH and os.path.exists(SUBMISSIONS_PATH):
    roots_to_scan.append(SUBMISSIONS_PATH)


def _scan_roots(roots):
    found = []
    for root in roots:
        for path in _iter_csv_paths_pruned(root):
            if path in seen:
                continue
            seen.add(path)
            if _looks_like_submission_csv_fast(path):
                try:
                    head = pd.read_csv(path, nrows=1)
                    cols = set(head.columns)
                    if "image_id" in cols and all(c in cols for c in TARGET_COLS):
                        found.append(path)
                except Exception:
                    pass
    return found


submissions_all = _scan_roots(roots_to_scan)

if not submissions_all:
    fallback_roots = []
    if BASE_PATH and os.path.exists(BASE_PATH):
        fallback_roots.append(BASE_PATH)
    if ALT_BASE_PATH and os.path.exists(ALT_BASE_PATH):
        fallback_roots.append(ALT_BASE_PATH)
    submissions_all = _scan_roots(fallback_roots)

submissions_all.sort()
print("Found submission-like CSV files:", submissions_all)




## === cell 2
def ensemble(
    submissions_all, sub_idx, weights=None, sample_submission_path=SAMPLE_SUB_PATH
):
    if weights is None:
        weights = [1.0] * len(sub_idx)
    if len(sub_idx) != len(weights):
        raise ValueError(
            f"sub_idx (len={len(sub_idx)}) and weights (len={len(weights)}) must match."
        )

    sample = pd.read_csv(sample_submission_path)
    if "image_id" not in sample.columns:
        raise KeyError("sample_submission must contain image_id")

    submission_with_weight = []
    for i in range(len(sub_idx)):
        idx = sub_idx[i]
        if idx < 0 or idx >= len(submissions_all):
            raise IndexError(
                f"Requested submission index {idx}, but only {len(submissions_all)} files were found."
            )
        path = submissions_all[idx]
        print(f"I'm taking submission {path} with weight {weights[i]}")
        submission = pd.read_csv(path)

        missing = [
            c for c in (["image_id"] + TARGET_COLS) if c not in submission.columns
        ]
        if missing:
            raise KeyError(f"Submission {path} is missing columns: {missing}")

        submission["image_id"] = submission["image_id"].astype(str)
        sub_aligned = sample[["image_id"]].merge(
            submission[["image_id"] + TARGET_COLS],
            on="image_id",
            how="left",
            sort=False,
        )

        for c in TARGET_COLS:
            col = pd.to_numeric(sub_aligned[c], errors="coerce").astype("float64")
            if col.isna().any():
                fillv = float(col.mean()) if col.notna().any() else 0.25
                sub_aligned[c] = col.fillna(fillv)

        preds = sub_aligned[TARGET_COLS].astype("float64").values
        submission_with_weight.append(preds * float(weights[i]))

    submission_avg = sum(submission_with_weight) / float(sum(weights))
    return submission_avg




## === cell 3
def make_submission_file(submission_avg, sample_submission_path):
    submission_df = pd.read_csv(sample_submission_path)

    for c in TARGET_COLS:
        if c not in submission_df.columns:
            raise KeyError(f"sample_submission is missing expected column: {c}")

    submission_df.loc[:, TARGET_COLS] = submission_avg
    submission_df.loc[:, TARGET_COLS] = submission_df.loc[:, TARGET_COLS].clip(0.0, 1.0)

    submission_df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission_df.shape)
    print(submission_df.head())




## === cell 4
from functools import lru_cache

try:
    from PIL import Image
except Exception:
    Image = None


_DOWNSAMPLE_MAX_SIDE = (
    256  # fixed cap, deterministic; avoids huge 2000x2000+ leaf images cost
)


def _quantile_linear_from_sorted(sorted_1d: np.ndarray, p: float) -> float:
    n = sorted_1d.shape[0]
    if n == 0:
        return float("nan")
    if n == 1:
        return float(sorted_1d[0])
    pos = p * (n - 1)
    lo = int(pos)
    hi = lo + 1
    if hi >= n:
        return float(sorted_1d[lo])
    w = pos - lo
    return float(sorted_1d[lo] * (1.0 - w) + sorted_1d[hi] * w)


@lru_cache(maxsize=4096)
def _read_image_basic_stats(path: str):
    if Image is None:
        return None
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")

            w, h = im.size
            mx = max(w, h)
            if mx > _DOWNSAMPLE_MAX_SIDE:
                scale = _DOWNSAMPLE_MAX_SIDE / float(mx)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                im = im.resize((new_w, new_h), resample=Image.BILINEAR)

            arr = np.asarray(im, dtype=np.float32) * (1.0 / 255.0)  # H,W,3

            mean_rgb = arr.mean(axis=(0, 1))
            std_rgb = arr.std(axis=(0, 1))
            min_rgb = arr.min(axis=(0, 1))
            max_rgb = arr.max(axis=(0, 1))

            flat = arr.reshape(-1, 3)
            c0 = np.sort(flat[:, 0], axis=0)
            c1 = np.sort(flat[:, 1], axis=0)
            c2 = np.sort(flat[:, 2], axis=0)
            q10 = np.array(
                [
                    _quantile_linear_from_sorted(c0, 0.10),
                    _quantile_linear_from_sorted(c1, 0.10),
                    _quantile_linear_from_sorted(c2, 0.10),
                ],
                dtype=np.float32,
            )
            q50 = np.array(
                [
                    _quantile_linear_from_sorted(c0, 0.50),
                    _quantile_linear_from_sorted(c1, 0.50),
                    _quantile_linear_from_sorted(c2, 0.50),
                ],
                dtype=np.float32,
            )
            q90 = np.array(
                [
                    _quantile_linear_from_sorted(c0, 0.90),
                    _quantile_linear_from_sorted(c1, 0.90),
                    _quantile_linear_from_sorted(c2, 0.90),
                ],
                dtype=np.float32,
            )

            r, g, b = float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])
            eps = 1e-6
            r_ratio = r / (g + b + eps)
            g_ratio = g / (r + b + eps)
            b_ratio = b / (r + g + eps)

            gray2d = (
                0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]
            )
            gray_mean = float(gray2d.mean())
            gray_std = float(gray2d.std())

            gray_flat = np.sort(gray2d.reshape(-1), axis=0)
            gray_q10 = _quantile_linear_from_sorted(gray_flat, 0.10)
            gray_q50 = _quantile_linear_from_sorted(gray_flat, 0.50)
            gray_q90 = _quantile_linear_from_sorted(gray_flat, 0.90)

            dx = (
                float(np.abs(gray2d[:, 1:] - gray2d[:, :-1]).mean())
                if gray2d.shape[1] > 1
                else 0.0
            )
            dy = (
                float(np.abs(gray2d[1:, :] - gray2d[:-1, :]).mean())
                if gray2d.shape[0] > 1
                else 0.0
            )
            edge_mean = float(0.5 * (dx + dy))

            cmax = arr.max(axis=2)
            cmin = arr.min(axis=2)
            delta = cmax - cmin
            sat = delta / (cmax + eps)
            val = cmax
            sat_mean = float(sat.mean())
            sat_std = float(sat.std())
            val_mean = float(val.mean())
            val_std = float(val.std())

            ngrdi = (g - r) / (g + r + eps)  # normalized green-red difference
            ngbdi = (g - b) / (g + b + eps)  # normalized green-blue difference
            rg_norm = (r - g) / (
                r + g + eps
            )  # symmetric counterpart, can help scab/rust contrast

            feats = np.array(
                [
                    mean_rgb[0],
                    mean_rgb[1],
                    mean_rgb[2],
                    std_rgb[0],
                    std_rgb[1],
                    std_rgb[2],
                    min_rgb[0],
                    min_rgb[1],
                    min_rgb[2],
                    max_rgb[0],
                    max_rgb[1],
                    max_rgb[2],
                    r_ratio,
                    g_ratio,
                    b_ratio,
                    gray_mean,
                    gray_std,
                    q10[0],
                    q10[1],
                    q10[2],
                    q50[0],
                    q50[1],
                    q50[2],
                    q90[0],
                    q90[1],
                    q90[2],
                    sat_mean,
                    sat_std,
                    val_mean,
                    val_std,
                    edge_mean,
                    ngrdi,
                    ngbdi,
                    rg_norm,
                    gray_q10,
                    gray_q50,
                    gray_q90,
                ],
                dtype=np.float32,
            )
            return feats
    except Exception:
        return None


def build_features(image_ids, images_dir):
    n_feats = 37
    X = np.zeros((len(image_ids), n_feats), dtype=np.float32)

    default_feat = np.array(
        [
            0.5,
            0.5,
            0.5,  # mean rgb
            0.1,
            0.1,
            0.1,  # std rgb
            0.0,
            0.0,
            0.0,  # min rgb
            1.0,
            1.0,
            1.0,  # max rgb
            1.0,
            1.0,
            1.0,  # ratios
            0.5,
            0.1,  # gray mean/std
            0.2,
            0.2,
            0.2,  # q10 rgb
            0.5,
            0.5,
            0.5,  # q50 rgb
            0.8,
            0.8,
            0.8,  # q90 rgb
            0.3,
            0.1,  # sat mean/std
            0.6,
            0.1,  # val mean/std
            0.05,  # edge mean
            0.0,
            0.0,
            0.0,  # ngrdi, ngbdi, rg_norm
            0.2,
            0.5,
            0.8,  # gray q10/q50/q90
        ],
        dtype=np.float32,
    )

    join = os.path.join
    read_stats = _read_image_basic_stats
    imgdir = images_dir
    def_feat = default_feat

    def _one(i_img):
        i, img_id = i_img
        fname = str(img_id)
        if not fname.lower().endswith(".jpg"):
            fname = fname + ".jpg"
        path = join(imgdir, fname)
        feat = read_stats(path)
        if feat is None:
            return i, def_feat, 1
        return i, feat, 0

    from concurrent.futures import ThreadPoolExecutor

    cpu = os.cpu_count() or 4
    max_workers = min(16, cpu)
    chunksize = 128 if len(image_ids) >= 1024 else 32

    missing = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, feat, miss in ex.map(_one, enumerate(image_ids), chunksize=chunksize):
            X[i] = feat
            missing += miss
    return X, missing


def fallback_train_image_model_and_predict(
    train_path, test_path, sample_submission_path, images_dir
):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_df = pd.read_csv(sample_submission_path)

    train_df["image_id"] = train_df["image_id"].astype(str)
    test_df["image_id"] = test_df["image_id"].astype(str)
    sample_df["image_id"] = sample_df["image_id"].astype(str)

    test_ids = sample_df["image_id"].tolist()
    train_ids = train_df["image_id"].tolist()

    X_train, miss_tr = build_features(train_ids, images_dir)
    X_test, miss_te = build_features(test_ids, images_dir)

    y_train = train_df[TARGET_COLS].astype(int).values

    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    base_lr = LogisticRegression(
        solver="lbfgs",
        max_iter=800,
        C=2.0,
        class_weight=None,
        n_jobs=None,
        random_state=0,
    )
    clf = MultiOutputClassifier(
        Pipeline([("scaler", StandardScaler()), ("lr", base_lr)])
    )
    clf.fit(X_train, y_train)

    probs = np.zeros((len(test_ids), len(TARGET_COLS)), dtype=np.float64)
    for j, est in enumerate(clf.estimators_):
        p = est.predict_proba(X_test)
        if p.shape[1] == 2:
            probs[:, j] = p[:, 1]
        else:
            prev = float(train_df[TARGET_COLS[j]].mean())
            probs[:, j] = prev

    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)

    print(
        f"Fallback image-model: missing train images={miss_tr}, missing test images={miss_te}"
    )
    return probs




## === cell 5
if len(submissions_all) >= 2:
    submission_avg = ensemble(
        submissions_all, [0, 1], [0.23, 0.77], sample_submission_path=SAMPLE_SUB_PATH
    )
    make_submission_file(submission_avg, SAMPLE_SUB_PATH)
elif len(submissions_all) == 1:
    submission_avg = ensemble(
        submissions_all, [0], [1.0], sample_submission_path=SAMPLE_SUB_PATH
    )
    make_submission_file(submission_avg, SAMPLE_SUB_PATH)
else:
    if (
        os.path.exists(TRAIN_PATH)
        and os.path.exists(TEST_PATH)
        and IMAGES_DIR is not None
        and os.path.isdir(IMAGES_DIR)
    ):
        submission_avg = fallback_train_image_model_and_predict(
            TRAIN_PATH, TEST_PATH, SAMPLE_SUB_PATH, IMAGES_DIR
        )
        make_submission_file(submission_avg, SAMPLE_SUB_PATH)
    else:
        sample_df = pd.read_csv(SAMPLE_SUB_PATH)
        if os.path.exists(TRAIN_PATH):
            train_df = pd.read_csv(TRAIN_PATH)
            priors = train_df[TARGET_COLS].mean().astype("float64")
            priors = priors.clip(1e-6, 1.0 - 1e-6)
            for c in TARGET_COLS:
                sample_df.loc[:, c] = float(priors[c])
            print(
                "No external submissions and image-model unavailable; wrote prior-based submission using train label prevalence:",
                priors.to_dict(),
            )
        else:
            for c in TARGET_COLS:
                sample_df.loc[:, c] = 0.25
            print(
                "No external submissions found and train.csv missing; wrote uniform baseline (0.25) submission."
            )

        sample_df.to_csv("submission.csv", index=False)
        print("Wrote submission.csv with shape:", sample_df.shape)
        print(sample_df.head())
