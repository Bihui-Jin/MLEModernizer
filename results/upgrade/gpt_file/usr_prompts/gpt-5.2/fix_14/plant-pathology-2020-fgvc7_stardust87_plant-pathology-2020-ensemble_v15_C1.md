# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The crash happens because your notebook expects an external folder of prior submission files (`/kaggle/input/submissions/submissions/`) that doesn’t exist in this environment, so `submissions_all` is empty and indexing `[0,1]` fails. I keep your ensembling core logic, but add a safe fallback that generates two simple baseline “submissions” from the provided `sample_submission.csv` so the pipeline always runs end-to-end. I also add small validations (weights length, index bounds, row alignment) and ensure the output `submission.csv` has the required columns and correct row order for `test.csv`. This is primarily a correctness fix (your current score is “Not yielded”); the baseline is only to produce a valid submission file.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 score comes from producing near-constant class probabilities (fallback submissions), which yields weak ROC AUC. To move toward the 0.97 target with minimal logic change, I keep your “ensemble CSVs” approach but generate the fallback submissions from a simple, legitimate signal: the class priors from `train.csv`, which is a standard baseline and usually scores substantially better than uniform guesses. I also ensure column order and row alignment strictly follow `test.csv` and keep your weighting/ensembling intact (still ensembling two CSVs). This should improve score while preserving your pipeline semantics and producing a valid `submission.csv`.'
- What this solution (achieved 0.45767) has done: 'Your current 0.5 score is consistent with producing almost-constant probabilities (class priors and lightly smoothed priors), which yields weak mean ROC AUC. To move toward the 0.97 target while keeping your “ensemble CSVs” core logic intact, I keep the ensembling pipeline but make the fallback submissions use a legitimate per-image signal: simple image brightness statistics from the provided JPGs via PIL (no model/architecture changes, still just producing CSVs to ensemble). This create two complementary, non-constant prediction files (different calibrations) and then ensemble them with your existing weights, which should substantially improve ROC AUC versus constants. I also add strict alignment to `test.csv` order inside the fallback creation so the generated CSVs can’t silently misorder rows.'
- What this solution (achieved 0.56502) has done: 'Your current score is far below the 0.97 target, and the main limiter is that the fallback “submissions” are based on crude brightness/contrast heuristics that don’t capture disease cues. With minimal core-logic change (still: generate two CSVs → ensemble with your existing weights), I make the fallback predictions come from a lightweight, legitimate image feature pipeline using downsampled RGB pixels + a multinomial logistic regression trained on `train.csv`. This keeps the same ensembling semantics but injects a much stronger per-image signal while staying well within the 600s budget. I also add strict file existence checks and ensure both train/test image ordering matches `test.csv`, so the submission rows align correctly.'
- What this solution (achieved 0.52933) has done: 'The timeout is dominated by the fallback path: per-image PIL loading/resizing in Python loops (done twice) plus two `CalibratedClassifierCV(cv=3)` fits, which each refit the full pipeline multiple times. To keep the exact same modeling logic (same features, same models, same calibration), the key speedups are (1) cache the computed RGB features to disk so repeated runs don’t redo image decoding, (2) make featurization faster with a single reusable worker and vectorized memory writes, (3) avoid repeated CSV reads and repeated index alignment work, and (4) reduce Python overhead in ensembling by streaming the weighted sum instead of building a list. These changes are provably equivalent in outputs (up to negligible floating-point differences) and only remove redundant work; the training/calibration procedure remains unchanged.'
- What this solution (achieved 0.51699) has done: 'The timeout is dominated by training eight calibrated SVMs (2 pipelines × 4 targets) with `CalibratedClassifierCV(cv=3)`, which refits the base estimator multiple times per target and is expensive at 48×48×3=6912 features. To preserve identical semantics, we keep the same features, SVMs, and sigmoid calibration, but enable safe parallelism inside `CalibratedClassifierCV` via `n_jobs=-1` so the CV fits run concurrently. We also cut avoidable overhead by (1) caching features with memory mapping to reduce RAM pressure and load time, (2) using `float32` consistently for `X_*` and avoiding repeated Python work in `featurize`, and (3) reading only needed columns when ensembling and building the final submission. These changes are performance-only and keep the model family, calibration method, and outputs equivalent up to negligible floating-point differences.'
- What this solution (achieved 0.56693) has done: 'Your current gap to the 0.97 target is large, and the biggest likely issue is that your two fallback models may be producing weak or misaligned predictions due to (a) train/test feature extraction not being consistently ordered/aligned and (b) fragile image-path resolution (e.g., `Test_*.jpg` vs `test_*.jpg`, nested folders). I keep your core logic exactly the same (48×48 RGB features + two LinearSVC models + sigmoid calibration cv=3 + 2-CSV ensemble with the same weights), but make minimal fixes to ensure every image is actually found and featurized, and that every generated CSV is strictly aligned to `test.csv` order. I also add a tiny, semantics-preserving improvement: use `StratifiedKFold` in the calibration CV to stabilize probability calibration on imbalanced labels (still cv=3, still sigmoid), which typically improves ROC AUC without changing the model family or training approach. Finally, I make the cache key depend on the resolved images directory to prevent accidentally reusing stale features from a different path.'
- What this solution (achieved 0.56391) has done: 'Your current score is far below the 0.97 target, so we should improve signal quality while keeping the same core pipeline (48×48 RGB features → two LinearSVC+sigmoid calibration models → two CSVs → weighted ensemble). The most likely silent failure hurting AUC is that many image paths aren’t being resolved because the dataset’s JPGs are typically named `Train_*.jpg` / `Test_*.jpg` while CSV ids are often `Train_*.jpg` or without extension depending on version; missing images become zero-vectors and destroy performance. I make image resolution robust by mapping both stems and full filenames (case-insensitive) and by trying `img_id`, `img_id + ".jpg"`, and stem variants; this preserves the exact modeling logic but ensures we featurize real pixels. I also add a small safety: if the cache exists but was built with many missing images, we rebuild once (same semantics, just prevents reusing a bad cache), which should move the score meaningfully toward the target.'
- What this solution (achieved 0.53803) has done: 'The timeout is dominated by repeated model training inside `CalibratedClassifierCV`: your current fallback trains 2 multiclass + 2×4 binary calibrated models, each with 3-fold CV, causing many redundant refits and repeated scaling work. I keep the exact same features, models, calibration method, CV settings, and blending logic, but restructure training to reuse the already-fitted multiclass calibrated models to derive the per-target probabilities (equivalent because each target is mutually exclusive class membership), eliminating the 8 extra binary calibrated fits. I also speed up image path resolution by avoiding a full `os.walk` map build and instead using direct existence checks (with a tiny cache) and keep the feature caching exactly as before. Finally, I prevent oversubscription from nested parallelism by setting common thread env vars, which improves wall-time without changing results.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")



## === cell 1
BASE_PATH_CANDIDATES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
]
BASE_PATH = None
for p in BASE_PATH_CANDIDATES:
    if os.path.exists(os.path.join(p, "sample_submission.csv")) and os.path.exists(
        os.path.join(p, "test.csv")
    ):
        BASE_PATH = p
        break

if BASE_PATH is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv and test.csv in expected Kaggle paths."
    )

SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")
TEST_CSV_PATH = os.path.join(BASE_PATH, "test.csv")
TRAIN_CSV_PATH = os.path.join(BASE_PATH, "train.csv")

IMAGES_DIR_CANDIDATES = [
    os.path.join(BASE_PATH, "images"),
    os.path.join(BASE_PATH, "plant-pathology-2020-fgvc7", "images"),
]
IMAGES_DIR = None
for d in IMAGES_DIR_CANDIDATES:
    if os.path.isdir(d):
        IMAGES_DIR = d
        break
if IMAGES_DIR is None:
    raise FileNotFoundError(
        "Could not locate images directory under expected Kaggle paths."
    )

print("Using BASE_PATH:", BASE_PATH)
print("SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)
print("TEST_CSV_PATH:", TEST_CSV_PATH)
print("TRAIN_CSV_PATH:", TRAIN_CSV_PATH)
print("IMAGES_DIR:", IMAGES_DIR)



## === cell 2
SUBMISSIONS_PATH = "/kaggle/input/submissions/submissions/"

submissions_all = []
if os.path.exists(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print("Found submission files:", submissions_all)




## === cell 3
def _make_fallback_submissions(
    sample_sub_path: str, test_csv_path: str, train_csv_path: str, images_dir: str
) -> list:
    """
    Core logic preserved:
    - Same downsampled RGB feature extraction (48x48 RGB flattened)
    - Same model family (LinearSVC + CalibratedClassifierCV(sigmoid, cv=3))
    - Same 2-model ensemble idea and 4 independent target probabilities
    - Same blend with multiclass probabilities using alpha=0.35

    Minimal score-focused improvement:
    - Add deterministic center-crop before resize to reduce background/edge noise and
      better align leaves, which typically improves ROC AUC on this dataset while keeping
      the same feature type (RGB pixels) and the same modeling/training/calibration logic.
    - Bump cache key version so we don't reuse pre-crop cached features.
    """
    from PIL import Image
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedKFold
    import hashlib

    sample = pd.read_csv(sample_sub_path)
    test = pd.read_csv(test_csv_path)
    train = pd.read_csv(train_csv_path)

    required_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    missing = [c for c in required_cols if c not in sample.columns]
    if missing:
        raise ValueError(f"sample_submission.csv missing columns: {missing}")

    target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
    for c in target_cols:
        if c not in train.columns:
            raise ValueError(f"train.csv missing required target column: {c}")

    test_ids = test["image_id"].astype(str).tolist()
    train_ids = train["image_id"].astype(str).tolist()

    sample = sample.set_index("image_id").reindex(test_ids).reset_index()
    if sample["image_id"].isna().any():
        raise ValueError(
            "sample_submission is missing some image_id entries present in test.csv."
        )

    Y = train[target_cols].to_numpy(dtype=np.int64, copy=True)
    y_mc = np.argmax(Y, axis=1).astype(np.int64)

    _path_cache = {}

    def _resolve_path(img_id: str) -> str:
        p = _path_cache.get(img_id)
        if p is not None:
            return p

        direct = os.path.join(images_dir, img_id)
        if os.path.exists(direct):
            _path_cache[img_id] = direct
            return direct

        direct2 = os.path.join(images_dir, f"{img_id}.jpg")
        if os.path.exists(direct2):
            _path_cache[img_id] = direct2
            return direct2

        if img_id.lower().endswith(".jpg"):
            direct3 = os.path.join(images_dir, img_id)
            if os.path.exists(direct3):
                _path_cache[img_id] = direct3
                return direct3

        _path_cache[img_id] = ""
        return ""

    def _cache_paths(size: int, crop_frac: float):
        tag_base = f"pp2020_rgb{size}_crop{int(crop_frac*100)}_v4"
        h = hashlib.md5(os.path.abspath(images_dir).encode("utf-8")).hexdigest()[:10]
        xtr = os.path.abspath(f"{tag_base}_{h}_X_train.npy")
        xte = os.path.abspath(f"{tag_base}_{h}_X_test.npy")
        return xtr, xte

    def _center_crop(im: Image.Image, frac: float) -> Image.Image:
        if frac >= 0.999:
            return im
        w, h = im.size
        new_w = max(1, int(round(w * frac)))
        new_h = max(1, int(round(h * frac)))
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        return im.crop((left, top, left + new_w, top + new_h))

    def featurize(image_ids, size=48, crop_frac=0.90):
        X = np.zeros((len(image_ids), size * size * 3), dtype=np.float32)
        missing_count = 0

        img_open = Image.open
        bilinear = Image.BILINEAR
        asarray = np.asarray
        inv255 = np.float32(1.0 / 255.0)

        for i, img_id in enumerate(image_ids):
            img_path = _resolve_path(img_id)
            if not img_path:
                missing_count += 1
                continue
            with img_open(img_path) as im:
                im = im.convert("RGB")
                im = _center_crop(im, crop_frac=crop_frac).resize(
                    (size, size), resample=bilinear
                )
                arr = asarray(im, dtype=np.float32)
            X[i, :] = arr.reshape(-1) * inv255

        if missing_count:
            print(
                f"Warning: {missing_count}/{len(image_ids)} images were missing under {images_dir}. They were featurized as zeros."
            )
        return X

    def _zero_row_fraction(X):
        Xv = np.asarray(X)
        return float(np.mean(np.all(Xv == 0.0, axis=1)))

    crop_frac = 0.90
    cache_train, cache_test = _cache_paths(size=48, crop_frac=crop_frac)

    if os.path.exists(cache_train) and os.path.exists(cache_test):
        X_train = np.load(cache_train, mmap_mode="r")
        X_test = np.load(cache_test, mmap_mode="r")

        ztr = _zero_row_fraction(X_train)
        zte = _zero_row_fraction(X_test)
        if (ztr > 0.05) or (zte > 0.05):
            print(
                f"Cached features look suspiciously sparse (all-zero rows: train={ztr:.3f}, test={zte:.3f}). Rebuilding features once."
            )
            X_train = featurize(train_ids, size=48, crop_frac=crop_frac)
            X_test = featurize(test_ids, size=48, crop_frac=crop_frac)
            np.save(cache_train, X_train)
            np.save(cache_test, X_test)
        else:
            print(
                f"Loaded cached features (all-zero rows: train={ztr:.3f}, test={zte:.3f})."
            )
    else:
        X_train = featurize(train_ids, size=48, crop_frac=crop_frac)
        X_test = featurize(test_ids, size=48, crop_frac=crop_frac)
        np.save(cache_train, X_train)
        np.save(cache_test, X_test)

    base_svm1 = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("svm", LinearSVC(C=2.0, class_weight=None, random_state=0, max_iter=5000)),
        ]
    )
    base_svm2 = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("svm", LinearSVC(C=0.7, class_weight=None, random_state=1, max_iter=5000)),
        ]
    )

    P1 = np.zeros((X_test.shape[0], 4), dtype=np.float64)
    P2 = np.zeros((X_test.shape[0], 4), dtype=np.float64)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    mc_clf1 = CalibratedClassifierCV(base_svm1, method="sigmoid", cv=cv, n_jobs=-1)
    mc_clf2 = CalibratedClassifierCV(base_svm2, method="sigmoid", cv=cv, n_jobs=-1)
    mc_clf1.fit(X_train, y_mc)
    mc_clf2.fit(X_train, y_mc)
    mc_proba1 = mc_clf1.predict_proba(X_test).astype(np.float64, copy=False)
    mc_proba2 = mc_clf2.predict_proba(X_test).astype(np.float64, copy=False)

    mc_classes1 = list(mc_clf1.classes_)
    mc_classes2 = list(mc_clf2.classes_)
    mc_map1 = {c: i for i, c in enumerate(mc_classes1)}
    mc_map2 = {c: i for i, c in enumerate(mc_classes2)}

    alpha = 0.35

    for j, col in enumerate(["healthy", "multiple_diseases", "rust", "scab"]):
        yj = Y[:, j]
        if yj.min() == yj.max():
            prior = float(yj.mean())
            p_mc1 = mc_proba1[:, mc_map1.get(j, 0)]
            p_mc2 = mc_proba2[:, mc_map2.get(j, 0)]
            P1[:, j] = (1.0 - alpha) * prior + alpha * p_mc1
            P2[:, j] = (1.0 - alpha) * prior + alpha * p_mc2
        else:
            P1[:, j] = mc_proba1[:, mc_map1.get(j, 0)]
            P2[:, j] = mc_proba2[:, mc_map2.get(j, 0)]

    np.clip(P1, 1e-6, 1 - 1e-6, out=P1)
    np.clip(P2, 1e-6, 1 - 1e-6, out=P2)

    sub1 = sample.copy()
    sub2 = sample.copy()
    sub1.loc[:, ["healthy", "multiple_diseases", "rust", "scab"]] = P1
    sub2.loc[:, ["healthy", "multiple_diseases", "rust", "scab"]] = P2

    sub1 = sub1.set_index("image_id").reindex(test_ids).reset_index()
    sub2 = sub2.set_index("image_id").reindex(test_ids).reset_index()

    path1 = "fallback_submission_1.csv"
    path2 = "fallback_submission_2.csv"
    sub1.to_csv(path1, index=False)
    sub2.to_csv(path2, index=False)
    return [os.path.abspath(path1), os.path.abspath(path2)]


if len(submissions_all) == 0:
    submissions_all = _make_fallback_submissions(
        SAMPLE_SUB_PATH, TEST_CSV_PATH, TRAIN_CSV_PATH, IMAGES_DIR
    )
    print("Created fallback submissions:", submissions_all)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1936751432.py in <cell line: 0>()
    217 
    218 if len(submissions_all) == 0:
--> 219     submissions_all = _make_fallback_submissions(
    220         SAMPLE_SUB_PATH, TEST_CSV_PATH, TRAIN_CSV_PATH, IMAGES_DIR
    221     )

/tmp/ipykernel_11/1936751432.py in _make_fallback_submissions(sample_sub_path, test_csv_path, train_csv_path, images_dir)
    149             )
    150     else:
--> 151         X_train = featurize(train_ids, size=48, crop_frac=crop_frac)
    152         X_test = featurize(test_ids, size=48, crop_frac=crop_frac)
    153         np.save(cache_train, X_train)

/tmp/ipykernel_11/1936751432.py in featurize(image_ids, size, crop_frac)
    111                 im = im.convert("RGB")
    112                 # Change is directly score-relevant: remove border/background then resize.
--> 113                 im = _center_crop(im, crop_frac=crop_frac).resize(
    114                     (size, size), resample=bilinear
    115                 )

TypeError: _make_fallback_submissions.<locals>._center_crop() got an unexpected keyword argument 'crop_frac'

## === cell 4
def ensemble(submissions_all, sub_idx, weights=[]):
    if len(sub_idx) == 0:
        raise ValueError("sub_idx must contain at least one index.")
    if len(weights) == 0:
        weights = [1.0 / len(sub_idx)] * len(sub_idx)
    if len(weights) != len(sub_idx):
        raise ValueError(
            f"weights length ({len(weights)}) must match sub_idx length ({len(sub_idx)})."
        )

    cols = ["healthy", "multiple_diseases", "rust", "scab"]
    usecols = ["image_id"] + cols
    submission_sum = None
    base_len = None

    test_ids = (
        pd.read_csv(TEST_CSV_PATH, usecols=["image_id"])["image_id"]
        .astype(str)
        .tolist()
    )

    for i in range(len(sub_idx)):
        idx = sub_idx[i]
        if idx < 0 or idx >= len(submissions_all):
            raise IndexError(
                f"sub_idx[{i}]={idx} out of range for submissions_all (len={len(submissions_all)})."
            )

        w = float(weights[i])
        print(f"I'm taking submission {submissions_all[idx]} with weight {w}")
        submission = pd.read_csv(submissions_all[idx], usecols=usecols)

        missing_cols = [c for c in cols if c not in submission.columns]
        if missing_cols:
            raise ValueError(
                f"Submission {submissions_all[idx]} missing columns: {missing_cols}"
            )

        submission["image_id"] = submission["image_id"].astype(str)
        submission = submission.set_index("image_id").reindex(test_ids).reset_index()
        if submission["image_id"].isna().any():
            raise ValueError(
                f"Submission {submissions_all[idx]} is missing some image_ids from test.csv"
            )

        arr = submission.loc[:, cols].to_numpy(dtype=np.float64, copy=False)

        if base_len is None:
            base_len = arr.shape[0]
            submission_sum = np.zeros_like(arr, dtype=np.float64)
        elif arr.shape[0] != base_len:
            raise ValueError(
                f"Row count mismatch across submissions: expected {base_len}, got {arr.shape[0]}"
            )

        submission_sum += arr * w

    return submission_sum




## === cell 5
def make_submission_file(submission_avg, submissions_all):
    submission_df = pd.read_csv(
        SAMPLE_SUB_PATH,
        usecols=["image_id", "healthy", "multiple_diseases", "rust", "scab"],
    )
    test_df = pd.read_csv(TEST_CSV_PATH, usecols=["image_id"])

    test_ids = test_df["image_id"].astype(str).tolist()
    submission_df["image_id"] = submission_df["image_id"].astype(str)
    submission_df = submission_df.set_index("image_id").reindex(test_ids).reset_index()
    if submission_df["image_id"].isna().any():
        raise ValueError(
            "sample_submission is missing some image_id entries present in test.csv."
        )

    cols = ["healthy", "multiple_diseases", "rust", "scab"]
    if submission_avg.shape != (len(submission_df), len(cols)):
        raise ValueError(
            f"submission_avg has shape {submission_avg.shape}, expected {(len(submission_df), len(cols))}"
        )

    submission_df.loc[:, cols] = submission_avg
    submission_df[cols] = submission_df[cols].clip(0.0, 1.0)

    submission_df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission_df.shape)
    print("Columns:", list(submission_df.columns))
    print(submission_df.head())


submission_avg = ensemble(submissions_all, [0, 1], [0.24, 0.76])
make_submission_file(submission_avg, submissions_all)

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/3387453063.py in <cell line: 0>()
     29 
     30 
---> 31 submission_avg = ensemble(submissions_all, [0, 1], [0.24, 0.76])
     32 make_submission_file(submission_avg, submissions_all)

/tmp/ipykernel_11/3069738150.py in ensemble(submissions_all, sub_idx, weights)
     23         idx = sub_idx[i]
     24         if idx < 0 or idx >= len(submissions_all):
---> 25             raise IndexError(
     26                 f"sub_idx[{i}]={idx} out of range for submissions_all (len={len(submissions_all)})."
     27             )

IndexError: sub_idx[0]=0 out of range for submissions_all (len=0).
