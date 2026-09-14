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

0.969570399454271

# 6. Current score

0.78433

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'You’re getting `IndexError` because `/kaggle/input/submissions/` doesn’t exist in this environment, so `submissions_all` is empty and indexing `[1,2]` fails. I keep the ensemble logic intact, but add a safe fallback: if no external submissions are found, we generate a valid baseline submission from `sample_submission.csv` (uniform probabilities) so the notebook runs end-to-end and always writes `submission.csv`. I also add minimal sanity checks (weights length, indices in range, column presence) to prevent silent misalignment and ensure the output columns/order match the competition format.'
- What this solution (achieved 0.4477) has done: 'Your current 0.5 score is coming from the “no external submissions found” fallback that writes uniform 0.25 probabilities, which is close to random and far from the 0.9696 target. To move toward the target with minimal change and without altering the overall “generate a submission.csv from available files” core logic, I keep your ensemble pathway intact but upgrade the fallback to a legitimate baseline trained only from `train.csv` using simple tabular features derived from the `image_id` string (no image modeling). This stays within the installed packages (pandas + scikit-learn via sklearn-pandas dependency), writes a valid `submission.csv`, and should substantially improve AUC above 0.5, moving closer to your target. I also add deterministic seeding and ensure the column order exactly matches `sample_submission.csv`.'
- What this solution (achieved 0.64323) has done: 'Your current score (0.4477) is far below the target (0.9696), and the main reason is that the fallback model uses only `image_id`-derived tabular features, which contain essentially no signal for leaf disease labels. To move toward the target while keeping your “fallback trains a simple sklearn model and outputs submission.csv” core logic, I switch the fallback features to lightweight, legitimate image features computed from the provided JPGs (simple color statistics), then train the same per-target LogisticRegression with CV averaging as you already do. This is a minimal change that uses only local competition data (no leakage) and should substantially increase ROC AUC versus the current near-random baseline. I also add robust image-path resolution (both possible dataset directory layouts) and keep the submission column order exactly as in `sample_submission.csv`.'
- What this solution (achieved 0.69924) has done: 'Your current fallback uses very coarse global color stats, which limits separability and keeps AUC far from the 0.9696 target; we can move toward the target with a minimal, legitimate improvement by enriching the image features while keeping the same per-target LogisticRegression + 5-fold CV training loop. I add a few cheap but much more informative features: HSV channel statistics plus simple “excess green/red” indices and green-pixel fraction, which are commonly helpful for leaf disease appearance. I also make the LogisticRegression configuration consistent for binary targets by setting `class_weight="balanced"` (still the same model family/training approach) to improve calibration under class imbalance typical in this dataset. All I/O paths, submission schema, and the ensemble-vs-fallback logic remain unchanged, and the script still writes `submission.csv` end-to-end.'
- What this solution (achieved 0.76737) has done: 'The timeout is overwhelmingly caused by the fallback path: it reads and processes ~1.8k JPEGs in pure Python, does expensive per-image `np.quantile` calls, and then runs 4×(5-fold) LogisticRegression fits with a pipeline that re-scales each fold. I keep the exact feature set and CV/training semantics, but make the computations equivalent and faster by (1) using OpenCV for image decoding (much faster than matplotlib), (2) parallelizing feature extraction across CPU cores while preserving deterministic ordering, (3) replacing `np.quantile` with the equivalent `np.percentile` (same algorithmic intent, faster in NumPy), and (4) avoiding pandas `.iloc` overhead inside the CV loop by using NumPy arrays while keeping the same StratifiedKFold splits and model. These changes reduce constant factors substantially without changing the model, folds, features, or target handling.'
- What this solution (achieved 0.78433) has done: 'Your current score (0.767) is well below the target (0.9696), so we should legitimately increase model signal while keeping the same overall fallback approach (hand-crafted image features + per-target LogisticRegression with 5-fold CV averaging). The smallest high-impact change is to compute the same features on a more informative region: crop to the image center to reduce background and lighting artifacts, which typically helps this dataset without changing the model/training loop. I also add a couple of very cheap texture/color-distribution features (channel percentiles and saturation high-fraction) that often separate diseased vs healthy leaves, while keeping everything else (CV, model family, I/O, submission formatting) identical. These tweaks are expected to move AUC upward toward your target without changing the core pipeline.'

# 9. Code solution

## === cell 0
import os
import re
import numpy as np
import pandas as pd



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/"

DATA_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_CSV_PATH = os.path.join(DATA_DIR, "test.csv")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")



## === cell 2
submissions_all = []
if os.path.exists(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))

submissions_all.sort()
print(submissions_all)




## === cell 3
def ensemble(submissions_all, sub_idx, weights=[]):
    if len(sub_idx) == 0:
        raise ValueError("sub_idx is empty; nothing to ensemble.")
    if len(weights) != len(sub_idx):
        raise ValueError(
            f"weights length ({len(weights)}) must match sub_idx length ({len(sub_idx)})."
        )
    if len(submissions_all) == 0:
        raise FileNotFoundError(
            "No submission files found in SUBMISSIONS_PATH to ensemble."
        )
    if max(sub_idx) >= len(submissions_all) or min(sub_idx) < 0:
        raise IndexError(
            f"sub_idx {sub_idx} out of range for {len(submissions_all)} found submissions."
        )

    submission_with_weight = []
    for i in range(len(sub_idx)):
        path = submissions_all[sub_idx[i]]
        print(f"I'm taking submission {path} with weight {weights[i]}")
        submission = pd.read_csv(path)

        required_cols = ["healthy", "multiple_diseases", "rust", "scab"]
        missing = [c for c in required_cols if c not in submission.columns]
        if missing:
            raise ValueError(f"Submission {path} missing required columns: {missing}")

        submission = submission.loc[:, required_cols].values
        submission_with_weight.append(submission * weights[i])

    submission_avg = sum(submission_with_weight)
    return submission_avg




## === cell 4
def make_submission_file(submission_avg, base_submission_path):
    submission_df = pd.read_csv(base_submission_path)

    required_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    missing = [c for c in required_cols if c not in submission_df.columns]
    if missing:
        raise ValueError(f"Base submission missing required columns: {missing}")

    if submission_avg.shape != (len(submission_df), 4):
        raise ValueError(
            f"submission_avg has shape {submission_avg.shape}, expected ({len(submission_df)}, 4)."
        )

    submission_avg = np.clip(submission_avg, 0.0, 1.0)

    submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = submission_avg
    submission_df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv")




## === cell 5
def _resolve_images_dir(data_dir):
    candidates = [
        os.path.join(data_dir, "images"),
        os.path.join(data_dir, "plant-pathology-2020-fgvc7", "images"),
        "/kaggle/input/plant-pathology-2020-fgvc7/images",
        "/kaggle/input/plant-pathology-2020-fgvc7/plant-pathology-2020-fgvc7/images",
        "/kaggle/data/plant-pathology-2020-fgvc7/images",
        "/kaggle/data/plant-pathology-2020-fgvc7/plant-pathology-2020-fgvc7/images",
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(
        f"Could not find images directory from candidates: {candidates}"
    )


def _safe_imread_rgb(path):
    try:
        import cv2
    except Exception:
        cv2 = None

    if cv2 is not None:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        if img.ndim == 2:  # grayscale
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[-1] == 4:  # BGRA -> BGR
            img = img[..., :3]
        img = img[..., ::-1]
        img = img.astype(np.float32)
        if img.max() > 1.5:
            img /= 255.0
        return img

    import matplotlib.image as mpimg

    img = mpimg.imread(path)
    if img.ndim == 2:  # grayscale -> fake RGB
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:  # RGBA -> RGB
        img = img[..., :3]
    img = img.astype(np.float32)
    if img.max() > 1.5:
        img /= 255.0
    return img


def _rgb_to_hsv_fast(img_rgb):
    r = img_rgb[..., 0]
    g = img_rgb[..., 1]
    b = img_rgb[..., 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    h = np.zeros_like(cmax, dtype=np.float32)
    mask = delta > 1e-8

    mask_r = mask & (cmax == r)
    mask_g = mask & (cmax == g)
    mask_b = mask & (cmax == b)

    h[mask_r] = ((g[mask_r] - b[mask_r]) / (delta[mask_r] + 1e-8)) % 6.0
    h[mask_g] = ((b[mask_g] - r[mask_g]) / (delta[mask_g] + 1e-8)) + 2.0
    h[mask_b] = ((r[mask_b] - g[mask_b]) / (delta[mask_b] + 1e-8)) + 4.0
    h = h / 6.0  # [0,1)

    s = np.zeros_like(cmax, dtype=np.float32)
    s[cmax > 1e-8] = delta[cmax > 1e-8] / (cmax[cmax > 1e-8] + 1e-8)

    v = cmax.astype(np.float32)
    return h, s, v


def _center_crop(img, frac=0.8):
    if frac >= 0.999:
        return img
    H, W = img.shape[0], img.shape[1]
    if H < 4 or W < 4:
        return img
    ch = max(2, int(H * frac))
    cw = max(2, int(W * frac))
    y0 = (H - ch) // 2
    x0 = (W - cw) // 2
    return img[y0 : y0 + ch, x0 : x0 + cw]


def _extract_one_image_features(image_id, images_dir):
    img_path = os.path.join(images_dir, f"{image_id}.jpg")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Missing image file: {img_path}")

    img = _safe_imread_rgb(img_path)

    img = _center_crop(img, frac=0.8)

    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    h, s, v = _rgb_to_hsv_fast(img)
    exg = 2.0 * g - r - b  # excess green
    exr = 1.4 * r - g  # excess red (simple)
    ndgi = (g - r) / (g + r + 1e-8)  # normalized difference green-red
    green_frac = float(((g > r) & (g > b) & (g > 0.25)).mean())

    gx = np.diff(gray, axis=1)  # (H, W-1)
    gy = np.diff(gray, axis=0)  # (H-1, W)
    if gx.shape[0] == 0 or gx.shape[1] == 0 or gy.shape[0] == 0 or gy.shape[1] == 0:
        grad_mag_mean = 0.0
        grad_mag_std = 0.0
    else:
        Hm = min(gx.shape[0], gy.shape[0])
        Wm = min(gx.shape[1], gy.shape[1])
        gxi = gx[:Hm, :Wm]
        gyi = gy[:Hm, :Wm]
        grad_mag = np.sqrt(gxi * gxi + gyi * gyi)
        grad_mag_mean = float(grad_mag.mean())
        grad_mag_std = float(grad_mag.std())

    lap = (
        -4.0 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    lap_mean = float(lap.mean())
    lap_std = float(lap.std())
    lap_energy = float((lap * lap).mean())

    rg = r / (g + 1e-8)
    rb = r / (b + 1e-8)
    gb = g / (b + 1e-8)

    p10_gray, p50_gray, p90_gray = np.percentile(gray, [10, 50, 90])
    p90_exg = float(np.percentile(exg, 90))

    p10_r, p50_r, p90_r = np.percentile(r, [10, 50, 90])
    p10_g, p50_g, p90_g = np.percentile(g, [10, 50, 90])
    p10_b, p50_b, p90_b = np.percentile(b, [10, 50, 90])
    sat_hi_frac = float((s > 0.5).mean())

    f = {
        "mean_r": float(r.mean()),
        "mean_g": float(g.mean()),
        "mean_b": float(b.mean()),
        "std_r": float(r.std()),
        "std_g": float(g.std()),
        "std_b": float(b.std()),
        "p10_r": float(p10_r),
        "p50_r": float(p50_r),
        "p90_r": float(p90_r),
        "p10_g": float(p10_g),
        "p50_g": float(p50_g),
        "p90_g": float(p90_g),
        "p10_b": float(p10_b),
        "p50_b": float(p50_b),
        "p90_b": float(p90_b),
        "mean_gray": float(gray.mean()),
        "std_gray": float(gray.std()),
        "p10_gray": float(p10_gray),
        "p50_gray": float(p50_gray),
        "p90_gray": float(p90_gray),
        "mean_h": float(h.mean()),
        "std_h": float(h.std()),
        "mean_s": float(s.mean()),
        "std_s": float(s.std()),
        "sat_hi_frac": sat_hi_frac,
        "mean_v": float(v.mean()),
        "std_v": float(v.std()),
        "mean_exg": float(exg.mean()),
        "std_exg": float(exg.std()),
        "p90_exg": p90_exg,
        "mean_exr": float(exr.mean()),
        "std_exr": float(exr.std()),
        "mean_ndgi": float(ndgi.mean()),
        "std_ndgi": float(ndgi.std()),
        "green_frac": green_frac,
        "grad_mag_mean": grad_mag_mean,
        "grad_mag_std": grad_mag_std,
        "lap_mean": lap_mean,
        "lap_std": lap_std,
        "lap_energy": lap_energy,
        "mean_rg": float(rg.mean()),
        "std_rg": float(rg.std()),
        "mean_rb": float(rb.mean()),
        "std_rb": float(rb.std()),
        "mean_gb": float(gb.mean()),
        "std_gb": float(gb.std()),
    }
    return f


def _extract_image_features(df, images_dir):
    image_ids = df["image_id"].astype(str).tolist()

    from concurrent.futures import ThreadPoolExecutor

    max_workers = min(8, (os.cpu_count() or 2))
    feats = [None] * len(image_ids)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, f in enumerate(
            ex.map(lambda iid: _extract_one_image_features(iid, images_dir), image_ids)
        ):
            feats[i] = f

    return pd.DataFrame(feats, index=df.index)


def fallback_train_image_feature_model_predict(
    train_csv_path, test_csv_path, sample_sub_path, data_dir
):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression

    train = pd.read_csv(train_csv_path)
    test = pd.read_csv(test_csv_path)
    target_cols = ["healthy", "multiple_diseases", "rust", "scab"]

    images_dir = _resolve_images_dir(data_dir)
    X_train_df = _extract_image_features(train, images_dir)
    X_test_df = _extract_image_features(test, images_dir)
    y = train[target_cols].copy()

    X_train = X_train_df.to_numpy(dtype=np.float64, copy=False)
    X_test = X_test_df.to_numpy(dtype=np.float64, copy=False)

    def fit_predict_one_target(y_col):
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                (
                    "clf",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=3000,
                        C=1.0,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        preds_test = np.zeros(len(X_test), dtype=float)

        y_bin = y_col.values
        for tr_idx, va_idx in skf.split(X_train, y_bin):
            model.fit(X_train[tr_idx], y_bin[tr_idx])
            preds_test += model.predict_proba(X_test)[:, 1] / skf.n_splits

        return preds_test

    preds = np.column_stack([fit_predict_one_target(y[c]) for c in target_cols])

    sample = pd.read_csv(sample_sub_path)
    test_with_preds = test[["image_id"]].copy()
    for i, c in enumerate(target_cols):
        test_with_preds[c] = preds[:, i]

    merged = sample[["image_id"]].merge(test_with_preds, on="image_id", how="left")
    if merged[target_cols].isnull().any().any():
        raise ValueError("Prediction merge produced NaNs; image_id alignment issue.")

    return merged[target_cols].values




## === cell 6
if len(submissions_all) >= 3:
    submission_avg = ensemble(submissions_all, [1, 2], [0.7, 0.3])
    make_submission_file(submission_avg, submissions_all[0])
elif len(submissions_all) >= 1:
    idx = list(range(len(submissions_all)))
    w = [1.0 / len(idx)] * len(idx)
    submission_avg = ensemble(submissions_all, idx, w)
    make_submission_file(submission_avg, submissions_all[0])
else:
    submission_avg = fallback_train_image_feature_model_predict(
        TRAIN_CSV_PATH, TEST_CSV_PATH, SAMPLE_SUB_PATH, DATA_DIR
    )
    make_submission_file(submission_avg, SAMPLE_SUB_PATH)
