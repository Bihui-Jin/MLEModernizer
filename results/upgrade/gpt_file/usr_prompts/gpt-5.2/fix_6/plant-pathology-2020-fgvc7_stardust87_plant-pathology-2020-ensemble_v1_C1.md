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

0.9625763959938703

# 6. Current score

0.58302

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the shape-mismatch bug by ensuring every candidate submission is aligned to the competition test set (`test.csv`) by `image_id`, and I ignore any CSVs that look like train-sized predictions. I also normalize/renormalize ensemble weights so the averaged probabilities stay on a consistent scale (score-neutral but safer). Finally, I make the fallback produce a valid uniform-probability submission with the correct rows/columns and write `submission.csv` end-to-end.'
- What this solution (achieved 0.66866) has done: 'Your current 0.5 score strongly suggests the fallback uniform 0.25 submission is being used (or the discovered CSVs are not real model predictions), so the smallest safe way to move toward the 0.9626 target is to generate a real image-based model prediction inside this notebook rather than trying to ensemble unknown external CSVs. I keep your existing submission alignment/formatting logic, but add a minimal scikit-learn baseline that reads images, extracts simple color/texture features, trains a one-vs-rest logistic regression for the 4 labels, and predicts probabilities for the test set. This preserves evaluation semantics (probabilities per class) and should materially improve ROC AUC versus uniform guessing while staying within the installed-package constraints. If image reading fails for any reason, the code still fall back to your original CSV-discovery ensemble and then finally to uniform 0.25 so it always produces a valid `submission.csv`.'
- What this solution (achieved 0.58405) has done: 'Your current baseline uses only 10 global color statistics, which is usually too weak for this task and explains the big gap to the 0.9626 target. I keep the same scikit-learn OneVsRest logistic-regression pipeline, but strengthen the existing feature extractor in a minimal way by adding simple downsampled “thumbnail” pixel features (still just deterministic numeric features, no new model/training loop). I also add a tiny, deterministic image resize inside `_read_image_rgb` to make feature extraction consistent and faster, and increase `max_iter` slightly to ensure convergence rather than changing the learning approach. This should materially increase ROC AUC while preserving your overall structure and still writing a valid `submission.csv` end-to-end.'
- What this solution (achieved 0.58302) has done: 'Your current pipeline is already producing a valid submission, but the score gap to 0.9626 is large, so the most reliable minimal improvement is to strengthen the *same* scikit-learn baseline without changing its core approach (still deterministic hand-crafted features + OneVsRest logistic regression). I keep the exact training loop/model family, but (1) add a couple of very cheap texture/shape statistics (edge energy + Laplacian variance + center-vs-border color difference) that are highly relevant for leaf lesion patterns, and (2) use a slightly higher-resolution thumbnail (still small) to carry more spatial information. These changes typically improve ROC AUC materially versus global color stats while staying within your constraints and runtime. Everything else (alignment, clipping, fallback behavior, submission writing) stays the same.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/submissions/"

COMP_DATA_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
SAMPLE_SUB_PATH = os.path.join(COMP_DATA_DIR, "sample_submission.csv")
TEST_CSV_PATH = os.path.join(COMP_DATA_DIR, "test.csv")
TRAIN_CSV_PATH = os.path.join(COMP_DATA_DIR, "train.csv")

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

test_df = pd.read_csv(TEST_CSV_PATH)
test_ids = test_df["image_id"].astype(str)
n_test = len(test_df)

sample_df = pd.read_csv(SAMPLE_SUB_PATH)
assert (
    list(sample_df.columns) == ["image_id"] + TARGET_COLS
), "Unexpected sample_submission format"
assert (
    len(sample_df) == n_test
), "sample_submission and test.csv row counts differ unexpectedly"

print("n_test:", n_test)
print("sample_submission columns:", list(sample_df.columns))



## === cell 2
submissions_all = []


def _is_candidate_submission_csv(fp: str) -> bool:
    try:
        head = pd.read_csv(fp, nrows=5)
    except Exception:
        return False
    if "image_id" not in head.columns:
        return False
    if not all(c in head.columns for c in TARGET_COLS):
        return False
    return True


if os.path.exists(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                fp = os.path.join(dirname, filename)
                if _is_candidate_submission_csv(fp):
                    submissions_all.append(fp)

if len(submissions_all) == 0 and os.path.exists("/kaggle/input"):
    for dirname, _, filenames in os.walk("/kaggle/input"):
        for filename in filenames:
            if not filename.lower().endswith(".csv"):
                continue
            fp = os.path.join(dirname, filename)
            if _is_candidate_submission_csv(fp):
                submissions_all.append(fp)

print(f"Found {len(submissions_all)} candidate submission CSV(s).")
for p in submissions_all[:20]:
    print(p)




## === cell 3
def _load_and_align_submission(fp: str, test_ids: pd.Series) -> pd.DataFrame:
    """
    Load a submission-like CSV and align it to test_ids by image_id.
    Returns a DataFrame aligned to test_ids with columns TARGET_COLS.
    Raises ValueError if it cannot be aligned safely.
    """
    df = pd.read_csv(fp)
    missing = [c for c in ["image_id"] + TARGET_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"File {fp} missing columns: {missing}")

    df = df.loc[:, ["image_id"] + TARGET_COLS].copy()
    df["image_id"] = df["image_id"].astype(str)
    df = df.drop_duplicates(subset="image_id", keep="first")

    aligned = df.set_index("image_id").reindex(test_ids.values)

    n_missing_rows = int(aligned[TARGET_COLS].isna().any(axis=1).sum())
    if n_missing_rows > 0:
        raise ValueError(
            f"File {fp} cannot be aligned to test set (missing {n_missing_rows}/{len(test_ids)} image_id rows)."
        )

    for c in TARGET_COLS:
        aligned[c] = pd.to_numeric(aligned[c], errors="coerce")
    if aligned[TARGET_COLS].isna().any().any():
        raise ValueError(f"File {fp} has non-numeric predictions after alignment.")

    aligned[TARGET_COLS] = aligned[TARGET_COLS].clip(0.0, 1.0)
    return aligned[TARGET_COLS]


def ensemble(submissions_all, sub_idx, weights):
    if len(submissions_all) == 0:
        raise ValueError("No submission files available to ensemble.")
    if len(sub_idx) == 0:
        raise ValueError("sub_idx is empty.")
    if len(sub_idx) != len(weights):
        raise ValueError("sub_idx and weights must have the same length.")
    if any(i < 0 or i >= len(submissions_all) for i in sub_idx):
        raise IndexError(
            f"sub_idx contains out-of-range indices for submissions_all (len={len(submissions_all)})."
        )

    wsum = float(sum(weights))
    if wsum == 0.0:
        raise ValueError("Sum of weights must be non-zero.")
    weights = [float(w) / wsum for w in weights]

    submission_with_weight = None
    used_files = []
    for i, w in zip(sub_idx, weights):
        fp = submissions_all[i]
        preds = _load_and_align_submission(fp, test_ids).values  # shape (n_test, 4)
        if submission_with_weight is None:
            submission_with_weight = preds * w
        else:
            submission_with_weight += preds * w
        used_files.append(fp)

    print("Ensembled files:")
    for u in used_files:
        print(" -", u)

    return submission_with_weight




## === cell 4
def make_submission_file(submission_avg):
    submission_df = pd.read_csv(SAMPLE_SUB_PATH)

    if submission_avg is not None:
        if submission_avg.shape != (len(submission_df), len(TARGET_COLS)):
            raise ValueError(
                f"submission_avg has shape {submission_avg.shape}, expected {(len(submission_df), len(TARGET_COLS))}."
            )
        submission_df.loc[:, TARGET_COLS] = submission_avg

    submission_df["image_id"] = submission_df["image_id"].astype(str)
    for c in TARGET_COLS:
        submission_df[c] = (
            pd.to_numeric(submission_df[c], errors="coerce").fillna(0.25).clip(0.0, 1.0)
        )

    submission_df.to_csv("submission.csv", index=False)
    return submission_df




## === cell 5
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


def _find_images_dir(comp_dir: str) -> str:
    cand = os.path.join(comp_dir, "images")
    if os.path.isdir(cand):
        return cand
    for p in [
        "/kaggle/input/plant-pathology-2020-fgvc7/images",
        "/kaggle/data/plant-pathology-2020-fgvc7/images",
        "/kaggle/input/images",
        "/kaggle/data/images",
    ]:
        if os.path.isdir(p):
            return p
    return cand


IMAGES_DIR = _find_images_dir(COMP_DATA_DIR)
print("IMAGES_DIR:", IMAGES_DIR, "exists:", os.path.isdir(IMAGES_DIR))


def _read_image_rgb(path: str, resize_to=(128, 128)):
    try:
        from PIL import Image

        img = Image.open(path).convert("RGB")
        if resize_to is not None:
            img = img.resize(resize_to, resample=Image.BILINEAR)
        return np.asarray(img)
    except Exception:
        try:
            import matplotlib.image as mpimg

            img = mpimg.imread(path)
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            if img.shape[-1] == 4:
                img = img[..., :3]
            if img.dtype != np.uint8:
                img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
            return img
        except Exception as e:
            raise RuntimeError(f"Could not read image {path}: {e}")


def _image_features(rgb: np.ndarray, thumb_size=20) -> np.ndarray:
    x = rgb.astype(np.float32) / 255.0

    means = x.reshape(-1, 3).mean(axis=0)
    stds = x.reshape(-1, 3).std(axis=0)

    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    gr = (g - r).reshape(-1)
    br = (b - r).reshape(-1)

    y = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)

    dx = y[:, 1:] - y[:, :-1]
    dy = y[1:, :] - y[:-1, :]
    edge_energy = float(np.mean(np.abs(dx)) + np.mean(np.abs(dy)))

    lap = -4.0 * y[1:-1, 1:-1] + y[1:-1, :-2] + y[1:-1, 2:] + y[:-2, 1:-1] + y[2:, 1:-1]
    lap_var = float(np.var(lap)) if lap.size > 0 else 0.0

    h, w = y.shape
    ch0, ch1 = int(0.25 * h), int(0.75 * h)
    cw0, cw1 = int(0.25 * w), int(0.75 * w)
    center = x[ch0:ch1, cw0:cw1, :].reshape(-1, 3)
    border_mask = np.ones((h, w), dtype=bool)
    border_mask[ch0:ch1, cw0:cw1] = False
    border = x[border_mask].reshape(-1, 3)
    if center.size == 0 or border.size == 0:
        center_border = np.zeros(3, dtype=np.float32)
    else:
        center_border = (center.mean(axis=0) - border.mean(axis=0)).astype(np.float32)

    base = np.concatenate(
        [
            means,
            stds,
            [gr.mean(), gr.std(), br.mean(), br.std()],
            [edge_energy, lap_var],
            center_border,
        ]
    ).astype(np.float32)

    h2, w2, _ = x.shape
    th = min(thumb_size, h2)
    tw = min(thumb_size, w2)
    ch = (h2 // th) * th
    cw = (w2 // tw) * tw
    yy = x[:ch, :cw, :]
    yy = yy.reshape(th, ch // th, tw, cw // tw, 3).mean(axis=(1, 3))  # (th, tw, 3)
    thumb = yy.reshape(-1).astype(np.float32)

    feats = np.concatenate([base, thumb]).astype(np.float32)
    return feats


def _build_features(image_ids: pd.Series, images_dir: str, thumb_size=20) -> np.ndarray:
    image_ids_arr = image_ids.astype(str).values
    if len(image_ids_arr) == 0:
        return np.zeros((0, 10), dtype=np.float32)

    def _resolve_fp(img_id: str) -> str:
        fp = os.path.join(images_dir, f"{img_id}.jpg")
        if not os.path.exists(fp):
            g = glob.glob(os.path.join(images_dir, img_id + ".*"))
            if len(g) > 0:
                fp = g[0]
        return fp

    fp0 = _resolve_fp(image_ids_arr[0])
    rgb0 = _read_image_rgb(fp0, resize_to=(128, 128))
    f0 = _image_features(rgb0, thumb_size=thumb_size)
    d = int(f0.shape[0])

    feats = np.zeros((len(image_ids_arr), d), dtype=np.float32)
    feats[0] = f0
    for i in range(1, len(image_ids_arr)):
        fp = _resolve_fp(image_ids_arr[i])
        rgb = _read_image_rgb(fp, resize_to=(128, 128))
        feats[i] = _image_features(rgb, thumb_size=thumb_size)
    return feats


def train_and_predict_baseline(
    train_csv_path: str, test_ids: pd.Series, images_dir: str
) -> np.ndarray:
    train_df = pd.read_csv(train_csv_path)
    train_df["image_id"] = train_df["image_id"].astype(str)

    X_train = _build_features(train_df["image_id"], images_dir, thumb_size=20)
    y_train = train_df[TARGET_COLS].astype(int).values

    X_test = _build_features(test_ids, images_dir, thumb_size=20)

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "ovr",
                OneVsRestClassifier(
                    LogisticRegression(solver="lbfgs", max_iter=2000, C=2.0)
                ),
            ),
        ]
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)  # shape (n_test, 4)
    proba = np.clip(proba, 0.0, 1.0)
    return proba




## === cell 6
submission_avg = None

try:
    if os.path.exists(TRAIN_CSV_PATH) and os.path.isdir(IMAGES_DIR):
        submission_avg = train_and_predict_baseline(
            TRAIN_CSV_PATH, test_ids, IMAGES_DIR
        )
        print("Baseline image model predictions generated:", submission_avg.shape)
except Exception as e:
    print("Baseline image model failed, falling back. Reason:", repr(e))
    submission_avg = None

if submission_avg is None and len(submissions_all) >= 2:
    try:
        submission_avg = ensemble(submissions_all, [0, 1], [0.4, 0.6])
    except Exception as e:
        print("Ensembling first two candidates failed, falling back. Reason:", repr(e))
        submission_avg = None

if submission_avg is None and len(submissions_all) >= 1:
    for idx in range(len(submissions_all)):
        try:
            submission_avg = ensemble(submissions_all, [idx], [1.0])
            break
        except Exception as e:
            print(
                f"Single-file candidate at index {idx} failed alignment, skipping. Reason:",
                repr(e),
            )
            submission_avg = None

if submission_avg is None:
    submission_avg = pd.DataFrame(0.25, index=range(n_test), columns=TARGET_COLS).values

sub_df = make_submission_file(submission_avg)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
print("Columns:", list(sub_df.columns))
print("submission.csv exists:", os.path.exists("submission.csv"))
