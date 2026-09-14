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

0.9629218283013236

# 6. Current score

0.54788

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The crash happens because `/kaggle/input/submissions/` doesn’t exist in this environment, so `submissions_all` is empty and indexing `[0,2,4]` fails. I keep the ensemble logic intact, but add a safe fallback: if no external submission files are found, we generate a valid baseline submission from the provided `sample_submission.csv` (uniform probabilities). This guarantees the notebook runs end-to-end and always writes `submission.csv` with the correct columns and row order. I also add minimal guards to prevent mismatched indices/weights from causing runtime errors.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 score comes from outputting uniform probabilities (or failing to find any real model submissions), which is far from the 0.9629 target. The smallest legitimate improvement without changing the “core logic” (you’re not training any model here) is to generate stronger probabilities directly from the provided training labels: use the per-class prevalence as constant predictions for all test images, which usually beat uniform on mean ROC AUC. I keep the existing ensemble behavior unchanged when external submissions exist, but improve the no-submission fallback from uniform (0.25) to label-prior probabilities computed from `train.csv`. I also harden column ordering/alignment using `sample_submission.csv` so the output is always valid and correctly ordered.'
- What this solution (achieved 0.56205) has done: 'I fix the fallback model crash by correcting the image filename mapping: the `image_id` values in `train.csv`/`test.csv` don’t include the `.jpg` extension, so the loader was trying to open non-existent paths and loaded zero images. I add a minimal helper that resolves each `image_id` to an existing file in the images directory (trying `id`, `id.jpg`, `id.JPG`, etc.), and use it in both train/test feature loading. This preserves the core logic (pixel features + multinomial LogisticRegression fallback, and the ensemble path unchanged) while making the pipeline run end-to-end and produce a valid `submission.csv` with correctly ordered columns.'
- What this solution (achieved 0.57929) has done: 'Your current score is far below the target, so we should legitimately improve predictions while keeping your core approach (pixel features + multinomial LogisticRegression, same training loop/solver) intact. The biggest score issue is that you’re collapsing the 4-label multilabel targets into a single multiclass via `argmax`, which mismatches the competition’s per-column ROC AUC (multilabel) evaluation. I keep the same feature extraction and LR family, but train one LogisticRegression per target (one-vs-rest) and output 4 independent probabilities, which aligns predictions with the metric and usually yields a large AUC jump. I also make a minimal safety fix to ensure probabilities are always valid and aligned with `sample_submission.csv` row order.'
- What this solution (achieved 0.56237) has done: 'Your current 0.579 score suggests the fallback pixel+LogisticRegression pipeline is still underpowered rather than broken, so the smallest legitimate move toward the 0.963 target is to make the existing model’s inputs a bit more informative without changing the overall approach. I keep the same “flatten pixels → one LogisticRegression per label → predict_proba” core logic, but (1) increase image resolution moderately (64→96) to capture more disease texture and (2) add a simple per-image standardization (zero-mean/unit-std) inside the same feature extraction step to stabilize LR optimization. I also batch `predict_proba` over the full test matrix per label (same semantics, just faster/less overhead) to stay within the 600s runtime. Submission formatting and column alignment remain unchanged via `sample_submission.csv`.'
- What this solution (achieved 0.58066) has done: 'Your current score is far below the target, so we should improve the model’s signal while keeping your existing “flatten pixels → one LogisticRegression per label → predict_proba” approach intact. The biggest low-risk gain here is to add a standard scaler on the pixel features (fit on train, apply to test) so LogisticRegression is properly conditioned; this doesn’t change the core logic, but usually boosts AUC substantially. I also set `class_weight="balanced"` to reduce the impact of label imbalance per target (still the same LR model family/training loop), and I keep the submission formatting/alignment exactly tied to `sample_submission.csv`. No changes are made to the ensemble path; only the fallback training path is strengthened.'
- What this solution (achieved 0.58019) has done: 'Your current gap to the target (0.58066 → 0.9629) is large, so we should improve signal while keeping your exact core approach (flattened pixel features → StandardScaler → one LogisticRegression per label → predict_proba). The most impactful minimal change is to avoid training on a subset of images caused by silent load failures: I make image resolution consistent (keep 96×96), but speed and stabilize loading by enforcing RGB, handling truncated images, and parallelizing feature extraction with threads so more images are actually used within the time limit. I also ensure train/test feature matrices align exactly with the CSV row order (no dropped rows), using priors only when an image truly can’t be read. Finally, I keep the ensemble path untouched and still write a valid `submission.csv` aligned to `sample_submission.csv` columns and order.'
- What this solution (achieved 0.58101) has done: 'Your current score (0.58019) is far below the target (0.9629), so we should improve the fallback model’s predictive signal while keeping the same core approach (flattened pixel features → StandardScaler → one LogisticRegression per label → predict_proba). The most likely limiter is that raw resized pixels are a weak representation; a minimal, still-linear change is to concatenate a tiny set of handcrafted color/texture statistics to the same pixel vector, which typically boosts separability without changing the training loop or model family. I also ensure train/test features are built in a consistent dtype/shape and keep the submission aligned exactly to `sample_submission.csv`. The ensemble path remains untouched.'
- What this solution (achieved 0.55379) has done: 'Your current score (0.581) is far below the target (0.963), so we should improve predictive signal while keeping your exact core approach: flattened pixel features → StandardScaler → one LogisticRegression per label → predict_proba. The smallest high-impact change is to avoid washing out class-separating color cues: keep your per-image mean/std normalization (as you have), but also concatenate the original downsampled RGB pixels (in [0,1]) alongside the standardized pixels and your existing stats, so LR can exploit both absolute color and normalized texture. Additionally, I add a tiny amount of data augmentation at feature level by averaging features from the original image and its horizontal flip for both train and test; this keeps the same model/training loop but typically improves robustness on leaves. Finally, I keep the ensemble path untouched and preserve submission alignment/columns via `sample_submission.csv`.'
- What this solution (achieved 0.55379) has done: 'Your current score is far below the target, so we should improve the fallback model’s signal while keeping the same core pipeline (flip-averaged flattened pixel features + StandardScaler + one LogisticRegression per label). The smallest high-impact fix is to correct the image root: your `IMAGES_DIR` currently points to `/kaggle/input/plant-pathology-2020-fgvc7/images`, but your provided data lives under `/kaggle/data/...`; this likely caused many/most images to fail to load and collapse predictions toward priors. I also add a minimal “auto-detect data root” fallback that prefers the existing path if present, otherwise switches to the known `/kaggle/data/...` locations, and I add a lightweight sanity print of how many images were successfully loaded to ensure we’re actually training on real features. No model architecture, loss, or training approach is changed, and the script still always writes a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.58343) has done: 'Your current score is far below the target, so we need a legitimate boost while keeping your exact pipeline (flip-averaged pixel+stats features → StandardScaler → one LogisticRegression per label → predict_proba). The smallest high-impact fix is to correct feature construction: right now you scale features, but you also include global stats already on a reasonable scale; standardizing everything can wash out those stats, and the LR is learning from a very high-dimensional raw pixel vector without any dimensionality reduction. To move toward the target without changing model family or training loop, I keep the same feature extractor and LR-per-label approach but add a lightweight PCA (fit on train, apply to test) after scaling to improve conditioning and generalization on these dense correlated pixel features. I also increase `max_iter` modestly to ensure convergence with the transformed features (no early stopping), and keep submission alignment exactly tied to `sample_submission.csv`.'
- What this solution (achieved 0.5833) has done: 'We keep your exact pipeline (flip-averaged pixel+stats features → StandardScaler → PCA → one LogisticRegression per label → predict_proba) but fix one key mismatch: you currently `StandardScaler` across all features, which washes out your handcrafted stats and the absolute-color pixel block you intentionally concatenated. The minimal improvement is to standardize only the high-dimensional pixel parts (both standardized-pixels block and raw RGB block) while leaving the 10 global stats unscaled, preserving their signal without changing the model family or training approach. This should legitimately increase ROC AUC toward your target while keeping semantics identical (same features, same LR, same PCA, same prediction flow). Submission formatting/ordering remains locked to `sample_submission.csv`.'
- What this solution (achieved 0.54788) has done: 'Your current score (0.5833) is far below the target (0.9629), so we should legitimately improve signal while keeping your exact pipeline (flip-averaged pixel+stats features → partial scaling → PCA → one LogisticRegression per label → predict_proba). The most likely issue now is the PCA bottleneck: with 96×96×3 pixels (twice) you have ~55k pixel features, and reducing to 256 components is probably discarding too much disease-relevant variation for a linear model. I make the smallest change that preserves core logic by increasing `pca_components` (still PCA in the same place) to retain more information, and I also ensure the image resize uses a high-quality resampling filter for more stable features. Everything else (feature construction, scaling strategy, LR settings, submission formatting) stays the same, and it still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import pandas as pd

from PIL import Image, ImageFile
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from concurrent.futures import ThreadPoolExecutor

ImageFile.LOAD_TRUNCATED_IMAGES = True



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/"

_CANDIDATE_ROOTS = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
]
DATA_ROOT = next(
    (p for p in _CANDIDATE_ROOTS if os.path.isfile(os.path.join(p, "train.csv"))), None
)
if DATA_ROOT is None:
    DATA_ROOT = "/kaggle/input/plant-pathology-2020-fgvc7"

SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_PATH = os.path.join(DATA_ROOT, "train.csv")
TEST_PATH = os.path.join(DATA_ROOT, "test.csv")
IMAGES_DIR = os.path.join(DATA_ROOT, "images")

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

print("Using DATA_ROOT:", DATA_ROOT)
print("TRAIN_PATH exists:", os.path.isfile(TRAIN_PATH))
print("TEST_PATH exists:", os.path.isfile(TEST_PATH))
print("SAMPLE_SUB_PATH exists:", os.path.isfile(SAMPLE_SUB_PATH))
print("IMAGES_DIR exists:", os.path.isdir(IMAGES_DIR))



## === cell 2
submissions_all = []
if os.path.isdir(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print("Found submission files:", submissions_all)




## === cell 3
def ensemble(submissions_all, sub_idx, weights=[]):
    if len(sub_idx) != len(weights):
        raise ValueError(
            f"sub_idx and weights must have same length. Got {len(sub_idx)} and {len(weights)}"
        )

    if len(submissions_all) == 0:
        raise ValueError("No submission files available to ensemble.")

    submission_with_weight = []
    for i in range(len(sub_idx)):
        if sub_idx[i] < 0 or sub_idx[i] >= len(submissions_all):
            raise IndexError(
                f"sub_idx[{i}]={sub_idx[i]} is out of range for {len(submissions_all)} available files."
            )
        print(
            f"I'm taking submission {submissions_all[sub_idx[i]]} with weight {weights[i]}"
        )
        submission = pd.read_csv(submissions_all[sub_idx[i]])

        missing = [c for c in TARGET_COLS if c not in submission.columns]
        if missing:
            raise ValueError(
                f"Submission {submissions_all[sub_idx[i]]} missing columns: {missing}"
            )

        submission = submission.loc[:, TARGET_COLS].values
        submission_with_weight.append(submission * weights[i])

    submission_avg = sum(submission_with_weight)
    return submission_avg




## === cell 4
def make_submission_file(submission_avg):
    submission_df = pd.read_csv(SAMPLE_SUB_PATH)
    submission_df = submission_df.loc[:, ["image_id"] + TARGET_COLS].copy()

    submission_avg = np.asarray(submission_avg, dtype=np.float32)
    if submission_avg.shape != (len(submission_df), len(TARGET_COLS)):
        raise ValueError(
            f"submission_avg has shape {submission_avg.shape} but expected {(len(submission_df), len(TARGET_COLS))}"
        )
    submission_avg = np.clip(submission_avg, 0.0, 1.0)

    submission_df.loc[:, TARGET_COLS] = submission_avg
    submission_df.to_csv("submission.csv", index=False)




## === cell 5
def _resolve_image_path(image_dir, image_id):
    candidates = [
        image_id,
        f"{image_id}.jpg",
        f"{image_id}.JPG",
        f"{image_id}.jpeg",
        f"{image_id}.JPEG",
        f"{image_id}.png",
        f"{image_id}.PNG",
    ]
    for name in candidates:
        p = os.path.join(image_dir, name)
        if os.path.isfile(p):
            return p
    return None


def _img_stats(arr_rgb_01):
    ch_mean = arr_rgb_01.mean(axis=(0, 1))
    ch_std = arr_rgb_01.std(axis=(0, 1))
    gray = (
        0.2989 * arr_rgb_01[..., 0]
        + 0.5870 * arr_rgb_01[..., 1]
        + 0.1140 * arr_rgb_01[..., 2]
    ).astype(np.float32)
    g_mean = float(gray.mean())
    g_std = float(gray.std())
    dx = np.abs(gray[:, 1:] - gray[:, :-1]).mean() if gray.shape[1] > 1 else 0.0
    dy = np.abs(gray[1:, :] - gray[:-1, :]).mean() if gray.shape[0] > 1 else 0.0
    stats = np.array(
        [
            ch_mean[0],
            ch_mean[1],
            ch_mean[2],
            ch_std[0],
            ch_std[1],
            ch_std[2],
            g_mean,
            g_std,
            float(dx),
            float(dy),
        ],
        dtype=np.float32,
    )
    return stats


def _load_image_feature_flipavg(image_path, size=(96, 96)):
    try:
        resample = getattr(Image, "Resampling", Image).BILINEAR
        if hasattr(Image, "Resampling"):
            resample = Image.Resampling.BICUBIC
        else:
            resample = Image.BICUBIC

        with Image.open(image_path) as img:
            img = img.convert("RGB").resize(size, resample=resample)
            arr01 = (np.asarray(img, dtype=np.float32) / 255.0).astype(np.float32)
        arr01_flip = arr01[:, ::-1, :].copy()

        def _feat_from_arr(arr01_local):
            m = float(arr01_local.mean())
            s = float(arr01_local.std())
            if s < 1e-6:
                s = 1.0
            arr_std = (arr01_local - m) / s
            stats = _img_stats(arr01_local)
            return np.concatenate(
                [
                    arr_std.reshape(-1).astype(np.float32),
                    arr01_local.reshape(-1).astype(np.float32),
                    stats,
                ],
                axis=0,
            )

        f1 = _feat_from_arr(arr01)
        f2 = _feat_from_arr(arr01_flip)
        return ((f1 + f2) * 0.5).astype(np.float32)
    except Exception:
        return None


def train_predict_lr_on_pixels(
    train_df,
    test_df,
    image_dir,
    target_cols,
    size=(96, 96),
    seed=42,
    n_threads=8,
    pca_components=256,
):
    priors = train_df[target_cols].mean().astype(float).values  # (4,)

    train_ids = train_df["image_id"].tolist()
    test_ids = test_df["image_id"].tolist()

    train_paths = [_resolve_image_path(image_dir, iid) for iid in train_ids]
    test_paths = [_resolve_image_path(image_dir, iid) for iid in test_ids]

    n_train_resolved = sum(p is not None for p in train_paths)
    n_test_resolved = sum(p is not None for p in test_paths)
    print(
        f"Resolved image paths: train {n_train_resolved}/{len(train_paths)}, test {n_test_resolved}/{len(test_paths)}"
    )

    def _feat_or_none(p):
        if p is None:
            return None
        return _load_image_feature_flipavg(p, size=size)

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        train_feats = list(ex.map(_feat_or_none, train_paths))
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        test_feats = list(ex.map(_feat_or_none, test_paths))

    first_feat = next((f for f in train_feats if f is not None), None)
    if first_feat is None:
        raise RuntimeError(
            "No training images could be loaded; cannot build fallback model. "
            "Check IMAGES_DIR and image_id filename resolution."
        )
    d = int(first_feat.shape[0])

    X_train = np.zeros((len(train_df), d), dtype=np.float32)
    train_has_feat = np.zeros(len(train_df), dtype=bool)
    for i, f in enumerate(train_feats):
        if f is not None:
            X_train[i, :] = f
            train_has_feat[i] = True

    X_test = np.zeros((len(test_df), d), dtype=np.float32)
    test_has_feat = np.zeros(len(test_df), dtype=bool)
    for i, f in enumerate(test_feats):
        if f is not None:
            X_test[i, :] = f
            test_has_feat[i] = True

    print(
        f"Loaded image features: train {int(train_has_feat.sum())}/{len(train_has_feat)}, test {int(test_has_feat.sum())}/{len(test_has_feat)}"
    )

    stats_dim = 10
    if d <= stats_dim:
        raise RuntimeError(f"Unexpected feature dim d={d}; expected > {stats_dim}.")

    pix_dim_total = d - stats_dim
    pix1_dim = pix_dim_total // 2  # arr_std block
    pix2_dim = pix_dim_total - pix1_dim  # raw RGB block (same size)

    X_train_pix = X_train[:, :pix_dim_total]
    X_train_stats = X_train[:, pix_dim_total:]
    X_test_pix = X_test[:, :pix_dim_total]
    X_test_stats = X_test[:, pix_dim_total:]

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_pix_sc = scaler.fit_transform(X_train_pix)
    X_test_pix_sc = scaler.transform(X_test_pix)

    X_train_sc = np.concatenate([X_train_pix_sc, X_train_stats], axis=1).astype(
        np.float32
    )
    X_test_sc = np.concatenate([X_test_pix_sc, X_test_stats], axis=1).astype(np.float32)

    n_comp = int(min(pca_components, X_train_sc.shape[0] - 1, X_train_sc.shape[1]))
    if n_comp >= 2:
        pca = PCA(n_components=n_comp, random_state=seed, svd_solver="randomized")
        X_train_fin = pca.fit_transform(X_train_sc)
        X_test_fin = pca.transform(X_test_sc)
        print(f"Applied PCA: {X_train_sc.shape[1]} -> {X_train_fin.shape[1]}")
    else:
        X_train_fin = X_train_sc
        X_test_fin = X_test_sc
        print("Skipped PCA (too few components possible).")

    Y = train_df[target_cols].values.astype(int)  # (n_samples, 4)
    preds = np.zeros((len(test_df), len(target_cols)), dtype=np.float32)

    for j, col in enumerate(target_cols):
        y = Y[:, j].astype(int)

        if len(np.unique(y)) < 2:
            preds[:, j] = float(priors[j])
            continue

        clf = LogisticRegression(
            max_iter=1500,
            solver="lbfgs",
            n_jobs=1,
            random_state=seed,
            C=2.0,
            class_weight="balanced",
        )
        clf.fit(X_train_fin, y)

        proba = clf.predict_proba(X_test_fin)[:, 1].astype(np.float32)
        preds[:, j] = proba

        preds[~test_has_feat, j] = float(priors[j])

    return preds




## === cell 6
if len(submissions_all) == 0:
    if not os.path.isfile(TRAIN_PATH):
        raise FileNotFoundError(f"Could not find train.csv at {TRAIN_PATH}")
    if not os.path.isfile(TEST_PATH):
        raise FileNotFoundError(f"Could not find test.csv at {TEST_PATH}")
    if not os.path.isfile(SAMPLE_SUB_PATH):
        raise FileNotFoundError(
            f"Could not find sample_submission.csv at {SAMPLE_SUB_PATH}"
        )
    if not os.path.isdir(IMAGES_DIR):
        raise FileNotFoundError(f"Could not find images directory at {IMAGES_DIR}")

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    for c in ["image_id"] + TARGET_COLS:
        if c not in train.columns:
            raise ValueError(f"train.csv missing expected column: {c}")
    if "image_id" not in test.columns:
        raise ValueError("test.csv missing expected column: image_id")

    submission_avg = train_predict_lr_on_pixels(
        train_df=train,
        test_df=test,
        image_dir=IMAGES_DIR,
        target_cols=TARGET_COLS,
        size=(96, 96),
        seed=42,
        n_threads=8,
        pca_components=768,
    )
    make_submission_file(submission_avg)
else:
    submission_avg = ensemble(submissions_all, [0, 2, 4], [0.15, 0.8, 0.05])
    make_submission_file(submission_avg)

print("Wrote submission.csv")
print(pd.read_csv("submission.csv").head())
