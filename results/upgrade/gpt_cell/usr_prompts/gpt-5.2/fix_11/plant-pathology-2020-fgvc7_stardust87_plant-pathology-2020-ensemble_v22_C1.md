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

0.9699

# 6. Current score

0.6983

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Diagnosis: `submissions_all` is empty because `SUBMISSIONS_PATH` points to `/kaggle/input/submissions/`, which does not exist in this environment (the available submissions/sample files are under `/kaggle/data/...`). As a result, `ensemble(submissions_all,[0,1,2], ...)` tries to index into an empty list and crashes with `IndexError`. The minimal fix is to update cell 5 to gracefully handle missing/insufficient submission files by falling back to the provided `sample_submission.csv` (which matches the expected schema) and using only the available indices. This preserves the ensemble and output-file creation logic without changing model/metric semantics.

Patch summary: In cell 5 only, detect if `submissions_all` has fewer than 3 files; if so, set `submissions_all` to a list containing the known `sample_submission.csv` path and adjust `sub_idx/weights` to valid lengths before calling `ensemble`.

Updated cells:'
- What this solution (achieved 0.49301) has done: 'Your current 0.5 score comes from submitting essentially constant/placeholder probabilities (the fallback to `sample_submission.csv` has all 0.25s), so ROC AUC stays near random. To move toward the 0.9699 target with minimal logic change, I keep your ensemble structure but add a deterministic “fallback model” that creates non-constant predictions from the training labels’ class priors, so the submission is valid and non-degenerate. I also make the code robust to missing `/kaggle/input/submissions/` by searching known data locations and ensuring weights are renormalized to sum to 1 for proper averaging (same ensemble semantics, just correct scaling). This should increase score from 0.5 toward the target without changing the overall approach (still producing a weighted average submission).'
- What this solution (achieved 0.52875) has done: 'Your current score is low because the “prior fallback” produces almost-constant predictions (just tiny noise around class priors), which yields near-random ROC AUC. To move toward the 0.9699 target while keeping the ensemble core logic intact, I keep your ensemble/make-submission flow but replace the fallback predictions with a lightweight, label-derived heuristic: infer each test image’s label by matching its `image_id` to the closest train `image_id` in pixel-space (1-NN on downsampled grayscale), then output a smoothed one-hot probability vector. This stays within the same evaluation semantics (still probabilities per class), avoids changing any model/training loop (there is none), and should substantially improve AUC versus priors-only. I also make path resolution robust to your provided `/kaggle/data/...` locations and ensure the output submission rows align exactly with `test.csv`.'
- What this solution (achieved 0.5381) has done: 'Your score is far below the 0.9699 target, so we should improve predictions while keeping your “fallback heuristic + ensemble” core logic intact. The biggest issue is that your 1-NN distance computation recomputes `train_norms` inside the test loop and uses a fairly weak grayscale 32×32 feature; both can be improved without changing the overall approach. I (1) precompute `train_norms` once, (2) use a slightly higher-resolution feature (48×48) and RGB (still just a flattened downsample), and (3) use a small k-NN (k=5) soft voting with smoothing instead of strict 1-NN, which typically boosts AUC substantially while preserving the same heuristic idea. I also make the image-path resolution a bit more robust and keep submission row alignment strictly following `test.csv`.'
- What this solution (achieved 0.60975) has done: 'Your current score (0.5381) is far below the target (0.9699), so we should improve the fallback heuristic’s predictive signal while keeping the same overall “build a fallback submission then (optionally) ensemble CSVs” core logic. The smallest impactful change is to strengthen the k-NN image matching by using (1) multi-scale RGB features (48×48 and 96×96) concatenated, and (2) cosine distance (L2-normalized features) which is typically more robust than raw L2 for lighting/background variation. I also make the k-NN vote slightly sharper by using a softmax over negative distances (temperature) instead of 1/d, while keeping the same smoothing and probability formatting. Everything else (paths, ensembling semantics, submission columns/row order) is preserved so it still runs end-to-end and writes `submission.csv`.'
- What this solution (achieved 0.63457) has done: 'I keep your existing “fallback heuristic + (optional) CSV ensemble” approach, but strengthen the fallback’s image-feature signal while keeping the same basic k-NN/soft-voting semantics. The minimal impactful change is to use a lightweight color-aware feature that is more invariant to illumination: concatenate your existing multiscale RGB downsample with a simple per-channel color histogram, and normalize each part before concatenation. This should improve nearest-neighbor retrieval quality and therefore ROC AUC, moving your score upward toward 0.9699 without changing the overall pipeline structure. I also ensure the submission is aligned to `test.csv` order (already mostly true) and keep runtime under the limit by using small histograms and efficient numpy ops.'
- What this solution (achieved 0.63372) has done: 'I keep your current “fallback heuristic + (optional) CSV ensemble” structure, but strengthen the fallback’s image representation in a minimal way to improve the k-NN retrieval signal and thus ROC AUC toward the 0.9699 target. Specifically, I add a small texture feature (uniform LBP histogram on a downsampled grayscale image) and concatenate it to your existing multiscale RGB + color histogram feature, with per-part L2 normalization preserved. This keeps the same k-NN/softmax-voting and probability smoothing semantics, only improving the underlying similarity features. I also make the feature dimension computed from actual components (to avoid mismatch if you tweak bins/sizes) and keep the submission aligned to `test.csv` as you already do.'
- What this solution (achieved 0.68026) has done: 'Your current score (0.63372) is far below the target (0.9699), so we should increase predictive signal while keeping your same “fallback heuristic + optional CSV ensemble” core logic. The smallest impactful change is to make the k-NN voting use both cosine-similarity and a mild RBF on raw L2 distance (computed on the same existing features) to reduce cases where cosine alone picks visually similar but label-mismatched neighbors. I also add a simple per-image “strength” gating that slightly increases smoothing (eps) when the neighbor weights are diffuse (uncertain), which typically improves AUC without changing the overall semantics. Everything else (paths, feature extraction components, k-NN soft voting structure, and submission formatting/alignment) is preserved.'
- What this solution (achieved 0.68064) has done: 'We keep your exact pipeline (fallback heuristic + optional CSV ensemble) but make two minimal adjustments that should increase AUC toward the 0.9699 target by improving neighbor retrieval signal and probability calibration. First, we add a very small “center crop” variant to the existing multiscale RGB feature (no new model/training; same k-NN voting), which often helps because leaf occupies the center and reduces background noise. Second, we slightly tune the voting/calibration in a controlled way: use a small blend of the global label prior (instead of uniform) for smoothing under uncertainty, which typically improves mean ROC AUC without changing core semantics. Everything still runs end-to-end and writes a valid `submission.csv` with correct column order aligned to `test.csv`.'
- What this solution (achieved 0.6983) has done: 'Your current score (0.68064) is far below the target (0.9699), so we should increase predictive signal while keeping your same “fallback heuristic submission + optional CSV ensemble” structure intact. The smallest likely win is to make the neighbor search less noisy: (1) add simple test-time augmentation (original + horizontal flip) but still using the same k-NN soft-voting mechanism, and (2) reduce the impact of very-close/duplicate neighbors by using a slightly smaller k with the same weighting scheme. These are minimal changes that preserve your core logic (no model training, still feature-extraction → kNN → soft vote → submission) while typically improving mean ROC AUC on this dataset. Everything still runs end-to-end and writes a valid `submission.csv` aligned to `test.csv`.'

# 9. Code solution

## === cell 0
import os
import pandas as pd



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/"



## === cell 2
submissions_all = []
for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
    for filename in filenames:
        if filename.lower().endswith(".csv"):
            submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print(submissions_all)




## === cell 3
def ensemble(submissions_all, sub_idx, weights=[]):
    submission_with_weight = []
    for i in range(len(sub_idx)):
        print(
            f"I'm taking submission {submissions_all[sub_idx[i]]} with weight {weights[i]}"
        )
        submission = pd.read_csv(submissions_all[sub_idx[i]])
        submission = submission.loc[
            :, ["healthy", "multiple_diseases", "rust", "scab"]
        ].values
        submission_with_weight.append(submission * weights[i])
    submission_avg = sum(submission_with_weight)
    return submission_avg




## === cell 4
def make_submission_file(submission_avg, submissions_all):
    submission_df = pd.read_csv(submissions_all[0])
    submission_df.iloc[:, 1:] = 0
    submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = submission_avg
    submission_df.to_csv("submission.csv", index=False)




## === cell 5
def build_prior_fallback_submission(
    train_csv_path="/kaggle/data/train.csv",
    test_csv_path="/kaggle/data/test.csv",
    sample_sub_path="/kaggle/data/sample_submission.csv",
    out_path="prior_fallback_submission.csv",
):
    import numpy as np
    from PIL import Image

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    _ = pd.read_csv(sample_sub_path)

    target_cols = ["healthy", "multiple_diseases", "rust", "scab"]

    prior = train_df[target_cols].mean(axis=0).values.astype(np.float32)
    prior = prior / (float(prior.sum()) + 1e-12)

    def _find_images_dir(base_csv_path):
        base_dir = os.path.dirname(os.path.abspath(base_csv_path))
        candidates = [
            os.path.join(base_dir, "images"),
            os.path.join(base_dir, "plant-pathology-2020-fgvc7", "images"),
            "/kaggle/data/images",
            "/kaggle/data/plant-pathology-2020-fgvc7/images",
            "/kaggle/input/plant-pathology-2020-fgvc7/images",
            "/kaggle/data/plant-pathology-2020-fgvc7/plant-pathology-2020-fgvc7/images",
            "/kaggle/input/plant-pathology-2020-fgvc7/plant-pathology-2020-fgvc7/images",
        ]
        for c in candidates:
            if os.path.isdir(c):
                return c
        return None

    images_dir = _find_images_dir(train_csv_path)
    if images_dir is None:
        raise FileNotFoundError(
            "Could not locate images directory for fallback heuristic."
        )

    def _resolve_img_path(img_id: str):
        p1 = os.path.join(images_dir, img_id)
        p2 = os.path.join(images_dir, f"{img_id}.jpg")
        if os.path.exists(p1):
            return p1
        if os.path.exists(p2):
            return p2
        return p2

    def _l2_normalize(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n > 0:
            return (v / n).astype(np.float32)
        return v.astype(np.float32)

    def _center_crop_pil(im: Image.Image, crop_frac: float = 0.80) -> Image.Image:
        w, h = im.size
        cw = int(round(w * crop_frac))
        ch = int(round(h * crop_frac))
        left = max(0, (w - cw) // 2)
        top = max(0, (h - ch) // 2)
        return im.crop((left, top, left + cw, top + ch))

    def _img_feature(img_path, size, crop=False, hflip=False):
        im = Image.open(img_path).convert("RGB")
        if hflip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        if crop:
            im = _center_crop_pil(im, crop_frac=0.80)
        im = im.resize(size, Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0  # (H,W,3)
        return arr.reshape(-1).astype(np.float32)

    def _color_hist_feature(img_path, size=(128, 128), bins=16, hflip=False):
        im = Image.open(img_path).convert("RGB")
        if hflip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        im = im.resize(size, Image.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)  # (H,W,3) in [0,255]
        edges = np.linspace(0, 256, bins + 1, dtype=np.int32)
        hists = []
        for c in range(3):
            h, _ = np.histogram(arr[:, :, c].ravel(), bins=edges)
            hists.append(h.astype(np.float32))
        h = np.concatenate(hists, axis=0)
        h = h / (float(h.sum()) + 1e-12)
        return h.astype(np.float32)

    def _lbp_hist_feature(img_path, size=(96, 96), hflip=False):
        im = Image.open(img_path).convert("L")
        if hflip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        im = im.resize(size, Image.BILINEAR)
        g = np.asarray(im, dtype=np.uint8)
        c = g[1:-1, 1:-1].astype(np.int16)
        n0 = g[0:-2, 0:-2].astype(np.int16)
        n1 = g[0:-2, 1:-1].astype(np.int16)
        n2 = g[0:-2, 2:].astype(np.int16)
        n3 = g[1:-1, 2:].astype(np.int16)
        n4 = g[2:, 2:].astype(np.int16)
        n5 = g[2:, 1:-1].astype(np.int16)
        n6 = g[2:, 0:-2].astype(np.int16)
        n7 = g[1:-1, 0:-2].astype(np.int16)

        lbp = (
            ((n0 >= c) << 7)
            | ((n1 >= c) << 6)
            | ((n2 >= c) << 5)
            | ((n3 >= c) << 4)
            | ((n4 >= c) << 3)
            | ((n5 >= c) << 2)
            | ((n6 >= c) << 1)
            | ((n7 >= c) << 0)
        ).astype(np.uint8)

        hist = np.bincount(lbp.ravel(), minlength=256).astype(np.float32)
        hist = hist / (float(hist.sum()) + 1e-12)
        return hist.astype(np.float32)

    def _img_feature_multiscale(img_path, hflip=False):
        f1 = _img_feature(img_path, (48, 48), crop=False, hflip=hflip)
        f2 = _img_feature(img_path, (96, 96), crop=False, hflip=hflip)
        fc = _img_feature(img_path, (64, 64), crop=True, hflip=hflip)

        f_rgb = np.concatenate([f1, f2, fc], axis=0).astype(np.float32)
        f_rgb = _l2_normalize(f_rgb)

        f_hist = _color_hist_feature(img_path, size=(128, 128), bins=16, hflip=hflip)
        f_hist = _l2_normalize(f_hist)

        f_lbp = _lbp_hist_feature(img_path, size=(96, 96), hflip=hflip)
        f_lbp = _l2_normalize(f_lbp)

        f = np.concatenate([f_rgb, f_hist, f_lbp], axis=0).astype(np.float32)
        f = _l2_normalize(f)
        return f

    train_ids = train_df["image_id"].astype(str).tolist()

    _tmp_feat = _img_feature_multiscale(_resolve_img_path(train_ids[0]), hflip=False)
    feat_dim = int(_tmp_feat.shape[0])

    X_train = np.zeros((len(train_ids), feat_dim), dtype=np.float32)
    for i, img_id in enumerate(train_ids):
        img_path = _resolve_img_path(img_id)
        X_train[i] = _img_feature_multiscale(img_path, hflip=False)

    y_train = train_df[target_cols].values.astype(np.float32)

    test_ids = test_df["image_id"].astype(str).tolist()
    preds = np.zeros((len(test_ids), len(target_cols)), dtype=np.float32)

    k = 9

    tau_cos = 0.045
    sigma_l2 = 0.65
    eps_base = 0.02
    eps_max = 0.06

    def _knn_predict_for_feature(x: np.ndarray) -> np.ndarray:
        sims = X_train @ x
        d_cos = 1.0 - sims

        kk = min(k, d_cos.shape[0])
        nn_idx = np.argpartition(d_cos, kk - 1)[:kk]

        diff = X_train[nn_idx] - x[None, :]
        d_l2 = np.sqrt((diff * diff).sum(axis=1) + 1e-12).astype(np.float32)

        z = -d_cos[nn_idx].astype(np.float32) / tau_cos
        z = z - float(np.max(z))
        w_cos = np.exp(z).astype(np.float32)

        w_l2 = np.exp(-(d_l2 * d_l2) / (2.0 * (sigma_l2 * sigma_l2))).astype(np.float32)

        w = (w_cos * w_l2).astype(np.float32)
        w = w / (float(w.sum()) + 1e-12)

        p = (y_train[nn_idx] * w[:, None]).sum(axis=0)

        s = float(p.sum())
        if s <= 0:
            p = prior
        else:
            p = p / s

        neff = 1.0 / float((w * w).sum() + 1e-12)
        conf = (neff - 1.0) / max(1.0, (kk - 1.0))  # 0 confident, 1 diffuse
        eps = float(eps_base + (eps_max - eps_base) * conf)

        p = (1.0 - eps) * p + eps * prior
        p = np.clip(p, 1e-4, 1.0 - 1e-4)
        return p.astype(np.float32)

    for j, img_id in enumerate(test_ids):
        img_path = _resolve_img_path(img_id)

        x0 = _img_feature_multiscale(img_path, hflip=False)  # L2-normalized
        p0 = _knn_predict_for_feature(x0)

        x1 = _img_feature_multiscale(img_path, hflip=True)  # L2-normalized
        p1 = _knn_predict_for_feature(x1)

        p = 0.5 * (p0 + p1)
        p = np.clip(p, 1e-4, 1.0 - 1e-4)
        preds[j] = p

    sub = test_df[["image_id"]].copy()
    sub[target_cols] = preds
    sub.to_csv(out_path, index=False)
    return out_path




## === cell 6
if len(submissions_all) == 0:
    candidates = [
        "/kaggle/data/sample_submission.csv",
        "/kaggle/data/plant-pathology-2020-fgvc7/sample_submission.csv",
        "/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv",
    ]
    sample_path = None
    for p in candidates:
        if os.path.exists(p):
            sample_path = p
            break
    if sample_path is None:
        raise FileNotFoundError(
            "Could not find sample_submission.csv in expected Kaggle paths."
        )

    train_candidates = [
        "/kaggle/data/train.csv",
        "/kaggle/data/plant-pathology-2020-fgvc7/train.csv",
        "/kaggle/input/plant-pathology-2020-fgvc7/train.csv",
    ]
    test_candidates = [
        "/kaggle/data/test.csv",
        "/kaggle/data/plant-pathology-2020-fgvc7/test.csv",
        "/kaggle/input/plant-pathology-2020-fgvc7/test.csv",
    ]
    train_path = next((p for p in train_candidates if os.path.exists(p)), None)
    test_path = next((p for p in test_candidates if os.path.exists(p)), None)
    if train_path is None or test_path is None:
        raise FileNotFoundError(
            "Could not find train.csv/test.csv in expected Kaggle paths."
        )

    prior_sub_path = build_prior_fallback_submission(
        train_csv_path=train_path,
        test_csv_path=test_path,
        sample_sub_path=sample_path,
        out_path="prior_fallback_submission.csv",
    )
    submissions_all = [prior_sub_path]

sub_idx = [0, 1, 2]
weights = [0.15, 0.8, 0.05]

valid = [(i, w) for i, w in zip(sub_idx, weights) if i < len(submissions_all)]
sub_idx = [i for i, _ in valid]
weights = [w for _, w in valid]

if len(weights) == 0:
    raise ValueError("No valid submissions available to ensemble.")
wsum = sum(weights)
if wsum == 0:
    weights = [1.0 / len(weights)] * len(weights)
else:
    weights = [w / wsum for w in weights]

submission_avg = ensemble(submissions_all, sub_idx, weights)
make_submission_file(submission_avg, submissions_all)
print("Wrote submission.csv")
