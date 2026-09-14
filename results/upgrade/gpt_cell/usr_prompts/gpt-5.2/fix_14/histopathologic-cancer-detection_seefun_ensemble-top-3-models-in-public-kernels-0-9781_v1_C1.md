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
Given a dataset of images from digital pathology scans, predict if the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format:

```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.
```

## Dataset
Files are named with an image `id`. The `train_labels.csv` file provides the ground truth for the images in the `train` folder. You are predicting the labels for the images in the `test` folder.

# 2. Python version

3.7

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
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
        input/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
        working/
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
```

-> data/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> (stopped after 10 files for performance)

# 5. Target score

0.9799

# 6. Current score

0.86265

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Diagnosis: Cell 2 crashes because it tries to read three external Kaggle dataset paths under `../input/...` that are not present in this environment (only `../input/histopathologic-cancer-detection/...` and similar are available). The notebook expects `sub1/sub2/sub3` DataFrames with `id` and `label` columns, but the referenced submission files don’t exist here.  
Patch summary: In cell 2, add a tiny “safe read” helper that falls back to the provided `../input/histopathologic-cancer-detection/sample_submission.csv` (or `../input/sample_submission.csv`) when a blend source file is missing, preserving the `id,label` schema so downstream code continues to run.  
Updated cells: Only cell 2 is modified.  
Compatibility notes for cell k+1: `sub1`, `sub2`, and `sub3` always be valid pandas DataFrames with the expected columns; thus `sub1.head()` in cell 3 work unchanged.  
Assumptions: Later cells rely on the standard Kaggle submission format (`id`, `label`) and can tolerate using the sample submission as a placeholder when the intended blend inputs are unavailable.'
- What this solution (achieved 0.76381) has done: 'The timeout is dominated by the fallback path that reads tens of thousands of `.tif` files in pure-Python loops (PIL open/convert + numpy conversion) to build features for the test set (45k images). To keep the same core logic (central 32×32 mean RGB → logistic regression → predict_proba) but make it fast, I cache per-image features to disk and parallelize feature extraction across CPU cores using a process pool (purely equivalent computation, just concurrent). I also avoid repeated DataFrame index lookups and reduce overhead in the inner loop by computing the crop mean without reshaping. If the external submissions exist, behavior is unchanged; if they don’t, the fallback now finishes within the time budget and reuse cached features across runs.'
- What this solution (achieved 0.76451) has done: 'Your current score is far below the target, so we should improve model quality without changing the core approach (central 32×32 mean RGB → logistic regression → predict_proba). The smallest high-impact change is to use a much larger (but still feasible) training subset, and to set `class_weight="balanced"` to better handle label imbalance, which generally improves ROC-AUC for this competition while keeping identical semantics. I also make the feature cache key include `max_items` so you don’t accidentally reuse an older smaller training cache when you increase `max_train`. Finally, I keep the blend logic untouched; if the external submissions don’t exist, all three fallbacks now be stronger, moving the ensemble toward the target.'
- What this solution (achieved 0.76439) has done: 'We keep your core pipeline (central 32×32 mean RGB → logistic regression → averaged ensemble) unchanged, but fix a silent score-killer: when external blend files are missing, each fallback model is currently trained on a different random subset/order, so the three “models” become near-identical and add noise rather than useful diversity. I make the fallback training deterministic and slightly more stable by using a stratified sample (same class balance each run) and standardizing the 3 RGB features before logistic regression (still the same model family/semantics, just better-conditioned), which should lift AUC toward your target without changing architecture/loops. I also ensure all three fallbacks share the exact same `id` ordering as the sample submission before averaging, preventing any accidental misalignment. These are minimal, local changes in the fallback path only; if the external submissions exist, behavior stays as before.'
- What this solution (achieved 0.76472) has done: 'Your score is far below the target, so we should gently improve the fallback model quality while keeping the exact same core pipeline (central 32×32 mean RGB → StandardScaler → LogisticRegression → predict_proba → average). The biggest low-risk gain is to (1) train on more images (still within time due to cached/parallel features) and (2) add a tiny amount of diversity between the three fallback models by varying the crop size slightly (still “center crop mean RGB” features, same model family/semantics) so the averaging helps instead of being redundant. I also fix a subtle multiprocessing bug where `enumerate(pool.imap(...))` was using the enumeration index, not the returned id, which can silently corrupt feature ordering and hurt AUC; this is directly score-relevant and preserves the same computation. Finally, the submission writing remains identical (`ensemble.csv`, columns `id,label`) and ID alignment checks stay in place.'
- What this solution (achieved 0.76929) has done: 'We keep your core pipeline intact (center-crop mean RGB → StandardScaler → LogisticRegression → predict_proba → average) and only make score-relevant, minimal adjustments. The main lift come from using a more informative feature while preserving the same semantics: add per-channel standard deviation of the center crop (still “simple statistics of the center region”), which typically improves AUC substantially for this competition without changing the model family or training loop. We also ensure the multiprocessing feature builder preserves exact `id`→row alignment by using the returned `_id` to place features (a subtle ordering bug can silently hurt AUC). Finally, we keep your 3-model diversity (different crop sizes/seeds) and write the same `ensemble.csv` submission format.'
- What this solution (achieved 0.76929) has done: 'We’re far below the target AUC, so the smallest score-relevant improvement without changing your core pipeline is to slightly strengthen the logistic regression fallback model while keeping the exact same feature family (center-crop simple stats) and training loop. I (1) add `C=2.0` (less regularization; often helps when using 6 simple features) and (2) use `max_iter=800` for more reliable convergence; both preserve the same model/semantics. I also make sure the test feature cache tag includes the crop size (it already does) and add a guard to always output predictions in the sample submission order (already true) while keeping averaging logic unchanged. These are minimal, low-risk tweaks expected to lift AUC toward your target without altering the overall approach.'
- What this solution (achieved 0.78356) has done: 'We’re far below the target AUC, so we should improve the fallback model’s ranking quality while keeping the same core pipeline (center-crop simple stats → StandardScaler → LogisticRegression → predict_proba → average). The smallest high-impact change is to add a few more center-crop statistics (min/max per channel) while preserving the same feature-extraction semantics and model family, which typically improves ROC-AUC materially for this competition. To keep runtime within 600s, we train each fallback model on a smaller (but still strong) stratified subset while keeping cached/parallel feature extraction and the existing 3-crop diversity. Finally, we fix cache versioning and ensure consistent ID alignment/order so averaging can’t silently misalign predictions.'
- What this solution (achieved 0.78316) has done: 'Your current score (0.78356) is far below the target (0.9799), so the goal is to legitimately improve ranking quality with minimal, score-relevant changes while keeping the same pipeline (center-crop simple stats → StandardScaler → LogisticRegression → predict_proba → average). The smallest high-impact fix is to greatly expand the training set size for the fallback models (your feature extraction is already cached/parallel), since 30k samples is the main limiter on AUC here. I also slightly increase logistic regression capacity (higher `C`) to better fit the richer 12-dim stats, without changing the model family or training semantics. Finally, I ensure the three fallback models are trained on different stratified subsets (still deterministic) so averaging actually adds useful diversity rather than near-duplicates.'
- What this solution (achieved 0.78965) has done: 'Your current score (0.78316) is far below the target (0.9799), so we should improve the fallback model’s ranking quality while keeping the same core pipeline: center-crop simple stats → StandardScaler → LogisticRegression → averaged ensemble. The smallest high-impact fix is to add one more cheap, center-region-only statistic feature (per-channel median) which improves robustness to stain/lighting outliers without changing the model family or training semantics. I also increase the fallback training set size moderately (features are cached/parallelized already) so the logistic model sees more variability, and I slightly tune regularization (C) to fit the expanded feature vector. All changes are confined to the fallback path; external blend files (if present) are still used as-is, and the script still writes a valid `ensemble.csv` with `id,label`.'
- What this solution (achieved 0.86222) has done: 'Your AUC is far below the target, so we should improve ranking quality while preserving the exact same pipeline (center-crop simple stats → StandardScaler → LogisticRegression → averaged ensemble). The smallest high-impact change here is to make the fallback model non-linear in a minimal way by augmenting the same center-crop statistics with simple pairwise interaction terms (no new data, still center-only, still logistic regression), which typically boosts separability a lot for this task. To support the larger feature vector without slowing inference, we keep caching/parallel extraction intact and only slightly increase `max_iter` for stable convergence. Everything else (ID alignment, averaging, submission writing) stays unchanged.'
- What this solution (achieved 0.86265) has done: 'We’re well below the target AUC, so the best “minimal-change” path is to improve the fallback model ranking quality without changing the overall pipeline (center-crop stats → StandardScaler → LogisticRegression → averaged ensemble). The main score-limiter is training on too few examples due to feature-extraction cost; we switch to a fast exact computation of the same statistics using `PIL.ImageStat` directly on the center crop (no full image→numpy conversion), letting us safely raise `MAX_TRAIN_FALLBACK` and train each of the 3 fallback models on more data within the 600s budget. We keep the same 15 base stats + pairwise interaction terms (same semantics), same logistic regression, and same averaging and submission format. This should move AUC upward toward your target while keeping changes tightly scoped to feature extraction speed + using a larger training set.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import scipy.special

sigmoid = lambda x: scipy.special.expit(x)



## === cell 1
from PIL import Image, ImageStat
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import multiprocessing as mp

INPUT_ROOTS = [
    "../input/histopathologic-cancer-detection",
    "../input",
    "../kaggle/input/histopathologic-cancer-detection",
    "../kaggle/input",
]
WORKING_ROOTS = [
    "../working/histopathologic-cancer-detection",
    "../kaggle/working/histopathologic-cancer-detection",
]


def _find_existing_file(relpath_candidates):
    for p in relpath_candidates:
        if os.path.exists(p):
            return p
    return None


def _find_base_dir():
    for base in INPUT_ROOTS + WORKING_ROOTS:
        if os.path.exists(base) and os.path.isdir(base):
            ss = os.path.join(base, "sample_submission.csv")
            tl = os.path.join(base, "train_labels.csv")
            tr = os.path.join(base, "train")
            te = os.path.join(base, "test")
            if (
                os.path.exists(ss)
                and os.path.exists(tl)
                and os.path.isdir(tr)
                and os.path.isdir(te)
            ):
                return base
    for base in INPUT_ROOTS + WORKING_ROOTS:
        nested = os.path.join(base, "histopathologic-cancer-detection")
        ss = os.path.join(nested, "sample_submission.csv")
        tl = os.path.join(nested, "train_labels.csv")
        tr = os.path.join(nested, "train")
        te = os.path.join(nested, "test")
        if (
            os.path.exists(ss)
            and os.path.exists(tl)
            and os.path.isdir(tr)
            and os.path.isdir(te)
        ):
            return nested
    raise FileNotFoundError(
        "Could not locate histopathologic-cancer-detection dataset directory with train/test and CSVs."
    )


BASE_DIR = _find_base_dir()
SAMPLE_SUB_PATH = os.path.join(BASE_DIR, "sample_submission.csv")
TRAIN_LABELS_PATH = os.path.join(BASE_DIR, "train_labels.csv")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")


def _central_crop_box(img_w, img_h, crop=32):
    x0 = img_w // 2 - crop // 2
    y0 = img_h // 2 - crop // 2
    return (x0, y0, x0 + crop, y0 + crop)


def _central_crop_stats_rgb_pil(path, crop=32):
    """
    Change (score-relevant): compute the SAME stats as before (mean/std/min/max/median per channel)
    but avoid converting full image to numpy, using PIL/ImageStat on the center crop only.
    This makes feature extraction much faster, enabling a larger training set within timeout.
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        box = _central_crop_box(im.size[0], im.size[1], crop=crop)
        patch = im.crop(box)

        st = ImageStat.Stat(patch)  # per-channel statistics
        mean = np.asarray(st.mean, dtype=np.float32)  # (3,)
        std = np.asarray(st.stddev, dtype=np.float32)  # (3,)
        mn = np.asarray(st.extrema, dtype=np.float32)[:, 0]  # (3,) mins
        mx = np.asarray(st.extrema, dtype=np.float32)[:, 1]  # (3,) maxs

        try:
            med = np.asarray(st.median, dtype=np.float32)  # (3,)
        except Exception:
            arr = np.asarray(patch, dtype=np.uint8)
            med = np.median(arr, axis=(0, 1)).astype(np.float32)

        return np.concatenate([mean, std, mn, mx, med], axis=0)  # 15 dims


def _augment_with_pairwise_products(x15):
    x15 = np.asarray(x15, dtype=np.float32)
    prods = []
    for i in range(15):
        xi = x15[i]
        for j in range(i + 1, 15):
            prods.append(xi * x15[j])
    return np.concatenate([x15, np.asarray(prods, dtype=np.float32)], axis=0)


FEATURE_BASE_DIM = 15
FEATURE_DIM = 120  # 15 + 105 pairwise products
FEATURE_VERSION = "v4_meanstdminmaxmedian_pairwise_pilfast"


def _cache_path_for(folder, n, crop=32, tag=""):
    base = os.path.basename(folder.rstrip("/"))
    cache_dir = os.path.join(".", ".feature_cache")
    os.makedirs(cache_dir, exist_ok=True)
    tag_part = f"_{tag}" if tag else ""
    return os.path.join(
        cache_dir,
        f"{base}{tag_part}_n{n}_crop{crop}_{FEATURE_VERSION}_d{FEATURE_DIM}_float32.npz",
    )


def _feat_one(args):
    _id, folder, crop = args
    fp = os.path.join(folder, f"{_id}.tif")
    if not os.path.exists(fp):
        return _id, np.zeros(FEATURE_DIM, dtype=np.float32)
    base_feat = _central_crop_stats_rgb_pil(fp, crop=crop).astype(
        np.float32, copy=False
    )
    feat = _augment_with_pairwise_products(base_feat).astype(np.float32, copy=False)
    return _id, feat


def _build_features(ids, folder, max_items=None, crop=32, use_cache=True, cache_tag=""):
    n = len(ids) if max_items is None else min(len(ids), max_items)
    ids_use = list(ids[:n])

    if use_cache:
        cache_path = _cache_path_for(folder, n, crop=crop, tag=cache_tag)
        if os.path.exists(cache_path):
            try:
                z = np.load(cache_path, allow_pickle=False)
                X = z["X"]
                used_ids = z["ids"].astype(str).tolist()
                if X.shape == (n, FEATURE_DIM) and len(used_ids) == n:
                    return X, used_ids
            except Exception:
                pass  # fall back to rebuild

    X = np.zeros((n, FEATURE_DIM), dtype=np.float32)

    cpu = os.cpu_count() or 2
    workers = max(1, min(8, cpu))
    chunksize = 256

    if workers == 1:
        for i, _id in enumerate(ids_use):
            _, feat = _feat_one((_id, folder, crop))
            X[i] = feat
    else:
        id_to_i = {k: i for i, k in enumerate(ids_use)}
        ctx = mp.get_context("fork" if hasattr(os, "fork") else "spawn")
        with ctx.Pool(processes=workers) as pool:
            for _id_ret, feat in pool.imap(
                _feat_one,
                ((_id, folder, crop) for _id in ids_use),
                chunksize=chunksize,
            ):
                i = id_to_i.get(_id_ret, None)
                if i is not None:
                    X[i] = feat

    if use_cache:
        cache_path = _cache_path_for(folder, n, crop=crop, tag=cache_tag)
        try:
            np.savez_compressed(cache_path, X=X, ids=np.array(ids_use, dtype=object))
        except Exception:
            pass

    return X, ids_use


def _make_stratified_train_ids(train_df, max_train, seed):
    pos = train_df.loc[train_df["label"].values == 1, "id"].values
    neg = train_df.loc[train_df["label"].values == 0, "id"].values

    rng = np.random.RandomState(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)

    p = len(pos) / (len(pos) + len(neg))
    n_pos = int(round(max_train * p))
    n_pos = max(1, min(n_pos, len(pos)))
    n_neg = max_train - n_pos
    n_neg = max(1, min(n_neg, len(neg)))

    ids = np.concatenate([pos[:n_pos], neg[:n_neg]])
    rng.shuffle(ids)
    return ids.tolist()


def _train_and_predict_submission(
    seed=42, max_train=30000, crop=32, cache_tag_extra=""
):
    train_df = pd.read_csv(TRAIN_LABELS_PATH)
    sample_df = pd.read_csv(SAMPLE_SUB_PATH)

    train_ids = _make_stratified_train_ids(train_df, max_train=max_train, seed=seed)

    X_train, used_train_ids = _build_features(
        train_ids,
        TRAIN_DIR,
        max_items=max_train,
        crop=crop,
        use_cache=True,
        cache_tag=f"train_{FEATURE_VERSION}_seed{seed}_max{max_train}_crop{crop}{cache_tag_extra}",
    )

    y_map = dict(zip(train_df["id"].values, train_df["label"].astype(int).values))
    y_train = np.fromiter(
        (y_map[_id] for _id in used_train_ids),
        dtype=np.int64,
        count=len(used_train_ids),
    )

    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(
            solver="lbfgs",
            max_iter=1400,
            n_jobs=1,
            random_state=seed,
            class_weight="balanced",
            C=6.0,
        ),
    )
    clf.fit(X_train, y_train)

    test_ids = sample_df["id"].tolist()
    X_test, _ = _build_features(
        test_ids,
        TEST_DIR,
        max_items=None,
        crop=crop,
        use_cache=True,
        cache_tag=f"test_all_{FEATURE_VERSION}_crop{crop}",
    )
    preds = clf.predict_proba(X_test)[:, 1].astype(np.float64)

    out = pd.DataFrame({"id": test_ids, "label": preds})
    return out


def _safe_read_submission_or_make(path, seed=42, crop=32, max_train=30000):
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "id" in df.columns and "label" in df.columns and len(df) > 0:
            return df[["id", "label"]]
    return _train_and_predict_submission(
        seed=seed, max_train=max_train, crop=crop, cache_tag_extra=f"_fallback{seed}"
    )


MAX_TRAIN_FALLBACK = (
    174000  # near full dataset (174,464), but stratified sample will cap to available
)

sub1 = _safe_read_submission_or_make(
    "../input/blending-power-0-9754/sub.csv",
    seed=11,
    crop=32,
    max_train=MAX_TRAIN_FALLBACK,
)
sub2 = _safe_read_submission_or_make(
    "../input/tta-power-densenet169/submission_tta_64.csv",
    seed=22,
    crop=28,
    max_train=MAX_TRAIN_FALLBACK,
)
sub3 = _safe_read_submission_or_make(
    "../input/you-really-need-attention-pytorch/sub_tta.csv",
    seed=33,
    crop=36,
    max_train=MAX_TRAIN_FALLBACK,
)



## === cell 2
sub1.head()



## === cell 3
sub2.head()



## === cell 4
sub3.head()



## === cell 5
sub1 = sub1.copy()
sub2 = sub2.copy()
sub3 = sub3.copy()

sub1 = sub1.sort_values("id").reset_index(drop=True)
sub2 = sub2.sort_values("id").reset_index(drop=True)
sub3 = sub3.sort_values("id").reset_index(drop=True)

if (
    not (sub1["id"].values == sub2["id"].values).all()
    or not (sub1["id"].values == sub3["id"].values).all()
):
    raise ValueError(
        "Submission IDs are not aligned across sub1/sub2/sub3 after sorting; cannot safely average."
    )

sub1["label"] = (
    sub1["label"].values + sub2["label"].values + sub3["label"].values
) / 3.0



## === cell 6
sub1["label"] = sub1["label"].clip(0.0, 1.0)
sub1.to_csv("ensemble.csv", index=False)
print("Wrote ensemble.csv with shape:", sub1.shape)
print(sub1.head())
