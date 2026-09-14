# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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
    Compute center-crop stats via PIL/ImageStat.
    Feature semantics preserved: mean/std/min/max/median per channel on the center region.
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        box = _central_crop_box(im.size[0], im.size[1], crop=crop)
        patch = im.crop(box)

        st = ImageStat.Stat(patch)
        mean = np.asarray(st.mean, dtype=np.float32)
        std = np.asarray(st.stddev, dtype=np.float32)
        mn = np.asarray(st.extrema, dtype=np.float32)[:, 0]
        mx = np.asarray(st.extrema, dtype=np.float32)[:, 1]

        try:
            med = np.asarray(st.median, dtype=np.float32)
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
FEATURE_SINGLE_DIM = 120  # 15 + 105 pairwise products
FEATURE_VERSION = "v5_meanstdminmaxmedian_pairwise_pilfast_2scale"


def _cache_path_for(folder, n, crop=32, tag=""):
    base = os.path.basename(folder.rstrip("/"))
    cache_dir = os.path.join(".", ".feature_cache")
    os.makedirs(cache_dir, exist_ok=True)
    tag_part = f"_{tag}" if tag else ""
    return os.path.join(
        cache_dir,
        f"{base}{tag_part}_n{n}_crop{crop}_{FEATURE_VERSION}_float32.npz",
    )


def _feat_one(args):
    i, _id, folder, crops = args
    fp = os.path.join(folder, f"{_id}.tif")
    if not os.path.exists(fp):
        return i, _id, np.zeros(2 * FEATURE_SINGLE_DIM, dtype=np.float32)

    feats = []
    for c in crops:
        base_feat = _central_crop_stats_rgb_pil(fp, crop=c).astype(
            np.float32, copy=False
        )
        feats.append(
            _augment_with_pairwise_products(base_feat).astype(np.float32, copy=False)
        )
    feat = np.concatenate(feats, axis=0).astype(np.float32, copy=False)
    return i, _id, feat


def _build_features(ids, folder, max_items=None, crop=32, use_cache=True, cache_tag=""):
    """
    Change (runtime+correctness): keep exact same ordering by passing index through workers,
    and avoid building a large id->index dict for speed/memory on 174k items.
    """
    n = len(ids) if max_items is None else min(len(ids), max_items)
    ids_use = list(ids[:n])

    crops = (
        int(crop),
        int(max(12, crop - 4)),
    )  # tiny multi-scale; center-only semantics preserved
    feat_dim = 2 * FEATURE_SINGLE_DIM

    if use_cache:
        cache_path = _cache_path_for(folder, n, crop=crop, tag=cache_tag)
        if os.path.exists(cache_path):
            try:
                z = np.load(cache_path, allow_pickle=False)
                X = z["X"]
                used_ids = z["ids"].astype(str).tolist()
                if X.shape == (n, feat_dim) and len(used_ids) == n:
                    return X, used_ids
            except Exception:
                pass

    X = np.zeros((n, feat_dim), dtype=np.float32)

    cpu = os.cpu_count() or 2
    workers = max(1, min(8, cpu))
    chunksize = 256

    if workers == 1:
        for i, _id in enumerate(ids_use):
            _, _, feat = _feat_one((i, _id, folder, crops))
            X[i] = feat
    else:
        ctx = mp.get_context("fork" if hasattr(os, "fork") else "spawn")
        with ctx.Pool(processes=workers) as pool:
            for i_ret, _id_ret, feat in pool.imap_unordered(
                _feat_one,
                ((i, _id, folder, crops) for i, _id in enumerate(ids_use)),
                chunksize=chunksize,
            ):
                if 0 <= i_ret < n:
                    X[i_ret] = feat

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

    total = len(pos) + len(neg)
    if max_train is None or max_train >= total:
        ids = np.concatenate([pos, neg])
        rng.shuffle(ids)
        return ids.tolist()

    p = len(pos) / total
    n_pos = int(round(max_train * p))
    n_pos = max(0, min(n_pos, len(pos)))
    n_neg = max_train - n_pos
    n_neg = max(0, min(n_neg, len(neg)))

    if n_pos + n_neg != max_train:
        n_neg = max_train - n_pos
    if n_neg > len(neg):
        n_neg = len(neg)
        n_pos = max_train - n_neg
        n_pos = min(n_pos, len(pos))
    if n_pos > len(pos):
        n_pos = len(pos)
        n_neg = max_train - n_pos
        n_neg = min(n_neg, len(neg))

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


MAX_TRAIN_FALLBACK = 174464

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
