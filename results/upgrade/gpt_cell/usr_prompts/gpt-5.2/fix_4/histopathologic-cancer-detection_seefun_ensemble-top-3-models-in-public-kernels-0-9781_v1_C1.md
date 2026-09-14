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

0.76381

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Diagnosis: Cell 2 crashes because it tries to read three external Kaggle dataset paths under `../input/...` that are not present in this environment (only `../input/histopathologic-cancer-detection/...` and similar are available). The notebook expects `sub1/sub2/sub3` DataFrames with `id` and `label` columns, but the referenced submission files don’t exist here.  
Patch summary: In cell 2, add a tiny “safe read” helper that falls back to the provided `../input/histopathologic-cancer-detection/sample_submission.csv` (or `../input/sample_submission.csv`) when a blend source file is missing, preserving the `id,label` schema so downstream code continues to run.  
Updated cells: Only cell 2 is modified.  
Compatibility notes for cell k+1: `sub1`, `sub2`, and `sub3` always be valid pandas DataFrames with the expected columns; thus `sub1.head()` in cell 3 work unchanged.  
Assumptions: Later cells rely on the standard Kaggle submission format (`id`, `label`) and can tolerate using the sample submission as a placeholder when the intended blend inputs are unavailable.'
- What this solution (achieved 0.76381) has done: 'The timeout is dominated by the fallback path that reads tens of thousands of `.tif` files in pure-Python loops (PIL open/convert + numpy conversion) to build features for the test set (45k images). To keep the same core logic (central 32×32 mean RGB → logistic regression → predict_proba) but make it fast, I cache per-image features to disk and parallelize feature extraction across CPU cores using a process pool (purely equivalent computation, just concurrent). I also avoid repeated DataFrame index lookups and reduce overhead in the inner loop by computing the crop mean without reshaping. If the external submissions exist, behavior is unchanged; if they don’t, the fallback now finishes within the time budget and reuse cached features across runs.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import scipy.special

sigmoid = lambda x: scipy.special.expit(x)



## === cell 1
from PIL import Image
from sklearn.linear_model import LogisticRegression
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


def _central_crop_mean_rgb(img_arr, crop=32):
    h, w = img_arr.shape[0], img_arr.shape[1]
    y0 = h // 2 - crop // 2
    x0 = w // 2 - crop // 2
    patch = img_arr[y0 : y0 + crop, x0 : x0 + crop, :3]
    return patch.mean(axis=(0, 1))


def _read_tif_as_rgb(path):
    with Image.open(path) as im:
        im = im.convert("RGB")
        return np.asarray(im)


def _cache_path_for(folder, n, crop=32):
    base = os.path.basename(folder.rstrip("/"))
    cache_dir = os.path.join(".", ".feature_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{base}_n{n}_crop{crop}_meanrgb_float32.npz")


def _feat_one(args):
    _id, folder, crop = args
    fp = os.path.join(folder, f"{_id}.tif")
    if not os.path.exists(fp):
        return _id, np.zeros(3, dtype=np.float32)
    arr = _read_tif_as_rgb(fp)
    feat = _central_crop_mean_rgb(arr, crop=crop).astype(np.float32, copy=False)
    return _id, feat


def _build_features(ids, folder, max_items=None, crop=32, use_cache=True):
    n = len(ids) if max_items is None else min(len(ids), max_items)
    ids_use = list(ids[:n])

    if use_cache:
        cache_path = _cache_path_for(folder, n, crop=crop)
        if os.path.exists(cache_path):
            try:
                z = np.load(cache_path, allow_pickle=False)
                X = z["X"]
                used_ids = z["ids"].astype(str).tolist()
                if X.shape == (n, 3) and len(used_ids) == n:
                    return X, used_ids
            except Exception:
                pass  # fall back to rebuild

    X = np.zeros((n, 3), dtype=np.float32)

    cpu = os.cpu_count() or 2
    workers = max(1, min(8, cpu))
    chunksize = 256  # reduces IPC overhead

    if workers == 1:
        for i, _id in enumerate(ids_use):
            _, feat = _feat_one((_id, folder, crop))
            X[i] = feat
    else:
        ctx = mp.get_context("fork" if hasattr(os, "fork") else "spawn")
        with ctx.Pool(processes=workers) as pool:
            for i, (_id, feat) in enumerate(
                pool.imap(
                    _feat_one,
                    ((_id, folder, crop) for _id in ids_use),
                    chunksize=chunksize,
                )
            ):
                X[i] = feat

    if use_cache:
        cache_path = _cache_path_for(folder, n, crop=crop)
        try:
            np.savez_compressed(cache_path, X=X, ids=np.array(ids_use, dtype=object))
        except Exception:
            pass

    return X, ids_use


def _train_and_predict_submission():
    train_df = pd.read_csv(TRAIN_LABELS_PATH)
    sample_df = pd.read_csv(SAMPLE_SUB_PATH)

    train_ids = train_df["id"].tolist()

    max_train = 20000
    X_train, used_train_ids = _build_features(
        train_ids, TRAIN_DIR, max_items=max_train, crop=32, use_cache=True
    )

    y_map = dict(zip(train_df["id"].values, train_df["label"].astype(int).values))
    y_train = np.fromiter(
        (y_map[_id] for _id in used_train_ids),
        dtype=np.int64,
        count=len(used_train_ids),
    )

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=200,
        n_jobs=1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    test_ids = sample_df["id"].tolist()
    X_test, _ = _build_features(
        test_ids, TEST_DIR, max_items=None, crop=32, use_cache=True
    )
    preds = clf.predict_proba(X_test)[:, 1].astype(np.float64)

    out = pd.DataFrame({"id": test_ids, "label": preds})
    return out


def _safe_read_submission_or_make(path):
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "id" in df.columns and "label" in df.columns and len(df) > 0:
            return df[["id", "label"]]
    return _train_and_predict_submission()


sub1 = _safe_read_submission_or_make("../input/blending-power-0-9754/sub.csv")
sub2 = _safe_read_submission_or_make(
    "../input/tta-power-densenet169/submission_tta_64.csv"
)
sub3 = _safe_read_submission_or_make(
    "../input/you-really-need-attention-pytorch/sub_tta.csv"
)



## === cell 2
sub1.head()



## === cell 3
sub2.head()



## === cell 4
sub3.head()



## === cell 5
sub1 = sub1.copy()
sub1["label"] = (
    sub1["label"].values + sub2["label"].values + sub3["label"].values
) / 3.0



## === cell 6
sub1["label"] = sub1["label"].clip(0.0, 1.0)
sub1.to_csv("ensemble.csv", index=False)
print("Wrote ensemble.csv with shape:", sub1.shape)
print(sub1.head())
