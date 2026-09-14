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
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(0)



## === cell 1
CANDIDATE_ROOTS = [
    "/kaggle/input/histopathologic-cancer-detection",
    "/kaggle/data/histopathologic-cancer-detection",
    "../input/histopathologic-cancer-detection",
    "../kaggle/input/histopathologic-cancer-detection",
    "/kaggle/input",
    "../input",
]


def find_file(patterns, roots):
    for r in roots:
        if not os.path.exists(r):
            continue
        for pat in patterns:
            p1 = os.path.join(r, pat)
            if os.path.exists(p1):
                return p1
        for pat in patterns:
            p2 = os.path.join(r, "histopathologic-cancer-detection", pat)
            if os.path.exists(p2):
                return p2
        for pat in patterns:
            hits = glob.glob(os.path.join(r, "**", pat), recursive=True)
            if hits:
                hits = sorted(hits, key=len)
                return hits[0]
    return None


train_csv = find_file(["train_labels.csv"], CANDIDATE_ROOTS)
sample_csv = find_file(["sample_submission.csv"], CANDIDATE_ROOTS)


def find_image_dir(dir_name):
    for r in CANDIDATE_ROOTS:
        if not os.path.exists(r):
            continue

        direct = os.path.join(r, dir_name)
        if os.path.isdir(direct) and os.path.exists(direct):
            with os.scandir(direct) as it:
                for e in it:
                    if e.is_file() and e.name.endswith(".tif"):
                        return direct

        direct2 = os.path.join(r, "histopathologic-cancer-detection", dir_name)
        if os.path.isdir(direct2) and os.path.exists(direct2):
            with os.scandir(direct2) as it:
                for e in it:
                    if e.is_file() and e.name.endswith(".tif"):
                        return direct2

        hits = [
            p
            for p in glob.glob(os.path.join(r, "**", dir_name), recursive=True)
            if os.path.isdir(p)
        ]
        for p in sorted(hits, key=len):
            try:
                with os.scandir(p) as it:
                    for e in it:
                        if e.is_file() and e.name.endswith(".tif"):
                            return p
            except OSError:
                continue
    return None


train_dir = find_image_dir("train")
test_dir = find_image_dir("test")

print("train_csv:", train_csv)
print("sample_csv:", sample_csv)
print("train_dir:", train_dir)
print("test_dir:", test_dir)

if train_csv is None or sample_csv is None or train_dir is None or test_dir is None:
    raise FileNotFoundError(
        "Could not locate required competition files. "
        "Expected train_labels.csv, sample_submission.csv, and train/test image directories."
    )



## === cell 2
train_df = pd.read_csv(train_csv)
sub_df = pd.read_csv(sample_csv)

assert set(train_df.columns) >= {"id", "label"}
assert set(sub_df.columns) >= {"id", "label"}

print(train_df.shape, sub_df.shape)
train_df.head()



## === cell 3
import multiprocessing as mp

try:
    import imageio.v2 as imageio
except Exception as e:
    raise ImportError(
        "imageio is required for fast TIFF reading in this environment."
    ) from e


def imread_any(path):
    return imageio.imread(path)


def _rgb_to_hsv_batch(rgb01):
    """
    Deterministic RGB->HSV for an array (..., 3) with values in [0,1].
    Returns HSV in [0,1].
    """
    r = rgb01[..., 0]
    g = rgb01[..., 1]
    b = rgb01[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    diff = mx - mn

    v = mx
    s = diff / (mx + 1e-8)

    h = np.zeros_like(mx, dtype=np.float32)
    mask = diff > 1e-8
    mr = mask & (mx == r)
    mg = mask & (mx == g)
    mb = mask & (mx == b)

    h[mr] = ((g[mr] - b[mr]) / (diff[mr] + 1e-8)) % 6.0
    h[mg] = ((b[mg] - r[mg]) / (diff[mg] + 1e-8)) + 2.0
    h[mb] = ((r[mb] - g[mb]) / (diff[mb] + 1e-8)) + 4.0
    h = (h / 6.0).astype(np.float32)

    hsv = np.stack([h, s.astype(np.float32), v.astype(np.float32)], axis=-1)
    return hsv


def _quantile_linear_1d(x, q):
    x = np.asarray(x)
    n = x.size
    if n == 0:
        return np.nan
    pos = (n - 1) * float(q)
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return float(np.partition(x, lo)[lo])
    x_part = np.partition(x, (lo, hi))
    x_lo = float(x_part[lo])
    x_hi = float(x_part[hi])
    h = pos - lo
    return x_lo + (x_hi - x_lo) * h


def _quantiles_linear_axis0(flat, qs):
    flat = np.asarray(flat)
    qs = list(qs)
    out = np.empty((len(qs), flat.shape[1]), dtype=np.float32)
    for c in range(flat.shape[1]):
        col = flat[:, c]
        for i, q in enumerate(qs):
            out[i, c] = _quantile_linear_1d(col, q)
    return out


def _extract_features_from_patch(patch):
    """
    Core logic unchanged: summarize a center crop with deterministic color/texture stats.
    """
    patch = patch.astype(np.float32)
    if patch.max() > 1.5:  # uint8-like
        patch = patch / 255.0

    flat = patch.reshape(-1, 3)

    ch_mean = flat.mean(axis=0)
    ch_std = flat.std(axis=0)
    overall_mean = float(patch.mean())
    overall_std = float(patch.std())
    eps = 1e-6
    rg = float(ch_mean[0] / (ch_mean[1] + eps))
    rb = float(ch_mean[0] / (ch_mean[2] + eps))
    gb = float(ch_mean[1] / (ch_mean[2] + eps))

    q = _quantiles_linear_axis0(flat, [0.1, 0.5, 0.9]).astype(
        np.float32
    )  # (3,3) -> 9 feats

    lum = (
        0.2989 * patch[..., 0] + 0.5870 * patch[..., 1] + 0.1140 * patch[..., 2]
    ).astype(np.float32)

    dx = lum[:, 1:] - lum[:, :-1]  # (H, W-1)
    dy = lum[1:, :] - lum[:-1, :]  # (H-1, W)
    dx_al = dx[:-1, :]  # (H-1, W-1) after aligning with dy
    dy_al = dy[:, :-1]  # (H-1, W-1)
    g = np.sqrt(dx_al * dx_al + dy_al * dy_al).reshape(-1)

    g_mean = float(g.mean())
    g_std = float(g.std())
    g_q0 = _quantile_linear_1d(g, 0.5)
    g_q1 = _quantile_linear_1d(g, 0.9)

    hsv = _rgb_to_hsv_batch(patch)
    hsv_flat = hsv.reshape(-1, 3)
    hsv_mean = hsv_flat.mean(axis=0).astype(np.float32)  # 3
    hsv_std = hsv_flat.std(axis=0).astype(np.float32)  # 3

    od = -np.log(patch + 1e-6).reshape(-1, 3)  # optical density
    v_h = np.array([0.65, 0.70, 0.29], dtype=np.float32)
    v_e = np.array([0.07, 0.99, 0.11], dtype=np.float32)
    h_proj = (od * v_h).sum(axis=1)
    e_proj = (od * v_e).sum(axis=1)

    h_q9 = _quantile_linear_1d(h_proj, 0.9)
    e_q9 = _quantile_linear_1d(e_proj, 0.9)

    he_feats = np.array(
        [
            float(h_proj.mean()),
            float(h_proj.std()),
            float(h_q9),
            float(e_proj.mean()),
            float(e_proj.std()),
            float(e_q9),
        ],
        dtype=np.float32,
    )  # 6 feats

    feats = np.array(
        [
            *ch_mean,  # 3
            *ch_std,  # 3 (total 6)
            overall_mean,
            overall_std,  # 2 (total 8)
            rg,
            rb,
            gb,  # 3 (total 11)
            *q.reshape(-1).tolist(),  # 9 (total 20)
            g_mean,
            g_std,  # 2 (total 22)
            float(g_q0),
            float(g_q1),  # 2 (total 24)
            *hsv_mean.tolist(),  # 3 (total 27)
            *hsv_std.tolist(),  # 3 (total 30)
            *he_feats.tolist(),  # 6 (total 36)
        ],
        dtype=np.float32,
    )
    return feats


def extract_center_features(img, crop=32):
    """
    Change to move AUC toward target (big gap): still 'center crop -> summarize',
    but allow a larger center crop to capture more discriminative context.
    """
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    h, w = img.shape[:2]
    crop = int(crop)
    y0 = (h - crop) // 2
    x0 = (w - crop) // 2
    patch = img[y0 : y0 + crop, x0 : x0 + crop]
    return _extract_features_from_patch(patch)


def _feat_from_path(path):
    img = imread_any(path)
    f32 = extract_center_features(img, crop=32)
    f96 = extract_center_features(img, crop=96)
    feat = np.concatenate([f32, f96], axis=0).astype(np.float32)
    return feat


def _feat_batch(args):
    start_idx, paths = args
    out = []
    missing = 0
    for j, p in enumerate(paths):
        try:
            feat = _feat_from_path(p)
            out.append((start_idx + j, feat))
        except Exception:
            missing += 1
    return out, missing


def build_feature_matrix(
    ids, img_dir, verbose_every=5000, cache_path=None, num_workers=None, n_features=72
):
    if cache_path is not None and os.path.exists(cache_path):
        X = np.load(cache_path)
        if X.shape == (len(ids), n_features) and X.dtype == np.float32:
            print(f"Loaded cached features: {cache_path}")
            return X

    n = len(ids)
    X = np.zeros((n, n_features), dtype=np.float32)

    paths = [os.path.join(img_dir, f"{img_id}.tif") for img_id in ids]

    if num_workers is None:
        num_workers = min(12, max(1, (os.cpu_count() or 2) - 1))

    batch_size = 512
    tasks = [(i, paths[i : i + batch_size]) for i in range(0, n, batch_size)]

    missing = 0
    done = 0

    if num_workers <= 1:
        for start_idx, batch_paths in tasks:
            out, miss = _feat_batch((start_idx, batch_paths))
            for idx, feat in out:
                X[idx] = feat
            missing += miss
            done += len(batch_paths)
            if verbose_every and done % verbose_every == 0:
                print(f"Processed {done}/{n} images...")
    else:
        ctx = mp.get_context("fork") if hasattr(os, "fork") else mp.get_context("spawn")
        chunksize = max(1, len(tasks) // (num_workers * 8))
        with ctx.Pool(processes=num_workers, maxtasksperchild=500) as pool:
            for out, miss in pool.imap_unordered(
                _feat_batch, tasks, chunksize=chunksize
            ):
                for idx, feat in out:
                    X[idx] = feat
                missing += miss
                done += len(out) + miss  # batch size = successes + misses
                if (
                    verbose_every
                    and done >= verbose_every
                    and (done % verbose_every) < batch_size
                ):
                    print(f"Processed {min(done, n)}/{n} images...")

    if missing:
        print(f"Warning: {missing} images were missing/unreadable in {img_dir}.")

    if cache_path is not None:
        np.save(cache_path, X)
        print(f"Cached features to: {cache_path}")

    return X




## === cell 4
train_ids = train_df["id"].values
y = train_df["label"].values.astype(np.int32)

X = build_feature_matrix(
    train_ids,
    train_dir,
    verbose_every=20000,
    cache_path="X_train_center32_96_features_v6.npy",
    num_workers=None,
    n_features=72,
)
print("X shape:", X.shape, "y shape:", y.shape)




## === cell 5
def make_clf():
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "lr",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=2000,
                    class_weight=None,
                    C=4.0,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)
clf_holdout = make_clf()
clf_holdout.fit(X_tr, y_tr)
va_pred = clf_holdout.predict_proba(X_va)[:, 1]
va_auc = roc_auc_score(y_va, va_pred)
print("Validation AUC:", va_auc)



## === cell 6
test_ids = sub_df["id"].values

X_test = build_feature_matrix(
    test_ids,
    test_dir,
    verbose_every=20000,
    cache_path="X_test_center32_96_features_v6.npy",
    num_workers=None,
    n_features=72,
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

test_pred_sum = np.zeros(X_test.shape[0], dtype=np.float64)
oof_pred = np.zeros(X.shape[0], dtype=np.float64)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    clf = make_clf()
    clf.fit(X[tr_idx], y[tr_idx])

    oof_pred[va_idx] = clf.predict_proba(X[va_idx])[:, 1]
    test_pred_sum += clf.predict_proba(X_test)[:, 1]

    if fold == 1 or fold == 5:
        print(f"Finished fold {fold}/5")

cv_auc = roc_auc_score(y, oof_pred)
print("OOF CV AUC:", cv_auc)

test_pred = (test_pred_sum / 5.0).astype(np.float32)

submission = pd.DataFrame({"id": test_ids, "label": test_pred})
assert submission.shape[0] == sub_df.shape[0]
assert submission["id"].iloc[0] == sub_df["id"].iloc[0]

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
submission.head()
