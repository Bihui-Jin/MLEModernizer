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
Classify and localize common thoracic lung diseases and critical findings.

For each test image, you will be predicting a bounding box and class for all findings. If you predict that there are no findings, you should create a prediction of "14 1 0 0 1 1" (14 is the class ID for no finding, and this provides a one-pixel bounding box with a confidence of 1.0).

## Metric
PASCAL VOC 2010 [mean Average Precision (mAP)](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/devkit_doc_08-May-2010.pdf) at IoU > 0.4.

## Submission Format
Images in the test set may contain more than one object. For each object in a given test image, you must predict a class ID, `confidence` score, and bounding box in format `xmin ymin xmax ymax`. If you predict that there are NO objects in a given image, you should predict `14 1.0 0 0 1 1`, where `14` is the class ID for "No finding", 1.0 is the confidence, and `0 0 1 1` is a one-pixel bounding box.

The submission file should contain a header and have the following format:

```
ID,TARGET
004f33259ee4aef671c2b95d54e4be68,14 1 0 0 1 1
004f33259ee4aef671c2b95d54e4be69,11 0.5 100 100 200 200 13 0.7 10 10 20 20
etc.
```

## Dataset
The dataset comprises postero-anterior (PA) CXR scans in DICOM format.

All images were labeled for the presence of 14 critical radiographic findings as listed below:

```
0 - Aortic enlargement
1 - Atelectasis
2 - Calcification
3 - Cardiomegaly
4 - Consolidation
5 - ILD
6 - Infiltration
7 - Lung Opacity
8 - Nodule/Mass
9 - Other lesion
10 - Pleural effusion
11 - Pleural thickening
12 - Pneumothorax
13 - Pulmonary fibrosis
```

The "No finding" observation (`14`) was intended to capture the absence of all findings above.

### Files
- **train.csv** - the train set metadata, with one row for each object, including a class and a bounding box. Some images in both test and train have multiple objects.
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `image_id` - unique image identifier
- `class_name` - the name of the class of detected object (or "No finding")
- `class_id` - the ID of the class of detected object
- `rad_id` - the ID of the radiologist that made the observation
- `x_min` - minimum X coordinate of the object's bounding box
- `y_min` - minimum Y coordinate of the object's bounding box
- `x_max` - maximum X coordinate of the object's bounding box
- `y_max` - maximum Y coordinate of the object's bounding box

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
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
            description.md (132 lines)
            sample_submission.csv (1501 lines)
            sample_submission.csv.zip (30.9 kB)
            test.zip (12.7 GB)
            train.csv (61172 lines)
            train.csv.zip (1.7 MB)
            train.zip (114.6 GB)
            test/
                00575e3846ebd05a909d97ba59c53d30.dicom (12.9 MB)
                0059d21bef1793fa9522e4ec8cae1a1a.dicom (9.3 MB)
                ... and 1498 other files
                test/
            train/
                000434271f63a053c4128a0ba6352c7f.dicom (13.3 MB)
                00053190460d56c53cc3e57321387478.dicom (9.7 MB)
                ... and 13498 other files
                train/
            vinbigdata-chest-xray-abnormalities-detection/
                description.md (132 lines)
                sample_submission.csv (1501 lines)
                ... and 5 other files
                test/
                    00575e3846ebd05a909d97ba59c53d30.dicom (12.9 MB)
                    0059d21bef1793fa9522e4ec8cae1a1a.dicom (9.3 MB)
                    ... and 1498 other files
                    test/
                train/
                    000434271f63a053c4128a0ba6352c7f.dicom (13.3 MB)
                    00053190460d56c53cc3e57321387478.dicom (9.7 MB)
                    ... and 13498 other files
                    train/
                vinbigdata-chest-xray-abnormalities-detection/
        input/
            description.md (132 lines)
            sample_submission.csv (1501 lines)
            sample_submission.csv.zip (30.9 kB)
            test.zip (12.7 GB)
            train.csv (61172 lines)
            train.csv.zip (1.7 MB)
            train.zip (114.6 GB)
            test/
                00575e3846ebd05a909d97ba59c53d30.dicom (12.9 MB)
                0059d21bef1793fa9522e4ec8cae1a1a.dicom (9.3 MB)
                ... and 1498 other files
                test/
                    00575e3846ebd05a909d97ba59c53d30.dicom (12.9 MB)
                    0059d21bef1793fa9522e4ec8cae1a1a.dicom (9.3 MB)
                    ... and 1498 other files
                    test/
            train/
                000434271f63a053c4128a0ba6352c7f.dicom (13.3 MB)
                00053190460d56c53cc3e57321387478.dicom (9.7 MB)
                ... and 13498 other files
                train/
                    000434271f63a053c4128a0ba6352c7f.dicom (13.3 MB)
                    00053190460d56c53cc3e57321387478.dicom (9.7 MB)
                    ... and 13498 other files
                    train/
            vinbigdata-chest-xray-abnormalities-detection/
                description.md (132 lines)
                sample_submission.csv (1501 lines)
                ... and 5 other files
                test/
                    00575e3846ebd05a909d97ba59c53d30.dicom (12.9 MB)
                    0059d21bef1793fa9522e4ec8cae1a1a.dicom (9.3 MB)
                    ... and 1498 other files
                    test/
                train/
                    000434271f63a053c4128a0ba6352c7f.dicom (13.3 MB)
                    00053190460d56c53cc3e57321387478.dicom (9.7 MB)
                    ... and 13498 other files
                    train/
                vinbigdata-chest-xray-abnormalities-detection/
        working/
            vinbigdata-chest-xray-abnormalities-detection/
                description.md (132 lines)
                sample_submission.csv (1501 lines)
                ... and 5 other files
                test/
                    00575e3846ebd05a909d97ba59c53d30.dicom (12.9 MB)
                    0059d21bef1793fa9522e4ec8cae1a1a.dicom (9.3 MB)
                    ... and 1498 other files
                    test/
                train/
                    000434271f63a053c4128a0ba6352c7f.dicom (13.3 MB)
                    00053190460d56c53cc3e57321387478.dicom (9.7 MB)
                    ... and 13498 other files
                    train/
                vinbigdata-chest-xray-abnormalities-detection/
```

-> data/sample_submission.csv has 1500 rows and 2 columns.
The columns are: image_id, PredictionString

-> data/train.csv has 61171 rows and 8 columns.
The columns are: image_id, class_name, class_id, rad_id, x_min, y_min, x_max, y_max

-> data/vinbigdata-chest-xray-abnormalities-detection/sample_submission.csv has 1500 rows and 2 columns.
The columns are: image_id, PredictionString

-> data/vinbigdata-chest-xray-abnormalities-detection/train.csv has 61171 rows and 8 columns.
The columns are: image_id, class_name, class_id, rad_id, x_min, y_min, x_max, y_max

-> input/sample_submission.csv has 1500 rows and 2 columns.
The columns are: image_id, PredictionString

-> input/train.csv has 61171 rows and 8 columns.
The columns are: image_id, class_name, class_id, rad_id, x_min, y_min, x_max, y_max

-> (stopped after 10 files for performance)

# 5. Target score

0.2177142085841383

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0475) has done: 'The current notebook fails because it references two external Kaggle datasets/paths that are not available in your environment. I replace those reads with a self-contained fallback that uses the provided `sample_submission.csv` as the base detection submission and generates a reasonable `pred_2class` (image-level “normal” probability) from the provided `train.csv` priors, so the pipeline runs end-to-end and writes a valid `submission.csv`. I also make the merge robust to missing columns and ensure every test image has a non-empty `PredictionString` (defaulting to the required “No finding” string). These changes are minimal and keep your existing post-processing logic/threshold semantics intact while producing a valid submission.'
- What this solution (achieved 0.06606) has done: 'Your current score (0.0475) is far below the target (0.2177), so we should legitimately increase mAP with the smallest change that preserves your “sample_submission-as-base + add/replace No finding based on class0 thresholds” core logic. The main issue is that you currently start from *all No finding* detections, which yields near-zero true positives for abnormal boxes; instead, we can build a simple, legal baseline detector by copying *training-set consensus boxes* (grouped by class) onto each test image, producing non-empty abnormal predictions while keeping your thresholding/post-processing semantics intact. To avoid flooding with too many low-quality boxes (which can hurt mAP), we keep only the top few most frequent classes and use the class frequency as the confidence, clipped to a reasonable range. We also keep your “No finding” threshold logic, but we ensure the produced PredictionString is always valid and formatted correctly.'
- What this solution (achieved 0.06689) has done: 'I keep your “training-consensus template boxes + optional No finding append/replace” core logic intact, but slightly increase the amount of useful signal in the template so mAP can move toward the 0.2177 target from 0.0661. Concretely, I (1) use class-wise box *medians* as you do but also add a second, slightly different box per class (using 25th/75th percentiles) to better cover size variability without changing model type, (2) increase TOPK_CLASSES a bit and cap total predicted boxes per image to avoid flooding, and (3) make the No finding append safe by inserting a separator space only when needed. These are minimal, legal post-processing changes that should increase true positives relative to your current single-template-per-class approach while keeping runtime well under the limit and preserving your thresholding semantics.'
- What this solution (achieved 0.06711) has done: 'We’re far below the target (0.06689 vs 0.2177, higher-is-better), so we should increase true positives with minimal risk while keeping your “train-template boxes + No finding append/replace by class0 thresholds” logic intact. The smallest legitimate boost is to make the template boxes slightly more image-size-aware by scaling the consensus boxes to the typical train image dimensions (reduces systematic IoU mismatch on many images) and to include one more robust size-variation box per class (10th/90th) while keeping a strict cap on total boxes. I also adjust the template confidence very slightly upward (still clipped) because your current confidences are quite low and mAP is sensitive to ranking; this keeps semantics the same (still priors-based, no new model). Finally, I keep submission formatting/validity guarantees unchanged.'
- What this solution (achieved 0.06741) has done: 'We’re far below the target (0.06711 vs 0.2177, higher-is-better), so the smallest legitimate way to move mAP upward is to improve ranking/recall within your existing “train-consensus template boxes + optional No finding append/replace by class0 thresholds” logic rather than changing models. I keep the same template-generation approach but (1) expand TOPK_CLASSES slightly and (2) allocate a small fixed per-class box budget (median + IQR + tail) while enforcing a strict global cap, which increases recall without flooding too many low-quality boxes. I also slightly re-balance confidence scaling so the most frequent classes rank higher (mAP is sensitive to ranking), while keeping the same priors-based confidence semantics and your No-finding threshold logic untouched. The script still run end-to-end and write a valid `submission.csv` with guaranteed non-empty `PredictionString` for every test image.'
- What this solution (achieved 0.06727) has done: 'Your current score (0.06741) is far below the target (0.2177), so we should increase mAP by improving recall/ranking while preserving your existing “train-derived template boxes + optional No finding append/replace by class0 thresholds” core logic. The smallest effective change here is to generate the template boxes using *radiologist-consensus boxes* (group by `class_id, image_id` first, then take per-class quantiles), which reduces label-noise in the templates and tends to improve IoU alignment without changing the approach. I also modestly increase the per-image box cap and slightly raise the minimum confidence floor for frequent classes to improve ranking, while keeping your No-finding threshold logic and submission formatting intact. Everything still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.06729) has done: 'We’re far below the target (0.06727 vs 0.21771, higher-is-better), so the smallest legitimate way to move mAP upward without changing your core “train-derived template boxes + optional No finding append/replace” logic is to improve the template box coverage and ranking slightly. I (1) add one more moderate-variation template box per class (using 0.40/0.60 quantiles) for the most frequent classes to improve IoU chances without flooding, and (2) make the confidence mapping a bit more separative (higher for frequent classes, still clipped) to help mAP ranking. I keep your thresholds, formatting, and end-to-end submission writing identical, while enforcing the same global cap so runtime stays well under 600s. These changes are minimal and remain purely train-prior/template based (no new model, no new data).'
- What this solution (achieved 0.05214) has done: 'Your current score (0.06729) is far below the target (0.2177), so we should increase mAP by making the existing “train-derived template boxes + optional No Finding append/replace” produce more *image-specific* boxes without changing the overall approach. The smallest safe step is to derive a per-test-image template by matching each test image to a small set of most-similar *train* images based on the already-available metadata (`x_min..y_max` distributions per image), then copying those images’ consensus boxes into the PredictionString (still purely train-prior/template based, no new model). This tends to raise IoU/recall versus one global template while keeping your same post-processing and No Finding threshold logic intact. I also keep a strict per-image box cap and confidence mapping based on class frequency to avoid flooding (which can hurt mAP).'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_DIR_CANDIDATES = [
    "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection",
    "/kaggle/data/vinbigdata-chest-xray-abnormalities-detection",
    "/kaggle/input",
    "/kaggle/data",
]


def _first_existing(path_list):
    for p in path_list:
        if os.path.exists(p):
            return p
    return None


BASE_DIR = _first_existing(DATA_DIR_CANDIDATES)
if BASE_DIR is None:
    raise FileNotFoundError(
        "Could not locate Kaggle input data directory. Checked: "
        + str(DATA_DIR_CANDIDATES)
    )

sample_path = os.path.join(BASE_DIR, "sample_submission.csv")
train_path = os.path.join(BASE_DIR, "train.csv")

sample_df = pd.read_csv(sample_path)
train_df = pd.read_csv(train_path)

if "image_id" not in sample_df.columns:
    for c in sample_df.columns:
        if c.lower() == "id":
            sample_df = sample_df.rename(columns={c: "image_id"})
            break
if "PredictionString" not in sample_df.columns:
    for c in sample_df.columns:
        if c.lower() in ("predictionstring", "target"):
            sample_df = sample_df.rename(columns={c: "PredictionString"})
            break
if "image_id" not in sample_df.columns or "PredictionString" not in sample_df.columns:
    raise ValueError(
        f"Unexpected sample_submission columns: {sample_df.columns.tolist()}"
    )

test_image_ids = sample_df["image_id"].values

t = train_df.copy()
for col in ["class_id", "x_min", "y_min", "x_max", "y_max"]:
    t[col] = pd.to_numeric(t[col], errors="coerce")
t = t.dropna(subset=["class_id", "x_min", "y_min", "x_max", "y_max"])
t["class_id"] = t["class_id"].astype(int)

t_abn = t[t["class_id"] != 14].copy()

NORMAL = "14 1 0 0 1 1"


def _sanitize_box(x1, y1, x2, y2):
    xmin = max(0.0, min(float(x1), float(x2)))
    ymin = max(0.0, min(float(y1), float(y2)))
    xmax = max(xmin + 1.0, max(float(x1), float(x2)))
    ymax = max(ymin + 1.0, max(float(y1), float(y2)))
    return xmin, ymin, xmax, ymax


def _build_image_level_features(df_abn: pd.DataFrame) -> pd.DataFrame:
    g = df_abn.groupby("image_id")
    feat = pd.DataFrame(
        {
            "image_id": g.size().index,
            "n_box": g.size().values.astype(float),
            "x_min_med": g["x_min"].median().values,
            "y_min_med": g["y_min"].median().values,
            "x_max_med": g["x_max"].median().values,
            "y_max_med": g["y_max"].median().values,
            "w_med": (g["x_max"].median() - g["x_min"].median()).values,
            "h_med": (g["y_max"].median() - g["y_min"].median()).values,
        }
    )
    for c in feat.columns:
        if c != "image_id":
            feat[c] = pd.to_numeric(feat[c], errors="coerce").fillna(0.0)
    return feat


def _try_import_pydicom():
    try:
        import pydicom  # type: ignore

        return pydicom
    except Exception:
        return None


def _dicom_path(split_dir, image_id):
    return os.path.join(BASE_DIR, split_dir, f"{image_id}.dicom")


def _safe_dicom_features(image_ids, split_dir, max_images=None):
    """
    Returns DataFrame with columns:
      image_id, px_mean, px_std, com_x, com_y

    Validity-relevant: use CSV cache (pyarrow/fastparquet may be unavailable) so we
    don't crash before writing submission.csv.
    """
    pydicom = _try_import_pydicom()
    n = len(image_ids) if max_images is None else min(len(image_ids), max_images)
    ids = list(image_ids[:n])

    cache_path = os.path.join(
        "/kaggle/working",
        f"_dicom_feat_cache_{os.path.basename(BASE_DIR)}_{split_dir}_{n}.csv",
    )
    if os.path.exists(cache_path):
        try:
            cached = pd.read_csv(cache_path)
            if "image_id" in cached.columns and len(cached) == n:
                cached = cached.set_index("image_id").reindex(ids).reset_index()
                if cached["image_id"].isna().any() is False:
                    return cached
        except Exception:
            pass

    if pydicom is None:
        out = pd.DataFrame(
            {
                "image_id": np.asarray(ids),
                "px_mean": np.zeros(n, dtype=np.float32),
                "px_std": np.ones(n, dtype=np.float32),
                "com_x": np.full(n, 0.5, dtype=np.float32),
                "com_y": np.full(n, 0.5, dtype=np.float32),
            }
        )
        return out

    dcmread = pydicom.dcmread

    def _header_only_features(ds):
        try:
            wc = getattr(ds, "WindowCenter", None)
            ww = getattr(ds, "WindowWidth", None)
            if isinstance(wc, (list, tuple)):
                wc = wc[0] if wc else None
            if isinstance(ww, (list, tuple)):
                ww = ww[0] if ww else None
            if wc is None or ww is None:
                return None
            wc = float(wc)
            ww = float(ww)
            px_mean = wc
            px_std = abs(ww) / 3.4641016151377544  # sqrt(12)
            return px_mean, px_std
        except Exception:
            return None

    def _one(iid):
        fp = _dicom_path(split_dir, iid)
        try:
            ds = dcmread(fp, stop_before_pixels=True, force=True, defer_size="1 KB")
            hdr = _header_only_features(ds)
            if hdr is not None:
                m, s = hdr
                return iid, float(m), float(max(s, 1e-6)), 0.5, 0.5

            ds = dcmread(fp, stop_before_pixels=False, force=True, defer_size="1 KB")
            arr = ds.pixel_array.astype(np.float32, copy=False)

            if arr.size:
                m = float(arr.mean())
                s = float(arr.std())
                amin = float(arr.min())
                a = arr - amin
                mass = float(a.sum())
                if mass <= 1e-6:
                    cy, cx = 0.5, 0.5
                else:
                    h, w = a.shape[:2]
                    row_mass = a.sum(axis=1)
                    col_mass = a.sum(axis=0)
                    cy = float(
                        np.dot(row_mass, np.arange(h, dtype=np.float32)) / mass
                    ) / max(1.0, float(h - 1))
                    cx = float(
                        np.dot(col_mass, np.arange(w, dtype=np.float32)) / mass
                    ) / max(1.0, float(w - 1))
            else:
                m, s, cx, cy = 0.0, 0.0, 0.5, 0.5

            return iid, m, (s if s > 1e-6 else 1e-6), cx, cy
        except Exception:
            return iid, 0.0, 1.0, 0.5, 0.5

    try:
        from concurrent.futures import ThreadPoolExecutor
        import multiprocessing as mp

        max_workers = min(8, max(2, (mp.cpu_count() or 2)))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_one, ids))
    except Exception:
        results = [_one(iid) for iid in ids]

    out_ids, px_mean, px_std, com_x, com_y = (
        zip(*results) if results else ([], [], [], [], [])
    )
    out = pd.DataFrame(
        {
            "image_id": np.asarray(out_ids),
            "px_mean": np.asarray(px_mean, dtype=np.float32),
            "px_std": np.asarray(px_std, dtype=np.float32),
            "com_x": np.asarray(com_x, dtype=np.float32),
            "com_y": np.asarray(com_y, dtype=np.float32),
        }
    )

    try:
        out.to_csv(cache_path, index=False)
    except Exception:
        pass

    return out


pydicom_available = _try_import_pydicom() is not None

if len(t_abn) == 0:
    pred_det_df = sample_df.copy()
    if "PredictionString" not in pred_det_df.columns:
        pred_det_df["PredictionString"] = NORMAL
else:
    t_cons = (
        t_abn.groupby(["image_id", "class_id"], as_index=False)[
            ["x_min", "y_min", "x_max", "y_max"]
        ]
        .median()
        .copy()
    )

    cls_counts = t_cons["class_id"].value_counts().sort_values(ascending=False)
    total_abn = float(cls_counts.sum())

    TOPK_CLASSES = 13
    top_classes = cls_counts.head(TOPK_CLASSES).index.tolist()

    def _conf_from_count(c):
        p = float(c) / total_abn
        return float(np.clip(p, 0.20, 0.92))

    conf_map = {
        int(cid): _conf_from_count(int(cls_counts.loc[cid])) for cid in top_classes
    }

    train_feat_bbox = _build_image_level_features(t_abn)

    MAX_ANCHOR = 350
    train_image_ids_all = train_feat_bbox["image_id"].values
    n_anchor = min(MAX_ANCHOR, len(train_feat_bbox))

    rng = np.random.default_rng(RANDOM_SEED)
    if len(train_image_ids_all) > n_anchor:
        anchor_ids = rng.choice(train_image_ids_all, size=n_anchor, replace=False)
    else:
        anchor_ids = train_image_ids_all[:n_anchor]

    train_feat_px = _safe_dicom_features(
        anchor_ids.tolist(), split_dir="train", max_images=n_anchor
    )

    train_feat = pd.merge(
        train_feat_bbox.set_index("image_id").loc[anchor_ids].reset_index(),
        train_feat_px,
        on="image_id",
        how="left",
    )

    if pydicom_available:
        feat_cols = [c for c in train_feat.columns if c != "image_id"]
    else:
        feat_cols = ["px_mean", "px_std", "com_x", "com_y"]

    X = train_feat[feat_cols].to_numpy(dtype=np.float32)

    mu = X.mean(axis=0, keepdims=True)
    sig = X.std(axis=0, keepdims=True) + 1e-6
    X_anchor = (X - mu) / sig

    cons_by_img = t_cons[t_cons["class_id"].isin(top_classes)].groupby("image_id")
    anchor_pred = {}
    MAX_BOXES_PER_IMAGE = 24

    for aid in anchor_ids:
        if aid not in cons_by_img.groups:
            anchor_pred[aid] = NORMAL
            continue
        img_boxes = cons_by_img.get_group(aid)
        parts = []
        img_boxes = img_boxes.copy()
        img_boxes["cls_rank"] = img_boxes["class_id"].map(
            lambda c: -cls_counts.get(int(c), 0)
        )
        img_boxes = img_boxes.sort_values(["cls_rank", "class_id"])
        for _, r in img_boxes.iterrows():
            cid = int(r["class_id"])
            if cid not in conf_map:
                continue
            conf = conf_map[cid]
            xmin, ymin, xmax, ymax = _sanitize_box(
                r["x_min"], r["y_min"], r["x_max"], r["y_max"]
            )
            parts.append(
                f"{cid} {conf:.4f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
            )
            if len(parts) >= MAX_BOXES_PER_IMAGE:
                break
        s = " ".join(parts).strip()
        anchor_pred[aid] = s if s else NORMAL

    K_NEIGHBORS = 5

    test_feat_px = _safe_dicom_features(
        test_image_ids.tolist(), split_dir="test", max_images=None
    )

    test_feat_bbox = pd.DataFrame({"image_id": test_image_ids})
    for c in train_feat_bbox.columns:
        if c != "image_id":
            test_feat_bbox[c] = 0.0

    test_feat = pd.merge(test_feat_bbox, test_feat_px, on="image_id", how="left")
    test_feat = test_feat[["image_id"] + feat_cols]

    Xt = test_feat[feat_cols].to_numpy(dtype=np.float32)
    Xtn = (Xt - mu) / sig

    A = X_anchor  # (n_anchor, d)
    A2 = np.einsum("ij,ij->i", A, A).astype(np.float32)  # (n_anchor,)

    pred_strings = []
    chunk = 500
    kth = min(K_NEIGHBORS, A.shape[0] - 1)

    DIST_TEMP = 3.0  # keep same semantics
    for start in range(0, len(test_image_ids), chunk):
        end = min(len(test_image_ids), start + chunk)
        T = Xtn[start:end].astype(np.float32, copy=False)  # (m,d)
        T2 = np.einsum("ij,ij->i", T, T).astype(np.float32)  # (m,)
        d2 = (T2[:, None] + A2[None, :] - 2.0 * (T @ A.T)).astype(
            np.float32, copy=False
        )

        nn_idx = np.argpartition(d2, kth=kth, axis=1)[:, :K_NEIGHBORS]
        for i in range(nn_idx.shape[0]):
            idxs = nn_idx[i]
            aids = anchor_ids[idxs]
            d2s = d2[i, idxs]
            order = np.argsort(d2s)
            aids = aids[order]
            d2s = d2s[order]

            tokens = []
            for aid, dd in zip(aids, d2s):
                s = anchor_pred.get(aid, "")
                if (not s) or (s == NORMAL):
                    continue
                w = float(np.exp(-float(dd) / DIST_TEMP))
                fields = s.split()
                out_fields = []
                for j in range(0, len(fields) - 5, 6):
                    try:
                        cid = fields[j]
                        conf = float(fields[j + 1])
                        conf = float(np.clip(conf * (0.85 + 0.15 * w), 0.01, 0.999))
                        out_fields.extend(
                            [
                                cid,
                                f"{conf:.4f}",
                                fields[j + 2],
                                fields[j + 3],
                                fields[j + 4],
                                fields[j + 5],
                            ]
                        )
                    except Exception:
                        continue
                if out_fields:
                    tokens.append(" ".join(out_fields))

            merged = " ".join(tokens).strip()
            if not merged:
                merged = NORMAL
            else:
                fields = merged.split()
                max_fields = MAX_BOXES_PER_IMAGE * 6
                if len(fields) > max_fields:
                    fields = fields[:max_fields]
                merged = " ".join(fields)
            pred_strings.append(merged)

    pred_det_df = pd.DataFrame(
        {"image_id": test_image_ids, "PredictionString": pred_strings}
    )

class_id_num = (
    pd.to_numeric(train_df["class_id"], errors="coerce").fillna(14).astype(np.int32)
)
has_finding_per_image = (class_id_num != 14).groupby(train_df["image_id"]).any()
normal_prior = float((~has_finding_per_image).mean())

pred_2class = pd.DataFrame(
    {"image_id": pred_det_df["image_id"].values, "class0": normal_prior}
)

low_threshold = 0.0
high_threshold = 0.976

print(f"Using BASE_DIR={BASE_DIR}")
print(f"pydicom_available={pydicom_available}")
print(
    f"Built pred_2class with constant class0(normal) prior={normal_prior:.6f} for {len(pred_2class)} test images"
)
print("Example PredictionString:")
print(pred_det_df.head(1))



## === cell 1
NORMAL = "14 1 0 0 1 1"

if "PredictionString" not in pred_det_df.columns:
    for c in pred_det_df.columns:
        if c.lower() in ("predictionstring", "target"):
            pred_det_df = pred_det_df.rename(columns={c: "PredictionString"})
            break
if "PredictionString" not in pred_det_df.columns:
    pred_det_df["PredictionString"] = NORMAL

pred_det_df["PredictionString"] = pred_det_df["PredictionString"].fillna("").astype(str)
pred_det_df.loc[
    pred_det_df["PredictionString"].str.strip().eq(""), "PredictionString"
] = NORMAL

n_normal_before = int((pred_det_df["PredictionString"].values == NORMAL).sum())

merged_df = pd.merge(pred_det_df, pred_2class, on="image_id", how="left")

if "target" in merged_df.columns and "class0" not in merged_df.columns:
    merged_df["class0"] = 1 - merged_df["target"]

if "class0" not in merged_df.columns:
    merged_df["class0"] = 1.0
merged_df["class0"] = merged_df["class0"].fillna(1.0).astype(float)

low_threshold = 0.0
high_threshold = 0.976

p0 = merged_df["class0"].to_numpy(dtype=float)
mask_keep = p0 < low_threshold
mask_add = (p0 >= low_threshold) & (p0 < high_threshold)
mask_replace = p0 >= high_threshold

c0 = int(mask_keep.sum())
c1 = int(mask_add.sum())
c2 = int(mask_replace.sum())

base = merged_df["PredictionString"].astype(str).str.strip()
base = base.mask(base.eq(""), NORMAL)

if c1:
    conf_series = pd.Series(p0[mask_add], index=merged_df.index[mask_add]).map(
        lambda v: f"{float(v):.4f}"
    )
    sep = base.loc[mask_add].map(lambda s: "" if (s.endswith(" ") or s == "") else " ")
    merged_df.loc[mask_add, "PredictionString"] = (
        base.loc[mask_add] + sep + "14 " + conf_series + " 0 0 1 1"
    )

merged_df.loc[mask_replace, "PredictionString"] = NORMAL

merged_df["PredictionString"] = merged_df["PredictionString"].fillna("").astype(str)
merged_df.loc[merged_df["PredictionString"].str.strip().eq(""), "PredictionString"] = (
    NORMAL
)

n_normal_after = int((merged_df["PredictionString"].values == NORMAL).sum())
print(
    f"n_normal: {n_normal_before} -> {n_normal_after} with threshold {low_threshold} & {high_threshold}"
)
print(f"Keep {c0} Add {c1} Replace {c2}")

submission_filepath = "submission.csv"

submission_df = merged_df[["image_id", "PredictionString"]].copy()

submission_df = (
    sample_df[["image_id"]]
    .merge(submission_df, on="image_id", how="left")
    .assign(PredictionString=lambda d: d["PredictionString"].fillna(NORMAL).astype(str))
)

submission_df.loc[
    submission_df["PredictionString"].str.strip().eq(""), "PredictionString"
] = NORMAL

submission_df.to_csv(submission_filepath, index=False)

print(f"Saved to {submission_filepath}")
print(submission_df.head())
print("Submission columns:", submission_df.columns.tolist())
print(
    "Any empty PredictionString:",
    bool(submission_df["PredictionString"].astype(str).str.strip().eq("").any()),
)
print("Row count:", len(submission_df), "Expected:", len(sample_df))
