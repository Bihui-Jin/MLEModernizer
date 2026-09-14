# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import pandas as pd


def _try_read_csv(path: str):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


paths_a = [f"../input/vin-15-cnn-predict/submission{i}.csv" for i in range(10)]
paths_b = [f"../input/vin-15-cnn-predict-1/submission{i}.csv" for i in range(5)]
all_paths = paths_a + paths_b

dfs = []
for p in all_paths:
    d = _try_read_csv(p)
    if d is not None:
        dfs.append(d)

base_path = (
    "../input/vinbigdata-chest-xray-abnormalities-detection/sample_submission.csv"
)
if not os.path.exists(base_path):
    base_path = "../input/sample_submission.csv"
sample_sub = pd.read_csv(base_path)

if "image_id" not in sample_sub.columns:
    if "ID" in sample_sub.columns:
        sample_sub = sample_sub.rename(columns={"ID": "image_id"})
if "PredictionString" not in sample_sub.columns:
    if "TARGET" in sample_sub.columns:
        sample_sub = sample_sub.rename(columns={"TARGET": "PredictionString"})
    else:
        sample_sub["PredictionString"] = ""

if len(dfs) >= 2 and all(set(["image_id"]).issubset(d.columns) for d in dfs):
    list1 = [str(i) for i in range(15)]
    df0 = dfs[0].copy()

    if all(col in df0.columns for col in (["image_id", "PredictionString"] + list1)):
        valid_dfs = [
            d
            for d in dfs
            if all(
                col in d.columns for col in (["image_id", "PredictionString"] + list1)
            )
        ]
        w = 1.0 / len(valid_dfs)
        df0[list1] = 0.0
        for d in valid_dfs:
            df0[list1] = df0[list1] + d[list1] * w
        df = df0.copy()
    else:
        df = sample_sub.copy()
        df["PredictionString"] = "14 1 0 0 1 1"
else:
    df = sample_sub.copy()
    df["PredictionString"] = "14 1 0 0 1 1"

df.head()



## === cell 1
df3_path = "../input/0284-norm/cascade_rcnn_x101_32x4d_fpn_20e_OHEM_fpn.4_with_raw.4_ensemble.5.bbox.json.filter.norm (1).csv"
df3 = _try_read_csv(df3_path)

if df3 is None:
    df3 = df[["image_id"]].copy()
    for k in range(15):
        df3[str(k)] = 0.0

df3.head()



## === cell 2
df4 = pd.merge(df, df3, on="image_id", how="left")

if "PredictionString" not in df4.columns:
    ps_cols = [c for c in df4.columns if c.startswith("PredictionString")]
    if ps_cols:
        df4["PredictionString"] = df4[ps_cols[0]]
    else:
        df4["PredictionString"] = "14 1 0 0 1 1"

df4.head()



## === cell 3
import numpy as np

score_cols = [str(k) for k in range(15)]
for col in score_cols:
    if col not in df4.columns:
        df4[col] = 0.0

df4[score_cols] = (
    df4[score_cols]
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0.0)
    .astype(np.float64, copy=False)
)

df4.iloc[0][["image_id", "PredictionString"] + [str(i) for i in range(5)]]



## === cell 4
train_path = "../input/vinbigdata-chest-xray-abnormalities-detection/train.csv"
if not os.path.exists(train_path):
    train_path = "../input/train.csv"

train_df = None
if os.path.exists(train_path):
    train_df = pd.read_csv(train_path)

fallback_boxes = {}
fallback_boxes_alt = {}  # (cid, alt_idx) -> (xmin,ymin,xmax,ymax)

if train_df is not None:
    t = train_df[["class_id", "x_min", "y_min", "x_max", "y_max"]].copy()

    t["class_id"] = pd.to_numeric(t["class_id"], errors="coerce")
    for c in ["x_min", "y_min", "x_max", "y_max"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")

    t = t.dropna(subset=["class_id", "x_min", "y_min", "x_max", "y_max"])
    t = t[(t["class_id"] >= 0) & (t["class_id"] <= 13)]
    t = t[(t["x_max"] > t["x_min"]) & (t["y_max"] > t["y_min"])]

    if len(t) > 0:
        x_low = float(t["x_min"].quantile(0.02))
        y_low = float(t["y_min"].quantile(0.02))
        x_high = float(t["x_max"].quantile(0.98))
        y_high = float(t["y_max"].quantile(0.98))

        g_med = t.groupby("class_id", sort=False)[
            ["x_min", "y_min", "x_max", "y_max"]
        ].median()

        def _clamp_box(xmin, ymin, xmax, ymax):
            xmin = max(x_low, float(xmin))
            ymin = max(y_low, float(ymin))
            xmax = min(x_high, float(xmax))
            ymax = min(y_high, float(ymax))
            if xmax <= xmin:
                xmax = xmin + 1.0
            if ymax <= ymin:
                ymax = ymin + 1.0
            return (xmin, ymin, xmax, ymax)

        for cid, row in g_med.iterrows():
            fallback_boxes[int(cid)] = _clamp_box(
                row["x_min"], row["y_min"], row["x_max"], row["y_max"]
            )

        for cid, ct in t.groupby("class_id", sort=False):
            if len(ct) < 10:
                continue
            q25 = ct[["x_min", "y_min", "x_max", "y_max"]].quantile(0.25)
            q75 = ct[["x_min", "y_min", "x_max", "y_max"]].quantile(0.75)
            fallback_boxes_alt[(int(cid), 0)] = _clamp_box(
                q25["x_min"], q25["y_min"], q25["x_max"], q25["y_max"]
            )
            fallback_boxes_alt[(int(cid), 1)] = _clamp_box(
                q75["x_min"], q75["y_min"], q75["x_max"], q75["y_max"]
            )

no_external_mode = False
if "PredictionString" in df4.columns:
    frac_default = (
        df4["PredictionString"].astype(str).fillna("").str.strip() == "14 1 0 0 1 1"
    ).mean()
    score_sum = df4[score_cols].to_numpy().sum()
    if frac_default > 0.95 and score_sum == 0.0:
        no_external_mode = True

if no_external_mode and len(fallback_boxes) > 0:
    from functools import lru_cache

    try:
        import pydicom  # type: ignore
        from pydicom.pixel_data_handlers.util import apply_voi_lut  # type: ignore

        _HAS_PYDICOM = True
    except Exception:
        _HAS_PYDICOM = False

    test_dir = "../input/vinbigdata-chest-xray-abnormalities-detection/test"
    if not os.path.exists(test_dir):
        test_dir = "../input/test"

    priors = None
    if train_df is not None and "class_id" in train_df.columns:
        tc = train_df[["class_id"]].copy()
        tc["class_id"] = pd.to_numeric(tc["class_id"], errors="coerce")
        tc = tc.dropna(subset=["class_id"])
        tc = tc[(tc["class_id"] >= 0) & (tc["class_id"] <= 13)]
        cnt = tc["class_id"].value_counts().sort_index()
        priors = {int(k): float(v) for k, v in cnt.items()}
        s = sum(priors.values()) if priors else 0.0
        if s > 0:
            priors = {k: v / s for k, v in priors.items()}
    if not priors:
        priors = {k: 1.0 / 14.0 for k in range(14)}

    rng = np.random.RandomState(0)

    def _fast_percentile_1_99(x: np.ndarray):
        n = x.size
        if n == 0:
            return 0.0, 1.0
        k1 = int(0.01 * (n - 1))
        k99 = int(0.99 * (n - 1))
        if k99 < k1:
            k99 = k1
        part = np.partition(x, (k1, k99))
        return float(part[k1]), float(part[k99])

    @lru_cache(maxsize=4096)
    def _sparse_img_stats(dcm_path: str, step: int = 16):
        ds = pydicom.dcmread(
            dcm_path,
            force=True,
            stop_before_pixels=False,
            specific_tags=[
                "PixelData",
                "PhotometricInterpretation",
                "BitsStored",
                "BitsAllocated",
                "HighBit",
                "PixelRepresentation",
                "RescaleIntercept",
                "RescaleSlope",
                "WindowCenter",
                "WindowWidth",
                "VOILUTSequence",
                "SamplesPerPixel",
                "PlanarConfiguration",
                "Rows",
                "Columns",
                "TransferSyntaxUID",
            ],
        )

        arr = ds.pixel_array
        try:
            arr = apply_voi_lut(arr, ds)
        except Exception:
            pass

        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)

        arr = arr[::step, ::step]

        flat = arr.reshape(-1)
        n = flat.size
        if n == 0:
            return 0.5, 0.2, 0.1, 0.1

        m = 4096 if n > 4096 else n
        if m < n:
            idx = (np.linspace(0, n - 1, m)).astype(np.int64)
            samp = flat[idx]
        else:
            samp = flat

        p1, p99 = _fast_percentile_1_99(samp)

        if p99 > p1:
            arr = (arr - p1) / (p99 - p1)
        arr = np.clip(arr, 0.0, 1.0)
        mean = float(arr.mean())
        std = float(arr.std())
        hi = float((arr > 0.75).mean())
        lo = float((arr < 0.25).mean())
        return mean, std, hi, lo

    class_stat_mu = {cid: None for cid in range(14)}
    class_stat_sig = {cid: None for cid in range(14)}
    global_mu = np.array([0.5, 0.2, 0.1, 0.1], dtype=np.float32)
    global_sig = np.array([0.15, 0.10, 0.08, 0.08], dtype=np.float32)

    if _HAS_PYDICOM and train_df is not None:
        train_dir = "../input/vinbigdata-chest-xray-abnormalities-detection/train"
        if not os.path.exists(train_dir):
            train_dir = "../input/train"

        timg = train_df[["image_id", "class_id"]].copy()
        timg["class_id"] = pd.to_numeric(timg["class_id"], errors="coerce")
        timg = timg.dropna(subset=["class_id"])
        timg = timg[(timg["class_id"] >= 0) & (timg["class_id"] <= 13)]
        timg["class_id"] = timg["class_id"].astype(int)
        timg = timg.drop_duplicates(subset=["class_id", "image_id"], keep="first")

        per_class_imgs = (
            timg.groupby("class_id", sort=False)["image_id"].apply(list).to_dict()
        )

        N_PER_CLASS = 18
        feats_by_class = {}

        for cid in range(14):
            imgs = per_class_imgs.get(cid, [])
            if len(imgs) == 0:
                continue
            if len(imgs) > N_PER_CLASS:
                imgs = rng.choice(imgs, size=N_PER_CLASS, replace=False)
            feats = []
            for im in imgs:
                p = os.path.join(train_dir, f"{im}.dicom")
                try:
                    feats.append(_sparse_img_stats(p, step=20))
                except Exception:
                    continue
            if len(feats) >= 6:
                X = np.array(feats, dtype=np.float32)
                feats_by_class[cid] = X
                class_stat_mu[cid] = X.mean(axis=0)
                class_stat_sig[cid] = X.std(axis=0) + 1e-6

        if feats_by_class:
            allX = np.concatenate(list(feats_by_class.values()), axis=0)
            global_mu = allX.mean(axis=0)
            global_sig = allX.std(axis=0) + 1e-6

    @lru_cache(maxsize=4096)
    def _img_hash01(image_id: str) -> float:
        x = 0
        for ch in str(image_id):
            x = (x * 131 + ord(ch)) % 1000003
        return (x % 1000000) / 1000000.0

    image_ids = df4["image_id"].astype(str).to_numpy()
    out_ps = [None] * len(image_ids)
    test_paths = [os.path.join(test_dir, f"{iid}.dicom") for iid in image_ids]

    def _get_feats(idx_path):
        idx, pth = idx_path
        try:
            return idx, _sparse_img_stats(pth, step=20)
        except Exception:
            return idx, None

    feats_arr = [None] * len(test_paths)
    if _HAS_PYDICOM:
        from concurrent.futures import ThreadPoolExecutor

        max_workers = min(8, (os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for idx, feats in ex.map(_get_feats, enumerate(test_paths), chunksize=16):
                feats_arr[idx] = feats

    _priors_get = priors.get
    _fb = fallback_boxes
    _fb_alt = fallback_boxes_alt
    _mu = class_stat_mu
    _sig = class_stat_sig
    _sqrt = np.sqrt
    _mean = np.mean
    _clip = np.clip
    _exp = np.exp

    for idx, image_id in enumerate(image_ids):
        h = _img_hash01(image_id)

        feats = feats_arr[idx]
        if feats is not None:
            feats = np.array(feats, dtype=np.float32)

        scores = []
        if feats is None:
            for cid in range(14):
                base = _priors_get(cid, 0.0)
                sc = base * (0.85 + 0.30 * ((h + (cid * 0.07)) % 1.0))
                scores.append(float(sc))
        else:
            for cid in range(14):
                base = _priors_get(cid, 0.0)
                if _mu[cid] is None:
                    sc = base * (0.85 + 0.30 * ((h + (cid * 0.07)) % 1.0))
                else:
                    mu = _mu[cid]
                    sig = _sig[cid]
                    z = (feats - mu) / sig
                    dist = float(_sqrt(_mean(_clip(z, -4, 4) ** 2)))
                    like = float(_exp(-0.9 * dist))  # [0,1]
                    sc = 0.55 * like + 0.45 * base
                scores.append(float(sc))

        cls_scores = [(cid, float(scores[cid])) for cid in range(14)]
        cls_scores.sort(key=lambda x: x[1], reverse=True)

        obj_mass = sum(s for _, s in cls_scores[:5])
        if obj_mass < 0.095:
            out_ps[idx] = "14 1 0 0 1 1"
            continue

        if obj_mass > 0.24:
            K = 5
        elif obj_mass > 0.17:
            K = 4
        else:
            K = 3

        parts = []
        picked = 0
        used_cids = []

        for cid, sc in cls_scores:
            if picked >= K:
                break
            if cid not in _fb:
                continue
            conf = max(0.05, min(0.75, sc * 1.9))
            xmin, ymin, xmax, ymax = _fb[cid]
            parts.extend(
                [
                    str(cid),
                    f"{conf:.4f}",
                    f"{xmin:.1f}",
                    f"{ymin:.1f}",
                    f"{xmax:.1f}",
                    f"{ymax:.1f}",
                ]
            )
            used_cids.append(cid)
            picked += 1

        if used_cids:
            top_cid = int(used_cids[0])
            alt_key = (top_cid, 0 if h < 0.5 else 1)
            if alt_key in _fb_alt and picked < (K + 1):
                xmin, ymin, xmax, ymax = _fb_alt[alt_key]
                primary_conf = float(parts[1]) if len(parts) >= 2 else 0.2
                alt_conf = max(0.05, min(0.65, primary_conf * 0.85))
                parts.extend(
                    [
                        str(top_cid),
                        f"{alt_conf:.4f}",
                        f"{xmin:.1f}",
                        f"{ymin:.1f}",
                        f"{xmax:.1f}",
                        f"{ymax:.1f}",
                    ]
                )

        out_ps[idx] = " ".join(parts) if parts else "14 1 0 0 1 1"

    df4["PredictionString"] = out_ps

df4.head()



## === cell 5
_ = None
if len(df4.columns) > 16 and len(df4) > 1:
    _ = df4.iloc[1, 16]
_



## === cell 6
list1 = set(range(15))
score_mat = df4[score_cols].to_numpy()  # row-aligned with df4
hi_mask = score_mat >= 0.99

ps_series = df4["PredictionString"].astype(object).tolist()
_default_ps = "14 1 0 0 1 1"
_float = float
_int = int
_join = " ".join

for i in range(df4.shape[0]):
    ps = ps_series[i]
    if ps == _default_ps or (not isinstance(ps, str)) or ps.strip() == "":
        continue
    b = ps.split()
    if len(b) % 6 != 0:
        continue
    row_scores = score_mat[i]
    row_hi = hi_mask[i]
    n = len(b) // 6
    for j in range(n):
        cls = b[6 * j]
        try:
            cls_int = _int(cls)
        except Exception:
            continue
        if cls_int in list1 and row_hi[cls_int]:
            scv = _float(row_scores[cls_int])
            try:
                c = _float(b[6 * j + 1])
                b[6 * j + 1] = str(scv * 0.1 + c * 0.9)
            except Exception:
                pass
    ps_series[i] = _join(b)

df4["PredictionString"] = ps_series
df4.head()



## === cell 7
df_final = df4[["image_id", "PredictionString"]].copy()

df_final["PredictionString"] = df_final["PredictionString"].fillna("").astype(str)
df_final.loc[df_final["PredictionString"].str.strip().eq(""), "PredictionString"] = (
    "14 1 0 0 1 1"
)

out_path = "submission.csv"
df_final.to_csv(out_path, index=False)

out_path, df_final.shape, df_final.head()



## === cell 8
df_final
