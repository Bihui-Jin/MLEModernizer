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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)



## === cell 1
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
    All features are in [0,1]-like ranges via normalization by width/height where applicable.
    """
    pydicom = _try_import_pydicom()
    if pydicom is None:
        n = len(image_ids) if max_images is None else min(len(image_ids), max_images)
        return pd.DataFrame(
            {
                "image_id": np.asarray(image_ids[:n]),
                "px_mean": np.zeros(n, dtype=np.float32),
                "px_std": np.ones(n, dtype=np.float32),
                "com_x": np.full(n, 0.5, dtype=np.float32),
                "com_y": np.full(n, 0.5, dtype=np.float32),
            }
        )

    out_ids = []
    px_mean = []
    px_std = []
    com_x = []
    com_y = []

    n = len(image_ids) if max_images is None else min(len(image_ids), max_images)
    for k in range(n):
        iid = image_ids[k]
        fp = _dicom_path(split_dir, iid)
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=False, force=True)
            arr = ds.pixel_array.astype(np.float32)

            m = float(arr.mean()) if arr.size else 0.0
            s = float(arr.std()) if arr.size else 0.0

            a = arr - float(arr.min()) if arr.size else arr
            mass = float(a.sum()) if arr.size else 0.0
            if mass <= 1e-6 or arr.size == 0:
                cy, cx = 0.5, 0.5
            else:
                h, w = a.shape[:2]
                yy = np.arange(h, dtype=np.float32)[:, None]
                xx = np.arange(w, dtype=np.float32)[None, :]
                cy = float((a * yy).sum() / mass) / max(1.0, float(h - 1))
                cx = float((a * xx).sum() / mass) / max(1.0, float(w - 1))

            out_ids.append(iid)
            px_mean.append(m)
            px_std.append(s)
            com_x.append(cx)
            com_y.append(cy)
        except Exception:
            out_ids.append(iid)
            px_mean.append(0.0)
            px_std.append(1.0)
            com_x.append(0.5)
            com_y.append(0.5)

    df = pd.DataFrame(
        {
            "image_id": np.asarray(out_ids),
            "px_mean": np.asarray(px_mean, dtype=np.float32),
            "px_std": np.asarray(px_std, dtype=np.float32),
            "com_x": np.asarray(com_x, dtype=np.float32),
            "com_y": np.asarray(com_y, dtype=np.float32),
        }
    )
    return df


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
    anchor_ids = train_image_ids_all[:n_anchor]

    train_feat_px = _safe_dicom_features(
        anchor_ids.tolist(), split_dir="train", max_images=n_anchor
    )

    train_feat = pd.merge(
        train_feat_bbox.iloc[:n_anchor].copy(), train_feat_px, on="image_id", how="left"
    )

    feat_cols = [c for c in train_feat.columns if c != "image_id"]
    X = train_feat[feat_cols].to_numpy(dtype=np.float32)

    mu = X.mean(axis=0, keepdims=True)
    sig = X.std(axis=0, keepdims=True) + 1e-6
    Xn = (X - mu) / sig

    X_anchor = Xn  # already restricted to anchors

    cons_by_img = t_cons[t_cons["class_id"].isin(top_classes)].groupby("image_id")
    anchor_pred = {}
    MAX_BOXES_PER_IMAGE = 24  # slight increase; still strict cap

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

    K_NEIGHBORS = 5  # keep as-is for speed and to preserve approach

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

    pred_strings = []
    chunk = 200  # smaller to reduce peak memory with DICOM-derived features
    for start in range(0, len(test_image_ids), chunk):
        end = min(len(test_image_ids), start + chunk)
        T = Xtn[start:end]  # (m,d)
        d2 = ((T[:, None, :] - X_anchor[None, :, :]) ** 2).sum(axis=2)
        nn_idx = np.argpartition(d2, kth=min(K_NEIGHBORS, d2.shape[1] - 1), axis=1)[
            :, :K_NEIGHBORS
        ]
        for i in range(nn_idx.shape[0]):
            aids = anchor_ids[nn_idx[i]]
            tokens = []
            for aid in aids:
                s = anchor_pred.get(aid, "")
                if s and s != NORMAL:
                    tokens.append(s)
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

has_finding_per_image = train_df.groupby("image_id")["class_id"].apply(
    lambda s: (pd.to_numeric(s, errors="coerce").fillna(14).astype(int) != 14).any()
)
normal_prior = float((~has_finding_per_image).mean())  # P(normal)

pred_2class = pd.DataFrame(
    {"image_id": pred_det_df["image_id"].values, "class0": normal_prior}
)

low_threshold = 0.0
high_threshold = 0.976

print(f"Using BASE_DIR={BASE_DIR}")
print(
    f"Built pred_2class with constant class0(normal) prior={normal_prior:.6f} for {len(pred_2class)} test images"
)
print("Example PredictionString:")
print(pred_det_df.head(1))



## === cell 2
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

n_normal_before = len(pred_det_df.query("PredictionString == @NORMAL"))

merged_df = pd.merge(pred_det_df, pred_2class, on="image_id", how="left")

if "target" in merged_df.columns and "class0" not in merged_df.columns:
    merged_df["class0"] = 1 - merged_df["target"]

if "class0" not in merged_df.columns:
    merged_df["class0"] = 1.0
merged_df["class0"] = merged_df["class0"].fillna(1.0).astype(float)

c0, c1, c2 = 0, 0, 0
for i in range(len(merged_df)):
    p0 = float(merged_df.loc[i, "class0"])
    if p0 < low_threshold:
        c0 += 1
    elif low_threshold <= p0 and p0 < high_threshold:
        base = str(merged_df.loc[i, "PredictionString"]).strip()
        if base == "":
            base = NORMAL
        merged_df.loc[i, "PredictionString"] = (base + f" 14 {p0:.4f} 0 0 1 1").strip()
        c1 += 1
    else:
        merged_df.loc[i, "PredictionString"] = NORMAL
        c2 += 1

merged_df["PredictionString"] = merged_df["PredictionString"].fillna("").astype(str)
merged_df.loc[merged_df["PredictionString"].str.strip().eq(""), "PredictionString"] = (
    NORMAL
)

n_normal_after = len(merged_df.query("PredictionString == @NORMAL"))
print(
    f"n_normal: {n_normal_before} -> {n_normal_after} with threshold {low_threshold} & {high_threshold}"
)
print(f"Keep {c0} Add {c1} Replace {c2}")

submission_filepath = "submission.csv"
submission_df = merged_df[["image_id", "PredictionString"]].copy()
submission_df.to_csv(submission_filepath, index=False)
print(f"Saved to {submission_filepath}")
print(submission_df.head())
