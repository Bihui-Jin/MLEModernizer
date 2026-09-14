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

0.225945608732547

# 6. Current score

0.04891

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0475) has done: 'Your notebook is failing because it tries to ensemble multiple external submissions from `../input/...` paths that do not exist in this environment, so nothing defines `df/df2/df3/...` and all later cells crash. I keep the “core logic” intent (produce a valid `image_id,PredictionString` submission) but replace the missing-input ensemble with a robust fallback that uses the provided `sample_submission.csv` as the base and guarantees correct formatting for every test image. This run end-to-end, write `submission.csv` with the required columns, and avoid runtime errors. Since there is no current score, the priority is producing a valid submission; the fallback predicts “No finding” for all images (a common baseline), which is score-safe and valid.'
- What this solution (achieved 0.01849) has done: 'Your current score (0.0475) is far below the target (0.2259), so we should improve performance with minimal but meaningful changes while keeping the “predict strings per image” core logic intact. The smallest legitimate upgrade is to replace the all–“No finding” baseline with a simple per-class, per-image heuristic learned from `train.csv`: for each class, predict a small number of highest-prior classes per image using their empirical frequency, and use typical (median) boxes per class from the training annotations. This stays within the same submission semantics (just generating `PredictionString`) and should lift mAP above the no-finding baseline without introducing new model architectures or training loops. We also keep a conservative confidence scheme and always fall back to “No finding” if something goes wrong for an image.'
- What this solution (achieved 0.06606) has done: 'You’re far below the target mAP, so the smallest meaningful improvement is to make the heuristic less “one-size-fits-all” without changing the overall approach (still: learn simple statistics from `train.csv` and emit `PredictionString`). Concretely, we (1) build **per-class median boxes normalized by image size** and then scale to each test image’s DICOM dimensions, which is a direct fix for mismatch between fixed median pixels and varying image resolutions, and (2) predict a **small set of top classes** but with a slightly higher K and **confidence calibrated by class prior**, while also appending a low-confidence “No finding” to reduce harm on empty images. This keeps the same core submission semantics (string generation, no model training) but should move the score upward toward your target. The code also remains end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.04891) has done: 'Your current score (0.06606) is far below the target (0.22595), so we should improve mAP with the smallest changes that keep your “train.csv statistics → PredictionString” core logic intact. The most direct weakness is using a single median box per class, which rarely matches the true object location; we can improve localization without changing approach by emitting a few “typical” boxes per class (quantiles of centers/sizes) and slightly lowering per-box confidence to reduce false-positive damage. We also restrict predictions to the most frequent classes (smaller K) and avoid always appending “No finding” (only add it when overall evidence is weak), which usually helps mAP in this competition. All changes keep the same pipeline, still no model training, still produce a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

BASE_INPUT = "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection"
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(BASE_INPUT, "train.csv")

if not os.path.exists(SAMPLE_SUB_PATH):
    SAMPLE_SUB_PATH = "/kaggle/input/sample_submission.csv"
if not os.path.exists(TRAIN_CSV_PATH):
    TRAIN_CSV_PATH = "/kaggle/input/train.csv"

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

if "image_id" not in sample_sub.columns:
    raise ValueError(
        f"sample_submission.csv missing 'image_id' column. Columns: {list(sample_sub.columns)}"
    )

if "PredictionString" not in sample_sub.columns:
    if len(sample_sub.columns) >= 2:
        sample_sub = sample_sub.rename(
            columns={
                sample_sub.columns[0]: "image_id",
                sample_sub.columns[1]: "PredictionString",
            }
        )
    else:
        raise ValueError(
            f"sample_submission.csv has unexpected columns: {list(sample_sub.columns)}"
        )

sample_sub.head()



## === cell 1
NO_FINDING_STR = "14 1 0 0 1 1"

train = pd.read_csv(TRAIN_CSV_PATH)

if ("width" not in train.columns) or ("height" not in train.columns):
    wh = (
        train.groupby("image_id", sort=False)[["x_max", "y_max"]]
        .max()
        .rename(columns={"x_max": "width", "y_max": "height"})
    )
    wh["width"] = (wh["width"].fillna(0).astype(float) + 1.0).clip(lower=1.0)
    wh["height"] = (wh["height"].fillna(0).astype(float) + 1.0).clip(lower=1.0)
    train = train.merge(wh, left_on="image_id", right_index=True, how="left")

HAS_W_H = ("width" in train.columns) and ("height" in train.columns)

train_find = train[train["class_id"].between(0, 13)].copy()
class_counts = train_find["class_id"].value_counts().sort_index()
total = int(class_counts.sum())
class_prior = (class_counts / max(total, 1)).to_dict()


def _clip_box(xmin, ymin, xmax, ymax, W, H):
    xmin = float(np.clip(xmin, 0.0, max(0.0, W - 1.0)))
    ymin = float(np.clip(ymin, 0.0, max(0.0, H - 1.0)))
    xmax = float(np.clip(xmax, xmin + 1.0, max(1.0, W)))
    ymax = float(np.clip(ymax, ymin + 1.0, max(1.0, H)))
    return xmin, ymin, xmax, ymax


def _boxes_from_center_size_quantiles(df, qs=(0.2, 0.5, 0.8)):
    w = df["width"].astype(float).replace(0, np.nan)
    h = df["height"].astype(float).replace(0, np.nan)

    x1 = (df["x_min"].astype(float) / w).clip(0.0, 1.0)
    y1 = (df["y_min"].astype(float) / h).clip(0.0, 1.0)
    x2 = (df["x_max"].astype(float) / w).clip(0.0, 1.0)
    y2 = (df["y_max"].astype(float) / h).clip(0.0, 1.0)

    cx = ((x1 + x2) * 0.5).clip(0.0, 1.0)
    cy = ((y1 + y2) * 0.5).clip(0.0, 1.0)
    bw = (x2 - x1).clip(1e-6, 1.0)
    bh = (y2 - y1).clip(1e-6, 1.0)

    out = []
    for q in qs:
        cxi = float(np.nanquantile(cx, q))
        cyi = float(np.nanquantile(cy, q))
        bwi = float(np.nanquantile(bw, q))
        bhi = float(np.nanquantile(bh, q))

        xmin = float(np.clip(cxi - 0.5 * bwi, 0.0, 1.0))
        ymin = float(np.clip(cyi - 0.5 * bhi, 0.0, 1.0))
        xmax = float(np.clip(cxi + 0.5 * bwi, 0.0, 1.0))
        ymax = float(np.clip(cyi + 0.5 * bhi, 0.0, 1.0))

        xmin = float(np.clip(min(xmin, xmax - 1e-6), 0.0, 1.0))
        ymin = float(np.clip(min(ymin, ymax - 1e-6), 0.0, 1.0))
        xmax = float(np.clip(max(xmax, xmin + 1e-6), 0.0, 1.0))
        ymax = float(np.clip(max(ymax, ymin + 1e-6), 0.0, 1.0))
        out.append((xmin, ymin, xmax, ymax))
    return out


if not HAS_W_H:
    raise ValueError(
        "Failed to create width/height for train; cannot compute normalized boxes."
    )

boxes_norm_by_class = (
    train_find.groupby("class_id", sort=True)
    .apply(_boxes_from_center_size_quantiles)
    .to_dict()
)

K = 4
top_classes = [
    int(c) for c in class_counts.sort_values(ascending=False).head(K).index.tolist()
]

NBOX = 3  # number of typical boxes per class (quantiles), kept small for submission string size


def conf_from_prior(p):
    p = float(p)
    return float(np.clip(0.12 + 0.38 * np.sqrt(p), 0.12, 0.55))


top_base_confs = {c: conf_from_prior(class_prior.get(c, 0.0)) for c in top_classes}

NO_FINDING_CONF = 0.18
NO_FINDING_ADD_IF_MAXCONF_BELOW = 0.20

test_dir = os.path.join(BASE_INPUT, "test")
if not os.path.exists(test_dir):
    test_dir = "/kaggle/input/test"

try:
    import pydicom  # available on Kaggle for this competition environment
except Exception:
    pydicom = None


def get_test_hw(image_id):
    if pydicom is None:
        return None
    fp = os.path.join(test_dir, f"{image_id}.dicom")
    if not os.path.exists(fp):
        fp2 = os.path.join(test_dir, f"{image_id}.dcm")
        fp = fp2 if os.path.exists(fp2) else fp
    try:
        ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
        H = int(getattr(ds, "Rows", 0) or 0)
        W = int(getattr(ds, "Columns", 0) or 0)
        if H > 0 and W > 0:
            return H, W
    except Exception:
        return None
    return None


pred_strings = []
for _img_id in sample_sub["image_id"].values:
    parts = []

    hw = get_test_hw(_img_id)
    if hw is None:
        H, W = 1024.0, 1024.0
    else:
        H, W = float(hw[0]), float(hw[1])

    max_conf_emitted = 0.0

    for c in top_classes:
        base_conf = float(top_base_confs.get(c, 0.18))
        per_box_conf = float(np.clip(base_conf / NBOX, 0.05, 0.35))

        boxes = boxes_norm_by_class.get(c, [(0.0, 0.0, 1.0, 1.0)])
        for xmin_n, ymin_n, xmax_n, ymax_n in boxes[:NBOX]:
            xmin, ymin, xmax, ymax = xmin_n * W, ymin_n * H, xmax_n * W, ymax_n * H
            xmin, ymin, xmax, ymax = _clip_box(xmin, ymin, xmax, ymax, W, H)
            parts.append(
                f"{c} {per_box_conf:.4f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
            )
            if per_box_conf > max_conf_emitted:
                max_conf_emitted = per_box_conf

    if max_conf_emitted < NO_FINDING_ADD_IF_MAXCONF_BELOW:
        parts.append(f"14 {NO_FINDING_CONF:.4f} 0 0 1 1")

    pred_str = " ".join(parts).strip()
    if len(pred_str) == 0:
        pred_str = NO_FINDING_STR
    pred_strings.append(pred_str)

df_final = sample_sub[["image_id"]].copy()
df_final["PredictionString"] = pred_strings

assert (
    df_final.shape[0] == sample_sub.shape[0]
), "Row count must match sample submission."
assert df_final["PredictionString"].notna().all(), "All predictions must be non-null."
assert (
    df_final["PredictionString"].astype(str).str.len() > 0
).all(), "All predictions must be non-empty strings."

df_final.head()



## === cell 2
df_final.to_csv("submission.csv", index=False)

print(df_final.head(3))
print("Wrote:", os.path.abspath("submission.csv"))
print("File exists:", os.path.exists("submission.csv"))
