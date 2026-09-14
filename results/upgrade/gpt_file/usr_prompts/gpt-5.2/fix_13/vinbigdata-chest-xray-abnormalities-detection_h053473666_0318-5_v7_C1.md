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

0.2279101660552051

# 6. Current score

0.01307

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0475) has done: 'Your notebook currently fails because it tries to read several ensemble submission files that do not exist in this Kaggle environment, which prevents `df/df2/df3` from ever being created and cascades into `NameError`s. I replace those missing inputs with a safe, environment-available baseline built from `sample_submission.csv`, then keep your existing post-processing logic structure but make it robust to the actual 1500-row test set (instead of hardcoded 3000) and to empty/NaN prediction strings. Finally, I ensure the output submission has the exact required columns (`image_id`, `PredictionString`) and is saved as `submission.csv` in the working directory.'
- What this solution (achieved 0.0094) has done: 'Your current pipeline always starts from `sample_submission` (mostly “No finding”) and never produces any positive boxes, which caps mAP very low; to move toward the target score, the smallest legitimate improvement is to inject simple, dataset-derived “prior” boxes from `train.csv` (per class median box), then output a small number of high-priority class predictions for every test image. This preserves your overall submission-building logic (string-based postprocessing + “No finding” fallback) while making predictions non-trivial and valid for the metric. I also keep your existing filtering/merging structure but make it robust to the added predictions and ensure the final CSV columns/format match the competition requirements. The changes are intentionally minimal and deterministic, and they should raise the score toward your target without introducing new libraries or changing any ML training logic.'
- What this solution (achieved 0.01856) has done: 'Your current submission is hurt mostly by two post-processing steps that (a) duplicate the same predictions twice (concatenation of `df4` and `df5`) and (b) never apply the intended per-class score boosting because all class score columns are ~0.001, so the `0.92` gate is never reached. I keep your “train-derived prior boxes + fixed top classes per image” core approach, but make two minimal changes: avoid duplicating predictions and correctly populate the per-class columns from the confidences you already defined so the later calibration logic can actually act. This should raise mAP meaningfully from 0.0094 toward your target while preserving your overall string-based submission-building logic and the required output format. The resulting code still runs end-to-end and writes `submission.csv` with `image_id,PredictionString`.'
- What this solution (achieved 0.03473) has done: 'Your current score (0.01856) is far below the target (0.2279), so we should legitimately increase mAP with minimal, low-risk changes while keeping your “train-derived prior boxes + fixed classes per image + string postprocess” core logic intact. The biggest issue is that you predict the same 5 classes for every image and then explicitly strip out class 14, which tends to flood the evaluator with false positives; instead, we conditionally output “No finding” for images that look like they should have no boxes using only train-derived priors (no new model). Concretely, we (1) build a per-class prior confidence from train frequency, (2) reduce the number of per-image predicted classes (top-K) to cut false positives, and (3) add a deterministic per-image gate that outputs only `14 1 0 0 1 1` for a fixed fraction of images to better match the dataset’s no-finding prevalence—this typically raises mAP substantially versus “always positive”. All paths and output format stay the same, and we still write a valid `submission.csv`.'
- What this solution (achieved 0.02723) has done: 'We keep your “train-derived priors + fixed top classes + deterministic no-finding gate + string postprocess” core logic, but adjust two parameters that most directly control the false-positive/false-negative balance that mAP is highly sensitive to. Your current `NO_FINDING_RATE=0.55` is likely suppressing too many positives and capping recall; we reduce it toward the dataset’s typical no-finding prevalence so more images emit boxes. To avoid overcorrecting with too many false positives, we keep `TOP_K=2` (minimal change) but slightly recalibrate confidences upward based on class frequency so detections rank better without changing the model/feature logic. These are small, deterministic edits that should move the score upward toward your target.'
- What this solution (achieved 0.02201) has done: 'Your current score (0.02723) is far below the target (0.2279), so we should increase mAP with the smallest changes that mainly improve recall without exploding false positives. The biggest controllable lever in your logic is the deterministic “No finding” gate: at 0.30 it likely suppresses too many true-positive images, so we reduce it to emit boxes more often. To keep false positives in check while improving recall, we keep `TOP_K=2` unchanged but slightly raise the base confidence scale so your few emitted detections rank better in AP. All other submission-building/postprocessing remains intact, and we still write a valid `submission.csv` with `image_id,PredictionString`.'
- What this solution (achieved 0.02331) has done: 'Your current gap to the target is large (0.02201 vs 0.2279, higher-is-better), so we should increase mAP with the smallest change that improves recall without exploding false positives. The biggest limiter in your logic is that each predicted box uses a class-median box that can be far from the true object size/location, hurting IoU>0.4; we can keep your “train-derived priors + fixed top-K classes + deterministic no-finding gate” approach but switch the prior boxes to be larger, more “covering” boxes (train quantile-based per-class boxes), which should raise the chance of crossing IoU>0.4. We also make TOP_K=3 (small increase in recall) while slightly lowering per-box confidence to reduce the FP penalty, keeping everything deterministic and submission formatting unchanged. No new libraries or training loops are introduced, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.0058) has done: 'We make two minimal, metric-relevant tweaks to move your mAP up toward the target: (1) stop using the same class-quantile box for every image/class by switching to a simple per-class **union** box (min of mins / max of maxes) from `train.csv`, which increases the chance of IoU>0.4 without changing your “train-derived priors + top-K classes + deterministic no-finding gate” core logic; and (2) slightly reduce per-box confidence to mitigate false-positive ranking harm when using larger boxes. Everything else (no-finding gate, top-K predictions, submission string building, and CSV writing) stays the same and still produces `submission.csv` end-to-end.'
- What this solution (achieved 0.02122) has done: 'Your current score is far below the target, and the biggest limiter is that you’re emitting the same huge “union” boxes for a few classes, which almost never reaches IoU>0.4, collapsing mAP. I keep your exact “train-derived priors + deterministic no-finding gate + top-K classes + string postprocess” core logic, but switch the per-class prior from an extreme union box to a more robust quantile-based box (still derived only from `train.csv`), which materially increases the chance of IoU>0.4. I also slightly widen the deterministic coverage (reduce `NO_FINDING_RATE`) to improve recall toward the target while keeping TOP_K and the rest of your pipeline unchanged. The submission writing stays identical (`submission.csv` with `image_id,PredictionString`).'
- What this solution (achieved 0.01573) has done: 'Your current score (0.02122) is far below the target (0.2279), so we should increase mAP with the smallest, low-risk change that improves IoU>0.4 without changing your overall “train-derived priors + deterministic no-finding gate + top-K classes + string postprocess” structure. The main issue is that your per-class box uses fixed image-coordinate quantiles that often don’t align with actual object scale; we can keep the exact approach but compute quantile boxes in a **scale-normalized** way (relative to each image size inferred from train boxes), then map them back to absolute pixels—this tends to produce more realistic boxes and improves the chance of IoU>0.4. Additionally, we keep TOP_K=3 but modestly raise confidences (still frequency-based) so true positives rank higher, and we slightly lower NO_FINDING_RATE to improve recall while remaining deterministic. All submission formatting and downstream post-processing remain unchanged, and the script still writes `submission.csv`.'
- What this solution (achieved 0.01201) has done: 'Your current score is far below the target (0.0157 vs 0.2279), so we should increase mAP with the smallest changes that mainly improve IoU>0.4 while keeping your “train-derived prior boxes + deterministic no-finding gate + top-K classes + string postprocess” structure intact. The biggest issue is that you reuse a single global box scale (global median width/height) for all images, which makes many predicted boxes badly scaled and hurts IoU; we can instead build per-class normalized boxes and then map them to a per-image estimated width/height derived deterministically from `image_id` using the train distribution (no new model, no extra data). To avoid a flood of false positives, we keep TOP_K unchanged but slightly increase NO_FINDING_RATE to better match the dataset’s substantial no-finding share, and we also lower the calibration gate in cell 6 so your existing confidence-adjustment logic actually triggers with your ~0.2 confidences. These are minimal, metric-relevant tweaks that preserve your pipeline and still write a valid `submission.csv`.'
- What this solution (achieved 0.01307) has done: 'Your current score is far below the target, so we should increase mAP with minimal, metric-relevant tweaks that preserve your “train-derived prior boxes + deterministic no-finding gate + top-K classes + string postprocess” structure. The biggest controllable issue is that boxes are generated in a pseudo image-size space and can end up poorly scaled/placed, crushing IoU>0.4; we can instead anchor the normalized class box onto a per-image “center” derived deterministically from train (median normalized center), while also using a more conservative quantile box (25–75%) to avoid overly large boxes. Additionally, the current confidences (~0.18–0.34) almost never pass `CALIB_GATE=0.22` for some classes; lowering the gate slightly and blending less aggressively let your existing calibration step work without changing semantics. Finally, we keep TOP_K=3 and NO_FINDING_RATE as-is to avoid destabilizing the FP/FN balance; all changes are deterministic and still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import hashlib

BASE_DIR = "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection"
SAMPLE_SUB_PATH = os.path.join(BASE_DIR, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(BASE_DIR, "train.csv")

if not os.path.exists(SAMPLE_SUB_PATH):
    SAMPLE_SUB_PATH = "/kaggle/input/sample_submission.csv"
if not os.path.exists(TRAIN_CSV_PATH):
    TRAIN_CSV_PATH = "/kaggle/input/train.csv"

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
if "PredictionString" not in sample_sub.columns and "TARGET" in sample_sub.columns:
    sample_sub = sample_sub.rename(columns={"TARGET": "PredictionString"})
if "image_id" not in sample_sub.columns and "ID" in sample_sub.columns:
    sample_sub = sample_sub.rename(columns={"ID": "image_id"})

df = sample_sub[["image_id", "PredictionString"]].copy()
df["PredictionString"] = df["PredictionString"].fillna("14 1 0 0 1 1")

train_df = pd.read_csv(TRAIN_CSV_PATH)

train_pos = train_df[(train_df["class_id"] >= 0) & (train_df["class_id"] <= 13)].copy()
for c in ["x_min", "y_min", "x_max", "y_max"]:
    train_pos[c] = pd.to_numeric(train_pos[c], errors="coerce")
train_pos = train_pos.dropna(subset=["class_id", "x_min", "y_min", "x_max", "y_max"])

img_wh = train_pos.groupby("image_id", sort=False).agg(
    img_w=("x_max", "max"),
    img_h=("y_max", "max"),
)
train_pos = train_pos.merge(img_wh, on="image_id", how="left")
train_pos["img_w"] = train_pos["img_w"].clip(lower=1.0)
train_pos["img_h"] = train_pos["img_h"].clip(lower=1.0)

train_pos["nx_min"] = (train_pos["x_min"] / train_pos["img_w"]).clip(0.0, 1.0)
train_pos["ny_min"] = (train_pos["y_min"] / train_pos["img_h"]).clip(0.0, 1.0)
train_pos["nx_max"] = (train_pos["x_max"] / train_pos["img_w"]).clip(0.0, 1.0)
train_pos["ny_max"] = (train_pos["y_max"] / train_pos["img_h"]).clip(0.0, 1.0)

train_pos["ncx"] = ((train_pos["nx_min"] + train_pos["nx_max"]) * 0.5).clip(0.0, 1.0)
train_pos["ncy"] = ((train_pos["ny_min"] + train_pos["ny_max"]) * 0.5).clip(0.0, 1.0)

global_box = {
    "x_min": float(train_pos["x_min"].min()),
    "y_min": float(train_pos["y_min"].min()),
    "x_max": float(train_pos["x_max"].max()),
    "y_max": float(train_pos["y_max"].max()),
}

w_q10, w_q50, w_q90 = (img_wh["img_w"].quantile(q) for q in (0.10, 0.50, 0.90))
h_q10, h_q50, h_q90 = (img_wh["img_h"].quantile(q) for q in (0.10, 0.50, 0.90))
w_q10, w_q50, w_q90 = float(w_q10), float(w_q50), float(w_q90)
h_q10, h_q50, h_q90 = float(h_q10), float(h_q50), float(h_q90)


def _hash_u01(s: str) -> float:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _interp3(u: float, a: float, b: float, c: float) -> float:
    if u <= 0.5:
        t = u / 0.5
        return a * (1 - t) + b * t
    t = (u - 0.5) / 0.5
    return b * (1 - t) + c * t


def _img_size_from_id(image_id: str) -> tuple[float, float]:
    u1 = _hash_u01(image_id + "_w")
    u2 = _hash_u01(image_id + "_h")
    w = _interp3(u1, w_q10, w_q50, w_q90)
    h = _interp3(u2, h_q10, h_q50, h_q90)
    return max(1.0, w), max(1.0, h)


def _quantile_box_norm(g: pd.DataFrame, qlo: float = 0.25, qhi: float = 0.75) -> dict:
    x1 = float(g["nx_min"].quantile(qlo))
    y1 = float(g["ny_min"].quantile(qlo))
    x2 = float(g["nx_max"].quantile(qhi))
    y2 = float(g["ny_max"].quantile(qhi))
    return {"nx_min": x1, "ny_min": y1, "nx_max": x2, "ny_max": y2}


def _center_norm(g: pd.DataFrame) -> dict:
    return {
        "ncx": float(g["ncx"].median()),
        "ncy": float(g["ncy"].median()),
    }


class_box_norm = (
    train_pos.groupby("class_id", sort=True).apply(_quantile_box_norm).to_dict()
)

class_center_norm = (
    train_pos.groupby("class_id", sort=True).apply(_center_norm).to_dict()
)

class_counts = train_pos["class_id"].value_counts().sort_values(ascending=False)
top_classes = class_counts.head(5).index.tolist()

freq = (class_counts / class_counts.max()).to_dict()

class_conf = {
    int(cid): float(0.18 + 0.16 * freq.get(int(cid), 0.0)) for cid in top_classes
}


def _box_for_class(image_id: str, cid: int):
    b = class_box_norm.get(int(cid), None)
    cen = class_center_norm.get(int(cid), None)

    if b is None:
        x1 = float(global_box["x_min"])
        y1 = float(global_box["y_min"])
        x2 = float(global_box["x_max"])
        y2 = float(global_box["y_max"])
        return max(0.0, x1), max(0.0, y1), max(x1 + 1.0, x2), max(y1 + 1.0, y2)

    w, h = _img_size_from_id(image_id)

    nx1, ny1, nx2, ny2 = (
        float(b["nx_min"]),
        float(b["ny_min"]),
        float(b["nx_max"]),
        float(b["ny_max"]),
    )
    bw = max(1e-6, nx2 - nx1)
    bh = max(1e-6, ny2 - ny1)

    if cen is None:
        ncx = (nx1 + nx2) * 0.5
        ncy = (ny1 + ny2) * 0.5
    else:
        ncx = float(cen["ncx"])
        ncy = float(cen["ncy"])

    jx = (_hash_u01(image_id + f"_{cid}_jx") - 0.5) * 0.04
    jy = (_hash_u01(image_id + f"_{cid}_jy") - 0.5) * 0.04
    ncx = float(np.clip(ncx + jx, 0.0, 1.0))
    ncy = float(np.clip(ncy + jy, 0.0, 1.0))

    nx1a = float(np.clip(ncx - bw * 0.5, 0.0, 1.0))
    ny1a = float(np.clip(ncy - bh * 0.5, 0.0, 1.0))
    nx2a = float(np.clip(ncx + bw * 0.5, 0.0, 1.0))
    ny2a = float(np.clip(ncy + bh * 0.5, 0.0, 1.0))

    x1 = nx1a * w
    y1 = ny1a * h
    x2 = nx2a * w
    y2 = ny2a * h

    if x2 <= x1:
        x2 = x1 + 1.0
    if y2 <= y1:
        y2 = y1 + 1.0
    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = max(x1 + 1.0, x2)
    y2 = max(y1 + 1.0, y2)
    return x1, y1, x2, y2


TOP_K = 3
top_classes_k = top_classes[:TOP_K]

NO_FINDING_RATE = 0.12  # keep as-is for stability


def _should_no_finding(image_id: str, rate: float = NO_FINDING_RATE) -> bool:
    u = _hash_u01(image_id)
    return u < rate


def build_prediction_string(image_id: str):
    if _should_no_finding(image_id):
        return "14 1 0 0 1 1"
    parts = []
    for cid in top_classes_k:
        x1, y1, x2, y2 = _box_for_class(image_id, int(cid))
        conf = class_conf.get(int(cid), 0.18)
        parts.extend(
            [
                str(int(cid)),
                f"{conf:.5f}",
                f"{x1:.1f}",
                f"{y1:.1f}",
                f"{x2:.1f}",
                f"{y2:.1f}",
            ]
        )
    return " ".join(parts) if len(parts) else "14 1 0 0 1 1"


for k in range(15):
    df[str(k)] = 0.001
df["14"] = 0.55

for cid, conf in class_conf.items():
    df[str(int(cid))] = float(conf)

df["PredictionString"] = [build_prediction_string(iid) for iid in df["image_id"].values]

df1 = df.copy()
df_densenet = df.copy()

df[[str(i) for i in range(15)]] = (
    df[[str(i) for i in range(15)]] * 0.25
    + df1[[str(i) for i in range(15)]] * 0.5
    + df_densenet[[str(i) for i in range(15)]] * 0.25
)



## === cell 1
df2 = sample_sub[["image_id", "PredictionString"]].copy()
df3 = sample_sub[["image_id", "PredictionString"]].copy()

df2["PredictionString"] = df["PredictionString"].values
df3["PredictionString"] = df["PredictionString"].values

df2["PredictionString"] = df2["PredictionString"].fillna("14 1 0 0 1 1")
df3["PredictionString"] = df3["PredictionString"].fillna("14 1 0 0 1 1")



## === cell 2
df4 = pd.merge(df, df3, on="image_id", how="left", suffixes=("", "_y"))

if "PredictionString_y" in df4.columns:
    df4["PredictionString"] = df4["PredictionString"].fillna(df4["PredictionString_y"])
    df4 = df4.drop(columns=["PredictionString_y"])

df4["PredictionString"] = df4["PredictionString"].fillna("14 1 0 0 1 1")



## === cell 3
n_rows = df4.shape[0]
for i in range(n_rows):
    ps = df4.loc[i, "PredictionString"]
    if not isinstance(ps, str) or ps.strip() == "":
        df4.loc[i, "PredictionString"] = "14 1 0 0 1 1"
        continue

    list2 = ps.split()
    if len(list2) % 6 != 0:
        df4.loc[i, "PredictionString"] = "14 1 0 0 1 1"
        continue

    h_parts = []
    n_objs = len(list2) // 6
    for j in range(n_objs):
        cls = list2[0 + 6 * j]
        if cls in {"14"}:
            h_parts.extend(list2[0 + 6 * j : 6 + 6 * j])
        else:
            h_parts.extend(list2[0 + 6 * j : 6 + 6 * j])

    if len(h_parts) == 0:
        df4.loc[i, "PredictionString"] = "14 1 0 0 1 1"
    else:
        df4.loc[i, "PredictionString"] = " ".join(h_parts)

for i in range(n_rows):
    ps = df4.loc[i, "PredictionString"]
    if not isinstance(ps, str) or ps.strip() == "":
        df4.loc[i, "PredictionString"] = "14 1 0 0 1 1"
    else:
        df4.loc[i, "PredictionString"] = " ".join(ps.split())



## === cell 4
df5 = pd.merge(df, df2, on="image_id", how="left", suffixes=("", "_y"))

if "PredictionString_y" in df5.columns:
    df5["PredictionString"] = df5["PredictionString"].fillna(df5["PredictionString_y"])
    df5 = df5.drop(columns=["PredictionString_y"])

df5["PredictionString"] = df5["PredictionString"].fillna("14 1 0 0 1 1")



## === cell 5
df4["PredictionString"] = df4["PredictionString"].fillna("").astype(str).str.strip()
df4.loc[df4["PredictionString"].eq(""), "PredictionString"] = "14 1 0 0 1 1"



## === cell 6
list1 = list(range(15))

CALIB_GATE = 0.16  # was 0.22

for i in range(df4.shape[0]):
    if df4.loc[i, "PredictionString"] == "14 1 0 0 1 1":
        continue

    a = df4.loc[i, "PredictionString"]
    if not isinstance(a, str) or a.strip() == "":
        df4.loc[i, "PredictionString"] = "14 1 0 0 1 1"
        continue

    b = a.split()
    if len(b) % 6 != 0:
        df4.loc[i, "PredictionString"] = "14 1 0 0 1 1"
        continue

    n_objs = len(b) // 6
    for j in range(n_objs):
        try:
            cls_id = int(float(b[0 + 6 * j]))
        except Exception:
            continue

        for k in list1:
            if cls_id == k:
                col = str(k)
                if col not in df4.columns:
                    continue
                if float(df4.loc[i, col]) < CALIB_GATE:
                    continue
                c = b[0 + 6 * j + 1]
                try:
                    b[0 + 6 * j + 1] = str(
                        float(df4.loc[i, col]) * 0.55 + float(c) * 0.45
                    )
                except Exception:
                    pass

    df4.loc[i, "PredictionString"] = " ".join(b)



## === cell 7
pass



## === cell 8
df4["PredictionString"] = df4["PredictionString"].fillna("").astype(str).str.strip()
df4.loc[df4["PredictionString"].eq(""), "PredictionString"] = "14 1 0 0 1 1"

df_final = df4[["image_id", "PredictionString"]].copy()
df_final.to_csv("submission.csv", index=False)



## === cell 9
df_final
