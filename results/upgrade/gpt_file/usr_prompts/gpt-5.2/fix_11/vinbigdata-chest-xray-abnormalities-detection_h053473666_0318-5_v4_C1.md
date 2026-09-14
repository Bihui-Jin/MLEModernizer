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

0.2277198942841179

# 6. Current score

0.0505

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0475) has done: 'I remove the broken dependencies on non-existent external submission files and instead build a valid submission directly from the provided `sample_submission.csv`, so the notebook runs end-to-end and always writes `submission.csv`. This fixes the runtime `FileNotFoundError`/`NameError` chain while preserving the intended output format and “No finding” fallback rule. To keep changes minimal and score-neutral (since current score wasn’t yielded), I won’t add any modeling/training; I generate a safe baseline that produces a valid CSV with the correct columns. I also add small guards to ensure the output is never empty and always matches the required schema.'
- What this solution (achieved 0.00017) has done: 'Your current score (0.0475) is far below the target (0.2277), so we should improve performance with minimal, metric-aligned changes while keeping the “no model” core logic. The smallest legitimate boost is to replace the “all No finding” output with a data-driven prior: for each class, emit one “typical” box derived from train.csv (median box, scaled to the test image size) and use the class prevalence as the confidence, while still emitting “No finding” when the aggregated abnormal confidence is low. This keeps the approach simple (no training, no new dependencies) but yields non-trivial recall and should move mAP upward toward your target band. The script still runs end-to-end and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.00352) has done: 'Your current score (0.00017) is far below the target (0.2277), so we should improve recall/box alignment with minimal changes while keeping the same “train.csv prior → fixed per-image predictions” core approach. The biggest issue is using a hard-coded 1024×1024 test size, which makes boxes systematically wrong; we instead estimate each test DICOM’s width/height from file size (fast, no new deps) and scale the per-class median normalized boxes accordingly. To further move mAP upward without changing the approach, we emit a few more likely classes (slightly higher TOP_K) and set confidences directly from prevalence (bounded) rather than very small values, while still keeping the “No finding” fallback rule. All changes stay within your current logic (no model/training) and still produce a valid `submission.csv`.'
- What this solution (achieved 0.00352) has done: 'Your current heuristic is still far below the target mAP, so we should improve it with the smallest metric-aligned changes while keeping the same “train.csv priors → fixed per-image predictions” core logic. The biggest gain with minimal risk is to stop emitting the same set of classes for every image: instead, output per-image class priors derived from train (class co-occurrence given the most likely “anchor” class), which increases precision without adding any model/training. We also calibrate confidences to be a bit higher (still bounded) and add a small “No finding” mixture so images can still effectively be treated as empty when priors are weak. All changes keep the pipeline simple, deterministic, and still produce a valid `submission.csv`.'
- What this solution (achieved 0.0032) has done: 'Your current heuristic is still far below the target mAP, so the most score-efficient “minimal change” is to improve box sizing and per-image class selection without introducing any new model/training. I keep your same train-prior approach, but (1) derive per-class *normalized box shapes* (center + width/height) and rebuild boxes around the center to avoid the common “median corners” shrink/shift issue, (2) choose the per-image anchor class deterministically from the filename hash (instead of the same anchor for every image) so predictions vary across images and reduce systematic false positives, and (3) slightly recalibrate confidences toward a more reasonable range to improve ranking for mAP while still staying conservative. These changes preserve your core logic (train.csv priors → fixed predictions) and still write a valid `submission.csv`.'
- What this solution (achieved 0.04996) has done: 'Your current score (0.0032) is far below the target (0.2277), so we need a meaningful but still “same core logic” improvement: keep the train-prior heuristic, but make it *per-image* and better calibrated to mAP ranking. I (1) compute a train-derived “no finding prior” and use it to decide per-image whether to emit only class 14, instead of a single global threshold, (2) emit a small number of predictions for all images but downweight them when the image is likely empty, and (3) slightly improve box placement by using per-class median centers but per-class median box sizes with a mild enlargement factor to improve IoU>0.4 hit-rate. These are minimal, deterministic changes that preserve your approach (no model/training, still priors-only) and should increase recall/precision balance to move the score toward the target band. The script still runs end-to-end and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.05019) has done: 'We keep your priors-only submission core logic intact, but fix two score-limiting issues that are still “minimal changes”: (1) your train image width/height proxy uses `x_max/y_max` which systematically underestimates true image size and distorts normalized boxes; we instead normalize each box by the *per-image max of both x-coordinates and y-coordinates* (max of x_min/x_max and y_min/y_max), which is a safer proxy and improves IoU alignment. (2) your per-image “empty” decision is currently a fixed-probability hash draw; we keep it deterministic but make it *data-driven per-image* by using the anchor class’s train frequency and slightly reduce empty probability when the anchor is common, which should improve recall without changing the overall heuristic approach. Finally, we mildly enlarge boxes a bit more (still same mechanism) to better clear IoU>0.4, which is directly aligned to the metric and should move the mAP upward toward your target.'
- What this solution (achieved 0.04972) has done: 'Your current priors-only submission is still far below the target mAP, so we need a small but directly metric-aligned tweak that improves IoU>0.4 hit-rate and reduces false positives without changing the overall “train priors → fixed predictions” approach. I keep the same pipeline, but (1) make box shapes slightly larger for small classes and slightly smaller for very large classes using a simple, train-derived per-class area calibration (improves IoU robustness), (2) adjust confidence to be mildly class-conditional on how often that class is the anchor’s co-occurrence partner (better ranking for mAP), and (3) reduce the always-added class 14 confidence so it doesn’t suppress true positive ranking as much. These are minimal, deterministic changes that preserve your core logic and output format while aiming to move the score upward toward your target band.'
- What this solution (achieved 0.04989) has done: 'We need to increase your mAP from 0.04972 toward 0.2277 (higher is better), but we must keep the same “priors-only from train.csv → fixed per-image predictions” core logic. The biggest score limiter within that logic is the noisy per-image empty decision and overly uniform per-image class choice; I make both more data-driven using only train-derived priors (no new model/training). Specifically, I (1) compute a per-anchor “abnormal image rate” to reduce false “No finding” outputs when the anchor tends to be abnormal, and (2) make `_classes_for_image` return a small, anchor-conditioned top list with a stricter cap to reduce false positives, while (3) slightly recalibrating confidence with an anchor-conditioned multiplier to improve ranking for mAP without changing output semantics. All changes remain deterministic, fast, and still always produce a valid `submission.csv`.'
- What this solution (achieved 0.0505) has done: 'Your current priors-only heuristic is far below the target mAP, so the most “minimal but meaningful” improvement is to increase true-positive IoU hits while reducing obvious false positives, without changing the overall approach (train-derived priors → fixed per-image predictions). I (1) compute a train-derived per-class variability (IQR) of box width/height and use it to add a very small deterministic per-image jitter to box center/size so predictions aren’t identical across all test images (helping some boxes land closer to true objects at IoU>0.4), (2) add one extra predicted box for the anchor class only (a small/large variant) to increase recall with a minimal increase in false positives, and (3) make the “empty” decision slightly less aggressive for images whose anchor class is common, so fewer abnormal images get forced to “No finding”. These stay within your existing core logic (no model/training, same priors, same output format) and should move the score upward toward the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import zlib

DATA_DIR_CANDIDATES = [
    "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection",
    "/kaggle/data/input/vinbigdata-chest-xray-abnormalities-detection",
    "/kaggle/input",
    "/kaggle/data/input",
    "/kaggle/data",
]


def _first_existing(path_list):
    for p in path_list:
        if os.path.exists(p):
            return p
    return None


base_dir = _first_existing(DATA_DIR_CANDIDATES)
if base_dir is None:
    raise FileNotFoundError(
        "Could not locate Kaggle input data directory from known candidates."
    )

sample_path = os.path.join(base_dir, "sample_submission.csv")
train_path = os.path.join(base_dir, "train.csv")

if not os.path.exists(sample_path):
    alt = _first_existing(
        [os.path.join(p, "sample_submission.csv") for p in DATA_DIR_CANDIDATES]
    )
    if alt is None:
        raise FileNotFoundError(
            "sample_submission.csv not found in expected locations."
        )
    sample_path = alt

if not os.path.exists(train_path):
    alt = _first_existing([os.path.join(p, "train.csv") for p in DATA_DIR_CANDIDATES])
    if alt is None:
        raise FileNotFoundError("train.csv not found in expected locations.")
    train_path = alt

sub = pd.read_csv(sample_path)
if not {"image_id", "PredictionString"}.issubset(sub.columns):
    raise ValueError(f"Unexpected sample_submission columns: {sub.columns.tolist()}")

train = pd.read_csv(train_path)
required_train_cols = {"image_id", "class_id", "x_min", "y_min", "x_max", "y_max"}
if not required_train_cols.issubset(train.columns):
    raise ValueError(f"Unexpected train.csv columns: {train.columns.tolist()}")

sub.head()




## === cell 1
NO_FINDING_STR = "14 1 0 0 1 1"

_test_dir_candidates = [
    os.path.join(base_dir, "test"),
    "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/test",
    "/kaggle/data/input/vinbigdata-chest-xray-abnormalities-detection/test",
    "/kaggle/input/test",
    "/kaggle/data/input/test",
]
test_dir = _first_existing(_test_dir_candidates)
if test_dir is None:
    raise FileNotFoundError("Could not locate test/ directory from known candidates.")


def _size_from_filesize(bytes_size: int):
    mb = bytes_size / (1024 * 1024)
    if mb < 6.5:
        s = 1536
    elif mb < 10.0:
        s = 2048
    elif mb < 13.5:
        s = 2560
    else:
        s = 3072
    return int(s), int(s)


test_wh = {}
missing = 0
for img_id in sub["image_id"].astype(str).tolist():
    fp = os.path.join(test_dir, f"{img_id}.dicom")
    if os.path.exists(fp):
        try:
            sz = os.path.getsize(fp)
            test_wh[img_id] = _size_from_filesize(sz)
        except OSError:
            missing += 1
    else:
        missing += 1

if len(test_wh) > 0:
    med_side = int(pd.Series([wh[0] for wh in test_wh.values()]).median())
    DEFAULT_TEST_W, DEFAULT_TEST_H = med_side, med_side
else:
    DEFAULT_TEST_W, DEFAULT_TEST_H = 2048, 2048

abn = train[train["class_id"].between(0, 13)].copy()

all_img_cnt = train["image_id"].nunique()
abn_img_cnt = abn["image_id"].nunique()
no_finding_rate = float(max(0.0, min(1.0, 1.0 - (abn_img_cnt / max(all_img_cnt, 1)))))

cls_img_cnt = abn.groupby("class_id")["image_id"].nunique()
prevalence = (cls_img_cnt / max(all_img_cnt, 1)).to_dict()

img_classes = abn.groupby("image_id")["class_id"].apply(
    lambda s: set(map(int, s.tolist()))
)

anchor_abn_rate = {}
abn_imgs_set = set(abn["image_id"].unique().tolist())
for c in range(14):
    imgs_with_c = [img_id for img_id, s in img_classes.items() if c in s]
    if len(imgs_with_c) == 0:
        anchor_abn_rate[c] = float(abn_img_cnt / max(all_img_cnt, 1))
    else:
        anchor_abn_rate[c] = float(
            len([i for i in imgs_with_c if i in abn_imgs_set]) / len(imgs_with_c)
        )

img_size_proxy = abn.groupby("image_id").agg(
    {
        "x_min": "max",
        "x_max": "max",
        "y_min": "max",
        "y_max": "max",
    }
)
img_size_proxy["img_w"] = img_size_proxy[["x_min", "x_max"]].max(axis=1).clip(lower=1.0)
img_size_proxy["img_h"] = img_size_proxy[["y_min", "y_max"]].max(axis=1).clip(lower=1.0)
img_size_proxy = img_size_proxy[["img_w", "img_h"]].reset_index()

abn = abn.merge(img_size_proxy, on="image_id", how="left")

abn["x_min_n"] = (abn["x_min"] / abn["img_w"]).clip(0.0, 1.0)
abn["x_max_n"] = (abn["x_max"] / abn["img_w"]).clip(0.0, 1.0)
abn["y_min_n"] = (abn["y_min"] / abn["img_h"]).clip(0.0, 1.0)
abn["y_max_n"] = (abn["y_max"] / abn["img_h"]).clip(0.0, 1.0)

abn["cx_n"] = ((abn["x_min_n"] + abn["x_max_n"]) / 2.0).clip(0.0, 1.0)
abn["cy_n"] = ((abn["y_min_n"] + abn["y_max_n"]) / 2.0).clip(0.0, 1.0)
abn["bw_n"] = (abn["x_max_n"] - abn["x_min_n"]).clip(0.01, 1.0)
abn["bh_n"] = (abn["y_max_n"] - abn["y_min_n"]).clip(0.01, 1.0)
abn["area_n"] = (abn["bw_n"] * abn["bh_n"]).clip(1e-4, 1.0)

shape_med = (
    abn.groupby("class_id")[["cx_n", "cy_n", "bw_n", "bh_n", "area_n"]]
    .median()
    .reset_index()
)

shape_q = (
    abn.groupby("class_id")[["cx_n", "cy_n", "bw_n", "bh_n"]]
    .quantile([0.25, 0.75])
    .reset_index()
)
shape_iqr = {cid: (0.0, 0.0, 0.0, 0.0) for cid in range(14)}
for cid in range(14):
    q25 = shape_q[(shape_q["class_id"] == cid) & (shape_q["level_1"] == 0.25)]
    q75 = shape_q[(shape_q["class_id"] == cid) & (shape_q["level_1"] == 0.75)]
    if len(q25) == 1 and len(q75) == 1:
        a = q25.iloc[0]
        b = q75.iloc[0]
        shape_iqr[cid] = (
            float(max(0.0, b["cx_n"] - a["cx_n"])),
            float(max(0.0, b["cy_n"] - a["cy_n"])),
            float(max(0.0, b["bw_n"] - a["bw_n"])),
            float(max(0.0, b["bh_n"] - a["bh_n"])),
        )

default_shape = (0.5, 0.5, 0.5, 0.5)  # cx, cy, bw, bh
shape_by_class = {}
area_by_class = {}
for cid in range(14):
    row = shape_med.loc[shape_med["class_id"] == cid]
    if len(row) == 1:
        r = row.iloc[0]
        shape_by_class[cid] = (
            float(r["cx_n"]),
            float(r["cy_n"]),
            float(r["bw_n"]),
            float(r["bh_n"]),
        )
        area_by_class[cid] = float(r["area_n"])
    else:
        shape_by_class[cid] = default_shape
        area_by_class[cid] = 0.25

med_area_all = float(pd.Series(list(area_by_class.values())).median())
scale_by_class = {}
for cid in range(14):
    a = float(area_by_class.get(cid, med_area_all))
    ratio = (med_area_all / max(a, 1e-4)) ** 0.20  # mild correction exponent
    scale_by_class[cid] = float(max(0.98, min(1.28, ratio)))

pair_counts = {cid: {} for cid in range(14)}
for cls_set in img_classes.tolist():
    for a in cls_set:
        da = pair_counts.get(a)
        if da is None:
            continue
        for b in cls_set:
            da[b] = da.get(b, 0) + 1

cond_prob = {a: {} for a in range(14)}
for a in range(14):
    denom = pair_counts.get(a, {}).get(a, 0)
    if denom <= 0:
        continue
    for b, cnt in pair_counts[a].items():
        cond_prob[a][b] = cnt / denom

prev_series = pd.Series(prevalence).sort_values(ascending=False)
GLOBAL_TOP_K = 8
global_top_classes = prev_series.head(GLOBAL_TOP_K).index.astype(int).tolist()


def _conf(cid: int) -> float:
    p = float(prevalence.get(cid, 0.0))
    return float(min(0.90, max(0.05, 2.80 * p)))


ABN_PRIOR = float(
    min(0.95, max(0.05, sum(prevalence.get(c, 0.0) for c in global_top_classes[:5])))
)

NO_FINDING_BASE = float(min(0.95, max(0.05, no_finding_rate)))


def _clip_box(xmin, ymin, xmax, ymax, w, h):
    xmin = int(max(0, min(int(xmin), w - 2)))
    ymin = int(max(0, min(int(ymin), h - 2)))
    xmax = int(max(xmin + 1, min(int(xmax), w - 1)))
    ymax = int(max(ymin + 1, min(int(ymax), h - 1)))
    return xmin, ymin, xmax, ymax


def _anchor_for_image(img_id: str) -> int:
    if not global_top_classes:
        return 10
    hv = zlib.crc32(img_id.encode("utf-8")) & 0xFFFFFFFF
    return int(global_top_classes[hv % len(global_top_classes)])


def _classes_for_image(img_id: str):
    anchor = _anchor_for_image(img_id)

    cand = cond_prob.get(anchor, {})
    if cand:
        scored = []
        for b, pb_a in cand.items():
            if b == 14:
                continue
            scored.append((float(pb_a) * float(prevalence.get(b, 0.0)), int(b)))
        scored.sort(reverse=True)

        out = [anchor]
        for _, b in scored:
            if b not in out:
                out.append(b)
            if len(out) >= 5:
                break
        for b in global_top_classes:
            if b not in out:
                out.append(b)
            if len(out) >= 6:
                break
        return out[:6]
    else:
        return global_top_classes[:6]


def _shape_to_corners(cx_n, cy_n, bw_n, bh_n):
    x1 = cx_n - bw_n / 2.0
    y1 = cy_n - bh_n / 2.0
    x2 = cx_n + bw_n / 2.0
    y2 = cy_n + bh_n / 2.0
    return (
        float(max(0.0, min(1.0, x1))),
        float(max(0.0, min(1.0, y1))),
        float(max(0.0, min(1.0, x2))),
        float(max(0.0, min(1.0, y2))),
    )


def _u01_from_id(img_id: str) -> float:
    hv = zlib.crc32(img_id.encode("utf-8")) & 0xFFFFFFFF
    return float(hv) / float(2**32 - 1)


def _signed_u01(img_id: str, salt: str) -> float:
    hv = zlib.crc32((img_id + "|" + salt).encode("utf-8")) & 0xFFFFFFFF
    u = float(hv) / float(2**32 - 1)
    return 2.0 * u - 1.0  # [-1, 1]


def build_pred_string_for_image(img_id: str):
    w, h = test_wh.get(img_id, (DEFAULT_TEST_W, DEFAULT_TEST_H))

    anchor = _anchor_for_image(img_id)
    anchor_p = float(prevalence.get(anchor, 0.0))  # ~[0, 1]
    anchor_abn = float(anchor_abn_rate.get(anchor, 0.5))

    empty_adj = float(
        max(0.70, min(1.05, 1.00 - 0.40 * (anchor_abn - (1.0 - NO_FINDING_BASE))))
    )
    empty_adj = float(max(0.70, min(1.05, empty_adj)))

    p_empty = float(
        min(0.97, max(0.01, NO_FINDING_BASE * (1.02 - 0.55 * ABN_PRIOR) * empty_adj))
    )

    u = _u01_from_id(img_id)
    if u < p_empty:
        return NO_FINDING_STR

    parts = []

    ENLARGE_BASE = 1.16

    classes = _classes_for_image(img_id)

    for cid in classes:
        base_conf = _conf(cid)
        if base_conf <= 0:
            continue

        pb = float(cond_prob.get(anchor, {}).get(cid, 0.0))

        anchor_boost = float(min(1.12, max(0.92, 1.00 + 0.35 * (anchor_p - ABN_PRIOR))))
        conf = float(
            min(0.95, max(0.03, base_conf * (0.88 + 0.45 * pb) * anchor_boost))
        )

        cx_n, cy_n, bw_n, bh_n = shape_by_class.get(cid, default_shape)

        i_cx, i_cy, i_bw, i_bh = shape_iqr.get(cid, (0.0, 0.0, 0.0, 0.0))
        jx = 0.18 * i_cx * _signed_u01(img_id, f"cx:{cid}")
        jy = 0.18 * i_cy * _signed_u01(img_id, f"cy:{cid}")
        js = 0.10 * (i_bw + i_bh) * 0.5 * _signed_u01(img_id, f"s:{cid}")

        cx_n = float(min(1.0, max(0.0, cx_n + jx)))
        cy_n = float(min(1.0, max(0.0, cy_n + jy)))

        scl = float(scale_by_class.get(cid, 1.0))
        bw_n = float(min(1.0, max(0.01, bw_n * ENLARGE_BASE * scl * (1.0 + js))))
        bh_n = float(min(1.0, max(0.01, bh_n * ENLARGE_BASE * scl * (1.0 + js))))

        xmn, ymn, xmx, ymx = _shape_to_corners(cx_n, cy_n, bw_n, bh_n)

        xmin = round(xmn * w)
        ymin = round(ymn * h)
        xmax = round(xmx * w)
        ymax = round(ymx * h)
        xmin, ymin, xmax, ymax = _clip_box(xmin, ymin, xmax, ymax, w, h)

        parts.append(f"{cid} {conf:.4f} {xmin} {ymin} {xmax} {ymax}")

        if cid == anchor:
            v = _signed_u01(img_id, f"anchor_var:{cid}")
            mult = 1.18 if v >= 0 else 0.92
            conf2 = float(max(0.02, min(0.90, conf * 0.72)))

            bw2 = float(min(1.0, max(0.01, bw_n * mult)))
            bh2 = float(min(1.0, max(0.01, bh_n * mult)))
            xmn2, ymn2, xmx2, ymx2 = _shape_to_corners(cx_n, cy_n, bw2, bh2)

            xmin2 = round(xmn2 * w)
            ymin2 = round(ymn2 * h)
            xmax2 = round(xmx2 * w)
            ymax2 = round(ymx2 * h)
            xmin2, ymin2, xmax2, ymax2 = _clip_box(xmin2, ymin2, xmax2, ymax2, w, h)

            if (xmin2, ymin2, xmax2, ymax2) != (xmin, ymin, xmax, ymax):
                parts.append(f"{cid} {conf2:.4f} {xmin2} {ymin2} {xmax2} {ymax2}")

    if not parts:
        return NO_FINDING_STR

    nf_conf = float(min(0.22, max(0.02, 0.35 * NO_FINDING_BASE)))
    parts.append(f"14 {nf_conf:.4f} 0 0 1 1")

    return " ".join(parts)


sub["image_id"] = sub["image_id"].astype(str)
sub["PredictionString"] = [
    build_pred_string_for_image(i) for i in sub["image_id"].tolist()
]
sub["PredictionString"] = sub["PredictionString"].fillna(NO_FINDING_STR).astype(str)

sub.shape, sub.head()




## === cell 2
out_path = "submission.csv"
sub.to_csv(out_path, index=False)

assert os.path.exists(out_path), "submission.csv was not created"
check = pd.read_csv(out_path)
assert list(check.columns) == [
    "image_id",
    "PredictionString",
], f"Bad submission columns: {check.columns.tolist()}"
assert len(check) == len(sub), "Row count mismatch after saving submission"
check.head()




## === cell 3
sub
