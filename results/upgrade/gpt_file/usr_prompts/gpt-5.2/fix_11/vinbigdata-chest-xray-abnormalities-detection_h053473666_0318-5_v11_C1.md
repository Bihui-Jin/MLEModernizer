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

0.2283912054588589

# 6. Current score

0.02479

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0475) has done: 'Your notebook fails immediately because it tries to read several external Kaggle Dataset paths (e.g., `../input/vin-big-data-predit/...`) that are not present in your environment, so none of the downstream variables (`df`, `df4`, etc.) ever get defined. To make it run end-to-end and still produce a valid VinBigData submission, I replaced those missing inputs with a robust fallback that uses the provided `sample_submission.csv` as the base and outputs the required `image_id,PredictionString` format. I also kept your post-processing “core logic” structure intact where possible, but made it conditional so it won’t crash when the auxiliary columns/predictions don’t exist. The result is a guaranteed-valid `submission.csv` (with `.csv` suffix) that Kaggle accept.'
- What this solution (achieved 0.01849) has done: 'Your current score is far below the target, and the main reason is that the pipeline mostly outputs “No finding” (or strips most boxes) because the external prediction files you intended to blend aren’t available, and your filtering removes many classes from the already-empty `PredictionString`. To move the score up toward the target without changing the overall approach (still producing a `PredictionString` per image), I minimally switch to a simple, data-driven fallback that uses `train.csv` to compute per-class typical boxes and class priors, then emits a small number of plausible boxes per image (instead of “No finding”), which typically improves mAP versus all-negative submissions. I also stop the hard removal of most classes in cell 5 (which is actively harmful when you don’t have strong detectors) and slightly lower the overly-strict 0.92 threshold in cell 6 so the intended “confidence adjustment” logic can actually take effect. The output format, paths, and submission CSV schema stay the same.'
- What this solution (achieved 0.0187) has done: 'Your current score is far below target, so the smallest meaningful way to move toward 0.228 is to make the fallback predictions less “generic” while keeping your overall “PredictionString per image” approach intact. I keep all blending logic as-is (only used if external files exist), but improve the internal fallback by (1) using per-class *mean and std* box statistics and (2) emitting multiple candidate boxes per class (median/mean ± std) with conservative confidences. I also stop the class-score gating in cell 7 from effectively disabling all fallback boxes (because your per-class columns are all zeros here), so the fallback predictions are not unintentionally down-weighted/ignored. Finally, I add a strict sanitizer to ensure every row has a valid 6*k token PredictionString (or the required “14 1 0 0 1 1”).'
- What this solution (achieved 0.02443) has done: 'Your current score is far below the target, so we should improve the *fallback* predictions (used when external detector CSVs aren’t present) without changing your overall “one PredictionString per image” approach. The biggest low-risk gain here is to make the fallback boxes image-size aware and class-aware: use train-set per-class box statistics **normalized by image width/height**, then scale to each test image’s pixel size (read from DICOM header without decoding pixels). We also make sure we never emit “No finding” alongside other boxes, and we sanitize/clip boxes to valid image bounds to avoid invalid IoU behavior. These are minimal, metric-aligned changes that should move mAP upward toward your target while keeping the blending/post-processing structure intact.'
- What this solution (achieved 0.02457) has done: 'We keep your blending/post-processing structure intact and only strengthen the fallback boxes (which are driving performance because external detector CSVs are missing). Specifically, we (1) compute per-class normalized box priors more robustly by sampling train image shapes in a stratified way across classes (so rare classes don’t get poor/empty stats), (2) emit one additional “wider” proposal per class (still minimal) to improve IoU>0.4 hit-rate, and (3) apply a light confidence normalization so the extra boxes don’t overwhelm ranking. These are metric-aligned, small changes that should increase mAP from 0.024 toward your 0.228 target without changing the overall approach or output format. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.02538) has done: 'Your score is far below the target, and the main limiter is that (when external detector submissions are missing) the fallback emits only a few “top classes” with a small number of generic boxes, so recall is extremely low for a 14-class detection mAP. To move the score upward without changing the overall submission approach, I keep your blending/post-processing structure intact but strengthen only the fallback generation: increase class coverage (use all 14 classes with a small prior-based cap), and add one more proposal (“wide”) per class to raise the chance of IoU>0.4 matches. I also make the fallback confidences slightly more conservative and prior-shaped so extra boxes don’t swamp ranking, while keeping all your sanitization and “no finding” handling the same. All paths stay the same and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.02567) has done: 'Your current score (0.02538) is far below the target (0.22839), so we should cautiously improve recall/IoU of the fallback predictions (which are dominating because the external blend files don’t exist). I keep your overall “PredictionString per image” pipeline and sanitization intact, but (1) expand fallback class coverage from top-10 to all 14 classes, and (2) add a second, safer “centered” proposal per class (in addition to your existing ones) to raise the chance of IoU>0.4 without exploding box count too much. I also (3) slightly adjust fallback confidence shaping so the added boxes don’t overwhelm ranking, while leaving the rest of the blending/post-processing logic unchanged. These are minimal, metric-aligned changes intended to move mAP upward toward your target band.'
- What this solution (achieved 0.02471) has done: 'Your current score is far below the target, so the safest way to move upward without changing your overall “fallback priors + optional blending” approach is to make the fallback less over-predictive and more mAP-friendly. I keep your pipeline structure intact but (1) cap fallback predictions to a small set of highest-prior classes (instead of all 14) to reduce false positives that crush precision, (2) reduce proposals per class from 5 to 3 (median/centered/wide) to keep recall while improving ranking, and (3) make confidences more conservative and spaced so the evaluator can rank boxes better. These are minimal, metric-aligned tweaks to the existing fallback generation (the dominant component in your environment) and should increase mAP vs the current very dense generic predictions.'
- What this solution (achieved 0.02479) has done: 'Your current score is far below the target, so we should cautiously improve mAP by strengthening only the fallback predictions (which dominate when external blend files are missing) without changing your overall pipeline or output format. The smallest, metric-aligned gain is to slightly increase recall while keeping precision under control: expand fallback coverage from top-6 to top-8 classes and add one additional (but conservative) “expanded” proposal per class to increase the chance of IoU>0.4 matches. I also make the fallback confidences a bit more separated by proposal rank so extra boxes don’t overwhelm ranking, and I sanitize all rows (not just the first 3000—though test is 1500, this makes it robust) to ensure valid tokenization. Everything else (blending hooks, post-processing structure, paths, and CSV schema) remains the same and still produces a valid `submission.csv`.'
- What this solution (achieved 0.02479) has done: 'We keep your pipeline and optional blending intact, but make a minimal, metric-aligned improvement to the fallback (the dominant component in your environment) by choosing fallback classes per-image using the existing per-class score columns (when available) instead of always using the same global `top_classes`. This increases the chance that predicted classes match true findings while keeping prediction count controlled for precision. We also ensure fallback confidences are monotonically decreasing within an image and lightly boosted for classes that the per-image columns suggest are present, without changing any model/training logic. Finally, we preserve all your sanitization and submission formatting, and still output a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

BASE_INPUT = "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection"
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")

assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample_submission at {SAMPLE_SUB_PATH}"
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

if "PredictionString" not in sample_sub.columns and "TARGET" in sample_sub.columns:
    sample_sub = sample_sub.rename(columns={"TARGET": "PredictionString"})
if "ID" in sample_sub.columns and "image_id" not in sample_sub.columns:
    sample_sub = sample_sub.rename(columns={"ID": "image_id"})

sample_sub = sample_sub[["image_id", "PredictionString"]].copy()
sample_sub.head()



## === cell 1
df = sample_sub.copy()

for k in range(15):
    col = str(k)
    if col not in df.columns:
        df[col] = 0.0

df.shape, df.columns[:5].tolist()




## === cell 2
def try_read_csv(path: str):
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        return None
    return None


df1 = try_read_csv("../input/noisy-vin-big-data-predit/submission.csv")
df_densenet = try_read_csv("../input/densenet201-vin-big-data-predit/submission.csv")

blend_cols = [str(i) for i in range(15)]
if (
    df1 is not None
    and df_densenet is not None
    and all(c in df.columns for c in blend_cols)
    and all(c in df1.columns for c in blend_cols)
    and all(c in df_densenet.columns for c in blend_cols)
):
    df[blend_cols] = (
        df[blend_cols] * 0.25 + df1[blend_cols] * 0.5 + df_densenet[blend_cols] * 0.25
    )

df.head()



## === cell 3
df_heart_cnn = try_read_csv("../input/heart-efn-cnn-predict/submission.csv")

if (
    df_heart_cnn is not None
    and all(c in df_heart_cnn.columns for c in ["0", "3"])
    and all(c in df.columns for c in ["0", "3"])
):
    df[["0", "3"]] = df[["0", "3"]] * 0.75 + df_heart_cnn[["0", "3"]] * 0.25

df[["image_id", "PredictionString"]].head()



## === cell 4
df2 = try_read_csv("../input/vinbigdata-sub-23-211/submission (18).csv")
df3 = try_read_csv("../input/vinbigdata-sub-23-211/submission (17).csv")

if df3 is not None and "image_id" in df3.columns:
    df4 = pd.merge(df, df3, on="image_id", how="left", suffixes=("", "_df3"))
    if "PredictionString_df3" in df4.columns:
        df4["PredictionString"] = df4["PredictionString_df3"].fillna(
            df4["PredictionString"]
        )
        df4 = df4.drop(columns=["PredictionString_df3"])
else:
    df4 = df.copy()

if df2 is not None and "image_id" in df2.columns and "PredictionString" in df2.columns:
    df5 = pd.merge(
        df,
        df2[["image_id", "PredictionString"]],
        on="image_id",
        how="left",
        suffixes=("", "_df2"),
    )
    df4["PredictionString"] = (
        df4["PredictionString"].astype(str).fillna("")
        + " "
        + df5["PredictionString_df2"].astype(str).fillna("")
    ).str.strip()

df4.shape



## === cell 5
TRAIN_CSV_PATH = os.path.join(BASE_INPUT, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_INPUT, "train")
TEST_IMG_DIR = os.path.join(BASE_INPUT, "test")

train_df = pd.read_csv(TRAIN_CSV_PATH)


def _dicom_hw(path: str):
    try:
        import pydicom  # type: ignore

        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        h = int(getattr(ds, "Rows", 0) or 0)
        w = int(getattr(ds, "Columns", 0) or 0)
        if h > 0 and w > 0:
            return h, w
    except Exception:
        pass
    return None, None


train_obj = train_df[train_df["class_id"].between(0, 13)].copy()

rng = np.random.default_rng(123)
per_class_imgs = train_obj.groupby("class_id")["image_id"].unique().to_dict()

MAX_SHAPE_READS = 6500  # keep runtime safe
PER_CLASS_QUOTA = 420

selected_ids = []
for cid, ids in per_class_imgs.items():
    ids = [x for x in ids if isinstance(x, str)]
    if len(ids) == 0:
        continue
    take = min(PER_CLASS_QUOTA, len(ids))
    chosen = rng.choice(ids, size=take, replace=False).tolist()
    selected_ids.extend(chosen)

if len(selected_ids) > MAX_SHAPE_READS:
    selected_ids = rng.choice(
        selected_ids, size=MAX_SHAPE_READS, replace=False
    ).tolist()

train_shapes = {}
for img_id in selected_ids:
    p = os.path.join(TRAIN_IMG_DIR, f"{img_id}.dicom")
    if os.path.exists(p):
        h, w = _dicom_hw(p)
        if h is not None:
            train_shapes[img_id] = (h, w)

if len(train_shapes) > 0:
    shape_df = pd.DataFrame(
        [(k, v[0], v[1]) for k, v in train_shapes.items()],
        columns=["image_id", "img_h", "img_w"],
    )
    train_obj2 = train_obj.merge(shape_df, on="image_id", how="inner")
else:
    train_obj2 = train_obj.iloc[0:0].copy()

if train_obj2.shape[0] == 0:
    train_obj_abs = train_obj.copy()
    stats_abs = train_obj_abs.groupby("class_id")[
        ["x_min", "y_min", "x_max", "y_max"]
    ].agg(["median", "mean", "std"])
    stats_abs.columns = ["_".join(c) for c in stats_abs.columns.to_flat_index()]
    stats_abs = stats_abs.reset_index()
    stats_norm = None
else:
    for c in ["x_min", "x_max"]:
        train_obj2[c] = train_obj2[c] / train_obj2["img_w"].clip(lower=1)
    for c in ["y_min", "y_max"]:
        train_obj2[c] = train_obj2[c] / train_obj2["img_h"].clip(lower=1)

    stats_norm = train_obj2.groupby("class_id")[
        ["x_min", "y_min", "x_max", "y_max"]
    ].agg(["median", "mean", "std"])
    stats_norm.columns = ["_".join(c) for c in stats_norm.columns.to_flat_index()]
    stats_norm = stats_norm.reset_index()
    stats_abs = None

img_counts = train_obj.groupby("class_id")["image_id"].nunique()
total_imgs = train_df["image_id"].nunique()
prior = (img_counts / max(total_imgs, 1)).sort_values(ascending=False)

TOPK_FALLBACK_CLASSES = 8
top_classes = prior.index.tolist()[:TOPK_FALLBACK_CLASSES]
top_classes = [int(c) for c in top_classes if 0 <= int(c) <= 13]


def _get_row_val(r, name, default=0.0):
    v = r.get(name, default)
    try:
        if pd.isna(v):
            return float(default)
    except Exception:
        pass
    return float(v)


def _make_centered_proposal(box, scale=0.92):
    x1, y1, x2, y2 = map(float, box)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    w, h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    nw, nh = w * scale, h * scale
    return (cx - nw * 0.5, cy - nh * 0.5, cx + nw * 0.5, cy + nh * 0.5)


def _make_expanded_proposal(box, scale=1.08):
    x1, y1, x2, y2 = map(float, box)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    w, h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    nw, nh = w * scale, h * scale
    return (cx - nw * 0.5, cy - nh * 0.5, cx + nw * 0.5, cy + nh * 0.5)


box_map = {}
if stats_norm is not None:
    for _, r in stats_norm.iterrows():
        cid = int(r["class_id"])
        med = (
            _get_row_val(r, "x_min_median"),
            _get_row_val(r, "y_min_median"),
            _get_row_val(r, "x_max_median"),
            _get_row_val(r, "y_max_median"),
        )
        mean = (
            _get_row_val(r, "x_min_mean"),
            _get_row_val(r, "y_min_mean"),
            _get_row_val(r, "x_max_mean"),
            _get_row_val(r, "y_max_mean"),
        )
        std = (
            max(_get_row_val(r, "x_min_std"), 0.0),
            max(_get_row_val(r, "y_min_std"), 0.0),
            max(_get_row_val(r, "x_max_std"), 0.0),
            max(_get_row_val(r, "y_max_std"), 0.0),
        )
        jitter = tuple(min(s, 0.20) for s in std)

        wide = (
            mean[0] - 0.55 * jitter[0],
            mean[1] - 0.55 * jitter[1],
            mean[2] + 0.55 * jitter[2],
            mean[3] + 0.55 * jitter[3],
        )

        centered = _make_centered_proposal(med, scale=0.92)
        expanded = _make_expanded_proposal(med, scale=1.08)

        proposals = [med, centered, wide, expanded]
        box_map[cid] = proposals
else:
    for _, r in stats_abs.iterrows():
        cid = int(r["class_id"])
        med = (
            _get_row_val(r, "x_min_median"),
            _get_row_val(r, "y_min_median"),
            _get_row_val(r, "x_max_median"),
            _get_row_val(r, "y_max_median"),
        )
        mean = (
            _get_row_val(r, "x_min_mean"),
            _get_row_val(r, "y_min_mean"),
            _get_row_val(r, "x_max_mean"),
            _get_row_val(r, "y_max_mean"),
        )
        std = (
            max(_get_row_val(r, "x_min_std"), 0.0),
            max(_get_row_val(r, "y_min_std"), 0.0),
            max(_get_row_val(r, "x_max_std"), 0.0),
            max(_get_row_val(r, "y_max_std"), 0.0),
        )
        jitter = tuple(min(s, 80.0) for s in std)
        wide = (
            mean[0] - 0.70 * jitter[0],
            mean[1] - 0.70 * jitter[1],
            mean[2] + 0.70 * jitter[2],
            mean[3] + 0.70 * jitter[3],
        )
        centered = _make_centered_proposal(med, scale=0.92)
        expanded = _make_expanded_proposal(med, scale=1.08)
        proposals = [med, centered, wide, expanded]
        box_map[cid] = proposals

conf_map = {}
for cid in range(14):
    p = float(prior.get(cid, 0.0))
    conf = 0.05 + 0.12 * min(max(p / 0.20, 0.0), 1.0)
    conf_map[cid] = float(conf)


def is_empty_ps(ps: str) -> bool:
    ps = str(ps).strip()
    if ps == "" or ps.lower() == "nan":
        return True
    if ps == "14 1 0 0 1 1":
        return True
    if len(ps.split()) < 6:
        return True
    return False


test_shapes = {}
for img_id in df4["image_id"].tolist():
    p = os.path.join(TEST_IMG_DIR, f"{img_id}.dicom")
    if os.path.exists(p):
        h, w = _dicom_hw(p)
        if h is not None:
            test_shapes[img_id] = (h, w)


def _clip_box(x1, y1, x2, y2, w, h):
    x1 = float(np.clip(x1, 0, max(w - 1, 0)))
    y1 = float(np.clip(y1, 0, max(h - 1, 0)))
    x2 = float(np.clip(x2, 0, max(w - 1, 0)))
    y2 = float(np.clip(y2, 0, max(h - 1, 0)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 - x1 < 1:
        x2 = min(x1 + 1, max(w - 1, 0))
    if y2 - y1 < 1:
        y2 = min(y1 + 1, max(h - 1, 0))
    return x1, y1, x2, y2


def _pick_classes_for_row(row, default_top_classes, k=TOPK_FALLBACK_CLASSES):
    avail = []
    for cid in range(14):
        col = str(cid)
        if col in row.index:
            try:
                v = float(row[col])
            except Exception:
                v = 0.0
            avail.append((cid, v))
    if len(avail) == 0:
        return list(default_top_classes)

    scored = []
    for cid, v in avail:
        scored.append((float(v) + 0.03 * float(prior.get(cid, 0.0)), cid))
    scored.sort(reverse=True)
    picked = [cid for _, cid in scored[:k]]

    picked = [int(c) for c in picked if 0 <= int(c) <= 13 and int(c) in box_map]
    if len(picked) == 0:
        picked = [c for c in default_top_classes if c in box_map]
    return picked


for i in range(df4.shape[0]):
    if not is_empty_ps(df4.loc[i, "PredictionString"]):
        continue

    img_id = df4.loc[i, "image_id"]
    h, w = test_shapes.get(img_id, (None, None))

    row = df4.loc[i]
    chosen_classes = _pick_classes_for_row(row, top_classes, k=TOPK_FALLBACK_CLASSES)

    parts = []
    for cid in chosen_classes:
        if cid not in box_map:
            continue

        conf0 = conf_map.get(cid, 0.08)
        col = str(cid)
        try:
            v_img = float(row[col]) if col in row.index else 0.0
        except Exception:
            v_img = 0.0
        boost = 1.0 + 0.25 * float(np.clip(v_img, 0.0, 1.0))
        conf0 = float(conf0 * boost)

        props = box_map[cid][:4]
        for k_prop, (x1, y1, x2, y2) in enumerate(props):
            if k_prop == 0:
                mult = 1.00  # median
            elif k_prop == 1:
                mult = 0.78  # centered
            elif k_prop == 2:
                mult = 0.62  # wide
            else:
                mult = 0.54  # expanded (lowest)
            conf = conf0 * mult

            if stats_norm is not None and h is not None and w is not None:
                xx1, yy1, xx2, yy2 = x1 * w, y1 * h, x2 * w, y2 * h
            else:
                xx1, yy1, xx2, yy2 = x1, y1, x2, y2

            if h is not None and w is not None:
                xx1, yy1, xx2, yy2 = _clip_box(xx1, yy1, xx2, yy2, w, h)
            else:
                xx1, xx2 = (xx1, xx2) if xx1 <= xx2 else (xx2, xx1)
                yy1, yy2 = (yy1, yy2) if yy1 <= yy2 else (yy2, yy1)
                if xx2 - xx1 < 1:
                    xx2 = xx1 + 1
                if yy2 - yy1 < 1:
                    yy2 = yy1 + 1

            parts.append(f"{cid} {conf:.4f} {xx1:.1f} {yy1:.1f} {xx2:.1f} {yy2:.1f}")

    if len(parts) > 1:

        def _conf_from_part(p):
            try:
                return float(p.split()[1])
            except Exception:
                return -1.0

        parts = sorted(parts, key=_conf_from_part, reverse=True)

    df4.loc[i, "PredictionString"] = " ".join(parts).strip() or "14 1 0 0 1 1"

df4[["image_id", "PredictionString"]].head()




## === cell 6
def _sanitize_ps(ps: str) -> str:
    ps = str(ps).strip()
    if ps == "" or ps.lower() == "nan":
        return "14 1 0 0 1 1"
    toks = ps.split()
    if len(toks) < 6:
        return "14 1 0 0 1 1"
    k = (len(toks) // 6) * 6
    toks = toks[:k]
    if len(toks) == 0:
        return "14 1 0 0 1 1"
    return " ".join(toks)


for i in range(df4.shape[0]):
    df4.loc[i, "PredictionString"] = _sanitize_ps(df4.loc[i, "PredictionString"])

df4[["image_id", "PredictionString"]].head()



## === cell 7
list1 = list(range(15))
for i in range(df4.shape[0]):
    ps = str(df4.loc[i, "PredictionString"]).strip()
    if ps == "14 1 0 0 1 1" or ps == "" or len(ps.split()) < 6:
        continue

    b = ps.split()
    num_boxes = len(b) // 6

    for j in range(num_boxes):
        try:
            cls_id = int(float(b[0 + 6 * j]))
        except Exception:
            continue
        if cls_id not in list1:
            continue

        col = str(cls_id)
        if col not in df4.columns:
            continue

        try:
            col_val = float(df4.loc[i, col])
        except Exception:
            col_val = 0.0
        if col_val > 1e-6 and col_val < 0.30:
            continue

        try:
            c = float(b[0 + 6 * j + 1])
        except Exception:
            continue

        if col_val > 1e-6:
            b[0 + 6 * j + 1] = str(col_val * 0.4 + c * 0.6)
        else:
            b[0 + 6 * j + 1] = str(c)

    df4.loc[i, "PredictionString"] = " ".join(b)

df4[["image_id", "PredictionString"]].head()




## === cell 8
def _remove_no_finding_if_needed(ps: str) -> str:
    ps = _sanitize_ps(ps)
    toks = ps.split()
    if toks == ["14", "1", "0", "0", "1", "1"]:
        return ps
    out = []
    for i in range(0, len(toks), 6):
        try:
            cid = int(float(toks[i]))
        except Exception:
            continue
        if cid == 14:
            continue
        out.extend(toks[i : i + 6])
    if len(out) == 0:
        return "14 1 0 0 1 1"
    return " ".join(out)


if "14" not in df4.columns:
    df4["14"] = 0.0

for i in range(df4.shape[0]):
    if float(df4.loc[i, "14"]) > 0.999:
        df4.loc[i, "PredictionString"] = "14 1 0 0 1 1"
    df4.loc[i, "PredictionString"] = _remove_no_finding_if_needed(
        df4.loc[i, "PredictionString"]
    )
    df4.loc[i, "PredictionString"] = _sanitize_ps(df4.loc[i, "PredictionString"])

df_final = df4[["image_id", "PredictionString"]].copy()
df_final.to_csv("submission.csv", index=False)

df_final.head()



## === cell 9
assert os.path.exists("submission.csv")
sub = pd.read_csv("submission.csv")
assert list(sub.columns) == ["image_id", "PredictionString"]
assert sub.shape[0] == sample_sub.shape[0]  # should be 1500
sub.isna().sum(), sub.head()
