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

0.2279143772743763

# 6. Current score

0.04914

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0475) has done: 'I remove the dependencies on missing `../input/.../submission.csv` files (the root cause of the FileNotFoundError) and instead build a valid baseline submission directly from the provided `sample_submission.csv` so the notebook always runs end-to-end in this environment. I keep the original intent (produce `image_id` + prediction string) but make it robust to the actual column names in this competition (`image_id`, `PredictionString`) and always write `submission.csv` with the correct header. Since your current score is “Not yielded”, the primary goal is to generate a valid CSV; this baseline be low-scoring but successfully submit and unblock iteration. I also keep the cell structure and avoid adding any heavy modeling/training (fits within the 600s constraint).'
- What this solution (achieved 0.0499) has done: 'Your current code always submits the “No finding” box for every image, which caps mAP very low; to move toward the target score we need a slightly more informative baseline without changing the overall “generate PredictionString and write submission.csv” core. The smallest legitimate step up (still very lightweight and within constraints) is to add a simple per-class box prior learned from `train.csv` (median box per class) and then, for each test image, emit a small set of the most common classes with conservative confidences plus the required “No finding” fallback. This keeps the same submission semantics (string of `class conf xmin ymin xmax ymax` tokens), avoids any heavy model/training, and usually improves over the all-14 baseline because it produces some true positives with plausible boxes. We also clamp boxes to a valid range and keep the CSV schema exactly as required.'
- What this solution (achieved 0.05001) has done: 'Your current baseline repeats the same “top-K prior boxes” for every test image, which yields some true positives but also many false positives that depress mAP; to move your score upward toward 0.227, the smallest safe improvement is to tune the number of emitted boxes and the confidence levels to be more conservative. I keep the exact same core approach (learn per-class median box priors from `train.csv`, emit a fixed list of class predictions per image, and always include the required `14 1 0 0 1 1` fallback) and only adjust lightweight calibration knobs. Concretely: increase `TOPK_CLASSES` slightly (to cover more positives) while lowering confidences and making them frequency-aware, which typically improves precision/recall balance under VOC mAP. I also keep the submission schema identical and still write `submission.csv` end-to-end.'
- What this solution (achieved 0.06606) has done: 'Your current prior-box baseline is underperforming mainly because it outputs the exact same high-FP set of classes for every image; to move the score upward toward the 0.2279 target with minimal logic change, we reduce false positives by (1) filtering the training priors to only boxes that look valid and reasonably sized, and (2) emitting fewer classes (lower TOPK) with slightly higher-but-still-conservative confidences to better balance precision/recall under VOC mAP. This keeps the same core approach (median per-class box priors + fixed per-image PredictionString + always include “No finding”) and only adjusts calibration/priors selection. We also avoid assuming a fixed 1024 canvas by clamping to a generous range based on observed train box maxima, which prevents invalid coordinates. The script still runs end-to-end quickly and writes a valid `submission.csv` with the correct header/columns.'
- What this solution (achieved 0.06679) has done: 'Your current score (0.06606) is well below the target (0.2279), so we should safely increase recall while keeping false positives controlled. With minimal change to your existing “per-class median box priors + fixed per-image PredictionString + always include No finding” core, I (1) expand TOPK_CLASSES modestly and (2) use a smoother, rank-based confidence schedule with a slightly higher cap, which tends to improve VOC mAP by covering more true classes without overly aggressive scores. I also make the priors a bit more robust by using per-class quantiles (instead of pure medians) to slightly enlarge boxes, which often increases IoU>0.4 hit-rate while keeping the same overall approach. The script still run fast, stay deterministic, and write a valid `submission.csv`.'
- What this solution (achieved 0.06663) has done: 'We keep your exact “per-class prior boxes + fixed PredictionString for every test image + always include `14 1 0 0 1 1`” core, but reduce false positives and improve IoU hit-rate with very small calibration tweaks. Specifically, we (1) shrink `TOPK_CLASSES` a bit to cut the number of always-on classes (a major FP driver for VOC mAP), and (2) make the prior boxes slightly larger by widening the quantile span just a touch to increase the chance of IoU>0.4 without changing the approach. We also mildly lower the confidence range to avoid over-confident FPs, which typically helps mAP when predictions are constant across images. The script still runs fast and produces the same valid `submission.csv` format.'
- What this solution (achieved 0.06691) has done: 'Your current score is far below the target, so we should cautiously increase recall while keeping false positives controlled, without changing the core “fixed per-image prior boxes + always include No finding” approach. The smallest safe lever is to make the predicted boxes a bit more “forgiving” for IoU>0.4 by slightly expanding the learned prior boxes (while still clamping to valid ranges) and to very slightly increase class coverage (TOPK) with a conservative confidence schedule. I keep the exact same training-data-derived priors (quantiles per class) and the same submission semantics, only adjusting (a) TOPK_CLASSES and (b) a small box expansion factor applied to the quantile boxes. This should move mAP upward by improving the chance that at least one predicted box overlaps true findings enough, while avoiding a big FP explosion.'
- What this solution (achieved 0.06721) has done: 'We keep your exact “train-derived per-class prior boxes + fixed PredictionString for every test image + always include `14 1 0 0 1 1`” approach, and only make two small, score-relevant adjustments to move up from 0.0669 toward 0.2279: (1) modestly increase recall by emitting a couple more frequent classes (slightly higher TOPK), and (2) make the prior boxes a bit more IoU>0.4-friendly by widening the quantile span and applying a slightly larger expansion. To avoid a false-positive explosion (which hurts VOC mAP), we simultaneously soften confidences a bit with a slightly lower cap and a gentler rank schedule. Everything else (I/O paths, submission schema, constant-per-image predictions, and the required “No finding” token) stays the same, and it still runs quickly end-to-end and writes `submission.csv`.'
- What this solution (achieved 0.06708) has done: 'We keep your exact “train-derived per-class prior boxes + fixed PredictionString for every test image + always include `14 1 0 0 1 1`” approach, but make two minimal, score-relevant tweaks to improve mAP toward the 0.2279 target. First, we reduce false positives by lowering the confidence range and emitting slightly fewer always-on classes (constant predictions across all images are very FP-sensitive under VOC mAP). Second, we slightly increase IoU>0.4 hit-rate by a tiny additional box expansion (still derived from your same quantile priors), which can improve overlaps without changing the core logic. All paths, output schema, and runtime behavior remain the same and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.06707) has done: 'To move your mAP up toward the 0.2279 target while keeping the exact same “train-derived per-class prior boxes + fixed per-image PredictionString + always include No finding” core, I make two small, score-relevant adjustments: (1) increase recall slightly by emitting one more frequent class (TOPK 7→8), and (2) make the prior boxes a bit more IoU>0.4-friendly by a tiny increase in the quantile span and a tiny increase in expansion (both still derived purely from train boxes). To keep false positives from exploding (constant predictions are very FP-sensitive), I keep confidences conservative by only slightly lifting the cap. All I/O paths, submission schema, and runtime remain the same, and it still write a valid `submission.csv`.'
- What this solution (achieved 0.06673) has done: 'We keep your exact “train-derived per-class quantile box priors + fixed per-image PredictionString + always include `14 1 0 0 1 1`” baseline, and only make minimal calibration changes aimed at increasing recall a bit without a big false-positive explosion (constant predictions are very sensitive under VOC mAP). Concretely, we (1) emit one additional frequent class (TOPK 8→9) and (2) slightly widen/expand the prior boxes to improve the chance of IoU>0.4 matches, while (3) keeping confidences conservative by lowering the cap a touch so extra boxes don’t dominate the precision curve. All paths, schema, and runtime behavior stay the same, and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.06673) has done: 'Your current score (0.06673) is far below the target (0.2279), so we should cautiously increase recall while keeping the exact same “train-derived per-class quantile box priors + fixed per-image PredictionString + always include No finding” core. The smallest safe change that tends to help VOC mAP here is to emit a *second* prior box per class (a “tight” and a slightly “loose” variant) but at lower confidences, which increases the chance of hitting IoU>0.4 without changing the underlying approach or adding any modeling. To control false positives, we keep TOPK the same, apply a small confidence decay to the extra boxes, and keep the overall confidence caps conservative. The output format, paths, and runtime remain the same, and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.06701) has done: 'Your current score (0.06673) is far below the target (0.2279), so we should increase recall a bit while keeping false positives from exploding, without changing the core “train-derived per-class quantile box priors + fixed per-image PredictionString + always include No finding” approach. The smallest useful lever here is to emit one additional (third) box variant per class that is slightly shifted (not a new model), which can increase the chance of an IoU>0.4 match when the object tends to appear off-center relative to the prior. To limit extra false positives, the shifted box uses a lower confidence multiplier than your loose box, and we keep TOPK and the base confidence schedule unchanged. The submission schema, paths, and runtime remain the same and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.06664) has done: 'Your current baseline is far below the target, so we should nudge mAP upward by improving IoU>0.4 hit-rate and slightly improving precision, without changing the core “train-derived per-class priors + fixed per-image PredictionString + always include No finding” approach. The smallest safe gain here is to (1) make box priors more robust by computing them on *normalized* coordinates (relative to each image size) and then converting back using the *expected* test image size (so coordinates are not biased by varying train resolutions), and (2) add a tiny opposite-direction shifted variant (±shift) to increase match chance while keeping confidences conservative to limit FP impact. Everything else stays the same: we still output a constant set of priors for every test image and write a valid `submission.csv`.'
- What this solution (achieved 0.04914) has done: 'Your current score (0.06664) is far below the target (0.2279), so we need a small, legitimate change that increases true-positive chance without changing the core “train-derived per-class priors + fixed per-image PredictionString + always include No finding” approach. The biggest low-risk issue is that we convert normalized priors back to pixels using a single median `(TGT_W, TGT_H)`, but train/test DICOMs vary in resolution; this mis-scales boxes and hurts IoU>0.4. I keep the exact same priors and emitted classes/variants, but read each test DICOM’s pixel array shape to scale boxes per-image (fast enough for 1500 images), which should improve localization IoU and move mAP upward. Everything else (string format, confidences, variants, required “14 1 0 0 1 1”, output path) remains the same.'

# 9. Code solution

## === cell 0
import os
import pandas as pd

BASE_INPUT = "/kaggle/data/vinbigdata-chest-xray-abnormalities-detection"
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(BASE_INPUT, "train.csv")

if not os.path.exists(SAMPLE_SUB_PATH):
    SAMPLE_SUB_PATH = "/kaggle/data/sample_submission.csv"
if not os.path.exists(TRAIN_CSV_PATH):
    TRAIN_CSV_PATH = "/kaggle/data/train.csv"

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

if "image_id" not in sample_sub.columns:
    raise ValueError(
        f"sample_submission.csv must contain 'image_id'. Found columns: {list(sample_sub.columns)}"
    )

pred_col = (
    "PredictionString"
    if "PredictionString" in sample_sub.columns
    else ("TARGET" if "TARGET" in sample_sub.columns else None)
)
if pred_col is None:
    raise ValueError(
        f"sample_submission.csv must contain 'PredictionString' or 'TARGET'. Found columns: {list(sample_sub.columns)}"
    )

df_final = sample_sub[["image_id", pred_col]].copy()
df_final.columns = ["image_id", "PredictionString"]

train_df = pd.read_csv(TRAIN_CSV_PATH)

train_df = train_df[train_df["class_id"].between(0, 13)].copy()
train_df = train_df.dropna(subset=["x_min", "y_min", "x_max", "y_max"]).copy()
train_df = train_df[
    (train_df["x_max"] > train_df["x_min"]) & (train_df["y_max"] > train_df["y_min"])
].copy()

img_wh = (
    train_df.groupby("image_id")[["x_max", "y_max"]]
    .max()
    .rename(columns={"x_max": "img_w", "y_max": "img_h"})
    .reset_index()
)
train_df = train_df.merge(img_wh, on="image_id", how="left")
train_df = train_df.dropna(subset=["img_w", "img_h"]).copy()
train_df = train_df[(train_df["img_w"] > 1) & (train_df["img_h"] > 1)].copy()

for c, denom in [
    ("x_min", "img_w"),
    ("x_max", "img_w"),
    ("y_min", "img_h"),
    ("y_max", "img_h"),
]:
    train_df[c + "_n"] = (
        train_df[c].astype(float) / train_df[denom].astype(float)
    ).clip(0.0, 1.0)

w = (train_df["x_max_n"] - train_df["x_min_n"]).astype(float)
h = (train_df["y_max_n"] - train_df["y_min_n"]).astype(float)
area = w * h

train_df = train_df[(w >= 0.01) & (h >= 0.01) & (area >= 0.0002)].copy()


def _q(series, q):
    return float(series.quantile(q))


QLO, QHI = 0.26, 0.74

box_priors_n = (
    train_df.groupby("class_id")
    .apply(
        lambda g: pd.Series(
            {
                "x_min_n": _q(g["x_min_n"], QLO),
                "y_min_n": _q(g["y_min_n"], QLO),
                "x_max_n": _q(g["x_max_n"], QHI),
                "y_max_n": _q(g["y_max_n"], QHI),
            }
        )
    )
    .reset_index()
)

class_freq = (
    train_df["class_id"].value_counts().sort_values(ascending=False).reset_index()
)
class_freq.columns = ["class_id", "count"]

priors = pd.merge(class_freq, box_priors_n, on="class_id", how="inner").sort_values(
    "count", ascending=False
)

TOPK_CLASSES = 9
top_priors = priors.head(TOPK_CLASSES).copy()


def _rank_conf(i, k):
    start = 0.18
    end = 0.04
    if k <= 1:
        return start
    t = i / (k - 1)
    return float(start * (1 - t) + end * t)


max_count = float(top_priors["count"].max()) if len(top_priors) else 1.0
freq_scale = (top_priors["count"].astype(float) / max_count).clip(0.78, 1.0).tolist()

TGT_W = float(img_wh["img_w"].median())
TGT_H = float(img_wh["img_h"].median())
CLAMP_HI_X_DEFAULT = max(1024.0, TGT_W * 1.05)
CLAMP_HI_Y_DEFAULT = max(1024.0, TGT_H * 1.05)


def _clamp(v, lo=0.0, hi=1024.0):
    return float(max(lo, min(hi, float(v))))




## === cell 1
def _find_test_dicom_dir():
    candidates = [
        os.path.join(BASE_INPUT, "test"),
        "/kaggle/data/test",
        "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/test",
        "/kaggle/input/test",
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return None


TEST_DICOM_DIR = _find_test_dicom_dir()

try:
    import pydicom  # type: ignore
except Exception:
    pydicom = None


def _get_test_hw(image_id: str):
    """Return (W,H) for the given test image. If unavailable, return median fallback."""
    if TEST_DICOM_DIR is None or pydicom is None:
        return TGT_W, TGT_H

    dcm_path = os.path.join(TEST_DICOM_DIR, f"{image_id}.dicom")
    if not os.path.exists(dcm_path):
        return TGT_W, TGT_H

    try:
        ds = pydicom.dcmread(dcm_path, stop_before_pixels=False, force=True)
        if hasattr(ds, "Columns") and hasattr(ds, "Rows"):
            w = float(ds.Columns)
            h = float(ds.Rows)
            if w > 1 and h > 1:
                return w, h
        arr = ds.pixel_array
        if arr is not None and len(arr.shape) >= 2:
            h = float(arr.shape[0])
            w = float(arr.shape[1])
            if w > 1 and h > 1:
                return w, h
    except Exception:
        return TGT_W, TGT_H

    return TGT_W, TGT_H


def build_pred_string_for_wh(img_w: float, img_h: float):
    parts = []
    k = len(top_priors)

    EXPAND_TIGHT = 0.14
    EXPAND_LOOSE = 0.26
    EXTRA_BOX_CONF_MULT = 0.62

    SHIFT_FRAC = 0.06
    SHIFT_BOX_CONF_MULT = 0.40

    SHIFT_BOX2_CONF_MULT = 0.34  # keep FPs controlled

    clamp_hi_x = max(1024.0, float(img_w) * 1.02)
    clamp_hi_y = max(1024.0, float(img_h) * 1.02)

    for i, row in enumerate(top_priors.itertuples(index=False)):
        cls = int(row.class_id)
        base_conf = _rank_conf(i, k) * float(freq_scale[i])
        base_conf = float(max(0.02, min(0.26, base_conf)))

        x1 = float(row.x_min_n) * float(img_w)
        y1 = float(row.y_min_n) * float(img_h)
        x2 = float(row.x_max_n) * float(img_w)
        y2 = float(row.y_max_n) * float(img_h)

        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        bw = max(1.0, (x2 - x1))
        bh = max(1.0, (y2 - y1))

        variants = [
            (EXPAND_TIGHT, 1.0, 0.0, 0.0),
            (EXPAND_LOOSE, EXTRA_BOX_CONF_MULT, 0.0, 0.0),
            (EXPAND_LOOSE, SHIFT_BOX_CONF_MULT, +SHIFT_FRAC, +SHIFT_FRAC),
            (EXPAND_LOOSE, SHIFT_BOX2_CONF_MULT, -SHIFT_FRAC, -SHIFT_FRAC),
        ]

        for expand, conf_mult, shift_fx, shift_fy in variants:
            conf = float(max(0.01, min(0.26, base_conf * conf_mult)))

            bw2 = bw * (1.0 + expand)
            bh2 = bh * (1.0 + expand)

            cx2 = cx + shift_fx * bw
            cy2 = cy + shift_fy * bh

            xx1 = _clamp(cx2 - 0.5 * bw2, hi=clamp_hi_x)
            yy1 = _clamp(cy2 - 0.5 * bh2, hi=clamp_hi_y)
            xx2 = _clamp(cx2 + 0.5 * bw2, hi=clamp_hi_x)
            yy2 = _clamp(cy2 + 0.5 * bh2, hi=clamp_hi_y)

            if xx2 <= xx1:
                xx2 = min(clamp_hi_x, xx1 + 1.0)
            if yy2 <= yy1:
                yy2 = min(clamp_hi_y, yy1 + 1.0)

            parts.append(f"{cls} {conf:.4f} {xx1:.1f} {yy1:.1f} {xx2:.1f} {yy2:.1f}")

    parts.append("14 1 0 0 1 1")
    return " ".join(parts)


pred_strings = []
for image_id in df_final["image_id"].astype(str).tolist():
    w, h = _get_test_hw(image_id)
    pred_strings.append(build_pred_string_for_wh(w, h))

df_final["PredictionString"] = pred_strings




## === cell 2
df_final["PredictionString"] = df_final["PredictionString"].apply(
    lambda s: " ".join(str(s).split()) if isinstance(s, str) else "14 1 0 0 1 1"
)




## === cell 3
OUT_PATH = "submission.csv"
df_final.to_csv(OUT_PATH, index=False)

print(
    f"Wrote {OUT_PATH} with shape={df_final.shape} and columns={list(df_final.columns)}"
)
df_final.head()




## === cell 4
assert (
    df_final.shape[0] == sample_sub.shape[0]
), "Row count mismatch vs sample_submission."
assert (
    df_final["PredictionString"].astype(str).str.len() > 0
).all(), "Found empty prediction strings."
df_final.tail()
