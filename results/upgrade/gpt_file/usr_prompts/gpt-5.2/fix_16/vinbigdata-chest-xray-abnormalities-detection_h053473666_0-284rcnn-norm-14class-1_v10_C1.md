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

0.2615307101474436

# 6. Current score

0.01527

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0475) has done: 'Your notebook fails because it depends on multiple external Kaggle datasets (`vin-15-cnn-predict`, `vin-15-cnn-predict-1`, `0284-norm`) that are not available in this environment, so `read_csv` immediately raises `FileNotFoundError` and nothing downstream is defined. To make it run end-to-end and still produce a valid submission, I replace those missing-file reads with a robust fallback that uses the provided `sample_submission.csv` as the base and outputs the required `PredictionString` for every test image. This keeps the “submission-writing” semantics intact and guarantees a valid `submission.csv` is created within the constraints of available packages/data. Because no workable model predictions are present, this not reach the target score, but it unblock execution and produce a correctly formatted file you can submit.'
- What this solution (achieved 0.01849) has done: 'Your current pipeline mostly outputs “No finding” for every image because the external prediction files are missing, which caps mAP near your current score. To move toward the target with minimal change and without altering your overall structure, I add a lightweight, data-only fallback that derives per-class typical box sizes from `train.csv` and emits a small number of plausible boxes per image (instead of always “No finding”). This keeps the same submission semantics (PredictionString formatting) and only activates when the external ensemble inputs are unavailable, so it should improve score relative to 0.0475 while staying simple and stable. I also remove a small execution hazard (`_` potentially undefined) to guarantee end-to-end runs.'
- What this solution (achieved 0.0475) has done: 'Your current score is far below the target, so we should improve (not degrade) the predictions with the smallest, safest change that keeps your “fallback-only” structure when external files are missing. The simplest high-impact fix is to stop emitting the same 3 class boxes for every image and instead use the per-image class probabilities you already have in `df4[str(k)]` to choose the top classes and confidence scores (this aligns better with mAP than constant guesses). We still use train-derived “typical boxes”, but we make them class-specific and a bit more realistic by using robust medians and clamping, and we ensure we output “No finding” only when the model scores are very low. These changes keep your overall pipeline intact (read external if present → merge → fallback) and should move score upward toward the target.'
- What this solution (achieved 0.01788) has done: 'Your current score (0.0475) is far below the target (0.2615), so we should improve predictions while keeping your “external files if present, otherwise fallback” structure intact. The biggest low-risk gain is to make the fallback emit more realistic, image-specific detections by (1) estimating how many boxes to output from the learned score mass, (2) using per-class median boxes but *adding a small set of class-specific alternate boxes* (quantiles) so multiple predicted objects aren’t identical, and (3) preventing “No finding” from being output too often when some class scores are present. These are minimal changes localized to the fallback block and do not change your overall pipeline, I/O, or submission semantics. I also keep the existing post-processing in cell 7 unchanged so behavior stays consistent when external predictions exist.'
- What this solution (achieved 0.01706) has done: 'The timeout is dominated by the “no external mode” path which reads and decodes up to 1500 DICOMs and computes per-image stats; this is heavy I/O + CPU and can exceed 600s. I keep the exact prediction logic intact, but remove unnecessary work by (1) ensuring that expensive DICOM work never triggers unless it truly must, and (2) making DICOM reading as cheap as possible when it does trigger (avoid PixelData in the tag list, preallocate, and avoid Python overhead in the loop). Additionally, I replace a few slow pandas operations with equivalent vectorized/numpy operations (same semantics) and reduce merge/column handling overhead. The output submission format and all paths remain unchanged.'
- What this solution (achieved 0.01626) has done: 'Your current score is far below the target (0.01706 vs 0.2615), and the main issue is that in the “no external predictions available” path you only emit a few generic boxes per image, which yields very low recall and thus low mAP. I keep your overall fallback structure and formatting unchanged, but make the fallback produce more detections per image in a controlled way by (1) using train-derived per-class frequency priors to decide how many boxes to output, and (2) adding 2–3 deterministic “jittered” variants of each class median box so multiple instances aren’t identical and cover more IoU matches. I also prevent the fallback from outputting “No finding” too aggressively by lowering the no-object cutoff slightly, which should improve recall and move mAP upward toward the target without changing the core logic or requiring new packages/files. All paths remain the same and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.01527) has done: 'Your current score is far below the target, so we should improve (not degrade) the fallback predictions while keeping your overall “external predictions if present, otherwise fallback” structure intact. The smallest high-impact fix is that your fallback currently ignores the per-image per-class scores in `df4[str(k)]` whenever those columns exist-but-are-zero (due to missing external files), so it ends up using only global priors + hash noise; instead we should derive image-specific class scores from the already-available `PredictionString` detections when present, and otherwise fall back to priors. Concretely, we (1) parse each row’s existing `PredictionString` into a per-class confidence vector, (2) mix it with priors (light smoothing) to rank classes per image, and (3) emit a few more boxes only for the top classes with confidence tied to that mixed score. This keeps the same output semantics and data sources, and should raise mAP toward the target by increasing recall on images where your pipeline already produced some non-default detections.'

# 9. Code solution

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
df4 = pd.merge(df, df3, on="image_id", how="left", suffixes=("", "_df3"))

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

score_arr = df4[score_cols].to_numpy(copy=False)
score_arr = pd.DataFrame(score_arr).apply(pd.to_numeric, errors="coerce").to_numpy()
np.nan_to_num(score_arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
df4[score_cols] = score_arr.astype(np.float64, copy=False)

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
    ps = df4["PredictionString"].astype(str)
    frac_default = (ps.str.strip().to_numpy() == "14 1 0 0 1 1").mean()
    score_sum = float(np.asarray(df4[score_cols].to_numpy(copy=False)).sum())
    if frac_default > 0.95 and score_sum == 0.0:
        no_external_mode = True

if no_external_mode and len(fallback_boxes) > 0:
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

    def _img_hash01(image_id: str) -> float:
        x = 0
        for ch in str(image_id):
            x = (x * 131 + ord(ch)) % 1000003
        return (x % 1000000) / 1000000.0

    jitter_boxes = {}  # cid -> list of boxes
    xmins = [b[0] for b in fallback_boxes.values()]
    ymins = [b[1] for b in fallback_boxes.values()]
    xmaxs = [b[2] for b in fallback_boxes.values()]
    ymaxs = [b[3] for b in fallback_boxes.values()]
    if xmins:
        x_low = float(np.min(xmins))
        y_low = float(np.min(ymins))
        x_high = float(np.max(xmaxs))
        y_high = float(np.max(ymaxs))
    else:
        x_low, y_low, x_high, y_high = 0.0, 0.0, 1024.0, 1024.0

    def _clamp_box2(xmin, ymin, xmax, ymax):
        xmin = max(x_low, float(xmin))
        ymin = max(y_low, float(ymin))
        xmax = min(x_high, float(xmax))
        ymax = min(y_high, float(ymax))
        if xmax <= xmin:
            xmax = xmin + 1.0
        if ymax <= ymin:
            ymax = ymin + 1.0
        return (xmin, ymin, xmax, ymax)

    for cid, box in fallback_boxes.items():
        xmin, ymin, xmax, ymax = box
        w = max(2.0, xmax - xmin)
        h = max(2.0, ymax - ymin)
        dx = 0.08 * w
        dy = 0.08 * h
        variants = [
            _clamp_box2(xmin, ymin, xmax, ymax),
            _clamp_box2(xmin + dx, ymin + dy, xmax + dx, ymax + dy),
            _clamp_box2(xmin - dx, ymin - dy, xmax - dx, ymax - dy),
        ]
        if (cid, 0) in fallback_boxes_alt:
            variants.append(fallback_boxes_alt[(cid, 0)])
        if (cid, 1) in fallback_boxes_alt:
            variants.append(fallback_boxes_alt[(cid, 1)])
        jitter_boxes[cid] = variants

    image_ids = df4["image_id"].astype(str).to_numpy()
    out_ps = [None] * len(image_ids)

    ps_in = df4["PredictionString"].fillna("").astype(str).to_numpy()

    def _ps_to_vec(ps: str):
        v = np.zeros(14, dtype=np.float64)
        if not ps or ps.strip() == "" or ps.strip() == "14 1 0 0 1 1":
            return v
        b = ps.split()
        if len(b) % 6 != 0:
            return v
        n = len(b) // 6
        for j in range(n):
            try:
                cid = int(b[6 * j])
                conf = float(b[6 * j + 1])
            except Exception:
                continue
            if 0 <= cid <= 13:
                if conf > v[cid]:
                    v[cid] = conf
        return v

    _pri = priors.get
    for i, image_id in enumerate(image_ids):
        h = _img_hash01(image_id)

        ps_vec = _ps_to_vec(ps_in[i])

        cls_scores = []
        for cid in range(14):
            base = float(_pri(cid, 0.0))
            sc = 0.70 * base + 0.30 * float(ps_vec[cid])
            sc = sc * (0.95 + 0.10 * ((h + cid * 0.113) % 1.0))
            cls_scores.append((cid, sc))
        cls_scores.sort(key=lambda x: x[1], reverse=True)

        obj_mass = sum(s for _, s in cls_scores[:6])
        if obj_mass < 0.075:
            out_ps[i] = "14 1 0 0 1 1"
            continue

        if obj_mass > 0.35:
            K = 11
        elif obj_mass > 0.26:
            K = 9
        elif obj_mass > 0.18:
            K = 7
        else:
            K = 6

        parts = []
        picked = 0
        for rank, (cid, sc) in enumerate(cls_scores):
            if picked >= K:
                break
            if cid not in jitter_boxes:
                continue

            conf = max(0.03, min(0.65, (sc * 3.0) * (0.99 - 0.05 * rank)))

            v = int(((h * 1000.0) + cid * 7.0) % len(jitter_boxes[cid]))
            xmin, ymin, xmax, ymax = jitter_boxes[cid][v]

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
            picked += 1

        out_ps[i] = " ".join(parts) if parts else "14 1 0 0 1 1"

    df4["PredictionString"] = out_ps

df4.head()




## === cell 5
_ = None
if len(df4.columns) > 16 and len(df4) > 1:
    _ = df4.iloc[1, 16]
_




## === cell 6
list1 = set(range(15))
score_mat = df4[score_cols].to_numpy(copy=False)  # row-aligned with df4
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
