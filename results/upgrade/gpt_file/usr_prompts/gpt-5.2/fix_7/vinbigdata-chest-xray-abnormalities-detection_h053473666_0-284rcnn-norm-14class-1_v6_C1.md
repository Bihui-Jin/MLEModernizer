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

0.2455998704933153

# 6. Current score

0.03649

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0475) has done: 'Your notebook fails because it depends on external Kaggle Dataset inputs (`vin-15-cnn-predict`, `vin-15-cnn-predict-1`, `0284-norm`) that are not available in your environment, so the first `read_csv` raises `FileNotFoundError` and everything after that is undefined. To make it run end-to-end and still follow the competition submission format, I add a small “fallback” path: if those external files aren’t found, we create a valid submission directly from the provided `sample_submission.csv` (all “No finding”). This is score-minimal but guarantees a valid `submission.csv` is written with correct columns and formatting; it also preserves your original ensemble logic when the external inputs are present. I also fix the column-name mismatch (`PredictionString` vs `TARGET`) robustly and guard later cells so they don’t crash when optional files are missing.'
- What this solution (achieved 0.0) has done: 'Your current fallback always predicts “No finding”, which matches the submission format but caps performance far below your target. To move the score upward with minimal change and without adding any new model/training code, I keep your existing ensemble logic intact and add a lightweight, label-prior fallback that uses `train.csv` to estimate per-class prevalence and emits a small number of high-prior classes with full-image boxes for every test image. This remains a legitimate baseline (no leakage from test labels), produces a valid `submission.csv`, and should substantially improve over 0.0475 toward your target when the external prediction files aren’t available. I also make the sample-submission loader accept both `PredictionString` and `TARGET` robustly (unchanged behavior when already correct).'
- What this solution (achieved 0.0) has done: 'Your current score is 0.0 because the fallback creates invalid detections for this competition’s expected coordinate scale (you’re emitting a 1024×1024 box, which is often far outside the actual DICOM pixel dimensions, so IoU becomes ~0). I keep your ensemble path untouched, but improve only the fallback by (1) estimating a realistic “typical” image width/height from `train.csv` boxes, then (2) emitting full-image boxes using that typical size so boxes align better with ground truth scale. I also slightly increase `top_k` and use a conservative confidence derived from class prevalence to improve recall without overconfident spam. This stays within your existing “label-prior fallback” core idea and only changes how fallback boxes/confidences are formed.'
- What this solution (achieved 0.03354) has done: 'Your current 0.0 is consistent with the fallback producing the *same multi-class full-image prediction for every test image*, which badly hurts per-class AP ranking and can collapse mAP. I keep your ensemble path unchanged, and only adjust the fallback so it becomes a legitimate “prior + no-finding gating” heuristic: for each image, either predict `No finding` or predict a small set of common classes with full-image boxes. This keeps the same lightweight, no-model approach but makes predictions vary in confidence and reduces false positives, which should move the score up toward your target. I also compute the “typical” box size more robustly from train boxes and avoid ever emitting obviously out-of-range coordinates.'
- What this solution (achieved 0.03666) has done: 'Your current gap to the target is large (0.03354 → 0.2456), and the biggest issue is the fallback: it emits the same full-image boxes for every predicted class, so IoU is usually poor and AP collapses. I keep your ensemble path untouched and only improve the fallback by (1) making the pseudo-boxes look more like typical ground-truth boxes using train-set box width/height quantiles, and (2) using deterministic per-image jitter for box center/size so predictions aren’t identical across all test images. I also make the “no-finding vs findings” gate deterministic and slightly tune `top_k`/confidence so recall improves without flooding too many false positives. These are minimal, metric-aligned changes that should move mAP upward toward your target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.03649) has done: 'Your current score (0.03666) is far below the target (0.2456), and the biggest controllable lever (without changing your core ensemble logic) is the fallback: it currently guesses generic boxes/classes per image, which yields poor IoU and weak AP. I keep the external-ensemble path completely intact, and only change the fallback to generate pseudo-boxes by sampling from the empirical distribution of train boxes (size + position), so predicted boxes look more like real annotations and IoU improves. I also switch the per-image randomness from Python’s salted `hash()` (non-deterministic across runs) to a stable md5-based RNG, improving reproducibility and allowing consistent tuning. Finally, I reduce false positives by predicting fewer classes per positive image (top_k) and by using train-derived class priors for confidence calibration, which typically raises mAP compared with flooding many classes.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import hashlib

BASE_INPUT = "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection"
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(BASE_INPUT, "train.csv")


def load_sample_submission(path=SAMPLE_SUB_PATH):
    sub = pd.read_csv(path)
    if "PredictionString" not in sub.columns:
        if "TARGET" in sub.columns:
            sub = sub.rename(columns={"TARGET": "PredictionString"})
        else:
            raise ValueError(
                f"Sample submission missing PredictionString/TARGET columns: {sub.columns.tolist()}"
            )
    if "image_id" not in sub.columns:
        if "ID" in sub.columns:
            sub = sub.rename(columns={"ID": "image_id"})
        else:
            raise ValueError(
                f"Sample submission missing image_id/ID columns: {sub.columns.tolist()}"
            )
    return sub[["image_id", "PredictionString"]].copy()




## === cell 1
external_paths_0 = "../input/vin-15-cnn-predict"
external_paths_1 = "../input/vin-15-cnn-predict-1"

dfs = {}
missing = []

for i in range(10):
    p = os.path.join(external_paths_0, f"submission{i}.csv")
    if os.path.exists(p):
        dfs[i] = pd.read_csv(p)
    else:
        missing.append(p)

for i in range(10, 15):
    p = os.path.join(external_paths_1, f"submission{i - 10}.csv")
    if os.path.exists(p):
        dfs[i] = pd.read_csv(p)
    else:
        missing.append(p)

use_fallback = len(missing) > 0

if use_fallback:
    df = load_sample_submission()
else:
    list1 = [str(i) for i in range(15)]
    df0 = dfs[0]
    for k in range(1, 15):
        if k not in dfs:
            raise FileNotFoundError(f"Expected dfs[{k}] to exist but it does not.")
    for c in ["image_id"] + list1:
        if c not in df0.columns:
            raise ValueError(
                f"Expected column '{c}' in ensemble input but not found. Columns: {df0.columns.tolist()}"
            )
    df0[list1] = (
        dfs[0][list1] * 0.05
        + dfs[1][list1] * 0.05
        + dfs[2][list1] * 0.05
        + dfs[3][list1] * 0.05
        + dfs[4][list1] * 0.05
        + dfs[5][list1] * 0.1
        + dfs[6][list1] * 0.1
        + dfs[7][list1] * 0.1
        + dfs[8][list1] * 0.1
        + dfs[9][list1] * 0.1
        + dfs[10][list1] * 0.05
        + dfs[11][list1] * 0.05
        + dfs[12][list1] * 0.05
        + dfs[13][list1] * 0.05
        + dfs[14][list1] * 0.05
    )
    df = df0.copy()

df.head()



## === cell 2
df



## === cell 3
df3_path = "../input/0284-norm/cascade_rcnn_x101_32x4d_fpn_20e_OHEM_fpn.4_with_raw.4_ensemble.5.bbox.json.filter.norm (1).csv"

df3 = None
if os.path.exists(df3_path) and (not use_fallback):
    df3 = pd.read_csv(df3_path)



## === cell 4
if (df3 is not None) and ("image_id" in df.columns) and ("image_id" in df3.columns):
    df4 = pd.merge(df, df3, on="image_id", how="left")
else:
    df4 = df.copy()

df4.head()



## === cell 5
val = None
try:
    val = df4.iloc[1, 16]
except Exception:
    val = None
val



## === cell 6
if "PredictionString" not in df4.columns:
    df4["PredictionString"] = "14 1 0 0 1 1"

if use_fallback:
    train_usecols = ["image_id", "class_id", "x_min", "y_min", "x_max", "y_max"]
    train_df = pd.read_csv(TRAIN_CSV_PATH, usecols=train_usecols)

    pos = (
        train_df[train_df["class_id"] != 14]
        .dropna(subset=["x_min", "y_min", "x_max", "y_max"])
        .copy()
    )
    for c in ["x_min", "y_min", "x_max", "y_max"]:
        pos[c] = pd.to_numeric(pos[c], errors="coerce")
    pos = pos.dropna(subset=["x_min", "y_min", "x_max", "y_max"])

    pos["w"] = (pos["x_max"] - pos["x_min"]).clip(lower=1.0)
    pos["h"] = (pos["y_max"] - pos["y_min"]).clip(lower=1.0)

    per_image_max = pos.groupby("image_id")[["x_max", "y_max"]].max()
    typical_w = float(per_image_max["x_max"].quantile(0.95))
    typical_h = float(per_image_max["y_max"].quantile(0.95))
    if not np.isfinite(typical_w) or typical_w <= 2:
        typical_w = 1024.0
    if not np.isfinite(typical_h) or typical_h <= 2:
        typical_h = 1024.0
    typical_w = int(np.clip(round(typical_w), 256, 4096))
    typical_h = int(np.clip(round(typical_h), 256, 4096))

    pos["cx"] = (pos["x_min"] + pos["x_max"]) / 2.0
    pos["cy"] = (pos["y_min"] + pos["y_max"]) / 2.0

    pos_unique = pos.drop_duplicates(["image_id", "class_id"])
    n_images_with_any = max(pos_unique["image_id"].nunique(), 1)
    class_prior = (
        pos_unique["class_id"].value_counts() / n_images_with_any
    ).sort_values(ascending=False)

    top_k = 3
    top_classes = [int(c) for c in class_prior.index.tolist()[:top_k]]

    n_total_train_imgs = max(train_df["image_id"].nunique(), 1)
    frac_any = float(pos_unique["image_id"].nunique() / n_total_train_imgs)
    p_no = float(np.clip(1.0 - frac_any, 0.05, 0.95))

    def u01(s: str, salt: str = "") -> float:
        m = hashlib.md5((s + "|" + salt).encode("utf-8")).hexdigest()
        return (int(m[:8], 16) % 1_000_000) / 1_000_000.0

    global_cx = pos["cx"].to_numpy()
    global_cy = pos["cy"].to_numpy()
    global_w = pos["w"].to_numpy()
    global_h = pos["h"].to_numpy()
    n_global = len(pos)

    per_class = {}
    for cid in top_classes:
        sub = pos[pos["class_id"] == cid]
        if len(sub) >= 200:
            per_class[cid] = (
                sub["cx"].to_numpy(),
                sub["cy"].to_numpy(),
                sub["w"].to_numpy(),
                sub["h"].to_numpy(),
            )

    def pick_box_params(img_id: str, cid: int):
        r = u01(img_id, f"pick_{cid}")
        if cid in per_class:
            cx_arr, cy_arr, w_arr, h_arr = per_class[cid]
            n = len(cx_arr)
            idx = int(r * n) % n
            cx, cy, w, h = (
                float(cx_arr[idx]),
                float(cy_arr[idx]),
                float(w_arr[idx]),
                float(h_arr[idx]),
            )
        else:
            idx = int(r * n_global) % n_global if n_global > 0 else 0
            cx = float(global_cx[idx]) if n_global > 0 else typical_w / 2.0
            cy = float(global_cy[idx]) if n_global > 0 else typical_h / 2.0
            w = float(global_w[idx]) if n_global > 0 else typical_w * 0.3
            h = float(global_h[idx]) if n_global > 0 else typical_h * 0.3

        jx = (u01(img_id, f"jx_{cid}") - 0.5) * 0.10 * w
        jy = (u01(img_id, f"jy_{cid}") - 0.5) * 0.10 * h
        jsw = 1.0 + (u01(img_id, f"jsw_{cid}") - 0.5) * 0.15
        jsh = 1.0 + (u01(img_id, f"jsh_{cid}") - 0.5) * 0.15

        cx = cx + jx
        cy = cy + jy
        w = max(1.0, w * jsw)
        h = max(1.0, h * jsh)
        return cx, cy, w, h

    def make_box(img_id: str, cid: int):
        cx, cy, w, h = pick_box_params(img_id, cid)

        x0 = int(np.clip(np.floor(cx - w / 2.0), 0, typical_w - 2))
        y0 = int(np.clip(np.floor(cy - h / 2.0), 0, typical_h - 2))
        x1 = int(np.clip(np.ceil(cx + w / 2.0), x0 + 1, typical_w))
        y1 = int(np.clip(np.ceil(cy + h / 2.0), y0 + 1, typical_h))
        return x0, y0, x1, y1

    preds = []
    for img_id in df4["image_id"].astype(str).tolist():
        u = u01(img_id, "gate")
        if u < p_no:
            preds.append("14 1 0 0 1 1")
            continue

        parts = []
        for cid in top_classes:
            prior = float(class_prior.get(cid, 0.05))

            base = 0.08 + 0.55 * prior
            jitter = (u01(img_id, f"conf_{cid}") - 0.5) * 0.03
            conf = float(np.clip(base + jitter, 0.05, 0.35))

            x0, y0, x1, y1 = make_box(img_id, cid)
            parts.extend([str(cid), f"{conf:.6f}", str(x0), str(y0), str(x1), str(y1)])

        preds.append(" ".join(parts) if parts else "14 1 0 0 1 1")

    df4["PredictionString"] = preds

df4[["image_id", "PredictionString"]].head()



## === cell 7
list1_int = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
needed_cols = [str(k) for k in list1_int]

can_adjust = (
    (not use_fallback)
    and all(c in df4.columns for c in needed_cols)
    and ("PredictionString" in df4.columns)
)

can_adjust



## === cell 8
if can_adjust:
    for i in range(df4.shape[0]):
        if df4.loc[i, "PredictionString"] == "14 1 0 0 1 1":
            continue
        a = df4.loc[i, "PredictionString"]
        b = a.split()
        for j in range(int(len(a.split()) / 6)):
            for k in list1_int:
                if int(b[0 + 6 * j]) == k:
                    if df4.loc[i, f"{k}"] < 0.9:
                        continue
                    c = b[0 + 6 * j + 1]
                    b[0 + 6 * j + 1] = str(df4.loc[i, f"{k}"] * 0.15 + float(c) * 0.85)
        df4.loc[i, "PredictionString"] = " ".join(b)



## === cell 9
if (not use_fallback) and ("14" in df4.columns):
    for i in range(df4.shape[0]):
        if df4.loc[i, "14"] > 0.999:
            df4.loc[i, "PredictionString"] = "14 1 0 0 1 1"
else:
    df4["PredictionString"] = df4.get("PredictionString", "14 1 0 0 1 1").fillna(
        "14 1 0 0 1 1"
    )

df_final = df4[["image_id", "PredictionString"]].copy()
df_final.to_csv("submission.csv", index=False)

df_final.head()



## === cell 10
df_final
