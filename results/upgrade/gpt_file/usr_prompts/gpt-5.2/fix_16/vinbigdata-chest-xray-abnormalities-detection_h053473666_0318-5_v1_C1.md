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

0.2270991441671168

# 6. Current score

0.03231

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0475) has done: 'Your notebook is currently an “ensemble blender” that depends on multiple external Kaggle datasets (e.g., `vin-big-data-predit`, `densenet201-vin-big-data-predit`) that are not available in your environment, causing the `FileNotFoundError` and cascading `NameError`s. To make it run end-to-end and create a valid submission, I replace those missing inputs with the only guaranteed available file: the competition `sample_submission.csv`, and I output a valid “No finding” prediction for every test image (this is score-valid but likely low). I also fix the submission column names to match the competition’s expected format (`image_id,PredictionString`) and ensure `submission.csv` is written in the current working directory. This keeps the changes minimal and focuses on correctness/stability so you can submit successfully.'
- What this solution (achieved 0.0516) has done: 'Your current submission predicts “No finding” for every image, which is valid but far below the target mAP. To move the score toward 0.227, the smallest legitimate improvement (without changing any modeling/training logic) is to use the provided `train.csv` to compute per-class prior frequencies and output a small, fixed set of the most common classes with a conservative central bounding box for every test image (plus “No finding” with reduced confidence). This typically increases recall and therefore mAP compared with all-negative predictions, while keeping the pipeline simple and deterministic. I also keep the submission schema exactly as required and ensure the CSV is written end-to-end.'
- What this solution (achieved 0.06606) has done: 'You’re currently using a single global “median box” for all top classes, which tends to miss IoU>0.4 for many findings and caps mAP. To move the score upward toward 0.227 with minimal logic change, I keep the same prior-based heuristic but compute a separate median bounding box per predicted class (still fixed per class, not per image), which usually increases localization quality without changing the overall approach. I also slightly increase `top_k` from 3→4 to raise recall a bit while keeping confidences conservative to limit precision damage. The submission format and paths remain unchanged and the notebook still run end-to-end and write `submission.csv`.'
- What this solution (achieved 0.066) has done: 'Your current heuristic is limited mainly by localization: a single median box per class (over all images) doesn’t match the typical “where” for many classes, which suppresses IoU>0.4 and caps mAP. Keeping the same prior-based approach and fixed-per-class boxes, I compute class boxes after trimming extreme outliers (use 10–90% quantiles) to get more representative boxes, and I use separate medians for width/height and center to stabilize shapes. I also raise `top_k` slightly (4→5) but reduce per-class confidences a bit to limit precision loss from extra false positives, aiming to move the score upward toward the 0.227 target without changing the overall method. Output format, paths, and the “No finding” fallback remain unchanged and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.066) has done: 'Your current heuristic is recall-heavy but still limited by IoU>0.4 localization; the most “minimal” way to improve toward 0.227 without changing the overall approach is to make the fixed-per-class boxes more representative of each class’s typical spatial location/size. I keep the same prior-based top‑K class prediction, but compute *robust per-class boxes* using (1) clipping to the image canvas inferred from train boxes, and (2) using medians of *normalized* (cx,cy,w,h) by per-image extent to reduce scale variability, then mapping back to pixel space via global typical width/height. I also slightly tune confidences (a bit higher for the first 2 classes, slightly lower for the tail and No Finding) to balance precision/recall without changing evaluation semantics. The script still runs end-to-end, uses only available files, and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.06393) has done: 'You’re currently far below the target (0.066 vs 0.227, higher-is-better), so we need a small but meaningful boost without changing the core “prior + fixed per-class boxes” heuristic. The biggest low-risk gain is to predict **more than one box per class** (still fixed, still derived only from `train.csv`) by splitting each class’s boxes into a few size-based clusters and emitting 2–3 representative boxes for the top classes; this raises recall and improves the chance of IoU>0.4 matches while keeping the same overall approach. I also set confidences in a simple descending schedule across all emitted boxes and keep a low-confidence “No finding” fallback to preserve validity. All paths and the submission schema (`image_id,PredictionString`) remain unchanged and the script still writes `submission.csv` end-to-end.'
- What this solution (achieved 0.06367) has done: 'Your current heuristic is already stable but likely under-scoring because it predicts the same set of classes for every image (many false positives) and spreads confidence across many boxes, which hurts precision in mAP. To move the score upward toward the 0.227 target with minimal change to the same “priors + fixed per-class boxes from train.csv” core logic, I (1) compute per-class *No Finding rate* from training data and use it to reduce how often we emit findings for all images, and (2) slightly reallocate confidence mass: fewer total predicted boxes per image with a bit higher confidence for the first boxes. This keeps everything deterministic, uses only `train.csv` + `sample_submission.csv`, and still always includes a valid “No finding” fallback. The output path and submission schema stay identical and it still writes `submission.csv` end-to-end.'
- What this solution (achieved 0.04996) has done: 'Your current heuristic likely underperforms because it emits the same set of findings for every test image, creating many false positives that heavily depress precision in mAP. To move the score upward toward the 0.227 target while preserving the same “priors + fixed per-class boxes from train.csv” core logic, I (1) introduce a deterministic per-image gating mechanism using only the image_id hash to decide whether to emit findings vs mostly “No finding”, with the emission rate calibrated to the training set’s `p_any_finding`. Then (2) for images where we do emit findings, I slightly reduce the number of emitted boxes and concentrate confidence into fewer predictions to improve precision without changing any modeling/training approach. The rest (class priors, robust per-class box computation, submission format, paths) stays the same and still produces a valid `submission.csv`.'
- What this solution (achieved 0.05062) has done: 'Your current heuristic likely under-scores because the per-image “emit findings” gate is too conservative and also emits a low-confidence “No finding” even when it decides to emit findings, which can still suppress mAP. I keep the same core logic (priors + fixed per-class boxes + deterministic hash gating), but (1) raise the emit rate a bit toward the training `p_any_finding`, and (2) only include the “No finding” box when we *don’t* emit findings (or give it a very small confidence when emitting), which typically improves precision/recall balance for mAP. I also slightly concentrate confidence into fewer boxes by reducing `max_total_boxes_emit` by 1 to cut false positives while keeping the same top-classes and box construction. The script still run end-to-end and write a valid `submission.csv` with `image_id,PredictionString`.'
- What this solution (achieved 0.03196) has done: 'We keep your exact “priors + fixed per-class boxes + deterministic hash gating” heuristic, but tune it slightly to increase mAP toward the 0.227 target by improving the precision/recall balance. Concretely: (1) raise the emit rate a bit (your current gating is likely suppressing recall too much), (2) remove the “No finding” box when we emit findings (it can hurt ranking/precision), and (3) slightly increase early confidences while reducing the number of emitted boxes to cut false positives. These are minimal parameter/post-processing adjustments that preserve the core logic and still produce a valid `submission.csv` end-to-end.'
- What this solution (achieved 0.02946) has done: 'Your current score (0.03196) is far below the target (0.2271), so we should increase it with the smallest, safest adjustments that preserve your existing “priors + fixed per-class boxes + deterministic hash gating” heuristic. The main issue is that the current gate is still too conservative for recall and you emit too few boxes per emitted image, which is especially damaging for mAP in detection. I (1) raise the emit rate toward “always emit” (still deterministic), (2) slightly increase the per-image box budget, and (3) soften the class-prior filter so more valid classes can be included, while keeping the same box construction and formatting. This should increase recall and move mAP upward toward the target without changing the fundamental approach.'
- What this solution (achieved 0.02174) has done: 'We keep your exact “priors + fixed per-class boxes + deterministic hash gating” heuristic, but adjust only the few knobs that are currently suppressing recall too much (which is why the score fell to 0.029). Specifically, we (1) increase the emit_rate (more images output findings instead of only “No finding”), (2) slightly raise the per-image box budget and reduce the prior filter strictness so more classes/boxes can pass through, and (3) temper the confidence schedule so additional boxes don’t get unrealistically high confidence (to limit precision collapse). These are minimal parameter-level changes that preserve your core logic and still write a valid `submission.csv` end-to-end.'
- What this solution (achieved 0.01894) has done: 'Your current heuristic is recall-limited by a tight per-image emit gate and a slightly strict class-prior filter, which together cause too many images to output only “No finding” and too few classes/boxes to be considered—this is consistent with the drop to 0.02174. To move the mAP upward toward the 0.227 target while preserving the exact “priors + fixed per-class boxes + deterministic hash gating” core logic, I only adjust three knobs: increase `emit_rate`, slightly relax `min_class_prior`, and increase `max_total_boxes_emit` by a small amount. I also keep confidence decay conservative so added boxes don’t unrealistically dominate ranking, which helps avoid precision collapse. The pipeline, paths, and submission schema remain unchanged and it still writes a valid `submission.csv` end-to-end.'
- What this solution (achieved 0.03047) has done: 'We keep your exact “priors + fixed per-class boxes + deterministic hash gating” heuristic, but adjust the few knobs that are currently killing mAP by flooding too many images with identical findings (precision collapse) while also not matching the train distribution of “any finding.” Concretely, we (1) set the emit rate to the training `p_any_finding` (instead of ~0.999), (2) reduce the per-image box budget to cut false positives, and (3) slightly raise the minimum class prior threshold so only more common classes are emitted. These are minimal parameter/post-processing changes that preserve your core logic and should move the score upward from 0.01894 toward the 0.227 target. The script still runs end-to-end and writes a valid `submission.csv` with `image_id,PredictionString`.'
- What this solution (achieved 0.03231) has done: 'Your current score (0.03047) is far below the target (0.2271), so we should increase it with minimal, parameter-level changes that keep your exact “priors + fixed per-class boxes + deterministic hash gating” core heuristic intact. The biggest low-risk issue is precision collapse from emitting the same fixed set of findings on too many images; we reduce the emit rate and box budget slightly, while making “No finding” the default more often. To compensate a bit on recall without changing the approach, we modestly expand the candidate class pool (top_k) but keep a stricter per-image cap so we don’t flood each image. Finally, we slightly adjust the confidence schedule so fewer boxes get higher confidence (better ranking for mAP) while still writing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import zlib

BASE_DIR_CANDIDATES = [
    "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection",
    "/kaggle/data/input/vinbigdata-chest-xray-abnormalities-detection",
    "/kaggle/input",
    "/kaggle/data/input",
    "/kaggle/data/vinbigdata-chest-xray-abnormalities-detection",
    "/kaggle/data",
]


def find_file(filename: str):
    for base in BASE_DIR_CANDIDATES:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
    for path in [
        f"../input/vinbigdata-chest-xray-abnormalities-detection/{filename}",
        f"../input/{filename}",
        f"../data/{filename}",
        filename,
    ]:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find {filename} in known Kaggle paths.")


sample_path = find_file("sample_submission.csv")
sub = pd.read_csv(sample_path)

if "image_id" not in sub.columns:
    if "ID" in sub.columns:
        sub = sub.rename(columns={"ID": "image_id"})
if "PredictionString" not in sub.columns:
    if "TARGET" in sub.columns:
        sub = sub.rename(columns={"TARGET": "PredictionString"})

if list(sub.columns)[:2] != ["image_id", "PredictionString"]:
    sub = sub[["image_id", "PredictionString"]]

sub.head()




## === cell 1
train_path = find_file("train.csv")
train_df = pd.read_csv(train_path)

df_findings = train_df[train_df["class_id"].astype(int) != 14].copy()

if len(df_findings) == 0:
    sub["PredictionString"] = "14 1 0 0 1 1"
else:
    img_has_finding = (
        train_df.assign(class_id=train_df["class_id"].astype(int))
        .groupby("image_id")["class_id"]
        .apply(lambda s: int((s != 14).any()))
    )
    p_any_finding = float(img_has_finding.mean())  # fraction of images with any non-14

    class_img_counts = (
        df_findings.assign(class_id=df_findings["class_id"].astype(int))
        .groupby("class_id")["image_id"]
        .nunique()
        .sort_values(ascending=False)
    )
    n_train_imgs = int(train_df["image_id"].nunique())
    class_img_priors = (class_img_counts / max(1, n_train_imgs)).to_dict()

    class_counts = (
        df_findings["class_id"].astype(int).value_counts().sort_values(ascending=False)
    )

    top_k = 8
    top_classes = class_counts.head(top_k).index.tolist()

    x_canvas = float(df_findings[["x_min", "x_max"]].max(axis=1).quantile(0.995))
    y_canvas = float(df_findings[["y_min", "y_max"]].max(axis=1).quantile(0.995))
    x_canvas = max(1.0, x_canvas)
    y_canvas = max(1.0, y_canvas)

    def _robust_box_from_group(g: pd.DataFrame, x_canvas: float, y_canvas: float):
        x1 = g[["x_min", "x_max"]].min(axis=1)
        x2 = g[["x_min", "x_max"]].max(axis=1)
        y1 = g[["y_min", "y_max"]].min(axis=1)
        y2 = g[["y_min", "y_max"]].max(axis=1)

        w = (x2 - x1).clip(lower=1.0)
        h = (y2 - y1).clip(lower=1.0)
        cx = x1 + 0.5 * w
        cy = y1 + 0.5 * h

        cxn = (cx / x_canvas).clip(0.0, 1.0)
        cyn = (cy / y_canvas).clip(0.0, 1.0)
        wn = (w / x_canvas).clip(1e-6, 1.0)
        hn = (h / y_canvas).clip(1e-6, 1.0)

        tmp = pd.DataFrame({"cxn": cxn, "cyn": cyn, "wn": wn, "hn": hn})

        qlow, qhigh = 0.10, 0.90
        for col in ["cxn", "cyn", "wn", "hn"]:
            lo = float(tmp[col].quantile(qlow))
            hi = float(tmp[col].quantile(qhigh))
            tmp2 = tmp[(tmp[col] >= lo) & (tmp[col] <= hi)]
            if len(tmp2) >= max(50, int(0.2 * len(tmp))):
                tmp = tmp2

        cx_m = float(tmp["cxn"].median()) * x_canvas
        cy_m = float(tmp["cyn"].median()) * y_canvas
        w_m = float(tmp["wn"].median()) * x_canvas
        h_m = float(tmp["hn"].median()) * y_canvas

        xmin = cx_m - 0.5 * w_m
        ymin = cy_m - 0.5 * h_m
        xmax = cx_m + 0.5 * w_m
        ymax = cy_m + 0.5 * h_m

        xmin = float(max(0.0, min(xmin, x_canvas - 1.0)))
        ymin = float(max(0.0, min(ymin, y_canvas - 1.0)))
        xmax = float(max(xmin + 1.0, min(xmax, x_canvas)))
        ymax = float(max(ymin + 1.0, min(ymax, y_canvas)))
        return (xmin, ymin, xmax, ymax)

    class_boxes = {}
    for cls in top_classes:
        g0 = df_findings[df_findings["class_id"].astype(int) == int(cls)].copy()
        if len(g0) == 0:
            continue

        x1 = g0[["x_min", "x_max"]].min(axis=1)
        x2 = g0[["x_min", "x_max"]].max(axis=1)
        y1 = g0[["y_min", "y_max"]].min(axis=1)
        y2 = g0[["y_min", "y_max"]].max(axis=1)
        w = (x2 - x1).clip(lower=1.0)
        h = (y2 - y1).clip(lower=1.0)
        area = (w * h).clip(lower=1.0)

        g0 = g0.assign(_area=area.values)

        if len(g0) >= 1500:
            qs = [0.0, 0.33, 0.66, 1.0]
        elif len(g0) >= 600:
            qs = [0.0, 0.5, 1.0]
        else:
            qs = [0.0, 1.0]

        boxes = []
        for i in range(len(qs) - 1):
            lo = float(g0["_area"].quantile(qs[i]))
            hi = float(g0["_area"].quantile(qs[i + 1]))
            if i == len(qs) - 2:
                gg = g0[(g0["_area"] >= lo) & (g0["_area"] <= hi)]
            else:
                gg = g0[(g0["_area"] >= lo) & (g0["_area"] < hi)]
            if len(gg) < 80:
                continue
            boxes.append(_robust_box_from_group(gg, x_canvas, y_canvas))

        if len(boxes) == 0:
            boxes = [_robust_box_from_group(g0, x_canvas, y_canvas)]

        class_boxes[int(cls)] = boxes

    pred_entries = []
    for rank, cls in enumerate(top_classes):
        cls = int(cls)
        if cls not in class_boxes:
            continue
        for b_idx, (xmin, ymin, xmax, ymax) in enumerate(class_boxes[cls]):
            pred_entries.append((cls, rank, b_idx, xmin, ymin, xmax, ymax))

    pred_entries.sort(key=lambda t: (t[1], t[2]))

    def _u01_from_id(image_id: str) -> float:
        return (zlib.crc32(image_id.encode("utf-8")) & 0xFFFFFFFF) / 2**32

    emit_rate = float(min(0.80, max(0.45, 0.90 * p_any_finding)))

    max_total_boxes_emit = 5

    min_class_prior = 0.030

    filtered_entries = []
    for t in pred_entries:
        cls = int(t[0])
        if class_img_priors.get(cls, 0.0) >= min_class_prior:
            filtered_entries.append(t)
        if len(filtered_entries) >= max_total_boxes_emit:
            break
    if len(filtered_entries) == 0:
        filtered_entries = pred_entries[: min(max_total_boxes_emit, len(pred_entries))]

    conf_nf_noemit = 1.0

    base_conf = 0.62
    conf_decay = 0.72
    min_conf = 0.05

    pred_strings = []
    for image_id in sub["image_id"].astype(str).tolist():
        u = _u01_from_id(image_id)

        if u >= emit_rate or len(filtered_entries) == 0:
            pred_strings.append(f"14 {conf_nf_noemit:.1f} 0 0 1 1")
            continue

        parts = []
        for i, (cls, _rank, _bidx, xmin, ymin, xmax, ymax) in enumerate(
            filtered_entries
        ):
            conf = max(min_conf, base_conf * (conf_decay**i))
            parts.append(
                f"{int(cls)} {conf:.4f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
            )

        pred_strings.append(" ".join(parts))

    sub["PredictionString"] = pred_strings

sub.isna().sum()




## === cell 2
out_path = "submission.csv"
sub.to_csv(out_path, index=False)

assert os.path.exists(out_path), "submission.csv was not created."
check = pd.read_csv(out_path)
assert list(check.columns) == [
    "image_id",
    "PredictionString",
], "Submission columns are incorrect."
assert len(check) == len(sub), "Row count mismatch in submission."
check.head()




## === cell 3
sub
