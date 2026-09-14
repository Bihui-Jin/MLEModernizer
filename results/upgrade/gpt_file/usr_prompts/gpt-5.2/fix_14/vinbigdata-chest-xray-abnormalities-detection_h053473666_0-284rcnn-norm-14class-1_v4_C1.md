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

0.2451838026465415

# 6. Current score

0.03094

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0475) has done: 'The original notebook depends on external Kaggle datasets (multiple `../input/.../submission*.csv` ensemble files) that are not present in your environment, causing immediate `FileNotFoundError` and preventing any submission from being written. To make it run end-to-end, I replace the missing ensemble inputs with the competition’s provided `sample_submission.csv` as a stable base, then add the expected per-class probability columns (`'0'..'14'`) so later logic can execute without changing its semantics. I also guard the optional second merge (also pointing to a missing external file) so the pipeline continues even when that file is unavailable. Finally, I keep your original post-processing loop but ensure it runs on a valid dataframe and writes `submission.csv` with the correct columns and `.csv` suffix.'
- What this solution (achieved 0.01849) has done: 'Your current score is far below the target because your pipeline never actually injects any real detections (the missing ensemble file means `PredictionString` stays as “No finding” for all images), so mAP stays near zero. To move toward the target while preserving your core post-processing semantics, I generate a simple, legitimate baseline detector from `train.csv`: for each class, learn a typical (median) box and a class prior, then emit a small set of highest-prior classes per image with modest confidence, falling back to “No finding” when the learned “No finding” prior is high. This keeps your later probability-column smoothing and the final “No finding” override logic intact, but now provides non-trivial boxes/classes so mAP should increase substantially toward your target. I also make the code robust to the two possible sample submission column names (`PredictionString` vs `TARGET`) and ensure the submission CSV is written correctly.'
- What this solution (achieved 0.01923) has done: 'Your current score (0.01849) is far below the target (0.24518), so we need a small but legitimate boost without changing your core post-processing loops. The main issue is that the fallback detector emits the same few global “top classes” and a single median box per class for every image, which produces many false positives and hurts mAP. I keep the same train.csv-driven heuristic, but make it (1) class-balanced by choosing classes based on both priors and box “tightness”, and (2) more conservative by lowering default confidences and emitting fewer classes (TOPK=2), while also adding a small fraction of “No finding” predictions to reduce false positives. These are minimal edits confined to the fallback block; your later 0.92-smoothing and the final “14” override are preserved exactly.'
- What this solution (achieved 0.02814) has done: 'Your current score (0.01923) is far below the target (0.24518), so we should legitimately increase mAP while keeping your existing “train.csv-derived fallback + post-smoothing + no-finding override” core logic intact. The biggest issue is that the fallback predicts the same few classes for every image, producing overwhelming false positives; mAP improves most by being more selective. I keep the same median-box-per-class idea but add a very small, data-driven “detector gate”: only emit a class if its learned probability for that image exceeds a threshold, otherwise output “No finding”. To make that per-image probability meaningful without using pixels/models, I use only train metadata: learn per-class radiologist co-occurrence (conditional priors) and use the hash-based image group as a stable “pseudo-context” to vary predictions across images in a legitimate deterministic way; then cap to TOPK and keep your later 0.92 smoothing and final class-14 override unchanged.'
- What this solution (achieved 0.02949) has done: 'You’re far below the target (0.028 vs 0.245, higher-is-better), so we need a legitimate boost while keeping your existing “train.csv-derived fallback → 0.92 smoothing → class-14 override” structure intact. The biggest issue is that you output only a single “median box” per class, which is rarely aligned to the true boxes and caps IoU/mAP; we can improve box quality without changing your overall approach by using a small *mixture* of typical boxes per class learned from train (e.g., k-median-like quantiles) and deterministically selecting one per image/class. To reduce false positives (which heavily hurts mAP), we also make the “No finding” rate data-driven (based on the observed fraction of “No finding” in train) instead of a fixed 40%, and slightly tune emission threshold/TOPK conservatively. These are contained changes inside the fallback block; the later smoothing loop and final no-finding override remain the same evaluation semantics.'
- What this solution (achieved 0.03664) has done: 'Your current score (0.02949) is far below the target (0.24518, higher-is-better), so we need a legitimate boost while keeping your existing “train.csv-driven fallback → 0.92 smoothing → class-14 override” core flow intact. The main limiter is poor localization: using global quantile boxes per class often yields IoU < 0.4, so mAP stays low; we can improve expected IoU without changing the approach by learning a small set of *clustered* typical boxes per class (k-means on box centers/sizes) and deterministically selecting among them per image/class. To reduce false positives (which heavily depress mAP), we also make emission slightly more conservative by using TOPK=1 and a small confidence cap, while keeping your smoothing and no-finding override semantics unchanged. These changes are confined to the fallback block that runs when the external ensemble CSV is missing, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.03055) has done: 'Your current score (0.03664) is far below the target (0.24518), so we need a legitimate boost while keeping your existing “train.csv-driven fallback → 0.92 smoothing → class-14 override” flow intact. The main bottleneck is recall: with `TOPK=1` and a fairly high `EMIT_THR`, you emit too few boxes, so many true findings get no prediction and mAP stays very low. I make the fallback slightly less conservative by emitting up to 2 classes per image and lowering the emit threshold a bit, while simultaneously reducing false positives by (a) lowering the overall “No finding” rate slightly and (b) applying a small per-class confidence shrink so extra boxes don’t dominate. These are contained changes only inside the fallback block (the external-ensemble path, smoothing loop, and final submission schema remain unchanged).'
- What this solution (achieved 0.03733) has done: 'We’re far below the target (0.03055 vs 0.24518, higher-is-better), so the smallest meaningful move is to reduce obvious false positives while slightly improving localization without changing your overall fallback→smoothing→no-finding-override flow. I keep your exact structure, but make the fallback boxes more IoU-friendly by (a) using class-wise *mean* box (often closer to typical size/position than median for this dataset) and (b) clipping boxes to a realistic CXR canvas estimated from train boxes. To reduce FP (which heavily hurts mAP), I also make predictions a bit more conservative: emit at most 1 class per image, raise the emit threshold slightly, and lower the confidence cap a bit; this usually improves mAP at IoU>0.4 for heuristic detectors. All changes are confined to the fallback block; your later 0.92-based smoothing and the final class-14 override remain identical.'
- What this solution (achieved 0.03171) has done: 'Your current score (0.03733) is far below the target (0.24518), so we need a small, legitimate mAP lift without changing your overall “train.csv-driven fallback → smoothing loop → no-finding override” flow. The biggest lever inside your existing fallback is recall at IoU>0.4: emitting only 1 class per image with a relatively high threshold misses many true findings, so I minimally loosen this by allowing up to 2 classes per image and slightly lowering `EMIT_THR` while keeping the same proto-box selection/localization logic. To avoid blowing up false positives (which would negate gains), I keep your confidence cap conservative and lightly increase the deterministic “no finding” rate a touch so additional boxes don’t appear on too many images. All changes are confined to the fallback constants; the smoothing and final output formatting remain identical and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.03432) has done: 'Your current score (0.03171) is far below the target (0.24518), so we should move upward with the smallest safe changes inside your existing fallback-only block. The main bottleneck is excessive false positives and weak localization; to improve mAP@0.4 without changing the overall flow, I (1) make the “no finding” decision data-driven per image using the already-computed per-image class scores (instead of a fixed hash-rate), and (2) slightly tune the emission threshold/confidence cap to favor precision while keeping TOPK=2 for recall. I also ensure the probability columns reflect the chosen detections (so your later 0.92-smoothing can actually affect confidences) without altering your smoothing logic itself. All changes are confined to the fallback path (when the external ensemble CSV is missing) and still produce a valid `submission.csv`.'
- What this solution (achieved 0.02009) has done: 'To move your score upward toward 0.245 with minimal disruption, I’m keeping your existing fallback structure (train.csv-driven priors → proto boxes → PredictionString → 0.92 smoothing → class-14 override) but making the fallback emit slightly more informative detections. Concretely: (1) make the per-image class scores less “flat” by using the learned radiologist prior a bit more (still deterministic, still metadata-only), (2) slightly increase recall by allowing up to 3 predictions when there is strong evidence (dynamic TOPK), and (3) tune the emission threshold and confidence cap modestly to avoid a large false-positive explosion. These are small constant-weight changes only inside the fallback block; file paths, later smoothing, and the final submission schema remain unchanged.'
- What this solution (achieved 0.02362) has done: 'Your current score (0.02009) is far below the target (0.24518), so we should increase mAP by reducing obvious false positives and making the fallback predictions less “one-size-fits-all” while keeping your overall fallback→smoothing→no-finding-override flow unchanged. The smallest high-impact fix is to make the “No finding” decision more realistic using only train metadata: learn per-image “any finding” frequency from the number of labeled boxes in train, then deterministically assign test images to those frequencies via hashing, so many images emit no boxes (precision gain) while some emit multiple (recall gain). I also tune only fallback constants (EMIT_THR/CONF_CAP) and ensure that when we emit detections we explicitly keep class-14 probability low enough so your later override doesn’t wipe them out. All paths, post-smoothing (0.92 logic), and the final submission schema remain the same and it still writes `submission.csv`.'
- What this solution (achieved 0.03094) has done: 'Your current score (0.02362) is far below the target (0.24518), so we need a small, legitimate boost without changing your overall “train.csv-driven fallback → 0.92 smoothing → class-14 override” structure. The most direct low-risk improvement inside your existing fallback is to reduce false positives (which crush mAP) while keeping recall: we slightly increase `EMIT_THR`, slightly reduce `CONF_CAP`, and make the “how many boxes to emit” decision depend not just on the hashed desired_k but also on the strength gap between the top scores (emit 2 only when there’s clear evidence). This preserves all core semantics (still metadata-only fallback, same proto-box selection, same smoothing loop, same final override), but should improve precision enough to move the score upward. The submission format and paths remain unchanged and it still writes `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

DATA_DIR = "/kaggle/data"
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")

df = pd.read_csv(SAMPLE_SUB_PATH)
if "PredictionString" not in df.columns:
    if "TARGET" in df.columns:
        df = df.rename(columns={"TARGET": "PredictionString"})
    else:
        raise ValueError(
            f"sample submission must contain PredictionString/TARGET. Found columns={df.columns.tolist()}"
        )

list1 = [str(i) for i in range(15)]
for c in list1:
    df[c] = 0.0

df.loc[df["PredictionString"].astype(str).str.strip().eq("14 1 0 0 1 1"), "14"] = 1.0

df.head()



## === cell 1
df



## === cell 2
df3_path = "../input/0284-norm/cascade_rcnn_x101_32x4d_fpn_20e_OHEM_fpn.4_with_raw.4_ensemble.5.bbox.json.filter.norm (1).csv"
if os.path.exists(df3_path):
    df3 = pd.read_csv(df3_path)
else:
    df3 = None

df3



## === cell 3
if df3 is not None and "image_id" in df3.columns:
    df4 = pd.merge(df, df3, on="image_id", how="left", suffixes=("", "_y"))
    for c in list1:
        cy = f"{c}_y"
        if cy in df4.columns:
            df4[c] = df4[c].fillna(df4[cy])
            df4 = df4.drop(columns=[cy])
    if "PredictionString_y" in df4.columns:
        df4["PredictionString"] = df4["PredictionString"].fillna(
            df4["PredictionString_y"]
        )
        df4 = df4.drop(columns=["PredictionString_y"])
else:
    train_df = pd.read_csv(TRAIN_CSV_PATH)

    for col in ["x_min", "y_min", "x_max", "y_max", "class_id"]:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
    train_df = train_df.dropna(subset=["class_id", "x_min", "y_min", "x_max", "y_max"])
    train_df["class_id"] = train_df["class_id"].astype(int)

    train_boxes = train_df[train_df["class_id"].between(0, 13)].copy()

    class_counts = train_boxes["class_id"].value_counts().to_dict()
    total_objects = float(sum(class_counts.values())) if class_counts else 1.0
    class_prior = {k: (v / total_objects) for k, v in class_counts.items()}

    def _kmeans_np(X, k=3, iters=12, seed=0):
        X = np.asarray(X, dtype=np.float32)
        n = X.shape[0]
        if n == 0:
            return np.zeros((k, X.shape[1]), dtype=np.float32)
        k_eff = int(min(k, n))
        rs = np.random.RandomState(seed)
        init_idx = rs.choice(n, size=k_eff, replace=False)
        C = X[init_idx].copy()
        for _ in range(int(iters)):
            d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
            lab = d2.argmin(axis=1)
            newC = []
            for j in range(k_eff):
                pts = X[lab == j]
                if pts.shape[0] == 0:
                    newC.append(C[j])
                else:
                    newC.append(pts.mean(axis=0))
            C = np.stack(newC, axis=0)
        if k_eff < k:
            pad = np.repeat(C[-1][None, :], k - k_eff, axis=0)
            C = np.concatenate([C, pad], axis=0)
        return C

    proto_boxes = {}  # cid -> list of boxes dicts

    mean_boxes_df = train_boxes.groupby("class_id")[
        ["x_min", "y_min", "x_max", "y_max"]
    ].mean()
    mean_boxes = mean_boxes_df.to_dict(orient="index")

    x_canvas_max = float(
        np.nanquantile(train_boxes["x_max"].values.astype(np.float32), 0.995)
    )
    y_canvas_max = float(
        np.nanquantile(train_boxes["y_max"].values.astype(np.float32), 0.995)
    )
    x_canvas_max = float(np.clip(x_canvas_max, 512.0, 4096.0))
    y_canvas_max = float(np.clip(y_canvas_max, 512.0, 4096.0))

    for cid, g in train_boxes.groupby("class_id"):
        x1 = g["x_min"].values.astype(np.float32)
        y1 = g["y_min"].values.astype(np.float32)
        x2 = g["x_max"].values.astype(np.float32)
        y2 = g["y_max"].values.astype(np.float32)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w = np.maximum(1.0, np.abs(x2 - x1))
        h = np.maximum(1.0, np.abs(y2 - y1))
        X = np.stack([cx, cy, w, h], axis=1)

        n = X.shape[0]
        k = 4 if n >= 1200 else (3 if n >= 300 else 2)
        C = _kmeans_np(X, k=k, iters=12, seed=int(cid) + 13)

        boxes = []
        for j in range(C.shape[0]):
            cxj, cyj, wj, hj = [float(v) for v in C[j]]
            x1j = cxj - 0.5 * wj
            y1j = cyj - 0.5 * hj
            x2j = cxj + 0.5 * wj
            y2j = cyj + 0.5 * hj
            boxes.append(
                {
                    "x_min": x1j,
                    "y_min": y1j,
                    "x_max": x2j,
                    "y_max": y2j,
                }
            )
        proto_boxes[int(cid)] = boxes

    if "rad_id" in train_boxes.columns:
        rc = (
            train_boxes.groupby(["rad_id", "class_id"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=list(range(14)), fill_value=0)
        )
        rad_tot = rc.sum(axis=1).astype(float) + 14.0  # +alpha*K with alpha=1
        rad_prior = (rc + 1.0).div(rad_tot, axis=0)
        rad_ids = rad_prior.index.astype(str).tolist()
    else:
        rad_prior = None
        rad_ids = []

    areas = {}
    for cid, box in mean_boxes.items():
        w0 = max(
            1.0,
            float(max(box["x_min"], box["x_max"]) - min(box["x_min"], box["x_max"])),
        )
        h0 = max(
            1.0,
            float(max(box["y_min"], box["y_max"]) - min(box["y_min"], box["y_max"])),
        )
        areas[int(cid)] = w0 * h0
    mean_area = float(np.median(list(areas.values()))) if areas else 1.0

    def class_score(c):
        p = float(class_prior.get(c, 0.0))
        a = float(areas.get(c, mean_area))
        tight = float(np.clip(mean_area / a, 0.5, 2.0))
        return p * tight

    cls_0_13 = [c for c in range(14) if (c in class_prior) and (c in mean_boxes)]
    cls_0_13_sorted = sorted(cls_0_13, key=class_score, reverse=True)

    BASE_TOPK = 2
    candidate_classes = cls_0_13_sorted[: max(BASE_TOPK, 12)]

    fallback_box = {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0}

    df4 = df.copy()

    for c in range(14):
        p = float(class_prior.get(c, 0.0))
        df4[str(c)] = float(np.clip(0.010 + 0.24 * p, 0.005, 0.14))

    nf_frac = 0.0
    try:
        nf_frac = float((train_df["class_id"] == 14).mean())
    except Exception:
        nf_frac = 0.0

    df4["14"] = float(np.clip(0.28 + 0.80 * nf_frac, 0.30, 0.75))

    image_ids = df4["image_id"].astype(str).values
    h = pd.util.hash_pandas_object(pd.Series(image_ids), index=False).values.astype(
        np.uint64
    )

    if rad_prior is not None and len(rad_ids) > 0:
        rad_pick = (h % len(rad_ids)).astype(int)
        picked_rad_ids = np.array(rad_ids, dtype=object)[rad_pick]
    else:
        picked_rad_ids = None

    img_obj_counts = (
        train_boxes.groupby("image_id").size().astype(int)
        if len(train_boxes) > 0
        else pd.Series(dtype=int)
    )
    if len(img_obj_counts) > 0:
        img_any = train_df.groupby("image_id")["class_id"].apply(
            lambda s: int(
                np.any((s.values.astype(int) >= 0) & (s.values.astype(int) <= 13))
            )
        )
        img_k = (
            train_df.groupby("image_id")["class_id"]
            .apply(
                lambda s: int(
                    np.sum((s.values.astype(int) >= 0) & (s.values.astype(int) <= 13))
                )
            )
            .clip(upper=3)
        )
        k_counts = img_k.value_counts().to_dict()
        for kk in [0, 1, 2, 3]:
            k_counts.setdefault(kk, 0)
        total_imgs = (
            float(sum(k_counts.values())) if sum(k_counts.values()) > 0 else 1.0
        )
        p_k = {kk: k_counts[kk] / total_imgs for kk in [0, 1, 2, 3]}
    else:
        p_k = {0: 0.5, 1: 0.3, 2: 0.15, 3: 0.05}

    EMIT_THR = 0.115
    CONF_CAP = 0.20

    pred_strings = []
    for idx in range(df4.shape[0]):
        scores = {}
        for cls in candidate_classes:
            base = float(df4.loc[idx, str(cls)])

            if rad_prior is not None and picked_rad_ids is not None:
                rid = picked_rad_ids[idx]
                try:
                    radp = float(rad_prior.loc[rid, cls])
                except Exception:
                    radp = float(class_prior.get(cls, 0.0))
                s = 0.45 * base + 0.55 * float(np.clip(radp, 0.0, 1.0))
            else:
                s = base

            s = 0.90 * s
            scores[cls] = float(np.clip(s, 0.02, CONF_CAP))

        if len(scores) == 0:
            pred_strings.append("14 1 0 0 1 1")
            continue

        sorted_cls = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        best_score = float(scores[sorted_cls[0]])
        second_score = float(scores[sorted_cls[1]]) if len(sorted_cls) > 1 else 0.0

        u = (h[idx] & np.uint64((1 << 53) - 1)).astype(np.float64) / float(1 << 53)
        cum0 = p_k.get(0, 0.5)
        cum1 = cum0 + p_k.get(1, 0.3)
        cum2 = cum1 + p_k.get(2, 0.15)
        if u < cum0:
            desired_k = 0
        elif u < cum1:
            desired_k = 1
        elif u < cum2:
            desired_k = 2
        else:
            desired_k = 3

        if best_score < EMIT_THR or desired_k == 0:
            pred_strings.append("14 1 0 0 1 1")
            continue

        want_k = int(min(3, max(1, desired_k)))
        if want_k >= 2:
            if (second_score < (EMIT_THR + 0.01)) or (
                (best_score - second_score) > 0.035
            ):
                TOPK = 1
            else:
                TOPK = 2
        else:
            TOPK = 1
        if want_k >= 3:
            TOPK = min(2, TOPK)  # keep conservative; avoid FP explosion

        chosen = [c for c in sorted_cls if scores[c] >= EMIT_THR][:TOPK]

        if len(chosen) == 0:
            pred_strings.append("14 1 0 0 1 1")
            continue

        for cls in chosen:
            df4.loc[idx, str(cls)] = max(
                float(df4.loc[idx, str(cls)]), float(scores[cls])
            )
        df4.loc[idx, "14"] = float(min(float(df4.loc[idx, "14"]), 0.85))

        parts = []
        for cls in chosen:
            plist = proto_boxes.get(int(cls), None)
            if plist is not None and len(plist) > 0:
                sel = int((h[idx] ^ np.uint64(cls * 7919)) % np.uint64(len(plist)))
                box = plist[sel]
            else:
                box = mean_boxes.get(cls, fallback_box)

            conf = float(scores[cls])

            xmin_f = float(min(box["x_min"], box["x_max"]))
            ymin_f = float(min(box["y_min"], box["y_max"]))
            xmax_f = float(max(box["x_min"], box["x_max"]))
            ymax_f = float(max(box["y_min"], box["y_max"]))

            xmin = int(np.clip(xmin_f, 0.0, x_canvas_max - 2.0))
            ymin = int(np.clip(ymin_f, 0.0, y_canvas_max - 2.0))
            xmax = int(np.clip(xmax_f, xmin + 1.0, x_canvas_max))
            ymax = int(np.clip(ymax_f, ymin + 1.0, y_canvas_max))

            parts.extend(
                [
                    str(int(cls)),
                    f"{conf:.6f}",
                    str(xmin),
                    str(ymin),
                    str(xmax),
                    str(ymax),
                ]
            )

        pred_strings.append(" ".join(parts))

    df4["PredictionString"] = pred_strings

df4.head()



## === cell 4
pass



## === cell 5
if df4.shape[0] > 1 and df4.shape[1] > 16:
    df4.iloc[1, 16]
else:
    df4.head(2)



## === cell 6
pass



## === cell 7
pass



## === cell 8
list1_int = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

for i in range(df4.shape[0]):
    if str(df4.loc[i, "PredictionString"]).strip() == "14 1 0 0 1 1":
        continue
    a = str(df4.loc[i, "PredictionString"])
    b = a.split()
    for j in range(int(len(a.split()) / 6)):
        for k in list1_int:
            if int(b[0 + 6 * j]) == k:
                if float(df4.loc[i, f"{k}"]) < 0.92:
                    continue
                c = b[0 + 6 * j + 1]
                b[0 + 6 * j + 1] = str(
                    float(df4.loc[i, f"{k}"]) * 0.15 + float(c) * 0.85
                )

    df4.loc[i, "PredictionString"] = " ".join(b)

df4.head()



## === cell 9
pass



## === cell 10
for i in range(df4.shape[0]):
    if float(df4.loc[i, "14"]) > 0.999:
        df4.loc[i, "PredictionString"] = "14 1 0 0 1 1"

df_final = df4[["image_id", "PredictionString"]]

df_final.to_csv("submission.csv", index=False)

df_final.head()



## === cell 11
df_final
