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

0.228277105363432

# 6. Current score

0.0196

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0238) has done: 'The current notebook fails immediately because it tries to ensemble several external `../input/.../submission.csv` files that do not exist in this environment, so no valid `submission.csv` is ever written. To make it run end-to-end with minimal changes, I replace those missing inputs with the provided `sample_submission.csv` as a safe base and add the missing per-class score columns (`'0'..'14'`) so the later calibration logic does not crash. I also fix hard-coded loops (`range(3000)`) to iterate over the actual number of test rows (1500), and I make the “filter classes” step robust to empty/NaN prediction strings. This reliably produce a valid `submission.csv` with the required columns/format; since we cannot train a detector with the available packages here, the score likely be low but the pipeline execute correctly.'
- What this solution (achieved 0.01378) has done: 'Your current pipeline always starts from `sample_submission.csv`, so it effectively predicts “No finding” for every image and then even removes several classes, which strongly caps mAP. With the available packages (no detector training), the most direct way to move the score toward your target is to create a simple, deterministic “pseudo-detector” using priors learned from `train.csv`: emit a few common classes with realistic box sizes/positions and moderate confidences for every test image instead of always “No finding”. This preserves your core flow (build `PredictionString` then apply your existing filtering/calibration steps) while producing non-trivial predictions that typically score substantially higher than all-negative. I also keep the submission schema exactly as required and ensure the code still runs end-to-end under the 600s constraint by avoiding any DICOM parsing.'
- What this solution (achieved 0.01826) has done: 'Your current pipeline builds reasonable “prior” boxes, but then it heavily degrades them by (1) filtering out classes `{0,7,13,14}` (including very common class 7), and (2) duplicating the exact same predictions via concatenation (df4 + df5) which creates duplicate boxes that are usually penalized in mAP. To move your score upward toward the 0.228 target with minimal semantic change, I keep your same prior-based prediction construction and downstream calibration, but stop duplicating predictions and stop removing common classes (only remove class 14 from non-empty strings). I also ensure every row ends with a valid non-empty `PredictionString` and keep the submission format unchanged.'
- What this solution (achieved 0.01826) has done: 'Your current code already generates non-trivial “prior” boxes, but it still suppresses confidence updates because the per-class columns rarely exceed the hard threshold (0.92), and it also computes train-derived max_x/max_y in a way that can produce oversized, mis-scaled boxes that hurt IoU. I keep the same prior-based prediction construction and the same downstream “calibration” loop, but (1) compute robust per-dimension image extents from train boxes and clamp them to common CXR sizes, and (2) raise the per-class columns just enough so the existing 0.92 gate actually triggers, increasing box confidences without changing the overall approach. These are minimal, local changes aimed at moving mAP up toward your target while keeping runtime low and preserving submission format. The script still run end-to-end and write `submission.csv` with the required columns.'
- What this solution (achieved 0.01867) has done: 'Your current score is far below the target, so we should increase mAP with the smallest change that keeps your “prior-box” approach intact. The biggest low-risk gain here is to predict more than 5 classes (still with train-derived median boxes) and to use the existing confidence “calibration” gate more effectively by setting per-class columns proportionally to class frequency (so common classes get boosted and rare ones don’t). This preserves your exact flow (build `PredictionString` → remove class 14 → apply the 0.92 gate blending) while reducing false positives from overly-uniform boosting. I also remove unused merges that don’t change predictions (they currently just duplicate the same strings), but keep the final submission schema unchanged.'
- What this solution (achieved 0.01897) has done: 'I keep your prior-box “pseudo-detector” core logic intact, but make two small, metric-relevant adjustments to move mAP upward toward the 0.228 target: (1) increase recall by predicting more classes (top_k from 10 → 14, excluding 14) so more GT classes are covered, and (2) reduce the false-positive penalty by lowering confidences for the additional tail classes while keeping higher confidences for the most frequent ones. I also remove the unused merge/duplicate dataframe step that doesn’t affect predictions but risks confusion, without changing the downstream filtering/calibration semantics. The submission format and filename stay the same, and the code remains DICOM-free and fast.'
- What this solution (achieved 0.01886) has done: 'I keep your prior-box “pseudo-detector” approach intact but make two small metric-aligned changes to raise mAP toward the 0.228 target: (1) increase per-class confidence just enough so your existing `0.92` calibration gate in cell 7 actually triggers for the top classes (right now most classes are below 0.92 after blending), and (2) add a tiny deterministic per-image jitter to box centers (derived from `image_id`) to reduce the harm of predicting identical boxes for every test image. These changes preserve your architecture/training semantics (there is none), keep the same prediction construction and calibration loop, avoid DICOM parsing, and still write a valid `submission.csv`. Runtime remains well under the limit.'
- What this solution (achieved 0.01832) has done: 'We keep your prior-box pseudo-detector and the existing calibration/filtering semantics intact, but make two small, metric-relevant changes to raise mAP toward the 0.228 target. First, we stop forcing every class probability column above the 0.92 “gate” (which currently boosts *all* predicted boxes and inflates false positives); instead we only push the gate for a small set of most frequent classes. Second, we also restrict the emitted predicted classes per image to a small top-N (by train frequency) while keeping your same box/stat construction, which typically improves precision (and thus mAP) more than it hurts recall in this setting. The output remains a valid `submission.csv` with the required columns and format, and runtime stays DICOM-free and fast.'
- What this solution (achieved 0.01908) has done: 'Your score (0.01832) is far below the target (0.2283), so we should cautiously increase it with minimal, metric-relevant edits while preserving your prior-box pseudo-detector and the existing calibration/filtering flow. The biggest low-risk issue is that you currently emit the same median box sizes/centers for every test image, which tends to miss many GT boxes (low IoU/recall); we improve IoU coverage by sampling from train-derived per-class quantiles (still “priors”, no DICOM/model). To avoid blowing up false positives, we keep the same small per-image class set, but emit 2 boxes per top class (median + a smaller quantile) with slightly reduced confidences for the second box. All changes keep the output schema identical and still write a valid `submission.csv`.'
- What this solution (achieved 0.01939) has done: 'We keep your prior-based “pseudo-detector” and the existing filtering/calibration loops intact, but make the smallest metric-relevant adjustment to increase mAP toward the 0.228 target by improving IoU coverage without exploding false positives. Concretely, we emit one additional (third) box per emitted class using a larger quantile size (q70) and a slightly shifted center (q60), at a lower confidence, so more ground-truth boxes have a better chance to overlap at IoU>0.4. We not change your class selection (`emit_classes`), gating threshold (0.92), or the “remove class 14 from non-empty strings” semantics. The code still avoids DICOM parsing, runs fast, and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.0196) has done: 'We keep your prior-box “pseudo-detector” and all downstream filtering/calibration logic intact, but make a small, metric-relevant adjustment to improve IoU coverage without exploding false positives. Concretely, we (1) choose `emit_classes` based on class frequency *and* class box-size diversity (so we don’t over-focus on classes with very similar median boxes), and (2) slightly reduce the per-image jitter scale so boxes don’t drift too far from learned priors (helps IoU>0.4), while keeping the same 3-box-per-class structure and confidence scheme. These are localized edits in the prior construction only; submission format, columns, gates (0.92), and the “remove class 14 from non-empty strings” semantics stay unchanged. The script still run fast (no DICOM parsing) and write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

DATA_DIR_CANDIDATES = [
    "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection",
    "/kaggle/input",
    "/kaggle/data/input/vinbigdata-chest-xray-abnormalities-detection",
    "/kaggle/data/input",
]


def _find_file(filename: str):
    for d in DATA_DIR_CANDIDATES:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    for p in [
        f"../input/vinbigdata-chest-xray-abnormalities-detection/{filename}",
        f"../input/{filename}",
        f"/kaggle/data/input/vinbigdata-chest-xray-abnormalities-detection/{filename}",
        f"/kaggle/data/input/{filename}",
    ]:
        if os.path.exists(p):
            return p
    return None


sample_path = _find_file("sample_submission.csv")
train_path = _find_file("train.csv")
if sample_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in the provided input paths."
    )
if train_path is None:
    raise FileNotFoundError("Could not locate train.csv in the provided input paths.")

sample = pd.read_csv(sample_path)
train = pd.read_csv(train_path)

if "image_id" not in sample.columns or "PredictionString" not in sample.columns:
    raise ValueError(f"Unexpected sample_submission columns: {list(sample.columns)}")

train = train.copy()
train = train[train["class_id"].notna()].copy()
train["class_id"] = train["class_id"].astype(int)

train_findings = train[train["class_id"] != 14].copy()

train_findings["w"] = (train_findings["x_max"] - train_findings["x_min"]).clip(
    lower=1.0
)
train_findings["h"] = (train_findings["y_max"] - train_findings["y_min"]).clip(
    lower=1.0
)
train_findings["cx"] = (train_findings["x_min"] + train_findings["x_max"]) / 2.0
train_findings["cy"] = (train_findings["y_min"] + train_findings["y_max"]) / 2.0


def _robust_extent(arr, fallback):
    arr = np.asarray(arr, dtype="float64")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(fallback)
    q = float(np.quantile(arr, 0.995))
    if not np.isfinite(q) or q <= 1:
        return float(fallback)
    return q


max_x = _robust_extent(
    pd.concat([train["x_max"], train["x_min"]], axis=0).to_numpy(), 1024.0
)
max_y = _robust_extent(
    pd.concat([train["y_max"], train["y_min"]], axis=0).to_numpy(), 1024.0
)

max_x = float(np.clip(max_x, 512.0, 4096.0))
max_y = float(np.clip(max_y, 512.0, 4096.0))

class_counts = train_findings["class_id"].value_counts().sort_values(ascending=False)

top_k = 14
top_classes = [int(c) for c in class_counts.head(top_k).index.tolist() if int(c) != 14]

grp = train_findings.groupby("class_id")
class_stats = {}
for cid in top_classes:
    g = grp.get_group(cid)

    class_stats[cid] = {
        "w_med": float(g["w"].median()),
        "h_med": float(g["h"].median()),
        "cx_med": float(g["cx"].median()),
        "cy_med": float(g["cy"].median()),
        "w_q30": float(g["w"].quantile(0.30)),
        "h_q30": float(g["h"].quantile(0.30)),
        "cx_q40": float(g["cx"].quantile(0.40)),
        "cy_q40": float(g["cy"].quantile(0.40)),
        "w_q70": float(g["w"].quantile(0.70)),
        "h_q70": float(g["h"].quantile(0.70)),
        "cx_q60": float(g["cx"].quantile(0.60)),
        "cy_q60": float(g["cy"].quantile(0.60)),
    }

freq = class_counts / class_counts.sum()

base_conf = {}
for rank, cid in enumerate(top_classes):
    f = float(freq.get(cid, 0.0))
    if rank < 6:
        base_conf[cid] = float(np.clip(0.20 + 1.10 * f, 0.18, 0.42))
    else:
        base_conf[cid] = float(np.clip(0.11 + 0.55 * f, 0.08, 0.22))


def _make_box(cx, cy, w, h, max_x, max_y):
    x1 = float(np.clip(cx - w / 2.0, 0.0, max_x - 1.0))
    y1 = float(np.clip(cy - h / 2.0, 0.0, max_y - 1.0))
    x2 = float(np.clip(cx + w / 2.0, x1 + 1.0, max_x))
    y2 = float(np.clip(cy + h / 2.0, y1 + 1.0, max_y))
    return x1, y1, x2, y2


def _id_jitter(image_id: str, scale: float = 0.035):
    s = str(image_id)
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    u1 = ((h & 0xFFFF) / 65535.0) * 2.0 - 1.0
    u2 = (((h >> 16) & 0xFFFF) / 65535.0) * 2.0 - 1.0
    return float(u1 * scale), float(u2 * scale)


def _select_emit_classes(top_classes, class_stats, class_counts, n_emit=6):
    selected = []
    for cid in top_classes:
        if cid not in class_stats:
            continue
        if len(selected) == 0:
            selected.append(cid)
            if len(selected) >= n_emit:
                break
            continue

        area_c = class_stats[cid]["w_med"] * class_stats[cid]["h_med"]
        ar_c = class_stats[cid]["w_med"] / max(1.0, class_stats[cid]["h_med"])

        mind = 1e18
        for sid in selected:
            area_s = class_stats[sid]["w_med"] * class_stats[sid]["h_med"]
            ar_s = class_stats[sid]["w_med"] / max(1.0, class_stats[sid]["h_med"])
            d = (np.log1p(area_c) - np.log1p(area_s)) ** 2 + (
                np.log(ar_c) - np.log(ar_s)
            ) ** 2
            if d < mind:
                mind = d

        is_frequent = int(class_counts.get(cid, 0)) >= int(class_counts.max() * 0.12)
        if mind >= 0.12 or is_frequent:
            selected.append(cid)
            if len(selected) >= n_emit:
                break

    if len(selected) < n_emit:
        for cid in top_classes:
            if cid not in selected:
                selected.append(cid)
            if len(selected) >= n_emit:
                break
    return selected[:n_emit]


top_emit = min(6, len(top_classes))
emit_classes = _select_emit_classes(
    top_classes, class_stats, class_counts, n_emit=top_emit
)


def build_prior_prediction_string_for_image(image_id: str):
    parts = []
    jx, jy = _id_jitter(image_id, scale=0.022)

    for cid in emit_classes:
        st = class_stats[cid]
        conf = base_conf[cid]

        cx1 = st["cx_med"] + jx * st["w_med"]
        cy1 = st["cy_med"] + jy * st["h_med"]
        x1, y1, x2, y2 = _make_box(cx1, cy1, st["w_med"], st["h_med"], max_x, max_y)
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

        conf2 = float(np.clip(conf * 0.78, 0.06, 0.35))
        cx2 = st["cx_q40"] - jx * st["w_q30"]
        cy2 = st["cy_q40"] - jy * st["h_q30"]
        x1, y1, x2, y2 = _make_box(cx2, cy2, st["w_q30"], st["h_q30"], max_x, max_y)
        parts.extend(
            [
                str(int(cid)),
                f"{conf2:.5f}",
                f"{x1:.1f}",
                f"{y1:.1f}",
                f"{x2:.1f}",
                f"{y2:.1f}",
            ]
        )

        conf3 = float(np.clip(conf * 0.62, 0.05, 0.28))
        cx3 = st["cx_q60"] + (jx * 0.65) * st["w_q70"]
        cy3 = st["cy_q60"] + (jy * 0.65) * st["h_q70"]
        x1, y1, x2, y2 = _make_box(cx3, cy3, st["w_q70"], st["h_q70"], max_x, max_y)
        parts.extend(
            [
                str(int(cid)),
                f"{conf3:.5f}",
                f"{x1:.1f}",
                f"{y1:.1f}",
                f"{x2:.1f}",
                f"{y2:.1f}",
            ]
        )

    if len(parts) == 0:
        return "14 1 0 0 1 1"
    return " ".join(parts)


df = sample.copy()
df["PredictionString"] = [
    build_prior_prediction_string_for_image(iid) for iid in df["image_id"].tolist()
]

for k in range(15):
    col = str(k)
    if col not in df.columns:
        df[col] = 0.0

gate_classes = emit_classes[: min(3, len(emit_classes))]  # only the most frequent few
for cid in top_classes:
    f = float(freq.get(cid, 0.0))
    if cid in gate_classes:
        df[str(cid)] = float(np.clip(0.945 + 1.2 * f, 0.92, 0.995))
    else:
        df[str(cid)] = float(np.clip(0.55 + 0.6 * f, 0.20, 0.89))

df["14"] = 0.0

df1 = df.copy()
df_densenet = df.copy()
cols = [str(i) for i in range(15)]
df[cols] = df[cols] * 0.25 + df1[cols] * 0.5 + df_densenet[cols] * 0.25



## === cell 1
df2 = df[["image_id", "PredictionString"]].copy()
df3 = df[["image_id", "PredictionString"]].copy()



## === cell 2
df4 = df.copy()
df4["PredictionString"] = df4["PredictionString"].fillna("14 1 0 0 1 1").astype(str)



## === cell 3
n_rows = df4.shape[0]

for i in range(n_rows):
    ps = df4.at[i, "PredictionString"]
    if not isinstance(ps, str) or ps.strip() == "":
        df4.at[i, "PredictionString"] = "14 1 0 0 1 1"
        continue

    list2 = ps.split()
    if len(list2) < 6:
        df4.at[i, "PredictionString"] = "14 1 0 0 1 1"
        continue

    h_parts = []
    n_boxes = len(list2) // 6
    for j in range(n_boxes):
        cls = list2[0 + 6 * j]
        if cls == "14":
            continue
        h_parts.extend(list2[6 * j : 6 * j + 6])

    if len(h_parts) == 0:
        df4.at[i, "PredictionString"] = "14 1 0 0 1 1"
    else:
        df4.at[i, "PredictionString"] = " ".join(h_parts)

for i in range(n_rows):
    df4.at[i, "PredictionString"] = " ".join(str(df4.at[i, "PredictionString"]).split())



## === cell 4
_ = df4.iloc[1, df4.columns.get_loc("PredictionString")] if df4.shape[0] > 1 else None



## === cell 5
df4["PredictionString"] = df4["PredictionString"].fillna("").astype(str).str.strip()
df4["PredictionString"] = df4["PredictionString"].replace("", "14 1 0 0 1 1")



## === cell 6
list1 = list(range(15))

for i in range(df4.shape[0]):
    if df4.loc[i, "PredictionString"] == "14 1 0 0 1 1":
        continue
    a = df4.loc[i, "PredictionString"]
    b = a.split()
    if len(b) < 6:
        df4.loc[i, "PredictionString"] = "14 1 0 0 1 1"
        continue

    n_boxes = len(b) // 6
    for j in range(n_boxes):
        try:
            cls_id = int(float(b[0 + 6 * j]))
        except Exception:
            continue

        if cls_id in list1:
            col = str(cls_id)
            if col not in df4.columns:
                continue
            if float(df4.loc[i, col]) < 0.92:
                continue
            try:
                c = float(b[0 + 6 * j + 1])
                b[0 + 6 * j + 1] = str(float(df4.loc[i, col]) * 0.4 + c * 0.6)
            except Exception:
                continue

    df4.loc[i, "PredictionString"] = " ".join(b)



## === cell 7
if "14" not in df4.columns:
    df4["14"] = 0.0

for i in range(df4.shape[0]):
    if float(df4.loc[i, "14"]) > 0.999:
        df4.loc[i, "PredictionString"] = "14 1 0 0 1 1"

df_final = df4[["image_id", "PredictionString"]].copy()
df_final.to_csv("submission.csv", index=False)



## === cell 8
df_final
