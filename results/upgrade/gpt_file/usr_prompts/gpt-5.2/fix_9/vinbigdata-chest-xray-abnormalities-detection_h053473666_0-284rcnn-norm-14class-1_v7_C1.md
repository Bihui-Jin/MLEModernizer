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

0.2418996863632449

# 6. Current score

0.0475

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0475) has done: 'Your notebook fails because it depends on external Kaggle datasets (`vin-15-cnn-predict`, `0284-norm`, etc.) that are not present in this environment, so `pd.read_csv()` raises `FileNotFoundError` and all downstream variables are undefined. To make it run end-to-end and still produce a valid submission, I add a small fallback that detects missing input files and instead outputs a submission built from the provided `sample_submission.csv` (which is always valid format-wise). This keeps the “core logic” intact when the ensemble files exist, but guarantees a `.csv` submission is written even when they don’t. The resulting score won’t be competitive without the missing model outputs, but you get a valid submission file to submit.'
- What this solution (achieved 0.0475) has done: 'Your current pipeline mostly outputs “No finding” for every image because the external ensemble CSVs are missing, so the score is capped very low. To move the score toward your target with minimal change and without altering the overall “multiply confidences by per-class weights then force class-14 if high” logic, I derive those per-class multipliers from the provided `train.csv` class frequencies (a lightweight prior) instead of defaulting them to all-zeros/only-14. This keeps the same post-processing semantics (class-wise confidence scaling) but makes predictions non-degenerate and typically improves mAP versus always predicting class 14. I also make the loops robust to missing/NaN `PredictionString` and preserve the existing submission format (`image_id`, `PredictionString`) so it always writes `submission.csv`.'
- What this solution (achieved 0.01815) has done: 'Your score is far below the target, so we should (minimally) make the fallback predictions less degenerate while preserving your exact “multiply confidences by per-class weights, then optionally force class-14” semantics. The biggest issue is that when the external ensemble files are missing, you currently never generate any non–“No finding” boxes, so mAP stays near-zero; we instead build a lightweight, valid set of heuristic boxes per image directly from `train.csv` priors (most common classes + typical box size/center), then apply your existing multiplier logic. We also fix a format mismatch risk by always outputting the required submission columns (`image_id`, `PredictionString`) even if the sample uses different names. Finally, we keep your class-14 forcing step but make it non-triggering in the fallback path (so it doesn’t wipe the added boxes), without changing the step itself.'
- What this solution (achieved 0.03983) has done: 'Your current score is far below the target, and the main limiting factor is that the fallback path emits the same 3 “typical” boxes for every test image, which produces many false positives and tanks mAP. To move toward the target while keeping your core “class-wise confidence multiplier” and “optional class-14 forcing” semantics unchanged, I (1) estimate a per-image probability of “No finding” from `train.csv` and use it to decide whether to emit findings or emit only class 14, and (2) when emitting findings, use a small set of per-class *top-K* classes (not always the same 3) with conservative confidences so we reduce false positives. I also clip confidences to `[0, 1]` after multiplication to avoid invalid/overconfident scores, which is directly relevant to mAP stability. All file paths and the final `submission.csv` schema stay the same.'
- What this solution (achieved 0.04143) has done: 'Your current score (0.03983) is far below the target (0.2419), so we should increase mAP with the smallest changes that don’t alter your pipeline’s core semantics (per-class multiplier scaling + optional class-14 forcing). The biggest controllable lever in your fallback path is reducing false positives: right now you emit findings for too many images and with confidences that are too high for a heuristic generator. I (1) estimate “no finding” probability correctly at *image level* from `train.csv` (not object-row level), then (2) use a more conservative fallback that only emits findings for a smaller fraction of images and with lower confidences, while keeping the same multiplier application and submission format. This should move the score upward toward the target by improving precision without changing the overall structure of your solution.'
- What this solution (achieved 0.04334) has done: 'Your current gap to target is large (0.04143 vs 0.2419), and in this fallback-only setting the main lever is reducing false positives while keeping your existing “per-class multiplier scaling” and “optional class-14 forcing” steps unchanged. I keep the ensemble-loading logic intact, but improve the fallback generator so it emits findings only for a smaller, more realistic fraction of images and uses a tighter top-class pool and lower confidences (better precision tends to help mAP vs spraying boxes). I also fix one subtle bug in how `p_no_finding` is estimated (it currently treats “no boxes in train.csv for an image” incorrectly), which affects the fallback’s no-finding decision; this is a minimal, directly relevant correctness fix. Finally, I ensure the submission column name is exactly `PredictionString` and always written to `submission.csv` as you already do.'
- What this solution (achieved 0.0475) has done: 'Your current score (0.04334) is far below the target (0.2419), so we should cautiously increase mAP with the smallest change that reduces false positives while keeping your existing “(fallback) generate boxes → multiply confidences by per-class weights → optionally force class-14” semantics intact. The biggest issue in the fallback path is that it emits a positive box for too many images; on this competition that usually hurts mAP more than it helps. I adjust only the fallback’s `is_no_finding_image()` threshold to be more conservative (more images predicted as “No finding”), and I keep the rest of the pipeline (box priors, multipliers, confidence scaling, class-14 forcing, output format) unchanged. This should move the score upward toward your target by improving precision without changing the core logic or any training/modeling.'
- What this solution (achieved 0.0475) has done: 'We keep your existing fallback structure (heuristic boxes → per-class confidence multipliers → optional class-14 forcing) but reduce the main mAP killer: too many false positives on images that are actually “No finding.” Concretely, we (1) estimate `p_no_finding` correctly at image-level from `train.csv` (including images absent from `train.csv`, which are implicitly “No finding”), and (2) make the fallback “no finding” decision slightly more conservative using that improved estimate, without changing your box generator or multiplier application. This should move the score upward toward your target by improving precision while preserving your pipeline’s semantics and always writing a valid `submission.csv`. No external datasets are required; if the ensemble files exist, your original ensemble path remains unchanged.'

# 9. Code solution

## === cell 0
import os
import glob
import pandas as pd

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection",
    "/kaggle/data/vinbigdata-chest-xray-abnormalities-detection",
    "/kaggle/input",
    "/kaggle/data/input",
    "/kaggle/data",
]


def find_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


DATA_ROOT = find_first_existing(DATA_ROOT_CANDIDATES)


def read_csv_if_exists(path):
    if path is None:
        return None
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


sample_sub_path = None
if DATA_ROOT is not None:
    cand1 = os.path.join(DATA_ROOT, "sample_submission.csv")
    sample_sub_path = cand1 if os.path.exists(cand1) else None

if sample_sub_path is None:
    for root in ["/kaggle/input", "/kaggle/data", "/kaggle/working"]:
        hits = glob.glob(
            os.path.join(root, "**", "sample_submission.csv"), recursive=True
        )
        if hits:
            sample_sub_path = hits[0]
            break

if sample_sub_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in the provided environment."
    )

sample_sub = pd.read_csv(sample_sub_path)

if "PredictionString" not in sample_sub.columns and "TARGET" in sample_sub.columns:
    sample_sub = sample_sub.rename(columns={"TARGET": "PredictionString"})
if "image_id" not in sample_sub.columns and "ID" in sample_sub.columns:
    sample_sub = sample_sub.rename(columns={"ID": "image_id"})

if "image_id" not in sample_sub.columns:
    raise ValueError(
        f"sample_submission must contain image_id column. Got: {sample_sub.columns.tolist()}"
    )
if "PredictionString" not in sample_sub.columns:
    sample_sub["PredictionString"] = ""

list1 = [str(i) for i in range(15)]

train_csv_path = None
if DATA_ROOT is not None:
    cand = os.path.join(DATA_ROOT, "train.csv")
    if os.path.exists(cand):
        train_csv_path = cand
if train_csv_path is None:
    hits = glob.glob(os.path.join("/kaggle", "**", "train.csv"), recursive=True)
    train_csv_path = hits[0] if hits else None

class_prior = {c: 0.0 for c in list1}
box_stats = {c: {"w": None, "h": None, "cx": None, "cy": None} for c in list1}

p_no_finding = 0.5
mean_boxes_when_finding = 1.0

if train_csv_path is not None and os.path.exists(train_csv_path):
    tr_full = pd.read_csv(
        train_csv_path,
        usecols=["image_id", "class_id", "x_min", "y_min", "x_max", "y_max"],
    )

    vc = tr_full["class_id"].value_counts()
    total = float(vc.sum()) if float(vc.sum()) > 0 else 1.0
    for k in range(15):
        class_prior[str(k)] = float(vc.get(k, 0.0)) / total

    pri = pd.Series(class_prior)
    pri_nonzero = pri[pri.index != "14"]
    mean_nonzero = float(pri_nonzero.mean()) if float(pri_nonzero.mean()) > 0 else 1.0
    multipliers = {}
    for k in range(15):
        if k == 14:
            multipliers[str(k)] = 0.85
        else:
            ratio = class_prior[str(k)] / mean_nonzero if mean_nonzero > 0 else 1.0
            ratio = max(0.0, min(2.0, ratio))
            multipliers[str(k)] = 0.75 + 0.5 * (ratio / 2.0)  # [0.75, 1.25]

    tr = tr_full.dropna(subset=["class_id", "x_min", "y_min", "x_max", "y_max"]).copy()
    tr["w"] = (tr["x_max"] - tr["x_min"]).clip(lower=1.0)
    tr["h"] = (tr["y_max"] - tr["y_min"]).clip(lower=1.0)
    tr["cx"] = (tr["x_min"] + tr["x_max"]) / 2.0
    tr["cy"] = (tr["y_min"] + tr["y_max"]) / 2.0
    for k in range(15):
        sub = tr[tr["class_id"] == k]
        if len(sub) == 0:
            continue
        box_stats[str(k)]["w"] = float(sub["w"].median())
        box_stats[str(k)]["h"] = float(sub["h"].median())
        box_stats[str(k)]["cx"] = float(sub["cx"].median())
        box_stats[str(k)]["cy"] = float(sub["cy"].median())

    train_image_ids = tr_full["image_id"].dropna().astype(str).unique().tolist()
    per_image_classids = (
        tr_full.dropna(subset=["image_id", "class_id"])
        .assign(image_id=lambda d: d["image_id"].astype(str))
        .groupby("image_id")["class_id"]
        .apply(lambda s: [int(x) for x in s.dropna().tolist()])
    )
    no_finding_in_traincsv = per_image_classids.apply(
        lambda lst: (len(lst) > 0) and all(x == 14 for x in lst)
    )
    p_nf_seen = (
        float(no_finding_in_traincsv.mean()) if len(no_finding_in_traincsv) else 0.5
    )

    all_train_ids = set(train_image_ids)
    implicit_no_finding_frac = 0.0
    try:
        train_dicom_dir = None
        if DATA_ROOT is not None:
            cand_dir = os.path.join(DATA_ROOT, "train")
            if os.path.isdir(cand_dir):
                train_dicom_dir = cand_dir
        if train_dicom_dir is None:
            for root in ["/kaggle/input", "/kaggle/data"]:
                hits = glob.glob(os.path.join(root, "**", "train"), recursive=True)
                hits = [h for h in hits if os.path.isdir(h)]
                if hits:
                    train_dicom_dir = hits[0]
                    break
        if train_dicom_dir is not None:
            n_dicom = len(glob.glob(os.path.join(train_dicom_dir, "*.dicom")))
            if n_dicom > 0:
                implicit_no_finding_frac = max(
                    0.0, min(1.0, (n_dicom - len(all_train_ids)) / float(n_dicom))
                )
    except Exception:
        implicit_no_finding_frac = 0.0

    p_no_finding = p_nf_seen
    if implicit_no_finding_frac > 0:
        p_no_finding = 0.8 * p_nf_seen + 0.2 * max(p_nf_seen, implicit_no_finding_frac)
    p_no_finding = float(max(0.05, min(0.995, p_no_finding)))

    counts_non14 = tr_full[tr_full["class_id"] != 14].groupby("image_id").size()
    mean_boxes_when_finding = float(counts_non14.mean()) if len(counts_non14) else 1.0
    mean_boxes_when_finding = max(1.0, min(3.0, mean_boxes_when_finding))
else:
    multipliers = {c: 1.0 for c in list1}

ensemble_ok = True
dfs = {}

for i in range(10):
    p = f"../input/vin-15-cnn-predict/submission{i}.csv"
    if os.path.exists(p):
        dfs[i] = pd.read_csv(p)
    else:
        ensemble_ok = False
        break

if ensemble_ok:
    for i in range(10, 15):
        p = f"../input/vin-15-cnn-predict-1/submission{i - 10}.csv"
        if os.path.exists(p):
            dfs[i] = pd.read_csv(p)
        else:
            ensemble_ok = False
            break

if ensemble_ok:
    needed_cols = ["image_id"] + list1
    for i in range(15):
        missing = [c for c in needed_cols if c not in dfs[i].columns]
        if missing:
            ensemble_ok = False
            break

if ensemble_ok:
    df0 = dfs[0]
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
else:
    df = sample_sub[["image_id", "PredictionString"]].copy()

    common_classes_all = [str(k) for k in range(14)]
    common_classes_all = sorted(
        common_classes_all, key=lambda c: class_prior.get(c, 0.0), reverse=True
    )

    def build_box_for_class(k_str):
        bs = box_stats.get(k_str, {})
        w = bs.get("w", None)
        h = bs.get("h", None)
        cx = bs.get("cx", None)
        cy = bs.get("cy", None)

        if w is None or h is None or cx is None or cy is None:
            cx, cy, w, h = 1500.0, 1500.0, 800.0, 800.0

        x1 = max(0.0, cx - w / 2.0)
        y1 = max(0.0, cy - h / 2.0)
        x2 = max(x1 + 1.0, cx + w / 2.0)
        y2 = max(y1 + 1.0, cy + h / 2.0)

        return int(x1), int(y1), int(x2), int(y2)

    def base_conf_for_class(k_str):
        p = float(class_prior.get(k_str, 0.0))
        conf = 0.01 + 0.06 * min(1.0, p / 0.12)
        return float(max(0.005, min(0.12, conf)))

    def is_no_finding_image(image_id: str) -> bool:
        thr = min(0.995, max(0.05, p_no_finding + 0.34))
        h = 0
        for ch in str(image_id):
            h = (h * 131 + ord(ch)) % 1000003
        u = (h % 10000) / 10000.0
        return u < thr

    top_k = int(round(mean_boxes_when_finding))
    top_k = max(1, min(1, top_k))  # keep exactly 1 box per positive image (unchanged)

    top_classes = common_classes_all[:6]

    pred_strings = []
    for img_id in df["image_id"].astype(str).tolist():
        if is_no_finding_image(img_id):
            pred_strings.append("14 1 0 0 1 1")
            continue

        parts = []
        h = 0
        for ch in str(img_id):
            h = (h * 137 + ord(ch)) % 1000003
        start = h % len(top_classes)
        chosen = []
        for t in range(len(top_classes)):
            chosen.append(top_classes[(start + t) % len(top_classes)])
            if len(chosen) >= top_k:
                break

        for k_str in chosen:
            x1, y1, x2, y2 = build_box_for_class(k_str)
            conf = base_conf_for_class(k_str)
            parts.extend([k_str, f"{conf:.6f}", str(x1), str(y1), str(x2), str(y2)])
        ps = " ".join(parts) if parts else "14 1 0 0 1 1"
        pred_strings.append(ps)

    df["PredictionString"] = pred_strings

    for c in list1:
        df[c] = float(multipliers.get(c, 1.0))

    df["14"] = 0.0

df


## === cell 1
df


## === cell 2
path_df3 = "../input/0284-norm/cascade_rcnn_x101_32x4d_fpn_20e_OHEM_fpn.4_with_raw.4_ensemble.5.bbox.json.filter.norm (1).csv"
if os.path.exists(path_df3):
    df3 = pd.read_csv(path_df3)
else:
    df3 = df[["image_id"]].copy()
    for c in [str(i) for i in range(15)]:
        df3[c] = 1.0  # neutral multiplier (keeps existing semantics)

df3.head()


## === cell 3
df4 = pd.merge(df, df3, on="image_id", how="left", suffixes=("", "_norm"))
df4.head()


## === cell 4
if "PredictionString" not in df4.columns:
    df4["PredictionString"] = "14 1 0 0 1 1"

for k in [str(i) for i in range(15)]:
    if k not in df4.columns and f"{k}_norm" in df4.columns:
        df4[k] = df4[f"{k}_norm"]
    elif k in df4.columns and f"{k}_norm" in df4.columns:
        pass
    elif k not in df4.columns:
        df4[k] = 1.0

df4.shape


## === cell 5
df4.iloc[1, min(16, df4.shape[1] - 1)]


## === cell 6
None


## === cell 7
None


## === cell 8
list1_int = list(range(15))

for i in range(df4.shape[0]):
    ps = df4.loc[i, "PredictionString"]
    if not isinstance(ps, str) or ps.strip() == "" or ps == "nan":
        df4.loc[i, "PredictionString"] = "14 1 0 0 1 1"
        continue
    if ps == "14 1 0 0 1 1":
        continue
    b = ps.split()
    for j in range(int(len(b) / 6)):
        for k in list1_int:
            if int(float(b[0 + 6 * j])) == k:
                c = b[0 + 6 * j + 1]
                mult = df4.loc[i, f"{k}"]
                try:
                    mult = float(mult)
                except Exception:
                    mult = 1.0
                new_c = mult * float(c)
                if new_c < 0.0:
                    new_c = 0.0
                if new_c > 1.0:
                    new_c = 1.0
                b[0 + 6 * j + 1] = str(new_c)
    df4.loc[i, "PredictionString"] = " ".join(b)

df4[["image_id", "PredictionString"]].head()


## === cell 9
None


## === cell 10
if "14" not in df4.columns:
    df4["14"] = 0.0

for i in range(df4.shape[0]):
    try:
        if float(df4.loc[i, "14"]) > 0.999:
            df4.loc[i, "PredictionString"] = "14 1 0 0 1 1"
    except Exception:
        pass

df_final = df4[["image_id", "PredictionString"]].copy()

df_final.to_csv("submission.csv", index=False)

df_final.head()


## === cell 11
df_final
