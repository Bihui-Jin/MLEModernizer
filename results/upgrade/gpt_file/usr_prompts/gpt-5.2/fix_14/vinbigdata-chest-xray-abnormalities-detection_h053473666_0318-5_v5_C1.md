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

0.2274192586362992

# 6. Current score

0.01872

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0475) has done: 'Your notebook is failing because it tries to read external Kaggle “dataset submissions” that are not present in this environment (FileNotFoundError), and later cells depend on variables that never get created. To make it run end-to-end and still preserve the intent (produce a valid `submission.csv` for VinBigData), I replace those missing inputs with a robust fallback that uses the provided `sample_submission.csv` and outputs the required “No finding” prediction for every test image. This is score-limited but valid and yield a submission file with the correct columns and `.csv` suffix. I also keep your later post-processing cells from crashing by ensuring `df4/df5/df_final` exist with the expected columns.'
- What this solution (achieved 0.0475) has done: 'Your current pipeline always outputs “No finding” for every test image, which caps mAP very low; to move toward the target score we need to produce at least some non-empty detections while keeping the rest of your logic intact. The smallest safe change is to replace the hardcoded `PredictionString` with a simple, train-derived prior: for each class, compute a typical bounding box (median over training boxes) and its frequency, then emit the top few classes with moderate confidence for every test image. This keeps your later merging/filtering cells working (same columns, same loops) while increasing recall enough to raise mAP from ~0.05 toward ~0.23 without introducing new model code or changing the core post-processing structure. I also keep “No finding” as fallback when no boxes are available for a class.'
- What this solution (achieved 0.00954) has done: 'Your score is far below the target, so we need a small, legitimate change that increases mAP without changing your downstream post-processing logic. Right now cell 4 removes several classes entirely (including common ones), and cells 1/8/9 effectively force “No finding” for every image (class 14 prob = 1.0), which destroys recall. I keep your “train-derived typical boxes” idea and all later cells/loops, but (1) stop filtering out classes in cell 4, (2) generate per-class probabilities from training frequency so cell 8 can calibrate confidences, and (3) set class-14 probability low (not 1.0) so cell 9 doesn’t overwrite predictions. This should move the score upward toward the target while keeping architecture/training semantics unchanged (there is no model training here).'
- What this solution (achieved 0.00954) has done: 'Your current score is far below the target, so we should make a small change that increases recall without changing your downstream merging/loop structure. The main issue is that cell 8 almost never adjusts confidences because your per-class probabilities are capped below the hard threshold 0.92, so your emitted boxes keep very low confidences (~0.12–0.35) and contribute little to mAP. I keep your “train-derived typical boxes + class-frequency priors + same post-processing cells” core logic, but (1) raise the generated per-class probability columns so some classes exceed 0.92, and (2) slightly raise the base emitted box confidences (still moderate) so detections are more likely to count. I also keep class 14 (“No finding”) probability low so cell 9 won’t overwrite detections.'
- What this solution (achieved 0.00954) has done: 'You’re far below the target (0.00954 vs 0.2274), so we need a small but meaningful recall/confidence lift while preserving your current “train-derived typical boxes + later string-merging/weighting” pipeline. The main bottleneck is cell 8: it only boosts confidences when per-class probabilities exceed 0.92, but your current per-class columns rarely reach that, so predictions stay too low-confidence to score. I minimally adjust the per-class probability scaling in cell 1 so that a few frequent classes exceed 0.92 (without touching any downstream loops), and I slightly raise the emitted base box confidence cap so the boosted detections are more competitive. Everything else (file paths, merging logic, PredictionString formatting, and writing `submission.csv`) remains the same.'
- What this solution (achieved 0.00954) has done: 'Your current score (0.00954) is far below the target (0.2274), so the smallest lever that can legitimately raise mAP in your existing pipeline is to ensure cell 8 actually boosts confidences for frequent classes by getting some per-class probability columns above the hard 0.92 threshold. I keep your core “train-derived median boxes + frequency prior + same downstream merges/loops” logic intact, but adjust only the probability scaling/clipping so common classes cross 0.92 and can be blended upward. I also slightly raise the base box confidence (still moderate) to make boosted detections more competitive, while keeping class 14 (“No finding”) probability safely low so cell 9 won’t overwrite detections. All paths, submission schema, and the rest of your post-processing remain unchanged, and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.00954) has done: 'Your current score is far below the target, so we should make the smallest change that materially increases mAP while preserving your existing “train-derived typical boxes + downstream string-merging/weighting” core logic. The biggest bottleneck is cell 8: it only boosts confidences when a class probability exceeds 0.92, but your per-class probabilities rarely cross that, so boxes stay low-confidence and contribute little. I minimally rescale the per-class probability columns (still derived only from training frequency) so the most frequent classes exceed 0.92, and I slightly raise the emitted base box confidences so boosted detections are more competitive. All file paths, loops, merging, formatting, and writing `submission.csv` remain the same.'
- What this solution (achieved 0.00954) has done: 'Your current score is far below the target, so we should minimally increase recall/usable confidence without changing your overall “train-derived median boxes + later merging/weighting cells” pipeline. The main bottleneck is cell 8: it only boosts boxes when a per-class probability exceeds 0.92, but your per-class columns are usually too low, so almost nothing gets boosted and confidences stay uncompetitive. I keep all downstream loops/formatting intact and only (1) rescale the per-class probability columns so the most frequent classes cross 0.92, and (2) modestly raise the base emitted box confidences (still not extreme) so mAP can rise toward the target. The submission format, paths, and all later merging/processing semantics remain the same, and it still write a valid `submission.csv`.'
- What this solution (achieved 0.01884) has done: 'We keep your “train-derived median boxes + frequency prior + same downstream merge/weighting cells” pipeline intact, but fix the main scoring bottleneck: your final boxes are duplicated (df4 concatenates df5, both identical), which artificially inflates false positives and hurts mAP. We remove that duplication by concatenating df4 with df5 only when they differ; in this notebook they don’t, so we skip the append. Then we make cell 8’s confidence blending actually help by softening the hard 0.92 gate to 0.80 (still a thresholded boost, same semantics) so more common classes get their confidence nudged upward. This is a minimal change set that should increase mAP from 0.00954 toward the target band without changing the overall approach or submission schema.'
- What this solution (achieved 0.01856) has done: 'Your current gap to the target is large (0.01884 vs 0.2274, higher-is-better), so we need a small change that increases mAP without changing your overall “train-derived median boxes + downstream confidence blending/formatting” pipeline. The main limiter now is that you emit the same 8 classes for every image, creating many false positives; we reduce that with a minimal per-image gating rule using only the already-computed class prior columns in `df` (no new model/training). Concretely, we keep your default prediction generation but lower `K` (fewer boxes per image) and in cell 2 build each image’s `PredictionString` by keeping only classes whose per-image prior exceeds a threshold (otherwise fall back to “No finding”). This should materially reduce false positives and move the score upward toward the target while preserving your existing downstream merge and confidence-adjustment logic.'
- What this solution (achieved 0.01856) has done: 'We keep your existing “train-derived median boxes + class-frequency priors + downstream gating/blending” pipeline intact, but adjust two tight bottlenecks that are suppressing score: (1) the class-confidence boosting in cell 8 is currently too weak (it often *reduces* strong box confidences), and (2) the gating in cell 2 is too strict, causing many images to fall back to “No finding”. Minimally, we lower the gate a bit so predictions aren’t empty, and change the blend weights so when a class prior is high it nudges the box confidence upward (still bounded, same semantics). This should increase recall with reasonable confidence and move mAP upward toward the target without changing your overall approach or output format. The script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.01872) has done: 'We keep your overall “train-derived median boxes + class-frequency priors + downstream gating/blending/formatting” pipeline exactly the same, but make two minimal tweaks that should raise mAP toward the target by improving recall without exploding false positives. First, we slightly relax the per-image gate in cell 2 so more images emit a few detections instead of falling back to “No finding” too often. Second, we modestly increase the number of candidate classes (K) used to build the default prediction string, but cap per-image kept classes to a small max so we don’t flood each image with boxes. These are small, local changes that keep your later cells/loops intact and still produce a valid `submission.csv`.'
- What this solution (achieved 0.01872) has done: 'We keep your pipeline and post-processing intact, but make two small, targeted changes to improve mAP toward the 0.227 target by reducing systematic false positives and making confidences more meaningful. First, we stop forcing a relatively high confidence for every top-class box in `default_pred` (currently many are ~0.93), and instead use a more moderate base confidence so only truly boosted classes get high scores after cell 8. Second, we slightly tighten the per-image gating (keep fewer classes per image) while making the gate depend on the per-class prior ranking you already compute, which typically improves precision at IoU>0.4 when boxes are “generic medians”. These changes preserve the same “train-derived median boxes + priors + gating + confidence blending” semantics and still produce a valid `submission.csv`.'

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


def find_existing_file(relpath: str):
    for root in DATA_ROOT_CANDIDATES:
        p = os.path.join(root, relpath)
        if os.path.exists(p):
            return p
    return None


sample_path = find_existing_file("sample_submission.csv")
if sample_path is None:
    for root in DATA_ROOT_CANDIDATES:
        hits = glob.glob(
            os.path.join(root, "**", "sample_submission.csv"), recursive=True
        )
        if hits:
            sample_path = hits[0]
            break
if sample_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in the provided data paths."
    )

sample = pd.read_csv(sample_path)

if "image_id" not in sample.columns or "PredictionString" not in sample.columns:
    colmap = {}
    if "ID" in sample.columns:
        colmap["ID"] = "image_id"
    if "TARGET" in sample.columns:
        colmap["TARGET"] = "PredictionString"
    sample = sample.rename(columns=colmap)
    if "image_id" not in sample.columns or "PredictionString" not in sample.columns:
        raise ValueError(
            f"Unexpected sample submission columns: {list(sample.columns)}"
        )

train_path = find_existing_file("train.csv")
if train_path is None:
    for root in DATA_ROOT_CANDIDATES:
        hits = glob.glob(os.path.join(root, "**", "train.csv"), recursive=True)
        if hits:
            train_path = hits[0]
            break
if train_path is None:
    raise FileNotFoundError("Could not locate train.csv in the provided data paths.")

train = pd.read_csv(train_path)

box_cols = ["x_min", "y_min", "x_max", "y_max"]
train_boxes = train[(train["class_id"].astype(int) != 14)].copy()
train_boxes = train_boxes.dropna(subset=box_cols + ["class_id"])
train_boxes["x1"] = train_boxes[["x_min", "x_max"]].min(axis=1)
train_boxes["x2"] = train_boxes[["x_min", "x_max"]].max(axis=1)
train_boxes["y1"] = train_boxes[["y_min", "y_max"]].min(axis=1)
train_boxes["y2"] = train_boxes[["y_min", "y_max"]].max(axis=1)
train_boxes = train_boxes[
    (train_boxes["x2"] > train_boxes["x1"]) & (train_boxes["y2"] > train_boxes["y1"])
]

grp = train_boxes.groupby("class_id")
median_boxes = grp[["x1", "y1", "x2", "y2"]].median()
freq = grp.size().sort_values(ascending=False)

K = 6
top_classes = [int(c) for c in freq.head(K).index.tolist()]

total_boxes = float(len(train_boxes)) if len(train_boxes) else 1.0
pred_parts = []
for cid in top_classes:
    if cid in median_boxes.index:
        x1, y1, x2, y2 = median_boxes.loc[cid, ["x1", "y1", "x2", "y2"]].tolist()
        p = float(freq.loc[cid] / total_boxes)
        conf = max(0.35, min(0.78, (p**0.5) * 1.10))
        pred_parts.append(f"{cid} {conf:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}")

default_pred = " ".join(pred_parts).strip()
if default_pred == "":
    default_pred = "14 1 0 0 1 1"

df = sample.copy()

p_class = {i: 0.0 for i in range(15)}
if len(freq) > 0:
    denom = float(freq.sum())
    for cid, cnt in freq.items():
        p_class[int(cid)] = float(cnt) / denom

for k in range(15):
    df[str(k)] = 0.0

for k in range(14):
    base = p_class.get(k, 0.0) ** 0.5
    df[str(k)] = min(0.999, max(0.25, base * 7.80))

df["14"] = 0.15

df["PredictionString"] = default_pred

df1 = df.copy()
df_densenet = df.copy()



## === cell 1
cls_cols = [str(i) for i in range(15)]
df[cls_cols] = df[cls_cols] * 0.25 + df1[cls_cols] * 0.5 + df_densenet[cls_cols] * 0.25



## === cell 2
df2 = sample[["image_id", "PredictionString"]].copy()
df3 = sample[["image_id", "PredictionString"]].copy()

GATE_THR = 0.76
MAX_KEEP = 2

_default_tokens = default_pred.split()
_default_by_class = {}
if len(_default_tokens) % 6 == 0:
    for j in range(len(_default_tokens) // 6):
        cid = int(float(_default_tokens[0 + 6 * j]))
        _default_by_class[cid] = _default_tokens[6 * j : 6 * j + 6]


def build_gated_pred(row):
    kept = []
    scored = []
    for cid, tokens in _default_by_class.items():
        col = str(cid)
        if col in row.index:
            scored.append((float(row[col]), cid, tokens))
    scored.sort(reverse=True, key=lambda t: t[0])

    kept_n = 0
    for prior, cid, tokens in scored:
        if prior >= GATE_THR:
            kept.extend(tokens)
            kept_n += 1
            if kept_n >= MAX_KEEP:
                break

    if len(kept) == 0:
        return "14 1 0 0 1 1"
    return " ".join(kept)


df2["PredictionString"] = df.apply(build_gated_pred, axis=1)
df3["PredictionString"] = df2["PredictionString"].values



## === cell 3
df4 = pd.merge(df, df3, on="image_id", how="left", suffixes=("", "_df3"))

if "PredictionString_df3" in df4.columns:
    df4 = df4.drop(columns=["PredictionString_df3"])

df4["PredictionString"] = df4["PredictionString"].fillna("14 1 0 0 1 1")



## === cell 4
for i in range(df4.shape[0]):
    pred = df4.loc[i, "PredictionString"]
    if not isinstance(pred, str) or pred.strip() == "":
        df4.loc[i, "PredictionString"] = "14 1 0 0 1 1"
        continue

    list2 = pred.split()
    h = ""
    if len(list2) % 6 != 0:
        continue

    for j in range(int(len(list2) / 6)):
        h = (
            h
            + " "
            + list2[0 + 6 * j]
            + " "
            + list2[1 + 6 * j]
            + " "
            + list2[2 + 6 * j]
            + " "
            + list2[3 + 6 * j]
            + " "
            + list2[4 + 6 * j]
            + " "
            + list2[5 + 6 * j]
        )
    df4.loc[i, "PredictionString"] = h.strip() if h.strip() else "14 1 0 0 1 1"

for i in range(df4.shape[0]):
    list2 = str(df4.loc[i, "PredictionString"]).split()
    df4.loc[i, "PredictionString"] = " ".join(list2) if list2 else "14 1 0 0 1 1"



## === cell 5
_ = df4.iloc[1, min(16, df4.shape[1] - 1)]



## === cell 6
df5 = pd.merge(df, df2, on="image_id", how="left", suffixes=("", "_df2"))
if "PredictionString_df2" in df5.columns:
    df5 = df5.drop(columns=["PredictionString_df2"])
df5["PredictionString"] = df5["PredictionString"].fillna("14 1 0 0 1 1")



## === cell 7
same_pred = (
    df4["PredictionString"].astype(str).values
    == df5["PredictionString"].astype(str).values
).all()
if not same_pred:
    df4[["PredictionString"]] = (
        df4[["PredictionString"]] + " " + df5[["PredictionString"]]
    )

df4["PredictionString"] = df4["PredictionString"].astype(str).str.split().str.join(" ")



## === cell 8
list1 = list(range(15))

for i in range(df4.shape[0]):
    if df4.loc[i, "PredictionString"] == "14 1 0 0 1 1":
        continue
    a = df4.loc[i, "PredictionString"]
    b = a.split()
    if len(b) % 6 != 0:
        continue
    for j in range(int(len(b) / 6)):
        for k in list1:
            if int(float(b[0 + 6 * j])) == k:
                if float(df4.loc[i, f"{k}"]) < 0.80:
                    continue
                c = float(b[0 + 6 * j + 1])
                prior = float(df4.loc[i, f"{k}"])
                new_c = prior * 0.75 + c * 0.25
                new_c = max(c, new_c)  # never reduce an existing box confidence
                new_c = min(0.999, new_c)  # keep within valid range
                b[0 + 6 * j + 1] = str(new_c)
    df4.loc[i, "PredictionString"] = " ".join(b)



## === cell 9
for i in range(df4.shape[0]):
    if float(df4.loc[i, "14"]) > 0.999:
        df4.loc[i, "PredictionString"] = "14 1 0 0 1 1"

df_final = df4[["image_id", "PredictionString"]].copy()

df_final.to_csv("submission.csv", index=False)



## === cell 10
df_final
