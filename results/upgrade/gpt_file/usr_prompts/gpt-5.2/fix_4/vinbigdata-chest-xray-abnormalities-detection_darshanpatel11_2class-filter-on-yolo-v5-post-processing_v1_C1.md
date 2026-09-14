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
numpy==1.26.4
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

0.2177142085841383

# 6. Current score

0.06689

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0475) has done: 'The current notebook fails because it references two external Kaggle datasets/paths that are not available in your environment. I replace those reads with a self-contained fallback that uses the provided `sample_submission.csv` as the base detection submission and generates a reasonable `pred_2class` (image-level “normal” probability) from the provided `train.csv` priors, so the pipeline runs end-to-end and writes a valid `submission.csv`. I also make the merge robust to missing columns and ensure every test image has a non-empty `PredictionString` (defaulting to the required “No finding” string). These changes are minimal and keep your existing post-processing logic/threshold semantics intact while producing a valid submission.'
- What this solution (achieved 0.06606) has done: 'Your current score (0.0475) is far below the target (0.2177), so we should legitimately increase mAP with the smallest change that preserves your “sample_submission-as-base + add/replace No finding based on class0 thresholds” core logic. The main issue is that you currently start from *all No finding* detections, which yields near-zero true positives for abnormal boxes; instead, we can build a simple, legal baseline detector by copying *training-set consensus boxes* (grouped by class) onto each test image, producing non-empty abnormal predictions while keeping your thresholding/post-processing semantics intact. To avoid flooding with too many low-quality boxes (which can hurt mAP), we keep only the top few most frequent classes and use the class frequency as the confidence, clipped to a reasonable range. We also keep your “No finding” threshold logic, but we ensure the produced PredictionString is always valid and formatted correctly.'
- What this solution (achieved 0.06689) has done: 'I keep your “training-consensus template boxes + optional No finding append/replace” core logic intact, but slightly increase the amount of useful signal in the template so mAP can move toward the 0.2177 target from 0.0661. Concretely, I (1) use class-wise box *medians* as you do but also add a second, slightly different box per class (using 25th/75th percentiles) to better cover size variability without changing model type, (2) increase TOPK_CLASSES a bit and cap total predicted boxes per image to avoid flooding, and (3) make the No finding append safe by inserting a separator space only when needed. These are minimal, legal post-processing changes that should increase true positives relative to your current single-template-per-class approach while keeping runtime well under the limit and preserving your thresholding semantics.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)



## === cell 1
DATA_DIR_CANDIDATES = [
    "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection",
    "/kaggle/data/vinbigdata-chest-xray-abnormalities-detection",
    "/kaggle/input",
    "/kaggle/data",
]


def _first_existing(path_list):
    for p in path_list:
        if os.path.exists(p):
            return p
    return None


BASE_DIR = _first_existing(DATA_DIR_CANDIDATES)
if BASE_DIR is None:
    raise FileNotFoundError(
        "Could not locate Kaggle input data directory. Checked: "
        + str(DATA_DIR_CANDIDATES)
    )

sample_path = os.path.join(BASE_DIR, "sample_submission.csv")
train_path = os.path.join(BASE_DIR, "train.csv")

sample_df = pd.read_csv(sample_path)
train_df = pd.read_csv(train_path)

test_image_ids = sample_df["image_id"].values

t = train_df.copy()
for col in ["class_id", "x_min", "y_min", "x_max", "y_max"]:
    t[col] = pd.to_numeric(t[col], errors="coerce")
t = t.dropna(subset=["class_id", "x_min", "y_min", "x_max", "y_max"])
t["class_id"] = t["class_id"].astype(int)

t_abn = t[t["class_id"] != 14].copy()

NORMAL = "14 1 0 0 1 1"

if len(t_abn) == 0:
    pred_det_df = sample_df.rename(
        columns={"PredictionString": "PredictionString"}
    ).copy()
    if "PredictionString" not in pred_det_df.columns:
        pred_det_df["PredictionString"] = NORMAL
else:
    cls_counts = t_abn["class_id"].value_counts().sort_values(ascending=False)
    total_abn = float(cls_counts.sum())

    TOPK_CLASSES = 8
    top_classes = cls_counts.head(TOPK_CLASSES).index.tolist()

    gb = t_abn[t_abn["class_id"].isin(top_classes)].groupby("class_id")[
        ["x_min", "y_min", "x_max", "y_max"]
    ]

    q_med = gb.quantile(0.50).reset_index()
    q1 = gb.quantile(0.25).reset_index()
    q3 = gb.quantile(0.75).reset_index()

    def _conf_from_count(c):
        p = float(c) / total_abn
        return float(np.clip(p, 0.10, 0.75))

    conf_map = {
        int(cid): _conf_from_count(int(cls_counts.loc[cid])) for cid in top_classes
    }

    def _sanitize_box(x1, y1, x2, y2):
        xmin = max(0.0, min(float(x1), float(x2)))
        ymin = max(0.0, min(float(y1), float(y2)))
        xmax = max(xmin + 1.0, max(float(x1), float(x2)))
        ymax = max(ymin + 1.0, max(float(y1), float(y2)))
        return xmin, ymin, xmax, ymax

    parts = []

    for _, r in q_med.iterrows():
        cid = int(r["class_id"])
        conf = conf_map[cid]
        xmin, ymin, xmax, ymax = _sanitize_box(
            r["x_min"], r["y_min"], r["x_max"], r["y_max"]
        )
        parts.append(f"{cid} {conf:.4f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}")

    for cid in top_classes:
        r1 = q1[q1["class_id"] == cid].iloc[0]
        r3 = q3[q3["class_id"] == cid].iloc[0]
        conf = max(0.05, conf_map[int(cid)] * 0.75)
        xmin, ymin, xmax, ymax = _sanitize_box(
            r1["x_min"], r1["y_min"], r3["x_max"], r3["y_max"]
        )
        parts.append(
            f"{int(cid)} {conf:.4f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
        )

    MAX_BOXES_PER_IMAGE = 12
    parts = parts[:MAX_BOXES_PER_IMAGE]

    template_pred_str = " ".join(parts).strip()
    if template_pred_str == "":
        template_pred_str = NORMAL

    pred_det_df = pd.DataFrame(
        {"image_id": test_image_ids, "PredictionString": template_pred_str}
    )

has_finding_per_image = train_df.groupby("image_id")["class_id"].apply(
    lambda s: (pd.to_numeric(s, errors="coerce").fillna(14).astype(int) != 14).any()
)
normal_prior = float((~has_finding_per_image).mean())  # P(normal)

pred_2class = pd.DataFrame(
    {"image_id": pred_det_df["image_id"].values, "class0": normal_prior}
)

low_threshold = 0.0
high_threshold = 0.976

print(f"Using BASE_DIR={BASE_DIR}")
print(
    f"Built pred_2class with constant class0(normal) prior={normal_prior:.6f} for {len(pred_2class)} test images"
)
print(
    f"Detector template uses TOPK_CLASSES={TOPK_CLASSES}, MAX_BOXES_PER_IMAGE={MAX_BOXES_PER_IMAGE}."
)
print("Example PredictionString:")
print(pred_det_df.head(1))



## === cell 2
NORMAL = "14 1 0 0 1 1"

if "PredictionString" not in pred_det_df.columns:
    for c in pred_det_df.columns:
        if c.lower() in ("predictionstring", "target"):
            pred_det_df = pred_det_df.rename(columns={c: "PredictionString"})
            break
if "PredictionString" not in pred_det_df.columns:
    pred_det_df["PredictionString"] = NORMAL

pred_det_df["PredictionString"] = pred_det_df["PredictionString"].fillna("").astype(str)
pred_det_df.loc[
    pred_det_df["PredictionString"].str.strip().eq(""), "PredictionString"
] = NORMAL

n_normal_before = len(pred_det_df.query("PredictionString == @NORMAL"))

merged_df = pd.merge(pred_det_df, pred_2class, on="image_id", how="left")

if "target" in merged_df.columns and "class0" not in merged_df.columns:
    merged_df["class0"] = 1 - merged_df["target"]

if "class0" not in merged_df.columns:
    merged_df["class0"] = 1.0
merged_df["class0"] = merged_df["class0"].fillna(1.0).astype(float)

c0, c1, c2 = 0, 0, 0
for i in range(len(merged_df)):
    p0 = float(merged_df.loc[i, "class0"])
    if p0 < low_threshold:
        c0 += 1
    elif low_threshold <= p0 and p0 < high_threshold:
        base = str(merged_df.loc[i, "PredictionString"]).strip()
        if base == "":
            base = NORMAL
        merged_df.loc[i, "PredictionString"] = (base + f" 14 {p0:.4f} 0 0 1 1").strip()
        c1 += 1
    else:
        merged_df.loc[i, "PredictionString"] = NORMAL
        c2 += 1

merged_df["PredictionString"] = merged_df["PredictionString"].fillna("").astype(str)
merged_df.loc[merged_df["PredictionString"].str.strip().eq(""), "PredictionString"] = (
    NORMAL
)

n_normal_after = len(merged_df.query("PredictionString == @NORMAL"))
print(
    f"n_normal: {n_normal_before} -> {n_normal_after} with threshold {low_threshold} & {high_threshold}"
)
print(f"Keep {c0} Add {c1} Replace {c2}")

submission_filepath = "submission.csv"
submission_df = merged_df[["image_id", "PredictionString"]].copy()
submission_df.to_csv(submission_filepath, index=False)
print(f"Saved to {submission_filepath}")
print(submission_df.head())
