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

0.2551051440444985

# 6. Current score

0.0422

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0475) has done: 'Your notebook currently fails because it depends on external Kaggle datasets (`../input/vin-15-cnn-predict*`, `../input/0284-norm`) that are not present in this environment, so no `df`/`df4` is ever created and downstream cells crash. To make it run end-to-end and still produce a valid submission, I replace those missing reads with a robust fallback that uses the provided `sample_submission.csv` structure and outputs a valid “No finding” prediction for every test image (this is score-poor but correct-format and unblocks submission generation). I also fix the submission column name to match the actual competition (`image_id,PredictionString`) rather than the older `ID,TARGET` wording. All changes are targeted to removing FileNotFound/NameError failures and ensuring a `.csv` submission is written.'
- What this solution (achieved 0.01849) has done: 'Your current score (0.0475) is far below the target (0.2551), and the reason is that the solution submits “No finding” for every image, which is valid-format but systematically wrong. To move the score upward with minimal logic changes, we keep a lightweight, deterministic heuristic: use the train annotations to estimate per-class priors and typical box sizes, then emit a small number of high-prior class predictions per test image with moderate confidence. This preserves the “single-pass, no-model” nature of the notebook while making predictions less degenerate than always class 14, which should improve mAP toward the target without adding any heavy dependencies or training loops. We also keep the required fallback to class 14 for any image where prediction string would otherwise be empty.'
- What this solution (achieved 0.0165) has done: 'Your current submission predicts the same few median boxes/classes for every image, which severely hurts mAP because most images won’t contain those objects and false positives are heavily penalized. To move the score upward toward 0.255 with minimal changes and no model training, I (1) stop emitting global “top-K” objects for every image, and instead (2) infer an image-specific “objectness” score from the train set by using each image’s number of annotated boxes; then (3) only emit a small number of predictions for a matching fraction of test images while emitting “No finding” for the rest. This keeps your heuristic nature intact (still only using train priors/medians), but reduces false positives dramatically, which typically improves mAP from a degenerate all-images-positive baseline. I also add a deterministic per-image confidence/box jitter (still based on train box statistics) to avoid identical predictions across images, which can slightly help ranking without changing the core approach.'
- What this solution (achieved 0.03673) has done: 'Your current heuristic still emits the same top classes for a fairly large fraction of test images, creating many false positives that suppress mAP. To move the score upward toward the target with minimal changes, I (1) make the “predict any object” gate stricter by estimating the *any-object prevalence* correctly over all train images (including the “No finding” rows), and (2) reduce the number of emitted boxes to 1 per positive image (keeping the same median-box-per-class logic) to further cut false positives. I also tune the confidence to be a bit lower (to reduce the impact of remaining false positives on AP ranking) while keeping the same prediction-string construction and fallback to class 14. No model/training is added; this stays a pure train-prior heuristic and still writes a valid `submission.csv`.'
- What this solution (achieved 0.03991) has done: 'We keep your “train-prior median box + deterministic gating” heuristic intact, but make two minimal changes aimed at improving mAP by reducing false positives while adding a tiny bit more recall. Specifically: (1) compute `p_has_object` over *all* unique train images by explicitly adding the missing “No finding” images (those with only class 14 are not present as separate rows), which makes the gate meaningfully stricter and better calibrated; (2) for images that pass the gate, emit up to 2 boxes (instead of 1) but only when the second class is sufficiently common, so we increase recall without spraying false positives everywhere. Everything else (no model/training, same median boxes, same deterministic jitter, same submission format) remains the same and still writes `submission.csv`.'
- What this solution (achieved 0.0422) has done: 'Your current heuristic is dominated by false positives because it predicts objects for too many test images using a gate based on train prevalence, while the test set prevalence can differ a lot; that keeps mAP far below the target. To move the score upward with minimal changes and without changing the core “train-prior median box + deterministic gating + optional 2nd box” logic, I calibrate the gate on the sample submission itself by predicting objects for only a fixed fraction of test images (set close to 20%), using the same deterministic hash so it’s stable. I also slightly reduce the second-box emission by making the class-frequency ratio requirement stricter, which cuts false positives further while keeping some recall. Everything else (box source, jitter, formatting, and the required “No finding” fallback) stays the same and still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import hashlib
import pandas as pd



## === cell 1
CANDIDATE_SAMPLE_PATHS = [
    "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/sample_submission.csv",
    "/kaggle/data/vinbigdata-chest-xray-abnormalities-detection/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
]
sample_path = None
for p in CANDIDATE_SAMPLE_PATHS:
    if os.path.exists(p):
        sample_path = p
        break

if sample_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected Kaggle paths. "
        "Checked: " + ", ".join(CANDIDATE_SAMPLE_PATHS)
    )

sample = pd.read_csv(sample_path)

if "image_id" not in sample.columns or "PredictionString" not in sample.columns:
    raise ValueError(
        f"Unexpected sample submission columns: {list(sample.columns)}. "
        "Expected ['image_id','PredictionString']."
    )

sample.head()



## === cell 2
CANDIDATE_TRAIN_PATHS = [
    "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train.csv",
    "/kaggle/data/vinbigdata-chest-xray-abnormalities-detection/train.csv",
    "/kaggle/input/train.csv",
    "/kaggle/data/train.csv",
]
train_path = None
for p in CANDIDATE_TRAIN_PATHS:
    if os.path.exists(p):
        train_path = p
        break

if train_path is None:
    raise FileNotFoundError(
        "Could not find train.csv in expected Kaggle paths. "
        "Checked: " + ", ".join(CANDIDATE_TRAIN_PATHS)
    )

train = pd.read_csv(train_path)

required_cols = ["image_id", "class_id", "x_min", "y_min", "x_max", "y_max"]
missing = [c for c in required_cols if c not in train.columns]
if missing:
    raise ValueError(f"train.csv missing columns: {missing}")

train["image_id"] = train["image_id"].astype(str)
train["class_id"] = pd.to_numeric(train["class_id"], errors="coerce")

CANDIDATE_TRAIN_DIRS = [
    "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train",
    "/kaggle/data/vinbigdata-chest-xray-abnormalities-detection/train",
    "/kaggle/input/train",
    "/kaggle/data/train",
]
train_dir = next((d for d in CANDIDATE_TRAIN_DIRS if os.path.isdir(d)), None)
if train_dir is None:
    all_train_image_ids = train["image_id"].dropna().unique()
else:
    all_train_image_ids = sorted(
        [
            fn.replace(".dicom", "")
            for fn in os.listdir(train_dir)
            if fn.endswith(".dicom")
        ]
    )

obj_present_by_image = train.groupby("image_id")["class_id"].apply(
    lambda s: (s.astype(float) != 14).any()
)
obj_present_by_image = obj_present_by_image.reindex(all_train_image_ids).fillna(False)

p_has_object_train = (
    float(obj_present_by_image.mean()) if len(obj_present_by_image) else 0.0
)

train_obj = train[(train["class_id"] != 14)].copy()
for c in ["x_min", "y_min", "x_max", "y_max"]:
    train_obj[c] = pd.to_numeric(train_obj[c], errors="coerce")
train_obj = train_obj.dropna(subset=["class_id", "x_min", "y_min", "x_max", "y_max"])
train_obj = train_obj[
    (train_obj["x_max"] > train_obj["x_min"])
    & (train_obj["y_max"] > train_obj["y_min"])
]

class_counts = train_obj["class_id"].value_counts().sort_values(ascending=False)
median_boxes = (
    train_obj.groupby("class_id")[["x_min", "y_min", "x_max", "y_max"]]
    .median()
    .loc[class_counts.index]
)

TOPK = 2
top_classes_pool = class_counts.head(10).index.tolist()  # consider a small pool

base_confs_by_rank = [0.20, 0.12]

if len(top_classes_pool) >= 2:
    c1, c2 = top_classes_pool[0], top_classes_pool[1]
    ratio = (
        float(class_counts.loc[c2]) / float(class_counts.loc[c1])
        if class_counts.loc[c1]
        else 0.0
    )
    if ratio < 0.70:
        top_classes = [c1]
    else:
        top_classes = [c1, c2]
else:
    top_classes = class_counts.head(1).index.tolist()

base_confs = base_confs_by_rank[: len(top_classes)]

if not top_classes:
    top_classes = [14]
    base_confs = [1.0]

(p_has_object_train, TOPK, top_classes, base_confs)




## === cell 3
def _u01_from_id(s: str) -> float:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 16**8  # [0,1)


def _jitter_params(image_id: str, cls_id: int):
    u = _u01_from_id(f"{image_id}_{cls_id}")
    v = _u01_from_id(f"{cls_id}_{image_id}")
    dx = int(round((u - 0.5) * 16))  # [-8, 8]
    dy = int(round((v - 0.5) * 16))  # [-8, 8]
    cm = 0.90 + 0.20 * _u01_from_id(f"conf_{image_id}_{cls_id}")
    return dx, dy, cm


TARGET_POS_FRAC_TEST = (
    0.20  # tuned to be conservative; minimal change vs using p_has_object_train
)


def build_pred_string(image_id: str) -> str:
    if _u01_from_id(image_id) > TARGET_POS_FRAC_TEST:
        return "14 1 0 0 1 1"

    preds = []
    for cls_id, base_conf in zip(top_classes, base_confs):
        if int(cls_id) == 14:
            preds.append("14 1 0 0 1 1")
            continue

        box = median_boxes.loc[cls_id]
        xmin = float(box["x_min"])
        ymin = float(box["y_min"])
        xmax = float(box["x_max"])
        ymax = float(box["y_max"])

        dx, dy, cm = _jitter_params(image_id, int(cls_id))
        xmin = int(max(0, round(xmin + dx)))
        ymin = int(max(0, round(ymin + dy)))
        xmax = int(max(xmin + 1, round(xmax + dx)))
        ymax = int(max(ymin + 1, round(ymax + dy)))

        conf = float(base_conf) * float(cm)
        conf = max(0.02, min(0.70, conf))

        preds.append(f"{int(cls_id)} {conf:.4f} {xmin} {ymin} {xmax} {ymax}")

    ps = " ".join(preds).strip()
    return ps if ps else "14 1 0 0 1 1"


sample_ids = sample["image_id"].head(5).tolist()
[(i, build_pred_string(i)) for i in sample_ids]



## === cell 4
df_final = sample.copy()

df_final["PredictionString"] = df_final["image_id"].astype(str).map(build_pred_string)

df_final["PredictionString"] = df_final["PredictionString"].fillna("").astype(str)
df_final.loc[df_final["PredictionString"].str.strip().eq(""), "PredictionString"] = (
    "14 1 0 0 1 1"
)

df_final.head()



## === cell 5
out_path = "submission.csv"
df_final.to_csv(out_path, index=False)

print(
    f"Wrote {out_path} with shape={df_final.shape} and columns={list(df_final.columns)}\n"
    f"p_has_object(train-derived, incl. implicit no-finding images)={p_has_object_train:.4f}, "
    f"target_pos_frac_test(gating)={TARGET_POS_FRAC_TEST:.3f}, "
    f"n_pred_classes={len(top_classes)}, top_classes={top_classes}\n"
    f"Example PredictionString: {df_final.loc[0,'PredictionString'][:160]}"
)



## === cell 6
assert out_path.endswith(".csv")
assert df_final.shape[0] == sample.shape[0]
assert df_final["image_id"].isna().sum() == 0
assert df_final["PredictionString"].isna().sum() == 0
df_final.describe(include="all")
