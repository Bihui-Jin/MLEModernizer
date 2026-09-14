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
Categorize radiographs as negative for pneumonia or typical, indeterminate, or atypical for COVID-19.

For each test image, you will be predicting a bounding box and class for all findings. If you predict that there are no findings, you should create a prediction of "none 1 0 0 1 1" ("none" is the class ID for no finding, and this provides a one-pixel bounding box with a confidence of 1.0).


For each test study, you should make a determination within the following labels:

```
'Negative for Pneumonia'
'Typical Appearance'
'Indeterminate Appearance'
'Atypical Appearance'
```

## Metric
Standard PASCAL VOC 2010 mean Average Precision (mAP) at IoU > `0.5`. 

Make predictions at both a study (multi-image) and image level.

### Study-level labels
Studies in the test set may contain more than one label. They are as follows:

> "negative", "typical", "indeterminate", "atypical"

For each study in the test set, you should predict at least one of the above labels. The format for a given label's prediction would be a class ID from the above list, a `confidence` score, and `0 0 1 1` is a one-pixel bounding box.

### Image-level labels
Images in the test set may contain more than one object. For each object in a given test image, you must predict a class ID of "opacity", a `confidence` score, and bounding box in format `xmin ymin xmax ymax`. If you predict that there are NO objects in a given image, you should predict `none 1.0 0 0 1 1`, where `none` is the class ID for "No finding", 1.0 is the confidence, and `0 0 1 1` is a one-pixel bounding box.

## Submission Format
The submission file should contain a header and have the following format:

```
Id,PredictionString
2b95d54e4be65_study,negative 1 0 0 1 1
2b95d54e4be66_study,typical 1 0 0 1 1
2b95d54e4be67_study,indeterminate 1 0 0 1 1 atypical 1 0 0 1 1
2b95d54e4be68_image,none 1 0 0 1 1
2b95d54e4be69_image,opacity 0.5 100 100 200 200 opacity 0.7 10 10 20 20
etc.
```

## Dataset 
The train dataset comprises chest scans in DICOM format.

All images are stored in paths with the form `study`/`series`/`image`. The `study` ID here relates directly to the study-level predictions, and the `image` ID is the ID used for image-level predictions.

-   **train_study_level.csv** - the train study-level metadata, with one row for each study, including correct labels.
-   **train_image_level.csv** - the train image-level metadata, with one row for each image, including both correct labels and any bounding boxes in a dictionary format. Some images in both test and train have multiple bounding boxes.
-   **sample_submission.csv** - a sample submission file containing all image- and study-level IDs.

### Columns
**train_study_level.csv**

-   `id` - unique study identifier
-   `Negative for Pneumonia` - `1` if the study is negative for pneumonia, `0` otherwise
-   `Typical Appearance` - `1` if the study has this appearance, `0` otherwise
-   `Indeterminate Appearance`  - `1` if the study has this appearance, `0` otherwise
-   `Atypical Appearance`  - `1` if the study has this appearance, `0` otherwise

**train_image_level.csv**

-   `id` - unique image identifier
-   `boxes` - bounding boxes in easily-readable dictionary format
-   `label` - the correct prediction label for the provided bounding boxes

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
            description.md (345 lines)
            sample_submission.csv (1245 lines)
            sample_submission.csv.zip (10.7 kB)
            test.zip (7.6 GB)
            train.zip (67.7 GB)
            train_image_level.csv (5697 lines)
            train_image_level.csv.zip (405.9 kB)
            train_study_level.csv (5449 lines)
            train_study_level.csv.zip (45.4 kB)
            siim-covid19-detection/
                description.md (345 lines)
                sample_submission.csv (1245 lines)
                ... and 7 other files
                siim-covid19-detection/
                test/
                    000c9c05fd14/
                        e555410bd2cd/
                            ... (max depth reached)
                    00c74279c5b7/
                        ca867739fd1b/
                            ... (max depth reached)
                    ... and 605 other folders
                train/
                    00086460a852/
                        9e8302230c91/
                            ... (max depth reached)
                    00292f8c37bd/
                        73120b4a13cb/
                            ... (max depth reached)
                    ... and 5447 other folders
            test/
                000c9c05fd14/
                    e555410bd2cd/
                        51759b5579bc.dcm (17.6 MB)
                00c74279c5b7/
                    ca867739fd1b/
                        136af218f8df.dcm (15.7 MB)
                ... and 605 other folders
            train/
                00086460a852/
                    9e8302230c91/
                        65761e66de9f.dcm (13.0 MB)
                00292f8c37bd/
                    73120b4a13cb/
                        f6293b1c49e2.dcm (15.5 MB)
                ... and 5447 other folders
        input/
            description.md (345 lines)
            sample_submission.csv (1245 lines)
            sample_submission.csv.zip (10.7 kB)
            test.zip (7.6 GB)
            train.zip (67.7 GB)
            train_image_level.csv (5697 lines)
            train_image_level.csv.zip (405.9 kB)
            train_study_level.csv (5449 lines)
            train_study_level.csv.zip (45.4 kB)
            siim-covid19-detection/
                description.md (345 lines)
                sample_submission.csv (1245 lines)
                ... and 7 other files
                siim-covid19-detection/
                test/
                    000c9c05fd14/
                        e555410bd2cd/
                            ... (max depth reached)
                    00c74279c5b7/
                        ca867739fd1b/
                            ... (max depth reached)
                    ... and 605 other folders
                train/
                    00086460a852/
                        9e8302230c91/
                            ... (max depth reached)
                    00292f8c37bd/
                        73120b4a13cb/
                            ... (max depth reached)
                    ... and 5447 other folders
            test/
                000c9c05fd14/
                    e555410bd2cd/
                        51759b5579bc.dcm (17.6 MB)
                00c74279c5b7/
                    ca867739fd1b/
                        136af218f8df.dcm (15.7 MB)
                ... and 605 other folders
            train/
                00086460a852/
                    9e8302230c91/
                        65761e66de9f.dcm (13.0 MB)
                00292f8c37bd/
                    73120b4a13cb/
                        f6293b1c49e2.dcm (15.5 MB)
                ... and 5447 other folders
        working/
            siim-covid19-detection/
                description.md (345 lines)
                sample_submission.csv (1245 lines)
                ... and 7 other files
                siim-covid19-detection/
                test/
                    000c9c05fd14/
                        e555410bd2cd/
                            ... (max depth reached)
                    00c74279c5b7/
                        ca867739fd1b/
                            ... (max depth reached)
                    ... and 605 other folders
                train/
                    00086460a852/
                        9e8302230c91/
                            ... (max depth reached)
                    00292f8c37bd/
                        73120b4a13cb/
                            ... (max depth reached)
                    ... and 5447 other folders
```

-> data/sample_submission.csv has 1244 rows and 2 columns.
The columns are: id, PredictionString

-> data/siim-covid19-detection/sample_submission.csv has 1244 rows and 2 columns.
The columns are: id, PredictionString

-> data/siim-covid19-detection/train_image_level.csv has 5696 rows and 4 columns.
The columns are: id, boxes, label, StudyInstanceUID

-> data/siim-covid19-detection/train_study_level.csv has 5448 rows and 5 columns.
The columns are: id, Negative for Pneumonia, Typical Appearance, Indeterminate Appearance, Atypical Appearance

-> data/train_image_level.csv has 5696 rows and 4 columns.
The columns are: id, boxes, label, StudyInstanceUID

-> data/train_study_level.csv has 5448 rows and 5 columns.
The columns are: id, Negative for Pneumonia, Typical Appearance, Indeterminate Appearance, Atypical Appearance

-> input/sample_submission.csv has 1244 rows and 2 columns.
The columns are: id, PredictionString

-> (stopped after 10 files for performance)

# 5. Target score

0.0930975760146081

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import hashlib
import ast

DATA_DIR_PRIMARY = "/kaggle/input/siim-covid19-detection"
DATA_DIR_FALLBACK = "/kaggle/input/siim-covid19-detection/siim-covid19-detection"

RNG_SEED = 12345
rng = np.random.default_rng(RNG_SEED)

sample_path_candidates = [
    os.path.join(DATA_DIR_PRIMARY, "sample_submission.csv"),
    os.path.join(DATA_DIR_FALLBACK, "sample_submission.csv"),
    "/kaggle/data/sample_submission.csv",
    "/kaggle/data/siim-covid19-detection/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
]
sample_path = next((p for p in sample_path_candidates if os.path.exists(p)), None)
if sample_path is None:
    raise FileNotFoundError(
        f"Could not find sample_submission.csv in candidates: {sample_path_candidates}"
    )

df_sample_submit = pd.read_csv(sample_path)

df_submit = df_sample_submit.copy()
if "Id" in df_submit.columns and "id" not in df_submit.columns:
    df_submit = df_submit.rename(columns={"Id": "id"})
if "id" not in df_submit.columns:
    raise ValueError(
        f"sample_submission.csv must contain an 'id' or 'Id' column. Found columns: {df_sample_submit.columns.tolist()}"
    )
if "PredictionString" not in df_submit.columns:
    pred_col = [c for c in df_submit.columns if c.lower() == "predictionstring"]
    if pred_col:
        df_submit = df_submit.rename(columns={pred_col[0]: "PredictionString"})
    else:
        df_submit["PredictionString"] = ""

df_submit["id"] = df_submit["id"].astype(str)

is_study = df_submit["id"].str.endswith("_study")
is_image = df_submit["id"].str.endswith("_image")

study_csv_candidates = [
    os.path.join(DATA_DIR_PRIMARY, "train_study_level.csv"),
    os.path.join(DATA_DIR_FALLBACK, "train_study_level.csv"),
    "/kaggle/data/train_study_level.csv",
    "/kaggle/data/siim-covid19-detection/train_study_level.csv",
]
study_csv_path = next((p for p in study_csv_candidates if os.path.exists(p)), None)

if study_csv_path is None:
    priors = {
        "negative": 0.45,
        "typical": 0.35,
        "indeterminate": 0.15,
        "atypical": 0.05,
    }
else:
    df_study = pd.read_csv(study_csv_path)
    study_cols = [
        "Negative for Pneumonia",
        "Typical Appearance",
        "Indeterminate Appearance",
        "Atypical Appearance",
    ]
    for c in study_cols:
        if c not in df_study.columns:
            raise ValueError(f"Missing column {c} in {study_csv_path}")
    freqs = df_study[study_cols].mean(axis=0).to_dict()
    priors = {
        "negative": float(freqs["Negative for Pneumonia"]),
        "typical": float(freqs["Typical Appearance"]),
        "indeterminate": float(freqs["Indeterminate Appearance"]),
        "atypical": float(freqs["Atypical Appearance"]),
    }
    for k in list(priors.keys()):
        priors[k] = float(np.clip(priors[k], 0.03, 0.97))

labels = ["negative", "typical", "indeterminate", "atypical"]
prior_vec = np.array([priors[l] for l in labels], dtype=float)
prior_vec = np.clip(prior_vec, 1e-6, None)
prior_vec = prior_vec / prior_vec.sum()

conf_min, conf_max = 0.08, 0.55
conf_vec = conf_min + (conf_max - conf_min) * prior_vec
conf_vec = np.clip(conf_vec, 0.01, 0.99)

study_parts = [f"{lab} {float(c):.6f} 0 0 1 1" for lab, c in zip(labels, conf_vec)]
study_pred_str = " ".join(study_parts)
df_submit.loc[is_study, "PredictionString"] = study_pred_str

img_csv_candidates = [
    os.path.join(DATA_DIR_PRIMARY, "train_image_level.csv"),
    os.path.join(DATA_DIR_FALLBACK, "train_image_level.csv"),
    "/kaggle/data/train_image_level.csv",
    "/kaggle/data/siim-covid19-detection/train_image_level.csv",
]
img_csv_path = next((p for p in img_csv_candidates if os.path.exists(p)), None)

pos_rate = 0.25
train_boxes = None  # will hold (x1,y1,x2,y2) integer boxes sampled from train
if img_csv_path is not None:
    df_img = pd.read_csv(img_csv_path)
    if "label" in df_img.columns:
        pos_rate = float(
            (df_img["label"].astype(str).str.contains("opacity", regex=False)).mean()
        )
        pos_rate = float(np.clip(pos_rate, 0.05, 0.60))

    if "boxes" in df_img.columns:
        pos_rows = df_img[
            df_img["label"].astype(str).str.contains("opacity", regex=False)
        ][["boxes"]].copy()
        parsed = []
        for s in pos_rows["boxes"].astype(str).tolist():
            if not s or s == "nan":
                continue
            try:
                b = ast.literal_eval(s)
                if isinstance(b, dict):
                    b = [b]
                if isinstance(b, list):
                    for d in b:
                        if not isinstance(d, dict):
                            continue
                        x = float(d.get("x", np.nan))
                        y = float(d.get("y", np.nan))
                        w = float(d.get("width", np.nan))
                        h = float(d.get("height", np.nan))
                        if (
                            np.isfinite(x)
                            and np.isfinite(y)
                            and np.isfinite(w)
                            and np.isfinite(h)
                            and w > 0
                            and h > 0
                        ):
                            x1, y1 = int(round(x)), int(round(y))
                            x2, y2 = int(round(x + w)), int(round(y + h))
                            if x2 > x1 and y2 > y1:
                                parsed.append((x1, y1, x2, y2))
            except Exception:
                continue
        if len(parsed) >= 50:
            train_boxes = np.array(parsed, dtype=np.int32)

image_ids = df_submit.loc[is_image, "id"].astype(str).tolist()
n_images = len(image_ids)

forced_pos_rate = float(np.clip(pos_rate, 0.10, 0.28))
n_pos = int(round(forced_pos_rate * n_images))
n_pos = int(np.clip(n_pos, 0, n_images))


def _stable_score(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


ranked = sorted(
    [(i, _stable_score(_id)) for i, _id in enumerate(image_ids)], key=lambda x: x[1]
)
pos_idx = set(i for i, _ in ranked[:n_pos])

opacity_conf = 0.33


def _pick_box_for_id(_id: str) -> str:
    if train_boxes is None or len(train_boxes) == 0:
        return "256 256 768 768"
    j = _stable_score(_id) % len(train_boxes)
    x1, y1, x2, y2 = train_boxes[j].tolist()
    x1 = int(np.clip(x1, 0, 1023))
    y1 = int(np.clip(y1, 0, 1023))
    x2 = int(np.clip(x2, x1 + 1, 1024))
    y2 = int(np.clip(y2, y1 + 1, 1024))
    return f"{x1} {y1} {x2} {y2}"


pred_img = []
for i, _id in enumerate(image_ids):
    if i in pos_idx:
        box = _pick_box_for_id(_id)
        pred_img.append(f"opacity {opacity_conf:.6f} {box}")
    else:
        pred_img.append("none 1 0 0 1 1")

df_submit.loc[is_image, "PredictionString"] = pred_img

df_submit["PredictionString"] = (
    df_submit["PredictionString"].fillna("none 1 0 0 1 1").astype(str)
)

if df_submit["id"].isna().any():
    raise ValueError("Found NaN in id column after processing.")

study_label_default = max(priors, key=priors.get)



## === cell 1
out_path = "./submission.csv"

df_out = df_submit[["id", "PredictionString"]].copy()
df_out = df_out.rename(columns={"id": "Id"})
df_out["Id"] = df_out["Id"].astype(str)
df_out["PredictionString"] = (
    df_out["PredictionString"].fillna("none 1 0 0 1 1").astype(str)
)

df_sample_ids = df_sample_submit.copy()
if "id" in df_sample_ids.columns:
    sample_order = df_sample_ids["id"].astype(str).tolist()
elif "Id" in df_sample_ids.columns:
    sample_order = df_sample_ids["Id"].astype(str).tolist()
else:
    raise ValueError(
        f"sample_submission.csv missing id/Id column: {df_sample_ids.columns.tolist()}"
    )

if df_out["Id"].duplicated().any():
    df_out = df_out.drop_duplicates(subset=["Id"], keep="first")

df_out = df_out.set_index("Id").reindex(sample_order).reset_index()

missing = df_out["PredictionString"].isna()
if missing.any():
    miss_ids = df_out.loc[missing, "Id"].astype(str)
    miss_is_study = miss_ids.str.endswith("_study")
    df_out.loc[missing & miss_is_study, "PredictionString"] = study_pred_str
    df_out.loc[missing & ~miss_is_study, "PredictionString"] = "none 1 0 0 1 1"

if list(df_out.columns) != ["Id", "PredictionString"]:
    raise ValueError(
        f"Bad submission columns: {df_out.columns.tolist()} (expected ['Id','PredictionString'])"
    )
if len(df_out) != len(sample_order):
    raise ValueError(
        f"Bad submission row count: {len(df_out)} (expected {len(sample_order)})"
    )
if df_out["Id"].isna().any() or df_out["Id"].duplicated().any():
    raise ValueError(
        "Submission Ids contain NaNs or duplicates; Kaggle will reject this file."
    )
if not out_path.endswith(".csv"):
    raise ValueError("Output submission path must end with .csv")

df_out.to_csv(out_path, index=False)

print(df_out.head(10))
print(f"\nLoaded sample from: {sample_path}")
print(f"Wrote submission to: {out_path}")
print(f"Rows: {len(df_out)}, Columns: {df_out.columns.tolist()}")
print(f"Unique Ids: {df_out['Id'].nunique()}")
print(f"Study default label (for reference): {study_label_default}")
print(f"Study priors used: {priors}")
print(f"Study prediction string used: {study_pred_str}")
print(
    f"Image positive rate read (for reference): {pos_rate:.4f} => forced_pos_rate={forced_pos_rate:.4f} => forced {n_pos}/{n_images} images with opacity"
)
print(f"Opacity confidence used: {opacity_conf:.6f}")
print(
    f"Using empirical train boxes: {train_boxes is not None and len(train_boxes) > 0} (n={0 if train_boxes is None else len(train_boxes)})"
)
