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
import ast

DATA_DIR = "/kaggle/input/siim-covid19-detection"

sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
df_sample_submit = pd.read_csv(sample_path)

study_candidates = [
    "/kaggle/input/2cls06thr/2cls_0.6thre.csv",
    "/kaggle/input/2cls06thr/2cls_0.6thre.csv.zip",
]
image_candidates = [
    "/kaggle/input/qnsres/submit.csv",
    "/kaggle/input/qnsres/submit.csv.zip",
]


def try_read_csv(candidates):
    for p in candidates:
        if not os.path.exists(p):
            continue
        try:
            if p.endswith(".zip"):
                return pd.read_csv(p, compression="zip")
            return pd.read_csv(p)
        except Exception:
            continue
    return None


df_work_study = try_read_csv(study_candidates)
df_work_image = try_read_csv(image_candidates)

df_submit = df_sample_submit.copy()
cols0 = {c.lower(): c for c in df_submit.columns}
if "id" in cols0:
    df_submit = df_submit.rename(columns={cols0["id"]: "Id"})
if "predictionstring" in cols0:
    df_submit = df_submit.rename(
        columns={cols0["predictionstring"]: "PredictionString"}
    )
df_submit = df_submit[["Id", "PredictionString"]]

train_study_path = os.path.join(DATA_DIR, "train_study_level.csv")
train_image_path = os.path.join(DATA_DIR, "train_image_level.csv")

df_train_study = pd.read_csv(train_study_path)
df_train_image = pd.read_csv(train_image_path)

study_labels = [
    "Negative for Pneumonia",
    "Typical Appearance",
    "Indeterminate Appearance",
    "Atypical Appearance",
]
study_class_ids = {
    "Negative for Pneumonia": "negative",
    "Typical Appearance": "typical",
    "Indeterminate Appearance": "indeterminate",
    "Atypical Appearance": "atypical",
}
priors = df_train_study[study_labels].mean().to_dict()

has_opacity = (
    df_train_image["label"].astype(str).str.contains(r"\bopacity\b", regex=True)
)
p_opacity = float(has_opacity.mean())
p_none = 1.0 - p_opacity

STUDY_BOX = "0 0 1 1"

sorted_labels = sorted(study_labels, key=lambda k: priors.get(k, 0.0), reverse=True)
top1 = sorted_labels[0]


def _extract_train_opacity_boxes(df_img: pd.DataFrame, max_boxes: int = 2048):
    boxes_all = []
    opacity_rows = df_img.loc[
        df_img["label"].astype(str).str.contains(r"\bopacity\b", regex=True)
        & df_img["boxes"].notna(),
        "boxes",
    ].astype(str)

    for s in opacity_rows.tolist():
        try:
            obj = ast.literal_eval(s)
        except Exception:
            continue
        if isinstance(obj, dict):
            obj = [obj]
        if not isinstance(obj, (list, tuple)):
            continue
        for b in obj:
            if not isinstance(b, dict):
                continue
            try:
                x = float(b.get("x", np.nan))
                y = float(b.get("y", np.nan))
                w = float(b.get("width", np.nan))
                h = float(b.get("height", np.nan))
            except Exception:
                continue
            if not np.isfinite([x, y, w, h]).all():
                continue
            if w <= 1 or h <= 1:
                continue
            xmin = int(round(x))
            ymin = int(round(y))
            xmax = int(round(x + w))
            ymax = int(round(y + h))
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes_all.append((xmin, ymin, xmax, ymax))
            if len(boxes_all) >= max_boxes:
                return boxes_all
    return boxes_all


train_opacity_boxes = _extract_train_opacity_boxes(df_train_image, max_boxes=2048)

if len(train_opacity_boxes) == 0:
    train_opacity_boxes = [(256, 256, 768, 768)]


def make_study_pred_all4():
    parts = []
    for k in study_labels:
        c = float(priors.get(k, 0.25))
        c = float(np.clip(c, 0.05, 0.60))
        parts.append(f"{study_class_ids[k]} {c:.6f} {STUDY_BOX}")
    return " ".join(parts)


def make_study_pred_top1():
    k = top1
    c = float(np.clip(priors.get(k, 0.25), 0.20, 0.75))
    return f"{study_class_ids[k]} {c:.6f} {STUDY_BOX}"


def make_image_pred_none_only():
    c_none = float(np.clip(p_none, 0.70, 0.99))
    return f"none {c_none:.6f} 0 0 1 1"


def make_image_pred_mixture(df_sample_ids: pd.Series):
    ids = df_sample_ids.astype(str)
    is_image = ids.str.endswith("_image")
    image_ids = ids[is_image].tolist()

    target_frac = float(np.clip(p_opacity, 0.05, 0.60))
    thresh = int(target_frac * (2**32 - 1))

    def _hash_u32(s: str) -> int:
        h = (
            pd.util.hash_pandas_object(pd.Series([s]), index=False)
            .values[0]
            .astype(np.uint64)
        )
        return int(h & np.uint64(0xFFFFFFFF))

    def pick_opacity(img_id: str) -> bool:
        return _hash_u32(img_id) < thresh

    c_op = float(np.clip(p_opacity, 0.15, 0.45))
    c_none = float(np.clip(p_none, 0.70, 0.99))

    def pick_box(img_id: str):
        idx = _hash_u32(img_id + "_box") % len(train_opacity_boxes)
        return train_opacity_boxes[idx]

    preds = {}
    for img_id in image_ids:
        if pick_opacity(img_id):
            xmin, ymin, xmax, ymax = pick_box(img_id)
            preds[img_id] = f"opacity {c_op:.6f} {xmin} {ymin} {xmax} {ymax}"
        else:
            preds[img_id] = f"none {c_none:.6f} 0 0 1 1"
    return preds


default_study_pred = make_study_pred_top1()
default_image_pred = make_image_pred_none_only()
image_default_map = make_image_pred_mixture(df_submit["Id"])




## === cell 1
def _normalize_pred_df(df):
    cols = {c.lower(): c for c in df.columns}
    if "id" in cols:
        df = df.rename(columns={cols["id"]: "Id"})
    if "predictionstring" in cols:
        df = df.rename(columns={cols["predictionstring"]: "PredictionString"})
    if ("Id" not in df.columns) or ("PredictionString" not in df.columns):
        return None
    df = df[["Id", "PredictionString"]].dropna(subset=["Id"])
    df["Id"] = df["Id"].astype(str)

    df["PredictionString"] = df["PredictionString"].astype(str)
    df["PredictionString"] = df["PredictionString"].where(
        df["PredictionString"].str.strip().ne(""), np.nan
    )

    df = df.drop_duplicates(subset=["Id"], keep="last")
    return df


if df_work_study is not None:
    df_work_study = _normalize_pred_df(df_work_study)
    if df_work_study is not None:
        df_submit = df_submit.merge(
            df_work_study, on="Id", how="left", suffixes=("", "_study")
        )
        df_submit["PredictionString"] = df_submit[
            "PredictionString_study"
        ].combine_first(df_submit["PredictionString"])
        df_submit = df_submit.drop(columns=["PredictionString_study"])

if df_work_image is not None:
    df_work_image = _normalize_pred_df(df_work_image)
    if df_work_image is not None:
        df_submit = df_submit.merge(
            df_work_image, on="Id", how="left", suffixes=("", "_image")
        )
        df_submit["PredictionString"] = df_submit[
            "PredictionString_image"
        ].combine_first(df_submit["PredictionString"])
        df_submit = df_submit.drop(columns=["PredictionString_image"])

df_submit["PredictionString"] = df_submit["PredictionString"].astype(str)
empty_mask = (
    df_submit["PredictionString"].str.strip().eq("")
    | df_submit["PredictionString"].isna()
)

is_study = df_submit["Id"].astype(str).str.endswith("_study")
is_image = df_submit["Id"].astype(str).str.endswith("_image")

df_submit.loc[empty_mask & is_study, "PredictionString"] = default_study_pred

need_img = empty_mask & is_image
if int(need_img.sum()) > 0:
    df_submit.loc[need_img, "PredictionString"] = (
        df_submit.loc[need_img, "Id"].map(image_default_map).fillna(default_image_pred)
    )

df_submit.loc[empty_mask & ~(is_study | is_image), "PredictionString"] = (
    "none 1 0 0 1 1"
)

df_submit = df_submit[["Id", "PredictionString"]]
df_submit["Id"] = df_submit["Id"].astype(str)

sample_ids_true = pd.read_csv(sample_path)["id"].astype(str)
order = pd.DataFrame({"Id": sample_ids_true})
df_submit = order.merge(df_submit, on="Id", how="left")

df_submit["PredictionString"] = df_submit["PredictionString"].fillna("none 1 0 0 1 1")
df_submit["PredictionString"] = df_submit["PredictionString"].astype(str)
df_submit.loc[df_submit["PredictionString"].str.strip().eq(""), "PredictionString"] = (
    "none 1 0 0 1 1"
)

df_submit = df_submit[["Id", "PredictionString"]]
if df_submit["Id"].isna().any():
    raise ValueError("Found NA Id in submission.")
if df_submit["Id"].nunique() != len(df_submit):
    raise ValueError("Duplicate Ids in submission after ordering/merge.")
if (df_submit["PredictionString"].astype(str).str.strip() == "").any():
    raise ValueError("Empty PredictionString present after filling defaults.")




## === cell 2
print(df_submit.head(10))
print("Rows:", len(df_submit), "Cols:", list(df_submit.columns))

sample_ids_true_list = pd.read_csv(sample_path)["id"].astype(str).tolist()
print(
    "ID order matches sample_submission:",
    df_submit["Id"].astype(str).tolist() == sample_ids_true_list,
)
print("Unique IDs:", df_submit["Id"].nunique(), "Expected:", len(sample_ids_true_list))

out_path = "./submission.csv"
df_submit.to_csv(out_path, index=False)
print("Wrote:", out_path)

print(
    "Empty PredictionString rows:",
    int((df_submit["PredictionString"].astype(str).str.strip() == "").sum()),
)
print(
    "Study rows:",
    int(df_submit["Id"].astype(str).str.endswith("_study").sum()),
    "Image rows:",
    int(df_submit["Id"].astype(str).str.endswith("_image").sum()),
)
print("Default study pred used:", default_study_pred)
print("Default image pred used (fallback):", default_image_pred)
print("Opacity prevalence (train):", p_opacity)
print(
    "Opacity-predicted image rows (default mixture):",
    int(
        df_submit.loc[
            df_submit["Id"].astype(str).str.endswith("_image"), "PredictionString"
        ]
        .astype(str)
        .str.startswith("opacity ")
        .sum()
    ),
)
print("Train opacity boxes available:", len(train_opacity_boxes))
