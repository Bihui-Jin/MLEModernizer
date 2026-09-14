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

DATA_DIR = "../input/siim-covid19-detection"
WORK_DIR = "../input"


def _read_csv_if_exists(path: str):
    if path is None:
        return None
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def _normalize_submission_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return None
    if "Id" in df.columns:
        return df
    if "id" in df.columns:
        return df.rename(columns={"id": "Id"})
    return df


sample_paths = [
    os.path.join(DATA_DIR, "sample_submission.csv"),
    os.path.join(WORK_DIR, "sample_submission.csv"),
    "../input/sample_submission.csv",
]
df_sample_submit = None
for p in sample_paths:
    df_sample_submit = _read_csv_if_exists(p)
    if df_sample_submit is not None:
        break
if df_sample_submit is None:
    raise FileNotFoundError(
        f"Could not find sample_submission.csv in any of: {sample_paths}"
    )

study_path = None
image_path = None

df_work_study = _read_csv_if_exists(study_path)
df_work_image = _read_csv_if_exists(image_path)

df_sample_submit = _normalize_submission_cols(df_sample_submit)
df_work_study = _normalize_submission_cols(df_work_study)
df_work_image = _normalize_submission_cols(df_work_image)

if (
    "Id" not in df_sample_submit.columns
    or "PredictionString" not in df_sample_submit.columns
):
    raise ValueError(
        "sample_submission must contain columns ['Id', 'PredictionString']."
    )

df_sample_submit = df_sample_submit[["Id", "PredictionString"]].copy()
df_sample_submit = df_sample_submit.set_index("Id")
df_submit = df_sample_submit.copy()

if df_work_study is not None:
    if "PredictionString" not in df_work_study.columns:
        raise ValueError(f"{study_path} exists but has no PredictionString column.")
    df_work_study = df_work_study.set_index("Id")
    common = df_work_study.index.intersection(df_submit.index)
    df_submit.loc[common, "PredictionString"] = df_work_study.loc[
        common, "PredictionString"
    ].values

if df_work_image is not None:
    if "PredictionString" not in df_work_image.columns:
        raise ValueError(f"{image_path} exists but has no PredictionString column.")
    df_work_image = df_work_image.set_index("Id")
    common_idx = df_work_image.index.intersection(df_submit.index)
    df_submit.loc[common_idx, "PredictionString"] = df_work_image.loc[
        common_idx, "PredictionString"
    ].values

train_study_paths = [
    os.path.join(DATA_DIR, "train_study_level.csv"),
    os.path.join(WORK_DIR, "train_study_level.csv"),
    "../input/train_study_level.csv",
]
df_train_study = None
for p in train_study_paths:
    df_train_study = _read_csv_if_exists(p)
    if df_train_study is not None:
        break

train_image_paths = [
    os.path.join(DATA_DIR, "train_image_level.csv"),
    os.path.join(WORK_DIR, "train_image_level.csv"),
    "../input/train_image_level.csv",
]
df_train_image = None
for p in train_image_paths:
    df_train_image = _read_csv_if_exists(p)
    if df_train_image is not None:
        break

labels = [
    "Negative for Pneumonia",
    "Typical Appearance",
    "Indeterminate Appearance",
    "Atypical Appearance",
]
label_to_classid = {
    "Negative for Pneumonia": "negative",
    "Typical Appearance": "typical",
    "Indeterminate Appearance": "indeterminate",
    "Atypical Appearance": "atypical",
}

if df_train_study is not None and all(c in df_train_study.columns for c in labels):
    priors = df_train_study[labels].mean(axis=0).astype(float).values
    priors = np.clip(priors, 1e-6, None)
    priors = priors / priors.sum()
else:
    priors = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)

study_default = " ".join(
    [f"{label_to_classid[lab]} {priors[i]:.6f} 0 0 1 1" for i, lab in enumerate(labels)]
)

study_opacity_rate = None
if df_train_image is not None and {"StudyInstanceUID", "label"}.issubset(
    df_train_image.columns
):
    tmp = df_train_image[["StudyInstanceUID", "label"]].copy()
    tmp["has_opacity"] = (
        tmp["label"]
        .astype(str)
        .str.contains("opacity", case=False, na=False)
        .astype(float)
    )
    study_opacity_rate = tmp.groupby("StudyInstanceUID")["has_opacity"].mean().to_dict()


def _study_prediction_string_for_uid(study_uid: str) -> str:
    if not study_opacity_rate or study_uid not in study_opacity_rate:
        return study_default
    r = float(study_opacity_rate[study_uid])
    p_neg = np.clip(0.75 - 0.6 * r, 1e-6, 1.0)
    p_typ = np.clip(0.15 + 0.5 * r, 1e-6, 1.0)
    p_ind = np.clip(0.07 + 0.08 * r, 1e-6, 1.0)
    p_aty = np.clip(0.03 + 0.02 * r, 1e-6, 1.0)
    probs = np.array([p_neg, p_typ, p_ind, p_aty], dtype=float)
    probs = probs / probs.sum()
    return " ".join(
        [
            f"{label_to_classid[lab]} {probs[i]:.6f} 0 0 1 1"
            for i, lab in enumerate(labels)
        ]
    )


def _image_default_prediction_string(study_uid: str) -> str:
    if not study_opacity_rate or study_uid not in study_opacity_rate:
        return "none 1 0 0 1 1"
    r = float(study_opacity_rate[study_uid])
    if r >= 0.55:
        conf = float(np.clip(0.30 + 0.25 * (r - 0.55) / 0.45, 0.30, 0.55))
        xmin, ymin, xmax, ymax = 256, 256, 768, 768
        return f"opacity {conf:.3f} {xmin} {ymin} {xmax} {ymax}"
    return "none 1 0 0 1 1"


pred = df_submit["PredictionString"].astype("string")
is_missing = pred.isna() | (pred.str.strip() == "")
idx_missing = df_submit.index[is_missing]

if len(idx_missing) > 0:
    idx_image = [i for i in idx_missing if str(i).endswith("_image")]
    idx_study = [i for i in idx_missing if str(i).endswith("_study")]

    if len(idx_image) > 0:
        for i in idx_image:
            study_uid = str(i).replace("_image", "")
            df_submit.loc[i, "PredictionString"] = _image_default_prediction_string(
                study_uid
            )
    if len(idx_study) > 0:
        for i in idx_study:
            study_uid = str(i).replace("_study", "")
            df_submit.loc[i, "PredictionString"] = _study_prediction_string_for_uid(
                study_uid
            )

df_submit["PredictionString"] = df_submit["PredictionString"].fillna("").astype(str)
still_empty = df_submit["PredictionString"].str.strip().eq("")
if still_empty.any():
    for row_id in df_submit.index[still_empty].astype(str).tolist():
        if row_id.endswith("_image"):
            study_uid = row_id.replace("_image", "")
            df_submit.loc[row_id, "PredictionString"] = (
                _image_default_prediction_string(study_uid)
            )
        else:
            study_uid = row_id.replace("_study", "")
            df_submit.loc[row_id, "PredictionString"] = (
                _study_prediction_string_for_uid(study_uid)
            )



## === cell 1
df_out = df_submit.reset_index(drop=False).rename(columns={"Id": "Id"})
df_out = df_out[["Id", "PredictionString"]].copy()

out_path = "./submission.csv"
df_out.to_csv(out_path, index=False)

print(df_out.head())
print(f"Wrote submission to: {out_path} with shape {df_out.shape}")
print(
    "Empty PredictionString rows:",
    int((df_out["PredictionString"].astype(str).str.strip() == "").sum()),
)
print("External study submission found:", df_work_study is not None)
print("External image submission found:", df_work_image is not None)
print("Using global study priors:", priors)
print("Train image-level loaded:", df_train_image is not None)
print("Has study_opacity_rate:", study_opacity_rate is not None)
