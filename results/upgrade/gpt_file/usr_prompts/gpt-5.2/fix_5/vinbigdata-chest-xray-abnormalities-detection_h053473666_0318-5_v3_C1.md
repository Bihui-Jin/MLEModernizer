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

0.2279394233788025

# 6. Current score

0.0238

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0475) has done: 'The original notebook depends on several external Kaggle Dataset submissions that are not available in your environment, causing immediate FileNotFoundError and cascading NameErrors. To make it run end-to-end while preserving the submission semantics, I replace those missing inputs with the provided `sample_submission.csv` as a safe baseline and create the expected per-class columns (`'0'..'14'`) so later cells don’t crash. I also fix hard-coded loop lengths (3000) to use the actual dataframe length (1500) and ensure `PredictionString` is always a string. This generate a valid `submission.csv` (with `.csv` suffix) in the correct format; score-wise it be a baseline (likely low) but at least yields a valid submission.'
- What this solution (achieved 0.0238) has done: 'Your current pipeline always falls back to the sample submission (mostly “No finding”), so it can’t reach the target mAP. To move the score toward 0.2279 with minimal semantic change, I replace the placeholder per-image per-class scores (`df['0'..'14']`) with real priors computed from `train.csv` (class frequency per image), then keep your existing merging/filtering/weighting logic intact so it can actually adjust confidences instead of doing nothing. I also make the row-iteration robust by using `.iloc` (avoids silent misalignment if the index isn’t 0..n-1) while preserving the same operations. This should legitimately improve detection confidence calibration (without changing your “model” structure) and increase mAP toward the target.'
- What this solution (achieved 0.01396) has done: 'Your current score is far below target, so we need a small but legitimate improvement that actually creates non-empty detections (mAP can’t improve if we always predict “No finding”). Without changing your overall pipeline structure, I inject simple, train-derived bounding box prototypes per class (median box) and generate a few per-image predicted boxes using your existing per-class prior scores as confidences. Then your existing filtering/merging/weighting logic operate on real boxes instead of mostly “14 1 0 0 1 1”, which should move mAP upward toward the 0.2279 target. I also keep the submission schema intact (`image_id`, `PredictionString`) and ensure `14` is only used when we truly have no predicted boxes.'
- What this solution (achieved 0.0238) has done: 'Your current score is far below target, so we need a minimal but legitimate change that increases mAP without altering your overall “priors + prototype boxes + post-filters” core pipeline. The biggest current limiter is that cell 3 deletes classes `0` and `7` entirely (and keeps only 1 box per class anyway), and cell 8 almost never triggers “No finding” because `df["14"]` is not near 1.0 for any row—so you often submit low-quality boxes for images that likely have none. I (1) stop removing classes `0` and `7` in the post-filter (only remove `14`), and (2) add a simple per-image “No finding gate” using your existing `df['14']` prior: if `p14` is higher than the best abnormal score, output only the required `14 1 0 0 1 1`. These are small, metric-aligned adjustments that should move the score upward toward the target while preserving your approach and submission semantics.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

BASE_DIR = "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection"

sample_path = os.path.join(BASE_DIR, "sample_submission.csv")
if not os.path.exists(sample_path):
    sample_path = "/kaggle/data/sample_submission.csv"

train_path = os.path.join(BASE_DIR, "train.csv")
if not os.path.exists(train_path):
    train_path = "/kaggle/data/train.csv"

sample = pd.read_csv(sample_path)
train_df = pd.read_csv(train_path)


def make_base_df(sample_df: pd.DataFrame) -> pd.DataFrame:
    out = sample_df.copy()
    if "PredictionString" not in out.columns:
        if "TARGET" in out.columns:
            out = out.rename(columns={"TARGET": "PredictionString"})
        elif "prediction" in out.columns:
            out = out.rename(columns={"prediction": "PredictionString"})
        else:
            out["PredictionString"] = "14 1 0 0 1 1"
    out["PredictionString"] = out["PredictionString"].fillna("14 1 0 0 1 1").astype(str)

    for k in range(15):
        out[str(k)] = 0.0
    out.loc[out["PredictionString"].str.strip().eq("14 1 0 0 1 1"), "14"] = 1.0
    return out


def compute_class_priors_per_image(train_df_: pd.DataFrame) -> pd.Series:
    img_cls = (
        train_df_.loc[train_df_["class_id"] != 14, ["image_id", "class_id"]]
        .drop_duplicates()
        .groupby("class_id")["image_id"]
        .nunique()
    )
    n_images = train_df_["image_id"].nunique()
    priors = (img_cls / max(n_images, 1)).reindex(range(14), fill_value=0.0)

    scaled = (priors * 3.5).clip(0.0, 0.999)

    nofind_imgs = (
        train_df_.loc[train_df_["class_id"] == 14, "image_id"]
        .drop_duplicates()
        .shape[0]
    )
    p14 = min(nofind_imgs / max(n_images, 1) * 1.5, 0.999)

    priors15 = pd.Series({str(k): float(scaled.loc[k]) for k in range(14)})
    priors15["14"] = float(p14)
    return priors15


def compute_class_box_prototypes(train_df_: pd.DataFrame) -> pd.DataFrame:
    cols = ["x_min", "y_min", "x_max", "y_max"]
    td = train_df_.copy()
    td = td.loc[td["class_id"] != 14].dropna(subset=cols + ["class_id"])
    for c in cols:
        td[c] = pd.to_numeric(td[c], errors="coerce")
    td = td.dropna(subset=cols)
    td = td.loc[(td["x_max"] > td["x_min"]) & (td["y_max"] > td["y_min"])]

    proto = td.groupby("class_id")[cols].median()
    proto = proto.reindex(range(14))

    fallback = pd.Series(
        {"x_min": 256.0, "y_min": 256.0, "x_max": 768.0, "y_max": 768.0}
    )
    for k in range(14):
        if k not in proto.index or proto.loc[k].isna().any():
            proto.loc[k] = fallback.values
    proto = proto.reset_index().rename(columns={"class_id": "class_id"})
    return proto


priors = compute_class_priors_per_image(train_df)
box_proto = compute_class_box_prototypes(train_df)

df = make_base_df(sample)
df1 = make_base_df(sample)
df_densenet = make_base_df(sample)

cols_0_14 = [str(i) for i in range(15)]

for k in cols_0_14:
    df[k] = priors.get(k, 0.0)
    df1[k] = priors.get(k, 0.0)
    df_densenet[k] = priors.get(k, 0.0)

df[cols_0_14] = (
    df[cols_0_14] * 0.25 + df1[cols_0_14] * 0.5 + df_densenet[cols_0_14] * 0.25
)

proto_map = box_proto.set_index("class_id")[
    ["x_min", "y_min", "x_max", "y_max"]
].to_dict("index")

K = 3
THR = 0.22  # small threshold so we emit some predictions but not for every class

NOFIND_MARGIN = 0.02  # small margin to avoid over-triggering "14"

pred_strings = []
for i in range(df.shape[0]):
    row = df.iloc[i]
    p14 = float(row["14"]) if "14" in row.index else 0.0

    class_scores = [(k, float(row[str(k)])) for k in range(14)]
    class_scores.sort(key=lambda x: x[1], reverse=True)
    best_abn = class_scores[0][1] if len(class_scores) else 0.0

    if p14 >= best_abn + NOFIND_MARGIN:
        pred_strings.append("14 1 0 0 1 1")
        continue

    chosen = [(k, s) for (k, s) in class_scores[:K] if s >= THR]

    if len(chosen) == 0:
        pred_strings.append("14 1 0 0 1 1")
        continue

    parts = []
    for k, s in chosen:
        bb = proto_map.get(k, {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0})
        conf = float(np.clip(s, 0.05, 0.95))
        parts.extend(
            [
                str(int(k)),
                f"{conf:.6f}",
                f"{float(bb['x_min']):.2f}",
                f"{float(bb['y_min']):.2f}",
                f"{float(bb['x_max']):.2f}",
                f"{float(bb['y_max']):.2f}",
            ]
        )
    pred_strings.append(" ".join(parts).strip())

df["PredictionString"] = pred_strings
df1["PredictionString"] = pred_strings
df_densenet["PredictionString"] = pred_strings



## === cell 1
df2 = sample.copy()
df3 = sample.copy()

if "PredictionString" not in df2.columns:
    df2 = (
        df2.rename(columns={"TARGET": "PredictionString"})
        if "TARGET" in df2.columns
        else df2
    )
if "PredictionString" not in df3.columns:
    df3 = (
        df3.rename(columns={"TARGET": "PredictionString"})
        if "TARGET" in df3.columns
        else df3
    )

df2["PredictionString"] = df2.get("PredictionString", "14 1 0 0 1 1")
df3["PredictionString"] = df3.get("PredictionString", "14 1 0 0 1 1")
df2["PredictionString"] = df2["PredictionString"].fillna("14 1 0 0 1 1").astype(str)
df3["PredictionString"] = df3["PredictionString"].fillna("14 1 0 0 1 1").astype(str)



## === cell 2
df4 = pd.merge(df, df3, on="image_id", how="left", suffixes=("", "_df3"))
if "PredictionString_df3" in df4.columns:
    df4["PredictionString"] = df4["PredictionString"].fillna(
        df4["PredictionString_df3"]
    )
    df4 = df4.drop(columns=["PredictionString_df3"])

df4["PredictionString"] = df4["PredictionString"].fillna("14 1 0 0 1 1").astype(str)



## === cell 3
n = df4.shape[0]
for i in range(n):
    ps = str(df4["PredictionString"].iloc[i]).strip()
    if ps == "" or ps.lower() == "nan":
        ps = "14 1 0 0 1 1"
    list2 = ps.split()

    h = ""
    num_boxes = len(list2) // 6
    for j in range(num_boxes):
        cls = list2[0 + 6 * j]
        if cls in ("14",):
            continue
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

    df4.loc[df4.index[i], "PredictionString"] = (
        h.strip() if h.strip() else "14 1 0 0 1 1"
    )

for i in range(n):
    list2 = str(df4["PredictionString"].iloc[i]).split()
    df4.loc[df4.index[i], "PredictionString"] = " ".join(list2)



## === cell 4
val = None
if df4.shape[0] > 1 and df4.shape[1] > 16:
    val = df4.iloc[1, 16]
val



## === cell 5
df5 = pd.merge(df, df2, on="image_id", how="left", suffixes=("", "_df2"))
if "PredictionString_df2" in df5.columns:
    df5["PredictionString"] = df5["PredictionString"].fillna(
        df5["PredictionString_df2"]
    )
    df5 = df5.drop(columns=["PredictionString_df2"])
df5["PredictionString"] = df5["PredictionString"].fillna("14 1 0 0 1 1").astype(str)



## === cell 6
df4[["PredictionString"]] = (
    df4[["PredictionString"]].astype(str) + " " + df5[["PredictionString"]].astype(str)
)
df4["PredictionString"] = df4["PredictionString"].str.strip()



## === cell 7
list1 = list(range(15))

for i in range(df4.shape[0]):
    if str(df4["PredictionString"].iloc[i]).strip() == "14 1 0 0 1 1":
        continue
    a = str(df4["PredictionString"].iloc[i])
    b = a.split()
    num_boxes = len(b) // 6
    for j in range(num_boxes):
        for k in list1:
            if int(float(b[0 + 6 * j])) == k:
                if str(k) not in df4.columns:
                    continue
                if float(df4.loc[df4.index[i], f"{k}"]) < 0.92:
                    continue
                c = b[0 + 6 * j + 1]
                b[0 + 6 * j + 1] = str(
                    float(df4.loc[df4.index[i], f"{k}"]) * 0.4 + float(c) * 0.6
                )
    df4.loc[df4.index[i], "PredictionString"] = " ".join(b).strip()



## === cell 8
for i in range(df4.shape[0]):
    if "14" in df4.columns and float(df4.loc[df4.index[i], "14"]) > 0.999:
        df4.loc[df4.index[i], "PredictionString"] = "14 1 0 0 1 1"

df4["PredictionString"] = df4["PredictionString"].astype(str).str.strip()
df4.loc[df4["PredictionString"].eq(""), "PredictionString"] = "14 1 0 0 1 1"

df_final = df4[["image_id", "PredictionString"]].copy()
df_final.to_csv("submission.csv", index=False)



## === cell 9
df_final
