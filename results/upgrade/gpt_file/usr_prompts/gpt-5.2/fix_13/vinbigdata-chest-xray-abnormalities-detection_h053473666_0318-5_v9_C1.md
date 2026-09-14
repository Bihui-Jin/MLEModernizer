# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

DATA_DIR_CANDIDATES = [
    "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection",
    "/kaggle/data/input/vinbigdata-chest-xray-abnormalities-detection",
    "/kaggle/input",
    "/kaggle/data/input",
]


def find_file(filename: str) -> str:
    for base in DATA_DIR_CANDIDATES:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
    for base in DATA_DIR_CANDIDATES:
        if os.path.exists(base):
            for root, _, files in os.walk(base):
                if filename in files:
                    return os.path.join(root, filename)
    raise FileNotFoundError(
        f"Could not find {filename} in candidates: {DATA_DIR_CANDIDATES}"
    )


sample_path = find_file("sample_submission.csv")
sample_sub = pd.read_csv(sample_path)

sample_cols = list(sample_sub.columns)

if "image_id" not in sample_sub.columns and "ID" in sample_sub.columns:
    sample_sub = sample_sub.rename(columns={"ID": "image_id"})
if "PredictionString" not in sample_sub.columns and "TARGET" in sample_sub.columns:
    sample_sub = sample_sub.rename(columns={"TARGET": "PredictionString"})

if "image_id" not in sample_sub.columns:
    raise ValueError(
        f"sample_submission.csv missing ID/image_id column. Columns: {sample_cols}"
    )
if "PredictionString" not in sample_sub.columns:
    raise ValueError(
        f"sample_submission.csv missing TARGET/PredictionString column. Columns: {sample_cols}"
    )

sample_sub["PredictionString"] = (
    sample_sub["PredictionString"].fillna("14 1 0 0 1 1").astype(str)
)
sample_sub.head()



## === cell 1
CLASS_COLS = [str(i) for i in range(15)]


def make_base_df_from_sample(sample: pd.DataFrame) -> pd.DataFrame:
    df = sample.copy()
    for c in CLASS_COLS:
        if c not in df.columns:
            df[c] = 0.0
    return df


df = make_base_df_from_sample(sample_sub)
df1 = make_base_df_from_sample(sample_sub)
df_densenet = make_base_df_from_sample(sample_sub)

df[CLASS_COLS] = (
    df[CLASS_COLS] * 0.25 + df1[CLASS_COLS] * 0.5 + df_densenet[CLASS_COLS] * 0.25
)



## === cell 2
df2 = make_base_df_from_sample(sample_sub)
df3 = make_base_df_from_sample(sample_sub)



## === cell 3
df4 = pd.merge(
    df,
    df3[["image_id", "PredictionString"]],
    on="image_id",
    how="left",
    suffixes=("", "_df3"),
)
df4["PredictionString"] = (
    df4["PredictionString_df3"].fillna(df4["PredictionString"]).astype(str)
)
df4 = df4.drop(columns=["PredictionString_df3"])

df4["PredictionString"] = df4["PredictionString"].fillna("14 1 0 0 1 1").astype(str)



## === cell 4
n_rows = df4.shape[0]

for i in range(n_rows):
    ps = df4.loc[i, "PredictionString"]
    if not isinstance(ps, str):
        ps = "14 1 0 0 1 1"
    tokens = ps.split()

    h_tokens = []
    n_preds = len(tokens) // 6
    for j in range(n_preds):
        cls = tokens[0 + 6 * j]
        if cls in {"0", "14", "7", "13", "12"}:
            continue
        h_tokens.extend(tokens[0 + 6 * j : 6 + 6 * j])

    df4.loc[i, "PredictionString"] = " ".join(h_tokens).strip()

for i in range(n_rows):
    df4.loc[i, "PredictionString"] = " ".join(
        str(df4.loc[i, "PredictionString"]).split()
    ).strip()



## === cell 5
_ = df4.iloc[1, 16] if (df4.shape[0] > 1 and df4.shape[1] > 16) else None
_



## === cell 6
df5 = pd.merge(
    df,
    df2[["image_id", "PredictionString"]],
    on="image_id",
    how="left",
    suffixes=("", "_df2"),
)
df5["PredictionString"] = (
    df5["PredictionString_df2"].fillna(df5["PredictionString"]).astype(str)
)
df5 = df5.drop(columns=["PredictionString_df2"])



## === cell 7
df4["PredictionString"] = df4["PredictionString"].fillna("14 1 0 0 1 1").astype(str)
df4["PredictionString"] = df4["PredictionString"].apply(
    lambda x: " ".join(str(x).split()).strip()
)



## === cell 8
list1 = list(range(15))

for i in range(df4.shape[0]):
    if df4.loc[i, "PredictionString"] == "14 1 0 0 1 1":
        continue
    a = str(df4.loc[i, "PredictionString"])
    b = a.split()
    n_preds = len(b) // 6
    for j in range(n_preds):
        cls = b[0 + 6 * j]
        try:
            cls_int = int(cls)
        except Exception:
            continue

        if cls_int in list1:
            score_col = str(cls_int)
            if score_col in df4.columns and float(df4.loc[i, score_col]) >= 0.92:
                c = b[0 + 6 * j + 1]
                try:
                    new_conf = float(df4.loc[i, score_col]) * 0.4 + float(c) * 0.6
                    b[0 + 6 * j + 1] = str(new_conf)
                except Exception:
                    pass

    df4.loc[i, "PredictionString"] = " ".join(b).strip()



## === cell 9
train_path = find_file("train.csv")
train_df = pd.read_csv(train_path)

train_df = train_df[
    pd.to_numeric(train_df["class_id"], errors="coerce").notnull()
].copy()
train_df["class_id"] = train_df["class_id"].astype(int)
train_df = train_df[(train_df["class_id"] >= 0) & (train_df["class_id"] <= 13)].copy()

img_stats = train_df.groupby("image_id").agg(
    img_w=("x_max", "max"),
    img_h=("y_max", "max"),
)
train_df = train_df.merge(img_stats, on="image_id", how="left")

train_df["img_w"] = train_df["img_w"].replace(0, np.nan)
train_df["img_h"] = train_df["img_h"].replace(0, np.nan)

for c in ["x_min", "x_max"]:
    train_df[f"{c}_n"] = (train_df[c] / train_df["img_w"]).clip(0, 1)
for c in ["y_min", "y_max"]:
    train_df[f"{c}_n"] = (train_df[c] / train_df["img_h"]).clip(0, 1)

typ = (
    train_df.groupby("class_id")
    .agg(
        x_min_n=("x_min_n", "median"),
        y_min_n=("y_min_n", "median"),
        x_max_n=("x_max_n", "median"),
        y_max_n=("y_max_n", "median"),
    )
    .reset_index()
)

all_cls = pd.DataFrame({"class_id": list(range(14))})
typ = all_cls.merge(typ, on="class_id", how="left")
typ = typ.fillna({"x_min_n": 0.35, "y_min_n": 0.35, "x_max_n": 0.65, "y_max_n": 0.65})

test_dir = None
for cand in DATA_DIR_CANDIDATES:
    td = os.path.join(cand, "test")
    if os.path.isdir(td):
        test_dir = td
        break
if test_dir is None:
    test_dir = ""


def _try_read_dicom_shape(path: str):
    try:
        import pydicom  # type: ignore
    except Exception:
        return None
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=False, force=True)
        arr = ds.pixel_array
        if arr is None:
            return None
        if arr.ndim == 2:
            h, w = arr.shape
        else:
            h, w = arr.shape[-2], arr.shape[-1]
        if int(h) <= 1 or int(w) <= 1:
            return None
        return int(w), int(h)
    except Exception:
        return None


DEFAULT_CANVAS = 1024
test_sizes = {}
if test_dir and os.path.isdir(test_dir):
    for image_id in sample_sub["image_id"].astype(str).tolist():
        p = os.path.join(test_dir, f"{image_id}.dicom")
        if os.path.exists(p):
            shp = _try_read_dicom_shape(p)
            if shp is not None:
                test_sizes[image_id] = shp

cls_counts = train_df["class_id"].value_counts().sort_index()
max_cnt = float(cls_counts.max()) if len(cls_counts) else 1.0
cls_conf = {}
for cid in range(14):
    cnt = float(cls_counts.get(cid, 0.0))
    cls_conf[cid] = float(0.10 + 0.12 * (cnt / max_cnt if max_cnt > 0 else 0.0))

PRED_CLASSES = [10, 3, 1]  # effusion, cardiomegaly, atelectasis
N_APPLY = df4.shape[0]

for i in range(N_APPLY):
    image_id = str(df4.loc[i, "image_id"])
    w, h = test_sizes.get(image_id, (DEFAULT_CANVAS, DEFAULT_CANVAS))

    preds = []
    for cid in PRED_CLASSES:
        r = typ.loc[typ["class_id"] == cid].iloc[0]
        x1 = int(round(float(r["x_min_n"]) * w))
        y1 = int(round(float(r["y_min_n"]) * h))
        x2 = int(round(float(r["x_max_n"]) * w))
        y2 = int(round(float(r["y_max_n"]) * h))

        x1 = int(np.clip(x1, 0, w - 1))
        x2 = int(np.clip(x2, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        y2 = int(np.clip(y2, 0, h - 1))
        if x2 <= x1:
            x2 = min(w - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(h - 1, y1 + 1)

        conf = cls_conf.get(cid, 0.15)
        preds.append(f"{cid} {conf} {x1} {y1} {x2} {y2}")
    df4.loc[i, "PredictionString"] = " ".join(preds)

df4["PredictionString"] = df4["PredictionString"].apply(
    lambda x: " ".join(str(x).split()).strip()
)




## === cell 10
def _sanitize_predstring(ps: str, canvas: int = 1024) -> str:
    ps = " ".join(str(ps).split()).strip()
    if ps == "":
        return "14 1 0 0 1 1"
    toks = ps.split()
    if len(toks) % 6 != 0:
        return "14 1 0 0 1 1"

    out = []
    for k in range(0, len(toks), 6):
        try:
            cid = int(float(toks[k]))
            conf = float(toks[k + 1])
            x1 = int(float(toks[k + 2]))
            y1 = int(float(toks[k + 3]))
            x2 = int(float(toks[k + 4]))
            y2 = int(float(toks[k + 5]))
        except Exception:
            continue

        if cid < 0 or cid > 14:
            continue

        if not np.isfinite(conf):
            continue
        conf = float(np.clip(conf, 1e-6, 1.0))

        x1 = int(np.clip(x1, 0, canvas - 1))
        y1 = int(np.clip(y1, 0, canvas - 1))
        x2 = int(np.clip(x2, 0, canvas - 1))
        y2 = int(np.clip(y2, 0, canvas - 1))
        if x2 <= x1:
            x2 = min(canvas - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(canvas - 1, y1 + 1)

        out.extend([str(cid), f"{conf:.6f}", str(x1), str(y1), str(x2), str(y2)])

    if len(out) == 0:
        return "14 1 0 0 1 1"
    if len(out) % 6 != 0:
        return "14 1 0 0 1 1"
    return " ".join(out)


df4["PredictionString"] = df4["PredictionString"].map(_sanitize_predstring)



## === cell 11
df4.loc[df4["PredictionString"].isna(), "PredictionString"] = "14 1 0 0 1 1"
df4["PredictionString"] = df4["PredictionString"].map(_sanitize_predstring)

df_final = df4[["image_id", "PredictionString"]].copy()
df_final = sample_sub[["image_id"]].merge(df_final, on="image_id", how="left")

df_final["PredictionString"] = (
    df_final["PredictionString"].fillna("14 1 0 0 1 1").astype(str)
)
df_final["PredictionString"] = df_final["PredictionString"].map(_sanitize_predstring)

submission = df_final[["image_id", "PredictionString"]].copy()
submission.to_csv("submission.csv", index=False)

assert submission.shape[0] == sample_sub.shape[0]
assert list(submission.columns) == ["image_id", "PredictionString"]
assert (
    submission["image_id"]
    .astype(str)
    .reset_index(drop=True)
    .equals(sample_sub["image_id"].astype(str).reset_index(drop=True))
)
assert (
    submission["PredictionString"]
    .map(lambda s: isinstance(s, str) and len(s.split()) % 6 == 0)
    .all()
)
assert submission["PredictionString"].map(lambda s: len(s.strip()) > 0).all()

submission.head()
