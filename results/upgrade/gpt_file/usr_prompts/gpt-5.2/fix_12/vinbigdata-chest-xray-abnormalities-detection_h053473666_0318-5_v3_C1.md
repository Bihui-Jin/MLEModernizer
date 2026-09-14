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


def compute_image_class_scores(train_df_: pd.DataFrame) -> pd.DataFrame:
    """
    Keep your exact smoothed per-image voting scores (core logic),
    but ensure index alignment between abnormal scores and p14.
    """
    t = train_df_.copy()

    rad_n = t.groupby("image_id")["rad_id"].nunique().rename("n_rad").clip(lower=1)

    abn = t.loc[
        t["class_id"] != 14, ["image_id", "class_id", "rad_id"]
    ].drop_duplicates()
    num = (
        abn.groupby(["image_id", "class_id"])["rad_id"]
        .nunique()
        .rename("n_pos")
        .reset_index()
    )

    pivot_num = num.pivot_table(
        index="image_id",
        columns="class_id",
        values="n_pos",
        aggfunc="max",
        fill_value=0.0,
    )
    for k in range(14):
        if k not in pivot_num.columns:
            pivot_num[k] = 0.0
    pivot_num = pivot_num[list(range(14))].astype(float)

    denom = rad_n.reindex(pivot_num.index).astype(float)

    alpha = 1.0
    frac = (pivot_num.add(alpha)).div(denom.values.reshape(-1, 1) + 2.0 * alpha)

    low, high = 0.03, 0.90
    scores = frac * (high - low) + low
    scores.columns = [str(c) for c in scores.columns]

    nofind_votes = (
        t.loc[t["class_id"] == 14, ["image_id", "rad_id"]]
        .drop_duplicates()
        .groupby("image_id")["rad_id"]
        .nunique()
        .rename("n_nf")
    )
    n_nf = nofind_votes.reindex(scores.index).fillna(0.0).astype(float)
    n_rad = rad_n.reindex(scores.index).fillna(1.0).astype(float)

    alpha14 = 1.0
    p14_frac = (n_nf + alpha14) / (n_rad + 2.0 * alpha14)
    p14 = (p14_frac * (high - low) + low).clip(0.03, 0.95)

    scores["14"] = p14

    return scores.reset_index()


def compute_image_class_box_prototypes(train_df_: pd.DataFrame) -> pd.DataFrame:
    cols = ["x_min", "y_min", "x_max", "y_max"]
    td = train_df_.copy()
    td = td.loc[td["class_id"] != 14].dropna(subset=["image_id", "class_id"] + cols)
    for c in cols:
        td[c] = pd.to_numeric(td[c], errors="coerce")
    td = td.dropna(subset=cols)
    td = td.loc[(td["x_max"] > td["x_min"]) & (td["y_max"] > td["y_min"])]

    proto = td.groupby(["image_id", "class_id"])[cols].median().reset_index()
    proto["class_id"] = proto["class_id"].astype(int)
    return proto


priors = compute_class_priors_per_image(train_df)
box_proto = compute_class_box_prototypes(train_df)
proto_map = box_proto.set_index("class_id")[
    ["x_min", "y_min", "x_max", "y_max"]
].to_dict("index")

img_scores = compute_image_class_scores(train_df)
img_boxes = compute_image_class_box_prototypes(train_df)

df = make_base_df(sample)
df1 = make_base_df(sample)
df_densenet = make_base_df(sample)

cols_0_14 = [str(i) for i in range(15)]

for k in cols_0_14:
    df[k] = priors.get(k, 0.0)
    df1[k] = priors.get(k, 0.0)
    df_densenet[k] = priors.get(k, 0.0)

df = df.merge(
    img_scores[["image_id"] + cols_0_14],
    on="image_id",
    how="left",
    suffixes=("", "_img"),
)
for k in cols_0_14:
    if f"{k}_img" in df.columns:
        df[k] = df[f"{k}_img"].fillna(df[k])
df = df.drop(columns=[c for c in df.columns if c.endswith("_img")])

df[cols_0_14] = (
    df[cols_0_14] * 0.25 + df1[cols_0_14] * 0.5 + df_densenet[cols_0_14] * 0.25
)

img_box_map = {}
for r in img_boxes.itertuples(index=False):
    img_box_map[(r.image_id, int(r.class_id))] = {
        "x_min": float(r.x_min),
        "y_min": float(r.y_min),
        "x_max": float(r.x_max),
        "y_max": float(r.y_max),
    }

K_BASE = 3
THR_BASE = 0.22

NOFIND_MARGIN_BASE = 0.02

pred_strings = []
for i in range(df.shape[0]):
    row = df.iloc[i]
    image_id = row["image_id"]
    p14 = float(row["14"]) if "14" in row.index else 0.0

    class_scores = [(k, float(row[str(k)])) for k in range(14)]
    class_scores.sort(key=lambda x: x[1], reverse=True)
    best_abn = class_scores[0][1] if len(class_scores) else 0.0
    second_abn = class_scores[1][1] if len(class_scores) > 1 else 0.0

    margin = NOFIND_MARGIN_BASE + (0.03 if p14 >= 0.6 else 0.0)
    if p14 >= best_abn + margin:
        pred_strings.append("14 1 0 0 1 1")
        continue

    gap12 = best_abn - second_abn
    if gap12 < 0.05:
        K = 1
        THR = max(THR_BASE, best_abn - 0.01)
    elif gap12 < 0.10:
        K = 2
        THR = THR_BASE + 0.03
    else:
        K = K_BASE
        THR = THR_BASE

    chosen = [(k, s) for (k, s) in class_scores[:K] if s >= THR]

    if len(chosen) == 0:
        pred_strings.append("14 1 0 0 1 1")
        continue

    parts = []
    for k, s in chosen:
        bb = img_box_map.get((image_id, int(k)))
        if bb is None:
            bb = proto_map.get(
                k, {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0}
            )
        conf = float(np.clip(s, 0.10, 0.95))
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
ps4 = df4["PredictionString"].astype(str).str.strip()
ps5 = df5["PredictionString"].astype(str).str.strip()
use5 = ps4.eq("14 1 0 0 1 1") & (~ps5.eq("14 1 0 0 1 1"))
df4.loc[use5, "PredictionString"] = ps5.loc[use5]
df4["PredictionString"] = df4["PredictionString"].astype(str).str.strip()



## === cell 7
list1 = list(range(14))  # 0..13 only

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
                if float(df4.loc[df4.index[i], f"{k}"]) < 0.75:
                    continue
                c = b[0 + 6 * j + 1]
                b[0 + 6 * j + 1] = str(
                    float(df4.loc[df4.index[i], f"{k}"]) * 0.4 + float(c) * 0.6
                )
    df4.loc[df4.index[i], "PredictionString"] = " ".join(b).strip()



## === cell 8
try:
    import pydicom  # available on Kaggle for this competition typically
except Exception:
    pydicom = None


def _dicom_path(image_id: str, is_train: bool) -> str:
    sub = "train" if is_train else "test"
    p = os.path.join(BASE_DIR, sub, f"{image_id}.dicom")
    if os.path.exists(p):
        return p
    p2 = os.path.join("/kaggle/data", sub, f"{image_id}.dicom")
    return p2


def dicom_brightness_feature(image_id: str, is_train: bool) -> float:
    if pydicom is None:
        return np.nan
    fp = _dicom_path(image_id, is_train=is_train)
    if not os.path.exists(fp):
        return np.nan
    try:
        ds = pydicom.dcmread(fp, stop_before_pixels=False, force=True)
        arr = ds.pixel_array.astype(np.float32)
        lo = np.percentile(arr, 1.0)
        hi = np.percentile(arr, 99.0)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return float(np.nan)
        x = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        return float(np.median(x))
    except Exception:
        return np.nan


train_img_ids = train_df["image_id"].drop_duplicates().tolist()
MAX_CALIB = 1200
train_img_ids_sub = train_img_ids[:MAX_CALIB]

train_bright = []
for iid in train_img_ids_sub:
    train_bright.append((iid, dicom_brightness_feature(iid, is_train=True)))
train_bright_df = pd.DataFrame(train_bright, columns=["image_id", "bright"]).dropna()

present = (
    train_df.loc[train_df["class_id"] != 14, ["image_id", "class_id"]]
    .drop_duplicates()
    .assign(present=1)
)
presence_pivot = present.pivot_table(
    index="image_id", columns="class_id", values="present", aggfunc="max", fill_value=0
)
for k in range(14):
    if k not in presence_pivot.columns:
        presence_pivot[k] = 0
presence_pivot = presence_pivot[list(range(14))].astype(int)
presence_pivot.columns = [str(c) for c in presence_pivot.columns]
presence_pivot = presence_pivot.reset_index()

train_cal = train_bright_df.merge(presence_pivot, on="image_id", how="left").fillna(0)

if train_cal.shape[0] >= 50:
    train_cal["q"] = pd.qcut(train_cal["bright"], q=5, duplicates="drop")
    q_means = train_cal.groupby("q")[cols_0_14[:-1]].mean()  # 0..13
    q_centers = train_cal.groupby("q")["bright"].mean()

    low, high = 0.03, 0.90
    q_scores = (q_means * (high - low) + low).clip(low, high)

    test_bright = []
    for iid in df4["image_id"].tolist():
        test_bright.append((iid, dicom_brightness_feature(iid, is_train=False)))
    test_bright_df = pd.DataFrame(test_bright, columns=["image_id", "bright"]).dropna()

    df4 = df4.merge(test_bright_df, on="image_id", how="left")

    def _nearest_q_score(b: float) -> pd.Series:
        if not np.isfinite(b) or q_centers.shape[0] == 0:
            return pd.Series({k: np.nan for k in cols_0_14[:-1]})
        idx = (q_centers - b).abs().values.argmin()
        return q_scores.iloc[idx]

    adj = df4["bright"].apply(_nearest_q_score)
    adj.columns = cols_0_14[:-1]
    adj = adj.astype(float)

    BLEND_W = 0.20
    for k in cols_0_14[:-1]:
        if k in df4.columns:
            df4[k] = np.where(
                adj[k].notna(),
                (1.0 - BLEND_W) * df4[k].astype(float) + BLEND_W * adj[k],
                df4[k].astype(float),
            )

    for i in range(df4.shape[0]):
        if "14" not in df4.columns:
            continue
        p14 = float(df4.loc[df4.index[i], "14"])
        best_abn = 0.0
        for kk in range(14):
            if str(kk) in df4.columns:
                v = float(df4.loc[df4.index[i], str(kk)])
                if v > best_abn:
                    best_abn = v
        margin = NOFIND_MARGIN_BASE + (0.03 if p14 >= 0.6 else 0.0)
        if p14 >= best_abn + margin:
            df4.loc[df4.index[i], "PredictionString"] = "14 1 0 0 1 1"

    df4 = df4.drop(columns=["bright"])
else:
    pass



## === cell 9
for i in range(df4.shape[0]):
    ps = str(df4.loc[df4.index[i], "PredictionString"]).strip()
    if ps == "" or ps.lower() == "nan" or ps == "14 1 0 0 1 1":
        df4.loc[df4.index[i], "PredictionString"] = "14 1 0 0 1 1"
        continue
    toks = ps.split()
    nb = len(toks) // 6
    if nb <= 2:
        continue
    boxes = []
    for j in range(nb):
        cls = toks[0 + 6 * j]
        conf = float(toks[1 + 6 * j])
        box = toks[2 + 6 * j : 6 + 6 * j]
        boxes.append((conf, cls, box))
    boxes.sort(reverse=True, key=lambda x: x[0])
    boxes = boxes[:2]
    out = []
    for conf, cls, box in boxes:
        out.extend([cls, str(conf)] + box)
    df4.loc[df4.index[i], "PredictionString"] = " ".join(out).strip()

for i in range(df4.shape[0]):
    if "14" in df4.columns:
        p14 = float(df4.loc[df4.index[i], "14"])
        best_abn = 0.0
        for k in range(14):
            if str(k) in df4.columns:
                v = float(df4.loc[df4.index[i], str(k)])
                if v > best_abn:
                    best_abn = v
        margin = NOFIND_MARGIN_BASE + (0.03 if p14 >= 0.6 else 0.0)
        if p14 >= best_abn + margin:
            df4.loc[df4.index[i], "PredictionString"] = "14 1 0 0 1 1"

df4["PredictionString"] = df4["PredictionString"].astype(str).str.strip()
df4.loc[df4["PredictionString"].eq(""), "PredictionString"] = "14 1 0 0 1 1"

df_final = df4[["image_id", "PredictionString"]].copy()
df_final.to_csv("submission.csv", index=False)



## === cell 10
df_final
