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
import time
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

T0 = time.time()
TIME_LIMIT_SEC = 600
SOFT_CUTOFF_SEC = 560  # leave time for final formatting+I/O


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
    t = train_df_

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
    td = train_df_
    td = td.loc[td["class_id"] != 14].dropna(subset=["image_id", "class_id"] + cols)
    td = td[["image_id", "class_id"] + cols].copy()
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
df1 = df.copy(deep=True)
df_densenet = df.copy(deep=True)

cols_0_14 = [str(i) for i in range(15)]

for k in cols_0_14:
    v = priors.get(k, 0.0)
    df[k] = v
    df1[k] = v
    df_densenet[k] = v

df = df.merge(
    img_scores[["image_id"] + cols_0_14],
    on="image_id",
    how="left",
    suffixes=("", "_img"),
)
for k in cols_0_14:
    cimg = f"{k}_img"
    if cimg in df.columns:
        df[k] = df[cimg].fillna(df[k])
df = df.drop(columns=[c for c in df.columns if c.endswith("_img")])

df[cols_0_14] = (
    df[cols_0_14] * 0.25 + df1[cols_0_14] * 0.5 + df_densenet[cols_0_14] * 0.25
)

_img_boxes_key = list(
    zip(img_boxes["image_id"].tolist(), img_boxes["class_id"].astype(int).tolist())
)
_img_boxes_val = list(
    zip(
        img_boxes["x_min"].astype(float).tolist(),
        img_boxes["y_min"].astype(float).tolist(),
        img_boxes["x_max"].astype(float).tolist(),
        img_boxes["y_max"].astype(float).tolist(),
    )
)
img_box_map = dict(zip(_img_boxes_key, _img_boxes_val))

K_BASE = 3
THR_BASE = 0.22
NOFIND_MARGIN_BASE = 0.02

score_cols_0_13 = [str(k) for k in range(14)]
score_mat = df[score_cols_0_13].to_numpy(dtype=np.float32, copy=False)
p14_arr = df["14"].to_numpy(dtype=np.float32, copy=False)
img_ids = df["image_id"].to_numpy()

pred_strings = []
for i, image_id in enumerate(img_ids):
    row_scores = score_mat[i]
    p14 = float(p14_arr[i])

    order = np.argsort(-row_scores)
    best_k = int(order[0])
    best_abn = float(row_scores[best_k])
    second_abn = float(row_scores[order[1]]) if order.size > 1 else 0.0

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

    chosen_idx = []
    for kk in order[:K]:
        if float(row_scores[kk]) >= THR:
            chosen_idx.append(int(kk))
    if not chosen_idx:
        pred_strings.append("14 1 0 0 1 1")
        continue

    parts = []
    for k in chosen_idx:
        s = float(row_scores[k])
        bb = img_box_map.get((image_id, int(k)))
        if bb is None:
            bb0 = proto_map.get(
                k, {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0}
            )
            bb = (
                float(bb0["x_min"]),
                float(bb0["y_min"]),
                float(bb0["x_max"]),
                float(bb0["y_max"]),
            )
        conf = float(np.clip(s, 0.10, 0.95))
        parts.extend(
            [
                str(int(k)),
                f"{conf:.6f}",
                f"{bb[0]:.2f}",
                f"{bb[1]:.2f}",
                f"{bb[2]:.2f}",
                f"{bb[3]:.2f}",
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
_ps3_map = df3.set_index("image_id")["PredictionString"]
df4 = df.copy()
df4["PredictionString"] = df4["PredictionString"].fillna(df4["image_id"].map(_ps3_map))
df4["PredictionString"] = df4["PredictionString"].fillna("14 1 0 0 1 1").astype(str)




## === cell 3
def _strip_class14(ps: str) -> str:
    ps = str(ps).strip()
    if ps == "" or ps.lower() == "nan":
        return "14 1 0 0 1 1"
    toks = ps.split()
    nb = len(toks) // 6
    if nb == 0:
        return "14 1 0 0 1 1"
    out = []
    for j in range(nb):
        cls = toks[6 * j]
        if cls == "14":
            continue
        out.extend(toks[6 * j : 6 * j + 6])
    return " ".join(out).strip() if out else "14 1 0 0 1 1"


df4["PredictionString"] = df4["PredictionString"].map(_strip_class14).astype(str)
df4["PredictionString"] = df4["PredictionString"].astype(str).str.split().str.join(" ")



## === cell 4
val = None
if df4.shape[0] > 1 and df4.shape[1] > 16:
    val = df4.iloc[1, 16]
val



## === cell 5
_ps2_map = df2.set_index("image_id")["PredictionString"]
df5 = df.copy()
df5["PredictionString"] = df5["PredictionString"].fillna(df5["image_id"].map(_ps2_map))
df5["PredictionString"] = df5["PredictionString"].fillna("14 1 0 0 1 1").astype(str)



## === cell 6
ps4 = df4["PredictionString"].astype(str).str.strip()
ps5 = df5["PredictionString"].astype(str).str.strip()
use5 = ps4.eq("14 1 0 0 1 1") & (~ps5.eq("14 1 0 0 1 1"))
df4.loc[use5, "PredictionString"] = ps5.loc[use5]
df4["PredictionString"] = df4["PredictionString"].astype(str).str.strip()



## === cell 7
list1 = list(range(14))  # 0..13 only

score_cols = [str(k) for k in range(14)]
row_scores_df = df4[["image_id"] + score_cols].set_index("image_id")
row_scores_np = row_scores_df.to_numpy(dtype=np.float32, copy=False)
row_scores_index = row_scores_df.index.to_numpy()
row_scores_pos = {iid: i for i, iid in enumerate(row_scores_index)}


def _blend_conf(ps: str, image_id: str) -> str:
    if str(ps).strip() == "14 1 0 0 1 1":
        return "14 1 0 0 1 1"
    b = str(ps).split()
    nb = len(b) // 6
    if nb == 0:
        return "14 1 0 0 1 1"
    idx = row_scores_pos.get(image_id)
    if idx is None:
        return " ".join(b).strip()
    scores = row_scores_np[idx]
    for j in range(nb):
        cls = int(float(b[6 * j]))
        if cls in list1:
            sv = float(scores[cls])
            if sv >= 0.75:
                c = float(b[6 * j + 1])
                b[6 * j + 1] = str(sv * 0.4 + c * 0.6)
    return " ".join(b).strip()


df4["PredictionString"] = [
    _blend_conf(ps, iid)
    for ps, iid in zip(df4["PredictionString"].to_numpy(), df4["image_id"].to_numpy())
]



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
    return os.path.join("/kaggle/data", sub, f"{image_id}.dicom")


_bright_cache = {}


def _hist_percentile_uint(arr_u: np.ndarray, q: float) -> float:
    if arr_u.size == 0:
        return float("nan")
    maxv = int(arr_u.max())
    hist = np.bincount(arr_u, minlength=maxv + 1)
    cdf = np.cumsum(hist)
    target = q * (arr_u.size - 1)
    idx = int(np.searchsorted(cdf, target, side="right"))
    if idx <= 0:
        return 0.0
    if idx >= cdf.size:
        return float(cdf.size - 1)
    prev = cdf[idx - 1]
    curr = cdf[idx]
    if curr == prev:
        return float(idx)
    frac = (target - prev) / (curr - prev)
    return float((idx - 1) + frac)


def dicom_brightness_feature(image_id: str, is_train: bool) -> float:
    key = ("tr" if is_train else "te", image_id)
    if key in _bright_cache:
        return _bright_cache[key]
    if pydicom is None:
        _bright_cache[key] = np.nan
        return np.nan
    fp = _dicom_path(image_id, is_train=is_train)
    if not os.path.exists(fp):
        _bright_cache[key] = np.nan
        return np.nan
    try:
        ds = pydicom.dcmread(fp, stop_before_pixels=False, force=True)
        arr = ds.pixel_array
        if arr is None:
            _bright_cache[key] = np.nan
            return np.nan
        if np.issubdtype(arr.dtype, np.integer):
            arr_u = np.asarray(arr, dtype=np.uint16).ravel()
            lo = _hist_percentile_uint(arr_u, 0.01)
            hi = _hist_percentile_uint(arr_u, 0.99)
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                _bright_cache[key] = float("nan")
                return float("nan")
            x = (arr_u.astype(np.float32) - float(lo)) / float(hi - lo)
            x = np.clip(x, 0.0, 1.0)
            m = float(np.median(x))
            _bright_cache[key] = m
            return m
        else:
            arrf = np.asarray(arr, dtype=np.float32)
            lo = np.percentile(arrf, 1.0)
            hi = np.percentile(arrf, 99.0)
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                _bright_cache[key] = float("nan")
                return float("nan")
            x = np.clip((arrf - lo) / (hi - lo), 0.0, 1.0)
            m = float(np.median(x))
            _bright_cache[key] = m
            return m
    except Exception:
        _bright_cache[key] = np.nan
        return np.nan


train_img_ids = train_df["image_id"].drop_duplicates().to_numpy()
MAX_CALIB = 1200
train_img_ids_sub = train_img_ids[:MAX_CALIB]

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

if (time.time() - T0) < SOFT_CUTOFF_SEC:
    train_bright = [
        (iid, dicom_brightness_feature(str(iid), is_train=True))
        for iid in train_img_ids_sub
    ]
    train_bright_df = pd.DataFrame(
        train_bright, columns=["image_id", "bright"]
    ).dropna()

    train_cal = train_bright_df.merge(presence_pivot, on="image_id", how="left").fillna(
        0
    )

    if train_cal.shape[0] >= 50 and (time.time() - T0) < SOFT_CUTOFF_SEC:
        train_cal["q"] = pd.qcut(train_cal["bright"], q=5, duplicates="drop")
        q_means = train_cal.groupby("q")[cols_0_14[:-1]].mean()  # 0..13
        q_centers = train_cal.groupby("q")["bright"].mean()

        low, high = 0.03, 0.90
        q_scores = (q_means * (high - low) + low).clip(low, high)

        test_ids_unique = df4["image_id"].drop_duplicates().to_numpy()
        if (time.time() - T0) < SOFT_CUTOFF_SEC:
            test_bright = [
                (iid, dicom_brightness_feature(str(iid), is_train=False))
                for iid in test_ids_unique
            ]
            test_bright_df = pd.DataFrame(
                test_bright, columns=["image_id", "bright"]
            ).dropna()

            _test_bright_map = test_bright_df.set_index("image_id")["bright"]
            df4["bright"] = df4["image_id"].map(_test_bright_map)

            centers = q_centers.values.astype(np.float32)
            qs = q_scores.values.astype(np.float32)  # shape (nq, 14)
            bright = df4["bright"].to_numpy(dtype=np.float32, copy=False)

            adj_mat = np.full((df4.shape[0], 14), np.nan, dtype=np.float32)
            finite_mask = np.isfinite(bright) & (centers.size > 0)
            if centers.size > 0 and finite_mask.any():
                d = np.abs(bright[finite_mask, None] - centers[None, :])
                idx = d.argmin(axis=1)
                adj_mat[finite_mask] = qs[idx]

            adj = pd.DataFrame(adj_mat, columns=cols_0_14[:-1], index=df4.index)

            BLEND_W = 0.20
            for k in cols_0_14[:-1]:
                if k in df4.columns:
                    basev = df4[k].astype(float)
                    df4[k] = np.where(
                        adj[k].notna(),
                        (1.0 - BLEND_W) * basev + BLEND_W * adj[k].astype(float),
                        basev,
                    )

            if "14" in df4.columns:
                p14v = df4["14"].astype(float).values
                abn_max = (
                    df4[[str(k) for k in range(14)]].astype(float).max(axis=1).values
                )
                margin = NOFIND_MARGIN_BASE + np.where(p14v >= 0.6, 0.03, 0.0)
                nf_mask = p14v >= (abn_max + margin)
                df4.loc[nf_mask, "PredictionString"] = "14 1 0 0 1 1"

            df4 = df4.drop(columns=["bright"])




## === cell 9
def _cap_to_top2(ps: str) -> str:
    ps = str(ps).strip()
    if ps == "" or ps.lower() == "nan" or ps == "14 1 0 0 1 1":
        return "14 1 0 0 1 1"
    toks = ps.split()
    nb = len(toks) // 6
    if nb <= 2:
        return " ".join(toks).strip()
    boxes = []
    for j in range(nb):
        cls = toks[6 * j]
        conf = float(toks[6 * j + 1])
        box = toks[6 * j + 2 : 6 * j + 6]
        boxes.append((conf, cls, box))
    boxes.sort(reverse=True, key=lambda x: x[0])
    boxes = boxes[:2]
    out = []
    for conf, cls, box in boxes:
        out.extend([cls, str(conf)] + box)
    return " ".join(out).strip()


df4["PredictionString"] = df4["PredictionString"].map(_cap_to_top2)

if "14" in df4.columns:
    p14v = df4["14"].astype(float).values
    abn_max = df4[[str(k) for k in range(14)]].astype(float).max(axis=1).values
    margin = NOFIND_MARGIN_BASE + np.where(p14v >= 0.6, 0.03, 0.0)
    nf_mask = p14v >= (abn_max + margin)
    df4.loc[nf_mask, "PredictionString"] = "14 1 0 0 1 1"

df4["PredictionString"] = df4["PredictionString"].astype(str).str.strip()
df4.loc[df4["PredictionString"].eq(""), "PredictionString"] = "14 1 0 0 1 1"

df_final = df4[["image_id", "PredictionString"]].copy()
df_final.to_csv("submission.csv", index=False)



## === cell 10
df_final
