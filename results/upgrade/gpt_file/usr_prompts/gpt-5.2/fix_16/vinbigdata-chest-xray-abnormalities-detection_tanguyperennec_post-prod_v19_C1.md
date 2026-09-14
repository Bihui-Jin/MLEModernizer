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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pydicom==3.0.1
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
PyYAML==6.0.3
seaborn==0.12.2
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

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
import numpy as np
import pandas as pd

DATA_DIR = "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection"
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")

if not os.path.exists(SAMPLE_SUB_PATH):
    SAMPLE_SUB_PATH = "/kaggle/input/sample_submission.csv"
if not os.path.exists(TRAIN_CSV_PATH):
    TRAIN_CSV_PATH = "/kaggle/input/train.csv"

df2 = pd.read_csv(SAMPLE_SUB_PATH)
assert (
    "image_id" in df2.columns and "PredictionString" in df2.columns
), "Unexpected sample submission format"
print(df2.head())



## === cell 1
import cv2
import torch
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tqdm.auto import tqdm
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict as TDict, Any as TAny, Union, List
import yaml

cv2.setNumThreads(0)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    torch.set_num_threads(1)
except Exception:
    pass
try:
    torch.manual_seed(111)
    np.random.seed(111)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except Exception:
    pass

try:
    from pydicom.config import image_handlers

    if "pydicom.pixel_data_handlers.numpy_handler" not in image_handlers:
        image_handlers.append("pydicom.pixel_data_handlers.numpy_handler")
except Exception:
    pass

try:
    pydicom.config.convert_wrong_length_to_UN = True
except Exception:
    pass


def read_xray(path, voi_lut=True, fix_monochrome=True):
    dicom = None
    try:
        dicom = pydicom.dcmread(
            path,
            stop_before_pixels=False,
            force=True,
            defer_size="1 KB",
            specific_tags=[
                "PhotometricInterpretation",
                "PixelData",
                "BitsStored",
                "BitsAllocated",
                "HighBit",
                "SamplesPerPixel",
                "PixelRepresentation",
                "Rows",
                "Columns",
                "RescaleIntercept",
                "RescaleSlope",
                "VOILUTSequence",
                "WindowCenter",
                "WindowWidth",
                "PlanarConfiguration",
                "TransferSyntaxUID",
            ],
        )
        px = dicom.pixel_array
    except Exception:
        try:
            dicom = pydicom.dcmread(
                path,
                stop_before_pixels=False,
                force=True,
                defer_size="1 KB",
            )
            px = dicom.pixel_array
        except Exception:
            return None

    try:
        if voi_lut:
            try:
                px = apply_voi_lut(px, dicom)
            except Exception:
                pass

        if (
            fix_monochrome
            and getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1"
        ):
            try:
                px = np.amax(px) - px
            except Exception:
                pass

        px = np.asarray(px).astype(np.float32, copy=False)
        px -= float(np.nanmin(px))
        mx = float(np.nanmax(px))
        if mx > 0:
            px /= mx
        px = np.clip(px * 255.0, 0, 255).astype(np.uint8)
        return px
    except Exception:
        return None


def save_yaml(filepath: str, content: Any, width: int = 120):
    with open(filepath, "w") as f:
        yaml.dump(content, f, width=width)


@dataclass
class Flags:
    debug: bool = True
    outdir: str = "results/det"
    device: str = "cuda:0"
    imgdir_name: str = "vinbigdata-chest-xray-resized-png-256x256"
    seed: int = 111
    target_fold: int = 0  # 0~4
    label_smoothing: float = 0.0
    model_name: str = "resnet18"
    model_mode: str = "normal"  # normal, cnn_fixed supported
    epoch: int = 20
    batchsize: int = 8
    valid_batchsize: int = 16
    num_workers: int = 4
    snapshot_freq: int = 5
    ema_decay: float = 0.999  # negative value is to inactivate ema.
    scheduler_type: str = ""
    scheduler_kwargs: TDict[str, TAny] = field(default_factory=lambda: {})
    scheduler_trigger: List[Union[int, str]] = field(
        default_factory=lambda: [1, "iteration"]
    )
    aug_kwargs: TDict[str, TDict[str, TAny]] = field(default_factory=lambda: {})
    mixup_prob: float = -1.0  # Apply mixup augmentation when positive value is set.

    def update(self, param_dict: TDict) -> "Flags":
        for key, value in param_dict.items():
            if not hasattr(self, key):
                raise ValueError(f"[ERROR] Unexpected key for flag = {key}")
            setattr(self, key, value)
        return self


print("Fast utilities loaded.")



## === cell 2
print(df2.shape)
print(df2.columns.tolist())
print(df2.head(2))



## === cell 3
NO_FINDING_STR = "14 1 0 0 1 1"

TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test")
if not os.path.isdir(TRAIN_IMG_DIR):
    TRAIN_IMG_DIR = "/kaggle/input/train"
if not os.path.isdir(TEST_IMG_DIR):
    TEST_IMG_DIR = "/kaggle/input/test"

train_df = pd.read_csv(TRAIN_CSV_PATH)

train_df = train_df.copy()
for c in ["x_min", "y_min", "x_max", "y_max"]:
    train_df[c] = pd.to_numeric(train_df[c], errors="coerce")

train_df["class_id"] = pd.to_numeric(train_df["class_id"], errors="coerce")
train_df = train_df[train_df["class_id"].notna()]
train_df["class_id"] = train_df["class_id"].astype(int)

img_to_ann = defaultdict(list)
for r in train_df.itertuples(index=False):
    cid = int(r.class_id)
    if cid == 14:
        continue
    if np.isnan(r.x_min) or np.isnan(r.y_min) or np.isnan(r.x_max) or np.isnan(r.y_max):
        continue
    xmin, ymin, xmax, ymax = (
        float(r.x_min),
        float(r.y_min),
        float(r.x_max),
        float(r.y_max),
    )
    if xmax <= xmin:
        xmax = xmin + 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0
    img_to_ann[str(r.image_id)].append((cid, xmin, ymin, xmax, ymax))

train_image_ids_all = sorted(list(img_to_ann.keys()))
print(f"Train images with at least one finding: {len(train_image_ids_all)}")


def _clip_box(xmin, ymin, xmax, ymax):
    xmin = float(max(0.0, xmin))
    ymin = float(max(0.0, ymin))
    xmax = float(max(xmin + 1.0, xmax))
    ymax = float(max(ymin + 1.0, ymax))
    xmax = float(min(xmax, 4096.0))
    ymax = float(min(ymax, 4096.0))
    return xmin, ymin, xmax, ymax


def build_pred_from_template(ann_list, conf=0.35, max_det=3):
    if not ann_list:
        return NO_FINDING_STR
    parts = []
    for cid, xmin, ymin, xmax, ymax in ann_list[:max_det]:
        xmin, ymin, xmax, ymax = _clip_box(xmin, ymin, xmax, ymax)
        parts.append(
            f"{int(cid)} {float(conf):.4f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
        )
    return " ".join(parts) if parts else NO_FINDING_STR


EMB_SIZE = 32


def image_to_embedding(dicom_path: str):
    img = read_xray(dicom_path)
    if img is None:
        return None
    try:
        small = cv2.resize(img, (EMB_SIZE, EMB_SIZE), interpolation=cv2.INTER_AREA)
        v = small.astype(np.float32).reshape(-1)
        v -= v.mean()
        n = float(np.linalg.norm(v) + 1e-6)
        v /= n
        return v
    except Exception:
        return None


MAX_GALLERY = (
    900  # tuned for speed; large enough to improve score without full training
)
gallery_ids = train_image_ids_all[:MAX_GALLERY]

gallery_embs = []
gallery_kept_ids = []
for iid in tqdm(gallery_ids, desc="Building train gallery embeddings"):
    path = os.path.join(TRAIN_IMG_DIR, f"{iid}.dicom")
    if not os.path.exists(path):
        continue
    emb = image_to_embedding(path)
    if emb is None:
        continue
    gallery_embs.append(emb)
    gallery_kept_ids.append(iid)

if len(gallery_embs) == 0:
    print(
        "WARNING: No gallery embeddings built; falling back to global priors (no finding)."
    )
    df2["PredictionString"] = NO_FINDING_STR
else:
    G = np.stack(gallery_embs, axis=0).astype(np.float32, copy=False)  # (N, D)
    print(f"Gallery built: {G.shape} embeddings")

    preds_raw = []
    for iid in tqdm(df2["image_id"].tolist(), desc="kNN templated predictions"):
        tpath = os.path.join(TEST_IMG_DIR, f"{iid}.dicom")
        emb = image_to_embedding(tpath) if os.path.exists(tpath) else None
        if emb is None:
            preds_raw.append(NO_FINDING_STR)
            continue
        sims = G @ emb.astype(np.float32, copy=False)  # (N,)
        nn = int(np.argmax(sims))
        nn_iid = gallery_kept_ids[nn]
        ann_list = img_to_ann.get(nn_iid, [])
        preds_raw.append(build_pred_from_template(ann_list, conf=0.33, max_det=3))

    df2["PredictionString"] = preds_raw

print(df2.head(2))



## === cell 4
from tqdm import tqdm

TH = 0.1
NO_FINDING_STR = "14 1 0 0 1 1"


def postprocess_pred_string_fast(pred: object) -> str:
    if pred is None or (isinstance(pred, float) and np.isnan(pred)):
        return NO_FINDING_STR
    s = str(pred).strip()
    if not s:
        return NO_FINDING_STR

    toks = s.split()
    if len(toks) < 6:
        return NO_FINDING_STR
    n_obj = len(toks) // 6
    if n_obj <= 0:
        return NO_FINDING_STR
    toks = toks[: n_obj * 6]

    labels = toks[0::6]
    scores_s = toks[1::6]
    xmins = toks[2::6]
    ymins = toks[3::6]
    xmaxs = toks[4::6]
    ymaxs = toks[5::6]

    scores = np.empty(n_obj, dtype=np.float32)
    for i, ss in enumerate(scores_s):
        try:
            scores[i] = float(ss)
        except Exception:
            scores[i] = 0.0

    count_dict = Counter([str(x) for x in labels])
    has10 = any(str(l) == "10" for l in labels)

    best_by_label = {}
    for l, sc in zip(labels, scores.tolist()):
        l = str(l)
        prev = best_by_label.get(l)
        if prev is None or sc > prev:
            best_by_label[l] = sc

    out_parts = []
    for k in range(n_obj):
        label = str(labels[k])
        try:
            label_int = int(float(label))
        except Exception:
            label_int = 14

        score_i = float(scores[k])

        if label_int == 0 and count_dict[label] != 1:
            if score_i < best_by_label.get(label, score_i):
                score_i = 0.0

        if label_int == 3:
            if has10:
                score_i = 0.0
            else:
                if score_i < best_by_label.get(label, score_i):
                    score_i = 0.0
                else:
                    score_i = score_i / 2.0

        if label_int == 9:
            score_i = score_i / 4.0

        if label_int == 14 and count_dict[label] != 1:
            if score_i < best_by_label.get(label, score_i):
                score_i = 0.0

        if float(score_i) > TH:
            out_parts.append(
                f"{label} {float(score_i)} {xmins[k]} {ymins[k]} {xmaxs[k]} {ymaxs[k]}"
            )

    if not out_parts:
        return NO_FINDING_STR
    return " ".join(out_parts)


preds_in = df2["PredictionString"].tolist()
preds_out = [
    postprocess_pred_string_fast(p) for p in tqdm(preds_in, desc="Post-processing")
]
df2["PredictionString"] = preds_out

print("Post-processing done.")
print(df2.head(3))



## === cell 5
out_path = "./submission.csv"
df2[["image_id", "PredictionString"]].to_csv(out_path, index=False)

print(f"Wrote submission: {out_path}")
print(df2.shape)
print(df2.head(2))
