# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

# 5. Target score

0.2121243007974857

# 6. Current score

0.0475

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0475) has done: 'I remove the failing reads of non-existent `../input/.../submission.csv` files and instead base the pipeline on the provided `sample_submission.csv`, which guarantees the right IDs and format. I keep your existing “post-processing of PredictionString” core logic, but make it robust to empty strings / NaNs and ensure every row outputs a valid fallback `14 1 0 0 1 1` when nothing passes the threshold. I also fix multiple NameErrors and a bug in `predict_bbox` (uses undefined variable `image`) so the script can run end-to-end without interactive-only visualization cells. Finally, I write `submission.csv` with the required columns (`image_id`, `PredictionString`) to the working directory.'
- What this solution (achieved 0.00957) has done: 'Your current pipeline is only post-processing the `sample_submission.csv`, which contains the default “No finding” for every image, so it cannot reach the target mAP regardless of thresholds. To move the score toward your target with minimal, metric-aligned change, I generate non-trivial predictions from the provided `train.csv` by building simple class-wise “priors” of typical box locations/sizes and frequencies, then emit a small number of plausible boxes per test image (plus the required fallback when empty). This preserves your existing core “PredictionString parsing + score filtering” semantics by keeping that stage intact and simply feeding it a more informative initial `PredictionString`. I also keep everything deterministic and fast (no training loops), and ensure the output CSV schema remains exactly `image_id,PredictionString`.'
- What this solution (achieved 0.0475) has done: 'I remove the largest bottleneck: repeatedly decoding full DICOMs and resizing them in Python loops for every train candidate and every test image. The core kNN retrieval logic stays identical, but embeddings are computed much faster by reading only the pixel data (skipping DICOM metadata) and using an optimized OpenCV resize + vectorized normalize, plus batching the similarity computation where possible. I also eliminate slow per-row pandas `.loc` assignments in the post-processing cell by parsing and processing prediction strings in pure Python and then assigning once, preserving exactly the same scoring rules and thresholding semantics. Finally, I cut unnecessary imports/plot utilities from the hot path and enforce deterministic threading settings to avoid overhead variability.'

# 9. Code solution

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


def read_xray(path, voi_lut=True, fix_monochrome=True):
    dicom = pydicom.dcmread(
        path,
        stop_before_pixels=False,
        force=True,
        specific_tags=[
            "PhotometricInterpretation",
            "PixelData",
            "BitsStored",
            "RescaleIntercept",
            "RescaleSlope",
            "VOILUTSequence",
            "WindowCenter",
            "WindowWidth",
        ],
    )
    px = dicom.pixel_array
    if voi_lut:
        px = apply_voi_lut(px, dicom)
    if (
        fix_monochrome
        and getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1"
    ):
        px = np.amax(px) - px
    px = px.astype(np.float32, copy=False)
    px -= float(px.min())
    mx = float(px.max())
    if mx > 0:
        px /= mx
    px = (px * 255.0).astype(np.uint8)
    return px


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

train_df = pd.read_csv(TRAIN_CSV_PATH)

TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test")
if not os.path.isdir(TRAIN_IMG_DIR):
    TRAIN_IMG_DIR = "/kaggle/input/train"
if not os.path.isdir(TEST_IMG_DIR):
    TEST_IMG_DIR = "/kaggle/input/test"

findings = train_df[train_df["class_id"].between(0, 13)].copy()
for c in ["x_min", "y_min", "x_max", "y_max", "class_id"]:
    findings[c] = pd.to_numeric(findings[c], errors="coerce")
findings = findings.dropna(
    subset=["image_id", "x_min", "y_min", "x_max", "y_max", "class_id"]
)
findings = findings[
    (findings["x_max"] > findings["x_min"]) & (findings["y_max"] > findings["y_min"])
]

img_to_boxes = defaultdict(list)
for r in findings.itertuples(index=False):
    img_to_boxes[str(r.image_id)].append(
        (
            int(r.class_id),
            float(r.x_min),
            float(r.y_min),
            float(r.x_max),
            float(r.y_max),
        )
    )

train_img_ids = np.array(sorted(img_to_boxes.keys()))
if len(train_img_ids) == 0:
    df2["PredictionString"] = NO_FINDING_STR
    print("No train findings found; falling back to all 'No finding'.")
else:
    def embed_dicom(path: str, out_hw: int = 32) -> np.ndarray:
        arr = read_xray(path)
        arr = cv2.resize(arr, (out_hw, out_hw), interpolation=cv2.INTER_AREA)
        v = arr.reshape(-1).astype(np.float32) * (1.0 / 255.0)
        v -= float(v.mean())
        n = float(np.linalg.norm(v) + 1e-6)
        return v / n

    MAX_TRAIN_CAND = 4000
    if len(train_img_ids) > MAX_TRAIN_CAND:
        step = int(np.ceil(len(train_img_ids) / MAX_TRAIN_CAND))
        train_cand_ids = train_img_ids[::step][:MAX_TRAIN_CAND]
    else:
        train_cand_ids = train_img_ids

    train_embs = np.zeros((len(train_cand_ids), 32 * 32), dtype=np.float32)
    missing_train = 0
    for i, iid in enumerate(tqdm(train_cand_ids, desc="Embedding train candidates")):
        p = os.path.join(TRAIN_IMG_DIR, f"{iid}.dicom")
        if not os.path.exists(p):
            missing_train += 1
            continue
        train_embs[i] = embed_dicom(p, out_hw=32)

    KNN = 5
    base_conf = 0.26
    neighbor_decay = 0.03
    max_boxes_per_image = 6  # keep small to reduce false positives

    all_train_zero = bool(np.all(train_embs == 0))

    pred_strings = []
    test_ids = df2["image_id"].astype(str).tolist()
    for test_id in tqdm(test_ids, desc="Building test predictions"):
        test_path = os.path.join(TEST_IMG_DIR, f"{test_id}.dicom")
        if not os.path.exists(test_path):
            pred_strings.append(NO_FINDING_STR)
            continue

        q = embed_dicom(test_path, out_hw=32)  # (1024,)
        sims = train_embs @ q  # (N,)

        if all_train_zero or np.all(sims == 0):
            pred_strings.append(NO_FINDING_STR)
            continue

        k_eff = min(KNN, len(sims))
        knn_idx = np.argpartition(-sims, k_eff - 1)[:k_eff]
        knn_idx = knn_idx[np.argsort(-sims[knn_idx])]

        box_candidates = []
        for rank, idx in enumerate(knn_idx):
            idx_i = int(idx)
            neighbor_id = str(train_cand_ids[idx_i])
            sim = float(sims[idx_i])
            for cid, x1, y1, x2, y2 in img_to_boxes.get(neighbor_id, []):
                conf = base_conf + 0.08 * max(0.0, sim) - rank * neighbor_decay
                box_candidates.append((cid, conf, x1, y1, x2, y2))

        if not box_candidates:
            pred_strings.append(NO_FINDING_STR)
            continue

        box_candidates.sort(key=lambda t: t[1], reverse=True)
        per_class_cap = 2
        per_class_count = Counter()
        parts = []
        for cid, conf, x1, y1, x2, y2 in box_candidates:
            if len(parts) >= max_boxes_per_image:
                break
            if per_class_count[cid] >= per_class_cap:
                continue
            per_class_count[cid] += 1

            x1i, y1i, x2i, y2i = (
                int(round(x1)),
                int(round(y1)),
                int(round(x2)),
                int(round(y2)),
            )
            if x2i <= x1i:
                x2i = x1i + 1
            if y2i <= y1i:
                y2i = y1i + 1

            conf = float(max(0.01, min(0.99, conf)))
            parts.append(f"{cid} {conf} {x1i} {y1i} {x2i} {y2i}")

        pred_strings.append(" ".join(parts) if parts else NO_FINDING_STR)

    df2["PredictionString"] = pred_strings
    print(
        "Built image-specific initial PredictionString via kNN retrieval from train GT boxes."
    )
    print(df2.head(2))




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/2504452087.py in <cell line: 0>()
     61             missing_train += 1
     62             continue
---> 63         train_embs[i] = embed_dicom(p, out_hw=32)
     64 
     65     KNN = 5

/tmp/ipykernel_55/2504452087.py in embed_dicom(path, out_hw)
     40     # --- Optimization: faster embedding using OpenCV resize + vectorized normalization; identical math.
     41     def embed_dicom(path: str, out_hw: int = 32) -> np.ndarray:
---> 42         arr = read_xray(path)
     43         arr = cv2.resize(arr, (out_hw, out_hw), interpolation=cv2.INTER_AREA)
     44         v = arr.reshape(-1).astype(np.float32) * (1.0 / 255.0)

/tmp/ipykernel_55/1147598673.py in read_xray(path, voi_lut, fix_monochrome)
     40         ],
     41     )
---> 42     px = dicom.pixel_array
     43     if voi_lut:
     44         px = apply_voi_lut(px, dicom)

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in __getattr__(self, name)
    916             return {}
    917         # Try the base class attribute getter (fix for issue 332)
--> 918         return object.__getattribute__(self, name)
    919 
    920     @property

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in pixel_array(self)
   2191             that iterates through the image frames.
   2192         """
-> 2193         self.convert_pixel_data()
   2194         return cast("numpy.ndarray", self._pixel_array)
   2195 

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in convert_pixel_data(self, handler_name)
   1724             # Use 'pydicom.pixels' backend
   1725             opts["decoding_plugin"] = name
-> 1726             self._pixel_array = pixel_array(self, **opts)
   1727             self._pixel_id = get_image_pixel_ids(self)
   1728         else:

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/utils.py in pixel_array(src, ds_out, specific_tags, index, raw, decoding_plugin, **kwargs)
   1428 
   1429         opts = as_pixel_options(ds, **kwargs)
-> 1430         return decoder.as_array(
   1431             ds,
   1432             index=index,

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py in as_array(self, src, index, validate, raw, decoding_plugin, **kwargs)
    988 
    989         if validate:
--> 990             runner.validate()
    991 
    992         if self.is_native:

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py in validate(self)
    751     def validate(self) -> None:
    752         """Validate the decoding options and source buffer (if any)."""
--> 753         self._validate_options()
    754         if self.is_dataset or self.is_buffer:
    755             self._validate_buffer()

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py in _validate_options(self)
    820     def _validate_options(self) -> None:
    821         """Validate the supplied options to ensure they meet minimum requirements."""
--> 822         super()._validate_options()
    823 
    824         # The Extended Offset Table is optional

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/common.py in _validate_options(self)
    573         prefix = "Missing required element: (0028"
    574         if self._opts.get("bits_allocated") is None:
--> 575             raise AttributeError(f"{prefix},0100) 'Bits Allocated'")
    576 
    577         if not 1 <= self.bits_allocated <= 64 or (

AttributeError: Missing required element: (0028,0100) 'Bits Allocated'

## === cell 4
from tqdm import tqdm

TH = 0.1
NO_FINDING_STR = "14 1 0 0 1 1"


def postprocess_pred_string(pred: object) -> str:
    if (
        pred is None
        or (isinstance(pred, float) and np.isnan(pred))
        or str(pred).strip() == ""
    ):
        return NO_FINDING_STR

    splited = str(pred).strip().split(" ")
    if len(splited) < 6:
        return NO_FINDING_STR

    n_obj = len(splited) // 6
    labels = [splited[i * 6] for i in range(n_obj)]
    scores = []
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    for i in range(n_obj):
        scores.append(splited[i * 6 + 1])
        xmins.append(splited[i * 6 + 2])
        ymins.append(splited[i * 6 + 3])
        xmaxs.append(splited[i * 6 + 4])
        ymaxs.append(splited[i * 6 + 5])

    score_f = []
    for s in scores:
        try:
            score_f.append(float(s))
        except Exception:
            score_f.append(0.0)

    count_dict = Counter([str(x) for x in labels])
    has10 = any(str(l) == "10" for l in labels)

    best_by_label = {}
    for l, s in zip(labels, score_f):
        l = str(l)
        prev = best_by_label.get(l)
        if prev is None or s > prev:
            best_by_label[l] = s

    out_parts = []
    for k in range(n_obj):
        label = str(labels[k])
        try:
            label_int = int(float(label))
        except Exception:
            label_int = 14

        score_i = float(score_f[k])

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
preds_out = [postprocess_pred_string(p) for p in tqdm(preds_in, desc="Post-processing")]
df2["PredictionString"] = preds_out

print("Post-processing done.")
print(df2.head(3))




## === cell 5
out_path = "./submission.csv"
df2[["image_id", "PredictionString"]].to_csv(out_path, index=False)

print(f"Wrote submission: {out_path}")
print(df2.shape)
print(df2.head(2))
