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
import os
from PIL import Image
from tqdm.auto import tqdm
from collections import Counter
from typing import Any, Dict
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import yaml
from dataclasses import dataclass, field
from typing import Dict as TDict, Any as TAny, Union, List


def read_xray(path, voi_lut=True, fix_monochrome=True):
    dicom = pydicom.dcmread(path)
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
    if (
        fix_monochrome
        and getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1"
    ):
        data = np.amax(data) - data
    data = data - np.min(data)
    mx = np.max(data)
    if mx > 0:
        data = data / mx
    data = (data * 255).astype(np.uint8)
    return data


def resize(array, size, keep_ratio=False, resample=Image.LANCZOS):
    im = Image.fromarray(array)
    if keep_ratio:
        im.thumbnail((size, size), resample)
    else:
        im = im.resize((size, size), resample)
    return im


def draw_bboxes(
    img,
    tl,
    br,
    rgb,
    score,
    label="",
    label_location="tl",
    opacity=0.1,
    line_thickness=0,
    font_scale=0.2,
    font_thickness=1,
):
    """Draw bounding boxes in an image"""
    tl = (int(tl[0]), int(tl[1]))
    br = (int(br[0]), int(br[1]))
    h, w = img.shape[:2]
    tl = (max(0, min(w - 1, tl[0])), max(0, min(h - 1, tl[1])))
    br = (max(1, min(w, br[0])), max(0, min(h, br[1])))
    if br[0] <= tl[0] or br[1] <= tl[1]:
        return img

    box = np.uint8(np.ones((br[1] - tl[1], br[0] - tl[0], 3)) * rgb)
    sub_combo = cv2.addWeighted(
        img[tl[1] : br[1], tl[0] : br[0], :], 1 - opacity, box, opacity, 1.0
    )
    img[tl[1] : br[1], tl[0] : br[0], :] = sub_combo
    if line_thickness > 0:
        img = cv2.rectangle(img, tuple(tl), tuple(br), rgb, line_thickness)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_line_type = cv2.LINE_AA
        label = str(label).upper()
        text_width, text_height = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )[0]
        label_origin = {"tl": tl, "br": br, "tr": (br[0], tl[1]), "bl": (tl[0], br[1])}[
            label_location
        ]
        label_offset = {
            "tl": np.array([0, -10]),
            "br": np.array([-text_width, text_height + 10]),
            "tr": np.array([-text_width, -10]),
            "bl": np.array([0, text_height + 10]),
        }[label_location]
        org = tuple((np.array(label_origin) + label_offset).tolist())
        img = cv2.putText(
            img,
            label + "(" + str(round(float(score), 2)) + ")",
            org,
            font,
            font_scale,
            rgb,
            font_thickness,
            font_line_type,
        )
    return img


def show_xray(
    *img, title: list or str = "", axis: bool = False, size: tuple = (20, 13)
):
    plt.figure(figsize=size)
    for n in range(len(img)):
        plt.subplot(len(img), 2, n + 1)
        plt.axis(axis)
        plt.imshow(img[n], cmap="gray")
        if isinstance(title, str):
            titre = title
        elif isinstance(title, (list, tuple)) and len(title) == len(img):
            titre = title[n]
        else:
            titre = ""
        plt.title(titre)
    plt.show()


def predict_bbox(image_to_predict, predictor, resized_width=256, resized_height=256):
    """Return predictions with labels, scores and bboxes.

    Bugfix: original code used undefined variable `image` instead of `img`.
    """
    with torch.no_grad():
        inputs_list = []
        img = image_to_predict.copy()
        if getattr(predictor, "input_format", "") == "RGB":
            img = img[:, :, ::-1]
        height, width = img.shape[:2]
        inputs = {"image": img, "height": height, "width": width}
        inputs_list.append(inputs)
        predictions = predictor.model(inputs_list)

    instances = predictions[0]["instances"]
    if len(instances) == 0:
        pred_classes = np.array([14], dtype=np.int64)
        pred_boxes = np.array([[0, 0, 1, 1]], dtype=np.float32)
        pred_scores = np.array([1.0], dtype=np.float32)
    else:
        fields: Dict[str, Any] = instances.get_fields()
        pred_classes = fields["pred_classes"]
        pred_scores = fields["scores"]
        pred_boxes = fields["pred_boxes"].tensor
        h_ratio, w_ratio = height / resized_height, width / resized_width
        pred_boxes[:, [0, 2]] *= w_ratio
        pred_boxes[:, [1, 3]] *= h_ratio
        pred_classes = pred_classes.cpu().numpy()
        pred_boxes = pred_boxes.cpu().numpy()
        pred_scores = pred_scores.cpu().numpy()
    return pred_classes, pred_boxes, pred_scores


class Xray:
    def __init__(
        self,
        path: str = "",
        folder: str = "",
        name: str = "",
        extension: str = "",
        th: float = 0.25,
        palette: str = "icefire",
        predictor=False,
    ):
        if extension == "":
            self.extension = path
        else:
            self.extension = extension
        if name == "":
            self.name = path
        else:
            self.name = name
        if folder == "":
            self.folder = path
        else:
            self.folder = folder
        self.path = path
        self.image = self.extension
        self.shape = self.image.shape
        self.height, self.width = self.shape[0], self.shape[1]
        self.th = th
        self.palette = [
            tuple([int(x) for x in np.array(c) * (255, 255, 255)])
            for c in sns.color_palette(palette, 15)
        ]
        self.predictor = predictor

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, ext):
        if ext == "png" or ext == "jpg":
            self._image = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        elif ext == "dicom":
            self._image = read_xray(self.path)
        else:
            raise ValueError("extention is not possible")

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        if value == "":
            if self.folder != "" and self.name != "" and self.extension != "":
                self._path = self.folder + "/" + self.name + "." + self.extension
            else:
                raise ValueError(
                    "Please provide a complete path or name, folder and extension values"
                )
        else:
            self._path = value

    @property
    def extension(self):
        return self._extension

    @extension.setter
    def extension(self, value):
        possible_extensions = ["png", "dicom", "jpg"]
        possible_extensions_txt = "png, dicom, jpg"
        ext = value.split(".")
        if len(ext) > 1:
            ext = ext[-1]
        if ext in possible_extensions:
            self._extension = ext
        else:
            raise ValueError(
                "Please enter a valid extension (possible extensions : "
                + possible_extensions_txt
            )

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        split_value = value.split(".")
        if len(split_value) > 1:
            name = split_value[-2]
            self._name = name.split("/")[-1]
        else:
            self._name = str(split_value)

    @property
    def folder(self):
        return self._folder

    @folder.setter
    def folder(self, value):
        split = value.split(".")
        split = "".join(split[: (len(split) - 1)])  # getting rid of the extension
        split = split.split("/")
        fold = "/".join(split[: (len(split) - 1)])  # getting rid of the name
        self._folder = fold

    @property
    def predictor(self):
        return self._predictor

    @predictor.setter
    def predictor(self, value):
        self._predictor = value
        return self._predictor

    def show(self):
        show_xray(self.image, title=self.name)

    def predict_bbox(self, resized_width: int = 256, resized_height: int = 256):
        if not self.predictor:
            raise ValueError(
                "Predictor is missing. Please provide one using Xray.predictor = predictor"
            )
        return predict_bbox(self.image, self.predictor, resized_width, resized_height)

    def process_prediction(self, th=False):
        if not th:
            th = self.th
        labels, boxes, scores = self.predict_bbox()
        processed_scores, processed_labels, processed_boxes = [], [], []
        if len(labels) > 1:
            count_dict = Counter(labels.tolist())
        else:
            count_dict = Counter(labels.tolist())
        for score, box, label in zip(scores, boxes, labels):
            score_i = score
            if int(label) == 0 and count_dict[label] != 1:
                best_score = np.max(scores[np.where(labels == label)])
                if score < best_score:
                    score_i = 0
            if int(label) == 3:
                score_i = score / 2
                if np.any(labels == 10):
                    score_i = 0
            if int(label) == 9:
                score_i = score / 1.3
            processed_scores.append(score_i)
            processed_labels.append(label)
            processed_boxes.append(box)
        return processed_labels, processed_boxes, processed_scores


class Xray_dataset:
    def __init__(self, files):
        self.files = [Xray(file) for file in files]


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


print("Utilities loaded.")



## === cell 2
print(df2.shape)
print(df2.columns.tolist())
print(df2.head(2))



## === cell 3
from collections import defaultdict

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
        v = arr.astype(np.float32).reshape(-1) / 255.0
        v = v - float(v.mean())
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
            train_embs[i] = 0
            continue
        train_embs[i] = embed_dicom(p, out_hw=32)

    KNN = 5
    base_conf = 0.26
    neighbor_decay = 0.03
    max_boxes_per_image = 6  # keep small to reduce false positives

    pred_strings = []
    for test_id in tqdm(
        df2["image_id"].astype(str).tolist(), desc="Building test predictions"
    ):
        test_path = os.path.join(TEST_IMG_DIR, f"{test_id}.dicom")
        if not os.path.exists(test_path):
            pred_strings.append(NO_FINDING_STR)
            continue

        q = embed_dicom(test_path, out_hw=32)  # (1024,)
        sims = train_embs @ q  # (N,)
        if np.all(train_embs == 0) or np.all(sims == 0):
            pred_strings.append(NO_FINDING_STR)
            continue

        knn_idx = np.argpartition(-sims, min(KNN, len(sims) - 1))[:KNN]
        knn_idx = knn_idx[np.argsort(-sims[knn_idx])]

        box_candidates = []
        for rank, idx in enumerate(knn_idx):
            neighbor_id = str(train_cand_ids[int(idx)])
            sim = float(sims[int(idx)])
            for cid, x1, y1, x2, y2 in img_to_boxes.get(neighbor_id, []):
                conf = base_conf + 0.08 * max(0.0, sim) - rank * neighbor_decay
                box_candidates.append((cid, conf, x1, y1, x2, y2))

        if len(box_candidates) == 0:
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

        if len(parts) == 0:
            pred_strings.append(NO_FINDING_STR)
        else:
            pred_strings.append(" ".join(parts))

    df2["PredictionString"] = pred_strings
    print(
        "Built image-specific initial PredictionString via kNN retrieval from train GT boxes."
    )
    print(df2.head(2))



## === cell 4
from tqdm import tqdm
from collections import Counter

TH = 0.1
NO_FINDING_STR = "14 1 0 0 1 1"

for i in tqdm(range(len(df2))):
    pred = df2.loc[i, "PredictionString"]

    if (
        pred is None
        or (isinstance(pred, float) and np.isnan(pred))
        or str(pred).strip() == ""
    ):
        df2.loc[i, "PredictionString"] = NO_FINDING_STR
        continue

    splited = str(pred).strip().split(" ")
    if len(splited) < 6:
        df2.loc[i, "PredictionString"] = NO_FINDING_STR
        continue

    result = {"label": [], "score": [], "xmin": [], "ymin": [], "xmax": [], "ymax": []}
    for n in range(len(splited) // 6):
        result["label"].append(splited[n * 6])
        result["score"].append(splited[n * 6 + 1])
        result["xmin"].append(splited[n * 6 + 2])
        result["ymin"].append(splited[n * 6 + 3])
        result["xmax"].append(splited[n * 6 + 4])
        result["ymax"].append(splited[n * 6 + 5])

    result_df = pd.DataFrame(result)
    if len(result_df) == 0:
        df2.loc[i, "PredictionString"] = NO_FINDING_STR
        continue

    result_df["score"] = (
        pd.to_numeric(result_df["score"], errors="coerce").fillna(0.0).astype(float)
    )
    labels = result_df["label"].astype(str).values
    count_dict = Counter(labels)

    resultSTR = ""
    for k in range(len(result_df)):
        label = str(result_df.loc[k, "label"])
        try:
            label_int = int(float(label))
        except Exception:
            label_int = 14

        score = float(result_df.loc[k, "score"])
        score_i = score

        if label_int == 0 and count_dict[label] != 1:
            best_score = result_df[result_df["label"].astype(str) == label][
                "score"
            ].max()
            if score < best_score:
                score_i = 0.0

        if label_int == 3:
            if np.any(result_df["label"].astype(str).values == "10"):
                score_i = 0.0
            else:
                best_score = result_df[result_df["label"].astype(str) == label][
                    "score"
                ].max()
                if score < best_score:
                    score_i = 0.0
                else:
                    score_i = score_i / 2.0

        if label_int == 9:
            score_i = score_i / 4.0

        if label_int == 14 and count_dict[label] != 1:
            best_score = result_df[result_df["label"].astype(str) == label][
                "score"
            ].max()
            if score < best_score:
                score_i = 0.0

        result_df.loc[k, "score"] = float(score_i)

        if float(result_df.loc[k, "score"]) > TH:
            resultSTR = (
                resultSTR
                + " "
                + str(result_df.loc[k, "label"])
                + " "
                + str(float(result_df.loc[k, "score"]))
                + " "
                + str(result_df.loc[k, "xmin"])
                + " "
                + str(result_df.loc[k, "ymin"])
                + " "
                + str(result_df.loc[k, "xmax"])
                + " "
                + str(result_df.loc[k, "ymax"])
            )

    resultSTR = resultSTR.strip()
    if resultSTR == "":
        resultSTR = NO_FINDING_STR
    df2.loc[i, "PredictionString"] = resultSTR

print("Post-processing done.")
print(df2.head(3))



## === cell 5
out_path = "./submission.csv"
df2[["image_id", "PredictionString"]].to_csv(out_path, index=False)

print(f"Wrote submission: {out_path}")
print(df2.shape)
print(df2.head(2))
