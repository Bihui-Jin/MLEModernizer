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

0.2190343062328492

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0475) has done: 'I remove the broken reads of non-existent input submissions and instead start from the competition’s provided `sample_submission.csv`, which guarantees correct IDs and column names. Then I apply your existing post-processing logic (thresholding and a couple of class-specific score rules) in a safe way that handles empty/NaN PredictionStrings and avoids `NameError`/`UnboundLocalError`. Finally, I write a valid `submission.csv` with the required `image_id,PredictionString` columns to the working directory so Kaggle can score it.'
- What this solution (achieved 0.00969) has done: 'Your current pipeline never generates real detections; it only post-processes the (empty) `sample_submission.csv`, so it collapses to mostly “No finding” and caps mAP very low. To move your score upward toward the target with minimal disruption, I keep all your existing utilities and post-processing logic, but actually run inference on the test DICOMs using a lightweight, deterministic baseline derived from train-set box statistics (no new model/architecture/training loop). Concretely, we predict a few plausible boxes per image (using mean boxes per class from training) with conservative confidences and then apply your same thresholding/class rules; this typically improves over all-“No finding” while staying simple and within time. The output still be a valid `submission.csv` with `image_id,PredictionString`.'
- What this solution (achieved 0.0475) has done: 'The timeout is dominated by repeatedly fully decoding 1500 DICOM pixel arrays in `_abnormality_proxy`, which is extremely expensive I/O + CPU work and unnecessary for this “center-crop mean” proxy. I keep the exact proxy logic/thresholding/output formatting intact, but speed it up by reading only the needed tags and then decoding pixels via `pydicom.dcmread(..., specific_tags=...)` plus `ds.pixel_array` once, avoiding a second full read and avoiding extra intermediate copies. I also remove avoidable per-image Python overhead in the main loop by precomputing a few lookup structures and using faster list construction, while preserving the same filtering semantics and string formatting. Paths, thresholds, and the overall heuristic logic remain unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

INPUT_DIR = "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection"
SAMPLE_SUB_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_DIR = os.path.join(INPUT_DIR, "test")

if not os.path.exists(SAMPLE_SUB_PATH):
    SAMPLE_SUB_PATH = "/kaggle/input/sample_submission.csv"
if not os.path.exists(TRAIN_CSV_PATH):
    TRAIN_CSV_PATH = "/kaggle/input/train.csv"
if not os.path.exists(TEST_DIR):
    TEST_DIR = "/kaggle/input/test"

df2 = pd.read_csv(SAMPLE_SUB_PATH)

if "image_id" not in df2.columns:
    raise ValueError(
        f"sample_submission missing 'image_id' column. columns={df2.columns.tolist()}"
    )
if "PredictionString" not in df2.columns:
    if "TARGET" in df2.columns:
        df2 = df2.rename(columns={"TARGET": "PredictionString"})
    else:
        raise ValueError(
            f"sample_submission missing 'PredictionString' column. columns={df2.columns.tolist()}"
        )

df2["PredictionString"] = df2["PredictionString"].fillna("").astype(str)

print("Loaded sample submission:", df2.shape)
print(df2.head(2))



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
        img = cv2.putText(
            img,
            label + "(" + str(round(float(score), 2)) + ")",
            tuple(label_origin + label_offset),
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
    """Show with matplotlib plot as many X-rays as passed as arguments"""
    plt.figure(figsize=size)
    for n in range(len(img)):
        plt.subplot(1, len(img), n + 1)
        plt.axis(axis)
        plt.imshow(img[n], cmap="gray")
        if isinstance(title, list):
            if n < len(title):
                plt.title(title[n])
        elif isinstance(title, str):
            plt.title(title)
    plt.show()


def predict_bbox(image_to_predict, predictor, resized_width=256, resized_height=256):
    """Return predictions with labels, scores and bboxes"""
    with torch.no_grad():
        inputs_list = []
        img = image_to_predict.copy()
        if predictor.input_format == "RGB":
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
        pred_classes, pred_scores, pred_boxes = (
            fields["pred_classes"],
            fields["scores"],
            fields["pred_boxes"].tensor,
        )
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
        if ext == "png":
            self._image = cv2.imread(self.path)
        elif ext == "dicom":
            self._image = read_xray(self.path)
        elif ext == "jpg":
            self._image = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        else:
            raise ValueError("extension is not possible")

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
        ext = value.split(".")
        if len(ext) > 1:
            ext = ext[-1]
        else:
            ext = ext[0]
        if ext in possible_extensions:
            self._extension = ext
        else:
            raise ValueError(
                "Please enter a valid extension (possible extensions : png, dicom, jpg)"
            )

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        split_value = str(value).split(".")
        if len(split_value) > 1:
            name = split_value[-2]
            self._name = name.split("/")[-1]
        else:
            self._name = split_value[0]

    @property
    def folder(self):
        return self._folder

    @folder.setter
    def folder(self, value):
        split = str(value).split(".")
        split = "".join(split[: (len(split) - 1)])
        split = split.split("/")
        fold = "/".join(split[: (len(split) - 1)])
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


class Xray_dataset:
    def __init__(self, files):
        self.files = [Xray(file) for file in files]




## === cell 2
from collections import Counter
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
import glob

TH = 0.05  # keep user's threshold


def _format_prediction_items(items):
    if not items:
        return "14 1 0 0 1 1"
    out = []
    for it in items:
        out.extend(
            [it["label"], it["score"], it["xmin"], it["ymin"], it["xmax"], it["ymax"]]
        )
    return " ".join(out)


def _clip_box(xmin, ymin, xmax, ymax, w, h):
    xmin = float(np.clip(xmin, 0, w - 1))
    ymin = float(np.clip(ymin, 0, h - 1))
    xmax = float(np.clip(xmax, xmin + 1, w))
    ymax = float(np.clip(ymax, ymin + 1, h))
    return xmin, ymin, xmax, ymax


def _abnormality_proxy(dicom_path: str):
    try:
        ds = pydicom.dcmread(
            dicom_path,
            force=True,
            specific_tags=["Rows", "Columns", "PhotometricInterpretation", "PixelData"],
        )
        h = int(getattr(ds, "Rows", 1024) or 1024)
        w = int(getattr(ds, "Columns", 1024) or 1024)

        arr = ds.pixel_array.astype(np.float32, copy=False)
        mn = float(arr.min())
        arr = arr - mn
        mx = float(arr.max())
        if mx > 0.0:
            arr = arr / mx

        ch0 = int(h * 0.40)
        ch1 = int(h * 0.60)
        cw0 = int(w * 0.40)
        cw1 = int(w * 0.60)
        crop = arr[ch0:ch1, cw0:cw1]
        if crop.size == 0:
            return 0.0, h, w
        return float(crop.mean()), h, w
    except Exception:
        return 0.0, 1024, 1024


train_df = pd.read_csv(TRAIN_CSV_PATH)
train_df = train_df[train_df["class_id"].between(0, 13)].copy()
train_df = train_df[
    (train_df["x_max"] > train_df["x_min"]) & (train_df["y_max"] > train_df["y_min"])
]

class_priors = {}
for cid, g in train_df.groupby("class_id"):
    q = g[["x_min", "y_min", "x_max", "y_max"]].quantile([0.25, 0.5, 0.75])
    med = q.loc[0.5].values.astype(float)
    iqr = (q.loc[0.75] - q.loc[0.25]).values.astype(float)
    class_priors[int(cid)] = {"med": med, "iqr": iqr}

common_classes = train_df["class_id"].value_counts().head(8).index.astype(int).tolist()
if 10 not in common_classes and 10 in class_priors:
    common_classes = common_classes[:7] + [10]

print("Using common_classes:", common_classes)

base_scores = {cid: 0.095 for cid in common_classes}
if len(common_classes) > 0:
    base_scores[common_classes[0]] = 0.115  # top class a bit higher

PROXY_LOW = 0.22
PROXY_HIGH = 0.50

try:
    test_ids = {
        os.path.basename(p)[:-6] for p in glob.glob(os.path.join(TEST_DIR, "*.dicom"))
    }
except Exception:
    test_ids = set()

image_ids = df2["image_id"].tolist()
dicom_paths = [
    os.path.join(TEST_DIR, f"{iid}.dicom") if iid in test_ids else None
    for iid in image_ids
]

max_workers = min(8, (os.cpu_count() or 2))


def _proxy_or_default(p):
    if p is None:
        return (0.0, 1024, 1024)
    return _abnormality_proxy(p)


with ThreadPoolExecutor(max_workers=max_workers) as ex:
    proxies = list(
        tqdm(
            ex.map(_proxy_or_default, dicom_paths, chunksize=8),
            total=len(dicom_paths),
        )
    )

pred_strings = [""] * len(df2)

_common = []
for cid in common_classes:
    pr = class_priors.get(cid)
    if pr is None:
        continue
    med = pr["med"]
    base = base_scores.get(cid, 0.095)
    _common.append((cid, med, base))

for i in range(len(df2)):
    proxy, h, w = proxies[i]

    if proxy <= PROXY_LOW:
        scale = 0.0
    elif proxy >= PROXY_HIGH:
        scale = 1.0
    else:
        scale = (proxy - PROXY_LOW) / (PROXY_HIGH - PROXY_LOW)

    items = []
    if scale > 0:
        mult = 0.78 + 0.42 * scale
        for cid, med, base in _common:
            xmin, ymin, xmax, ymax = _clip_box(med[0], med[1], med[2], med[3], w, h)
            sc = base * mult
            items.append(
                {
                    "label": str(int(cid)),
                    "score": f"{float(sc):.6f}".rstrip("0").rstrip("."),
                    "xmin": f"{xmin:.1f}".rstrip("0").rstrip("."),
                    "ymin": f"{ymin:.1f}".rstrip("0").rstrip("."),
                    "xmax": f"{xmax:.1f}".rstrip("0").rstrip("."),
                    "ymax": f"{ymax:.1f}".rstrip("0").rstrip("."),
                }
            )

    if not items:
        pred_strings[i] = "14 1 0 0 1 1"
        continue

    labels = [it["label"] for it in items]
    count_dict = Counter(labels)

    best_score_per_label = {}
    for it in items:
        lab = it["label"]
        sc = float(it["score"])
        prev = best_score_per_label.get(lab)
        if (prev is None) or (sc > prev):
            best_score_per_label[lab] = sc

    has_pleural_effusion_10 = any(l == "10" for l in labels)

    kept = []
    for it in items:
        label = it["label"]
        score = float(it["score"])
        score_i = score

        if label == "0" and count_dict[label] != 1:
            if score < best_score_per_label[label]:
                score_i = 0.0

        if label == "3":
            if has_pleural_effusion_10:
                score_i = 0.0
            else:
                if score < best_score_per_label[label]:
                    score_i = 0.0

        if score_i > TH:
            it2 = dict(it)
            it2["score"] = f"{score_i:.6f}".rstrip("0").rstrip(".")
            kept.append(it2)

    pred_strings[i] = _format_prediction_items(kept)

df2["PredictionString"] = pd.Series(pred_strings, index=df2.index).replace(
    "", "14 1 0 0 1 1"
)
print(df2.head(3))



## === cell 3
SUB_PATH = "./submission.csv"
sub = df2[["image_id", "PredictionString"]].rename(
    columns={"image_id": "ID", "PredictionString": "TARGET"}
)
sub.to_csv(SUB_PATH, index=False)
print("Wrote:", SUB_PATH, "rows:", len(sub))
print(sub.sample(2, random_state=0))
