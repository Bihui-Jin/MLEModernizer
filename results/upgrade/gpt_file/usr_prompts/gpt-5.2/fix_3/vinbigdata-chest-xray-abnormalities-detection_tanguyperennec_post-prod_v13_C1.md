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

0.2121747901708085

# 6. Current score

0.01872

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0475) has done: 'I remove the dependency on missing external submissions (the `../input/results-vinbin/...` and `../input/241solution/...` paths) and instead start from the competition’s provided `sample_submission.csv`, so the notebook can run in this environment. Then I keep your existing “PredictionString filtering / score-adjustment” core logic, but make it robust to empty strings and type mismatches (strings vs ints/floats) that currently cause `NameError`/logic bugs. Finally, I ensure every test image has a valid prediction string (fallback to `14 1 0 0 1 1`) and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.01872) has done: 'Your current score is low because the notebook never generates real detections: it overwrites every row with `"14 1 0 0 1 1"` and then only “filters/adjusts” that constant string, so mAP stays near zero. To move toward the target score with minimal change and without introducing a new modeling approach, I keep your existing PredictionString parsing + score-adjustment logic, but replace the unconditional overwrite with a simple, legitimate heuristic baseline built from `train.csv`: for each class (0–13), predict a small number of high-frequency “anchor” boxes (median boxes from train) with priors (class frequency). Then your existing filtering code run on those predictions, and we still fall back to “No finding” when nothing passes the threshold, producing a valid submission.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

DATA_DIR = "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection"
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

if not os.path.exists(SAMPLE_SUB_PATH):
    SAMPLE_SUB_PATH = "/kaggle/input/sample_submission.csv"

df2 = pd.read_csv(SAMPLE_SUB_PATH)

if "image_id" not in df2.columns:
    raise ValueError(
        f"sample_submission missing 'image_id'. columns={df2.columns.tolist()}"
    )
if "PredictionString" not in df2.columns:
    if "TARGET" in df2.columns:
        df2 = df2.rename(columns={"TARGET": "PredictionString"})
    else:
        raise ValueError(
            f"sample_submission missing 'PredictionString'. columns={df2.columns.tolist()}"
        )

TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
if not os.path.exists(TRAIN_CSV_PATH):
    TRAIN_CSV_PATH = "/kaggle/input/train.csv"
train_df = pd.read_csv(TRAIN_CSV_PATH)

train_findings = train_df[train_df["class_id"].astype(int) != 14].copy()

if len(train_findings) == 0:
    df2["PredictionString"] = "14 1 0 0 1 1"
else:
    grp = train_findings.groupby("class_id")
    box_median = grp[["x_min", "y_min", "x_max", "y_max"]].median()
    class_counts = grp.size().sort_values(ascending=False)
    total = float(class_counts.sum())
    class_prior = (class_counts / total).to_dict()

    priors = np.array(list(class_prior.values()), dtype=np.float32)
    p_min, p_max = float(priors.min()), float(priors.max())

    def prior_to_score(p: float) -> float:
        if p_max <= p_min:
            return 0.20
        return 0.12 + (float(p) - p_min) / (p_max - p_min) * (0.55 - 0.12)

    TOPK_CLASSES = 6
    top_classes = [
        int(c)
        for c in class_counts.index.tolist()[:TOPK_CLASSES]
        if int(c) in box_median.index
    ]

    preds = []
    for _ in range(len(df2)):
        parts = []
        for cid in top_classes:
            x_min, y_min, x_max, y_max = box_median.loc[cid].tolist()
            x1 = int(round(float(x_min)))
            y1 = int(round(float(y_min)))
            x2 = int(round(float(x_max)))
            y2 = int(round(float(y_max)))
            if x2 <= x1:
                x2 = x1 + 1
            if y2 <= y1:
                y2 = y1 + 1
            score = prior_to_score(class_prior.get(cid, 0.0))
            parts.extend(
                [
                    str(int(cid)),
                    f"{float(score):.6f}".rstrip("0").rstrip("."),
                    str(x1),
                    str(y1),
                    str(x2),
                    str(y2),
                ]
            )
        preds.append(" ".join(parts).strip())
    df2["PredictionString"] = preds

df2.head()



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
    data = data.astype(np.float32)
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
    br = (max(br[0], tl[0] + 1), max(br[1], tl[1] + 1))

    box = np.uint8(np.ones((br[1] - tl[1], br[0] - tl[0], 3)) * rgb)
    sub_combo = cv2.addWeighted(
        img[tl[1] : br[1], tl[0] : br[0], :], 1 - opacity, box, opacity, 1.0
    )
    img[tl[1] : br[1], tl[0] : br[0], :] = sub_combo
    if line_thickness > 0:
        img = cv2.rectangle(img, tl, br, rgb, line_thickness)
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
            f"{label}({round(float(score), 2)})",
            tuple((np.array(label_origin) + label_offset).tolist()),
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
        plt.subplot(1, len(img), n + 1)
        plt.axis(axis)
        plt.imshow(img[n], cmap="gray" if (img[n].ndim == 2) else None)
        if isinstance(title, list):
            titre = title[n] if n < len(title) else ""
        elif isinstance(title, str):
            titre = title
        else:
            titre = ""
        plt.title(titre)
    plt.show()


def predict_bbox(image_to_predict, predictor, resized_width=256, resized_height=256):
    with torch.no_grad():
        inputs_list = []
        img = image_to_predict.copy()
        if getattr(predictor, "input_format", "BGR") == "RGB":
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
        pred_boxes = pred_boxes.clone()
        pred_boxes[:, [0, 2]] *= w_ratio
        pred_boxes[:, [1, 3]] *= h_ratio
        pred_classes = pred_classes.detach().cpu().numpy()
        pred_boxes = pred_boxes.detach().cpu().numpy()
        pred_scores = pred_scores.detach().cpu().numpy()
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
        self.path = path
        self.extension = extension if extension != "" else path
        self.name = name if name != "" else path
        self.folder = folder if folder != "" else path

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
        ext = value.split(".")
        if len(ext) > 1:
            ext = ext[-1]
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
        split_value = value.split(".")
        if len(split_value) > 1:
            name = split_value[-2]
            self._name = name.split("/")[-1]
        else:
            self._name = (
                split_value[0] if isinstance(split_value, list) else str(split_value)
            )

    @property
    def folder(self):
        return self._folder

    @folder.setter
    def folder(self, value):
        split = value.split(".")
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




## === cell 2
TEST_DIR = os.path.join(DATA_DIR, "test")
first_id = df2.loc[0, "image_id"]
test_path = os.path.join(TEST_DIR, f"{first_id}.dicom")
if os.path.exists(test_path):
    xray = Xray(test_path)
    xray.shape, xray.height, xray.width
else:
    ("missing_test_path", test_path)




## === cell 3
def parse_predstring(pred: str):
    pred = "" if pred is None else str(pred).strip()
    if pred == "":
        return pd.DataFrame(
            {"label": [], "score": [], "xmin": [], "ymin": [], "xmax": [], "ymax": []}
        )

    parts = pred.split()
    if len(parts) % 6 != 0:
        return pd.DataFrame(
            {"label": [], "score": [], "xmin": [], "ymin": [], "xmax": [], "ymax": []}
        )

    arr = np.array(parts, dtype=object).reshape(-1, 6)
    df = pd.DataFrame(arr, columns=["label", "score", "xmin", "ymin", "xmax", "ymax"])
    return df


example_df = parse_predstring(df2.loc[0, "PredictionString"])
example_df.head()



## === cell 4
from collections import Counter
from tqdm.auto import tqdm

TH = 0.1

for i in tqdm(range(len(df2))):
    pred = df2.loc[i, "PredictionString"]
    result_df = parse_predstring(pred)

    if len(result_df) == 0:
        df2.loc[i, "PredictionString"] = "14 1 0 0 1 1"
        continue

    try:
        result_df["label"] = result_df["label"].astype(int)
        result_df["score"] = result_df["score"].astype(float)
        for c in ["xmin", "ymin", "xmax", "ymax"]:
            result_df[c] = result_df[c].astype(float)
    except Exception:
        df2.loc[i, "PredictionString"] = "14 1 0 0 1 1"
        continue

    labels = result_df["label"].tolist()
    count_dict = Counter(labels)

    for k in range(len(result_df)):
        label = int(result_df.loc[k, "label"])
        score = float(result_df.loc[k, "score"])
        score_i = score

        if label == 0 and count_dict[label] != 1:
            best_score = float(
                result_df.loc[result_df["label"] == label, "score"].max()
            )
            if score < best_score:
                score_i = 0.0

        if label == 3:  # cardiomegaly
            if (
                10 in count_dict
            ):  # cardiomegaly + pleural effusion => remove cardiomegaly
                score_i = 0.0
            else:
                best_score = float(
                    result_df.loc[result_df["label"] == label, "score"].max()
                )
                if score < best_score:
                    score_i = 0.0
                else:
                    score_i = score_i / 2.0

        if label == 9:  # other lesion
            score_i = score_i / 4.0

        if label == 14 and count_dict[label] != 1:
            best_score = float(
                result_df.loc[result_df["label"] == label, "score"].max()
            )
            if score < best_score:
                score_i = 0.0

        result_df.loc[k, "score"] = float(score_i)

    kept = result_df[result_df["score"] > TH].copy()

    if len(kept) == 0:
        df2.loc[i, "PredictionString"] = "14 1 0 0 1 1"
        continue

    parts = []
    for _, row in kept.iterrows():
        parts.extend(
            [
                str(int(row["label"])),
                f"{float(row['score']):.6f}".rstrip("0").rstrip("."),
                str(int(round(float(row["xmin"])))),
                str(int(round(float(row["ymin"])))),
                str(int(round(float(row["xmax"])))),
                str(int(round(float(row["ymax"])))),
            ]
        )
    df2.loc[i, "PredictionString"] = " ".join(parts)

df2["PredictionString"] = df2["PredictionString"].fillna("").astype(str).str.strip()
df2.loc[df2["PredictionString"] == "", "PredictionString"] = "14 1 0 0 1 1"

df2.head()



## === cell 5
out_path = "./submission.csv"
df2.to_csv(out_path, index=False)

print("Saved:", out_path)
print("Columns:", df2.columns.tolist())
print("Rows:", len(df2))
df2.head()
