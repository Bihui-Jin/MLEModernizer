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

0.0475

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.0475) has done: 'I remove the hardcoded reads of missing `../input/.../submission.csv` files and instead load the provided `sample_submission.csv` as the base so the notebook always runs in this environment. Then I fix several NameErrors and logic bugs in the post-processing loop (e.g., undefined `df2`, string-vs-float comparisons, and `count_dict` use when there is only one box) while preserving the same intended filtering/calibration behavior. Finally, I ensure the output file is written as `./submission.csv` with the correct columns (`image_id`, `PredictionString`) so Kaggle accepts it. Since no current score exists (no valid submission previously), the goal is to produce a valid, stable submission; the post-processing remains essentially identical to your original intent.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

BASE_DIR = "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection"

sample_path = os.path.join(BASE_DIR, "sample_submission.csv")
if not os.path.exists(sample_path):
    sample_path = "/kaggle/input/sample_submission.csv"

df2 = pd.read_csv(sample_path)

expected_cols = {"image_id", "PredictionString"}
missing = expected_cols - set(df2.columns)
if missing:
    raise ValueError(f"sample_submission missing columns: {missing}")

df2.head()



## === cell 1
import os
from PIL import Image
from tqdm.auto import tqdm
from collections import Counter
from typing import Any, Dict
import cv2
import matplotlib.pyplot as plt
import numpy as np
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
    tl = (max(int(tl[0]), 0), max(int(tl[1]), 0))
    br = (max(int(br[0]), tl[0] + 1), max(int(br[1]), tl[1] + 1))
    h, w = img.shape[:2]
    br = (min(br[0], w), min(br[1], h))

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
            tuple((np.array(label_origin) + label_offset).astype(int)),
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
    """Show with matplotlib plot as many X-rays than passed as argument"""
    plt.figure(figsize=size)
    for n in range(len(img)):
        plt.subplot(1, len(img), n + 1)
        plt.axis(axis)
        plt.imshow(img[n], cmap="gray" if img[n].ndim == 2 else None)
        if isinstance(title, list):
            titre = title[n] if n < len(title) else ""
        elif isinstance(title, str):
            titre = title
        else:
            titre = ""
        plt.title(titre)
    plt.show()


def predict_bbox(image_to_predict, predictor, resized_width=256, resized_height=256):
    """Return predictions with labels, scores and bboxes"""
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
        split_value = value.split(".")
        if len(split_value) > 1:
            name = split_value[-2]
            self._name = name.split("/")[-1]
        else:
            self._name = value.split("/")[-1]

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

    def process_prediction(self, th=False):
        if not th:
            th = self.th
        labels, boxes, scores = self.predict_bbox()
        processed_scores, processed_labels, processed_boxes = [], [], []
        count_dict = (
            Counter(labels.tolist()) if len(labels) > 1 else Counter(labels.tolist())
        )
        for score, box, label in zip(scores, boxes, labels):
            score_i = float(score)
            if int(label) == 0 and count_dict[int(label)] != 1:
                best_score = np.max(scores[np.where(labels == label)])
                if score < best_score:
                    score_i = 0.0
            if int(label) == 3:
                score_i = score_i / 2.0
                if np.any(labels == 10):
                    score_i = 0.0
            if int(label) == 9:
                score_i = score_i / 1.3
            processed_scores.append(score_i)
            processed_labels.append(label)
            processed_boxes.append(box)
        return processed_labels, processed_boxes, processed_scores


class Xray_dataset:
    def __init__(self, files):
        self.files = [Xray(file) for file in files]


from typing import Any
import yaml


def save_yaml(filepath: str, content: Any, width: int = 120):
    with open(filepath, "w") as f:
        yaml.dump(content, f, width=width)


from dataclasses import dataclass, field
from typing import Dict, Any, Union, List


@dataclass
class Flags:
    debug: bool = True
    outdir: str = "results/det"
    device: str = "cuda:0"
    imgdir_name: str = "vinbigdata-chest-xray-resized-png-256x256"
    seed: int = 111
    target_fold: int = 0
    label_smoothing: float = 0.0
    model_name: str = "resnet18"
    model_mode: str = "normal"
    epoch: int = 20
    batchsize: int = 8
    valid_batchsize: int = 16
    num_workers: int = 4
    snapshot_freq: int = 5
    ema_decay: float = 0.999
    scheduler_type: str = ""
    scheduler_kwargs: Dict[str, Any] = field(default_factory=lambda: {})
    scheduler_trigger: List[Union[int, str]] = field(
        default_factory=lambda: [1, "iteration"]
    )
    aug_kwargs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {})
    mixup_prob: float = -1.0

    def update(self, param_dict: Dict) -> "Flags":
        for key, value in param_dict.items():
            if not hasattr(self, key):
                raise ValueError(f"[ERROR] Unexpected key for flag = {key}")
            setattr(self, key, value)
        return self


if __name__ == "__main__":
    print("tests went good")



## === cell 2
n = 5
name = df2.loc[n, "image_id"]
pred = (
    str(df2.loc[n, "PredictionString"])
    if pd.notna(df2.loc[n, "PredictionString"])
    else ""
)

dicom_path = os.path.join(BASE_DIR, "test", f"{name}.dicom")
if os.path.exists(dicom_path):
    xray = Xray(dicom_path)
    if pred.strip():
        splited = pred.split(" ")
        result = {
            "label": [],
            "score": [],
            "xmin": [],
            "ymin": [],
            "xmax": [],
            "ymax": [],
        }
        for j in range(len(splited) // 6):
            result["label"].append(splited[j * 6])
            result["score"].append(splited[j * 6 + 1])
            result["xmin"].append(splited[j * 6 + 2])
            result["ymin"].append(splited[j * 6 + 3])
            result["xmax"].append(splited[j * 6 + 4])
            result["ymax"].append(splited[j * 6 + 5])
        result_df = pd.DataFrame(result)
    else:
        result_df = pd.DataFrame(
            columns=["label", "score", "xmin", "ymin", "xmax", "ymax"]
        )
else:
    result_df = pd.DataFrame(columns=["label", "score", "xmin", "ymin", "xmax", "ymax"])

result_df.head()



## === cell 3
if "xray" in globals() and len(result_df) > 0:
    palette = "icefire"
    palette = [
        tuple([int(x) for x in np.array(c) * (255, 255, 255)])
        for c in sns.color_palette(palette, 15)
    ]
    predicted = xray.image
    if predicted.ndim == 2:
        predicted = cv2.cvtColor(predicted, cv2.COLOR_GRAY2RGB)
    for j in range(len(result_df)):
        predicted = draw_bboxes(
            predicted,
            tl=(
                int(float(result_df.loc[j, "xmin"])),
                int(float(result_df.loc[j, "ymin"])),
            ),
            br=(
                int(float(result_df.loc[j, "xmax"])),
                int(float(result_df.loc[j, "ymax"])),
            ),
            rgb=palette[int(float(result_df.loc[j, "label"]))],
            score=float(result_df.loc[j, "score"]),
            label=str(result_df.loc[j, "label"]),
            label_location="tl",
            opacity=0.1,
            line_thickness=0,
            font_scale=0.2,
            font_thickness=1,
        )



## === cell 4
if "xray" in globals() and "predicted" in globals():
    pass



## === cell 5
from collections import Counter
from tqdm import tqdm

TH = 0.1

df_out = df2.copy()

for i in tqdm(range(len(df_out))):
    pred = df_out.loc[i, "PredictionString"]
    pred = "" if pd.isna(pred) else str(pred).strip()

    if pred == "":
        df_out.loc[i, "PredictionString"] = ""
        continue

    splited = pred.split(" ")
    if len(splited) % 6 != 0:
        df_out.loc[i, "PredictionString"] = ""
        continue

    labels = [splited[j * 6] for j in range(len(splited) // 6)]
    scores = [splited[j * 6 + 1] for j in range(len(splited) // 6)]
    xmins = [splited[j * 6 + 2] for j in range(len(splited) // 6)]
    ymins = [splited[j * 6 + 3] for j in range(len(splited) // 6)]
    xmaxs = [splited[j * 6 + 4] for j in range(len(splited) // 6)]
    ymaxs = [splited[j * 6 + 5] for j in range(len(splited) // 6)]

    result_df = pd.DataFrame(
        {
            "label": labels,
            "score": scores,
            "xmin": xmins,
            "ymin": ymins,
            "xmax": xmaxs,
            "ymax": ymaxs,
        }
    )

    result_df["label_int"] = result_df["label"].astype(int)
    result_df["score_f"] = result_df["score"].astype(float)

    count_dict = (
        Counter(result_df["label_int"].tolist())
        if len(result_df) > 1
        else Counter(result_df["label_int"].tolist())
    )

    for k in range(len(result_df)):
        label_int = int(result_df.loc[k, "label_int"])
        score = float(result_df.loc[k, "score_f"])
        score_i = score

        if label_int == 0 and count_dict[label_int] != 1:
            best_score = float(
                result_df.loc[result_df["label_int"] == label_int, "score_f"].max()
            )
            if score < best_score:
                score_i = 0.0

        if label_int == 3:
            if (result_df["label_int"] == 10).any():
                score_i = 0.0
            else:
                best_score = float(
                    result_df.loc[result_df["label_int"] == label_int, "score_f"].max()
                )
                if score < best_score:
                    score_i = 0.0
                else:
                    score_i = score_i / 2.0

        result_df.loc[k, "score_f"] = score_i

    kept = result_df.loc[
        result_df["score_f"] > TH,
        ["label_int", "score_f", "xmin", "ymin", "xmax", "ymax"],
    ]

    if len(kept) == 0:
        df_out.loc[i, "PredictionString"] = ""
    else:
        parts = []
        for _, r in kept.iterrows():
            parts.extend(
                [
                    str(int(r["label_int"])),
                    f"{float(r['score_f']):.6f}",
                    str(int(float(r["xmin"]))),
                    str(int(float(r["ymin"]))),
                    str(int(float(r["xmax"]))),
                    str(int(float(r["ymax"]))),
                ]
            )
        df_out.loc[i, "PredictionString"] = " ".join(parts)

empty_mask = df_out["PredictionString"].fillna("").str.strip().eq("")
df_out.loc[empty_mask, "PredictionString"] = "14 1 0 0 1 1"

df_out.head()



## === cell 6
out_path = "./submission.csv"
df_out[["image_id", "PredictionString"]].to_csv(out_path, index=False)

print("Wrote:", out_path)
print(df_out.shape)
print(df_out.columns.tolist())
print(df_out.head(3))
