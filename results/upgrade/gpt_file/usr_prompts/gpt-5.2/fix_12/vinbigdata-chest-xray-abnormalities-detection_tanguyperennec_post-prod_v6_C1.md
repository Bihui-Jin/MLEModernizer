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

0.2031557757398293

# 6. Current score

0.0475

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0475) has done: 'I remove the non-existent input file reads and instead load the provided `sample_submission.csv` as the base submission template. Then I generate a valid `PredictionString` for every test image (defaulting to the required “No finding” string), so the pipeline always produces a correct CSV even without a detector model. Finally, I keep your thresholding/post-processing structure but make it robust to empty strings/NaNs and ensure the output columns match Kaggle’s expected format (`image_id,PredictionString`) and the file is saved as `submission.csv` in the working directory.'
- What this solution (achieved 0.0475) has done: 'The timeout is dominated by repeatedly decoding 1500 large DICOMs in Python and doing extra work inside each loop (full `apply_voi_lut`, repeated object creation, repeated min/max passes). I keep the exact heuristic logic and thresholds, but speed up the pipeline by (1) using a much faster DICOM read path for this use case (read only minimal tags + `pixel_array` and avoid `apply_voi_lut`), (2) computing mean/std on a small central downsample directly (no full-image normalization pass), and (3) parallelizing per-image processing with a deterministic thread pool (I/O bound + pydicom decompression benefits from threads). I also remove expensive plotting/debug cells from the critical path by gating them behind a `DEBUG` flag (defaults to False) so submission generation runs fast while preserving the algorithm and outputs. These changes are provably equivalent to the current heuristic’s semantics because the heuristic only depends on relative brightness/contrast of the central area and image shape, and we keep the same crop, resize, mean/std, and box formulas.'
- What this solution (achieved 0.0475) has done: 'Your current score (0.0475) is far below the target (0.2032), so we should improve score with minimal, low-risk changes while keeping your heuristic “detector” core logic intact. The biggest lever without changing the approach is aligning post-processing to mAP: avoid suppressing too many boxes and ensure every image has at least one prediction (your heuristic already does, but failures/empty strings can still occur). I (1) slightly lower the score threshold (keeps the same heuristic predictions but retains more boxes, typically improving recall and mAP), (2) make the fallback more robust by forcing a valid “No finding” string when any parsing issue occurs, and (3) ensure the submission uses the competition’s expected column names (`image_id,PredictionString`) exactly (already correct) while keeping runtime within limits.'
- What this solution (achieved 0.0475) has done: 'We keep your exact heuristic “detector” approach, but adjust post-processing to better match mAP by improving recall: don’t drop low-confidence heuristic boxes too aggressively, and ensure the “No finding” fallback only happens when truly needed. Concretely, we lower the score filter threshold slightly (from 0.05 to 0.01) and also keep the top-k predictions per image after filtering to avoid overly long/duplicated strings while preserving your existing ordering/format. We also clamp and sanitize bbox coordinates (xmin<xmax, ymin<ymax, inside image bounds when available) to prevent invalid boxes that can hurt evaluation. All changes are small, preserve your core logic, and still write a valid `submission.csv`.'
- What this solution (achieved 0.0475) has done: 'Your current heuristic always emits the same large “whole-image” boxes with mid confidences, which tends to score poorly on mAP because boxes don’t match GT localization and extra false positives hurt precision. To move toward the target with minimal semantic change, I only adjust post-processing to (1) include an explicit high-confidence “No finding” prediction alongside your heuristic boxes (this helps many normal images without preventing abnormal detections) and (2) slightly raise the score threshold and reduce per-image top-k to cut false positives, which usually increases mAP for weak localizers. I keep the heuristic generation logic, DICOM reading, parallelism, and submission format unchanged, and still guarantee every row has a valid PredictionString. The script continue to run end-to-end and write `./submission.csv`.'
- What this solution (achieved 0.0475) has done: 'We keep your heuristic detector and overall pipeline unchanged, but adjust the final post-processing to reduce false positives that are likely depressing mAP: only add the “No finding” class when there are no kept abnormal predictions (instead of always), and slightly tighten the confidence threshold while keeping top-k the same. This preserves the evaluation semantics and file format while shifting predictions toward higher precision, which is typically what a weak heuristic needs to move from ~0.05 toward ~0.20 mAP. We also ensure the fallback “14 1 0 0 1 1” is always emitted if parsing/filtering removes everything. The script still runs end-to-end and writes `./submission.csv` with `image_id,PredictionString`.'
- What this solution (achieved 0.0475) has done: 'We keep your heuristic “detector” exactly as-is and only make minimal post-processing changes that should increase mAP by reducing false positives and improving localization validity. Concretely: (1) tighten the confidence filter a bit and reduce per-image TOPK so fewer low-quality boxes are submitted (better precision), (2) ensure we never output class 14 (“No finding”) alongside abnormal boxes (it hurts precision), and (3) clamp/sanitize box coordinates more robustly and add a small box for very low-confidence cases only when empty. These are small, low-risk changes that preserve your pipeline and should move the score upward from 0.0475 toward the 0.203 target while still always producing a valid `submission.csv`.'
- What this solution (achieved 0.0475) has done: 'Your current score (0.0475) is far below the target (0.2032), so we should improve mAP by reducing obvious false positives while keeping your heuristic detector unchanged. The smallest lever is post-processing: your current settings keep 2 boxes per image at a fairly low threshold, which likely hurts precision badly for this weak localizer. I minimally tighten the confidence threshold and reduce TOPK to 1 so fewer low-quality boxes are submitted per image, while still guaranteeing a valid “No finding” prediction when no abnormal boxes pass the filter. I also add a tiny safety clamp so any malformed/degenerate boxes don’t slip through and poison evaluation.'
- What this solution (achieved 0.0475) has done: 'We keep your exact heuristic detector and overall pipeline, but tune the final post-processing to reduce false positives, which is the most likely reason mAP is stuck around ~0.05. Concretely, we raise the confidence threshold moderately and keep TOPK=1 so each image submits at most one abnormal box, otherwise we fall back to the required “No finding” prediction. We also add a tiny, deterministic “minimum score lift” for your kept box (without changing which box is chosen) so the single submitted prediction per image is not overly low-confidence after thresholding. These are minimal, metric-aligned changes that should move score upward toward the 0.203 target while preserving your core logic and still producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

INPUT_DIR = "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection"
SAMPLE_SUB_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

df2 = pd.read_csv(SAMPLE_SUB_PATH)

expected_cols = ["image_id", "PredictionString"]
missing = [c for c in expected_cols if c not in df2.columns]
if missing:
    raise ValueError(
        f"sample_submission.csv missing columns: {missing}. Found: {df2.columns.tolist()}"
    )

df2["PredictionString"] = ""
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

cv2.setNumThreads(0)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

_DICOM_TAGS_MIN = [
    "PhotometricInterpretation",
    "WindowCenter",
    "WindowWidth",
    "RescaleIntercept",
    "RescaleSlope",
    "VOILUTSequence",
    "PixelRepresentation",
    "BitsStored",
    "BitsAllocated",
    "HighBit",
    "SamplesPerPixel",
    "PlanarConfiguration",
    "Rows",
    "Columns",
    "PixelData",
]


def read_xray(path, voi_lut=True, fix_monochrome=True):
    dicom = pydicom.dcmread(
        path,
        force=True,
        stop_before_pixels=False,
        specific_tags=_DICOM_TAGS_MIN,
        defer_size="512 KB",
    )
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    if (
        fix_monochrome
        and getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1"
    ):
        data = np.amax(data) - data

    data = data.astype(np.float32, copy=False)
    data -= float(np.min(data))
    mx = float(np.max(data))
    if mx > 0.0:
        data /= mx
    data = (data * 255.0).astype(np.uint8, copy=False)
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
    """Draw bounding boxes in an image."""
    tl = (max(int(tl[0]), 0), max(int(tl[1]), 0))
    br = (max(int(br[0]), tl[0] + 1), max(int(br[1]), tl[1] + 1))
    h, w = img.shape[:2]
    tl = (min(tl[0], w - 1), min(tl[1], h - 1))
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
            tuple(np.array(label_origin) + label_offset),
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
    """Show with matplotlib plot as many X-rays than passed as argument."""
    plt.figure(figsize=size)
    for n in range(len(img)):
        plt.subplot(1, len(img), n + 1)
        plt.axis("off" if not axis else "on")
        plt.imshow(img[n], cmap="gray")
        if isinstance(title, str):
            titre = title
        elif isinstance(title, (list, tuple)) and n < len(title):
            titre = title[n]
        else:
            titre = ""
        plt.title(titre)
    plt.show()


def predict_bbox(image_to_predict, predictor, resized_width=256, resized_height=256):
    """Return predictions with labels, scores and bboxes."""
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
        pred_classes = fields["pred_classes"].detach().cpu().numpy()
        pred_scores = fields["scores"].detach().cpu().numpy()
        pred_boxes = fields["pred_boxes"].tensor.detach().cpu().numpy()

        h_ratio, w_ratio = height / resized_height, width / resized_width
        pred_boxes[:, [0, 2]] *= w_ratio
        pred_boxes[:, [1, 3]] *= h_ratio

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
            self._image = cv2.imread(self.path)
        elif ext == "dicom":
            self._image = read_xray(self.path)
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
            self._name = split_value[0]

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
        count_dict = Counter(labels.tolist()) if len(labels) > 1 else Counter()

        for score, box, label in zip(scores, boxes, labels):
            score_i = float(score)
            if int(label) == 0 and count_dict.get(int(label), 1) != 1:
                best_score = float(np.max(scores[np.where(labels == label)]))
                if score < best_score:
                    score_i = 0.0
            if int(label) == 3:
                score_i = score_i / 2.0
                if np.any(labels == 10):
                    score_i = 0.0
            if int(label) == 9:
                score_i = score_i / 1.3
            processed_scores.append(score_i)
            processed_labels.append(int(label))
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


def _read_dicom_for_heuristic(path: str):
    dcm = pydicom.dcmread(
        path,
        force=True,
        stop_before_pixels=False,
        specific_tags=["PhotometricInterpretation", "Rows", "Columns", "PixelData"],
        defer_size="512 KB",
    )
    arr = dcm.pixel_array
    if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = np.max(arr) - arr
    return arr


def heuristic_prediction_string(dicom_path: str) -> str:
    try:
        img = _read_dicom_for_heuristic(dicom_path)
        h, w = int(img.shape[0]), int(img.shape[1])

        y0, y1 = int(0.20 * h), int(0.80 * h)
        x0, x1 = int(0.20 * w), int(0.80 * w)
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            crop = img

        crop = crop.astype(np.float32, copy=False)
        mn = float(np.min(crop))
        crop = crop - mn
        mx = float(np.max(crop))
        if mx > 0.0:
            crop = crop / mx
        crop_u8 = (crop * 255.0).astype(np.uint8, copy=False)

        crop_small = cv2.resize(crop_u8, (64, 64), interpolation=cv2.INTER_AREA)
        mean = float(np.mean(crop_small) / 255.0)
        std = float(np.std(crop_small) / 255.0)

        xmin = int(0.10 * w)
        ymin = int(0.10 * h)
        xmax = int(0.90 * w)
        ymax = int(0.90 * h)

        preds = []
        base = 0.18 + 0.20 * min(std, 0.8)

        if mean < 0.55 or std > 0.18:
            score10 = float(np.clip(base + 0.05, 0.05, 0.60))
            preds.append((10, score10, xmin, int(0.55 * h), xmax, int(0.95 * h)))

        if mean < 0.60:
            score3 = float(np.clip(base, 0.05, 0.50))
            preds.append(
                (3, score3, int(0.20 * w), int(0.20 * h), int(0.80 * w), int(0.85 * h))
            )

        score7 = float(np.clip(0.15 + 0.25 * min(std, 0.8), 0.05, 0.55))
        preds.append((7, score7, xmin, ymin, xmax, ymax))

        preds = sorted(preds, key=lambda x: x[1], reverse=True)[:3]

        parts = []
        for cls, sc, xmn, ymn, xmx, ymx in preds:
            parts.extend(
                [
                    str(int(cls)),
                    str(float(sc)),
                    str(int(xmn)),
                    str(int(ymn)),
                    str(int(xmx)),
                    str(int(ymx)),
                ]
            )
        return " ".join(parts).strip()
    except Exception:
        return ""


if __name__ == "__main__":
    print("tests went good")



## === cell 2
test_dir = os.path.join(INPUT_DIR, "test")

image_ids = df2["image_id"].to_numpy()
dicom_paths = [os.path.join(test_dir, f"{iid}.dicom") for iid in image_ids]

from concurrent.futures import ThreadPoolExecutor

max_workers = min(8, (os.cpu_count() or 4))

pred_strings = [None] * len(dicom_paths)
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for i, ps in enumerate(
        tqdm(ex.map(heuristic_prediction_string, dicom_paths), total=len(dicom_paths))
    ):
        pred_strings[i] = ps

df2["PredictionString"] = pred_strings
df2.head()



## === cell 3
DEBUG = False

if DEBUG:
    n = 5
    name = df2.loc[n, "image_id"]
    pred = df2.loc[n, "PredictionString"]

    test_dicom_path = os.path.join(INPUT_DIR, "test", f"{name}.dicom")
    xray = Xray(test_dicom_path)

    splited = str(pred).strip().split(" ") if isinstance(pred, str) else []
    result = {"label": [], "score": [], "xmin": [], "ymin": [], "xmax": [], "ymax": []}

    for j in range(len(splited) // 6):
        result["label"].append(splited[j * 6])
        result["score"].append(splited[j * 6 + 1])
        result["xmin"].append(splited[j * 6 + 2])
        result["ymin"].append(splited[j * 6 + 3])
        result["xmax"].append(splited[j * 6 + 4])
        result["ymax"].append(splited[j * 6 + 5])

    result_df = pd.DataFrame(result)
    result_df.head()



## === cell 4
if DEBUG:
    palette = "icefire"
    palette = [
        tuple([int(x) for x in np.array(c) * (255, 255, 255)])
        for c in sns.color_palette(palette, 15)
    ]

    predicted = xray.image
    if predicted.ndim == 2:
        predicted = cv2.cvtColor(predicted, cv2.COLOR_GRAY2RGB)

    for i in range(len(result_df)):
        predicted = draw_bboxes(
            predicted,
            tl=(
                int(float(result_df.loc[i, "xmin"])),
                int(float(result_df.loc[i, "ymin"])),
            ),
            br=(
                int(float(result_df.loc[i, "xmax"])),
                int(float(result_df.loc[i, "ymax"])),
            ),
            rgb=palette[int(float(result_df.loc[i, "label"]))],
            score=float(result_df.loc[i, "score"]),
            label=str(result_df.loc[i, "label"]),
            label_location="tl",
            opacity=0.1,
            line_thickness=0,
            font_scale=0.2,
            font_thickness=1,
        )



## === cell 5
if DEBUG:
    try:
        show_xray(xray.image, predicted, size=(12, 6))
    except Exception as e:
        print("Skipping display:", repr(e))



## === cell 6
TH = 0.40
TOPK_PER_IMAGE = 1

INCLUDE_NOFINDING_WHEN_EMPTY = True
NOFINDING_SCORE = 1.0

MIN_KEPT_CONFIDENCE = 0.45


def _postprocess_predstring(pred, th: float, topk: int) -> str:
    pred = (
        ""
        if (pred is None or (isinstance(pred, float) and np.isnan(pred)))
        else str(pred).strip()
    )

    splited = pred.split() if pred != "" else []
    items = []
    if len(splited) % 6 == 0 and len(splited) > 0:
        for j in range(len(splited) // 6):
            try:
                cls = int(float(splited[j * 6]))
                sc = float(splited[j * 6 + 1])
                x1 = int(float(splited[j * 6 + 2]))
                y1 = int(float(splited[j * 6 + 3]))
                x2 = int(float(splited[j * 6 + 4]))
                y2 = int(float(splited[j * 6 + 5]))

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = max(0, x2)
                y2 = max(0, y2)
                if x2 <= x1:
                    x2 = x1 + 1
                if y2 <= y1:
                    y2 = y1 + 1

                if cls == 14:
                    continue

                if sc >= th:
                    items.append((sc, cls, x1, y1, x2, y2))
            except Exception:
                continue

    if not items and INCLUDE_NOFINDING_WHEN_EMPTY:
        return f"14 {float(NOFINDING_SCORE)} 0 0 1 1"

    if not items:
        return "14 1 0 0 1 1"

    items.sort(key=lambda t: t[0], reverse=True)
    items = items[: int(topk)]

    out = []
    for sc, cls, x1, y1, x2, y2 in items:
        sc = float(max(float(sc), float(MIN_KEPT_CONFIDENCE)))
        out.extend(
            [
                str(int(cls)),
                str(float(sc)),
                str(int(x1)),
                str(int(y1)),
                str(int(x2)),
                str(int(y2)),
            ]
        )

    return " ".join(out).strip()


pred_arr = df2["PredictionString"].to_numpy()
preds_pp = [
    _postprocess_predstring(p, TH, TOPK_PER_IMAGE)
    for p in tqdm(pred_arr, total=len(pred_arr))
]
df2["PredictionString"] = preds_pp



## === cell 7
sub_path = "./submission.csv"
df2[["image_id", "PredictionString"]].to_csv(sub_path, index=False)
print("Wrote submission to:", sub_path)
print(df2.head())
