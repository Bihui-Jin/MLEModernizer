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

0.01057

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0475) has done: 'I remove the dependency on missing external submissions (the `../input/results-vinbin/...` and `../input/241solution/...` paths) and instead start from the competition’s provided `sample_submission.csv`, so the notebook can run in this environment. Then I keep your existing “PredictionString filtering / score-adjustment” core logic, but make it robust to empty strings and type mismatches (strings vs ints/floats) that currently cause `NameError`/logic bugs. Finally, I ensure every test image has a valid prediction string (fallback to `14 1 0 0 1 1`) and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.01872) has done: 'Your current score is low because the notebook never generates real detections: it overwrites every row with `"14 1 0 0 1 1"` and then only “filters/adjusts” that constant string, so mAP stays near zero. To move toward the target score with minimal change and without introducing a new modeling approach, I keep your existing PredictionString parsing + score-adjustment logic, but replace the unconditional overwrite with a simple, legitimate heuristic baseline built from `train.csv`: for each class (0–13), predict a small number of high-frequency “anchor” boxes (median boxes from train) with priors (class frequency). Then your existing filtering code run on those predictions, and we still fall back to “No finding” when nothing passes the threshold, producing a valid submission.'
- What this solution (achieved 0.0095) has done: 'The timeout is dominated by repeatedly DICOM-decoding every test image in `image_stats_u8()` and by per-row pandas DataFrame construction/parsing in the post-filter loop. To keep identical prediction semantics while making it fast, the script below (1) replaces `image_stats_u8()` with a metadata-only DICOM reader that computes mean/std from `WindowCenter/Width` (VOI LUT) when present and falls back to full pixel decoding only if needed, and (2) replaces `parse_predstring()` + per-row DataFrame operations with an equivalent pure-Python/numpy string pipeline that applies the same filtering rules but without pandas overhead. All paths, scoring logic, box math, and thresholds are unchanged; the only differences are performance-oriented and should be within negligible floating-point formatting differences.'
- What this solution (achieved 4e-05) has done: 'Your current baseline is too weak because it always predicts the same “median boxes for top classes” regardless of image content, so it rarely overlaps real findings at IoU>0.4. To move the score upward toward your target while keeping the same overall approach (train.csv-derived anchor boxes + per-image gating + the same post-filter rules), I make the anchors less “one-size-fits-all” by using multiple anchors per class (mixture via k-means on train boxes) and predict a small top-N per image, which increases the chance of matching a true box without changing model/training logic. I also compute per-image gating from cheap DICOM metadata only (no pixel decoding) to keep runtime under control. Finally, I keep your exact filtering/adjustment semantics and ensure the submission format remains valid for every image.'
- What this solution (achieved 1e-05) has done: 'Your current score (4e-05) is far below the target (0.212), so we need a meaningful improvement while keeping your core “train.csv-derived anchors → per-image gating → post-filtering” approach intact. The biggest issue is that you’re generating many low-quality boxes per image (top 10 classes × 2 anchors = 20 boxes/image) with modest confidence, which creates lots of false positives and tanks mAP; we move toward the target by sharply reducing predictions to only the most plausible classes and requiring higher confidence before emitting boxes. Concretely: (1) use fewer classes (TOPK_CLASSES) and only 1 anchor per class, (2) raise the filtering threshold TH, and (3) slightly lower the baseline score mapping so fewer boxes survive unless strongly supported by class prior + gating. This keeps the same semantics (anchors + gating + your exact per-class score adjustments) but should substantially reduce false positives and increase mAP toward the target.'
- What this solution (achieved 1e-05) has done: 'Your current score (1e-05) is far below the target (0.212), so we need a real uplift while keeping your same “train.csv-derived anchors → per-image gating → post-filtering” core logic. The minimal high-impact fix is to stop predicting only the top-6 most frequent classes with one anchor each (this is too low-recall), and instead increase recall by predicting more plausible classes and multiple anchors per class while keeping your exact filtering semantics to control false positives. Concretely, we (1) expand `TOPK_CLASSES` moderately, (2) increase `PER_CLASS_TOP` to use more of the existing k-means anchors, and (3) slightly lower `TH` so some additional boxes survive the filter—these are small parameter changes, not a change in approach. Everything else (paths, anchor construction, gating, and score-adjustment rules) stays the same, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 2e-05) has done: 'Your current score (1e-05) is far below the target (0.212, higher-is-better), so we need a meaningful uplift while keeping your same core approach (train.csv-derived anchors → per-image gating → your exact post-filter rules). The smallest high-impact change is to raise recall by (a) using more classes and (b) emitting more anchors per class, while controlling false positives by only keeping the top-N predictions per image before applying your unchanged `_process_predstring()` thresholding/adjustments. This keeps the same prediction semantics (anchors + gating + filtering), but avoids drowning mAP in many weak boxes and avoids the “too low recall” regime. The submission writing stays identical and still guarantees a valid fallback `"14 1 0 0 1 1"`.'
- What this solution (achieved 1e-05) has done: 'Your current score is far below the target (higher-is-better), and the main reason is that the baseline emits many medium-confidence boxes for many classes in every image, producing overwhelming false positives that crush mAP. To move the score upward toward the target while preserving the same core “train.csv-derived anchors → per-image gating → same post-filter rules” approach, I (1) reduce the number of candidate classes and total boxes per image, and (2) raise the filtering threshold so only the strongest predictions survive. These are small parameter changes (not a new model or new training), and they directly target precision/recall balance under mAP@IoU>0.4. The submission writing stays identical and still guarantees a valid fallback `"14 1 0 0 1 1"` for every test image.'
- What this solution (achieved 1e-05) has done: 'Your current score is far below the target (0.212), and the main issue is recall: you emit at most 5 boxes/image from only 8 classes and only 1 anchor per class, so most true findings never get any overlapping prediction at IoU>0.4. To move the score up toward the target while preserving the same core approach (train.csv-derived k-means anchors → per-image gating → the exact same post-filter rules), I only adjust three small parameters to increase recall in a controlled way: expand `TOPK_CLASSES`, increase `PER_CLASS_TOP`, and increase `MAX_TOTAL_PREDS`. I keep your thresholding/adjustment logic unchanged and still guarantee the required fallback `"14 1 0 0 1 1"` for every image. No new model, no new training loop, and no new feature extraction are introduced—just slightly richer anchor emission.'
- What this solution (achieved 2e-05) has done: 'Your current score is far below the target, so we should cautiously increase recall without flooding mAP with too many false positives. I keep your exact core pipeline (train.csv-derived k-means anchors → per-image gating from DICOM metadata → the same `_process_predstring` rules), but make three minimal, high-impact parameter adjustments: emit slightly more candidate classes, allow one more anchor per class, and slightly raise the pre-filter cap of total boxes per image. This should meaningfully reduce the “no overlap” failure mode while still letting your unchanged post-filter/thresholding suppress weak predictions. The submission schema and fallback `"14 1 0 0 1 1"` behavior remain unchanged.'
- What this solution (achieved 1e-05) has done: 'Your current score is far below the target (higher-is-better), so we need a meaningful lift while keeping the same core “train.csv-derived anchors → per-image gating → post-filtering” pipeline. The biggest likely issue is precision collapse from emitting many boxes per image and letting several medium-confidence boxes survive the filter; we reduce false positives by (1) limiting to the most frequent finding classes, (2) emitting fewer anchors per class, and (3) keeping only the top few boxes per image before applying your unchanged `_process_predstring()` rules. This is purely parameter-level control around the same anchor-generation logic (no new model/training/feature extraction) and should move mAP upward toward your target. We also keep the required fallback `"14 1 0 0 1 1"` and still write a valid `submission.csv`.'
- What this solution (achieved 0.01057) has done: 'Your current score (1e-05) is far below the target (0.212, higher-is-better), so we should increase mAP by improving localization/recall without changing the overall “train.csv-derived anchors → simple per-image gating → your exact post-filter rules” pipeline. The biggest low-risk issue is that anchors are built assuming a fixed 1024×1024 scale, but VinBigData images are not all 1024, so the emitted boxes often land in the wrong location/size and fail IoU>0.4; we fix this by learning anchors in a scale-invariant way using each training image’s real width/height from DICOM metadata (stop_before_pixels) and then converting back per-test-image using its own width/height. This keeps the same approach (k-means anchors), but makes boxes consistent with each image’s geometry and should substantially lift IoU matches (and thus mAP) toward your target. Everything else stays the same: same k-means, same gating, same TH and post-processing semantics, and it still writes a valid `submission.csv`.'

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
    class_counts = grp.size().sort_values(ascending=False)
    total = float(class_counts.sum())
    class_prior = (class_counts / total).to_dict()

    priors = np.array(list(class_prior.values()), dtype=np.float32)
    p_min, p_max = float(priors.min()), float(priors.max())

    def prior_to_score(p: float) -> float:
        if p_max <= p_min:
            return 0.16
        return 0.08 + (float(p) - p_min) / (p_max - p_min) * (0.45 - 0.08)

    TOPK_CLASSES = 10
    top_classes = [int(c) for c in class_counts.index.tolist()[:TOPK_CLASSES]]

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


def _get_first_float(x, default=None):
    if x is None:
        return default
    try:
        if isinstance(x, (list, tuple)):
            x = x[0]
        if (
            hasattr(x, "__iter__")
            and not isinstance(x, (str, bytes))
            and not isinstance(x, (float, int))
        ):
            x = list(x)[0]
        return float(x)
    except Exception:
        try:
            return float(getattr(x, "value", default))
        except Exception:
            return default


def image_stats_u8(path: str):
    try:
        ds = pydicom.dcmread(
            path,
            stop_before_pixels=True,
            force=True,
            specific_tags=["Rows", "Columns", "WindowCenter", "WindowWidth"],
        )
        w = int(getattr(ds, "Columns", 1024) or 1024)
        h = int(getattr(ds, "Rows", 1024) or 1024)
        wc = _get_first_float(getattr(ds, "WindowCenter", None), default=None)
        ww = _get_first_float(getattr(ds, "WindowWidth", None), default=None)

        if wc is not None and ww is not None and ww > 0:
            mu_n = 0.5
            sd_n = float(np.clip(ww / 4096.0, 0.0, 1.0))
            return float(mu_n), float(sd_n), w, h

        return 0.5, 0.5, w, h
    except Exception:
        return 0.5, 0.5, 1024, 1024


def image_wh(path: str):
    try:
        ds = pydicom.dcmread(
            path,
            stop_before_pixels=True,
            force=True,
            specific_tags=["Rows", "Columns"],
        )
        w = int(getattr(ds, "Columns", 1024) or 1024)
        h = int(getattr(ds, "Rows", 1024) or 1024)
        if w <= 0 or h <= 0:
            return 1024, 1024
        return w, h
    except Exception:
        return 1024, 1024




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

TH = 0.22


def _kmeans_np(X: np.ndarray, k: int, iters: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n == 0:
        return np.zeros((0, X.shape[1]), dtype=np.float32)
    if n <= k:
        return X.astype(np.float32, copy=False)
    idx = rng.choice(n, size=k, replace=False)
    C = X[idx].astype(np.float32, copy=True)
    for _ in range(iters):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        a = d2.argmin(axis=1)
        C_new = C.copy()
        for j in range(k):
            m = a == j
            if m.any():
                C_new[j] = X[m].mean(axis=0)
        if np.allclose(C, C_new, rtol=0, atol=1e-6):
            break
        C = C_new
    return C.astype(np.float32, copy=False)


anchors_by_class = {}
base_score_by_class = {}

if (
    "top_classes" in globals()
    and "train_findings" in globals()
    and len(top_classes) > 0
):
    train_dir = os.path.join(DATA_DIR, "train")

    unique_train_ids = train_findings["image_id"].dropna().astype(str).unique().tolist()
    wh_map = {}
    for image_id in tqdm(unique_train_ids, desc="Reading train (w,h) metadata"):
        dcm_path = os.path.join(train_dir, f"{image_id}.dicom")
        wh_map[image_id] = image_wh(dcm_path)

    for cid in top_classes:
        cid = int(cid)
        dfc = train_findings[train_findings["class_id"].astype(int) == cid]
        if len(dfc) == 0:
            continue

        img_ids = dfc["image_id"].astype(str).to_numpy()
        w_arr = np.empty(len(dfc), dtype=np.float32)
        h_arr = np.empty(len(dfc), dtype=np.float32)
        for i, iid in enumerate(img_ids):
            w_i, h_i = wh_map.get(iid, (1024, 1024))
            w_arr[i] = float(w_i)
            h_arr[i] = float(h_i)
        w_arr = np.maximum(w_arr, 1.0)
        h_arr = np.maximum(h_arr, 1.0)

        x1 = dfc["x_min"].astype(float).to_numpy()
        y1 = dfc["y_min"].astype(float).to_numpy()
        x2 = dfc["x_max"].astype(float).to_numpy()
        y2 = dfc["y_max"].astype(float).to_numpy()

        cx = 0.5 * (x1 + x2) / w_arr
        cy = 0.5 * (y1 + y2) / h_arr
        bw = (x2 - x1) / w_arr
        bh = (y2 - y1) / h_arr
        X = np.stack([cx, cy, bw, bh], axis=1).astype(np.float32)

        if len(X) < 200:
            k = 2
        elif len(X) < 1000:
            k = 3
        else:
            k = 4

        C = _kmeans_np(X, k=k, iters=25, seed=cid + 123)
        anchors_by_class[cid] = C  # (k,4) of cx,cy,bw,bh normalized to that image
        base_score_by_class[cid] = float(prior_to_score(class_prior.get(cid, 0.0)))

if len(anchors_by_class) > 0:
    preds = []
    for image_id in tqdm(
        df2["image_id"].tolist(), desc="Building baseline PredictionString"
    ):
        dcm_path = os.path.join(TEST_DIR, f"{image_id}.dicom")
        mu_n, sd_n, w, h = image_stats_u8(dcm_path)

        gate = (
            0.85
            + 0.25 * np.clip(sd_n, 0.0, 1.0)
            - 0.10 * np.clip(abs(mu_n - 0.5) * 2.0, 0.0, 1.0)
        )
        gate = float(np.clip(gate, 0.70, 1.10))

        scale = 0.90 + 0.25 * np.clip(sd_n, 0.0, 1.0)
        scale = float(np.clip(scale, 0.85, 1.15))

        parts_records = []  # list of (score, cid, x1,y1,x2,y2)

        PER_CLASS_TOP = 2
        MAX_TOTAL_PREDS = 6

        for cid in top_classes:
            cid = int(cid)
            if cid not in anchors_by_class:
                continue
            C = anchors_by_class[cid]
            base_score = base_score_by_class[cid]

            take = min(PER_CLASS_TOP, C.shape[0])
            for j in range(take):
                cx_n, cy_n, bw_n, bh_n = map(float, C[j].tolist())
                bw_px = max(2.0, (bw_n * w) * scale)
                bh_px = max(2.0, (bh_n * h) * scale)
                cx_px = cx_n * w
                cy_px = cy_n * h

                x1p = int(round(cx_px - 0.5 * bw_px))
                y1p = int(round(cy_px - 0.5 * bh_px))
                x2p = int(round(cx_px + 0.5 * bw_px))
                y2p = int(round(cy_px + 0.5 * bh_px))

                x1p = max(0, min(x1p, w - 2))
                y1p = max(0, min(y1p, h - 2))
                x2p = max(x1p + 1, min(x2p, w - 1))
                y2p = max(y1p + 1, min(y2p, h - 1))

                score = float(np.clip(base_score * gate * (0.92**j), 0.01, 0.99))
                parts_records.append((score, int(cid), x1p, y1p, x2p, y2p))

        parts_records.sort(key=lambda t: t[0], reverse=True)
        parts_records = parts_records[:MAX_TOTAL_PREDS]

        parts = []
        for score, cid, x1p, y1p, x2p, y2p in parts_records:
            parts.extend(
                [
                    str(int(cid)),
                    f"{float(score):.6f}".rstrip("0").rstrip("."),
                    str(int(x1p)),
                    str(int(y1p)),
                    str(int(x2p)),
                    str(int(y2p)),
                ]
            )

        preds.append(" ".join(parts).strip())
    df2["PredictionString"] = preds


def _format_score(s: float) -> str:
    return f"{float(s):.6f}".rstrip("0").rstrip(".")


def _process_predstring(pred: str, th: float = TH) -> str:
    pred = "" if pred is None else str(pred).strip()
    if not pred:
        return "14 1 0 0 1 1"
    parts = pred.split()
    if len(parts) % 6 != 0:
        return "14 1 0 0 1 1"

    n = len(parts) // 6
    labels = [0] * n
    scores = [0.0] * n
    x1s = [0.0] * n
    y1s = [0.0] * n
    x2s = [0.0] * n
    y2s = [0.0] * n
    try:
        j = 0
        for i in range(n):
            labels[i] = int(float(parts[j]))
            j += 1
            scores[i] = float(parts[j])
            j += 1
            x1s[i] = float(parts[j])
            j += 1
            y1s[i] = float(parts[j])
            j += 1
            x2s[i] = float(parts[j])
            j += 1
            y2s[i] = float(parts[j])
            j += 1
    except Exception:
        return "14 1 0 0 1 1"

    count_dict = Counter(labels)

    best0 = max((scores[i] for i in range(n) if labels[i] == 0), default=-1.0)
    best3 = max((scores[i] for i in range(n) if labels[i] == 3), default=-1.0)
    best14 = max((scores[i] for i in range(n) if labels[i] == 14), default=-1.0)

    has10 = 10 in count_dict

    for i in range(n):
        label = labels[i]
        score_i = float(scores[i])

        if label == 0 and count_dict[label] != 1:
            if scores[i] < best0:
                score_i = 0.0

        if label == 3:  # cardiomegaly
            if has10:  # cardiomegaly + pleural effusion => remove cardiomegaly
                score_i = 0.0
            else:
                if scores[i] < best3:
                    score_i = 0.0
                else:
                    score_i = score_i / 2.0

        if label == 9:  # other lesion
            score_i = score_i / 4.0

        if label == 14 and count_dict[label] != 1:
            if scores[i] < best14:
                score_i = 0.0

        scores[i] = float(score_i)

    kept_idx = [i for i in range(n) if scores[i] > th]
    if not kept_idx:
        return "14 1 0 0 1 1"

    out = []
    for i in kept_idx:
        out.extend(
            [
                str(int(labels[i])),
                _format_score(scores[i]),
                str(int(round(float(x1s[i])))),
                str(int(round(float(y1s[i])))),
                str(int(round(float(x2s[i])))),
                str(int(round(float(y2s[i])))),
            ]
        )
    return " ".join(out)


df2["PredictionString"] = [
    _process_predstring(p, TH)
    for p in tqdm(
        df2["PredictionString"].tolist(), desc="Filtering/adjusting predictions"
    )
]

df2["PredictionString"] = df2["PredictionString"].fillna("").astype(str).str.strip()
df2.loc[df2["PredictionString"] == "", "PredictionString"] = "14 1 0 0 1 1"

df2.head()




## === cell 5
out_path = "./submission.csv"

df_out = df2[["image_id", "PredictionString"]].copy()
df_out.to_csv(out_path, index=False)

print("Saved:", out_path)
print("Columns:", df_out.columns.tolist())
print("Rows:", len(df_out))
df_out.head()
