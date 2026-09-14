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

0.01087

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0475) has done: 'I remove the hardcoded reads of missing `../input/.../submission.csv` files and instead load the provided `sample_submission.csv` as the base so the notebook always runs in this environment. Then I fix several NameErrors and logic bugs in the post-processing loop (e.g., undefined `df2`, string-vs-float comparisons, and `count_dict` use when there is only one box) while preserving the same intended filtering/calibration behavior. Finally, I ensure the output file is written as `./submission.csv` with the correct columns (`image_id`, `PredictionString`) so Kaggle accepts it. Since no current score exists (no valid submission previously), the goal is to produce a valid, stable submission; the post-processing remains essentially identical to your original intent.'
- What this solution (achieved 0.01854) has done: 'Your current submission is effectively the sample submission (mostly “No finding”), which caps mAP very low; to move toward the 0.212 target we need to actually generate some plausible boxes from train statistics while keeping your overall post-processing/filtering idea intact. I add a minimal “prior-based detector”: for each class (0–13) we compute a few representative boxes from `train.csv` (median box, plus a couple quantiles) and use class frequency to assign small confidences; then we write those predictions into `df2` before your existing thresholding/calibration cell runs. This keeps the rest of your pipeline unchanged (same filtering loop, same final “No finding” fallback), but provides non-empty detections so mAP should increase substantially from 0.0475 toward the target band. All paths remain under `/kaggle/input/...`, runtime stays small (only CSV aggregation), and it still writes `./submission.csv` with the required columns.'
- What this solution (achieved 0.00065) has done: 'Your current “prior-based detector” is generating the same few absolute pixel boxes for every test image, but test DICOMs have varying widths/heights, so many boxes end up badly placed and IoU is usually poor—this is a primary reason mAP stays far below the 0.212 target. I keep your core approach (train.csv box priors + the same post-filtering loop) but compute class representative boxes in *normalized coordinates* (relative to image width/height) and then scale them per-test-image using each test DICOM’s shape. I also make the scores safely survive your `TH=0.1` filter by slightly raising the base confidence range (still low, not trying to “maximize”, just to get non-empty predictions through your unchanged filtering). All paths remain the same, runtime stays within limits by only reading each test DICOM once for shape (no pixel decode), and it still writes `./submission.csv` with `image_id,PredictionString`.'
- What this solution (achieved 0.02449) has done: 'Your current score (0.00065) is far below the target (0.21217), so we should improve mAP by making your existing “train prior boxes → per-test scaling → existing filtering → no-finding fallback” pipeline produce more *reasonable* and *diverse* detections without changing the overall approach. The main issue is that estimating train image width/height from `x_max/y_max` is very inaccurate, which corrupts your normalized boxes; we instead read true `(Rows, Columns)` for train images (metadata only, no pixels) for a small, deterministic subset per class to stay within the time limit. Then we generate representative normalized boxes per class from those true dimensions and predict a few top classes per image with confidences that reliably survive your unchanged `TH=0.1` filter. Everything else (post-processing logic, thresholds, submission schema/paths) stays the same.'
- What this solution (achieved 0.0141) has done: 'We’re far below the target (0.02449 vs 0.21217), so we should improve mAP by making your existing “train prior boxes → per-test scaling → existing TH=0.1 filter + label rules → no-finding fallback” pipeline produce more class coverage and better box overlap, without changing the post-processing core logic. The smallest effective lever is in the prior generation: instead of always predicting only 5 most frequent classes with 2 boxes each, we predict a few representative boxes for more classes (still capped to keep runtime and output size sane) and set confidences so they reliably survive your unchanged TH filter. We also add a small, deterministic per-image jitter (scale/shift) to the normalized boxes to increase diversity and chance IoU>0.4 across varying findings, while keeping the same overall approach (no model training, still using train-derived priors and test DICOM shapes only). All file paths stay the same and the script still writes a valid `./submission.csv` with `image_id,PredictionString`.'
- What this solution (achieved 0.01196) has done: 'We’re far below the target (0.0141 vs 0.2122), so the smallest safe way to move mAP upward is to improve box plausibility/overlap while keeping your same “train-derived priors → per-test scaling → TH=0.1 filtering → No finding fallback” core intact. I (1) build per-class *mixture* priors by clustering normalized boxes (k=3) instead of using simple quantiles (still no learning on test labels, just better representative boxes), (2) use a deterministic per-image selection of the best prior component per class, and (3) apply a tiny amount of per-image confidence jitter while ensuring scores still survive your unchanged TH=0.1 filter. These changes keep the overall pipeline identical (no model training, same post-processing semantics), but should yield noticeably better IoU alignment and thus improve mAP toward the target band. The script still writes `./submission.csv` with `image_id,PredictionString`.'
- What this solution (achieved 0.01245) has done: 'Your current score (0.01196) is far below the target (0.21217), so we need a modest, safe boost without changing the overall “train-derived priors → per-test scaling → TH=0.1 filtering → No finding fallback” pipeline. The main low-risk lever is to make the priors better match typical object extents per class by clustering in a more stable space: center/size (cx,cy,w,h) rather than raw corners, then converting back to corners per test image. To avoid overpredicting many low-quality boxes (which can hurt mAP), we slightly reduce the number of predicted classes and boxes per image while keeping confidences above the unchanged TH filter. These are minimal edits confined to prior generation; your post-processing/filtering and submission writing stay the same.'
- What this solution (achieved 0.01282) has done: 'We need to move your score up toward 0.212 (current 0.01245), so the smallest safe lever is to improve box plausibility/IoU while keeping your same “train-derived priors → per-test scaling → TH=0.1 filter → No finding fallback” pipeline. I keep all post-processing logic intact and only adjust prior generation to (1) build per-class priors using a slightly larger, more reliable train subset (still metadata-only DICOM reads), and (2) pick per-test-image the closest prior component by matching each class’s typical box area to the test image size, reducing the harmful jitter that likely hurts IoU>0.4. I also slightly rebalance confidence so more useful classes survive your unchanged TH filter without flooding too many boxes (which can reduce mAP). The script still run end-to-end and write `./submission.csv` with `image_id,PredictionString`.'
- What this solution (achieved 0.00119) has done: 'Your current score (0.01282) is far below the target (0.21217), so we should improve mAP by making the existing “train-derived priors → per-test scaling → TH=0.1 filtering → No finding fallback” pipeline produce boxes that overlap better with typical findings while keeping your post-processing intact. The smallest high-impact fix is to generate priors from train boxes in a way that’s more robust: (1) compute per-class normalized priors using all available boxes (no train DICOM reads needed), (2) expand `top_classes` coverage (more classes predicted) but keep a strict cap on total boxes/image to avoid mAP harm from flooding, and (3) slightly reduce the per-image jitter/scale perturbations that likely break IoU>0.4. These edits are confined to the prior-generation cell and keep your TH filter, label rules, and submission writing unchanged. Runtime also improves because we remove the expensive loop that reads hundreds/thousands of train DICOM headers.'
- What this solution (achieved 0.0087) has done: 'We’re far below the target (0.00119 vs 0.21217), so we should increase mAP by improving IoU and class relevance while keeping your same “train-derived priors → per-test scaling → TH=0.1 filtering → No finding fallback” pipeline intact. The highest-impact minimal fix is that your “normalization” currently uses per-image max box extents (`Wm/Hm`), which severely distorts normalized boxes; instead we estimate true train image sizes from DICOM metadata for a small deterministic subset and build normalized priors from that. Then we keep the same k-means (cx,cy,w,h) prior generation and the same post-filtering logic, but adjust which classes/boxes are emitted (a few more classes, fewer boxes per class) to avoid flooding while increasing coverage. Finally, we keep confidences safely above TH without changing TH or the later filtering rules, and still write `./submission.csv` with the required columns.'
- What this solution (achieved 0.01329) has done: 'Your current score (0.0087) is far below the target (0.2122), so we should increase mAP with the smallest safe change that improves box IoU and class relevance while keeping your core pipeline (train-derived priors → per-test scaling → TH=0.1 filter → No finding fallback) intact. The biggest issue is that you only emit 1 prior box per class, which is unlikely to overlap real findings at IoU>0.4; we emit up to 2 representative k-means components per class (still capped by the same MAX_BOXES_PER_IMAGE) to increase overlap chances without flooding predictions. To avoid adding noisy low-value classes, we keep the same `top_classes` list but make per-image class ordering deterministic and slightly vary which components are chosen per image (deterministically from the image_id hash), improving diversity without changing any downstream filtering semantics. All paths remain unchanged, runtime stays within limits (no extra DICOM reads beyond existing header-only shape read), and the script still writes a valid `./submission.csv`.'
- What this solution (achieved 0.01259) has done: 'Your current score (0.01329) is far below the target (0.21217), so we need a modest mAP lift without changing your overall pipeline (train-derived priors → per-test scaling → TH=0.1 filtering → No finding fallback). The biggest low-risk issue is that your priors include many boxes that are extremely large/small and/or very close to the image borders, which tends to miss IoU>0.4 on real findings and can also flood low-quality predictions. I keep your k-means prior approach and post-filtering logic intact, but (1) filter training boxes used for priors using robust size/area constraints, (2) make per-class base confidence slightly higher so more boxes survive the unchanged TH=0.1 filter, and (3) slightly reduce the per-image jitter/scale so boxes don’t drift away from the learned centers. These are minimal edits confined to prior generation and should move the score upward toward the target band while keeping runtime and I/O unchanged.'
- What this solution (achieved 0.01087) has done: 'Your current score (0.01259) is far below the target (0.21217), so we should increase mAP with the smallest change that improves IoU plausibility while keeping your exact pipeline (train-derived priors → per-test scaling → TH=0.1 filtering → No finding fallback) intact. The main weakness is that the priors ignore where findings typically occur relative to the lungs; many classes in VinBigData are strongly side-biased (left/right), so symmetric “global” priors often miss IoU>0.4. I compute simple per-class position priors from train boxes (normalized center x), split each class into left/right components when the data supports it, and then deterministically choose the appropriate component per test image by hashing image_id—this preserves your no-model approach and only tweaks prior generation. I also add a tiny, deterministic horizontal flip choice per image to better match left/right distribution without changing downstream filtering or thresholds, and keep the total boxes/image capped exactly as before.'

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
train_path = os.path.join(BASE_DIR, "train.csv")
if not os.path.exists(train_path):
    train_path = "/kaggle/input/train.csv"
train_df = pd.read_csv(train_path)

train_obj = train_df.loc[train_df["class_id"].between(0, 13)].copy()
for c in ["x_min", "y_min", "x_max", "y_max"]:
    train_obj[c] = pd.to_numeric(train_obj[c], errors="coerce")
train_obj = train_obj.dropna(subset=["class_id", "x_min", "y_min", "x_max", "y_max"])
train_obj = train_obj.loc[
    (train_obj["x_max"] > train_obj["x_min"])
    & (train_obj["y_max"] > train_obj["y_min"])
].copy()

train_obj["image_id"] = train_obj["image_id"].astype(str)
class_counts = train_obj["class_id"].value_counts().to_dict()
max_count = max(class_counts.values()) if class_counts else 1

train_img_ids = train_obj["image_id"].unique().tolist()


def _hash_u32(s: str) -> int:
    x = 2166136261
    for ch in s:
        x ^= ord(ch)
        x = (x * 16777619) & 0xFFFFFFFF
    return int(x)


MAX_TRAIN_DICOM_HEADERS = 2200
train_img_ids_sorted = sorted(train_img_ids, key=lambda s: _hash_u32(s))
train_img_ids_sub = train_img_ids_sorted[
    : min(MAX_TRAIN_DICOM_HEADERS, len(train_img_ids_sorted))
]

train_sizes = {}
train_dir = os.path.join(BASE_DIR, "train")
if not os.path.exists(train_dir):
    train_dir = "/kaggle/input/train"

for image_id in tqdm(train_img_ids_sub, desc="Reading train DICOM sizes (header-only)"):
    dpath = os.path.join(train_dir, f"{image_id}.dicom")
    H, W = None, None
    try:
        dcm = pydicom.dcmread(dpath, stop_before_pixels=True)
        if hasattr(dcm, "Rows") and hasattr(dcm, "Columns"):
            H, W = int(dcm.Rows), int(dcm.Columns)
    except Exception:
        H, W = None, None
    if H is not None and W is not None and H > 1 and W > 1:
        train_sizes[image_id] = (H, W)

sizes_df = pd.DataFrame(
    [{"image_id": k, "H": v[0], "W": v[1]} for k, v in train_sizes.items()]
)
if len(sizes_df) == 0:
    img_scale = (
        train_obj.groupby("image_id")[["x_max", "y_max"]]
        .max()
        .rename(columns={"x_max": "Wm", "y_max": "Hm"})
    )
    train_obj = train_obj.join(img_scale, on="image_id", how="left")
    train_obj["Wm"] = train_obj["Wm"].clip(lower=1.0)
    train_obj["Hm"] = train_obj["Hm"].clip(lower=1.0)
else:
    medH = float(sizes_df["H"].median())
    medW = float(sizes_df["W"].median())
    train_obj = train_obj.merge(sizes_df, on="image_id", how="left")
    train_obj["H"] = train_obj["H"].fillna(medH).clip(lower=2.0)
    train_obj["W"] = train_obj["W"].fillna(medW).clip(lower=2.0)
    train_obj = train_obj.rename(columns={"W": "Wm", "H": "Hm"})

train_obj["x_min_n"] = (train_obj["x_min"] / train_obj["Wm"]).clip(0.0, 1.0)
train_obj["x_max_n"] = (train_obj["x_max"] / train_obj["Wm"]).clip(0.0, 1.0)
train_obj["y_min_n"] = (train_obj["y_min"] / train_obj["Hm"]).clip(0.0, 1.0)
train_obj["y_max_n"] = (train_obj["y_max"] / train_obj["Hm"]).clip(0.0, 1.0)

w_n = (train_obj["x_max_n"] - train_obj["x_min_n"]).clip(lower=0.0, upper=1.0)
h_n = (train_obj["y_max_n"] - train_obj["y_min_n"]).clip(lower=0.0, upper=1.0)
area_n = w_n * h_n

asp = (w_n / np.clip(h_n, 1e-6, None)).clip(0.0, 1e6)
train_obj = train_obj.loc[
    (w_n >= 0.03)
    & (h_n >= 0.03)
    & (area_n >= 0.002)
    & (area_n <= 0.80)
    & (asp >= 0.15)
    & (asp <= 6.5)
].copy()


def _kmeans_np(
    X: np.ndarray, k: int = 3, iters: int = 35, seed: int = 111
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    if n == 0:
        return np.zeros((0, X.shape[1]), dtype=np.float32)
    k = int(min(k, n))
    rs = np.random.default_rng(seed)
    init_idx = rs.choice(n, size=k, replace=False)
    C = X[init_idx].copy()
    for _ in range(iters):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        lbl = d2.argmin(axis=1)
        C_new = C.copy()
        for j in range(k):
            m = lbl == j
            if m.any():
                C_new[j] = X[m].mean(axis=0)
        if np.max(np.abs(C_new - C)) < 1e-6:
            C = C_new
            break
        C = C_new
    return C


rep_boxes_n = {}
rep_area_n = {}

rep_boxes_n_lr = {}  # cid -> dict('L': reps, 'R': reps, 'A': reps)
rep_area_n_lr = {}

for cid in range(14):
    dfc = train_obj.loc[
        train_obj["class_id"] == cid, ["x_min_n", "y_min_n", "x_max_n", "y_max_n"]
    ]
    if len(dfc) == 0:
        continue

    Xcorn = dfc.to_numpy(dtype=np.float32)
    x1n, y1n, x2n, y2n = Xcorn[:, 0], Xcorn[:, 1], Xcorn[:, 2], Xcorn[:, 3]
    cx = 0.5 * (x1n + x2n)
    cy = 0.5 * (y1n + y2n)
    w = np.clip(x2n - x1n, 1e-4, 1.0)
    h = np.clip(y2n - y1n, 1e-4, 1.0)
    X = np.stack([cx, cy, w, h], axis=1).astype(np.float32)

    centers = _kmeans_np(X, k=3, iters=35, seed=111 + int(cid))
    reps = []
    areas = []
    for row in centers:
        cxj, cyj, wj, hj = [float(v) for v in row.tolist()]
        cxj = float(np.clip(cxj, 0.0, 1.0))
        cyj = float(np.clip(cyj, 0.0, 1.0))
        wj = float(np.clip(wj, 1e-3, 1.0))
        hj = float(np.clip(hj, 1e-3, 1.0))
        x1 = float(np.clip(cxj - 0.5 * wj, 0.0, 1.0))
        x2 = float(np.clip(cxj + 0.5 * wj, 0.0, 1.0))
        y1 = float(np.clip(cyj - 0.5 * hj, 0.0, 1.0))
        y2 = float(np.clip(cyj + 0.5 * hj, 0.0, 1.0))
        if x2 <= x1:
            x2 = min(1.0, x1 + 1e-3)
        if y2 <= y1:
            y2 = min(1.0, y1 + 1e-3)
        reps.append((x1, y1, x2, y2))
        areas.append(float((x2 - x1) * (y2 - y1)))
    rep_boxes_n[cid] = reps
    rep_area_n[cid] = areas

    cx_all = cx.astype(np.float32)
    left_mask = cx_all < 0.5
    right_mask = ~left_mask
    rep_boxes_n_lr[cid] = {}
    rep_area_n_lr[cid] = {}

    def _make_reps_from_mask(msk: np.ndarray, tag: str):
        Xm = X[msk]
        if Xm.shape[0] < 120:
            return None, None
        cent = _kmeans_np(
            Xm, k=2, iters=35, seed=222 + int(cid) + (0 if tag == "L" else 1)
        )
        reps_m = []
        areas_m = []
        for row2 in cent:
            cxj, cyj, wj, hj = [float(v) for v in row2.tolist()]
            cxj = float(np.clip(cxj, 0.0, 1.0))
            cyj = float(np.clip(cyj, 0.0, 1.0))
            wj = float(np.clip(wj, 1e-3, 1.0))
            hj = float(np.clip(hj, 1e-3, 1.0))
            x1 = float(np.clip(cxj - 0.5 * wj, 0.0, 1.0))
            x2 = float(np.clip(cxj + 0.5 * wj, 0.0, 1.0))
            y1 = float(np.clip(cyj - 0.5 * hj, 0.0, 1.0))
            y2 = float(np.clip(cyj + 0.5 * hj, 0.0, 1.0))
            if x2 <= x1:
                x2 = min(1.0, x1 + 1e-3)
            if y2 <= y1:
                y2 = min(1.0, y1 + 1e-3)
            reps_m.append((x1, y1, x2, y2))
            areas_m.append(float((x2 - x1) * (y2 - y1)))
        return reps_m, areas_m

    repsL, areasL = _make_reps_from_mask(left_mask, "L")
    repsR, areasR = _make_reps_from_mask(right_mask, "R")
    if repsL is not None and repsR is not None:
        rep_boxes_n_lr[cid]["L"] = repsL
        rep_boxes_n_lr[cid]["R"] = repsR
        rep_area_n_lr[cid]["L"] = areasL
        rep_area_n_lr[cid]["R"] = areasR

    rep_boxes_n_lr[cid]["A"] = reps
    rep_area_n_lr[cid]["A"] = areas


def class_base_conf(cid: int) -> float:
    cnt = class_counts.get(cid, 0)
    return float(0.26 + 0.30 * (cnt / max_count))


top_classes = sorted(
    [c for c in class_counts.keys() if 0 <= c <= 13],
    key=lambda c: class_counts[c],
    reverse=True,
)[:12]
if len(top_classes) == 0:
    top_classes = [10]


def _hash01(s: str) -> float:
    x = 2166136261
    for ch in s:
        x ^= ord(ch)
        x = (x * 16777619) & 0xFFFFFFFF
    return (x % 1000000) / 1000000.0


pred_strings = []
test_dir = os.path.join(BASE_DIR, "test")
if not os.path.exists(test_dir):
    test_dir = "/kaggle/input/test"

for image_id in tqdm(df2["image_id"].tolist(), desc="Building prior predictions"):
    dicom_path = os.path.join(test_dir, f"{image_id}.dicom")

    H, W = 1024, 1024
    try:
        dcm = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        if hasattr(dcm, "Rows") and hasattr(dcm, "Columns"):
            H, W = int(dcm.Rows), int(dcm.Columns)
    except Exception:
        pass

    u = _hash01(image_id)

    shift = (u - 0.5) * 0.0010
    scale = 1.0 + (u - 0.5) * 0.0020
    conf_jit = 1.0 + (u - 0.5) * 0.02

    side_pick = "L" if ((u * 131071) % 1.0) < 0.5 else "R"
    do_flip = ((u * 524287) % 1.0) < 0.35  # modest chance; deterministic per image

    parts = []
    total_boxes = 0
    MAX_BOXES_PER_IMAGE = 18

    if len(top_classes) > 1:
        rot = int((u * 9973) % len(top_classes))
        classes_this = top_classes[rot:] + top_classes[:rot]
    else:
        classes_this = top_classes

    for rank, cid in enumerate(classes_this):
        if cid not in rep_boxes_n_lr:
            continue
        if total_boxes >= MAX_BOXES_PER_IMAGE:
            break

        base = class_base_conf(cid) * conf_jit

        if ("L" in rep_boxes_n_lr[cid]) and ("R" in rep_boxes_n_lr[cid]):
            boxes = rep_boxes_n_lr[cid].get(side_pick, rep_boxes_n_lr[cid]["A"])
            areas = rep_area_n_lr[cid].get(side_pick, rep_area_n_lr[cid]["A"])
        else:
            boxes = rep_boxes_n_lr[cid]["A"]
            areas = rep_area_n_lr[cid]["A"]

        if len(areas) >= 1:
            target_area_n = float(np.median(np.asarray(areas, dtype=np.float32)))
            order = sorted(
                range(len(areas)), key=lambda j: abs(areas[j] - target_area_n)
            )
        else:
            order = [0]

        if len(order) >= 2:
            start = int((u * 10007) % len(order))
            use_idxs = [order[start], order[(start + 1) % len(order)]]
        else:
            use_idxs = order[:1]

        use_idxs = use_idxs[:2]

        for bi, bidx in enumerate(use_idxs):
            if total_boxes >= MAX_BOXES_PER_IMAGE:
                break
            x1n, y1n, x2n, y2n = boxes[bidx]

            cx = 0.5 * (x1n + x2n) + shift
            cy = 0.5 * (y1n + y2n) + shift
            bw = (x2n - x1n) * scale
            bh = (y2n - y1n) * scale
            x1nj = np.clip(cx - 0.5 * bw, 0.0, 1.0)
            x2nj = np.clip(cx + 0.5 * bw, 0.0, 1.0)
            y1nj = np.clip(cy - 0.5 * bh, 0.0, 1.0)
            y2nj = np.clip(cy + 0.5 * bh, 0.0, 1.0)

            if do_flip:
                x1nj, x2nj = (1.0 - x2nj), (1.0 - x1nj)
                x1nj, x2nj = float(np.clip(min(x1nj, x2nj), 0.0, 1.0)), float(
                    np.clip(max(x1nj, x2nj), 0.0, 1.0)
                )

            x1 = x1nj * W
            x2 = x2nj * W
            y1 = y1nj * H
            y2 = y2nj * H

            x1i, y1i = int(max(0, round(x1))), int(max(0, round(y1)))
            x2i = int(max(x1i + 1, min(W, round(x2))))
            y2i = int(max(y1i + 1, min(H, round(y2))))

            score = base * (0.93**rank) * (0.92**bi)
            parts.extend(
                [str(int(cid)), f"{score:.6f}", str(x1i), str(y1i), str(x2i), str(y2i)]
            )
            total_boxes += 1

    pred_strings.append(" ".join(parts).strip())

df2["PredictionString"] = pred_strings
df2.head()



## === cell 3
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



## === cell 4
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



## === cell 5
if "xray" in globals() and "predicted" in globals():
    pass



## === cell 6
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



## === cell 7
out_path = "./submission.csv"
df_out[["image_id", "PredictionString"]].to_csv(out_path, index=False)

print("Wrote:", out_path)
print(df_out.shape)
print(df_out.columns.tolist())
print(df_out.head(3))
