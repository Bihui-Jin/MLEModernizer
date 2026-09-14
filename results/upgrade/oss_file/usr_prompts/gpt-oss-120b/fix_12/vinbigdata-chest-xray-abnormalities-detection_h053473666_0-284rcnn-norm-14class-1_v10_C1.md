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

# 5. Target score

0.2615307101474436

# 6. Current score

0.0475

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0475) has done: 'I replace the failing ensemble loading with a simple, reliable pipeline that reads the provided `sample_submission.csv`, ensures the required “no finding” format is present, and writes it out as `submission.csv`. This removes missing‑file errors, guarantees a correctly‑structured CSV, and produces a valid submission without altering any core modeling logic.'
- What this solution (achieved 0.0475) has done: 'I replace the placeholder “no‑finding” predictions with a simple heuristic that uses the training data: for each test image we predict the three most frequent disease classes, assigning each a confidence proportional to its overall frequency and using the average bounding box for that class across the whole training set. This adds informative predictions while keeping the overall pipeline unchanged, which should raise the mAP from the current 0.0475 toward the target 0.2615.'
- What this solution (achieved 0.0475) has done: 'I fixed the NaN‑conversion error when computing average bounding boxes by filling missing values before casting to integers. This ensures the pipeline runs end‑to‑end and produces a valid `submission.csv`. The rest of the logic (using the three most frequent classes with frequency‑based confidence) is kept unchanged, which should improve the mAP toward the target.'
- What this solution (achieved 0.0475) has done: 'I increase the number of frequent disease classes predicted for every test image from the previous three to the top ten most common classes. This adds more potentially correct detections while keeping the same confidence‑frequency weighting and bbox averaging, so it should raise the mAP toward the target without altering the core pipeline.'
- What this solution (achieved 0.0475) has done: 'We raise the mAP by reducing the number of generic predictions per image, limiting them to the most frequent disease classes that have a meaningful prior (frequency ≥ 0.01) and capping the list at three classes. This cuts many false positives while still providing useful detections, moving the score toward the target. The core logic of loading data, computing frequencies, and averaging bboxes stays unchanged; only the selection of classes and the handling of empty predictions are tweaked.'
- What this solution (achieved 0.0475) has done: 'I lower the frequency threshold to keep every class and raise the maximum number of predicted classes per image from 3 to 10. This adds many more reasonable dete‑ctions for each test image while retaining the original averaging and confidence logic, moving the mAP closer to the target value.'
- What this solution (achieved 0.0475) has done: 'I tighten the heuristic by predicting only the most common disease classes (frequency ≥ 0.01) and limiting the per‑image list to the top 5 classes instead of up to 10. This reduces many unlikely false‑positive boxes, which should improve precision and therefore raise the mAP toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.0475) has done: 'I raise the number of globally‑predicted disease classes and remove the frequency‑filtering so that each test image receives more plausible boxes, which should increase recall and move the mAP upward toward the target score. The only modifications are the `FREQ_THRESHOLD` and `MAX_CLASSES` constants in the preprocessing cell.'
- What this solution (achieved 0.0475) has done: 'I diversify the predictions per image instead of assigning the same ten classes to every test case. By selecting a deterministic subset of the most frequent classes based on each image’s ID, we increase the chance of matching true objects while keeping the original heuristic and bounding‑box logic unchanged. This small change should raise the mAP toward the target without altering the core pipeline.'
- What this solution (achieved 0.0475) has done: 'I increase the number of predicted classes per image and broaden the global pool, then replace the deterministic consecutive‑selection with a hash‑seeded random sample. This adds more relevant detections while keeping the heuristic deterministic, which is expected to raise the mAP toward the target score.'
- What this solution (achieved 0.0475) has done: 'I increased the recall of the heuristic by expanding the global pool of frequent disease classes (MAX_GLOBAL_CLASSES = 20) and letting each test image receive up to ten predictions (PER_IMAGE_MAX = 10).  To obtain tighter bounding boxes I switched the pooled box coordinates from a mean to a median, which typically yields boxes that overlap true objects better and thus improves mAP.  These small parameter tweaks keep the overall pipeline unchanged while moving the score closer to the target.'

# 9. Code solution

## === cell 0
import pandas as pd
from pathlib import Path
import hashlib
import random  # added for deterministic random sampling



## === cell 1
sample_path = Path("../input/sample_submission.csv")
if not sample_path.is_file():
    sample_path = Path(
        "../input/vinbigdata-chest-xray-abnormalities-detection/sample_submission.csv"
    )
submission_df = pd.read_csv(sample_path)



## === cell 2
train_path = Path("../input/train.csv")
if not train_path.is_file():
    train_path = Path(
        "../input/vinbigdata-chest-xray-abnormalities-detection/train.csv"
    )
train_df = pd.read_csv(train_path)

class_counts = train_df["class_id"].value_counts().sort_values(ascending=False)
total_counts = class_counts.sum()
class_freq = class_counts / total_counts  # relative frequency

avg_bbox = (
    train_df.groupby("class_id")[["x_min", "y_min", "x_max", "y_max"]]
    .median()
    .round()
    .fillna(0)  # replace NaN with 0 to allow integer casting
    .astype(int)
)

FREQ_THRESHOLD = 0.0  # keep every class
MAX_GLOBAL_CLASSES = 20  # larger global pool
PER_IMAGE_MAX = 10  # predict more classes per image

filtered_classes = [cls for cls, freq in class_freq.items() if freq >= FREQ_THRESHOLD]
top_classes = filtered_classes[:MAX_GLOBAL_CLASSES]  # global list used for all images


def select_classes_for_image(image_id: str):
    """
    Deterministically pick a subset of `top_classes` for the given image.
    Uses an MD5 hash of the image_id as a seed for a Random instance,
    then samples up to PER_IMAGE_MAX classes without replacement.
    """
    if not top_classes:
        return []
    h = int(hashlib.md5(image_id.encode()).hexdigest(), 16)
    rng = random.Random(h)
    k = min(PER_IMAGE_MAX, len(top_classes))
    return rng.sample(top_classes, k)


def build_prediction_string(image_id: str):
    """
    Build a prediction string for a single image using the selected classes.
    If the global list is empty, return the required “no finding” format.
    """
    chosen = select_classes_for_image(image_id)
    if not chosen:
        return "14 1 0 0 1 1"
    parts = []
    for cls in chosen:
        conf = round(
            class_freq.get(cls, 0.0), 3
        )  # confidence limited to three decimals
        bbox = avg_bbox.loc[cls]
        parts.extend(
            [
                str(cls),
                str(conf),
                str(bbox["x_min"]),
                str(bbox["y_min"]),
                str(bbox["x_max"]),
                str(bbox["y_max"]),
            ]
        )
    return " ".join(parts)


def apply_heuristic(row):
    pred = row["PredictionString"]
    if isinstance(pred, str) and pred.strip():
        return pred
    return build_prediction_string(row["image_id"])


submission_df["PredictionString"] = submission_df.apply(apply_heuristic, axis=1)



## === cell 3
submission_df["PredictionString"] = submission_df["PredictionString"].replace(
    r"^\s*$", "14 1 0 0 1 1", regex=True
)



## === cell 4
output_path = Path("submission.csv")
submission_df.to_csv(output_path, index=False)
