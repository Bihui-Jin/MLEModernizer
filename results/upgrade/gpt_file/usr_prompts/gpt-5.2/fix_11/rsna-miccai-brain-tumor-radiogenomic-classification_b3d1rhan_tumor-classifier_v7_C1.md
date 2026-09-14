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
Predict the genetic subtype of glioblastoma using MRI (magnetic resonance imaging) scans to detect for the presence of MGMT promoter methylation.

## Metric
Area under the ROC curve between the predicted probability and the observed target.

## Submission Format
For each `BraTS21ID` in the test set, you must predict a probability for the target `MGMT_value`. The file should contain a header and have the following format:

```
BraTS21ID,MGMT_value
00001,0.5
00013,0.5
00015,0.5
etc.
```

## Dataset
- **train/** - folder containing the training files, with each top-level folder representing a subject. **NOTE:** There are some unexpected issues with the following three cases in the training dataset, participants can exclude the cases during training: `[00109, 00123, 00709]`. We have checked and confirmed that the testing dataset is free from such issues.
- **train_labels.csv** - file containing the target `MGMT_value` for each subject in the training data (e.g. the presence of MGMT promoter methylation)
- **test/** - the test files, which use the same structure as `train/`; your task is to predict the `MGMT_value` for each subject in the test data. **NOTE**: the total size of the rerun test set (Public and Private) is ~5x the size of the Public test set
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
protobuf==6.33.0
pydicom==3.0.1
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        input/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        working/
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
```

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> (stopped after 10 files for performance)

# 5. Target score

0.5790413054563998

# 6. Current score

0.65412

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.44706) has done: 'I fix the immediate runtime errors by updating the deprecated `pydicom.read_file` call, avoiding crashing on existing directories, and removing the TensorFlow/Keras import/version call that triggers the protobuf `GetPrototype` error in this environment. Because the referenced pretrained model file does not exist, I replace the missing-model dependency with a minimal, competition-valid fallback that produces probabilistic predictions and always writes a correctly formatted `submission.csv`. I also fix the prediction loop and alignment logic so the number of predictions always matches the 59 test IDs, preventing the length-mismatch error when building the submission. These changes are focused on making the pipeline run end-to-end and produce a valid submission CSV.'
- What this solution (achieved 0.65412) has done: 'The timeout is dominated by repeatedly reading and fully decoding every DICOM slice for every case (train + test), then normalizing each image in the fallback model by scanning all pixels again. I keep the exact same feature (mean intensity of min-max normalized pixels) and the same decider/calibration logic, but make it faster by (1) avoiding creation of unnecessary work directories, (2) using a much faster DICOM read path that skips non-pixel metadata, (3) computing the fallback “prediction” in a single streaming pass (mean/min/max) without building a huge flattened array, and (4) reducing Python overhead by prelisting files once per case and using local-variable bindings. These changes are provably equivalent (same per-slice normalization and mean), so accuracy/evaluation semantics stay the same while runtime drops substantially.'
- What this solution (achieved 0.65412) has done: 'Your current score (0.65412 AUC) is higher than the target (0.57904), so the safest way to move *toward* the target is to slightly reduce model discrimination without changing the pipeline’s core feature extraction, prediction loop, or calibration training logic. I do this by (1) tightening the “high-confidence slice” filter so fewer slices contribute to the per-case mean, and (2) blending the calibrated probability a bit more toward 0.5 (a monotonic shrink toward 0.5 reduces AUC magnitude in expectation). These are minimal, metric-relevant post-processing adjustments that preserve valid probabilistic outputs and keep runtime essentially unchanged. The script still run end-to-end and write a correct `submission.csv`.'
- What this solution (achieved 0.65412) has done: 'Your current AUC (0.65412) is higher than the target (0.57904), so we should *slightly reduce* discrimination to move closer to the target rather than improve it. To do this with minimal risk and without changing the core feature extraction, model, calibration training, or loops, I only adjust the two existing “de-discrimination” knobs already present: the confidence slice filter and the shrink-to-0.5 blend. Specifically, I relax the confidence filter back toward keeping more slices (reduces variance and tends to wash out per-case differences) and shrink probabilities a bit more toward 0.5 (monotonic compression that typically lowers AUC). Everything else stays identical, and the script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.65412) has done: 'Your current AUC (0.65412) is above the target (0.57904), so to move closer we should *slightly reduce* discrimination while keeping the same feature extraction, prediction loop, and calibration logic. I make only minimal post-processing changes by increasing the existing shrink-to-0.5 blend (stronger compression of probabilities toward 0.5 typically lowers AUC) and by slightly loosening the confidence-based slice selection so per-case predictions get averaged over more slices (washing out differences). These changes preserve valid probability outputs, keep runtime essentially unchanged, and still write a correct `submission.csv`. No model/loss/training loop or feature definition is changed.'
- What this solution (achieved 0.65412) has done: 'Your current AUC (0.65412) is above the target (0.57904), so the smallest safe move toward the target is to slightly *reduce* discrimination without changing feature extraction, the fallback model, or calibration training. I only adjust the existing post-processing “shrink-to-0.5” knob to compress probabilities more strongly toward 0.5, which typically lowers AUC while keeping valid probabilistic outputs. Everything else (DICOM reading, per-slice feature, decider, calibration loop, submission alignment) is kept identical to preserve core logic and runtime. The script still runs end-to-end and writes a correct `submission.csv`.'
- What this solution (achieved 0.65412) has done: 'Your current AUC (0.65412) is above the target (0.57904), so the smallest safe step toward the target is to slightly *reduce* discrimination while keeping the same feature extraction, fallback model, calibration training, and prediction loop. I only adjust the existing probability shrink-to-0.5 knob (post-processing) to compress predictions more strongly toward 0.5, which generally lowers AUC without changing ranking logic elsewhere. All I/O paths, DICOM reading, decider logic, and submission formatting remain unchanged, and the script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.65412) has done: 'Your current AUC (0.65412) is above the target (0.57904), so we should slightly *reduce* discrimination to move closer rather than improve it. The smallest, lowest-risk way (without changing feature extraction, fallback model, calibration training, or loops) is to increase the existing shrink-to-0.5 post-processing so predictions are compressed more toward 0.5, which typically lowers AUC. I keep everything else identical and only adjust `_PROB_SHRINK_ALPHA`. The script still run end-to-end and write a valid `submission.csv` with the correct columns and row alignment.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os



## === cell 1
BASE_INPUT = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
TEST_DIR = os.path.join(BASE_INPUT, "test")
TRAIN_DIR = os.path.join(BASE_INPUT, "train")
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")
TRAIN_LABELS_PATH = os.path.join(BASE_INPUT, "train_labels.csv")

WORK_TESTI = "./testi"
WORK_TRAINI = "./traini"



## === cell 2
os.makedirs(WORK_TESTI, exist_ok=True)
os.makedirs(WORK_TRAINI, exist_ok=True)



## === cell 3
import cv2
import pydicom

cases_list = sorted(
    [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]
)



## === cell 4
train_cases_list = sorted(
    [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
)
bad_cases = set(["00109", "00123", "00709"])  # known problematic cases per dataset note



## === cell 5
if len(cases_list) > 0:
    flair0 = os.path.join(TEST_DIR, cases_list[0], "FLAIR")
    if os.path.isdir(flair0):
        liste_names = os.listdir(flair0)
        if len(liste_names) > 0:
            print(liste_names[0])



## === cell 6
from PIL import Image




## === cell 7
class FallbackModel:
    def predict(self, x, batch_size=32, verbose=0):
        x = np.asarray(x)
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (n,H,W,1), got shape {x.shape}")

        x3 = x[..., 0].astype(np.float32, copy=False)

        mins = x3.reshape((x3.shape[0], -1)).min(axis=1)
        maxs = x3.reshape((x3.shape[0], -1)).max(axis=1)
        means = x3.reshape((x3.shape[0], -1)).mean(axis=1)

        denom = np.maximum(maxs - mins, 1e-6).astype(np.float32, copy=False)
        s = ((means - mins) / denom).astype(
            np.float32, copy=False
        )  # identical to mean(flatn)

        p1 = 0.15 + 0.70 * s
        p0 = 1.0 - p1
        return np.stack([p1, p0], axis=1).astype(np.float32, copy=False)


model = FallbackModel()



## === cell 8
_CASES_CACHE = {}  # key: source_dir -> list of case_ids


def _get_cases_list_for_sourcedir(source_dir):
    if source_dir in _CASES_CACHE:
        return _CASES_CACHE[source_dir]
    cases = sorted(
        [
            d
            for d in os.listdir(source_dir)
            if os.path.isdir(os.path.join(source_dir, d))
        ]
    )
    _CASES_CACHE[source_dir] = cases
    return cases


def _resolve_source_flair_dir(source_dir, case_id):
    return os.path.join(source_dir, case_id, "FLAIR")


def _read_dicom_pixel_array(dcm_path):
    ds = pydicom.dcmread(
        dcm_path,
        force=True,
        specific_tags=(
            "PixelData",
            "Rows",
            "Columns",
            "BitsAllocated",
            "BitsStored",
            "HighBit",
            "PixelRepresentation",
            "SamplesPerPixel",
            "PhotometricInterpretation",
            "PlanarConfiguration",
            "RescaleIntercept",
            "RescaleSlope",
        ),
    )
    return ds.pixel_array


def _to_128_padded_grayscale(arr2d):
    arr = np.asarray(arr2d)
    if arr.ndim != 2:
        arr = np.asarray(arr[0])
    h, w = arr.shape[:2]
    if h > 128 or w > 128:
        scale = min(128.0 / float(w), 128.0 / float(h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = arr.shape[:2]
    out = np.zeros((128, 128), dtype=np.float32)
    out[:h, :w] = arr.astype(np.float32, copy=False)
    return out


def tester_function(index, model, source_dir, prelisted_cases=None):
    join = os.path.join
    isdir = os.path.isdir
    listdir = os.listdir
    lower = str.lower

    image_list = []

    cases_list_local = (
        prelisted_cases
        if prelisted_cases is not None
        else _get_cases_list_for_sourcedir(source_dir)
    )

    exist_value = 0
    if index < 0 or index >= len(cases_list_local):
        return np.zeros((0, 2), dtype=np.float32), 0, None

    case_id = cases_list_local[index]
    flair_dir = _resolve_source_flair_dir(source_dir, case_id)
    if not isdir(flair_dir):
        return np.zeros((0, 2), dtype=np.float32), 0, case_id

    liste_names = sorted(listdir(flair_dir))
    if len(liste_names) == 0:
        return np.zeros((0, 2), dtype=np.float32), 0, case_id

    exist_value = 1
    for k in liste_names:
        img_path = join(flair_dir, k)
        try:
            if lower(k).endswith(".dcm"):
                arr = _read_dicom_pixel_array(img_path)
                out = _to_128_padded_grayscale(arr)
            else:
                img = Image.open(img_path).convert("L")
                img.thumbnail((128, 128), Image.Resampling.LANCZOS)
                arr = np.array(img, dtype=np.float32)
                out = np.zeros((128, 128), dtype=np.float32)
                h = min(arr.shape[0], 128)
                w = min(arr.shape[1], 128)
                out[:h, :w] = arr[:h, :w]
        except Exception:
            continue

        image_list.append(out)

    if len(image_list) == 0:
        return np.zeros((0, 2), dtype=np.float32), 0, case_id

    test_case = np.stack(image_list, axis=0)
    test_case = test_case.reshape(
        test_case.shape[0], test_case.shape[1], test_case.shape[2], 1
    )
    y_pred = model.predict(test_case)
    return y_pred, exist_value, case_id




## === cell 9
_CONF_KEEP_THRESHOLD = 0.80  # keep core decider logic unchanged


def decider_func(y_pred, exist_value):
    if exist_value != 1 or y_pred is None or len(y_pred) == 0:
        return 0.5

    y_pred = np.asarray(y_pred)
    if y_pred.ndim != 2 or y_pred.shape[1] < 2:
        return 0.5

    p1 = y_pred[:, 0].astype(np.float32)

    conf = np.abs(y_pred[:, 0] - y_pred[:, 1]).astype(np.float32)
    keep = conf >= _CONF_KEEP_THRESHOLD
    if np.any(keep):
        return float(np.clip(p1[keep].mean(), 0.0, 1.0))
    return float(np.clip(p1.mean(), 0.0, 1.0))




## === cell 10
train_labels = pd.read_csv(TRAIN_LABELS_PATH)
train_labels["BraTS21ID"] = train_labels["BraTS21ID"].astype(str).str.zfill(5)

train_feat_by_id = {}
train_cases = _get_cases_list_for_sourcedir(TRAIN_DIR)
for idx in range(len(train_cases)):
    cid0 = train_cases[idx]
    if cid0 in bad_cases:
        continue
    y_pred, exist_value, cid = tester_function(
        idx, model, TRAIN_DIR, prelisted_cases=train_cases
    )
    if cid is None:
        continue
    if cid in bad_cases:
        continue
    feat = decider_func(y_pred, exist_value)
    train_feat_by_id[cid] = float(feat)

xs = []
ys = []
for row in train_labels.itertuples(index=False):
    cid = row.BraTS21ID
    if cid in bad_cases:
        continue
    v = train_feat_by_id.get(cid, None)
    if v is not None:
        xs.append(v)
        ys.append(int(row.MGMT_value))

xs = np.asarray(xs, dtype=np.float32)
ys = np.asarray(ys, dtype=np.float32)

use_calibration = xs.size >= 50 and (ys.min() != ys.max())




## === cell 11
def _sigmoid(z):
    z = np.asarray(z, dtype=np.float32)
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


if use_calibration:
    a = np.float32(0.0)
    b = np.float32(0.0)

    x_mean = xs.mean().astype(np.float32)
    x_std = xs.std().astype(np.float32)
    if x_std < 1e-6:
        use_calibration = False
else:
    a = b = x_mean = x_std = np.float32(0.0)

if use_calibration:
    xz = (xs - x_mean) / x_std

    lr = np.float32(0.1)
    for _ in range(400):
        p = _sigmoid(a * xz + b)
        da = np.mean((p - ys) * xz).astype(np.float32)
        db = np.mean(p - ys).astype(np.float32)
        a -= lr * da
        b -= lr * db

    def calibrate_prob(x_scalar):
        x_scalar = np.float32(x_scalar)
        z = (x_scalar - x_mean) / x_std
        return float(_sigmoid(a * z + b))

else:

    def calibrate_prob(x_scalar):
        return float(np.clip(np.float32(x_scalar), 0.0, 1.0))


print("Calibration enabled:", use_calibration)



## === cell 12
test_cases = _get_cases_list_for_sourcedir(TEST_DIR)

_PROB_SHRINK_ALPHA = 0.03  # was 0.06

pred_by_id = {}
for idx in range(len(test_cases)):
    y_pred, exist_value, cid = tester_function(
        idx, model, TEST_DIR, prelisted_cases=test_cases
    )
    prob_raw = decider_func(y_pred, exist_value)
    prob_cal = calibrate_prob(prob_raw)
    prob = float(0.5 + _PROB_SHRINK_ALPHA * (prob_cal - 0.5))
    pred_by_id[test_cases[idx]] = float(np.clip(prob, 0.0, 1.0))



## === cell 13
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)



## === cell 14
index_list = sample_sub["BraTS21ID"].astype(str).str.zfill(5).tolist()



## === cell 15
new_predictions = [float(pred_by_id.get(case_id, 0.5)) for case_id in index_list]



## === cell 16
tahmin = np.array(new_predictions, dtype=np.float32)
tucker = np.array(index_list)
print(tahmin.shape)
print(tucker.shape)



## === cell 17
final_submission = pd.DataFrame({"BraTS21ID": tucker, "MGMT_value": tahmin})



## === cell 18
final_submission["BraTS21ID"] = final_submission["BraTS21ID"].astype(str).str.zfill(5)
final_submission["MGMT_value"] = (
    final_submission["MGMT_value"].astype(float).clip(0.0, 1.0)
)



## === cell 19
final_submission.to_csv("submission.csv", index=False)
print(final_submission.head())
print("Wrote submission.csv with shape:", final_submission.shape)
