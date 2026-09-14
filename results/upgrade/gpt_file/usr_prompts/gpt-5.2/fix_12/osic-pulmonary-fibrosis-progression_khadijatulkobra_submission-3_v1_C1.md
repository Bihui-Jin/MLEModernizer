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
Predict a patient’s severity of decline in lung function based on a CT scan of their lungs. Lung function is assessed based on output from a spirometer, which measures the forced vital capacity (`FVC`), i.e. the volume of air exhaled.

## Metric
A modified version of the Laplace Log Likelihood. 

For each true FVC measurement, you will predict both an FVC and a confidence measure (standard deviation 𝜎𝜎). The metric is computed as:

$$
\begin{gathered}
\sigma_{\text {clipped }}=\max (\sigma, 70), \\
\Delta=\min \left(\left|F V C_{\text {true }}-F V C_{\text {predicted }}\right|, 1000\right), \\
\text { metric }=-\frac{\sqrt{2} \Delta}{\sigma_{\text {clipped }}}-\ln \left(\sqrt{2} \sigma_{\text {clipped }}\right) .
\end{gathered}
$$

The error is thresholded at 1000 ml to avoid large errors adversely penalizing results, while the confidence values are clipped at 70 ml to reflect the approximate measurement uncertainty in FVC. The final score is calculated by averaging the metric across all test set `Patient_Week`s (three per patient). 

Metric values will be negative and higher is better.

## Submission Format
For each `Patient_Week`, you must predict the `FVC` and a confidence. You are asked to predict every patient's `FVC` measurement for every possible week. Those weeks which are not in the final three visits are ignored in scoring.

The file should contain a header and have the following format:

```
Patient_Week,FVC,Confidence
ID00002637202176704235138_1,2000,100
ID00002637202176704235138_2,2000,100
ID00002637202176704235138_3,2000,100
etc.

```

## Dataset
In the dataset, you are provided with a baseline chest CT scan and associated clinical information for a set of patients. A patient has an image acquired at time `Week = 0` and has numerous follow up visits over the course of approximately 1-2 years, at which time their `FVC` is measured.

- In the training set, you are provided with an anonymized, baseline CT scan and the entire history of FVC measurements.
- In the test set, you are provided with a baseline CT scan and only the initial FVC measurement. **You are asked to predict the final three `FVC` measurements for each patient, as well as a confidence value in your prediction.**

- **train.csv** - the training set, contains full history of clinical information
- **test.csv** - the test set, contains only the baseline measurement
- **train/** - contains the training patients' baseline CT scan in DICOM format
- **test/** - contains the test patients' baseline CT scan in DICOM format
- **sample_submission.csv** - demonstrates the submission format

**train.csv and test.csv**

- `Patient`a unique Id for each patient (also the name of the patient's DICOM folder)
- `Weeks`the relative number of weeks pre/post the baseline CT (may be negative)
- `FVC` - the recorded lung capacity in ml
- `Percent`a computed field which approximates the patient's FVC as a percent of the typical FVC for a person of similar characteristics
- `Age`
- `Sex`
- `SmokingStatus`

**sample submission.csv**

- `Patient_Week` - a unique Id formed by concatenating the `Patient` and `Weeks` columns (i.e. ABC_22 is a prediction for patient ABC at week 22)
- `FVC` - the predicted FVC in ml
- `Confidence` - a confidence value of your prediction (also has units of ml)

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        input/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        working/
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
```

-> data/osic-pulmonary-fibrosis-progression/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/osic-pulmonary-fibrosis-progression/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/osic-pulmonary-fibrosis-progression/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> (stopped after 10 files for performance)

# 5. Target score

-6.8779462154133455

# 6. Current score

-24.3347

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -24.65932) has done: 'I fix the training pipeline so it no longer crashes when CT images fail to load (the current empty `train_imgs` leads to `np.stack` on an empty list). To preserve the existing model/training logic, I add a safe fallback that uses a single zero-volume CT for all training rows if no images can be loaded, ensuring the model object is always defined and inference can run. I also make the DICOM loader more robust by ignoring non-DICOM files and sorting by DICOM metadata when available, which should allow real images to load in the Kaggle environment and improve score versus the all-zero fallback. Finally, I keep the submission formatting identical and ensure `submission.csv` is always written.'
- What this solution (achieved -24.65932) has done: 'Your current score is far below target, so the safest way to move it up is to fix a metric-semantics mismatch: the model outputs three quantiles, but you’re submitting the middle output as FVC and using (q75−q25) as “Confidence” without enforcing the required ordering/positivity. I keep the same architecture/training loop/loss, but at inference I (1) sort the three outputs so q25≤q50≤q75, (2) use q50 as FVC, and (3) compute Confidence as (q75−q25) with the competition’s sigma clipping. This is a minimal post-processing change that typically improves OSIC LaplaceLL substantially without altering training. I also cast predictions to float and ensure submission alignment stays identical.'
- What this solution (achieved -24.65932) has done: 'Your current score is far below the target, so we should increase it with the smallest changes that keep your model/training intact. The biggest avoidable score loss here is that the test-side features are built differently than the train-side features: in training, `Healthy-FVC` is derived from the row’s `FVC` (the visit FVC), while in test you derive it from `base_FVC`; that mismatch harms generalization. I align test preprocessing to exactly match `csv_preprocess` semantics by using `Healthy-FVC = base_FVC*100/Percent` (same as train’s baseline row after preprocessing) and, crucially, by ordering/including the exact same feature columns in the same order as in training. I also ensure `Confidence` is strictly positive before clipping (avoid pathological cases when quantiles collapse), without changing the model or loss.'
- What this solution (achieved -24.78914) has done: 'Your current score is far below the target (higher is better), and the biggest “minimal change” gain is to fix a train/inference mismatch: you train on every visit as a separate row (same baseline CT repeated) but your inference only uses the baseline CT once, without modeling patient-specific progression. Without changing your model/loss/loop, we can improve predictions by computing a per-patient linear trend (slope) from the training set and then applying that slope to adjust the model’s baseline-centered prediction across weeks, which typically moves OSIC scores up a lot. We also calibrate `Confidence` per patient using the residual spread of that slope-fit (clipped to the competition minimum), which is metric-aligned and stable. All changes are confined to post-processing at inference time and add a small, legitimate clinical prior while preserving your architecture and training semantics.'
- What this solution (achieved -24.3347) has done: 'Your current score is far below the target (higher is better), so we should improve it with the smallest changes that keep your model/loss/training loop intact. The biggest avoidable issue in your current inference is double-counting disease progression: your NN already sees `Week` as an input feature, then you add an extra `slope * Week` adjustment on top, which typically over-corrects and hurts LaplaceLL. I keep the same model and slope fit, but change the post-processing to use the slope only to build a patient-specific confidence (sigma) and to softly nudge the NN prediction toward a linear prior via a small blend, rather than adding the full slope term. This preserves your existing approach while making the inference calibration more metric-aligned and usually much less error-prone.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import pydicom
import scipy.ndimage

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = (
        True  # speeds up fixed-shape Conv3D without changing semantics
    )




## === cell 1
def resolve_base_dir():
    """
    Resolve the directory that directly contains train.csv/test.csv/sample_submission.csv.
    """
    candidates = [
        "/kaggle/input/osic-pulmonary-fibrosis-progression",
        "/kaggle/data/osic-pulmonary-fibrosis-progression",
        "/kaggle/data/input/osic-pulmonary-fibrosis-progression",
        "../input/osic-pulmonary-fibrosis-progression",
        "../data/osic-pulmonary-fibrosis-progression",
        "../data",
        "/kaggle/input",
        "/kaggle/data",
    ]

    def is_base_dir(p):
        return (
            os.path.exists(os.path.join(p, "train.csv"))
            and os.path.exists(os.path.join(p, "test.csv"))
            and os.path.exists(os.path.join(p, "sample_submission.csv"))
        )

    for c in candidates:
        if not os.path.exists(c):
            continue

        if os.path.isdir(c) and is_base_dir(c):
            return c

        nested = os.path.join(c, "osic-pulmonary-fibrosis-progression")
        if os.path.isdir(nested) and is_base_dir(nested):
            return nested

    raise FileNotFoundError(
        "Could not resolve OSIC dataset base dir from known candidates."
    )


BASE_DIR = resolve_base_dir()
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")
TRAIN_FOLDER = os.path.join(BASE_DIR, "train")
TEST_FOLDER = os.path.join(BASE_DIR, "test")

print("BASE_DIR:", BASE_DIR)
print("TRAIN_CSV:", TRAIN_CSV, "exists:", os.path.exists(TRAIN_CSV))
print("TEST_CSV:", TEST_CSV, "exists:", os.path.exists(TEST_CSV))
print("SAMPLE_SUB:", SAMPLE_SUB, "exists:", os.path.exists(SAMPLE_SUB))
print("TRAIN_FOLDER:", TRAIN_FOLDER, "exists:", os.path.exists(TRAIN_FOLDER))
print("TEST_FOLDER:", TEST_FOLDER, "exists:", os.path.exists(TEST_FOLDER))

CACHE_DIR = os.path.join("/kaggle/working", "ct_cache_z100_y200_x200_uint8")
os.makedirs(CACHE_DIR, exist_ok=True)
print("CACHE_DIR:", CACHE_DIR)




## === cell 2
def load_scan(path):
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    files.sort()
    dsets = []
    for s in files:
        fp = os.path.join(path, s)
        try:
            ds = pydicom.dcmread(
                fp,
                force=True,
                stop_before_pixels=True,
                specific_tags=[
                    "ImagePositionPatient",
                    "InstanceNumber",
                    "RescaleIntercept",
                    "RescaleSlope",
                ],
            )
            ds._fp = fp  # stash path for later full read
            dsets.append(ds)
        except Exception:
            continue

    if len(dsets) == 0:
        return []

    try:
        dsets.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except Exception:
        try:
            dsets.sort(key=lambda x: int(getattr(x, "InstanceNumber", 0)))
        except Exception:
            pass

    slices = []
    for ds in dsets:
        try:
            full = pydicom.dcmread(
                ds._fp,
                force=True,
                stop_before_pixels=False,
                specific_tags=[
                    "ImagePositionPatient",
                    "InstanceNumber",
                    "RescaleIntercept",
                    "RescaleSlope",
                    "PixelData",
                ],
            )
            slices.append(full)
        except Exception:
            continue
    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices]).astype(np.int16, copy=False)
    try:
        image[image <= -2000] = 0

        slopes = np.array(
            [float(getattr(s, "RescaleSlope", 1.0)) for s in slices], dtype=np.float32
        )
        intercepts = np.array(
            [float(getattr(s, "RescaleIntercept", 0.0)) for s in slices],
            dtype=np.float32,
        )

        if not np.all(slopes == 1.0) or not np.all(intercepts == 0.0):
            img_f = image.astype(np.float32, copy=False)
            img_f *= slopes[:, None, None]
            img_f += intercepts[:, None, None]
            image = img_f.astype(np.int16, copy=False)
    except Exception:
        print("HU conversion Failed!!")
    return np.array(image, dtype=np.int16)


def resize_along_allaxis(
    slices, target_dimensionZ=30, target_dimensionY=100, target_dimensionX=100
):
    present_dimensionZ, present_dimensionY, present_dimensionX = (
        slices.shape[0],
        slices.shape[1],
        slices.shape[2],
    )
    if (
        target_dimensionZ == present_dimensionZ
        and target_dimensionY == present_dimensionY
        and target_dimensionX == present_dimensionX
    ):
        return slices
    zoom_factorZ = float(target_dimensionZ) / float(present_dimensionZ)
    zoom_factorY = float(target_dimensionY) / float(present_dimensionY)
    zoom_factorX = float(target_dimensionX) / float(present_dimensionX)
    resize_image = scipy.ndimage.zoom(
        slices,
        [zoom_factorZ, zoom_factorY, zoom_factorX],
        mode="nearest",
        order=1,
        prefilter=False,
    )
    return resize_image


MIN_BOUND = -1000.0
MAX_BOUND = 400.0


def image_normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.0
    image[image < 0] = 0.0
    return image


def read_image(dir_name, patientid, Z=100, Y=200, X=200):
    path = os.path.join(dir_name, patientid)
    slices = load_scan(path)
    if len(slices) == 0:
        return None
    try:
        image_array = get_pixels_hu(slices)
        ctimage_resizedAll = resize_along_allaxis(
            image_array, target_dimensionX=X, target_dimensionY=Y, target_dimensionZ=Z
        )
        image = (image_normalize(ctimage_resizedAll) * 255.0).astype("uint8")
        return image
    except Exception:
        print("PatientId:%s couldnt be converted" % (patientid))
        return None


def read_image_cached(dir_name, patientid, Z=100, Y=200, X=200, cache_dir=CACHE_DIR):
    """
    Fix: np.save('file.npy.tmp', ...) actually writes 'file.npy.tmp.npy'.
    We save to a base tmp name and then atomically rename the produced .npy.
    """
    fn = f"{os.path.basename(dir_name)}_{patientid}_z{Z}_y{Y}_x{X}_uint8.npy"
    fp = os.path.join(cache_dir, fn)
    if os.path.exists(fp):
        try:
            return np.load(fp, allow_pickle=False, mmap_mode="r")
        except Exception:
            pass

    img = read_image(dir_name, patientid, Z=Z, Y=Y, X=X)
    if img is None:
        return None

    tmp_base = fp + ".tmp"
    tmp_npy = tmp_base + ".npy"
    try:
        if os.path.exists(tmp_npy):
            os.remove(tmp_npy)
        if os.path.exists(tmp_base):
            os.remove(tmp_base)
    except Exception:
        pass

    np.save(tmp_base, img, allow_pickle=False)
    os.replace(tmp_npy, fp)
    return img




## === cell 3
def csv_preprocess(data):
    data = data.copy()
    data["Healthy-FVC"] = round((data["FVC"] * 100) / data["Percent"])
    FE = []
    FE.append("Healthy-FVC")

    COLS = ["Sex", "SmokingStatus"]
    for col in COLS:
        for mod in data[col].unique():
            FE.append(mod)
            data[mod] = (data[col] == mod).astype(int)

    data = data[["Patient", "Weeks", "FVC", "Age"] + FE]
    data = data.sort_values(["Patient", "Weeks"], ascending=True).reset_index(drop=True)

    FE1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]
    rename_col = {"Weeks": "base_Weeks", "FVC": "base_FVC"}
    data = data.rename(columns=rename_col)

    first_idx = data.groupby("Patient", sort=False).head(1).index.to_numpy()
    counts = (
        data["Patient"]
        .value_counts(sort=False)
        .reindex(data.loc[first_idx, "Patient"].values)
        .to_numpy()
    )

    base_rows = data.loc[first_idx].reset_index(drop=True)
    repeated = base_rows.loc[base_rows.index.repeat(counts)].reset_index(drop=True)

    repeated["Week"] = data["base_Weeks"].to_numpy()
    repeated["actual_FVC"] = data["base_FVC"].to_numpy()

    npData = repeated[
        ["Patient", "base_Weeks", "base_FVC", "Age"]
        + FE1
        + ["Week", "Healthy-FVC", "actual_FVC"]
    ].copy()

    npData = npData.reset_index(drop=True)
    npData = npData.fillna(0)

    npData["Week"] = npData["Week"] - npData["base_Weeks"]
    npData["base_Weeks"] = 0.0
    return npData




## === cell 4
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ds_3d_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, kernels_per_layer):
        super(ds_3d_conv, self).__init__()
        self.depthwise = nn.Conv3d(
            nin,
            nin * kernels_per_layer,
            kernel_size=kernel_size,
            padding=padding,
            groups=nin,
        )
        self.pointwise = nn.Conv3d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class SIGMA(nn.Module):
    def __init__(self):
        super(SIGMA, self).__init__()
        self.data_net1 = nn.Sequential(
            nn.Linear(41, 64), nn.ReLU(), nn.Linear(64, 119), nn.ReLU()
        )
        self.data_net2 = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 503), nn.ReLU()
        )
        self.data_net3 = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 119), nn.ReLU()
        )
        self.data_net4 = nn.Sequential(
            nn.Linear(750, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.ReLU(),
        )

    def forward(self, data_i, image_o):
        x = torch.cat((data_i, image_o), dim=-1)
        out1 = self.data_net1(x)
        out2 = torch.cat((data_i, out1), dim=-1)
        out2 = self.data_net2(out2)
        out3 = torch.cat((data_i, out2), dim=-1)
        out3 = self.data_net3(out3)
        out4 = torch.cat((data_i, out1, out2, out3), dim=-1)
        out = self.data_net4(out4)
        return out


class IMAGE(nn.Module):
    def __init__(
        self, channel_number=[32, 64, 128, 256, 256, 64], output_dim=16, dropout=True
    ):
        super(IMAGE, self).__init__()
        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = channel_number[i - 1]
            out_channel = channel_number[i]
            if i < n_layer - 1:
                self.feature_extractor.add_module(
                    "conv_%d" % i,
                    self.conv_layer(
                        in_channel,
                        out_channel,
                        maxpool=True,
                        kernel_size=3,
                        padding=1,
                        kernels_per_layer=1,
                        maxpool_stride=2,
                    ),
                )
            else:
                self.feature_extractor.add_module(
                    "conv_%d" % i,
                    self.conv_layer(
                        in_channel,
                        out_channel,
                        maxpool=False,
                        kernel_size=1,
                        padding=0,
                        kernels_per_layer=1,
                    ),
                )
        self.classifier = nn.Sequential()
        if dropout is True:
            self.classifier.add_module("dropout", nn.Dropout(0.5))
        i = n_layer
        in_channel = channel_number[-1]
        out_channel = output_dim
        self.classifier.add_module(
            "conv_%d" % i, nn.Conv3d(in_channel, out_channel, padding=0, kernel_size=1)
        )
        self.flat = nn.Sequential(
            Flatten(),
            nn.Linear(1728, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
        )

    @staticmethod
    def conv_layer(
        in_channel,
        out_channel,
        maxpool=True,
        kernel_size=3,
        padding=1,
        kernels_per_layer=1,
        maxpool_stride=2,
    ):
        if maxpool is True:
            layer = nn.Sequential(
                ds_3d_conv(
                    in_channel, out_channel, kernel_size, padding, kernels_per_layer
                ),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                ds_3d_conv(
                    in_channel, out_channel, kernel_size, padding, kernels_per_layer
                ),
                nn.BatchNorm3d(out_channel),
                nn.ReLU(),
            )
        return layer

    def forward(self, image_i):
        image_o = self.feature_extractor(image_i)
        image_o = self.classifier(image_o)
        image_o = self.flat(image_o)
        return image_o


class Combined_NET(nn.Module):
    def __init__(self):
        super(Combined_NET, self).__init__()
        self.image = IMAGE()
        self.data = SIGMA()

    def forward(self, image_i, data_i):
        image_o = self.image(image_i)
        data_o = self.data(data_i, image_o)
        return data_o




## === cell 5
C1, C2 = torch.tensor(70, dtype=torch.float32), torch.tensor(1000, dtype=torch.float32)


def score(y_true, y_pred):
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    c1_same_shape = torch.ones(sigma.size(), device=y_pred.device) * C1
    sigma_clip = torch.max(sigma, c1_same_shape)
    delta = (y_true[:, 0] - fvc_pred).abs()

    c2_same_shape = torch.ones(delta.size(), device=y_pred.device) * C2
    delta = torch.min(delta, c2_same_shape)

    sq2 = torch.tensor(2.0, device=y_pred.device).sqrt()
    metric = (delta / sigma_clip) * sq2 + (sigma_clip * sq2).log()
    return (metric).mean()


def qloss(y_true, y_pred):
    qs = [0.25, 0.50, 0.75]
    q = torch.tensor(np.array([qs]), device=y_pred.device, dtype=torch.float32)
    e = y_true - y_pred
    v = torch.max(q * e, (q - 1) * e)
    return v.mean()


def quartile_loss(y_true, y_pred, _lambda=0.65):
    loss = _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)
    return loss




## === cell 6
def make_eval_data(
    npEval,
    model,
    device="cuda",
    batch_size=16,
    pid_to_slope=None,
    pid_to_sigma=None,
    global_slope=0.0,
):
    x_features = npEval[
        [
            "base_FVC",
            "Age",
            "Male",
            "Female",
            "Ex-smoker",
            "Never smoked",
            "Currently smokes",
            "Week",
            "Healthy-FVC",
        ]
    ].values.astype(np.float32)
    x_patientids = npEval["Patient"].values

    unique_patients = pd.unique(x_patientids)
    loaded_images = {}
    for unique_patient in unique_patients:
        img = read_image_cached(TEST_FOLDER, unique_patient, Z=100, Y=200, X=200)
        if img is None:
            img = np.zeros((100, 200, 200), dtype=np.uint8)
        loaded_images[unique_patient] = img

    use_cuda = torch.cuda.is_available() and device == "cuda"
    if use_cuda:
        model = model.cuda()
    model.eval()

    N = len(npEval)
    preds = np.empty((N, 3), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            pids_batch = x_patientids[start:end]
            img_batch = np.stack([loaded_images[pid] for pid in pids_batch]).astype(
                np.float32
            )
            feat_batch = x_features[start:end]

            xb_img = torch.from_numpy(img_batch).unsqueeze(1)  # (B,1,Z,Y,X)
            xb_feat = torch.from_numpy(feat_batch)

            if use_cuda:
                xb_img = xb_img.pin_memory().cuda(non_blocking=True)
                xb_feat = xb_feat.pin_memory().cuda(non_blocking=True)

            out = model(xb_img, xb_feat).detach().cpu().numpy()
            preds[start:end] = out

    preds_sorted = np.sort(preds.astype(np.float32, copy=False), axis=1)

    fvc_nn = preds_sorted[:, 1].astype(np.float32, copy=False)
    conf_nn = (preds_sorted[:, 2] - preds_sorted[:, 0]).astype(
        np.float32, copy=False
    ) + 1e-3

    week = npEval["Week"].values.astype(np.float32, copy=False)

    if pid_to_slope is not None:
        slopes = np.array(
            [pid_to_slope.get(p, global_slope) for p in x_patientids], dtype=np.float32
        )
        base_fvc = npEval["base_FVC"].values.astype(np.float32, copy=False)
        fvc_linear = base_fvc + slopes * week

        alpha = 0.20  # small, stable blend; keeps core logic, reduces large systematic errors
        fvc = (1.0 - alpha) * fvc_nn + alpha * fvc_linear
    else:
        fvc = fvc_nn

    conf = conf_nn
    if pid_to_sigma is not None:
        sigmas = np.array(
            [pid_to_sigma.get(p, np.nan) for p in x_patientids], dtype=np.float32
        )
        has = np.isfinite(sigmas)
        beta = (
            0.70  # lean more on empirical per-patient sigma for LaplaceLL calibration
        )
        conf = np.where(has, (1.0 - beta) * conf_nn + beta * sigmas, conf_nn)

    npEval["FVC"] = fvc
    npEval["Confidence"] = conf
    return npEval




## === cell 7
data_train = pd.read_csv(TRAIN_CSV)
data_test = pd.read_csv(TEST_CSV)
sample_submission = pd.read_csv(SAMPLE_SUB)

print(data_train.shape, data_test.shape, sample_submission.shape)



## === cell 8
sub = sample_submission.copy()
sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: x.split("_")[1]).astype(int)
sub = sub.reset_index(drop=True)

merge = pd.merge(
    data_test,
    sub[["Patient_Week", "Patient", "Weeks"]],
    on=["Patient"],
    how="left",
)
merge = merge.rename(
    columns={"Weeks_x": "base_Weeks", "Weeks_y": "Week", "FVC": "base_FVC"}
)
merge = merge.sort_values(["Patient", "Week"]).reset_index(drop=True)

data_test_merged = merge.loc[
    :,
    [
        "Patient",
        "base_Weeks",
        "base_FVC",
        "Percent",
        "Age",
        "Sex",
        "SmokingStatus",
        "Week",
    ],
].copy()

submission_df = merge.loc[:, ["Patient_Week"]].copy()
submission_df["FVC"] = 0.0
submission_df["Confidence"] = 0.0

data = data_test_merged.copy()
data["Healthy-FVC"] = np.round((data["base_FVC"] * 100.0) / data["Percent"]).astype(
    np.float32
)

for colname in ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]:
    data[colname] = 0

data["Male"] = (data["Sex"] == "Male").astype(int)
data["Female"] = (data["Sex"] == "Female").astype(int)
data["Ex-smoker"] = (data["SmokingStatus"] == "Ex-smoker").astype(int)
data["Never smoked"] = (data["SmokingStatus"] == "Never smoked").astype(int)
data["Currently smokes"] = (data["SmokingStatus"] == "Currently smokes").astype(int)

data["Week"] = data["Week"] - data["base_Weeks"]
data["base_Weeks"] = 0.0

data_test_proc = data[
    [
        "Patient",
        "base_Weeks",
        "base_FVC",
        "Age",
        "Male",
        "Female",
        "Ex-smoker",
        "Never smoked",
        "Currently smokes",
        "Week",
        "Healthy-FVC",
    ]
].copy()

print("data_test_proc shape:", data_test_proc.shape)




## === cell 9
def build_train_examples(data_train_df):
    npTrain = csv_preprocess(data_train_df)

    X_feat = npTrain[
        [
            "base_FVC",
            "Age",
            "Male",
            "Female",
            "Ex-smoker",
            "Never smoked",
            "Currently smokes",
            "Week",
            "Healthy-FVC",
        ]
    ].values.astype(np.float32)
    y_q = npTrain[["actual_FVC"]].values.astype(np.float32)
    y_q = np.concatenate([y_q, y_q, y_q], axis=1)  # (N,3) for qloss

    patients = npTrain["Patient"].values
    return npTrain, patients, X_feat, y_q


def load_images_for_patients(patient_ids, folder, Z=100, Y=200, X=200):
    imgs = {}
    for pid in sorted(set(patient_ids)):
        img = read_image_cached(folder, pid, Z=Z, Y=Y, X=X)
        if img is None:
            continue
        imgs[pid] = img
    return imgs


class PatientIndexedTrainDataset(Dataset):
    def __init__(self, img_bank_np, pid_to_index, pids_per_row, X_feat_np, y_np):
        self.img_bank = torch.from_numpy(img_bank_np.astype(np.float32)).unsqueeze(
            1
        )  # (P,1,Z,Y,X)
        self.pid_idx = torch.from_numpy(
            np.array([pid_to_index[p] for p in pids_per_row], dtype=np.int64)
        )
        self.X_feat = torch.from_numpy(X_feat_np.astype(np.float32))
        self.y = torch.from_numpy(y_np.astype(np.float32))

    def __len__(self):
        return self.X_feat.shape[0]

    def __getitem__(self, i):
        return self.img_bank[self.pid_idx[i]], self.X_feat[i], self.y[i]


npTrain, train_pids, X_feat_train, y_train = build_train_examples(data_train)

train_imgs = load_images_for_patients(train_pids, TRAIN_FOLDER, Z=100, Y=200, X=200)
if len(train_imgs) == 0:
    print(
        "WARNING: No train CTs could be loaded. Falling back to all-zero CT volumes for training."
    )
    unique_pids = ["DUMMY_PATIENT"]
    pid_to_index = {"DUMMY_PATIENT": 0}
    img_bank_np = np.zeros((1, 100, 200, 200), dtype=np.float32)
    train_pids_f = np.array(["DUMMY_PATIENT"] * len(train_pids), dtype=object)
else:
    mask = np.array([pid in train_imgs for pid in train_pids], dtype=bool)
    X_feat_train = X_feat_train[mask]
    y_train = y_train[mask]
    train_pids_f = train_pids[mask]

    unique_pids = sorted(set(train_pids_f))
    pid_to_index = {pid: i for i, pid in enumerate(unique_pids)}
    img_bank_np = np.stack([np.asarray(train_imgs[pid]) for pid in unique_pids]).astype(
        np.float32
    )

dataset = PatientIndexedTrainDataset(
    img_bank_np, pid_to_index, train_pids_f, X_feat_train, y_train
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Combined_NET().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

num_workers = 2 if os.cpu_count() and os.cpu_count() > 2 else 0
loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
)

model.train()
epochs = 1
for ep in range(epochs):
    running = 0.0
    for xb_img, xb_feat, yb in loader:
        xb_img = xb_img.to(device, non_blocking=True)
        xb_feat = xb_feat.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        pred = model(xb_img, xb_feat)
        loss = quartile_loss(yb, pred)
        loss.backward()
        opt.step()
        running += loss.item()
    print(f"epoch {ep+1}/{epochs} loss: {running/len(loader):.5f}")




## === cell 10
def fit_patient_slope_and_sigma(train_df: pd.DataFrame):
    pid_to_slope = {}
    pid_to_sigma = {}

    for pid, g in train_df.groupby("Patient", sort=False):
        gg = g.sort_values("Weeks")
        w = gg["Weeks"].values.astype(np.float32)
        f = gg["FVC"].values.astype(np.float32)

        if len(gg) < 2 or np.all(w == w[0]):
            pid_to_slope[pid] = 0.0
            pid_to_sigma[pid] = np.nan
            continue

        b, a = np.polyfit(w, f, deg=1)
        pid_to_slope[pid] = float(b)

        pred = a + b * w
        resid = f - pred
        mae = float(np.mean(np.abs(resid)))
        pid_to_sigma[pid] = float(max(1e-3, mae * np.sqrt(2.0)))

    slopes = np.array(list(pid_to_slope.values()), dtype=np.float32)
    global_slope = float(np.median(slopes[np.isfinite(slopes)])) if len(slopes) else 0.0

    sigmas = np.array(
        [s for s in pid_to_sigma.values() if np.isfinite(s)], dtype=np.float32
    )
    global_sigma = float(np.median(sigmas)) if len(sigmas) else 200.0

    return pid_to_slope, pid_to_sigma, global_slope, global_sigma


pid_to_slope_train, pid_to_sigma_train, global_slope, global_sigma = (
    fit_patient_slope_and_sigma(data_train)
)
test_patients = pd.unique(data_test_proc["Patient"].values)
pid_to_slope = {pid: pid_to_slope_train.get(pid, global_slope) for pid in test_patients}
pid_to_sigma = {pid: pid_to_sigma_train.get(pid, global_sigma) for pid in test_patients}

test_pred = make_eval_data(
    data_test_proc.copy(),
    model,
    device=device,
    batch_size=16,
    pid_to_slope=pid_to_slope,
    pid_to_sigma=pid_to_sigma,
    global_slope=global_slope,
)

test_pred["Confidence"] = test_pred["Confidence"].clip(lower=70.0)

test_pred["FVC"] = pd.to_numeric(test_pred["FVC"], errors="coerce").fillna(0.0)
test_pred["Confidence"] = pd.to_numeric(
    test_pred["Confidence"], errors="coerce"
).fillna(70.0)



## === cell 11
submission_df.loc[:, "FVC"] = test_pred["FVC"].values
submission_df.loc[:, "Confidence"] = test_pred["Confidence"].values

final_sub = sample_submission[["Patient_Week"]].merge(
    submission_df, on="Patient_Week", how="left", validate="one_to_one"
)
final_sub["FVC"] = final_sub["FVC"].fillna(0.0)
final_sub["Confidence"] = final_sub["Confidence"].fillna(70.0)
final_sub = final_sub[["Patient_Week", "FVC", "Confidence"]]

final_sub.to_csv("submission.csv", index=False)
print(final_sub.head())
print("Wrote submission.csv with shape:", final_sub.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))
