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

-6.856507064793506

# 6. Current score

-7.89794

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -9.38451) has done: 'I fix the crash caused by CT reads returning `None` by making `read_image` always return a valid zero-filled volume when conversion fails, so training/inference never hits `NoneType`. I also make the checkpoint path robust for Kaggle by searching under `/kaggle/input` for the `.pth` file, while keeping the exact same loading/training logic. Finally, I ensure the evaluation path uses the same safe CT loading behavior and always writes a correctly formatted `submission.csv` with the required columns.'
- What this solution (achieved -9.24778) has done: 'Your current score is well below the target (gap = -9.38451 − (-6.8565) = -2.53; higher is better), so we should make a small, low-risk improvement. The biggest scoring issue in your pipeline is that the model’s first output is unconstrained and often produces very large Confidence values, which the metric penalizes via the `-log(sigma)` term; clipping only the minimum at 70 does not prevent overly-large sigma. I keep the same model/weights/training logic, but post-process predictions to clip Confidence into a reasonable range and also ensure FVC is within plausible bounds; this typically improves the Laplace log-likelihood without changing core semantics. I also fix a subtle feature-mismatch bug in `make_eval_data` (it currently uses `base_FVC` instead of `Healthy-FVC` for the last feature), which is a minimal correctness fix that should directly improve predictions.'
- What this solution (achieved -7.5648) has done: 'We need to move the score up toward the target (current -9.24778 vs target -6.8565; higher is better), so we make the smallest changes that directly improve Laplace log-likelihood without changing the model or training. The biggest controllable lever here is Confidence calibration: your current hard upper-clip at 300 is still often too large, which hurts the `-log(sigma)` term; we replace it with a patient-specific calibrated sigma based on in-model residuals on that patient’s known baseline point (Week==base_Weeks), keeping the baseline FVC exactly fixed as you already do. This keeps core logic intact (same network outputs) but adjusts post-processing to better match the metric by shrinking sigma when the model is accurate and increasing it when it’s not. We also ensure the merge/sort alignment between `submission` and `test` is strictly consistent before writing the CSV to avoid any accidental row mismatch.'
- What this solution (achieved -8.05771) has done: 'We need to move the score up (current -7.5648 vs target -6.8565; higher is better), so the smallest safe lever is better Confidence calibration without changing the model/training. I keep your patient-specific sigma idea but calibrate it using the baseline residual plus the model’s predicted sigma at baseline, and then apply a mild week-distance inflation so far-future weeks aren’t under-confident (which is heavily penalized when wrong). I also change the FVC post-processing to anchor predictions to the known baseline FVC using a per-patient shift (preserves the model’s week-to-week trend but fixes systematic bias), which typically improves |FVC_true−FVC_pred| while keeping logic intact. Finally, I keep submission alignment the same and still write a valid `submission.csv`.'
- What this solution (achieved -7.88587) has done: 'Your current score (-8.05771) is below the target (-6.85651) so we should improve it with minimal, low-risk changes focused on the metric. The easiest lever is Confidence calibration: your current cap at 200 and week inflation can still be miscalibrated; we compute a per-patient sigma from the model’s own predicted sigma plus a conservative floor tied to the baseline residual, then use a gentler week-distance inflation and a slightly higher sigma_max to avoid severe under-confidence penalties. We also ensure the baseline anchoring shift uses the baseline point after the shift is applied consistently, and keep all model/training/feature logic unchanged. Finally, we keep the submission alignment logic intact and still write a valid `submission.csv`.'
- What this solution (achieved -8.1243) has done: 'We need to move the score up toward the target (current -7.88587 vs target -6.8565; higher is better), so we make a minimal, metric-aligned change focused on Confidence calibration without touching the model, training loop, or features. Your current per-patient sigma is derived only from the baseline residual and then inflated by week distance; this can still be under-confident for far weeks (big penalty when wrong) and over-confident when baseline happens to be easy. I instead estimate a per-patient week-dependent sigma from the model’s own predicted sigma across that patient’s requested weeks, add a small floor based on baseline residual, and then apply a slightly stronger-but-bounded week-distance inflation; this preserves your existing baseline anchoring of FVC exactly. Finally, we keep the same submission alignment/sorting and still write `submission.csv`.'
- What this solution (achieved -7.89794) has done: 'Your current score (-8.1243) is worse than the target (-6.8565), so we should make a small, low-risk improvement focused on the metric without changing the model or training. The biggest lever left in your pipeline is Confidence calibration: right now Confidence depends mostly on the model’s raw sigma distribution plus a weak week-distance inflation, which can still be under-confident for far weeks (heavy penalty when wrong). I keep your exact FVC baseline anchoring, but replace the Confidence formula with a more conservative, per-patient sigma that (a) uses the model’s predicted week-to-week FVC spread as an uncertainty proxy and (b) adds a bounded week-distance inflation; this typically improves the Laplace log-likelihood by avoiding overly-small sigma. I also ensure the baseline row’s Confidence is set consistently (not accidentally inflated/deflated) while keeping all file paths and submission alignment unchanged.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import pydicom
import scipy.ndimage
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

pydicom.config.settings.reading_validation_mode = "IGNORE"



## === cell 1
BASE_PATH = "../data/osic-pulmonary-fibrosis-progression"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TEST_CSV = os.path.join(BASE_PATH, "test.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_FOLDER = os.path.join(BASE_PATH, "train")
TEST_FOLDER = os.path.join(BASE_PATH, "test")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
device




## === cell 2
def load_scan(path):
    entries = [e.name for e in os.scandir(path) if e.is_file()]
    slices = []
    try:
        for fn in entries:
            ds = pydicom.dcmread(os.path.join(path, fn), stop_before_pixels=True)
            slices.append((float(ds.ImagePositionPatient[2]), fn))
        slices.sort(key=lambda x: x[0])
        files_sorted = [fn for _, fn in slices]
    except Exception:
        files_sorted = sorted(entries)

    return [
        pydicom.dcmread(os.path.join(path, fn), stop_before_pixels=False)
        for fn in files_sorted
    ]


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices]).astype(np.int16)

    try:
        image[image <= -2000] = 0

        intercept = np.array(
            [float(s.RescaleIntercept) for s in slices], dtype=np.float32
        )
        slope = np.array([float(s.RescaleSlope) for s in slices], dtype=np.float32)

        img = image.astype(np.float32)
        img *= slope[:, None, None]
        img = img.astype(np.int16)
        img = (img.astype(np.int32) + intercept[:, None, None].astype(np.int32)).astype(
            np.int16
        )
        return img
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
    path = dir_name + os.sep + patientid
    try:
        slices = load_scan(path)
        image_array = get_pixels_hu(slices)
        ctimage_resizedAll = resize_along_allaxis(
            image_array, target_dimensionX=X, target_dimensionY=Y, target_dimensionZ=Z
        )
        image = (image_normalize(ctimage_resizedAll) * 255.0).astype("uint8")
        return image
    except Exception:
        print(f"PatientId:{patientid} couldnt be converted; using zeros volume.")
        return np.zeros((Z, Y, X), dtype=np.uint8)




## === cell 3
def csv_preprocess(data):
    data["Healthy-FVC"] = round((data["FVC"] * 100) / data["Percent"])
    FE = ["Healthy-FVC"]

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

    out_rows = []
    for pid, grp in data.groupby("Patient", sort=False):
        base = grp.iloc[[0]].copy()
        weeks = grp["base_Weeks"].to_numpy()
        fvcs = grp["base_FVC"].to_numpy()
        for w, f in zip(weeks, fvcs):
            r = base.copy()
            r["Week"] = w
            r["actual_FVC"] = f
            out_rows.append(r)

    npData = pd.concat(out_rows, axis=0, ignore_index=True, sort=False)
    npData = npData.fillna(0)

    for c in FE1:
        if c not in npData.columns:
            npData[c] = 0
    if "Healthy-FVC" not in npData.columns:
        npData["Healthy-FVC"] = 0
    return npData[
        ["Patient", "base_Weeks", "base_FVC", "Age"]
        + FE1
        + ["Week", "Healthy-FVC", "actual_FVC"]
    ]




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
            nn.Linear(42, 64), nn.ReLU(), nn.Linear(64, 118), nn.ReLU()
        )
        self.data_net2 = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 502), nn.ReLU()
        )
        self.data_net3 = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 118), nn.ReLU()
        )
        self.data_net4 = nn.Sequential(
            nn.Linear(780, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.ReLU(),
        )

    def forward(self, data_i, image_o):
        x = torch.cat((data_i, image_o), dim=-1)
        out1 = self.data_net1(x)
        out2 = torch.cat((data_i, out1), dim=-1)
        out2 = self.data_net2(out2)
        out3 = torch.cat((data_i, out2), dim=-1)
        out3 = self.data_net3(out3)
        out4 = torch.cat((x, out1, out2, out3), dim=-1)
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
            in_channel = 1 if i == 0 else channel_number[i - 1]
            out_channel = channel_number[i]
            if i < n_layer - 1:
                self.feature_extractor.add_module(
                    f"conv_{i}",
                    self.conv_layer(
                        in_channel,
                        out_channel,
                        maxpool=True,
                        kernel_size=3,
                        padding=1,
                        kernels_per_layer=1,
                    ),
                )
            else:
                self.feature_extractor.add_module(
                    f"conv_{i}",
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
            f"conv_{i}", nn.Conv3d(in_channel, out_channel, padding=0, kernel_size=1)
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
    sigma = y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = torch.clamp(sigma, min=float(C1))
    delta = (y_true[:, 0] - fvc_pred).abs()
    delta = torch.clamp(delta, max=float(C2))

    sq2 = torch.sqrt(torch.tensor(2.0, device=y_pred.device))
    metric = (delta / sigma_clip) * sq2 + (sigma_clip * sq2).log()
    return metric.mean()


def quartile_loss(y_true, y_pred):
    loss = score(y_true, y_pred)
    return loss




## === cell 6
def _load_one_patient_ct(args):
    dir_name, pid = args
    return pid, read_image(dir_name, pid, Z=100, Y=200, X=200)


def _load_cts_for_patients(dir_name, patient_ids, desc):
    from concurrent.futures import ThreadPoolExecutor

    patient_ids = list(patient_ids)
    loaded_images = {}
    max_workers = min(8, (os.cpu_count() or 2))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for pid, img in tqdm(
            ex.map(_load_one_patient_ct, [(dir_name, p) for p in patient_ids]),
            total=len(patient_ids),
            desc=desc,
        ):
            loaded_images[pid] = img
    return loaded_images


def make_eval_data(npEval, model, device="cuda"):
    feat_cols = [
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
    x_features = torch.tensor(npEval[feat_cols].values, dtype=torch.float32)
    x_patientids = npEval["Patient"].values

    unique_patients = pd.unique(x_patientids)
    loaded_images = _load_cts_for_patients(
        TEST_FOLDER, unique_patients, desc="Loading test CTs"
    )

    model = model.to(device)
    model.eval()

    preds = np.empty((len(npEval), 2), dtype=np.float32)

    pid_to_indices = {}
    for i, pid in enumerate(x_patientids):
        pid_to_indices.setdefault(pid, []).append(i)

    with torch.no_grad():
        for pid, idxs in pid_to_indices.items():
            img_u8 = loaded_images[pid]
            x_image_np = (img_u8.astype(np.float32) / 255.0)[None, None, ...]
            x_image = torch.from_numpy(x_image_np).to(device)

            image_o = model.image(x_image)

            bx_tab = x_features[idxs].to(device)
            image_o_b = image_o.expand(bx_tab.shape[0], -1).contiguous()

            out = (
                model.data(bx_tab, image_o_b).detach().cpu().numpy().astype(np.float32)
            )
            preds[np.asarray(idxs, dtype=np.int64)] = out

    npEval["FVC"] = preds[:, 1]
    npEval["Confidence"] = preds[:, 0]
    return npEval




## === cell 7
data_train = pd.read_csv(TRAIN_CSV)
data_test = pd.read_csv(TEST_CSV)
submission = pd.read_csv(SAMPLE_SUB)

submission["Patient"] = submission["Patient_Week"].apply(lambda x: x.split("_")[0])
submission["Weeks"] = (
    submission["Patient_Week"].apply(lambda x: x.split("_")[1]).astype(int)
)
submission = submission.sort_values(
    by=["Patient", "Weeks"], ascending=True
).reset_index(drop=True)

merge = (
    pd.merge(data_test, submission, on=["Patient"], how="left")
    .sort_values(["Patient", "Weeks_y"])
    .reset_index(drop=True)
)
merge = merge.drop(["FVC_y"], axis=1)
merge = merge.rename(
    columns={"FVC_x": "base_FVC", "Weeks_y": "Week", "Weeks_x": "base_Weeks"}
)

data_test = merge.loc[
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
]
submission = merge.loc[:, ["Patient_Week", "base_FVC", "Confidence"]]
submission = submission.rename(columns={"base_FVC": "FVC"})



## === cell 8
data = data_test.copy()
data["Healthy-FVC"] = round((data["base_FVC"] * 100) / data["Percent"])
FE = ["Healthy-FVC"]

COLS = ["Sex", "SmokingStatus"]
for col in COLS:
    for mod in data[col].unique():
        FE.append(mod)
        data[mod] = (data[col] == mod).astype(int)

FE1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]
npData = pd.DataFrame(
    columns=["Patient", "base_Weeks", "base_FVC", "Age", "Healthy-FVC"] + FE1 + ["Week"]
)
npData = pd.concat([npData, data], axis=0, ignore_index=True, sort=True)
npData = npData.fillna(0)

data_test = npData[
    [
        "Patient",
        "base_Weeks",
        "base_FVC",
        "Age",
        "Healthy-FVC",
        "Male",
        "Female",
        "Ex-smoker",
        "Never smoked",
        "Currently smokes",
        "Week",
    ]
].copy()




## === cell 9
def build_train_set(train_df):
    npTrain = csv_preprocess(train_df.copy())
    x_features = npTrain[
        [
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
    ].astype(np.float32)
    y_true = npTrain[["actual_FVC"]].astype(np.float32).values  # shape (N,1)

    unique_patients = npTrain.Patient.unique()

    loaded_images = _load_cts_for_patients(
        TRAIN_FOLDER, unique_patients, desc="Loading train CTs"
    )

    images = np.zeros((len(npTrain), 1, 100, 200, 200), dtype=np.float32)
    for i, pid in enumerate(npTrain["Patient"].values):
        images[i, 0] = loaded_images[pid].astype(np.float32)

    images = images / 255.0

    y = torch.from_numpy(y_true)
    x_tab = torch.from_numpy(x_features.values)
    x_img = torch.from_numpy(images)
    return x_img, x_tab, y, npTrain


def train_fallback_model(model, train_df, device):
    x_img, x_tab, y, _ = build_train_set(train_df)
    y_true = y  # (N,1)

    ds = TensorDataset(x_img, x_tab, y_true)
    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)

    model = model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    n_epochs = 1
    for epoch in range(n_epochs):
        pbar = tqdm(dl, desc=f"Training epoch {epoch+1}/{n_epochs}")
        for bx_img, bx_tab, by in pbar:
            bx_img = bx_img.to(device).float()
            bx_tab = bx_tab.to(device).float()
            by = by.to(device).float()

            pred = model(bx_img, bx_tab)
            loss = quartile_loss(by, pred)

            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.detach().cpu().item()))
    return model




## === cell 10
def find_ckpt(preferred_path: str):
    if os.path.exists(preferred_path):
        return preferred_path
    search_root = "/kaggle/input"
    target_name = os.path.basename(preferred_path)
    if os.path.isdir(search_root):
        for root, _, files in os.walk(search_root):
            if target_name in files:
                return os.path.join(root, target_name)
    return preferred_path


model = Combined_NET()

ckpt_path = "../input/tt-05-690/Epoch5_Score6.906373289247222_Acc0.9272943559861341.pth"
ckpt_path = find_ckpt(ckpt_path)

if os.path.exists(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
else:
    model = train_fallback_model(model, data_train, device)

test = make_eval_data(data_test.copy(), model, device=device)



## === cell 11
sigma_min = 70.0
sigma_max = (
    350.0  # slightly higher cap to avoid severe under-confidence when far from baseline
)

for pid in test.Patient.unique():
    grp = test[test.Patient == pid].copy()

    base_idx = grp[grp.Week == grp.base_Weeks].index.values
    if len(base_idx) == 0:
        continue
    bi = int(base_idx[0])

    true_base_fvc = float(test.loc[bi, "base_FVC"])
    pred_base_fvc = float(test.loc[bi, "FVC"])

    shift = true_base_fvc - pred_base_fvc
    test.loc[grp.index, "FVC"] = test.loc[grp.index, "FVC"] + shift
    test.loc[bi, "FVC"] = true_base_fvc

    abs_err = abs(true_base_fvc - pred_base_fvc)

    fvc_vals = test.loc[grp.index, "FVC"].astype(float).values
    spread = float(np.std(fvc_vals))  # already after baseline anchoring

    sigma0 = 0.80 * spread + (sigma_min + 0.35 * min(abs_err, 1000.0))
    sigma0 = float(np.clip(sigma0, sigma_min, sigma_max))

    week_dist = (
        (test.loc[grp.index, "Week"] - float(test.loc[bi, "base_Weeks"]))
        .abs()
        .astype(float)
        .values
    )
    sigma_week = sigma0 * (1.0 + 0.013 * week_dist)
    sigma_week = np.minimum(sigma_week, sigma0 * 2.5)

    sigma_week = np.clip(sigma_week, sigma_min, sigma_max)
    test.loc[grp.index, "Confidence"] = sigma_week
    test.loc[bi, "Confidence"] = sigma0

test.loc[test.Confidence < sigma_min, "Confidence"] = sigma_min
test.loc[test.Confidence > sigma_max, "Confidence"] = sigma_max

test.loc[test.FVC < 500, "FVC"] = 500
test.loc[test.FVC > 6000, "FVC"] = 6000



## === cell 12
test = test.sort_values(["Patient", "Week"]).reset_index(drop=True)

submission_tmp = submission.copy()
submission_tmp["Patient"] = submission_tmp["Patient_Week"].apply(
    lambda x: x.split("_")[0]
)
submission_tmp["Week"] = submission_tmp["Patient_Week"].apply(
    lambda x: int(x.split("_")[1])
)
submission_tmp = submission_tmp.sort_values(["Patient", "Week"]).reset_index(drop=True)

submission_tmp.loc[:, "FVC"] = test["FVC"].values
submission_tmp.loc[:, "Confidence"] = test["Confidence"].values

submission_out = submission_tmp[["Patient_Week", "FVC", "Confidence"]]
submission_out.to_csv("submission.csv", index=False)

print(submission_out.head())
print("Wrote submission.csv with shape:", submission_out.shape)
