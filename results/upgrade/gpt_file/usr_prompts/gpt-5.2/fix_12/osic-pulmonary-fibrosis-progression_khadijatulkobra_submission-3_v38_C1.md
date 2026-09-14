# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

-6.864623391191141

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -24.65417) has done: 'I fix the FileNotFoundError coming from the image caching logic by correcting the temporary `.npy` save path (NumPy appends `.npy`, so the current `os.replace` points to a non-existent file). I also make the cache write robust under DataLoader multiprocessing by using a unique temp filename and an atomic rename, falling back safely if another worker already wrote the cache. These changes are score-neutral but unblock training/inference so a valid `submission.csv` is produced end-to-end. Finally, I keep the model and training loop intact and only add a minimal guard to ensure Confidence is valid (>=70) as required by the metric.'
- What this solution (achieved -24.79812) has done: 'Most of the timeout comes from repeatedly decoding/processing large DICOM volumes (train preload of ~176 patients and test preload) and expensive 3D resizing; the model training itself is only 1 epoch but still pays the full I/O + preprocessing cost. I keep the exact same image preprocessing math and model/training logic, but make it faster by (1) reading DICOMs with `stop_before_pixels=True` for sorting (then decoding PixelData once in correct order), (2) using `os.scandir` and caching slice order to avoid repeated directory work, (3) saving cached arrays with `np.save(..., allow_pickle=False)` and loading via `mmap` as before, and (4) eliminating slow pandas `.apply` and a Python loop in the postprocessing step via vectorized operations. These are provably equivalent transformations (same images/features/loops), just less overhead.'
- What this solution (achieved -24.79812) has done: 'Your current score gap to target is large (−24.80 vs −6.86; higher is better), and the dominant issue is that the training loss is accidentally optimizing the *negative* of the competition metric (your `score()` returns a positive value and you minimize it), so the model is trained to get worse. I keep your model, data pipeline, and training loop identical, but fix the sign so training maximizes the intended Laplace log-likelihood by minimizing its negative (this should move the score strongly upward toward the target). I also make the prediction head consistent with the metric by forcing `sigma` to be positive at inference/training time via a non-destructive transform (absolute value + epsilon), while keeping the same output structure `[sigma, fvc]`. Finally, I keep your required postprocessing (`Confidence >= 70` and baseline-week overwrite) and ensure the submission is still written as `submission.csv`.'
- What this solution (achieved -24.79812) has done: 'Your score is far below the target (−24.80 vs −6.86; higher is better), and the main reason is that the training objective is currently the *wrong sign* of the competition metric (you’re minimizing a quantity that should be maximized), which actively trains the model to perform worse. I make the minimal fix: change `score()` to return the **negative** Laplace log-likelihood (so minimizing it correctly maximizes the Kaggle metric), while keeping the same model, data pipeline, and 1-epoch training loop. I also align the label tensor shape with what `score()` expects (2D `[B,1]`) to avoid unintended broadcasting and stabilize training without changing semantics. Everything else (CT preprocessing, caching, architecture, inference, baseline-week overwrite, confidence clipping, and submission writing) stays the same.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import scipy.ndimage

import pydicom
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def seed_everything(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

BASE_PATH = "../data/osic-pulmonary-fibrosis-progression"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TEST_CSV = os.path.join(BASE_PATH, "test.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_FOLDER = os.path.join(BASE_PATH, "train")
TEST_FOLDER = os.path.join(BASE_PATH, "test")

CACHE_DIR = os.path.join("../data", "osic_cache_uint8_z100_y200_x200")
os.makedirs(CACHE_DIR, exist_ok=True)

_MAX_PRELOAD_WORKERS = max(1, min(8, (os.cpu_count() or 2)))

_DICOM_SORT_CACHE = {}

_TABULAR_DIM_REQUIRED = 42


def _pad_tabular_to_required_dim(
    x_2d: torch.Tensor, required_dim: int = _TABULAR_DIM_REQUIRED
):
    """Pad last dimension with zeros so x.shape[-1] == required_dim."""
    cur = x_2d.shape[-1]
    if cur == required_dim:
        return x_2d
    if cur > required_dim:
        return x_2d[:, :required_dim]
    pad = torch.zeros(
        (x_2d.shape[0], required_dim - cur), dtype=x_2d.dtype, device=x_2d.device
    )
    return torch.cat([x_2d, pad], dim=-1)




## === cell 1
def load_scan(path):
    cached = _DICOM_SORT_CACHE.get(path)
    if cached is None:
        entries = [e.name for e in os.scandir(path) if e.is_file()]
        entries.sort()

        zpos = []
        need_fallback = False
        for name in entries:
            fp = os.path.join(path, name)
            try:
                ds = pydicom.dcmread(
                    fp,
                    force=True,
                    stop_before_pixels=True,
                    specific_tags=["ImagePositionPatient"],
                )
                ipp = getattr(ds, "ImagePositionPatient", None)
                if ipp is None:
                    need_fallback = True
                    break
                zpos.append(float(ipp[2]))
            except Exception:
                need_fallback = True
                break

        if not need_fallback and len(zpos) == len(entries):
            order = np.argsort(np.asarray(zpos, dtype=np.float32), kind="mergesort")
            entries = [entries[i] for i in order]
        _DICOM_SORT_CACHE[path] = entries
    else:
        entries = cached

    slices = []
    for name in entries:
        fp = os.path.join(path, name)
        try:
            ds = pydicom.dcmread(
                fp,
                force=True,
                stop_before_pixels=False,
                specific_tags=[
                    "PixelData",
                    "RescaleIntercept",
                    "RescaleSlope",
                    "ImagePositionPatient",
                ],
            )
        except Exception:
            ds = pydicom.dcmread(fp, force=True)
        slices.append(ds)
    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices]).astype(np.int16)
    try:
        image[image <= -2000] = 0
        intercepts = np.array(
            [float(getattr(s, "RescaleIntercept", 0.0)) for s in slices],
            dtype=np.float32,
        )
        slopes = np.array(
            [float(getattr(s, "RescaleSlope", 1.0)) for s in slices], dtype=np.float32
        )

        if not np.allclose(slopes, 1.0):
            img_f = image.astype(np.float32)
            img_f *= slopes[:, None, None]
            img_f += intercepts[:, None, None]
            image = img_f.astype(np.int16)
        else:
            image = image.astype(np.int32, copy=False)
            image += intercepts.astype(np.int32)[:, None, None]
            image = image.astype(np.int16, copy=False)
    except Exception:
        pass
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
        slices, [zoom_factorZ, zoom_factorY, zoom_factorX], mode="nearest"
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
    cache_key = f"{os.path.basename(dir_name)}__{patientid}__Z{Z}_Y{Y}_X{X}.npy"
    cache_path = os.path.join(CACHE_DIR, cache_key)
    if os.path.exists(cache_path):
        return np.load(cache_path, mmap_mode="r")

    path = dir_name + os.sep + patientid
    try:
        slices = load_scan(path)
        image_array = get_pixels_hu(slices)
        ctimage_resizedAll = resize_along_allaxis(
            image_array, target_dimensionX=X, target_dimensionY=Y, target_dimensionZ=Z
        )
        image = (image_normalize(ctimage_resizedAll) * 255.0).astype("uint8")
    except Exception:
        image = np.full((Z, Y, X), 128, dtype=np.uint8)

    tmp_path = f"{cache_path}.tmp_{os.getpid()}.npy"
    np.save(tmp_path, image, allow_pickle=False)
    try:
        os.replace(tmp_path, cache_path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass

    return np.load(cache_path, mmap_mode="r")




## === cell 2
def csv_preprocess(data):
    data = data.copy()
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
    data = data.rename(columns={"Weeks": "base_Weeks", "FVC": "base_FVC"})

    first_rows = data.groupby("Patient", sort=False).head(1).copy()
    counts = data.groupby("Patient", sort=False).size().rename("n")
    first_rows = first_rows.merge(
        counts, left_on="Patient", right_index=True, how="left"
    )

    obs = data[["Patient", "base_Weeks", "base_FVC"]].copy()
    obs = obs.rename(columns={"base_Weeks": "Week", "base_FVC": "actual_FVC"})

    repeated = first_rows.loc[first_rows.index.repeat(first_rows["n"])].drop(
        columns=["n"]
    )
    repeated = repeated.reset_index(drop=True)

    obs = obs.reset_index(drop=True)
    repeated["Week"] = obs["Week"].to_numpy()
    repeated["actual_FVC"] = obs["actual_FVC"].to_numpy()

    for col in FE1:
        if col not in repeated.columns:
            repeated[col] = 0
    repeated = repeated.fillna(0)

    return repeated[
        ["Patient", "base_Weeks", "base_FVC", "Age"]
        + FE1
        + ["Week", "Healthy-FVC", "actual_FVC"]
    ]




## === cell 3
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
            nn.Linear(74, 64), nn.ReLU(), nn.Linear(64, 118), nn.ReLU()
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
                    "conv_%d" % i,
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




## === cell 4
C1, C2 = 70.0, 1000.0


def score(y_true, y_pred):
    sigma_raw = y_pred[:, 0]
    fvc_pred = y_pred[:, 1]
    sigma = sigma_raw.abs() + 1e-6

    sigma_clip = torch.clamp(sigma, min=C1)
    delta = (y_true[:, 0] - fvc_pred).abs()
    delta = torch.clamp(delta, max=C2)

    key = (y_pred.device, y_pred.dtype)
    if not hasattr(score, "_sq2_cache"):
        score._sq2_cache = {}
    if key not in score._sq2_cache:
        score._sq2_cache[key] = torch.sqrt(
            torch.tensor(2.0, device=y_pred.device, dtype=y_pred.dtype)
        )
    sq2 = score._sq2_cache[key]

    ll = -(sq2 * delta / sigma_clip) - torch.log(sq2 * sigma_clip)
    return (-ll).mean()


def quartile_loss(y_true, y_pred):
    return score(y_true, y_pred)




## === cell 5
class OSICDataset(Dataset):
    def __init__(
        self,
        df,
        image_dir,
        Z=100,
        Y=200,
        X=200,
        cache_images=True,
        preloaded_images=None,
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.Z, self.Y, self.X = Z, Y, X
        self.cache_images = cache_images
        self._cache = {} if preloaded_images is None else preloaded_images

        self.feature_cols = [
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

    def __len__(self):
        return len(self.df)

    def _get_image(self, patient):
        if patient in self._cache:
            return self._cache[patient]
        img = read_image(self.image_dir, patient, Z=self.Z, Y=self.Y, X=self.X)
        if self.cache_images:
            self._cache[patient] = img
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient = row["Patient"]
        img = self._get_image(patient)

        x_image = (
            torch.from_numpy(np.asarray(img)).to(torch.float32).div(255.0).unsqueeze(0)
        )  # [1,Z,Y,X]

        x_feat_10 = torch.from_numpy(
            row[self.feature_cols].values.astype(np.float32)
        ).unsqueeze(
            0
        )  # [1,10]
        x_feat_42 = _pad_tabular_to_required_dim(x_feat_10).squeeze(0)  # [42]

        y = torch.tensor([row["actual_FVC"]], dtype=torch.float32)  # [1]
        return x_image, x_feat_42, y


def make_eval_data(npEval, model, device="cuda", batch_size=8):
    from concurrent.futures import ThreadPoolExecutor

    feature_cols = [
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

    x_features_10 = torch.tensor(
        npEval[feature_cols].values, dtype=torch.float32
    )  # [N,10]
    x_features = _pad_tabular_to_required_dim(x_features_10)  # [N,42]

    patients = npEval["Patient"].values

    unique_patients = pd.unique(patients)
    loaded_images = {}
    dir_name_of_patientid = TEST_FOLDER

    def _load_one(p):
        return p, read_image(dir_name_of_patientid, p, Z=100, Y=200, X=200)

    with ThreadPoolExecutor(max_workers=_MAX_PRELOAD_WORKERS) as ex:
        for p, img in tqdm(
            ex.map(_load_one, unique_patients),
            total=len(unique_patients),
            desc="preload test images",
        ):
            loaded_images[p] = img

    use_cuda = torch.cuda.is_available() and device == "cuda"
    if use_cuda:
        model.to("cuda")

    model.eval()
    preds = np.zeros((len(npEval), 2), dtype=np.float32)

    with torch.no_grad():
        start = 0
        while start < len(npEval):
            end = min(start + batch_size, len(npEval))
            batch_patients = patients[start:end]
            imgs = np.stack(
                [np.asarray(loaded_images[p]) for p in batch_patients], axis=0
            )
            x_image = torch.from_numpy(imgs).to(torch.float32).div(255.0).unsqueeze(1)
            x_feat = x_features[start:end]

            if use_cuda:
                x_image = x_image.cuda(non_blocking=True)
                x_feat = x_feat.cuda(non_blocking=True)

            out = model(x_image, x_feat).detach().cpu().numpy()
            preds[start:end] = out
            start = end

    npEval["FVC"] = preds[:, 1]
    npEval["Confidence"] = np.abs(preds[:, 0]) + 1e-6
    return npEval




## === cell 6
data_train = pd.read_csv(TRAIN_CSV)
data_test = pd.read_csv(TEST_CSV)
submission = pd.read_csv(SAMPLE_SUB)



## === cell 7
pw = submission["Patient_Week"].str.split("_", n=1, expand=True)
submission["Patient"] = pw[0]
submission["Weeks"] = pw[1].astype(np.int16)

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
train_np = csv_preprocess(data_train)
train_np = train_np[
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
        "actual_FVC",
    ]
].copy()

for col in ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]:
    if col not in train_np.columns:
        train_np[col] = 0
for col in ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]:
    if col not in data_test.columns:
        data_test[col] = 0



## === cell 10
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Combined_NET().to(device)

from concurrent.futures import ThreadPoolExecutor

unique_train_patients = pd.unique(train_np["Patient"].values)
preloaded_train_images = {}


def _load_train_one(p):
    return p, read_image(TRAIN_FOLDER, p, Z=100, Y=200, X=200)


with ThreadPoolExecutor(max_workers=_MAX_PRELOAD_WORKERS) as ex:
    for p, img in tqdm(
        ex.map(_load_train_one, unique_train_patients),
        total=len(unique_train_patients),
        desc="preload train images",
    ):
        preloaded_train_images[p] = img

train_ds = OSICDataset(
    train_np,
    image_dir=TRAIN_FOLDER,
    Z=100,
    Y=200,
    X=200,
    cache_images=True,
    preloaded_images=preloaded_train_images,
)

num_workers = min(4, os.cpu_count() or 2)
train_loader = DataLoader(
    train_ds,
    batch_size=1,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=(device == "cuda"),
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

model.train()
for epoch in range(1):
    pbar = tqdm(train_loader, total=len(train_loader))
    running = 0.0
    for x_img, x_feat, y in pbar:
        x_img = x_img.to(device, non_blocking=True)  # [B,1,Z,Y,X]
        x_feat = x_feat.to(device, non_blocking=True)  # [B,42]
        y = y.to(device, non_blocking=True)  # [B,1]

        pred = model(x_img, x_feat)  # [B,2] => [sigma, fvc]
        loss = quartile_loss(y, pred)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running += loss.item()
        pbar.set_description(f"epoch {epoch+1} loss {running / (pbar.n+1):.4f}")



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3187860679.py in <cell line: 0>()
     52         y = y.to(device, non_blocking=True)  # [B,1]
     53 
---> 54         pred = model(x_img, x_feat)  # [B,2] => [sigma, fvc]
     55         loss = quartile_loss(y, pred)
     56 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_55/2076509389.py in forward(self, image_i, data_i)
    156     def forward(self, image_i, data_i):
    157         image_o = self.image(image_i)
--> 158         data_o = self.data(data_i, image_o)
    159         return data_o
    160 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_55/2076509389.py in forward(self, data_i, image_o)
     49         out1 = self.data_net1(x)
     50         out2 = torch.cat((data_i, out1), dim=-1)
---> 51         out2 = self.data_net2(out2)
     52         out3 = torch.cat((data_i, out2), dim=-1)
     53         out3 = self.data_net3(out3)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/linear.py in forward(self, input)
    123 
    124     def forward(self, input: Tensor) -> Tensor:
--> 125         return F.linear(input, self.weight, self.bias)
    126 
    127     def extra_repr(self) -> str:

RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x160 and 128x256)

## === cell 11
test = make_eval_data(data_test.copy(), model, device=device, batch_size=8)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/1456412379.py in <cell line: 0>()
----> 1 test = make_eval_data(data_test.copy(), model, device=device, batch_size=8)
      2 

/tmp/ipykernel_55/2685607266.py in make_eval_data(npEval, model, device, batch_size)
    121                 x_feat = x_feat.cuda(non_blocking=True)
    122 
--> 123             out = model(x_image, x_feat).detach().cpu().numpy()
    124             preds[start:end] = out
    125             start = end

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_55/2076509389.py in forward(self, image_i, data_i)
    156     def forward(self, image_i, data_i):
    157         image_o = self.image(image_i)
--> 158         data_o = self.data(data_i, image_o)
    159         return data_o
    160 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_55/2076509389.py in forward(self, data_i, image_o)
     49         out1 = self.data_net1(x)
     50         out2 = torch.cat((data_i, out1), dim=-1)
---> 51         out2 = self.data_net2(out2)
     52         out3 = torch.cat((data_i, out2), dim=-1)
     53         out3 = self.data_net3(out3)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/linear.py in forward(self, input)
    123 
    124     def forward(self, input: Tensor) -> Tensor:
--> 125         return F.linear(input, self.weight, self.bias)
    126 
    127     def extra_repr(self) -> str:

RuntimeError: mat1 and mat2 shapes cannot be multiplied (8x160 and 128x256)

## === cell 12
mask_baseweek = test["Week"].to_numpy() == test["base_Weeks"].to_numpy()
if mask_baseweek.any():
    test.loc[mask_baseweek, "FVC"] = test.loc[mask_baseweek, "base_FVC"].to_numpy()
    test.loc[mask_baseweek, "Confidence"] = 70.0

test.loc[test.Confidence < 70, "Confidence"] = 70.0

submission.loc[:, "FVC"] = test.FVC.values
submission.loc[:, "Confidence"] = test.Confidence.values
submission = submission[["Patient_Week", "FVC", "Confidence"]]
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1807468129.py in <cell line: 0>()
      1 # Ensure baseline week rows exactly match provided baseline FVC and minimum confidence.
----> 2 mask_baseweek = test["Week"].to_numpy() == test["base_Weeks"].to_numpy()
      3 if mask_baseweek.any():
      4     test.loc[mask_baseweek, "FVC"] = test.loc[mask_baseweek, "base_FVC"].to_numpy()
      5     test.loc[mask_baseweek, "Confidence"] = 70.0

NameError: name 'test' is not defined
