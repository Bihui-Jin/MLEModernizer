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

-6.879888052224982

# 6. Current score

-12.55535

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -13.35832) has done: 'The timeout is dominated by repeatedly decoding and resizing large DICOM volumes inside `__getitem__` and again for test inference; each epoch/worker ends up re-reading the same patient scan many times. I keep the exact same preprocessing and model, but cache CT volumes aggressively in memory during the run (in addition to your existing on-disk `.npy` cache) and make the DataLoader reuse workers efficiently. I also avoid extra per-batch tensor/constant constructions in the loss and reduce Python overhead in image assembly for inference by preallocating and bulk-filling from the cache. These changes are runtime-only and preserve identical inputs to the network (same images/normalization/resize) and identical training/inference semantics.'
- What this solution (achieved -13.35832) has done: 'I fix the “Not yielded” issue by ensuring the submission rows exactly match `sample_submission.csv` (same `Patient_Week` set and order), because generating a different template can lead to an invalid submission or unintended scoring behavior. I keep your model and preprocessing intact, but adjust inference to run on the canonical sample submission weeks by merging in the baseline clinical fields from `test.csv`. I also make a minimal, metric-aligned post-processing tweak: set `Confidence` to at least 70 and optionally calibrate it using the model’s sigma output, while leaving `FVC` predictions unchanged except for the required baseline-week overwrite. These are small changes focused on producing a valid submission and nudging score upward without changing core training/model logic.'
- What this solution (achieved -14.69969) has done: 'Your current score is far below the target (gap ≈ -6.48, higher-is-better), so we should improve it with minimal, metric-aligned changes while keeping your model and preprocessing intact. The biggest score issue here is a label/metric mismatch: your training target `y_true` is built as `[actual_FVC, 0]` but `score()` reads `y_true[:,0]` as the true FVC (OK) while it expects prediction columns as `[sigma, fvc]`; however you never train sigma to be positive/meaningful and you also don’t apply the correct (negative) Laplace log-likelihood sign, which can push optimization the wrong way. I keep the same architecture and training loop, but fix `y_true` to be a single FVC target and update the loss to correctly implement the competition’s metric form (as a minimization objective) while leaving inference/output schema unchanged. I also ensure `sigma` is strictly positive at inference (via `softplus`) and still clipped to [70, 500], which usually improves the public metric without changing the model structure.'
- What this solution (achieved -14.69969) has done: 'Your current score (-14.70) is far below the target (-6.88), so we should improve it with minimal, metric-aligned changes without touching the model architecture or training loop. The biggest issue is that the network’s output is treated as `[sigma, FVC]` but your training loss also assumes that order; however at inference you apply `softplus` twice to `Confidence` (once implicitly in the loss during training via `softplus(sigma_raw)`, then again in cell 12), which tends to inflate sigma and hurts the Laplace log-likelihood score. I (1) remove the second `softplus` in inference and instead only clip sigma to the required bounds, and (2) add a small, deterministic sigma calibration on the training set (using the already-trained model) to better match the competition’s sigma scale, which typically boosts the metric while preserving core logic. These changes keep inputs/outputs and submission format identical and should move the score upward toward your target.'
- What this solution (achieved -13.38643) has done: 'Your current score is far below the target (gap ≈ -7.82; higher is better), so we should improve it with the smallest metric-aligned change that doesn’t alter your model/training core. The biggest avoidable hit is that you apply `softplus` to `Confidence` a second time at inference even though training already interprets the first output as `sigma_raw` and the loss applies `softplus`; this double-softplus inflates sigma and lowers the Laplace log-likelihood. I remove the second `softplus` and instead apply the same transform as in the loss exactly once: `sigma = softplus(sigma_raw)` then scale and clip. I also calibrate sigma using the median of *clipped* sigma (consistent with the metric’s sigma_clipped) to avoid pushing many predictions below 70 and to better match the evaluation semantics.'
- What this solution (achieved -12.71292) has done: 'Your gap to the target is large (current -13.386 vs target -6.880; higher is better), so we should improve score with the smallest metric-aligned changes that keep your model and training intact. The most leverage here is post-processing: (1) enforce the known baseline FVC constraint for all baseline-week rows (not just the first duplicate), (2) replace the current median-ratio sigma scaling with a direct, deterministic grid-search over a single global sigma scale that maximizes the Laplace log-likelihood on a small, fixed calibration subset, and (3) optionally apply a tiny “shrink-to-baseline” factor for non-baseline weeks (a safe linear blend) tuned on the same subset to reduce large deltas without changing the model. These are lightweight, deterministic, and preserve architecture/training while directly optimizing the competition metric. The submission generation and row alignment with `sample_submission.csv` remain unchanged.'
- What this solution (achieved -12.55535) has done: 'I keep your model/training exactly as-is and only adjust the post-processing calibration that directly targets the competition metric. Your current sigma calibration searches only a few coarse scales and clips sigma to 500, which can be overly restrictive and leave metric on the table; I (1) widen the sigma clip upper bound (still respecting the metric’s lower clip at 70) and (2) replace the coarse grid with a deterministic two-stage search (coarse + local refine) to choose a better single global sigma_scale. I also tune the shrink-to-baseline factor on a slightly richer grid (still tiny/cheap) using the same fixed calibration subset, keeping the baseline-week overwrite intact. These changes are minimal, deterministic, and should move the score upward toward your target without changing core logic.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import scipy.ndimage

import pydicom

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset

os.environ.setdefault("PYTHONHASHSEED", "0")
torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False



## === cell 1
BASE_INPUT = "../input/osic-pulmonary-fibrosis-progression"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "../data/osic-pulmonary-fibrosis-progression"

TRAIN_FOLDER = os.path.join(BASE_INPUT, "train")
TEST_FOLDER = os.path.join(BASE_INPUT, "test")


def load_scan(path):  # path == .../train/patientId
    files = os.listdir(path)
    if not files:
        return []
    full_paths = [os.path.join(path, s) for s in files]

    slices = []
    for fp in full_paths:
        try:
            ds_h = pydicom.dcmread(
                fp,
                stop_before_pixels=True,
                specific_tags=["ImagePositionPatient", "InstanceNumber"],
            )
            if (
                hasattr(ds_h, "ImagePositionPatient")
                and ds_h.ImagePositionPatient is not None
            ):
                key = float(ds_h.ImagePositionPatient[2])
            else:
                key = int(getattr(ds_h, "InstanceNumber", 0))
        except Exception:
            key = 0
        slices.append((key, fp))

    slices.sort(key=lambda t: t[0])

    out = []
    for _, fp in slices:
        try:
            ds = pydicom.dcmread(
                fp, specific_tags=["PixelData", "RescaleIntercept", "RescaleSlope"]
            )
            out.append(ds)
        except Exception:
            try:
                out.append(pydicom.dcmread(fp))
            except Exception:
                pass
    return out


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices]).astype(np.int16, copy=False)
    try:
        image[image <= -2000] = 0

        intercept = np.array(
            [float(getattr(s, "RescaleIntercept", 0.0)) for s in slices],
            dtype=np.float32,
        )
        slope = np.array(
            [float(getattr(s, "RescaleSlope", 1.0)) for s in slices], dtype=np.float32
        )

        out = image.astype(np.float32, copy=False)
        out *= slope[:, None, None]
        out += intercept[:, None, None]
        image = out.astype(np.int16, copy=False)
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


_CACHE_DIR = os.path.join("../working", "osic_ct_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)


def _cache_path(patientid, Z, Y, X):
    return os.path.join(_CACHE_DIR, f"{patientid}_Z{Z}_Y{Y}_X{X}.npy")


_CT_MEM_CACHE = (
    {}
)  # key: (dir_name, patientid, Z, Y, X) -> np.ndarray(uint8) shape (Z,Y,X)


def read_image(dir_name, patientid, Z=100, Y=200, X=200):
    mkey = (dir_name, patientid, int(Z), int(Y), int(X))
    cached = _CT_MEM_CACHE.get(mkey, None)
    if cached is not None:
        return cached

    cpath = _cache_path(patientid, Z, Y, X)
    if os.path.exists(cpath):
        try:
            arr = np.load(cpath, allow_pickle=False, mmap_mode="r")
            arr = np.asarray(arr)
            _CT_MEM_CACHE[mkey] = arr
            return arr
        except Exception:
            pass

    path = dir_name + os.sep + patientid
    slices = load_scan(path)
    try:
        image_array = get_pixels_hu(slices)
        ctimage_resizedAll = resize_along_allaxis(
            image_array, target_dimensionX=X, target_dimensionY=Y, target_dimensionZ=Z
        )
        image = (image_normalize(ctimage_resizedAll) * 255.0).astype(
            "uint8", copy=False
        )
        try:
            np.save(cpath, image, allow_pickle=False)
        except Exception:
            pass
        _CT_MEM_CACHE[mkey] = image
        return image
    except Exception:
        print(f"PatientId:{patientid} couldnt be converted")
        return None




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

    base = data.groupby("Patient", sort=False).head(1).copy()

    wk = data[["Patient", "base_Weeks", "base_FVC"]].rename(
        columns={"base_Weeks": "Week", "base_FVC": "actual_FVC"}
    )

    counts = wk["Patient"].value_counts(sort=False)
    base_rep = base.loc[
        base.index.repeat(base["Patient"].map(counts).values)
    ].reset_index(drop=True)
    wk_sorted = wk.sort_values(["Patient", "Week"], ascending=True).reset_index(
        drop=True
    )
    base_rep["Week"] = wk_sorted["Week"].values
    base_rep["actual_FVC"] = wk_sorted["actual_FVC"].values

    out_cols = (
        ["Patient", "base_Weeks", "base_FVC", "Age"]
        + FE1
        + ["Week", "Healthy-FVC", "actual_FVC"]
    )
    for c in FE1:
        if c not in base_rep.columns:
            base_rep[c] = 0
    if "Healthy-FVC" not in base_rep.columns:
        base_rep["Healthy-FVC"] = 0

    npData = base_rep[out_cols].fillna(0).reset_index(drop=True)
    return npData




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
            nn.Linear(42, 64), nn.ReLU(), nn.Linear(64, 118), nn.ReLU()
        )
        self.data_net2 = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 502), nn.ReLU()
        )
        self.data_net3 = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 118), nn.ReLU()
        )

        self.data_net4 = nn.Sequential(
            nn.Linear(812, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.ReLU(),
        )

    @staticmethod
    def _fit_to_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
        if x.shape[-1] == target_dim:
            return x
        if x.shape[-1] > target_dim:
            return x[..., :target_dim]
        pad = torch.zeros(
            x.shape[0], target_dim - x.shape[-1], dtype=x.dtype, device=x.device
        )
        return torch.cat([x, pad], dim=-1)

    def forward(self, data_i, image_o):
        out1 = self.data_net1(data_i)  # expects 42
        x = torch.cat((data_i, image_o), dim=-1)  # 42 + 32 = 74
        out2 = torch.cat((data_i, out1), dim=-1)  # 42 + 118 = 160
        out2 = self._fit_to_dim(out2, 128)
        out2 = self.data_net2(out2)  # expects 128
        out3 = torch.cat((data_i, out2), dim=-1)  # 42 + 502 = 544
        out3 = self._fit_to_dim(out3, 512)
        out3 = self.data_net3(out3)  # expects 512
        out4 = torch.cat((x, out1, out2, out3), dim=-1)  # 812 dims
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
            "conv_%d" % i,
            nn.Conv3d(in_channel, out_channel, padding=0, kernel_size=1),
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
C1 = 70.0
C2 = 1000.0
SQRT2 = float(np.sqrt(2.0))


def laplace_nll_loss(y_true_fvc: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    y_true_fvc: shape (B, 1) or (B,)
    y_pred: shape (B, 2) with columns [sigma_raw, fvc_pred]
    """
    if y_true_fvc.dim() == 2:
        y_true_fvc = y_true_fvc[:, 0]

    sigma_raw = y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma = F.softplus(sigma_raw) + 1e-6
    sigma_clip = torch.clamp(sigma, min=C1)

    delta = (y_true_fvc - fvc_pred).abs()
    delta = torch.clamp(delta, max=C2)

    sq2 = sigma_clip.new_tensor(SQRT2)
    nll = (sq2 * delta) / sigma_clip + torch.log(sq2 * sigma_clip)
    return nll.mean()


def quartile_loss(y_true, y_pred):
    return laplace_nll_loss(y_true, y_pred)




## === cell 5
def pad_features_to_42(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 2:
        raise ValueError(f"Expected 2D tensor for features, got shape {tuple(x.shape)}")
    need = 42 - x.shape[1]
    if need < 0:
        return x[:, :42]
    if need == 0:
        return x
    pad = torch.zeros((x.shape[0], need), dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=1)




## === cell 6
_TAB_FEATURES_10 = [
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


def make_eval_data(npEval, model, device="cuda", batch_size=8):
    npEval = npEval.copy()

    missing = [c for c in _TAB_FEATURES_10 if c not in npEval.columns]
    if missing:
        raise KeyError(f"Missing required feature columns for inference: {missing}")

    x_features_df = npEval[_TAB_FEATURES_10].copy()
    x_features = torch.tensor(
        x_features_df.to_numpy(dtype=np.float32, copy=True)
    ).float()
    x_features = pad_features_to_42(x_features)

    x_patientids_name = npEval[["Patient"]].values.squeeze(1)

    unique_patients = npEval.Patient.unique()
    loaded_images = {}
    dir_name_of_patientid = TEST_FOLDER
    for unique_patient in unique_patients:
        loaded_images[unique_patient] = read_image(
            dir_name_of_patientid, unique_patient, Z=100, Y=200, X=200
        )

    x_images = np.empty((len(npEval), 1, 100, 200, 200), dtype=np.float32)
    zero_img = np.zeros((100, 200, 200), dtype=np.float32)
    for i, pid in enumerate(x_patientids_name):
        x_image = loaded_images.get(pid, None)
        if x_image is None:
            x_images[i, 0] = zero_img
        else:
            x_images[i, 0] = x_image.astype(np.float32, copy=False)

    x_images = torch.from_numpy(x_images)

    use_cuda = torch.cuda.is_available() and device == "cuda"
    if use_cuda:
        model.to("cuda")
    model.eval()

    ds = TensorDataset(x_images, x_features)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
    )

    preds = []
    with torch.no_grad():
        for xb_img, xb_feat in dl:
            if use_cuda:
                xb_img = xb_img.cuda(non_blocking=True)
                xb_feat = xb_feat.cuda(non_blocking=True)
            out = model(xb_img, xb_feat)
            preds.append(out.detach().cpu().numpy())
    predictions = np.concatenate(preds, axis=0)

    npEval["FVC"] = predictions[:, 1]
    npEval["Confidence"] = predictions[:, 0]  # sigma_raw; transform later exactly once
    return npEval




## === cell 7
data_train = pd.read_csv(os.path.join(BASE_INPUT, "train.csv"))
data_test = pd.read_csv(os.path.join(BASE_INPUT, "test.csv"))
submission = pd.read_csv(os.path.join(BASE_INPUT, "sample_submission.csv"))



## === cell 8
sub_tpl = submission[["Patient_Week"]].copy()
sub_tpl[["Patient", "Week"]] = sub_tpl["Patient_Week"].str.rsplit("_", n=1, expand=True)
sub_tpl["Week"] = sub_tpl["Week"].astype(int)

base = data_test.rename(columns={"Weeks": "base_Weeks", "FVC": "base_FVC"}).copy()
merge = sub_tpl.merge(base, on="Patient", how="left")

data_test_eval = merge.loc[
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
        "Patient_Week",
    ],
].copy()

submission_out = merge.loc[:, ["Patient_Week"]].copy()
submission_out["FVC"] = merge["base_FVC"].values
submission_out["Confidence"] = 70.0

del merge, sub_tpl, base



## === cell 9
data = data_test_eval.copy()
data["Healthy-FVC"] = np.round((data["base_FVC"] * 100) / data["Percent"]).astype(
    np.float32
)

for col in ["Sex", "SmokingStatus"]:
    for mod in data[col].unique():
        data[mod] = (data[col] == mod).astype(np.int32)

for c in ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]:
    if c not in data.columns:
        data[c] = 0

data_test_eval = data[
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
        "Patient_Week",
    ]
].copy()

del data




## === cell 10
class OSICTrainDataset(Dataset):
    def __init__(self, npTrain: pd.DataFrame):
        self.npTrain = npTrain.reset_index(drop=True)

        x_features = self.npTrain[_TAB_FEATURES_10]
        self.x_features = pad_features_to_42(
            torch.tensor(x_features.to_numpy(dtype=np.float32, copy=True)).float()
        )

        self.y = torch.tensor(
            self.npTrain[["actual_FVC"]].to_numpy(dtype=np.float32, copy=True)
        ).float()

        self.pids = self.npTrain["Patient"].values

    def __len__(self):
        return len(self.npTrain)

    def __getitem__(self, idx):
        pid = self.pids[idx]
        img = read_image(TRAIN_FOLDER, pid, Z=100, Y=200, X=200)
        if img is None:
            img = np.zeros((100, 200, 200), dtype=np.uint8)
        x_img = torch.from_numpy(img.astype(np.float32, copy=False))[None, ...]
        x_feat = self.x_features[idx]
        y = self.y[idx]
        return x_img, x_feat, y


def build_train_dataset(data_train_df, max_patients_for_images=None):
    npTrain = csv_preprocess(data_train_df)

    if max_patients_for_images is not None:
        keep = npTrain["Patient"].unique()[:max_patients_for_images]
        npTrain = npTrain[npTrain["Patient"].isin(keep)].reset_index(drop=True)

    return OSICTrainDataset(npTrain)




## === cell 11
torch.manual_seed(0)
np.random.seed(0)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Combined_NET()

weights_path = (
    "../input/ww-5-677/Epoch5_Score6.774192634481468_Acc0.9327437837787022.pth"
)
loaded = False
if os.path.exists(weights_path):
    state = torch.load(weights_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    loaded = True
    print("Loaded weights non-strictly.")
    if missing:
        print(
            "Missing keys (initialized randomly):",
            missing[:5],
            "..." if len(missing) > 5 else "",
        )
    if unexpected:
        print(
            "Unexpected keys (ignored):",
            unexpected[:5],
            "..." if len(unexpected) > 5 else "",
        )

if not loaded:
    ds = build_train_dataset(data_train, max_patients_for_images=None)

    if device == "cuda":
        batch_size = 4
        num_workers = min(8, os.cpu_count() or 1)
    else:
        batch_size = 1
        num_workers = 0

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    epochs = 1
    for ep in range(epochs):
        for xb_img, xb_feat, yb in dl:
            xb_img = xb_img.to(device, non_blocking=True)
            xb_feat = xb_feat.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            pred = model(xb_img, xb_feat)
            loss = quartile_loss(yb, pred)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

    model.to("cpu")




## === cell 12
def _metric_np(fvc_true, fvc_pred, sigma_pred):
    sigma_clipped = np.maximum(sigma_pred, C1)
    delta = np.minimum(np.abs(fvc_true - fvc_pred), C2)
    return (-SQRT2 * delta / sigma_clipped) - np.log(SQRT2 * sigma_clipped)


def _calibrate_postprocess_from_train(
    model,
    data_train_df,
    device="cuda",
    n_calib=192,  # keep small for speed; deterministic subset
    batch_size=4,
):
    npTrain = csv_preprocess(data_train_df)

    if len(npTrain) == 0:
        return 1.0, 0.0
    npCal = npTrain.iloc[: min(n_calib, len(npTrain))].copy().reset_index(drop=True)

    ds = OSICTrainDataset(npCal)
    use_cuda = torch.cuda.is_available() and device == "cuda"
    model.eval()
    if use_cuda:
        model.to("cuda")

    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=use_cuda
    )

    fvc_true_all, fvc_pred_all, sigma_raw_all, base_fvc_all, week_all, base_week_all = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    with torch.no_grad():
        for xb_img, xb_feat, yb in dl:
            if use_cuda:
                xb_img = xb_img.cuda(non_blocking=True)
                xb_feat = xb_feat.cuda(non_blocking=True)
                yb = yb.cuda(non_blocking=True)
            out = model(xb_img, xb_feat)
            sigma_raw_all.append(out[:, 0].detach().cpu().numpy())
            fvc_pred_all.append(out[:, 1].detach().cpu().numpy())
            fvc_true_all.append(yb[:, 0].detach().cpu().numpy())
            base_fvc_all.append(
                xb_feat[:, 1].detach().cpu().numpy()
            )  # base_FVC is feature index 1 in _TAB_FEATURES_10
            base_week_all.append(
                xb_feat[:, 0].detach().cpu().numpy()
            )  # base_Weeks is feature index 0
            week_all.append(
                xb_feat[:, 8].detach().cpu().numpy()
            )  # Week is feature index 8

    if use_cuda:
        model.to("cpu")

    fvc_true = np.concatenate(fvc_true_all).astype(np.float32)
    fvc_pred = np.concatenate(fvc_pred_all).astype(np.float32)
    sigma_raw = np.concatenate(sigma_raw_all).astype(np.float32)
    base_fvc = np.concatenate(base_fvc_all).astype(np.float32)
    base_week = np.concatenate(base_week_all).astype(np.float32)
    week = np.concatenate(week_all).astype(np.float32)

    sigma0 = F.softplus(torch.from_numpy(sigma_raw)).numpy().astype(np.float32) + 1e-6

    is_base = week == base_week
    fvc_pred_adj0 = fvc_pred.copy()
    fvc_pred_adj0[is_base] = base_fvc[is_base]

    SIGMA_MAX = 1000.0

    coarse = np.linspace(0.4, 2.4, 21, dtype=np.float32)  # step 0.1
    best_scale = 1.0
    best_score = -1e18
    for s in coarse:
        sig = np.clip(sigma0 * s, 70.0, SIGMA_MAX)
        sc = float(_metric_np(fvc_true, fvc_pred_adj0, sig).mean())
        if sc > best_score:
            best_score = sc
            best_scale = float(s)

    refine = np.linspace(
        max(0.1, best_scale - 0.2), best_scale + 0.2, 41, dtype=np.float32
    )  # step 0.01
    for s in refine:
        sig = np.clip(sigma0 * s, 70.0, SIGMA_MAX)
        sc = float(_metric_np(fvc_true, fvc_pred_adj0, sig).mean())
        if sc > best_score:
            best_score = sc
            best_scale = float(s)

    alphas = np.array([0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18], dtype=np.float32)
    best_alpha = 0.0
    best_score2 = -1e18
    sig_best = np.clip(sigma0 * np.float32(best_scale), 70.0, SIGMA_MAX)
    for a in alphas:
        fvc_pp = fvc_pred_adj0.copy()
        nb = ~is_base
        fvc_pp[nb] = base_fvc[nb] + (1.0 - a) * (fvc_pp[nb] - base_fvc[nb])
        sc = float(_metric_np(fvc_true, fvc_pp, sig_best).mean())
        if sc > best_score2:
            best_score2 = sc
            best_alpha = float(a)

    return float(best_scale), float(best_alpha), float(SIGMA_MAX)


sigma_scale, fvc_shrink_alpha, _SIGMA_MAX = _calibrate_postprocess_from_train(
    model, data_train, device=device, n_calib=192, batch_size=4
)
print(
    "Chosen sigma_scale:",
    sigma_scale,
    "Chosen fvc_shrink_alpha:",
    fvc_shrink_alpha,
    "SigmaMax:",
    _SIGMA_MAX,
)

test_pred = make_eval_data(data_test_eval.copy(), model, device=device, batch_size=8)

is_base_test = test_pred["Week"].to_numpy() == test_pred["base_Weeks"].to_numpy()
test_pred.loc[is_base_test, "FVC"] = test_pred.loc[is_base_test, "base_FVC"].to_numpy()
test_pred.loc[is_base_test, "Confidence"] = np.float32(70.0)

sigma_raw = torch.tensor(test_pred["Confidence"].to_numpy(np.float32))
conf = (F.softplus(sigma_raw) + 1e-6).numpy().astype(np.float32)
conf = conf * np.float32(sigma_scale)
test_pred["Confidence"] = conf

if fvc_shrink_alpha > 0:
    nb = ~is_base_test
    base_fvc = test_pred.loc[nb, "base_FVC"].to_numpy(np.float32)
    fvc = test_pred.loc[nb, "FVC"].to_numpy(np.float32)
    test_pred.loc[nb, "FVC"] = base_fvc + (1.0 - np.float32(fvc_shrink_alpha)) * (
        fvc - base_fvc
    )

test_pred["Confidence"] = np.clip(
    test_pred["Confidence"].astype(np.float32), 70.0, np.float32(_SIGMA_MAX)
)

test_pred["FVC"] = test_pred["FVC"].astype(np.float32)
test_pred.loc[test_pred["FVC"] < 0, "FVC"] = 0.0

sub_final = submission[["Patient_Week"]].copy()
sub_final = sub_final.merge(
    test_pred[["Patient_Week", "FVC", "Confidence"]],
    on="Patient_Week",
    how="left",
)

if sub_final[["FVC", "Confidence"]].isna().any().any():
    tmp = data_test.rename(columns={"Weeks": "base_Weeks", "FVC": "base_FVC"})[
        ["Patient", "base_FVC"]
    ].copy()
    tmp["Patient_Week"] = (
        tmp["Patient"].astype(str) + "_" + data_test["Weeks"].astype(int).astype(str)
    )
    base_map = dict(zip(tmp["Patient"], tmp["base_FVC"]))
    pw_patient = sub_final["Patient_Week"].str.split("_", n=1, expand=True)[0]
    sub_final["FVC"] = sub_final["FVC"].fillna(pw_patient.map(base_map)).fillna(0.0)
    sub_final["Confidence"] = sub_final["Confidence"].fillna(70.0)

sub_final = sub_final[["Patient_Week", "FVC", "Confidence"]].copy()
sub_final.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_final.shape)
print(sub_final.head())
