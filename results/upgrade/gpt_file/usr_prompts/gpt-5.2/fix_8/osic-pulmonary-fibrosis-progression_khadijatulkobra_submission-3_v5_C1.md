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

-6.939142401520759

# 6. Current score

-7.78135

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -24.59128) has done: 'The timeout is dominated by repeated heavy DICOM I/O + 3D resampling inside `__getitem__` and again during test-time inference, plus per-row Python loops in `csv_preprocess` and per-sample inference loops. I keep the exact same model, losses, epochs, and image preprocessing math, but (1) cache CT volumes to disk as `.npy` so they are decoded/resized once per patient, (2) precompute and store tensors in the dataset to avoid repeated pandas/NumPy conversions, (3) enable multi-worker DataLoader with pinned memory and persistent workers to overlap CPU I/O with GPU, and (4) batch test inference so the model runs once per batch while reusing cached per-patient images. These changes preserve evaluation semantics (same inputs -> same outputs up to negligible float differences) but remove redundant work and drastically cut wall time.'
- What this solution (achieved -24.64601) has done: 'The timeout is dominated by repeated, expensive CT DICOM decoding + 3D resampling, and by avoidable CPU↔GPU syncs inside the training loop. I make CT preprocessing provably equivalent but much faster by (1) reading only required DICOM tags/pixels, (2) vectorizing HU conversion across slices, (3) switching the 3D resize to `scipy.ndimage.zoom(..., order=1, prefilter=False)` (same interpolation as default but less overhead), and (4) aggressively caching to `.npy` (already present) while also avoiding redundant loads in test-time batching. I also reduce per-iteration overhead by moving constant tensors onto the correct device once, avoiding `.cpu()` sync each batch, and using `optimizer.zero_grad(set_to_none=True)` (already) with faster accumulation. Core model, losses, features, epochs, and semantics remain unchanged.'
- What this solution (achieved -24.64601) has done: 'You’re hitting a runtime error because deterministic algorithms are enabled without setting the required `CUBLAS_WORKSPACE_CONFIG`, so the backward pass fails on CUDA. I fix this by setting that environment variable before importing/using torch (and falling back to non-deterministic only if needed), which preserves the training loop/model semantics while unblocking execution. I also fix a key logic bug in `csv_preprocess`: it mistakenly uses baseline weeks/FVC for all visits (so “Week” becomes 0 everywhere), which severely damages learning and explains the poor score; the fix keeps the same intended feature set but correctly uses each row’s `Weeks` and `FVC` as the visit target. Finally, I make test-time image loading reuse the existing `ImageCache` to avoid redundant disk reads and keep the end-to-end runtime stable while writing a valid `submission.csv`.'
- What this solution (achieved -7.78135) has done: 'I fix the runtime error by disabling strict deterministic algorithms (MaxPool3d backward on CUDA is not deterministic) while keeping seeds and CuDNN deterministic flags so training remains stable. This unblocks the backward pass without changing the model, loss, features, or training loop structure. I also ensure the script runs end-to-end and writes a valid `submission.csv` with the required columns, and I keep all I/O paths unchanged.'
- What this solution (achieved -7.78135) has done: 'Your current score (-7.78135) is worse than the target (-6.93914), so we should improve it with minimal semantic changes. The largest low-risk gain for this competition is to align the training objective with the evaluation metric by using the **negative** Laplace log-likelihood (your `score()` currently has the wrong sign, so the model is incentivized to get worse on the Kaggle metric). I fix `score()` to be the proper NLL (minimize NLL == maximize Kaggle metric) while keeping the same model, features, and training loop. I also clip the predicted `Confidence` at inference to be positive and then apply the same 70ml floor used by the metric to avoid pathological sigmas hurting score.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
import numpy as np
import pandas as pd
import scipy.ndimage

import pydicom

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE_INPUT = "/kaggle/input/osic-pulmonary-fibrosis-progression"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "../input/osic-pulmonary-fibrosis-progression"

TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
TEST_CSV = os.path.join(BASE_INPUT, "test.csv")
SAMPLE_SUB = os.path.join(BASE_INPUT, "sample_submission.csv")
TRAIN_FOLDER = os.path.join(BASE_INPUT, "train")
TEST_FOLDER = os.path.join(BASE_INPUT, "test")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def seed_worker(worker_id):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


seed_everything(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
device



## === cell 1
_DCM_TAGS = [
    "ImagePositionPatient",
    "InstanceNumber",
    "RescaleIntercept",
    "RescaleSlope",
]


def _safe_float_z(ds):
    try:
        ipp = ds.ImagePositionPatient
        return float(ipp[2])
    except Exception:
        try:
            return float(getattr(ds, "InstanceNumber", 0))
        except Exception:
            return 0.0


def load_scan(path):
    files = os.listdir(path)
    meta = []
    for fn in files:
        fp = path + os.sep + fn
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True, specific_tags=_DCM_TAGS)
            meta.append((fn, _safe_float_z(ds)))
        except Exception:
            meta.append((fn, np.nan))

    if all(np.isfinite(z) for _, z in meta):
        meta.sort(key=lambda t: t[1])
    else:
        meta.sort(key=lambda t: t[0])

    slices = []
    for fn, _ in meta:
        fp = path + os.sep + fn
        slices.append(pydicom.dcmread(fp))
    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices]).astype(np.int16, copy=False)

    image[image <= -2000] = 0

    try:
        slopes = np.array(
            [float(getattr(s, "RescaleSlope", 1.0)) for s in slices], dtype=np.float32
        )
        intercepts = np.array(
            [float(getattr(s, "RescaleIntercept", 0.0)) for s in slices],
            dtype=np.float32,
        )

        if not np.all(slopes == 1.0):
            image = image.astype(np.float32, copy=False)
            image *= slopes[:, None, None]
            image = image.astype(np.int16, copy=False)

        image = (image + intercepts[:, None, None].astype(np.int16)).astype(
            np.int16, copy=False
        )
    except Exception:
        pass

    return np.asarray(image, dtype=np.int16)


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


def _cache_dir_for(folder, Z, Y, X):
    safe = folder.strip("/").replace("/", "_")
    return os.path.join("/kaggle/working", f"ct_cache_{safe}_{Z}x{Y}x{X}")


def _blank_ct(Z, Y, X, value=0, dtype=np.uint8):
    return (np.zeros((Z, Y, X), dtype=dtype) + np.array(value, dtype=dtype)).astype(
        dtype, copy=False
    )


def read_image(dir_name, patientid, Z=100, Y=200, X=200):
    cache_dir = _cache_dir_for(dir_name, Z, Y, X)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{patientid}.npy")

    if os.path.exists(cache_path):
        try:
            arr = np.load(cache_path, mmap_mode=None)
            if arr.shape == (Z, Y, X):
                return arr
        except Exception:
            pass  # regenerate

    path = dir_name + os.sep + patientid
    try:
        slices = load_scan(path)
        image_array = get_pixels_hu(slices)
        ctimage_resizedAll = resize_along_allaxis(
            image_array, target_dimensionX=X, target_dimensionY=Y, target_dimensionZ=Z
        )
        image = (image_normalize(ctimage_resizedAll) * 255.0).astype("uint8")
    except Exception:
        image = _blank_ct(Z, Y, X, value=0, dtype=np.uint8)

    np.save(cache_path, image, allow_pickle=False)
    return image




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

    base = data.groupby("Patient", as_index=False).nth(0).reset_index(drop=True)
    base = base.rename(columns={"Weeks": "base_Weeks", "FVC": "base_FVC"})

    visits = data[["Patient", "Weeks", "FVC"]].copy()
    visits = visits.rename(columns={"Weeks": "Week", "FVC": "actual_FVC"})

    npData = visits.merge(base, on="Patient", how="left")

    keep_cols = (
        ["Patient", "base_Weeks", "base_FVC", "Age"]
        + FE1
        + ["Week", "Healthy-FVC", "actual_FVC"]
    )
    for c in FE1:
        if c not in npData.columns:
            npData[c] = 0
    npData = npData[keep_cols].copy()

    npData.reset_index(inplace=True, drop=True)
    npData = npData.fillna(0)

    npData["Week"] = npData["Week"] - npData["base_Weeks"]
    npData["base_Weeks"] = 0.0

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
C1 = torch.tensor(70.0, dtype=torch.float32, device=device)
C2 = torch.tensor(1000.0, dtype=torch.float32, device=device)
SQRT2 = torch.sqrt(torch.tensor(2.0, dtype=torch.float32, device=device))


def score(y_true, y_pred):
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = torch.clamp(sigma, min=float(C1))
    delta = (y_true[:, 0] - fvc_pred).abs()
    delta = torch.clamp(delta, max=float(C2))

    nll = (delta / sigma_clip) * SQRT2 + (sigma_clip * SQRT2).log()
    return nll.mean()


def qloss(y_true, y_pred):
    qs = [0.25, 0.50, 0.75]
    q = torch.tensor(np.array([qs]), device=y_pred.device, dtype=torch.float32)
    e = y_true - y_pred
    v = torch.max(q * e, (q - 1) * e)
    return v.mean()


def quartile_loss(y_true, y_pred, _lambda=0.65):
    loss = _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)
    return loss




## === cell 5
raw_train = pd.read_csv(TRAIN_CSV)
train_df = csv_preprocess(raw_train)

FEATURE_COLS = [
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
TARGET_COL = "actual_FVC"

train_df[FEATURE_COLS] = train_df[FEATURE_COLS].astype(np.float32)
train_df[TARGET_COL] = train_df[TARGET_COL].astype(np.float32)

train_df.head()




## === cell 6
class ImageCache:
    def __init__(self, folder, Z=100, Y=200, X=200):
        self.folder = folder
        self.Z, self.Y, self.X = Z, Y, X
        self.cache = {}

    def get(self, patient_id):
        if patient_id not in self.cache:
            self.cache[patient_id] = read_image(
                self.folder, patient_id, Z=self.Z, Y=self.Y, X=self.X
            )
        return self.cache[patient_id]


class OSICDataset(Dataset):
    def __init__(self, df, image_cache):
        self.df = df.reset_index(drop=True)
        self.image_cache = image_cache

        self.pids = self.df["Patient"].astype(str).values
        self.x = np.ascontiguousarray(self.df[FEATURE_COLS].values, dtype=np.float32)
        self.y = np.ascontiguousarray(self.df[TARGET_COL].values, dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pid = self.pids[idx]
        img = self.image_cache.get(pid).astype(np.float32, copy=False)
        img = torch.from_numpy(img).unsqueeze(0)  # [1, Z, Y, X]
        feats = torch.from_numpy(self.x[idx])
        y = torch.tensor([self.y[idx]], dtype=torch.float32)
        return img, feats, y


patients = train_df["Patient"].unique().tolist()
rng = np.random.RandomState(42)
rng.shuffle(patients)
n_valid = max(1, int(0.2 * len(patients)))
valid_patients = set(patients[:n_valid])
train_patients = set(patients[n_valid:])

df_tr = train_df[train_df["Patient"].isin(train_patients)].copy()
df_va = train_df[train_df["Patient"].isin(valid_patients)].copy()

img_cache_train = ImageCache(TRAIN_FOLDER, Z=100, Y=200, X=200)
train_ds = OSICDataset(df_tr, img_cache_train)
valid_ds = OSICDataset(df_va, img_cache_train)

num_workers = min(4, (os.cpu_count() or 2))
pin = torch.cuda.is_available()
g = torch.Generator()
g.manual_seed(42)

train_loader = DataLoader(
    train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin,
    persistent_workers=(num_workers > 0),
    worker_init_fn=seed_worker,
    generator=g,
)
valid_loader = DataLoader(
    valid_ds,
    batch_size=2,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin,
    persistent_workers=(num_workers > 0),
    worker_init_fn=seed_worker,
)

len(train_ds), len(valid_ds)




## === cell 7
def df_append_compat(df, other, sort=False):
    return pd.concat([df, other], ignore_index=True, sort=sort)




## === cell 8
model = Combined_NET().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def train_one_epoch(model, loader):
    model.train()
    total = 0.0
    n = 0
    for img, feats, y in loader:
        img = img.to(device, non_blocking=True)
        feats = feats.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(img, feats)  # [B,3]
        loss = quartile_loss(y, pred)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = img.size(0)
        total += loss.detach().item() * bs
        n += bs
    return total / max(n, 1)


@torch.no_grad()
def valid_epoch(model, loader):
    model.eval()
    total = 0.0
    n = 0
    for img, feats, y in loader:
        img = img.to(device, non_blocking=True)
        feats = feats.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(img, feats)
        loss = quartile_loss(y, pred)
        bs = img.size(0)
        total += loss.detach().item() * bs
        n += bs
    return total / max(n, 1)


EPOCHS = 2
for ep in range(1, EPOCHS + 1):
    tr_loss = train_one_epoch(model, train_loader)
    va_loss = valid_epoch(model, valid_loader)
    print(f"epoch {ep}/{EPOCHS} train_loss={tr_loss:.5f} valid_loss={va_loss:.5f}")



## === cell 9
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
].copy()
sub_base = merge.loc[:, ["Patient_Week", "base_FVC", "Confidence"]].copy()
sub_base = sub_base.rename(columns={"base_FVC": "FVC"})

data = data_test.copy()
data["Healthy-FVC"] = round((data["base_FVC"] * 100) / data["Percent"])
COLS = ["Sex", "SmokingStatus"]
for col in COLS:
    for mod in data[col].unique():
        data[mod] = (data[col] == mod).astype(int)

FE1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]
for c in FE1:
    if c not in data.columns:
        data[c] = 0

data = data.fillna(0)
data["Week"] = data["Week"] - data["base_Weeks"]
data["base_Weeks"] = 0.0

data_test_features = data[
    ["Patient", "base_Weeks", "base_FVC", "Age", "Healthy-FVC"] + FE1 + ["Week"]
].copy()
data_test_features[FEATURE_COLS] = data_test_features[FEATURE_COLS].astype(np.float32)

data_test_features.head()




## === cell 10
@torch.no_grad()
def make_eval_data(npEval, model, image_cache, device=device, batch_size=4):
    x_features = np.ascontiguousarray(npEval[FEATURE_COLS].values, dtype=np.float32)
    pids = npEval["Patient"].astype(str).values

    model = model.to(device)
    model.eval()

    n = len(npEval)
    preds = np.empty((n, 3), dtype=np.float32)

    imgs = np.empty((batch_size, 1, 100, 200, 200), dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        bs = end - start

        for j in range(bs):
            imgs[j, 0] = image_cache.get(pids[start + j]).astype(np.float32, copy=False)

        x_image = torch.from_numpy(imgs[:bs]).to(device, non_blocking=True)
        x_feature = torch.from_numpy(x_features[start:end]).to(
            device, non_blocking=True
        )

        prediction = model(x_image, x_feature)
        preds[start:end] = prediction.detach().cpu().numpy()

    out = npEval.copy()
    out["FVC"] = preds[:, 1]

    conf = preds[:, 2] - preds[:, 0]
    conf = np.maximum(conf, 1.0)
    out["Confidence"] = conf
    return out


img_cache_test = ImageCache(TEST_FOLDER, Z=100, Y=200, X=200)
test_pred_df = make_eval_data(data_test_features.copy(), model, img_cache_test)
test_pred_df[["FVC", "Confidence"]].describe()



## === cell 11
pred_with_key = pd.concat(
    [
        data_test_features[["Patient"]].reset_index(drop=True),
        merge[["Patient_Week"]].reset_index(drop=True),
        test_pred_df[["FVC", "Confidence"]].reset_index(drop=True),
    ],
    axis=1,
)

final_sub = pd.read_csv(SAMPLE_SUB)[["Patient_Week"]].copy()
final_sub = final_sub.merge(
    pred_with_key[["Patient_Week", "FVC", "Confidence"]], on="Patient_Week", how="left"
)

final_sub["FVC"] = pd.to_numeric(final_sub["FVC"], errors="coerce")
final_sub["Confidence"] = pd.to_numeric(final_sub["Confidence"], errors="coerce")

final_sub["FVC"] = final_sub["FVC"].fillna(final_sub["FVC"].median())
final_sub["Confidence"] = final_sub["Confidence"].fillna(200.0)

final_sub["Confidence"] = final_sub["Confidence"].clip(lower=70.0)

final_sub.head(), final_sub.isna().sum()



## === cell 12
final_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final_sub.shape)
print(final_sub.columns.tolist())
print(final_sub.head(3).to_string(index=False))
