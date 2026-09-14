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

-6.898329111219623

# 6. Current score

-24.64668

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -24.64601) has done: 'I fix the runtime error by removing the non-picklable nested `_load_one` function used with `multiprocessing.Pool` in `make_eval_data`, switching to a safe sequential image-loading path for test-time (only 18 patients, so it stays fast and stable). This unblocks inference so `test` is created and the submission-writing cells run without `NameError`. I also add a small safety fallback so if any test CT fails to load, the code substitutes a zero-volume tensor to still produce a valid submission. These changes are execution/stability fixes and should be score-neutral aside from allowing the model to actually generate predictions.'
- What this solution (achieved -24.64675) has done: 'Your score is far below the target, so we should improve (increase) it with the smallest changes that don’t alter the model/training core logic. The biggest issue is that the model’s FVC head is unconstrained, and at inference you only clip Confidence but not the implied quantile ordering; this can produce negative/unstable sigmas and badly calibrated confidence, which the Laplace metric heavily penalizes. I add a minimal, metric-aligned post-processing step in `make_eval_data` to (1) enforce monotonic quantiles `q25<=q50<=q75`, (2) convert them into a valid `(FVC, Confidence)` with `Confidence=max(q75-q25, 70)`, and (3) clamp FVC to a reasonable physiological range. This keeps architecture/training intact and only fixes inference semantics to match the evaluation, which should move the score substantially toward your target.'
- What this solution (achieved -24.64668) has done: 'I fix the test-frame construction so it matches the training feature schema expected by `make_eval_data` (the current merge creates `Weeks_x/Weeks_y` and then tries to sort by a non-existent `Weeks`, cascading into missing `base_FVC` and missing engineered columns). I minimally change cells 7–8 to build `data_test` using the same `csv_preprocess` function used for training, but with the submission weeks injected as the per-row `Weeks` and the baseline row replicated per week. This preserves the model/training logic and only corrects inference data alignment so the pipeline runs end-to-end and writes a valid `submission.csv`. With the corrected features, inference execute and should materially improve score versus the broken/placeholder output (previously “Not yielded”).'
- What this solution (achieved -24.64668) has done: 'Your current score is far below the target (higher is better), so we should improve it with the smallest changes that don’t alter the model/training core. The main score-killer here is that your model outputs “quantiles” but the loss/metric assumes the 0.5 output is the FVC and the spread implies sigma; without enforcing valid quantile ordering inside the loss, training can learn inconsistent (crossing) quantiles and then the metric gets heavily penalized. I minimally enforce `q25<=q50<=q75` inside `qloss` and `score` (training-time, not changing architecture/loops), and keep your existing inference post-processing. I also make the confidence computation consistent with the Laplace sigma implied by IQR by using `sigma = (q75-q25)/2` in both training score term and inference Confidence (still clipped at 70), which aligns directly to the evaluation metric and should move the score toward your target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import pydicom
import scipy.ndimage
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from tqdm.auto import tqdm

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    torch.set_num_threads(1)
except Exception:
    pass



## === cell 1
BASE_PATH = "/kaggle/data/osic-pulmonary-fibrosis-progression"
TRAIN_FOLDER = os.path.join(BASE_PATH, "train")
TEST_FOLDER = os.path.join(BASE_PATH, "test")

_DCMREAD_KW = dict(
    force=True,
    stop_before_pixels=False,
    specific_tags=[
        "PixelData",
        "ImagePositionPatient",
        "RescaleIntercept",
        "RescaleSlope",
    ],
)

_CACHE_DIR = os.path.join("/kaggle/working", "ct_cache_uint8_Z100_Y200_X200")
os.makedirs(_CACHE_DIR, exist_ok=True)


def _cache_path(patientid: str) -> str:
    return os.path.join(_CACHE_DIR, f"{patientid}.npy")


def load_scan(path):  # path == .../train/patientId or .../test/patientId
    files = os.listdir(path)
    try:
        slices = [pydicom.dcmread(os.path.join(path, s), **_DCMREAD_KW) for s in files]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except Exception:
        files.sort()
        slices = [pydicom.dcmread(os.path.join(path, s), **_DCMREAD_KW) for s in files]
    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices]).astype(np.int16, copy=False)
    try:
        image[image <= -2000] = 0
        intercepts = np.array(
            [getattr(s, "RescaleIntercept", 0.0) for s in slices], dtype=np.float32
        )
        slopes = np.array(
            [getattr(s, "RescaleSlope", 1.0) for s in slices], dtype=np.float32
        )

        need = slopes != 1.0
        if need.any():
            image_f = image.astype(np.float32, copy=False)
            image_f[need] = image_f[need] * slopes[need, None, None]
            image = image_f.astype(np.int16, copy=False)

        image = (
            image.astype(np.int32, copy=False)
            + intercepts[:, None, None].astype(np.int32)
        ).astype(np.int16, copy=False)
    except Exception:
        print("HU conversion Failed!!")
    return np.array(image, dtype=np.int16, copy=False)


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
    if Z == 100 and Y == 200 and X == 200:
        cp = _cache_path(patientid)
        if os.path.exists(cp):
            try:
                return np.load(cp, allow_pickle=False, mmap_mode=None)
            except Exception:
                pass

    path = dir_name + os.sep + patientid
    slices = load_scan(path)
    try:
        image_array = get_pixels_hu(slices)
        ctimage_resizedAll = resize_along_allaxis(
            image_array, target_dimensionX=X, target_dimensionY=Y, target_dimensionZ=Z
        )
        image = (image_normalize(ctimage_resizedAll) * 255.0).astype("uint8")
        if Z == 100 and Y == 200 and X == 200:
            try:
                np.save(_cache_path(patientid), image, allow_pickle=False)
            except Exception:
                pass
        return image
    except Exception:
        print(f"PatientId:{patientid} couldnt be converted")
        return None




## === cell 2
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
    data = data.rename(columns={"Weeks": "base_Weeks", "FVC": "base_FVC"})

    g = data.groupby("Patient", sort=False)
    base = g.nth(0).reset_index()  # one row per patient (same as index[0] in original)

    sizes = g.size().values
    rep_idx = np.repeat(np.arange(len(base)), sizes)
    out = base.iloc[rep_idx].reset_index(drop=True)

    out["Week"] = data["base_Weeks"].to_numpy()
    out["actual_FVC"] = data["base_FVC"].to_numpy()

    out = out.fillna(0)
    out["Week"] = out["Week"] - out["base_Weeks"]
    out["base_Weeks"] = 0.0

    keep_cols = (
        ["Patient", "base_Weeks", "base_FVC", "Age"]
        + FE1
        + ["Week", "Healthy-FVC", "actual_FVC"]
    )
    for c in keep_cols:
        if c not in out.columns:
            out[c] = 0
    return out[keep_cols]




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
            nn.Linear(41, 64),
            nn.ReLU(),
            nn.Linear(64, 119),
            nn.ReLU(),
        )
        self.data_net2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 503),
            nn.ReLU(),
        )
        self.data_net3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 119),
            nn.ReLU(),
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
C1 = 70.0
C2 = 1000.0


def _enforce_quantile_order(y_pred):
    q25 = y_pred[:, 0]
    q50 = y_pred[:, 1]
    q75 = y_pred[:, 2]
    q25m = torch.minimum(q25, q75)
    q75m = torch.maximum(q25, q75)
    q50m = torch.minimum(torch.maximum(q50, q25m), q75m)
    return torch.stack([q25m, q50m, q75m], dim=1)


def score(y_true, y_pred):
    y_pred = _enforce_quantile_order(y_pred)
    sigma = (y_pred[:, 2] - y_pred[:, 0]) * 0.5
    fvc_pred = y_pred[:, 1]

    sigma_clip = torch.clamp(sigma, min=C1)
    delta = (y_true[:, 0] - fvc_pred).abs()
    delta = torch.clamp(delta, max=C2)

    sq2 = torch.sqrt(torch.tensor(2.0, device=y_pred.device, dtype=y_pred.dtype))
    metric = (delta / sigma_clip) * sq2 + torch.log(sigma_clip * sq2)
    return metric.mean()


def qloss(y_true, y_pred):
    y_pred = _enforce_quantile_order(y_pred)
    qs = [0.25, 0.50, 0.75]
    q = torch.tensor(np.array([qs]), device=y_pred.device, dtype=torch.float32)
    e = y_true - y_pred
    v = torch.max(q * e, (q - 1) * e)
    return v.mean()


def quartile_loss(y_true, y_pred, _lambda=0.65):
    loss = _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)
    return loss




## === cell 5
def make_eval_data(npEval, model, device="cuda", batch_size=8):
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
    for pid in unique_patients:
        img = read_image(TEST_FOLDER, pid, Z=100, Y=200, X=200)
        if img is None:
            img = np.zeros((100, 200, 200), dtype=np.uint8)
        loaded_images[pid] = img

    model = model.to(device)
    model.eval()

    preds = np.empty((len(npEval), 3), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(npEval), batch_size):
            end = min(start + batch_size, len(npEval))
            pids = x_patientids[start:end]
            imgs = [loaded_images[pid] for pid in pids]
            xb_img = torch.from_numpy(
                np.stack(imgs, axis=0).astype(np.float32)
            ).unsqueeze(1)
            xb = torch.from_numpy(x_features[start:end])

            xb_img = xb_img.to(device, non_blocking=True)
            xb = xb.to(device, non_blocking=True)

            out = model(xb_img, xb).detach().cpu().numpy()
            preds[start:end] = out

    q25 = preds[:, 0].astype(np.float32)
    q50 = preds[:, 1].astype(np.float32)
    q75 = preds[:, 2].astype(np.float32)

    q25m = np.minimum(q25, q75)
    q75m = np.maximum(q25, q75)
    q50m = np.minimum(np.maximum(q50, q25m), q75m)

    fvc = q50m
    sigma = 0.5 * (q75m - q25m)

    conf = np.maximum(sigma, 70.0).astype(np.float32)
    fvc = np.clip(fvc, 0.0, 8000.0).astype(np.float32)

    npEval["FVC"] = fvc
    npEval["Confidence"] = conf
    return npEval




## === cell 6
data_train = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
data_test = pd.read_csv(os.path.join(BASE_PATH, "test.csv"))
submission = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"))



## === cell 7
sub_weeks = submission["Patient_Week"].str.split("_", expand=True)
sub_frame = (
    pd.DataFrame(
        {
            "Patient": sub_weeks[0].values,
            "Weeks": sub_weeks[1].astype(int).values,
            "Patient_Week": submission["Patient_Week"].values,
        }
    )
    .sort_values(["Patient", "Weeks"], ascending=True)
    .reset_index(drop=True)
)

base = (
    data_test.sort_values(["Patient", "Weeks"], ascending=True)
    .groupby("Patient", sort=False)
    .nth(0)
    .reset_index()
)
base = base.drop(columns=["Weeks"], errors="ignore")

infer_raw = sub_frame.merge(base, on="Patient", how="left")

proc = csv_preprocess(
    infer_raw[
        ["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]
    ].copy()
)

proc = proc.sort_values(["Patient", "Week"], ascending=True).reset_index(drop=True)
submission = sub_frame.copy()

data_test = proc[
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

assert len(submission) == len(data_test)




## === cell 8
def build_train_frame(train_df):
    tr = csv_preprocess(train_df.copy())
    tr = tr.fillna(0)
    X = tr[
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
    y = tr[["actual_FVC"]].values.astype(np.float32)
    p = tr["Patient"].values
    return tr, X, y, p


train_proc, X_tab, y_fvc, train_patients = build_train_frame(data_train)

unique_train_patients = pd.unique(train_patients)


def _load_train_one(pid):
    return pid, read_image(TRAIN_FOLDER, pid, Z=100, Y=200, X=200)


train_images = {}
if len(unique_train_patients) > 1:
    import multiprocessing as mp

    nproc = min(4, os.cpu_count() or 1, len(unique_train_patients))
    ctx = mp.get_context("fork") if hasattr(mp, "get_context") else mp
    with ctx.Pool(processes=nproc) as pool:
        for pid, img in tqdm(
            pool.imap_unordered(_load_train_one, unique_train_patients, chunksize=1),
            total=len(unique_train_patients),
            desc="Loading train CTs (parallel+cache)",
        ):
            if img is not None:
                train_images[pid] = img
else:
    for pid in tqdm(unique_train_patients, desc="Loading train CTs"):
        img = read_image(TRAIN_FOLDER, pid, Z=100, Y=200, X=200)
        if img is not None:
            train_images[pid] = img

valid_mask = np.array([pid in train_images for pid in train_patients], dtype=bool)
X_tab = X_tab[valid_mask]
y_fvc = y_fvc[valid_mask]
train_patients = train_patients[valid_mask]

for pid in list(train_images.keys()):
    train_images[pid] = np.ascontiguousarray(
        train_images[pid].astype(np.float32), dtype=np.float32
    )



## === cell 9
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Combined_NET().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, X_tab, y_fvc, patients, train_images):
        self.X = torch.from_numpy(X_tab)  # float32
        self.y = torch.from_numpy(y_fvc)  # float32
        self.patients = patients
        self.imgs = train_images

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        pid = self.patients[idx]
        img = self.imgs[pid]  # already float32 contiguous (0..255)
        img_t = torch.from_numpy(img).unsqueeze(0)  # [1, Z, Y, X]
        return img_t, self.X[idx], self.y[idx]


batch_size = 4
n = len(train_patients)
indices = np.arange(n)
rng = np.random.default_rng(42)
rng.shuffle(indices)

ds = TrainDataset(X_tab, y_fvc, train_patients, train_images)
sampler = torch.utils.data.SubsetRandomSampler(indices.tolist())

num_workers = 2 if (os.cpu_count() or 1) > 2 else 0
dl = torch.utils.data.DataLoader(
    ds,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=num_workers,
    pin_memory=(device == "cuda"),
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
)

epochs = 1  # keep identical intent as provided script
model.train()
for epoch in range(epochs):
    running = 0.0
    for xb_img, xb, yb in tqdm(
        dl,
        desc=f"Training epoch {epoch+1}/{epochs}",
        total=(n + batch_size - 1) // batch_size,
    ):
        xb_img = xb_img.to(device, non_blocking=True)
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        pred = model(xb_img, xb)
        loss = quartile_loss(yb, pred)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running += float(loss.detach().cpu().item())



## === cell 10
test = make_eval_data(data_test.copy(), model, device=device, batch_size=8)



## === cell 11
pred_conf = test["Confidence"].astype(float).values
pred_conf = np.maximum(pred_conf, 70.0)
test["Confidence"] = pred_conf



## === cell 12
submission = submission.sort_values(["Patient", "Weeks"], ascending=True).reset_index(
    drop=True
)
submission.loc[:, "FVC"] = test.FVC.values
submission.loc[:, "Confidence"] = test.Confidence.values

submission = submission[["Patient_Week", "FVC", "Confidence"]]
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
