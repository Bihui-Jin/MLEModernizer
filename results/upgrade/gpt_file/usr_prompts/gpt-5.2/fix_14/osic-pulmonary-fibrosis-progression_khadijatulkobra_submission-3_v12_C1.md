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

-6.855675335414687

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -12.941) has done: 'The timeout is dominated by CT preprocessing: `_warm_cache()` forces a full pass over every DICOM slice for all train+test patients, and `read_image()` does two DICOM reads per slice (metadata sort + full pixel read). To fit within 600s without changing the model/training logic, I remove the eager warm-cache step (cache still be populated on-demand) and make `load_scan()` do a single full DICOM read per file while extracting sort keys from the already-loaded datasets. I also speed up directory listing/sorting and avoid extra tensor construction work in evaluation by pre-stacking per-patient images once. These changes preserve identical inputs to the network (same resizing/normalization) while eliminating redundant I/O and computation.'
- What this solution (achieved -11.1441) has done: 'Your current gap to the target is large (about 89% relative), so we need a legitimate score uplift while keeping your core model and CT pipeline intact. The smallest high-impact fix is to correct a sign error in the `score()` term: you currently *add* the Laplace NLL components, but the competition metric is the *negative* of that expression, so training is pushed in the wrong direction. I only change `score()` to match the competition’s formula (keeping the same outputs, loss structure, and training loop), and keep the rest identical so runtime and semantics remain stable. This should move the score upward substantially toward the target without changing architecture or feature extraction.'
- What this solution (achieved -13.48027) has done: 'We keep your model and CT pipeline intact and make only two score-relevant fixes that usually give a meaningful uplift on this competition: (1) train on the target that matches inference (your loss currently compares `actual_FVC` to raw network outputs, but at test time you use `pred[:,1]` as FVC; we change the training loss to use `pred[:,1]` and `pred[:,2]-pred[:,0]` consistently), and (2) output confidence in a safer, metric-friendly way by enforcing `sigma >= 70` and using `softplus`-style positivity (via a minimal clamp) to avoid negative/near-zero sigma that gets clipped and hurts the Laplace term. These are minimal semantic corrections rather than architecture changes, and they should move your score upward toward the target. Submission writing and paths remain unchanged, and runtime stays within limits.'
- What this solution (achieved -12.67606) has done: 'Your current score (-13.48027) is far below the target (-6.8557), so we should improve (increase) the metric with minimal, metric-aligned changes. The main issue is that the model is trained on the *first observed week per patient* (after sorting), which often isn’t the baseline week (0) and can be negative; this misaligns training with the test setting where only the baseline measurement is given. I change `csv_preprocess()` to explicitly pick the per-patient row whose `Weeks` is closest to 0 as the baseline, and then build training examples by pairing that baseline with every observed week’s true FVC—same features, same model, same loss, just correct baseline semantics. This is a small but high-impact fix for this competition and should move the score substantially upward toward your target while keeping runtime and the rest of the pipeline intact.'
- What this solution (achieved -12.34969) has done: 'We make two minimal, score-relevant fixes without changing your model architecture or training loop: (1) ensure the one-hot feature columns are consistent between train and test by explicitly creating the full expected set (`Male/Female` and the 3 smoking categories) even if a category is missing in `test.csv`, and (2) align the test-side “baseline” row to the same definition used in training (week closest to 0 per patient), instead of always using the single provided row as-is. Both changes reduce train/test feature-semantic mismatch and should improve the LaplaceLL metric (move your score up toward the target) while keeping runtime within limits. The submission format/path stays the same and the script still produces `submission.csv`.'
- What this solution (achieved -14.77958) has done: 'Your current score (-12.34969) is well below the target (-6.8557), so we should increase performance with minimal, metric-aligned changes while preserving the same model and training loop. The biggest remaining score drag is that the model is trained only for 1 epoch, which typically underfits badly on this task; increasing epochs is a minimal change that keeps the exact same architecture, loss, and optimization method while improving predictions. To keep runtime within 600s and avoid data-loading overhead changes, we keep the CT pipeline intact and only (a) train for a few more epochs and (b) add a tiny, metric-safe confidence clamp at inference to avoid pathological confidence values (without changing evaluation semantics). These changes should move the score upward toward the target band without rewriting core logic.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import pydicom
import scipy.ndimage

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



## === cell 1
BASE_INPUT = "/kaggle/input/osic-pulmonary-fibrosis-progression"
TRAIN_FOLDER = os.path.join(BASE_INPUT, "train")
TEST_FOLDER = os.path.join(BASE_INPUT, "test")


def load_scan(path):
    try:
        entries = list(os.scandir(path))
    except FileNotFoundError:
        return []
    files = [e.path for e in entries if e.is_file()]
    if not files:
        return []

    meta = []
    for fpath in files:
        z = 0.0
        try:
            ds_hdr = pydicom.dcmread(
                fpath,
                stop_before_pixels=True,
                force=True,
                specific_tags=["ImagePositionPatient", "InstanceNumber"],
            )
            if (
                hasattr(ds_hdr, "ImagePositionPatient")
                and ds_hdr.ImagePositionPatient is not None
            ):
                z = float(ds_hdr.ImagePositionPatient[2])
            elif (
                hasattr(ds_hdr, "InstanceNumber") and ds_hdr.InstanceNumber is not None
            ):
                z = float(ds_hdr.InstanceNumber)
        except Exception:
            z = 0.0
        meta.append((z, fpath))

    if not meta:
        return []

    meta.sort(key=lambda t: float(t[0]))
    slices = []
    for _, fpath in meta:
        try:
            ds = pydicom.dcmread(fpath, force=True)
            slices.append(ds)
        except Exception:
            continue
    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices]).astype(np.int16, copy=False)
    try:
        image[image <= -2000] = 0
        intercept = np.array(
            [float(getattr(s, "RescaleIntercept", 0.0)) for s in slices],
            dtype=np.float32,
        )
        slope = np.array(
            [float(getattr(s, "RescaleSlope", 1.0)) for s in slices],
            dtype=np.float32,
        )

        slope_b = slope[:, None, None]
        intercept_b = intercept[:, None, None]

        img_f = image.astype(np.float32, copy=False) * slope_b + intercept_b
        image = img_f.astype(np.int16)
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


_CACHE_DIR = "/kaggle/working/osic_ct_cache"
os.makedirs(_CACHE_DIR, exist_ok=True)


def _cache_path(patientid, Z, Y, X):
    return os.path.join(_CACHE_DIR, f"{patientid}_Z{Z}_Y{Y}_X{X}.npy")


def read_image(dir_name, patientid, Z=100, Y=200, X=200):
    cpath = _cache_path(patientid, Z, Y, X)
    if os.path.exists(cpath):
        try:
            return np.load(cpath, mmap_mode="r")
        except Exception:
            pass  # fall back to recompute if cache corrupted

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

        tmp_path = cpath + ".tmp"
        with open(tmp_path, "wb") as f:
            np.save(f, np.asarray(image))
        os.replace(tmp_path, cpath)

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

    data = data[["Patient", "Weeks", "FVC", "Age"] + FE].copy()
    data = data.reset_index(drop=True)

    FE1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]

    data["_abs_week"] = data["Weeks"].abs()
    base_idx = (
        data.sort_values(
            ["Patient", "_abs_week", "Weeks"], ascending=[True, True, True]
        )
        .groupby("Patient", sort=False)
        .head(1)
        .index
    )
    base = data.loc[base_idx].copy()
    base = base.rename(columns={"Weeks": "base_Weeks", "FVC": "base_FVC"})
    base = base[["Patient", "base_Weeks", "base_FVC", "Age"] + FE1 + ["Healthy-FVC"]]

    obs = data[["Patient", "Weeks", "FVC"]].copy()
    obs = obs.rename(columns={"Weeks": "Week", "FVC": "actual_FVC"})

    out = obs.merge(base, on="Patient", how="left", sort=False)
    out = out[
        ["Patient", "base_Weeks", "base_FVC", "Age"]
        + FE1
        + ["Week", "Healthy-FVC", "actual_FVC"]
    ].fillna(0)

    out = out.sort_values(["Patient", "Week"], ascending=True).reset_index(drop=True)
    return out




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
            nn.Linear(780, 256),
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
C1 = torch.tensor(70.0, dtype=torch.float32)
C2 = torch.tensor(1000.0, dtype=torch.float32)
SQRT2_CPU = float(np.sqrt(2.0))


def score(y_true, y_pred):
    q25 = y_pred[:, 0]
    q50 = y_pred[:, 1]
    q75 = y_pred[:, 2]

    sigma = (q75 - q25) / 2.0
    sigma_clip = torch.clamp(sigma, min=float(C1))

    delta = (y_true[:, 0] - q50).abs()
    delta = torch.clamp(delta, max=float(C2))

    sq2 = torch.tensor(SQRT2_CPU, device=y_pred.device, dtype=y_pred.dtype)
    metric = -((delta / sigma_clip) * sq2 + (sigma_clip * sq2).log())
    return metric.mean()


def qloss(y_true, y_pred):
    qs = [0.25, 0.50, 0.75]
    q = torch.tensor(np.array([qs]), device=y_pred.device, dtype=torch.float32)
    e = y_true - y_pred
    v = torch.max(q * e, (q - 1) * e)
    return v.mean()


def quartile_loss(y_true_fvc, y_pred_raw, _lambda=0.65):
    loss = _lambda * qloss(y_true_fvc, y_pred_raw) + (1 - _lambda) * score(
        y_true_fvc, y_pred_raw
    )
    return loss




## === cell 5
FEATURE_COLS = [
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


class OSICDataset(Dataset):
    def __init__(self, df, image_dir, cache_images=True):
        self.df = df.reset_index(drop=True).copy()
        self.image_dir = image_dir
        self.cache_images = cache_images
        self._img_cache = {} if cache_images else None

        self._x_feat_np = self.df[FEATURE_COLS].to_numpy(dtype=np.float32, copy=True)
        self._y_np = self.df["actual_FVC"].to_numpy(dtype=np.float32, copy=True)
        self._pid = self.df["Patient"].to_numpy(copy=False)

    def __len__(self):
        return len(self.df)

    def _get_image_tensor(self, patient_id):
        if self.cache_images and patient_id in self._img_cache:
            return self._img_cache[patient_id]
        img = read_image(self.image_dir, patient_id, Z=100, Y=200, X=200)
        if img is None:
            img = np.zeros((100, 200, 200), dtype=np.uint8)

        x_image = torch.from_numpy(np.asarray(img)).to(torch.float32).unsqueeze(0)
        if self.cache_images:
            self._img_cache[patient_id] = x_image
        return x_image

    def __getitem__(self, idx):
        pid = self._pid[idx]
        x_image = self._get_image_tensor(pid)
        x_feat = torch.from_numpy(self._x_feat_np[idx])
        y = torch.tensor([self._y_np[idx]], dtype=torch.float32)
        return x_image, x_feat, y




## === cell 6
def make_eval_data(npEval, model, device="cuda", batch_size=8):
    x_features = torch.tensor(npEval[FEATURE_COLS].values, dtype=torch.float32)
    patient_ids = npEval["Patient"].values
    unique_patients = pd.unique(patient_ids)

    dir_name_of_patientid = TEST_FOLDER  # keep path as intended

    imgs = []
    for pid in unique_patients:
        img = read_image(dir_name_of_patientid, pid, Z=100, Y=200, X=200)
        if img is None:
            img = np.zeros((100, 200, 200), dtype=np.uint8)
        imgs.append(torch.from_numpy(np.asarray(img)).to(torch.float32).unsqueeze(0))
    img_bank = torch.stack(imgs, dim=0)

    use_cuda = torch.cuda.is_available() and device == "cuda"
    if use_cuda:
        model.to("cuda")
        img_bank = img_bank.cuda(non_blocking=True)
        x_features = x_features.cuda(non_blocking=True)

    pid_to_idx = {pid: i for i, pid in enumerate(unique_patients)}
    idxs = np.fromiter(
        (pid_to_idx[pid] for pid in patient_ids), dtype=np.int64, count=len(patient_ids)
    )

    model.eval()
    preds = np.empty((len(npEval), 3), dtype=np.float32)

    with torch.no_grad():
        n = len(npEval)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = torch.from_numpy(idxs[start:end])
            if use_cuda:
                batch_idx = batch_idx.cuda(non_blocking=True)
            x_image = img_bank.index_select(0, batch_idx)
            x_feat = x_features[start:end]
            out = model(x_image, x_feat).detach().cpu().numpy()
            preds[start:end] = out

    npEval["FVC"] = preds[:, 1]
    conf = (preds[:, 2] - preds[:, 0]) / 2.0
    conf = np.where(np.isfinite(conf), conf, 70.0)
    conf = np.maximum(conf, 70.0)
    npEval["Confidence"] = conf
    return npEval




## === cell 7
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
device



## === cell 8
data_train = pd.read_csv(os.path.join(BASE_INPUT, "train.csv"))
data_test = pd.read_csv(os.path.join(BASE_INPUT, "test.csv"))
sample_submission = pd.read_csv(os.path.join(BASE_INPUT, "sample_submission.csv"))

submission = sample_submission.copy()
pw = submission["Patient_Week"].str.split("_", n=1, expand=True)
submission["Patient"] = pw[0]
submission["Weeks"] = pw[1].astype(int)
submission = submission.sort_values(
    by=["Patient", "Weeks"], ascending=True
).reset_index(drop=True)

data_test_b = data_test.copy()
data_test_b["_abs_week"] = data_test_b["Weeks"].abs()
base_idx_t = (
    data_test_b.sort_values(
        ["Patient", "_abs_week", "Weeks"], ascending=[True, True, True]
    )
    .groupby("Patient", sort=False)
    .head(1)
    .index
)
data_test_b = (
    data_test_b.loc[base_idx_t].drop(columns=["_abs_week"]).reset_index(drop=True)
)

merge = (
    pd.merge(data_test_b, submission, on=["Patient"], how="left")
    .sort_values(["Patient", "Weeks_y"])
    .reset_index(drop=True)
)
merge = merge.drop(["FVC_y"], axis=1)
merge = merge.rename(
    columns={"FVC_x": "base_FVC", "Weeks_y": "Week", "Weeks_x": "base_Weeks"}
)

data_test_expanded = merge.loc[
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

submission_out = merge.loc[:, ["Patient_Week"]].copy()
submission_out["FVC"] = 0.0
submission_out["Confidence"] = 100.0



## === cell 9
data = data_test_expanded.copy()
data["Healthy-FVC"] = round((data["base_FVC"] * 100) / data["Percent"])

FE1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]
data["Male"] = (data["Sex"] == "Male").astype(int)
data["Female"] = (data["Sex"] == "Female").astype(int)
data["Ex-smoker"] = (data["SmokingStatus"] == "Ex-smoker").astype(int)
data["Never smoked"] = (data["SmokingStatus"] == "Never smoked").astype(int)
data["Currently smokes"] = (data["SmokingStatus"] == "Currently smokes").astype(int)

npData = pd.DataFrame(
    columns=["Patient", "base_Weeks", "base_FVC", "Age", "Healthy-FVC"] + FE1 + ["Week"]
)
npData = pd.concat([npData, data], axis=0, sort=True, ignore_index=True)
npData = npData.fillna(0)

data_test_model = npData[
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

del data, npData, data_test_expanded




## === cell 10
def _warm_cache(patient_ids, image_dir, Z=100, Y=200, X=200, workers=None):
    patient_ids = list(patient_ids)
    if workers is None:
        cpu = os.cpu_count() or 2
        workers = min(8, cpu)  # cap to avoid oversubscription on Kaggle
    todo = [pid for pid in patient_ids if not os.path.exists(_cache_path(pid, Z, Y, X))]
    if not todo:
        return

    from concurrent.futures import ProcessPoolExecutor

    def _one(pid):
        _ = read_image(image_dir, pid, Z=Z, Y=Y, X=X)
        return pid

    chunksize = max(1, len(todo) // (workers * 4) if workers > 0 else 1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_one, todo, chunksize=chunksize))




## === cell 11
train_np = csv_preprocess(data_train)

train_pids = pd.unique(train_np["Patient"].values)
test_pids = pd.unique(data_test_model["Patient"].values)

_warm_cache(train_pids, TRAIN_FOLDER, Z=100, Y=200, X=200, workers=None)
_warm_cache(test_pids, TEST_FOLDER, Z=100, Y=200, X=200, workers=None)

if torch.cuda.is_available():
    num_workers = min(4, (os.cpu_count() or 2))
else:
    num_workers = min(4, (os.cpu_count() or 2))

train_ds = OSICDataset(train_np, image_dir=TRAIN_FOLDER, cache_images=False)
train_loader = DataLoader(
    train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
)

model = Combined_NET().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()

epochs = 4
for ep in range(epochs):
    running = 0.0
    for x_img, x_feat, y in train_loader:
        x_img = x_img.to(device, non_blocking=True)
        x_feat = x_feat.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x_img, x_feat)

        loss = quartile_loss(y, pred)
        loss.backward()
        optimizer.step()

        running += float(loss.detach().cpu().item())



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
_RemoteTraceback                          Traceback (most recent call last)
_RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/lib/python3.11/multiprocessing/queues.py", line 244, in _feed
    obj = _ForkingPickler.dumps(obj)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/multiprocessing/reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
AttributeError: Can't pickle local object '_warm_cache.<locals>._one'
"""

The above exception was the direct cause of the following exception:

AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/3147333421.py in <cell line: 0>()
      4 test_pids = pd.unique(data_test_model["Patient"].values)
      5 
----> 6 _warm_cache(train_pids, TRAIN_FOLDER, Z=100, Y=200, X=200, workers=None)
      7 _warm_cache(test_pids, TEST_FOLDER, Z=100, Y=200, X=200, workers=None)
      8 

/tmp/ipykernel_55/589608620.py in _warm_cache(patient_ids, image_dir, Z, Y, X, workers)
     23     chunksize = max(1, len(todo) // (workers * 4) if workers > 0 else 1)
     24     with ProcessPoolExecutor(max_workers=workers) as ex:
---> 25         list(ex.map(_one, todo, chunksize=chunksize))
     26 
     27 

/usr/lib/python3.11/concurrent/futures/process.py in _chain_from_iterable_of_lists(iterable)
    618     careful not to keep references to yielded objects.
    619     """
--> 620     for element in iterable:
    621         element.reverse()
    622         while element:

/usr/lib/python3.11/concurrent/futures/_base.py in result_iterator()
    617                     # Careful not to keep a reference to the popped future
    618                     if timeout is None:
--> 619                         yield _result_or_cancel(fs.pop())
    620                     else:
    621                         yield _result_or_cancel(fs.pop(), end_time - time.monotonic())

/usr/lib/python3.11/concurrent/futures/_base.py in _result_or_cancel(***failed resolving arguments***)
    315     try:
    316         try:
--> 317             return fut.result(timeout)
    318         finally:
    319             fut.cancel()

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    454                     raise CancelledError()
    455                 elif self._state == FINISHED:
--> 456                     return self.__get_result()
    457                 else:
    458                     raise TimeoutError()

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

/usr/lib/python3.11/multiprocessing/queues.py in _feed(buffer, notempty, send_bytes, writelock, reader_close, writer_close, ignore_epipe, onerror, queue_sem)
    242 
    243                         # serialize the data before acquiring the lock
--> 244                         obj = _ForkingPickler.dumps(obj)
    245                         if wacquire is None:
    246                             send_bytes(obj)

/usr/lib/python3.11/multiprocessing/reduction.py in dumps(cls, obj, protocol)
     49     def dumps(cls, obj, protocol=None):
     50         buf = io.BytesIO()
---> 51         cls(buf, protocol).dump(obj)
     52         return buf.getbuffer()
     53 

AttributeError: Can't pickle local object '_warm_cache.<locals>._one'

## === cell 12
test_pred_df = make_eval_data(
    data_test_model.copy(), model, device=device, batch_size=8
)

submission_out["FVC"] = test_pred_df["FVC"].values.astype(float)
submission_out["Confidence"] = test_pred_df["Confidence"].values.astype(float)

submission_out = submission_out[["Patient_Week", "FVC", "Confidence"]]
submission_out.to_csv("submission.csv", index=False)

submission_out.head()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1247099540.py in <cell line: 0>()
      1 test_pred_df = make_eval_data(
----> 2     data_test_model.copy(), model, device=device, batch_size=8
      3 )
      4 
      5 submission_out["FVC"] = test_pred_df["FVC"].values.astype(float)

NameError: name 'model' is not defined

## === cell 13
submission_out.shape
