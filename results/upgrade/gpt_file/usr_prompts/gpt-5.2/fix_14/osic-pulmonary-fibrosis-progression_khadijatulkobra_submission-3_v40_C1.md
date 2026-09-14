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

-6.91367988174805

# 6. Current score

-18.88211

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -24.58978) has done: 'The timeout is dominated by repeated DICOM decoding + resizing inside the Dataset, especially if multiple DataLoader workers each rebuild their own cache; this causes redundant work and heavy I/O/CPU. I make CT preprocessing provably equivalent but faster by (1) caching per-patient CT volumes to disk in a single-process precompute step (so it happens once), (2) forcing the DataLoader to use `num_workers=0` to avoid per-worker cache duplication and extra disk contention, and (3) vectorizing HU conversion (remove per-slice Python loop) while keeping identical math. The model, inputs, and inference semantics remain the same; only data loading/feature prep is made more efficient.'
- What this solution (achieved -24.58978) has done: 'The timeout is dominated by CT DICOM decoding + 3D resampling; your script does that twice (an up-front precompute pass and then again inside the Dataset, plus slow per-slice `pydicom.dcmread` and `scipy.ndimage.zoom`). To finish under 600s without changing the model or prediction semantics, the replacement removes redundant precompute work, accelerates DICOM loading using `pydicom.read_file(..., stop_before_pixels=True)` for sorting and only decodes pixel data once, and caches each patient volume in RAM for the whole run (test set is tiny) while still keeping the existing on-disk `.npy` cache path. It also uses multi-worker DataLoader to overlap CPU image loading with GPU inference/training (same data, same order), and makes the metric function avoid repeated tensor allocations. All changes are provably equivalent: same CT preprocessing outputs, same model forward, same loss/eval logic, just less duplicated work and better parallelism.'
- What this solution (achieved -24.58978) has done: 'We fix the crash by making deterministic mode compatible with CUDA/cuBLAS: set `CUBLAS_WORKSPACE_CONFIG` early and fall back to non-deterministic algorithms only if needed, so training can run. We also correct the loss/metric sign (your current `score()` returns the negative of the competition metric, so training optimizes the wrong direction), which should move the score substantially toward the target without changing the model or data semantics. Finally, we make the label tensor shape consistent with what `score()` expects and ensure inference writes a properly aligned `submission.csv` with the required columns.'
- What this solution (achieved -18.82215) has done: 'I fix the crash in training by switching PyTorch deterministic mode to `warn_only=True`, which preserves the same training logic while avoiding the non-deterministic `max_pool3d_with_indices_backward_cuda` hard error. I also make the training/inference path robust by safely handling missing pretrained weights (continue to train) and ensuring `DataLoader` arguments are compatible with the Kaggle runtime (e.g., only set `prefetch_factor` when `num_workers>0`). Finally, I keep submission alignment identical to `sample_submission.csv` and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved -18.82215) has done: 'Your current score is far below the target (higher is better), so we should make a small, metric-aligned improvement without changing the model or training loop. The biggest gain with minimal risk is to enforce a strictly positive, reasonable `sigma` at inference (your model’s last layer uses ReLU so sigma can be near-zero, which the metric heavily penalizes via `-log(sigma)`), while keeping the competition’s clip-at-70 behavior. I also apply the same positivity enforcement inside the loss to keep training/inference semantics consistent (still the same Laplace metric objective), and keep the baseline-week overwrite as you already do. These are tiny post-processing/metric-stability tweaks that typically move OSIC scores substantially upward toward your target band.'
- What this solution (achieved -18.82215) has done: 'Your current score (-18.82) is far below the target (-6.91), so we should improve (increase) the metric with minimal, metric-aligned changes. The biggest likely issue is that your model outputs are being interpreted as `[sigma, fvc]` but the pretrained weights (and many OSIC baselines) use `[fvc, sigma]`; swapping these at inference (and in the metric used for training) can dramatically improve score without changing architecture or training loop. I implement a tiny “output mapping” helper that chooses the most plausible ordering via a fast sanity check on a small batch, then apply it consistently in both `laplace_metric` and `make_eval_data`. I also ensure confidence is strictly positive and then clipped to 70 only at submission time (same evaluation semantics), keeping everything else unchanged.'
- What this solution (achieved -19.13528) has done: 'Your current score (-18.82) is far below the target (-6.91), so we should increase the metric with the smallest, metric-aligned changes. The biggest likely issue is that your “sigma” output is not constrained to be a sensible uncertainty scale during both training-loss computation and submission (you only clip at the very end), which can overly penalize via the `-log(sigma)` term. I (1) enforce the competition’s `sigma_clipped=max(sigma,70)` consistently inside `laplace_metric` (so the objective matches the leaderboard metric), and (2) add a minimal per-patient linear-trend calibration on the predicted FVC using train-derived average slope vs week (keeps core model, just adjusts predictions toward typical progression). These changes preserve architecture/training loop and should move the score substantially upward toward the target band.'
- What this solution (achieved -18.88211) has done: 'Your current score (-19.135) is far below the target (-6.914), so we should increase it with minimal, metric-aligned changes. The biggest low-risk gain is to make the model’s predicted uncertainty behave like the leaderboard metric: enforce `sigma_clipped=max(sigma,70)` not only in the loss but also in the values written to `Confidence`, and prevent pathological huge confidences by adding a conservative upper clamp (keeps semantics, improves stability of the `-log(sigma)` term). Next, your per-patient linear trend adjustment currently adds a global slope on top of the model’s own week-dependent output, which often over-corrects; we keep the same calibration idea but damp it with a small factor so it nudges rather than dominates. These changes don’t alter the architecture, data extraction, or training loop, and they still write a valid `submission.csv` in the required format.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
import numpy as np
import pandas as pd

import pydicom
import scipy.ndimage

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE_INPUT = "../input/osic-pulmonary-fibrosis-progression"
TRAIN_DICOM_FOLDER = os.path.join(BASE_INPUT, "train")
TEST_DICOM_FOLDER = os.path.join(BASE_INPUT, "test")

TRAIN_CSV_PATH = os.path.join(BASE_INPUT, "train.csv")
TEST_CSV_PATH = os.path.join(BASE_INPUT, "test.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    except Exception:
        pass


seed_everything(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
device




## === cell 1
def load_scan(path):
    entries = [e for e in os.scandir(path) if e.is_file()]
    entries.sort(key=lambda e: e.name)

    z_positions = []
    for e in entries:
        try:
            ds_hdr = pydicom.dcmread(e.path, force=True, stop_before_pixels=True)
            z_positions.append(float(ds_hdr.ImagePositionPatient[2]))
        except Exception:
            z_positions.append(np.nan)

    if np.isnan(z_positions).any():
        order = range(len(entries))
    else:
        order = np.argsort(z_positions)

    dsets = []
    for i in order:
        ds = pydicom.dcmread(entries[i].path, force=True)
        dsets.append(ds)
    return dsets


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices]).astype(np.int16)
    try:
        image[image <= -2000] = 0

        intercept = np.array(
            [getattr(s, "RescaleIntercept", 0.0) for s in slices], dtype=np.float32
        )
        slope = np.array(
            [getattr(s, "RescaleSlope", 1.0) for s in slices], dtype=np.float32
        )

        img_f = image.astype(np.float32, copy=False)
        img_f = img_f * slope[:, None, None] + intercept[:, None, None]
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
    path = dir_name + os.sep + patientid
    slices = load_scan(path)
    try:
        image_array = get_pixels_hu(slices)
        ctimage_resizedAll = resize_along_allaxis(
            image_array, target_dimensionX=X, target_dimensionY=Y, target_dimensionZ=Z
        )
        image = (image_normalize(ctimage_resizedAll) * 255.0).astype("uint8")
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

    baseline = data.groupby("Patient", sort=False).nth(0).reset_index()

    out = data[
        ["Patient", "base_Weeks", "base_FVC", "Age"] + FE1 + ["Healthy-FVC"]
    ].merge(
        baseline[["Patient", "base_Weeks", "base_FVC", "Age"] + FE1 + ["Healthy-FVC"]],
        on="Patient",
        how="left",
        suffixes=("", "_base"),
        sort=False,
    )

    npData = pd.DataFrame(
        {
            "Patient": out["Patient"].values,
            "base_Weeks": out["base_Weeks_base"].values,
            "base_FVC": out["base_FVC_base"].values,
            "Age": out["Age_base"].values,
            "Male": out.get("Male_base", 0).values if "Male_base" in out else 0,
            "Female": out.get("Female_base", 0).values if "Female_base" in out else 0,
            "Ex-smoker": (
                out.get("Ex-smoker_base", 0).values if "Ex-smoker_base" in out else 0
            ),
            "Never smoked": (
                out.get("Never smoked_base", 0).values
                if "Never smoked_base" in out
                else 0
            ),
            "Currently smokes": (
                out.get("Currently smokes_base", 0).values
                if "Currently smokes_base" in out
                else 0
            ),
            "Week": data["base_Weeks"].values,
            "Healthy-FVC": out["Healthy-FVC_base"].values,
            "actual_FVC": data["base_FVC"].values,
        }
    )

    npData = npData.fillna(0).reset_index(drop=True)
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
            nn.Linear(42, 64),
            nn.ReLU(),
            nn.Linear(64, 118),
            nn.ReLU(),
        )
        self.data_net2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 502),
            nn.ReLU(),
        )
        self.data_net3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 118),
            nn.ReLU(),
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




## === cell 4
C1 = 70.0
C2 = 1000.0

OUTPUT_ORDER = None  # will be set to "sigma_fvc" or "fvc_sigma"


def _split_pred(y_pred, order: str):
    if order == "sigma_fvc":
        sigma = y_pred[:, 0]
        fvc_pred = y_pred[:, 1]
    elif order == "fvc_sigma":
        fvc_pred = y_pred[:, 0]
        sigma = y_pred[:, 1]
    else:
        raise ValueError("Unknown order")
    return sigma, fvc_pred


def laplace_metric(y_true_fvc, y_pred, order: str = None):
    if y_true_fvc.ndim == 2:
        y_true_fvc = y_true_fvc[:, 0]

    if order is None:
        order = OUTPUT_ORDER if OUTPUT_ORDER is not None else "sigma_fvc"

    sigma, fvc_pred = _split_pred(y_pred, order)

    sigma = torch.clamp(sigma, min=1.0)
    sigma_clip = torch.clamp(sigma, min=C1)

    delta = (y_true_fvc - fvc_pred).abs()
    delta = torch.clamp(delta, max=C2)

    sq2 = 2.0**0.5
    metric = -sq2 * delta / sigma_clip - torch.log(sq2 * sigma_clip)
    return metric.mean()


def quartile_loss(y_true_fvc, y_pred):
    return -laplace_metric(
        y_true_fvc, y_pred, order=OUTPUT_ORDER if OUTPUT_ORDER else "sigma_fvc"
    )




## === cell 5
data_train = pd.read_csv(TRAIN_CSV_PATH)
data_test = pd.read_csv(TEST_CSV_PATH)
submission = pd.read_csv(SAMPLE_SUB_PATH)

pw = submission["Patient_Week"].str.split("_", n=1, expand=True)
submission["Patient"] = pw[0]
submission["Weeks"] = pw[1].astype(int)
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
submission = merge.loc[:, ["Patient_Week", "base_FVC", "Confidence"]].rename(
    columns={"base_FVC": "FVC"}
)

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
npData = pd.concat([npData, data], ignore_index=True, sort=True).fillna(0)

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
del npData, data, merge

data_train.shape, data_test.shape, submission.shape



## === cell 6
train_np = csv_preprocess(data_train)
train_np = train_np[
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
        "actual_FVC",
    ]
].copy()

train_np.head()




## === cell 7
class OSICDataset(Dataset):
    def __init__(self, df, dicom_root, image_cache=None, disk_cache_dir=None):
        self.df = df.reset_index(drop=True)
        self.dicom_root = dicom_root
        self.cache = {} if image_cache is None else image_cache

        if disk_cache_dir is None:
            disk_cache_dir = os.path.join(
                "/kaggle/working",
                "osic_ct_npy_cache",
                "train" if dicom_root.endswith("train") else "test",
            )
        self.disk_cache_dir = disk_cache_dir
        try:
            os.makedirs(self.disk_cache_dir, exist_ok=True)
            self.disk_cache_ok = True
        except Exception:
            self.disk_cache_ok = False

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

        self._X_feat = self.df[self.feature_cols].to_numpy(dtype=np.float32, copy=True)
        self._patients = self.df["Patient"].to_numpy()

        if "actual_FVC" in self.df.columns:
            self._y = self.df["actual_FVC"].to_numpy(dtype=np.float32, copy=True)
        else:
            self._y = np.zeros((len(self.df),), dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def _disk_path(self, patient):
        return os.path.join(self.disk_cache_dir, f"{patient}_Z100_Y200_X200_uint8.npy")

    def _atomic_save_npy(self, path, arr):
        tmp = path + f".tmp_{os.getpid()}"
        with open(tmp, "wb") as f:
            np.save(f, arr, allow_pickle=False)
        os.replace(tmp, path)

    def _get_image(self, patient):
        if patient in self.cache:
            return self.cache[patient]

        if self.disk_cache_ok:
            p = self._disk_path(patient)
            if os.path.exists(p):
                img = np.load(p, allow_pickle=False)
                self.cache[patient] = img
                return img

        img = read_image(self.dicom_root, patient, Z=100, Y=200, X=200)
        if img is None:
            img = np.zeros((100, 200, 200), dtype=np.uint8)

        if self.disk_cache_ok:
            p = self._disk_path(patient)
            try:
                self._atomic_save_npy(p, img)
            except Exception:
                pass

        self.cache[patient] = img
        return img

    def __getitem__(self, idx):
        patient = self._patients[idx]
        img = self._get_image(patient)

        x_img = torch.from_numpy(img).unsqueeze(0).float()  # [1,Z,Y,X]
        x_feat = torch.from_numpy(self._X_feat[idx])
        y = torch.tensor([self._y[idx]], dtype=torch.float32)
        return x_img, x_feat, y




## === cell 8
TRAIN_CT_CACHE_DIR = os.path.join("/kaggle/working", "osic_ct_npy_cache", "train")
TEST_CT_CACHE_DIR = os.path.join("/kaggle/working", "osic_ct_npy_cache", "test")
os.makedirs(TRAIN_CT_CACHE_DIR, exist_ok=True)
os.makedirs(TEST_CT_CACHE_DIR, exist_ok=True)

_test_img_cache = {}
for p in data_test["Patient"].unique():
    ds_tmp = OSICDataset(
        pd.DataFrame(
            {
                "Patient": [p],
                "base_Weeks": [0],
                "base_FVC": [0],
                "Age": [0],
                "Male": [0],
                "Female": [0],
                "Ex-smoker": [0],
                "Never smoked": [0],
                "Currently smokes": [0],
                "Week": [0],
                "Healthy-FVC": [0],
            }
        ),
        TEST_DICOM_FOLDER,
        image_cache=_test_img_cache,
        disk_cache_dir=TEST_CT_CACHE_DIR,
    )
    _ = ds_tmp._get_image(p)
del ds_tmp
len(_test_img_cache)




## === cell 9
def make_eval_data(npEval, model, device="cuda", batch_size=8):
    eval_ds = OSICDataset(
        npEval,
        TEST_DICOM_FOLDER,
        image_cache=_test_img_cache,
        disk_cache_dir=TEST_CT_CACHE_DIR,
    )

    cpu = os.cpu_count() or 2
    num_workers = min(4, max(1, cpu // 2))
    pin = torch.cuda.is_available() and device == "cuda"

    dl_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = 2

    eval_loader = DataLoader(eval_ds, **dl_kwargs)

    model = model.to(
        device if (torch.cuda.is_available() and device == "cuda") else "cpu"
    )
    model.eval()

    preds = []
    with torch.no_grad():
        for x_img, x_feat, _ in eval_loader:
            if torch.cuda.is_available() and device == "cuda":
                x_img = x_img.cuda(non_blocking=True)
                x_feat = x_feat.cuda(non_blocking=True)
            y_pred = model(x_img, x_feat)
            preds.append(y_pred.detach().cpu().numpy())

    predictions = np.concatenate(preds, axis=0)

    order = OUTPUT_ORDER if OUTPUT_ORDER is not None else "sigma_fvc"
    if order == "sigma_fvc":
        conf = predictions[:, 0]
        fvc = predictions[:, 1]
    else:  # "fvc_sigma"
        fvc = predictions[:, 0]
        conf = predictions[:, 1]

    npEval = npEval.copy()
    npEval["FVC"] = fvc
    npEval["Confidence"] = conf
    return npEval




## === cell 10
WEIGHTS_PATH = (
    "../input/ww-15-674/Epoch15_Score6.744856326943202_Acc0.9344794751792554.pth"
)

model = Combined_NET()

if os.path.exists(WEIGHTS_PATH):
    state = torch.load(WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state)
    trained = False
else:
    trained = True
    model = model.to(device)

    train_cache = {}  # on-demand cache (disk+RAM)
    train_ds = OSICDataset(
        train_np,
        TRAIN_DICOM_FOLDER,
        image_cache=train_cache,
        disk_cache_dir=TRAIN_CT_CACHE_DIR,
    )

    cpu = os.cpu_count() or 2
    num_workers = min(4, max(1, cpu // 2))
    pin = torch.cuda.is_available()

    dl_kwargs = dict(
        batch_size=2,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, **dl_kwargs)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    EPOCHS = 1  # unchanged
    for epoch in range(EPOCHS):
        running = 0.0
        for x_img, x_feat, y in train_loader:
            x_img = x_img.to(device, non_blocking=True)
            x_feat = x_feat.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            y_pred = model(x_img, x_feat)

            loss = quartile_loss(y, y_pred)

            try:
                loss.backward()
            except RuntimeError as e:
                msg = str(e)
                if "deterministic" in msg.lower():
                    try:
                        torch.use_deterministic_algorithms(False)
                        loss.backward()
                    except Exception:
                        raise
                else:
                    raise

            optimizer.step()
            running += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} - loss: {running/len(train_loader):.5f}")

model.to(device)
trained




## === cell 11
def choose_output_order(model, device):
    ds = OSICDataset(
        train_np.head(8),
        TRAIN_DICOM_FOLDER,
        image_cache={},
        disk_cache_dir=TRAIN_CT_CACHE_DIR,
    )
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    model.eval()
    with torch.no_grad():
        x_img, x_feat, _ = next(iter(dl))
        x_img = x_img.to(device)
        x_feat = x_feat.to(device)
        y_pred = model(x_img, x_feat).detach().cpu().numpy()

    a = y_pred[:, 0]
    b = y_pred[:, 1]

    def fvc_like(x):
        return float(np.mean((x > 500) & (x < 7000)))

    if fvc_like(a) >= fvc_like(b):
        return "fvc_sigma"
    return "sigma_fvc"


def compute_global_slope(train_df):
    df = train_df[["Patient", "Weeks", "FVC"]].dropna().copy()
    slopes = []
    for _, g in df.groupby("Patient"):
        if len(g) < 2:
            continue
        x = g["Weeks"].to_numpy(dtype=np.float32)
        y = g["FVC"].to_numpy(dtype=np.float32)
        x0 = x - x.mean()
        denom = float(np.sum(x0 * x0))
        if denom <= 1e-6:
            continue
        slope = float(np.sum(x0 * (y - y.mean())) / denom)  # ml per week
        if np.isfinite(slope):
            slopes.append(slope)
    if len(slopes) == 0:
        return -10.0
    return float(np.median(slopes))


GLOBAL_SLOPE = compute_global_slope(data_train)
OUTPUT_ORDER = choose_output_order(model.to(device), device)
print("Selected OUTPUT_ORDER =", OUTPUT_ORDER)
print("GLOBAL_SLOPE (ml/week) =", GLOBAL_SLOPE)

test = make_eval_data(
    data_test.copy(),
    model,
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch_size=8,
)

test = test.sort_values(["Patient", "Week"]).reset_index(drop=True)
base_pred = (
    test.loc[test["Week"].values == test["base_Weeks"].values, ["Patient", "FVC"]]
    .rename(columns={"FVC": "pred_base"})
    .reset_index(drop=True)
)
base_true = (
    test.loc[test["Week"].values == test["base_Weeks"].values, ["Patient", "base_FVC"]]
    .rename(columns={"base_FVC": "base_FVC_val"})
    .reset_index(drop=True)
)
base_map = pd.merge(base_true, base_pred, on="Patient", how="left")
test = test.merge(base_map, on="Patient", how="left", sort=False)

week_delta = (test["Week"].values - test["base_Weeks"].values).astype(np.float32)
pred_centered = (test["FVC"].values - test["pred_base"].values).astype(np.float32)

SLOPE_DAMP = 0.35
test["FVC"] = (
    test["base_FVC_val"].values.astype(np.float32)
    + pred_centered
    + (SLOPE_DAMP * GLOBAL_SLOPE) * week_delta
)

mask = test["Week"].values == test["base_Weeks"].values
test.loc[mask, "FVC"] = test.loc[mask, "base_FVC"].values
test.loc[mask, "Confidence"] = 70.0

test["Confidence"] = test["Confidence"].astype(float)
test.loc[test.Confidence < 1.0, "Confidence"] = 1.0
test.loc[test.Confidence < 70.0, "Confidence"] = 70.0
test.loc[test.Confidence > 500.0, "Confidence"] = 500.0

test[["Patient", "Week", "FVC", "Confidence"]].head()



## === cell 12
submission = pd.read_csv(SAMPLE_SUB_PATH)
submission["FVC"] = test["FVC"].values
submission["Confidence"] = test["Confidence"].values

submission["FVC"] = submission["FVC"].astype(float)
submission["Confidence"] = submission["Confidence"].astype(float)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
