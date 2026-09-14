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

-6.933046029604298

# 6. Current score

-10.44284

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -24.64816) has done: 'I fix the pandas `DataFrame.append` crash by replacing it with `pd.concat`, which is the direct modern equivalent and keeps the same data flow. Then I remove the hardcoded missing pretrained-weight path and instead run inference with the untrained model weights (so the notebook always runs end-to-end in this environment), using `torch.no_grad()` for correctness and speed. I also fix the wrong test DICOM folder path (`../input/...`) to the provided dataset path under `../data/osic-pulmonary-fibrosis-progression/test` so image loading works. Finally, I ensure the submission is aligned to `sample_submission.csv` order and always writes a valid `submission.csv` with the required columns.'
- What this solution (achieved -24.65932) has done: 'Your current poor score is mainly because the model is completely untrained (random weights), so predictions are essentially noise; the smallest legitimate way to move toward the target is to actually use the provided `train.csv` to fit the existing network with the same loss you already defined, then run the same inference pipeline. I add a minimal training step that uses the same clinical feature construction already present (no new feature logic), and I train on a small, fixed number of samples/epochs to improve score substantially without aiming for best performance. I also ensure `Confidence` stays valid by using a `softplus` transform during loss/inference so sigma stays positive (this matches the metric’s intent and avoids negative/zero sigma issues) while still respecting your core output semantics (sigma, fvc). Submission writing and alignment stay the same.'
- What this solution (achieved -11.12699) has done: 'The timeout is dominated by repeatedly reading and resizing hundreds of DICOM slices per patient with slow Python/pydicom loops, plus redundant tensor/image replication during training and inference. I keep the exact model, loss, and training loop semantics, but speed up CT preprocessing by (1) using pydicom’s `stop_before_pixels=True` to sort slices without decoding pixels, (2) using vectorized HU conversion (apply intercept/slope per-slice with numpy broadcasting), (3) replacing `scipy.ndimage.zoom` with much faster `torch.nn.functional.interpolate` on CPU for 3D resizing (same trilinear/nearest-family semantics), and (4) caching the processed per-patient volume to `.npy` under the same patient folder so repeated runs don’t redo DICOM work. I also remove per-batch Python list/stack overhead by precomputing a dense `(N,Z,Y,X)` image tensor aligned to rows, and during evaluation avoid expanding the same image B times by running once per patient and broadcasting the result (mathematically identical because the image and network are the same for all that patient’s rows). These changes are purely performance-focused and preserve the algorithm’s logic and predictions up to negligible floating-point differences.'
- What this solution (achieved -11.15535) has done: 'I fix the CUDA out-of-memory error in `make_eval_data` by removing the expensive `repeat()` of the 3D CT tensor and instead expanding it as a view (`expand`), which preserves the same computations without allocating a huge `(B,1,Z,Y,X)` copy. I also make inference run in small chunks per patient to keep peak activation memory bounded, while keeping the model, features, and postprocessing exactly the same. Finally, I ensure the pipeline always reaches submission-writing by fixing the cascade `NameError` (caused by the earlier crash) and writing `submission.csv` with the required columns and ordering.'
- What this solution (achieved -10.44284) has done: 'Your current score (-11.15535) is worse than the target (-6.9330), so we should cautiously improve performance with minimal semantic changes. The biggest issue is that your training table uses each row as its own “baseline” (Week==base_Weeks), which does not match the test setting (only baseline Week=0 is known); I keep the same features/loss/model, but rebuild the training table to use a consistent per-patient baseline (closest to Week 0) and then create training pairs across all weeks, matching the evaluation setup. This is a data-prep fix (not a new model/feature) and typically gives a large, stable lift toward the target. I also make the train-time target tensor match the loss’ expected shape (N,2) with the true FVC in column 0 (sigma column unused by the loss), which is a correctness fix and avoids unintended broadcasting.'
- What this solution (achieved -10.44284) has done: 'We’re currently well below the target (current -10.44 vs target -6.93; higher is better), so the smallest safe improvement is to better align training with the competition metric without changing the model or data features. Your current `quartile_loss/score` is missing the leading negative sign from the Laplace log-likelihood, so training is effectively optimizing in the wrong direction; fixing that keeps the same semantics but makes the model actually learn what the leaderboard scores. I also correct the `y_true` column order so `y_true[:,0]` is the true FVC as expected by `score()`, and I keep everything else (CT preprocessing, architecture, training loop structure, inference, submission writing) unchanged. These changes are minimal but should move the score upward toward the target band.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import pydicom
import scipy.ndimage
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

TRAIN_FOLDER = "../data/train"


def load_scan(path):
    files = os.listdir(path)
    dsets = []
    for s in files:
        fp = path + os.sep + s
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
        except Exception:
            ds = None
        if ds is not None:
            ds._fp = fp  # stash path for later pixel decode
            dsets.append(ds)
    if not dsets:
        files.sort()
        return [pydicom.dcmread(path + os.sep + s, force=True) for s in files]

    try:
        dsets.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except Exception:
        dsets.sort(key=lambda x: os.path.basename(getattr(x, "_fp", "")))
    return dsets


def get_pixels_hu(slices):
    imgs = [
        (
            pydicom.dcmread(getattr(s, "_fp", ""), force=True).pixel_array
            if not hasattr(s, "PixelData")
            else s.pixel_array
        )
        for s in slices
    ]
    image = np.stack(imgs).astype(np.int16, copy=False)

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
        if not np.all(slope == 1.0):
            out = out * slope[:, None, None]
        out = out + intercept[:, None, None]
        image = out.astype(np.int16, copy=False)
    except Exception:
        print("HU conversion Failed!!")
    return np.array(image, dtype=np.int16, copy=False)


def plot_show_slice(slices):
    if not isinstance(slices, type(np.array([]))):
        first_patient_pixels = get_pixels_hu(slices)
    else:
        first_patient_pixels = slices

    print("Number of Total Slices in this Scan:", len(slices))
    try:
        print("Shape of the the Image is:", slices.shape[1], slices.shape[2])
    except Exception:
        print("Shape of the the Image is:BLANK")
    fig = plt.figure(figsize=(10, 10))
    for i, sl in enumerate(first_patient_pixels[:16]):
        y = fig.add_subplot(4, 4, i + 1)
        y.imshow(sl, cmap="gray")
    plt.show()


def resize_along_zaxis(slices, target_dimension=30):
    present_dimension = len(slices)
    if target_dimension == present_dimension:
        return slices
    zoom_factor = float(target_dimension) / float(present_dimension)
    resize_image = scipy.ndimage.zoom(slices, [zoom_factor, 1.0, 1.0])
    return resize_image


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

    vol = torch.from_numpy(slices.astype(np.float32, copy=False))[None, None, ...]
    vol = F.interpolate(
        vol,
        size=(target_dimensionZ, target_dimensionY, target_dimensionX),
        mode="trilinear",
        align_corners=False,
    )
    return vol[0, 0].numpy().astype(slices.dtype, copy=False)


MIN_BOUND = -1000.0
MAX_BOUND = 400.0


def image_normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.0
    image[image < 0] = 0.0
    return image


def load_array(path):
    try:
        if path.endswith(".npy"):
            image_array = np.load(path)
        else:
            path = path + ".npy"
            image_array = np.load(path)
    except Exception:
        print(
            "The file in the Path:%s doesnetexists!!"
            % (path.split("\\")[-1].split("/")[-1].split(".")[0])
        )
        return []
    return image_array


def read_image(dir_name, patientid, Z=100, Y=200, X=200):
    path = dir_name + os.sep + patientid
    cache_path = path + os.sep + f"ct_{Z}_{Y}_{X}.npy"
    try:
        if os.path.exists(cache_path):
            return np.load(cache_path, mmap_mode="r")
    except Exception:
        pass

    slices = load_scan(path)
    try:
        image_array = get_pixels_hu(slices)
        ctimage_resizedAll = resize_along_allaxis(
            image_array, target_dimensionX=X, target_dimensionY=Y, target_dimensionZ=Z
        )
        image = (image_normalize(ctimage_resizedAll) * 255.0).astype("uint8")
        try:
            np.save(cache_path, image)
        except Exception:
            pass
        return image
    except Exception:
        print("PatientId:%s couldnt be converted" % (patientid))
        return None




## === cell 1
def csv_preprocess(data):
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

    npData = pd.DataFrame(
        columns=["Patient", "base_Weeks", "base_FVC", "Age"]
        + FE1
        + ["Week", "Healthy-FVC", "actual_FVC"]
    )

    for pid in data["Patient"].unique():
        weeks = data.loc[data["Patient"] == pid].base_Weeks
        fvc = data.loc[data["Patient"] == pid].base_FVC
        index = data.loc[data["Patient"] == pid].index
        weeks.reset_index(inplace=True, drop=True)
        fvc.reset_index(inplace=True, drop=True)
        for k in range(len(weeks)):
            npData = pd.concat([npData, data.loc[data.index == index[0]]], sort=False)
            npData.iloc[-1, npData.columns.get_loc("Week")] = weeks[k]
            npData.iloc[-1, npData.columns.get_loc("actual_FVC")] = fvc[k]
    npData.reset_index(inplace=True, drop=True)
    npData = npData.fillna(0)
    return npData




## === cell 2
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




## === cell 3
C1_VAL, C2_VAL = 70.0, 1000.0


def score(y_true, y_pred):
    sigma = y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = torch.clamp(sigma, min=C1_VAL)
    delta = (y_true[:, 0] - fvc_pred).abs()
    delta = torch.clamp(delta, max=C2_VAL)

    sq2 = torch.sqrt(torch.tensor(2.0, device=y_pred.device, dtype=y_pred.dtype))
    metric = -((delta / sigma_clip) * sq2 + (sigma_clip * sq2).log())
    return metric.mean()


def quartile_loss(y_true, y_pred):
    return -score(y_true, y_pred)




## === cell 4
def postprocess_pred(raw_pred):
    raw_sigma = raw_pred[:, 0:1]
    raw_fvc = raw_pred[:, 1:2]
    sigma = F.softplus(raw_sigma) + 1e-3
    return torch.cat([sigma, raw_fvc], dim=1)




## === cell 5
def make_eval_data(npEval, model, device="cuda", patient_chunk_size=2):
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
    x_features = torch.tensor(npEval[feature_cols].values, dtype=torch.float32)
    x_patientids_name = npEval[["Patient"]].values

    unique_patients = npEval.Patient.unique()
    dir_name_of_patientid = "../data/osic-pulmonary-fibrosis-progression/test"

    loaded_images_t = {}
    for unique_patient in unique_patients:
        img = read_image(dir_name_of_patientid, unique_patient, Z=100, Y=200, X=200)
        if img is None:
            img = np.zeros((100, 200, 200), dtype=np.uint8)
        loaded_images_t[unique_patient] = torch.from_numpy(np.asarray(img)).to(
            dtype=torch.float32
        )  # (Z,Y,X)

    use_cuda = torch.cuda.is_available() and device == "cuda"
    if use_cuda:
        model.to("cuda")
    model.eval()

    pid_to_indices = {}
    for i, pid_arr in enumerate(x_patientids_name):
        pid = pid_arr[0]
        pid_to_indices.setdefault(pid, []).append(i)

    if use_cuda:
        x_features = x_features.pin_memory()

    predictions = np.empty((len(npEval), 2), dtype=np.float32)
    with torch.no_grad():
        for pid, idxs in pid_to_indices.items():
            img1 = loaded_images_t[pid].unsqueeze(0).unsqueeze(0)  # (1,1,Z,Y,X)
            if use_cuda:
                img1 = img1.cuda(non_blocking=True)

            idxs = list(idxs)
            for start in range(0, len(idxs), patient_chunk_size):
                sub = idxs[start : start + patient_chunk_size]
                feat = x_features[sub]  # (b,10)
                if use_cuda:
                    feat = feat.cuda(non_blocking=True)

                b = len(sub)
                img_b = img1.expand(b, -1, -1, -1, -1)  # (b,1,Z,Y,X) view

                raw_prediction = model(img_b, feat)
                prediction = postprocess_pred(raw_prediction).detach().cpu().numpy()
                predictions[np.array(sub, dtype=np.int64)] = prediction

    npEval["FVC"] = predictions[:, 1]
    npEval["Confidence"] = predictions[:, 0]
    return npEval




## === cell 6
data_train = pd.read_csv("../data/osic-pulmonary-fibrosis-progression/train.csv")
data_test = pd.read_csv("../data/osic-pulmonary-fibrosis-progression/test.csv")
submission = pd.read_csv(
    "../data/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)




## === cell 7
def build_train_table(train_df):
    df = train_df.copy()
    df["Healthy-FVC"] = round((df["FVC"] * 100) / df["Percent"])
    for col in ["Sex", "SmokingStatus"]:
        for mod in df[col].unique():
            df[mod] = (df[col] == mod).astype(int)
    for c in ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]:
        if c not in df.columns:
            df[c] = 0

    df = df.sort_values(["Patient", "Weeks"]).reset_index(drop=True)

    rows = []
    for pid, g in df.groupby("Patient", sort=False):
        g = g.sort_values("Weeks").reset_index(drop=True)

        bidx = (g["Weeks"].abs()).values.argmin()
        base = g.loc[bidx]

        tmp = g.copy()
        tmp["base_Weeks"] = base["Weeks"]
        tmp["base_FVC"] = base["FVC"]
        tmp["Week"] = tmp["Weeks"]
        tmp["actual_FVC"] = tmp["FVC"]

        tmp = tmp[
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
        ].fillna(0)
        rows.append(tmp)

    out = pd.concat(rows, axis=0, ignore_index=True)
    return out


def train_model(model, train_df, device):
    use_cuda = torch.cuda.is_available() and device == "cuda"
    if use_cuda:
        model.to("cuda")
    model.train()

    patients = sorted(train_df["Patient"].unique().tolist())
    train_df = train_df[train_df["Patient"].isin(patients)].reset_index(drop=True)

    train_img_dir = "../data/osic-pulmonary-fibrosis-progression/train"
    loaded_images_t = {}
    for pid in tqdm(patients, desc="Loading train CTs"):
        img = read_image(train_img_dir, pid, Z=100, Y=200, X=200)
        if img is None:
            img = np.zeros((100, 200, 200), dtype=np.uint8)
        loaded_images_t[pid] = torch.from_numpy(np.asarray(img)).to(dtype=torch.float32)

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
    x_features = torch.tensor(train_df[feature_cols].values, dtype=torch.float32)

    y_fvc = torch.tensor(train_df[["actual_FVC"]].values, dtype=torch.float32)
    y_true = torch.cat([y_fvc, torch.zeros_like(y_fvc)], dim=1)

    idx_to_pid = train_df["Patient"].values

    imgs_np = np.empty((len(train_df), 100, 200, 200), dtype=np.float32)
    for i, pid in enumerate(idx_to_pid):
        imgs_np[i] = loaded_images_t[pid].numpy()
    x_images = torch.from_numpy(imgs_np)  # (N,Z,Y,X)

    ds = TensorDataset(x_features, y_true, x_images)
    dl = DataLoader(
        ds,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=use_cuda,
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = 3

    for _ in range(epochs):
        for xb, yb, imgzb in dl:
            x_image = imgzb.unsqueeze(1)  # (B,1,Z,Y,X)

            if use_cuda:
                xb = xb.cuda(non_blocking=True)
                yb = yb.cuda(non_blocking=True)
                x_image = x_image.cuda(non_blocking=True)

            opt.zero_grad(set_to_none=True)
            raw_pred = model(x_image, xb)
            pred = postprocess_pred(raw_pred)

            loss = quartile_loss(yb, pred)
            loss.backward()
            opt.step()

    model.eval()
    return model


device = "cuda" if torch.cuda.is_available() else "cpu"
train_table = build_train_table(data_train)
model = Combined_NET()
model = train_model(model, train_table, device=device)



## === cell 8
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

del data_test
del submission

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



## === cell 9
data = data_test.copy()
data["Healthy-FVC"] = round((data["base_FVC"] * 100) / data["Percent"])
FE = []
FE.append("Healthy-FVC")

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

del data_test, data
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
]
del npData



## === cell 10
test = make_eval_data(
    data_test.copy(),
    model,
    device="cuda" if torch.cuda.is_available() else "cpu",
    patient_chunk_size=2,
)



## === cell 11
for nid in test.Patient.unique():
    index = test[(test.Patient == nid) & (test.Week == test.base_Weeks)].index.values
    if len(index) > 0:
        test.iloc[index[0], test.columns.get_loc("FVC")] = test.iloc[
            index[0], test.columns.get_loc("base_FVC")
        ]



## === cell 12
test.loc[test.Confidence < 70, "Confidence"] = 70



## === cell 13
sub_out = pd.read_csv(
    "../data/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)
sub_out["Patient"] = sub_out["Patient_Week"].apply(lambda x: x.split("_")[0])
sub_out["Weeks"] = sub_out["Patient_Week"].apply(lambda x: x.split("_")[1]).astype(int)
sub_out = sub_out.sort_values(by=["Patient", "Weeks"], ascending=True).reset_index(
    drop=True
)

test_sorted = test.copy()
test_sorted = test_sorted.merge(
    sub_out[["Patient_Week", "Patient", "Weeks"]],
    left_on=["Patient", "Week"],
    right_on=["Patient", "Weeks"],
    how="right",
)
test_sorted = test_sorted.sort_values(by=["Patient", "Weeks"]).reset_index(drop=True)

sub_out.loc[:, "FVC"] = test_sorted["FVC"].values
sub_out.loc[:, "Confidence"] = test_sorted["Confidence"].values
sub_out = sub_out[["Patient_Week", "FVC", "Confidence"]]
sub_out.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub_out.shape)
print(sub_out.head())
