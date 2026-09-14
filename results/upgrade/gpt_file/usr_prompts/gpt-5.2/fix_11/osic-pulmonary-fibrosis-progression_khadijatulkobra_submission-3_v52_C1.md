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

-7.137627295073682

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -24.65417) has done: 'I fix the immediate runtime break in pandas by replacing the removed `DataFrame.append` with a `pd.concat` equivalent. Then I remove the dependency on a missing external `.pth` file by running the existing architecture with random weights (same forward pass semantics) so the notebook can complete and produce a valid `submission.csv`. I also make the data/image paths consistent with the provided dataset layout (`../input/osic-pulmonary-fibrosis-progression/...`) and add a safe CPU fallback plus `no_grad()` for inference stability. Finally, I ensure predictions are aligned to `sample_submission.csv` order and apply the required confidence clipping/base-week correction so the submission format is valid.'
- What this solution (achieved -24.65417) has done: 'The timeout is dominated by repeatedly reading/decoding hundreds of DICOM slices per training sample and per test row, plus heavy pandas row-wise concatenation in `csv_preprocess`. I keep the same model, loss, and 1-epoch training loop, but make data access equivalent and far cheaper by (1) precomputing one resized/normalized CT volume per patient once and caching it (train + test), (2) switching the dataset to index into prebuilt numpy feature/label arrays (no per-row pandas/torch tensor creation overhead), (3) vectorizing `csv_preprocess` to remove the O(N²) concat-in-loop pattern, and (4) batching evaluation so the model runs on GPU efficiently without changing predictions. These changes preserve the exact inputs to the model (same `read_image` pipeline, same feature columns/values, same training steps), but eliminate redundant I/O and Python overhead to fit within 600 seconds.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import pydicom
import scipy.ndimage
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(max(1, os.cpu_count() // 2))




## === cell 1
TRAIN_FOLDER = "../input/osic-pulmonary-fibrosis-progression/train"


def load_scan(path):
    try:
        slices = [pydicom.dcmread(path + os.sep + s) for s in os.listdir(path)]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except Exception:
        files = os.listdir(path)
        files.sort()
        slices = [pydicom.dcmread(path + os.sep + s) for s in files]
    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    try:
        image[image <= -2000] = 0
        for slice_number in range(len(slices)):
            intercept = slices[slice_number].RescaleIntercept
            slope = slices[slice_number].RescaleSlope

            if slope != 1:
                image[slice_number] = slope * image[slice_number].astype(np.float64)
                image[slice_number] = image[slice_number].astype(np.int16)
            image[slice_number] += np.int16(intercept)
    except Exception:
        print("HU conversion Failed!!")
    return np.array(image, dtype=np.int16)


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
    for i, slc in enumerate(first_patient_pixels[:16]):
        y = fig.add_subplot(4, 4, i + 1)
        y.imshow(slc, cmap="gray")
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


def save_array(patientID_folder, output_folder, Z=100, Y=200, X=200):
    patientIDs = os.listdir(patientID_folder)
    patientIDs.sort()
    Save_dir = output_folder
    if not os.path.exists(Save_dir):
        print("The output directory doesnt exists")
        raise Exception

    for i, patientID in enumerate(patientIDs):
        path = TRAIN_FOLDER + os.sep + patientIDs[i]
        slices = load_scan(path)
        try:
            image_array = get_pixels_hu(slices)
            ctimage_resizedAll = resize_along_allaxis(
                image_array,
                target_dimensionX=X,
                target_dimensionY=Y,
                target_dimensionZ=Z,
            )
            image = (image_normalize(ctimage_resizedAll) * 255.0).astype("uint8")
            np.save(Save_dir + os.sep + patientID + ".npy", image)
        except Exception:
            print("PatientId:%s couldnt be save and converted" % (patientID))


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
    slices = load_scan(path)
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




## === cell 2
def csv_preprocess(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["Healthy-FVC"] = np.round((data["FVC"] * 100) / data["Percent"])

    COLS = ["Sex", "SmokingStatus"]
    for col in COLS:
        for mod in data[col].dropna().unique():
            data[mod] = (data[col] == mod).astype(int)

    data = data.sort_values(["Patient", "Weeks"], ascending=True).reset_index(drop=True)

    FE1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]
    for col in FE1:
        if col not in data.columns:
            data[col] = 0

    base = (
        data.groupby("Patient", sort=False)
        .first()
        .reset_index()[["Patient", "Weeks", "FVC"]]
        .rename(columns={"Weeks": "base_Weeks", "FVC": "base_FVC"})
    )

    out = data.merge(base, on="Patient", how="left", sort=False)
    out = out.rename(columns={"Weeks": "Week", "FVC": "actual_FVC"})

    out = out[
        ["Patient", "base_Weeks", "base_FVC", "Age"]
        + FE1
        + ["Week", "Healthy-FVC", "actual_FVC"]
    ].fillna(0)

    return out.reset_index(drop=True)




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
            nn.Linear(106, 64), nn.ReLU(), nn.Linear(64, 118), nn.ReLU()
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




## === cell 4
C1 = 70.0
C2 = 1000.0


def score(y_true, y_pred):
    sigma = y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = torch.clamp(
        sigma, min=torch.as_tensor(C1, device=y_pred.device, dtype=y_pred.dtype)
    )
    delta = (y_true[:, 0] - fvc_pred).abs()
    delta = torch.minimum(
        delta, torch.as_tensor(C2, device=y_pred.device, dtype=y_pred.dtype)
    )

    sq2 = torch.as_tensor(2.0, device=y_pred.device, dtype=y_pred.dtype).sqrt()
    metric = (delta / sigma_clip) * sq2 + (sigma_clip * sq2).log()
    return metric.mean()


def quartile_loss(y_true, y_pred):
    loss = score(y_true, y_pred)
    return loss




## === cell 5
FE1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]
BASE_FEAT_COLS_10 = [
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

PAD_COLS_64 = [f"pad_{i:02d}" for i in range(64)]
FEAT_COLS_74 = BASE_FEAT_COLS_10 + PAD_COLS_64


def ensure_feat74(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in BASE_FEAT_COLS_10:
        if c not in df.columns:
            df[c] = 0
    for c in PAD_COLS_64:
        if c not in df.columns:
            df[c] = 0.0
    keep_cols = (
        ["Patient"] + FEAT_COLS_74 + [c for c in df.columns if c == "actual_FVC"]
    )
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df.loc[:, keep_cols].copy()
    if "Patient" in df.columns:
        df = df[
            ["Patient"]
            + [c for c in FEAT_COLS_74 if c in df.columns]
            + [c for c in df.columns if c == "actual_FVC"]
        ]
    return df


def build_train_dataframe(train_csv: pd.DataFrame) -> pd.DataFrame:
    npTrain = csv_preprocess(train_csv.copy())
    for col in FE1:
        if col not in npTrain.columns:
            npTrain[col] = 0
    npTrain = npTrain[
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
    npTrain = ensure_feat74(npTrain)
    return npTrain


class OSICTrainDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, patient_to_img: dict):
        self.df = df.reset_index(drop=True)
        self.patient_to_img = patient_to_img

        self.patients = self.df["Patient"].to_numpy()
        self.X_feat = self.df[FEAT_COLS_74].to_numpy(dtype=np.float32, copy=True)
        self.y = self.df["actual_FVC"].to_numpy(dtype=np.float32, copy=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        pid = self.patients[idx]
        img = self.patient_to_img.get(pid)
        if img is None:
            img = np.zeros((100, 200, 200), dtype=np.uint8)

        x_image = (
            torch.from_numpy(img.astype(np.float32, copy=False)).unsqueeze(0) / 255.0
        )
        x_feat = torch.from_numpy(self.X_feat[idx])
        y = torch.tensor([self.y[idx]], dtype=torch.float32)
        return x_image, x_feat, y


def _preload_patient_images(
    patient_ids, image_root, Z=100, Y=200, X=200, desc="preload"
):
    out = {}
    for pid in tqdm(list(patient_ids), desc=desc):
        img = read_image(image_root, pid, Z=Z, Y=Y, X=X)
        if img is None:
            img = np.zeros((Z, Y, X), dtype=np.uint8)
        out[pid] = img
    return out


def train_one_run(model, train_df, device="cuda", batch_size=2, epochs=1, lr=1e-4):
    use_cuda = torch.cuda.is_available() and device == "cuda"
    device_t = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device_t)
    model.train()

    train_patient_imgs = _preload_patient_images(
        train_df["Patient"].unique(),
        TRAIN_FOLDER,
        Z=100,
        Y=200,
        X=200,
        desc="preload train CT",
    )

    ds = OSICTrainDataset(train_df, patient_to_img=train_patient_imgs)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=use_cuda,
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        for x_img, x_feat, y in tqdm(dl, desc="training", leave=False):
            x_img = x_img.to(device_t, non_blocking=True)
            x_feat = x_feat.to(device_t, non_blocking=True)
            y = y.to(device_t, non_blocking=True)

            pred = model(x_img, x_feat)
            y_true = y.view(-1, 1)
            loss = quartile_loss(y_true, pred)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    return model




## === cell 6
def make_eval_data(npEval: pd.DataFrame, model, device="cuda", batch_size=8):
    npEval = npEval.copy()
    npEval = ensure_feat74(npEval)

    x_features = npEval[FEAT_COLS_74].to_numpy(dtype=np.float32, copy=True)
    patients = npEval["Patient"].to_numpy()

    unique_patients = pd.unique(patients)
    dir_name_of_patientid = "../input/osic-pulmonary-fibrosis-progression/test"

    loaded_images = _preload_patient_images(
        unique_patients,
        dir_name_of_patientid,
        Z=100,
        Y=200,
        X=200,
        desc="preload test CT",
    )

    use_cuda = torch.cuda.is_available() and device == "cuda"
    device_t = torch.device("cuda" if use_cuda else "cpu")
    model.to(device_t)
    model.eval()

    imgs = np.stack([loaded_images.get(pid) for pid in patients], axis=0).astype(
        np.float32, copy=False
    )
    imgs_t = torch.from_numpy(imgs).unsqueeze(1) / 255.0  # [N,1,Z,Y,X]
    feats_t = torch.from_numpy(x_features)  # [N,74]

    preds = np.empty((len(npEval), 2), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(npEval), batch_size):
            end = min(len(npEval), start + batch_size)
            x_image = imgs_t[start:end].to(device_t, non_blocking=True)
            x_feature = feats_t[start:end].to(device_t, non_blocking=True)

            prediction = model(x_image, x_feature)
            prediction[:, 0] = torch.clamp(prediction[:, 0], min=1.0)
            preds[start:end] = prediction.detach().to("cpu").numpy()

    out = npEval.copy()
    out["FVC"] = preds[:, 1]
    out["Confidence"] = preds[:, 0]
    return out




## === cell 7
data_train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
data_test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
submission = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)




## === cell 8
model = Combined_NET()

train_df = build_train_dataframe(data_train)
try:
    model = train_one_run(
        model, train_df, device="cuda", batch_size=2, epochs=1, lr=1e-4
    )
except Exception as e:
    print(
        "Training failed; continuing with untrained model to still produce a submission."
    )
    print("Training error:", repr(e))
    model.eval()




## === cell 9
sub = submission.copy()
sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Week"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[1]))
sub = sub.sort_values(["Patient", "Week"]).reset_index(drop=True)

base = data_test.copy()
base = base.rename(columns={"Weeks": "base_Weeks", "FVC": "base_FVC"})
base = base[
    ["Patient", "base_Weeks", "base_FVC", "Percent", "Age", "Sex", "SmokingStatus"]
]

test_join = sub[["Patient_Week", "Patient", "Week"]].merge(
    base, on="Patient", how="left", sort=False
)

test_join["Healthy-FVC"] = np.round(
    (test_join["base_FVC"] * 100) / test_join["Percent"]
)

COLS = ["Sex", "SmokingStatus"]
for col in COLS:
    for mod in test_join[col].dropna().unique():
        test_join[mod] = (test_join[col] == mod).astype(int)

for col in FE1:
    if col not in test_join.columns:
        test_join[col] = 0

data_test_feat = test_join[
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

data_test_feat = ensure_feat74(data_test_feat)
submission_out = test_join[["Patient_Week"]].copy()

test_pred = make_eval_data(data_test_feat.copy(), model, device="cuda", batch_size=8)

for nid in test_join["Patient"].unique():
    base_week = float(test_join.loc[test_join["Patient"] == nid, "base_Weeks"].iloc[0])
    base_fvc = float(test_join.loc[test_join["Patient"] == nid, "base_FVC"].iloc[0])

    idxs = test_pred[
        (test_pred.Patient == nid)
        & (test_pred.base_Weeks == base_week)
        & (test_pred.Week == base_week)
    ].index.values
    if len(idxs) > 0:
        i0 = idxs[0]
        test_pred.iloc[i0, test_pred.columns.get_loc("FVC")] = base_fvc
        test_pred.iloc[i0, test_pred.columns.get_loc("Confidence")] = 70.0

test_pred.loc[test_pred.Confidence < 70, "Confidence"] = 70.0

test_pred["Patient_Week"] = (
    test_pred["Patient"].astype(str) + "_" + test_pred["Week"].astype(int).astype(str)
)
pred_map = test_pred[["Patient_Week", "FVC", "Confidence"]]

submission_final = submission_out.merge(pred_map, on="Patient_Week", how="left")

submission_final["FVC"] = submission_final["FVC"].fillna(
    submission_final["FVC"].median() if submission_final["FVC"].notna().any() else 2000
)
submission_final["Confidence"] = submission_final["Confidence"].fillna(70.0)
submission_final.loc[submission_final["Confidence"] < 70, "Confidence"] = 70.0

submission_final.to_csv("submission.csv", index=False)
print(submission_final.head())
print("Wrote submission.csv with shape:", submission_final.shape)
print("Columns:", list(submission_final.columns))

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_56/3255983612.py in <cell line: 0>()
     46 submission_out = test_join[["Patient_Week"]].copy()
     47 
---> 48 test_pred = make_eval_data(data_test_feat.copy(), model, device="cuda", batch_size=8)
     49 
     50 for nid in test_join["Patient"].unique():

/tmp/ipykernel_56/3829159402.py in make_eval_data(npEval, model, device, batch_size)
     36             x_feature = feats_t[start:end].to(device_t, non_blocking=True)
     37 
---> 38             prediction = model(x_image, x_feature)
     39             prediction[:, 0] = torch.clamp(prediction[:, 0], min=1.0)
     40             preds[start:end] = prediction.detach().to("cpu").numpy()

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

/tmp/ipykernel_56/1830947498.py in forward(self, image_i, data_i)
    157     def forward(self, image_i, data_i):
    158         image_o = self.image(image_i)
--> 159         data_o = self.data(data_i, image_o)
    160         return data_o
    161 

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

/tmp/ipykernel_56/1830947498.py in forward(self, data_i, image_o)
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

RuntimeError: mat1 and mat2 shapes cannot be multiplied (8x192 and 128x256)
