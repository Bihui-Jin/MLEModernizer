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

-7.10343049416784

# 6. Current score

-10.06049

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -12.16418) has done: 'I fix the CT cache writing bug that crashes in multi-threaded mode: `np.save()` appends “.npy”, so saving to a “.tmp” name and then `os.replace()` fails because the expected temp file doesn’t exist. I make the cache directory writable by moving it to `/kaggle/working` (the input directory is read-only) and write atomically using a temp file with the right extension. I also filter out any patients whose CT failed to load (None volumes) so the Dataset/DataLoader won’t crash, and then the rest of the pipeline (training → inference → submission.csv) run end-to-end with the same model/loss/core logic.'
- What this solution (achieved -10.06049) has done: 'Your current score (-12.16418) is far below the target (-7.1034), so we should improve performance with minimal risk while keeping the same model/loss/feature logic. The biggest gain with minimal semantic change is to train longer (still the same loop/objective) because EPOCHS=1 is severely underfitting; increasing epochs is a direct way to move toward the target without changing architecture or metric alignment. I also ensure the predicted Confidence is always valid by clipping both in the submission step (and making it strictly positive before clipping) to avoid any accidental negative/zero sigma effects. All paths and the submission schema remain unchanged, and the script still runs end-to-end within constraints.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import scipy.ndimage
import pydicom

import torch
import torch.nn as nn

from tqdm.auto import tqdm


def seed_everything(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_INPUT = "../input/osic-pulmonary-fibrosis-progression"
TRAIN_FOLDER = os.path.join(BASE_INPUT, "train")
TEST_FOLDER = os.path.join(BASE_INPUT, "test")




## === cell 1
_DICOM_TAGS = [
    "ImagePositionPatient",
    "RescaleIntercept",
    "RescaleSlope",
    "PixelData",
    "BitsAllocated",
    "BitsStored",
    "HighBit",
    "PixelRepresentation",
    "SamplesPerPixel",
    "PhotometricInterpretation",
    "Rows",
    "Columns",
]


def load_scan(path):
    try:
        files = os.listdir(path)
        slices = [
            pydicom.dcmread(path + os.sep + s, specific_tags=_DICOM_TAGS) for s in files
        ]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except Exception:
        files = os.listdir(path)
        files.sort()
        slices = [
            pydicom.dcmread(path + os.sep + s, specific_tags=_DICOM_TAGS) for s in files
        ]
    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices]).astype(np.int16)
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

    COLS = ["Sex", "SmokingStatus"]
    for col in COLS:
        for mod in data[col].unique():
            data[mod] = (data[col] == mod).astype(int)

    data = data[
        ["Patient", "Weeks", "FVC", "Age", "Healthy-FVC"]
        + list(data["Sex"].unique())
        + list(data["SmokingStatus"].unique())
    ]
    data = data.sort_values(["Patient", "Weeks"], ascending=True).reset_index(drop=True)

    FE1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]
    for c in FE1:
        if c not in data.columns:
            data[c] = 0

    data = data.rename(columns={"Weeks": "base_Weeks", "FVC": "base_FVC"})

    base = data.groupby("Patient", sort=False).head(1).copy()

    meas = data[["Patient", "base_Weeks", "base_FVC"]].copy()
    meas = meas.rename(columns={"base_Weeks": "Week", "base_FVC": "actual_FVC"})

    base_rep = base.merge(meas, on="Patient", how="right", sort=False)

    base_rep = base_rep.reset_index(drop=True).fillna(0)
    base_rep["Week"] = base_rep["Week"] - base_rep["base_Weeks"]
    base_rep["base_Weeks"] = 0.0

    cols = (
        ["Patient", "base_Weeks", "base_FVC", "Age"]
        + FE1
        + ["Week", "Healthy-FVC", "actual_FVC"]
    )
    for c in cols:
        if c not in base_rep.columns:
            base_rep[c] = 0
    return base_rep[cols].copy()




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
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = torch.clamp(sigma, min=C1)
    delta = (y_true[:, 0] - fvc_pred).abs()
    delta = torch.clamp(delta, max=C2)

    sq2 = torch.sqrt(torch.tensor(2.0, device=y_pred.device))
    metric = (delta / sigma_clip) * sq2 + (sigma_clip * sq2).log()
    return metric.mean()


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
def make_eval_data(npEval, model, image_cache, device=DEVICE, batch_size=8):
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
    ].to_numpy(dtype=np.float32, copy=False)
    patients = npEval["Patient"].to_numpy()

    model = model.to(device)
    model.eval()
    preds = np.empty((len(npEval), 3), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, len(npEval), batch_size):
            end = min(start + batch_size, len(npEval))
            batch_patients = patients[start:end]
            batch_imgs = np.stack([image_cache[p] for p in batch_patients]).astype(
                np.float32, copy=False
            )  # [B,Z,Y,X]
            x_img = (
                torch.from_numpy(batch_imgs).unsqueeze(1).to(device, non_blocking=True)
            )
            x_feat = torch.from_numpy(x_features[start:end]).to(
                device, non_blocking=True
            )

            out = model(x_img, x_feat).detach().cpu().numpy()
            preds[start:end] = out

    npEval["FVC"] = preds[:, 1]
    npEval["Confidence"] = preds[:, 2] - preds[:, 0]
    return npEval




## === cell 6
data_train = pd.read_csv(os.path.join(BASE_INPUT, "train.csv"))
data_test = pd.read_csv(os.path.join(BASE_INPUT, "test.csv"))
sample_sub = pd.read_csv(os.path.join(BASE_INPUT, "sample_submission.csv"))




## === cell 7
submission = sample_sub.copy()
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
submission_out = merge.loc[:, ["Patient_Week"]].copy()




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
for c in FE1:
    if c not in data.columns:
        data[c] = 0

npData = pd.DataFrame(
    columns=["Patient", "base_Weeks", "base_FVC", "Age", "Healthy-FVC"] + FE1 + ["Week"]
)
npData = pd.concat([npData, data], axis=0, ignore_index=True, sort=True)
npData = npData.fillna(0)

npData["Week"] = npData["Week"] - npData["base_Weeks"]
npData["base_Weeks"] = 0.0

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

train_np = train_np[
    train_np["Patient"].apply(lambda p: os.path.isdir(os.path.join(TRAIN_FOLDER, p)))
].reset_index(drop=True)




## === cell 10
from concurrent.futures import ThreadPoolExecutor, as_completed

_CT_CACHE_DIR = os.path.join("/kaggle/working", "_ct_cache_uint8_z100_y200_x200")
os.makedirs(_CT_CACHE_DIR, exist_ok=True)


def _cache_path(patient_id: str) -> str:
    return os.path.join(_CT_CACHE_DIR, f"{patient_id}.npy")


def _load_or_build_one(patient_id: str, image_root: str, Z: int, Y: int, X: int):
    p = _cache_path(patient_id)
    if os.path.exists(p):
        arr = np.load(p, mmap_mode=None)  # uint8 [Z,Y,X]
        return patient_id, arr
    arr = read_image(image_root, patient_id, Z=Z, Y=Y, X=X)
    if arr is None:
        return patient_id, None
    tmp = p + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, p)
    return patient_id, arr


def build_image_cache(patient_ids, image_root, Z=100, Y=200, X=200):
    patient_ids = list(patient_ids)
    cache = {}
    max_workers = min(8, (os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [
            ex.submit(_load_or_build_one, pid, image_root, Z, Y, X)
            for pid in patient_ids
        ]
        for fut in tqdm(
            as_completed(futs),
            total=len(futs),
            desc=f"Loading CT volumes from {image_root}",
        ):
            pid, arr = fut.result()
            cache[pid] = arr
    return cache


train_patient_ids = train_np["Patient"].unique()
test_patient_ids = data_test["Patient"].unique()

train_image_cache = build_image_cache(
    train_patient_ids, TRAIN_FOLDER, Z=100, Y=200, X=200
)
test_image_cache = build_image_cache(test_patient_ids, TEST_FOLDER, Z=100, Y=200, X=200)

bad_train = {p for p, v in train_image_cache.items() if v is None}
if bad_train:
    train_np = train_np[~train_np["Patient"].isin(bad_train)].reset_index(drop=True)
    train_image_cache = {p: v for p, v in train_image_cache.items() if v is not None}

bad_test = {p for p, v in test_image_cache.items() if v is None}
if bad_test:
    data_test = data_test[~data_test["Patient"].isin(bad_test)].reset_index(drop=True)
    submission_out = submission_out.loc[data_test.index].reset_index(drop=True)
    test_image_cache = {p: v for p, v in test_image_cache.items() if v is not None}

print("Train patients:", train_np["Patient"].nunique(), "rows:", len(train_np))
print("Test patients:", data_test["Patient"].nunique(), "rows:", len(data_test))




## === cell 11
class OSICDataset(torch.utils.data.Dataset):
    def __init__(self, df, image_cache):
        self.df = df.reset_index(drop=True)

        patients = self.df["Patient"].to_numpy()
        self.patients = patients

        self.x_feat = self.df[
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
        ].to_numpy(dtype=np.float32, copy=True)
        self.y = np.repeat(
            self.df[["actual_FVC"]].to_numpy(dtype=np.float32, copy=True), 3, axis=1
        )

        self.images = [image_cache[p] for p in patients]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x_feat = torch.from_numpy(self.x_feat[idx])
        img = self.images[idx]
        x_img = torch.from_numpy(img.astype(np.float32, copy=False)).unsqueeze(
            0
        )  # [1,Z,Y,X]
        y = torch.from_numpy(self.y[idx])
        return x_img, x_feat, y


train_ds = OSICDataset(train_np, train_image_cache)

_num_workers = min(4, os.cpu_count() or 1)
train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=1,
    shuffle=True,
    num_workers=_num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_num_workers > 0),
    prefetch_factor=2 if _num_workers > 0 else None,
)




## === cell 12
model = Combined_NET().to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr=1e-4)

model.train()

EPOCHS = 5

for epoch in range(EPOCHS):
    running = 0.0
    n = 0
    for x_img, x_feat, y in tqdm(
        train_loader, desc=f"Training epoch {epoch+1}/{EPOCHS}"
    ):
        x_img = x_img.to(DEVICE, non_blocking=True)
        x_feat = x_feat.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        pred = model(x_img, x_feat)
        loss = quartile_loss(y, pred)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        running += float(loss.detach().cpu().item())
        n += 1
    if n > 0:
        print(f"Epoch {epoch+1} mean loss: {running/n:.6f}")




## === cell 13
test_pred_df = make_eval_data(
    data_test.copy(), model, image_cache=test_image_cache, device=DEVICE, batch_size=8
)

test_pred_df["Confidence"] = test_pred_df["Confidence"].astype(np.float32)
test_pred_df["Confidence"] = np.maximum(test_pred_df["Confidence"].values, 1.0).astype(
    np.float32
)
test_pred_df["Confidence"] = test_pred_df["Confidence"].clip(lower=70.0)




## === cell 14
sub = sample_sub.copy()
sub["FVC"] = test_pred_df["FVC"].values
sub["Confidence"] = test_pred_df["Confidence"].values

sub["FVC"] = sub["FVC"].astype(np.float32)
sub["Confidence"] = sub["Confidence"].astype(np.float32)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
