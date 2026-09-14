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

-6.879888052224982

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import scipy.ndimage

import pydicom

import torch
import torch.nn as nn
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


def read_image(dir_name, patientid, Z=100, Y=200, X=200):
    cpath = _cache_path(patientid, Z, Y, X)
    if os.path.exists(cpath):
        try:
            return np.load(cpath, allow_pickle=False)
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
C1, C2 = torch.tensor(70, dtype=torch.float32), torch.tensor(1000, dtype=torch.float32)


def score(y_true, y_pred):
    sigma = y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = torch.clamp(sigma, min=float(C1.item()))
    delta = (y_true[:, 0] - fvc_pred).abs()
    delta = torch.clamp(delta, max=float(C2.item()))

    sq2 = torch.tensor(2.0, device=y_pred.device).sqrt()
    metric = (delta / sigma_clip) * sq2 + (sigma_clip * sq2).log()
    return metric.mean()


def quartile_loss(y_true, y_pred):
    loss = score(y_true, y_pred)
    return loss




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
def make_eval_data(npEval, model, device="cuda", batch_size=8):
    x_features = npEval[
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
    ]
    x_features = torch.tensor(x_features.values).float()
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
    for i, pid in enumerate(x_patientids_name):
        x_image = loaded_images.get(pid, None)
        if x_image is None:
            x_image = np.zeros((100, 200, 200), dtype=np.uint8)
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
    npEval["Confidence"] = predictions[:, 0]
    return npEval




## === cell 7
data_train = pd.read_csv(os.path.join(BASE_INPUT, "train.csv"))
data_test = pd.read_csv(os.path.join(BASE_INPUT, "test.csv"))
submission = pd.read_csv(os.path.join(BASE_INPUT, "sample_submission.csv"))




## === cell 8
pw = submission["Patient_Week"].str.split("_", n=1, expand=True)
submission["Patient"] = pw[0].values
submission["Weeks"] = pw[1].astype(int).values
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
submission_out = merge.loc[:, ["Patient_Week", "base_FVC", "Confidence"]]
submission_out = submission_out.rename(columns={"base_FVC": "FVC"})




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
npData = pd.concat([npData, data], axis=0, sort=True, ignore_index=True)
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
class OSICTrainDataset(Dataset):
    def __init__(self, npTrain: pd.DataFrame):
        self.npTrain = npTrain.reset_index(drop=True)

        x_features = self.npTrain[
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
        ]
        self.x_features = pad_features_to_42(torch.tensor(x_features.values).float())
        self.y = torch.tensor(self.npTrain[["actual_FVC"]].values).float()
        self.pids = self.npTrain["Patient"].values

    def __len__(self):
        return len(self.npTrain)

    def __getitem__(self, idx):
        pid = self.pids[idx]
        img = read_image(TRAIN_FOLDER, pid, Z=100, Y=200, X=200)
        if img is None:
            img = np.zeros((100, 200, 200), dtype=np.uint8)
        x_img = torch.from_numpy(img.astype(np.float32, copy=False))[
            None, ...
        ]  # (1,100,200,200)
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
    model.load_state_dict(state)
    loaded = True

if not loaded:
    ds = build_train_dataset(data_train, max_patients_for_images=None)

    if device == "cuda":
        batch_size = 4
        num_workers = min(4, os.cpu_count() or 1)
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
        prefetch_factor=2 if num_workers > 0 else None,
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




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_56/3127326829.py in <cell line: 0>()
     49             yb = yb.to(device, non_blocking=True)
     50 
---> 51             pred = model(xb_img, xb_feat)
     52             loss = quartile_loss(yb, pred)
     53 

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

/tmp/ipykernel_56/61133390.py in forward(self, image_i, data_i)
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

/tmp/ipykernel_56/61133390.py in forward(self, data_i, image_o)
     45     def forward(self, data_i, image_o):
     46         x = torch.cat((data_i, image_o), dim=-1)
---> 47         out1 = self.data_net1(x)
     48         out2 = torch.cat((data_i, out1), dim=-1)
     49         out2 = self.data_net2(out2)

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

RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x74 and 42x64)

## === cell 12
test = make_eval_data(data_test.copy(), model, device=device, batch_size=8)

for nid in test.Patient.unique():
    idx = test[(test.Patient == nid) & (test.Week == test.base_Weeks)].index.values
    if len(idx) > 0:
        test.iloc[idx[0], test.columns.get_loc("FVC")] = test.iloc[
            idx[0], test.columns.get_loc("base_FVC")
        ]
        test.iloc[idx[0], test.columns.get_loc("Confidence")] = 70




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_56/3861885008.py in <cell line: 0>()
----> 1 test = make_eval_data(data_test.copy(), model, device=device, batch_size=8)
      2 
      3 for nid in test.Patient.unique():
      4     idx = test[(test.Patient == nid) & (test.Week == test.base_Weeks)].index.values
      5     if len(idx) > 0:

/tmp/ipykernel_56/4130119369.py in make_eval_data(npEval, model, device, batch_size)
     58                 xb_img = xb_img.cuda(non_blocking=True)
     59                 xb_feat = xb_feat.cuda(non_blocking=True)
---> 60             out = model(xb_img, xb_feat)
     61             preds.append(out.detach().cpu().numpy())
     62     predictions = np.concatenate(preds, axis=0)

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

/tmp/ipykernel_56/61133390.py in forward(self, image_i, data_i)
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

/tmp/ipykernel_56/61133390.py in forward(self, data_i, image_o)
     45     def forward(self, data_i, image_o):
     46         x = torch.cat((data_i, image_o), dim=-1)
---> 47         out1 = self.data_net1(x)
     48         out2 = torch.cat((data_i, out1), dim=-1)
     49         out2 = self.data_net2(out2)

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

RuntimeError: mat1 and mat2 shapes cannot be multiplied (8x74 and 42x64)

## === cell 13
test.loc[test.Confidence < 70, "Confidence"] = 70




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/1567086473.py in <cell line: 0>()
----> 1 test.loc[test.Confidence < 70, "Confidence"] = 70
      2 
      3 

NameError: name 'test' is not defined

## === cell 14
submission_out.loc[:, "FVC"] = test.FVC.values
submission_out.loc[:, "Confidence"] = test.Confidence.values
submission_out = submission_out[["Patient_Week", "FVC", "Confidence"]]
submission_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_out.shape)
print(submission_out.head())

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/429482956.py in <cell line: 0>()
----> 1 submission_out.loc[:, "FVC"] = test.FVC.values
      2 submission_out.loc[:, "Confidence"] = test.Confidence.values
      3 submission_out = submission_out[["Patient_Week", "FVC", "Confidence"]]
      4 submission_out.to_csv("submission.csv", index=False)
      5 print("Wrote submission.csv with shape:", submission_out.shape)

NameError: name 'test' is not defined
