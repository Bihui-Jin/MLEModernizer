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

-6.848780787703139

# 6. Current score

-8.49531

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -24.65669) has done: 'I fixed the pandas `append` deprecation, added a safe fallback when the pretrained model file is missing, wrapped model inference in `torch.no_grad`, guarded the baseline‑week correction loop against missing rows, and kept the original workflow intact so a valid `submission.csv` is written.'
- What this solution (achieved -9.11845) has done: 'I replace the random‑weight model inference with a simple, data‑driven heuristic: compute a per‑patient (or global) linear trend of FVC over weeks from the training set and use it to predict future weeks. This keeps the original architecture untouched, removes the reliance on missing pretrained weights, and should raise the score toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved -8.49531) has done: 'I compute a per‑patient estimate of the typical error (standard deviation of residuals) from the linear fit on the training data and use it as the confidence value in the heuristic predictions (with a minimum of 70). This keeps the overall workflow unchanged while providing a more realistic σ, which should raise the Laplace‑Log‑Likelihood score toward the target.'

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
from skimage import measure, morphology
from torch.utils.data import DataLoader, TensorDataset




## === cell 1
TRAIN_FOLDER = "../data/train"


def load_scan(path):
    try:
        slices = [pydicom.dcmread(os.path.join(path, s)) for s in os.listdir(path)]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except Exception:
        files = os.listdir(path)
        files.sort()
        slices = [pydicom.dcmread(os.path.join(path, s)) for s in files]
    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices]).astype(np.int16)
    try:
        image[image <= -2000] = 0
        for i, s in enumerate(slices):
            intercept = s.RescaleIntercept
            slope = s.RescaleSlope
            if slope != 1:
                image[i] = (slope * image[i].astype(np.float64)).astype(np.int16)
            image[i] += np.int16(intercept)
    except Exception:
        print("HU conversion Failed!!")
    return np.array(image, dtype=np.int16)


def resize_along_allaxis(
    slices, target_dimensionZ=30, target_dimensionY=100, target_dimensionX=100
):
    dz, dy, dx = slices.shape
    if (dz, dy, dx) == (target_dimensionZ, target_dimensionY, target_dimensionX):
        return slices
    zoom = (target_dimensionZ / dz, target_dimensionY / dy, target_dimensionX / dx)
    return scipy.ndimage.zoom(slices, zoom, mode="nearest")


MIN_BOUND = -1000.0
MAX_BOUND = 400.0


def image_normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image = np.clip(image, 0, 1)
    return image


def read_image(dir_name, patientid, Z=100, Y=200, X=200):
    path = os.path.join(dir_name, patientid)
    slices = load_scan(path)
    try:
        img = get_pixels_hu(slices)
        img = resize_along_allaxis(
            img, target_dimensionX=X, target_dimensionY=Y, target_dimensionZ=Z
        )
        img = (image_normalize(img) * 255).astype("uint8")
        return img
    except Exception:
        print(f"PatientId:{patientid} couldn't be converted")
        return np.zeros((Z, Y, X), dtype="uint8")




## === cell 2
def csv_preprocess(data):
    data["Healthy-FVC"] = round((data["FVC"] * 100) / data["Percent"])
    fe = ["Healthy-FVC"]
    for col in ["Sex", "SmokingStatus"]:
        for mod in data[col].unique():
            fe.append(mod)
            data[mod] = (data[col] == mod).astype(int)
    data = data[["Patient", "Weeks", "FVC", "Age"] + fe]
    data = data.sort_values(["Patient", "Weeks"]).reset_index(drop=True)
    data = data.rename(columns={"Weeks": "base_Weeks", "FVC": "base_FVC"})
    npdata = pd.DataFrame(
        columns=[
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
    )
    for pid in data["Patient"].unique():
        sub = data[data["Patient"] == pid]
        weeks = sub["base_Weeks"].reset_index(drop=True)
        fvc = sub["base_FVC"].reset_index(drop=True)
        idx = sub.index
        for k in range(len(weeks)):
            row = sub.loc[idx[0]].copy()
            row["Week"] = weeks[k]
            row["actual_FVC"] = fvc[k]
            npdata = pd.concat([npdata, pd.DataFrame([row])], ignore_index=True)
    npdata = npdata.fillna(0)
    return npdata




## === cell 3
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ds_3d_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, kernels_per_layer):
        super().__init__()
        self.depthwise = nn.Conv3d(
            nin,
            nin * kernels_per_layer,
            kernel_size=kernel_size,
            padding=padding,
            groups=nin,
        )
        self.pointwise = nn.Conv3d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class SIGMA(nn.Module):
    def __init__(self):
        super().__init__()
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
        return self.data_net4(out4)


class IMAGE(nn.Module):
    def __init__(
        self, channel_number=[32, 64, 128, 256, 256, 64], output_dim=16, dropout=True
    ):
        super().__init__()
        self.feature_extractor = nn.Sequential()
        for i, out_ch in enumerate(channel_number):
            in_ch = 1 if i == 0 else channel_number[i - 1]
            self.feature_extractor.add_module(
                f"conv_{i}",
                self.conv_layer(
                    in_ch,
                    out_ch,
                    maxpool=(i < len(channel_number) - 1),
                    kernel_size=3,
                    padding=1,
                    kernels_per_layer=1,
                ),
            )
        self.classifier = nn.Sequential()
        if dropout:
            self.classifier.add_module("dropout", nn.Dropout(0.5))
        self.classifier.add_module(
            "conv_final",
            nn.Conv3d(channel_number[-1], output_dim, kernel_size=1, padding=0),
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
        in_ch,
        out_ch,
        maxpool=True,
        kernel_size=3,
        padding=1,
        kernels_per_layer=1,
        maxpool_stride=2,
    ):
        layers = [
            ds_3d_conv(in_ch, out_ch, kernel_size, padding, kernels_per_layer),
            nn.BatchNorm3d(out_ch),
        ]
        if maxpool:
            layers.append(nn.MaxPool3d(2, stride=maxpool_stride))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, image_i):
        x = self.feature_extractor(image_i)
        x = self.classifier(x)
        return self.flat(x)


class Combined_NET(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = IMAGE()
        self.data = SIGMA()

    def forward(self, image_i, data_i):
        img_o = self.image(image_i)
        return self.data(data_i, img_o)




## === cell 4
C1, C2 = torch.tensor(70.0, dtype=torch.float32), torch.tensor(
    1000.0, dtype=torch.float32
)


def score(y_true, y_pred):
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]
    sigma_clip = torch.max(sigma, C1)
    delta = torch.abs(y_true[:, 0] - fvc_pred)
    delta = torch.min(delta, C2)
    sq2 = torch.sqrt(torch.tensor(2.0))
    metric = (delta / sigma_clip) * sq2 + torch.log(sigma_clip * sq2)
    return metric.mean()


def qloss(y_true, y_pred):
    qs = torch.tensor([0.25, 0.5, 0.75], device=y_pred.device, dtype=torch.float32)
    e = y_true - y_pred
    v = torch.max(qs * e, (qs - 1) * e)
    return v.mean()


def quartile_loss(y_true, y_pred, _lambda=0.65):
    return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)




## === cell 5
def make_eval_data(npEval, model, device="cuda"):
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
    patients = npEval["Patient"].values
    unique_patients = npEval["Patient"].unique()
    img_dir = "../input/osic-pulmonary-fibrosis-progression/test"
    loaded_images = {
        pid: read_image(img_dir, pid, Z=100, Y=200, X=200) for pid in unique_patients
    }
    if torch.cuda.is_available() and device == "cuda":
        model.to("cuda")
    model.eval()
    preds = []
    with torch.no_grad():
        for i, pid in enumerate(patients):
            img = loaded_images[pid]
            img_tensor = (
                torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            )
            feat_tensor = x_features[i].unsqueeze(0)
            if torch.cuda.is_available() and device == "cuda":
                img_tensor = img_tensor.cuda()
                feat_tensor = feat_tensor.cuda()
            pred = model(img_tensor, feat_tensor)
            preds.append(pred.cpu().numpy()[0])
    preds = np.array(preds)
    npEval["FVC"] = preds[:, 1]
    npEval["Confidence"] = preds[:, 2] - preds[:, 0]
    return npEval




## === cell 6
data_train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
data_test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
submission = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)




## === cell 7
submission["Patient"] = submission["Patient_Week"].apply(lambda x: x.split("_")[0])
submission["Weeks"] = submission["Patient_Week"].apply(lambda x: int(x.split("_")[1]))
submission = submission.sort_values(["Patient", "Weeks"]).reset_index(drop=True)

merge = pd.merge(data_test, submission, on="Patient", how="left")
merge = merge.drop(columns=["FVC_y"])
merge = merge.rename(
    columns={"FVC_x": "base_FVC", "Weeks_y": "Week", "Weeks_x": "base_Weeks"}
)
data_test = merge[
    [
        "Patient",
        "base_Weeks",
        "base_FVC",
        "Percent",
        "Age",
        "Sex",
        "SmokingStatus",
        "Week",
    ]
].copy()
submission = merge[["Patient_Week", "base_FVC", "Confidence"]].rename(
    columns={"base_FVC": "FVC"}
)




## === cell 8
data = data_test.copy()
data["Healthy-FVC"] = round((data["base_FVC"] * 100) / data["Percent"])
fe = ["Healthy-FVC"]
for col in ["Sex", "SmokingStatus"]:
    for mod in data[col].unique():
        fe.append(mod)
        data[mod] = (data[col] == mod).astype(int)
fe1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]
npData = pd.DataFrame(
    columns=["Patient", "base_Weeks", "base_FVC", "Age", "Healthy-FVC"] + fe1 + ["Week"]
)
npData = pd.concat([npData, data], ignore_index=True)
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
def compute_patient_slopes(df):
    slopes = {}
    for pid, grp in df.groupby("Patient"):
        if len(grp) > 1:
            weeks = grp["Weeks"].values
            fvc = grp["FVC"].values
            slope = np.polyfit(weeks, fvc, 1)[0]
            slopes[pid] = slope
    return slopes


def compute_patient_sigmas(df):
    """Standard deviation of residuals from linear fit per patient."""
    sigmas = {}
    for pid, grp in df.groupby("Patient"):
        if len(grp) > 1:
            weeks = grp["Weeks"].values
            fvc = grp["FVC"].values
            a, b = np.polyfit(weeks, fvc, 1)
            preds = a * weeks + b
            residuals = fvc - preds
            sigma = np.std(residuals)
            sigmas[pid] = max(sigma, 70.0)
    return sigmas


patient_slopes = compute_patient_slopes(data_train)
global_slope = np.mean(list(patient_slopes.values())) if patient_slopes else 0.0

patient_sigmas = compute_patient_sigmas(data_train)
global_sigma = np.median(list(patient_sigmas.values())) if patient_sigmas else 70.0


def heuristic_predict(df):
    preds_fvc = []
    preds_conf = []
    for _, row in df.iterrows():
        pid = row["Patient"]
        base_week = row["base_Weeks"]
        base_fvc = row["base_FVC"]
        target_week = row["Week"]
        slope = patient_slopes.get(pid, global_slope)
        fvc_pred = base_fvc + slope * (target_week - base_week)
        fvc_pred = max(fvc_pred, 0)
        preds_fvc.append(fvc_pred)
        conf = patient_sigmas.get(pid, global_sigma)
        preds_conf.append(conf)
    df = df.copy()
    df["FVC"] = preds_fvc
    df["Confidence"] = preds_conf
    return df


model = Combined_NET()
weight_path = (
    "../input/other681/Epoch4_Score6.818399303638382_Acc0.9305297150911874.pth"
)
if os.path.exists(weight_path):
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    test = make_eval_data(data_test.copy(), model)
else:
    print("Warning: pretrained weights not found – using heuristic predictions.")
    test = heuristic_predict(data_test.copy())




## === cell 10
for nid in test["Patient"].unique():
    mask = (test["Patient"] == nid) & (test["Week"] == test["base_Weeks"])
    idx = test[mask].index
    if not idx.empty:
        i = idx[0]
        test.at[i, "FVC"] = test.at[i, "base_FVC"]
        test.at[i, "Confidence"] = 70




## === cell 11
test.loc[test["Confidence"] < 70, "Confidence"] = 70




## === cell 12
submission.loc[:, "FVC"] = test["FVC"].values
submission.loc[:, "Confidence"] = test["Confidence"].values
submission.to_csv("submission.csv", index=False)
