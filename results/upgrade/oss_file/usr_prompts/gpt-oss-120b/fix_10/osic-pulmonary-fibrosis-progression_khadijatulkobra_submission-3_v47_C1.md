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

-6.934720565384507

# 6. Current score

-9.07915

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -10.81761) has done: 'I add a `Patient_Week` column to the test feature table before merging so that the join on `Patient_Week` succeeds, and then the final CSV contain the required `Patient_Week`, `FVC`, `Confidence` columns.'
- What this solution (achieved -10.81761) has done: 'I add a simple per‑patient linear trend derived from the training data to predict FVC instead of just copying the baseline value. The trend (slope + intercept) is computed once after loading the training set and stored in a global dictionary. The `make_eval_data` function now uses this dictionary: if a patient’s trend is known it predicts FVC as slope × Week + intercept, otherwise it falls back to the baseline FVC. Confidence remains the required 70. This small, targeted change should raise the score toward the target without altering the core model architecture.'
- What this solution (achieved -11.38566) has done: 'I add a simple global week‑adjustment based on the average progression observed in the training set and use it when a patient‑specific trend is unavailable. This modest change keeps the existing per‑patient linear model while providing a better fallback prediction, which should raise the score toward the target. I also compute the week‑delta after building `patient_trends` so it is available to the prediction function.'
- What this solution (achieved -14.97987) has done: 'I add a simple global linear regression that uses age, week, sex and smoking one‑hot features to predict FVC when a patient‑specific trend is unavailable. This keeps the existing per‑patient trend logic unchanged, only enriching the fallback prediction, and retains the required 70 confidence. The changes are limited to the data‑preparation cell and the `make_eval_data` function.'
- What this solution (achieved -11.16437) has done: 'I add a lightweight per‑patient confidence estimate based on the residual standard deviation of the linear trend (or the global regression when a patient‑specific trend is missing). This keeps the original trend‑based FVC predictions but replaces the fixed confidence 70 with a data‑driven value that is still clipped at 70, which should raise the Laplace‑Log‑Likelihood toward the target without altering the core model logic.'
- What this solution (achieved -9.07915) has done: 'I adjust the fallback prediction in `make_eval_data` to use the baseline FVC + average weekly change (`week_delta`) instead of the generic global regression. This leverages the observed overall progression pattern and should raise the Laplace‑Log‑Likelihood toward the target while keeping the core modeling unchanged. The confidence handling remains the same.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pydicom
import scipy.ndimage




## === cell 1
def load_scan(path):
    try:
        slices = [pydicom.dcmread(os.path.join(path, s)) for s in os.listdir(path)]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except Exception:
        files = sorted(os.listdir(path))
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
    return image.astype(np.int16)


def resize_along_allaxis(
    slices, target_dimensionZ=30, target_dimensionY=100, target_dimensionX=100
):
    dz, dy, dx = slices.shape
    if (dz, dy, dx) == (target_dimensionZ, target_dimensionY, target_dimensionX):
        return slices
    zoom_factors = (
        target_dimensionZ / dz,
        target_dimensionY / dy,
        target_dimensionX / dx,
    )
    return scipy.ndimage.zoom(slices, zoom_factors, mode="nearest")


MIN_BOUND, MAX_BOUND = -1000.0, 400.0


def image_normalize(image):
    img = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    img = np.clip(img, 0, 1)
    return img


def read_image(dir_name, patientid, Z=100, Y=200, X=200):
    path = os.path.join(dir_name, patientid)
    slices = load_scan(path)
    try:
        img = get_pixels_hu(slices)
        img = resize_along_allaxis(img, Z, Y, X)
        img = (image_normalize(img) * 255).astype("uint8")
        return img
    except Exception:
        print(f"PatientId:{patientid} couldn't be converted")
        return np.zeros((Z, Y, X), dtype="uint8")




## === cell 2
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ds_3d_conv(nn.Module):
    def __init__(self, nin, nout, kernel, pad, kpl):
        super().__init__()
        self.depthwise = nn.Conv3d(nin, nin * kpl, kernel, padding=pad, groups=nin)
        self.pointwise = nn.Conv3d(nin * kpl, nout, kernel_size=1)

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
        return self.data_net4(out4)


class IMAGE(nn.Module):
    def __init__(self, channels=[32, 64, 128, 256, 256, 64], out_dim=16):
        super().__init__()
        layers = []
        for i, out_c in enumerate(channels):
            in_c = 1 if i == 0 else channels[i - 1]
            last = i == len(channels) - 1
            layers.append(
                self.conv_layer(
                    in_c,
                    out_c,
                    maxpool=not last,
                    kernel_size=3 if not last else 1,
                    padding=1 if not last else 0,
                )
            )
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Sequential()
        self.classifier.add_module("dropout", nn.Dropout(0.5))
        self.classifier.add_module(
            "conv_last", nn.Conv3d(channels[-1], out_dim, kernel_size=1)
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
        in_c,
        out_c,
        maxpool=True,
        kernel_size=3,
        padding=1,
        kernels_per_layer=1,
        maxpool_stride=2,
    ):
        seq = [
            ds_3d_conv(in_c, out_c, kernel_size, padding, kernels_per_layer),
            nn.BatchNorm3d(out_c),
        ]
        if maxpool:
            seq.append(nn.MaxPool3d(2, stride=maxpool_stride))
        seq.append(nn.ReLU())
        return nn.Sequential(*seq)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return self.flat(x)


class Combined_NET(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = IMAGE()
        self.data = SIGMA()

    def forward(self, img, data):
        img_o = self.image(img)
        return self.data(data, img_o)




## === cell 3
C1 = torch.tensor(70.0, dtype=torch.float32)
C2 = torch.tensor(1000.0, dtype=torch.float32)


def score(y_true, y_pred):
    sigma = y_pred[:, 0]
    fvc_pred = y_pred[:, 1]
    sigma_clip = torch.max(sigma, C1)
    delta = torch.abs(y_true[:, 0] - fvc_pred)
    delta = torch.min(delta, C2)
    sq2 = torch.tensor(2.0).sqrt()
    metric = (delta / sigma_clip) * sq2 + torch.log(sigma_clip * sq2)
    return metric.mean()


def quartile_loss(y_true, y_pred):
    return score(y_true, y_pred)




## === cell 4
def make_eval_data(npEval):
    """Predict FVC using per‑patient linear trends when available;
    otherwise fall back to a simple baseline+average weekly change.
    Confidence is set per‑patient based on residual variance, clipped at 70."""
    out = npEval.copy()

    def predict_row(row):
        pid = row["Patient"]
        week = row["Week"]
        if pid in patient_trends:
            slope, intercept = patient_trends[pid]
            fvc_pred = float(slope * week + intercept)
        else:
            if "base_FVC" in row and week in week_delta:
                fvc_pred = float(row["base_FVC"] + week_delta[week])
            else:
                feats = np.array(
                    [
                        1.0,  # bias
                        week,
                        row["Age"],
                        row.get("Male", 0),
                        row.get("Female", 0),
                        row.get("Ex-smoker", 0),
                        row.get("Never smoked", 0),
                        row.get("Currently smokes", 0),
                        row.get("Healthy-FVC", 0.0),
                    ]
                )
                fvc_pred = float(feats @ global_reg_coef)

        sigma = patient_sigma.get(pid, global_sigma)
        sigma = max(sigma, 70.0)  # enforce competition minimum
        return pd.Series({"FVC": fvc_pred, "Confidence": sigma})

    out[["FVC", "Confidence"]] = out.apply(predict_row, axis=1)
    return out




## === cell 5
data_train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
data_test_raw = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
submission = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)

patient_trends = {}
patient_residuals = {}  # store residuals for sigma calculation
global_residuals = []

for pid, grp in data_train.groupby("Patient"):
    if len(grp) > 1:
        weeks = grp["Weeks"].values.astype(np.float64)
        fvc = grp["FVC"].values.astype(np.float64)
        try:
            slope, intercept = np.polyfit(weeks, fvc, 1)
        except Exception:
            slope, intercept = 0.0, float(grp["FVC"].iloc[0])
        preds = slope * weeks + intercept
        resid = fvc - preds
        patient_residuals[pid] = resid
        global_residuals.extend(resid.tolist())
    else:
        slope, intercept = 0.0, float(grp["FVC"].iloc[0])
    patient_trends[pid] = (slope, intercept)

patient_sigma = {}
for pid, resid in patient_residuals.items():
    if len(resid) > 0:
        patient_sigma[pid] = float(np.std(resid, ddof=1))
    else:
        patient_sigma[pid] = 70.0  # fallback

global_sigma = float(np.std(global_residuals, ddof=1)) if global_residuals else 70.0

week_mean_fvc = data_train.groupby("Weeks")["FVC"].mean()
baseline_week0 = week_mean_fvc.get(
    0, data_train[data_train["Weeks"] == 0]["FVC"].mean()
)
week_delta = (week_mean_fvc - baseline_week0).to_dict()


def add_one_hot(df):
    for col in ["Sex", "SmokingStatus"]:
        for mod in df[col].unique():
            df[mod] = (df[col] == mod).astype(int)


add_one_hot(data_train)  # encode training set
add_one_hot(data_test_raw)  # encode test raw set (used later for features)

reg_features = [
    "Weeks",
    "Age",
    "Male",
    "Female",
    "Ex-smoker",
    "Never smoked",
    "Currently smokes",
    "Healthy-FVC",
]

data_train["Healthy-FVC"] = round((data_train["FVC"] * 100) / data_train["Percent"])
X_train = data_train[reg_features].fillna(0).values
y_train = data_train["FVC"].values
X_aug = np.column_stack([np.ones(X_train.shape[0]), X_train])  # bias term
global_reg_coef, _, _, _ = np.linalg.lstsq(X_aug, y_train, rcond=None)

submission["Patient"] = submission["Patient_Week"].apply(lambda x: x.split("_")[0])
submission["Week"] = submission["Patient_Week"].apply(lambda x: int(x.split("_")[1]))

df = submission[["Patient", "Week"]].merge(data_test_raw, on="Patient", how="left")
df = df.rename(columns={"Weeks": "base_Weeks", "FVC": "base_FVC"})

df["Healthy-FVC"] = round((df["base_FVC"] * 100) / df["Percent"])

data_test = df[
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
].fillna(0)

data_test["Patient_Week"] = data_test["Patient"] + "_" + data_test["Week"].astype(str)




## === cell 6
test = make_eval_data(data_test.copy())




## === cell 7
submission = submission.drop(columns=["FVC", "Confidence"], errors="ignore")

submission = submission.merge(
    test[["Patient_Week", "FVC", "Confidence"]],
    on="Patient_Week",
    how="left",
)

submission["FVC"] = submission["FVC"].fillna(test["base_FVC"])
submission["Confidence"] = submission["Confidence"].fillna(70.0)




## === cell 8
submission.to_csv("submission.csv", index=False)
