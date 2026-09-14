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

-7.019832695103727

# 6. Current score

-24.79812

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -19.09388) has done: 'I fix the failure in the linear‑regression prediction step by removing the non‑numeric **Patient** column from the test features so the model receives only numeric inputs. This resolves the `ValueError` and allows the `test` DataFrame to be created, after which the submission file is written correctly.'
- What this solution (achieved -24.79812) has done: 'I keep the overall pipeline but improve the linear‑regression step by (1) scaling the numeric features with StandardScaler to give the model a better‑conditioned input space, and (2) using the minimum allowed confidence 70 instead of a constant 100 so the scoring formula is closer to the competition’s optimum. These minimal changes keep the core logic intact while moving the metric toward the target score.'
- What this solution (achieved -12.58075) has done: 'I keep the overall pipeline unchanged but give the linear‑regression model richer input features and a more realistic confidence estimate. By expanding the numeric features with second‑order polynomial terms (still a linear model) and setting the confidence to the training residual standard deviation (or the required minimum 70 ml, whichever is larger), the predictions become more accurate and the confidence better matches the model’s error, moving the metric toward the target score.'
- What this solution (achieved -12.59138) has done: 'I enhance the linear‑regression pipeline by (1) adding a reconstructed `Percent` feature (derived from `base_FVC` and `Healthy‑FVC`) to both training and test data, and (2) increasing the polynomial expansion degree from 2 to 3 so the model can capture richer interactions while still using the same scaled linear regression approach. These minimal changes keep the core logic intact but should produce predictions that are closer to the target metric, moving the score toward ‑7.02.'
- What this solution (achieved -10.3569) has done: 'I adjust the linear‑regression pipeline to use a regularized Ridge model (which often generalises better than plain LinearRegression) and set a slightly larger, data‑driven confidence value. The confidence is computed as the sum of the residual standard deviation and the mean absolute residual, then clipped at the required minimum of 70 ml. This small change keeps the overall architecture unchanged while should improve the Laplace Log Likelihood score, moving it closer to the target.'
- What this solution (achieved -24.79812) has done: 'I keep the overall pipeline unchanged but improve the score by (1) using a tighter regularisation (Ridge α = 0.1) which usually yields more accurate FVC predictions, and (2) setting the confidence to the minimum allowed value 70 ml (instead of a larger data‑driven estimate). A lower, constant confidence reduces the penalty term in the Laplace Log‑Likelihood while still satisfying the required lower bound, moving the metric closer to the target.'
- What this solution (achieved -12.84137) has done: 'I added the missing imports, guarded the deep‑learning parts so they’re skipped if torch is unavailable, and fixed the data‑loading and preprocessing steps so the ridge‑regression pipeline runs without errors. The script now creates a valid submission.csv containing the required columns.'
- What this solution (achieved -24.79812) has done: 'I keep the overall pipeline unchanged but set the confidence used for all predictions to the minimum allowed value 70 instead of the data‑driven standard‑deviation estimate, which reduces the penalty term in the Laplace Log Likelihood and moves the score closer to the target. The only modification is in cell 7 where the confidence value is defined.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import scipy.ndimage

try:
    import pydicom
except Exception:
    pydicom = None


def load_scan(path):
    try:
        if pydicom is None:
            raise ImportError
        slices = [pydicom.dcmread(os.path.join(path, s)) for s in os.listdir(path)]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except Exception:
        files = sorted(os.listdir(path))
        slices = (
            [pydicom.dcmread(os.path.join(path, s)) for s in files] if pydicom else []
        )
    return slices


def get_pixels_hu(slices):
    if not slices:
        return np.zeros((0, 0, 0), dtype=np.int16)
    image = np.stack([s.pixel_array for s in slices]).astype(np.int16)
    try:
        image[image <= -2000] = 0
        for i, s in enumerate(slices):
            intercept = s.RescaleIntercept
            slope = s.RescaleSlope
            if slope != 1:
                image[i] = slope * image[i].astype(np.float64)
                image[i] = image[i].astype(np.int16)
            image[i] += np.int16(intercept)
    except Exception:
        print("HU conversion Failed!!")
    return np.array(image, dtype=np.int16)


def resize_along_allaxis(
    slices, target_dimensionZ=30, target_dimensionY=100, target_dimensionX=100
):
    dz, dy, dx = slices.shape
    if (target_dimensionZ, target_dimensionY, target_dimensionX) == (dz, dy, dx):
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
    img = np.clip(img, 0.0, 1.0)
    return img


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




## === cell 1
def csv_preprocess(data):
    data["Healthy-FVC"] = round((data["FVC"] * 100) / data["Percent"])
    FE = ["Healthy-FVC"]
    COLS = ["Sex", "SmokingStatus"]
    for col in COLS:
        for mod in data[col].unique():
            FE.append(mod)
            data[mod] = (data[col] == mod).astype(int)
    data = data[["Patient", "Weeks", "FVC", "Age"] + FE]
    data = data.sort_values(["Patient", "Weeks"]).reset_index(drop=True)
    data = data.rename(columns={"Weeks": "base_Weeks", "FVC": "base_FVC"})
    FE1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]
    npData = pd.DataFrame(
        columns=["Patient", "base_Weeks", "base_FVC", "Age"]
        + FE1
        + ["Week", "Healthy-FVC", "actual_FVC"]
    )
    for pid in data["Patient"].unique():
        weeks = data.loc[data["Patient"] == pid].base_Weeks.reset_index(drop=True)
        fvc = data.loc[data["Patient"] == pid].base_FVC.reset_index(drop=True)
        idx = data.loc[data["Patient"] == pid].index
        for k in range(len(weeks)):
            npData = pd.concat([npData, data.loc[[idx[0]]]], ignore_index=True)
            npData.loc[npData.index[-1], "Week"] = weeks[k]
            npData.loc[npData.index[-1], "actual_FVC"] = fvc[k]
    npData = npData.fillna(0)
    npData["Week"] = npData["Week"] - npData["base_Weeks"]
    npData["base_Weeks"] = 0.0
    return npData




## === cell 2
try:
    import torch
    import torch.nn as nn

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
            return self.pointwise(self.depthwise(x))

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
            return self.data_net4(out4)

    class IMAGE(nn.Module):
        def __init__(
            self,
            channel_number=[32, 64, 128, 256, 256, 64],
            output_dim=16,
            dropout=True,
        ):
            super(IMAGE, self).__init__()
            self.feature_extractor = nn.Sequential()
            for i, out_ch in enumerate(channel_number):
                in_ch = 1 if i == 0 else channel_number[i - 1]
                self.feature_extractor.add_module(
                    f"conv_{i}",
                    self.conv_layer(
                        in_ch,
                        out_ch,
                        maxpool=i < len(channel_number) - 1,
                        kernel_size=3 if i < len(channel_number) - 1 else 1,
                        padding=1 if i < len(channel_number) - 1 else 0,
                        kernels_per_layer=1,
                    ),
                )
            self.classifier = nn.Sequential()
            if dropout:
                self.classifier.add_module("dropout", nn.Dropout(0.5))
            self.classifier.add_module(
                "conv_final", nn.Conv3d(channel_number[-1], output_dim, kernel_size=1)
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
            layers = [
                ds_3d_conv(
                    in_channel, out_channel, kernel_size, padding, kernels_per_layer
                ),
                nn.BatchNorm3d(out_channel),
            ]
            if maxpool:
                layers.append(nn.MaxPool3d(2, stride=maxpool_stride))
            layers.append(nn.ReLU())
            return nn.Sequential(*layers)

        def forward(self, image_i):
            o = self.feature_extractor(image_i)
            o = self.classifier(o)
            return self.flat(o)

    class Combined_NET(nn.Module):
        def __init__(self):
            super(Combined_NET, self).__init__()
            self.image = IMAGE()
            self.data = SIGMA()

        def forward(self, image_i, data_i):
            img_o = self.image(image_i)
            return self.data(data_i, img_o)

except Exception:
    pass




## === cell 3
try:
    import torch

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
        metric = (delta / sigma_clip) * sq2 + (sigma_clip * sq2).log()
        return metric.mean()

    def qloss(y_true, y_pred):
        qs = torch.tensor([0.25, 0.5, 0.75], device=y_pred.device, dtype=torch.float32)
        e = y_true - y_pred
        v = torch.max(qs * e, (qs - 1) * e)
        return v.mean()

    def quartile_loss(y_true, y_pred, _lambda=0.65):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)

except Exception:
    pass




## === cell 4
data_train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
data_test_raw = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
submission = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)




## === cell 5
submission["Patient"] = submission["Patient_Week"].apply(lambda x: x.split("_")[0])
submission["Weeks"] = submission["Patient_Week"].apply(lambda x: int(x.split("_")[1]))
submission = submission.sort_values(["Patient", "Weeks"]).reset_index(drop=True)

merge = pd.merge(data_test_raw, submission, on="Patient", how="left")
merge = merge.drop(columns=["FVC_y"]).rename(
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




## === cell 6
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
    columns=[
        "Patient",
        "base_Weeks",
        "base_FVC",
        "Age",
        "Healthy-FVC",
    ]
    + FE1
    + ["Week"]
)
npData = pd.concat([npData, data], ignore_index=True)
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
]

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

train_pre = csv_preprocess(data_train)

train_pre["Percent"] = np.where(
    train_pre["Healthy-FVC"] != 0,
    (train_pre["base_FVC"] * 100) / train_pre["Healthy-FVC"],
    0.0,
)

data_test["Percent"] = np.where(
    data_test["Healthy-FVC"] != 0,
    (data_test["base_FVC"] * 100) / data_test["Healthy-FVC"],
    0.0,
)

X_train = train_pre.drop(columns=["Patient", "actual_FVC"]).values
y_train = train_pre["actual_FVC"].values

poly = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly = poly.fit_transform(X_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)

ridge = Ridge(alpha=0.1, random_state=42)
ridge.fit(X_train_scaled, y_train)

train_residuals = y_train - ridge.predict(X_train_scaled)
std_res = np.std(train_residuals)

confidence_val = 70.0

X_test = data_test.drop(columns=["Patient"]).values
X_test_poly = poly.transform(X_test)
X_test_scaled = scaler.transform(X_test_poly)

pred_fvc = ridge.predict(X_test_scaled)

test = pd.DataFrame(
    {
        "FVC": pred_fvc,
        "Confidence": np.full_like(pred_fvc, confidence_val),
    }
)




## === cell 7
submission.loc[:, "FVC"] = test["FVC"]
submission.loc[:, "Confidence"] = test["Confidence"]
submission.to_csv("submission.csv", index=False)
