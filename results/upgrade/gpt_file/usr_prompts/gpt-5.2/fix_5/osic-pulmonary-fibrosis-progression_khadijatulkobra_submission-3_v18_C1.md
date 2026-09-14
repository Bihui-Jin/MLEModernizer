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

-6.970834191744311

# 6. Current score

-8.3127

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -24.6542) has done: 'I fix the pandas `.append` deprecation by replacing it with `pd.concat`, which unblocks the test feature dataframe creation. Next, I remove the dependency on a missing external `.pth` file by keeping the same model architecture but running it with randomly initialized weights (so the pipeline runs end-to-end and still produces a valid submission). I also fix the incorrect CT directory path (`../input/...`) to the provided dataset path so DICOM loading works, and add safe fallbacks so a submission is always produced even if image loading fails. Finally, I ensure confidence is valid (≥70) and that predictions are aligned back to `sample_submission.csv` rows before writing `submission.csv`.'
- What this solution (achieved -8.3127) has done: 'The timeout is dominated by repeatedly decoding and resampling hundreds of DICOM slices per patient and by building huge in-RAM 3D tensors (float32) for every measurement row, which explodes memory and slows training. I keep the same CT preprocessing (HU conversion → resize to Z=100,Y=200,X=200 → normalize → uint8) and the same model/loss/training loops, but I (1) make caching deterministic and much faster by saving/loading compressed `.npz` memmaps and using `mmap_mode='r'`, (2) avoid duplicating CT volumes per row by using a Dataset that maps each row to a patient volume on-the-fly (zero-copy from memmap) and (3) speed up CSV submission parsing and small pandas ops via vectorization. These changes preserve identical inputs to the model (same resized/normalized voxels and same tabular features) and therefore preserve evaluation semantics, while drastically reducing I/O and RAM overhead to fit under 600 seconds.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import pydicom  # this one is to read the dicom files
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage import measure, morphology
from sklearn.preprocessing import normalize

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, Dataset

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))




## === cell 1
DATA_ROOT = "../data/osic-pulmonary-fibrosis-progression"
TRAIN_FOLDER = os.path.join(DATA_ROOT, "train")
TEST_FOLDER = os.path.join(DATA_ROOT, "test")

CACHE_ROOT = os.path.join(DATA_ROOT, "_ct_cache_z100_y200_x200_uint8")
os.makedirs(CACHE_ROOT, exist_ok=True)


def _dcmread_fast(fp):
    return pydicom.dcmread(
        fp,
        stop_before_pixels=False,
        specific_tags=[
            "PixelData",
            "RescaleIntercept",
            "RescaleSlope",
            "ImagePositionPatient",
            "InstanceNumber",
            "Rows",
            "Columns",
        ],
    )


def load_scan(path):  # path == (.../train/patientId)
    files = os.listdir(path)
    slices = []
    for s in files:
        fp = path + os.sep + s
        try:
            ds = _dcmread_fast(fp)
            slices.append(ds)
        except Exception:
            continue

    def _sort_key(ds):
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) >= 3:
            try:
                return float(ipp[2])
            except Exception:
                pass
        inst = getattr(ds, "InstanceNumber", None)
        if inst is not None:
            try:
                return float(inst)
            except Exception:
                pass
        return 0.0

    slices.sort(key=_sort_key)
    return slices


def make_dict_with_slices(patientIDs):  # huge memory consuming . don't use it
    patient_slices_dict = {}
    failed_slices_patiendIDs = []
    for patientID in patientIDs:
        path = TRAIN_FOLDER + os.sep + patientID
        try:
            patient_slices_dict[patientID] = load_scan(path)
        except Exception:
            failed_slices_patiendIDs.append(patientID)
    return patient_slices_dict, failed_slices_patiendIDs


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices]).astype(np.int16, copy=False)
    try:
        image[image <= -2000] = 0
        for slice_number in range(len(slices)):
            intercept = float(getattr(slices[slice_number], "RescaleIntercept", 0.0))
            slope = float(getattr(slices[slice_number], "RescaleSlope", 1.0))
            if slope != 1:
                image[slice_number] = (
                    slope * image[slice_number].astype(np.float32)
                ).astype(np.int16, copy=False)
            image[slice_number] = image[slice_number] + np.int16(intercept)
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
    cache_fp = os.path.join(CACHE_ROOT, f"{patientid}_z{Z}_y{Y}_x{X}.npy")
    if os.path.exists(cache_fp):
        return np.load(cache_fp, mmap_mode="r")

    path = dir_name + os.sep + patientid
    slices = load_scan(path)
    try:
        image_array = get_pixels_hu(slices)
        ctimage_resizedAll = resize_along_allaxis(
            image_array, target_dimensionX=X, target_dimensionY=Y, target_dimensionZ=Z
        )
        image = (image_normalize(ctimage_resizedAll) * 255.0).astype("uint8")
        np.save(cache_fp, image)
        return np.load(cache_fp, mmap_mode="r")
    except Exception:
        print("PatientId:%s couldnt be converted" % (patientid))
        return None




## === cell 2
def csv_preprocess(data):
    data = data.copy()
    data["Healthy-FVC"] = np.round((data["FVC"] * 100) / data["Percent"]).astype(
        np.float32
    )

    COLS = ["Sex", "SmokingStatus"]
    for col in COLS:
        for mod in data[col].unique():
            data[mod] = (data[col] == mod).astype(np.int8)

    data = data[
        ["Patient", "Weeks", "FVC", "Age", "Healthy-FVC"]
        + list(data["Sex"].unique())
        + list(data["SmokingStatus"].unique())
    ]
    data = data.sort_values(["Patient", "Weeks"], ascending=True).reset_index(drop=True)

    FE1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]
    rename_col = {"Weeks": "base_Weeks", "FVC": "base_FVC"}
    data = data.rename(columns=rename_col)

    for c in FE1:
        if c not in data.columns:
            data[c] = 0

    base = data.groupby("Patient", sort=False).first().reset_index()
    meas = data[["Patient", "base_Weeks", "base_FVC"]].copy()
    meas = meas.rename(columns={"base_Weeks": "Week", "base_FVC": "actual_FVC"})

    npData = meas.merge(
        base[["Patient", "base_Weeks", "base_FVC", "Age", "Healthy-FVC"] + FE1],
        on="Patient",
        how="left",
        sort=False,
    )

    npData = npData[
        ["Patient", "base_Weeks", "base_FVC", "Age"]
        + FE1
        + ["Week", "Healthy-FVC", "actual_FVC"]
    ]
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
C1, C2 = 70.0, 1000.0


def score(y_true, y_pred):
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]
    sigma_clip = torch.clamp(sigma, min=C1)
    delta = (y_true[:, 0] - fvc_pred).abs()
    delta = torch.clamp(delta, max=C2)
    sq2 = (y_pred.new_tensor(2.0)).sqrt()
    metric = (delta / sigma_clip) * sq2 + (sigma_clip * sq2).log()
    return (metric).mean()


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
from concurrent.futures import ThreadPoolExecutor, as_completed


def build_patient_image_cache(
    patient_ids, folder, Z=100, Y=200, X=200, max_workers=None
):
    cache = {}

    def _load_one(pid):
        img = read_image(folder, pid, Z=Z, Y=Y, X=X)
        if (
            img is None
            or isinstance(img, list)
            or (hasattr(img, "size") and img.size == 0)
        ):
            img = np.zeros((Z, Y, X), dtype=np.uint8)
        return pid, img

    if max_workers is None:
        max_workers = min(8, (os.cpu_count() or 2))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_load_one, pid) for pid in patient_ids]
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Loading CT from {os.path.basename(folder)}",
        ):
            pid, img = fut.result()
            cache[pid] = img
    return cache


class PatientRowDataset(Dataset):
    def __init__(self, df, image_cache, with_target=True):
        self.df = df.reset_index(drop=True)
        self.image_cache = image_cache
        self.with_target = with_target

        self.x_features = self.df[
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
        ].values.astype(np.float32, copy=False)

        self.patient_ids = self.df["Patient"].values
        self.Z, self.Y, self.X = next(iter(image_cache.values())).shape

        if with_target:
            self.y = self.df[["actual_FVC"]].values.astype(np.float32, copy=False)
        else:
            self.y = None

        self._last_pid = None
        self._last_img = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        if pid == self._last_pid:
            img = self._last_img
        else:
            img = self.image_cache.get(pid, None)
            if img is None or (hasattr(img, "size") and img.size == 0):
                img = np.zeros((self.Z, self.Y, self.X), dtype=np.uint8)
            self._last_pid = pid
            self._last_img = img

        x_img = torch.from_numpy(
            np.asarray(img, dtype=np.float32, order="C")
        ).unsqueeze(0)
        x_feat = torch.from_numpy(self.x_features[idx])

        if self.with_target:
            y = torch.from_numpy(self.y[idx])
            return x_img, x_feat, y
        else:
            return x_img, x_feat


def train_model(model, train_loader, valid_loader, device, epochs=2, lr=1e-4):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        n = 0
        for x_img, x_feat, y in tqdm(
            train_loader, desc=f"Train epoch {ep+1}/{epochs}", leave=False
        ):
            x_img = x_img.to(device, non_blocking=True)
            x_feat = x_feat.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            pred = model(x_img, x_feat)
            loss = quartile_loss(y, pred)
            loss.backward()
            opt.step()

            bs = y.size(0)
            tr_loss += loss.item() * bs
            n += bs

        model.eval()
        va_loss = 0.0
        vn = 0
        with torch.no_grad():
            for x_img, x_feat, y in tqdm(
                valid_loader, desc=f"Valid epoch {ep+1}/{epochs}", leave=False
            ):
                x_img = x_img.to(device, non_blocking=True)
                x_feat = x_feat.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                pred = model(x_img, x_feat)
                loss = quartile_loss(y, pred)
                bs = y.size(0)
                va_loss += loss.item() * bs
                vn += bs
        print(
            f"Epoch {ep+1}/{epochs} - train_loss={tr_loss/max(n,1):.4f} valid_loss={va_loss/max(vn,1):.4f}"
        )
    return model


def make_eval_data(npEval, model, image_cache, device="cuda", batch_size=16):
    ds = PatientRowDataset(npEval, image_cache, with_target=False)
    use_cuda = torch.cuda.is_available() and device == "cuda"
    dev = "cuda" if use_cuda else "cpu"
    model.to(dev)
    model.eval()

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
    )

    preds = []
    with torch.no_grad():
        for xb_img, xb_feat in loader:
            xb_img = xb_img.to(dev, non_blocking=True)
            xb_feat = xb_feat.to(dev, non_blocking=True)
            pb = model(xb_img, xb_feat).to("cpu").numpy()
            preds.append(pb)
    predictions = np.concatenate(preds, axis=0)

    npEval = npEval.copy()
    npEval["FVC"] = predictions[:, 1]
    sigma = np.abs(predictions[:, 2] - predictions[:, 0]).astype(np.float32)
    sigma = np.maximum(sigma, 70.0)
    npEval["Confidence"] = sigma
    return npEval




## === cell 6
data_train = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))
data_test = pd.read_csv(os.path.join(DATA_ROOT, "test.csv"))
sample_sub = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))




## === cell 7
train_np = csv_preprocess(data_train.copy())

patients = train_np["Patient"].unique()
rng = np.random.RandomState(42)
rng.shuffle(patients)
split = int(0.8 * len(patients))
tr_patients = set(patients[:split])
va_patients = set(patients[split:])

train_df = train_np[train_np["Patient"].isin(tr_patients)].reset_index(drop=True)
valid_df = train_np[train_np["Patient"].isin(va_patients)].reset_index(drop=True)

train_image_cache = build_patient_image_cache(
    sorted(train_df["Patient"].unique()), TRAIN_FOLDER, Z=100, Y=200, X=200
)
valid_image_cache = build_patient_image_cache(
    sorted(valid_df["Patient"].unique()), TRAIN_FOLDER, Z=100, Y=200, X=200
)

use_cuda = torch.cuda.is_available()

train_ds = PatientRowDataset(train_df, train_image_cache, with_target=True)
valid_ds = PatientRowDataset(valid_df, valid_image_cache, with_target=True)

num_workers = min(2, os.cpu_count() or 1)
train_loader = DataLoader(
    train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=use_cuda,
    persistent_workers=bool(num_workers > 0),
)
valid_loader = DataLoader(
    valid_ds,
    batch_size=2,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=use_cuda,
    persistent_workers=bool(num_workers > 0),
)




## === cell 8
sub_expanded = sample_sub.copy()
tmp = sub_expanded["Patient_Week"].str.split("_", n=1, expand=True)
sub_expanded["Patient"] = tmp[0]
sub_expanded["Weeks"] = tmp[1].astype(int)
sub_expanded = sub_expanded.sort_values(
    by=["Patient", "Weeks"], ascending=True
).reset_index(drop=True)

merge = (
    pd.merge(data_test, sub_expanded, on=["Patient"], how="left")
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




## === cell 9
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
npData = pd.concat([npData, data], ignore_index=True, sort=True)
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




## === cell 10
model = Combined_NET()
device = "cuda" if torch.cuda.is_available() else "cpu"

model = train_model(model, train_loader, valid_loader, device=device, epochs=2, lr=1e-4)

test_image_cache = build_patient_image_cache(
    sorted(data_test["Patient"].unique()), TEST_FOLDER, Z=100, Y=200, X=200
)
test_pred = make_eval_data(
    data_test.copy(), model, test_image_cache, device=device, batch_size=16
)




## === cell 11
for nid in test_pred.Patient.unique():
    idx = test_pred[
        (test_pred.Patient == nid) & (test_pred.Week == test_pred.base_Weeks)
    ].index.values
    if len(idx) > 0:
        test_pred.iloc[idx[0], test_pred.columns.get_loc("FVC")] = test_pred.iloc[
            idx[0], test_pred.columns.get_loc("base_FVC")
        ]
        test_pred.iloc[idx[0], test_pred.columns.get_loc("Confidence")] = 70

test_pred.loc[test_pred.Confidence < 70, "Confidence"] = 70

submission.loc[:, "FVC"] = test_pred["FVC"].values
submission.loc[:, "Confidence"] = test_pred["Confidence"].values

submission = submission[["Patient_Week", "FVC", "Confidence"]]
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
