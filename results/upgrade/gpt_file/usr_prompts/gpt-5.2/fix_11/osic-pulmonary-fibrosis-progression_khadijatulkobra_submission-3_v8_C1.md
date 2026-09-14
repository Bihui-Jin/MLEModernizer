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

-7.019832695103727

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import scipy.ndimage
import pydicom
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

try:
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
except Exception:
    pass




## === cell 1
def find_osic_root():
    candidates = [
        "/kaggle/input/osic-pulmonary-fibrosis-progression",
        "/kaggle/data/osic-pulmonary-fibrosis-progression",
        "../input/osic-pulmonary-fibrosis-progression",
        "../data/osic-pulmonary-fibrosis-progression",
        "../input",
        "../data",
    ]
    for c in candidates:
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "train.csv")):
            return c
    base = "/kaggle/input"
    if os.path.isdir(base):
        for name in os.listdir(base):
            cand = os.path.join(base, name)
            if (
                os.path.isdir(cand)
                and os.path.exists(os.path.join(cand, "train.csv"))
                and os.path.exists(os.path.join(cand, "sample_submission.csv"))
            ):
                return cand
    raise FileNotFoundError(
        "Could not locate OSIC dataset root containing train.csv/test.csv/sample_submission.csv"
    )


OSIC_ROOT = find_osic_root()
TRAIN_FOLDER = os.path.join(OSIC_ROOT, "train")
TEST_FOLDER = os.path.join(OSIC_ROOT, "test")

print("OSIC_ROOT:", OSIC_ROOT)
print("TRAIN_FOLDER:", TRAIN_FOLDER)
print("TEST_FOLDER:", TEST_FOLDER)




## === cell 2
def load_scan(path):
    try:
        files = [f for f in os.listdir(path) if f.lower().endswith(".dcm")]
    except Exception:
        files = os.listdir(path)
    files.sort()
    if not files:
        return []

    metas = []
    for f in files:
        fp = os.path.join(path, f)
        try:
            ds_meta = pydicom.dcmread(
                fp,
                stop_before_pixels=True,
                force=True,
                specific_tags=[
                    "ImagePositionPatient",
                    "InstanceNumber",
                    "RescaleIntercept",
                    "RescaleSlope",
                ],
            )
            z = None
            try:
                z = float(ds_meta.ImagePositionPatient[2])
            except Exception:
                z = None
            inst = None
            try:
                inst = int(getattr(ds_meta, "InstanceNumber", 0))
            except Exception:
                inst = 0
            metas.append((z, inst, fp))
        except Exception:
            continue

    if not metas:
        return []

    if all(m[0] is not None for m in metas):
        metas.sort(key=lambda t: t[0])
    else:
        metas.sort(key=lambda t: t[1])

    slices = []
    for _, __, fp in metas:
        try:
            ds = pydicom.dcmread(fp, force=True)
            slices.append(ds)
        except Exception:
            continue
    return slices


def get_pixels_hu(slices):
    if len(slices) == 0:
        return np.zeros((0, 0, 0), dtype=np.int16)

    image = np.stack([s.pixel_array for s in slices]).astype(np.int16, copy=False)
    image[image <= -2000] = 0

    try:
        intercept = np.array(
            [float(getattr(s, "RescaleIntercept", 0.0)) for s in slices],
            dtype=np.float32,
        )
        slope = np.array(
            [float(getattr(s, "RescaleSlope", 1.0)) for s in slices], dtype=np.float32
        )

        img_f = image.astype(np.float32, copy=False)
        img_f *= slope[:, None, None]
        img_f += intercept[:, None, None]
        image = img_f.astype(np.int16, copy=False)
    except Exception:
        print("HU conversion Failed!!")
    return np.asarray(image, dtype=np.int16)


def resize_along_allaxis(
    slices, target_dimensionZ=30, target_dimensionY=100, target_dimensionX=100
):
    present_dimensionZ, present_dimensionY, present_dimensionX = slices.shape
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


def read_image(dir_name, patientid, Z=100, Y=200, X=200, cache_dir=None):
    cache_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{patientid}_{Z}x{Y}x{X}.npy")
        if os.path.exists(cache_path):
            try:
                return np.load(cache_path, mmap_mode="r")
            except Exception:
                pass  # fall back to recompute

    path = os.path.join(dir_name, patientid)
    slices = load_scan(path)
    try:
        image_array = get_pixels_hu(slices)
        ctimage_resizedAll = resize_along_allaxis(
            image_array, target_dimensionX=X, target_dimensionY=Y, target_dimensionZ=Z
        )
        image = (image_normalize(ctimage_resizedAll) * 255.0).astype(
            "uint8", copy=False
        )

        if cache_dir is not None and cache_path is not None:
            try:
                tmp_path = cache_path + f".{os.getpid()}.tmp.npy"
                np.save(tmp_path, np.asarray(image), allow_pickle=False)
                os.replace(tmp_path, cache_path)
            except Exception:
                pass
        return image
    except Exception:
        print(f"PatientId:{patientid} couldnt be converted")
        return None




## === cell 3
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

    grp = data.groupby("Patient", sort=False, observed=True)
    base = grp.nth(0).reset_index()  # one row per patient, baseline fields
    counts = grp.size().to_numpy(dtype=np.int64)
    pid_rep = np.repeat(base["Patient"].to_numpy(), counts)

    base_rep = base.set_index("Patient").loc[pid_rep].reset_index(drop=False)
    base_rep["Week"] = data["base_Weeks"].to_numpy()
    base_rep["actual_FVC"] = data["base_FVC"].to_numpy()

    npData = base_rep.fillna(0).reset_index(drop=True)
    npData["Week"] = npData["Week"] - npData["base_Weeks"]
    npData["base_Weeks"] = 0.0

    for col in FE1:
        if col not in npData.columns:
            npData[col] = 0

    return npData




## === cell 4
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




## === cell 5
C1 = 70.0
C2 = 1000.0
SQRT2 = float(np.sqrt(2.0))


def score(y_true, y_pred):
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = torch.clamp(sigma, min=C1)
    delta = (y_true[:, 0] - fvc_pred).abs()
    delta = torch.clamp(delta, max=C2)

    sq2 = SQRT2
    metric = (delta / sigma_clip) * sq2 + (sigma_clip * sq2).log()
    return metric.mean()


def qloss(y_true, y_pred):
    qs = [0.25, 0.50, 0.75]
    q = torch.tensor(np.array([qs]), device=y_pred.device, dtype=torch.float32)
    e = y_true - y_pred
    v = torch.max(q * e, (q - 1) * e)
    return v.mean()


def quartile_loss(y_true, y_pred, _lambda=0.65):
    return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)




## === cell 6
data_train = pd.read_csv(os.path.join(OSIC_ROOT, "train.csv"))
data_test = pd.read_csv(os.path.join(OSIC_ROOT, "test.csv"))
sample_submission = pd.read_csv(os.path.join(OSIC_ROOT, "sample_submission.csv"))

train_np = csv_preprocess(data_train)

for col in ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]:
    if col not in train_np.columns:
        train_np[col] = 0

FEAT_COLS = [
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

X_train = train_np[FEAT_COLS].values.astype(np.float32)
y_train = train_np[["actual_FVC"]].values.astype(np.float32)
p_train = train_np["Patient"].values

print("Train rows:", len(train_np), "unique patients:", train_np["Patient"].nunique())



## === cell 7
CACHE_DIR = os.path.join("/kaggle/working", "osic_preprocessed_cache")


def _cache_path(cache_dir, pid, Z, Y, X):
    return (
        None if cache_dir is None else os.path.join(cache_dir, f"{pid}_{Z}x{Y}x{X}.npy")
    )


def _load_from_cache_if_present(cache_dir, pid, Z, Y, X):
    if cache_dir is None:
        return None
    cp = _cache_path(cache_dir, pid, Z, Y, X)
    if cp is None or not os.path.exists(cp):
        return None
    try:
        arr = np.load(cp, mmap_mode="r")
        return np.array(arr, copy=False)
    except Exception:
        return None


def _preload_one(image_root, pid, Z, Y, X, cache_dir):
    arr = _load_from_cache_if_present(cache_dir, pid, Z, Y, X)
    if arr is None:
        arr = read_image(image_root, pid, Z=Z, Y=Y, X=X, cache_dir=cache_dir)
    if arr is None:
        arr = np.zeros((Z, Y, X), dtype="uint8")
    img = (
        torch.from_numpy(np.ascontiguousarray(arr)).to(dtype=torch.float32).unsqueeze(0)
    )
    return pid, img


def preload_patient_images(
    image_root, patient_ids, zyx=(100, 200, 200), cache_dir=None
):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    Z, Y, X = zyx
    patient_ids = list(dict.fromkeys(patient_ids))  # preserve order, unique
    if len(patient_ids) == 0:
        return {}

    max_workers = max(1, min(8, (os.cpu_count() or 1)))
    cache = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [
            ex.submit(_preload_one, image_root, pid, Z, Y, X, cache_dir)
            for pid in patient_ids
        ]
        for fut in as_completed(futs):
            pid, img = fut.result()
            cache[pid] = img

    return {pid: cache[pid] for pid in patient_ids}


class OSICDatasetPreloaded(Dataset):
    def __init__(self, X, y, patients, image_cache):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.patients = patients
        self.image_cache = image_cache

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        pid = self.patients[idx]
        img = self.image_cache[pid]  # float32 tensor [1,Z,Y,X]
        feats = self.X[idx]
        target = self.y[idx]
        return img, feats, target


device = "cuda" if torch.cuda.is_available() else "cpu"

unique_train_patients = pd.unique(p_train).tolist()
train_image_cache = preload_patient_images(
    TRAIN_FOLDER, unique_train_patients, zyx=(100, 200, 200), cache_dir=CACHE_DIR
)

train_ds = OSICDatasetPreloaded(X_train, y_train, p_train, train_image_cache)

g = torch.Generator()
g.manual_seed(SEED)

num_workers = 2 if device == "cuda" else 0
train_loader = DataLoader(
    train_ds,
    batch_size=4 if device == "cuda" else 2,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=(device == "cuda"),
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
    generator=g,
)



## === cell 8
model = Combined_NET().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

model.train()
EPOCHS = 2
for ep in range(EPOCHS):
    running = 0.0
    n = 0
    for img, feats, y in train_loader:
        if device == "cuda":
            img = img.to(device, non_blocking=True)
            feats = feats.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
        else:
            img = img.to(device)
            feats = feats.to(device)
            y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(img, feats)
        loss = quartile_loss(y, pred)
        loss.backward()
        optimizer.step()

        running += float(loss.detach().cpu().item())
        n += 1
    print(f"Epoch {ep+1}/{EPOCHS} - train_loss: {running/max(n,1):.5f}")




## === cell 9
def build_test_expanded(sample_sub, test_df):
    pw = sample_sub["Patient_Week"].str.split("_", n=1, expand=True)
    sub = sample_sub.copy()
    sub["Patient"] = pw[0]
    sub["Weeks_target"] = pw[1].astype(np.int32)

    base = test_df.copy()
    base = base.rename(columns={"Weeks": "Weeks_base", "FVC": "base_FVC"})
    cols_needed = [
        "Patient",
        "Weeks_base",
        "base_FVC",
        "Percent",
        "Age",
        "Sex",
        "SmokingStatus",
    ]
    base = base[cols_needed]

    exp = sub.merge(base, on="Patient", how="left", validate="many_to_one")
    exp = exp.rename(columns={"Weeks_target": "Week"})  # absolute target week
    exp["base_Weeks"] = exp["Weeks_base"]
    exp.drop(columns=["Weeks_base"], inplace=True)

    exp["Healthy-FVC"] = np.round((exp["base_FVC"] * 100.0) / exp["Percent"]).astype(
        np.float32
    )

    for col in ["Sex", "SmokingStatus"]:
        for mod in exp[col].dropna().unique():
            exp[str(mod)] = (exp[col] == mod).astype(np.int32)

    for col in ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]:
        if col not in exp.columns:
            exp[col] = 0

    exp["Week"] = exp["Week"].astype(np.float32) - exp["base_Weeks"].astype(np.float32)
    exp["base_Weeks"] = 0.0

    exp = exp[
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
            "Patient_Week",
        ]
    ].copy()

    return exp


data_test_expanded = build_test_expanded(sample_submission, data_test)

print(
    "Expanded test rows:",
    len(data_test_expanded),
    "unique patients:",
    data_test_expanded["Patient"].nunique(),
)




## === cell 10
def make_eval_data(npEval, model, device="cuda", cache_dir=None, infer_bs=1):
    feat_cols = FEAT_COLS.copy()

    npEval = npEval.copy()
    for c in feat_cols:
        if c not in npEval.columns:
            npEval[c] = 0.0

    unique_patients = npEval.Patient.unique().tolist()
    img_cache = preload_patient_images(
        TEST_FOLDER, unique_patients, zyx=(100, 200, 200), cache_dir=cache_dir
    )

    use_cuda = torch.cuda.is_available() and device == "cuda"
    compute_device = torch.device("cuda") if use_cuda else torch.device("cpu")
    model = model.to(compute_device)
    model.eval()

    preds = np.zeros((len(npEval), 3), dtype=np.float32)

    with torch.no_grad():
        for pid in unique_patients:
            idx = npEval.index[npEval["Patient"] == pid].to_numpy()
            x_feature_all = torch.as_tensor(
                npEval.loc[idx, feat_cols].values, dtype=torch.float32
            )
            base_img = img_cache[pid]  # [1,Z,Y,X] float32 on CPU

            base_img = base_img.unsqueeze(0).to(
                compute_device, non_blocking=use_cuda
            )  # [1,1,Z,Y,X]

            n = len(idx)
            for start in range(0, n, infer_bs):
                end = min(n, start + infer_bs)
                xb = x_feature_all[start:end].to(compute_device, non_blocking=use_cuda)
                ximg = base_img.repeat(end - start, 1, 1, 1, 1)
                pb = model(ximg, xb).detach().cpu().numpy()
                preds[idx[start:end]] = pb

    out = npEval.copy()
    out["FVC"] = preds[:, 1]
    out["Confidence"] = (preds[:, 2] - preds[:, 0]).astype(np.float32)
    out["Confidence"] = out["Confidence"].abs()
    out["Confidence"] = out["Confidence"].clip(lower=70)
    return out


infer_bs = 1 if device == "cuda" else 2
test_pred = make_eval_data(
    data_test_expanded.copy(),
    model,
    device=("cuda" if torch.cuda.is_available() else "cpu"),
    cache_dir=CACHE_DIR,
    infer_bs=infer_bs,
)

out = sample_submission.copy()
out = out.merge(
    test_pred[["Patient_Week", "FVC", "Confidence"]], on="Patient_Week", how="left"
)

out["FVC"] = out["FVC"].fillna(sample_submission["FVC"])
out["Confidence"] = (
    out["Confidence"].fillna(sample_submission["Confidence"]).clip(lower=70)
)

out = out[["Patient_Week", "FVC", "Confidence"]]
out.to_csv("submission.csv", index=False)
print(out.head())
print("Wrote submission.csv with rows:", len(out))
print(
    "submission.csv exists:",
    os.path.exists("submission.csv"),
    "size:",
    os.path.getsize("submission.csv"),
)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'FVC'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/3468052200.py in <cell line: 0>()
     65 )
     66 
---> 67 out["FVC"] = out["FVC"].fillna(sample_submission["FVC"])
     68 out["Confidence"] = (
     69     out["Confidence"].fillna(sample_submission["Confidence"]).clip(lower=70)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'FVC'
