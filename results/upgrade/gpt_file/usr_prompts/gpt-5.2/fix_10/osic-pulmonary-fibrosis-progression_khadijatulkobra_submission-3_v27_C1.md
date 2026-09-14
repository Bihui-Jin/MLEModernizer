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

-6.952725958196671

# 6. Current score

-8.35712

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -24.6542) has done: 'I fix the pandas runtime error by replacing the removed `DataFrame.append` call with `pd.concat`, which is score-neutral but unblocks the pipeline. I also fix the missing pretrained weight path by adding a safe fallback: if the `.pth` file isn’t available in the Kaggle environment, the code still run end-to-end by using the untrained model (and still producing a valid submission). To prevent further NameErrors, I ensure `test` is always defined before post-processing and submission writing. Finally, I keep the output format exactly as required (`Patient_Week,FVC,Confidence`) and clip confidence to at least 70 to match the metric’s clipping behavior.'
- What this solution (achieved -10.60485) has done: 'I fix the CUDA OOM by preventing the `(batch,1,100,200,200)` CT tensor expansion during inference: we compute the image embedding once per patient and then run only the lightweight tabular head in batches, which preserves the same model and weights but drastically reduces memory. I also make inference robust by automatically selecting a safe `feat_batch_size` for the GPU/CPU, and I ensure `test` is always defined so post-processing and submission writing never raise `NameError`. Finally, I keep the required submission schema and enforce the competition’s minimum confidence clipping (≥70) plus baseline-row override for `Week == base_Weeks`.'
- What this solution (achieved -10.60485) has done: 'We make two minimal inference-time fixes that preserve your model and training logic but should improve the public score toward the target by reducing avoidable metric penalties. First, we align `data_test` row order to exactly match `submission` so `submission["FVC"]=test.FVC.values` cannot silently mis-assign predictions to the wrong `Patient_Week` (a common hidden score killer). Second, we enforce valid/non-negative confidence by applying the competition clip on `sigma` and using a slightly safer confidence construction (still `pred[:,2]-pred[:,0]`, just clipped), preventing accidental negative/too-small values that hurt the Laplace LL. Everything else (architecture, weights, loss, CT embedding reuse) stays unchanged and the script still writes `submission.csv`.'
- What this solution (achieved -10.86799) has done: 'I fix the `KeyError: 'Weeks'` by sorting on the correct post-merge column names (`Weeks_x/Weeks_y`) and by renaming them before sorting, so the test expansion aligns correctly to `sample_submission`. I also make the pipeline resilient to earlier failures by ensuring `data_test` always contains `Patient_Week`, and by dropping `Patient_Week` with `errors="ignore"` so inference can run regardless of minor column state. Finally, I guard submission writing so `test` is always defined (or a safe fallback is used) and always write a valid `submission.csv` with the exact required columns and confidence clipped to ≥70 (score-neutral correctness fix).'
- What this solution (achieved -8.35712) has done: 'Your current score (-10.86799) is below the target (-6.9527), so we should improve it cautiously without changing the model/training core. The biggest minimal gain usually comes from aligning predictions with the metric: make sure predicted quantiles are ordered (so Confidence is meaningful/non-explosive) and set Confidence using a robust scale tied to the quantile spread, then clip to ≥70. Additionally, we ensure the `Patient_Week` merge/order is strictly aligned to `sample_submission` (avoid silent row mis-assignment), and we enforce baseline-week override deterministically. These are inference/post-processing changes only and keep architecture/training unchanged while reducing avoidable metric penalties.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import pydicom  # read the dicom files
import scipy.ndimage
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn as nn

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

TRAIN_FOLDER = "../data/train"

try:
    import pydicom.config

    pydicom.config.image_handlers = ["gdcm", "pylibjpeg", "numpy"]
except Exception:
    pass

CACHE_DIR = "../working/osic_ct_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(dir_name, patientid, Z, Y, X):
    base = os.path.basename(os.path.normpath(dir_name))
    return os.path.join(CACHE_DIR, f"{base}_{patientid}_Z{Z}_Y{Y}_X{X}.npy")


def load_scan(path):
    files = os.listdir(path)
    file_paths = [os.path.join(path, s) for s in files]

    slices = []
    try:
        for fp in file_paths:
            ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
            slices.append(ds)
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except Exception:
        file_paths.sort()
        slices = [
            pydicom.dcmread(fp, stop_before_pixels=True, force=True)
            for fp in file_paths
        ]

    for ds, fp in zip(slices, [getattr(ds, "filename", None) for ds in slices]):
        if fp is None:
            pass
    return slices, file_paths


def make_dict_with_slices(patientIDs):
    patient_slices_dict = {}
    failed_slices_patiendIDs = []

    for patientID in patientIDs:
        path = TRAIN_FOLDER + os.sep + patientID
        try:
            patient_slices_dict[patientID] = load_scan(path)
        except Exception:
            failed_slices_patiendIDs.append(patientID)

    return patient_slices_dict, failed_slices_patiendIDs


def get_pixels_hu(slices_and_paths):
    header_slices, file_paths = slices_and_paths

    sorted_paths = []
    for ds in header_slices:
        fp = getattr(ds, "filename", None)
        if fp is not None and os.path.exists(fp):
            sorted_paths.append(fp)
        else:
            sorted_paths = file_paths
            break

    px_list = []
    for fp in sorted_paths:
        ds = pydicom.dcmread(fp, force=True)
        px_list.append(ds.pixel_array)
    image = np.stack(px_list).astype(np.int16)

    try:
        image[image <= -2000] = 0
        for slice_number in range(len(header_slices)):
            intercept = header_slices[slice_number].RescaleIntercept
            slope = header_slices[slice_number].RescaleSlope

            if slope != 1:
                image[slice_number] = slope * image[slice_number].astype(np.float64)
                image[slice_number] = image[slice_number].astype(np.int16)
            image[slice_number] += np.int16(intercept)
    except Exception:
        print("HU conversion Failed!!")

    return np.array(image, dtype=np.int16)


def plot_show_slice(slices):
    if not isinstance(slices, type(np.array([]))):
        first_patient_pixels = get_pixels_hu(slices)  # conversion to np array and HU
    else:
        first_patient_pixels = slices

    print("Number of Total Slices in this Scan:", len(slices))
    try:
        print("Shape of the the Image is:", slices.shape[1], slices.shape[2])
    except Exception:
        print("Shape of the the Image is:BLANK")
    fig = plt.figure(figsize=(10, 10))
    for i, slice_ in enumerate(first_patient_pixels[:16]):
        y = fig.add_subplot(4, 4, i + 1)
        y.imshow(slice_, cmap="gray")
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
    cpath = _cache_path(dir_name, patientid, Z, Y, X)
    if os.path.exists(cpath):
        return np.load(cpath, mmap_mode=None)

    path = dir_name + os.sep + patientid
    slices = load_scan(path)
    try:
        image_array = get_pixels_hu(slices)
        ctimage_resizedAll = resize_along_allaxis(
            image_array, target_dimensionX=X, target_dimensionY=Y, target_dimensionZ=Z
        )
        image = (image_normalize(ctimage_resizedAll) * 255.0).astype("uint8")
        np.save(cpath, image)
        return image
    except Exception:
        print("PatientId:%s couldnt be converted" % (patientid))
        return None




## === cell 1
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

    g = data.groupby("Patient", sort=False)
    base = g.nth(0).reset_index()  # baseline row per patient (first after sort)
    sizes = g.size().to_numpy()
    base_rep = base.loc[base.index.repeat(sizes)].reset_index(drop=True)

    base_rep["Week"] = data["base_Weeks"].to_numpy()
    base_rep["actual_FVC"] = data["base_FVC"].to_numpy()

    for c in FE1:
        if c not in base_rep.columns:
            base_rep[c] = 0

    cols = (
        ["Patient", "base_Weeks", "base_FVC", "Age"]
        + FE1
        + ["Week", "Healthy-FVC", "actual_FVC"]
    )
    base_rep = base_rep[cols].fillna(0).reset_index(drop=True)
    return base_rep




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




## === cell 3
C1_VALUE, C2_VALUE = 70.0, 1000.0

_SQRT2_CPU_F32 = float(np.sqrt(2.0))


def score(y_true, y_pred):
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]
    sigma_clip = torch.clamp(sigma, min=C1_VALUE)
    delta = (y_true[:, 0] - fvc_pred).abs()
    delta = torch.clamp(delta, max=C2_VALUE)
    sq2 = torch.as_tensor(_SQRT2_CPU_F32, device=y_pred.device, dtype=y_pred.dtype)
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




## === cell 4
def _enforce_quantile_monotonicity(preds_3):
    preds_3 = np.asarray(preds_3, dtype=np.float32)
    q25 = preds_3[:, 0]
    q50 = preds_3[:, 1]
    q75 = preds_3[:, 2]
    q25n = np.minimum(q25, q75)
    q75n = np.maximum(q25, q75)
    q50n = np.clip(q50, q25n, q75n)
    out = np.stack([q25n, q50n, q75n], axis=1)
    return out


def make_eval_data(npEval, model, device="cuda", feat_batch_size=128):
    feats_cols = [
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
    x_features_np = npEval[feats_cols].to_numpy(dtype=np.float32, copy=False)
    patients_np = npEval["Patient"].to_numpy()

    unique_patients = pd.unique(patients_np)
    dir_name_of_patientid = "../input/osic-pulmonary-fibrosis-progression/test"

    loaded_images = {}
    max_workers = min(8, (os.cpu_count() or 2))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(read_image, dir_name_of_patientid, pid, 100, 200, 200): pid
            for pid in unique_patients
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Loading test CTs"):
            pid = futs[fut]
            img = fut.result()
            if img is None:
                raise RuntimeError(
                    f"Failed to read DICOM stack for patient {pid} from {dir_name_of_patientid}"
                )
            loaded_images[pid] = img

    model.to(device)
    model.eval()

    if device == "cuda":
        feat_batch_size = int(min(max(1, feat_batch_size), 8))
    else:
        feat_batch_size = int(min(max(1, feat_batch_size), 256))

    preds = np.empty((len(npEval), 3), dtype=np.float32)

    with torch.no_grad():
        for pid in unique_patients:
            idx = np.flatnonzero(patients_np == pid)

            x_image = torch.from_numpy(loaded_images[pid]).to(
                device=device, dtype=torch.float32
            )
            x_image = (x_image / 255.0).unsqueeze(0).unsqueeze(0)  # (1,1,Z,Y,X)

            image_emb_1 = model.image(x_image)  # (1, 32)

            for start in range(0, len(idx), feat_batch_size):
                b = idx[start : start + feat_batch_size]
                x_feat = torch.from_numpy(x_features_np[b]).to(
                    device=device, dtype=torch.float32
                )
                img_emb_b = image_emb_1.expand(x_feat.size(0), -1)
                out = model.data(x_feat, img_emb_b).detach().to("cpu").numpy()
                preds[b] = out

    preds = _enforce_quantile_monotonicity(preds)

    npEval = npEval.copy()
    npEval["FVC"] = preds[:, 1]

    iqr = (preds[:, 2] - preds[:, 0]).astype(np.float32)
    conf = (iqr * (np.float32(np.sqrt(2.0)) / (2.0 * np.float32(np.log(2.0))))).astype(
        np.float32
    )
    conf = np.maximum(conf, C1_VALUE).astype(np.float32)
    npEval["Confidence"] = conf
    return npEval




## === cell 5
data_train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
data_test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
sample_sub = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)



## === cell 6
FE1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]


def add_onehot_columns(df):
    df = df.copy()
    for col, mods in {
        "Sex": ["Male", "Female"],
        "SmokingStatus": ["Ex-smoker", "Never smoked", "Currently smokes"],
    }.items():
        for mod in mods:
            df[mod] = (df[col] == mod).astype(int)
    return df


npTrain = csv_preprocess(data_train.copy())
for c in FE1:
    if c not in npTrain.columns:
        npTrain[c] = 0

pw = sample_sub["Patient_Week"].str.split("_", n=1, expand=True)
sub_keys = pd.DataFrame(
    {
        "Patient_Week": sample_sub["Patient_Week"].values,
        "Patient": pw[0].values,
        "Weeks": pw[1].astype(np.int32).values,
    }
)
sub_keys = sub_keys.sort_values(by=["Patient", "Weeks"], ascending=True).reset_index(
    drop=True
)

merge = pd.merge(data_test, sub_keys, on=["Patient"], how="left")
merge = merge.rename(
    columns={"Weeks_x": "base_Weeks", "Weeks_y": "Week", "FVC": "base_FVC"}
)
merge = merge.sort_values(["Patient", "Week"], ascending=True).reset_index(drop=True)

merge = merge[
    [
        "Patient",
        "base_Weeks",
        "base_FVC",
        "Percent",
        "Age",
        "Sex",
        "SmokingStatus",
        "Week",
        "Patient_Week",
    ]
].copy()

data = merge.copy()
data["Healthy-FVC"] = round((data["base_FVC"] * 100) / data["Percent"])
data = add_onehot_columns(data)

npData = pd.concat(
    [
        pd.DataFrame(
            columns=["Patient", "base_Weeks", "base_FVC", "Age", "Healthy-FVC"]
            + FE1
            + ["Week", "Patient_Week"]
        ),
        data,
    ],
    ignore_index=True,
    sort=True,
).fillna(0)

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
        "Patient_Week",
    ]
].copy()



## === cell 7
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Combined_NET()

ckpt_path = "../input/3other683/Epoch3_Score6.832028585396066_Acc0.9297884854654602.pth"
loaded_ckpt = False
if os.path.exists(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    loaded_ckpt = True
else:
    print(
        f"Warning: checkpoint not found at {ckpt_path}. Will run a small fallback training on train.csv/train CTs."
    )

if not loaded_ckpt:
    train_dir = "../input/osic-pulmonary-fibrosis-progression/train"

    feats_cols = [
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

    available_patients = set(os.listdir(train_dir))
    npTrain = npTrain[npTrain["Patient"].isin(available_patients)].reset_index(
        drop=True
    )

    unique_patients = npTrain["Patient"].unique().tolist()
    loaded_images_train = {}

    max_workers = min(8, (os.cpu_count() or 2))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(read_image, train_dir, pid, 100, 200, 200): pid
            for pid in unique_patients
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Loading train CTs"):
            pid = futs[fut]
            img = fut.result()
            if img is None:
                continue
            loaded_images_train[pid] = img

    npTrain = npTrain[
        npTrain["Patient"].isin(list(loaded_images_train.keys()))
    ].reset_index(drop=True)

    X_feat = torch.tensor(npTrain[feats_cols].values, dtype=torch.float32)
    y = torch.tensor(npTrain[["actual_FVC"]].values, dtype=torch.float32)
    y_q = y.repeat(1, 3)

    pid_arr = npTrain["Patient"].values

    model.to(device)
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 2
    epochs = 2

    idxs = np.arange(len(npTrain))

    for ep in range(epochs):
        np.random.shuffle(idxs)
        running = 0.0
        steps = 0
        for start in tqdm(
            range(0, len(idxs), batch_size), desc=f"Training epoch {ep+1}/{epochs}"
        ):
            bidx = idxs[start : start + batch_size]
            imgs = [loaded_images_train[pid] for pid in pid_arr[bidx]]

            x_img = (
                torch.tensor(np.stack(imgs, axis=0), dtype=torch.float32) / 255.0
            ).unsqueeze(1)

            x_feat = X_feat[bidx]
            yb = y_q[bidx]

            x_img = x_img.to(device)
            x_feat = x_feat.to(device)
            yb = yb.to(device)

            pred = model(x_img, x_feat)
            loss = quartile_loss(yb, pred)

            optim.zero_grad()
            loss.backward()
            optim.step()

            running += float(loss.detach().cpu().item())
            steps += 1

        print(f"Epoch {ep+1}: train_loss={running/max(steps,1):.5f}")



## === cell 8
test_feats = data_test.copy()

eval_df = test_feats.drop(columns=["Patient_Week"], errors="ignore").copy()

test = make_eval_data(
    eval_df,
    model,
    device=device,
    feat_batch_size=128,
)

if "Patient_Week" in test_feats.columns and len(test_feats) == len(test):
    test["Patient_Week"] = test_feats["Patient_Week"].values
else:
    test = test.merge(
        test_feats[["Patient", "Week", "Patient_Week"]],
        on=["Patient", "Week"],
        how="left",
    )



## === cell 9
if "test" not in globals() or test is None or len(test) == 0:
    final = sample_sub.copy()
    final["FVC"] = 2000.0
    final["Confidence"] = 70.0
else:
    base_mask = test["Week"].to_numpy() == test["base_Weeks"].to_numpy()
    if base_mask.any():
        test.loc[base_mask, "FVC"] = test.loc[base_mask, "base_FVC"].to_numpy()
        test.loc[base_mask, "Confidence"] = 70.0

    test.loc[test.Confidence < 70.0, "Confidence"] = 70.0

    pred_out = test[["Patient_Week", "FVC", "Confidence"]].copy()

    final = sample_sub[["Patient_Week"]].merge(pred_out, on="Patient_Week", how="left")

    base_map = data_test.drop_duplicates("Patient")[["Patient", "base_FVC"]].copy()
    pw2 = final["Patient_Week"].str.split("_", n=1, expand=True)
    final["Patient"] = pw2[0].values
    final = final.merge(base_map, on="Patient", how="left")

    final["FVC"] = final["FVC"].fillna(final["base_FVC"]).astype(np.float32)
    final["Confidence"] = final["Confidence"].fillna(70.0).astype(np.float32)
    final.loc[final["Confidence"] < 70.0, "Confidence"] = 70.0

    final = final[["Patient_Week", "FVC", "Confidence"]]

final.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final.shape)
print(final.head())
