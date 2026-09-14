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

-6.916795942216862

# 6. Current score

-10.81761

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -10.81761) has done: 'I fix the pandas runtime error by replacing the removed `DataFrame.append` call with `pd.concat`, which is score-neutral but unblocks the pipeline. I also fix the missing pretrained weight file error by making weight loading optional and falling back to a deterministic, metric-safe baseline predictor (use `base_FVC` for all weeks) when the `.pth` is not available, ensuring a valid submission is always produced. Additionally, I correct the test DICOM folder path to the Kaggle dataset path used elsewhere in the notebook and make `make_eval_data` robust to missing/failed image reads by using zeros. Finally, I guarantee the submission columns/row order exactly match `sample_submission.csv` and clip confidence to the required minimum (70).'
- What this solution (achieved -10.81761) has done: 'Your current score is far below the target, and the biggest (minimal) improvement lever without changing the model is to actually use the trained weights instead of falling back to the constant baseline when the external `.pth` file is missing. I switch weight loading to look for a local `weights.pth` (or common variants) placed next to the notebook/script, while keeping the baseline fallback if no weights are available so a valid submission is always produced. I also fix a small but important path inconsistency: your image loader points to `../input/.../test`, but your environment paths show the dataset under `../data/...`, so I make the code automatically pick the correct existing test DICOM directory. Finally, I keep submission ordering identical to `sample_submission.csv` to avoid any accidental misalignment penalties.'
- What this solution (achieved -10.81761) has done: 'We need to move the score up from -10.82 toward -6.92, so the smallest safe gain is to make the existing network inference actually work as intended and not be silently degraded by train-mode layers. I keep your model/feature logic intact, but (1) load weights more robustly (handle common checkpoint formats like `{'state_dict': ...}`), (2) set deterministic evaluation behavior and correct device placement, and (3) fix a key inference mismatch: your IMAGE branch expects normalized float input (0–1), but `read_image()` returns `uint8` 0–255; converting to 0–1 at eval is a minimal, metric-aligned change that typically improves predictions without changing architecture/training. I also clip the model’s predicted confidence to be positive before the required `>=70` clip to avoid pathological sigma values hurting the Laplace metric, while preserving your existing final clipping step. All I/O paths and submission formatting remain the same.'
- What this solution (achieved -10.81761) has done: 'Your score is well below the target (need to improve from -10.82 toward about -6.92), so we should make the smallest changes that improve inference quality without changing the model or training. The biggest low-risk win here is fixing a feature-order mismatch: `make_eval_data()` currently feeds features in a different column order than the network was trained on (42-dim input), which can severely degrade predictions. I also make the data columns explicitly numeric `float32` before converting to tensors (avoids silent object-dtype issues) and add a safe fallback to baseline predictions if weights exist but inference fails, so you always get a valid submission. Paths and submission formatting remain unchanged.'
- What this solution (achieved -10.81761) has done: 'Your current score (-10.8176) is much worse than the target (-6.9168), so we should improve inference quality without changing the model/training logic. The smallest high-impact fix is to feed the full 42 clinical feature vector that `SIGMA` expects (your current eval uses only 10 features, which breaks the network’s learned mapping). I mirror the original one-hot feature construction from `csv_preprocess` for test-time and enforce a stable, explicit feature column order + float32 conversion. I also keep your existing weight-loading, image normalization, and submission alignment, only adding a safe fallback if any expected one-hot columns are missing.'
- What this solution (achieved -10.81761) has done: 'We need to move the score up (from -10.8176 toward -6.9168), so the smallest safe improvement is to stop feeding “wrong/mostly-zero” clinical features into the already-trained network. I keep your exact model and inference loop, but I rebuild the 42-dim tabular input to match what `csv_preprocess()` produced during training: base fields + the 5 one-hots + `Week` plus the remaining dims as zeros (so the network sees the intended signal instead of a misordered/padded vector). I also fix a subtle column issue where `Percent` was missing from `data_test` before `make_eval_data()` tries to recompute `Healthy-FVC`, which could silently degrade predictions or trigger fallback. Everything else (paths, image reading, weight loading, submission alignment, confidence clipping) stays the same.'
- What this solution (achieved -10.81761) has done: 'We need to raise the score from -10.8176 toward -6.9168 (higher is better), so the smallest safe gain is to make your inference use a saner confidence (sigma) and avoid pathological outputs that heavily penalize the Laplace metric. I keep your model and feature construction intact, but change only post-processing: interpret the first head as a *positive* sigma and clamp it to a reasonable range (>=70 and not absurdly large), and clip FVC to a plausible physiological range to avoid rare extreme predictions. This preserves your core logic (same network, same weights, same inputs) while improving metric behavior directly. I also make the dataset path selection consistent for CSV loading (your environment has `../data/...`), which prevents accidental fallback behavior due to missing files.'
- What this solution (achieved -10.81761) has done: 'Your current score (-10.8176) is much worse than the target (-6.9168), so we need a modest but reliable lift without changing the model/training logic. The biggest minimal win here is to fix a likely submission alignment bug: you currently overwrite `submission` rows with `test["FVC"].values` assuming identical row order, but `test` and `submission` can be ordered differently (different sorts/merges), which can destroy score. I change only the final assembly to merge predictions back by the true key (`Patient_Week`) instead of positional assignment, while keeping your model, feature construction, and inference identical. I also ensure `Patient_Week` exists in `test` so the key-merge is lossless and deterministic.'
- What this solution (achieved -10.81761) has done: 'We need to move the score up (from -10.82 toward -6.92), so the smallest safe lever is improving prediction calibration without changing the model or training: your current post-processing clamps `sigma` too aggressively (upper-bounded at 500) and treats the first head as “sigma” without ensuring it’s numerically stable. I keep the exact model and inference loop, but change only metric-aligned post-processing: compute `sigma` as a strictly-positive value via `softplus` (still monotonic), then clip to `[70, 1000]` (matching the metric’s natural error cap scale), which typically improves Laplace log-likelihood when the model’s raw sigma is poorly scaled. I also ensure the output head ordering is consistent (if the checkpoint was trained with `[FVC, sigma]` instead of `[sigma, FVC]`, the current code can be catastrophically wrong), by selecting the ordering that gives reasonable ranges on the *test baseline week* rows (no labels used) and then applying the same ordering everywhere. Submission writing and row alignment remain unchanged.'
- What this solution (achieved -10.81761) has done: 'Your current score is far below the target (need a sizable lift from -10.82 toward about -6.92), so the most likely minimal win is fixing inference-time postprocessing that can badly hurt the Laplace metric. I keep your model, inputs, and loops intact, but change only the final head interpretation: instead of guessing output order via a tiny probe, I robustly pick the head order that yields more plausible baseline-week behavior (reasonable FVC range and sigma range) across all baseline rows, which is still label-free. I also remove the hard upper-clip of sigma at 1000 (keeping the required lower clip at 70) because overly-large sigma is already penalized by `-log(sigma)` and forcing it can be miscalibrating; we instead enforce positivity with `softplus` and a gentle upper cap at 3000 to avoid numerical explosions. Finally, I ensure the baseline-week overwrite uses the correct `Patient_Week` key and keep submission alignment exactly matching `sample_submission.csv`.'
- What this solution (achieved -10.81761) has done: 'Your current score (-10.8176) is well below the target (-6.9168), so we should make a small, metric-aligned improvement without changing your model or training loop. The biggest low-risk gain is to fix the head post-processing: your network’s final layer uses `ReLU`, so outputs are already non-negative; applying `softplus` to the “sigma head” can unnecessarily inflate sigma and hurt the Laplace log-likelihood. I keep the same head-order selection logic, but decode sigma as a direct positive value (with a small epsilon), and set only gentle, metric-consistent clipping (lower bound 70, cap at 1000). This change is confined to inference post-processing and should move the score upward toward the target while preserving the rest of your pipeline and submission alignment.'
- What this solution (achieved -10.81761) has done: 'We need to raise your score from -10.8176 toward the target -6.9168 (higher is better), so the smallest safe win is to improve metric-aligned post-processing without touching the model, training, or feature/image pipelines. Your current inference clamps sigma too tightly and uses a head-order heuristic that only looks at absolute FVC error; both can yield poorly calibrated confidences that the Laplace metric punishes. I keep your model forward pass identical, but (1) select the output-head ordering by directly maximizing the Laplace metric on baseline-week rows (using only the known baseline FVC available in test, which is allowed), and (2) replace the hard sigma cap of 1000 with a gentle cap (3000) while still enforcing the required minimum 70—this typically improves the likelihood term when the model’s sigma scale is off. Submission format, ordering, and baseline-week overwrite remain unchanged.'
- What this solution (achieved -10.81761) has done: 'I keep your model and inference loop intact and make only metric-aligned post-processing changes that are likely to lift the score from -10.82 toward the -6.92 target. The main adjustment is to recalibrate the predicted `Confidence` (sigma): instead of using the raw network sigma (often mis-scaled), we set sigma based on the per-row absolute deviation from the known baseline FVC for that patient (available in test), which directly optimizes the Laplace log-likelihood without using any future labels. FVC predictions remain exactly what the model outputs (still clipped to a plausible range), and we keep the baseline-week overwrite and the submission merge-by-key to avoid misalignment. This is a small, robust change localized to inference post-processing and should improve the metric substantially when the original sigma head is poorly calibrated.'
- What this solution (achieved -10.81761) has done: 'Your current score (-10.8176) is worse than the target (-6.9168), so we should make a small, metric-aligned change that increases the score without changing the model or training. The biggest low-risk lever is the `Confidence` (sigma): right now it’s set proportional to per-row deviation from the baseline FVC, which can over-penalize via the `-log(sigma)` term when delta is small. I keep your model predictions exactly as-is, but recalibrate sigma to a simple patient-level constant based on the model’s own predicted spread across weeks (no label leakage), then apply the metric-required lower clip (70). This typically improves Laplace log-likelihood by avoiding tiny sigmas when the model isn’t perfectly accurate and avoiding overly large sigmas that lose `-log(sigma)`.'
- What this solution (achieved -10.81761) has done: 'Your score is far below the target, so we should make a small, metric-aligned improvement without touching the model architecture or training logic. The current inference recalibrates `Confidence` using only the model’s predicted spread, which often underestimates uncertainty and gets heavily penalized by the Laplace metric when FVC is off. I keep your FVC predictions exactly the same, but recalibrate `Confidence` using a patient-level blend of (a) the model spread and (b) the model’s own baseline-week absolute error against the known baseline FVC in `test.csv` (allowed information), then clip at 70. This is localized to post-processing, keeps paths and submission alignment intact, and should move the score upward toward your target.'

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
from torch.utils.data import TensorDataset



## === cell 1
TRAIN_FOLDER = "../data/train"


def load_scan(
    path,
):  # Here path == (../input/osic-pulmonary-fibrosis-progression/train/patientId)
    try:
        slices = [pydicom.dcmread(path + os.sep + s) for s in os.listdir(path)]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except:
        files = os.listdir(path)
        files.sort()
        slices = [pydicom.dcmread(path + os.sep + s) for s in files]
    return slices


def make_dict_with_slices(
    patientIDs,
):  # this will be very huge memory consuming . don't use it
    patient_slices_dict = {}
    failed_slices_patiendIDs = []
    for patientID in patientIDs:
        path = TRAIN_FOLDER + os.sep + patientID
        try:
            patient_slices_dict[patientID] = load_scan(path)
        except:
            failed_slices_patiendIDs.append(patientID)
    return patient_slices_dict, failed_slices_patiendIDs


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
    except:
        print("HU conversion Failed!!")
    return np.array(image, dtype=np.int16)


def plot_show_slice(slices):
    if not isinstance(slices, type(np.array([]))):
        first_patient_pixels = get_pixels_hu(
            slices
        )  ## conversion to np array and HU unit
    else:
        first_patient_pixels = slices
    print("Number of Total Slices in this Scan:", len(slices))
    try:
        print("Shape of the the Image is:", slices.shape[1], slices.shape[2])
    except:
        print("Shape of the the Image is:BLANK")
    fig = plt.figure(figsize=(10, 10))
    for i, slice in enumerate(first_patient_pixels[:16]):
        y = fig.add_subplot(4, 4, i + 1)
        y.imshow(slice, cmap="gray")
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


def save_array(
    patientID_folder, output_folder, Z=100, Y=200, X=200
):  # patientID_folder, output_folder
    patientIDs = os.listdir(patientID_folder)
    patientIDs.sort()
    Save_dir = output_folder  # './trainset'
    if not os.path.exists(Save_dir):
        print("The output directory doesnt exists")
        raise Exception
    for i, patientID in enumerate(patientIDs):
        path = TRAIN_FOLDER + os.sep + patientIDs[i]
        slices = load_scan(path)
        try:
            image_array = get_pixels_hu(slices)  # HU unit conversion + nparray
            ctimage_resizedAll = resize_along_allaxis(
                image_array,
                target_dimensionX=X,
                target_dimensionY=Y,
                target_dimensionZ=Z,
            )
            image = (image_normalize(ctimage_resizedAll) * 255.0).astype("uint8")
            np.save(Save_dir + os.sep + patientID + ".npy", image)
        except:
            print("PatientId:%s couldnt be save and converted" % (patientID))


def load_array(path):  ### path='./traindataset/ID00052637202186188008618.npy'
    try:
        if path.endswith(".npy"):
            image_array = np.load(path)
        else:
            path = path + ".npy"
            image_array = np.load(path)
    except:
        print(
            "The file in the Path:%s doesnetexists!!"
            % (path.split("\\")[-1].split("/")[-1].split(".")[0])
        )
        return []
    return image_array


def read_image(
    dir_name, patientid, Z=100, Y=200, X=200
):  # patientID_folder, output_folder
    path = dir_name + os.sep + patientid
    slices = load_scan(path)
    try:
        image_array = get_pixels_hu(slices)  # HU unit conversion + nparray
        ctimage_resizedAll = resize_along_allaxis(
            image_array, target_dimensionX=X, target_dimensionY=Y, target_dimensionZ=Z
        )
        image = (image_normalize(ctimage_resizedAll) * 255.0).astype("uint8")
        return image
    except:
        print("PatientId:%s couldnt be converted" % (patientid))
        return None




## === cell 2
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
C1, C2 = torch.tensor(70, dtype=torch.float32), torch.tensor(1000, dtype=torch.float32)


def score(y_true, y_pred):
    sigma = y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    c1_same_shape = torch.ones(sigma.size(), device=y_pred.device) * C1
    sigma_clip = torch.max(sigma, c1_same_shape)
    delta = (y_true[:, 0] - fvc_pred).abs()

    c2_same_shape = torch.ones(delta.size(), device=y_pred.device) * C2
    delta = torch.min(delta, c2_same_shape)

    sq2 = torch.tensor(2.0).sqrt()
    metric = (delta / sigma_clip) * sq2 + (sigma_clip * sq2).log()
    return (metric).mean()


def quartile_loss(y_true, y_pred):  # 0.65
    loss = score(y_true, y_pred)  # 0.35
    return loss




## === cell 5
def make_eval_data(npEval, model, device="cuda"):
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    device = torch.device(device)

    npEval = npEval.copy()

    if "Percent" not in npEval.columns:
        npEval["Percent"] = np.nan

    npEval["Healthy-FVC"] = npEval.get("Healthy-FVC", np.nan)
    if npEval["Healthy-FVC"].isna().any():
        npEval["Healthy-FVC"] = np.round((npEval["base_FVC"] * 100) / npEval["Percent"])

    FE1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]
    if "Sex" in npEval.columns:
        npEval["Male"] = (npEval["Sex"] == "Male").astype(int)
        npEval["Female"] = (npEval["Sex"] == "Female").astype(int)
    else:
        npEval["Male"] = 0
        npEval["Female"] = 0
    if "SmokingStatus" in npEval.columns:
        npEval["Ex-smoker"] = (npEval["SmokingStatus"] == "Ex-smoker").astype(int)
        npEval["Never smoked"] = (npEval["SmokingStatus"] == "Never smoked").astype(int)
        npEval["Currently smokes"] = (
            npEval["SmokingStatus"] == "Currently smokes"
        ).astype(int)
    else:
        npEval["Ex-smoker"] = 0
        npEval["Never smoked"] = 0
        npEval["Currently smokes"] = 0

    core_feature_cols = [
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
    for c in core_feature_cols:
        if c not in npEval.columns:
            npEval[c] = 0

    x_core = npEval[core_feature_cols].to_numpy(dtype=np.float32)
    x_padded = np.concatenate(
        [x_core, np.zeros((x_core.shape[0], 42 - x_core.shape[1]), dtype=np.float32)],
        axis=1,
    )
    x_features = torch.tensor(x_padded, dtype=torch.float32)
    x_patientids_name = npEval[["Patient"]].values

    unique_patients = npEval.Patient.unique()
    loaded_images = {}

    candidate_test_dirs = [
        "../input/osic-pulmonary-fibrosis-progression/test",
        "../data/osic-pulmonary-fibrosis-progression/test",
        "../data/test",
    ]
    dir_name_of_patientid = None
    for d in candidate_test_dirs:
        if os.path.isdir(d):
            dir_name_of_patientid = d
            break
    if dir_name_of_patientid is None:
        dir_name_of_patientid = candidate_test_dirs[0]  # will fall back to zeros

    for unique_patient in unique_patients:
        img = read_image(dir_name_of_patientid, unique_patient, Z=100, Y=200, X=200)
        if img is None or (isinstance(img, (list, tuple)) and len(img) == 0):
            img = np.zeros((100, 200, 200), dtype=np.uint8)
        loaded_images[unique_patient] = img

    model = model.to(device)
    model.eval()

    def _decode_raw(raw2, order_choice):
        a, b = float(raw2[0]), float(raw2[1])
        if order_choice == 1:  # raw[0]=sigma, raw[1]=fvc
            sigma = a
            fvc = b
        else:  # swapped
            sigma = b
            fvc = a

        sigma = float(max(sigma, 1e-3))
        sigma = float(np.clip(sigma, 1e-3, 3000.0))
        return sigma, fvc

    def _laplace_metric(fvc_true, fvc_pred, sigma):
        sigma_clip = max(float(sigma), 70.0)
        delta = min(abs(float(fvc_true) - float(fvc_pred)), 1000.0)
        return -(np.sqrt(2.0) * delta) / sigma_clip - np.log(np.sqrt(2.0) * sigma_clip)

    baseline_mask = npEval["Week"].to_numpy() == npEval["base_Weeks"].to_numpy()
    baseline_indices = np.where(baseline_mask)[0]

    order_choice = 1
    if len(baseline_indices) > 0:
        metrics = {1: [], 2: []}
        with torch.no_grad():
            for i in baseline_indices:
                pid = x_patientids_name[i][0]
                x_image = torch.tensor(loaded_images[pid], dtype=torch.float32) / 255.0
                x_image = x_image.unsqueeze(0).unsqueeze(0).to(device)
                x_feature = x_features[i].unsqueeze(0).to(device)
                raw = model(x_image, x_feature)[0].detach().to("cpu").numpy()

                base_fvc = float(npEval.iloc[i]["base_FVC"])
                for oc in (1, 2):
                    sigma, fvc = _decode_raw(raw, oc)
                    metrics[oc].append(_laplace_metric(base_fvc, fvc, sigma))

        m1 = float(np.mean(metrics[1])) if len(metrics[1]) else -1e9
        m2 = float(np.mean(metrics[2])) if len(metrics[2]) else -1e9
        order_choice = 2 if m2 > m1 else 1

    raw_preds = np.zeros((len(npEval), 2), dtype=np.float32)
    with torch.no_grad():
        for i, patientid in enumerate(x_patientids_name):
            pid = patientid[0]
            x_image = torch.tensor(loaded_images[pid], dtype=torch.float32) / 255.0
            x_image = x_image.unsqueeze(0).unsqueeze(0).to(device)
            x_feature = x_features[i].unsqueeze(0).to(device)
            raw = model(x_image, x_feature)[0].detach().to("cpu").numpy()
            sigma_raw, fvc = _decode_raw(raw, order_choice)
            raw_preds[i, 0] = sigma_raw
            raw_preds[i, 1] = fvc

    dfp = npEval[["Patient", "Week", "base_Weeks", "base_FVC"]].copy()
    dfp["fvc_pred"] = raw_preds[:, 1].astype(np.float32)

    spread = dfp.groupby("Patient")["fvc_pred"].agg(
        fvc_std="std", fvc_min="min", fvc_max="max"
    )
    spread["fvc_std"] = spread["fvc_std"].fillna(0.0)

    base_rows = dfp[dfp["Week"].to_numpy() == dfp["base_Weeks"].to_numpy()].copy()
    base_rows["abs_err_base"] = (base_rows["fvc_pred"] - base_rows["base_FVC"]).abs()
    base_err = base_rows.groupby("Patient")["abs_err_base"].mean().fillna(0.0)

    spread = spread.join(base_err.rename("abs_err_base"), how="left")
    spread["abs_err_base"] = spread["abs_err_base"].fillna(
        spread["abs_err_base"].median() if len(spread) else 0.0
    )

    spread_term = 2.0 * spread["fvc_std"].to_numpy()
    base_term = np.sqrt(2.0) * spread["abs_err_base"].to_numpy() + 70.0
    sigma_pat = 0.5 * spread_term + 0.5 * base_term
    spread["sigma_pat"] = np.clip(sigma_pat, 70.0, 3000.0)
    sigma_map = spread["sigma_pat"].to_dict()

    predictions = []
    for i in range(len(npEval)):
        pid = str(npEval.iloc[i]["Patient"])
        fvc = float(raw_preds[i, 1])
        sigma_cal = float(sigma_map.get(pid, 200.0))
        fvc = float(np.clip(fvc, 500.0, 6000.0))
        predictions.append([sigma_cal, fvc])

    predictions = np.array(predictions, dtype=np.float32)
    npEval["FVC"] = predictions[:, 1]
    npEval["Confidence"] = predictions[:, 0]
    return npEval




## === cell 6
candidate_base_dirs = [
    "../data/osic-pulmonary-fibrosis-progression",
    "../input/osic-pulmonary-fibrosis-progression",
]
BASE_DIR = None
for d in candidate_base_dirs:
    if os.path.isdir(d):
        BASE_DIR = d
        break
if BASE_DIR is None:
    BASE_DIR = candidate_base_dirs[-1]

data_train = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
data_test = pd.read_csv(os.path.join(BASE_DIR, "test.csv"))
submission = pd.read_csv(os.path.join(BASE_DIR, "sample_submission.csv"))



## === cell 7
submission_template = submission.copy()

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
        "Patient_Week",
    ],
]
submission = merge.loc[:, ["Patient_Week", "base_FVC", "Confidence"]]
submission = submission.rename(columns={"base_FVC": "FVC"})



## === cell 8
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
    columns=["Patient", "base_Weeks", "base_FVC", "Age", "Healthy-FVC"]
    + FE1
    + ["Week", "Percent", "Sex", "SmokingStatus", "Patient_Week"]
)
npData = pd.concat([npData, data], ignore_index=True, sort=False)
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
        "Percent",
        "Sex",
        "SmokingStatus",
        "Patient_Week",
    ]
]
del npData



## === cell 9
model = Combined_NET()

candidate_weight_paths = [
    "../input/tt-03-694/Epoch3_Score6.942932319956899_Acc0.9245264870441513.pth",
    "./weights.pth",
    "./model.pth",
    "./best.pth",
]
weight_path = None
for p in candidate_weight_paths:
    if os.path.exists(p):
        weight_path = p
        break

has_weights = weight_path is not None
if has_weights:
    state = torch.load(weight_path, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            new_state[nk] = v
        state = new_state
    model.load_state_dict(state, strict=True)

print("Using weights:", has_weights, "| path:", weight_path)



## === cell 10
if has_weights:
    try:
        test = make_eval_data(data_test.copy(), model)
    except Exception as e:
        print("Inference failed, falling back to baseline predictor. Error:", repr(e))
        test = data_test.copy()
        test["FVC"] = test["base_FVC"].astype(float)
        test["Confidence"] = 70.0
else:
    test = data_test.copy()
    test["FVC"] = test["base_FVC"].astype(float)
    test["Confidence"] = 70.0



## === cell 11
for nid in test.Patient.unique():
    index = test[(test.Patient == nid) & (test.Week == test.base_Weeks)].index.values
    if len(index) > 0:
        test.iloc[index[0], test.columns.get_loc("FVC")] = float(
            test.iloc[index[0], test.columns.get_loc("base_FVC")]
        )
        test.iloc[index[0], test.columns.get_loc("Confidence")] = 70.0



## === cell 12
test.loc[test.Confidence < 70, "Confidence"] = 70.0



## === cell 13
pred_keyed = test[["Patient_Week", "FVC", "Confidence"]].copy()
pred_keyed["FVC"] = pred_keyed["FVC"].astype(float)
pred_keyed["Confidence"] = pred_keyed["Confidence"].astype(float)
pred_keyed.loc[pred_keyed.Confidence < 70, "Confidence"] = 70.0

submission = submission_template[["Patient_Week"]].merge(
    pred_keyed, on="Patient_Week", how="left"
)
submission["FVC"] = submission["FVC"].astype(float).fillna(submission["FVC"].median())
submission["Confidence"] = submission["Confidence"].astype(float).fillna(70.0)
submission.loc[submission.Confidence < 70, "Confidence"] = 70.0

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
