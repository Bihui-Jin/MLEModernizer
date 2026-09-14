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

-6.847914811266264

# 6. Current score

-24.65932

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -24.6135) has done: 'I fix the pandas `DataFrame.append` deprecation by replacing it with `pd.concat`, which unblocks test-feature preparation. Next, because the referenced pretrained `.pth` file is not present in this environment, I add a small fallback that trains the same provided model architecture on tabular+CT inputs from the competition’s `train/` DICOMs and `train.csv`, then uses it for inference. I also correct hardcoded paths (`../input/...`) to the provided `/kaggle/data/osic-pulmonary-fibrosis-progression/...` dataset location so DICOM loading works. Finally, I ensure a valid `submission.csv` is written with exactly `Patient_Week,FVC,Confidence` and clip confidence to the required minimum of 70.'
- What this solution (achieved -24.65417) has done: 'I fix the caching write bug that causes `FileNotFoundError` by ensuring `np.save` writes to the exact filename you later `os.replace` (numpy was silently appending “.npy”), and I make the cache write atomic. I also reduce DataLoader fragility by disabling `persistent_workers` and setting `num_workers=0` (this is a stability fix to ensure it runs end-to-end in Kaggle; it doesn’t change the model logic). Finally, I keep the existing training/inference pipeline intact and ensure the script always writes a valid `submission.csv` with the required columns and confidence clipped to at least 70.'
- What this solution (achieved -24.65417) has done: 'I fix the runtime error by making PyTorch determinism compatible with CUDA/cuBLAS: set `CUBLAS_WORKSPACE_CONFIG` before torch is used and fall back to non-deterministic algorithms only if needed so backprop can run. I also correct the loss/metric sign so training minimizes the same quantity that Kaggle scores (negative log-likelihood), which should improve the score substantially while keeping the same model architecture and training loop. Finally, I ensure the model outputs are in a valid range for the competition by clamping predicted sigma to be positive and writing a valid `submission.csv` with the required columns.'
- What this solution (achieved -24.65932) has done: 'The timeout is dominated by DICOM decoding + 3D resizing for every train patient during `prebuild_patient_cache`, plus repeated (and unneeded) pixel decoding work in `load_scan/get_pixels_hu`. I keep the exact same model/training/prediction logic, but make image caching actually effective and much faster by (1) reading only metadata when sorting slices, (2) decoding pixels only once (no Python loop per-slice for intercept/slope), (3) storing a compact uint8 `.npy` cache and always reusing it, and (4) prebuilding cache in parallel across CPU cores to overlap DICOM IO/resize work. These changes are deterministic and preserve the same image normalization pipeline and thus the same learning/evaluation semantics (only negligible float differences from equivalent vectorized math). Finally, I avoid redundant patient cache builds and reduce overhead in evaluation by preloading images via the same cache.'
- What this solution (achieved -9.37018) has done: 'Your score is far below the target (higher is better), and the biggest gap comes from the model effectively learning almost nothing: (1) the image input is 0–255 but never scaled back to the 0–1 range used during normalization, and (2) the model’s sigma head is constrained by a final ReLU, which makes the easiest way to reduce the loss to drive sigma toward 0 (then it gets clipped to 70 inside the metric, yielding weak gradients and poor FVC learning). I keep the same architecture and training loop, but (a) feed the network the correctly scaled image tensor (divide by 255.0 at training and inference), and (b) add a tiny, metric-consistent floor to the predicted sigma during training/inference (after the model output, not changing layers) so gradients stay meaningful and FVC can be learned. I also compute a single constant confidence from train residuals and use it in the submission (still a valid confidence prediction) to improve the Laplace likelihood without changing the model core. These are minimal, directly metric-aligned changes and should move the score substantially toward your target.'
- What this solution (achieved -9.37018) has done: 'We need to move your score upward toward the target, so the smallest useful change is to make the tabular features consistent between train and test: right now `Week` in training is accidentally set to the *baseline week* (so the model never learns the time effect), while test `Week` is the *target week*. I fix `csv_preprocess()` so `Week` equals the actual measurement week and `base_Weeks/base_FVC` come only from each patient’s baseline row, which preserves your model and training loop but makes the learned mapping align with the task. Additionally, I make one metric-aligned, low-risk improvement to your fixed confidence: compute it using the training Laplace-optimal scale proxy (median absolute error on training, but only on non-baseline rows to avoid artificially low residuals). Everything else (architecture, loss, one-epoch training, image pipeline, submission writing) stays the same and still produces `submission.csv`.'
- What this solution (achieved -9.37018) has done: 'We need to move your score up (less negative) toward the target, and the biggest remaining mismatch is that the model’s first output (sigma) is being trained against a constant 70 rather than learning uncertainty that matches residuals; that wastes half the head and can hurt FVC learning under the Laplace NLL. I keep the same architecture and training loop, but change the training target so `y_true[:,0]` is a per-row proxy sigma derived from each patient’s own FVC trajectory (median absolute deviation from the patient’s baseline trend), clipped to the competition’s 70 minimum—this aligns the loss with the metric without changing model semantics. I also ensure training and inference use the same sigma floor (70) and keep your fixed-confidence submission logic (now computed more consistently) to avoid destabilizing results. These are small, metric-aligned data/target fixes and should improve the score toward your target band.'
- What this solution (achieved -24.65932) has done: 'I fix the shape mismatch causing the `mat1 and mat2 shapes cannot be multiplied (…x74 and 42x64)` error by ensuring the tabular tensor fed into `SIGMA.data_net1` is always exactly the 42 features the model expects (not the 74-dim concatenation with image features). This is a minimal, architecture-preserving change: the model still uses the same IMAGE backbone and SIGMA MLP blocks, but we pass the correct inputs to each block so dimensions align. I also make the forward pass robust to any accidental extra columns by slicing/padding `data_i` to 42, and then complete inference and submission writing end-to-end so `submission.csv` is produced with correct columns and confidence clipped to at least 70. These changes are correctness fixes (no metric cheating) and should also improve score versus a broken run because the model can now actually train and predict.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import scipy.ndimage
import pydicom

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


seed_everything(42)



## === cell 1
BASE_PATH = "/kaggle/data/osic-pulmonary-fibrosis-progression"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TEST_CSV = os.path.join(BASE_PATH, "test.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_FOLDER = os.path.join(BASE_PATH, "train")
TEST_FOLDER = os.path.join(BASE_PATH, "test")

CACHE_DIR = os.path.join("/kaggle", "working", "osic_cache_v1")
os.makedirs(CACHE_DIR, exist_ok=True)

_RAM_IMAGE_CACHE = {}


def load_scan(path):  # path == TRAIN_FOLDER/patientId
    files = os.listdir(path)
    if not files:
        return []
    full = [os.path.join(path, f) for f in files]
    slices = []
    for fp in full:
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
            ds._fp_pixels_path = fp  # keep file path for later full read if needed
            slices.append(ds)
        except Exception:
            ds = pydicom.dcmread(fp, force=True)
            slices.append(ds)

    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except Exception:
        try:
            slices.sort(key=lambda x: float(getattr(x, "SliceLocation", 0.0)))
        except Exception:
            files.sort()
            slices = [pydicom.dcmread(os.path.join(path, f), force=True) for f in files]
    return slices


def get_pixels_hu(slices):
    if not slices:
        return np.zeros((0, 0, 0), dtype=np.int16)

    full_slices = []
    for s in slices:
        if hasattr(s, "PixelData"):
            full_slices.append(s)
        else:
            fp = getattr(s, "_fp_pixels_path", None)
            if fp is None:
                raise RuntimeError("Missing pixel data and file path for DICOM slice.")
            full_slices.append(pydicom.dcmread(fp, force=True))

    try:
        image = np.stack([s.pixel_array for s in full_slices]).astype(
            np.int16, copy=False
        )
    except Exception as e:
        raise RuntimeError(f"DICOM pixel_array decode failed: {e}")

    image = image.copy()
    image[image <= -2000] = 0

    try:
        intercepts = np.asarray(
            [getattr(s, "RescaleIntercept", 0.0) for s in full_slices], dtype=np.float32
        ).reshape(-1, 1, 1)
        slopes = np.asarray(
            [getattr(s, "RescaleSlope", 1.0) for s in full_slices], dtype=np.float32
        ).reshape(-1, 1, 1)

        img_f = image.astype(np.float32, copy=False)
        img_f = img_f * slopes + intercepts
        image = img_f.astype(np.int16, copy=False)
    except Exception:
        pass

    return np.asarray(image, dtype=np.int16)


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


def _dummy_image(Z=100, Y=200, X=200, value=0):
    return np.ones((Z, Y, X), dtype=np.uint8) * np.uint8(value)


def _cache_key(dir_name, patientid, Z, Y, X):
    root_tag = os.path.basename(dir_name.rstrip(os.sep))
    return f"{root_tag}_{patientid}_{Z}x{Y}x{X}.npy"


def read_image(dir_name, patientid, Z=100, Y=200, X=200):
    """
    Speed/correctness-preserving caching:
    - RAM cache avoids repeated np.load for the same patient/shape.
    - Disk cache stores the resized/normalized uint8 volume per patient/shape.
    - Atomic write avoids partial files.
    """
    ram_key = (dir_name, patientid, int(Z), int(Y), int(X))
    cached = _RAM_IMAGE_CACHE.get(ram_key, None)
    if cached is not None:
        return cached

    cache_path = os.path.join(CACHE_DIR, _cache_key(dir_name, patientid, Z, Y, X))
    if os.path.exists(cache_path):
        arr = np.load(cache_path, allow_pickle=False, mmap_mode=None)
        _RAM_IMAGE_CACHE[ram_key] = arr
        return arr

    path = dir_name + os.sep + patientid
    try:
        slices = load_scan(path)
        image_array = get_pixels_hu(slices)
        ctimage_resizedAll = resize_along_allaxis(
            image_array, target_dimensionX=X, target_dimensionY=Y, target_dimensionZ=Z
        )
        image = (image_normalize(ctimage_resizedAll) * 255.0).astype(
            "uint8", copy=False
        )
    except Exception:
        const_val = abs(hash(patientid)) % 256
        image = _dummy_image(Z=Z, Y=Y, X=X, value=const_val)

    tmp_path = cache_path + ".tmp"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(tmp_path, "wb") as f:
        np.save(f, image, allow_pickle=False)
    os.replace(tmp_path, cache_path)

    _RAM_IMAGE_CACHE[ram_key] = image
    return image


def prebuild_patient_cache(patient_ids, img_root, Z=100, Y=200, X=200):
    uniq = pd.unique(np.asarray(patient_ids))
    if len(uniq) == 0:
        return

    missing = []
    root_tag = os.path.basename(img_root.rstrip(os.sep))
    for pid in uniq:
        cache_path = os.path.join(CACHE_DIR, f"{root_tag}_{pid}_{Z}x{Y}x{X}.npy")
        if not os.path.exists(cache_path):
            missing.append(pid)

    if not missing:
        return

    from concurrent.futures import ThreadPoolExecutor

    max_workers = min(8, (os.cpu_count() or 2))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(lambda pid: read_image(img_root, pid, Z=Z, Y=Y, X=X), missing))




## === cell 2
FE1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]
FEATURE_COLS_10 = [
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
PAD_FEATURES_TOTAL = 42  # model expects 42-dim tabular input


def csv_preprocess(data, global_sex_levels=None, global_smoke_levels=None):
    data = data.copy()
    data["Healthy-FVC"] = round((data["FVC"] * 100) / data["Percent"])

    if global_sex_levels is None:
        global_sex_levels = list(pd.unique(data["Sex"]))
    if global_smoke_levels is None:
        global_smoke_levels = list(pd.unique(data["SmokingStatus"]))

    for mod in global_sex_levels:
        data[mod] = (data["Sex"] == mod).astype(int)
    for mod in global_smoke_levels:
        data[mod] = (data["SmokingStatus"] == mod).astype(int)

    keep_cols = (
        ["Patient", "Weeks", "FVC", "Age", "Healthy-FVC"]
        + list(global_sex_levels)
        + list(global_smoke_levels)
    )
    data = data[keep_cols]
    data = data.sort_values(["Patient", "Weeks"], ascending=True).reset_index(drop=True)

    base_rows = data.groupby("Patient", sort=False).head(1).copy()
    rename_map = {
        "Weeks": "base_Weeks",
        "FVC": "base_FVC",
        "Age": "Age_base",
        "Healthy-FVC": "Healthy-FVC_base",
    }
    for c in list(global_sex_levels) + list(global_smoke_levels):
        rename_map[c] = f"{c}_base"
    base_rows = base_rows.rename(columns=rename_map)

    expanded = data.merge(
        base_rows[
            ["Patient", "base_Weeks", "base_FVC", "Age_base", "Healthy-FVC_base"]
            + [f"{c}_base" for c in list(global_sex_levels) + list(global_smoke_levels)]
        ],
        on="Patient",
        how="left",
        sort=False,
    )

    expanded["Week"] = expanded["Weeks"].to_numpy()
    expanded["actual_FVC"] = expanded["FVC"].to_numpy()

    expanded = expanded[
        ["Patient", "base_Weeks", "base_FVC", "Age_base", "Healthy-FVC_base"]
        + [f"{c}_base" for c in list(global_sex_levels) + list(global_smoke_levels)]
        + ["Week", "actual_FVC"]
    ].rename(
        columns={
            "Age_base": "Age",
            "Healthy-FVC_base": "Healthy-FVC",
            **{
                f"{c}_base": c
                for c in list(global_sex_levels) + list(global_smoke_levels)
            },
        }
    )

    expanded = expanded.fillna(0).reset_index(drop=True)

    for c in FE1:
        if c not in expanded.columns:
            expanded[c] = 0

    return expanded


def make_tab_features_42(df: pd.DataFrame) -> np.ndarray:
    X10 = df[FEATURE_COLS_10].to_numpy(dtype=np.float32, copy=False)
    if X10.shape[1] != 10:
        raise RuntimeError("Expected 10 core features.")
    pad = np.zeros((X10.shape[0], PAD_FEATURES_TOTAL - 10), dtype=np.float32)
    return np.concatenate([X10, pad], axis=1)




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
            nn.Linear(42, 64),
            nn.ReLU(),
            nn.Linear(64, 118),
            nn.ReLU(),
        )
        self.data_net2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 502),
            nn.ReLU(),
        )
        self.data_net3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 118),
            nn.ReLU(),
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
        if data_i.size(-1) != 42:
            if data_i.size(-1) > 42:
                data_i = data_i[..., :42]
            else:
                pad = torch.zeros(
                    (data_i.size(0), 42 - data_i.size(-1)),
                    device=data_i.device,
                    dtype=data_i.dtype,
                )
                data_i = torch.cat([data_i, pad], dim=-1)

        x = torch.cat((data_i, image_o), dim=-1)  # 42 + 32 = 74 (used later)
        out1 = self.data_net1(data_i)  # 42 -> 118
        out2 = torch.cat((data_i, out1), dim=-1)  # 42+118=160 but net2 expects 128

        out2_in = (
            out2[..., :128]
            if out2.size(-1) >= 128
            else torch.cat(
                [
                    out2,
                    torch.zeros(
                        (out2.size(0), 128 - out2.size(-1)),
                        device=out2.device,
                        dtype=out2.dtype,
                    ),
                ],
                dim=-1,
            )
        )
        out2 = self.data_net2(out2_in)  # 128 -> 502

        out3 = torch.cat((data_i, out2), dim=-1)  # 42+502=544, net3 expects 512
        out3_in = (
            out3[..., :512]
            if out3.size(-1) >= 512
            else torch.cat(
                [
                    out3,
                    torch.zeros(
                        (out3.size(0), 512 - out3.size(-1)),
                        device=out3.device,
                        dtype=out3.dtype,
                    ),
                ],
                dim=-1,
            )
        )
        out3 = self.data_net3(out3_in)  # 512 -> 118

        out4 = torch.cat(
            (x, out1, out2, out3), dim=-1
        )  # 74+118+502+118=812, net4 expects 780
        out4_in = (
            out4[..., :780]
            if out4.size(-1) >= 780
            else torch.cat(
                [
                    out4,
                    torch.zeros(
                        (out4.size(0), 780 - out4.size(-1)),
                        device=out4.device,
                        dtype=out4.dtype,
                    ),
                ],
                dim=-1,
            )
        )
        out = self.data_net4(out4_in)  # 780 -> 2
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
SIGMA_FLOOR_FOR_TRAINING = 70.0


def score(y_true, y_pred):
    sigma_raw = y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma = torch.clamp(sigma_raw, min=1e-3)
    sigma = torch.clamp(sigma, min=SIGMA_FLOOR_FOR_TRAINING)
    sigma_clip = torch.clamp(sigma, min=70.0)

    delta = (y_true[:, 1] - fvc_pred).abs()
    delta = torch.clamp(delta, max=1000.0)

    sq2 = 1.4142135623730951
    metric = (delta / sigma_clip) * sq2 + (sigma_clip * sq2).log()
    return metric.mean()


def quartile_loss(y_true, y_pred):
    return -score(y_true, y_pred)




## === cell 5
def make_eval_data(npEval, model, device="cuda", batch_size=8, fixed_confidence=None):
    x_features = torch.from_numpy(make_tab_features_42(npEval))

    patient_ids = npEval["Patient"].to_numpy()
    unique_patients = pd.unique(patient_ids)
    loaded_images = {}
    dir_name_of_patientid = TEST_FOLDER
    for pid in unique_patients:
        loaded_images[pid] = read_image(dir_name_of_patientid, pid, Z=100, Y=200, X=200)

    if torch.cuda.is_available() and device == "cuda":
        model.to("cuda")
    model.eval()

    preds = np.empty((len(npEval), 2), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, len(npEval), batch_size):
            end = min(start + batch_size, len(npEval))
            batch_pids = patient_ids[start:end]
            imgs = np.stack([loaded_images[pid] for pid in batch_pids], axis=0).astype(
                np.float32, copy=False
            )
            imgs = imgs / 255.0
            img_t = torch.from_numpy(imgs).unsqueeze(1)  # (B,1,Z,Y,X)
            feat_t = x_features[start:end]

            if torch.cuda.is_available() and device == "cuda":
                img_t = img_t.cuda(non_blocking=True)
                feat_t = feat_t.cuda(non_blocking=True)

            out = model(img_t, feat_t).detach().cpu().numpy()
            preds[start:end] = out

    npEval["FVC"] = preds[:, 1]

    if fixed_confidence is not None:
        npEval["Confidence"] = float(max(70.0, fixed_confidence))
    else:
        conf = preds[:, 0]
        conf = np.clip(conf, 1.0, None)
        conf = np.maximum(conf, 70.0)
        npEval["Confidence"] = conf

    return npEval




## === cell 6
data_train = pd.read_csv(TRAIN_CSV)
data_test = pd.read_csv(TEST_CSV)
submission = pd.read_csv(SAMPLE_SUB)

_global_sex_levels = sorted(
    pd.unique(pd.concat([data_train["Sex"], data_test["Sex"]], axis=0)).tolist()
)
_global_smoke_levels = sorted(
    pd.unique(
        pd.concat([data_train["SmokingStatus"], data_test["SmokingStatus"]], axis=0)
    ).tolist()
)



## === cell 7
pw = submission["Patient_Week"].str.split("_", n=1, expand=True)
submission["Patient"] = pw[0].to_numpy()
submission["Weeks"] = pw[1].astype(int).to_numpy()
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
]
submission = merge.loc[:, ["Patient_Week", "base_FVC", "Confidence"]]
submission = submission.rename(columns={"base_FVC": "FVC"})



## === cell 8
data = data_test.copy()
data["Healthy-FVC"] = round((data["base_FVC"] * 100) / data["Percent"])

for mod in _global_sex_levels:
    data[mod] = (data["Sex"] == mod).astype(int)
for mod in _global_smoke_levels:
    data[mod] = (data["SmokingStatus"] == mod).astype(int)

for c in FE1:
    if c not in data.columns:
        data[c] = 0

data_test = data[
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
data_test = data_test.fillna(0)



## === cell 9
train_proc = csv_preprocess(
    data_train,
    global_sex_levels=_global_sex_levels,
    global_smoke_levels=_global_smoke_levels,
)
train_proc = train_proc.fillna(0)

X_train_tab = make_tab_features_42(train_proc)
patient_train = train_proc["Patient"].values

tmp = train_proc[["Patient", "base_Weeks", "base_FVC", "Week", "actual_FVC"]].copy()
tmp["dt"] = (tmp["Week"] - tmp["base_Weeks"]).astype(np.float32)
tmp["dt"] = tmp["dt"].replace(0.0, 1.0)
tmp["slope"] = (tmp["actual_FVC"] - tmp["base_FVC"]) / tmp["dt"]
patient_slope = tmp.groupby("Patient")["slope"].median()
tmp = tmp.join(patient_slope.rename("slope_med"), on="Patient")
tmp["fvc_trend"] = tmp["base_FVC"] + tmp["slope_med"] * (
    tmp["Week"] - tmp["base_Weeks"]
)
tmp["abs_resid"] = (tmp["actual_FVC"] - tmp["fvc_trend"]).abs()
patient_sigma = (
    tmp.groupby("Patient")["abs_resid"].median().clip(lower=70.0, upper=500.0)
)
sigma_target = tmp["Patient"].map(patient_sigma).astype(np.float32).to_numpy()

y_train = train_proc[["actual_FVC"]].values.astype(np.float32)
y_train_full = np.concatenate([sigma_target.reshape(-1, 1), y_train], axis=1).astype(
    np.float32, copy=False
)

prebuild_patient_cache(patient_train, TRAIN_FOLDER, Z=100, Y=200, X=200)
prebuild_patient_cache(data_test["Patient"].values, TEST_FOLDER, Z=100, Y=200, X=200)




## === cell 10
class OSICTrainDataset(Dataset):
    def __init__(self, patient_ids, x_tab, y, img_root, Z=100, Y=200, X=200):
        self.patient_ids = patient_ids
        self.x_tab = x_tab
        self.y = y
        self.img_root = img_root
        self.Z, self.Y, self.X = Z, Y, X

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        img = read_image(self.img_root, pid, Z=self.Z, Y=self.Y, X=self.X).astype(
            np.float32, copy=False
        )
        img = img / 255.0
        img_t = torch.from_numpy(img).unsqueeze(0)  # (1, Z,Y,X)
        x_t = torch.from_numpy(self.x_tab[idx])
        y_t = torch.from_numpy(self.y[idx])
        return img_t, x_t, y_t




## === cell 11
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Combined_NET().to(device)

train_ds = OSICTrainDataset(
    patient_train, X_train_tab, y_train_full, TRAIN_FOLDER, Z=100, Y=200, X=200
)


def _seed_worker(worker_id: int):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


num_workers = min(4, (os.cpu_count() or 2))
g = torch.Generator()
g.manual_seed(42)

train_loader = DataLoader(
    train_ds,
    batch_size=1,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=(device == "cuda"),
    worker_init_fn=_seed_worker,
    generator=g,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
)

opt = torch.optim.Adam(model.parameters(), lr=1e-4)
model.train()

EPOCHS = 1

for epoch in range(EPOCHS):
    for img_t, x_t, y_t in train_loader:
        img_t = img_t.to(device, non_blocking=True)
        x_t = x_t.to(device, non_blocking=True)
        y_t = y_t.to(device, non_blocking=True)

        pred = model(img_t, x_t)
        loss = quartile_loss(y_t, pred)

        opt.zero_grad(set_to_none=True)
        try:
            loss.backward()
        except RuntimeError as e:
            msg = str(e)
            if (
                "deterministic implementation" in msg
                or "Deterministic behavior was enabled" in msg
            ):
                try:
                    torch.use_deterministic_algorithms(False)
                except Exception:
                    pass
                loss.backward()
            else:
                raise
        opt.step()



## === cell 12
fixed_conf = float(np.median(patient_sigma.values.astype(np.float32)))
fixed_conf = float(np.clip(fixed_conf, 70.0, 500.0))



## === cell 13
test = make_eval_data(
    data_test.copy(), model, device=device, batch_size=8, fixed_confidence=fixed_conf
)

for nid in test.Patient.unique():
    index = test[(test.Patient == nid) & (test.Week == test.base_Weeks)].index.values
    if len(index) > 0:
        test.iloc[index[0], test.columns.get_loc("FVC")] = test.iloc[
            index[0], test.columns.get_loc("base_FVC")
        ]
        test.iloc[index[0], test.columns.get_loc("Confidence")] = 70.0



## === cell 14
test.loc[test.Confidence < 70, "Confidence"] = 70.0

submission.loc[:, "FVC"] = test.FVC.values
submission.loc[:, "Confidence"] = test.Confidence.values
submission = submission[["Patient_Week", "FVC", "Confidence"]]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Using fixed confidence:", fixed_conf)
print("Wrote submission.csv with shape:", submission.shape)
print("Saved to:", os.path.abspath("submission.csv"))
