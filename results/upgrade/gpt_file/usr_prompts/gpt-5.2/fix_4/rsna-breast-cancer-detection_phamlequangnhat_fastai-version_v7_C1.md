# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect breast cancer in mammograms.

## Metric
[Probabilistic F1 score](https://aclanthology.org/2020.eval4nlp-1.9.pdf) (pF1). This extension of the traditional F score accepts probabilities instead of binary classifications. 

With pX as the probabilistic version of X:

$$
pF_1 = 2 \frac{pPrecision \cdot pRecall}{pPrecision + pRecall}
$$

where:

$$
pPrecision = \frac{pTP}{pTP + pFP}
$$

$$
pRecall = \frac{pTP}{TP + FN}
$$

## Submission Format
For each `prediction_id`, you should predict the likelihood of cancer in the corresponding `cancer` column. The submission file should have the following format:

```
prediction_id,cancer
0-L,0
0-R,0.5
0-R,0.5
1-L,1
...
# Dataset

**[train/test]_images/[patient_id]/[image_id].dcm** The mammograms, in dicom format. You can expect roughly 8,000 patients in the hidden test set. There are usually but not always 4 images per patient. Note that many of the images use the jpeg 2000 format which may you may need special libraries to load.

**sample_submission.csv** A valid sample submission.

**[train/test].csv** Metadata for each patient and image. Only the first few rows of the test set are available for download.

- `site_id` - ID code for the source hospital.
- `patient_id` - ID code for the patient.
- `image_id` - ID code for the image.
- `laterality` - Whether the image is of the left or right breast.
- `view` - The orientation of the image. The default for a screening exam is to capture two views per breast.
- `age` - The patient's age in years.
- `implant` - Whether or not the patient had breast implants. Site 1 only provides breast implant information at the patient level, not at the breast level.
- `density` - A rating for how dense the breast tissue is, with A being the least dense and D being the most dense. Extremely dense tissue can make diagnosis more difficult. Only provided for train.
- `machine_id` - An ID code for the imaging device.
- `cancer` - Whether or not the breast was positive for malignant cancer. The target value. Only provided for train.
- `biopsy` - Whether or not a follow-up biopsy was performed on the breast. Only provided for train.
- `invasive` - If the breast is positive for cancer, whether or not the cancer proved to be invasive. Only provided for train.
- `BIRADS` - 0 if the breast required follow-up, 1 if the breast was rated as negative for cancer, and 2 if the breast was rated as normal. Only provided for train.
- `prediction_id` - The ID for the matching submission row. Multiple images will share the same prediction ID. Test only.
- `difficult_negative_case` - True if the case was unusually difficult. Only provided for train.

# 2. Python version

3.11

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (190 lines)
            sample_submission.csv (2385 lines)
            sample_submission.csv.zip (6.5 kB)
            test.csv (5475 lines)
            test.csv.zip (60.6 kB)
            test.zip (160 Bytes)
            test_images.zip (29.1 GB)
            train.csv (49233 lines)
            train.csv.zip (513.7 kB)
            train.zip (162 Bytes)
            train_images.zip (260.7 GB)
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
            test_images/
                10116/
                    1470873094.dcm (8.5 MB)
                    472095321.dcm (5.6 MB)
                    ... and 2 other files
                10130/
                    1013166704.dcm (9.7 MB)
                    1165309236.dcm (8.7 MB)
                    ... and 5 other files
                ... and 1191 other folders
            train_images/
                10006/
                    1459541791.dcm (4.4 MB)
                    1864590858.dcm (4.0 MB)
                    ... and 2 other files
                10011/
                    1031443799.dcm (2.1 MB)
                    220375232.dcm (1.7 MB)
                    ... and 2 other files
                ... and 10720 other folders
        input/
            description.md (190 lines)
            sample_submission.csv (2385 lines)
            sample_submission.csv.zip (6.5 kB)
            test.csv (5475 lines)
            test.csv.zip (60.6 kB)
            test.zip (160 Bytes)
            test_images.zip (29.1 GB)
            train.csv (49233 lines)
            train.csv.zip (513.7 kB)
            train.zip (162 Bytes)
            train_images.zip (260.7 GB)
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
            test_images/
                10116/
                    1470873094.dcm (8.5 MB)
                    472095321.dcm (5.6 MB)
                    ... and 2 other files
                10130/
                    1013166704.dcm (9.7 MB)
                    1165309236.dcm (8.7 MB)
                    ... and 5 other files
                ... and 1191 other folders
            train_images/
                10006/
                    1459541791.dcm (4.4 MB)
                    1864590858.dcm (4.0 MB)
                    ... and 2 other files
                10011/
                    1031443799.dcm (2.1 MB)
                    220375232.dcm (1.7 MB)
                    ... and 2 other files
                ... and 10720 other folders
        working/
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
```

-> data/rsna-breast-cancer-detection/sample_submission.csv has 2384 rows and 2 columns.
The columns are: prediction_id, cancer

-> data/rsna-breast-cancer-detection/test.csv has 5474 rows and 9 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, implant, machine_id, prediction_id

-> data/rsna-breast-cancer-detection/train.csv has 49232 rows and 14 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, cancer, biopsy, invasive, BIRADS, implant, density, machine_id, difficult_negative_case

-> data/sample_submission.csv has 2384 rows and 2 columns.
The columns are: prediction_id, cancer

-> data/test.csv has 5474 rows and 9 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, implant, machine_id, prediction_id

-> data/train.csv has 49232 rows and 14 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, cancer, biopsy, invasive, BIRADS, implant, density, machine_id, difficult_negative_case

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os, sys, subprocess, textwrap, shutil, glob, json, math, random, time
from pathlib import Path


def run(cmd, cwd=None, check=True):
    """Run a shell command reliably in a .py/script environment."""
    print(f"[cmd] {cmd}")
    p = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(p.stdout)
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed with code {p.returncode}: {cmd}")
    return p


KAGGLE_INPUT = "/kaggle/input/rsna-breast-cancer-detection"
assert os.path.exists(KAGGLE_INPUT), f"Missing expected input dir: {KAGGLE_INPUT}"


## === cell 1
try:
    import pylibjpeg  # noqa: F401
except Exception as e:
    print("pylibjpeg not available; continuing without it. Error:", repr(e))


## === cell 2
timm_bundle = "/kaggle/input/timm-with-dependencies"
dicomsdl_whl = "/kaggle/input/discom/discom/dicomsdl-0.109.1-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl"

if os.path.exists(timm_bundle):
    os.makedirs("timm-with-dependencies", exist_ok=True)
    run(
        f"unzip -q '{timm_bundle}/timm with dependencies/timm_all' -d timm-with-dependencies",
        check=False,
    )
    run("pip install --no-index --find-links timm-with-dependencies timm", check=False)
else:
    print("Optional timm bundle not found; skipping timm install.")

if os.path.exists(dicomsdl_whl):
    print("dicomsdl wheel found but is not compatible with Python 3.11; skipping.")
else:
    print("Optional dicomsdl wheel not found; skipping.")


## === cell 3
DEBUG = False
SEED = 42
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


## === cell 4
import cv2
import numpy as np
import pandas as pd
import pydicom

from tqdm.auto import tqdm
from joblib import Parallel, delayed

print("Versions:")
print("python:", sys.version.split()[0])
print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("pydicom:", pydicom.__version__)
print("cv2:", cv2.__version__)


## === cell 5
test_images = glob.glob(f"{KAGGLE_INPUT}/test_images/*/*.dcm")
if DEBUG:
    test_images = glob.glob(f"{KAGGLE_INPUT}/train_images/10042/*.dcm")

print("Number of images :", len(test_images))
assert len(test_images) > 0, "No DICOM images found; check input paths."


## === cell 6
SAVE_FOLDER = "/kaggle/tmp/output/"
SIZE = (1024, 512)  # (width, height) for cv2.resize
EXTENSION = "png"

os.makedirs(SAVE_FOLDER, exist_ok=True)




## === cell 7
def _safe_normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    mn = float(np.min(img))
    mx = float(np.max(img))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(img, dtype=np.float32)
    return (img - mn) / (mx - mn)


def _read_dicom_pixel_array(path: str) -> tuple[np.ndarray, str]:
    """
    Returns (image_array_float01, photometric_interpretation).
    If decode fails, returns (None, None).
    """
    try:
        ds = pydicom.dcmread(path, force=True)
    except Exception:
        return None, None

    phot = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2"))

    try:
        img = ds.pixel_array  # may fail if compressed without handlers
    except Exception:
        return None, None

    img = _safe_normalize(img)

    if phot.upper() == "MONOCHROME1":
        img = 1.0 - img

    return img, phot


_PNG_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 1]


def process(f, out_path, size=(1024, 512), extension="png"):
    if os.path.exists(out_path):
        return True

    img, _ = _read_dicom_pixel_array(f)
    if img is None:
        return False

    if img.ndim > 2:
        img = img.squeeze()
        if img.ndim > 2:
            img = img[..., 0]

    img = cv2.resize(img, (int(size[0]), int(size[1])), interpolation=cv2.INTER_AREA)

    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    ok = cv2.imwrite(out_path, (img * 255).clip(0, 255).astype(np.uint8), _PNG_PARAMS)
    return bool(ok)




## === cell 8
to_convert = []
patients = set()

for f in test_images:
    patient = os.path.basename(os.path.dirname(f))
    image = os.path.splitext(os.path.basename(f))[0]
    patients.add(patient)
    out_path = os.path.join(SAVE_FOLDER, patient, f"{image}.{EXTENSION}")
    if not os.path.exists(out_path):
        to_convert.append((f, out_path))

for p in patients:
    os.makedirs(os.path.join(SAVE_FOLDER, p), exist_ok=True)

print(f"Already converted: {len(test_images) - len(to_convert)} / {len(test_images)}")
print(f"Remaining to convert: {len(to_convert)} / {len(test_images)}")


## === cell 9
if len(to_convert) > 0:
    n_jobs = min(8, max(1, (os.cpu_count() or 4)))
    batch_size = 16 if len(to_convert) > 256 else 4
    results = Parallel(n_jobs=n_jobs, prefer="threads", batch_size=batch_size)(
        delayed(process)(f, out_path, size=SIZE, extension=EXTENSION)
        for (f, out_path) in tqdm(to_convert, desc="Converting DICOM->PNG")
    )
    n_ok = int(np.sum(results)) + (len(test_images) - len(to_convert))
else:
    n_ok = len(test_images)

print(f"Converted {n_ok}/{len(test_images)} images to {EXTENSION} under {SAVE_FOLDER}")

if n_ok == 0:
    print(
        "Warning: 0 images converted. Likely due to missing JPEG2000 decode support. Will use fallback submission if needed."
    )


## === cell 10
TEST_OUTDIR = "/kaggle/tmp/test"
os.makedirs(TEST_OUTDIR, exist_ok=True)
print("Test output dir:", TEST_OUTDIR)


## === cell 11
FASTAI_BUNDLE = "/kaggle/input/fastai/fastai"
MODELS_DST = "/kaggle/tmp/models"

if os.path.exists(FASTAI_BUNDLE):
    os.makedirs(MODELS_DST, exist_ok=True)
    src_models = os.path.join(FASTAI_BUNDLE, "models")
    if os.path.exists(src_models):
        for item in os.listdir(src_models):
            s = os.path.join(src_models, item)
            d = os.path.join(MODELS_DST, item)
            if os.path.isdir(s):
                if os.path.exists(d):
                    shutil.rmtree(d)
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
        print("Copied models to:", MODELS_DST)
    else:
        print("fastai bundle found but models folder missing:", src_models)
else:
    print(
        "fastai bundle not found; inference script may not run. Will use fallback if no submission generated."
    )


## === cell 12
infer_py = os.path.join(FASTAI_BUNDLE, "infer.py")
submission_path = os.path.join(TEST_OUTDIR, "submission.csv")

if os.path.exists(infer_py):
    cmd = (
        f"python '{infer_py}' "
        f"--model '{MODELS_DST}' "
        f"--data '{SAVE_FOLDER}' "
        f"--csv '{submission_path}' "
        f"--threshold 0.25 "
        f"--split 4 "
        f"--stt 0 "
        f"--test_csv '{KAGGLE_INPUT}/test.csv'"
    )
    run(cmd, cwd=FASTAI_BUNDLE, check=False)
else:
    print("infer.py not found at:", infer_py)

print(
    "Expected inference submission:",
    submission_path,
    "exists?",
    os.path.exists(submission_path),
)


## === cell 13
sample_sub_path = f"{KAGGLE_INPUT}/sample_submission.csv"
test_csv_path = f"{KAGGLE_INPUT}/test.csv"

sample_sub = pd.read_csv(sample_sub_path)
assert {"prediction_id", "cancer"}.issubset(sample_sub.columns)

if os.path.exists(submission_path):
    df_1 = pd.read_csv(submission_path)
    if not {"prediction_id", "cancer"}.issubset(df_1.columns):
        print(
            "Inference submission missing required columns; using fallback sample submission."
        )
        sub = sample_sub.copy()
    else:
        sub = (
            df_1[["prediction_id", "cancer"]]
            .groupby("prediction_id", as_index=False)
            .max()
        )
        sub = sample_sub[["prediction_id"]].merge(sub, on="prediction_id", how="left")
        sub["cancer"] = sub["cancer"].fillna(0.0).clip(0.0, 1.0)
else:
    print("No inference submission produced; writing fallback submission (all zeros).")
    sub = sample_sub.copy()
    sub["cancer"] = 0.0

out_final = "/kaggle/working/submission.csv"
sub.to_csv(out_final, index=False)
print("Wrote:", out_final, "shape:", sub.shape)
print(sub.head())
assert out_final.endswith(".csv") and os.path.exists(out_final)
assert list(sub.columns) == ["prediction_id", "cancer"]
assert len(sub) == len(sample_sub), "Submission row count must match sample_submission."
