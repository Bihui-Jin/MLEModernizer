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

# 5. Target score

0.1157787128830565

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I fix the root cause of the crashes: your notebook is running in an environment where importing `numpy` (and thus `pandas`) is broken, so any code relying on them cannot run. To make the solution run end-to-end and still produce a valid `.csv` submission, I remove the hard dependency on NumPy/Pandas and write the submission using Python’s standard library (`csv`) by copying `sample_submission.csv` and setting `cancer` to 0.0 (a safe fallback when inference can’t run). I also keep your existing DICOM/PNG conversion and fastai inference attempt, but guard imports so the pipeline continues even if optional pieces fail. The final result always write `/kaggle/working/submission.csv` with the exact required columns.'

# 9. Code solution

## === cell 0
import os, subprocess, sys, shutil, csv, glob


def _run(cmd):
    try:
        subprocess.check_call(cmd, shell=True)
        return True
    except Exception as e:
        print(f"[WARN] Command failed (continuing): {cmd}\n  -> {e}")
        return False


for bad_path in ["timm-with-dependencies"]:
    if os.path.isdir(bad_path):
        shutil.rmtree(bad_path, ignore_errors=True)

try:
    import pylibjpeg  # noqa: F401
except Exception:
    _run("pip -q install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg")



## === cell 1
DEBUG = False



## === cell 2

try:
    import cv2  # optional (used for PNG conversion)
except Exception as e:
    cv2 = None
    print("[WARN] cv2 import failed; PNG conversion will be skipped:", e)

try:
    import pydicom  # optional (used for DICOM reading)
except Exception as e:
    pydicom = None
    print("[WARN] pydicom import failed; PNG conversion will be skipped:", e)

try:
    from tqdm.notebook import tqdm
except Exception:
    try:
        from tqdm import tqdm
    except Exception:

        def tqdm(x, **kwargs):  # minimal fallback
            return x


try:
    from joblib import Parallel, delayed
except Exception as e:
    Parallel = None
    delayed = None
    print("[WARN] joblib import failed; conversion will be single-threaded:", e)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/numpy/_core/__init__.py in <module>
     22 
     23 try:
---> 24     from . import multiarray
     25 except ImportError as exc:
     26     import sys

/usr/local/lib/python3.11/dist-packages/numpy/_core/multiarray.py in <module>
      9 import functools
     10 
---> 11 from . import _multiarray_umath, overrides
     12 from ._multiarray_umath import *  # noqa: F403
     13 

AttributeError: module 'numpy._globals' has no attribute '_signature_descriptor'

## === cell 3
DATA_ROOT = "/kaggle/input/rsna-breast-cancer-detection"

test_images = glob.glob(f"{DATA_ROOT}/test_images/*/*.dcm")
if DEBUG:
    test_images = glob.glob(f"{DATA_ROOT}/train_images/10042/*.dcm")

print("Number of images :", len(test_images))
assert len(test_images) > 0, "No DICOM files found. Check DATA_ROOT/path."



## === cell 4
SAVE_FOLDER = "/kaggle/tmp/output/"
SIZE = (1024, 512)  # (width, height)
EXTENSION = "png"

os.makedirs(SAVE_FOLDER, exist_ok=True)




## === cell 5
def process(f, size=(1024, 512), save_folder="", extension="png"):
    if pydicom is None or cv2 is None:
        return False
    try:
        patient = f.split("/")[-2]
        image = os.path.basename(f)[:-4]

        dicom = pydicom.dcmread(f, force=True)
        img = dicom.pixel_array  # may require pylibjpeg/openjpeg for JP2K

        mn, mx = float(img.min()), float(img.max())
        if mx > mn:
            img = (img - mn) / (mx - mn)
        else:
            img = img * 0.0

        if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
            img = 1.0 - img

        img = cv2.resize(img, (int(size[0]), int(size[1])))

        path = os.path.join(save_folder, f"{patient}")
        os.makedirs(path, exist_ok=True)
        out_path = os.path.join(path, f"{image}.{extension}")
        cv2.imwrite(out_path, (img * 255).clip(0, 255).astype("uint8"))
        return True
    except Exception:
        return False




## === cell 6
ok = 0
if pydicom is not None and cv2 is not None:
    if Parallel is not None and delayed is not None:
        results = Parallel(n_jobs=4)(
            delayed(process)(
                uid, size=SIZE, save_folder=SAVE_FOLDER, extension=EXTENSION
            )
            for uid in tqdm(test_images)
        )
        ok = sum(1 for r in results if r)
        print(f"Converted {ok}/{len(results)} images to PNG under {SAVE_FOLDER}")
    else:
        for uid in tqdm(test_images):
            ok += (
                1
                if process(uid, size=SIZE, save_folder=SAVE_FOLDER, extension=EXTENSION)
                else 0
            )
        print(f"Converted {ok}/{len(test_images)} images to PNG under {SAVE_FOLDER}")
else:
    print("[WARN] Skipping PNG conversion (missing pydicom/cv2).")



## === cell 7
if os.path.isdir(SAVE_FOLDER):
    patients = sorted(
        [
            d
            for d in os.listdir(SAVE_FOLDER)
            if os.path.isdir(os.path.join(SAVE_FOLDER, d))
        ]
    )
    print("Example patient folders:", patients[:5])
else:
    print("[WARN] SAVE_FOLDER not found:", SAVE_FOLDER)



## === cell 8
TEST_OUTDIR = "/kaggle/tmp/test"
os.makedirs(TEST_OUTDIR, exist_ok=True)



## === cell 9
_run("cp -r /kaggle/input/fastai/fastai/models /kaggle/tmp/models")



## === cell 10
infer_py = "/kaggle/input/fastai/fastai/infer.py"
model_dir = "/kaggle/tmp/models"
png_data_dir = SAVE_FOLDER
out_csv = f"{TEST_OUTDIR}/submission.csv"
test_csv = f"{DATA_ROOT}/test.csv"

infer_ok = False
if os.path.exists(infer_py) and os.path.isdir(model_dir) and os.path.exists(test_csv):
    cmd = (
        f'python "{infer_py}" '
        f'--model "{model_dir}" '
        f'--data "{png_data_dir}" '
        f'--csv "{out_csv}" '
        f"--threshold 0.41 "
        f"--split 4 "
        f"--stt 0 "
        f'--test_csv "{test_csv}"'
    )
    infer_ok = _run(cmd)
else:
    print("[WARN] Missing infer.py/model_dir/test_csv; skipping model inference.")
    print(" infer.py exists:", os.path.exists(infer_py))
    print(" model_dir exists:", os.path.isdir(model_dir))
    print(" test_csv exists:", os.path.exists(test_csv))

print("Inference attempted:", infer_ok, "-> output exists:", os.path.exists(out_csv))



## === cell 11
sample_path = f"{DATA_ROOT}/sample_submission.csv"
final_path = "/kaggle/working/submission.csv"


def read_submission_csv(path):
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        cols = reader.fieldnames or []
    return cols, rows


def write_submission_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prediction_id", "cancer"])
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {"prediction_id": r["prediction_id"], "cancer": r["cancer"]}
            )


sample_cols, sample_rows = read_submission_csv(sample_path)
if "prediction_id" not in sample_cols:
    raise RuntimeError("sample_submission.csv missing prediction_id column")
sample_pred_ids = [r["prediction_id"] for r in sample_rows]

pred_map = None
if os.path.exists(out_csv):
    out_cols, out_rows = read_submission_csv(out_csv)
    if ("prediction_id" in out_cols) and ("cancer" in out_cols):
        pred_map = {}
        sums = {}
        cnts = {}
        for r in out_rows:
            pid = str(r.get("prediction_id", ""))
            try:
                val = float(r.get("cancer", 0.0))
            except Exception:
                val = 0.0
            if val != val:  # NaN check
                val = 0.0
            if val < 0.0:
                val = 0.0
            if val > 1.0:
                val = 1.0
            sums[pid] = sums.get(pid, 0.0) + val
            cnts[pid] = cnts.get(pid, 0) + 1
        pred_map = {pid: (sums[pid] / max(cnts[pid], 1)) for pid in sums.keys()}
    else:
        print("[WARN] infer output missing required columns; falling back to zeros.")

final_rows = []
missing = 0
for pid in sample_pred_ids:
    if pred_map is None:
        val = 0.0
    else:
        if pid in pred_map:
            val = pred_map[pid]
        else:
            val = 0.0
            missing += 1
    final_rows.append({"prediction_id": pid, "cancer": float(val)})

write_submission_csv(final_path, final_rows)
print(
    "Wrote:",
    final_path,
    "rows:",
    len(final_rows),
    "missing_pred_ids_from_infer:",
    missing,
)
print("Head:")
for r in final_rows[:5]:
    print(r)
