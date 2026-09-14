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

0.04268

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the root cause of the crashes: your notebook is running in an environment where importing `numpy` (and thus `pandas`) is broken, so any code relying on them cannot run. To make the solution run end-to-end and still produce a valid `.csv` submission, I remove the hard dependency on NumPy/Pandas and write the submission using Python’s standard library (`csv`) by copying `sample_submission.csv` and setting `cancer` to 0.0 (a safe fallback when inference can’t run). I also keep your existing DICOM/PNG conversion and fastai inference attempt, but guard imports so the pipeline continues even if optional pieces fail. The final result always write `/kaggle/working/submission.csv` with the exact required columns.'
- What this solution (achieved 0.0) has done: 'I fix the runtime crash caused by the broken NumPy import by preventing `tqdm.notebook` (and other optional imports) from being imported in a way that pulls in NumPy. Then I make sure the pipeline always produces a valid `/kaggle/working/submission.csv` by reading prediction_ids directly from the official sample submission and filling predictions from `infer.py` if it ran, otherwise falling back safely. Finally, I keep your existing DICOM→PNG conversion and `fastai` inference call intact, but add small guards so missing/failed conversions don’t break the submission creation.'
- What this solution (achieved 0.0) has done: 'I stop the `numpy` crash by preventing imports that transitively pull NumPy (notably `tqdm` and `joblib`) and by providing small standard-library fallbacks. Then I make the DICOM→PNG conversion run without OpenCV by using `pydicom` + Pillow (both commonly available) and a safe fallback to skip conversion if image decoding isn’t possible, keeping the same core preprocessing intent. Finally, I fix the inference stage so it doesn’t hard-depend on a non-existent `/kaggle/input/fastai` dataset; instead it call `infer.py` only if it exists, and otherwise still produce a valid `/kaggle/working/submission.csv` using the official `sample_submission.csv` prediction_ids (so you never get score 0.0 due to a missing/invalid submission file). These changes are primarily correctness/stability fixes; if inference is available they should also improve score over the all-zeros fallback.'
- What this solution (achieved 0.02212) has done: 'Your 0.0 score is consistent with a “mostly/all zeros” submission, which happens when inference doesn’t run or when prediction IDs don’t align and get filled with 0.0. I keep your pipeline and submission-writing logic, but (1) switch the DICOM conversion loop to only convert the minimal subset of DICOMs needed to cover the `prediction_id`s present in `sample_submission.csv` (so inference has the required PNGs and finishes within the time limit), (2) add a robust fallback that builds predictions directly from `test.csv` metadata when `infer.py` output is missing/empty, using a small positive prior (non-zero) to move pF1 above 0 toward your target, and (3) ensure the final submission always exactly matches the sample’s `prediction_id` order and row count. These are minimal, score-relevant changes that preserve your overall approach and avoid touching model architecture/training.'
- What this solution (achieved 0.02257) has done: 'Your current score (0.02212) is far below the target (0.11578), so we should cautiously increase it without changing your model/inference core. The biggest issue is that your fallback produces almost-constant low probabilities (≈0.02), which tends to yield a very low pF1; we can improve pF1 substantially by calibrating a simple probability transform on train metadata only (no leakage) and applying it to test as a fallback when `infer.py` predictions are missing. Concretely, we keep your pipeline intact, but replace the constant prior fallback with a tiny logistic-regression (implemented in pure Python) using `age`, `implant`, `site_id`, `machine_id`, `laterality`, and `view`, and we also blend any available `infer.py` predictions with this metadata model to stabilize/improve coverage. This is minimal, runs fast, uses only provided CSVs, preserves submission format/order, and should move the score toward your target.'
- What this solution (achieved 0.0041) has done: 'We keep your inference and metadata-logreg core intact, but tune the *post-processing* to better match the pF1 metric, since your current output is likely too “flat” (near the base rate) which suppresses pF1. Concretely, we (1) train the same pure-Python logistic regression but with a couple more epochs (still extremely fast) so the metadata fallback is meaningfully non-constant, and (2) apply a tiny, deterministic “sharpening” calibration to increase separation between low/high risks (without changing ranking drastically). Finally, we blend infer/meta as before, but adjust the blend weight slightly toward the (usually stronger) image-model output when present, while keeping the submission row order/format exactly identical.'
- What this solution (achieved 0.00754) has done: 'Your score (0.0041) is far below the target (0.11578), so we should increase pF1 with minimal, metric-aligned post-processing while keeping your existing pipeline (DICOM→PNG, optional `infer.py`, pure-Python metadata logreg, and blending) intact. The biggest score-limiter is likely that your probabilities are too “flat” around the prior; pF1 typically improves when predictions are more concentrated near 0 for most and higher for a small fraction. I therefore (1) keep the same logreg but train it a bit longer and with a slightly smaller LR for stability, and (2) replace the fixed gamma “sharpen” with a deterministic, pF1-friendly *top-k boosting* (only based on your own predicted risks) to create a small high-probability tail without changing any model architecture or using leakage. Finally, I keep the submission schema and sample order exactly the same and still write `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.04383) has done: 'We keep your pipeline (DICOM→PNG, optional `infer.py` inference, pure-Python metadata logreg, and blending) unchanged in structure, but make two minimal score-relevant adjustments to better align outputs with pF1. First, we make the metadata model training slightly more stable without changing the model by using a deterministic shuffle and a more pF1-friendly positive class weighting (still logistic regression, same loss family) so it produces a more useful risk spread than the current “flat” outputs. Second, we replace the current aggressive `topk_boost` (which forces almost everything to ~0, often hurting pF1) with a softer, prior-preserving boost that only lifts the highest-risk tail while leaving the majority near their model probabilities. These changes are small, fast, and should move your score upward toward the 0.1158 target while preserving submission format/order.'
- What this solution (achieved 0.04268) has done: 'We keep your pipeline intact (DICOM→PNG conversion, optional `infer.py` predictions, pure-Python metadata logistic regression, blending, and tail-boosting), but make two small, metric-aligned adjustments to move pF1 upward toward the 0.1158 target. First, we set the `infer.py` `--threshold` to `0.0` to avoid accidentally zeroing probabilities inside the external inference script (this commonly collapses predictions and hurts pF1), without changing your inference approach. Second, we make the `topk_boost` a bit less aggressive by (a) slightly reducing the forced-high probability `hi` and (b) allowing the non-tail majority to keep more of their original model probability instead of being clamped down to a low “base,” which tends to improve pF1 when the model already has some signal. Everything still writes a valid `/kaggle/working/submission.csv` matching the sample submission order/shape.'

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
def tqdm(x, **kwargs):
    return x


try:
    import pydicom  # optional
except Exception as e:
    pydicom = None
    print("[WARN] pydicom import failed; DICOM reading will be skipped:", e)

try:
    from PIL import Image
except Exception as e:
    Image = None
    print("[WARN] Pillow import failed; PNG writing will be skipped:", e)




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
    if pydicom is None or Image is None:
        return False
    try:
        patient = f.split("/")[-2]
        image = os.path.basename(f)[:-4]

        dicom = pydicom.dcmread(f, force=True)
        img = dicom.pixel_array  # may require pylibjpeg/openjpeg for JP2K

        mn = float(img.min())
        mx = float(img.max())
        if mx > mn:
            img = (img - mn) / (mx - mn)
        else:
            img = img * 0.0

        if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
            img = 1.0 - img

        img_u8 = (img * 255.0).clip(0, 255).astype("uint8")

        pil = Image.fromarray(img_u8)
        pil = pil.resize((int(size[0]), int(size[1])), resample=Image.BILINEAR)

        path = os.path.join(save_folder, f"{patient}")
        os.makedirs(path, exist_ok=True)
        out_path = os.path.join(path, f"{image}.{extension}")
        pil.save(out_path)
        return True
    except Exception:
        return False




## === cell 6
sample_path = f"{DATA_ROOT}/sample_submission.csv"
test_csv_path = f"{DATA_ROOT}/test.csv"


def read_csv_rows(path):
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        cols = reader.fieldnames or []
    return cols, rows


sample_cols, sample_rows = read_csv_rows(sample_path)
if "prediction_id" not in (sample_cols or []):
    raise RuntimeError("sample_submission.csv missing prediction_id column")
sample_pred_ids = set(str(r["prediction_id"]) for r in sample_rows)

test_cols, test_rows = read_csv_rows(test_csv_path)
need_cols = {"patient_id", "image_id", "prediction_id"}
if not need_cols.issubset(set(test_cols or [])):
    raise RuntimeError(
        f"test.csv missing required columns: {need_cols - set(test_cols or [])}"
    )

need_dicoms = []
for r in test_rows:
    pid = str(r.get("prediction_id", ""))
    if pid in sample_pred_ids:
        patient_id = str(r["patient_id"])
        image_id = str(r["image_id"])
        dcm_path = f"{DATA_ROOT}/test_images/{patient_id}/{image_id}.dcm"
        if os.path.exists(dcm_path):
            need_dicoms.append(dcm_path)

if DEBUG:
    need_dicoms = need_dicoms[:50]

print("DICOMs needed for sample_submission prediction_ids:", len(need_dicoms))

ok = 0
if pydicom is not None and Image is not None:
    for uid in tqdm(need_dicoms):
        ok += (
            1
            if process(uid, size=SIZE, save_folder=SAVE_FOLDER, extension=EXTENSION)
            else 0
        )
    print(f"Converted {ok}/{len(need_dicoms)} needed images to PNG under {SAVE_FOLDER}")
else:
    print("[WARN] Skipping PNG conversion (missing pydicom/Pillow).")




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
possible_infer = [
    "/kaggle/input/fastai/fastai/infer.py",
    "/kaggle/input/rsna-breast-cancer-detection/infer.py",
    "/kaggle/input/infer.py",
]
infer_py = next((p for p in possible_infer if os.path.exists(p)), None)

possible_model_dirs = [
    "/kaggle/input/fastai/fastai/models",
    "/kaggle/input/rsna-breast-cancer-detection/models",
    "/kaggle/input/models",
]
src_model_dir = next((p for p in possible_model_dirs if os.path.isdir(p)), None)

model_dir = "/kaggle/tmp/models"
if src_model_dir is not None:
    os.makedirs(model_dir, exist_ok=True)
    _run(f'cp -r "{src_model_dir}/." "{model_dir}/"')
else:
    print(
        "[WARN] No model directory found in inputs; inference will likely be skipped."
    )

png_data_dir = SAVE_FOLDER
out_csv = f"{TEST_OUTDIR}/submission.csv"
test_csv = f"{DATA_ROOT}/test.csv"

infer_ok = False
if infer_py is not None and os.path.isdir(model_dir) and os.path.exists(test_csv):
    cmd = (
        f'python "{infer_py}" '
        f'--model "{model_dir}" '
        f'--data "{png_data_dir}" '
        f'--csv "{out_csv}" '
        f"--threshold 0.0 "
        f"--split 4 "
        f"--stt 0 "
        f'--test_csv "{test_csv}"'
    )
    infer_ok = _run(cmd)
else:
    print("[WARN] Missing infer.py/model_dir/test_csv; skipping model inference.")
    print(" infer.py exists:", infer_py is not None)
    print(" model_dir exists:", os.path.isdir(model_dir))
    print(" test_csv exists:", os.path.exists(test_csv))

print("Inference attempted:", infer_ok, "-> output exists:", os.path.exists(out_csv))




## === cell 10
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


import math


def _sigmoid(z):
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


def _clip01(x):
    if x != x:
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _to_float(x, default=0.0):
    try:
        if x is None:
            return default
        xs = str(x).strip()
        if xs == "" or xs.lower() in ("nan", "none"):
            return default
        return float(xs)
    except Exception:
        return default


def _hash_bucket(s, mod):
    s = str(s)
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h % mod)


def featurize_meta(row):
    feats = {}
    age = _to_float(row.get("age", ""), default=0.0)
    feats["age"] = age / 100.0  # scale
    feats["age2"] = (age / 100.0) * (age / 100.0)

    implant = row.get("implant", "")
    imp = 1.0 if str(implant).strip().lower() in ("1", "true", "t", "yes") else 0.0
    feats["implant"] = imp

    site_id = row.get("site_id", "")
    feats[f"site_{site_id}"] = 1.0

    machine_id = row.get("machine_id", "")
    feats[f"machine_{machine_id}"] = 1.0

    laterality = row.get("laterality", "")
    feats[f"lat_{laterality}"] = 1.0

    view = row.get("view", "")
    feats[f"view_{view}"] = 1.0

    combo = f"{site_id}_{machine_id}_{view}_{laterality}"
    feats[f"combo_{_hash_bucket(combo, 64)}"] = 1.0
    return feats


def train_logreg(train_rows, lr=0.02, epochs=12, l2=0.8, pos_weight=4.0, seed=1337):
    w = {"__bias__": 0.0}

    s = 0.0
    n = 0
    for r in train_rows:
        y = _to_float(r.get("cancer", 0.0), 0.0)
        s += 1.0 if y >= 0.5 else 0.0
        n += 1
    base = s / max(n, 1)
    base = min(max(base, 1e-4), 1.0 - 1e-4)
    w["__bias__"] = math.log(base / (1.0 - base))

    idx = list(range(len(train_rows)))

    def _rand_u32(x):
        x = (x ^ (x >> 16)) & 0xFFFFFFFF
        x = (x * 0x7FEB352D) & 0xFFFFFFFF
        x = (x ^ (x >> 15)) & 0xFFFFFFFF
        x = (x * 0x846CA68B) & 0xFFFFFFFF
        x = (x ^ (x >> 16)) & 0xFFFFFFFF
        return x

    for ep in range(epochs):
        idx.sort(key=lambda i, ep=ep: _rand_u32((seed + 97 * ep + i) & 0xFFFFFFFF))
        for i in idx:
            r = train_rows[i]
            y = 1.0 if _to_float(r.get("cancer", 0.0), 0.0) >= 0.5 else 0.0
            x = featurize_meta(r)

            z = w.get("__bias__", 0.0)
            for k, v in x.items():
                z += w.get(k, 0.0) * v
            p = _sigmoid(z)

            wt = pos_weight if y >= 0.5 else 1.0
            err = (p - y) * wt

            w["__bias__"] = w.get("__bias__", 0.0) - lr * err

            for k, v in x.items():
                wk = w.get(k, 0.0)
                grad = err * v + l2 * wk
                w[k] = wk - lr * grad
    return w


def predict_logreg(w, row):
    x = featurize_meta(row)
    z = w.get("__bias__", 0.0)
    for k, v in x.items():
        z += w.get(k, 0.0) * v
    return _sigmoid(z)


def topk_boost(
    pred_rows,
    k_frac=0.03,
    hi=0.65,
    tail2_mult=1.25,
    prior_floor=0.0035,
    keep_frac=0.60,
):
    n = len(pred_rows)
    if n <= 0:
        return pred_rows

    k = int(max(1, min(n, round(n * float(k_frac)))))
    order = sorted(range(n), key=lambda i: float(pred_rows[i]["cancer"]), reverse=True)

    vals = [float(r["cancer"]) for r in pred_rows]
    vals_sorted = sorted(vals)
    cut = int(max(1, round(0.9 * n))) - 1
    base = (
        vals_sorted[cut]
        if cut < len(vals_sorted)
        else (vals_sorted[-1] if vals_sorted else 0.01)
    )
    base = max(prior_floor, min(base, 0.06))

    for rank, idx in enumerate(order):
        p = _clip01(float(pred_rows[idx]["cancer"]))
        if rank < k:
            pred_rows[idx]["cancer"] = max(p, hi)
        elif rank < 4 * k:
            pred_rows[idx]["cancer"] = _clip01(p * tail2_mult)
        else:
            pred_rows[idx]["cancer"] = max(
                prior_floor, min(p, 1.0) * keep_frac + base * (1.0 - keep_frac)
            )
    return pred_rows


sample_cols, sample_rows = read_submission_csv(sample_path)
if "prediction_id" not in sample_cols:
    raise RuntimeError("sample_submission.csv missing prediction_id column")
sample_pred_ids_ordered = [str(r["prediction_id"]) for r in sample_rows]

pred_map = None
if os.path.exists(out_csv):
    out_cols, out_rows = read_submission_csv(out_csv)
    if ("prediction_id" in out_cols) and ("cancer" in out_cols) and len(out_rows) > 0:
        sums, cnts = {}, {}
        for r in out_rows:
            pid = str(r.get("prediction_id", ""))
            try:
                val = float(r.get("cancer", 0.0))
            except Exception:
                val = 0.0
            val = _clip01(val)
            sums[pid] = sums.get(pid, 0.0) + val
            cnts[pid] = cnts.get(pid, 0) + 1
        pred_map = {pid: (sums[pid] / max(cnts[pid], 1)) for pid in sums.keys()}
        print("[INFO] Loaded infer predictions for prediction_ids:", len(pred_map))
    else:
        print(
            "[WARN] infer output missing required columns/rows; falling back to metadata model."
        )
else:
    print("[WARN] infer output not found; falling back to metadata model.")

train_csv_path = f"{DATA_ROOT}/train.csv"
meta_pred_map = {}
prior = 0.02

if os.path.exists(train_csv_path):
    try:
        tcols, trows = read_csv_rows(train_csv_path)
        if "cancer" in (tcols or []) and len(trows) > 0:
            train_rows_for_fit = trows if not DEBUG else trows[:5000]

            w = train_logreg(
                train_rows_for_fit, lr=0.02, epochs=12, l2=0.8, pos_weight=4.0
            )

            s = 0.0
            n = 0
            for r in trows:
                try:
                    s += float(r.get("cancer", 0.0))
                    n += 1
                except Exception:
                    continue
            if n > 0:
                prior = s / n

            sums, cnts = {}, {}
            for r in test_rows:
                pid = str(r.get("prediction_id", ""))
                if pid in sample_pred_ids:
                    p = predict_logreg(w, r)
                    sums[pid] = sums.get(pid, 0.0) + p
                    cnts[pid] = cnts.get(pid, 0) + 1
            meta_pred_map = {
                pid: (sums[pid] / max(cnts[pid], 1)) for pid in sums.keys()
            }
            print(
                "[INFO] Built metadata predictions for prediction_ids:",
                len(meta_pred_map),
            )
    except Exception as e:
        print("[WARN] Could not train metadata model from train.csv:", e)

if prior != prior or prior <= 0.0:
    prior = 0.02
prior = max(0.005, min(0.06, float(prior)))
print("[INFO] Using prior fallback (only if no meta/model preds):", prior)

alpha = 0.80

final_rows = []
missing_infer = 0
missing_meta = 0

for pid in sample_pred_ids_ordered:
    have_infer = pred_map is not None and pid in pred_map
    have_meta = pid in meta_pred_map

    if have_infer and have_meta:
        val = alpha * float(pred_map[pid]) + (1.0 - alpha) * float(meta_pred_map[pid])
    elif have_infer:
        val = float(pred_map[pid])
    elif have_meta:
        val = float(meta_pred_map[pid])
        missing_infer += 1
    else:
        val = float(prior)
        missing_infer += 1
        missing_meta += 1

    val = _clip01(val)
    final_rows.append({"prediction_id": pid, "cancer": float(val)})

final_rows = topk_boost(
    final_rows,
    k_frac=0.03,
    hi=0.65,
    tail2_mult=1.25,
    prior_floor=0.0035,
    keep_frac=0.60,
)

write_submission_csv(final_path, final_rows)
print(
    "Wrote:",
    final_path,
    "rows:",
    len(final_rows),
    "missing_infer_pred_ids:",
    missing_infer,
    "missing_meta_pred_ids:",
    missing_meta,
)
print("Head:")
for r in final_rows[:5]:
    print(r)
print(
    "Submission exists:",
    os.path.exists(final_path),
    "size_bytes:",
    os.path.getsize(final_path) if os.path.exists(final_path) else -1,
)
