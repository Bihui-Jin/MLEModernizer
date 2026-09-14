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

0.0927070457354758

# 6. Current score

0.05232

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The timeout is dominated by DICOM decoding + PNG writing for every test image; the rest (copying models + running inference) is comparatively small. I keep the same conversion logic and inference call, but make conversion faster by (1) avoiding expensive `dcmread`/decompression for images already converted, (2) using `pydicom.dcmread(..., stop_before_pixels=True)` to detect un-decodable compressed syntaxes before attempting pixel decode, and (3) switching joblib to process-based parallelism (threads don’t help much because decoding is CPU-bound and often holds the GIL). I also remove per-patient directory pre-creation (it’s redundant because `process()` already creates directories) and reduce Python overhead in the file loop by vectorizing path parsing and using `os.path.isfile`. These changes are runtime-focused and preserve identical outputs for successfully decoded images and the same fallback behavior if decoding fails.'
- What this solution (achieved 0.01781) has done: 'Your 0.0 score is consistent with the pipeline falling back to an all-zeros submission, which happens when `infer.py` can’t run (missing fastai bundle/models) and/or when DICOM decoding produces too few usable PNGs. To move the score toward the ~0.093 target with minimal core-logic change, I keep your conversion + external inference approach, but add a lightweight, fully-metadata-based fallback model that produces non-zero probabilities when inference fails. This fallback trains a simple logistic regression on `train.csv` tabular metadata (no images), then predicts per test `prediction_id` and fills missing values safely; it should reliably beat 0.0 without touching your main model logic. I also add a small guard: only use the fallback if the inference submission is missing/invalid, or if it contains (near) all zeros.'
- What this solution (achieved 0.05232) has done: 'Your current score (0.01781) is far below the target (0.0927), so we should improve performance while keeping your main DICOM->PNG + `infer.py` path intact. The most likely issue is that the fallback tabular model is trained at the *image* level while the metric/submission is at the *prediction_id* level; fixing this alignment usually gives a noticeable lift without changing the overall approach. I therefore change only the fallback to train on `prediction_id`-aggregated labels and aggregated features (matching how you already aggregate test predictions), and I also add a tiny blend: when `infer.py` outputs are present but weak, blend them with the fallback instead of hard-switching. This preserves your core logic and only affects the safety net used when inference is missing/degenerate, pushing the score upward toward the target.'
- What this solution (achieved 0.05232) has done: 'We’re currently below the target (0.05232 vs 0.09271, higher is better), so we should make the smallest changes that legitimately improve pF1 without altering your core image pipeline. The biggest low-risk gain is to tune the blending weight `alpha` using a quick out-of-fold pF1 estimate on the training set at the same `prediction_id` granularity (no change to architectures, just choosing a better mix of your existing two predictors). To keep this minimal and stable, I fit the same tabular fallback as you already do, then learn an OOF blend weight between the tabular model and a simple prior derived from training prevalence; this avoids needing image inference on train but still gives a more pF1-friendly calibration than a fixed `alpha=0.85`. Finally, I keep your existing “infer is degenerate” guard, but when infer is OK I use the learned `alpha` (clipped to a conservative range) instead of a hardcoded value.'

# 9. Code solution

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


_UNSUPPORTED_TS_UIDS = {
    "1.2.840.10008.1.2.4.90",
    "1.2.840.10008.1.2.4.91",
    "1.2.840.10008.1.2.4.80",
    "1.2.840.10008.1.2.4.81",
    "1.2.840.10008.1.2.4.50",
    "1.2.840.10008.1.2.4.51",
    "1.2.840.10008.1.2.4.57",
    "1.2.840.10008.1.2.4.70",
    "1.2.840.10008.1.2.5",
}


def _is_probably_unsupported_transfer_syntax(path: str) -> bool:
    try:
        ds = pydicom.dcmread(
            path,
            stop_before_pixels=True,
            force=True,
            specific_tags=["TransferSyntaxUID"],
        )
        ts = str(getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", "") or "")
        return ts in _UNSUPPORTED_TS_UIDS
    except Exception:
        return False


def _read_dicom_pixel_array(path: str) -> tuple[np.ndarray, str]:
    """
    Returns (image_array_float01, photometric_interpretation).
    If decode fails, returns (None, None).
    """
    if _is_probably_unsupported_transfer_syntax(path):
        return None, None

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

for f in test_images:
    patient = os.path.basename(os.path.dirname(f))
    image = os.path.splitext(os.path.basename(f))[0]
    out_path = os.path.join(SAVE_FOLDER, patient, f"{image}.{EXTENSION}")
    if not os.path.isfile(out_path):
        to_convert.append((f, out_path))

print(f"Already converted: {len(test_images) - len(to_convert)} / {len(test_images)}")
print(f"Remaining to convert: {len(to_convert)} / {len(test_images)}")




## === cell 9
if len(to_convert) > 0:
    n_jobs = min(8, max(1, (os.cpu_count() or 4)))
    batch_size = 8 if len(to_convert) > 512 else 4
    results = Parallel(n_jobs=n_jobs, prefer="processes", batch_size=batch_size)(
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
def build_tabular_fallback_submission(
    train_csv_path: str, test_csv_path: str, sample_sub_path: str
) -> pd.DataFrame:
    """
    Train/predict at the same granularity as the submission/metric (prediction_id).
    - Train labels aggregated per prediction_id (max cancer).
    - Features aggregated per prediction_id (numeric median; categorical mode).
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    train = pd.read_csv(train_csv_path)
    test = pd.read_csv(test_csv_path)
    sample_sub = pd.read_csv(sample_sub_path)

    if "prediction_id" not in train.columns:
        train = train.copy()
        train["prediction_id"] = (
            train["patient_id"].astype(str) + "-" + train["laterality"].astype(str)
        )

    feature_cols = [
        "site_id",
        "patient_id",
        "laterality",
        "view",
        "age",
        "implant",
        "machine_id",
    ]
    feature_cols = [
        c for c in feature_cols if (c in train.columns and c in test.columns)
    ]

    def _mode_series(s: pd.Series):
        s = s.dropna()
        if len(s) == 0:
            return np.nan
        vc = s.value_counts(dropna=True)
        return vc.index[0] if len(vc) else np.nan

    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]

    agg_dict_train = {c: "median" for c in num_cols}
    for c in cat_cols:
        agg_dict_train[c] = _mode_series
    agg_dict_train["cancer"] = "max"

    train_g = (
        train[["prediction_id"] + feature_cols + ["cancer"]]
        .groupby("prediction_id", as_index=False)
        .agg(agg_dict_train)
    )

    X_train = train_g[feature_cols].copy()
    y_train = train_g["cancer"].astype(int).values

    agg_dict_test = {c: "median" for c in num_cols}
    for c in cat_cols:
        agg_dict_test[c] = _mode_series

    test_g = (
        test[["prediction_id"] + feature_cols]
        .groupby("prediction_id", as_index=False)
        .agg(agg_dict_test)
    )

    X_test = test_g[feature_cols].copy()

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        (
                            "oh",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=500,
        C=1.0,
        class_weight="balanced",
        n_jobs=None,
    )

    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1].astype(np.float32)
    pred = test_g[["prediction_id"]].copy()
    pred["cancer"] = np.clip(proba, 0.0, 1.0)

    sub = sample_sub[["prediction_id"]].merge(pred, on="prediction_id", how="left")
    sub["cancer"] = sub["cancer"].fillna(0.0).clip(0.0, 1.0)
    return sub




## === cell 14
def _pf1_score(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12) -> float:
    """
    Probabilistic F1 as described for the competition (probabilistic precision/recall).
    Using pTP=sum(p*y), pFP=sum(p*(1-y)), FN=sum((1-p)*y).
    """
    y_true = y_true.astype(np.float64)
    y_prob = np.clip(y_prob.astype(np.float64), 0.0, 1.0)

    pTP = float(np.sum(y_prob * y_true))
    pFP = float(np.sum(y_prob * (1.0 - y_true)))
    FN = float(np.sum((1.0 - y_prob) * y_true))

    pPrec = pTP / (pTP + pFP + eps)
    pRec = pTP / (pTP + FN + eps)
    return float(2.0 * pPrec * pRec / (pPrec + pRec + eps))


def _learn_blend_alpha_oof(train_csv_path: str) -> float:
    """
    CHANGE (score improvement, minimal): choose blend weight via quick OOF pF1 on train at prediction_id level.
    We only tune alpha for mixing:
        blended = alpha * "infer_like" + (1-alpha) * tabular_proba
    Since we don't have infer predictions for train, we use a conservative proxy ("infer_like"=constant prevalence),
    so alpha tuning mostly calibrates how much to trust tabular vs a prior, which improves pF1 stability.
    This keeps the core image inference untouched and only replaces a fixed alpha with a data-driven one.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    train = pd.read_csv(train_csv_path)
    if "prediction_id" not in train.columns:
        train = train.copy()
        train["prediction_id"] = (
            train["patient_id"].astype(str) + "-" + train["laterality"].astype(str)
        )

    feature_cols = [
        "site_id",
        "patient_id",
        "laterality",
        "view",
        "age",
        "implant",
        "machine_id",
    ]
    feature_cols = [c for c in feature_cols if c in train.columns]

    def _mode_series(s: pd.Series):
        s = s.dropna()
        if len(s) == 0:
            return np.nan
        vc = s.value_counts(dropna=True)
        return vc.index[0] if len(vc) else np.nan

    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]

    agg_dict_train = {c: "median" for c in num_cols}
    for c in cat_cols:
        agg_dict_train[c] = _mode_series
    agg_dict_train["cancer"] = "max"

    g = (
        train[["prediction_id"] + feature_cols + ["cancer"]]
        .groupby("prediction_id", as_index=False)
        .agg(agg_dict_train)
    )
    X = g[feature_cols].copy()
    y = g["cancer"].astype(int).values
    prevalence = float(np.mean(y))

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        (
                            "oh",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=500,
        C=1.0,
        class_weight="balanced",
        n_jobs=None,
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float32)
    for tr_idx, va_idx in skf.split(X, y):
        pipe.fit(X.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = pipe.predict_proba(X.iloc[va_idx])[:, 1].astype(np.float32)

    prior = np.full_like(oof, prevalence, dtype=np.float32)

    alphas = np.linspace(0.0, 1.0, 11, dtype=np.float32)
    best_alpha = 0.0
    best_score = -1.0
    for a in alphas:
        blended = a * prior + (1.0 - a) * oof
        s = _pf1_score(y, blended)
        if s > best_score:
            best_score = s
            best_alpha = float(a)

    best_alpha = float(np.clip(best_alpha, 0.05, 0.50))
    print(
        f"Learned OOF alpha proxy (prior vs tabular): {best_alpha:.3f} (best OOF pF1={best_score:.5f}, prevalence={prevalence:.5f})"
    )
    return best_alpha




## === cell 15
sample_sub_path = f"{KAGGLE_INPUT}/sample_submission.csv"
test_csv_path = f"{KAGGLE_INPUT}/test.csv"
train_csv_path = f"{KAGGLE_INPUT}/train.csv"

sample_sub = pd.read_csv(sample_sub_path)
assert {"prediction_id", "cancer"}.issubset(sample_sub.columns)

sub_infer = None
infer_ok = False

if os.path.exists(submission_path):
    try:
        df_1 = pd.read_csv(submission_path)
        if {"prediction_id", "cancer"}.issubset(df_1.columns):
            sub_infer = (
                df_1[["prediction_id", "cancer"]]
                .groupby("prediction_id", as_index=False)
                .max()
            )
            sub_infer = sample_sub[["prediction_id"]].merge(
                sub_infer, on="prediction_id", how="left"
            )
            sub_infer["cancer"] = sub_infer["cancer"].fillna(0.0).clip(0.0, 1.0)

            mean_pred = float(sub_infer["cancer"].mean())
            max_pred = float(sub_infer["cancer"].max())
            infer_ok = not (max_pred <= 1e-6 or mean_pred <= 1e-6)
            if not infer_ok:
                print(
                    f"Inference predictions look degenerate (mean={mean_pred:.3g}, max={max_pred:.3g})."
                )
        else:
            print("Inference submission missing required columns.")
    except Exception as e:
        print("Failed to read inference submission; error:", repr(e))
else:
    print("No inference submission produced.")

sub_fallback = build_tabular_fallback_submission(
    train_csv_path, test_csv_path, sample_sub_path
)

alpha_proxy = _learn_blend_alpha_oof(train_csv_path)

if sub_infer is None:
    sub = sub_fallback
else:
    mean_pred = float(sub_infer["cancer"].mean())
    max_pred = float(sub_infer["cancer"].max())
    if infer_ok:
        alpha = float(
            np.clip(0.80 + 0.30 * alpha_proxy, 0.80, 0.95)
        )  # weight on inference
        sub = sub_infer.copy()
        sub["cancer"] = (
            alpha * sub_infer["cancer"].values
            + (1.0 - alpha) * sub_fallback["cancer"].values
        ).astype(np.float32)
        sub["cancer"] = sub["cancer"].clip(0.0, 1.0)
        print(
            f"Using blended submission: alpha={alpha:.3f} (infer OK: mean={mean_pred:.4f}, max={max_pred:.4f}; alpha_proxy={alpha_proxy:.3f})"
        )
    else:
        sub = sub_fallback
        print("Using fallback submission (inference degenerate).")

out_final = "/kaggle/working/submission.csv"
sub.to_csv(out_final, index=False)
print("Wrote:", out_final, "shape:", sub.shape)
print(sub.head())
assert out_final.endswith(".csv") and os.path.exists(out_final)
assert list(sub.columns) == ["prediction_id", "cancer"]
assert len(sub) == len(sample_sub), "Submission row count must match sample_submission."
print(
    "Submission stats: min/mean/max =",
    float(sub.cancer.min()),
    float(sub.cancer.mean()),
    float(sub.cancer.max()),
)
