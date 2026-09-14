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
Predict whether a lesion is malignant (0 denotes **benign**, and 1 indicates **malignant**).

## Metric
Area under the ROC curve.

## Submission Format
For each `image_name` in the test set, you must predict the probability (`target`) that the sample is **malignant**. The file should contain a header and have the following format:

```
image_name,target
ISIC_0052060,0.7
ISIC_0052349,0.9
ISIC_0058510,0.8
ISIC_0073313,0.5
ISIC_0073502,0.5
etc.
```

## Dataset 
The images are provided in DICOM format.

Images are also provided in JPEG and TFRecord format (in the `jpeg` and `tfrecords` directories, respectively). Images in TFRecord format have been resized to a uniform 1024x1024.

Metadata is also provided outside of the DICOM format, in CSV files. See the `Columns` section for a description.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `image_name` - unique identifier, points to filename of related DICOM image
- `patient_id` - unique patient identifier
- `sex` - the sex of the patient (when unknown, will be blank)
- `age_approx` - approximate patient age at time of imaging
- `anatom_site_general_challenge` - location of imaged site
- `diagnosis` - detailed diagnosis information (train only)
- `benign_malignant` - indicator of malignancy of imaged lesion
- `target` - binarized version of the target variable

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
        input/
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
        working/
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
```

-> data/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/siim-isic-melanoma-classification/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> data/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> input/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> (stopped after 10 files for performance)

# 5. Target score

0.9332

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Diagnosis: Cell 1 crashes with `FileNotFoundError` because it tries to load multiple CSVs from `../input/public-submission-melanoma-95/`, but that dataset directory does not exist in this environment (available paths are under `/kaggle/data` and `/kaggle/input`). We must keep the ensemble logic intact for cell 2, so the fix is to resolve the correct base directory for the existing `siim-isic-melanoma-classification` files and safely fall back when the external “public-submission-melanoma-95” CSVs are unavailable. The minimal deterministic approach is to (a) read the sample submission from a path that exists, and (b) when the external submission files are missing, set each corresponding `public_sub_*` DataFrame to a copy of `sub` (so `.target` exists with the correct length), allowing cell 2 to run without changing its semantics beyond the unavoidable absence of those external files.

Patch summary: Update cell 1 to locate the real `sample_submission.csv` in the provided data tree, and wrap the missing external CSV reads with existence checks; if absent, replace them with `sub.copy()` to preserve required variables and column interfaces (`image_name`, `target`) for the weighted averaging in cell 2.

Updated cells: Only cell 1 is modified.

Compatibility notes for cell k+1: Cell 2 expects `public_sub_meta_ens_9577`, `public_sub_9619`, `public_sub_9606`, `public_sub_tabular`, and `sub` to exist and each to have a `.target` Series of the same length; this patch guarantees that interface and length compatibility.

Assumptions: The competition sample submission at `/kaggle/data/siim-isic-melanoma-classification/sample_submission.csv` (or `/kaggle/input/...`) is present (as shown), and when the external ensemble CSVs are missing we must still produce a valid submission-shaped placeholder to unblock execution.'
- What this solution (achieved 0.66859) has done: 'Your current 0.5 score is consistent with producing near-constant predictions (your fallback makes every missing ensemble file equal to `sample_submission`, which is all zeros), so AUC collapses to random. To move toward the 0.9332 target with minimal change and without altering your ensemble core logic, I keep the same weighted-averaging in cell 2 but make the fallback produce a non-constant, data-driven signal using only provided metadata (age/sex/anatom site) from `train.csv`/`test.csv`. This replaces the “all zeros” placeholder with a simple, deterministic prior-by-group probability that usually yields a meaningful AUC lift while staying lightweight and within the installed packages. The submission schema and file writing remain unchanged.'
- What this solution (achieved 0.64986) has done: 'Your current ensemble collapses to a single metadata-prior signal whenever the external submission files are missing, so the weighted averaging in cell 2 doesn’t add diversity and AUC plateaus (~0.67). To move the score upward toward 0.9332 with minimal logic change, I keep your exact ensemble formula but make each fallback “public_sub_*” produce a slightly different, deterministic metadata-based prediction (different groupings/prior strengths), so the ensemble becomes a real blend instead of identical inputs. I also add a safe alignment step to guarantee every fallback is ordered exactly like `sub` (prevents silent row misalignment from harming AUC). The script still runs end-to-end and writes `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd



## === cell 1
import os


def _first_existing_path(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of the expected paths exist: {paths}")


sample_sub_path = _first_existing_path(
    [
        "/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv",
        "/kaggle/data/siim-isic-melanoma-classification/sample_submission.csv",
        "../input/siim-isic-melanoma-classification/sample_submission.csv",
        "../data/siim-isic-melanoma-classification/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
    ]
)
sub = pd.read_csv(sample_sub_path)

train_csv_path = _first_existing_path(
    [
        "/kaggle/input/siim-isic-melanoma-classification/train.csv",
        "/kaggle/data/siim-isic-melanoma-classification/train.csv",
        "../input/siim-isic-melanoma-classification/train.csv",
        "../data/siim-isic-melanoma-classification/train.csv",
        "/kaggle/input/train.csv",
        "/kaggle/data/train.csv",
    ]
)
test_csv_path = _first_existing_path(
    [
        "/kaggle/input/siim-isic-melanoma-classification/test.csv",
        "/kaggle/data/siim-isic-melanoma-classification/test.csv",
        "../input/siim-isic-melanoma-classification/test.csv",
        "../data/siim-isic-melanoma-classification/test.csv",
        "/kaggle/input/test.csv",
        "/kaggle/data/test.csv",
    ]
)

train_df = pd.read_csv(
    train_csv_path,
    usecols=[
        "image_name",
        "sex",
        "age_approx",
        "anatom_site_general_challenge",
        "target",
    ],
    dtype={
        "image_name": "string",
        "sex": "string",
        "anatom_site_general_challenge": "string",
        "target": "int8",
    },
)
test_df = pd.read_csv(
    test_csv_path,
    usecols=["image_name", "sex", "age_approx", "anatom_site_general_challenge"],
    dtype={
        "image_name": "string",
        "sex": "string",
        "anatom_site_general_challenge": "string",
    },
)


def _prep_meta(tr, te):
    tr = tr.copy()
    te = te.copy()

    tr["sex"] = tr["sex"].fillna("unknown").replace("", "unknown")
    te["sex"] = te["sex"].fillna("unknown").replace("", "unknown")

    tr["anatom_site_general_challenge"] = (
        tr["anatom_site_general_challenge"].fillna("unknown").replace("", "unknown")
    )
    te["anatom_site_general_challenge"] = (
        te["anatom_site_general_challenge"].fillna("unknown").replace("", "unknown")
    )

    tr["age_approx"] = pd.to_numeric(tr["age_approx"], errors="coerce")
    te["age_approx"] = pd.to_numeric(te["age_approx"], errors="coerce")

    bins = [-np.inf, 10, 20, 30, 40, 50, 60, 70, 80, np.inf]
    labels = ["<10", "10s", "20s", "30s", "40s", "50s", "60s", "70s", "80+"]
    tr["age_bin"] = pd.cut(tr["age_approx"], bins=bins, labels=labels)
    te["age_bin"] = pd.cut(te["age_approx"], bins=bins, labels=labels)

    for c in ["sex", "anatom_site_general_challenge", "age_bin"]:
        if c in tr.columns:
            tr[c] = tr[c].astype("category")
        if c in te.columns:
            te[c] = te[c].astype("category")

    return tr, te


_tr_meta, _te_meta = _prep_meta(train_df, test_df)


def _make_meta_prior_submission_prepped(tr, te, sub_template, group_cols, m):
    global_mean = float(tr["target"].mean())

    if len(group_cols) == 1:
        key_tr = tr[group_cols[0]].cat.codes.to_numpy(np.int32, copy=False)
        key_te = te[group_cols[0]].cat.codes.to_numpy(np.int32, copy=False)
    else:
        codes_tr = [tr[c].cat.codes.to_numpy(np.int32, copy=False) for c in group_cols]
        codes_te = [te[c].cat.codes.to_numpy(np.int32, copy=False) for c in group_cols]
        radices = []
        for c in group_cols:
            radices.append(int(tr[c].cat.categories.size) + 2)
        key_tr = np.zeros(len(tr), dtype=np.int64)
        key_te = np.zeros(len(te), dtype=np.int64)
        mul = 1
        for r, ctr, cte in zip(radices, codes_tr, codes_te):
            key_tr += (ctr.astype(np.int64) + 1) * mul
            key_te += (cte.astype(np.int64) + 1) * mul
            mul *= r

    y = tr["target"].to_numpy(np.float64, copy=False)
    df_key = pd.DataFrame({"k": key_tr, "y": y})
    grp = df_key.groupby("k", sort=False, observed=True)["y"].agg(["mean", "count"])
    smoothed = (grp["mean"] * grp["count"] + global_mean * float(m)) / (
        grp["count"] + float(m)
    )

    te_sm = pd.Series(key_te, index=te["image_name"].to_numpy(), name="k").map(smoothed)
    te_target = te_sm.fillna(global_mean).astype(float).clip(0.0, 1.0)

    out = sub_template[["image_name"]].copy()
    out["target"] = out["image_name"].map(te_target).fillna(global_mean).astype(float)
    out["target"] = out["target"].clip(0.0, 1.0)
    return out


meta_prior_full_m50 = _make_meta_prior_submission_prepped(
    _tr_meta,
    _te_meta,
    sub,
    group_cols=["sex", "anatom_site_general_challenge", "age_bin"],
    m=50.0,
)
meta_prior_noage_m30 = _make_meta_prior_submission_prepped(
    _tr_meta, _te_meta, sub, group_cols=["sex", "anatom_site_general_challenge"], m=30.0
)
meta_prior_ageonly_m100 = _make_meta_prior_submission_prepped(
    _tr_meta, _te_meta, sub, group_cols=["age_bin"], m=100.0
)
meta_prior_siteonly_m20 = _make_meta_prior_submission_prepped(
    _tr_meta, _te_meta, sub, group_cols=["anatom_site_general_challenge"], m=20.0
)

public_dir_candidates = [
    "/kaggle/input/public-submission-melanoma-95",
    "/kaggle/data/public-submission-melanoma-95",
    "../input/public-submission-melanoma-95",
    "../data/public-submission-melanoma-95",
]
public_dir = next((d for d in public_dir_candidates if os.path.isdir(d)), None)



## === cell 2
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression

import hashlib
from concurrent.futures import ThreadPoolExecutor


def _first_existing_dir(paths):
    for p in paths:
        if os.path.isdir(p):
            return p
    return None


jpeg_train_dir = _first_existing_dir(
    [
        "/kaggle/input/siim-isic-melanoma-classification/jpeg/train",
        "/kaggle/data/siim-isic-melanoma-classification/jpeg/train",
        "/kaggle/input/jpeg/train",
        "/kaggle/data/jpeg/train",
    ]
)
jpeg_test_dir = _first_existing_dir(
    [
        "/kaggle/input/siim-isic-melanoma-classification/jpeg/test",
        "/kaggle/data/siim-isic-melanoma-classification/jpeg/test",
        "/kaggle/input/jpeg/test",
        "/kaggle/data/jpeg/test",
    ]
)


def _path_from_name(name, base_dir):
    if base_dir is None:
        return None
    return os.path.join(base_dir, f"{name}.jpg")


try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False


def _jpeg_feat_one_path_pil(p, size):
    feat = np.zeros((12,), dtype=np.float32)
    if p is None:
        return feat
    try:
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if im.size[0] > size[0] * 2 and im.size[1] > size[1] * 2:
                im.thumbnail((size[0] * 2, size[1] * 2), resample=Image.BILINEAR)
            im = im.resize(size, resample=Image.BILINEAR)
            arr_u8 = np.asarray(im, dtype=np.uint8)
        arr = arr_u8.astype(np.float32) * (1.0 / 255.0)

        ch_mean = arr.mean(axis=(0, 1))
        ch_std = arr.std(axis=(0, 1))

        gray = arr.mean(axis=2)
        g_mean = float(gray.mean())
        g_std = float(gray.std())
        dark = float((gray < 0.25).mean())
        bright = float((gray > 0.75).mean())

        feat[0:3] = ch_mean
        feat[3:6] = ch_std
        feat[6] = g_mean
        feat[7] = g_std
        feat[8] = dark
        feat[9] = bright
        feat[10] = float(ch_mean.max() - ch_mean.min())
        feat[11] = float(ch_std.mean())
    except Exception:
        pass
    return feat


def _jpeg_feat_one_path_cv2(p, size):
    feat = np.zeros((12,), dtype=np.float32)
    if p is None:
        return feat
    try:
        img_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return feat
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        h, w = img_rgb.shape[:2]
        if w > size[0] * 2 and h > size[1] * 2:
            img_rgb = cv2.resize(
                img_rgb, (size[0] * 2, size[1] * 2), interpolation=cv2.INTER_LINEAR
            )

        img_rgb = cv2.resize(img_rgb, size, interpolation=cv2.INTER_LINEAR)
        arr = img_rgb.astype(np.float32) * (1.0 / 255.0)

        ch_mean = arr.mean(axis=(0, 1))
        ch_std = arr.std(axis=(0, 1))

        gray = arr.mean(axis=2)
        g_mean = float(gray.mean())
        g_std = float(gray.std())
        dark = float((gray < 0.25).mean())
        bright = float((gray > 0.75).mean())

        feat[0:3] = ch_mean
        feat[3:6] = ch_std
        feat[6] = g_mean
        feat[7] = g_std
        feat[8] = dark
        feat[9] = bright
        feat[10] = float(ch_mean.max() - ch_mean.min())
        feat[11] = float(ch_std.mean())
    except Exception:
        pass
    return feat


def _jpeg_feat_one_path(p, size):
    if _HAS_CV2:
        return _jpeg_feat_one_path_cv2(p, size)
    return _jpeg_feat_one_path_pil(p, size)


def _extract_jpeg_features(image_names, base_dir, size=(64, 64)):
    n = len(image_names)
    feats = np.zeros((n, 12), dtype=np.float32)
    if n == 0 or base_dir is None:
        return feats

    cpu = os.cpu_count() or 2
    max_workers = min(24, cpu * 2)

    _size = size
    chunksize = 2048 if n >= 20000 else 1024

    def _worker(name):
        p = os.path.join(base_dir, f"{name}.jpg")
        return _jpeg_feat_one_path(p, _size)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, f in enumerate(ex.map(_worker, image_names, chunksize=chunksize)):
            feats[i] = f
    return feats


def _standardize(train_X, test_X):
    mu = train_X.mean(axis=0, keepdims=True)
    sd = train_X.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return (train_X - mu) / sd, (test_X - mu) / sd


_tr_names = train_df["image_name"].to_numpy()
_te_names = test_df["image_name"].to_numpy()


def _cache_path(tag):
    base = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    return os.path.join(base, tag)


def _compute_cache_key(train_dir, test_dir, size, ntr, nte, backend):
    s = f"{train_dir}|{test_dir}|{size[0]}x{size[1]}|{ntr}|{nte}|{backend}"
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


if jpeg_train_dir is not None and jpeg_test_dir is not None:
    _backend = "cv2" if _HAS_CV2 else "pil"
    _key = _compute_cache_key(
        jpeg_train_dir,
        jpeg_test_dir,
        (64, 64),
        len(_tr_names),
        len(_te_names),
        _backend,
    )
    _cache_npz = _cache_path(f"jpeg_feats_64x64_{_key}.npz")
    if os.path.exists(_cache_npz):
        d = np.load(_cache_npz)
        _Xtr_full = d["Xtr"]
        _Xte_full = d["Xte"]
        d.close()
    else:
        _Xtr_full = _extract_jpeg_features(_tr_names, jpeg_train_dir, size=(64, 64))
        _Xte_full = _extract_jpeg_features(_te_names, jpeg_test_dir, size=(64, 64))
        np.savez_compressed(_cache_npz, Xtr=_Xtr_full, Xte=_Xte_full)
else:
    _Xtr_full, _Xte_full = None, None


_sub_image_names = sub["image_name"].to_numpy()


def _fit_predict_jpeg_lr(train_df, test_df, feature_idx=None, C=1.0):
    if (
        jpeg_train_dir is None
        or jpeg_test_dir is None
        or _Xtr_full is None
        or _Xte_full is None
    ):
        return meta_prior_full_m50.copy()

    Xtr = _Xtr_full if feature_idx is None else _Xtr_full[:, feature_idx]
    Xte = _Xte_full if feature_idx is None else _Xte_full[:, feature_idx]

    Xtr, Xte = _standardize(Xtr, Xte)
    y = train_df["target"].to_numpy(dtype=np.int32, copy=False)

    clf = LogisticRegression(
        C=float(C),
        solver="liblinear",
        max_iter=200,
        random_state=0,
    )
    clf.fit(Xtr, y)
    proba = clf.predict_proba(Xte)[:, 1].astype(float)

    global_mean = float(train_df["target"].mean())

    out = sub[["image_name"]].copy()
    if (
        len(out) == len(test_df)
        and (out["image_name"].values == test_df["image_name"].values).all()
    ):
        out["target"] = np.clip(proba, 0.0, 1.0)
    else:
        te_pred = pd.Series(proba, index=test_df["image_name"].to_numpy())
        out["target"] = (
            out["image_name"].map(te_pred).fillna(global_mean).astype(float).to_numpy()
        )
        out["target"] = np.clip(out["target"], 0.0, 1.0).astype(float)
    return out


jpeg_lr_all_C1 = _fit_predict_jpeg_lr(train_df, test_df, feature_idx=None, C=1.0)
jpeg_lr_means_only = _fit_predict_jpeg_lr(
    train_df, test_df, feature_idx=[0, 1, 2, 6, 10], C=0.8
)
jpeg_lr_std_only = _fit_predict_jpeg_lr(
    train_df, test_df, feature_idx=[3, 4, 5, 7, 11], C=0.8
)
jpeg_lr_contrast_only = _fit_predict_jpeg_lr(
    train_df, test_df, feature_idx=[8, 9, 10, 11], C=0.6
)




## === cell 3
def _read_or_fallback(filename, fallback_df):
    if public_dir is not None:
        path = os.path.join(public_dir, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "image_name" in df.columns and "target" in df.columns:
                s = pd.Series(df["target"].values, index=df["image_name"].values)
                out = sub[["image_name"]].copy()
                out["target"] = out["image_name"].map(s)
                out["target"] = (
                    out["target"].fillna(fallback_df["target"].values).astype(float)
                )
                out["target"] = out["target"].clip(0.0, 1.0)
                return out
    return fallback_df.copy()


public_sub_mean_9533 = _read_or_fallback("submission_mean.csv", jpeg_lr_means_only)
public_sub_median_9533 = _read_or_fallback("submission_median.csv", jpeg_lr_std_only)
public_sub_meta_ens_9577 = _read_or_fallback(
    "external_meta_ensembled.csv", jpeg_lr_all_C1
)
public_sub_9581 = _read_or_fallback("submission_9581.csv", jpeg_lr_all_C1)
public_sub_tabular = _read_or_fallback(
    "submission_tabular_only.csv", jpeg_lr_contrast_only
)
public_sub_9619 = _read_or_fallback("submission_9619.csv", jpeg_lr_means_only)
public_sub_9606 = _read_or_fallback("submission_9606.csv", jpeg_lr_std_only)
public_sub_9603 = _read_or_fallback("submission_9603.csv", jpeg_lr_contrast_only)



## === cell 4
sub.target = (
    public_sub_meta_ens_9577.target * 0.35
    + public_sub_9619.target * 0.175
    + public_sub_9606.target * 0.175
    + public_sub_tabular.target * 0.30
)



## === cell 5
sub.head()
sub.to_csv("submission.csv", index=False)



## === cell 6
sub.head
