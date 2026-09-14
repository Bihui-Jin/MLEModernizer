# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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
)
test_df = pd.read_csv(
    test_csv_path,
    usecols=["image_name", "sex", "age_approx", "anatom_site_general_challenge"],
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

    return tr, te


def _make_meta_prior_submission(train_df, test_df, sub_template, group_cols, m):
    tr, te = _prep_meta(train_df, test_df)

    global_mean = float(tr["target"].mean())

    grp = (
        tr.groupby(group_cols, dropna=False)["target"]
        .agg(["mean", "count"])
        .reset_index()
    )
    grp["smoothed"] = (grp["mean"] * grp["count"] + global_mean * float(m)) / (
        grp["count"] + float(m)
    )

    te = te.merge(grp[group_cols + ["smoothed"]], on=group_cols, how="left")
    te["target"] = te["smoothed"].fillna(global_mean).astype(float)

    out = sub_template[["image_name"]].merge(
        te[["image_name", "target"]], on="image_name", how="left"
    )
    out["target"] = out["target"].fillna(global_mean).astype(float)
    out["target"] = out["target"].clip(0.0, 1.0)

    out = out.set_index("image_name").reindex(sub_template["image_name"]).reset_index()
    out["target"] = out["target"].astype(float)
    return out


meta_prior_full_m50 = _make_meta_prior_submission(
    train_df,
    test_df,
    sub,
    group_cols=["sex", "anatom_site_general_challenge", "age_bin"],
    m=50.0,
)
meta_prior_noage_m30 = _make_meta_prior_submission(
    train_df, test_df, sub, group_cols=["sex", "anatom_site_general_challenge"], m=30.0
)
meta_prior_ageonly_m100 = _make_meta_prior_submission(
    train_df, test_df, sub, group_cols=["age_bin"], m=100.0
)
meta_prior_siteonly_m20 = _make_meta_prior_submission(
    train_df, test_df, sub, group_cols=["anatom_site_general_challenge"], m=20.0
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


def _img_path(img_name, is_train):
    base = jpeg_train_dir if is_train else jpeg_test_dir
    if base is None:
        return None
    return os.path.join(base, f"{img_name}.jpg")


def _build_paths(image_names, is_train):
    base = jpeg_train_dir if is_train else jpeg_test_dir
    if base is None:
        return [None] * len(image_names)
    return [os.path.join(base, f"{n}.jpg") for n in image_names]


def _extract_jpeg_features_from_paths(paths, size=(64, 64)):
    feats = np.zeros((len(paths), 12), dtype=np.float32)
    for i, p in enumerate(paths):
        if p is None or (not os.path.exists(p)):
            continue
        try:
            with Image.open(p) as im:
                im = (
                    ImageOps.exif_transpose(im)
                    .convert("RGB")
                    .resize(size, Image.BILINEAR)
                )
                arr = np.asarray(im, dtype=np.float32) / 255.0  # (H,W,3)
            ch_mean = arr.mean(axis=(0, 1))
            ch_std = arr.std(axis=(0, 1))
            gray = arr.mean(axis=2)
            g_mean = float(gray.mean())
            g_std = float(gray.std())
            dark = float((gray < 0.25).mean())
            bright = float((gray > 0.75).mean())
            feats[i, 0:3] = ch_mean
            feats[i, 3:6] = ch_std
            feats[i, 6] = g_mean
            feats[i, 7] = g_std
            feats[i, 8] = dark
            feats[i, 9] = bright
            feats[i, 10] = float(ch_mean.max() - ch_mean.min())
            feats[i, 11] = float(ch_std.mean())
        except Exception:
            pass
    return feats


def _standardize(train_X, test_X):
    mu = train_X.mean(axis=0, keepdims=True)
    sd = train_X.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return (train_X - mu) / sd, (test_X - mu) / sd


_tr_names = train_df["image_name"].values
_te_names = test_df["image_name"].values
_tr_paths = _build_paths(_tr_names, is_train=True)
_te_paths = _build_paths(_te_names, is_train=False)

if jpeg_train_dir is not None and jpeg_test_dir is not None:
    _Xtr_full = _extract_jpeg_features_from_paths(_tr_paths, size=(64, 64))
    _Xte_full = _extract_jpeg_features_from_paths(_te_paths, size=(64, 64))
else:
    _Xtr_full, _Xte_full = None, None


def _fit_predict_jpeg_lr(train_df, test_df, feature_idx=None, C=1.0):
    if (
        jpeg_train_dir is None
        or jpeg_test_dir is None
        or _Xtr_full is None
        or _Xte_full is None
    ):
        return meta_prior_full_m50.copy()

    Xtr = _Xtr_full
    Xte = _Xte_full

    if feature_idx is not None:
        Xtr = Xtr[:, feature_idx]
        Xte = Xte[:, feature_idx]

    Xtr, Xte = _standardize(Xtr, Xte)
    y = train_df["target"].values.astype(int)

    clf = LogisticRegression(
        C=float(C),
        solver="liblinear",
        max_iter=200,
        random_state=0,
    )
    clf.fit(Xtr, y)
    proba = clf.predict_proba(Xte)[:, 1].astype(float)

    out = sub[["image_name"]].merge(
        pd.DataFrame({"image_name": test_df["image_name"].values, "target": proba}),
        on="image_name",
        how="left",
    )
    global_mean = float(train_df["target"].mean())
    out["target"] = out["target"].fillna(global_mean).astype(float).clip(0.0, 1.0)
    out = out.set_index("image_name").reindex(sub["image_name"]).reset_index()
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
                df = sub[["image_name"]].merge(
                    df[["image_name", "target"]], on="image_name", how="left"
                )
                df["target"] = (
                    df["target"].fillna(fallback_df["target"].values).astype(float)
                )
                df = df.set_index("image_name").reindex(sub["image_name"]).reset_index()
                df["target"] = df["target"].astype(float).clip(0.0, 1.0)
                return df
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
