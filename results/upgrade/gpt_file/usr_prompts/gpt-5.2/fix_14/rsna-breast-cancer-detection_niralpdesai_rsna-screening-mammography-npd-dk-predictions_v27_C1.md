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
import os
import time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import cv2
import pydicom

from concurrent.futures import ThreadPoolExecutor, as_completed

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def _progress(i, n, every=200):
    if i % every == 0 or i == n:
        print(f"Processed {i}/{n}")




## === cell 1
csv_train = pd.read_csv("/kaggle/input/rsna-breast-cancer-detection/train.csv")
csv_test = pd.read_csv("/kaggle/input/rsna-breast-cancer-detection/test.csv")

train_images_folder = "/kaggle/input/rsna-breast-cancer-detection/train_images"
test_images_folder = "/kaggle/input/rsna-breast-cancer-detection/test_images"

csv_train.shape, csv_test.shape



## === cell 2
data_test = csv_test.copy()
data_test["age"] = data_test["age"].fillna(data_test["age"].mean())



## === cell 3
test_im_dir = "/kaggle/input/rsna-breast-cancer-detection/test_images"
patient_str = data_test["patient_id"].astype(str).to_numpy()
image_str = data_test["image_id"].astype(np.int64).astype(str).to_numpy()
paths = [
    os.path.join(test_im_dir, p, f"{img}.dcm") for p, img in zip(patient_str, image_str)
]

len(paths), paths[0]




## === cell 4
def cut_empty_space_ray(im, T=100, cutedge=10, show=False):
    h, w = im.shape[:2]
    if h <= 2 * cutedge + 1 or w <= 2 * cutedge + 1:
        return im

    impx_cv2_raw = im[cutedge:-cutedge, cutedge:-cutedge]

    im_float = impx_cv2_raw.astype(np.float32)
    _, impx_cv2 = cv2.threshold(im_float, float(T), 255, cv2.THRESH_BINARY)

    im_u8 = np.clip(impx_cv2, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(im_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return impx_cv2_raw

    boundary = max(contours, key=cv2.contourArea)
    x, y, w2, h2 = cv2.boundingRect(boundary)

    if w2 <= 1 or h2 <= 1:
        return impx_cv2_raw

    impx_mask = np.zeros(im_u8.shape, dtype="uint8")
    cv2.drawContours(impx_mask, [boundary], -1, 255, cv2.FILLED)
    impx_cv2_masked = cv2.bitwise_and(impx_cv2_raw, impx_cv2_raw, mask=impx_mask)

    out = impx_cv2_masked[y : y + h2, x : x + w2]
    if show:
        plt.imshow(out, cmap="gray")
        plt.show()
        plt.close()
    return out


def process_im_ray(im, image_size=256, show=False):
    out = im.astype(np.float32)

    minval = float(np.min(out))
    maxval = float(np.max(out))
    if maxval <= minval:
        out = np.zeros((image_size, image_size), dtype=np.float32)
        return out

    Threshold = maxval / 5.0
    out = cut_empty_space_ray(out, T=Threshold, show=show)

    minval2 = float(np.min(out))
    maxval2 = float(np.max(out))
    if maxval2 > minval2:
        out = (out - minval2) / (maxval2 - minval2)
    else:
        out = np.zeros_like(out, dtype=np.float32)

    out = cv2.resize(out, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return out


def read_dicom_pixel_array(path):
    ds = pydicom.dcmread(
        path,
        force=True,
        stop_before_pixels=False,
        specific_tags=["PixelData", "PhotometricInterpretation"],
    )
    arr = ds.pixel_array.astype(np.float32)

    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = arr[..., 0]

    photometric = getattr(ds, "PhotometricInterpretation", None)
    if photometric == "MONOCHROME1":
        arr = np.max(arr) - arr

    return arr




## === cell 5
def ensure_pngs_from_csv(
    df,
    dcm_root,
    workdir,
    image_size,
    max_workers=8,
    patient_col="patient_id",
    image_col="image_id",
):
    start = time.time()
    save_dir = os.path.join(workdir, f"processed_{image_size}")
    os.makedirs(save_dir, exist_ok=True)

    p_str = df[patient_col].astype(str).to_numpy()
    i_str = df[image_col].astype(np.int64).astype(str).to_numpy()

    dcm_paths = [
        os.path.join(dcm_root, p, f"{img}.dcm") for p, img in zip(p_str, i_str)
    ]
    out_dirs = [os.path.join(save_dir, p) for p in p_str]
    out_paths = [os.path.join(d, f"{img}.png") for d, img in zip(out_dirs, i_str)]

    for p in np.unique(p_str):
        os.makedirs(os.path.join(save_dir, p), exist_ok=True)

    exists_mask = np.fromiter(
        (os.path.exists(p) for p in out_paths), dtype=bool, count=len(out_paths)
    )
    todo_idx = np.where(~exists_mask)[0]

    unreadable = 0
    written = int(exists_mask.sum())

    def _convert_one(i):
        pth = dcm_paths[i]
        out_path = out_paths[i]
        try:
            arr = read_dicom_pixel_array(pth)
            img = process_im_ray(arr, image_size=image_size)
            img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            ok = cv2.imwrite(out_path, img_u8)
            if not ok:
                return (False, True)
            return (True, False)
        except Exception:
            return (False, True)

    if len(todo_idx) > 0:
        max_workers = min(int(max_workers), (os.cpu_count() or 4))
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_convert_one, int(i)) for i in todo_idx]
            for fut in as_completed(futs):
                ok, bad = fut.result()
                if ok:
                    written += 1
                if bad and (not ok):
                    unreadable += 1
                completed += 1
                if completed % 200 == 0 or completed == len(futs):
                    print(
                        f"Converted {completed}/{len(futs)} (written={written}, unreadable={unreadable})"
                    )

    elapsed = time.time() - start
    print(
        f"PNG conversion done. written={written}, unreadable={unreadable}, elapsed_sec={elapsed:.1f}"
    )
    print("save_dir:", save_dir)
    return save_dir


image_size = 512
pngfolder = ensure_pngs_from_csv(
    data_test,
    dcm_root="/kaggle/input/rsna-breast-cancer-detection/test_images",
    workdir="/kaggle/working/test/",
    image_size=image_size,
    max_workers=8,
)




## === cell 6
def onehot(df, encode):
    temp = pd.get_dummies(df[encode], prefix="", prefix_sep="")
    df = df.drop([encode], axis=1)
    df = pd.concat([df, temp], axis=1)
    return df


data_train_relevant = csv_train.copy()
both = pd.concat(
    [data_train_relevant.assign(_is_train=1), data_test.assign(_is_train=0)],
    axis=0,
    ignore_index=True,
)
for col in ["laterality", "view"]:
    both = onehot(both, col)

data_train_relevant = (
    both[both["_is_train"] == 1].drop(columns=["_is_train"]).reset_index(drop=True)
)
data_test = (
    both[both["_is_train"] == 0].drop(columns=["_is_train"]).reset_index(drop=True)
)

data_test["implant"] = data_test["implant"].fillna(0).astype("uint8")
data_test.head()



## === cell 7
data_test["file"] = (
    pngfolder
    + "/"
    + data_test["patient_id"].astype(np.int64).astype(str)
    + "/"
    + data_test["image_id"].astype(np.int64).astype(str)
    + ".png"
)

data_test["png_exists"] = pd.Index(data_test["file"]).map(os.path.exists).to_numpy()
data_test[["file", "png_exists"]].head(), data_test["png_exists"].mean()



## === cell 8
base_rate = float(csv_train["cancer"].mean())

age_train = csv_train["age"].copy()
age_mean = float(age_train.fillna(age_train.mean()).mean())
age_std = float(age_train.fillna(age_mean).std() + 1e-6)

x = ((age_train.fillna(age_mean) - age_mean) / age_std).astype(np.float64).to_numpy()
y = csv_train["cancer"].astype(np.float64).to_numpy()


def _sigmoid(z):
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


beta0 = float(np.log(base_rate / (1.0 - base_rate)))
beta1 = 0.15

for _ in range(8):
    z = beta0 + beta1 * x
    p = _sigmoid(z)
    w = p * (1.0 - p) + 1e-12

    g0 = np.sum(p - y)
    g1 = np.sum((p - y) * x)

    h00 = np.sum(w)
    h01 = np.sum(w * x)
    h11 = np.sum(w * x * x) + 1e-12

    det = h00 * h11 - h01 * h01
    if not np.isfinite(det) or det <= 0:
        break

    step0 = (h11 * g0 - h01 * g1) / det
    step1 = (-h01 * g0 + h00 * g1) / det

    beta0 -= float(step0)
    beta1 -= float(step1)

print(
    "Fitted age logit params:", {"beta0": beta0, "beta1": beta1, "base_rate": base_rate}
)

age_z_test = ((data_test["age"].fillna(age_mean) - age_mean) / age_std).astype(
    np.float64
)
logit = beta0 + beta1 * age_z_test
prob = _sigmoid(logit.to_numpy())

prob = pd.Series(prob, index=data_test.index).where(data_test["png_exists"], base_rate)

data_test["cancer_pred_image"] = prob.astype(np.float32)
data_test[["prediction_id", "cancer_pred_image"]].head()




## === cell 9
def probabilistic_f1(y_true, y_prob, eps=1e-12):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    pTP = np.sum(y_true * y_prob)
    pFP = np.sum((1.0 - y_true) * y_prob)
    pFN = np.sum(y_true * (1.0 - y_prob))
    pPrec = pTP / (pTP + pFP + eps)
    pRec = pTP / (pTP + pFN + eps)
    return 2.0 * pPrec * pRec / (pPrec + pRec + eps)


def _fit_age_logistic_newton(x_tr, y_tr, base_rate_in, n_iter=8):
    beta0_loc = float(np.log(base_rate_in / (1.0 - base_rate_in)))
    beta1_loc = 0.15
    x_tr = np.asarray(x_tr, dtype=np.float64)
    y_tr = np.asarray(y_tr, dtype=np.float64)

    for _ in range(n_iter):
        z = beta0_loc + beta1_loc * x_tr
        p = _sigmoid(z)
        w = p * (1.0 - p) + 1e-12

        g0 = np.sum(p - y_tr)
        g1 = np.sum((p - y_tr) * x_tr)

        h00 = np.sum(w)
        h01 = np.sum(w * x_tr)
        h11 = np.sum(w * x_tr * x_tr) + 1e-12

        det = h00 * h11 - h01 * h01
        if not np.isfinite(det) or det <= 0:
            break

        step0 = (h11 * g0 - h01 * g1) / det
        step1 = (-h01 * g0 + h00 * g1) / det

        beta0_loc -= float(step0)
        beta1_loc -= float(step1)

    return float(beta0_loc), float(beta1_loc)


def _patient_folds(patient_ids, n_folds, seed=RANDOM_SEED):
    p = np.asarray(patient_ids, dtype=np.int64)
    x = (p.astype(np.uint64) * np.uint64(2654435761) + np.uint64(seed)) % np.uint64(
        n_folds
    )
    return x.astype(np.int64)


def _png_mean_feature(path):
    try:
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            return np.nan
        return float(im.mean() / 255.0)
    except Exception:
        return np.nan


def _build_train_png_paths(csv_train_in, png_root, image_size_in):
    return (
        png_root
        + "/"
        + csv_train_in["patient_id"].astype(np.int64).astype(str)
        + "/"
        + csv_train_in["image_id"].astype(np.int64).astype(str)
        + ".png"
    )


def _ensure_train_pngs(csv_train_in, image_size_in=512, max_workers=8):
    return ensure_pngs_from_csv(
        csv_train_in,
        dcm_root="/kaggle/input/rsna-breast-cancer-detection/train_images",
        workdir="/kaggle/working/train/",
        image_size=image_size_in,
        max_workers=max_workers,
        patient_col="patient_id",
        image_col="image_id",
    )


def _fit_age_img_logistic_newton(x1_tr, x2_tr, y_tr, base_rate_in, n_iter=10):
    b0 = float(np.log(base_rate_in / (1.0 - base_rate_in)))
    b1 = 0.15
    b2 = 0.0
    x1_tr = np.asarray(x1_tr, dtype=np.float64)
    x2_tr = np.asarray(x2_tr, dtype=np.float64)
    y_tr = np.asarray(y_tr, dtype=np.float64)

    for _ in range(n_iter):
        z = b0 + b1 * x1_tr + b2 * x2_tr
        p = _sigmoid(z)
        w = p * (1.0 - p) + 1e-12

        g0 = np.sum(p - y_tr)
        g1 = np.sum((p - y_tr) * x1_tr)
        g2 = np.sum((p - y_tr) * x2_tr)

        h00 = np.sum(w)
        h01 = np.sum(w * x1_tr)
        h02 = np.sum(w * x2_tr)
        h11 = np.sum(w * x1_tr * x1_tr) + 1e-12
        h12 = np.sum(w * x1_tr * x2_tr)
        h22 = np.sum(w * x2_tr * x2_tr) + 1e-12

        H = np.array(
            [[h00, h01, h02], [h01, h11, h12], [h02, h12, h22]], dtype=np.float64
        )
        g = np.array([g0, g1, g2], dtype=np.float64)

        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break

        if not np.all(np.isfinite(step)):
            break

        b0 -= float(step[0])
        b1 -= float(step[1])
        b2 -= float(step[2])

    return float(b0), float(b1), float(b2)


def _compute_png_means_cached(paths_list, exists_mask, max_workers=8):
    paths_list = list(paths_list)
    exists_mask = np.asarray(exists_mask, dtype=bool)
    means = np.full(len(paths_list), np.nan, dtype=np.float64)

    idx = np.where(exists_mask)[0]
    if len(idx) == 0:
        return means

    max_workers = min(int(max_workers), (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_png_mean_feature, paths_list[int(i)]): int(i) for i in idx}
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                means[i] = float(fut.result())
            except Exception:
                means[i] = np.nan
            done += 1
            if done % 1000 == 0 or done == len(futs):
                print(f"[png mean] computed {done}/{len(futs)}")
    return means


def fit_age_img_coeff_oof(
    csv_train_in, age_mean_in, age_std_in, image_size_in=512, n_folds=5
):
    png_root_tr = _ensure_train_pngs(
        csv_train_in, image_size_in=image_size_in, max_workers=8
    )

    df = csv_train_in[["patient_id", "image_id", "age", "cancer"]].copy()
    df["age"] = df["age"].fillna(age_mean_in)
    df["age_z"] = (df["age"] - age_mean_in) / age_std_in

    df["file"] = _build_train_png_paths(df, png_root_tr, image_size_in)
    df["png_exists"] = pd.Index(df["file"]).map(os.path.exists).to_numpy()

    means = _compute_png_means_cached(
        df["file"].tolist(), df["png_exists"].to_numpy(), max_workers=8
    )
    mean_fill = float(np.nanmean(means)) if np.isfinite(np.nanmean(means)) else 0.5
    means = np.where(np.isfinite(means), means, mean_fill)

    img_mean_std = float(np.std(means) + 1e-6)
    img_mean_mean = float(np.mean(means))
    df["img_mean_z"] = (means - img_mean_mean) / img_mean_std

    folds = _patient_folds(df["patient_id"].to_numpy(), n_folds, seed=RANDOM_SEED)
    y_all = df["cancer"].astype(np.float64).to_numpy()
    x1_all = df["age_z"].astype(np.float64).to_numpy()
    x2_all = df["img_mean_z"].astype(np.float64).to_numpy()

    oof_logits = np.empty_like(y_all, dtype=np.float64)

    for k in range(n_folds):
        tr_mask = folds != k
        va_mask = folds == k
        y_tr = y_all[tr_mask]
        br_k = float(np.mean(y_tr))
        br_k = min(max(br_k, 1e-6), 1.0 - 1e-6)
        b0_k, b1_k, b2_k = _fit_age_img_logistic_newton(
            x1_all[tr_mask], x2_all[tr_mask], y_tr, br_k, n_iter=10
        )
        oof_logits[va_mask] = b0_k + b1_k * x1_all[va_mask] + b2_k * x2_all[va_mask]

    br_full = min(max(float(np.mean(y_all)), 1e-6), 1.0 - 1e-6)
    b0_full, b1_full, b2_full = _fit_age_img_logistic_newton(
        x1_all, x2_all, y_all, br_full, n_iter=10
    )

    return {
        "b0": b0_full,
        "b1": b1_full,
        "b2": b2_full,
        "img_mean_mean": img_mean_mean,
        "img_mean_std": img_mean_std,
        "mean_fill": mean_fill,
        "oof_logits": oof_logits,
        "y_true": y_all,
        "png_root_train": png_root_tr,
        "train_png_means": means,  # cached for reuse
    }


age_img_fit = fit_age_img_coeff_oof(
    csv_train, age_mean, age_std, image_size_in=image_size, n_folds=5
)
print(
    "Fitted age+img logistic params (full):",
    {k: age_img_fit[k] for k in ["b0", "b1", "b2"]},
)

test_means = _compute_png_means_cached(
    data_test["file"].tolist(), data_test["png_exists"].to_numpy(), max_workers=8
)
test_means = np.where(
    np.isfinite(test_means), test_means, float(age_img_fit["mean_fill"])
)
img_mean_z_test = (test_means - float(age_img_fit["img_mean_mean"])) / float(
    age_img_fit["img_mean_std"]
)

logit = (
    float(age_img_fit["b0"])
    + float(age_img_fit["b1"]) * age_z_test.to_numpy(dtype=np.float64)
    + float(age_img_fit["b2"]) * img_mean_z_test
)
prob = _sigmoid(logit)

prob = pd.Series(prob, index=data_test.index).where(data_test["png_exists"], base_rate)
data_test["cancer_pred_image"] = prob.astype(np.float32)




## === cell 10
def fit_scale_shift_oof_pf1_from_logits(oof_logits, y_true):
    oof_logits = np.asarray(oof_logits, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)

    scales = np.array(
        [
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
            1.00,
            1.10,
            1.25,
            1.40,
            1.60,
            1.90,
            2.30,
            2.80,
            3.40,
        ],
        dtype=np.float64,
    )
    shifts = np.array(
        [
            -4.0,
            -3.5,
            -3.0,
            -2.6,
            -2.4,
            -2.2,
            -2.0,
            -1.8,
            -1.6,
            -1.45,
            -1.3,
            -1.15,
            -1.0,
            -0.9,
            -0.8,
            -0.7,
            -0.6,
            -0.5,
            -0.45,
            -0.4,
            -0.35,
            -0.3,
            -0.25,
            -0.2,
            -0.15,
            -0.1,
            -0.05,
            0.0,
            0.05,
            0.1,
            0.15,
            0.2,
            0.3,
            0.4,
            0.5,
            0.65,
            0.8,
        ],
        dtype=np.float64,
    )

    zz = oof_logits[None, None, :] * scales[:, None, None] + shifts[None, :, None]
    pp = _sigmoid(zz)

    y = y_true[None, None, :]
    pTP = np.sum(y * pp, axis=2)
    pFP = np.sum((1.0 - y) * pp, axis=2)
    pFN = np.sum(y * (1.0 - pp), axis=2)
    eps = 1e-12
    pPrec = pTP / (pTP + pFP + eps)
    pRec = pTP / (pTP + pFN + eps)
    f1 = 2.0 * pPrec * pRec / (pPrec + pRec + eps)

    flat_idx = int(np.nanargmax(f1))
    si, bi = np.unravel_index(flat_idx, f1.shape)
    return float(scales[si]), float(shifts[bi]), float(f1[si, bi])


cal_s, cal_b, oof_pf1 = fit_scale_shift_oof_pf1_from_logits(
    age_img_fit["oof_logits"], age_img_fit["y_true"]
)
print(
    "Chosen calibration (true OOF, on age+img logits):",
    {"scale": cal_s, "shift": cal_b, "OOF pF1": oof_pf1},
)

logit_cal = logit * float(cal_s) + float(cal_b)
prob_cal = _sigmoid(logit_cal)
prob_cal = pd.Series(prob_cal, index=data_test.index).where(
    data_test["png_exists"], base_rate
)
data_test["cancer_pred_image"] = prob_cal.astype(np.float32)




## === cell 11
def _pool_power_mean(probs, p):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, 1.0)
    if p == 0:
        return float(np.exp(np.mean(np.log(np.clip(probs, 1e-12, 1.0)))))
    return float(np.power(np.mean(np.power(probs, p)), 1.0 / p))


def _build_oof_image_probs(
    csv_train_in, age_mean_in, age_std_in, cal_s_in, cal_b_in, n_folds=5
):
    png_root_tr = age_img_fit["png_root_train"]  # already ensured/converted

    df = csv_train_in[["patient_id", "image_id", "laterality", "age", "cancer"]].copy()
    df["age"] = df["age"].fillna(age_mean_in)
    df["age_z"] = (df["age"] - age_mean_in) / age_std_in

    df["file"] = _build_train_png_paths(df, png_root_tr, image_size)
    df["png_exists"] = pd.Index(df["file"]).map(os.path.exists).to_numpy()

    means = np.asarray(age_img_fit["train_png_means"], dtype=np.float64)
    means = np.where(np.isfinite(means), means, float(age_img_fit["mean_fill"]))
    img_z = (means - float(age_img_fit["img_mean_mean"])) / float(
        age_img_fit["img_mean_std"]
    )
    df["img_mean_z"] = img_z

    folds = _patient_folds(df["patient_id"].to_numpy(), n_folds, seed=RANDOM_SEED)

    x1_all = df["age_z"].astype(np.float64).to_numpy()
    x2_all = df["img_mean_z"].astype(np.float64).to_numpy()
    y_true = df["cancer"].astype(np.float64).to_numpy()

    oof_logits = np.empty_like(y_true, dtype=np.float64)

    for k in range(n_folds):
        tr_mask = folds != k
        va_mask = folds == k

        y_tr = y_true[tr_mask]
        br_k = float(np.mean(y_tr))
        br_k = min(max(br_k, 1e-6), 1.0 - 1e-6)

        b0_k, b1_k, b2_k = _fit_age_img_logistic_newton(
            x1_all[tr_mask], x2_all[tr_mask], y_tr, br_k, n_iter=10
        )
        oof_logits[va_mask] = b0_k + b1_k * x1_all[va_mask] + b2_k * x2_all[va_mask]

    oof_prob_img = _sigmoid(oof_logits * float(cal_s_in) + float(cal_b_in))
    df["oof_prob_img"] = oof_prob_img.astype(np.float64)

    df["pid_lat"] = (
        df["patient_id"].astype(np.int64).astype(str)
        + "-"
        + df["laterality"].astype(str)
    )
    return df


def tune_pooling_exponent_pf1(
    csv_train_in, age_mean_in, age_std_in, cal_s_in, cal_b_in, n_folds=5
):
    df_oof = _build_oof_image_probs(
        csv_train_in, age_mean_in, age_std_in, cal_s_in, cal_b_in, n_folds=n_folds
    )

    p_grid = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        8.0,
        10.0,
        12.0,
        15.0,
        18.0,
        20.0,
        22.0,
        25.0,
        30.0,
        40.0,
        60.0,
    ]

    grp = df_oof.groupby("pid_lat", sort=False)
    y_pid = grp["cancer"].max().astype(np.float64)
    prob_lists = grp["oof_prob_img"].apply(lambda s: s.values)

    best_p = 20.0
    best_score = -1.0

    for p in p_grid:
        pooled = prob_lists.apply(lambda arr: _pool_power_mean(arr, p)).astype(
            np.float64
        )
        score = probabilistic_f1(y_pid.values, pooled.reindex(y_pid.index).values)
        if score > best_score:
            best_score = float(score)
            best_p = float(p)

    return best_p, best_score


pool_p, pool_oof_pf1 = tune_pooling_exponent_pf1(
    csv_train, age_mean, age_std, cal_s, cal_b, n_folds=5
)
print("Chosen pooling exponent:", {"p": pool_p, "OOF pF1 (breast-level)": pool_oof_pf1})



## === cell 12
if float(pool_p) >= 59.999:
    pred_by_pid = (
        data_test.groupby("prediction_id")["cancer_pred_image"].max().reset_index()
    )
else:
    pred_by_pid = (
        data_test.groupby("prediction_id")["cancer_pred_image"]
        .apply(lambda s: _pool_power_mean(s.values, float(pool_p)))
        .reset_index()
        .rename(columns={"cancer_pred_image": "cancer_pred_image"})
    )

pred_by_pid = pred_by_pid.rename(columns={"cancer_pred_image": "cancer"})

sample_sub = pd.read_csv(
    "/kaggle/input/rsna-breast-cancer-detection/sample_submission.csv"
)
sub = sample_sub[["prediction_id"]].merge(pred_by_pid, on="prediction_id", how="left")
sub["cancer"] = sub["cancer"].fillna(base_rate).astype(np.float32)

sub.head(), sub.shape



## === cell 13
assert list(sub.columns) == ["prediction_id", "cancer"]
assert sub["cancer"].between(0, 1).all()
assert sub.isna().sum().sum() == 0
print("Submission rows:", len(sub))



## === cell 14
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
print(sub.head(10))
