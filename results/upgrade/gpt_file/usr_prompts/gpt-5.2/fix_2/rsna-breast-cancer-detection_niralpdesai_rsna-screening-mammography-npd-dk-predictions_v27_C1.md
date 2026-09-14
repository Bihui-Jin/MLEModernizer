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
import glob
import time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import cv2

import pydicom

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)



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
test_im_dir = "/kaggle/input/rsna-breast-cancer-detection/test_images/"

paths = [
    os.path.join(test_im_dir, str(patient), f"{int(image)}.dcm")
    for patient, image in zip(data_test["patient_id"], data_test["image_id"])
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
    """
    Read a DICOM file using pydicom. Returns float32 2D array or raises.
    Some Kaggle RSNA images are JPEG2000; if decoder is unavailable, this may fail.
    """
    ds = pydicom.dcmread(path, force=True)
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
start = time.time()
image_size = 512
run_dcm_to_png = True

workdir = "/kaggle/working/test/"
save_dir = os.path.join(workdir, f"processed_{image_size}")
os.makedirs(save_dir, exist_ok=True)

unreadable = 0
written = 0

if run_dcm_to_png:
    for p in tqdm(paths, total=len(paths)):
        patient_id = os.path.basename(os.path.dirname(p))
        image_id = os.path.splitext(os.path.basename(p))[0]

        out_folder = os.path.join(save_dir, patient_id)
        os.makedirs(out_folder, exist_ok=True)
        out_path = os.path.join(out_folder, f"{image_id}.png")

        if os.path.exists(out_path):
            written += 1
            continue

        try:
            arr = read_dicom_pixel_array(p)
            img = process_im_ray(arr, image_size=image_size)
            img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            cv2.imwrite(out_path, img_u8)
            written += 1
        except Exception:
            unreadable += 1
            continue

elapsed = time.time() - start
print(
    f"PNG conversion done. written={written}, unreadable={unreadable}, elapsed_sec={elapsed:.1f}"
)
print("save_dir:", save_dir)




## === cell 6
def onehot(df, encode):
    temp = pd.get_dummies(df[encode], prefix="", prefix_sep="")
    df = df.drop([encode], axis=1)
    df = pd.concat([df, temp], axis=1)
    return df


data_train_relevant = csv_train.copy()
for col in ["laterality", "view"]:
    data_test = onehot(data_test, col)
    data_train_relevant = onehot(data_train_relevant, col)

for col in data_train_relevant.columns:
    if col not in data_test.columns:
        data_test[col] = 0

data_test["implant"] = data_test["implant"].fillna(0).astype("uint8")
data_test.head()



## === cell 7
pngfolder = save_dir  # produced in cell 6

data_test["file"] = data_test.apply(
    lambda x: os.path.join(
        pngfolder, str(int(x["patient_id"])), f"{int(x['image_id'])}.png"
    ),
    axis=1,
)

data_test["png_exists"] = data_test["file"].apply(os.path.exists)
data_test[["file", "png_exists"]].head(), data_test["png_exists"].mean()



## === cell 8

base_rate = float(csv_train["cancer"].mean())

age_mean = float(csv_train["age"].fillna(csv_train["age"].mean()).mean())
age_std = float(csv_train["age"].fillna(age_mean).std() + 1e-6)

age_z = (data_test["age"].fillna(age_mean) - age_mean) / age_std
logit = np.log(base_rate / (1 - base_rate)) + 0.15 * age_z
prob = 1 / (1 + np.exp(-logit))

prob = prob.where(data_test["png_exists"], base_rate)

data_test["cancer_pred_image"] = prob.astype(np.float32)
data_test[["prediction_id", "cancer_pred_image"]].head()



## === cell 9
pred_by_pid = (
    data_test.groupby("prediction_id")["cancer_pred_image"].max().reset_index()
)
pred_by_pid = pred_by_pid.rename(columns={"cancer_pred_image": "cancer"})

sample_sub = pd.read_csv(
    "/kaggle/input/rsna-breast-cancer-detection/sample_submission.csv"
)
sub = sample_sub[["prediction_id"]].merge(pred_by_pid, on="prediction_id", how="left")
sub["cancer"] = sub["cancer"].fillna(base_rate).astype(np.float32)

sub.head(), sub.shape



## === cell 10
assert list(sub.columns) == ["prediction_id", "cancer"]
assert sub["cancer"].between(0, 1).all()
assert sub.isna().sum().sum() == 0
print("Submission rows:", len(sub))



## === cell 11
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
print(sub.head(10))
