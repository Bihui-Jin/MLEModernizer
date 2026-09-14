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

0.0387540746106483

# 6. Current score

0.02212

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.02212) has done: 'The timeout is dominated by the per-image DICOM decode + OpenCV processing + PNG write loop, plus repeated filesystem checks and per-row pandas `apply`. I keep the exact image processing logic and prediction formula, but remove unnecessary disk I/O and repeated directory creation, speed up DICOM reading by skipping non-pixel metadata, parallelize the conversion with a deterministic thread pool, and vectorize file-path construction and existence checks. This preserves the same core logic and outputs (aside from negligible floating-point noise) while cutting wall time substantially. I also avoid importing notebook-only tqdm and ensure the pipeline never re-processes already-converted files.'
- What this solution (achieved 0.02212) has done: 'Your current score is below target (0.02212 vs 0.03875, higher-is-better), so the smallest safe way to move toward the target is to improve the probabilistic calibration without changing the core heuristic. I keep your exact image pipeline and the same “age-adjusted base-rate” idea, but fit the age coefficient on the training CSV (no image labels needed) via a tiny logistic regression solved with Newton’s method, and then apply it to test. I also align one-hot columns between train/test more safely (ensuring both get the same dummy columns) so the fitted coefficient is not distorted by accidental column mismatches. Everything still runs end-to-end and writes `submission.csv` in the required format.'
- What this solution (achieved 0.02212) has done: 'Your current score (0.02212) is below target (0.03875), so we should improve pF1 by increasing recall at low prevalence with minimal semantic changes. I keep your exact DICOM→PNG preprocessing and the same age-only logistic model, but add a single global calibration step that directly optimizes the probabilistic F1 on out-of-fold (patient-grouped) train predictions by learning one scalar multiplier on the logit (temperature scaling). This doesn’t change the model/features/training approach (still the same fitted beta0/beta1), but it better matches the competition metric and typically boosts pF1 by correcting probability sharpness. Finally, I apply the learned logit scale to test, keep the same `groupby(prediction_id).max()` aggregation, and write a valid `submission.csv`.'

# 9. Code solution

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
start = time.time()
image_size = 512
run_dcm_to_png = True

workdir = "/kaggle/working/test/"
save_dir = os.path.join(workdir, f"processed_{image_size}")
os.makedirs(save_dir, exist_ok=True)

out_patient = patient_str  # from cell 4
out_image = image_str
out_dirs = [os.path.join(save_dir, p) for p in out_patient]
out_paths = [os.path.join(d, f"{img}.png") for d, img in zip(out_dirs, out_image)]

for d in set(out_dirs):
    os.makedirs(d, exist_ok=True)

exists_mask = np.fromiter(
    (os.path.exists(p) for p in out_paths), dtype=bool, count=len(out_paths)
)
todo_idx = np.where(~exists_mask)[0]

unreadable = 0
written = int(exists_mask.sum())


def _convert_one(i):
    p = paths[i]
    out_path = out_paths[i]
    try:
        arr = read_dicom_pixel_array(p)
        img = process_im_ray(arr, image_size=image_size)
        img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        ok = cv2.imwrite(out_path, img_u8)
        if not ok:
            return (False, True)  # unreadable=True (treat as failure)
        return (True, False)
    except Exception:
        return (False, True)


if run_dcm_to_png and len(todo_idx) > 0:
    max_workers = min(8, (os.cpu_count() or 4))
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
pngfolder = save_dir  # produced in cell 6
data_test["file"] = (
    pngfolder
    + "/"
    + data_test["patient_id"].astype(np.int64).astype(str)
    + "/"
    + data_test["image_id"].astype(np.int64).astype(str)
    + ".png"
)

data_test["png_exists"] = data_test["file"].map(os.path.exists)
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


def fit_logit_scale_oof_pf1(
    csv_train_in, beta0_in, beta1_in, age_mean_in, age_std_in, n_folds=5
):
    df = csv_train_in[["patient_id", "age", "cancer"]].copy()
    df["age"] = df["age"].fillna(age_mean_in)
    df["age_z"] = (df["age"] - age_mean_in) / age_std_in

    pids = df["patient_id"].astype(np.int64).to_numpy()
    uniq = np.unique(pids)
    order = np.argsort(uniq)
    uniq = uniq[order]
    fold_id = (np.arange(len(uniq)) % n_folds).astype(np.int64)
    pid_to_fold = dict(zip(uniq.tolist(), fold_id.tolist()))
    folds = np.vectorize(pid_to_fold.get, otypes=[np.int64])(pids)

    logits = (beta0_in + beta1_in * df["age_z"].astype(np.float64).to_numpy()).astype(
        np.float64
    )
    y_true = df["cancer"].astype(np.float64).to_numpy()

    scales = np.array([0.5, 0.7, 0.85, 1.0, 1.15, 1.35, 1.6, 2.0], dtype=np.float64)

    best_s = 1.0
    best_score = -1.0
    for s in scales:
        oof_prob = np.empty_like(y_true, dtype=np.float64)
        for k in range(n_folds):
            m = folds == k
            oof_prob[m] = _sigmoid(logits[m] * s)
        sc = probabilistic_f1(y_true, oof_prob)
        if sc > best_score:
            best_score = sc
            best_s = float(s)

    return best_s, best_score


logit_scale, oof_pf1 = fit_logit_scale_oof_pf1(
    csv_train, beta0, beta1, age_mean, age_std, n_folds=5
)
print("Chosen logit_scale:", logit_scale, "OOF pF1:", oof_pf1)

logit_scaled = logit * logit_scale
prob_scaled = _sigmoid(logit_scaled.to_numpy())
prob_scaled = pd.Series(prob_scaled, index=data_test.index).where(
    data_test["png_exists"], base_rate
)
data_test["cancer_pred_image"] = prob_scaled.astype(np.float32)



## === cell 10
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



## === cell 11
assert list(sub.columns) == ["prediction_id", "cancer"]
assert sub["cancer"].between(0, 1).all()
assert sub.isna().sum().sum() == 0
print("Submission rows:", len(sub))



## === cell 12
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
print(sub.head(10))
