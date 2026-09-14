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
import sys
import math
import random
import warnings

import numpy as np
import pandas as pd

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def seed_everything(seed: int = 534):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(534)



## === cell 1
data_dir = "/kaggle/input/rsna-breast-cancer-detection"

target_size = [224, 224]
batch_size = 16
num_epochs = 6



## === cell 2
try:
    import pydicom

    _HAS_PYDICOM = True
except Exception:
    _HAS_PYDICOM = False


def _blank_image_uint8() -> np.ndarray:
    return np.zeros((target_size[0], target_size[1]), dtype=np.uint8)


def _read_dicom_pixel_array(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    if not _HAS_PYDICOM:
        return _blank_image_uint8().astype(np.float32)

    dcm = pydicom.dcmread(path, force=True)

    try:
        arr = dcm.pixel_array.astype(np.float32)

        slope = float(getattr(dcm, "RescaleSlope", 1.0))
        intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept

        pi = str(getattr(dcm, "PhotometricInterpretation", "")).upper()
        if pi == "MONOCHROME1":
            arr = arr.max() - arr

        return arr
    except Exception:
        pass

    try:
        from pydicom.encaps import generate_pixel_data_frame

        frames = list(generate_pixel_data_frame(dcm.PixelData))
        if len(frames) < 1:
            raise RuntimeError("No frames found in encapsulated PixelData.")
        frame0 = frames[0]

        buf = np.frombuffer(frame0, dtype=np.uint8)

        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img_color = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img_color is None:
                raise RuntimeError("OpenCV failed to decode encapsulated frame bytes.")
            img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        arr = img.astype(np.float32)

        pi = str(getattr(dcm, "PhotometricInterpretation", "")).upper()
        if pi == "MONOCHROME1":
            arr = arr.max() - arr

        return arr
    except Exception as e:
        raise RuntimeError(
            f"Failed to decode DICOM pixel data (likely compressed without plugins): {path}. Error: {e}"
        ) from e


def normalize_xray(path):
    dicom_arr = _read_dicom_pixel_array(path)
    dicom_arr = dicom_arr - np.min(dicom_arr)
    mx = np.max(dicom_arr)
    if mx > 0:
        dicom_arr = dicom_arr / mx
    dicom_arr = np.clip(dicom_arr, 0, 1)
    dicom_arr = (dicom_arr * 255).astype(np.uint8)
    return dicom_arr


def crop_and_resize(image, crop_size=5):
    if (
        crop_size > 0
        and image.shape[0] > 2 * crop_size
        and image.shape[1] > 2 * crop_size
    ):
        image = image[crop_size:-crop_size, crop_size:-crop_size]
    image = cv2.resize(image, target_size[::-1], cv2.INTER_LINEAR)
    return image


_CACHE_ROOT = "/kaggle/working/rsna_png_cache_224"
os.makedirs(_CACHE_ROOT, exist_ok=True)


def _cache_png_path_from_dcm(file_path: str) -> str:
    rel = file_path.split("/", 4)[-1]
    if rel.endswith(".dcm"):
        rel = rel[:-4] + ".png"
    else:
        rel = rel + ".png"
    return os.path.join(_CACHE_ROOT, rel)


_PNG_WRITE_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 1]


def preprocess_and_save(file_path):
    png_path = _cache_png_path_from_dcm(file_path)
    try:
        if os.path.exists(png_path):
            img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError("Cached PNG unreadable")
            h, w = img.shape[:2]
        else:
            image = normalize_xray(file_path)
            h, w = image.shape[:2]
            img = crop_and_resize(image)
            os.makedirs(os.path.dirname(png_path), exist_ok=True)
            cv2.imwrite(png_path, img, _PNG_WRITE_PARAMS)
    except Exception:
        h, w = target_size[0], target_size[1]
        img = _blank_image_uint8()

    sub_path = file_path.split("/", 4)[-1].split(".dcm")[0] + ".png"
    infos = sub_path.split("/")
    pid = infos[-2]
    iid = infos[-1].replace(".png", "")
    return pid, iid, h, w, img


def pfbeta_torch(labels, preds, beta=1):
    preds = np.clip(preds, 0, 1)
    y_true_count = labels.sum()
    ctp = preds[labels == 1].sum()
    cfp = preds[labels == 0].sum()
    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp + 1e-12)
    c_recall = ctp / (y_true_count + 1e-12)
    if c_precision > 0 and c_recall > 0:
        result = (
            (1 + beta_squared)
            * (c_precision * c_recall)
            / (beta_squared * c_precision + c_recall + 1e-12)
        )
        return float(result)
    else:
        return 0.0


def pfbeta_torch_tensor(
    labels: torch.Tensor, preds: torch.Tensor, beta: float = 1.0
) -> float:
    labels = labels.view(-1)
    preds = preds.view(-1).clamp(0.0, 1.0)
    y_true_count = labels.sum()
    ctp = preds[labels == 1].sum()
    cfp = preds[labels == 0].sum()
    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp + 1e-12)
    c_recall = ctp / (y_true_count + 1e-12)
    if (c_precision > 0).item() and (c_recall > 0).item():
        result = (
            (1 + beta_squared)
            * (c_precision * c_recall)
            / (beta_squared * c_precision + c_recall + 1e-12)
        )
        return float(result.item())
    return 0.0




## === cell 3
train_df = pd.read_csv(f"{data_dir}/train.csv")
train_df["dcm_path"] = (
    data_dir
    + "/train_images/"
    + train_df["patient_id"].astype(str)
    + "/"
    + train_df["image_id"].astype(str)
    + ".dcm"
)
print("train_df:", train_df.shape)
print(train_df.head(2))

test_df = pd.read_csv(f"{data_dir}/test.csv")
test_df["dcm_path"] = (
    data_dir
    + "/test_images/"
    + test_df["patient_id"].astype(str)
    + "/"
    + test_df["image_id"].astype(str)
    + ".dcm"
)
print("test_df:", test_df.shape)
print(test_df.head(2))



## === cell 4
train_df["png_path"] = train_df["dcm_path"].map(_cache_png_path_from_dcm)
test_df["png_path"] = test_df["dcm_path"].map(_cache_png_path_from_dcm)


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]


def _cv_gray_to_normalized_tensor(img_gray_uint8: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(img_gray_uint8).to(torch.float32).div_(255.0)  # [H,W]
    x = x.unsqueeze(0).repeat(3, 1, 1)  # [3,H,W]
    x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
    return x


def _apply_train_aug_cv2(
    img_gray_uint8: np.ndarray, rng: np.random.RandomState
) -> np.ndarray:
    img = img_gray_uint8

    if rng.rand() < 0.5:
        img = cv2.flip(img, 1)

    if rng.rand() < 0.5:
        img = cv2.flip(img, 0)

    b = 1.0 + (rng.rand() * 0.2 - 0.1)
    c = 1.0 + (rng.rand() * 0.2 - 0.1)
    img = img.astype(np.float32) * c + (b - 1.0) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)

    h, w = img.shape[:2]
    angle = rng.uniform(-45.0, 45.0)
    scale = rng.uniform(0.7, 1.3)
    tx = rng.uniform(-0.3, 0.3) * w
    ty = rng.uniform(-0.3, 0.3) * h
    M = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    img = cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )

    if rng.rand() < 0.7:
        ds = 0.3
        d = ds * min(h, w)
        src = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
        )
        dst = src + rng.uniform(-d, d, size=(4, 2)).astype(np.float32)
        P = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(
            img, P, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )

    return img


def _random_erasing_inplace(
    x_chw: torch.Tensor,
    rng: np.random.RandomState,
    p=0.5,
    scale=(0.1, 0.5),
    ratio=(0.3, 3.3),
) -> torch.Tensor:
    if rng.rand() >= p:
        return x_chw
    _, H, W = x_chw.shape
    area = H * W
    erase_area = rng.uniform(scale[0], scale[1]) * area
    aspect = rng.uniform(ratio[0], ratio[1])
    h = int(round(math.sqrt(erase_area * aspect)))
    w = int(round(math.sqrt(erase_area / aspect)))
    if h <= 0 or w <= 0 or h >= H or w >= W:
        return x_chw
    y1 = rng.randint(0, H - h + 1)
    x1 = rng.randint(0, W - w + 1)
    x_chw[:, y1 : y1 + h, x1 : x1 + w] = 0.0
    return x_chw


class MyDataset(Dataset):
    def __init__(
        self,
        df,
        transform=None,
        fast_val_test_tensor=False,
        train_aug_cv2=False,
        seed=534,
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.fast_val_test_tensor = fast_val_test_tensor
        self.train_aug_cv2 = train_aug_cv2
        self.seed = int(seed)

        self._cancer = self.df["cancer"].astype(np.float32).values
        self._dcm_path = self.df["dcm_path"].astype(str).values
        self._png_path = self.df["png_path"].astype(str).values

    def __len__(self):
        return len(self._cancer)

    def __getitem__(self, index):
        cancer = torch.tensor(float(self._cancer[index]), dtype=torch.float32)

        png_path = self._png_path[index]
        img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            _, _, _, _, img = preprocess_and_save(self._dcm_path[index])

        if self.fast_val_test_tensor:
            img = _cv_gray_to_normalized_tensor(img)
            return {"cancer": cancer, "images": img}

        if self.train_aug_cv2:
            rng = np.random.RandomState(self.seed + index)
            img = _apply_train_aug_cv2(img, rng)

            x = (
                torch.from_numpy(img)
                .to(torch.float32)
                .div_(255.0)
                .unsqueeze(0)
                .repeat(3, 1, 1)
            )
            x = _random_erasing_inplace(
                x, rng, p=0.5, scale=(0.1, 0.5), ratio=(0.3, 3.3)
            )
            x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
            return {"cancer": cancer, "images": x}

        img = Image.fromarray(img).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"cancer": cancer, "images": img}




## === cell 5
class PretrainedBinaryClassifier(nn.Module):
    def __init__(self):
        super(PretrainedBinaryClassifier, self).__init__()
        self.model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.model(x)
        return x




## === cell 6
random_seed = 534
n0 = min(55, int((train_df.cancer == 0).sum()))
n1 = min(45, int((train_df.cancer == 1).sum()))

train_subset_0 = train_df[train_df.cancer == 0].sample(n=n0, random_state=random_seed)
train_subset_1 = train_df[train_df.cancer == 1].sample(n=n1, random_state=random_seed)
train_combined = pd.concat([train_subset_0, train_subset_1]).reset_index(drop=True)

print("train_combined:", train_combined.shape)
print(train_combined.laterality.value_counts())
print(train_combined.cancer.value_counts())

training_set, validation_set = train_test_split(
    train_combined, test_size=0.2, random_state=276, stratify=train_combined["cancer"]
)
print("training_set:", training_set.shape)
print("validation_set:", validation_set.shape)



## === cell 7
from concurrent.futures import ProcessPoolExecutor


def _build_png_cache_for_df(df: pd.DataFrame, desc: str, max_workers: int):
    dcm_paths = df["dcm_path"].astype(str).tolist()
    png_paths = df["png_path"].astype(str).tolist()

    missing_dcm = [p for p, pp in zip(dcm_paths, png_paths) if not os.path.exists(pp)]
    m = len(missing_dcm)
    print(f"[cache] {desc}: {len(dcm_paths)} total, {m} missing")

    if m == 0:
        return

    cpu = os.cpu_count() or 2
    workers = max(1, min(max_workers, cpu))
    chunksize = 32 if workers > 1 else 1

    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for _ in ex.map(preprocess_and_save, missing_dcm, chunksize=chunksize):
            done += 1
            if done % 250 == 0 or done == m:
                print(f"[cache] {desc}: {done}/{m} newly cached")


_cache_workers = min(4, (os.cpu_count() or 2))
_build_png_cache_for_df(training_set, "train", max_workers=_cache_workers)
_build_png_cache_for_df(validation_set, "val", max_workers=_cache_workers)


class ToTensorWithErasing(object):
    def __call__(self, img):
        img_tensor = transforms.functional.to_tensor(img)
        img_tensor = transforms.RandomErasing(
            p=0.5, scale=(0.1, 0.5), ratio=(0.3, 3.3)
        )(img_tensor)
        return img_tensor


train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=45, translate=(0.3, 0.3), scale=(0.7, 1.3)),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.7),
        ToTensorWithErasing(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = MyDataset(
    training_set,
    transform=train_transform,
    fast_val_test_tensor=False,
    train_aug_cv2=True,
    seed=534,
)

val_dataset = MyDataset(
    validation_set, transform=val_transform, fast_val_test_tensor=True
)

print("The training dataset contains", len(train_dataset), "samples.")
print("The val dataset contains", len(val_dataset), "samples.")
print("Training dataset transform is:", train_dataset.transform)
print("Validation dataset transform is:", val_dataset.transform)




## === cell 8
def _seed_worker(worker_id: int):
    worker_seed = (534 + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(534)

num_workers = min(4, (os.cpu_count() or 2))
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
    worker_init_fn=_seed_worker,
    generator=g,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
    worker_init_fn=_seed_worker,
    generator=g,
)

print("Number of training batches", len(train_loader))
print("Number of validation batches", len(val_loader))

print("Number of training examples before augmentation:", len(training_set))
print("Number of training examples after augmentation:", len(train_dataset))



## === cell 9
print(torch.cuda.is_available())
print(torch.cuda.device_count())

model = PretrainedBinaryClassifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.BCEWithLogitsLoss().to(device)

learning_rate = 0.0001
weight_decay = 0.01
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)



## === cell 10
train_losses = []
train_acc_metric = []
train_pf1_metric = []

val_losses = []
val_acc_metric = []
val_pf1_metric = []

for epoch in range(num_epochs):
    running_train_loss = 0.0
    running_train_acc = 0.0

    running_train_pf1 = 0.0

    model.train()
    for i, data in enumerate(train_loader, 0):
        inputs, targets = data["images"], data["cancer"]
        targets = targets.view(-1, 1)

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)  # logits
        loss = criterion(outputs, targets)

        probs = torch.sigmoid(outputs)
        predicted = torch.round(probs)
        correct = (predicted == targets).sum().item()
        accuracy = correct / targets.size(0)

        pf1_metric = pfbeta_torch_tensor(targets.detach(), probs.detach())

        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()
        running_train_acc += accuracy
        running_train_pf1 += pf1_metric

    avg_train_loss = running_train_loss / max(1, len(train_loader))
    avg_train_acc = running_train_acc / max(1, len(train_loader))
    avg_train_pf1 = running_train_pf1 / max(1, len(train_loader))
    train_losses.append(avg_train_loss)
    train_acc_metric.append(avg_train_acc)
    train_pf1_metric.append(avg_train_pf1)
    print(
        "Epoch %d, avg training loss: %.3f, avg training accuracy: %.3f, avg training pf1: %.3f"
        % (epoch + 1, avg_train_loss, avg_train_acc, avg_train_pf1)
    )

    model.eval()
    running_val_loss = 0.0
    running_val_acc = 0.0
    running_val_pf1 = 0.0

    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs, targets = data["images"], data["cancer"]
            targets = targets.view(-1, 1)

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)  # logits
            loss = criterion(outputs, targets)

            probs = torch.sigmoid(outputs)
            predicted = torch.round(probs)
            correct = (predicted == targets).sum().item()
            accuracy = correct / targets.size(0)

            pf1_metric = pfbeta_torch_tensor(targets.detach(), probs.detach())

            running_val_loss += loss.item()
            running_val_acc += accuracy
            running_val_pf1 += pf1_metric

    avg_val_loss = running_val_loss / max(1, len(val_loader))
    avg_val_acc = running_val_acc / max(1, len(val_loader))
    avg_val_pf1 = running_val_pf1 / max(1, len(val_loader))
    val_losses.append(avg_val_loss)
    val_acc_metric.append(avg_val_acc)
    val_pf1_metric.append(avg_val_pf1)
    print(
        "Epoch %d, avg validation loss: %.3f, avg validation accuracy: %.3f, avg validation pf1: %.3f"
        % (epoch + 1, avg_val_loss, avg_val_acc, avg_val_pf1)
    )




## === cell 11
class TestDataset(Dataset):
    def __init__(self, df, transform=None, fast_tensor=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.fast_tensor = fast_tensor
        self._pred_id = self.df["prediction_id"].astype(str).values
        self._dcm_path = self.df["dcm_path"].astype(str).values
        self._png_path = self.df["png_path"].astype(str).values

    def __len__(self):
        return len(self._pred_id)

    def __getitem__(self, index):
        png_path = self._png_path[index]
        img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            _, _, _, _, img = preprocess_and_save(self._dcm_path[index])

        if self.fast_tensor:
            img = _cv_gray_to_normalized_tensor(img)
            return self._pred_id[index], img

        img = Image.fromarray(img).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return self._pred_id[index], img


test_batch_size = 32
test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_dataset = TestDataset(test_df, transform=test_transform, fast_tensor=True)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=test_batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
    worker_init_fn=_seed_worker,
    generator=g,
)

model.eval()

pred_ids_all = []
preds_all = []

with torch.inference_mode():
    for prediction_ids, images in test_dataloader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)

        pred_ids_all.extend(list(prediction_ids))
        preds_all.append(probs)

preds_all = np.concatenate(preds_all, axis=0)
pred_df = pd.DataFrame(
    {"prediction_id": pred_ids_all, "cancer": preds_all.astype(float)}
)

sub = pred_df.groupby("prediction_id", as_index=False)["cancer"].mean()

sample_sub = pd.read_csv(f"{data_dir}/sample_submission.csv")
sub = sample_sub[["prediction_id"]].merge(sub, on="prediction_id", how="left")
sub["cancer"] = sub["cancer"].fillna(0.0).clip(0.0, 1.0)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Columns:", list(sub.columns))
