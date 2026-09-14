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
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import PIL
from PIL import Image

from torchvision.models.efficientnet import efficientnet_b0
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import cv2

os.environ.setdefault("PYTHONHASHSEED", "0")
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
import pydicom




## === cell 2
def read_dicom_resized_rgb_u8(fp, size=512):
    try:
        dicom = pydicom.dcmread(fp, force=True)
        photo = getattr(dicom, "PhotometricInterpretation", "")

        img = dicom.pixel_array.astype(np.float32, copy=False)

        img_min = float(img.min())
        img_max = float(img.max())
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img, dtype=np.float32)

        if photo == "MONOCHROME1":
            img = 1.0 - img

        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        img_u8 = (img * 255.0).clip(0, 255).astype(np.uint8, copy=False)

        return np.repeat(img_u8[:, :, None], 3, axis=2)
    except Exception:
        return np.zeros((size, size, 3), dtype=np.uint8)




## === cell 3
class RSNABreast(Dataset):
    def __init__(self, dataframe, dicom_size=512, cache_dir=None):
        df = dataframe.reset_index(drop=True)
        self.filepaths = df["Filepath"].to_numpy()
        self.pred_ids = df["prediction_id"].to_numpy()
        self.dicom_size = int(dicom_size)

        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

        self._mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[
            :, None, None
        ]
        self._std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[
            :, None, None
        ]

    def __len__(self):
        return len(self.filepaths)

    def _cache_path(self, fp):
        safe = fp.replace(":", "_").replace("\\", "_").replace("/", "_")
        if len(safe) > 180:
            safe = safe[:180] + f"__{abs(hash(fp))}"
        return os.path.join(self.cache_dir, f"{safe}_s{self.dicom_size}.pt")

    def __getitem__(self, idx):
        fp = self.filepaths[idx]

        if self.cache_dir is not None:
            cpath = self._cache_path(fp)
            try:
                x = torch.load(cpath, map_location="cpu", weights_only=False)
                return x, idx
            except Exception:
                pass

        img = read_dicom_resized_rgb_u8(fp, size=self.dicom_size)  # HWC u8 RGB
        x = torch.from_numpy(img).permute(2, 0, 1).to(dtype=torch.float32).div_(255.0)
        x.sub_(self._mean).div_(self._std)

        if self.cache_dir is not None:
            try:
                torch.save(x, cpath, _use_new_zipfile_serialization=False)
            except Exception:
                pass

        return x, idx




## === cell 4
DATA_ROOT = "/kaggle/input/rsna-breast-cancer-detection"



## === cell 5
samp_sub = pd.read_csv(
    os.path.join(DATA_ROOT, "sample_submission.csv"),
    usecols=["prediction_id"],
    dtype={"prediction_id": "string"},
)
test_csv = pd.read_csv(
    os.path.join(DATA_ROOT, "test.csv"),
    usecols=["patient_id", "image_id", "prediction_id"],
    dtype={"patient_id": np.int64, "image_id": np.int64, "prediction_id": "string"},
)

test_csv["Filepath"] = (
    DATA_ROOT
    + "/test_images/"
    + test_csv["patient_id"].astype(str)
    + "/"
    + test_csv["image_id"].astype(str)
    + ".dcm"
)

test_csv.head()



## === cell 6
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(1280, 1, bias=True)

model = model.to(device)
model.eval()



## === cell 7
cpu_cnt = os.cpu_count() or 2
num_workers = min(8, max(2, cpu_cnt // 2))

cache_dir = "/kaggle/working/dicom_cache_b0_512"
ds_loader = RSNABreast(test_csv, dicom_size=512, cache_dir=cache_dir)

batch_size = 64 if torch.cuda.is_available() else 32

dl_l = DataLoader(
    ds_loader,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)

pred_ids_arr = test_csv["prediction_id"].to_numpy()

n = len(ds_loader)
pred_score = np.empty(n, dtype=np.float32)

with torch.inference_mode():
    for xb, idxs in tqdm(dl_l, desc="Inference"):
        xb = xb.to(device, non_blocking=True)
        outM = model(xb)
        sigV = (
            torch.sigmoid(outM)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(np.float32, copy=False)
        )
        idxs_np = idxs.numpy()
        pred_score[idxs_np] = sigV



## === cell 8
pred_df = pd.DataFrame({"prediction_id": pred_ids_arr, "cancer": pred_score})

pred_df = pred_df.groupby("prediction_id", as_index=False)["cancer"].max()

sub = samp_sub[["prediction_id"]].merge(pred_df, on="prediction_id", how="left")
sub["cancer"] = sub["cancer"].fillna(0.0).astype(np.float32)

sub.to_csv("submission.csv", index=False)
sub.head()



## === cell 9
print("submission.csv written:", os.path.abspath("submission.csv"))
print("Rows:", len(sub), "Cols:", list(sub.columns))
print(sub.describe(include="all"))
