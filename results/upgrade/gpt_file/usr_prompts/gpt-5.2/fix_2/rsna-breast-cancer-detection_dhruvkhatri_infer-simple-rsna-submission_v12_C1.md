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

from torch.utils.data import Dataset, DataLoader
from torchvision.models.efficientnet import efficientnet_b0
from torchvision.transforms import transforms

from tqdm import tqdm
import PIL
from PIL import Image

import cv2


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)




## === cell 1
class RSNABreast(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        row = self.dataframe.iloc[idx]
        prediction_id = row["prediction_id"]
        if self.transform:
            img = self.transform(img)
        return img, prediction_id

    def load_image(self, image_index):
        img = PIL.Image.open(self.dataframe.loc[image_index, "Filepath"])
        img = img.convert("RGB")
        return img




## === cell 2
import pydicom
import pydicom.config

pydicom.config.image_handlers = [
    pydicom.pixel_data_handlers.gdcm_handler,
    pydicom.pixel_data_handlers.pylibjpeg_handler,
    pydicom.pixel_data_handlers.numpy_handler,
]

try:
    import pylibjpeg  # noqa: F401
except Exception:
    pass
try:
    import gdcm  # noqa: F401
except Exception:
    pass

from joblib import Parallel, delayed


def dicom_to_uint8(dicom: pydicom.dataset.FileDataset) -> np.ndarray:
    """Convert DICOM pixel array to uint8 [0,255] with photometric fix and min/max scaling."""
    img = dicom.pixel_array.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx > mn:
        img = (img - mn) / (mx - mn)
    else:
        img = np.zeros_like(img, dtype=np.float32)

    if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = 1.0 - img

    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img


def process(f, size=256, save_folder="", extension="png"):
    patient = f.split("/")[-2]
    image = os.path.splitext(os.path.basename(f))[0]

    dicom = pydicom.dcmread(f, force=True)

    img = dicom_to_uint8(dicom)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    out_path = os.path.join(save_folder, f"{patient}_{image}.{extension}")
    cv2.imwrite(out_path, img)


test_images = glob.glob(
    "/kaggle/input/rsna-breast-cancer-detection/test_images/*/*.dcm"
)
os.makedirs("/kaggle/working/test", exist_ok=True)


def safe_process(uid):
    try:
        process(uid, size=256, save_folder="/kaggle/working/test", extension="png")
        return True
    except Exception:
        return False


_ = Parallel(n_jobs=4, backend="loky")(
    delayed(safe_process)(uid)
    for uid in tqdm(test_images, desc="Converting DICOM->PNG")
)



## === cell 3
weightPath = "/kaggle/input/rsnabreastscancer/rsnaB.pt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(1280, 1, bias=True)

if os.path.exists(weightPath):
    state = torch.load(weightPath, map_location="cpu")
    model.load_state_dict(state, strict=True)

model = model.to(device)
model.eval()



## === cell 4
root_path = "/kaggle/working/test"


def getpath(root_path, row_df):
    return os.path.join(root_path, f"{row_df['patient_id']}_{row_df['image_id']}.png")




## === cell 5
samp_sub = pd.read_csv(
    "/kaggle/input/rsna-breast-cancer-detection/sample_submission.csv"
)
samp_sub.head()



## === cell 6
test_csv = pd.read_csv("/kaggle/input/rsna-breast-cancer-detection/test.csv")
test_csv["Filepath"] = test_csv.apply(lambda row: getpath(root_path, row), axis=1)

test_csv["exists"] = test_csv["Filepath"].apply(os.path.exists)
missing = (~test_csv["exists"]).sum()
if missing > 0:
    test_csv = (
        test_csv.loc[test_csv["exists"]].drop(columns=["exists"]).reset_index(drop=True)
    )
else:
    test_csv = test_csv.drop(columns=["exists"])

test_csv.head()



## === cell 7
valid_transform = transforms.Compose(
    [
        transforms.Resize([128, 128]),
        transforms.ToTensor(),
        transforms.Normalize(([0.485, 0.456, 0.406]), ([0.229, 0.224, 0.225])),
    ]
)

ds_loader = RSNABreast(test_csv, valid_transform)
dl_l = DataLoader(
    ds_loader,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

list_prediction_id = []
list_cancer_score = []

sigmoid = nn.Sigmoid()

for xb, pred_ids in tqdm(dl_l, desc="Infer"):
    xb = xb.to(device, non_blocking=True)
    with torch.no_grad():
        outM = model(xb)
        sigV = sigmoid(outM).detach().cpu().view(-1)

    list_prediction_id.extend(list(pred_ids))
    list_cancer_score.extend(sigV.numpy().tolist())



## === cell 8
pred_df = pd.DataFrame(
    {"prediction_id": list_prediction_id, "cancer": list_cancer_score}
)
pred_df = pred_df.groupby("prediction_id", as_index=False)["cancer"].max()

sub = samp_sub[["prediction_id"]].merge(pred_df, on="prediction_id", how="left")
sub["cancer"] = sub["cancer"].fillna(0.0).astype(np.float32)

sub.to_csv("submission.csv", index=False)
sub.head()



## === cell 9
print("submission.csv written:", os.path.abspath("submission.csv"))
print("rows:", len(sub), "cols:", list(sub.columns))
print(
    "cancer stats:",
    float(sub["cancer"].min()),
    float(sub["cancer"].mean()),
    float(sub["cancer"].max()),
)
