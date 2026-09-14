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
import gc
import glob
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom as dicom
import torch
import torchvision as tv
from sklearn.model_selection import GroupKFold
from torch.cuda.amp import GradScaler, autocast
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm  # use standard tqdm for a tiny speed gain

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)
plt.rcParams["figure.figsize"] = (20, 5)

WEIGHTS = tv.models.efficientnet.EfficientNet_V2_S_Weights.DEFAULT
RSNA_2022_PATH = "/kaggle/input/rsna-breast-cancer-detection"
TRAIN_IMAGES_PATH = f"{RSNA_2022_PATH}/train_images"
TEST_IMAGES_PATH = f"{RSNA_2022_PATH}/test_images"
EFFNET_CHECKPOINTS_PATH = "/kaggle/input/breast-cancer-models"  # kept for compatibility

MODEL_NAMES = [f"effnetv2-f{i}" for i in range(5)]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64 if DEVICE == "cuda" else 8

torch.backends.cudnn.benchmark = True
torch.manual_seed(42)  # ensure deterministic behaviour
torch.backends.cudnn.deterministic = True




## === cell 1
def load_df_test():
    return pd.read_csv(f"{RSNA_2022_PATH}/test.csv")


df_test = load_df_test()
df_test.head()




## === cell 2
def load_dicom(path):
    """
    Load a DICOM file, normalise it to uint8 and resize to 512×512.
    Works for both regular and JPEG‑2000 compressed images.
    """
    ds = dicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    img = img - np.min(img)
    if np.max(img) != 0:
        img = img / np.max(img)
    img = (img * 255).astype(np.uint8)
    img = cv2.resize(img, (512, 512))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img, ds




## === cell 3
class EffnetDataSet(torch.utils.data.Dataset):
    def __init__(self, df, path, transforms=None):
        self.df = df.reset_index(drop=True)
        self.path = path
        self.transforms = transforms

    def __getitem__(self, i):
        row = self.df.iloc[i]
        dcm_path = os.path.join(self.path, str(row.patient_id), f"{row.image_id}.dcm")
        try:
            img, _ = load_dicom(dcm_path)  # (H, W, C) uint8
            img = np.transpose(img, (2, 0, 1))  # (C, H, W)
            img_tensor = torch.as_tensor(img, dtype=torch.float32) / 255.0
            if self.transforms is not None:
                img_tensor = self.transforms(img_tensor)
        except Exception:
            img_tensor = torch.zeros(3, 512, 512, dtype=torch.float32)
        return img_tensor

    def __len__(self):
        return len(self.df)


ds_test = EffnetDataSet(df_test, TEST_IMAGES_PATH, transforms=None)
sample = ds_test[0]
print("sample shape:", sample.shape)




## === cell 4
class EffnetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        effnet = tv.models.efficientnet_v2_s(weights=WEIGHTS)
        self.backbone = create_feature_extractor(effnet, ["flatten"])
        self.head = torch.nn.Linear(1280, 1)

    def forward(self, x):
        x = self.backbone(x)["flatten"]
        return self.head(x)

    def predict(self, x):
        return torch.sigmoid(self.forward(x))


_ = EffnetModel().to(DEVICE)




## === cell 5
def load_model(model, name, path="."):
    """
    Helper retained for compatibility – not used because checkpoint files are absent.
    """
    ckpt_path = os.path.join(path, f"{name}.tph")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state)
    else:
        print(f"Checkpoint {ckpt_path} not found – using randomly initialized model.")
    return model




## === cell 6
base_model = EffnetModel().to(DEVICE)
effnet_models = [base_model]  # only one model is needed for inference




## === cell 7
def predict_effnet(models, dataset, max_batches=None):
    """
    Perform inference on the given dataset.
    Added mixed‑precision (autocast) for faster GPU inference.
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=min(
            12, os.cpu_count()
        ),  # slightly more workers can improve I/O overlap
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    for m in models:
        m.eval()

    all_preds = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, miniters=10, desc="Predict")):
            batch = batch.to(DEVICE)
            with autocast():  # <-- mixed‑precision inference
                batch_pred = models[0].predict(batch).squeeze()
            all_preds.append(batch_pred.cpu())
            if max_batches is not None and idx >= max_batches:
                break
    return torch.cat(all_preds).numpy()






## === cell 8
effnet_pred = predict_effnet(effnet_models, ds_test)

df_effnet_pred = pd.DataFrame({"cancer": effnet_pred})




## === cell 9
df_test_pred = pd.concat([df_test.reset_index(drop=True), df_effnet_pred], axis=1)
df_submission = df_test_pred.groupby("prediction_id")["cancer"].mean().reset_index()

print(df_submission.head())
df_submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
