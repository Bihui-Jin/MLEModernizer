# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.0381862601745652

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I make the notebook run end-to-end in the Kaggle Python 3.11 environment by removing the failing `pip install` cell, adding robust imports, and providing a safe fallback DICOM reader when `dicomsdl` isn’t available. I also fix the core training/inference logic bugs without changing the model architecture: the model currently applies `sigmoid` inside `forward` but trains with `BCEWithLogitsLoss` (incorrect), so I switch the loss to `BCELoss` to match the existing semantics. Finally, I ensure predictions are generated for all test rows, merged back correctly, aggregated by `prediction_id`, aligned to `sample_submission.csv`, and saved as a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys
import warnings

warnings.filterwarnings("ignore")



## === cell 1
import numpy as np
import pandas as pd

try:
    import cv2
except Exception as e:
    raise ImportError(
        "cv2 is required in this solution but is not available in the environment."
    ) from e

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.models as models
from PIL import Image

from sklearn.model_selection import train_test_split

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import dicomsdl  # noqa: F401

    _HAVE_DICOMSDL = True
except Exception:
    _HAVE_DICOMSDL = False

try:
    import pydicom

    _HAVE_PYDICOM = True
except Exception:
    _HAVE_PYDICOM = False

print("dicomsdl available:", _HAVE_DICOMSDL)
print("pydicom available:", _HAVE_PYDICOM)
print("torch:", torch.__version__)



## === cell 2
data_dir = "/kaggle/input/rsna-breast-cancer-detection"
assert os.path.exists(data_dir), f"Expected data_dir not found: {data_dir}"

target_size = [216, 216]
batch_size = 64
num_epochs = 5



## === cell 3
train_df = pd.read_csv(f"{data_dir}/train.csv")
test_df = pd.read_csv(f"{data_dir}/test.csv")
sample_sub = pd.read_csv(f"{data_dir}/sample_submission.csv")

train_df["dcm_path"] = (
    data_dir
    + "/train_images/"
    + train_df["patient_id"].astype(str)
    + "/"
    + train_df["image_id"].astype(str)
    + ".dcm"
)

test_df["dcm_path"] = (
    data_dir
    + "/test_images/"
    + test_df["patient_id"].astype(str)
    + "/"
    + test_df["image_id"].astype(str)
    + ".dcm"
)

print(
    "train_df:",
    train_df.shape,
    "test_df:",
    test_df.shape,
    "sample_sub:",
    sample_sub.shape,
)
train_df.head(2)




## === cell 4
def _read_dicom_pixels(path):
    """
    Bug fix: dicomsdl is not available in this environment; provide a robust fallback.
    Returns: (pixel_array_float32, photometric_interpretation_str_or_None)
    """
    if _HAVE_DICOMSDL:
        dcm = dicomsdl.open(path)
        arr = dcm.pixelData(storedvalue=False)  # float32
        pi = getattr(dcm, "PhotometricInterpretation", None)
        return arr.astype(np.float32, copy=False), pi

    if _HAVE_PYDICOM:
        dcm = pydicom.dcmread(path, force=True)
        arr = dcm.pixel_array.astype(np.float32, copy=False)

        slope = float(getattr(dcm, "RescaleSlope", 1.0))
        intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept

        pi = str(getattr(dcm, "PhotometricInterpretation", "") or "")
        return arr, pi

    raise ImportError("No DICOM reader available (dicomsdl or pydicom).")


def normalize_xray(path, fix_monochrome=True):
    dicom_arr, photometric = _read_dicom_pixels(path)

    dicom_arr = dicom_arr - np.min(dicom_arr)
    mx = np.max(dicom_arr)
    if mx > 0:
        dicom_arr = dicom_arr / mx

    if fix_monochrome and photometric == "MONOCHROME1":
        dicom_arr = np.amax(dicom_arr) - dicom_arr

    dicom_arr = dicom_arr - np.min(dicom_arr)
    mx = np.max(dicom_arr)
    if mx > 0:
        dicom_arr = dicom_arr / mx

    dicom_arr = (dicom_arr * 255.0).astype(np.uint8)
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


def preprocess_and_save(file_path):
    image = normalize_xray(file_path)
    h, w = image.shape[:2]
    image = crop_and_resize(image)
    sub_path = file_path.split("/", 4)[-1].split(".dcm")[0] + ".png"
    infos = sub_path.split("/")
    pid = infos[-2]
    iid = infos[-1].replace(".png", "")
    return pid, iid, h, w, image


def pfbeta_torch(labels, preds, beta=1):
    preds = np.clip(preds, 0, 1)
    y_true_count = labels.sum()
    ctp = preds[labels == 1].sum()
    cfp = preds[labels == 0].sum()
    beta_squared = beta * beta
    if (ctp + cfp) == 0 or y_true_count == 0:
        return 0.0
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if c_precision > 0 and c_recall > 0:
        return (
            (1 + beta_squared)
            * (c_precision * c_recall)
            / (beta_squared * c_precision + c_recall)
        )
    return 0.0




## === cell 5
class MyDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        cancer = torch.tensor(
            int(row["cancer"]), dtype=torch.float32
        )  # float for BCELoss
        full_path = row["dcm_path"]

        pid, iid, h, w, img = preprocess_and_save(full_path)
        img = Image.fromarray(img).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"cancer": cancer, "images": img}




## === cell 6
class PretrainedBinaryClassifier(nn.Module):
    def __init__(self):
        super(PretrainedBinaryClassifier, self).__init__()
        self.model = models.resnet50(pretrained=False)

        weights_path = "/kaggle/input/my-requirement-files/resnet50-0676ba61.pth"
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            self.model.load_state_dict(state_dict)

        for param in self.model.parameters():
            param.requires_grad = False

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        x = self.model(x)
        x = torch.sigmoid(x)
        return x




## === cell 7
train_subset_0 = train_df[train_df.cancer == 0].iloc[:55, :]
train_subset_1 = train_df[train_df.cancer == 1].iloc[:45, :]
train_combined = pd.concat([train_subset_0, train_subset_1], axis=0).reset_index(
    drop=True
)

print("train_combined:", train_combined.shape)
print(train_combined["cancer"].value_counts())

training_set, validation_set = train_test_split(
    train_combined, test_size=0.2, random_state=42, stratify=train_combined["cancer"]
)
print("training_set:", training_set.shape, "validation_set:", validation_set.shape)



## === cell 8
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = MyDataset(training_set, transform=transform)
val_dataset = MyDataset(validation_set, transform=transform)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
)

print("len(train_loader):", len(train_loader), "len(val_loader):", len(val_loader))



## === cell 9
model = PretrainedBinaryClassifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

print("device:", device)



## === cell 10
train_losses, train_acc_metric, train_pf1_metric = [], [], []
val_losses, val_acc_metric, val_pf1_metric = [], [], []

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

        optimizer.zero_grad()
        outputs = model(inputs)  # already sigmoid probabilities in [0,1]
        loss = criterion(outputs, targets)

        predicted = torch.round(outputs)
        correct = (predicted == targets).sum().item()
        accuracy = correct / targets.size(0)

        pf1_metric = pfbeta_torch(
            targets.detach().cpu().numpy(), outputs.detach().cpu().numpy()
        )

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
        f"Epoch {epoch+1}, avg training loss: {avg_train_loss:.3f}, avg training accuracy: {avg_train_acc:.3f}, avg training pf1: {avg_train_pf1:.3f}"
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

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            predicted = torch.round(outputs)
            correct = (predicted == targets).sum().item()
            accuracy = correct / targets.size(0)

            pf1_metric = pfbeta_torch(
                targets.detach().cpu().numpy(), outputs.detach().cpu().numpy()
            )

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
        f"Epoch {epoch+1}, avg validation loss: {avg_val_loss:.3f}, avg validation accuracy: {avg_val_acc:.3f}, avg validation pf1: {avg_val_pf1:.3f}"
    )




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2514558914.py in <cell line: 0>()
      8 
      9     model.train()
---> 10     for i, data in enumerate(train_loader, 0):
     11         inputs, targets = data["images"], data["cancer"]
     12         targets = targets.view(-1, 1)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/tmp/ipykernel_55/2684919804.py in __getitem__(self, index)
     14         full_path = row["dcm_path"]
     15 
---> 16         pid, iid, h, w, img = preprocess_and_save(full_path)
     17         img = Image.fromarray(img).convert("RGB")
     18         if self.transform:

/tmp/ipykernel_55/3191759740.py in preprocess_and_save(file_path)
     58 
     59 def preprocess_and_save(file_path):
---> 60     image = normalize_xray(file_path)
     61     h, w = image.shape[:2]
     62     image = crop_and_resize(image)

/tmp/ipykernel_55/3191759740.py in normalize_xray(path, fix_monochrome)
     26 
     27 def normalize_xray(path, fix_monochrome=True):
---> 28     dicom_arr, photometric = _read_dicom_pixels(path)
     29 
     30     # normalize to [0, 1]

/tmp/ipykernel_55/3191759740.py in _read_dicom_pixels(path)
     12     if _HAVE_PYDICOM:
     13         dcm = pydicom.dcmread(path, force=True)
---> 14         arr = dcm.pixel_array.astype(np.float32, copy=False)
     15 
     16         # Apply rescale if present (common in DICOM)

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in pixel_array(self)
   2191             that iterates through the image frames.
   2192         """
-> 2193         self.convert_pixel_data()
   2194         return cast("numpy.ndarray", self._pixel_array)
   2195 

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in convert_pixel_data(self, handler_name)
   1724             # Use 'pydicom.pixels' backend
   1725             opts["decoding_plugin"] = name
-> 1726             self._pixel_array = pixel_array(self, **opts)
   1727             self._pixel_id = get_image_pixel_ids(self)
   1728         else:

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/utils.py in pixel_array(src, ds_out, specific_tags, index, raw, decoding_plugin, **kwargs)
   1428 
   1429         opts = as_pixel_options(ds, **kwargs)
-> 1430         return decoder.as_array(
   1431             ds,
   1432             index=index,

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py in as_array(self, src, index, validate, raw, decoding_plugin, **kwargs)
    980             cast(
    981                 dict[str, "DecodeFunction"],
--> 982                 self._validate_plugins(decoding_plugin),
    983             ),
    984         )

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/common.py in _validate_plugins(self, plugin)
    255         missing = "\n".join([f"\t{s}" for s in self.missing_dependencies])
    256         if self._decoder:
--> 257             raise RuntimeError(
    258                 f"Unable to decompress '{self.UID.name}' pixel data because all "
    259                 f"plugins are missing dependencies:\n{missing}"

RuntimeError: Unable to decompress 'JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1])' pixel data because all plugins are missing dependencies:
	gdcm - requires gdcm>=3.0.10
	pylibjpeg - requires pylibjpeg>=2.0 and pylibjpeg-libjpeg>=2.1

## === cell 11
class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        full_path = row["dcm_path"]
        pid, iid, h, w, img = preprocess_and_save(full_path)
        img = Image.fromarray(img).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return (
            row["prediction_id"],
            img,
        )  # return prediction_id for correct alignment/aggregation




## === cell 12
test_transform = transforms.Compose([transforms.ToTensor()])
test_dataset = TestDataset(test_df, transform=test_transform)

test_batch_size = 128
test_dataloader = DataLoader(
    test_dataset,
    batch_size=test_batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

pred_predid = []
pred_prob = []

model.eval()
with torch.no_grad():
    for batch in test_dataloader:
        prediction_ids, images = batch
        images = images.to(device, non_blocking=True)
        outputs = model(images)  # probabilities
        probs = outputs.squeeze(1).detach().cpu().numpy()
        pred_predid.extend(list(prediction_ids))
        pred_prob.extend(list(probs))

print(
    "num test rows:",
    len(test_df),
    "num preds:",
    len(pred_prob),
    "num pred_ids:",
    len(pred_predid),
)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/1350749483.py in <cell line: 0>()
     18 model.eval()
     19 with torch.no_grad():
---> 20     for batch in test_dataloader:
     21         prediction_ids, images = batch
     22         images = images.to(device, non_blocking=True)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/tmp/ipykernel_55/3071476035.py in __getitem__(self, index)
     10         row = self.df.iloc[index]
     11         full_path = row["dcm_path"]
---> 12         pid, iid, h, w, img = preprocess_and_save(full_path)
     13         img = Image.fromarray(img).convert("RGB")
     14         if self.transform:

/tmp/ipykernel_55/3191759740.py in preprocess_and_save(file_path)
     58 
     59 def preprocess_and_save(file_path):
---> 60     image = normalize_xray(file_path)
     61     h, w = image.shape[:2]
     62     image = crop_and_resize(image)

/tmp/ipykernel_55/3191759740.py in normalize_xray(path, fix_monochrome)
     26 
     27 def normalize_xray(path, fix_monochrome=True):
---> 28     dicom_arr, photometric = _read_dicom_pixels(path)
     29 
     30     # normalize to [0, 1]

/tmp/ipykernel_55/3191759740.py in _read_dicom_pixels(path)
     12     if _HAVE_PYDICOM:
     13         dcm = pydicom.dcmread(path, force=True)
---> 14         arr = dcm.pixel_array.astype(np.float32, copy=False)
     15 
     16         # Apply rescale if present (common in DICOM)

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in pixel_array(self)
   2191             that iterates through the image frames.
   2192         """
-> 2193         self.convert_pixel_data()
   2194         return cast("numpy.ndarray", self._pixel_array)
   2195 

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in convert_pixel_data(self, handler_name)
   1724             # Use 'pydicom.pixels' backend
   1725             opts["decoding_plugin"] = name
-> 1726             self._pixel_array = pixel_array(self, **opts)
   1727             self._pixel_id = get_image_pixel_ids(self)
   1728         else:

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/utils.py in pixel_array(src, ds_out, specific_tags, index, raw, decoding_plugin, **kwargs)
   1428 
   1429         opts = as_pixel_options(ds, **kwargs)
-> 1430         return decoder.as_array(
   1431             ds,
   1432             index=index,

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py in as_array(self, src, index, validate, raw, decoding_plugin, **kwargs)
    980             cast(
    981                 dict[str, "DecodeFunction"],
--> 982                 self._validate_plugins(decoding_plugin),
    983             ),
    984         )

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/common.py in _validate_plugins(self, plugin)
    255         missing = "\n".join([f"\t{s}" for s in self.missing_dependencies])
    256         if self._decoder:
--> 257             raise RuntimeError(
    258                 f"Unable to decompress '{self.UID.name}' pixel data because all "
    259                 f"plugins are missing dependencies:\n{missing}"

RuntimeError: Unable to decompress 'JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1])' pixel data because all plugins are missing dependencies:
	gdcm - requires gdcm>=3.0.10
	pylibjpeg - requires pylibjpeg>=2.0 and pylibjpeg-libjpeg>=2.1

## === cell 13
pred_df = pd.DataFrame({"prediction_id": pred_predid, "cancer": pred_prob})
sub = pred_df.groupby("prediction_id", as_index=False)["cancer"].mean()

sub = sample_sub[["prediction_id"]].merge(sub, on="prediction_id", how="left")
sub["cancer"] = sub["cancer"].fillna(0.0).clip(0.0, 1.0)

print(sub.head())
print("submission rows:", len(sub), "expected:", len(sample_sub))



## === cell 14
out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "size:", os.path.getsize(out_path))
