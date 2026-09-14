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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scipy==1.15.3
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8931701420368692

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import glob
import re

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import softmax

import torch
import torch.nn as nn
from torchvision import models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)



## === cell 1
resnet_model_path = "../input/rn-wc-tta-calr-clahe-cutmix/model(24).pth"
effnet_model_path = "../input/en-b4-tta-calr-clahe-v2-8/model(15).pth"
effnet_model_folds_path = [
    "../input/en-b4-tta-calr-clahe-v3-5-folds/EFFICIENT_NET_B4_0_11.pth",
    "../input/en-b4-tta-calr-clahe-v3-5-folds/EFFICIENT_NET_B4_1_10.pth",
    "../input/en-b4-tta-calr-clahe-v3-5-folds/EFFICIENT_NET_B4_2_13.pth",
    "../input/en-b4-tta-calr-clahe-v3-5-folds/EFFICIENT_NET_B4_3_12.pth",
    "../input/en-b4-tta-calr-clahe-v3-5-folds/EFFICIENT_NET_B4_4_11.pth",
]

sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"




## === cell 2
def _find_pth_by_regex(pattern: str, root: str = "/kaggle/input") -> str | None:
    cand = []
    for p in glob.glob(os.path.join(root, "**", "*.pth"), recursive=True):
        if re.search(pattern, os.path.basename(p), flags=re.IGNORECASE):
            cand.append(p)
    return sorted(cand)[0] if cand else None


def _ensure_path(path: str, fallback_pattern: str) -> str:
    if path and os.path.exists(path):
        return path
    found = _find_pth_by_regex(fallback_pattern)
    if found is None:
        raise FileNotFoundError(
            f"Model weights not found. Missing path: {path}. "
            f"Also couldn't find any .pth under /kaggle/input matching regex: {fallback_pattern!r}."
        )
    return found


resnet_model_path = _ensure_path(
    resnet_model_path, r"resn|resnext|rn|next50|32x4d|model.*\.pth"
)
resolved_folds = []
for p in effnet_model_folds_path:
    if os.path.exists(p):
        resolved_folds.append(p)
if len(resolved_folds) == 0:
    all_b4 = sorted(glob.glob("/kaggle/input/**/**", recursive=True))
    b4_pths = sorted(glob.glob("/kaggle/input/**/*.pth", recursive=True))
    b4_pths = [
        p
        for p in b4_pths
        if re.search(r"b4|efficient.*b4|eff.*b4", os.path.basename(p), re.IGNORECASE)
    ]
    if len(b4_pths) == 0:
        b4_pths = sorted(glob.glob("/kaggle/input/**/*.pth", recursive=True))
    resolved_folds = b4_pths[:5]
if len(resolved_folds) == 0:
    raise FileNotFoundError(
        "Could not locate any EfficientNet fold .pth files under /kaggle/input."
    )

effnet_model_folds_path = resolved_folds

print("Resolved resnet weights:", resnet_model_path)
print("Resolved effnet fold weights:", effnet_model_folds_path)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2618810227.py in <cell line: 0>()
     23 
     24 # Try to resolve paths; patterns are conservative to avoid wrong matches.
---> 25 resnet_model_path = _ensure_path(
     26     resnet_model_path, r"resn|resnext|rn|next50|32x4d|model.*\.pth"
     27 )

/tmp/ipykernel_55/2618810227.py in _ensure_path(path, fallback_pattern)
     15     if found is None:
     16         # Provide a clear error that explains what is missing.
---> 17         raise FileNotFoundError(
     18             f"Model weights not found. Missing path: {path}. "
     19             f"Also couldn't find any .pth under /kaggle/input matching regex: {fallback_pattern!r}."

FileNotFoundError: Model weights not found. Missing path: ../input/rn-wc-tta-calr-clahe-cutmix/model(24).pth. Also couldn't find any .pth under /kaggle/input matching regex: 'resn|resnext|rn|next50|32x4d|model.*\\.pth'.

## === cell 3
def _clean_state_dict(sd):
    if (
        isinstance(sd, dict)
        and "state_dict" in sd
        and isinstance(sd["state_dict"], dict)
    ):
        sd = sd["state_dict"]
    new_sd = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        new_sd[nk] = v
    return new_sd


resnet_model = models.resnext50_32x4d(weights=None)
resnet_model.fc = nn.Linear(2048, 5)
resnet_model.to(DEVICE)
resnet_sd = torch.load(resnet_model_path, map_location="cpu")
resnet_sd = _clean_state_dict(resnet_sd)
resnet_model.load_state_dict(resnet_sd, strict=False)
resnet_model.eval()

effnet_model = models.efficientnet_b4(weights=None)
if isinstance(effnet_model.classifier, nn.Sequential):
    in_features = effnet_model.classifier[-1].in_features
    effnet_model.classifier[-1] = nn.Linear(in_features, 5)
else:
    in_features = effnet_model.classifier.in_features
    effnet_model.classifier = nn.Linear(in_features, 5)

effnet_model.to(DEVICE)
effnet_model.eval()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/323221334.py in <cell line: 0>()
     21 resnet_model.fc = nn.Linear(2048, 5)
     22 resnet_model.to(DEVICE)
---> 23 resnet_sd = torch.load(resnet_model_path, map_location="cpu")
     24 resnet_sd = _clean_state_dict(resnet_sd)
     25 resnet_model.load_state_dict(resnet_sd, strict=False)

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1423         pickle_load_args["encoding"] = "utf-8"
   1424 
-> 1425     with _open_file_like(f, "rb") as opened_file:
   1426         if _is_zipfile(opened_file):
   1427             # The zipfile reader is going to advance the current file position.

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _open_file_like(name_or_buffer, mode)
    749 def _open_file_like(name_or_buffer, mode):
    750     if _is_path(name_or_buffer):
--> 751         return _open_file(name_or_buffer, mode)
    752     else:
    753         if "w" in mode:

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in __init__(self, name, mode)
    730 class _open_file(_opener):
    731     def __init__(self, name, mode):
--> 732         super().__init__(open(name, mode))
    733 
    734     def __exit__(self, *args):

FileNotFoundError: [Errno 2] No such file or directory: '../input/rn-wc-tta-calr-clahe-cutmix/model(24).pth'

## === cell 4
to_tensor = transforms.ToTensor()


def load_image_np(path: str) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        return np.array(im)




## === cell 5
sub_aug = A.Compose(
    [
        A.RandomResizedCrop(size=(512, 512), scale=(0.5, 1.0)),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.8),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)




## === cell 6
def predict_model_tta(
    model: torch.nn.Module, image_paths: list[str], tta_count: int
) -> np.ndarray:
    preds = np.zeros((len(image_paths), 5), dtype=np.float32)
    for i, path in enumerate(image_paths):
        image_pred = np.zeros((1, 5), dtype=np.float32)
        img0 = load_image_np(path)
        for _ in range(tta_count):
            img = sub_aug(image=img0)["image"]
            x = to_tensor(np.array(img)).to(DEVICE)
            out = model(x.unsqueeze(0))
            image_pred += out.detach().float().cpu().numpy()
        image_pred /= float(tta_count)
        preds[i] = image_pred[0]
    return preds




## === cell 7
sample_sub = pd.read_csv(sample_sub_path)
image_paths = [
    os.path.join(test_images_path, iid) for iid in sample_sub["image_id"].tolist()
]

tta_count = 5
resnet_predictions = predict_model_tta(resnet_model, image_paths, tta_count=tta_count)

print("ResNeXt preds shape:", resnet_predictions.shape)



## === cell 8
tta_count = 1
effnet_fold_preds = []

for model_path in effnet_model_folds_path:
    sd = torch.load(model_path, map_location="cpu")
    sd = _clean_state_dict(sd)
    effnet_model.load_state_dict(sd, strict=False)
    effnet_model.to(DEVICE)
    effnet_model.eval()

    fold_preds = predict_model_tta(effnet_model, image_paths, tta_count=tta_count)
    effnet_fold_preds.append(fold_preds)

effnet_predictions = np.mean(np.stack(effnet_fold_preds, axis=0), axis=0)
print("EffNet preds shape:", effnet_predictions.shape)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1139588490.py in <cell line: 0>()
      4 
      5 for model_path in effnet_model_folds_path:
----> 6     sd = torch.load(model_path, map_location="cpu")
      7     sd = _clean_state_dict(sd)
      8     effnet_model.load_state_dict(sd, strict=False)

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1423         pickle_load_args["encoding"] = "utf-8"
   1424 
-> 1425     with _open_file_like(f, "rb") as opened_file:
   1426         if _is_zipfile(opened_file):
   1427             # The zipfile reader is going to advance the current file position.

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _open_file_like(name_or_buffer, mode)
    749 def _open_file_like(name_or_buffer, mode):
    750     if _is_path(name_or_buffer):
--> 751         return _open_file(name_or_buffer, mode)
    752     else:
    753         if "w" in mode:

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in __init__(self, name, mode)
    730 class _open_file(_opener):
    731     def __init__(self, name, mode):
--> 732         super().__init__(open(name, mode))
    733 
    734     def __exit__(self, *args):

FileNotFoundError: [Errno 2] No such file or directory: '../input/en-b4-tta-calr-clahe-v3-5-folds/EFFICIENT_NET_B4_0_11.pth'

## === cell 9
combine_logits = (effnet_predictions * 0.5) + (resnet_predictions * 0.5)
combine_preds = softmax(combine_logits, axis=1).argmax(axis=1).astype(int)

sub_df = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values, "label": combine_preds}
)
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Saved submission.csv with shape:", sub_df.shape)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1061068438.py in <cell line: 0>()
      1 # Bugfix: ensure numeric arrays and correct blending; then argmax labels and write valid submission.csv
----> 2 combine_logits = (effnet_predictions * 0.5) + (resnet_predictions * 0.5)
      3 combine_preds = softmax(combine_logits, axis=1).argmax(axis=1).astype(int)
      4 
      5 sub_df = pd.DataFrame(

NameError: name 'effnet_predictions' is not defined
