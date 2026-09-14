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

0.8689936536718041

# 6. Current score

0.11024

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11061) has done: 'I remove the failing `pip install` dependency and instead build the exact EfficientNet-B4 architecture using `torchvision` so the provided `.pth` can load without `efficientnet_pytorch`. I fix the Albumentations v2 API break by using the correct `size=(H,W)` signature and replacing removed transforms (`Cutout`) with their current equivalents so TTA runs. I also fix inference-time bugs that would silently hurt accuracy (wrong averaging divisor, missing `torch.no_grad()`, and missing softmax/logit handling), while keeping the same core approach: EfficientNet-B4 + 5x TTA over the sample submission list and argmax for labels. Finally, the script always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.11024) has done: 'I fix the immediate runtime failure by making the checkpoint path robust: the current `model_path` points to a dataset that isn’t present, so I search common Kaggle input locations for `model_50.pth` (and fail fast with a clear message if it truly doesn’t exist). I also make the model weight loading tolerant to minor key mismatches (`strict=False`) while keeping the same EfficientNet-B4 architecture, which prevents silent “all-random-weights” behavior and should raise accuracy substantially toward your target. Finally, I fix a subtle inference bug: Albumentations `Normalize` already outputs float images, so applying `torchvision.transforms.ToTensor()` rescales by 1/255 again; I replace it with a correct HWC->CHW tensor conversion to avoid destroying the input scale and hurting score. The rest of the core logic (EfficientNet-B4 + 5x TTA over `sample_submission.csv` and argmax) stays the same.'

# 9. Code solution

## === cell 0
import os
import random
import glob

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models

import albumentations as A



## === cell 1
model_path = "../input/en-b4-tta-calr-50/model_50.pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def resolve_checkpoint_path(p: str) -> str:
    if os.path.exists(p):
        return p

    fname = os.path.basename(p)
    search_roots = [
        "../input",
        "/kaggle/input",
        "/kaggle/data/input",
    ]
    candidates = []
    for root in search_roots:
        candidates.extend(glob.glob(os.path.join(root, "**", fname), recursive=True))

    preferred = [c for c in candidates if "en-b4-tta-calr-50" in c.replace("\\", "/")]
    if preferred:
        return preferred[0]
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"Checkpoint not found. Tried '{p}' and searched for '{fname}' under: {search_roots}"
    )


model_path = resolve_checkpoint_path(model_path)
print("Using model checkpoint:", model_path)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3036485663.py in <cell line: 0>()
     43 
     44 
---> 45 model_path = resolve_checkpoint_path(model_path)
     46 print("Using model checkpoint:", model_path)
     47 

/tmp/ipykernel_55/3036485663.py in resolve_checkpoint_path(p)
     38         return candidates[0]
     39 
---> 40     raise FileNotFoundError(
     41         f"Checkpoint not found. Tried '{p}' and searched for '{fname}' under: {search_roots}"
     42     )

FileNotFoundError: Checkpoint not found. Tried '../input/en-b4-tta-calr-50/model_50.pth' and searched for 'model_50.pth' under: ['../input', '/kaggle/input', '/kaggle/data/input']

## === cell 2
model = models.efficientnet_b4(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 5)
model = model.to(device)

ckpt = torch.load(model_path, map_location=device, weights_only=False)

state_dict = ckpt
if isinstance(ckpt, dict):
    for key in ["state_dict", "model_state_dict", "model", "net"]:
        if key in ckpt and isinstance(ckpt[key], dict):
            state_dict = ckpt[key]
            break

if isinstance(state_dict, dict):
    new_sd = {}
    for k, v in state_dict.items():
        nk = k[7:] if isinstance(k, str) and k.startswith("module.") else k
        new_sd[nk] = v
    state_dict = new_sd

missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(
    f"Loaded weights with strict=False. Missing keys: {len(missing)} Unexpected keys: {len(unexpected)}"
)
if len(missing) > 0:
    print("First missing keys:", missing[:10])
if len(unexpected) > 0:
    print("First unexpected keys:", unexpected[:10])

model.eval()



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1198572379.py in <cell line: 0>()
      7 # Robust state_dict loading (handles checkpoints that wrap state dict under a key)
      8 # Bugfix: use weights_only=False to support older checkpoints; also allow non-strict load to avoid key prefix mismatches.
----> 9 ckpt = torch.load(model_path, map_location=device, weights_only=False)
     10 
     11 state_dict = ckpt

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/en-b4-tta-calr-50/model_50.pth'

## === cell 3
sub_aug = A.Compose(
    [
        A.RandomResizedCrop(size=(256, 256), scale=(0.5, 1.0), p=1.0),
        A.Transpose(p=0.8),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.8),
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=0,
            p=0.5,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1),
            contrast_limit=(-0.1, 0.1),
            p=0.5,
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        A.CoarseDropout(
            num_holes_range=(1, 20),
            hole_height_range=(1, 10),
            hole_width_range=(1, 10),
            p=0.5,
        ),
    ],
    p=1.0,
)


def alb_to_tensor(img_hwc: np.ndarray) -> torch.Tensor:
    if img_hwc.dtype != np.float32:
        img_hwc = img_hwc.astype(np.float32)
    return torch.from_numpy(img_hwc).permute(2, 0, 1).contiguous()




## === cell 4
sample_sub = pd.read_csv(sample_sub_path)

predictions = []
tta_n = 5  # original code looped 5 times

with torch.no_grad():
    for _, sample_row in sample_sub.iterrows():
        img_path = os.path.join(test_images_path, sample_row.image_id)
        image_np = np.array(Image.open(img_path).convert("RGB"))

        image_logits = 0.0
        for _ in range(tta_n):
            aug = sub_aug(image=image_np)["image"]  # HWC float32 normalized
            img_t = alb_to_tensor(aug).to(device)  # CHW float32 (already normalized)
            outputs = model(img_t.unsqueeze(0))  # logits
            image_logits = image_logits + outputs

        image_logits = image_logits / float(tta_n)
        pred_label = int(torch.argmax(image_logits, dim=1).item())
        predictions.append([sample_row.image_id, pred_label])

sub_df = pd.DataFrame(predictions, columns=["image_id", "label"])
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
print("Label distribution:\n", sub_df["label"].value_counts(dropna=False).sort_index())
