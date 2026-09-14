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

0.8327289211242067

# 6. Current score

0.11659

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.56764) has done: 'I fix the missing EfficientNet dependency by removing the external wheel install and switching to a torchvision EfficientNet-B4 model with an updated classification head so the checkpoint can load (with a safe fallback to non-strict loading if key names differ). I update the Albumentations transforms to the v2 API (notably `RandomResizedCrop(size=(h,w))` and replacing removed `Cutout` with `CoarseDropout`) so augmentation runs without validation errors. I also correct the TTA averaging logic (currently divides by 10 while looping 5 times) and ensure inference runs under `torch.no_grad()` with proper device handling and RGB conversion. Finally, I ensure `submission.csv` is written with the required `image_id,label` columns and correct row order from `sample_submission.csv`.'
- What this solution (achieved 0.11659) has done: 'The immediate failure is the missing checkpoint path, so I make checkpoint discovery robust by searching common Kaggle input locations for a `.pth`/`.pt` file and only asserting after we’ve tried to find a valid one. To move accuracy toward the target, I also fix the inference preprocessing to match EfficientNet’s expected input: use deterministic resize/center-crop (no random augmentations) and use the official EfficientNet-B4 normalization; then apply lightweight deterministic TTA (flips) with softmax-probability averaging (more stable than averaging logits). Finally, I keep the submission ordering exactly as `sample_submission.csv` and always write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import warnings

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)



## === cell 1
model_path = "../input/en-b4-tta-calr-15/model_15.pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"

assert os.path.exists(sample_sub_path), f"Missing: {sample_sub_path}"
assert os.path.isdir(test_images_path), f"Missing dir: {test_images_path}"


def _find_checkpoint(preferred_path: str) -> str:
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path

    search_roots = [
        "../input",
        "/kaggle/input",
        "/kaggle/data/input",
        "/kaggle/data/cassava-leaf-disease-classification",
        "/kaggle/data",
    ]
    exts = (".pth", ".pt", ".bin")

    candidates = []
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(exts):
                    candidates.append(os.path.join(dirpath, fn))

    def score(p):
        pl = p.lower()
        s = 0
        if "b4" in pl or "efficientnet" in pl:
            s += 5
        if "model" in pl or "checkpoint" in pl or "ckpt" in pl:
            s += 2
        if "tta" in pl:
            s += 1
        s -= pl.count(os.sep) * 0.01
        return s

    if not candidates:
        return preferred_path  # will fail with clear assert below

    candidates = sorted(candidates, key=score, reverse=True)
    return candidates[0]


resolved_model_path = _find_checkpoint(model_path)
print("Resolved model_path:", resolved_model_path)
assert os.path.exists(resolved_model_path), (
    f"Missing checkpoint. Tried preferred '{model_path}' and auto-search; "
    f"no .pth/.pt found under expected input directories."
)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_56/1879095061.py in <cell line: 0>()
     55 resolved_model_path = _find_checkpoint(model_path)
     56 print("Resolved model_path:", resolved_model_path)
---> 57 assert os.path.exists(resolved_model_path), (
     58     f"Missing checkpoint. Tried preferred '{model_path}' and auto-search; "
     59     f"no .pth/.pt found under expected input directories."

AssertionError: Missing checkpoint. Tried preferred '../input/en-b4-tta-calr-15/model_15.pth' and auto-search; no .pth/.pt found under expected input directories.

## === cell 2
model = models.efficientnet_b4(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 5)
model.to(DEVICE)



## === cell 3
ckpt = torch.load(resolved_model_path, map_location="cpu")
state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

if isinstance(state_dict, dict):
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    if any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}

try:
    model.load_state_dict(state_dict, strict=True)
    print("Loaded checkpoint with strict=True")
except Exception as e:
    print("Strict load failed; retrying with strict=False. Error:", repr(e))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"Loaded with strict=False. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}"
    )

model.eval()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_56/1141447785.py in <cell line: 0>()
      1 # Bug fix: load checkpoint robustly across different training codebases (state_dict nesting/key prefixes).
----> 2 ckpt = torch.load(resolved_model_path, map_location="cpu")
      3 state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
      4 
      5 if isinstance(state_dict, dict):

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/en-b4-tta-calr-15/model_15.pth'

## === cell 4

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

infer_aug = A.Compose(
    [
        A.SmallestMaxSize(max_size=380, p=1.0),
        A.CenterCrop(height=380, width=380, p=1.0),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
    ],
    p=1.0,
)

to_tensor = transforms.ToTensor()

sample_sub = pd.read_csv(sample_sub_path)

pred_labels = []
use_tta = True

with torch.no_grad():
    for _, row in sample_sub.iterrows():
        image_path = os.path.join(test_images_path, row.image_id)

        img = Image.open(image_path).convert("RGB")
        img = np.array(img)

        imgs = [img]
        if use_tta:
            imgs = [
                img,
                np.ascontiguousarray(img[:, ::-1, :]),
                np.ascontiguousarray(img[::-1, :, :]),
                np.ascontiguousarray(img[::-1, ::-1, :]),
            ]

        prob_sum = None
        for im in imgs:
            aug_im = infer_aug(image=im)["image"]  # HWC float32 normalized
            x = to_tensor(aug_im).to(DEVICE)  # CHW float32
            logits = model(x.unsqueeze(0))  # [1,5]
            probs = torch.softmax(logits, dim=1)  # [1,5]
            prob_sum = probs if prob_sum is None else (prob_sum + probs)

        prob_mean = prob_sum / float(len(imgs))
        pred = int(torch.argmax(prob_mean, dim=1).item())
        pred_labels.append(pred)

sub_df = sample_sub.copy()
sub_df["label"] = pred_labels
sub_df.to_csv("submission.csv", index=False)

print(sub_df.head())
print("Saved submission.csv with", len(sub_df), "rows")
print("Unique labels predicted:", sorted(sub_df["label"].unique().tolist()))
