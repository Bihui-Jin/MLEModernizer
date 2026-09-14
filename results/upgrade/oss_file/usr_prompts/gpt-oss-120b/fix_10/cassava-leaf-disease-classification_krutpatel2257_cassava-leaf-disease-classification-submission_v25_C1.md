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

0.7601994560290117

# 6. Current score

0.07549

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.08146) has done: 'The fix replaces the invalid `RandomResizedCrop` (which caused a validation error) with a simple `Resize` operation, ensuring the augmentation pipeline is valid and `sub_aug` is defined. No other logic is altered, so the model loading, inference, and CSV output remain unchanged, allowing the script to run end‑to‑end and generate a proper `submission.csv`.'
- What this solution (achieved 0.08146) has done: 'The fix removes the double‑normalisation that was collapsing the model’s inputs. In the augmentation pipeline we drop the Albumentations Normalize step and instead apply the standard torchvision normalisation after converting the image to a tensor. This keeps the core model and inference logic unchanged while providing the correct scaling, which should raise the accuracy from the very low 0.08 toward the target 0.76.'
- What this solution (achieved 0.34193) has done: 'I load the checkpoint with `strict=False` so any mismatched classifier weights are still applied, and replace the random test‑time augmentations with a deterministic resize (no flips or color changes). I also set TTA to a single deterministic pass, which improves accuracy when the model was trained on original images. These minimal changes keep the core model and inference flow unchanged while moving the validation score much closer to the target.'
- What this solution (achieved 0.21936) has done: 'The update adds a small deterministic test‑time augmentation ensemble: after the basic resize, the image is also evaluated in its horizontal‑flipped, vertical‑flipped, and both‑flipped forms. The predictions from these four versions are averaged (tta_count = 4), which commonly raises image‑classification accuracy without altering the core model or training logic.'
- What this solution (achieved 0.06278) has done: 'I adjust the inference pipeline to better match the EfficientNet‑B4 expected input size and avoid the potentially harmful test‑time flips that previously reduced accuracy. The resize is changed from 512 × 512 to the native 380 × 380 resolution, and the test‑time augmentation count is set to 1 with the flip logic removed. These minimal changes keep the model loading and submission logic unchanged while moving the validation score much closer to the target.'
- What this solution (achieved 0.1704) has done: 'I add a small deterministic test‑time augmentation ensemble (original, horizontal flip, vertical flip, both flips) and set `tta_count = 4`. This keeps the model and preprocessing unchanged while providing the model with four slightly different views of each image and averaging their logits, which is known to raise classification accuracy and move the current score much closer to the target.'
- What this solution (achieved 0.07549) has done: 'I add a few extra deterministic test‑time augmentations (90°, 180° and 270° rotations) to the existing TTA pipeline so the model sees more varied but still valid views of each image. This keeps the model architecture and inference flow unchanged, only expands the augmentation list, and the aggregated predictions are still averaged over the larger set, which should raise the validation accuracy toward the target.'

# 9. Code solution

## === cell 0
try:
    import subprocess, sys, os

    wheel_path = (
        "../input/efficientnet-pytorch-070/efficientnet_pytorch-0.7.0-py3-none-any.whl"
    )
    if os.path.exists(wheel_path):
        subprocess.check_call([sys.executable, "-m", "pip", "install", wheel_path])
except Exception:
    pass



## === cell 1
import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models

from torchvision.models import efficientnet_b4

import albumentations



## === cell 2
model_path = "../input/en-b4-tta-calr-clahe/model(14).pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = efficientnet_b4(pretrained=True)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 5)

model.to(device)

if os.path.exists(model_path):
    try:
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state, strict=False)
    except Exception as e:
        print(f"Warning: failed to load weights from {model_path}: {e}")
else:
    print(
        f"Info: weight file not found at {model_path}; using ImageNet pretrained model."
    )

model.eval()



## === cell 3
base_aug = albumentations.Compose([albumentations.Resize(height=380, width=380)], p=1.0)

hflip_aug = albumentations.Compose(
    [
        albumentations.Resize(height=380, width=380),
        albumentations.HorizontalFlip(p=1.0),
    ],
    p=1.0,
)

vflip_aug = albumentations.Compose(
    [
        albumentations.Resize(height=380, width=380),
        albumentations.VerticalFlip(p=1.0),
    ],
    p=1.0,
)

hvflip_aug = albumentations.Compose(
    [
        albumentations.Resize(height=380, width=380),
        albumentations.HorizontalFlip(p=1.0),
        albumentations.VerticalFlip(p=1.0),
    ],
    p=1.0,
)

rot90_aug = albumentations.Compose(
    [
        albumentations.Resize(height=380, width=380),
        albumentations.Rotate(limit=[90, 90], p=1.0),
    ],
    p=1.0,
)

rot180_aug = albumentations.Compose(
    [
        albumentations.Resize(height=380, width=380),
        albumentations.Rotate(limit=[180, 180], p=1.0),
    ],
    p=1.0,
)

rot270_aug = albumentations.Compose(
    [
        albumentations.Resize(height=380, width=380),
        albumentations.Rotate(limit=[270, 270], p=1.0),
    ],
    p=1.0,
)

tta_augs = [
    base_aug,
    hflip_aug,
    vflip_aug,
    hvflip_aug,
    rot90_aug,
    rot180_aug,
    rot270_aug,
]

torch_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)



## === cell 4
sample_sub = pd.read_csv(sample_sub_path)

tta_count = len(tta_augs)  # now 7 deterministic augmentations

predictions = []
for _, sample_row in sample_sub.iterrows():
    agg_output = torch.zeros(5, device=device)
    img_path = os.path.join(test_images_path, sample_row.image_id)
    image = np.array(Image.open(img_path).convert("RGB"))
    for aug in tta_augs:
        aug_image = aug(image=image)["image"]
        tensor = transforms.ToTensor()(aug_image).to(device)
        tensor = torch_normalize(tensor)

        with torch.no_grad():
            out = model(tensor.unsqueeze(0)).squeeze(0)
        agg_output += out
    agg_output /= tta_count
    pred_label = torch.argmax(agg_output).item()
    predictions.append([sample_row.image_id, pred_label])

sub_df = pd.DataFrame(predictions, columns=["image_id", "label"])
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
