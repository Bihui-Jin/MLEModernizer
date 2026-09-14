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

0.8327289211242067

# 6. Current score

0.10725

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.37668) has done: 'I fix the runtime errors by correcting the augmentation pipeline: the installed Albumentations version does not contain the `Cutout` transform, so I remove that step. This allows the `sub_aug` object to be created properly, eliminating the `NameError` in the inference loop. No other logic is changed, preserving the model and inference behavior while ensuring a valid `submission.csv` is written.'
- What this solution (achieved 0.16143) has done: 'I replace the random test‑time augmentation pipeline with a deterministic resize‑and‑normalize transform (removing random crops, flips, and dropout). This keeps the model architecture unchanged but provides stable inputs during inference, which is expected to raise the validation accuracy toward the target without altering the core logic.'
- What this solution (achieved 0.19581) has done: 'I adjust the test‑time preprocessing to match EfficientNet‑B4’s expected input size (380 × 380). Using the correct resize should provide much richer features and raise the validation accuracy toward the target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.1364) has done: 'I add a few simple deterministic test‑time augmentations (horizontal flip, vertical flip, transpose) to the existing resize‑and‑normalize pipeline, keeping the EfficientNet‑B4 core unchanged. By averaging predictions over these five complementary views we expect a higher validation accuracy, moving the score toward the target while still producing a correct `submission.csv`.'
- What this solution (achieved 0.25336) has done: 'The fix corrects the softmax dimension error that caused a runtime exception during inference. By defining the softmax over dimension 0 (appropriate for the 1‑D tensor after squeezing) the model’s probability outputs are computed correctly, allowing the script to generate a valid `submission.csv`. This minimal change restores end‑to‑end execution while preserving the original architecture and inference logic.'
- What this solution (achieved 0.20179) has done: 'The update fixes the checkpoint loading to correctly pull the fine‑tuned weights (checking for common keys and using strict loading) and aligns the softmax to the proper tensor dimension, which yields more accurate class probabilities while preserving the original model and inference pipeline. These minimal adjustments are expected to raise the validation accuracy toward the target score.'
- What this solution (achieved 0.20142) has done: 'I relax the checkpoint loading to `strict=False` so that any matching fine‑tuned weights are loaded even if the state dict keys do not exactly match the model definition. This small change keeps the core architecture unchanged but allows the pretrained fine‑tuned parameters (when present) to be used, which should raise the validation accuracy toward the target score.'
- What this solution (achieved 0.10725) has done: 'I clean the checkpoint state‑dict keys (e.g., remove a possible “module.” prefix) before loading it, so the fine‑tuned weights are correctly applied to the EfficientNet‑B4 model. This small fix keeps the core logic unchanged while allowing the model to use the intended parameters, which should raise the validation accuracy toward the target score.'

# 9. Code solution

## === cell 0
import subprocess, sys

try:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "../input/efficientnet-pytorch-070/efficientnet_pytorch-0.7.0-py3-none-any.whl",
        ]
    )
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

try:
    from efficientnet_pytorch import EfficientNet
except ModuleNotFoundError:

    class EfficientNetWrapper(nn.Module):
        def __init__(self, num_classes=5):
            super().__init__()
            self.backbone = models.efficientnet_b4(
                weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
            )
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

        def forward(self, x):
            return self.backbone(x)

    EfficientNet = EfficientNetWrapper




## === cell 2
model_path = "../input/en-b4-tta-calr-15/model_15.pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"




## === cell 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if hasattr(EfficientNet, "from_pretrained"):
    model = EfficientNet.from_pretrained("efficientnet-b4", num_classes=5)
else:
    model = EfficientNet(num_classes=5)

model = model.to(device)

abs_model_path = os.path.abspath(model_path)
if os.path.exists(abs_model_path):
    try:
        state = torch.load(abs_model_path, map_location=device)
        if isinstance(state, dict):
            if "model" in state:
                state = state["model"]
            elif "state_dict" in state:
                state = state["state_dict"]
            cleaned_state = {}
            for k, v in state.items():
                new_k = k
                if k.startswith("module."):
                    new_k = k[7:]  # strip "module."
                cleaned_state[new_k] = v
            state = cleaned_state
        model.load_state_dict(state, strict=False)
        print("Checkpoint loaded with cleaned keys (strict=False).")
    except Exception as e:
        print(f"Warning: failed to load checkpoint ({e}); using pretrained weights.")
else:
    print("Checkpoint not found; using pretrained EfficientNet weights.")

model.eval()
torch.set_grad_enabled(False)




## === cell 4
import albumentations as A

base_transform = A.Compose(
    [
        A.Resize(height=380, width=380, p=1.0),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

tta_transforms = [
    base_transform,
    A.Compose(
        [
            A.HorizontalFlip(p=1.0),
            A.Resize(height=380, width=380, p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ],
        p=1.0,
    ),
    A.Compose(
        [
            A.VerticalFlip(p=1.0),
            A.Resize(height=380, width=380, p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ],
        p=1.0,
    ),
    A.Compose(
        [
            A.Transpose(p=1.0),  # rotate 90°
            A.Resize(height=380, width=380, p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ],
        p=1.0,
    ),
    base_transform,  # repeat identity to keep 5 runs
]




## === cell 5
sample_sub = pd.read_csv(sample_sub_path)

predictions = []
tta_runs = len(tta_transforms)  # 5 deterministic views

softmax = torch.nn.Softmax(dim=1)  # apply across class dimension

for _, sample_row in sample_sub.iterrows():
    image_path = os.path.join(test_images_path, sample_row.image_id)
    if not os.path.exists(image_path):
        print(f"Warning: image {image_path} not found, skipping.")
        continue
    raw_image = np.array(Image.open(image_path).convert("RGB"))

    prob_sum = torch.zeros(5, device=device)  # accumulate probabilities per class
    for aug in tta_transforms:
        aug_image = aug(image=raw_image)["image"]
        tensor_image = torch.from_numpy(aug_image).permute(2, 0, 1).float().to(device)
        outputs = model(tensor_image.unsqueeze(0))  # (1, 5)
        probs = softmax(outputs).squeeze(0)  # (5,)
        prob_sum += probs
    avg_probs = prob_sum / tta_runs
    pred_label = torch.argmax(avg_probs).item()
    predictions.append([sample_row.image_id, pred_label])

sub_df = pd.DataFrame(predictions, columns=["image_id", "label"])
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
