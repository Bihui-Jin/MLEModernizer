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

0.8930190389845875

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11734) has done: 'The fix updates the augmentation pipeline to use a valid Albumentations API (replacing the outdated RandomResizedCrop with Resize) so the transform compiles and the `sub_aug` variable is defined. No other logic is altered, preserving the original model loading, inference, and submission generation. This resolves the runtime errors and enables the script to produce a proper `submission.csv` that can achieve the target accuracy.'
- What this solution (achieved 0.54484) has done: 'I remove the random test‑time augmentations and set TTA to a single deterministic pass. Using only a resize + normalization (the same preprocessing the models were trained with) prevents the noisy flips and rotations that were hurting the predictions, which should raise the validation accuracy toward the target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.2728) has done: 'I increase the test‑time augmentation to a modest deterministic set (a resize to the model’s native 224 px size plus a horizontal flip) and run a few augmentations per image (tta_count = 4). This better matches the training preprocessing, should raise validation accuracy toward the target while keeping the model architecture and inference logic unchanged.'
- What this solution (achieved 0.1151) has done: 'I make the inference deterministic by removing the random horizontal flip from the test‑time augmentation and limiting TTA to a single pass. This aligns the preprocessing exactly with what the models were trained on and eliminates noise that was dragging accuracy down, moving the score much closer to the target while keeping the original architecture and logic unchanged.'
- What this solution (achieved 0.11622) has done: 'I keep the overall architecture, loading and inference flow unchanged but modify the way model outputs are combined: each model’s raw logits be turned into probabilities with a soft‑max before averaging (instead of averaging logits). This small adjustment aligns the ensemble with how the models were trained and is expected to raise the validation accuracy toward the target without altering any core logic.'
- What this solution (achieved 0.12145) has done: 'I add a modest deterministic test‑time augmentation (horizontal flip) and increase the TTA count from 1 to 8 so the model predictions are averaged over several transformed views of each image. This small change keeps the original architecture and inference logic intact while providing additional informative variations that are known to raise accuracy, moving the score toward the target without altering any core components.'
- What this solution (achieved 0.61099) has done: 'I make the inference deterministic (remove the random horizontal flip and run a single pass) and add a simple class‑frequency prior derived from the training labels to bias the weak ImageNet‑pretrained models toward the most common classes. These small, targeted changes keep the original architecture and loading logic intact while giving the ensemble a better calibrated prediction, which should raise the validation accuracy toward the target.'
- What this solution (achieved 0.61099) has done: 'I increase deterministic test‑time augmentation by adding a guaranteed horizontal‑flip view and use both views (original + flipped) for ensemble averaging, which usually raises accuracy. I also simplify the class‑frequency prior to raw label frequencies (removing the aggressive square‑root scaling) so the model predictions are less over‑biased. These minimal adjustments keep the original architecture and inference flow unchanged while moving the validation score closer to the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import softmax

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

import albumentations  # augmentation library
from albumentations import HorizontalFlip

torch.manual_seed(42)
np.random.seed(42)
import random

random.seed(42)

try:
    from efficientnet_pytorch import EfficientNet
except ModuleNotFoundError:
    EfficientNet = None



## === cell 1
resnet_model_path = "../input/rn-wc-tta-calr-clahe-cutmix/model(24).pth"
effnet_model_path = "../input/en-b4-tta-calr-clahe-v3/eff_epoch_11.pth"

sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"



## === cell 2
resnet_model = models.resnext50_32x4d(pretrained=False)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_model = resnet_model.to(device)

if os.path.exists(resnet_model_path):
    resnet_model.load_state_dict(torch.load(resnet_model_path, map_location=device))
else:
    resnet_model = models.resnext50_32x4d(pretrained=True)
    resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 5)
    resnet_model = resnet_model.to(device)

resnet_model.eval()



## === cell 3
if EfficientNet is not None:
    effnet_model = EfficientNet.from_name("efficientnet-b4", num_classes=5)
else:
    effnet_model = models.efficientnet_b4(pretrained=False)
    effnet_model.classifier[1] = nn.Linear(effnet_model.classifier[1].in_features, 5)

effnet_model = effnet_model.to(device)

if os.path.exists(effnet_model_path):
    effnet_model.load_state_dict(torch.load(effnet_model_path, map_location=device))
else:
    if EfficientNet is None:
        effnet_model = models.efficientnet_b4(pretrained=True)
        effnet_model.classifier[1] = nn.Linear(
            effnet_model.classifier[1].in_features, 5
        )
    effnet_model = effnet_model.to(device)

effnet_model.eval()



## === cell 4
base_transforms = [
    albumentations.Resize(height=224, width=224),
    albumentations.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_pixel_value=255.0,
        p=1.0,
    ),
]

sub_aug = albumentations.Compose(base_transforms, p=1.0)
flip_aug = albumentations.Compose([HorizontalFlip(p=1.0)] + base_transforms, p=1.0)

tta_augs = [sub_aug, flip_aug]



## === cell 5
train_df = pd.read_csv(train_csv_path)
class_counts = train_df["label"].value_counts().sort_index()
class_prior = class_counts.values.astype(np.float32)
class_prior = class_prior / class_prior.sum()  # probabilities
epsilon = 1e-6

sample_sub = pd.read_csv(sample_sub_path)

predictions = []
for _, sample_row in sample_sub.iterrows():
    prob_sum = np.zeros(5, dtype=np.float32)  # accumulate probabilities
    for aug in tta_augs:  # deterministic TTA
        img_path = os.path.join(test_images_path, sample_row.image_id)
        image = np.array(Image.open(img_path).convert("RGB"))
        aug_out = aug(image=image)
        aug_image = aug_out["image"]  # normalized float32 array

        tensor_image = torch.from_numpy(aug_image).permute(2, 0, 1).float().to(device)

        with torch.no_grad():
            out_resnet = resnet_model(tensor_image.unsqueeze(0))
            out_effnet = effnet_model(tensor_image.unsqueeze(0))

        prob_resnet = F.softmax(out_resnet.squeeze(0), dim=0).cpu().numpy()
        prob_effnet = F.softmax(out_effnet.squeeze(0), dim=0).cpu().numpy()

        prob_sum += prob_resnet
        prob_sum += prob_effnet

    prob_avg = prob_sum / (len(tta_augs) * 2)  # average over models & TTA

    prob_adj = prob_avg * class_prior + epsilon
    prob_adj = prob_adj / prob_adj.sum()

    predictions.append(prob_adj)

predictions = np.stack(predictions)  # shape (N, 5)
final_labels = predictions.argmax(axis=1)

sub_df = pd.DataFrame({"image_id": sample_sub.image_id, "label": final_labels})
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
