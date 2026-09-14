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

3.13

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
tqdm==4.67.1

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

0.8747355696585071

# 6. Current score

0.14013

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11136) has done: 'I fix the script so it always creates a valid `submission.csv`. The changes load the sample‑submission file using an absolute path, wrap model weight loading in a try/except (falling back to ImageNet‑pretrained weights if the custom checkpoints are missing), and write the output to the standard Kaggle working directory. These minimal adjustments keep the original architecture and ensemble logic while ensuring the pipeline runs end‑to‑end and produces a submission file.'
- What this solution (achieved 0.13303) has done: 'I correct the checkpoint paths (fix the typo “casava‑aug” → “cassava‑aug” and add a fallback to a generic “cassava‑leaf‑disease‑classification” folder) so the fine‑tuned weights can be loaded, and then quickly evaluate both models on a small validation split of the training data. Using the validation accuracies I recompute the ensemble weights (higher‑performing model gets a larger weight). This keeps the original architecture and inference pipeline but should raise the accuracy toward the target without changing any core logic.'
- What this solution (achieved 0.14013) has done: 'I keep the overall pipeline unchanged but modify the ensemble weighting logic so that the model with the higher validation accuracy receives all the weight (the other gets zero). This simple change aligns the predictions with the better‑performing model and is expected to move the Kaggle score closer to the target while preserving the core architecture and inference code.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.models import EfficientNet_V2_S_Weights
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
import copy
import cv2
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2




## === cell 2
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"




## === cell 3
sample_submission_path = os.path.join(
    "/kaggle/input",
    "cassava-leaf-disease-classification",
    "sample_submission.csv",
)
test_df = pd.read_csv(sample_submission_path)
test_df.head()




## === cell 4
resnet_transforms = A.Compose(
    [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(224, 224),  # Resize to match ResNet input size
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)
efficientnet_transforms = A.Compose(
    [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(384, 384),  # Resize to match EfficientNetV2-S input size
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 5
class CassavaTestDataset(Dataset):
    def __init__(
        self, dataframe, image_dir, transform_resnet=None, transform_efficientnet=None
    ):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform_resnet = transform_resnet
        self.transform_efficientnet = transform_efficientnet

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]  # Image ID
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform_resnet:
            augmented_resnet = self.transform_resnet(image=image)
            image_resnet = augmented_resnet["image"]
        else:
            image_resnet = None

        if self.transform_efficientnet:
            augmented_efficientnet = self.transform_efficientnet(image=image)
            image_efficientnet = augmented_efficientnet["image"]
        else:
            image_efficientnet = None

        return image_resnet, image_efficientnet, img_name




## === cell 6
test_dataset = CassavaTestDataset(
    test_df,
    test_image_dir,
    transform_resnet=resnet_transforms,
    transform_efficientnet=efficientnet_transforms,
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)




## === cell 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 8
resnet_model = models.resnet50(pretrained=False)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 5)

possible_resnet_paths = [
    "/kaggle/input/cassava-aug/pytorch/default/1/cassava_leaf_best_model_fine_aug.pth",
    "/kaggle/input/cassava-leaf-disease-classification/cassava_leaf_best_model_fine_aug.pth",
]
state_dict = None
for p in possible_resnet_paths:
    if os.path.exists(p):
        try:
            state_dict = torch.load(p, map_location=device)
            print(f"Loaded ResNet checkpoint from {p}")
            break
        except Exception as e:
            print(f"Failed loading ResNet from {p}: {e}")

if state_dict is None:
    print("Could not find custom ResNet weights, using ImageNet pretrained model.")
    resnet_model = models.resnet50(pretrained=True)
    num_ftrs = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_ftrs, 5)

resnet_model = resnet_model.to(device)
resnet_model.eval()




## === cell 9
efficientnet_model = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model.classifier[1].in_features
efficientnet_model.classifier[1] = nn.Linear(num_features_efficientnet, 5)

possible_eff_paths = [
    "/kaggle/input/eff-t/pytorch/default/1/Eff.pth",
    "/kaggle/input/cassava-leaf-disease-classification/Eff.pth",
]
state_dict = None
for p in possible_eff_paths:
    if os.path.exists(p):
        try:
            state_dict = torch.load(p, map_location=device)
            print(f"Loaded EfficientNet checkpoint from {p}")
            break
        except Exception as e:
            print(f"Failed loading EfficientNet from {p}: {e}")

if state_dict is None:
    print(
        "Could not find custom EfficientNet weights, using ImageNet pretrained weights."
    )
    efficientnet_model = models.efficientnet_v2_s(
        weights=EfficientNet_V2_S_Weights.DEFAULT
    )
    num_features_efficientnet = efficientnet_model.classifier[1].in_features
    efficientnet_model.classifier[1] = nn.Linear(num_features_efficientnet, 5)

efficientnet_model = efficientnet_model.to(device)
efficientnet_model.eval()




## === cell 10
train_csv_path = os.path.join(
    "/kaggle/input",
    "cassava-leaf-disease-classification",
    "train.csv",
)
train_df = pd.read_csv(train_csv_path)

train_df, _ = train_test_split(
    train_df, test_size=0.9, stratify=train_df["label"], random_state=42
)
val_df = train_df.sample(n=min(2000, len(train_df)), random_state=42)

val_dataset = CassavaTestDataset(
    val_df,
    "/kaggle/input/cassava-leaf-disease-classification/train_images",
    transform_resnet=resnet_transforms,
    transform_efficientnet=efficientnet_transforms,
)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)


def evaluate(model, loader, device, use_resnet=True):
    correct = 0
    total = 0
    with torch.no_grad():
        for img_resnet, img_eff, img_names in loader:
            inputs = img_resnet.to(device) if use_resnet else img_eff.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            true = val_df.iloc[total : total + len(preds), 1].values
            correct += (preds == true).sum()
            total += len(preds)
    return correct / total if total > 0 else 0.0


acc_resnet = evaluate(resnet_model, val_loader, device, use_resnet=True)
acc_eff = evaluate(efficientnet_model, val_loader, device, use_resnet=False)

print(f"Validation accuracy – ResNet: {acc_resnet:.4f}, EfficientNet: {acc_eff:.4f}")

if acc_resnet >= acc_eff:
    weight_resnet = 1.0
    weight_efficientnet = 0.0
else:
    weight_resnet = 0.0
    weight_efficientnet = 1.0

print(
    f"Ensemble weights – ResNet: {weight_resnet:.3f}, EfficientNet: {weight_efficientnet:.3f}"
)




## === cell 11
ensemble_predictions = []
image_names = []

with torch.no_grad():
    for images_resnet, images_efficientnet, img_names in test_loader:
        if images_resnet is None:
            images_resnet = torch.zeros((len(img_names), 3, 224, 224), device=device)
        if images_efficientnet is None:
            images_efficientnet = torch.zeros(
                (len(img_names), 3, 384, 384), device=device
            )

        images_resnet = images_resnet.to(device)
        images_efficientnet = images_efficientnet.to(device)

        outputs_resnet = resnet_model(images_resnet)
        probs_resnet = F.softmax(outputs_resnet, dim=1)

        outputs_efficientnet = efficientnet_model(images_efficientnet)
        probs_efficientnet = F.softmax(outputs_efficientnet, dim=1)

        combined_probs = (weight_resnet * probs_resnet) + (
            weight_efficientnet * probs_efficientnet
        )

        preds = combined_probs.argmax(dim=1).cpu().numpy()

        ensemble_predictions.extend(preds)
        image_names.extend(img_names)




## === cell 12
submission_df = pd.DataFrame({"image_id": image_names, "label": ensemble_predictions})

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Submission file saved as '{submission_path}'")
