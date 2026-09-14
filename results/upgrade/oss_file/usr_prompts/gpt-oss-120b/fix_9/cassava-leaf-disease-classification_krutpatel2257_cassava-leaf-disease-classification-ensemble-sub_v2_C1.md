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

0.8931701420368692

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.09193) has done: 'Implemented fixes:
- Replaced the invalid `RandomResizedCrop` with a simple `Resize` to satisfy Albumentations validation.
- Added a custom `to_tensor` lambda that converts the already‑normalized NumPy image directly to a torch tensor (avoiding double normalization).
- Ensured the augmentation pipeline (`sub_aug`) is defined correctly before use.
- Keeping the rest of the logic unchanged so the ensemble predictions are computed and a proper `submission.csv` is written.'
- What this solution (achieved 0.61099) has done: 'I add a simple class‑frequency prior derived from the training labels and blend it with the model‑based probabilities. This inexpensive calibration nudges predictions toward the most common classes, which should raise accuracy from the very low baseline toward the target without altering the core model architecture or training logic.'
- What this solution (achieved 0.4929) has done: 'I reduce the influence of the class‑frequency prior (which was pulling predictions toward the most common classes) and give the more powerful EfficientNet model a higher weight in the ensemble. These tiny adjustments keep the original architecture and training untouched while nudging the predictions toward the target accuracy.'
- What this solution (achieved 0.51271) has done: 'I increase the test‑time augmentation count, compute class probabilities for each model before ensembling, give the stronger EfficientNet model a higher weight, and reduce the influence of the class‑frequency prior. These small, targeted changes keep the original architecture and training untouched while nudging the predictions toward higher accuracy, moving the score closer to the target.'
- What this solution (achieved 0.0994) has done: 'The changes batch the forward passes for both models and share the augmentation work between them, turning ~64 k single‑image inference calls into 12 batched calls per model (plus lightweight augmentation loops). This keeps the exact TTA count, model architecture, and weighting while dramatically reducing Python‑level overhead and GPU kernel launches, fitting comfortably inside the 600 s limit.'
- What this solution (achieved 0.61099) has done: 'I incorporate the class‑frequency prior that was computed but never used. By blending a small weight of this prior with the model‑based probabilities before taking the arg‑max, we bias predictions toward the more common classes, which should raise the accuracy from the very low baseline toward the target while leaving the model architecture, training, and TTA untouched.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from scipy.special import softmax

import torch
import torch.nn as nn
from torchvision import models, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 1
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"



## === cell 2
resnet_model = models.resnext50_32x4d(pretrained=True)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 5)
resnet_model = resnet_model.to(device)
resnet_model.eval()

effnet_model = models.efficientnet_b4(pretrained=True)
effnet_model.classifier[1] = nn.Linear(effnet_model.classifier[1].in_features, 5)
effnet_model = effnet_model.to(device)
effnet_model.eval()



## === cell 3
sub_aug = A.Compose(
    [
        A.Resize(height=512, width=512),
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

sample_sub = pd.read_csv(sample_sub_path)

tta_count = 12

to_tensor = lambda img: torch.from_numpy(img).permute(2, 0, 1).float()

train_path = "../input/cassava-leaf-disease-classification/train.csv"
train_df = pd.read_csv(train_path)
class_counts = train_df["label"].value_counts().sort_index()
prior_probs = class_counts.values.astype(np.float32)
prior_probs = prior_probs / prior_probs.sum()  # shape (5,)



## === cell 4
image_paths = [
    os.path.join(test_images_path, img_id) for img_id in sample_sub["image_id"]
]
raw_images = []
for p in image_paths:
    with Image.open(p) as im:
        raw_images.append(np.array(im.convert("RGB")))

N = len(raw_images)
batch_size = 64  # adjust to fit GPU memory

resnet_sum = np.zeros((N, 5), dtype=np.float32)
effnet_sum = np.zeros((N, 5), dtype=np.float32)

with torch.no_grad():
    for _ in range(tta_count):
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_imgs = raw_images[start:end]

            aug_tensors = []
            for img in batch_imgs:
                aug_img = sub_aug(image=img)["image"]
                aug_tensors.append(to_tensor(aug_img))
            batch_tensor = torch.stack(aug_tensors).to(device)  # (B, C, H, W)

            res_out = resnet_model(batch_tensor).cpu().numpy()  # (B, 5)
            eff_out = effnet_model(batch_tensor).cpu().numpy()  # (B, 5)

            resnet_sum[start:end] += res_out
            effnet_sum[start:end] += eff_out

resnet_predictions = resnet_sum / tta_count
effnet_predictions = effnet_sum / tta_count



## === cell 5
resnet_probs = softmax(resnet_predictions, axis=1)
effnet_probs = softmax(effnet_predictions, axis=1)

combined = (resnet_probs * 0.15) + (effnet_probs * 0.85)

combined = combined / combined.sum(axis=1, keepdims=True)

prior_weight = 0.15  # modest influence of class prior
prior_matrix = np.broadcast_to(prior_probs, (N, 5))
combined = (combined * (1 - prior_weight)) + (prior_matrix * prior_weight)

combined = combined / combined.sum(axis=1, keepdims=True)

pred_labels = combined.argmax(axis=1)

submission_df = pd.DataFrame({"image_id": sample_sub["image_id"], "label": pred_labels})
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Submission written to {submission_path}")
print(submission_df.head())
