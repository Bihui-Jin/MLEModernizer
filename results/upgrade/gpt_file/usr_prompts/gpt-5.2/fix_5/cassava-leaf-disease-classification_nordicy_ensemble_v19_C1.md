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

0.8756421879721971

# 6. Current score

0.61734

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.36958) has done: 'I fix the Albumentations `RandomResizedCrop` API error by using the v2 signature (`size=(h,w)`), which unblocks definition of `tta_transform`. Then I remove hard dependencies on missing external weight files by falling back to a standard ImageNet-pretrained EfficientNetV2-S when those files are not present, so the notebook runs end-to-end and produces predictions. Finally, I fix the submission length/format issue by ensuring `image_names` are plain strings (not tuples) and that the output `submission.csv` matches exactly the `sample_submission.csv` order and length.'
- What this solution (achieved 0.61846) has done: 'Your score is far below the target, and the main cause is that you are effectively submitting predictions from a randomly-initialized 5-class head (because the external fine-tuned checkpoints usually aren’t available), so accuracy collapses. The smallest change that preserves your core approach (same model family, same TTA loop, same argmax) is to keep EfficientNetV2-S ImageNet weights but replace the random 5-class classifier with a deterministic “ImageNet → Cassava” label mapping derived from the training set (majority cassava label per ImageNet top-1 class). I also fix the subtle but important bug where `img_name` becomes `"('xxx.jpg',)"` due to the custom collate, which breaks the mapping and forces many fillna(0) labels. These changes should move accuracy substantially upward toward your target while keeping the inference-only pipeline and TTA logic intact and still producing a valid `submission.csv`.'
- What this solution (achieved 0.61061) has done: 'We keep your exact inference-only + TTA + “ImageNet→Cassava majority map” core logic, but strengthen the mapping step so it generalizes better and reduces the gap to your target. The smallest high-impact change is to build the mapping from the full training set (not a 4k sample) and to do it efficiently on GPU by accumulating counts with `torch.bincount`, which fits within the time budget and avoids the slow Python loop. We also align preprocessing between mapping and prediction by using the same CLAHE+Resize pipeline (without random ops) for mapping so the ImageNet top-1 classes are more consistent with what you feed at test time. These changes should increase accuracy toward ~0.875 without changing model architectures, losses, or training loops (none exist here).'
- What this solution (achieved 0.61734) has done: 'Your current score is far below the target, so we should improve accuracy with the smallest possible change that keeps your inference-only + TTA + “ImageNet→Cassava majority map” approach intact. The main weakness is that the mapping uses only ImageNet top-1; we can make it more robust by accumulating Cassava label counts from ImageNet top-k (k=5) predictions per training image, then mapping each ImageNet class to the Cassava label it most often co-occurs with. This preserves your exact pipeline (no cassava training, same EfficientNetV2-S ImageNet backbone, same argmax-at-the-end semantics) but reduces mapping noise and typically boosts accuracy. I also keep submission alignment exactly to `sample_submission.csv` and keep all file paths unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import random

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(42)



## === cell 1
num_tta = 5



## === cell 2
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"



## === cell 3
test_df = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
test_df.head()



## === cell 4
efficientnet_transforms = A.Compose(
    [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 5
class CassavaTestDataset(Dataset):
    def __init__(self, dataframe, image_dir):
        self.dataframe = dataframe
        self.image_dir = image_dir

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]  # image_id
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, img_name




## === cell 6
tta_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.RandomResizedCrop(size=(384, 384), scale=(0.8, 1.0), p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 7
def tta_predict_single_model(model, image, tta_transform, device, n_tta=5):
    model.eval()
    tta_predictions = []

    if image.shape[-1] != 3:
        raise ValueError("Image must have 3 channels (H, W, 3)")

    with torch.no_grad():
        for _ in range(n_tta):
            augmented = tta_transform(image=image)["image"]
            augmented = augmented.unsqueeze(0).to(device)

            output = model(augmented)
            probs = F.softmax(output, dim=1)
            tta_predictions.append(probs)

    avg_probs = torch.mean(torch.stack(tta_predictions), dim=0)
    return avg_probs




## === cell 8
def identity_collate(batch):
    return batch


test_dataset = CassavaTestDataset(test_df, test_image_dir)
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=identity_collate,
)



## === cell 9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 10
train_df = pd.read_csv(train_csv_path)

map_image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"

map_df = train_df.reset_index(drop=True)

mapping_transform = A.Compose(
    [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


class CassavaMapDataset(Dataset):
    def __init__(self, dataframe, image_dir):
        self.df = dataframe
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "image_id"]
        y = int(self.df.loc[idx, "label"])
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x = mapping_transform(image=image)["image"]
        return x, y


map_loader = DataLoader(
    CassavaMapDataset(map_df, map_image_dir),
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

imagenet_effnet = models.efficientnet_v2_s(
    weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
).to(device)
imagenet_effnet.eval()

topk_for_mapping = 5

counts = torch.zeros((1000, 5), dtype=torch.int32, device=device)

with torch.no_grad():
    for xb, yb in tqdm(
        map_loader, total=len(map_loader), desc="Building ImageNet->Cassava map (top-k)"
    ):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True).long()  # [B]

        logits = imagenet_effnet(xb)  # [B,1000]
        topk = logits.topk(k=topk_for_mapping, dim=1).indices.long()  # [B,k]

        yb_rep = yb[:, None].expand(-1, topk_for_mapping).reshape(-1)  # [B*k]
        topk_flat = topk.reshape(-1)  # [B*k]

        idx = topk_flat * 5 + yb_rep  # [B*k]
        bc = torch.bincount(idx, minlength=1000 * 5).to(torch.int32).view(1000, 5)
        counts += bc

global_majority = int(train_df["label"].value_counts().idxmax())

mapping = counts.argmax(dim=1).detach().cpu().numpy().astype(int)
seen = counts.sum(dim=1).detach().cpu().numpy()
mapping[seen == 0] = global_majority

mapping[:10], global_majority



## === cell 11
resnet_model = models.resnet50(weights=None)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 5)

resnet_ckpt_path = (
    "/kaggle/input/casava-aug/pytorch/default/1/cassava_leaf_best_model_fine_aug.pth"
)
if os.path.exists(resnet_ckpt_path):
    resnet_model.load_state_dict(torch.load(resnet_ckpt_path, map_location=device))
resnet_model = resnet_model.to(device)
resnet_model.eval()



## === cell 12
efficientnet_model_1 = models.efficientnet_v2_s(
    weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
)
num_features_efficientnet = efficientnet_model_1.classifier[1].in_features
efficientnet_model_1.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)

eff1_ckpt_path = "/kaggle/input/eff-5/pytorch/default/1/Eff_best5.pth"
if os.path.exists(eff1_ckpt_path):
    efficientnet_model_1.load_state_dict(
        torch.load(eff1_ckpt_path, map_location=device)
    )

efficientnet_model_1 = efficientnet_model_1.to(device)
efficientnet_model_1.eval()



## === cell 13
efficientnet_model_7 = models.efficientnet_v2_s(
    weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
)
num_features_efficientnet7 = efficientnet_model_7.classifier[1].in_features
efficientnet_model_7.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet7, 5)
)

eff7_ckpt_path = "/kaggle/input/eff-7-last/pytorch/default/1/Eff7_0.8799.pth"
if os.path.exists(eff7_ckpt_path):
    efficientnet_model_7.load_state_dict(
        torch.load(eff7_ckpt_path, map_location=device)
    )

efficientnet_model_7 = efficientnet_model_7.to(device)
efficientnet_model_7.eval()



## === cell 14
efficientnet_model_8 = models.efficientnet_v2_s(
    weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
)
num_features_efficientnet8 = efficientnet_model_8.classifier[1].in_features
efficientnet_model_8.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet8, 5)
)

eff8_ckpt_path = "/kaggle/input/eff-6/pytorch/default/1/Eff_best6.pth"
if os.path.exists(eff8_ckpt_path):
    efficientnet_model_8.load_state_dict(
        torch.load(eff8_ckpt_path, map_location=device)
    )

efficientnet_model_8 = efficientnet_model_8.to(device)
efficientnet_model_8.eval()




## === cell 15
def _unwrap_img_name(x):
    if isinstance(x, (list, tuple)) and len(x) == 1:
        return str(x[0])
    return str(x)


def tta_predict_imagenet_to_cassava(
    imagenet_model, image, tta_transform, device, mapping_arr, n_tta=5
):
    imagenet_model.eval()
    tta_logits = []
    with torch.no_grad():
        for _ in range(n_tta):
            augmented = tta_transform(image=image)["image"].unsqueeze(0).to(device)
            logits = imagenet_model(augmented)  # [1,1000]
            tta_logits.append(logits.detach().cpu())
    avg_logits = torch.mean(torch.stack(tta_logits), dim=0)  # [1,1000]
    top1 = int(avg_logits.argmax(dim=1).item())
    return int(mapping_arr[top1])


ensemble_predictions = []
image_names = []

for batch in tqdm(test_loader, total=len(test_loader), desc="Predicting"):
    image, img_name = batch[0]

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    img_name = _unwrap_img_name(img_name)

    pred = tta_predict_imagenet_to_cassava(
        imagenet_effnet, image, tta_transform, device, mapping, n_tta=num_tta
    )

    ensemble_predictions.append(int(pred))
    image_names.append(img_name)

len(image_names), len(ensemble_predictions), len(test_df)



## === cell 16
pred_map = dict(zip(image_names, ensemble_predictions))
submission_df = test_df.copy()
submission_df["label"] = submission_df["image_id"].map(pred_map)

submission_df["label"] = submission_df["label"].fillna(global_majority).astype(int)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Saved: {submission_path}")
print(submission_df.head())
print(
    "Rows:", len(submission_df), "Missing labels:", submission_df["label"].isna().sum()
)
