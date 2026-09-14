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

0.8859171955273496

# 6. Current score

0.65994

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.06054) has done: 'The changes fix the missing‑file errors by loading pretrained ImageNet weights when the custom checkpoints aren’t available, correct the sample‑submission path, and add a lightweight ensemble (ResNet + EfficientNet) to modestly boost the accuracy while keeping the original workflow untouched.'
- What this solution (achieved 0.58969) has done: 'The changes remove the unnecessary second‑epoch fine‑tuning of both models, which was the dominant cost, and skip creation of the training DataLoaders. The models are loaded from the provided checkpoints, the backbone layers are frozen, and the models are set to evaluation mode directly, preserving the exact inference logic and predictions while cutting runtime well under the 600‑second limit.'
- What this solution (achieved 0.07586) has done: 'I fix the missing‑checkpoint typo so the ResNet can load its fine‑tuned weights, and I use a resize of 224 × 224 for ResNet (its native input size) while keeping the 384 × 384 resize for EfficientNet. These minimal changes keep the overall architecture and inference flow unchanged but should improve the model accuracy, moving the score toward the target.'
- What this solution (achieved 0.27578) has done: 'I add a lightweight fine‑tuning stage that trains the classifier heads of the frozen ResNet‑50 and EfficientNet‑V2‑S on the provided training CSV for a couple of epochs. This supplies task‑specific weights (instead of only ImageNet weights) and should raise the validation‑style accuracy dramatically toward the target while keeping the original inference pipeline unchanged.'
- What this solution (achieved 0.11061) has done: 'I fixed the Albumentations `RandomResizedCrop` parameter error by passing a size tuple, and I modestly extended training to five epochs with a slightly lower learning rate to improve validation accuracy while keeping the original workflow intact. These changes resolve the runtime exception and should move the model’s score closer to the target.'
- What this solution (achieved 0.65994) has done: 'We cut the training epochs from 5 to 2 and enable persistent workers on the data loaders, which removes the overhead of repeatedly spawning worker processes each epoch. This keeps the same models, transforms, loss, and optimizer logic intact while substantially reducing total runtime, keeping predictions unchanged.'

# 9. Code solution

## === cell 0
pass



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
num_tta = 5



## === cell 3
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"



## === cell 4
test_df = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
test_df.head()



## === cell 5
efficientnet_transforms = A.Compose(
    [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(384, 384),  # Resize to match EfficientNetV2-S input size
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 6
class CassavaTestDataset(Dataset):
    def __init__(self, dataframe, image_dir):
        self.dataframe = dataframe
        self.image_dir = image_dir

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]  # Image ID
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, img_name




## === cell 7
tta_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

resnet_tta_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 8
def tta_predict_single_model(model, image, tta_transform, device, n_tta=5):
    """
    Perform TTA by creating a batch of `n_tta` augmented versions of the same image,
    running a **single** forward pass, and averaging the softmax probabilities.
    This yields identical predictions to the original loop but with far less GPU
    overhead.
    """
    model.eval()
    if image.shape[-1] != 3:
        raise ValueError("Image must have 3 channels (H, W, 3)")

    aug_batch = []
    for _ in range(n_tta):
        aug = tta_transform(image=image)["image"]
        aug_batch.append(aug)
    aug_batch = torch.stack(aug_batch).to(device)  # (n_tta, C, H, W)

    with torch.no_grad():
        outputs = model(aug_batch)  # (n_tta, num_classes)
        probs = F.softmax(outputs, dim=1)  # (n_tta, num_classes)

    avg_probs = torch.mean(probs, dim=0, keepdim=True)  # (1, num_classes)
    return avg_probs




## === cell 9
def identity_collate(batch):
    return batch


test_dataset = CassavaTestDataset(
    test_df,
    test_image_dir,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=identity_collate,  # Use custom collate function
)



## === cell 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True



## === cell 11
resnet_model = models.resnet50(pretrained=False)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 5)

resnet_path = (
    "/kaggle/input/cassava-aug/pytorch/default/1/cassava_leaf_best_model_fine_aug.pth"
)
if os.path.exists(resnet_path):
    try:
        resnet_model.load_state_dict(torch.load(resnet_path, map_location=device))
    except Exception:
        resnet_model = models.resnet50(pretrained=True)
        resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 5)
else:
    resnet_model = models.resnet50(pretrained=True)
    resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 5)

resnet_model = resnet_model.to(device)
resnet_model.eval()

efficientnet_model_7 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_7.classifier[1].in_features
efficientnet_model_7.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)

eff7_path = "/kaggle/input/effbest10temp/pytorch/default/1/Eff_best10.pth"
if os.path.exists(eff7_path):
    try:
        efficientnet_model_7.load_state_dict(torch.load(eff7_path, map_location=device))
    except Exception:
        efficientnet_model_7 = models.efficientnet_v2_s(
            weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )
        efficientnet_model_7.classifier = nn.Sequential(
            nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
        )
else:
    efficientnet_model_7 = models.efficientnet_v2_s(
        weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1
    )
    efficientnet_model_7.classifier = nn.Sequential(
        nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
    )

efficientnet_model_7 = efficientnet_model_7.to(device)
efficientnet_model_7.eval()




## === cell 12
def freeze_backbone(model):
    for name, param in model.named_parameters():
        if "fc" not in name and "classifier" not in name:
            param.requires_grad = False


freeze_backbone(resnet_model)
freeze_backbone(efficientnet_model_7)

resnet_model.eval()
efficientnet_model_7.eval()



## === cell 13
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_img_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"

train_df = pd.read_csv(train_csv_path)

train_df, val_df = train_test_split(
    train_df, test_size=0.1, stratify=train_df["label"], random_state=42
)

skip_training = os.path.exists(resnet_path) and os.path.exists(eff7_path)


class CassavaTrainDataset(Dataset):
    def __init__(self, dataframe, image_dir, transforms):
        self.df = dataframe
        self.img_dir = image_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["image_id"]
        label = int(row["label"])
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        aug = self.transforms(image=image)
        tensor = aug["image"]  # already a torch Tensor
        return tensor, label


resnet_train_tf = A.Compose(
    [
        A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

efficientnet_train_tf = A.Compose(
    [
        A.RandomResizedCrop(size=(384, 384), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

train_dataset_resnet = CassavaTrainDataset(train_df, train_img_dir, resnet_train_tf)
val_dataset_resnet = CassavaTrainDataset(val_df, train_img_dir, resnet_train_tf)

train_dataset_eff = CassavaTrainDataset(train_df, train_img_dir, efficientnet_train_tf)
val_dataset_eff = CassavaTrainDataset(val_df, train_img_dir, efficientnet_train_tf)

train_loader_resnet = DataLoader(
    train_dataset_resnet,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)
val_loader_resnet = DataLoader(
    val_dataset_resnet,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
)

train_loader_eff = DataLoader(
    train_dataset_eff,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)
val_loader_eff = DataLoader(
    val_dataset_eff,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
)

criterion = nn.CrossEntropyLoss()
optimizer_resnet = optim.Adam(resnet_model.fc.parameters(), lr=5e-4)
optimizer_eff = optim.Adam(efficientnet_model_7.classifier.parameters(), lr=5e-4)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


epochs = 2
if not skip_training:
    for epoch in range(epochs):
        _ = train_one_epoch(resnet_model, train_loader_resnet, optimizer_resnet, device)
        _ = train_one_epoch(
            efficientnet_model_7, train_loader_eff, optimizer_eff, device
        )



## === cell 14
ensemble_predictions = []
image_names = []

with torch.no_grad():
    for batch in test_loader:
        image, img_name = batch[0]  # batch[0] gives (image, img_name)

        if isinstance(image, torch.Tensor):
            image = image.numpy()

        probs_eff7 = tta_predict_single_model(
            efficientnet_model_7, image, tta_transform, device, n_tta=num_tta
        )

        probs_resnet = tta_predict_single_model(
            resnet_model, image, resnet_tta_transform, device, n_tta=num_tta
        )

        combined_probs = (probs_eff7 + probs_resnet) / 2.0

        final_pred = combined_probs.argmax(dim=1).cpu().item()  # final class index

        ensemble_predictions.append(final_pred)
        image_names.append(img_name)



## === cell 15
submission_df = pd.DataFrame(
    {
        "image_id": image_names,  # Image IDs from the test set
        "label": ensemble_predictions,  # Predicted labels for each image
    }
)

submission_df.to_csv("submission.csv", index=False)

print("Submission file saved as 'submission.csv'")
