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

geopandas==0.14.4
imbalanced-learn==0.13.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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

0.7878513145965549

# 6. Current score

0.10762

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'Your notebook currently can’t “yield” a score because it depends on a missing pretrained weights file (`../input/512image/cassava_net_512.pth`) and it also risks creating an invalid submission due to test filename order and non-deterministic TTA. I make the smallest changes that (1) ensure the model weights path resolves from the competition dataset (fallback to random init if not found, so a submission is always produced), (2) ensure inference is deterministic and fast enough by disabling gradients and setting fixed seeds while keeping your exact model and TTA transforms, and (3) guarantee the submission rows align 1:1 with `sample_submission.csv` image order (this alone can drastically improve accuracy vs. mismatched ordering). These changes preserve your core logic (ResNet34 + TTA mean logits + argmax) and only fix execution/submission correctness issues. The result always write a valid `submission.csv`.'
- What this solution (achieved 0.10762) has done: 'I first fix the immediate runtime error by removing the unused `imblearn`/`SMOTE` import that is incompatible with the provided scikit-learn version. Then I fix a critical dataset/transform logic bug: your `CassavaDataset` ignores the passed `transform` and always applies random augmentations (even for validation), which breaks correctness and reproducibility; I preserve your augmentations by moving them to the DataLoader-time transform and keeping inference unchanged. Finally, I keep your submission-row ordering aligned exactly to `sample_submission.csv` (already mostly done) and make inference deterministic by reseeding right before TTA so the random TTA transforms are stable run-to-run (score-neutral but avoids accidental variance).'
- What this solution (achieved 0.10762) has done: 'Your score is far below target, so we should improve genuine predictive performance while keeping your core approach intact (ResNet34 + mean-logits TTA + argmax). The biggest issue is that your TTA normalization stats differ from training/standard ImageNet stats, which can severely break a pretrained ResNet’s input distribution and collapse accuracy; I align TTA normalization to the same ImageNet stats you already use for train/valid. I also remove randomness from TTA transforms (keep the same set of TTA “views” but make them deterministic via fixed flip/crop variants), because stochastic crops can hurt accuracy when averaged only 5 times and adds instability. Finally, I add a safe fallback to load weights even if they were saved from a `DataParallel` model (common “module.” prefix), which can otherwise silently prevent proper weight loading in many notebooks; this keeps your exact architecture and evaluation semantics.'
- What this solution (achieved 0.10762) has done: 'Your current score (0.10762) is far below the target (0.78785), and the main reason is that the model is almost certainly running with randomly initialized weights because `cassava_net_512.pth` isn’t present in the provided data paths. To move toward the target without changing the core logic (ResNet34 + mean-logits TTA + argmax), I switch the backbone initialization to ImageNet-pretrained ResNet34 (same architecture, just proper initialization) and keep your existing custom-weight loading as an override if the file exists. I also fix a small but important bug where `df_train` (not `dfx`) is used after you overwrite it, which can cause unintended training dataset selection if you later extend training; this is score-safety with minimal change. Everything else (transforms/TTA/inference order/submission format) stays the same, and it still always produce `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

from sklearn import metrics, model_selection, preprocessing
from tqdm.auto import tqdm


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)



## === cell 1
torch.cuda.is_available()



## === cell 2
dfx = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")

df_train, df_valid = model_selection.train_test_split(
    dfx, test_size=0.1, random_state=42, stratify=dfx.label.values
)



## === cell 3
df_train = df_train.reset_index(drop=True)
df_valid = df_valid.reset_index(drop=True)

image_path = "../input/cassava-leaf-disease-classification/train_images/"
train_image_paths = [os.path.join(image_path, x) for x in df_train.image_id.values]
valid_image_paths = [os.path.join(image_path, x) for x in df_valid.image_id.values]

train_targets = df_train.label.values
valid_targets = df_valid.label.values



## === cell 4
dfx_full = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")

image_path_full = "../input/cassava-leaf-disease-classification/train_images/"
full_train_image_paths = [
    os.path.join(image_path_full, x) for x in dfx_full.image_id.values
]
full_train_targets = dfx_full.label.values



## === cell 5
len(train_image_paths), len(train_targets)



## === cell 6
len(valid_image_paths), len(valid_targets)



## === cell 7
"""torch module dataset"""


class CassavaDataset(Dataset):  # Override torch.utils.data.Dataset
    def __init__(self, data, targets, transform=None):
        self.files = data
        self.targets = targets
        self.classes = list(set(targets))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.files[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = int(self.targets[idx])
        return image, label




## === cell 8
"""Dataset Initialization"""

input_size = 512
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(*imagenet_stats),
    ]
)

valid_transform = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.CenterCrop((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(*imagenet_stats),
    ]
)

cassava_data = CassavaDataset(
    train_image_paths, train_targets, transform=train_transform
)
cassava_test = CassavaDataset(
    valid_image_paths, valid_targets, transform=valid_transform
)



## === cell 9
batch_size = 16

cassava_loader = DataLoader(
    cassava_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
)
classes = ("0", "1", "2", "3", "4")

test_loader = DataLoader(
    cassava_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)




## === cell 10
def denormalize(images, means, stds):
    if len(images.shape) == 3:
        images = images.unsqueeze(0)
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means


def show_image(img_tensor, label):
    print("Label:", cassava_data.classes[label], "(" + str(label) + ")")
    img_tensor = denormalize(img_tensor, *imagenet_stats)[0].permute((1, 2, 0))
    plt.imshow(img_tensor)
    plt.axis("off")


def imshow(img, label):
    npimg = img.numpy()
    print("Label:", cassava_data.classes[label], "(" + str(label) + ")")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")
    plt.show()




## === cell 11
def reset_weights(m):
    """
    Try resetting model weights to avoid
    weight leakage.
    """
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()




## === cell 12
def _strip_module_prefix_if_present(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    keys = list(state_dict.keys())
    if len(keys) > 0 and all(k.startswith("module.") for k in keys):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


CANDIDATE_PATHS = [
    "../input/512image/cassava_net_512.pth",  # original path (may not exist)
    "/kaggle/input/512image/cassava_net_512.pth",
    "../input/cassava_net_512.pth",
    "/kaggle/input/cassava_net_512.pth",
]
PATH = None
for pth in CANDIDATE_PATHS:
    if os.path.exists(pth):
        PATH = pth
        break

try:
    resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
except Exception:
    resnet = models.resnet34(pretrained=True)

num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 5)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet.to(device)

if PATH is not None:
    state = torch.load(PATH, map_location=device)
    state = _strip_module_prefix_if_present(state)
    missing, unexpected = resnet.load_state_dict(state, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        print(
            "WARNING: Non-strict weight load. Missing:",
            len(missing),
            "Unexpected:",
            len(unexpected),
        )
else:
    print(
        "WARNING: Pretrained cassava weights not found; using ImageNet-pretrained backbone + random FC. "
        "This is substantially better than full random init and should move accuracy toward the target."
    )

resnet.eval()



## === cell 13
submission_df = pd.read_csv(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)
submission_df.head()



## === cell 14
"""TTA"""

input_size = 512
stats = imagenet_stats

transform = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ]
)

trans1 = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.Pad(8, padding_mode="reflect"),
        transforms.CenterCrop((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ]
)

trans2 = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ]
)

trans3 = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ]
)

trans4 = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ]
)
transs = [transform, trans1, trans2, trans3, trans4]



## === cell 15
"""Inference"""

seed_everything(42)

test_path = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

test_images = submission_df["image_id"].tolist()

y_preds = []

resnet.eval()
with torch.no_grad():
    for i in tqdm(test_images, total=len(test_images)):
        image = Image.open(os.path.join(test_path, i)).convert("RGB")

        outs = torch.zeros((len(transs), 5), device="cpu")
        for k, trans in enumerate(transs):
            img = trans(image)
            img = img.unsqueeze(0).to(device, non_blocking=True)
            out = resnet(img).detach().float().cpu().squeeze(0)
            outs[k, :] = out

        out = outs.mean(dim=0)
        predicted = int(torch.argmax(out).item())
        y_preds.append(predicted)



## === cell 16
df_sub = pd.DataFrame({"image_id": test_images, "label": y_preds})
df_sub.head()



## === cell 17
df_sub["label"] = df_sub["label"].astype(int)
df_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_sub.shape)
print(df_sub.head())
print("submission.csv exists:", os.path.exists("submission.csv"))
