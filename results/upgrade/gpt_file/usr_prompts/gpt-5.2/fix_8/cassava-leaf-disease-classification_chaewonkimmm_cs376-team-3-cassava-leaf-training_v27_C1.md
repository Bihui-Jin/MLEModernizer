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

- What this solution (achieved 0.61099) has done: 'I fix the inference crash by filtering out directories inside `test_images` (the dataset includes a nested `test_images/` folder) so `PIL.Image.open` only sees real image files. Then I ensure predictions are generated for exactly the same `image_id` list as `sample_submission.csv`, which prevents length mismatches and guarantees correct submission ordering. Finally, I keep the model and TTA logic unchanged, but make the dataloaders deterministic for validation and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.54783) has done: 'Your current score is far below the target, and the biggest likely cause is a normalization mismatch: the model weights are for a ResNet34 trained with ImageNet normalization, but your TTA inference uses CIFAR-like mean/std, which heavily degrade accuracy. I keep the model, TTA structure (5 transforms + averaging), and inference loop intact, but switch the inference normalization to ImageNet stats (matching the training/weights) and remove randomness from TTA (replace random flip/crop with deterministic equivalents) so predictions are stable and aligned with evaluation. I also add a small safety check to ensure submission rows match `sample_submission.csv` ordering exactly (without changing paths). These are minimal changes aimed to move accuracy upward toward the target band without altering core architecture or training.'
- What this solution (achieved 0.10762) has done: 'Your current score is well below the target, so we should improve accuracy with minimal, low-risk changes that don’t alter the model or training loop. The biggest remaining issue is that your `CassavaDataset` ignores the passed `transform` and instead applies a hard-coded random training augmentation even for validation; this makes validation noisy and is inconsistent with your deterministic inference pipeline. I (1) fix the dataset to use the provided transform (and only apply a default transform when none is provided), (2) keep training transforms random and validation transforms deterministic with the same ImageNet normalization, and (3) batch TTA inference via a DataLoader for correct, faster, and more stable predictions while preserving the exact same 5-view TTA averaging logic and submission ordering.'
- What this solution (achieved 0.10762) has done: 'Your score is far below the target, and the most likely reason (without changing your model/loops) is that you are loading weights into a plain `resnet34()` with no matching pretrained backbone, so the head weights alone won’t produce meaningful predictions. I keep the exact same architecture (ResNet34 + Linear(…,5)), the same transforms, and the same 5-view TTA averaging, but I load an ImageNet-pretrained ResNet34 backbone first and then overlay your checkpoint with `strict=False` so the classifier weights are used when present. I also set deterministic seeds and DataLoader settings to stabilize outputs without changing training/inference semantics. This is a minimal, low-risk change expected to move accuracy up toward the target band while preserving your pipeline and submission format.'
- What this solution (achieved 0.10762) has done: 'Your current score (0.10762) is far below the target (0.78785), so we should fix the most likely correctness issue rather than “tune” the model. The biggest bug is that your checkpoint path points to `../input/512image/cassava_net_512.pth`, which is not in the provided data paths; that means you’re effectively submitting an ImageNet ResNet34 with a random 5-class head (near-random accuracy). I keep the exact same model (ResNet34 + Linear(…,5)) and the same TTA/inference logic, but make checkpoint loading robust by searching common Kaggle input locations for `cassava_net_512.pth` (and related `.pth` files) and loading the first match. This is a minimal change that should immediately move accuracy upward toward your target while preserving evaluation semantics and producing the same `submission.csv`.'
- What this solution (achieved 0.10762) has done: 'Your current score is far below the target, so we should fix a likely “silent correctness” issue rather than tune: right now you split train/valid but then you accidentally train on the full dataset (data leakage in your local validation), so you have no reliable signal and may have saved/loaded mismatched weights. I make the smallest change to ensure the training DataLoader uses `df_train` (not the full train) while keeping the same model, transforms, and inference/TTA unchanged. I also add a lightweight checkpoint existence print that makes it obvious whether you’re actually loading a cassava-trained checkpoint (since submitting an ImageNet backbone + random head scores near-random). These changes preserve core logic and should move accuracy upward toward your target band by ensuring you’re using the intended training split and (if you retrain) producing a meaningful checkpoint.'

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

from sklearn import model_selection



## === cell 1
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available(), device



## === cell 2
dfx = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")

df_train, df_valid = model_selection.train_test_split(
    dfx, test_size=0.1, random_state=42, stratify=dfx.label.values
)



## === cell 3
_train = df_train.reset_index(drop=True)
df_valid = df_valid.reset_index(drop=True)

image_path = "../input/cassava-leaf-disease-classification/train_images/"
train_image_paths = [os.path.join(image_path, x) for x in df_train.image_id.values]
valid_image_paths = [os.path.join(image_path, x) for x in df_valid.image_id.values]

train_targets = df_train.label.values
valid_targets = df_valid.label.values



## === cell 4
train_image_paths_full = train_image_paths
train_targets_full = train_targets



## === cell 5
len(train_image_paths_full), len(train_targets_full)



## === cell 6
len(valid_image_paths), len(valid_targets)



## === cell 7
"""torch module dataset"""


class CassavaDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.files = data
        self.targets = targets
        self.classes = list(set(targets)) if targets is not None else []
        self.transform = transform

        input_size = 512
        imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.default_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(*imagenet_stats),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        image = Image.open(img_name).convert("RGB")

        transform = (
            self.transform if self.transform is not None else self.default_transform
        )
        image = transform(image)

        label = self.targets[idx] if self.targets is not None else -1
        return image, label




## === cell 8
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
        transforms.ToTensor(),
        transforms.Normalize(*imagenet_stats),
    ]
)



## === cell 9
"""Dataset Initialization"""
cassava_data = CassavaDataset(
    train_image_paths_full, train_targets_full, transform=train_transform
)
cassava_test = CassavaDataset(
    valid_image_paths, valid_targets, transform=valid_transform
)



## === cell 10
batch_size = 16
g = torch.Generator()
g.manual_seed(SEED)

cassava_loader = DataLoader(
    cassava_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    generator=g,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True,
)
classes = ("0", "1", "2", "3", "4")

test_loader = DataLoader(
    cassava_test,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True,
)




## === cell 11
def denormalize(images, means, stds):
    if len(images.shape) == 3:
        images = images.unsqueeze(0)
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means


imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def show_image(img_tensor, label):
    print("Label:", cassava_data.classes[label], "(" + str(label) + ")")
    img_tensor = denormalize(img_tensor, *imagenet_stats)[0].permute((1, 2, 0))
    plt.imshow(img_tensor)


def imshow(img, label):
    npimg = img.numpy()
    print("Label:", cassava_data.classes[label], "(" + str(label) + ")")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()




## === cell 12
def reset_weights(m):
    """
    Try resetting model weights to avoid
    weight leakage.
    """
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()




## === cell 13
def find_checkpoint(preferred_path: str, search_roots=None, name_hints=None):
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path

    if search_roots is None:
        search_roots = ["../input", "/kaggle/input"]
    if name_hints is None:
        name_hints = [
            "cassava_net_512.pth",
            "cassava",
            "net_512",
            "512",
            ".pth",
        ]

    candidates = []
    for root in search_roots:
        if not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if not fn.lower().endswith((".pth", ".pt", ".bin")):
                    continue
                fn_l = fn.lower()
                if any(h.lower() in fn_l for h in name_hints):
                    candidates.append(os.path.join(dirpath, fn))

    for c in candidates:
        if os.path.basename(c) == "cassava_net_512.pth":
            return c

    candidates = sorted(candidates, key=lambda x: (len(x), x))
    return candidates[0] if candidates else None


PREFERRED_PATH = "../input/512image/cassava_net_512.pth"
FOUND_PATH = find_checkpoint(PREFERRED_PATH)

try:
    resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
except Exception:
    resnet = models.resnet34(pretrained=True)

num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 5)
resnet.to(device)

if FOUND_PATH is not None and os.path.exists(FOUND_PATH):
    state = torch.load(FOUND_PATH, map_location=device)

    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if isinstance(state, dict):
        cleaned = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            cleaned[nk] = v
        state = cleaned

    missing, unexpected = resnet.load_state_dict(state, strict=False)
    print("Loaded checkpoint from:", FOUND_PATH)
    if len(unexpected) > 0:
        print("WARNING: Unexpected keys in checkpoint:", unexpected[:10])
    if len(missing) > 0:
        print(
            "WARNING: Missing keys when loading checkpoint (showing first 10):",
            missing[:10],
        )
else:
    print(
        f"WARNING: Could not find checkpoint. Tried {PREFERRED_PATH} and searched /kaggle/input. "
        "Submitting with ImageNet-pretrained backbone + random head (will score poorly)."
    )

resnet.eval()



## === cell 14
submission_df = pd.read_csv(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)
submission_df.head()



## === cell 15
"""TTA

Keep identical core logic: 5 deterministic views + average logits.
"""
input_size = 512
stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

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
        transforms.RandomHorizontalFlip(p=1.0),  # deterministic flip view
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ]
)

trans3 = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.RandomVerticalFlip(p=1.0),  # deterministic flip view
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



## === cell 16
"""Inference (bugfix retained: ignore nested directories; align strictly to sample_submission order)

Batch TTA inference via DataLoader while preserving the same 5-view TTA and averaging.
"""

test_path = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

all_entries = os.listdir(test_path)
test_images_files = sorted(
    [f for f in all_entries if os.path.isfile(os.path.join(test_path, f))]
)

sub_ids = submission_df["image_id"].tolist()
if set(sub_ids) == set(test_images_files):
    test_images = sub_ids
else:
    test_images = test_images_files


class CassavaTTADataset(Dataset):
    def __init__(self, root, image_ids, tta_transforms):
        self.root = root
        self.image_ids = image_ids
        self.tta_transforms = tta_transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        fname = self.image_ids[idx]
        image = Image.open(os.path.join(self.root, fname)).convert("RGB")
        views = [t(image) for t in self.tta_transforms]  # list of [C,H,W]
        views = torch.stack(views, dim=0)  # [T,C,H,W]
        return views, fname


tta_ds = CassavaTTADataset(test_path, test_images, transs)
tta_loader = DataLoader(
    tta_ds,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True,
)

y_preds = []
pred_ids = []

resnet.eval()
with torch.inference_mode():
    for views, fnames in tta_loader:
        b, t, c, h, w = views.shape
        views = views.view(b * t, c, h, w).to(device, non_blocking=True)

        logits = resnet(views)  # [B*T,5]
        logits = logits.view(b, t, -1).mean(dim=1)  # [B,5] average over TTA
        preds = logits.argmax(dim=1).detach().cpu().numpy().astype(int).tolist()

        y_preds.extend(preds)
        pred_ids.extend(list(fnames))

len(y_preds), y_preds[:5]



## === cell 17
assert len(pred_ids) == len(y_preds), (len(pred_ids), len(y_preds))

df_sub = pd.DataFrame({"image_id": pred_ids, "label": y_preds})

if set(df_sub["image_id"]) == set(submission_df["image_id"]):
    df_sub = submission_df[["image_id"]].merge(df_sub, on="image_id", how="left")
    assert df_sub["label"].isna().sum() == 0

df_sub.head()



## === cell 18
df_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_sub.shape)
print(df_sub.head())
