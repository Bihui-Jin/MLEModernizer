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

2.7

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-image==0.25.2
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

0.8529767301299487

# 6. Current score

0.2145

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I remove the dependency on the missing `efficientnet_pytorch` package by switching to `torchvision.models.efficientnet_b3`, which preserves the EfficientNet-B3 architecture and keeps inference logic the same. I also fix the Albumentations API break (`A.Flip` doesn’t exist in v2) by replacing it with `A.HorizontalFlip` and make the custom tensor transform compatible with Albumentations’ `Compose` by returning a dict. Finally, I harden paths, loading (`map_location`), and DataLoader collation so the loop defines `names/predicted` reliably and always writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.23318) has done: 'I fix the Albumentations v2 compatibility error by replacing the custom `ToTensor` with `albumentations.pytorch.ToTensorV2`, which provides the required interface and returns a dict as expected by `Compose`. I also fix the missing checkpoint path by adding a safe fallback: if the external model file isn’t present, we still run end-to-end with the same EfficientNet-B3 architecture (random weights), producing a valid `submission.csv`. These changes are minimal, unblock execution, and raise the score from near-random only if a valid pretrained checkpoint is actually available; otherwise the run is correctness-focused and deterministic. Finally, I ensure `testloader` is defined by making the transform/dataloader cell error-free and keep the submission aligned to `sample_submission.csv`.'
- What this solution (achieved 0.1136) has done: 'I fix the DataLoader crash by filtering `test_images` to include only real image files and by making image reading more robust (skip/replace unreadable items instead of throwing). This addresses the root cause shown in the traceback: a directory path (nested `test_images/`) is being passed into `imread`. I also set deterministic flags and keep the existing model/transform/inference logic unchanged, so score only improves insofar as the model can now run through all test images and produce a complete submission. Finally, I ensure the submission is aligned to `sample_submission.csv` and always writes a valid `submission.csv`.'
- What this solution (achieved 0.2145) has done: 'Your current score is far below the target because the model is almost certainly running with random weights (the checkpoint path points to a dataset that likely doesn’t exist), so we need to load real pretrained weights while keeping the exact same EfficientNet-B3 inference core. I minimally change the model initialization to use `torchvision`’s built-in ImageNet pretrained EfficientNet-B3 weights as a safe fallback when the competition checkpoint is missing, and I replace the invalid `CenterCrop(512)` (test images aren’t guaranteed to be ≥512) with a deterministic `Resize(512,512)` to avoid implicit failures/garbage inputs. These two changes preserve the overall architecture and inference loop, but should substantially increase accuracy toward your target. The submission writing/alignment logic remain the same and still produce a valid `submission.csv`.'

# 9. Code solution

## === cell 0
from __future__ import print_function, division

import os
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage import io

warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

torch.manual_seed(42)
np.random.seed(42)
if use_cuda:
    torch.cuda.manual_seed_all(42)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

MODEL_PATH = "/kaggle/input/effnetmodel/efficientnet-b3-e10.pt"




## === cell 1
class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        all_entries = sorted(os.listdir(root_dir))
        self.images = []
        for f in all_entries:
            fp = os.path.join(root_dir, f)
            if os.path.isfile(fp) and f.lower().endswith(exts):
                self.images.append(f)

        if len(self.images) == 0:
            raise RuntimeError(
                "No image files found in {} (found {} entries).".format(
                    root_dir, len(all_entries)
                )
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)

        try:
            image = io.imread(img_path)
        except Exception:
            image = np.zeros((512, 512, 3), dtype=np.uint8)

        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        elif image.ndim != 3:
            image = np.zeros((512, 512, 3), dtype=np.uint8)

        if self.transform:
            out = self.transform(image=image)
            image = out["image"] if isinstance(out, dict) else out

        return img_name, image




## === cell 2
transform = A.Compose(
    [
        A.Resize(height=512, width=512, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(),
    ]
)

test_ds = TestDataset(root_dir=TEST_IMG_DIR, transform=transform)
testloader = DataLoader(
    test_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=use_cuda
)

print("Test images:", len(test_ds))




## === cell 3
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

try:
    default_weights = EfficientNet_B3_Weights.DEFAULT
except Exception:
    default_weights = None

if os.path.exists(MODEL_PATH):
    model = efficientnet_b3(num_classes=5).to(device)

    ckpt = torch.load(MODEL_PATH, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    new_state = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        new_state[nk] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print("Loaded checkpoint:", MODEL_PATH)
    print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
else:
    print("WARNING: checkpoint not found at:", MODEL_PATH)
    if default_weights is not None:
        base = efficientnet_b3(weights=default_weights)
    else:
        base = efficientnet_b3(weights=None)

    in_features = base.classifier[1].in_features
    base.classifier[1] = nn.Linear(in_features, 5)
    model = base.to(device)

    print(
        "Using torchvision ImageNet-pretrained EfficientNet-B3 backbone with new 5-class head."
    )

model.eval()




## === cell 4
names = []
predicted = []

with torch.no_grad():
    for names_batch, images_batch in testloader:
        images_batch = images_batch.to(device, non_blocking=True).float()
        logits = model(images_batch)
        preds = torch.argmax(logits, dim=1).cpu().numpy().astype(int)

        names.extend(list(names_batch))
        predicted.extend(list(preds))

print("Predicted:", len(predicted))




## === cell 5
sub = pd.DataFrame({"image_id": names, "label": predicted})

sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    sub = sample[["image_id"]].merge(sub, on="image_id", how="left")
    sub["label"] = sub["label"].fillna(0).astype(int)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Unique predicted labels:", sorted(sub["label"].unique().tolist()))
