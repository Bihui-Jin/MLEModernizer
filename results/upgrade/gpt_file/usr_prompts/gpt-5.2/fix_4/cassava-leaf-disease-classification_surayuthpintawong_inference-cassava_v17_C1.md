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

0.8750377757630704

# 6. Current score

0.09342

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I remove the notebook-only `%cd` magics and fix the missing `efficientnet_pytorch` dependency by switching to the built-in `torchvision.models.efficientnet_b4` while keeping the same EfficientNet-B4 architecture and 5-class head. I also fix the Albumentations API break (`A.Flip` no longer exists) by using `A.HorizontalFlip`, and correct the custom `ToTensor` so it works with Albumentations’ dict interface and returns a proper float tensor. Finally, I make model weight loading robust (CPU/GPU map_location and common checkpoint formats) and ensure we always write a valid `submission.csv` aligned to `sample_submission.csv` image order.'
- What this solution (achieved 0.13677) has done: 'I fix the Albumentations runtime error by making the custom tensor transform compatible with Albumentations v2 (it must accept/return a dict and expose `available_keys`). Then I fix the missing-weights crash by searching for the checkpoint in common Kaggle input locations (including the dataset folder you actually have) and only failing with a clear message if nothing is found. Finally, I ensure the dataloader/model cells run in order and always write a valid `submission.csv` aligned to `sample_submission.csv`. These changes are correctness/stability fixes; they don’t change the model architecture or inference semantics beyond making the pipeline actually run.'
- What this solution (achieved 0.09342) has done: 'I fix two execution blockers: (1) the test image directory contains a nested `test_images/` folder, so the dataset currently tries to “read a folder as an image”; I robustly collect only actual image files (jpg/png/jpeg) recursively and keep names consistent with `sample_submission.csv`. (2) your run crashes when the expected checkpoint isn’t present; to ensure an end-to-end runnable pipeline that produces a valid `submission.csv`, I fall back to ImageNet pretrained EfficientNet-B4 weights if the competition checkpoint cannot be found (this should also improve the score versus random initialization, moving it toward your target). All other logic (EfficientNet-B4 head with 5 classes, transforms, inference, submission alignment) is kept the same.'

# 9. Code solution

## === cell 0
from __future__ import print_function, division

import os
import glob
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from skimage import io

import albumentations as A

warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TEST_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

model_full_name = "efficientnet-b4-e0"
model_name = "efficientnet-b4"
folder_name = "effnetmodelb44"
WEIGHTS_PATH = os.path.join("/kaggle/input", folder_name, model_full_name + ".pt")

print("device:", device)
print(
    "TEST_DIR exists:",
    os.path.isdir(TEST_DIR),
    "n_entries:",
    (len(os.listdir(TEST_DIR)) if os.path.isdir(TEST_DIR) else 0),
)
print("SAMPLE_SUB_PATH exists:", os.path.exists(SAMPLE_SUB_PATH))
print("Initial WEIGHTS_PATH:", WEIGHTS_PATH)




## === cell 1
class ToTensorV2Compatible(object):
    available_keys = {"image"}

    def __call__(self, force_apply=False, **data):
        image = data.get("image", None)
        if image is None:
            return data
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.shape[-1] == 4:
            image = image[:, :, :3]

        image = image.transpose(2, 0, 1)
        image = np.ascontiguousarray(image)
        data["image"] = torch.from_numpy(image)
        return data




## === cell 2
def list_image_files(root_dir):
    """
    Bugfix: Kaggle dataset sometimes has nested `test_images/test_images/`.
    If we just os.listdir(root_dir), we may include subfolders and crash when reading them as images.
    We only return actual image files (recursively), but keep only the basename for submission alignment.
    """
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(root_dir, "**", "*" + ext), recursive=True))
        paths.extend(
            glob.glob(os.path.join(root_dir, "**", "*" + ext.upper()), recursive=True)
        )
    paths = sorted(set(paths))
    return paths


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = list_image_files(root_dir)
        if len(self.image_paths) == 0:
            entries = [os.path.join(root_dir, x) for x in sorted(os.listdir(root_dir))]
            self.image_paths = [
                p
                for p in entries
                if os.path.isfile(p)
                and p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ]

        self.images = [os.path.basename(p) for p in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        img_name = self.images[idx]
        image = io.imread(img_path)

        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.shape[-1] == 4:
            image = image[:, :, :3]

        if self.transform:
            out = self.transform(image=image)
            image = out["image"]

        return img_name, image




## === cell 3
transform = A.Compose(
    [
        A.CenterCrop(width=512, height=512),
        A.HorizontalFlip(p=0.5),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2Compatible(),
    ]
)

test_ds = TestDataset(root_dir=TEST_DIR, transform=transform)
testloader = DataLoader(
    test_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=use_cuda
)

print("Test dataset size:", len(test_ds))
print("First 5 test images (basenames):", test_ds.images[:5])



## === cell 4
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights


def resolve_weights_path(initial_path):
    if os.path.exists(initial_path):
        return initial_path

    candidates = []
    candidates += glob.glob(os.path.join("/kaggle/input", folder_name, "*.pt"))
    candidates += glob.glob(os.path.join(DATA_ROOT, "**", "*.pt"), recursive=True)
    candidates += glob.glob(os.path.join("/kaggle/input", folder_name, "*.pth"))
    candidates += glob.glob(os.path.join(DATA_ROOT, "**", "*.pth"), recursive=True)

    base = os.path.basename(initial_path)
    for c in candidates:
        if os.path.basename(c) == base:
            return c

    candidates = sorted(set(candidates))
    return candidates[0] if candidates else None


resolved = resolve_weights_path(WEIGHTS_PATH)
use_imagenet_fallback = resolved is None

if use_imagenet_fallback:
    print(
        "WARNING: Competition checkpoint not found at/under expected locations.\n"
        "Falling back to torchvision ImageNet pretrained EfficientNet-B4 weights to ensure an end-to-end run."
    )
    model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
else:
    WEIGHTS_PATH = resolved
    print("Using WEIGHTS_PATH:", WEIGHTS_PATH)
    model = efficientnet_b4(weights=None)

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 5)
model = model.to(device)

if not use_imagenet_fallback:
    ckpt = torch.load(WEIGHTS_PATH, map_location="cpu")
    state_dict = None
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    clean_sd = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        clean_sd[nk] = v

    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    print(
        "Loaded weights. Missing keys:",
        len(missing),
        "Unexpected keys:",
        len(unexpected),
    )

model.eval()
model = model.to(device)



## === cell 5
names = []
predicted = []

with torch.no_grad():
    for names_batch, images_batch in testloader:
        images_batch = images_batch.to(device, non_blocking=True).float()
        output = model(images_batch)
        pred = torch.argmax(output, dim=1).detach().cpu().numpy()
        names.extend(list(names_batch))
        predicted.extend(pred.tolist())

print("Preds:", len(predicted), "Names:", len(names))



## === cell 6
sample = pd.read_csv(SAMPLE_SUB_PATH)

pred_map = dict(zip(names, predicted))
sample["label"] = sample["image_id"].map(pred_map)

sample["label"] = sample["label"].fillna(0).astype(int)

sample.to_csv("submission.csv", index=False)

print(sample.head())
print("Wrote submission.csv with shape:", sample.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))
print("Unique predicted labels:", sorted(sample["label"].unique().tolist()))
