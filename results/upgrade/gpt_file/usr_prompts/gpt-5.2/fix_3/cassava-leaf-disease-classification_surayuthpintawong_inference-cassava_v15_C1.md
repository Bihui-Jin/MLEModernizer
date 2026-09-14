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

0.8788153520701119

# 6. Current score

0.08707

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.08707) has done: 'I fix the missing checkpoint issue by removing the dependency on a non-existent `../input/effnetmodelb43/...pt` and instead load an EfficientNet-B4 backbone using torchvision’s built-in pretrained weights (same architecture, just a different weight source so it can run). I also make the test image ordering deterministic and match `sample_submission.csv` ordering to avoid row misalignment, which can otherwise tank accuracy. Finally, I make the Albumentations → tensor conversion robust (ensuring contiguous float tensors) and guarantee a valid `submission.csv` is always written end-to-end.'

# 9. Code solution

## === cell 0
from __future__ import print_function, division

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from skimage import io

import albumentations as A
import torchvision

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

TEST_DIR = "../input/cassava-leaf-disease-classification/test_images/"
SAMPLE_SUB_PATH = "../input/cassava-leaf-disease-classification/sample_submission.csv"

folder_name = "effnetmodelb43"
model_full_name = "efficientnet-b4-e10"
model_name = "efficientnet-b4"
WEIGHTS_PATH = "../input/" + folder_name + "/" + model_full_name + ".pt"

if not os.path.isdir(TEST_DIR):
    alt = "/kaggle/input/cassava-leaf-disease-classification/test_images"
    if os.path.isdir(alt):
        TEST_DIR = alt

if not os.path.exists(SAMPLE_SUB_PATH):
    alt = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
    if os.path.exists(alt):
        SAMPLE_SUB_PATH = alt




## === cell 1
class EfficientNet(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

    @classmethod
    def from_name(cls, name, num_classes=5, use_pretrained=True):
        if name != "efficientnet-b4":
            raise ValueError(
                "Only efficientnet-b4 is supported in this environment fix."
            )
        weights = (
            torchvision.models.EfficientNet_B4_Weights.DEFAULT
            if use_pretrained
            else None
        )
        model = torchvision.models.efficientnet_b4(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return cls(model)


def _clean_state_dict_keys(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("backbone."):
            nk = nk[len("backbone.") :]
        cleaned[nk] = v
    return cleaned


def _try_load_external_checkpoint(model, weights_path, device):
    """
    Optional: try to load the originally-referenced checkpoint if it exists.
    Keeps core logic but avoids hard failure when file is missing.
    """
    if weights_path is None or (not os.path.exists(weights_path)):
        return False, "Checkpoint not found; using torchvision pretrained weights."
    ckpt = torch.load(weights_path, map_location=device)
    state_dict = _clean_state_dict_keys(ckpt)
    try:
        model.backbone.load_state_dict(state_dict, strict=False)
        return True, "Loaded checkpoint into model.backbone (strict=False)."
    except Exception:
        try:
            model.load_state_dict(state_dict, strict=False)
            return True, "Loaded checkpoint into model (strict=False)."
        except Exception as e:
            return (
                False,
                "Failed to load checkpoint; using torchvision pretrained weights. Error: %s"
                % str(e),
            )




## === cell 2
class ToTensor(object):
    def __call__(self, image, force_apply=True):
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        if image.shape[-1] == 4:
            image = image[:, :, :3]
        output = np.ascontiguousarray(image.transpose((2, 0, 1)))
        return torch.from_numpy(output)


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None, images=None):
        self.root_dir = root_dir
        self.transform = transform
        if images is None:
            self.images = sorted(os.listdir(root_dir))
        else:
            self.images = list(images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = io.imread(img_path)

        if self.transform:
            out = self.transform(image=image)
            image = out["image"] if isinstance(out, dict) and "image" in out else out

        return img_name, image




## === cell 3
transform = A.Compose(
    [
        A.CenterCrop(width=512, height=512),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)


class AlbumentationsWithTensor(object):
    def __init__(self, a_transform):
        self.a_transform = a_transform
        self.to_tensor = ToTensor()

    def __call__(self, image):
        out = self.a_transform(image=image)
        img = out["image"]
        return self.to_tensor(img)


sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_ids = sample_sub["image_id"].tolist()

test_image = TestDataset(
    root_dir=TEST_DIR, transform=AlbumentationsWithTensor(transform), images=test_ids
)
testloader = DataLoader(
    test_image, batch_size=16, shuffle=False, num_workers=2, pin_memory=use_cuda
)

model = EfficientNet.from_name(model_name, num_classes=5, use_pretrained=True).to(
    device
)

loaded, msg = _try_load_external_checkpoint(model, WEIGHTS_PATH, device)
print(msg)

model.eval()

names = []
predicted = []
with torch.no_grad():
    for names_batch, images_batch in testloader:
        images_batch = images_batch.to(device).float()
        output = model(images_batch)
        output = torch.max(output, 1)[1].cpu().numpy()
        names.extend(list(names_batch))
        predicted.extend(output.tolist())




## === cell 4
pred_map = dict(zip(names, predicted))
final_pred = [int(pred_map[iid]) for iid in test_ids]

result = pd.DataFrame({"image_id": test_ids, "label": final_pred})
result.to_csv("submission.csv", index=False)

print(result.head())
print("Wrote submission.csv with {} rows".format(len(result)))
assert len(result) == len(
    sample_sub
), "Submission row count mismatch vs sample_submission.csv"
assert list(result.columns) == ["image_id", "label"], "Submission columns mismatch"
