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

0.8806

# 6. Current score

0.10762

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'Diagnosis: The crash comes from an Albumentations v2 API change: `A.RandomResizedCrop(256,256)` is no longer interpreted as `(height, width)` and instead triggers schema validation expecting tuple-valued parameters (`size`/`scale`). This makes the transform construction fail before any data loading/training.  
Patch summary: Update only the `RandomResizedCrop` call in cell 5 to use the v2 signature with an explicit `size=(image_size, image_size)`, preserving the same crop output resolution and leaving all other augmentations unchanged.  
Updated cells: Only cell 5 is modified.  
Compatibility notes for cell k+1: `transform` remains an `A.Compose` producing tensors via `ToTensorV2`, so dataset/dataloader usage in the next cell stays identical.  
Assumptions: Albumentations 2.0.8 is installed (as listed) and `image_size` is defined earlier (cell 1).'
- What this solution (achieved 0.10762) has done: 'Your current score is far below the target, so we should improve accuracy with minimal, low-risk edits while keeping the same model and inference approach. The biggest issue hurting performance is that you’re applying *training augmentations* (random resized crop + flips/transpose/brightness) during test-time inference, which makes predictions unstable and usually much worse; we switch test-time transforms to deterministic resize+center-crop with the same normalization. We also ensure the loaded model is compatible with PyTorch 2.6’s safer `torch.load` behavior by loading as a `state_dict` when needed, without changing architecture. Finally, we keep your multi-pass inference loop intact but make it deterministic (since transforms are deterministic, multiple passes won’t add noise).'
- What this solution (achieved 0.61099) has done: 'You’re far below the target (0.10762 vs 0.8806), so we should make a small change that can realistically recover a large chunk of lost accuracy without changing your model. The most likely cause is a train/inference preprocessing mismatch: this competition’s strong baselines nearly always used `RandomResizedCrop` during training, and swapping it to `CenterCrop` at test time can severely hurt because the model learned scale/position statistics from random crops. I keep your architecture, softmax/averaging inference loop, and I/O identical, and only switch the test transform to a deterministic `Resize -> RandomResizedCrop(..., p=1.0)` (using Albumentations v2 signature) so inference matches what the checkpoint likely expects. This should increase accuracy substantially and move closer to the target while remaining stable/deterministic.'
- What this solution (achieved 0.10762) has done: 'We keep your exact model and multi-pass inference logic, but fix a key source of accuracy loss: your current test-time transform uses `RandomResizedCrop`, which introduces randomness and effectively performs test-time augmentation without calibration, often hurting accuracy a lot. To move the score up toward the target, we make the test transform deterministic (Resize + CenterCrop) while keeping the same normalization and tensor conversion, so predictions are stable and aligned with typical ResNet preprocessing. We also set seeds and deterministic flags so the run is reproducible and the 10 inference passes become identical (no semantic change, just stability). Everything else (paths, architecture, softmax averaging, submission writing) stays the same.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("bmh")
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2



## === cell 1
OUTPUT_DIR = "./"
num_classes = 5
image_size = 256
batch_size = 32



## === cell 2
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device



## === cell 3
candidate_paths = [
    "../input/pretrained-model/sgd_aug.pt",
    "/kaggle/input/pretrained-model/sgd_aug.pt",
    "/kaggle/data/pretrained-model/sgd_aug.pt",
    "/kaggle/input/cassava-leaf-disease-classification/sgd_aug.pt",
    "/kaggle/input/cassava-leaf-disease-classification/pretrained-model/sgd_aug.pt",
    "/kaggle/data/cassava-leaf-disease-classification/sgd_aug.pt",
    "/kaggle/data/cassava-leaf-disease-classification/pretrained-model/sgd_aug.pt",
]

model_path = next((p for p in candidate_paths if os.path.exists(p)), None)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)

if model_path is not None:
    obj = None
    try:
        obj = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(model_path, map_location="cpu")
    except Exception:
        obj = torch.load(model_path, map_location="cpu")

    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            state = obj["state_dict"]
        else:
            state = obj
        new_state = {}
        for k, v in state.items():
            nk = k[len("module.") :] if k.startswith("module.") else k
            new_state[nk] = v
        missing, unexpected = model.load_state_dict(new_state, strict=False)
    else:
        model = obj

model = model.to(device)




## === cell 4
class CassavaDataset(Dataset):
    def __init__(self, data_dir, ids, labels, transform=None):
        self.data_dir = data_dir
        self.ids = ids
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.data_dir, self.ids[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        label = self.labels[idx]

        return (image, label)




## === cell 5
transform = A.Compose(
    [
        A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR),
        A.CenterCrop(height=image_size, width=image_size, p=1.0),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(p=1.0),
    ]
)



## === cell 6
test_df = pd.read_csv(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)
test_dir = "../input/cassava-leaf-disease-classification/test_images"
ids = test_df["image_id"].values
labels = test_df["label"].values
test_df



## === cell 7
test_dataset = CassavaDataset(test_dir, ids, labels, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)



## === cell 8
softmax = nn.Softmax(dim=1)



## === cell 9
num_inferences = 10
inferences = []

model.eval()
with torch.no_grad():
    for i in range(num_inferences):
        inf = []
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            outputs = softmax(model(inputs))
            outputs = outputs.cpu().numpy()
            inf += list(outputs)
        inferences.append(np.array(inf))



## === cell 10
preds = np.zeros((inferences[0].shape))
for inf in inferences:
    preds += inf
preds = preds / num_inferences
preds = list(np.argmax(preds, axis=1))



## === cell 11
test_df["label"] = preds
test_df.to_csv(OUTPUT_DIR + "submission.csv", index=False)



## === cell 12
pd.read_csv(OUTPUT_DIR + "submission.csv")
