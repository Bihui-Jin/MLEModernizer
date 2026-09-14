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

0.870806890299184

# 6. Current score

0.17115

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.07399) has done: 'I fix the Albumentations pipeline so it returns a proper tensor and doesn’t crash on `available_keys`, while keeping the same augmentation intent (center crop + optional horizontal flip + normalize). I also make the weights loading robust by falling back to ImageNet pretrained EfficientNet-B4 if the external `.pt` file isn’t present, so the notebook always runs end-to-end and yields a valid submission. Finally, I rebuild predictions strictly in the order of `sample_submission.csv` to guarantee the submission length and ordering match Kaggle’s expected test set, eliminating the “same length as the answers” error. These changes are minimal and primarily correctness/stability oriented, with a sensible fallback that should also improve accuracy versus random weights.'
- What this solution (achieved 0.12855) has done: 'Your score (0.07399) is far below the target (0.8708), so the biggest likely issue is not “tuning” but that inference-time preprocessing doesn’t match what EfficientNet expects, making predictions almost random. I keep the same model and inference loop, but switch the transform from a hard center-crop to the standard EfficientNet resize+center-crop pipeline (still deterministic and minimal) and ensure RGB conversion for any non-RGB JPEGs. I also use the exact EfficientNet-B4 ImageNet normalization values from torchvision weights to better align the fallback backbone and any custom weights trained with the usual pipeline. These are small, evaluation-semantic-preserving changes that typically move accuracy dramatically upward without changing the core approach.'
- What this solution (achieved 0.13901) has done: 'Your current score is far below the target, so the most likely issue is a preprocessing mismatch that makes the loaded (or fallback) EfficientNet behave almost randomly. I keep the same model and inference loop, but make the transform match the canonical EfficientNet-B4 inference pipeline: resize the shorter side to 384 then center-crop 380 (same model input size you already use), and remove test-time random horizontal flip which injects unnecessary randomness and typically hurts accuracy for a single-pass submission. I also force RGB conversion for any non-RGB images to avoid silent channel-order/shape issues. These are minimal changes that preserve the overall approach while materially improving accuracy toward the target.'
- What this solution (achieved 0.17115) has done: 'Your score is far below the target, so we should focus on a minimal correctness fix that can materially improve accuracy without changing the model or training approach. The biggest likely remaining issue is that `skimage.io.imread` yields RGB, but `albumentations`/OpenCV-style transforms are typically BGR-based; this color-channel mismatch can make an EfficientNet behave close to random even with good weights. I minimally switch image loading to OpenCV (`cv2.imread`) and convert BGR→RGB explicitly (keeping the same resize/crop/normalize/tensor steps), which preserves your core inference logic while aligning preprocessing with what torchvision EfficientNet expects. I also bump `num_workers` slightly (safe) but keep everything deterministic and single-pass.'

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
import cv2
import torchvision

warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

model_full_name = "efficientnet-b4-e10"
model_name = "efficientnet-b4"
folder_name = "effnetmodelv18"

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TEST_DIR = os.path.join(DATA_ROOT, "test_images")
WEIGHTS_PATH = os.path.join("/kaggle/input", folder_name, model_full_name + ".pt")

SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

print("DATA_ROOT:", DATA_ROOT)
print("TEST_DIR exists:", os.path.isdir(TEST_DIR))
print("WEIGHTS_PATH:", WEIGHTS_PATH, "| exists:", os.path.exists(WEIGHTS_PATH))
print("SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH, "| exists:", os.path.exists(SAMPLE_SUB_PATH))




## === cell 1
def to_tensor_fn(image, **kwargs):
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    image = np.ascontiguousarray(image.transpose(2, 0, 1))
    return torch.from_numpy(image)




## === cell 2
class TestDataset(Dataset):
    def __init__(self, root_dir, image_ids=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        if image_ids is not None:
            self.images = list(image_ids)
        else:
            self.images = sorted(
                [f for f in os.listdir(root_dir) if f.lower().endswith(".jpg")]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise IOError("Failed to read image: %s" % img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            out = self.transform(image=image)
            image = out["image"]

        return img_name, image




## === cell 3
weights_enum = torchvision.models.EfficientNet_B4_Weights.IMAGENET1K_V1
mean = tuple(weights_enum.transforms().mean)
std = tuple(weights_enum.transforms().std)

transform = A.Compose(
    [
        A.SmallestMaxSize(max_size=384, interpolation=cv2.INTER_LINEAR),
        A.CenterCrop(width=380, height=380),
        A.Normalize(
            mean=mean,
            std=std,
            max_pixel_value=255.0,
            p=1.0,
        ),
        A.Lambda(image=to_tensor_fn),
    ]
)

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_image_ids = sample_sub["image_id"].tolist()

test_ds = TestDataset(root_dir=TEST_DIR, image_ids=test_image_ids, transform=transform)

testloader = DataLoader(
    test_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=use_cuda
)

print("Test dataset size:", len(test_ds))
print("Sample submission size:", len(sample_sub))



## === cell 4
model = torchvision.models.efficientnet_b4(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 5)
model = model.to(device)

loaded_custom = False
if os.path.exists(WEIGHTS_PATH):
    ckpt = torch.load(WEIGHTS_PATH, map_location="cpu")
    if (
        isinstance(ckpt, dict)
        and "state_dict" in ckpt
        and isinstance(ckpt["state_dict"], dict)
    ):
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        state_dict = None

    if state_dict is not None:
        new_state = {}
        for k, v in state_dict.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            new_state[nk] = v
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        loaded_custom = True
        print("Loaded custom weights.")
        print("Missing keys (truncated):", missing[:10], "count:", len(missing))
        print(
            "Unexpected keys (truncated):", unexpected[:10], "count:", len(unexpected)
        )

if not loaded_custom:
    print(
        "Custom weights not found/loaded; using ImageNet-pretrained backbone as fallback."
    )
    pretrained = torchvision.models.efficientnet_b4(weights=weights_enum)
    model.features.load_state_dict(pretrained.features.state_dict(), strict=True)

model.eval()



## === cell 5
names = []
predicted = []

with torch.no_grad():
    for names_batch, images_batch in testloader:
        images_batch = images_batch.to(device).float()
        logits = model(images_batch)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int)

        names.extend(list(names_batch))
        predicted.extend(list(preds))

print(
    "Predictions made:", len(predicted), "Unique labels:", sorted(set(predicted))[:10]
)



## === cell 6
pred_map = dict(zip(names, predicted))
ordered_preds = [int(pred_map[iid]) for iid in test_image_ids]

result = pd.DataFrame({"image_id": test_image_ids, "label": ordered_preds})
result.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", result.shape)
print(result.head())
print("Submission length matches sample:", len(result) == len(sample_sub))
