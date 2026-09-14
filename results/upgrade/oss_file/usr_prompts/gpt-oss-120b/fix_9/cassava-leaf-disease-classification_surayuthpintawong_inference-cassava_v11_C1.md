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

0.0725

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.43535) has done: 'The code failed because it used a relative “./input/…” path that does not exist in the Kaggle runtime; the correct base directory is `/kaggle/input`. I introduced a constant `INPUT_ROOT` pointing to that location and rebuilt the checkpoint and test‑image paths with it. I also saved the submission to the Kaggle working folder (`/kaggle/working/submission.csv`). These small, targeted fixes let the script run end‑to‑end and generate a proper `.csv` submission.'
- What this solution (achieved 0.11061) has done: 'The fix filters out sub‑folders and non‑image files when building the test dataset, loads the checkpoint more robustly (handling possible nested keys), and sets `num_workers=0` to avoid multiprocessing file‑read errors. These changes unblock the DataLoader, allow the pretrained/fine‑tuned EfficientNet‑B4 model to run on all test images, and produce a correct `submission.csv` that can achieve a much higher accuracy score.'
- What this solution (achieved 0.55904) has done: 'I make the checkpoint loading more robust by searching for either “.pt” or “.pth” files with the expected base name, handling common key names (`model`, `state_dict`, `model_state_dict`) and adding a small resize transform (instead of a center‑crop) that better matches the pretrained EfficientNet input size. These adjustments keep the original architecture and inference pipeline untouched while allowing the fine‑tuned weights to be found and applied, which should raise the validation‑style accuracy from the current 0.11 toward the target 0.8788.'
- What this solution (achieved 0.12631) has done: 'I make the inference deterministic by removing the random horizontal flip from the test transform, and I add a simple test‑time augmentation that averages the model logits for the original and horizontally‑flipped version of each image. This keeps the same architecture and training weights but is expected to raise the validation‑style accuracy toward the target without introducing new randomness or changing the core logic.'
- What this solution (achieved 0.19021) has done: 'I make the checkpoint loading more tolerant by loading with `strict=False` so that any mismatched classifier weights don’t abort the load, ensuring the pretrained feature extractor is actually used instead of falling back to pure ImageNet weights. I also add a safer recursive search for the checkpoint file if the expected folder isn’t present. These minimal changes keep the model architecture and inference pipeline unchanged while likely improving accuracy toward the target.'
- What this solution (achieved 0.0725) has done: 'We make the checkpoint loading more robust: search for any file whose name starts with the expected base, strip possible “module.” prefixes from the saved state dict, and prioritize loading the full model weights (including the classifier). This ensures the fine‑tuned classifier is actually used, which should raise accuracy toward the target while keeping the rest of the pipeline unchanged.'

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
import torchvision
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage import io

warnings.filterwarnings("ignore")

INPUT_ROOT = "/kaggle/input/cassava-leaf-disease-classification"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 5)
model = model.to(device)

folder_name = "effnetmodelb4"
model_base_name = "efficientnet-b4-e10"
chk_dir = os.path.join(INPUT_ROOT, folder_name)

checkpoint_path = None
for root, _, files in os.walk(chk_dir):
    for f in files:
        if f.startswith(model_base_name) and f.lower().endswith((".pt", ".pth")):
            checkpoint_path = os.path.join(root, f)
            break
    if checkpoint_path:
        break

if checkpoint_path and os.path.isfile(checkpoint_path):
    try:
        state = torch.load(checkpoint_path, map_location=device)
        if isinstance(state, dict):
            for key in ["model", "state_dict", "model_state_dict", "weights"]:
                if key in state:
                    state = state[key]
                    break
            if isinstance(state, dict):
                new_state = {}
                for k, v in state.items():
                    new_key = k
                    if k.startswith("module."):
                        new_key = k[len("module.") :]
                    new_state[new_key] = v
                state = new_state
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")
    except Exception as e:
        print(f"Failed to load checkpoint ({e}); using ImageNet weights.")
else:
    print("Checkpoint not found; using ImageNet pretrained weights.")

model.eval()




## === cell 1
class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [
            f
            for f in sorted(os.listdir(root_dir))
            if os.path.isfile(os.path.join(root_dir, f))
            and f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = io.imread(img_path)

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return img_name, image




## === cell 2
transform = A.Compose(
    [
        A.Resize(512, 512),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(),
    ]
)

test_dir = os.path.join(INPUT_ROOT, "test_images")
if not os.path.isdir(test_dir):
    raise FileNotFoundError(f"Test images directory not found: {test_dir}")

test_dataset = TestDataset(root_dir=test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

names = []
predicted = []

with torch.no_grad():
    for batch_names, batch_imgs in test_loader:
        batch_imgs = batch_imgs.to(device)

        logits_orig = model(batch_imgs)

        batch_imgs_flipped = torch.flip(batch_imgs, dims=[3])  # flip width dimension
        logits_flip = model(batch_imgs_flipped)

        logits_avg = (logits_orig + logits_flip) / 2.0

        preds = torch.argmax(logits_avg, dim=1).cpu().numpy()
        names.extend(batch_names)
        predicted.extend(preds.tolist())

assert len(names) == len(predicted), "Mismatch between image IDs and predictions"




## === cell 3
submission = pd.DataFrame({"image_id": names, "label": predicted})
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
