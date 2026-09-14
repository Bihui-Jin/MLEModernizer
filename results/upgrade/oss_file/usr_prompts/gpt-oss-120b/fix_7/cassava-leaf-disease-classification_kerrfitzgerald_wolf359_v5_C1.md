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

No external packages required in the script and installed.

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

0.7840737382895134

# 6. Current score

0.22833

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.13416) has done: 'Implemented a robust test dataset loader that recursively collects only image files, avoiding directory‑as‑file errors. The loader now filters for JPEGs, walks subfolders when needed, and correctly returns image tensors with their filenames. This resolves the `IsADirectoryError`, allowing the inference loop to run and generate a proper `submission.csv`. No other logic was altered, preserving the original model and training setup.'
- What this solution (achieved 0.18049) has done: 'I add a fallback load for the alternative checkpoint (`model_path`) when the primary fine‑tuned weights are missing, so the model uses a more relevant pretrained state instead of plain ImageNet weights. This small change preserves the original architecture and training logic while likely raising accuracy toward the target score.'
- What this solution (achieved 0.11286) has done: 'Implemented a tolerant checkpoint loader that uses `strict=False` so backbone weights from the custom ViT checkpoint are applied even when classifier shapes differ, preserving the pretrained features. Added a lightweight test‑time augmentation during inference by averaging model logits on the original and horizontally‑flipped image, which typically yields a modest accuracy gain without altering the core model or training pipeline. These minimal adjustments keep the original architecture intact while moving the validation score closer to the target.'
- What this solution (achieved 0.22833) has done: 'I add a lightweight second model that loads the alternate pretrained checkpoint (if available) and average its predictions with the primary model during inference. This keeps the original architecture and training untouched while giving the ensemble a modest boost in accuracy, moving the validation score closer to the target without major changes. The updated cells load the secondary model, put both on the same device, and modify the inference loop to combine logits from the two models before taking the arg‑max.'

# 9. Code solution

## === cell 0
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid

import os
import time
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt



## === cell 1
try:
    import timm  # noqa: F401
except ImportError:
    try:
        import subprocess, sys

        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "../input/timm034/timm-0.3.4-py3-none-any.whl",
            ]
        )
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "timm"])
    finally:
        import timm


## === cell 2
import timm



## === cell 3
print("Available ViT Models: ")
timm.list_models("vit*")


## === cell 4
data_path = "../input/cassava-leaf-disease-classification/"
train_path = "../input/cassava-leaf-disease-classification/train_images/"
test_path = "../input/cassava-leaf-disease-classification/test_images/"
model_path = "../input/vitbase16224/jx_vit_base_p16_224-80ecf9dd.pth"
Cassava_model = (
    "../input/cassavaaugmtp99epochs2/CassavaViT_Augm_TP99_Epochs2_LR1-75e05.pt"
)




## === cell 5
class ViTBase16(nn.Module):
    def __init__(self, n_classes, pretrained=False):
        super(ViTBase16, self).__init__()
        self.model = timm.create_model("vit_base_patch16_224", pretrained=pretrained)
        if pretrained and os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location="cpu")
                self.model.load_state_dict(state_dict, strict=False)
                print("Custom checkpoint loaded with strict=False.")
            except Exception as e:
                print(f"Warning: failed to load custom checkpoint: {e}")
        self.model.head = nn.Linear(self.model.head.in_features, n_classes)

    def forward(self, x):
        return self.model(x)




## === cell 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cassava_model = ViTBase16(n_classes=5, pretrained=True)
if os.path.exists(Cassava_model):
    try:
        cassava_model.load_state_dict(torch.load(Cassava_model, map_location=device))
        print("Loaded fine‑tuned Cassava model weights.")
    except Exception as e:
        print(f"Warning: failed to load fine‑tuned weights: {e}")
        if os.path.exists(model_path):
            try:
                cassava_model.load_state_dict(
                    torch.load(model_path, map_location=device), strict=False
                )
                print("Loaded alternative pretrained checkpoint as fallback.")
            except Exception as e2:
                print(f"Warning: failed to load alternative checkpoint: {e2}")
else:
    print("Fine‑tuned checkpoint not found; attempting alternative checkpoint.")
    if os.path.exists(model_path):
        try:
            cassava_model.load_state_dict(
                torch.load(model_path, map_location=device), strict=False
            )
            print("Loaded alternative pretrained checkpoint.")
        except Exception as e2:
            print(f"Warning: failed to load alternative checkpoint: {e2}")
    else:
        print("Alternative checkpoint not found; using ImageNet‑pretrained model.")
cassava_gpu_model = cassava_model.to(device)

cassava_model_2 = ViTBase16(n_classes=5, pretrained=False)
if os.path.exists(model_path):
    try:
        cassava_model_2.load_state_dict(
            torch.load(model_path, map_location=device), strict=False
        )
        print("Loaded secondary model from custom checkpoint.")
    except Exception as e:
        print(f"Warning: failed to load secondary checkpoint: {e}")
else:
    print("Secondary checkpoint not found; secondary model will use ImageNet weights.")
cassava_gpu_model_2 = cassava_model_2.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cassava_gpu_model.parameters(), lr=1.5e-05)


## === cell 7
test_img_names = []
for folder, subfolders, filenames in os.walk(test_path):
    for img in filenames:
        if img.lower().endswith(".jpg"):
            test_img_names.append(img)
print("Testing Images:", len(test_img_names))




## === cell 8
class TestSet2(Dataset):
    """Cassava Disease Test Dataset"""

    def __init__(self, root_dir, test_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.test_dir = test_dir
        self.transform = transform

        self.image_files = [
            os.path.join(self.test_dir, f)
            for f in os.listdir(self.test_dir)
            if f.lower().endswith(".jpg")
            and os.path.isfile(os.path.join(self.test_dir, f))
        ]

        if not self.image_files:
            for subdir, _, files in os.walk(self.test_dir):
                for f in files:
                    if f.lower().endswith(".jpg"):
                        self.image_files.append(os.path.join(subdir, f))

        print(f"Cassava Disease Test Dataset Length = {len(self.image_files)}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_name = os.path.basename(img_path)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, img_name




## === cell 9
test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


## === cell 10
testset = TestSet2(root_dir="", test_dir=test_path, transform=test_transform)
print(testset)


## === cell 11
test_batch_size = 1
test_loader = DataLoader(
    dataset=testset, batch_size=test_batch_size, shuffle=False, pin_memory=True
)


## === cell 12
print(test_loader)

for images, names in test_loader:
    break

im = make_grid(images, nrow=4)
print(names)
plt.figure(figsize=(12, 4))
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))


## === cell 13
col_names = ["image_id", "label"]
submission_df = pd.DataFrame(columns=col_names)
tic = time.time()
cassava_gpu_model.eval()
cassava_gpu_model_2.eval()
with torch.no_grad():
    for b, (X_test, name) in enumerate(test_loader):
        X_test = X_test.to(device)

        logits1 = cassava_gpu_model(X_test)

        logits2 = cassava_gpu_model_2(X_test)

        X_flip = torch.flip(X_test, dims=[-1])
        logits1_flip = cassava_gpu_model(X_flip)
        logits2_flip = cassava_gpu_model_2(X_flip)

        avg_logits = (logits1 + logits2 + logits1_flip + logits2_flip) / 4.0
        predicted = torch.argmax(avg_logits, dim=1).item()

        submission_df.loc[len(submission_df)] = {
            "image_id": name[0],
            "label": int(predicted),
        }
toc = time.time() - tic
print("Time for model inference:", toc)
submission_df.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")
print(submission_df.head())
