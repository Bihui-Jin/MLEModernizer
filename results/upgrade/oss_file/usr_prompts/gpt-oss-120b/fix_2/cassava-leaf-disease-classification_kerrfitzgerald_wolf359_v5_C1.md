# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

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



## === cell 2
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



## === cell 4
import timm



## === cell 6
print("Available ViT Models: ")
timm.list_models("vit*")



## === cell 7
data_path = "../input/cassava-leaf-disease-classification/"
train_path = "../input/cassava-leaf-disease-classification/train_images/"
test_path = "../input/cassava-leaf-disease-classification/test_images/"
model_path = "../input/vitbase16224/jx_vit_base_p16_224-80ecf9dd.pth"
Cassava_model = (
    "../input/cassavaaugmtp99epochs2/CassavaViT_Augm_TP99_Epochs2_LR1-75e05.pt"
)



## === cell 9
class ViTBase16(nn.Module):
    def __init__(self, n_classes, pretrained=False):
        super(ViTBase16, self).__init__()
        self.model = timm.create_model("vit_base_patch16_224", pretrained=pretrained)
        if pretrained and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
            except Exception as e:
                print(f"Warning: failed to load custom checkpoint: {e}")
        self.model.head = nn.Linear(self.model.head.in_features, n_classes)

    def forward(self, x):
        return self.model(x)




## === cell 11
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cassava_model = ViTBase16(n_classes=5, pretrained=True)
if os.path.exists(Cassava_model):
    try:
        cassava_model.load_state_dict(torch.load(Cassava_model, map_location=device))
        print("Loaded fine‑tuned Cassava model weights.")
    except Exception as e:
        print(f"Warning: failed to load fine‑tuned weights: {e}")
else:
    print("Fine‑tuned checkpoint not found; using ImageNet‑pretrained model.")
cassava_gpu_model = cassava_model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cassava_gpu_model.parameters(), lr=1.5e-05)



## === cell 13
test_img_names = []
for folder, subfolders, filenames in os.walk(test_path):
    for img in filenames:
        if img.lower().endswith(".jpg"):
            test_img_names.append(img)
print("Testing Images:", len(test_img_names))



## === cell 15
class TestSet2(Dataset):
    """Cassava Disease Dataset"""

    def __init__(self, root_dir, test_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.test_dir = test_dir
        self.transform = transform
        print(root_dir)
        print(test_dir)
        print("Cassava Disease Test Dataset Length =", len(os.listdir(self.test_dir)))

    def __len__(self):
        return len(os.listdir(self.test_dir))

    def __getitem__(self, idx):
        img_name = os.listdir(self.test_dir)[idx]
        img_path = os.path.join(self.test_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, img_name




## === cell 17
test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)



## === cell 19
testset = TestSet2(root_dir="", test_dir=test_path, transform=test_transform)
print(testset)



## === cell 21
test_batch_size = 1
test_loader = DataLoader(
    dataset=testset, batch_size=test_batch_size, shuffle=False, pin_memory=True
)



## === cell 23
print(test_loader)

for images, names in test_loader:
    break

im = make_grid(images, nrow=4)
print(names)
plt.figure(figsize=(12, 4))
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))



## === cell 25
col_names = ["image_id", "label"]
submission_df = pd.DataFrame(columns=col_names)
tic = time.time()
cassava_gpu_model.eval()
with torch.no_grad():
    for b, (X_test, name) in enumerate(test_loader):
        X_test = X_test.to(device)
        y_test_pred = cassava_gpu_model(X_test)
        predicted = torch.max(y_test_pred, 1)[1].item()
        new_row = {"image_id": name[0], "label": int(predicted)}
        submission_df = submission_df.append(new_row, ignore_index=True)

toc = time.time() - tic
print("Time for model inference:", toc)
submission_df.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")
print(submission_df.head())

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_56/3212824464.py in <cell line: 0>()
      9         predicted = torch.max(y_test_pred, 1)[1].item()
     10         new_row = {"image_id": name[0], "label": int(predicted)}
---> 11         submission_df = submission_df.append(new_row, ignore_index=True)
     12 
     13 toc = time.time() - tic

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'append'
