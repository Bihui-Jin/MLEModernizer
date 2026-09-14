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

0.852825627077667

# 6. Current score

0.27877

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.13266) has done: 'I replace the deprecated `DataFrame.append` call with a list‑accumulation approach to build the submission DataFrame, which resolves the AttributeError on newer pandas versions. The rest of the pipeline stays unchanged, preserving the model logic and ensuring a correctly formatted CSV is written.'
- What this solution (achieved 0.26682) has done: 'We load the provided fine‑tuned ViT weights (model_path) when the timm library is available, because those weights are far more suitable for the task than the default ImageNet‑only backbone. Additionally, we add a simple test‑time augmentation (horizontal flip) during inference and combine the logits, which gives a modest boost in accuracy without altering the core model or training pipeline. These two minimal changes keep the original structure intact while moving the validation score much closer to the target.'
- What this solution (achieved 0.27877) has done: 'I add a lightweight ResNet‑18 model (pre‑trained on ImageNet) and combine its logits with the ViT logits during inference, plus an extra vertical‑flip test‑time augmentation. This keeps the original pipeline but gives a modest boost in accuracy, moving the score toward the target while preserving all existing logic.'

# 9. Code solution

## === cell 0
import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.utils import make_grid
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



## === cell 1
try:
    import timm
except Exception:
    timm = None
    print("timm not available – will use a torchvision model as fallback.")



## === cell 2
if timm:
    print("Available ViT Models (first 5):")
    print(timm.list_models("vit*")[:5])
else:
    print("timm not installed; skipping ViT model listing.")



## === cell 3
base_path = "/kaggle/input/cassava-leaf-disease-classification/"
train_path = os.path.join(base_path, "train_images")
test_path = os.path.join(base_path, "test_images")
model_path = "/kaggle/input/vitbase16224/jx_vit_base_p16_224-80ecf9dd.pth"
cassava_weights_path = (
    "/kaggle/input/cassavanewaugtp95epochs3/CassavaViT_newaug_TP95_Epochs3_LR1-75e05.pt"
)




## === cell 4
class ViTBase16(nn.Module):
    def __init__(self, n_classes, use_pretrained=True):
        super(ViTBase16, self).__init__()
        if timm:
            self.model = timm.create_model(
                "vit_base_patch16_224", pretrained=use_pretrained
            )
        else:
            backbone = models.resnet18(pretrained=use_pretrained)
            self.model = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
            self._feat_dim = backbone.fc.in_features
        if timm:
            self.model.head = nn.Linear(self.model.head.in_features, n_classes)
        else:
            self.model = nn.Sequential(self.model, nn.Linear(self._feat_dim, n_classes))

    def forward(self, x):
        return self.model(x)


class ResNetBaseline(nn.Module):
    def __init__(self, n_classes, use_pretrained=True):
        super(ResNetBaseline, self).__init__()
        backbone = models.resnet18(pretrained=use_pretrained)
        backbone.fc = nn.Linear(backbone.fc.in_features, n_classes)
        self.model = backbone

    def forward(self, x):
        return self.model(x)




## === cell 5
cassava_model = ViTBase16(n_classes=5, use_pretrained=True)

if timm and os.path.isfile(model_path):
    try:
        state_dict = torch.load(model_path, map_location=device)
        cassava_model.load_state_dict(state_dict, strict=False)
        print("Loaded fine‑tuned ViT weights from model_path.")
    except Exception as e:
        print(f"Failed to load ViT weights from model_path: {e}")

if os.path.isfile(cassava_weights_path):
    try:
        state_dict = torch.load(cassava_weights_path, map_location=device)
        cassava_model.load_state_dict(state_dict, strict=False)
        print("Loaded custom fine‑tuned weights.")
    except Exception as e:
        print(f"Failed to load custom weights: {e}")
else:
    print("Custom weight file not found – using pretrained model only.")

resnet_model = ResNetBaseline(n_classes=5, use_pretrained=True)

cassava_gpu_model = cassava_model.to(device)
resnet_gpu_model = resnet_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cassava_gpu_model.parameters(), lr=1.5e-05)



## === cell 6
test_img_names = []
for folder, subfolders, filenames in os.walk(test_path):
    for img in filenames:
        if img.lower().endswith(".jpg"):
            test_img_names.append(img)
print("Number of test images found:", len(test_img_names))




## === cell 7
class TestSet2(Dataset):
    """Cassava Disease Test Dataset"""

    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.images = sorted(
            [f for f in os.listdir(self.test_dir) if f.lower().endswith(".jpg")]
        )
        print(f"Test dataset initialized with {len(self.images)} images.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.test_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, img_name




## === cell 8
test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)



## === cell 9
testset = TestSet2(test_dir=test_path, transform=test_transform)
print(testset)



## === cell 10
test_batch_size = 1
test_loader = DataLoader(
    dataset=testset, batch_size=test_batch_size, shuffle=False, pin_memory=True
)

print(f"Test loader created with {len(test_loader)} batches.")



## === cell 11
for images, names in test_loader:
    im = make_grid(images, nrow=4)
    plt.figure(figsize=(12, 4))
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
    plt.title(f"Sample batch – first image: {names[0]}")
    plt.axis("off")
    plt.show()
    break



## === cell 12
submission_path = "submission.csv"
col_names = ["image_id", "label"]
rows = []

tic = time.time()
cassava_gpu_model.eval()
resnet_gpu_model.eval()
with torch.no_grad():
    for X_test, name in test_loader:
        X_test = X_test.to(device)

        logits_vit = cassava_gpu_model(X_test)

        logits_resnet = resnet_gpu_model(X_test)

        X_flip = torch.flip(X_test, dims=[-1])
        logits_vit_flip = cassava_gpu_model(X_flip)
        logits_resnet_flip = resnet_gpu_model(X_flip)

        X_flip_v = torch.flip(X_test, dims=[-2])
        logits_vit_flip_v = cassava_gpu_model(X_flip_v)
        logits_resnet_flip_v = resnet_gpu_model(X_flip_v)

        combined_logits = (
            logits_vit
            + logits_resnet
            + logits_vit_flip
            + logits_resnet_flip
            + logits_vit_flip_v
            + logits_resnet_flip_v
        )
        pred = torch.argmax(combined_logits, dim=1).item()
        rows.append({"image_id": name[0], "label": int(pred)})

toc = time.time() - tic
print(f"Inference completed in {toc:.2f} seconds.")

submission_df = pd.DataFrame(rows, columns=col_names)
submission_df.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
print(submission_df.head())
