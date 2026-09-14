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

0.8354487760652766

# 6. Current score

0.12108

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.12444) has done: 'The test dataset loader was trying to open a sub‑directory as an image, causing an IsADirectoryError.  
I updated **TestSet2** to pre‑collect only “.jpg” files, use that list for length and indexing, and skip any non‑image entries. This fixes the loading error, enables inference, and ensures a correctly‑formatted `submission.csv` is written.'
- What this solution (achieved 0.10613) has done: 'I adjust the model loading logic so that if the Cassava‑specific checkpoint (`CassavaViT_Augm_TP98_Epochs1_LR1-75e05.pt`) exists it be used instead of only the generic ImageNet weights. This keeps the architecture unchanged while likely improving the predictions and moving the validation accuracy closer to the target.'
- What this solution (achieved 0.10501) has done: 'I fix the checkpoint file paths so that the fine‑tuned Cassava ViT weights are correctly loaded (or fall back to the generic ViT checkpoint). Using the proper absolute paths under /kaggle/input ensures the model starts from a well‑trained state, which should raise the validation accuracy toward the target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.14425) has done: 'We adjust the model initialization so the classification head is set **before** loading any checkpoint. This prevents the fine‑tuned head from being overwritten by the generic ImageNet weights, which should raise prediction accuracy toward the target while keeping the original architecture unchanged.'
- What this solution (achieved 0.12668) has done: 'I add a lightweight test‑time augmentation (horizontal flip) during inference and combine the logits of the original and flipped images before taking the argmax. This keeps the model architecture unchanged while often improving classification accuracy, moving the score toward the target. The change is limited to the inference loop, preserving all other logic.'
- What this solution (achieved 0.10314) has done: 'I make the checkpoint discovery more robust and load the weights with `strict=False` so the fine‑tuned Cassava ViT parameters are correctly applied even if the classification head size differs. This ensures the model uses the specialised weights instead of falling back to the generic ImageNet checkpoint, which should raise the validation accuracy toward the target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.39948) has done: 'I adjust the model initialization so that the fine‑tuned Cassava checkpoint is loaded correctly and keeps its original classification head. The ViT is created without ImageNet weights, then the checkpoint (if present) is loaded with strict matching; otherwise we fall back to the generic pretrained model and replace the head. This change restores the proper learned weights, which should raise the validation accuracy toward the target while leaving the rest of the pipeline untouched.'
- What this solution (achieved 0.12108) has done: 'I adjust the checkpoint loading to use `strict=False` so that any mismatched keys (e.g., a different classification head) do not prevent loading the fine‑tuned weights, and I guarantee that the model’s final linear layer has exactly five output units. During inference I enrich the test‑time augmentation by also adding a 90° rotation, summing the logits from the original, horizontally‑flipped and rotated images before taking the argmax. These small, targeted changes keep the original architecture unchanged while expectedly raising the validation accuracy toward the target score.'

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
    pass




## === cell 2
import timm




## === cell 3
_base_input = "/kaggle/input"

data_path = "/kaggle/input/cassava-leaf-disease-classification/"
train_path = os.path.join(data_path, "train_images")
test_path = os.path.join(data_path, "test_images")

generic_candidates = [
    os.path.join(_base_input, "vitbase16224", "jx_vit_base_p16_224-80ecf9dd.pth"),
    "../input/vitbase16224/jx_vit_base_p16_224-80ecf9dd.pth",
]
model_path = next((p for p in generic_candidates if os.path.exists(p)), None)

cassava_candidates = [
    os.path.join(
        _base_input,
        "cassavaaugmtp98epochs1lr175",
        "CassavaViT_Augm_TP98_Epochs1_LR1-75e05.pt",
    ),
    "../input/cassavaaugmtp98epochs1lr175/CassavaViT_Augm_TP98_Epochs1_LR1-75e05.pt",
]
Cassava_model = next((p for p in cassava_candidates if os.path.exists(p)), None)

print("ViT generic checkpoint:", model_path, "exists:", bool(model_path))
print(
    "Cassava fine‑tuned checkpoint:",
    Cassava_model,
    "exists:",
    bool(Cassava_model),
)




## === cell 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")




## === cell 5
class ViTBase16(nn.Module):
    def __init__(self, n_classes, pretrained=False):
        super(ViTBase16, self).__init__()
        self.model = timm.create_model("vit_base_patch16_224", pretrained=False)

        checkpoint_path = None
        if Cassava_model and os.path.exists(Cassava_model):
            checkpoint_path = Cassava_model
            print("Attempting to load Cassava‑specific checkpoint.")
        elif model_path and os.path.exists(model_path):
            checkpoint_path = model_path
            print("Attempting to load generic ViT checkpoint.")
        else:
            print(
                "No external checkpoint found – will use ImageNet pretrained weights later."
            )

        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location=device)
            load_msg = self.model.load_state_dict(state_dict, strict=False)
            print("Checkpoint loaded with strict=False:", load_msg)
        else:
            self.model = timm.create_model("vit_base_patch16_224", pretrained=True)

        if (
            getattr(self.model, "head", None) is None
            or self.model.head.in_features != n_classes
        ):
            self.model.head = nn.Linear(self.model.head.in_features, n_classes)
            print("Replaced model head to have", n_classes, "outputs.")
        else:
            print(
                "Model head already matches", n_classes, "outputs; kept loaded weights."
            )

    def forward(self, x):
        return self.model(x)




## === cell 6
cassava_model = ViTBase16(n_classes=5, pretrained=False)
cassava_gpu_model = cassava_model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cassava_gpu_model.parameters(), lr=1.5e-05)
print(cassava_gpu_model)




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
        self.image_names = [
            f for f in os.listdir(self.test_dir) if f.lower().endswith(".jpg")
        ]
        print(f"Cassava Disease Test Dataset Length = {len(self.image_names)}")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.test_dir, img_name)
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
for images, names in test_loader:
    im = make_grid(images, nrow=4)
    plt.figure(figsize=(12, 4))
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
    plt.title(f"Example batch – first image: {names[0]}")
    plt.axis("off")
    plt.show()
    break




## === cell 13
col_names = ["image_id", "label"]
rows = []  # collect rows here

tic = time.time()
cassava_gpu_model.eval()
with torch.no_grad():
    for X_test, name in test_loader:
        X_test = X_test.to(device)

        logits_orig = cassava_gpu_model(X_test)

        X_flip = torch.flip(X_test, dims=[3])  # flip width dimension
        logits_flip = cassava_gpu_model(X_flip)

        X_rot = torch.rot90(X_test, k=1, dims=[2, 3])  # rotate height<->width
        logits_rot = cassava_gpu_model(X_rot)

        combined_logits = logits_orig + logits_flip + logits_rot
        predicted = torch.argmax(combined_logits, dim=1).item()

        rows.append({"image_id": name[0], "label": int(predicted)})

toc = time.time() - tic
print("Inference time:", toc)

submission_df = pd.DataFrame(rows, columns=col_names)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Saved submission to", submission_path)
display(pd.read_csv(submission_path).head())
