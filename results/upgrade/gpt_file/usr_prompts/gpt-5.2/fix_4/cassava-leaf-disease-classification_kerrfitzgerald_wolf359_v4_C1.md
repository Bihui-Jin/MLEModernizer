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

0.10949

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.10874) has done: 'I fix the runtime failure by removing the hard dependency on an external checkpoint that isn’t present in this environment and instead fall back to a timm ImageNet-pretrained ViT (same architecture) when the custom `.pt` file can’t be found. This keeps the core model and inference logic intact while ensuring the notebook runs end-to-end and always defines `cassava_gpu_model`. I also make the weight-loading more robust (handle common checkpoint key formats) and ensure the submission is written as `submission.csv` with the required columns and row count aligned to the sample submission ordering.'
- What this solution (achieved 0.10949) has done: 'Your current score (0.10874) is far below the target (0.83545), and the biggest likely cause is that you are not actually loading the trained Cassava fine-tuned checkpoint (so predictions are essentially from a generic ImageNet ViT head). I make a minimal, robust checkpoint-loading fix that correctly maps common key patterns (including `state_dict` with `model.` prefixes and mismatched `head.*` keys) so the 5-class head weights load when available, without changing the model architecture or inference logic. I also switch inference to batched prediction (larger `batch_size`) and enable AMP autocast to stay well under the time limit; this should not change evaluation semantics beyond negligible floating-point differences. Finally, I keep the submission alignment to `sample_submission.csv` exactly as you already do.'

# 9. Code solution

## === cell 0
import os
import time
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

import matplotlib.pyplot as plt


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)



## === cell 1
pass



## === cell 2
import sys
import subprocess

wheel_path = "../input/timm034/timm-0.3.4-py3-none-any.whl"
try:
    if os.path.exists(wheel_path):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", wheel_path]
        )
    else:
        print(
            f"Wheel not found at {wheel_path}; will try using installed timm if available."
        )
except Exception as e:
    print(
        "pip install attempt failed; will try using installed timm if available. Error:",
        repr(e),
    )



## === cell 3
pass



## === cell 4
import timm

print("timm version:", getattr(timm, "__version__", "unknown"))



## === cell 5
pass



## === cell 6
print("Available ViT Models (first 20):")
print(timm.list_models("vit*")[:20])



## === cell 7
data_path = "../input/cassava-leaf-disease-classification/"
train_path = "../input/cassava-leaf-disease-classification/train_images/"
test_path = "../input/cassava-leaf-disease-classification/test_images/"
model_path = "../input/vitbase16224/jx_vit_base_p16_224-80ecf9dd.pth"
Cassava_model = (
    "../input/cassavaaugmtp98epochs1lr175/CassavaViT_Augm_TP98_Epochs1_LR1-75e05.pt"
)

print("Paths:")
print(" data_path :", data_path)
print(" train_path:", train_path)
print(" test_path :", test_path)
print(" model_path exists?  ", os.path.exists(model_path), model_path)
print(" Cassava_model exists?", os.path.exists(Cassava_model), Cassava_model)



## === cell 8
pass




## === cell 9
class ViTBase16(nn.Module):
    def __init__(self, n_classes, pretrained=False):
        super(ViTBase16, self).__init__()

        timm_pretrained = False
        if pretrained and (not os.path.exists(model_path)):
            timm_pretrained = True
            print(
                f"WARNING: base model weights not found at {model_path}. "
                f"Falling back to timm pretrained weights."
            )

        self.model = timm.create_model(
            "vit_base_patch16_224", pretrained=timm_pretrained
        )

        if pretrained and os.path.exists(model_path) and (not timm_pretrained):
            state = torch.load(model_path, map_location="cpu")
            if (
                isinstance(state, dict)
                and "state_dict" in state
                and isinstance(state["state_dict"], dict)
            ):
                state = state["state_dict"]
            if (
                isinstance(state, dict)
                and "model" in state
                and isinstance(state["model"], dict)
            ):
                state = state["model"]
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(
                    f"Loaded base weights with strict=False. Missing: {len(missing)}, Unexpected: {len(unexpected)}"
                )

        self.model.head = nn.Linear(self.model.head.in_features, n_classes)

    def forward(self, x):
        return self.model(x)




## === cell 10
pass




## === cell 11
def _unwrap_checkpoint(ckpt_obj):
    if not isinstance(ckpt_obj, dict):
        return ckpt_obj
    for key in ["state_dict", "model_state_dict", "model", "net"]:
        if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
            return ckpt_obj[key]
    return ckpt_obj


def _clean_state_dict_keys(state):
    cleaned = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        cleaned[nk] = v
    return cleaned


cassava_model = ViTBase16(n_classes=5, pretrained=True)

if os.path.exists(Cassava_model):
    ckpt = torch.load(Cassava_model, map_location="cpu")
    ckpt = _unwrap_checkpoint(ckpt)
    if isinstance(ckpt, dict):
        ckpt = _clean_state_dict_keys(ckpt)

        model_sd = cassava_model.state_dict()
        remapped = {}
        for k, v in ckpt.items():
            rk = k
            if rk.startswith("head.") and ("model." + rk) in model_sd:
                rk = "model." + rk
            remapped[rk] = v
        ckpt = remapped

        missing, unexpected = cassava_model.load_state_dict(ckpt, strict=False)
        print(
            f"Loaded trained Cassava checkpoint. Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        )
        head_loaded = ("model.head.weight" not in missing) and (
            "model.head.bias" not in missing
        )
        print("Head weights loaded?", head_loaded)
else:
    print(
        f"WARNING: Trained Cassava checkpoint not found: {Cassava_model}\n"
        "Proceeding with ViT backbone initialization (timm pretrained if available)."
    )

cassava_gpu_model = cassava_model.to(DEVICE).eval()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cassava_gpu_model.parameters(), lr=1.5e-05)

print(cassava_gpu_model.__class__.__name__, "ready on", DEVICE)



## === cell 12
pass



## === cell 13
test_img_names = []
for folder, subfolders, filenames in os.walk(test_path):
    for img in filenames:
        if img.lower().endswith(".jpg"):
            test_img_names.append(img)

print("Testing Images:", len(test_img_names))
print("First 5 test images:", sorted(test_img_names)[:5])



## === cell 14
pass




## === cell 15
class TestSet2(Dataset):
    """Cassava Disease Test Dataset"""

    def __init__(self, root_dir, test_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.test_dir = test_dir
        self.transform = transform

        self.files = sorted(
            [f for f in os.listdir(self.test_dir) if f.lower().endswith(".jpg")]
        )

        print(root_dir)
        print(test_dir)
        print("Cassava Disease Test Dataset Length = ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.test_dir, fname)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(img)
        else:
            image = transforms.ToTensor()(img)

        return (image, fname)




## === cell 16
pass



## === cell 17
test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)



## === cell 18
pass



## === cell 19
testset = TestSet2(root_dir="", test_dir=test_path, transform=test_transform)
print(testset)



## === cell 20
pass



## === cell 21
test_batch_size = 32 if torch.cuda.is_available() else 8
test_loader = DataLoader(
    dataset=testset,
    batch_size=test_batch_size,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    num_workers=2,
)



## === cell 22
pass



## === cell 23
print(test_loader)

for images, names in test_loader:
    break

im = make_grid(images[: min(len(images), 16)], nrow=4)
print("Example batch names:", names[:4])

plt.figure(figsize=(12, 4))
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
plt.axis("off")
plt.show()



## === cell 24
pass



## === cell 25
rows = []
tic = time.time()

cassava_gpu_model.eval()
use_amp = torch.cuda.is_available()
with torch.no_grad():
    for b, (X_test, names) in enumerate(test_loader):
        X_test = X_test.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            y_test_pred = cassava_gpu_model(X_test)
        predicted = torch.argmax(y_test_pred, dim=1).detach().cpu().numpy().astype(int)
        for fname, lab in zip(list(names), list(predicted)):
            rows.append((fname, int(lab)))

toc = time.time() - tic
print("Time for test inference is", toc, "seconds")

submission_df = pd.DataFrame(rows, columns=["image_id", "label"])

assert submission_df.shape[0] == len(
    testset
), "Submission rows != number of test images"
assert list(submission_df.columns) == ["image_id", "label"]

sample_path = os.path.join(data_path, "sample_submission.csv")
if os.path.exists(sample_path):
    sample_df = pd.read_csv(sample_path)
    submission_df = sample_df[["image_id"]].merge(
        submission_df, on="image_id", how="left"
    )
    if submission_df["label"].isna().any():
        submission_df["label"] = submission_df["label"].fillna(0).astype(int)
else:
    submission_df = submission_df.sort_values("image_id").reset_index(drop=True)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

df_sub_test = pd.read_csv(submission_path)
print(df_sub_test.head())
print("Wrote:", submission_path, "rows:", len(df_sub_test))

print("Files in current directory:")
print(os.listdir("."))
