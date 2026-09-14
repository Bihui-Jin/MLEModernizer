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

geopandas==0.14.4
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

0.8655182834693261

# 6. Current score

0.10575

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.10575) has done: 'The baseline code fails because it relies on external packages/modules (`data`, `model`, `configs`, etc.) that are not present in this Kaggle environment, so nothing gets imported, the model never loads, and the written CSV ends up incomplete/incorrect. I replace those missing dependencies with a minimal, self-contained PyTorch inference pipeline using `torchvision`’s ResNet-50 (same general backbone family as your config name suggests) and standard image transforms, then generate predictions for every row in `sample_submission.csv` to guarantee correct length/order. I also make the code robust to the actual dataset path (both `/kaggle/input/...` and the provided `../input/...`-style paths), and ensure the submission is written as `submission.csv` with exactly `image_id,label` and 2676 rows. This should run end-to-end and produce a valid submission file without changing the overall “single CNN image classifier → argmax label” evaluation semantics.'

# 9. Code solution

## === cell 0
import os
import sys
import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(42)

BASE_INPUT_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification",
    "/kaggle/input",
    "/kaggle/data",
    "../input",
]


def find_existing_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


base_path = find_existing_path(BASE_INPUT_CANDIDATES)
if base_path is None:
    raise FileNotFoundError(
        "Could not locate Kaggle input data directory from known candidates."
    )

if os.path.exists(os.path.join(base_path, "sample_submission.csv")):
    data_root = base_path
else:
    nested = os.path.join(base_path, "cassava-leaf-disease-classification")
    if os.path.exists(os.path.join(nested, "sample_submission.csv")):
        data_root = nested
    else:
        raise FileNotFoundError(
            "Could not find sample_submission.csv under expected input paths."
        )

sample_sub_path = os.path.join(data_root, "sample_submission.csv")
test_img_dir = os.path.join(data_root, "test_images")
assert os.path.exists(sample_sub_path), f"Missing: {sample_sub_path}"
assert os.path.isdir(test_img_dir), f"Missing dir: {test_img_dir}"

sample_sub = pd.read_csv(sample_sub_path)
print("sample_submission shape:", sample_sub.shape)
print(
    "test_images files:",
    len([f for f in os.listdir(test_img_dir) if f.lower().endswith(".jpg")]),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)



## === cell 1
num_classes = 5
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, num_classes)
model = model.to(device)

weights_candidates = [
    "../input/baseline-weights/epoch_16.pth",
    "/kaggle/input/baseline-weights/epoch_16.pth",
]
weights_path = None
for wp in weights_candidates:
    if os.path.exists(wp):
        weights_path = wp
        break

if weights_path is not None:
    ckpt = torch.load(weights_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    model_sd = model.state_dict()
    loaded = 0
    for k, v in ckpt.items() if isinstance(ckpt, dict) else []:
        if k in model_sd and model_sd[k].shape == v.shape:
            model_sd[k] = v
            loaded += 1
        elif (
            k.startswith("module.")
            and k[7:] in model_sd
            and model_sd[k[7:]].shape == v.shape
        ):
            model_sd[k[7:]] = v
            loaded += 1
    model.load_state_dict(model_sd, strict=True)
    print(
        f"Loaded weights from: {weights_path} (matched tensors: {loaded}/{len(model_sd)})"
    )
else:
    print("WARNING: baseline weights not found; proceeding without finetuned weights.")

model.eval()




## === cell 2
class CassavaTestDataset(Dataset):
    def __init__(self, img_dir, image_ids, transform=None):
        self.img_dir = img_dir
        self.image_ids = list(image_ids)
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        fn = self.image_ids[idx]
        fp = os.path.join(self.img_dir, fn)
        img = Image.open(fp).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, fn


img_size = 512
test_tfms = transforms.Compose(
    [
        transforms.Resize(
            int(img_size * 1.14), interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_ds = CassavaTestDataset(
    img_dir=test_img_dir,
    image_ids=sample_sub["image_id"].values,
    transform=test_tfms,
)

loader = DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

x0, fn0 = test_ds[0]
print("first image tensor shape:", x0.shape, "first fn:", fn0)



## === cell 3
pred_labels = []
pred_image_ids = []

with torch.no_grad():
    for images, fns in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int).tolist()
        pred_labels.extend(preds)
        pred_image_ids.extend(list(fns))

submission = pd.DataFrame({"image_id": pred_image_ids, "label": pred_labels})

submission = sample_sub[["image_id"]].merge(submission, on="image_id", how="left")
missing = submission["label"].isna().sum()
if missing > 0:
    submission["label"] = submission["label"].fillna(0).astype(int)
    print(f"WARNING: {missing} missing predictions were filled with label 0.")
else:
    submission["label"] = submission["label"].astype(int)

out_path = "./submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print("submission shape:", submission.shape)
print(submission.head())



## === cell 4
check = pd.read_csv("./submission.csv")
assert list(check.columns) == ["image_id", "label"]
assert len(check) == len(sample_sub), (len(check), len(sample_sub))
assert check["label"].between(0, 4).all()
print("Submission OK:", check.shape)
print(check.tail())
