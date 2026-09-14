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

3.13

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

0.8135388334844363

# 6. Current score

0.09417

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.09417) has done: 'I remove the hard dependency on the missing external checkpoint by falling back to a standard ResNet50 initialization that exists in the Kaggle environment, so the notebook runs end-to-end and always produces `submission.csv`. I also fix the cascading errors (undefined `model`, mismatched `preds` length) by ensuring `preds` is generated for every `image_id` and by adding a safe, batched inference loop for speed/stability without changing the model family. Finally, I keep the exact required submission columns/order and add minimal sanity checks so the file is valid.'

# 9. Code solution

## === cell 0
import os
import time
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

main_model_preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

CKPT_PATH = "/kaggle/input/resnet50_70_2/pytorch/default/1/Resnet50_70_2.pth"

assert os.path.exists(TEST_IMG_DIR), f"Missing test_images dir: {TEST_IMG_DIR}"
assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample_submission.csv: {SAMPLE_SUB_PATH}"

print("Device:", device)
print("Checkpoint exists:", os.path.exists(CKPT_PATH), "->", CKPT_PATH)




## === cell 1
def _build_resnet50_num_classes(
    num_classes: int = 5, pretrained: bool = True
) -> nn.Module:
    if pretrained:
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    else:
        m = resnet50(weights=None)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    return m


def load_resnet_checkpoint_or_fallback(
    ckpt_path: str, device: torch.device
) -> nn.Module:
    """
    Fix: The provided CKPT_PATH does not exist in this Kaggle environment.
    Fallback to an ImageNet-pretrained ResNet50 head-replaced to 5 classes so inference runs and yields a valid submission.
    """
    if not (ckpt_path and os.path.exists(ckpt_path)):
        model = _build_resnet50_num_classes(num_classes=5, pretrained=True).to(device)
        model.eval()
        return model

    obj = torch.load(ckpt_path, map_location=device)

    if isinstance(obj, nn.Module):
        model = obj.to(device)
        model.eval()
        return model

    state_dict = None
    if isinstance(obj, dict):
        for k in ["state_dict", "model_state_dict", "model", "net"]:
            if k in obj and isinstance(obj[k], dict):
                state_dict = obj[k]
                break
        if state_dict is None:
            if any(
                isinstance(k, str) and (".weight" in k or ".bias" in k)
                for k in obj.keys()
            ):
                state_dict = obj

    if state_dict is None:
        model = _build_resnet50_num_classes(num_classes=5, pretrained=True).to(device)
        model.eval()
        return model

    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k[len("module.") :]] = v if k.startswith("module.") else v

    model = _build_resnet50_num_classes(num_classes=5, pretrained=False).to(device)
    model.load_state_dict(cleaned, strict=False)
    model.eval()
    return model


model = load_resnet_checkpoint_or_fallback(CKPT_PATH, device)
print("Model ready. Params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")



## === cell 2
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
assert "image_id" in sample_sub.columns and "label" in sample_sub.columns
image_ids = sample_sub["image_id"].tolist()

preds = []
t0 = time.time()

BATCH_SIZE = 32


def _load_image_tensor(img_path: str) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    return main_model_preprocess(img)


with torch.no_grad():
    batch = []
    batch_names = []
    for i, image_name in enumerate(image_ids):
        img_path = os.path.join(TEST_IMG_DIR, image_name)
        batch.append(_load_image_tensor(img_path))
        batch_names.append(image_name)

        if len(batch) == BATCH_SIZE or i == (len(image_ids) - 1):
            x = torch.stack(batch, dim=0).to(device, non_blocking=True)
            out = model(x)
            batch_preds = (
                torch.argmax(out, dim=1).detach().cpu().numpy().astype(int).tolist()
            )
            preds.extend(batch_preds)

            batch.clear()
            batch_names.clear()
            del x, out

            if device.type == "cuda":
                torch.cuda.empty_cache()

elapsed = time.time() - t0
print(f"Inference done: {len(preds)} predictions in {elapsed:.1f}s")

assert len(preds) == len(
    image_ids
), f"Pred length {len(preds)} != image_ids length {len(image_ids)}"



## === cell 3
submission = pd.DataFrame({"image_id": image_ids, "label": preds})

assert len(submission) == len(
    sample_sub
), "Submission length mismatch vs sample_submission"
assert submission["image_id"].isnull().sum() == 0
assert submission["label"].isnull().sum() == 0
assert list(submission.columns) == ["image_id", "label"]

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
