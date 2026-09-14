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

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
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
tqdm==4.67.1

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

0.8845572680568148

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.25673) has done: 'I make the pipeline run end-to-end without requiring an external weights dataset by falling back to a torchvision pretrained ViT when the requested `.pth` file isn’t available, keeping the same ViT family and inference flow. I also fix the device/dtype mismatch that caused `Input type ... and weight type ... should be the same` by ensuring the model is moved to `device` after any potential weight loading and that inputs use the same device. Finally, I keep the submission formatting aligned to `sample_submission.csv` and always write a valid `submission.csv` in the working directory.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.25673) is far below the target (0.88456), and the biggest issue is that your fallback path uses an ImageNet-pretrained ViT with a freshly initialized 5-class head, so predictions are essentially near-random. To move the score sharply toward the target while keeping the same core ViT inference pipeline, the minimal fix is to *require* a valid cassava fine-tuned `.pth/.pt` and load it correctly (including common checkpoint key-prefix patterns like `module.`). I also switch test-time preprocessing to the official ViT_H_14 weights’ transforms when using torchvision weights (kept as a last-resort fallback), and I add a lightweight batch inference (no change to semantics) to reduce overhead and stay within runtime. The output submission format/path stays identical and still always write `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

import torch
from torchvision import transforms, models
from tqdm import tqdm
from PIL import Image

test_data_directory = "/kaggle/input/cassava-leaf-disease-classification/test_images"
sample_submission_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

model_path = "/kaggle/input/vit_l_cassava/pytorch/default/1/model_weights_3.pth"

image_size = 518
num_classes = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


def find_fallback_weights_path(preferred_path: str) -> str | None:
    """
    Score-improvement fix (minimal, keeps same ViT inference flow):
    - The current low score is consistent with using ImageNet weights + random 5-class head.
      That produces near-random cassava predictions.
    - So we aggressively search for a cassava fine-tuned checkpoint under /kaggle/input,
      and only fall back to torchvision weights if none exists.
    """
    if os.path.isfile(preferred_path):
        return preferred_path

    candidates = []
    for pattern in ["/kaggle/input/**/*.pth", "/kaggle/input/**/*.pt"]:
        candidates.extend(glob.glob(pattern, recursive=True))

    def score(p: str) -> int:
        base = os.path.basename(p).lower()
        full = p.lower()
        s = 0
        if "cassava" in full:
            s += 10
        if "leaf" in full:
            s += 2
        if "vit" in base or "vit" in full:
            s += 5
        if "weight" in base or "weights" in base:
            s += 2
        if "model" in base:
            s += 1
        if "imagenet" in full:
            s -= 3
        return s

    candidates = sorted(set(candidates), key=lambda p: (score(p), p), reverse=True)
    if not candidates:
        return None
    return candidates[0]


def torch_load_state_dict(path: str, map_location):
    """
    Compatibility: environments differ on torch.load(weights_only=...).
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def unwrap_state_dict(state):
    """
    Score-improvement fix: handle common checkpoint wrappers & key prefixes so we actually
    load the trained weights into the ViT (otherwise accuracy collapses).
    """
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state_dict = state["state_dict"]
    elif (
        isinstance(state, dict)
        and "model" in state
        and isinstance(state["model"], dict)
    ):
        state_dict = state["model"]
    else:
        state_dict = state

    if isinstance(state_dict, dict):
        new_sd = {}
        for k, v in state_dict.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            new_sd[nk] = v
        state_dict = new_sd

    return state_dict


weights_path = find_fallback_weights_path(model_path)
if weights_path is None:
    print(
        "Warning: No external cassava fine-tuned .pth/.pt found under /kaggle/input.\n"
        "Falling back to torchvision pretrained ViT (expected to score low because head is not cassava-trained)."
    )
else:
    print(f"Using weights: {weights_path}")



## === cell 1
if weights_path is None:
    tv_weights = models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
    val_transforms = tv_weights.transforms()
else:
    val_transforms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

if weights_path is None:
    model = models.vit_h_14(weights=tv_weights, image_size=image_size)
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
else:
    model = models.vit_h_14(weights=None, image_size=image_size)
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)

    state = torch_load_state_dict(weights_path, map_location="cpu")
    state_dict = unwrap_state_dict(state)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            f"Warning: non-strict load_state_dict. Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}"
        )

model = model.to(device)
model.eval()



## === cell 2
sample_sub = pd.read_csv(sample_submission_path)
image_ids = sample_sub["image_id"].tolist()

batch_size = 16 if device.type == "cuda" else 8

test_predictions = []
batch_imgs = []
batch_names = []

with torch.no_grad():
    for image_name in tqdm(image_ids, desc="Test"):
        image_path = os.path.join(test_data_directory, image_name)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Test image not found: {image_path}")

        img = Image.open(image_path).convert("RGB")
        img_t = val_transforms(img)
        batch_imgs.append(img_t)
        batch_names.append(image_name)

        if len(batch_imgs) == batch_size:
            x = torch.stack(batch_imgs, dim=0).to(device)
            logits = model(x)
            preds = (
                torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int).tolist()
            )
            test_predictions.extend(preds)
            batch_imgs, batch_names = [], []

    if batch_imgs:
        x = torch.stack(batch_imgs, dim=0).to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int).tolist()
        test_predictions.extend(preds)

if len(test_predictions) != len(image_ids):
    raise RuntimeError(
        f"Prediction length mismatch: {len(test_predictions)} vs {len(image_ids)}"
    )

submission_df = pd.DataFrame({"image_id": image_ids, "label": test_predictions})
submission_df = submission_df[["image_id", "label"]]
submission_df.to_csv("submission.csv", index=False)

print("Submission file created: submission.csv")
print(submission_df.head())
print("Rows:", len(submission_df))
