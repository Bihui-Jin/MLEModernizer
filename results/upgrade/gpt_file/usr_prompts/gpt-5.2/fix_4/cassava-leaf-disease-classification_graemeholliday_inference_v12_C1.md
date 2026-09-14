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

albumentations==2.0.8
geopandas==0.14.4
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

0.8765488062858869

# 6. Current score

0.09006

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I remove the offline `pip install efficientnet_pytorch` dependency and switch to the equivalent `torchvision` EfficientNet backbone so imports work in this environment. I also fix inference-time albumentations errors by replacing deprecated/changed transforms (e.g., `RandomResizedCrop` now requires `size`, and `Cutout` is not available in albumentations v2) while keeping the same inference-TTA intent. Finally, I make the code run on both CPU and GPU (no hard CUDA assert) and ensure the submission rows exactly match `sample_submission.csv` ordering/length so Kaggle accepts the file.'
- What this solution (achieved 0.08483) has done: 'I fix the immediate runtime failure by making checkpoint loading robust to missing files and by pointing `checkpoint_path` to the actual dataset directory structure (and searching common locations) instead of a non-existent `../input/focalloss`. To move accuracy toward your target (your current 0.11584 is far below), I also load ImageNet pretrained EfficientNet weights when no custom checkpoint is found, which is a minimal change that keeps the same model architecture and inference flow but yields sane predictions. I additionally make the inference transforms deterministic/valid for test-time inference by replacing the stochastic `RandomResizedCrop` with a standard `Resize` (still keeping flips/transpose TTA as intended) to avoid harmful random crops at inference. Finally, I ensure the submission is written with the correct columns/order and `.csv` suffix.'
- What this solution (achieved 0.09006) has done: 'Your score is far below the target, so the smallest safe way to move accuracy up (without changing the model/loop) is to fix the normalization mismatch: torchvision EfficientNet pretrained weights expect inputs normalized with its own mean/std (not generic ImageNet values). I switch the `Normalize` parameters to `EfficientNet_B4_Weights`’ built-in transforms (mean/std), and keep your existing TTA/epoch-averaging inference exactly the same. I also set `cudnn.benchmark=False` when `deterministic=True` to avoid nondeterministic kernels that can slightly destabilize predictions run-to-run (this shouldn’t hurt accuracy, but improves reproducibility). All paths and submission formatting stay unchanged.'

# 9. Code solution

## === cell 0
import os
import gc
import time
import random
import numpy as np
import pandas as pd
import cv2

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from tqdm import tqdm
from torchvision import models



## === cell 1
image_size = 384



## === cell 2
config = dict(
    seed=22,
    experiment_name="focal_loss",
    test_location="../input/cassava-leaf-disease-classification/test_images",
    checkpoint_path="../input/cassava-leaf-disease-classification",
    checkpoint="focal_loss.pt",
    model="efficientnet-b4",  # keep semantic name; we will map to torchvision efficientnet_b4
    epochs=5,
    batch_size=32,
    workers=2,
    inference_augmentations=[
        dict(name="Resize", params=dict(height=image_size, width=image_size)),
        dict(name="HorizontalFlip", params=dict(always_apply=False, p=0.5)),
        dict(name="VerticalFlip", params=dict(always_apply=False, p=0.5)),
        dict(name="Transpose", params=dict(p=0.5)),
        dict(
            name="Normalize",
            params=dict(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
        ),
    ],
)



## === cell 3
sample_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test = pd.read_csv(sample_path)
test.head()




## === cell 4
def seed(seed=22):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




## === cell 5
seed(config["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)



## === cell 6
_EFFNET_B4_DEFAULT_WEIGHTS = models.EfficientNet_B4_Weights.IMAGENET1K_V1
_EFFNET_B4_MEAN = list(_EFFNET_B4_DEFAULT_WEIGHTS.transforms().mean)
_EFFNET_B4_STD = list(_EFFNET_B4_DEFAULT_WEIGHTS.transforms().std)

for aug in config["inference_augmentations"]:
    if aug.get("name") == "Normalize":
        aug["params"]["mean"] = _EFFNET_B4_MEAN
        aug["params"]["std"] = _EFFNET_B4_STD
        aug["params"]["max_pixel_value"] = 255.0


def _build_backbone_from_config(pretrained: bool = False):
    name = config["model"].lower()
    if name in ["efficientnet-b4", "efficientnet_b4"]:
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b4(weights=weights)
    elif name in ["efficientnet-b0", "efficientnet_b0"]:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
    else:
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b4(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 5)
    return model


def _find_checkpoint_path():
    candidates = []

    candidates.append(os.path.join(config["checkpoint_path"], config["checkpoint"]))

    candidates.append(os.path.join("../input", config["checkpoint"]))
    candidates.append(
        os.path.join(
            "../input/cassava-leaf-disease-classification", config["checkpoint"]
        )
    )

    search_root = "../input"
    for root, _, files in os.walk(search_root):
        if config["checkpoint"] in files:
            candidates.append(os.path.join(root, config["checkpoint"]))

    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


def load_model():
    ckpt_path = _find_checkpoint_path()

    if ckpt_path is None:
        print(
            f"WARNING: checkpoint '{config['checkpoint']}' not found. "
            "Falling back to ImageNet-pretrained EfficientNet weights."
        )
        model = _build_backbone_from_config(pretrained=True)
        return model.to(device)

    model = _build_backbone_from_config(pretrained=False)
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    state_dict = None
    if isinstance(checkpoint, dict):
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            state_dict = checkpoint["state_dict"]
        else:
            if all(isinstance(k, str) for k in checkpoint.keys()):
                state_dict = checkpoint

    if state_dict is None:
        raise ValueError(f"Unrecognized checkpoint format at: {ckpt_path}")

    new_state = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        new_state[nk] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print("Loading model from checkpoint:", ckpt_path)
    if missing:
        print(
            "Missing keys (non-fatal):", missing[:5], "..." if len(missing) > 5 else ""
        )
    if unexpected:
        print(
            "Unexpected keys (non-fatal):",
            unexpected[:5],
            "..." if len(unexpected) > 5 else "",
        )
    return model.to(device)




## === cell 7
def get_transforms():
    transforms = [
        getattr(A, item["name"])(**item["params"])
        for item in config["inference_augmentations"]
    ]
    return A.Compose(transforms)




## === cell 8
class CassavaDataset(Dataset):
    def __init__(self, images, transforms):
        self.images = images
        self.transforms = transforms

    def __getitem__(self, n):
        img_name = self.images[n]
        img_path = os.path.join(config["test_location"], img_name)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)["image"]
        image = np.moveaxis(image, -1, 0)
        image = torch.tensor(image, dtype=torch.float32)
        return image

    def __len__(self):
        return len(self.images)




## === cell 9
def get_dataloader():
    transforms = get_transforms()
    test_data = np.array(test["image_id"].values)

    data = CassavaDataset(test_data, transforms)
    dataloader = DataLoader(
        data,
        shuffle=False,
        batch_size=config["batch_size"],
        pin_memory=torch.cuda.is_available(),
        num_workers=config["workers"],
    )
    return dataloader




## === cell 10
def infer(model, dataloader):
    print("Running inference...")
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            batch = batch.to(device, non_blocking=True)
            batch_hat = model(batch)
            predictions.append(batch_hat.detach().cpu())

    return torch.cat(predictions, dim=0)




## === cell 11
if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dataloader = get_dataloader()
    model = load_model()
    predictions = None
    print("Inferring experiment", config["experiment_name"])

    for epoch in range(config["epochs"]):
        print("Epoch:", epoch)
        start_time = time.time()

        if epoch == 0:
            predictions = infer(model, dataloader)
        else:
            predictions += infer(model, dataloader)

        print("Time:", time.time() - start_time)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    predictions /= config["epochs"]
    results = predictions.numpy()
    test["label"] = np.argmax(results, axis=-1).astype(int)



## === cell 12
sub = test[["image_id", "label"]].copy()
sample = pd.read_csv(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)
sub = sample[["image_id"]].merge(sub, on="image_id", how="left")
assert len(sub) == len(sample), "Submission length mismatch"
assert sub["label"].isna().sum() == 0, "Some test images missing predictions"
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
