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

0.8912058023572076

# 6. Current score

0.1562

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.1577) has done: 'I fix the Albumentations v2 API mismatch that prevents transforms from constructing (the `RandomResizedCrop` transform now expects `size=(h,w)` instead of `height`/`width`). I also make the transform builder backward/forward compatible so it won’t crash if the environment changes, while keeping the same augmentation intent and inference loop. Finally, I ensure the submission is always written after successful inference and matches the required `image_id,label` format.'
- What this solution (achieved 0.1562) has done: 'Your current score (0.1577) is far below the target (0.8912), and the most likely cause is that the checkpoint is not being found/loaded, so you’re effectively running an ImageNet-pretrained EfficientNet with a random 5-class head (or a mismatched head), which yields near-random accuracy. I make the checkpoint search/load stricter and transparent, and ensure we correctly extract the `state_dict` from common checkpoint formats (e.g., `state_dict`, `model_state_dict`, PyTorch Lightning `state_dict`) so the trained weights actually load. I also enforce `strict=True` once the right state dict is identified (to avoid silently missing most weights), falling back to ImageNet only if loading truly fails. These are minimal changes that preserve your exact inference-time augmentation and TTA loop, but should move the score sharply upward toward the target.'

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
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from tqdm import tqdm

import torchvision



## === cell 1
image_size = 380



## === cell 2
config = dict(
    seed=22,
    experiment_name="enb7",
    test_location="../input/cassava-leaf-disease-classification/test_images",
    checkpoint_path="../input/savedmodels2",
    checkpoint="enb7best.pt",
    model="efficientnet-b7",
    epochs=10,
    batch_size=16,
    workers=8,
    inference_augmentations=[
        dict(
            name="RandomResizedCrop",
            params=dict(
                height=image_size,
                width=image_size,
                scale=(0.8, 1.0),
                ratio=(0.75, 1.3333333333),
                p=1.0,
            ),
        ),
        dict(name="HorizontalFlip", params=dict(p=0.5)),
        dict(name="Transpose", params=dict(p=0.5)),
        dict(name="VerticalFlip", params=dict(p=0.5)),
        dict(
            name="HueSaturationValue",
            params=dict(
                hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
            ),
        ),
        dict(
            name="RandomBrightnessContrast",
            params=dict(
                brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
            ),
        ),
        dict(
            name="Normalize",
            params=dict(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ),
        dict(
            name="CoarseDropout",
            params=dict(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.5,
            ),
        ),
    ],
)



## === cell 3
sample_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
sample = pd.read_csv(sample_path)
test = sample[["image_id"]].copy()
test.head()




## === cell 4
def seed(seed=22):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)




## === cell 5
seed(config["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)




## === cell 6
def _build_model_torchvision(use_imagenet_weights: bool = False):
    weights = (
        torchvision.models.EfficientNet_B7_Weights.DEFAULT
        if use_imagenet_weights
        else None
    )
    model = torchvision.models.efficientnet_b7(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, 5)
    return model


def _strip_module_prefix(state_dict: dict):
    new_state = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        new_state[nk] = v
    return new_state


def _extract_state_dict(ckpt_obj):
    """
    Change rationale (score toward target): the previous loader often fed the *entire checkpoint dict*
    into load_state_dict(strict=False), silently missing most weights and producing near-random accuracy.
    We now robustly extract the real state_dict from common keys and only then load it.
    """
    if isinstance(ckpt_obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
        if len(ckpt_obj) > 0 and all(
            hasattr(v, "shape") or torch.is_tensor(v) for v in ckpt_obj.values()
        ):
            return ckpt_obj
    return None


def _load_state_dict_flex(model, ckpt_obj):
    state = _extract_state_dict(ckpt_obj)
    if state is None:
        raise ValueError(
            "Could not extract a state_dict from the checkpoint object. "
            "Expected keys like 'state_dict'/'model_state_dict'/'model'."
        )

    state = _strip_module_prefix(state)

    try:
        model.load_state_dict(state, strict=True)
        print("Loaded checkpoint with strict=True.")
        return model
    except RuntimeError as e:
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            "WARNING: strict=True failed; loaded with strict=False.\n"
            f"RuntimeError: {e}\n"
            f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}"
        )
        return model


def _find_checkpoint_path(checkpoint_name: str, preferred_dir: str):
    candidates = []

    if preferred_dir is not None:
        candidates.append(os.path.join(preferred_dir, checkpoint_name))

    search_roots = [
        "../input",
        "/kaggle/input",
        "../input/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification",
    ]

    for root in search_roots:
        if os.path.isdir(root):
            candidates.append(os.path.join(root, checkpoint_name))

    for p in candidates:
        if p and os.path.isfile(p):
            return p

    for root in search_roots:
        if os.path.isdir(root):
            for dirpath, _, filenames in os.walk(root):
                if checkpoint_name in filenames:
                    return os.path.join(dirpath, checkpoint_name)

    return None


def load_model():
    ckpt_path = _find_checkpoint_path(
        config["checkpoint"], config.get("checkpoint_path")
    )

    if ckpt_path is None:
        print(
            f"WARNING: Checkpoint '{config['checkpoint']}' not found under '{config.get('checkpoint_path')}'.\n"
            f"Falling back to ImageNet pretrained EfficientNet-B7 weights (architecture unchanged)."
        )
        model = _build_model_torchvision(use_imagenet_weights=True)
        model = model.to(device)
        model.eval()
        return model

    print("Found checkpoint:", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        for k in ["epoch", "train_loss", "val_loss", "metrics", "lr"]:
            if k in checkpoint:
                print(f"{k}:", checkpoint[k])

    model = _build_model_torchvision(use_imagenet_weights=False)
    try:
        model = _load_state_dict_flex(model, checkpoint)
    except Exception as e:
        print(
            "WARNING: Failed to load weights from checkpoint; falling back to ImageNet weights.\n"
            f"Reason: {repr(e)}"
        )
        model = _build_model_torchvision(use_imagenet_weights=True)

    model = model.to(device)
    model.eval()
    return model




## === cell 7
def _make_albu_transform(name: str, params: dict):
    cls = getattr(A, name)

    p = dict(params) if params is not None else {}

    if name == "RandomResizedCrop":
        if "size" not in p and ("height" in p and "width" in p):
            h, w = p.pop("height"), p.pop("width")
            p["size"] = (h, w)

        try:
            return cls(**p)
        except Exception:
            p2 = dict(params)
            if "size" in p2 and ("height" not in p2 and "width" not in p2):
                h, w = p2["size"]
                p2.pop("size")
                p2["height"] = h
                p2["width"] = w
            return cls(**p2)

    return cls(**p)


def get_transforms():
    transforms = [
        _make_albu_transform(item["name"], item["params"])
        for item in config["inference_augmentations"]
    ]
    comp = A.Compose(transforms)
    return comp




## === cell 8
class CassavaDataset(Dataset):
    def __init__(self, images, transforms):
        self.images = images
        self.transforms = transforms

    def __getitem__(self, n):
        img_path = os.path.join(config["test_location"], self.images[n])
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)["image"]
        image = np.moveaxis(image, -1, 0)  # HWC -> CHW
        image = torch.tensor(image, dtype=torch.float32)
        return image

    def __len__(self):
        return len(self.images)




## === cell 9
def get_dataloader():
    transforms = get_transforms()
    test_data = np.array(test["image_id"])
    data = CassavaDataset(test_data, transforms)

    workers = config["workers"]
    if device.type == "cpu":
        workers = 0

    dataloader = DataLoader(
        data,
        shuffle=False,
        batch_size=config["batch_size"],
        pin_memory=(device.type == "cuda"),
        num_workers=workers,
    )
    return dataloader




## === cell 10
def infer(model, dataloader):
    print("Running inference...")
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(device, non_blocking=(device.type == "cuda"))
            batch_hat = model(batch)
            predictions.append(batch_hat.detach().cpu())

    return torch.cat(predictions, dim=0)




## === cell 11
if __name__ == "__main__":
    if device.type == "cuda":
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
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    predictions /= config["epochs"]
    results = predictions.numpy()
    test["label"] = np.argmax(results, axis=-1).astype(int)

    submission = sample[["image_id"]].merge(
        test[["image_id", "label"]], on="image_id", how="left"
    )
    if submission["label"].isna().any():
        missing = submission[submission["label"].isna()]["image_id"].head().tolist()
        raise RuntimeError(f"Missing predictions for some test images, e.g.: {missing}")

    submission["label"] = submission["label"].astype(int)
    submission.to_csv("submission.csv", index=False)
    print(submission.head())
    print("Wrote submission.csv with rows:", len(submission))
    print("Saved at:", os.path.abspath("submission.csv"))
