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

0.14686

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I filter the test image list to only JPEG files (using the provided sample submission order) and adjust the transform builder so that `RandomResizedCrop` receives the correct height‑width arguments instead of a single integer. These fixes remove the Albumentations validation error and ensure the generated CSV has the exact number of rows required for a valid submission.'
- What this solution (achieved 0.61099) has done: 'I fix the Albumentations `RandomResizedCrop` error and remove stochastic augmentations during inference, replacing it with a deterministic `Resize` + `Normalize`. This resolves the validation exception, ensures the model receives correctly‑sized inputs, and yields a stable prediction that can reach the target accuracy.'
- What this solution (achieved 0.14686) has done: 'I enable ImageNet pretrained weights for EfficientNet‑B7 (the model was previously loaded with random weights) and add a simple deterministic test‑time augmentation: for each batch we also evaluate the horizontally‑flipped images and average the logits with the original predictions. These tiny changes keep the original architecture and training logic intact while providing a more informative model and a modest boost in validation accuracy, moving the score closer to the target.'
- What this solution (achieved 0.14686) has done: 'I adjust the model‑loading logic so that the fine‑tuned EfficientNet‑B7 checkpoint is correctly found in the Kaggle `/kaggle/input` directory (the original relative path often points to a non‑existent location). By loading the proper checkpoint the model’s learned weights for the five disease classes are restored, which should raise the validation accuracy substantially and move the score toward the target. No other parts of the pipeline are changed, preserving the original architecture, training‑free inference flow, and deterministic preprocessing.'

# 9. Code solution

## === cell 0
import numpy as np
import cv2
import gc
import random
import torch
import os
import pandas as pd
from torch import optim, nn
import albumentations as A
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm
from torchvision.models import efficientnet_b7




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
            name="RandomResizedCrop", params=dict(height=image_size, width=image_size)
        ),
        dict(
            name="HorizontalFlip",
            params=dict(
                always_apply=False,
                p=0.5,
            ),
        ),
        dict(name="Transpose", params=dict(p=0.5)),
        dict(name="VerticalFlip", params=dict(always_apply=False, p=0.5)),
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
        dict(name="CoarseDropout", params=dict(p=0.5)),
        dict(name="Cutout", params=dict(p=0.5)),
    ],
)




## === cell 3
sample_sub_path = os.path.join(
    "..", "input", "cassava-leaf-disease-classification", "sample_submission.csv"
)
test = pd.read_csv(sample_sub_path)
valid_files = set(
    f for f in os.listdir(config["test_location"]) if f.lower().endswith(".jpg")
)
test = test[test["image_id"].isin(valid_files)].reset_index(drop=True)
test.head()




## === cell 4
def seed(seed=22):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)




## === cell 5
seed(config["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




## === cell 6
def load_model():
    model = efficientnet_b7(pretrained=True)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 5)

    possible_paths = [
        os.path.join(config["checkpoint_path"], config["checkpoint"]),
        os.path.join("/kaggle/input", "savedmodels2", config["checkpoint"]),
    ]
    ckpt_path = next((p for p in possible_paths if os.path.exists(p)), None)

    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location=device)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
            epoch = checkpoint.get("epoch", "N/A")
            train_loss = checkpoint.get("train_loss", "N/A")
            val_loss = checkpoint.get("val_loss", "N/A")
            metrics = checkpoint.get("metrics", "N/A")
            lr = checkpoint.get("lr", "N/A")
        else:
            model.load_state_dict(checkpoint)
            epoch = train_loss = val_loss = metrics = lr = "N/A"

        print("Loading model from checkpoint:", ckpt_path)
        print("Epoch", epoch)
        print("Train loss", train_loss)
        print("Validation loss", val_loss)
        print("Accuracy", metrics)
        print("Learning rate", lr)
    else:
        print("Checkpoint not found, using pretrained ImageNet weights only.")

    model.eval()
    return model.to(device)




## === cell 7
def get_transforms():
    """
    Build a deterministic inference pipeline.
    RandomResizedCrop is replaced with a fixed Resize,
    and all stochastic augmentations are omitted.
    """
    transforms = []
    for item in config["inference_augmentations"]:
        name = item["name"]
        params = item["params"].copy()

        if name == "RandomResizedCrop":
            name = "Resize"
            params = {"height": image_size, "width": image_size}

        if name not in ("Resize", "Normalize"):
            continue

        transforms.append(getattr(A, name)(**params))

    return A.Compose(transforms)




## === cell 8
class CassavaDataset(Dataset):
    def __init__(self, images, transforms):
        self.images = images
        self.transforms = transforms

    def __getitem__(self, n):
        img_path = os.path.join(config["test_location"], self.images[n])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)["image"]
        image = np.moveaxis(image, -1, 0)  # C x H x W
        return torch.FloatTensor(image)

    def __len__(self):
        return len(self.images)




## === cell 9
def get_dataloader():
    transforms = get_transforms()
    test_data = np.array(test["image_id"])
    dataset = CassavaDataset(test_data, transforms)
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=config["batch_size"],
        pin_memory=True if device.type == "cuda" else False,
        num_workers=config["workers"],
    )
    return dataloader




## === cell 10
def infer(model, dataloader):
    """
    Perform inference with a simple deterministic TTA:
    average predictions on the original and horizontally‑flipped images.
    """
    print("Running inference with TTA (horizontal flip)...")
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(device)

            logits_orig = model(batch)

            batch_flipped = torch.flip(batch, dims=[-1])
            logits_flip = model(batch_flipped)

            logits = (logits_orig + logits_flip) / 2.0

            predictions.append(logits.cpu())

    return torch.cat(predictions, dim=0)




## === cell 11
if __name__ == "__main__":
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
        torch.cuda.empty_cache()
        gc.collect()

    predictions /= config["epochs"]
    results = predictions.numpy()
    test["label"] = np.argmax(results, axis=-1).astype(int)




## === cell 12
submission_path = "submission.csv"
test.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
