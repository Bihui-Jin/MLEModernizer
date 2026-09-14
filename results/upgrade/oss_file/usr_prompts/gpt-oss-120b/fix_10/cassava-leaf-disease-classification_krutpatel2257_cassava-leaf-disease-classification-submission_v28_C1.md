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

0.8884859474161378

# 6. Current score

0.63416

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0938) has done: 'I fix the file‑path error by dynamically selecting the correct input directory (e.g., `/kaggle/input/...`), and adjust the image‑to‑tensor conversion so it works with the already‑normalized NumPy arrays from Albumentations. These minimal changes ensure the script can read the sample submission, run TTA inference, and write a valid `submission.csv` without altering the core model logic.'
- What this solution (achieved 0.23767) has done: 'I replace the heavy random augmentation used during inference with a deterministic resize → center‑crop pipeline and set TTA = 1, so the model receives images in the size it was pretrained on (224 × 224). This small, safe change keeps the core model untouched but usually raises the accuracy dramatically from the near‑random 0.09 toward the target.'
- What this solution (achieved 0.10389) has done: 'I switch the model to a pretrained EfficientNet‑B4 (available in torchvision) so the architecture matches the checkpoint that was originally trained, and I keep loading the checkpoint only if its keys fit the new model. This change is expected to raise accuracy markedly toward the target while preserving the rest of the pipeline.'
- What this solution (achieved 0.18946) has done: 'I add a lightweight training step that runs only when the checkpoint file is missing, so the model gets fine‑tuned on the provided training images for a few epochs. This keeps the original architecture unchanged, uses the same preprocessing, and modestly improves accuracy toward the target. I also add a small TTA (horizontal flip) to the inference loop to gain a bit more performance.'
- What this solution (achieved 0.63416) has done: 'I fixed the Albumentations Resize calls (they now receive height and width as separate arguments) in the validation, test‑time, and TTA pipelines, which removed the schema validation errors and allowed the transforms to be built. With the transforms defined correctly, `tta_transforms` is created, so the inference loop can run and produce a proper `submission.csv`. No other logic was changed, preserving the original model architecture and training flow while enabling the script to finish end‑to‑end.'

# 9. Code solution

## === cell 0
import os
import albumentations
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

try:
    from efficientnet_pytorch import EfficientNet
except ImportError:
    EfficientNet = None


## === cell 1
config = {
    "DATA": {
        "IMAGES": "train_images",
        "LABELS": "train.csv",
        "SUB_IMAGES": "test_images",
        "SUB_LABELS": "sample_submission.csv",
        "SUB_OUTPUT": "submission.csv",
    },
    "DEVICE": "cuda",
    "NUM_GPU": torch.cuda.device_count(),
    "TRAIN_BATCH_SIZE": 32,
    "VAL_BATCH_SIZE": 16,
    "CLASSES": 5,
    "CV_FOLDS": 5,
    "NUM_EPOCHS": 3,  # short fine‑tuning
    "MODEL_PATH": "model.pth",
    "SGD": {"LR": 0.0005, "MOMENTUM": 0.9, "WEIGHT_DECAY": 0.001},
    "COS_ANN_LR": {"ETA_MIN": 0.00001},
    "MODEL_TYPE": "EFFNET_B4",
}


## === cell 2
candidate_dirs = [
    "./input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification",
    "./working/cassava-leaf-disease-classification",
]
base_dir = None
for d in candidate_dirs:
    if os.path.isdir(d):
        base_dir = d
        break
if base_dir is None:
    raise FileNotFoundError(
        "Could not locate the 'cassava-leaf-disease-classification' data directory."
    )

sample_sub_path = os.path.join(base_dir, "sample_submission.csv")
test_images_path = os.path.join(base_dir, "test_images")
model_path = "../input/en-b4-tta-calr-clahe-v2-12-14/model(18).pth"


## === cell 3
if config["MODEL_TYPE"] == "EFFNET_B4":
    model = models.efficientnet_b4(pretrained=True)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, config["CLASSES"])
elif config["MODEL_TYPE"] == "EFFICIENT_NET_B4":
    if EfficientNet is None:
        raise ImportError("EfficientNet requested but not installed.")
    model = EfficientNet.from_pretrained(
        "efficientnet-b4", num_classes=config["CLASSES"]
    )
elif config["MODEL_TYPE"] == "RESNET_50":
    model = models.resnext50_32x4d(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, config["CLASSES"])
else:
    raise ValueError(f"Unsupported MODEL_TYPE: {config['MODEL_TYPE']}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

checkpoint_loaded = False
if os.path.exists(model_path):
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        checkpoint_loaded = True
        print("Checkpoint loaded (partial if necessary).")
    except Exception as e:
        print(f"Warning: failed to load checkpoint ({e}). Using pretrained weights.")
else:
    print(
        f"Warning: model checkpoint not found at {model_path}. Using pretrained weights."
    )

model.eval()


## === cell 4
if not checkpoint_loaded:
    print("Starting lightweight fine‑tuning...")

    train_csv_path = os.path.join(base_dir, config["DATA"]["LABELS"])
    train_images_dir = os.path.join(base_dir, config["DATA"]["IMAGES"])
    train_df = pd.read_csv(train_csv_path)

    class CassavaDataset(torch.utils.data.Dataset):
        def __init__(self, df, img_dir, transforms):
            self.df = df
            self.img_dir = img_dir
            self.transforms = transforms

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_path = os.path.join(self.img_dir, row["image_id"])
            image = np.array(Image.open(img_path).convert("RGB"))
            if self.transforms:
                image = self.transforms(image=image)["image"]
            tensor = torch.from_numpy(image).permute(2, 0, 1).float()
            label = int(row["label"])
            return tensor, label

    train_aug = albumentations.Compose(
        [
            albumentations.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ],
        p=1.0,
    )

    val_aug = albumentations.Compose(
        [
            albumentations.Resize(256, 256),
            albumentations.CenterCrop(224, 224),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ],
        p=1.0,
    )

    dataset = CassavaDataset(train_df, train_images_dir, train_aug)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["TRAIN_BATCH_SIZE"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["SGD"]["LR"],
        momentum=config["SGD"]["MOMENTUM"],
        weight_decay=config["SGD"]["WEIGHT_DECAY"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["NUM_EPOCHS"],
        eta_min=config["COS_ANN_LR"]["ETA_MIN"],
    )

    model.train()
    for epoch in range(config["NUM_EPOCHS"]):
        epoch_loss = 0.0
        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)

        scheduler.step()
        avg_loss = epoch_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{config['NUM_EPOCHS']}] - Loss: {avg_loss:.4f}")

    model.eval()
    print("Fine‑tuning completed.")


## === cell 5
sub_aug = albumentations.Compose(
    [
        albumentations.Resize(256, 256),
        albumentations.CenterCrop(224, 224),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

tta_transforms = [
    lambda x: x,
    lambda x: albumentations.HorizontalFlip(p=1.0)(image=x)["image"],
]


## === cell 6
sample_sub = pd.read_csv(sample_sub_path)
predictions = []
for _, sample_row in sample_sub.iterrows():
    img_path = os.path.join(test_images_path, sample_row.image_id)
    image = np.array(Image.open(img_path).convert("RGB"))

    logits_sum = torch.zeros(config["CLASSES"], device=device)
    for tta in tta_transforms:
        aug_img = sub_aug(image=image)
        tta_img = tta(aug_img["image"])
        tensor_img = torch.from_numpy(tta_img).permute(2, 0, 1).float().to(device)

        with torch.no_grad():
            logits = model(tensor_img.unsqueeze(0)).squeeze(0)
        logits_sum += logits

    avg_logits = logits_sum / len(tta_transforms)
    _, pred_label = torch.max(avg_logits, 0)
    predictions.append([sample_row.image_id, int(pred_label.item())])

sub_df = pd.DataFrame(predictions, columns=["image_id", "label"])
sub_df.to_csv(config["DATA"]["SUB_OUTPUT"], index=False)
print(sub_df.head())
