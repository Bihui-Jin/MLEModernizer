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

3.12

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
timm==1.0.19
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

0.8970988213961922

# 6. Current score

0.11323

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.58857) has done: 'I fix the Albumentations RandomResizedCrop definition in the test augmentation pipeline (it expects a `size` tuple, not `height`/`width`). This resolves the validation error, defines `test_augs` correctly, and allows the subsequent inference cells to run, producing a proper CSV submission.'
- What this solution (achieved 0.50785) has done: 'I increase the test‑time augmentation repetitions (TTA) from 5 to 8 and replace the L2‑normalisation of model logits with a standard softmax before ensembling. Using probabilities rather than L2‑scaled vectors generally yields a better calibrated ensemble, helping the accuracy move closer to the target score while keeping the original architecture and training logic unchanged.'
- What this solution (achieved 0.5213) has done: 'I correct the input directory path so the script can locate the test images (and any checkpoint files) in the standard Kaggle `/kaggle/input` location. This fixes the `FileNotFoundError`, allowing the inference loop to run and the submission CSV to be generated. No changes are made to the model architecture or training logic, keeping the core approach intact.'
- What this solution (achieved 0.11323) has done: 'The update batches image loading and model inference, using a DataLoader with multiple workers to parallelize I/O and reduce per‑image Python overhead while keeping the same number of test‑time augmentations and averaging logic. Random seeds and deterministic CuDNN settings are added to ensure reproducible results. No model architecture or training logic is changed.'
- What this solution (achieved 0.11323) has done: 'I improve the validation‑accuracy estimate and the test‑time predictions by averaging **probabilities** instead of raw logits during TTA. This small change keeps the exact model architecture and training untouched, but usually yields a better‑calibrated ensemble, raising the validation accuracy and moving the final Kaggle score toward the target. The rest of the pipeline (loading checkpoints, data handling, CSV output) stays unchanged.'

# 9. Code solution

## === cell 0
import os
import random
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

BASE_INPUT_DIR = "/kaggle/input"
if not os.path.isdir(BASE_INPUT_DIR):
    BASE_INPUT_DIR = os.path.abspath("input")

ROOT_DIR = os.path.abspath(".")
INPUT_PATH = os.path.join(BASE_INPUT_DIR, "cassava-leaf-disease-classification")
TRAIN_CSV_PATH = os.path.join(INPUT_PATH, "train.csv")
TRAIN_IMAGE_PATH = os.path.join(INPUT_PATH, "train_images")
TEST_IMAGE_PATH = os.path.join(INPUT_PATH, "test_images")
SUBMISSION_PATH = "submission.csv"

RESNEXT_PATH = "1022_res50.pth"
B4_PATH = "1022_b4ns.pth"

DEVICES = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
DEVICE = DEVICES[0] if DEVICES else torch.device("cpu")

OUT_FEATURES = 5
BATCH_SIZE = 32
IMAGE_SIZE = 512
TTA = 8  # test‑time augmentations
VAL_TTA = 4  # validation augmentations
NUM_WORKERS = min(4, os.cpu_count() or 1)

test_augs = A.Compose(
    [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ]
)



## === cell 1
my_model_1 = torch.hub.load("pytorch/vision:v0.15.2", "resnet50", pretrained=True)
my_model_1.fc = nn.Linear(my_model_1.fc.in_features, OUT_FEATURES)

my_model_2 = torch.hub.load(
    "pytorch/vision:v0.15.2", "efficientnet_b4", pretrained=True
)
my_model_2.classifier[1] = nn.Linear(my_model_2.classifier[1].in_features, OUT_FEATURES)


def load_checkpoint(model, ckpt_path):
    """Load a checkpoint into a model if the file exists."""
    if os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=DEVICE)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
    return model


my_model_1 = load_checkpoint(my_model_1, os.path.join(INPUT_PATH, RESNEXT_PATH))
my_model_2 = load_checkpoint(my_model_2, os.path.join(INPUT_PATH, B4_PATH))

if torch.cuda.is_available():
    my_model_1 = nn.DataParallel(my_model_1).to(DEVICE)
    my_model_2 = nn.DataParallel(my_model_2).to(DEVICE)
else:
    my_model_1 = my_model_1.to(DEVICE)
    my_model_2 = my_model_2.to(DEVICE)

my_model_1.eval()
my_model_2.eval()

val_df = pd.read_csv(TRAIN_CSV_PATH)
val_df = val_df.sample(frac=0.1, random_state=42).reset_index(drop=True)


class CassavaValDataset(Dataset):
    def __init__(self, df, image_dir, transform):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["image_id"])
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        img_tensor = self.transform(image=img_np)["image"]
        label = int(row["label"])
        return img_tensor, label, idx


val_dataset = CassavaValDataset(val_df, TRAIN_IMAGE_PATH, test_augs)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)


def compute_accuracy_batch(model, loader, true_labels):
    """
    Compute validation accuracy by averaging **softmax probabilities**
    over VAL_TTA augmentations (instead of raw logits).
    """
    model.eval()
    probs_sum = torch.zeros(len(true_labels), OUT_FEATURES, device=DEVICE)
    with torch.no_grad():
        for _ in range(VAL_TTA):
            for imgs, _, idxs in loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                logits = model(imgs)
                probs = torch.softmax(logits, dim=1)
                probs_sum[idxs] += probs
    probs_avg = probs_sum / VAL_TTA
    preds = probs_avg.argmax(dim=1)
    acc = (preds == true_labels).float().mean().item()
    return acc


labels_tensor = torch.tensor(val_df["label"].values, dtype=torch.long, device=DEVICE)

acc1 = compute_accuracy_batch(my_model_1, val_loader, labels_tensor)
acc2 = compute_accuracy_batch(my_model_2, val_loader, labels_tensor)

print(f"Validation accuracy - Model 1 (ResNet50): {acc1:.4f}")
print(f"Validation accuracy - Model 2 (EfficientNet B4): {acc2:.4f}")

if acc1 + acc2 > 0:
    w1 = acc1 / (acc1 + acc2)
    w2 = acc2 / (acc1 + acc2)
else:
    w1 = w2 = 0.5
print(f"Ensemble weights -> Model1: {w1:.3f}, Model2: {w2:.3f}")



## === cell 2
test_image_list = sorted(
    [
        img
        for img in os.listdir(TEST_IMAGE_PATH)
        if img.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
)


class CassavaTestDataset(Dataset):
    def __init__(self, image_list, image_dir, transform):
        self.image_list = image_list
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        img_tensor = self.transform(image=img_np)["image"]
        return img_tensor, idx


test_dataset = CassavaTestDataset(test_image_list, TEST_IMAGE_PATH, test_augs)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

probs_sum_1 = torch.zeros(len(test_dataset), OUT_FEATURES, device=DEVICE)
probs_sum_2 = torch.zeros(len(test_dataset), OUT_FEATURES, device=DEVICE)

with torch.no_grad():
    for _ in range(TTA):
        for imgs, idxs in test_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            out1 = my_model_1(imgs)
            out2 = my_model_2(imgs)
            probs1 = torch.softmax(out1, dim=1)
            probs2 = torch.softmax(out2, dim=1)
            probs_sum_1[idxs] += probs1
            probs_sum_2[idxs] += probs2

prob_1 = probs_sum_1 / TTA
prob_2 = probs_sum_2 / TTA



## === cell 3
final_pred = (prob_1 * w1) + (prob_2 * w2)
label = final_pred.argmax(dim=1).cpu().numpy()

df_submission = pd.DataFrame({"image_id": test_image_list, "label": label})
df_submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved to {SUBMISSION_PATH}")
