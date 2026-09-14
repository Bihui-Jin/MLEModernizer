# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

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
TTA = 8  # number of test‑time augmentations
VAL_TTA = 4  # augmentations for validation weighting

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
val_df = val_df.sample(frac=0.1, random_state=42).reset_index(
    drop=True
)  # 10% validation


def compute_accuracy(model, df):
    correct = 0
    total = len(df)
    for _, row in df.iterrows():
        img_path = os.path.join(TRAIN_IMAGE_PATH, row["image_id"])
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        logits = torch.zeros(OUT_FEATURES, device=DEVICE)
        for _ in range(VAL_TTA):
            aug = test_augs(image=img_np)["image"].unsqueeze(0).to(DEVICE)
            logits += model(aug).squeeze(0)
        logits /= VAL_TTA
        pred = logits.argmax().item()
        if pred == int(row["label"]):
            correct += 1
    return correct / total if total > 0 else 0.0


acc1 = compute_accuracy(my_model_1, val_df)
acc2 = compute_accuracy(my_model_2, val_df)

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

preds_1 = []
preds_2 = []

for single_image_name in test_image_list:
    with torch.no_grad():
        ans1 = torch.zeros(OUT_FEATURES, device=DEVICE)
        ans2 = torch.zeros(OUT_FEATURES, device=DEVICE)

        img = Image.open(os.path.join(TEST_IMAGE_PATH, single_image_name)).convert(
            "RGB"
        )
        img_np = np.array(img)

        for _ in range(TTA):
            aug = test_augs(image=img_np)["image"]
            aug = aug.unsqueeze(0).to(DEVICE)

            ans1 += my_model_1(aug).squeeze(0)
            ans2 += my_model_2(aug).squeeze(0)

        ans1 /= TTA
        ans2 /= TTA

        preds_1.append(ans1.cpu())
        preds_2.append(ans2.cpu())

predictions_1 = torch.stack(preds_1, dim=0)
predictions_2 = torch.stack(preds_2, dim=0)

prob_1 = torch.softmax(predictions_1, dim=1)
prob_2 = torch.softmax(predictions_2, dim=1)

torch.cuda.empty_cache()



## === cell 3
final_pred = (prob_1 * w1) + (prob_2 * w2)
label = final_pred.argmax(dim=1).numpy()

df_submission = pd.DataFrame({"image_id": test_image_list, "label": label})
df_submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved to {SUBMISSION_PATH}")
