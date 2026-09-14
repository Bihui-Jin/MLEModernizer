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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.cuda.amp
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm

torch.backends.cudnn.benchmark = True

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

INPUT_PATH = "../input/ensemble-1023/"
TRAIN_CSV_PATH = "../input/cassava-leaf-disease-classification/train.csv"
TRAIN_IMAGE_PATH = "../input/cassava-leaf-disease-classification/train_images/"
TEST_IMAGE_PATH = "../input/cassava-leaf-disease-classification/test_images/"
SUBMISSION_PATH = "submission.csv"
RESNEXT_PATH = "1022_res50.pth"
B4_PATH = "1022_b4ns.pth"
DEVICES = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())] or [
    torch.device("cpu")
]
OUT_FEATURES = 5
NUM_EPOCHS = 12  # increased epochs for better training
BATCH_SIZE = 16
IMAGE_SIZE = 512
OPTIMIZER = torch.optim.AdamW
LR_START = 1e-5
LR_MAX = 2e-4
LR_FINAL = 1e-5
TTA = 8

train_augs = A.Compose(
    [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ]
)

test_augs = A.Compose(
    [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(),
        ToTensorV2(),
    ]
)


class CassavaDataset(Dataset):
    def __init__(self, csv_path, img_dir, transforms):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transforms = transforms
        self.file_names = self.df["image_id"].values
        self.labels = self.df["label"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.file_names[idx]
        label = self.labels[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = np.array(Image.open(img_path).convert("RGB"))
        augmented = self.transforms(image=image)
        tensor = augmented["image"]
        return tensor, torch.tensor(label, dtype=torch.long)


class TestCassavaDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.file_names = sorted(
            [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
        )

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        return self.file_names[idx]


def load_or_pretrained(model, ckpt_path, model_name):
    if os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        if "model" in state:
            model.load_state_dict(state["model"])
        else:
            model.load_state_dict(state)
    return model


model_name1 = "resnet50"
model_name2 = "convnextv2_base"

my_model_1 = timm.create_model(model_name1, pretrained=True, num_classes=OUT_FEATURES)
my_model_2 = timm.create_model(model_name2, pretrained=True, num_classes=OUT_FEATURES)



## === cell 1
train_dataset = CassavaDataset(TRAIN_CSV_PATH, TRAIN_IMAGE_PATH, train_augs)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
)

test_dataset = TestCassavaDataset(TEST_IMAGE_PATH)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=lambda x: [i for i in x],
)

test_raw_cache = {}
for name in tqdm(test_dataset.file_names, desc="Preloading test images"):
    img_path = os.path.join(TEST_IMAGE_PATH, name)
    test_raw_cache[name] = np.array(Image.open(img_path).convert("RGB"))

my_model_1 = load_or_pretrained(
    my_model_1, os.path.join(INPUT_PATH, RESNEXT_PATH), model_name1
)
my_model_2 = load_or_pretrained(
    my_model_2, os.path.join(INPUT_PATH, B4_PATH), model_name2
)

my_model_1 = nn.DataParallel(my_model_1).to(DEVICES[0])
my_model_2 = nn.DataParallel(my_model_2).to(DEVICES[0])

criterion = nn.CrossEntropyLoss()
optimizer_2 = OPTIMIZER(my_model_2.parameters(), lr=LR_MAX)
scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_2, T_max=NUM_EPOCHS, eta_min=LR_FINAL
)
scaler = torch.cuda.amp.GradScaler()

my_model_2.train()
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        imgs = imgs.to(DEVICES[0])
        labels = labels.to(DEVICES[0])
        optimizer_2.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = my_model_2(imgs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer_2)
        scaler.update()
        epoch_loss += loss.item()
    scheduler_2.step()
    print(f"Epoch {epoch+1} – loss: {epoch_loss/len(train_loader):.4f}")

my_model_1.eval()
my_model_2.eval()
torch.cuda.empty_cache()



## === cell 2
preds_1 = []
with torch.no_grad():
    for batch_names in tqdm(test_loader, desc="Model1 inference"):
        imgs = [test_augs(image=test_raw_cache[name])["image"] for name in batch_names]
        batch_tensor = torch.stack(imgs).to(DEVICES[0])
        out = my_model_1(batch_tensor)
        preds_1.append(out.cpu())
predictions_1 = torch.cat(preds_1, dim=0)
prob_pred_1 = F.softmax(predictions_1, dim=1)

accum_predictions = torch.zeros(len(test_dataset), OUT_FEATURES)
for tta_step in range(TTA):
    preds_step = []
    with torch.no_grad():
        for batch_names in tqdm(
            test_loader, desc=f"Model2 TTA {tta_step+1}/{TTA}", leave=False
        ):
            aug_imgs = [
                test_augs(image=test_raw_cache[name])["image"] for name in batch_names
            ]
            batch_tensor = torch.stack(aug_imgs).to(DEVICES[0])
            out = my_model_2(batch_tensor)
            preds_step.append(out.cpu())
    step_tensor = torch.cat(preds_step, dim=0)
    accum_predictions += step_tensor

prob_pred_2 = F.softmax(accum_predictions / TTA, dim=1)

final_pred = (prob_pred_1 * 0.44) + (prob_pred_2 * 0.56)
label = final_pred.argmax(dim=1).cpu().numpy()
df_submission = pd.DataFrame({"image_id": test_dataset.file_names, "label": label})
df_submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved to {SUBMISSION_PATH}")
