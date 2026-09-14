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
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

torch.backends.cudnn.benchmark = True

INPUT_PATH = "../input/ensemble-1023/"
TRAIN_CSV_PATH = "../input/cassava-leaf-disease-classification/train.csv"
TRAIN_IMAGE_PATH = "../input/cassava-leaf-disease-classification/train_images/"
TEST_IMAGE_PATH = "../input/cassava-leaf-disease-classification/test_images/"
SUBMISSION_PATH = "submission.csv"
RESNEXT_PATH = "1022_res50.pth"
B4_PATH = "1022_b4ns.pth"
OUT_FEATURES = 5
NUM_EPOCHS = 3  # short fine‑tune to boost accuracy
BATCH_SIZE = 32
IMAGE_SIZE = 512
OPTIMIZER = torch.optim.AdamW
SEED = 42
LR_START = 1e-4
LR_MAX = 2e-4
LR_FINAL = 1e-5
TTA = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

my_model_1 = timm.create_model(
    "resnext50_32x4d", pretrained=True, num_classes=OUT_FEATURES
)
my_model_2 = timm.create_model(
    "tf_efficientnet_b4_ns", pretrained=True, num_classes=OUT_FEATURES
)


def load_checkpoint(model, path):
    if os.path.isfile(path):
        model_param = torch.load(path, map_location=device)
        new_model_param = {
            k[7:]: v for k, v in model_param.items() if k.startswith("module.")
        }
        if not new_model_param:
            new_model_param = model_param
        model.load_state_dict(new_model_param)
        return True
    return False


resnext_path_full = os.path.join(INPUT_PATH, RESNEXT_PATH)
if not load_checkpoint(my_model_1, resnext_path_full):
    fallback_path = os.path.join(
        "../input/cassava-leaf-disease-classification/", RESNEXT_PATH
    )
    load_checkpoint(my_model_1, fallback_path)

b4_path_full = os.path.join(INPUT_PATH, B4_PATH)
if not load_checkpoint(my_model_2, b4_path_full):
    fallback_path = os.path.join(
        "../input/cassava-leaf-disease-classification/", B4_PATH
    )
    load_checkpoint(my_model_2, fallback_path)

my_model_1 = nn.DataParallel(my_model_1).to(device)
my_model_2 = nn.DataParallel(my_model_2).to(device)

if not (os.path.isfile(resnext_path_full) or os.path.isfile(b4_path_full)):

    class CassavaDataset(torch.utils.data.Dataset):
        def __init__(self, df, img_dir, transforms):
            self.df = df.reset_index(drop=True)
            self.img_dir = img_dir
            self.transforms = transforms

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_path = os.path.join(self.img_dir, row["image_id"])
            img = np.array(Image.open(img_path).convert("RGB"))
            aug = self.transforms(image=img)
            return aug["image"], row["label"]

    train_df = pd.read_csv(TRAIN_CSV_PATH)
    train_transforms = A.Compose(
        [
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    train_dataset = CassavaDataset(train_df, TRAIN_IMAGE_PATH, train_transforms)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer_1 = OPTIMIZER(my_model_1.parameters(), lr=LR_START)
    optimizer_2 = OPTIMIZER(my_model_2.parameters(), lr=LR_START)

    for epoch in range(NUM_EPOCHS):
        my_model_1.train()
        my_model_2.train()
        epoch_loss = 0.0
        for imgs, labels in tqdm(
            train_loader, desc=f"Fine‑tune Epoch {epoch+1}/{NUM_EPOCHS}"
        ):
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer_1.zero_grad()
            logits_1 = my_model_1(imgs)
            loss1 = criterion(logits_1, labels)
            loss1.backward()
            optimizer_1.step()

            optimizer_2.zero_grad()
            logits_2 = my_model_2(imgs)
            loss2 = criterion(logits_2, labels)
            loss2.backward()
            optimizer_2.step()

            epoch_loss += loss1.item() + loss2.item()
        print(f"Epoch {epoch+1} average loss: {epoch_loss/len(train_loader):.4f}")

test_augs = A.Compose(
    [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ],
    p=1.0,
)




## === cell 1
preds_1 = []
preds_2 = []
image_names = []

my_model_1.eval()
my_model_2.eval()

test_image_list = np.asarray(
    sorted([f for f in os.listdir(TEST_IMAGE_PATH) if f.lower().endswith(".jpg")])
)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, img_list, transforms):
        self.img_dir = img_dir
        self.img_list = img_list
        self.transforms = transforms

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = np.array(Image.open(img_path).convert("RGB"))
        aug = self.transforms(image=img)
        return aug["image"], img_name


test_dataset = TestDataset(TEST_IMAGE_PATH, test_image_list, test_augs)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

for batch_imgs, batch_names in tqdm(test_loader, desc="Inference"):
    batch_imgs = batch_imgs.to(device)

    with torch.no_grad():
        logits_1 = my_model_1(batch_imgs)  # (B, OUT_FEATURES)
        logits_2 = my_model_2(batch_imgs)  # (B, OUT_FEATURES)

    preds_1.append(logits_1.cpu())
    preds_2.append(logits_2.cpu())
    image_names.extend(batch_names)

predictions_1 = torch.cat(preds_1, dim=0)  # shape (N_test, OUT_FEATURES)
predictions_2 = torch.cat(preds_2, dim=0)

w1, w2 = 0.5, 0.5
prob_1 = F.softmax(predictions_1, dim=1)
prob_2 = F.softmax(predictions_2, dim=1)
final_prob = prob_1 * w1 + prob_2 * w2

label = final_prob.argmax(dim=1).numpy()

df_submission = pd.DataFrame({"image_id": image_names, "label": label})
df_submission.to_csv(SUBMISSION_PATH, index=False)
