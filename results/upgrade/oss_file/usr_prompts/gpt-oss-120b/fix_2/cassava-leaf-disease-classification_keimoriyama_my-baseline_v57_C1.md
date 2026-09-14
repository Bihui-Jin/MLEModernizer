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

3.9

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
import random
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split, KFold
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import timm



## === cell 1
base_path = "../input/cassava-leaf-disease-classification"
train_csv_path = os.path.join(base_path, "train.csv")
train_images_path = os.path.join(base_path, "train_images")
test_images_path = os.path.join(base_path, "test_images")
sample_submission_path = os.path.join(base_path, "sample_submission.csv")



## === cell 2
df = pd.read_csv(train_csv_path)
df["path"] = df["image_id"].apply(lambda x: os.path.join(train_images_path, x))
df = df.drop(columns=["image_id"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)



## === cell 3
train_df, valid_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"].values
)




## === cell 4
class CassavaDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.loc[idx, "path"]
        label = self.df.loc[idx, "label"]
        with open(path, "rb") as f:
            image = Image.open(f).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(label)




## === cell 5
class make_mask_image:
    def __init__(self, p, mask_size=50):
        self.p = p
        self.mask_size = mask_size

    def __call__(self, image):
        if random.random() < self.p:
            draw = ImageDraw.Draw(image)
            w, h = image.size
            for _ in range(10):
                x = random.randrange(0, w - self.mask_size)
                y = random.randrange(0, h - self.mask_size)
                draw.rectangle(
                    (x, y, x + self.mask_size, y + self.mask_size),
                    fill=(0, 0, 0),
                    outline=(0, 0, 0),
                )
        return image




## === cell 6
image_size = 512
train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomResizedCrop(image_size),
        make_mask_image(p=0.3, mask_size=50),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

valid_transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)



## === cell 7
full_dataset = CassavaDataset(df, transform=train_transform)



## === cell 8
epoch = 3
batch_size = 16
num_classes = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



## === cell 9
resNet = timm.create_model("resnet50", pretrained=True, num_classes=num_classes)
resNet = resNet.to(device)

ef_model = timm.create_model(
    "tf_efficientnet_b2_ns", pretrained=True, num_classes=num_classes
)
ef_model = ef_model.to(device)



## === cell 10
resNet_optimizer = torch.optim.AdamW(resNet.parameters(), lr=1e-4, weight_decay=1e-4)
ef_optimizer = torch.optim.AdamW(ef_model.parameters(), lr=1e-4, weight_decay=1e-4)

resNet_scheduler = torch.optim.lr_scheduler.StepLR(
    resNet_optimizer, step_size=2, gamma=0.1
)
ef_scheduler = torch.optim.lr_scheduler.StepLR(ef_optimizer, step_size=2, gamma=0.1)

criterion = nn.CrossEntropyLoss()




## === cell 11
def plot_losses(title, train_losses, valid_losses):
    plt.figure()
    plt.plot(train_losses, label="train loss")
    plt.plot(valid_losses, label="valid loss")
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()




## === cell 12
def train_model(model, dataset, optimizer, scheduler, model_path):
    best_state = None
    best_loss = float("inf")
    train_losses, valid_losses = [], []

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, valid_idx) in enumerate(kf.split(dataset)):
        print(f"\n=== Fold {fold+1} ===")
        train_sub = Subset(dataset, train_idx)
        valid_sub = Subset(dataset, valid_idx)

        train_loader = DataLoader(
            train_sub, batch_size=batch_size, shuffle=True, num_workers=2
        )
        valid_loader = DataLoader(
            valid_sub, batch_size=batch_size, shuffle=False, num_workers=2
        )

        for ep in range(1, epoch + 1):
            model.train()
            t_loss = 0.0
            for imgs, targets in train_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, targets)
                loss.backward()
                optimizer.step()
                t_loss += loss.item() * imgs.size(0)
            t_loss /= len(train_loader.dataset)
            train_losses.append(t_loss)

            model.eval()
            v_loss = 0.0
            correct = 0
            with torch.no_grad():
                for imgs, targets in valid_loader:
                    imgs, targets = imgs.to(device), targets.to(device)
                    out = model(imgs)
                    loss = criterion(out, targets)
                    v_loss += loss.item() * imgs.size(0)
                    correct += (out.argmax(1) == targets).sum().item()
            v_loss /= len(valid_loader.dataset)
            valid_losses.append(v_loss)

            acc = correct / len(valid_loader.dataset)
            print(
                f"Epoch {ep} | Train loss: {t_loss:.4f} | Valid loss: {v_loss:.4f} | Acc: {acc:.4f}"
            )

            scheduler.step()

            if v_loss < best_loss:
                best_loss = v_loss
                best_state = model.state_dict()

    torch.save(best_state, model_path)
    return train_losses, valid_losses




## === cell 13
resNet_path = "./resnet50_cassava.pth"
ef_path = "./efnet_cassava.pth"

print("Training ResNet50...")
resnet_train_losses, resnet_valid_losses = train_model(
    resNet, full_dataset, resNet_optimizer, resNet_scheduler, resNet_path
)

print("Training EfficientNet B2...")
ef_train_losses, ef_valid_losses = train_model(
    ef_model, full_dataset, ef_optimizer, ef_scheduler, ef_path
)

plot_losses("ResNet50 losses", resnet_train_losses, resnet_valid_losses)
plot_losses("EfficientNet B2 losses", ef_train_losses, ef_valid_losses)



## === cell 14
resNet.load_state_dict(torch.load(resNet_path, map_location=device))
ef_model.load_state_dict(torch.load(ef_path, map_location=device))




## === cell 15
class CassavaEnsemble(nn.Module):
    def __init__(self, model_a, model_b, weight_a=0.3):
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b
        self.weight_a = weight_a

    def forward(self, x):
        out_a = self.model_a(x)
        out_b = self.model_b(x)
        return self.weight_a * out_a + (1 - self.weight_a) * out_b

    def predict(self, x, weight_a):
        out_a = self.model_a(x)
        out_b = self.model_b(x)
        return weight_a * out_a + (1 - weight_a) * out_b




## === cell 16
classifier = CassavaEnsemble(resNet, ef_model, weight_a=0.3).to(device)
classifier.eval()



## === cell 17
test_files = [
    f
    for f in os.listdir(test_images_path)
    if os.path.isfile(os.path.join(test_images_path, f))
    and f.lower().endswith((".jpg", ".png"))
]

image_ids = []
preds = []

for fname in test_files:
    img_path = os.path.join(test_images_path, fname)
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        continue
    img_tensor = valid_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = classifier(img_tensor)
        pred_label = out.argmax(1).item()
    image_ids.append(fname)
    preds.append(pred_label)



## === cell 18
submission = pd.DataFrame({"image_id": image_ids, "label": preds})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path} with {len(submission)} rows.")
