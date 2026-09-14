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
import pandas as pd
import timm
import torch

torch.manual_seed(42)  # ensure reproducibility




## === cell 1
path = "../input/cassava-leaf-disease-classification"
print("Data path contents:", os.listdir(path))




## === cell 2
df = pd.read_csv(os.path.join(path, "train.csv"))
print(df.shape, df.head())




## === cell 3
df["path"] = df["image_id"].map(lambda x: os.path.join(path, "train_images", x))
df = df.drop(columns=["image_id"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)




## === cell 4
from sklearn import model_selection

train_df, valid_df = model_selection.train_test_split(
    df, test_size=0.2, random_state=42, stratify=df.label.values
)




## === cell 5
train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)




## === cell 6
from PIL import Image, ImageDraw




## === cell 7
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler  # <-- added for mixed precision

torch.backends.cudnn.benchmark = True




## === cell 8
class CassavaDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.iloc[idx]["path"]
        label = int(self.df.iloc[idx]["label"])
        with open(path, "rb") as f:
            image = Image.open(f).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label




## === cell 9
import random


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




## === cell 10
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




## === cell 11
train_dataset = CassavaDataset(train_df, transform=train_transform)




## === cell 12
epoch = 3
batch_size = 64  # increased batch size to reduce number of steps
num_classes = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




## === cell 13
resNet = timm.create_model("resnet50", pretrained=True)
resNet.fc = nn.Linear(resNet.fc.in_features, num_classes)
resNet = resNet.to(device)

ef_model = timm.create_model("tf_efficientnet_b2_ns", pretrained=True)
ef_model.classifier = nn.Linear(ef_model.classifier.in_features, num_classes)
ef_model = ef_model.to(device)




## === cell 14
ef_optimizer = torch.optim.AdamW(ef_model.parameters(), lr=1e-4, weight_decay=1e-4)
ef_scheduler = torch.optim.lr_scheduler.StepLR(ef_optimizer, step_size=2, gamma=0.1)

resNet_optimizer = torch.optim.AdamW(resNet.parameters(), lr=1e-4, weight_decay=1e-4)
resNet_scheduler = torch.optim.lr_scheduler.StepLR(
    resNet_optimizer, step_size=2, gamma=0.1
)

criterion = nn.CrossEntropyLoss()




## === cell 15
def train_one_model(model, optimizer, scheduler, model_title):
    best_state = None
    best_loss = float("inf")
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)

    loader_kwargs = {
        "batch_size": batch_size,
        "pin_memory": True,
        "num_workers": min(12, os.cpu_count() or 4),  # more workers for faster loading
        "persistent_workers": True,
        "prefetch_factor": 2,
    }

    scaler = GradScaler()  # mixed‑precision scaler

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_dataset)):
        print(f"\nTraining fold {fold+1}/5")
        train_sub = Subset(train_dataset, train_idx)
        valid_sub = Subset(train_dataset, valid_idx)

        train_loader = DataLoader(train_sub, shuffle=True, **loader_kwargs)
        valid_loader = DataLoader(valid_sub, shuffle=False, **loader_kwargs)

        for ep in range(1, epoch + 1):
            model.train()
            train_loss = 0.0
            for imgs, targets in train_loader:
                imgs, targets = imgs.to(device, non_blocking=True), targets.to(
                    device, non_blocking=True
                )
                optimizer.zero_grad()
                with autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item() * imgs.size(0)

            train_loss /= len(train_loader.dataset)

            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for imgs, targets in valid_loader:
                    imgs, targets = imgs.to(device, non_blocking=True), targets.to(
                        device, non_blocking=True
                    )
                    with autocast():
                        outputs = model(imgs)
                        loss = criterion(outputs, targets)
                    valid_loss += loss.item() * imgs.size(0)

            valid_loss /= len(valid_loader.dataset)

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_state = model.state_dict()

            scheduler.step()
            print(
                f"Epoch {ep}/{epoch} | Train loss: {train_loss:.4f} | Valid loss: {valid_loss:.4f}"
            )

    torch.save(best_state, model_title)
    model.load_state_dict(best_state)
    return model




## === cell 16
resNet = train_one_model(resNet, resNet_optimizer, resNet_scheduler, "./res_model.pth")
ef_model = train_one_model(ef_model, ef_optimizer, ef_scheduler, "./ef_model.pth")




## === cell 17
class CassavaClassifier(nn.Module):
    def __init__(self, model_a, model_b):
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b

    def forward(self, x):
        out_a = self.model_a(x)
        out_b = self.model_b(x)
        return 0.3 * out_a + 0.7 * out_b

    def predict(self, x, weight_a=0.3):
        out_a = self.model_a(x)
        out_b = self.model_b(x)
        return weight_a * out_a + (1 - weight_a) * out_b




## === cell 18
classifier = CassavaClassifier(resNet, ef_model).to(device)
classifier.eval()




## === cell 19
def evaluate_model(model, df):
    model.eval()
    correct = 0
    total = len(df)
    for _, row in df.iterrows():
        img = Image.open(row["path"]).convert("RGB")
        img = valid_transform(img).unsqueeze(0).to(device)
        with torch.no_grad(), autocast():
            pred = model(img).argmax(1).item()
        if pred == int(row["label"]):
            correct += 1
    return correct / total




## === cell 20
val_acc = evaluate_model(classifier, valid_df)
print(f"Validation accuracy (quick check): {val_acc:.4f}")




## === cell 21
test_path = os.path.join(path, "test_images")
image_path_list = []
image_id_list = []
for fname in os.listdir(test_path):
    full_path = os.path.join(test_path, fname)
    if os.path.isfile(full_path) and fname.lower().endswith(".jpg"):
        image_id_list.append(fname)
        image_path_list.append(full_path)

print(f"Found {len(image_path_list)} test images.")




## === cell 22
preds = []
with torch.no_grad():
    for img_path in image_path_list:
        img = Image.open(img_path).convert("RGB")
        img = valid_transform(img).unsqueeze(0).to(device)
        with autocast():
            pred = classifier(img).argmax(1).item()
        preds.append(pred)




## === cell 23
submission = pd.DataFrame({"image_id": image_id_list, "label": preds})
print(submission.head())




## === cell 24
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
