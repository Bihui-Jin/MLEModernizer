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
import pandas as pd
import timm
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from tqdm import tqdm



## === cell 1
path = "../input/cassava-leaf-disease-classification"
print(os.listdir(path))


## === cell 2
df = pd.read_csv(os.path.join(path, "train.csv"))


## === cell 3
print(df.head())


## === cell 4
df["path"] = df["image_id"].map(lambda x: os.path.join(path, "train_images", x))
df = df.drop(columns=["image_id"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)


## === cell 5
print(df.head())


## === cell 6
from sklearn import model_selection

train_df, valid_df = model_selection.train_test_split(
    df, test_size=0.2, random_state=42, stratify=df.label.values
)


## === cell 7
train_df.label.value_counts().plot(kind="bar")
plt.show()


## === cell 8
valid_df.label.value_counts().plot(kind="bar")
plt.show()


## === cell 9
train_df = train_df.reset_index(drop=True)
print(train_df.head())


## === cell 10
valid_df = valid_df.reset_index(drop=True)
print(valid_df.head())


## === cell 11
im = Image.open(train_df["path"][0])
im.show()


## === cell 12
im


## === cell 13
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold




## === cell 14
class CassavaDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        path = self.df["path"].iloc[index]
        label = self.df["label"].iloc[index]
        with open(path, "rb") as f:
            image = Image.open(f).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label




## === cell 15
import random




## === cell 16
class make_mask_image:
    def __init__(self, p, mask_size=50):
        self.p = p
        self.mask_size = mask_size

    def __call__(self, image):
        if random.random() < self.p:
            draw = ImageDraw.Draw(image)
            width, height = image.size
            for _ in range(10):
                x = random.randrange(0, width - self.mask_size)
                y = random.randrange(0, height - self.mask_size)
                draw.rectangle(
                    (x, y, x + self.mask_size, y + self.mask_size),
                    fill=(0, 0, 0),
                    outline=(0, 0, 0),
                )
        return image




## === cell 17
image_size = 512
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomResizedCrop(image_size),
        make_mask_image(p=0.5, mask_size=50),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

valid_transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)


## === cell 18
dataset = CassavaDataset(train_df, train_transform)


## === cell 19
import json

label_map_path = os.path.join(path, "label_num_to_disease_map.json")
with open(label_map_path, "r") as f:
    label_to_name = json.load(f)




## === cell 20
class Unnormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor




## === cell 21
unnorm = Unnormalize(mean, std)




## === cell 22
def display_img(img_tensor, unnorm=None, label=None):
    if unnorm is not None:
        img_tensor = unnorm(img_tensor)
    plt.imshow(img_tensor.permute(1, 2, 0).cpu().numpy())
    if label is not None:
        plt.title(label_to_name[str(label)])
    plt.show()




## === cell 23
def display_batch(batch, unnorm=None):
    imgs, _ = batch
    if unnorm:
        imgs = [unnorm(img) for img in imgs]
    plt.figure(figsize=(16, 8))
    plt.axis("off")
    plt.imshow(make_grid(imgs, nrow=8).permute(1, 2, 0).cpu().numpy())
    plt.show()




## === cell 24
tensor, label = dataset[0]
display_img(tensor, unnorm, label)


## === cell 25
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)
display_batch(next(iter(loader)), unnorm)


## === cell 26
import torch.nn as nn



## === cell 27
epoch = 8
batch_size = 16
num_classes = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


## === cell 28
resNet = timm.create_model("resnet50", pretrained=True)
resNet.fc = nn.Linear(resNet.fc.in_features, num_classes)
resNet = resNet.to(device)


## === cell 29
ef_model = timm.create_model("tf_efficientnet_b2_ns", pretrained=True)
ef_model.classifier = nn.Linear(ef_model.classifier.in_features, num_classes)
ef_model = ef_model.to(device)


## === cell 30
resNet_optimizer = torch.optim.AdamW(resNet.parameters(), lr=1e-4, weight_decay=1e-4)
resNet_scheduler = torch.optim.lr_scheduler.StepLR(
    resNet_optimizer, step_size=2, gamma=0.1
)

ef_optimizer = torch.optim.AdamW(ef_model.parameters(), lr=1e-4, weight_decay=1e-4)
ef_scheduler = torch.optim.lr_scheduler.StepLR(ef_optimizer, step_size=2, gamma=0.1)

criterion = nn.CrossEntropyLoss()




## === cell 31
def calc_correction(model, df):
    model.eval()
    paths = df["path"].values
    labels = df["label"].values
    correct = 0
    pred_counts = [0] * num_classes
    with torch.no_grad():
        for img_path, true_label in zip(paths, labels):
            img = Image.open(img_path).convert("RGB")
            img = valid_transform(img).unsqueeze(0).to(device)
            pred = model(img).argmax(1).item()
            pred_counts[pred] += 1
            if pred == true_label:
                correct += 1
    accuracy = correct / len(paths)
    return accuracy, pred_counts




## === cell 32
def plot_losses(max_epoch, title, train_losses, valid_losses):
    epochs = list(range(1, len(train_losses) + 1))
    plt.plot(epochs, train_losses, label="train loss")
    plt.plot(epochs, valid_losses, label="valid loss")
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()




## === cell 33
def train_model(
    model, dataset, batch_size, optimizer, criterion, scheduler, epochs, model_path
):
    best_model_wts = None
    best_loss = float("inf")
    train_losses, valid_losses = [], []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold+1}/5")
        train_subset = Subset(dataset, train_idx)
        valid_subset = Subset(dataset, valid_idx)

        train_loader = DataLoader(train_subset, batch_size, shuffle=True, num_workers=2)
        valid_loader = DataLoader(
            valid_subset, batch_size, shuffle=False, num_workers=2
        )

        for epoch_num in range(1, epochs + 1):
            model.train()
            running_train_loss = 0.0
            for imgs, targets in train_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item() * imgs.size(0)
            epoch_train_loss = running_train_loss / len(train_loader.dataset)
            train_losses.append(epoch_train_loss)

            model.eval()
            running_valid_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for imgs, targets in valid_loader:
                    imgs, targets = imgs.to(device), targets.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, targets)
                    running_valid_loss += loss.item() * imgs.size(0)

                    preds = outputs.argmax(1)
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)

            epoch_valid_loss = running_valid_loss / len(valid_loader.dataset)
            valid_losses.append(epoch_valid_loss)

            epoch_acc = correct / total
            print(
                f"Epoch {epoch_num}/{epochs} | TrainLoss: {epoch_train_loss:.4f} "
                f"| ValidLoss: {epoch_valid_loss:.4f} | Acc: {epoch_acc:.4f}"
            )

            if epoch_valid_loss < best_loss:
                best_loss = epoch_valid_loss
                best_model_wts = model.state_dict()

            scheduler.step()

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), model_path)
    return model, train_losses, valid_losses




## === cell 34
def train_models(resNet, ef_model):
    res_path = "./res_model.pth"
    resNet, res_train_losses, res_valid_losses = train_model(
        resNet,
        dataset,
        batch_size,
        resNet_optimizer,
        criterion,
        resNet_scheduler,
        epoch,
        res_path,
    )
    print("ResNet validation accuracy:", calc_correction(resNet, valid_df)[0])
    plot_losses(epoch, "ResNet losses", res_train_losses, res_valid_losses)

    ef_path = "./ef_model.pth"
    ef_model, ef_train_losses, ef_valid_losses = train_model(
        ef_model,
        dataset,
        batch_size,
        ef_optimizer,
        criterion,
        ef_scheduler,
        epoch,
        ef_path,
    )
    print("EfficientNet validation accuracy:", calc_correction(ef_model, valid_df)[0])
    plot_losses(epoch, "EfficientNet losses", ef_train_losses, ef_valid_losses)




## === cell 35
class CassaveClassifier(nn.Module):
    def __init__(self, model_a, model_b):
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b

    def forward(self, x):
        return 0.5 * self.model_a(x) + 0.5 * self.model_b(x)

    def test(self, x, rate):
        return rate * self.model_a(x) + (1 - rate) * self.model_b(x)




## === cell 36
classifier = CassaveClassifier(resNet, ef_model).to(device)


## === cell 37
train_models(resNet, ef_model)


## === cell 38
test_path = "../input/cassava-leaf-disease-classification/test_images"


## === cell 39
image_ids = []
image_paths = []
for fname in os.listdir(test_path):
    full_path = os.path.join(test_path, fname)
    if os.path.isfile(full_path):  # skip directories
        image_ids.append(fname)
        image_paths.append(full_path)


## === cell 40
preds = []
classifier.eval()
with torch.no_grad():
    for img_path in tqdm(image_paths):
        img = Image.open(img_path).convert("RGB")
        img = valid_transform(img).unsqueeze(0).to(device)
        pred = classifier(img).argmax(1).item()
        preds.append(pred)


## === cell 41
print(f"Generated predictions for {len(preds)} images.")


## === cell 42
submission = pd.DataFrame({"image_id": image_ids, "label": preds})


## === cell 43
print(submission.head())


## === cell 44
submission.to_csv("submission.csv", index=False)
print("Saved submission to submission.csv")
